#!/usr/bin/env python3
"""
Evaluate predictive performance, calibration, uncertainty quality, and selective prediction.

Outputs JSON to evaluation/metrics_<method>_<split>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms
from torchvision.datasets import PCAM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.load_model import get_device, load_hf_image_classifier
from uncertainty_lab.metrics.core import (
    apply_uncertainty_thresholds,
    fit_uncertainty_thresholds,
    json_safe as _json_safe,
    optimize_temperature,
    slide_level_proxy_from_probs,
    summarize_from_logits,
)
from uncertainty_lab.metrics.plots import save_reliability_plot


def get_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--method", default="confidence", choices=["confidence", "mc_dropout", "deep_ensemble"])
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--ensemble_size", type=int, default=3, help="Deep ensemble members")
    p.add_argument("--ensemble_run_ids", type=str, default="", help="Comma-separated run IDs for deep ensemble")
    p.add_argument("--max_samples", type=int, default=2000, help="Evaluate on at most this many samples")
    p.add_argument("--proxy_bag_size", type=int, default=16, help="PCAM patch-to-slide proxy bag size")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=str, default="", help="Optional checkpoints/run_*/best.pt override")
    p.add_argument(
        "--fit_temperature_on_val",
        action="store_true",
        help="Fit temperature on validation set and report calibrated metrics on target split",
    )
    p.add_argument(
        "--fit_deferral_on_val",
        action="store_true",
        help="Fit uncertainty deferral thresholds on validation split and report transfer to target split",
    )
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def _new_model(cfg: dict):
    model, _, _ = load_hf_image_classifier(
        model_id=cfg["model"]["model_id"],
        num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    return model


def _resolve_ckpts(run_id: str, ensemble_run_ids: list[str], ensemble_size: int) -> list[Path]:
    ckpts: list[Path] = []
    if ensemble_run_ids:
        for rid in ensemble_run_ids:
            p = REPO_ROOT / "checkpoints" / rid / "best.pt"
            if p.exists():
                ckpts.append(p)
        return ckpts

    if run_id:
        p = REPO_ROOT / "checkpoints" / run_id / "best.pt"
        if p.exists():
            return [p]

    base = REPO_ROOT / "checkpoints"
    if base.exists():
        runs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
        for d in runs:
            p = d / "best.pt"
            if p.exists():
                ckpts.append(p)
            if len(ckpts) >= max(1, ensemble_size):
                break
    if not ckpts:
        default = REPO_ROOT / "checkpoints" / "best.pt"
        if default.exists():
            ckpts = [default]
    return ckpts


def load_models(cfg: dict, method: str, run_id: str, ensemble_run_ids: list[str], ensemble_size: int) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    if method == "deep_ensemble":
        ckpts = _resolve_ckpts(run_id=run_id, ensemble_run_ids=ensemble_run_ids, ensemble_size=ensemble_size)
        for ckpt in ckpts:
            m = _new_model(cfg)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                m.load_state_dict(state["model_state_dict"], strict=False)
            models.append(m)
        if models:
            return models
    model = _new_model(cfg)
    ckpt = None
    if run_id:
        candidate = REPO_ROOT / "checkpoints" / run_id / "best.pt"
        if candidate.exists():
            ckpt = candidate
    else:
        default = REPO_ROOT / "checkpoints" / "best.pt"
        if default.exists():
            ckpt = default
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
    models.append(model)
    return models


def collect_logits_and_labels(
    models: list[torch.nn.Module],
    split: str,
    transform,
    data_root: Path,
    max_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    method: str,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = PCAM(root=str(data_root), split=split, download=False, transform=transform)
    n_total = len(ds)
    n_use = min(max(1, max_samples), n_total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_use, replace=False)
    subset = torch.utils.data.Subset(ds, indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=max(1, batch_size), shuffle=False, num_workers=0)

    logits_list = []
    y_list = []
    first_model = models[0]
    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if method == "mc_dropout":
            first_model.train()
            probs_stack = []
            with torch.no_grad():
                for _ in range(max(2, mc_samples)):
                    logits = first_model(pixel_values=batch).logits
                    probs_stack.append(torch.softmax(logits, dim=1))
            probs = torch.stack(probs_stack, dim=0).mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
            first_model.eval()
        elif method == "deep_ensemble":
            probs_members = []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    logits_m = m(pixel_values=batch).logits
                    probs_members.append(torch.softmax(logits_m, dim=1))
            probs = torch.stack(probs_members, dim=0).mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
        else:
            first_model.eval()
            with torch.no_grad():
                logits = first_model(pixel_values=batch).logits
        logits_list.append(logits.detach().cpu())
        y_list.append(labels.detach().cpu())

    logits_np = torch.cat(logits_list, dim=0).numpy().astype(np.float64)
    y_np = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    return logits_np, y_np


def main() -> int:
    args = parse_args()
    cfg = get_config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    image_size = tuple(cfg["data"]["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ensemble_run_ids = [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()]
    models = load_models(
        cfg,
        method=args.method,
        run_id=args.run_id,
        ensemble_run_ids=ensemble_run_ids,
        ensemble_size=max(1, args.ensemble_size),
    )
    models = [m.to(device) for m in models]
    n_bins = int(cfg.get("evaluation", {}).get("calibration_bins", 15))
    logits_eval, y_eval = collect_logits_and_labels(
        models=models,
        split=args.split,
        transform=transform,
        data_root=REPO_ROOT / cfg["data"]["root"],
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        method=args.method,
        mc_samples=args.mc_samples,
    )
    base_summary = summarize_from_logits(logits_eval, y_eval, n_bins=n_bins)

    calibration_report = {"uncalibrated": base_summary["calibration"], "temperature_scaling": None}
    if args.fit_temperature_on_val or args.fit_deferral_on_val:
        val_logits, val_y = collect_logits_and_labels(
            models=models,
            split="val",
            transform=transform,
            data_root=REPO_ROOT / cfg["data"]["root"],
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed + 1,
            device=device,
            method=args.method,
            mc_samples=args.mc_samples,
        )
        if args.fit_temperature_on_val:
            temperature = optimize_temperature(val_logits, val_y)
            cal_logits_eval = logits_eval / temperature
            cal_summary = summarize_from_logits(cal_logits_eval, y_eval, n_bins=n_bins)
            calibration_report["temperature_scaling"] = {
                "temperature": float(temperature),
                "calibrated": cal_summary["calibration"],
                "delta_ece": float(cal_summary["calibration"]["ece"] - base_summary["calibration"]["ece"]),
                "delta_nll": float(cal_summary["calibration"]["nll"] - base_summary["calibration"]["nll"]),
                "delta_brier": float(cal_summary["calibration"]["brier"] - base_summary["calibration"]["brier"]),
            }
        if args.fit_deferral_on_val:
            val_summary = summarize_from_logits(val_logits, val_y, n_bins=n_bins)
            val_unc = np.array(val_summary["internals"]["uncertainty_one_minus_msp"], dtype=np.float64)
            val_err = np.array(val_summary["internals"]["error"], dtype=np.int64)
            eval_unc = np.array(base_summary["internals"]["uncertainty_one_minus_msp"], dtype=np.float64)
            eval_err = np.array(base_summary["internals"]["error"], dtype=np.int64)
            thresholds = fit_uncertainty_thresholds(val_unc, val_err, targets=[0.01, 0.02, 0.05], min_coverage=0.2)
            transfer = apply_uncertainty_thresholds(eval_unc, eval_err, thresholds)
            base_summary["selective_prediction"]["validation_fitted_deferral"] = {
                "source_split": "val",
                "method": "uncertainty_one_minus_msp",
                "fit_on_val": thresholds,
                "applied_on_eval": transfer,
                "reference": "Dolezal et al. 2022 style thresholding concept: defer/remove low-confidence predictions.",
            }

    probs_eval = torch.softmax(torch.tensor(logits_eval, dtype=torch.float32), dim=1).numpy()
    bag_size = max(2, int(args.proxy_bag_size))
    slide_proxy = slide_level_proxy_from_probs(probs_eval, y_eval, bag_size=bag_size)

    out = {
        "config": {
            "split": args.split,
            "method": args.method,
            "mc_samples": int(args.mc_samples if args.method == "mc_dropout" else 1),
            "ensemble_size": len(models) if args.method == "deep_ensemble" else 1,
            "max_samples": int(len(y_eval)),
            "dataset_size": int(len(PCAM(root=str(REPO_ROOT / cfg["data"]["root"]), split=args.split, download=False))),
            "run_id": args.run_id or None,
            "model_id": cfg["model"]["model_id"],
            "fit_temperature_on_val": bool(args.fit_temperature_on_val),
            "fit_deferral_on_val": bool(args.fit_deferral_on_val),
        },
        "predictive_performance": base_summary["predictive_performance"],
        "calibration": base_summary["calibration"],
        "uncertainty_quality": base_summary["uncertainty_quality"],
        "selective_prediction": base_summary["selective_prediction"],
        "calibration_report": calibration_report,
        "pathology_reporting": {
            "slide_level_proxy": {
                "dataset": "pcam",
                "proxy_type": "pcam_fixed_bag_aggregation",
                "note": "PCAM has no real slide IDs; this is a pseudo-slide proxy by fixed-size patch bags.",
                **slide_proxy,
            }
        },
        "literature_alignment": {
            "thresholding_deferral_reference": "Dolezal et al. (2022) DOI: 10.1200/JCO.2022.40.16_suppl.8549",
            "near_far_ood_context_references": [
                "Linmans et al. (2022) DOI: 10.1016/j.media.2022.102655",
                "Thagaard et al. (2020) DOI: 10.1007/978-3-030-59710-8_80",
            ],
        },
    }

    # Remove internals before writing final JSON payload.
    out["predictive_performance"] = base_summary["predictive_performance"]
    out["calibration"] = base_summary["calibration"]
    out["uncertainty_quality"] = base_summary["uncertainty_quality"]
    out["selective_prediction"] = base_summary["selective_prediction"]

    if args.out:
        out_path = Path(args.out)
    else:
        out_name = f"metrics_{args.method}_{args.split}.json"
        out_path = REPO_ROOT / "evaluation" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = _json_safe(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    rel_plot = out_path.with_name(out_path.stem + "_reliability.png")
    save_reliability_plot(out["calibration"]["reliability_bins"], rel_plot, title=f"Reliability ({args.method}, {args.split})")
    out["calibration"]["reliability_plot_path"] = str(rel_plot)
    out = _json_safe(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    print(json.dumps(out["predictive_performance"], indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
