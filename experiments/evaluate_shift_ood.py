#!/usr/bin/env python3
"""
Evaluate ID vs synthetic-shift robustness and simple OOD detection.
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
from PIL import ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torchvision import transforms
from torchvision.datasets import PCAM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.load_model import get_device, load_hf_image_classifier


def get_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        ece += abs(acc - c) * (cnt / n)
    return float(ece)


def apply_shift(img, shift: str, severity: int):
    s = max(1, min(5, int(severity)))
    if shift == "id":
        return img
    if shift == "blur":
        radius = [0.5, 1.0, 1.5, 2.0, 2.5][s - 1]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if shift == "noise":
        arr = np.asarray(img).astype(np.float32)
        sigma = [6, 12, 18, 24, 30][s - 1]
        noisy = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        return transforms.ToPILImage()(np.clip(noisy, 0, 255).astype(np.uint8))
    if shift == "jpeg":
        # Approximate compression artifact by down/up sampling.
        scale = [0.95, 0.85, 0.75, 0.6, 0.45][s - 1]
        w, h = img.size
        w2, h2 = max(8, int(w * scale)), max(8, int(h * scale))
        return img.resize((w2, h2)).resize((w, h))
    if shift == "color":
        color = [0.95, 0.9, 0.8, 0.7, 0.6][s - 1]
        contrast = [1.05, 1.1, 1.15, 1.2, 1.25][s - 1]
        out = ImageEnhance.Color(img).enhance(color)
        out = ImageEnhance.Contrast(out).enhance(contrast)
        return out
    return img


def run_split(
    models: list[torch.nn.Module],
    split: str,
    shift: str,
    severity: int,
    max_samples: int,
    batch_size: int,
    seed: int,
    device,
    tfm,
    method: str,
    mc_samples: int,
):
    ds = PCAM(root=str(REPO_ROOT / "data" / "raw"), split=split, download=False, transform=None)
    n = len(ds)
    k = min(max(1, max_samples), n)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=k, replace=False)

    y_true, p1, conf = [], [], []
    first_model = models[0]
    with torch.no_grad():
        for start in range(0, k, batch_size):
            batch_idxs = idxs[start : start + batch_size]
            xs = []
            ys = []
            for i in batch_idxs:
                img, y = ds[int(i)]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                img = apply_shift(img, shift, severity)
                xs.append(tfm(img))
                ys.append(int(y))
            x = torch.stack(xs, dim=0).to(device)
            if method == "mc_dropout":
                first_model.train()
                probs_stack = []
                for _ in range(max(2, int(mc_samples))):
                    logits = first_model(pixel_values=x).logits
                    probs_stack.append(torch.softmax(logits, dim=1))
                probs_t = torch.stack(probs_stack, dim=0).mean(dim=0)
                probs = probs_t.cpu().numpy()
                first_model.eval()
            elif method == "deep_ensemble":
                probs_members = []
                for m in models:
                    m.eval()
                    logits_m = m(pixel_values=x).logits
                    probs_members.append(torch.softmax(logits_m, dim=1))
                probs_t = torch.stack(probs_members, dim=0).mean(dim=0)
                probs = probs_t.cpu().numpy()
            else:
                first_model.eval()
                logits = first_model(pixel_values=x).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(ys)
            p1.extend(probs[:, 1].tolist())
            conf.extend(np.max(probs, axis=1).tolist())

    y_true = np.array(y_true, dtype=np.int64)
    p1 = np.array(p1, dtype=np.float64)
    conf = np.array(conf, dtype=np.float64)
    y_pred = (p1 >= 0.5).astype(np.int64)
    correct = (y_pred == y_true).astype(np.int64)
    err = 1 - correct
    uncertainty = 1.0 - conf

    return {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "roc_auc": safe_auc(y_true, p1),
        "pr_auc": safe_auprc(y_true, p1),
        "ece": compute_ece(conf, correct, n_bins=15),
        "nll": float(log_loss(y_true, np.vstack([1.0 - p1, p1]).T, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p1)),
        "mean_uncertainty": float(uncertainty.mean()),
        "error_rate": float(err.mean()),
        "uncertainty_scores": uncertainty.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--method", default="mc_dropout", choices=["confidence", "mc_dropout", "deep_ensemble"])
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--ensemble_size", type=int, default=3)
    p.add_argument("--ensemble_run_ids", type=str, default="", help="Comma-separated run IDs for deep ensemble")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--shifts", type=str, default="id,blur,jpeg,color,noise")
    p.add_argument("--severities", type=str, default="1,3,5")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def _resolve_ckpts(run_id: str, ensemble_run_ids: list[str], ensemble_size: int) -> list[Path]:
    ckpts: list[Path] = []
    if ensemble_run_ids:
        for rid in ensemble_run_ids:
            p = REPO_ROOT / "checkpoints" / rid / "best.pt"
            if p.exists():
                ckpts.append(p)
        return ckpts
    if run_id:
        c = REPO_ROOT / "checkpoints" / run_id / "best.pt"
        if c.exists():
            return [c]
    base = REPO_ROOT / "checkpoints"
    if base.exists():
        runs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
        for d in runs:
            c = d / "best.pt"
            if c.exists():
                ckpts.append(c)
            if len(ckpts) >= max(1, ensemble_size):
                break
    if not ckpts:
        c = REPO_ROOT / "checkpoints" / "best.pt"
        if c.exists():
            ckpts = [c]
    return ckpts


def load_models(method: str, run_id: str, ensemble_run_ids: list[str], ensemble_size: int):
    cfg = get_config()
    def _new_model():
        m, _, _ = load_hf_image_classifier(
            model_id=cfg["model"]["model_id"],
            num_labels=2,
            dropout=cfg["model"].get("dropout", 0.1),
        )
        return m
    models = []
    if method == "deep_ensemble":
        ckpts = _resolve_ckpts(run_id, ensemble_run_ids, ensemble_size)
        for ckpt in ckpts:
            m = _new_model()
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                m.load_state_dict(state["model_state_dict"], strict=False)
            models.append(m)
        if models:
            return models
    m = _new_model()
    ckpts = _resolve_ckpts(run_id, [], 1)
    if ckpts:
        state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            m.load_state_dict(state["model_state_dict"], strict=False)
    models.append(m)
    return models


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    cfg = get_config()
    image_size = tuple(cfg["data"]["image_size"])
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ensemble_run_ids = [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()]
    models = load_models(args.method, args.run_id, ensemble_run_ids, max(1, args.ensemble_size))
    models = [m.to(device) for m in models]

    shifts = [s.strip() for s in args.shifts.split(",") if s.strip()]
    severities = [int(x) for x in args.severities.split(",") if x.strip()]
    if "id" not in shifts:
        shifts = ["id"] + shifts

    results = {}
    id_unc = None
    for shift in shifts:
        sev_list = [0] if shift == "id" else severities
        for sev in sev_list:
            key = f"{shift}_s{sev}"
            res = run_split(
                models=models,
                split=args.split,
                shift=shift,
                severity=sev,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                seed=args.seed + sev + len(shift),
                device=device,
                tfm=tfm,
                method=args.method,
                mc_samples=args.mc_samples,
            )
            results[key] = {k: v for k, v in res.items() if k != "uncertainty_scores"}
            if shift == "id":
                id_unc = np.array(res["uncertainty_scores"], dtype=np.float64)
            else:
                if id_unc is not None:
                    ood_unc = np.array(res["uncertainty_scores"], dtype=np.float64)
                    m = min(len(id_unc), len(ood_unc))
                    y = np.concatenate([np.zeros(m, dtype=np.int64), np.ones(m, dtype=np.int64)])
                    score = np.concatenate([id_unc[:m], ood_unc[:m]])
                    results[key]["ood_detection_auroc"] = safe_auc(y, score)
                    results[key]["ood_detection_auprc"] = safe_auprc(y, score)

    near_keys = []
    far_keys = []
    for k in results.keys():
        if k.startswith("id_"):
            continue
        try:
            sev = int(k.split("_s")[-1])
        except Exception:
            continue
        # For this PCAM pipeline, treat severity <=3 as near-OOD and >=4 as far-OOD.
        if sev <= 3:
            near_keys.append(k)
        elif sev >= 4:
            far_keys.append(k)

    def _avg(keys, field):
        vals = [results[x].get(field) for x in keys if results[x].get(field) is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    grouped = {
        "near_ood": {
            "conditions": near_keys,
            "n_conditions": len(near_keys),
            "mean_ood_auroc": _avg(near_keys, "ood_detection_auroc"),
            "mean_ood_auprc": _avg(near_keys, "ood_detection_auprc"),
            "mean_accuracy": _avg(near_keys, "accuracy"),
            "mean_ece": _avg(near_keys, "ece"),
        },
        "far_ood": {
            "conditions": far_keys,
            "n_conditions": len(far_keys),
            "mean_ood_auroc": _avg(far_keys, "ood_detection_auroc"),
            "mean_ood_auprc": _avg(far_keys, "ood_detection_auprc"),
            "mean_accuracy": _avg(far_keys, "accuracy"),
            "mean_ece": _avg(far_keys, "ece"),
        },
    }

    out = {
        "config": {
            "split": args.split,
            "method": args.method,
            "mc_samples": int(args.mc_samples if args.method == "mc_dropout" else 1),
            "ensemble_size": len(models) if args.method == "deep_ensemble" else 1,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "run_id": args.run_id or None,
            "shifts": shifts,
            "severities": severities,
        },
        "results": results,
        "grouped_summary": grouped,
        "literature_alignment": {
            "references": [
                "Linmans et al. (2022) DOI: 10.1016/j.media.2022.102655",
                "Thagaard et al. (2020) DOI: 10.1007/978-3-030-59710-8_80",
            ],
            "note": "Near-vs-far OOD grouping is reported to reflect literature observations that method ranking may differ by shift regime.",
        },
    }

    out_path = Path(args.out) if args.out else (REPO_ROOT / "evaluation" / f"shift_ood_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
