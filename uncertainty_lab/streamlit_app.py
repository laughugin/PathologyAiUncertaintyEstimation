"""
Uncertainty Lab — Streamlit UI (local).

Workflow: Home → Setup → Run → Results

Run from repository root:
  streamlit run uncertainty_lab/streamlit_app.py
"""
from __future__ import annotations

import copy
import io
import json
import sys
import zipfile
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
import yaml

from uncertainty_lab.streamlit_preview import (
    fetch_hf_model_info,
    preview_binary_folder,
    preview_csv,
    preview_pcam,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _nav_index() -> int:
    return int(st.session_state.get("_nav", 0))


def _set_nav(i: int) -> None:
    st.session_state["_nav"] = max(0, min(3, i))


def init_session_defaults() -> None:
    """One-time defaults for widget-backed session keys."""
    d = {
        "ulab_ds_type": "pcam",
        "ulab_ds_root": "data/raw",
        "ulab_eval_split": "test",
        "ulab_max_samples": 512,
        "ulab_val_frac": 0.15,
        "ulab_test_frac": 0.15,
        "ulab_csv_path": "data/manifest.csv",
        "ulab_path_col": "path",
        "ulab_label_col": "label",
        "ulab_csv_base": "",
        "ulab_model_source": "huggingface",
        "ulab_model_id": "google/vit-base-patch16-224",
        "ulab_local_ckpt": "",
        "ulab_ens_paths": "",
        "ulab_mc_samples": 30,
        "ulab_pipeline_mode": "evaluate",
        "ulab_run_name": "ui_run",
        "ulab_plot_rel": True,
        "ulab_plot_rc": True,
        "ulab_plot_hist": True,
        "ulab_top_k": 12,
        "ulab_uq_conf": True,
        "ulab_uq_mc": False,
        "ulab_uq_ens": False,
        "ulab_show_acc": True,
        "ulab_show_auc": True,
        "ulab_show_ece": True,
        "ulab_show_brier": True,
        "ulab_show_rc": True,
        "ulab_log": "",
        "ulab_seed": 42,
        "ulab_batch_size": 32,
        "ulab_num_workers": 0,
        "ulab_device": "auto",
        "ulab_ckpt_upload_path": "",
        "ulab_model_meta": None,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_example_pcam_config() -> None:
    """Preset for thesis-style PCAM exploration."""
    st.session_state["ulab_ds_type"] = "pcam"
    st.session_state["ulab_ds_root"] = "data/raw"
    st.session_state["ulab_eval_split"] = "test"
    st.session_state["ulab_max_samples"] = 256
    st.session_state["ulab_model_id"] = "google/vit-base-patch16-224"
    st.session_state["ulab_uq_conf"] = True
    st.session_state["ulab_uq_mc"] = False
    st.session_state["ulab_uq_ens"] = False
    st.session_state["ulab_pipeline_mode"] = "evaluate"
    st.session_state["ulab_run_name"] = "pcam_demo"


def collect_config() -> dict:
    from uncertainty_lab.config import deep_merge, load_config

    base = load_config(None, repo_root=REPO_ROOT)
    s = st.session_state

    methods: list[str] = []
    if s.get("ulab_uq_conf", True):
        methods.append("confidence")
    if s.get("ulab_uq_mc"):
        methods.append("mc_dropout")
    if s.get("ulab_uq_ens"):
        methods.append("deep_ensemble")
    if not methods:
        methods = ["confidence"]

    ds = {
        "type": s["ulab_ds_type"],
        "root": s["ulab_ds_root"],
        "eval_split": s["ulab_eval_split"],
        "max_eval_samples": int(s["ulab_max_samples"]),
        "val_fraction": float(s["ulab_val_frac"]),
        "test_fraction": float(s["ulab_test_frac"]),
        "batch_size": int(s.get("ulab_batch_size", 32)),
        "num_workers": int(s.get("ulab_num_workers", 0)),
    }
    if ds["type"] == "csv":
        ds["csv_path"] = s["ulab_csv_path"]
        ds["path_column"] = s["ulab_path_col"]
        ds["label_column"] = s["ulab_label_col"]
        cb = str(s.get("ulab_csv_base") or "").strip()
        ds["csv_base_dir"] = cb if cb else None

    md = {
        "source": s["ulab_model_source"],
        "model_id": s["ulab_model_id"],
        "num_labels": 2,
        "dropout": 0.1,
        "local_checkpoint": None,
        "ensemble_checkpoints": [],
    }
    up = str(s.get("ulab_ckpt_upload_path") or "").strip()
    ck = str(s.get("ulab_local_ckpt") or "").strip()
    if up and Path(up).is_file():
        md["local_checkpoint"] = str(Path(up).resolve())
    elif s["ulab_model_source"] == "local" and ck:
        md["local_checkpoint"] = ck
    elif ck:
        md["local_checkpoint"] = ck

    if "deep_ensemble" in methods:
        md["ensemble_checkpoints"] = [ln.strip() for ln in str(s.get("ulab_ens_paths") or "").splitlines() if ln.strip()]

    unc = {
        "method": methods[0],
        "mc_dropout_n_samples": int(s["ulab_mc_samples"]),
    }

    ev = {
        "top_k_uncertain": int(s["ulab_top_k"]),
        "plots": {
            "reliability": bool(s.get("ulab_plot_rel", True)),
            "risk_coverage": bool(s.get("ulab_plot_rc", True)),
            "uncertainty_histograms": bool(s.get("ulab_plot_hist", True)),
        },
    }

    run = {
        "name": str(s.get("ulab_run_name") or "ui_run"),
        "repo_root": str(REPO_ROOT),
        "output_base": str(REPO_ROOT / "runs"),
        "device": str(s.get("ulab_device", "auto")).lower().strip(),
    }

    patch = {
        "seed": int(s.get("ulab_seed", 42)),
        "dataset": ds,
        "model": md,
        "uncertainty": unc,
        "evaluation": ev,
        "pipeline": {"mode": s["ulab_pipeline_mode"]},
        "run": run,
        "_ulab_methods": methods,
    }
    return deep_merge(base, patch)


def render_nav_bar() -> None:
    idx = _nav_index()
    st.markdown("### Uncertainty Lab")
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        if st.button("Home", use_container_width=True, type="primary" if idx == 0 else "secondary"):
            _set_nav(0)
            st.rerun()
    with c1:
        if st.button("Setup", use_container_width=True, type="primary" if idx == 1 else "secondary"):
            _set_nav(1)
            st.rerun()
    with c2:
        if st.button("Run", use_container_width=True, type="primary" if idx == 2 else "secondary"):
            _set_nav(2)
            st.rerun()
    with c3:
        if st.button("Results", use_container_width=True, type="primary" if idx == 3 else "secondary"):
            _set_nav(3)
            st.rerun()
    st.divider()


def render_home() -> None:
    st.markdown("## Uncertainty Lab for Vision Models")
    st.markdown("Evaluate how confident your model is — and when it is wrong.")

    st.markdown("")
    if st.button("Start New Run", type="primary", key="home_start_run"):
        _set_nav(1)
        st.rerun()

    st.markdown("---")
    st.markdown("**What this tool does**")
    st.markdown(
        """
- Load your own dataset (folder, CSV, or PCAM)  
- Use a pretrained Hugging Face model or your own checkpoint  
- Run uncertainty estimation (confidence, MC Dropout, deep ensemble)  
- Analyze calibration and reliability (ECE, Brier, risk–coverage)  
"""
    )

    st.markdown("---")
    st.markdown("**Workflow**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Setup**  \nUpload data, choose model, pick methods & metrics.")
    with c2:
        st.markdown("**2. Run**  \nExecute evaluation on your machine.")
    with c3:
        st.markdown("**3. Results**  \nInspect plots, calibration, and uncertain samples.")

    st.markdown("---")
    st.markdown("**Try a demo**")
    st.caption("Use Patch Camelyon (PCAM) if you have already downloaded it under `data/raw`.")
    if st.button("Load example config (PCAM)", key="home_example_pcam"):
        apply_example_pcam_config()
        _set_nav(1)
        st.rerun()

    st.markdown("---")
    st.markdown("**Why uncertainty matters**")
    st.info(
        "A model can be accurate but poorly calibrated. Uncertainty estimation helps "
        "identify when predictions should not be trusted — central to safe use in practice."
    )

    res = st.session_state.get("last_result")
    bench = st.session_state.get("last_benchmark")
    if res:
        st.markdown("---")
        st.markdown("**Last run**")
        met = res.get("metrics") or {}
        perf = met.get("predictive_performance", {})
        cal = met.get("calibration", {})
        mcols = st.columns(4)
        mcols[0].metric("Accuracy", f"{perf.get('accuracy', 0):.4f}" if perf.get("accuracy") is not None else "—")
        mcols[1].metric("ECE", f"{cal.get('ece', 0):.4f}" if cal.get("ece") is not None else "—")
        roc = perf.get("roc_auc")
        mcols[2].metric("AUROC", f"{roc:.4f}" if roc is not None else "—")
        method = met.get("uncertainty_method") or res.get("uncertainty_method") or "—"
        mcols[3].metric("Method", str(method)[:24])
        if st.button("View Results", key="home_view_results"):
            _set_nav(3)
            st.rerun()
    elif bench:
        st.markdown("---")
        st.markdown("**Last benchmark**")
        rows = bench.get("rows") or []
        if rows:
            import pandas as pd

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if st.button("View Results", key="home_view_bench_results"):
            _set_nav(3)
            st.rerun()


def _resolve_data_path(p: str) -> Path:
    path = Path(p.strip())
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _dataset_preview() -> dict:
    s = st.session_state
    t = s.get("ulab_ds_type", "pcam")
    try:
        if t == "folder":
            return preview_binary_folder(_resolve_data_path(str(s.get("ulab_ds_root", ""))))
        if t == "csv":
            cp = _resolve_data_path(str(s.get("ulab_csv_path", "data/manifest.csv")))
            bd_raw = str(s.get("ulab_csv_base") or "").strip()
            bd = _resolve_data_path(bd_raw) if bd_raw else None
            return preview_csv(cp, s["ulab_path_col"], s["ulab_label_col"], bd)
        if t == "pcam":
            return preview_pcam(_resolve_data_path(str(s.get("ulab_ds_root", "data/raw"))), str(s.get("ulab_eval_split", "test")))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Unknown dataset type."}


def _validate_before_proceed() -> tuple[bool, str]:
    s = st.session_state
    pr = _dataset_preview()
    if not pr["ok"]:
        return False, pr.get("error", "Dataset validation failed.")

    if s.get("ulab_ds_type") == "csv":
        ncls = len(pr.get("class_counts") or {})
        if ncls != 2:
            return False, f"Binary classification requires exactly **two** label values in the CSV; found {ncls}."

    if not (s.get("ulab_uq_conf") or s.get("ulab_uq_mc") or s.get("ulab_uq_ens")):
        return False, "Select at least one uncertainty method."

    if s.get("ulab_uq_ens"):
        lines = [ln.strip() for ln in str(s.get("ulab_ens_paths") or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            return False, "Deep ensemble needs at least two checkpoint paths (one per line)."

    if s.get("ulab_model_source") == "local":
        up = str(s.get("ulab_ckpt_upload_path") or "").strip()
        ck = str(s.get("ulab_local_ckpt") or "").strip()
        has = (up and Path(up).is_file()) or bool(ck)
        if not has:
            return False, "Local model source requires an uploaded `.pt` file or a checkpoint path."

    return True, ""


def _friendly_run_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    low = msg.lower()
    if "need exactly" in low and "two" in low:
        return f"Invalid dataset structure: {msg}"
    if "not a directory" in low or "no such file" in low or "filenotfound" in low:
        return f"Path or file not found: {msg}"
    if "cuda" in low and ("out of memory" in low or "oom" in low):
        return f"GPU out of memory. Try a smaller batch size or CPU in Setup → Advanced. ({msg})"
    if "huggingface" in low or "repository not found" in low or "401" in msg:
        return f"Model loading failed (check Hugging Face ID and network): {msg}"
    return msg


def _dataset_type_label(t: str) -> str:
    return {"pcam": "PCAM", "folder": "Custom folder", "csv": "CSV manifest"}.get(t, t)


def _method_summary_line(method_id: str, cfg: dict) -> str:
    u = cfg.get("uncertainty", {}) or {}
    mc = int(u.get("mc_dropout_n_samples", 30))
    if method_id == "confidence":
        return "Confidence (baseline)"
    if method_id == "mc_dropout":
        return f"MC Dropout ({mc} runs)"
    if method_id == "deep_ensemble":
        n = len((cfg.get("model") or {}).get("ensemble_checkpoints") or [])
        return f"Deep ensemble ({n} checkpoints)"
    return method_id


def _metrics_summary_labels() -> str:
    s = st.session_state
    parts: list[str] = []
    if s.get("ulab_show_acc", True):
        parts.append("Accuracy")
    if s.get("ulab_show_auc", True):
        parts.append("AUROC")
    if s.get("ulab_show_ece", True):
        parts.append("ECE")
    if s.get("ulab_show_brier", True):
        parts.append("Brier")
    if s.get("ulab_show_rc", True):
        parts.append("Risk–coverage")
    return ", ".join(parts) if parts else "(none selected for Results table)"


def _render_run_completion_section(*, just_finished: bool) -> None:
    """Single-run metrics, benchmark table, and View Results (persists when revisiting Run)."""
    res = st.session_state.get("last_result")
    bench = st.session_state.get("last_benchmark")
    if bench and bench.get("rows"):
        st.markdown("### Benchmark summary" if just_finished else "### Latest benchmark")
        import pandas as pd

        st.dataframe(pd.DataFrame(bench["rows"]), use_container_width=True)
    elif res and res.get("metrics"):
        met = res["metrics"]
        perf = met.get("predictive_performance", {})
        cal = met.get("calibration", {})
        st.markdown("### Run complete" if just_finished else "### Latest completed run")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Accuracy",
            f"{perf['accuracy']:.4f}" if perf.get("accuracy") is not None else "—",
        )
        c2.metric("ECE", f"{cal['ece']:.4f}" if cal.get("ece") is not None else "—")
        c3.metric(
            "Brier score",
            f"{cal['brier']:.4f}" if cal.get("brier") is not None else "—",
        )
    else:
        return

    if st.button("View Results", type="primary", key="run_tab_view_results"):
        _set_nav(3)
        st.rerun()


def _render_run_configuration_summary(cfg: dict, methods: list[str]) -> None:
    ds = cfg.get("dataset", {})
    md = cfg.get("model", {})
    stats = st.session_state.get("ulab_run_dataset_stats") or {}
    dtype = str(ds.get("type", ""))
    n_line = ""
    if stats.get("n_total") is not None:
        n_line = f" · **{int(stats['n_total'])}** samples (from Setup preview)"
    else:
        cap = ds.get("max_eval_samples")
        if cap is not None:
            n_line = f" · up to **{cap}** eval samples"

    root = ds.get("root", "")
    csv_p = ds.get("csv_path", "")
    loc = f"`{csv_p}`" if dtype == "csv" and csv_p else f"`{root}`"

    st.markdown("### Configuration summary")
    lines = [
        f"- **Dataset:** {_dataset_type_label(dtype)}{n_line} · {loc}",
        f"- **Model:** `{md.get('model_id', '')}`"
        + (f" · checkpoint loaded" if md.get("local_checkpoint") else ""),
        f"- **Pipeline:** `{cfg.get('pipeline', {}).get('mode', 'evaluate')}`"
        f" · device `{cfg.get('run', {}).get('device', 'auto')}`",
    ]
    if len(methods) > 1:
        lines.append(f"- **Methods (benchmark):** " + "; ".join(_method_summary_line(m, cfg) for m in methods))
    else:
        lines.append(f"- **Method:** {_method_summary_line(methods[0], cfg)}")
    lines.append(f"- **Metrics highlighted in Results:** {_metrics_summary_labels()}")
    st.markdown("\n".join(lines))


def render_setup() -> None:
    st.markdown("## Setup")
    st.caption("Define the full experiment in one place: dataset → model → methods → metrics → Run.")

    # ----- 1. Dataset -----
    st.markdown("### 1. Dataset")
    st.caption("Binary classification only: two classes (two folders, or 0/1 labels in CSV).")
    st.selectbox("Dataset source", ["pcam", "folder", "csv"], key="ulab_ds_type")

    zip_up = st.file_uploader(
        "Upload dataset as ZIP (optional)",
        type=["zip"],
        help="Archive should contain one root folder with **two** class subfolders, or two subfolders at top level.",
    )
    if zip_up is not None:
        tmp = Path(mkdtemp(prefix="ulab_zip_"))
        z = zipfile.ZipFile(io.BytesIO(zip_up.read()))
        z.extractall(tmp)
        subs = [p for p in tmp.iterdir() if p.is_dir()]
        root = subs[0] if len(subs) == 1 else tmp
        st.session_state["ulab_ds_type"] = "folder"
        st.session_state["ulab_ds_root"] = str(root)
        st.success(f"ZIP extracted. Using folder: `{root}`")
        st.rerun()

    st.text_input(
        "Local path (PCAM root, folder dataset, or CSV file)",
        key="ulab_ds_root",
        help="Examples: `data/raw` for PCAM; `/path/to/data` with `class0/` & `class1/` for folder.",
    )

    st.selectbox("Eval split (PCAM / folder splits)", ["train", "val", "test"], key="ulab_eval_split")
    st.number_input("Max samples to evaluate", min_value=16, max_value=100000, key="ulab_max_samples")

    if st.session_state.get("ulab_ds_type") in ("folder", "csv"):
        st.slider("Validation fraction", 0.05, 0.4, key="ulab_val_frac")
        st.slider("Test fraction", 0.05, 0.4, key="ulab_test_frac")

    if st.session_state.get("ulab_ds_type") == "csv":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("CSV path (if different from field above)", key="ulab_csv_path")
        with c2:
            st.text_input("Base directory for relative image paths", key="ulab_csv_base")
        c3, c4 = st.columns(2)
        with c3:
            st.text_input("Column: image path", key="ulab_path_col")
        with c4:
            st.text_input("Column: label (0 or 1)", key="ulab_label_col")

    st.markdown("**Preview**")
    info = _dataset_preview()
    if info.get("ok"):
        m1, m2 = st.columns(2)
        with m1:
            m1.metric("Total samples", int(info["n_total"]))
        with m2:
            st.markdown("**Class distribution**")
            for cls, cnt in sorted(info.get("class_counts", {}).items()):
                st.caption(f"- `{cls}`: **{cnt}**")
        if info.get("note"):
            st.caption(info["note"])
        prev_paths = info.get("preview_paths") or []
        pil_imgs = info.get("previews_pil") or []
        if prev_paths:
            cols = st.columns(min(4, len(prev_paths)))
            for i, p in enumerate(prev_paths[:4]):
                with cols[i % len(cols)]:
                    st.image(str(p), use_container_width=True)
        elif pil_imgs:
            cols = st.columns(min(4, len(pil_imgs)))
            for i, im in enumerate(pil_imgs[:4]):
                with cols[i % len(cols)]:
                    st.image(im, use_container_width=True)
    else:
        st.warning(info.get("error", "Could not load dataset preview."))

    st.divider()

    # ----- 2. Model -----
    st.markdown("### 2. Model")
    st.radio("Load model from", ["huggingface", "local"], horizontal=True, key="ulab_model_source")
    st.text_input("Hugging Face model ID (architecture + weights)", key="ulab_model_id")

    ckpt_bytes = st.file_uploader("Or upload checkpoint (.pt / .pth)", type=["pt", "pth"], key="ulab_ckpt_uploader")
    if ckpt_bytes is not None:
        tdir = Path(st.session_state.get("ulab_ckpt_tmp") or mkdtemp(prefix="ulab_ckpt_"))
        st.session_state["ulab_ckpt_tmp"] = str(tdir)
        dest = tdir / "weights.pt"
        dest.write_bytes(ckpt_bytes.getvalue())
        st.session_state["ulab_ckpt_upload_path"] = str(dest)
        st.caption(f"Using uploaded checkpoint: `{dest}`")

    st.text_input("Optional: path to fine-tuned `.pt` (instead of or in addition to upload)", key="ulab_local_ckpt")

    b1, _ = st.columns([1, 3])
    with b1:
        if st.button("Look up model info", key="setup_model_info_btn"):
            st.session_state["ulab_model_meta"] = fetch_hf_model_info(str(st.session_state.get("ulab_model_id", "")))
    meta = st.session_state.get("ulab_model_meta")
    if isinstance(meta, dict) and meta.get("ok"):
        st.success(
            f"**{meta.get('name', '')}** · labels: `{meta.get('num_labels')}` · "
            f"input size: `{meta.get('input_size')}`"
        )
    elif isinstance(meta, dict) and meta.get("error"):
        st.caption(f"Model info: {meta['error']}")

    st.divider()

    # ----- 3. Uncertainty -----
    st.markdown("### 3. Uncertainty methods")
    st.checkbox("Confidence (1 − max probability) — baseline", key="ulab_uq_conf")
    st.checkbox("MC Dropout", key="ulab_uq_mc")
    if st.session_state.get("ulab_uq_mc"):
        st.slider("MC Dropout — stochastic forward passes", 5, 50, key="ulab_mc_samples")
    st.checkbox("Deep ensemble", key="ulab_uq_ens")
    if st.session_state.get("ulab_uq_ens"):
        st.text_area(
            "Checkpoint paths, one per line (same architecture as HF model ID)",
            key="ulab_ens_paths",
            height=100,
        )

    st.divider()

    # ----- 4. Metrics (display + plots; all metrics computed in pipeline) -----
    st.markdown("### 4. Metrics & plots")
    st.caption("All listed metrics are computed internally; checkboxes control what **Results** highlights and which plots are saved.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Show in Results table**")
        st.checkbox("Accuracy", key="ulab_show_acc")
        st.checkbox("AUROC", key="ulab_show_auc")
        st.checkbox("ECE (calibration)", key="ulab_show_ece")
    with c2:
        st.markdown("**Show in Results table**")
        st.checkbox("Brier score", key="ulab_show_brier")
        st.checkbox("Risk–coverage (AURC)", key="ulab_show_rc")
    st.markdown("**Generate plot files**")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.checkbox("Reliability diagram", key="ulab_plot_rel")
    with pc2:
        st.checkbox("Risk–coverage curve", key="ulab_plot_rc")
    with pc3:
        st.checkbox("Uncertainty histograms", key="ulab_plot_hist")

    # ----- 5. Advanced -----
    with st.expander("Advanced settings", expanded=False):
        st.number_input("Random seed", 0, 2**31 - 1, key="ulab_seed")
        st.number_input("Batch size", 1, 512, key="ulab_batch_size")
        st.number_input("DataLoader workers", 0, 16, key="ulab_num_workers")
        st.selectbox("Device", ["auto", "cpu", "cuda"], key="ulab_device")
        st.selectbox("Pipeline mode", ["evaluate", "train", "train_evaluate"], key="ulab_pipeline_mode")
        st.text_input("Run name (optional)", key="ulab_run_name")
        st.number_input("Top-K uncertain samples (for Results listing)", 0, 200, key="ulab_top_k")

    st.divider()

    # ----- 6. Config preview -----
    st.markdown("### Config preview")
    full_cfg = copy.deepcopy(collect_config())
    methods = list(full_cfg.pop("_ulab_methods", []))
    preview_doc = {
        "seed": full_cfg.get("seed"),
        "dataset": {k: v for k, v in full_cfg.get("dataset", {}).items() if not str(k).startswith("_")},
        "model": {
            "model_id": full_cfg.get("model", {}).get("model_id"),
            "source": full_cfg.get("model", {}).get("source"),
            "local_checkpoint": full_cfg.get("model", {}).get("local_checkpoint"),
        },
        "uncertainty_methods": methods,
        "mc_dropout_passes": full_cfg.get("uncertainty", {}).get("mc_dropout_n_samples"),
        "pipeline_mode": full_cfg.get("pipeline", {}).get("mode"),
        "run": {"device": full_cfg.get("run", {}).get("device"), "name": full_cfg.get("run", {}).get("name")},
    }
    st.code(yaml.safe_dump(preview_doc, default_flow_style=False, sort_keys=False), language="yaml")

    # ----- 7. Proceed -----
    st.markdown("### Next step")
    if st.button("Proceed to Run", type="primary", key="setup_proceed_run"):
        ok, err = _validate_before_proceed()
        if not ok:
            st.error(err)
        else:
            cfg = collect_config()
            st.session_state["ulab_pending_config"] = copy.deepcopy(cfg)
            pv = _dataset_preview()
            if pv.get("ok"):
                st.session_state["ulab_run_dataset_stats"] = {
                    "n_total": int(pv["n_total"]),
                    "ds_type": st.session_state.get("ulab_ds_type", ""),
                }
            _set_nav(2)
            st.rerun()


def render_run() -> None:
    st.markdown("## Run")
    st.caption("Execute the pipeline and monitor progress. Change settings on **Setup** — not here.")

    pending = st.session_state.get("ulab_pending_config")
    if pending is not None:
        st.caption("Configuration is frozen from **Setup → Proceed to Run**.")
        cfg = copy.deepcopy(pending)
    else:
        st.info(
            "You have not clicked **Proceed to Run** on Setup yet. "
            "This tab will use your current Setup widget values."
        )
        cfg = collect_config()

    methods = list(cfg.pop("_ulab_methods", ["confidence"]))
    st.session_state["_pending_methods"] = methods

    _render_run_configuration_summary(cfg, methods)

    if len(methods) > 1:
        st.warning(
            f"Multiple methods: **{', '.join(methods)}**. "
            "The lab will run a **benchmark** (one evaluation folder per method)."
        )

    st.text_input(
        "Run name (optional)",
        key="ulab_run_name",
        help="Used for the output folder under `runs/`. You can also set this in Setup → Advanced.",
    )
    cfg["run"]["name"] = str(st.session_state.get("ulab_run_name") or "ui_run").strip() or "ui_run"

    status_ph = st.empty()
    progress_ph = st.empty()

    run_clicked = st.button("Run evaluation", type="primary", key="run_eval_primary_btn")
    run_succeeded = False

    if run_clicked:
        from uncertainty_lab.pipeline.run import run_benchmark, run_pipeline

        log_lines: list[str] = []

        def progress_cb(message: str, fraction: float) -> None:
            status_ph.markdown(f"**{message}**")
            progress_ph.progress(min(1.0, max(0.0, fraction)))

        def log_cb(line: str) -> None:
            log_lines.append(line)

        progress_cb("Starting…", 0.0)

        try:
            if len(methods) > 1:
                bench_cfg = copy.deepcopy(cfg)
                bench_cfg["benchmark"] = {"methods": methods}
                bench_cfg.setdefault("uncertainty", {})["method"] = methods[0]
                out = run_benchmark(bench_cfg, progress_callback=progress_cb, log_callback=log_cb)
                log_lines.append(json.dumps(out, indent=2))
                st.session_state["last_benchmark"] = out
                st.session_state["last_result"] = None
                st.session_state["ulab_log"] = "\n".join(log_lines)
                st.session_state["ulab_last_run_error"] = None

                progress_cb("Benchmark complete.", 1.0)
                st.success("**Benchmark completed successfully.**")
                run_succeeded = True
            else:
                run_cfg = copy.deepcopy(cfg)
                run_cfg.setdefault("uncertainty", {})["method"] = methods[0]
                result = run_pipeline(run_cfg, progress_callback=progress_cb, log_callback=log_cb)
                log_lines.append(f"run_dir: {result.get('run_dir')}")
                st.session_state["ulab_log"] = "\n".join(log_lines)
                st.session_state["ulab_last_run_error"] = None

                if result.get("status") == "trained":
                    progress_cb("Training finished (evaluate-only mode not run).", 1.0)
                    st.success(
                        "**Training finished.** Pipeline mode was **train** only — no evaluation metrics. "
                        "Switch to **train_evaluate** or **evaluate** on Setup for metrics and plots."
                    )
                    st.session_state["last_result"] = None
                    st.session_state["last_benchmark"] = None
                    run_succeeded = True
                else:
                    st.session_state["last_result"] = result
                    st.session_state["last_benchmark"] = None
                    progress_cb("Run complete.", 1.0)
                    st.success("**Run completed successfully.**")
                    run_succeeded = True

        except Exception as e:
            progress_ph.progress(1.0)
            status_ph.markdown("")
            friendly = _friendly_run_error(e)
            st.error(f"**Run failed:** {friendly}")
            st.session_state["ulab_last_run_error"] = friendly
            log_lines.append(f"ERROR: {e!r}")
            st.session_state["ulab_log"] = "\n".join(log_lines)

    err_prev = st.session_state.get("ulab_last_run_error")
    if err_prev and not run_clicked:
        st.warning(f"Last run error: {err_prev}")

    with st.expander("Show logs", expanded=False):
        st.code(st.session_state.get("ulab_log") or "(no logs yet)", language="text")

    res = st.session_state.get("last_result")
    bench = st.session_state.get("last_benchmark")
    has_eval_output = (res and res.get("metrics")) or (bench and bench.get("rows"))
    if has_eval_output and (run_succeeded or not run_clicked):
        _render_run_completion_section(just_finished=bool(run_succeeded))


def render_results() -> None:
    st.markdown("## Results")
    bench = st.session_state.get("last_benchmark")
    res = st.session_state.get("last_result")

    if bench and not res:
        st.markdown("### Benchmark comparison")
        rows = bench.get("rows") or []
        if rows:
            import pandas as pd

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(f"Comparison file: `{bench.get('comparison_path', '')}`")
        return

    if not res:
        st.info("No results yet. Run an evaluation from the **Run** tab, or load an example on **Home**.")
        return

    met = res.get("metrics") or {}
    perf = met.get("predictive_performance", {})
    cal = met.get("calibration", {})
    sel = met.get("selective_prediction", {})

    st.markdown("### Metrics summary")
    s = st.session_state
    rows = []
    if s.get("ulab_show_acc", True) and perf.get("accuracy") is not None:
        rows.append({"Metric": "Accuracy", "Value": f"{perf['accuracy']:.6f}"})
    if s.get("ulab_show_auc", True) and perf.get("roc_auc") is not None:
        rows.append({"Metric": "AUROC", "Value": f"{perf['roc_auc']:.6f}"})
    if s.get("ulab_show_ece", True) and cal.get("ece") is not None:
        rows.append({"Metric": "ECE", "Value": f"{cal['ece']:.6f}"})
    if s.get("ulab_show_brier", True) and cal.get("brier") is not None:
        rows.append({"Metric": "Brier", "Value": f"{cal['brier']:.6f}"})
    if s.get("ulab_show_rc", True) and sel.get("aurc") is not None:
        rows.append({"Metric": "AURC (risk–coverage)", "Value": f"{sel['aurc']:.6f}"})
    if rows:
        import pandas as pd

        st.table(pd.DataFrame(rows))
    else:
        st.write("Enable metric checkboxes under Setup → Metrics & plots.")

    st.markdown("### Plots")
    rd = Path(res["run_dir"])
    for name, title in (
        ("reliability.png", "Reliability diagram"),
        ("risk_coverage.png", "Risk–coverage"),
        ("uncertainty_histograms.png", "Uncertainty histogram"),
    ):
        p = rd / "plots" / name
        if p.is_file():
            st.image(str(p), caption=title)

    st.markdown("### Sample analysis")
    hi = met.get("highlighted_samples")
    if hi:
        st.write("Most uncertain / error-prone indices (from Setup Top-K).")
        st.json(hi)
    else:
        st.caption("Increase “Top-K uncertain indices” in Setup to list samples here.")

    mp = res.get("metrics_path")
    if mp:
        st.download_button("Download metrics.json", json.dumps(met, indent=2), file_name="metrics.json")


def main() -> None:
    st.set_page_config(page_title="Uncertainty Lab", layout="wide", initial_sidebar_state="collapsed")
    init_session_defaults()
    render_nav_bar()

    idx = _nav_index()
    if idx == 0:
        render_home()
    elif idx == 1:
        render_setup()
    elif idx == 2:
        render_run()
    else:
        render_results()


if __name__ == "__main__":
    main()
