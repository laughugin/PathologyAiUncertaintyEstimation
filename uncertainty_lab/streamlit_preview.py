"""Dataset inspection helpers for the Streamlit Setup tab (binary classification only)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def preview_binary_folder(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {"ok": False, "error": f"Not a directory: {root}"}
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(subdirs) != 2:
        return {
            "ok": False,
            "error": f"Need exactly **two** class subfolders; found {len(subdirs)}.",
        }
    counts: dict[str, int] = {}
    preview_paths: list[Path] = []
    for sd in subdirs:
        imgs = sorted([p for p in sd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT])
        counts[sd.name] = len(imgs)
        for p in imgs[:2]:
            if len(preview_paths) < 4:
                preview_paths.append(p)
    n_total = sum(counts.values())
    if n_total == 0:
        return {"ok": False, "error": "No images found in the two class folders."}
    return {"ok": True, "n_total": n_total, "class_counts": counts, "preview_paths": preview_paths, "previews_pil": []}


def preview_csv(
    csv_path: Path,
    path_column: str,
    label_column: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    try:
        from uncertainty_lab.data.csv_dataset import read_csv_samples

        samples = read_csv_samples(csv_path, path_column, label_column, base_dir)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    ctr = Counter(y for _, y in samples)
    counts = {str(k): ctr[k] for k in sorted(ctr)}
    preview_paths = [p for p, _ in samples[:4]]
    return {
        "ok": True,
        "n_total": len(samples),
        "class_counts": counts,
        "preview_paths": preview_paths,
        "previews_pil": [],
    }


def preview_pcam(root: Path, split: str) -> dict[str, Any]:
    try:
        from torchvision.datasets import PCAM

        ds = PCAM(root=str(root), split=split, download=False)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    n = len(ds)
    if n == 0:
        return {"ok": False, "error": "PCAM split is empty."}
    take = min(3000, n)
    ctr: Counter[int] = Counter()
    for i in range(take):
        _, y = ds[i]
        ctr[int(y)] += 1
    counts = {f"class_{k}": v for k, v in sorted(ctr.items())}
    previews_pil = [ds[i][0] for i in range(min(4, n))]
    return {
        "ok": True,
        "n_total": n,
        "class_counts": counts,
        "preview_paths": [],
        "previews_pil": previews_pil,
        "note": f"Class counts from first {take} indices (approximate balance).",
    }


def fetch_hf_model_info(model_id: str) -> dict[str, Any]:
    """Lightweight Hub fetch (config + processor size only)."""
    mid = (model_id or "").strip()
    if not mid:
        return {"ok": False, "error": "Enter a model ID."}
    try:
        from transformers import AutoConfig, AutoImageProcessor

        cfg = AutoConfig.from_pretrained(mid)
        proc = AutoImageProcessor.from_pretrained(mid)
        num_labels = getattr(cfg, "num_labels", None)
        id2label = getattr(cfg, "id2label", None)
        name = getattr(cfg, "name_or_path", None) or mid
        size = None
        if hasattr(proc, "size") and proc.size:
            size = (proc.size.get("height"), proc.size.get("width"))
        return {
            "ok": True,
            "model_id": mid,
            "name": name,
            "num_labels": num_labels,
            "id2label": id2label,
            "input_size": size,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
