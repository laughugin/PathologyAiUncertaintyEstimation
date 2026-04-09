"""
Local web UI for the Uncertainty Estimation in Digital Pathology project.
Run: python web/app.py   or   flask --app web.app run
"""
from pathlib import Path
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import yaml
import numpy as np

# Project root (parent of web/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, render_template, send_file, jsonify, request, Response

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["REPO_ROOT"] = REPO_ROOT
app.config["RUN_TIMEOUT"] = 600  # seconds for run tasks

MODEL_CATALOG = [
    {"id": "google/vit-base-patch16-224", "name": "ViT-Base (Patch16-224)", "input_size": [224, 224]},
    {"id": "microsoft/beit-base-patch16-224", "name": "BEiT-Base (Patch16-224)", "input_size": [224, 224]},
    {"id": "facebook/deit-base-patch16-224", "name": "DeiT-Base (Patch16-224)", "input_size": [224, 224]},
]

# Lazy-loaded datasets per split (avoid loading all at startup)
_pcam_cache = {}


def get_pcam(split: str):
    """Get or create PCAM dataset for split. Raises if data missing."""
    global _pcam_cache
    if split not in _pcam_cache:
        from torchvision.datasets import PCAM
        root = REPO_ROOT / "data" / "raw"
        _pcam_cache[split] = PCAM(root=str(root), split=split, download=False)
    return _pcam_cache[split]


def get_dataset(dataset_id: str, split: str):
    """
    Central dataset router.
    Current implementation supports only `pcam` (thesis scaffold for future datasets).
    """
    dataset_id = (dataset_id or "pcam").strip().lower()
    if dataset_id != "pcam":
        raise ValueError(f"Unsupported dataset: {dataset_id}. Only 'pcam' is supported.")
    return get_pcam(split)


def get_config():
    """Load default config YAML."""
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@app.route("/")
def index():
    return render_template("index.html", config=get_config())


@app.route("/project")
def project():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    config = get_config()
    import json
    config_str = json.dumps(config, indent=2) if config else "{}"
    return render_template("project.html", readme=readme, config=config, config_str=config_str)


@app.route("/dataset")
def dataset_page():
    config = get_config()
    return render_template("dataset.html", config=config)


@app.route("/run")
def run_page():
    tasks = {k: v[1] for k, v in RUN_TASKS.items()}
    return render_template("run.html", config=get_config(), run_tasks=tasks)


@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html", config=get_config())


@app.route("/setup")
def setup_page():
    """Single hub: links to Lab, dataset browser, and project docs (nav stays 4 items)."""
    return render_template("setup.html", config=get_config())


@app.route("/api/dataset/<split>/info")
def api_dataset_info(split):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        return jsonify({"split": split, "size": len(ds)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/sample/<int:idx>")
def api_dataset_sample(split, idx):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        img, label = ds[idx]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/label/<int:idx>")
def api_dataset_label(split, idx):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        _, label = ds[idx]
        return jsonify({"idx": idx, "label": int(label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Cached per-split stats (size, class counts) — computed from H5 labels (fast)
_dataset_stats_cache = {}

_PCAM_Y_FILES = {
    "train": "camelyonpatch_level_2_split_train_y.h5",
    "val": "camelyonpatch_level_2_split_valid_y.h5",
    "test": "camelyonpatch_level_2_split_test_y.h5",
}


_pcam_label_cache = {}


def _pcam_labels_for_split(split: str):
    """
    Load PCAM H5 label array once per split (tiny: ~262k labels).
    Returns dict with:
      - y: array shape (N,) with labels {0,1}
      - indices0 / indices1: arrays of indices for each class
      - size: N
    """
    global _pcam_label_cache
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split}")
    if split in _pcam_label_cache:
        return _pcam_label_cache[split]

    import h5py

    base = REPO_ROOT / "data" / "raw" / "pcam"
    path = base / _PCAM_Y_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"PCAM H5 label file not found: {path}")

    with h5py.File(path, "r") as f:
        y = f["y"][:]
        y = y.ravel()

    # Ensure compact dtype for faster indexing; labels are 0/1.
    y = y.astype(np.int64, copy=False)
    indices0 = np.flatnonzero(y == 0).astype(np.int64, copy=False)
    indices1 = np.flatnonzero(y == 1).astype(np.int64, copy=False)

    info = {"y": y, "indices0": indices0, "indices1": indices1, "size": int(y.shape[0])}
    _pcam_label_cache[split] = info
    return info


def _dataset_stats_from_h5(split: str):
    """Read label counts directly from PCAM H5 file (fast, no Python loop)."""
    if split not in ("train", "val", "test"):
        return None
    import h5py
    base = REPO_ROOT / "data" / "raw" / "pcam"
    path = base / _PCAM_Y_FILES[split]
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        y = f["y"][:]  # shape (N, 1, 1, 1)
        y = y.ravel()
        n = len(y)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
    return {
        "split": split,
        "size": n,
        "count_normal": n0,
        "count_metastasis": n1,
        "ratio_normal": round(n0 / n, 4) if n else 0,
        "ratio_metastasis": round(n1 / n, 4) if n else 0,
    }


@app.route("/api/dataset/<split>/stats")
def api_dataset_stats(split):
    """Return dataset statistics: size, count_normal (0), count_metastasis (1). Read from H5 (fast)."""
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    dataset_id = request.args.get("dataset", "pcam")
    if (dataset_id or "").strip().lower() != "pcam":
        return jsonify({"error": "Dataset stats only implemented for 'pcam' in this repo."}), 501
    global _dataset_stats_cache
    cache_key = (dataset_id, split)
    if cache_key not in _dataset_stats_cache:
        try:
            st = _dataset_stats_from_h5(split)
            if st is None:
                ds = get_dataset(dataset_id, split)
                n = len(ds)
                n0 = sum(1 for i in range(n) if ds[i][1] == 0)
                n1 = n - n0
                st = {
                    "split": split,
                    "size": n,
                    "count_normal": n0,
                    "count_metastasis": n1,
                    "ratio_normal": round(n0 / n, 4) if n else 0,
                    "ratio_metastasis": round(n1 / n, 4) if n else 0,
                }
            _dataset_stats_cache[cache_key] = st
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(_dataset_stats_cache[cache_key])


@app.route("/api/dataset/verify")
def api_dataset_verify():
    """Verify that the configured dataset is fully labeled: for each split, size = count_normal + count_metastasis."""
    dataset_id = request.args.get("dataset", "pcam")
    if (dataset_id or "").strip().lower() != "pcam":
        return jsonify({"error": "Dataset verification only implemented for 'pcam' in this repo."}), 501
    result = {"datasets": [], "all_labeled": True}
    for split in ("train", "val", "test"):
        try:
            st = _dataset_stats_from_h5(split)
            if st is None:
                ds = get_dataset(dataset_id, split)
                n = len(ds)
                n0 = sum(1 for i in range(n) if ds[i][1] == 0)
                n1 = n - n0
                st = {"split": split, "size": n, "count_normal": n0, "count_metastasis": n1}
            total_labeled = st["count_normal"] + st["count_metastasis"]
            all_labeled = total_labeled == st["size"] and st["size"] > 0
            result["datasets"].append({
                "split": split,
                "size": st["size"],
                "count_normal": st["count_normal"],
                "count_metastasis": st["count_metastasis"],
                "all_labeled": all_labeled,
                "label_domain": [0, 1],
            })
            if not all_labeled:
                result["all_labeled"] = False
        except Exception as e:
            result["datasets"].append({"split": split, "error": str(e)})
            result["all_labeled"] = False
    return jsonify(result)


@app.route("/api/datasets")
def api_datasets():
    """
    Return datasets for UI.

    Fields:
    - can_browse: Dataset preview/stats implemented in Dataset tab
    - can_train: Dataset selection implemented for training runs
    """
    cfg = get_config()
    dataset_id = (cfg.get("data") or {}).get("dataset", "pcam")
    # PCAM sizes from official splits (thesis defaults)
    # Notes for thesis UI:
    # - Browsing/preview is implemented only for PCAM (patches stored as H5).
    # - Additional trusted datasets may be downloadable but not yet previewable.
    available = [
        {
            "id": "pcam",
            "name": "Patch Camelyon (PCAM)",
            "description": "Binary metastasis detection; 96×96 patches from CAMELYON16.",
            "splits": ["train", "val", "test"],
            "max_train": 262144,
            "max_val": 32768,
            "max_test": 32768,
            "can_browse": True,
            "can_train": True,
            "download_url": "https://patchcamelyon.grand-challenge.org/Download/",
            "resource_url": "https://patchcamelyon.grand-challenge.org/Download/",
        },
        {
            "id": "nct_crc_he_100k",
            "name": "NCT-CRC-HE-100K",
            "description": "Trusted colorectal patch dataset (download available). Browsing/training not implemented yet.",
            "splits": ["train", "val", "test"],
            "can_browse": False,
            "can_train": False,
            "download_url": "https://zenodo.org/record/1214456",
            "resource_url": "https://zenodo.org/record/1214456",
        },
    ]
    return jsonify({"datasets": available, "default": dataset_id})


@app.route("/api/models")
def api_models():
    """Return available model IDs for comparative experiments."""
    cfg = get_config()
    default_model = ((cfg.get("model") or {}).get("model_id")) or "google/vit-base-patch16-224"
    return jsonify({"models": MODEL_CATALOG, "default": default_model})


@app.route("/api/dataset/<split>/indices")
def api_dataset_indices(split):
    """Return indices for browsing, optionally filtered by label. Query: label=all|0|1, offset=0, limit=24."""
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    dataset_id = request.args.get("dataset", "pcam")
    ds = get_dataset(dataset_id, split)
    label_arg = request.args.get("label", "all")
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(500, max(1, int(request.args.get("limit", 24))))
    except ValueError:
        offset, limit = 0, 24
    try:
        size = len(ds)
        if label_arg == "all":
            indices = list(range(offset, min(offset + limit, size)))
        else:
            target = int(label_arg)
            if target not in (0, 1):
                return jsonify({"error": "label must be 0, 1, or all"}), 400
            indices = []
            skipped = 0
            for i in range(min(size, 500000)):
                _, lab = ds[i]
                if int(lab) != target:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                indices.append(i)
                if len(indices) >= limit:
                    break
        return jsonify({"split": split, "indices": indices, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/samples")
def api_dataset_samples(split):
    """
    Return browseable samples in a single call.
    This is used by the Dataset tab to avoid N+1 label fetches.
    Query: dataset=<id>&label=all|0|1&offset=0&limit=24
    """
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400

    dataset_id = request.args.get("dataset", "pcam")
    try:
        label_arg = request.args.get("label", "all")
        mode = (request.args.get("mode", "offset") or "offset").strip().lower()
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(500, max(1, int(request.args.get("limit", 24))))
        random_n = int(request.args.get("random_n", limit))
        random_n = min(500, max(1, random_n))
    except ValueError:
        return jsonify({"error": "Invalid offset/limit"}), 400

    try:
        dataset_id_norm = (dataset_id or "").strip().lower()
        if dataset_id_norm != "pcam":
            return jsonify({"error": f"Dataset browse only implemented for 'pcam' (requested: {dataset_id})"}), 501

        # PCAM: use cached H5 label array for fast indexing + random sampling.
        pc = _pcam_labels_for_split(split)
        y = pc["y"]
        size = pc["size"]
        if size <= 0:
            return jsonify({"split": split, "dataset": dataset_id, "items": [], "size": 0})

        if label_arg == "all":
            if mode == "random":
                k = min(random_n, size)
                rng = np.random.default_rng()
                sel = rng.choice(size, size=k, replace=False)
                items = [{"idx": int(i), "label": int(y[int(i)])} for i in sel]
            else:
                start = min(offset, size)
                end = min(offset + limit, size)
                sel = np.arange(start, end, dtype=np.int64)
                items = [{"idx": int(i), "label": int(y[int(i)])} for i in sel]
        else:
            target = int(label_arg)
            if target not in (0, 1):
                return jsonify({"error": "label must be 0, 1, or all"}), 400
            pool = pc["indices0"] if target == 0 else pc["indices1"]
            pool_size = int(pool.shape[0])

            if pool_size <= 0:
                return jsonify({"split": split, "dataset": dataset_id, "items": [], "size": size})

            if mode == "random":
                k = min(random_n, pool_size)
                rng = np.random.default_rng()
                sel = rng.choice(pool, size=k, replace=False)
                items = [{"idx": int(i), "label": target} for i in sel]
            else:
                start = min(offset, pool_size)
                end = min(offset + limit, pool_size)
                sel = pool[start:end]
                items = [{"idx": int(i), "label": target} for i in sel]

        return jsonify({"split": split, "dataset": dataset_id, "items": items, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/project/structure")
def api_project_structure():
    """Return tree of key dirs/files for project management view."""
    root = REPO_ROOT
    ignore = {"venv", "__pycache__", ".git", ".h5", ".gz"}
    structure = []

    def walk(path: Path, prefix: str = ""):
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return
        dirs = [e for e in entries if e.is_dir() and e.name not in ignore]
        files = [e for e in entries if e.is_file() and not e.suffix in (".h5", ".gz")]
        for d in dirs:
            rel = d.relative_to(root)
            structure.append({"path": str(rel), "type": "dir"})
            if len([x for x in structure if x["path"].startswith(str(rel))]) < 50:
                walk(d, prefix + "  ")
        for f in files[:100]:  # cap files per dir
            structure.append({"path": str(f.relative_to(root)), "type": "file"})

    for top in [
        "configs",
        "data",
        "models",
        "uncertainty",
        "uncertainty_lab",
        "evaluation",
        "experiments",
        "web",
        "scripts",
    ]:
        p = root / top
        if p.exists():
            structure.append({"path": top, "type": "dir"})
            if p.is_dir():
                walk(p)
    return jsonify({"root": str(root), "entries": structure})


@app.route("/api/config")
def api_config():
    return jsonify(get_config())


@app.route("/api/device")
def api_device():
    """Return where the model runs: local (on this machine), and device (cuda/cpu)."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        name = None
        if cuda:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return jsonify({
            "model_location": "local",
            "device": device,
            "device_name": name,
            "note": "Model is downloaded once from Hugging Face, then all training and inference run locally on this machine. GPU = fast, CPU = slower.",
        })
    except Exception as e:
        return jsonify({"model_location": "local", "device": "unknown", "error": str(e)})


# ---------- Training with live log (SSE) ----------
_train_state = {
    "process": None,
    "queue": None,
    "thread": None,
    "epochs_total": None,
    "buffer": [],  # list[{seq:int, text:str}] ring buffer for UI replay
    "next_seq": 0,
}
_train_lock = threading.Lock()


def _train_reader(process, queue):
    """
    Read training stdout line-by-line and push into:
      - queue: for SSE streaming
      - buffer: for UI replay when user navigates away/back
    """
    try:
        for raw_line in iter(process.stdout.readline, ""):
            text = raw_line.replace("\n", " ").strip()
            if not text:
                continue
            with _train_lock:
                seq = _train_state["next_seq"]
                _train_state["next_seq"] = seq + 1
                _train_state["buffer"].append({"seq": seq, "text": text})
                # Cap buffer size to keep memory bounded.
                if len(_train_state["buffer"]) > 2000:
                    _train_state["buffer"] = _train_state["buffer"][-2000:]
            queue.put({"seq": seq, "text": text})
    except Exception:
        pass
    finally:
        queue.put(None)  # sentinel


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    """Start training in background. Body: { dataset, model_id, epochs, n_train, n_val, lr, batch_size }."""
    global _train_state
    if _train_state["process"] is not None and _train_state["process"].poll() is None:
        return jsonify({"ok": False, "error": "Training already running"}), 400
    data = request.get_json() or {}
    cfg = get_config()
    cmd = [sys.executable, "experiments/train.py"]
    if data.get("dataset") is not None:
        cmd.extend(["--dataset", str(data["dataset"]).strip()])
    if data.get("model_id") is not None and str(data.get("model_id")).strip():
        cmd.extend(["--model_id", str(data["model_id"]).strip()])
    if data.get("epochs") is not None:
        epochs_total = int(data.get("epochs"))
        cmd.extend(["--epochs", str(epochs_total)])
    else:
        epochs_total = int(((cfg or {}).get("train") or {}).get("epochs", 3))
    if data.get("n_train") is not None:
        cmd.extend(["--n_train", str(int(data["n_train"]))])
    if data.get("n_val") is not None:
        cmd.extend(["--n_val", str(int(data["n_val"]))])
    if data.get("lr") is not None:
        cmd.extend(["--lr", str(float(data["lr"]))])
    if data.get("batch_size") is not None:
        cmd.extend(["--batch_size", str(int(data["batch_size"]))])
    try:
        import queue as q
        proc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        que = q.Queue()
        th = threading.Thread(target=_train_reader, args=(proc, que), daemon=True)
        th.start()
        with _train_lock:
            _train_state["process"] = proc
            _train_state["queue"] = que
            _train_state["thread"] = th
            _train_state["epochs_total"] = epochs_total
            _train_state["buffer"] = []
            _train_state["next_seq"] = 0
        return jsonify({"ok": True, "task_id": "train", "status": "started"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/train/status")
def api_train_status():
    """Return whether training is running and recent log buffer for UI replay."""
    with _train_lock:
        proc = _train_state.get("process")
        running = proc is not None and proc.poll() is None
        epochs_total = _train_state.get("epochs_total")
        buf = list(_train_state.get("buffer") or [])
        last_seq = buf[-1]["seq"] if buf else -1
    return jsonify({
        "running": running,
        "epochs_total": epochs_total,
        "buffer": buf[-200:],  # keep status response small
        "last_seq": last_seq,
    })


@app.route("/api/train/runs")
def api_train_runs():
    """List saved training runs (checkpoints/run_*) with their metrics."""
    base = REPO_ROOT / "checkpoints"
    if not base.exists():
        return jsonify({"runs": []})
    runs = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        metrics_file = d / "metrics.json"
        if not metrics_file.exists():
            runs.append({"run_id": d.name, "run_dir": str(d), "best_val_acc": None})
            continue
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            runs.append({
                "run_id": m.get("run_id", d.name),
                "run_dir": m.get("run_dir", str(d)),
                "model_id": m.get("model_id"),
                "epochs": m.get("epochs"),
                "dataset": m.get("dataset"),
                "n_train": m.get("n_train"),
                "n_val": m.get("n_val"),
                "lr": m.get("lr"),
                "batch_size": m.get("batch_size"),
                "best_val_acc": m.get("best_val_acc"),
                "best_epoch": m.get("best_epoch"),
                "history": m.get("history", []),
            })
        except Exception:
            runs.append({"run_id": d.name, "run_dir": str(d), "best_val_acc": None})
    return jsonify({"runs": runs})


@app.route("/api/train/stream")
def api_train_stream():
    """Server-Sent Events: stream training log lines until process ends."""
    from_seq = int(request.args.get("from_seq", -1))
    def generate(from_seq=from_seq):
        queue = _train_state.get("queue")
        process = _train_state.get("process")
        if queue is None or process is None:
            yield "data: {\"type\":\"done\",\"text\":\"[No training running]\"}\n\n"
            return
        while True:
            try:
                item = queue.get(timeout=2)
                if item is None:
                    yield 'data: {"type":"done","text":"[DONE]"}\n\n'
                    break
                seq = item.get("seq", -1)
                text = item.get("text", "")
                if seq > from_seq and text:
                    # JSON payload for robustness on the client.
                    yield 'data: {"type":"log","seq":%d,"text":%s}\n\n' % (seq, json.dumps(text))
            except Exception:
                if process.poll() is not None:
                    yield 'data: {"type":"done","text":"[DONE]"}\n\n'
                    break
                yield ": keepalive\n\n"
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- Blind test: random sample, predict, reveal label ----------
_model_cache = None


def _resolve_model_id_for_run(run_id=None):
    """Resolve model id from run metrics or fallback to default config."""
    cfg = get_config()
    model_id = cfg.get("model", {}).get("model_id", "google/vit-base-patch16-224")
    if not run_id:
        return model_id
    metrics_path = REPO_ROOT / "checkpoints" / run_id / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            model_id = m.get("model_id") or model_id
        except Exception:
            pass
    return model_id


def _predict_probs(model, x, method="mc_dropout", mc_samples=30):
    """
    Return class probabilities and uncertainty fields.
    method: mc_dropout (preferred) | confidence
    """
    import torch
    method = (method or "mc_dropout").strip().lower()
    if method not in ("confidence", "mc_dropout"):
        method = "mc_dropout"

    if method == "mc_dropout":
        # Enable dropout stochasticity at test time.
        model.train()
        T = max(2, min(100, int(mc_samples)))
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                logits = model(pixel_values=x).logits
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        probs_stack = torch.stack(probs_list, dim=0)  # (T, B, C)
        probs_mean = probs_stack.mean(dim=0)
        # Epistemic proxy: mean predictive variance over classes.
        uncertainty_var = probs_stack.var(dim=0).mean(dim=1)  # (B,)
        # Predictive entropy.
        entropy = -(probs_mean * (probs_mean.clamp(min=1e-12)).log()).sum(dim=1)  # (B,)
        model.eval()
        return probs_mean, {"uncertainty_var": uncertainty_var, "entropy": entropy, "mc_samples": T}

    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)
    entropy = -(probs * (probs.clamp(min=1e-12)).log()).sum(dim=1)
    return probs, {"uncertainty_var": None, "entropy": entropy, "mc_samples": 1}


def _compute_calibration_metrics(results, n_bins=15):
    """
    Compute ECE and reliability bins.
    results items require: confidence, correct.
    """
    n_bins = max(2, min(50, int(n_bins)))
    if not results:
        return {"ece": 0.0, "bins": []}

    bins = []
    ece = 0.0
    total = float(len(results))

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Right-closed on last bin.
        in_bin = []
        for r in results:
            c = float(r["confidence"])
            if (c >= lo and c < hi) or (b == n_bins - 1 and c <= hi):
                in_bin.append(r)
        if not in_bin:
            bins.append({"bin": b, "lo": round(lo, 6), "hi": round(hi, 6), "count": 0, "acc": None, "conf": None})
            continue
        acc = sum(1.0 if x["correct"] else 0.0 for x in in_bin) / len(in_bin)
        conf = sum(float(x["confidence"]) for x in in_bin) / len(in_bin)
        frac = len(in_bin) / total
        ece += abs(acc - conf) * frac
        bins.append({"bin": b, "lo": round(lo, 6), "hi": round(hi, 6), "count": len(in_bin), "acc": round(acc, 6), "conf": round(conf, 6)})

    return {"ece": round(float(ece), 6), "bins": bins}


def _get_model_for_inference(run_id=None, model_id_override=None):
    """Load model from run checkpoint or HF. run_id e.g. 'run_20250315_120000'. Cached by run/model."""
    global _model_cache
    model_id = model_id_override or _resolve_model_id_for_run(run_id=run_id)
    cache_key = f"{run_id or 'default'}::{model_id}"
    if _model_cache is not None:
        if getattr(_model_cache, "_run_id", None) == cache_key:
            return _model_cache
        _model_cache = None
    import torch
    from models.load_model import load_hf_image_classifier, get_device
    cfg = get_config()
    model, _, _ = load_hf_image_classifier(
        model_id=model_id,
        num_labels=cfg.get("model", {}).get("num_labels", 2),
    )
    if run_id:
        ckpt = REPO_ROOT / "checkpoints" / run_id / "best.pt"
    else:
        ckpt = REPO_ROOT / "checkpoints" / "best.pt"
    if not ckpt.exists() and not run_id:
        # Fallback: latest run
        base = REPO_ROOT / "checkpoints"
        if base.exists():
            run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
            if run_dirs and (run_dirs[0] / "best.pt").exists():
                ckpt = run_dirs[0] / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
    device = get_device()
    model = model.to(device)
    model.eval()
    _model_cache = model
    _model_cache._run_id = cache_key
    return model


@app.route("/api/evaluate/random")
def api_evaluate_random():
    """Return a random patch index for blind evaluation.

    Query:
      - split: train|val|test (default: test)
    """
    try:
        split = (request.args.get("split", "test") or "test").strip().lower()
        if split not in ("train", "val", "test"):
            return jsonify({"error": "Invalid split"}), 400

        ds = get_pcam(split)
        n = len(ds)
        if n == 0:
            return jsonify({"error": "Test set empty"}), 500
        idx = random.randint(0, n - 1)
        return jsonify({"split": split, "idx": idx, "size": n})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/batch")
def api_evaluate_batch():
    """
    Run inference on N random patches and return results + summary.

    Query:
      - split: train|val|test (default: test)
      - n: number of random patches (default: 10, max: 50)
      - run_id: optional training run_id to load from checkpoints/run_*/best.pt
      - method: confidence|mc_dropout
      - mc_samples: number of stochastic passes for MC dropout
      - calibration_bins: number of ECE bins
    """
    split = (request.args.get("split", "test") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400

    run_id = request.args.get("run_id") or None
    method = (request.args.get("method", "mc_dropout") or "mc_dropout").strip().lower()
    try:
        mc_samples = int(request.args.get("mc_samples", 30))
    except ValueError:
        mc_samples = 30
    try:
        calibration_bins = int(request.args.get("calibration_bins", 15))
    except ValueError:
        calibration_bins = 15

    try:
        n = int(request.args.get("n", 10))
    except ValueError:
        n = 10
    n = max(1, min(50, n))

    try:
        ds = get_pcam(split)
        size = len(ds)
        if size == 0:
            return jsonify({"error": f"{split} set empty"}), 500

        if n >= size:
            indices = list(range(size))
        else:
            indices = random.sample(range(size), n)

        import torch
        from torchvision import transforms

        # Load model once per batch.
        model = _get_model_for_inference(run_id=run_id)
        device = next(model.parameters()).device
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        results = []
        correct = 0
        confs_correct = []
        confs_incorrect = []

        with torch.no_grad():
            for idx in indices:
                img, label = ds[idx]
                # Ensure RGB for image models.
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                x = transform(img).unsqueeze(0).to(device)
                probs_t, extra = _predict_probs(model, x, method=method, mc_samples=mc_samples)
                probs = probs_t[0].cpu().numpy()

                pred = int(probs.argmax())
                conf = float(probs[pred])
                gt = int(label)
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1
                    confs_correct.append(conf)
                else:
                    confs_incorrect.append(conf)

                label_name = "Metastasis" if pred == 1 else "Normal"

                results.append({
                    "idx": int(idx),
                    "gt": gt,
                    "pred": pred,
                    "pred_label_name": label_name,
                    "p_normal": float(probs[0]),
                    "p_metastasis": float(probs[1]),
                    "confidence": round(conf, 6),
                    "uncertainty": round(float(1.0 - conf), 6),
                    "entropy": round(float(extra["entropy"][0].cpu().item()), 6),
                    "uncertainty_var": None if extra["uncertainty_var"] is None else round(float(extra["uncertainty_var"][0].cpu().item()), 6),
                    "correct": is_correct,
                })

        mean_conf = float(sum([r["confidence"] for r in results]) / len(results)) if results else 0.0
        acc = correct / len(results) if results else 0.0
        mean_conf_correct = float(sum(confs_correct) / len(confs_correct)) if confs_correct else None
        mean_conf_incorrect = float(sum(confs_incorrect) / len(confs_incorrect)) if confs_incorrect else None

        calib = _compute_calibration_metrics(results, n_bins=calibration_bins)

        return jsonify({
            "split": split,
            "n": len(results),
            "run_id": run_id,
            "method": method,
            "mc_samples": max(2, min(100, int(mc_samples))) if method == "mc_dropout" else 1,
            "summary": {
                "accuracy": round(acc, 6),
                "mean_confidence": round(mean_conf, 6),
                "mean_confidence_correct": None if mean_conf_correct is None else round(mean_conf_correct, 6),
                "mean_confidence_incorrect": None if mean_conf_incorrect is None else round(mean_conf_incorrect, 6),
                "ece": calib["ece"],
            },
            "calibration": calib,
            "results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/predict")
def api_evaluate_predict():
    """Run model on one sample; return prediction and uncertainty fields."""
    split = request.args.get("split", "test")
    run_id = request.args.get("run_id") or None
    method = (request.args.get("method", "mc_dropout") or "mc_dropout").strip().lower()
    try:
        mc_samples = int(request.args.get("mc_samples", 30))
    except ValueError:
        mc_samples = 30
    try:
        idx = int(request.args.get("idx", 0))
    except ValueError:
        return jsonify({"error": "Invalid idx"}), 400
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        import torch
        from torchvision import transforms
        ds = get_pcam(split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        img, _ = ds[idx]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        x = transform(img).unsqueeze(0)
        model = _get_model_for_inference(run_id=run_id)
        device = next(model.parameters()).device
        x = x.to(device)
        probs_t, extra = _predict_probs(model, x, method=method, mc_samples=mc_samples)
        prob = probs_t[0].cpu().numpy()
        pred = int(prob.argmax())
        conf = float(prob[pred])
        label_name = "Metastasis" if pred == 1 else "Normal"
        return jsonify({
            "split": split,
            "idx": idx,
            "pred": pred,
            "prob": round(conf, 4),
            "uncertainty": round(float(1.0 - conf), 4),
            "entropy": round(float(extra["entropy"][0].cpu().item()), 6),
            "uncertainty_var": None if extra["uncertainty_var"] is None else round(float(extra["uncertainty_var"][0].cpu().item()), 6),
            "method": method,
            "mc_samples": int(extra.get("mc_samples") or 1),
            "label_name": label_name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/full", methods=["POST"])
def api_evaluate_full():
    """
    Run full thesis evaluation pipeline via experiments/evaluate_uncertainty.py.
    Body:
      - split: train|val|test
      - method: confidence|mc_dropout
      - mc_samples: int
      - max_samples: int
      - batch_size: int
      - run_id: optional
      - fit_temperature_on_val: bool
      - fit_deferral_on_val: bool
    """
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    method = (data.get("method") or "mc_dropout").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    if method not in ("confidence", "mc_dropout", "deep_ensemble"):
        return jsonify({"ok": False, "error": "Invalid method"}), 400

    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        max_samples = int(data.get("max_samples", 2000))
    except Exception:
        max_samples = 2000
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        seed = int(data.get("seed", 42))
    except Exception:
        seed = 42
    try:
        ensemble_size = int(data.get("ensemble_size", 3))
    except Exception:
        ensemble_size = 3

    run_id = (data.get("run_id") or "").strip()
    fit_temp = bool(data.get("fit_temperature_on_val", False))
    fit_deferral = bool(data.get("fit_deferral_on_val", True))
    out_path = REPO_ROOT / "evaluation" / f"metrics_{method}_{split}.json"

    cmd = [
        sys.executable,
        "experiments/evaluate_uncertainty.py",
        "--split",
        split,
        "--method",
        method,
        "--mc_samples",
        str(max(2, min(100, mc_samples))),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    if run_id:
        cmd.extend(["--run_id", run_id])
    if fit_temp:
        cmd.append("--fit_temperature_on_val")
    if fit_deferral:
        cmd.append("--fit_deferral_on_val")

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(120, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Evaluation script failed", "log": log}), 500
        if not out_path.exists():
            return jsonify({"ok": False, "error": f"Expected output not found: {out_path}", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Evaluation timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/shift", methods=["POST"])
def api_evaluate_shift():
    """Run synthetic shift/OOD evaluation pipeline."""
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    try:
        max_samples = int(data.get("max_samples", 512))
    except Exception:
        max_samples = 512
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        seed = int(data.get("seed", 42))
    except Exception:
        seed = 42

    shifts = (data.get("shifts") or "id,blur,jpeg,color,noise").strip()
    severities = (data.get("severities") or "1,3,5").strip()
    method = (data.get("method") or "mc_dropout").strip().lower()
    if method not in ("confidence", "mc_dropout", "deep_ensemble"):
        return jsonify({"ok": False, "error": "Invalid method"}), 400
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        ensemble_size = int(data.get("ensemble_size", 3))
    except Exception:
        ensemble_size = 3
    run_id = (data.get("run_id") or "").strip()
    out_path = REPO_ROOT / "evaluation" / f"shift_ood_{split}.json"

    cmd = [
        sys.executable,
        "experiments/evaluate_shift_ood.py",
        "--split",
        split,
        "--method",
        method,
        "--mc_samples",
        str(max(2, mc_samples)),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--seed",
        str(seed),
        "--shifts",
        shifts,
        "--severities",
        severities,
        "--out",
        str(out_path),
    ]
    if run_id:
        cmd.extend(["--run_id", run_id])

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(120, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Shift evaluation failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Shift evaluation timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/all", methods=["POST"])
def api_evaluate_all():
    """Run complete thesis bundle (core + shift) and return combined summary."""
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    try:
        max_samples = int(data.get("max_samples", 512))
    except Exception:
        max_samples = 512
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    run_id = (data.get("run_id") or "").strip()
    fit_temp = bool(data.get("fit_temperature_on_val", False))
    fit_deferral = bool(data.get("fit_deferral_on_val", True))
    include_deep_ensemble = bool(data.get("include_deep_ensemble", True))
    try:
        ensemble_size = int(data.get("ensemble_size", 3))
    except Exception:
        ensemble_size = 3
    shift_severities = (data.get("shift_severities") or "1,3,5").strip()

    out_path = REPO_ROOT / "evaluation" / "thesis_bundle_summary.json"
    cmd = [
        sys.executable,
        "experiments/run_thesis_bundle.py",
        "--split",
        split,
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--mc_samples",
        str(max(2, mc_samples)),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--shift_severities",
        shift_severities,
        "--out",
        str(out_path.relative_to(REPO_ROOT)),
    ]
    if fit_temp:
        cmd.append("--fit_temperature_on_val")
    if fit_deferral:
        cmd.append("--fit_deferral_on_val")
    if include_deep_ensemble:
        cmd.append("--include_deep_ensemble")
    if run_id:
        cmd.extend(["--run_id", run_id])

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(180, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Thesis bundle failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Thesis bundle timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/results")
def api_evaluate_results():
    """
    List saved evaluation JSON files and optionally load one.
    Query:
      - action: list | load
      - name: file name (required for action=load), e.g. metrics_confidence_test.json
    """
    action = (request.args.get("action") or "list").strip().lower()
    base = REPO_ROOT / "evaluation"
    base.mkdir(parents=True, exist_ok=True)

    if action == "list":
        files = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files:
            try:
                st = p.stat()
                out.append(
                    {
                        "name": p.name,
                        "path": str(p.relative_to(REPO_ROOT)),
                        "size_bytes": int(st.st_size),
                        "modified_ts": int(st.st_mtime),
                    }
                )
            except Exception:
                continue
        return jsonify({"ok": True, "files": out})

    if action == "load":
        name = (request.args.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Missing 'name' for load action"}), 400
        # basic filename hardening
        if "/" in name or "\\" in name or not name.endswith(".json"):
            return jsonify({"ok": False, "error": "Invalid file name"}), 400
        path = base / name
        if not path.exists():
            return jsonify({"ok": False, "error": f"File not found: {name}"}), 404
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return jsonify({"ok": True, "name": name, "path": str(path.relative_to(REPO_ROOT)), "result": payload})
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to read JSON: {e}"}), 500

    return jsonify({"ok": False, "error": "Unknown action. Use list or load."}), 400


# Allowed run tasks (command list; first arg is Python, relative paths to REPO_ROOT)
def _run_cmd(py_args):
    return [sys.executable] + py_args


RUN_TASKS = {
    "check_setup": (
        _run_cmd(["-c", "import torch; import torchvision; from torchvision.datasets import PCAM; print('OK: torch', torch.__version__, 'PCAM available')"]),
        "Verify environment: PyTorch, torchvision, PCAM dataset availability.",
    ),
    "download_pcam": (
        _run_cmd(["data/download_datasets.py", "--root", "data/raw", "--dataset", "pcam"]),
        "Download Patch Camelyon (train/val/test) to data/raw.",
    ),
    "download_nct_crc_he_100k": (
        _run_cmd(["data/download_datasets.py", "--root", "data/raw", "--dataset", "nct_crc_he_100k"]),
        "Download NCT-CRC-HE-100K (Zenodo zip) to data/raw/nct_crc_he_100k.",
    ),
    "cache_model": (
        _run_cmd(["models/load_model.py", "--model_id", "google/vit-base-patch16-224"]),
        "Download and cache Hugging Face ViT model for binary classification.",
    ),
    "uncertainty_lab_check": (
        _run_cmd(
            [
                "-c",
                "import uncertainty_lab; from uncertainty_lab.pipeline.run import run_pipeline; "
                "print('OK uncertainty_lab', getattr(uncertainty_lab, '__version__', '?'))",
            ]
        ),
        "Verify Uncertainty Lab package (import + run_pipeline symbol).",
    ),
}
# Training is started via POST /api/train/start and streamed via /api/train/stream (see Run page).


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run a named task (download_pcam, cache_model, check_setup). Returns log."""
    data = request.get_json() or {}
    task = data.get("task")
    if task not in RUN_TASKS:
        return jsonify({"ok": False, "error": "Unknown task", "allowed": list(RUN_TASKS.keys())}), 400
    cmd = RUN_TASKS[task][0]
    if task == "cache_model":
        model_id = (data.get("model_id") or "").strip()
        if model_id:
            cmd = _run_cmd(["models/load_model.py", "--model_id", model_id])
    cwd = str(REPO_ROOT)
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=app.config.get("RUN_TIMEOUT", 600),
        )
        out = (r.stdout or "") + (r.stderr or "")
        return jsonify({
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "log": out or "(no output)",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Task timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/lab")
def lab_page():
    """Uncertainty Lab: config-driven pipeline from the browser."""
    return render_template("lab.html")


@app.route("/api/lab/run", methods=["POST"])
def api_lab_run():
    """Merge JSON/YAML fragment into lab defaults and call ``run_pipeline``."""
    import traceback

    data = request.get_json() or {}
    raw = data.get("config")
    try:
        if isinstance(raw, str):
            patch = yaml.safe_load(raw) or {}
        elif isinstance(raw, dict):
            patch = raw
        else:
            patch = {}
        from uncertainty_lab.config import deep_merge, load_config
        from uncertainty_lab.pipeline.run import run_pipeline

        cfg = load_config(repo_root=REPO_ROOT)
        cfg = deep_merge(cfg, patch)
        cfg.setdefault("run", {})["repo_root"] = str(REPO_ROOT)
        r = run_pipeline(cfg)
        return jsonify({"ok": True, **r})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/runs/list")
def api_runs_list():
    """List subdirectories under ``runs/`` (Uncertainty Lab outputs)."""
    base = REPO_ROOT / "runs"
    if not base.is_dir():
        return jsonify({"ok": True, "runs": []})
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for d in dirs[:200]:
        mp = d / "metrics.json"
        cp = d / "config.yaml"
        out.append(
            {
                "name": d.name,
                "path": str(d.relative_to(REPO_ROOT)),
                "has_metrics": mp.exists(),
                "has_config": cp.exists(),
                "modified_ts": int(d.stat().st_mtime),
            }
        )
    return jsonify({"ok": True, "runs": out})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5000, help="Port")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
