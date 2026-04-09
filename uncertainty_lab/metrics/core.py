"""Classification, calibration, and selective-prediction metrics from logits (binary)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    ece = 0.0
    out_bins: list[dict] = []
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            out_bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "acc": None, "conf": None})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += abs(acc - conf) * (cnt / total)
        out_bins.append({"lo": float(lo), "hi": float(hi), "count": cnt, "acc": acc, "conf": conf})
    return float(ece), out_bins


def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def binary_roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    if len(np.unique(y_true)) < 2:
        return []
    fpr, tpr, thr = roc_curve(y_true, y_score)
    out = []
    for i in range(len(fpr)):
        th = float(thr[i])
        out.append({"fpr": float(fpr[i]), "tpr": float(tpr[i]), "threshold": None if not np.isfinite(th) else th})
    return out


def binary_pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    if len(np.unique(y_true)) < 2:
        return []
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    out = []
    for i in range(len(precision)):
        if i >= len(thr):
            threshold = None
        else:
            t = float(thr[i])
            threshold = None if not np.isfinite(t) else t
        out.append({"precision": float(precision[i]), "recall": float(recall[i]), "threshold": threshold})
    return out


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray) -> tuple[list[dict], float]:
    order = np.argsort(uncertainty)
    err_sorted = error[order].astype(np.float64)
    n = len(err_sorted)
    if n == 0:
        return [], 0.0
    cumsum_err = np.cumsum(err_sorted)
    curve = []
    for k in range(1, n + 1):
        coverage = k / n
        risk = cumsum_err[k - 1] / k
        curve.append({"coverage": float(coverage), "risk": float(risk)})
    coverage_arr = np.array([p["coverage"] for p in curve], dtype=np.float64)
    risk_arr = np.array([p["risk"] for p in curve], dtype=np.float64)
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risk_arr, coverage_arr))
    else:
        aurc = float(np.trapz(risk_arr, coverage_arr))
    return curve, aurc


def try_torch_uncertainty_aurc(error: np.ndarray, uncertainty: np.ndarray) -> float | None:
    try:
        import torch
        from torch_uncertainty.metrics.classification import AURC

        target = torch.tensor(error.astype(np.int64))
        score = torch.tensor(uncertainty.astype(np.float32))
        metric = AURC()
        val = metric(score, target)
        return float(val.detach().cpu().item())
    except Exception:
        return None


def target_risk_thresholds(rc_curve: list[dict], targets: list[float] | None = None) -> list[dict]:
    targets = targets or [0.01, 0.02, 0.05]
    out = []
    for t in targets:
        match = next((p for p in rc_curve if p["risk"] <= t), None)
        if match is None:
            out.append({"target_risk": float(t), "coverage": None, "deferral_rate": None})
        else:
            cov = float(match["coverage"])
            out.append({"target_risk": float(t), "coverage": cov, "deferral_rate": float(1.0 - cov)})
    return out


def confidence_diagnostics(conf: np.ndarray, correct: np.ndarray) -> dict:
    total = int(len(conf))
    if total == 0:
        return {
            "mean_confidence": None,
            "mean_confidence_correct": None,
            "mean_confidence_incorrect": None,
            "overconfidence_gap": None,
            "high_confidence_fraction_at_0_90": None,
            "high_confidence_fraction_at_0_95": None,
            "high_confidence_error_rate_at_0_90": None,
            "high_confidence_error_rate_at_0_95": None,
        }
    conf = conf.astype(np.float64, copy=False)
    correct = correct.astype(np.int64, copy=False)
    incorrect = 1 - correct

    def _safe_mean(arr: np.ndarray) -> float | None:
        if arr.size == 0:
            return None
        return float(arr.mean())

    conf_correct = conf[correct == 1]
    conf_incorrect = conf[incorrect == 1]
    mean_conf = float(conf.mean())
    mean_conf_correct = _safe_mean(conf_correct)
    mean_conf_incorrect = _safe_mean(conf_incorrect)
    over_gap = None
    if mean_conf_incorrect is not None:
        over_gap = float(mean_conf_incorrect - float(incorrect.mean()))

    def _subset_metrics(th: float) -> tuple[float, float | None]:
        mask = conf >= th
        frac = float(mask.mean())
        if int(mask.sum()) == 0:
            return frac, None
        err_rate = float((1 - correct[mask]).mean())
        return frac, err_rate

    frac90, err90 = _subset_metrics(0.90)
    frac95, err95 = _subset_metrics(0.95)
    return {
        "mean_confidence": mean_conf,
        "mean_confidence_correct": mean_conf_correct,
        "mean_confidence_incorrect": mean_conf_incorrect,
        "overconfidence_gap": over_gap,
        "high_confidence_fraction_at_0_90": frac90,
        "high_confidence_fraction_at_0_95": frac95,
        "high_confidence_error_rate_at_0_90": err90,
        "high_confidence_error_rate_at_0_95": err95,
    }


def summarize_from_logits(logits: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits_t, dim=1).numpy().astype(np.float64)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    p1 = probs[:, 1]
    conf = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    uncertainty_msp = 1.0 - conf
    error = (y_pred != y_true).astype(np.int64)
    correct = 1 - error

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    roc_auc = safe_binary_auc(y_true, p1)
    pr_auc = safe_binary_auprc(y_true, p1)
    ece, rel_bins = compute_ece(conf, correct, n_bins=n_bins)
    nll = float(log_loss(y_true, np.vstack([1.0 - p1, p1]).T, labels=[0, 1]))
    brier = float(brier_score_loss(y_true, p1))

    err_auc_entropy = safe_binary_auc(error, entropy)
    err_pr_entropy = safe_binary_auprc(error, entropy)
    err_auc_msp = safe_binary_auc(error, uncertainty_msp)
    err_pr_msp = safe_binary_auprc(error, uncertainty_msp)
    err_roc_entropy = binary_roc_curve_points(error, entropy)
    err_prc_entropy = binary_pr_curve_points(error, entropy)
    err_roc_msp = binary_roc_curve_points(error, uncertainty_msp)
    err_prc_msp = binary_pr_curve_points(error, uncertainty_msp)
    conf_diag = confidence_diagnostics(conf, correct)
    rc_curve, aurc = risk_coverage(uncertainty_msp, error)
    aurc_tu = try_torch_uncertainty_aurc(error, uncertainty_msp)
    risk_targets = target_risk_thresholds(rc_curve)

    return {
        "predictive_performance": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
        "calibration": {
            "ece": float(ece),
            "nll": nll,
            "brier": brier,
            "reliability_bins": rel_bins,
        },
        "uncertainty_quality": {
            "error_detection_entropy": {
                "auroc": err_auc_entropy,
                "auprc": err_pr_entropy,
                "roc_curve": err_roc_entropy,
                "pr_curve": err_prc_entropy,
            },
            "error_detection_one_minus_msp": {
                "auroc": err_auc_msp,
                "auprc": err_pr_msp,
                "roc_curve": err_roc_msp,
                "pr_curve": err_prc_msp,
            },
            "prediction_confidence_diagnostics": conf_diag,
            "notes": {
                "auroc_auprc_scope": "Ranking-only view of uncertainty quality; do not use alone for per-prediction confidence validity.",
                "msp_baseline_scope": "1-MSP is retained as a crude baseline and is typically overconfident.",
            },
        },
        "selective_prediction": {
            "aurc": float(aurc),
            "aurc_torch_uncertainty": aurc_tu,
            "risk_coverage_curve": rc_curve[:: max(1, len(rc_curve) // 200)],
            "target_risk_thresholds": risk_targets,
        },
        "internals": {
            "uncertainty_one_minus_msp": uncertainty_msp,
            "entropy": entropy,
            "confidence": conf,
            "error": error,
            "y_pred": y_pred,
            "probs_pos": p1,
        },
    }


def compute_metrics_bundle(
    logits: np.ndarray,
    y_true: np.ndarray,
    cfg: dict,
    *,
    for_json: bool = False,
) -> dict:
    ev = cfg.get("evaluation", {})
    n_bins = int(ev.get("calibration_bins", 15))
    summary = summarize_from_logits(logits, y_true, n_bins=n_bins)
    if for_json:
        internals = summary.pop("internals", {})
        summary["internals"] = {
            "uncertainty_one_minus_msp": internals["uncertainty_one_minus_msp"].tolist(),
            "error": internals["error"].tolist(),
        }
    return summary


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(torch.exp(log_t).detach().cpu().item())
    return max(1e-3, min(100.0, t))


def fit_uncertainty_thresholds(
    uncertainty: np.ndarray,
    error: np.ndarray,
    targets: list[float] | None = None,
    min_coverage: float = 0.2,
) -> list[dict]:
    targets = targets or [0.01, 0.02, 0.05]
    n = int(len(uncertainty))
    if n == 0:
        return [{"target_risk": float(t), "threshold": None, "coverage": None, "risk": None} for t in targets]

    order = np.argsort(uncertainty)
    u_sorted = uncertainty[order].astype(np.float64, copy=False)
    e_sorted = error[order].astype(np.float64, copy=False)
    cumsum_e = np.cumsum(e_sorted)

    out = []
    min_keep = max(1, int(np.ceil(float(min_coverage) * n)))
    for t in targets:
        best_idx = None
        for k in range(min_keep, n + 1):
            risk = float(cumsum_e[k - 1] / k)
            if risk <= float(t):
                best_idx = k
        if best_idx is None:
            out.append({"target_risk": float(t), "threshold": None, "coverage": None, "risk": None})
            continue
        thr = float(u_sorted[best_idx - 1])
        cov = float(best_idx / n)
        risk = float(cumsum_e[best_idx - 1] / best_idx)
        out.append({"target_risk": float(t), "threshold": thr, "coverage": cov, "risk": risk})
    return out


def apply_uncertainty_thresholds(uncertainty: np.ndarray, error: np.ndarray, thresholds: list[dict]) -> list[dict]:
    out = []
    n = int(len(uncertainty))
    if n == 0:
        return out
    for x in thresholds:
        thr = x.get("threshold")
        if thr is None:
            out.append(
                {
                    "target_risk": float(x.get("target_risk", 0.0)),
                    "threshold": None,
                    "coverage": None,
                    "risk": None,
                    "deferral_rate": None,
                }
            )
            continue
        mask = uncertainty <= float(thr)
        k = int(mask.sum())
        if k == 0:
            out.append(
                {
                    "target_risk": float(x.get("target_risk", 0.0)),
                    "threshold": float(thr),
                    "coverage": 0.0,
                    "risk": None,
                    "deferral_rate": 1.0,
                }
            )
            continue
        cov = float(k / n)
        risk = float(error[mask].mean())
        out.append(
            {
                "target_risk": float(x.get("target_risk", 0.0)),
                "threshold": float(thr),
                "coverage": cov,
                "risk": risk,
                "deferral_rate": float(1.0 - cov),
            }
        )
    return out


def slide_level_proxy_from_probs(probs: np.ndarray, y_true: np.ndarray, bag_size: int = 16) -> dict:
    n = len(y_true)
    if n == 0:
        return {"bag_size": bag_size, "n_bags": 0}
    m = n // bag_size
    if m == 0:
        return {"bag_size": bag_size, "n_bags": 0}
    probs = probs[: m * bag_size]
    y_true = y_true[: m * bag_size]
    probs_bag = probs.reshape(m, bag_size, 2).mean(axis=1)
    y_bag = y_true.reshape(m, bag_size)
    y_bag = (y_bag.mean(axis=1) >= 0.5).astype(np.int64)
    p1 = probs_bag[:, 1]
    y_pred = (p1 >= 0.5).astype(np.int64)
    return {
        "bag_size": int(bag_size),
        "n_bags": int(m),
        "accuracy": float(accuracy_score(y_bag, y_pred)),
        "roc_auc": safe_binary_auc(y_bag, p1),
        "pr_auc": safe_binary_auprc(y_bag, p1),
    }


def top_k_indices(uncertainty: np.ndarray, error: np.ndarray | None, k: int, prefer_errors: bool = True) -> np.ndarray:
    """Indices of most uncertain samples; optionally rank misclassified higher."""
    u = uncertainty.astype(np.float64)
    n = len(u)
    if n == 0:
        return np.array([], dtype=np.int64)
    k = min(k, n)
    if prefer_errors and error is not None:
        score = u + 1e6 * error.astype(np.float64)
    else:
        score = u
    order = np.argsort(-score)
    return order[:k]
