"""Matplotlib plots for uncertainty and calibration."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability(rel_bins: list[dict], out_path: Path, title: str = "Reliability diagram") -> None:
    xs, ys = [], []
    for b in rel_bins:
        if b["count"] > 0 and b["conf"] is not None and b["acc"] is not None:
            xs.append(float(b["conf"]))
            ys.append(float(b["acc"]))
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    if xs:
        plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


save_reliability_plot = plot_reliability


def plot_risk_coverage(curve: list[dict], out_path: Path, title: str = "Risk–coverage") -> None:
    if not curve:
        return
    cov = [p["coverage"] for p in curve]
    risk = [p["risk"] for p in curve]
    plt.figure(figsize=(5, 4))
    plt.plot(cov, risk, linewidth=1.5)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate among kept)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, max(0.05, max(risk) * 1.05))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_uncertainty_histograms(
    uncertainty: np.ndarray,
    correct_mask: np.ndarray,
    out_path: Path,
    title: str = "Uncertainty (1 − max prob)",
) -> None:
    u = uncertainty.astype(np.float64)
    plt.figure(figsize=(5, 4))
    c = correct_mask.astype(bool)
    if c.any():
        plt.hist(u[c], bins=30, alpha=0.6, label="Correct", density=True)
    if (~c).any():
        plt.hist(u[~c], bins=30, alpha=0.6, label="Incorrect", density=True)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.title("Uncertainty distribution")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
