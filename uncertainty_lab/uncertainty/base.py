"""Pluggable uncertainty methods (binary classification)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class UncertaintyMethod(ABC):
    """Each method produces averaged logits per sample (N, num_classes) for downstream metrics."""

    method_id: str = "base"

    @abstractmethod
    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return logits (N, C), labels (N,) as float64 / int64 numpy.

        ``on_batch(current, total_batches)`` is invoked after each batch (1-based index).
        """


class ConfidenceMethod(UncertaintyMethod):
    method_id = "confidence"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = models[0]
        m.eval()
        logits_list, y_list = [], []
        n_batches = len(loader)
        with torch.no_grad():
            for bi, (batch, labels) in enumerate(loader, start=1):
                batch = batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if batch.dim() == 3:
                    batch = batch.unsqueeze(0)
                logits = m(pixel_values=batch).logits
                logits_list.append(logits.detach().cpu())
                y_list.append(labels.detach().cpu())
                if on_batch is not None:
                    on_batch(bi, n_batches)
        return _cat_logits_labels(logits_list, y_list)


class MCDropoutMethod(UncertaintyMethod):
    method_id = "mc_dropout"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = models[0]
        T = max(2, int(mc_samples))
        logits_list, y_list = [], []
        n_batches = len(loader)
        for bi, (batch, labels) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            m.train()
            probs_stack = []
            with torch.no_grad():
                for _ in range(T):
                    lg = m(pixel_values=batch).logits
                    probs_stack.append(torch.softmax(lg, dim=1))
            probs = torch.stack(probs_stack, dim=0).mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
            m.eval()
            logits_list.append(logits.detach().cpu())
            y_list.append(labels.detach().cpu())
            if on_batch is not None:
                on_batch(bi, n_batches)
        return _cat_logits_labels(logits_list, y_list)


class DeepEnsembleMethod(UncertaintyMethod):
    method_id = "deep_ensemble"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits_list, y_list = [], []
        n_batches = len(loader)
        for bi, (batch, labels) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            probs_members = []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    lg = m(pixel_values=batch).logits
                    probs_members.append(torch.softmax(lg, dim=1))
            probs = torch.stack(probs_members, dim=0).mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
            logits_list.append(logits.detach().cpu())
            y_list.append(labels.detach().cpu())
            if on_batch is not None:
                on_batch(bi, n_batches)
        return _cat_logits_labels(logits_list, y_list)


def _cat_logits_labels(logits_list, y_list) -> tuple[np.ndarray, np.ndarray]:
    logits_np = torch.cat(logits_list, dim=0).numpy().astype(np.float64)
    y_np = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    return logits_np, y_np


_METHODS: dict[str, type[UncertaintyMethod]] = {
    "confidence": ConfidenceMethod,
    "mc_dropout": MCDropoutMethod,
    "deep_ensemble": DeepEnsembleMethod,
}


def get_method(method_id: str) -> UncertaintyMethod:
    key = str(method_id).lower().strip()
    if key not in _METHODS:
        raise ValueError(f"Unknown uncertainty method '{method_id}'. Choose from {list(_METHODS)}")
    return _METHODS[key]()
