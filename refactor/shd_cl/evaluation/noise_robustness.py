"""Evaluate test accuracy under Gaussian noise injected into ``W_rec``.

Noise is additive, zero-mean, with absolute standard deviation ``mu`` applied
independently to every recurrent synapse: ``W_rec += N(0, mu)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from ..models.snn import ReservoirSNN
from .metrics import accuracy
from .predict import argmax_predict, collect_both_logits


@torch.no_grad()
def w_rec_std(model: ReservoirSNN) -> float:
    return float(model.W_rec.detach().std().item())


@torch.no_grad()
def make_w_rec_noise(weight: torch.Tensor, mu: float, *,
                     rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample ``N(0, mu)`` noise with the same shape/device/dtype as ``weight``."""
    if mu <= 0:
        return torch.zeros_like(weight)
    return torch.randn(weight.shape, device=weight.device, dtype=weight.dtype,
                       generator=rng) * mu


@torch.no_grad()
def apply_w_rec_noise(model: ReservoirSNN, noise: torch.Tensor) -> None:
    model.W_rec.data.add_(noise)


@torch.no_grad()
def inject_w_rec_noise(model: ReservoirSNN, mu: float, *,
                       rng: Optional[torch.Generator] = None) -> float:
    """Add ``N(0, mu)`` to each ``W_rec`` element in-place; returns ``mu``."""
    if mu > 0:
        apply_w_rec_noise(model, make_w_rec_noise(model.W_rec.data, mu, rng=rng))
    return float(mu)


@torch.no_grad()
def restore_w_rec(model: ReservoirSNN, backup: torch.Tensor) -> None:
    model.W_rec.data.copy_(backup)


@torch.no_grad()
def evaluate_test_accuracy(model: ReservoirSNN, X_te: torch.Tensor, y_te: torch.Tensor,
                           batch_size: int, device: torch.device, *,
                           use_linear_readout: bool) -> float:
    """Match baseline training eval: ridge -> linear logits, BPTT -> output logits."""
    out_te, lin_te = collect_both_logits(model, X_te, batch_size, device)
    logits = lin_te if use_linear_readout else out_te
    pred = argmax_predict(logits, None)
    return accuracy(y_te.cpu().numpy().astype(np.int64), pred)


@torch.no_grad()
def sweep_w_rec_noise(model: ReservoirSNN, X_te: torch.Tensor, y_te: torch.Tensor, *,
                      mu_values: Sequence[float], batch_size: int, device: torch.device,
                      use_linear_readout: bool, seed: int = 0,
                      shared_noise: Optional[Sequence[Optional[torch.Tensor]]] = None
                      ) -> List[dict]:
    """Run test eval at each absolute noise level; restores ``W_rec`` before returning."""
    backup = model.W_rec.data.clone()
    rng = torch.Generator(device=backup.device)
    rng.manual_seed(seed)
    rows: List[dict] = []
    for i, mu in enumerate(mu_values):
        restore_w_rec(model, backup)
        if shared_noise is not None and shared_noise[i] is not None:
            apply_w_rec_noise(model, shared_noise[i])
        else:
            inject_w_rec_noise(model, float(mu), rng=rng)
        acc = evaluate_test_accuracy(
            model, X_te, y_te, batch_size, device, use_linear_readout=use_linear_readout)
        rows.append({"mu": float(mu), "test_acc": acc})
    restore_w_rec(model, backup)
    return rows
