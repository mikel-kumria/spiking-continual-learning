"""Closed-form (optionally class-balanced weighted) ridge regression of ``W_out``.

Unweighted objective (no bias)::

    min_W || X W - Y ||_F^2 + lambda ||W||_F^2
    =>  W = (X^T X + lambda I)^-1 X^T Y

Weighted objective with per-sample weights ``w_i`` (``D = diag(w)``)::

    min_W || sqrt(D)(X W - Y) ||_F^2 + lambda ||W||_F^2
    =>  W = (X^T D X + lambda I)^-1 X^T D Y

Implemented in the numerically stable ``sqrt(w)`` form: pre-scale the rows of
``X`` and ``Y`` by ``sqrt(w_i)`` and solve the ordinary normal equations. The
solve is done in float64 on CPU (MPS has no float64).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import NUM_CLASSES

RIDGE_WEIGHTING_MODES = ("none", "inverse_class_count", "normalized_inverse_class_count")


def compute_sample_weights(y: np.ndarray, mode: str) -> Tuple[np.ndarray, int]:
    """Per-sample weights for weighted ridge -> ``(weights [N], n_present_classes)``.

    * ``none``                          : w_i = 1
    * ``inverse_class_count``           : w_i = 1 / n_class[y_i]   (each class total 1)
    * ``normalized_inverse_class_count``: w_i = N / (C * n_class[y_i])
      (each class equal total weight, mean sample weight ~ 1 -> lambda comparable
      across replay ratios). This is the CIL default.
    """
    y = np.asarray(y)
    if mode not in RIDGE_WEIGHTING_MODES:
        raise ValueError(f"unknown ridge_weighting {mode!r}; choose {RIDGE_WEIGHTING_MODES}")
    classes, counts = np.unique(y, return_counts=True)
    n_present = int(len(classes))
    count_of = {int(c): int(n) for c, n in zip(classes, counts)}
    N = int(len(y))
    if mode == "none":
        w = np.ones(N, dtype=np.float64)
    elif mode == "inverse_class_count":
        w = np.array([1.0 / count_of[int(v)] for v in y], dtype=np.float64)
    else:  # normalized_inverse_class_count
        w = np.array([N / (n_present * count_of[int(v)]) for v in y], dtype=np.float64)
    return w, n_present


def one_hot(y: np.ndarray, columns: Sequence[int]) -> torch.Tensor:
    """Float64 one-hot ``[N, len(columns)]`` where column j corresponds to labels
    ``columns[j]``. Labels not in ``columns`` must not appear."""
    col_of = {int(c): j for j, c in enumerate(columns)}
    Y = torch.zeros((len(y), len(columns)), dtype=torch.float64)
    for i, v in enumerate(np.asarray(y).tolist()):
        Y[i, col_of[int(v)]] = 1.0
    return Y


def solve_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float,
                sample_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """Solve (weighted) ridge -> ``(W [H,K], info)``. All math in float64 on CPU."""
    Xd = X.detach().to("cpu", torch.float64)
    Yd = Y.detach().to("cpu", torch.float64)
    H = Xd.shape[1]
    if sample_weight is not None:
        sw = sample_weight.detach().to("cpu", torch.float64).clamp_min(0.0)
        sqrt_w = torch.sqrt(sw)[:, None]                 # [N,1]
        Xw = Xd * sqrt_w
        Yw = Yd * sqrt_w
    else:
        Xw, Yw = Xd, Yd
    A = Xw.T @ Xw + float(lam) * torch.eye(H, dtype=torch.float64)
    B = Xw.T @ Yw
    solve_status = "cholesky"
    try:
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(B, L)
    except RuntimeError:
        solve_status = "lstsq_fallback"
        W = torch.linalg.solve(A, B)
    info = {"solve_status": solve_status, "lambda": float(lam),
            "H": int(H), "K": int(Y.shape[1])}
    try:  # condition number is cheap-ish and useful provenance
        info["cond_number"] = float(torch.linalg.cond(A).item())
    except RuntimeError:
        info["cond_number"] = None
    return W, info


def fit_readout(spike_sum: torch.Tensor, y: np.ndarray, *, columns: Sequence[int],
                nb_outputs: int, lam: float, weighting: str = "none",
                zero_columns: Optional[Sequence[int]] = None
                ) -> Tuple[torch.Tensor, dict]:
    """Fit ``W_out [H, nb_outputs]`` by ridge on the given target ``columns``.

    * ``columns`` are the ORIGINAL label ids the readout is trained for (e.g. the
      19 active classes for pretraining, or all 20 for baseline/CIL).
    * Targets are one-hot over ``columns``; the solved ``[H, len(columns)]`` block is
      scattered into the full ``[H, nb_outputs]`` matrix. Any ``zero_columns`` (e.g.
      the removed class during 19-class pretraining) are forced to exactly 0.
    """
    weights_np, n_present = compute_sample_weights(y, weighting)
    sw = torch.from_numpy(weights_np) if weighting != "none" else None
    Y = one_hot(y, columns)
    Wk, info = solve_ridge(spike_sum, Y, lam, sample_weight=sw)
    H = spike_sum.shape[1]
    W_full = torch.zeros((H, nb_outputs), dtype=torch.float64)
    col_idx = torch.tensor(list(columns), dtype=torch.long)
    W_full[:, col_idx] = Wk
    if zero_columns:
        for c in zero_columns:
            W_full[:, int(c)] = 0.0
    info.update({"ridge_weighting": weighting, "n_present_classes": n_present,
                 "columns": [int(c) for c in columns],
                 "zeroed_columns": [int(c) for c in (zero_columns or [])]})
    return W_full, info


def apply_readout(model, W_full: torch.Tensor) -> None:
    """Copy a solved ``[H, nb_outputs]`` readout into ``model.W_out``."""
    with torch.no_grad():
        model.W_out.copy_(W_full.to(model.W_out.device, model.W_out.dtype))
