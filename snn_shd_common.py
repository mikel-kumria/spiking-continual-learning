"""Shared helpers for the two-stage SHD continual-learning pipeline.

This module is the single source of truth for everything that
``pretrain_snn_shd.py`` (Stage 1) and ``class_incremental_snn_shd.py``
(Stage 2) have in common, so the two scripts cannot drift apart.

The recurrent SNN architecture itself is **imported from**
``train_snn_shd.py`` rather than re-implemented here, which guarantees that
the model pretrained in Stage 1 and the model adapted in Stage 2 share the
*exact* same dynamics:

    input [N, T, C]
      -> W_in
      -> single recurrent hidden layer (second-order LIF), W_rec
      -> hidden spikes over time
      -> Phi = mean_t(hidden_spikes)          # shape [B, H]
      -> logits = Phi @ W_out                 # shape [B, nb_outputs]

Importing ``train_snn_shd`` is side-effect free: its ``main()`` only runs
under ``if __name__ == "__main__"``.

Canonical data-shape contract (NEVER transposed):
    X : uint8/float array [N, T, C]   (C = channel = last axis)
    y : int64 array       [N]         (original SHD labels in [0, 19])
    speaker : int64 array [N]         (or all -1 if unavailable)
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# The model is the canonical one defined in train_snn_shd.py. Re-export it so
# both new scripts build/reload the identical architecture.
from train_snn_shd import ReservoirSNN, SurrGradSpike  # noqa: F401  (re-export)

NUM_CLASSES = 20  # SHD: spoken digits 0-9 in English + German


# =============================================================================
# Runtime: device + determinism
# =============================================================================


def resolve_device(choice: str) -> torch.device:
    """Mirror of ``train_snn_shd.resolve_device`` (auto/cpu/cuda)."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_determinism(seed: int) -> None:
    """Seed Python/NumPy/Torch RNGs for reproducible splits, init and streams."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Channel compression (channel axis only) -- ported from
# references/legacy/build_shd_compressed_dataset.py and
# dataset_preprocessing/shd_channel_compression.ipynb. Operates on NumPy so it
# composes with .npz output. Time (axis -2) is NEVER touched.
# =============================================================================


def validate_compression_factor(nb_inputs: int, n_compressed: int) -> int:
    """Return the integer compression factor or raise a clear ``ValueError``.

    ``n_compressed`` must divide ``nb_inputs`` exactly (contiguous equal-sized
    channel groups); we never silently truncate or pad.
    """
    if n_compressed <= 0:
        raise ValueError(f"n_compressed_channels must be positive, got {n_compressed}")
    if n_compressed > nb_inputs:
        raise ValueError(
            f"n_compressed_channels={n_compressed} cannot exceed nb_inputs={nb_inputs}")
    if nb_inputs % n_compressed != 0:
        divisors = [d for d in range(1, nb_inputs + 1)
                    if nb_inputs % d == 0 and d <= 100]
        raise ValueError(
            f"n_compressed_channels={n_compressed} must divide nb_inputs={nb_inputs} "
            f"exactly (remainder {nb_inputs % n_compressed}); choose a divisor of "
            f"{nb_inputs} (e.g. {divisors} ...).")
    return nb_inputs // n_compressed


def _group_count(x: np.ndarray, factor: int) -> np.ndarray:
    """``[..., C] -> [..., C//factor]`` spike count per adjacent channel group."""
    *lead, c = x.shape
    if c % factor != 0:
        raise ValueError(f"channel count {c} not divisible by factor {factor}")
    target = c // factor
    return x.reshape(*lead, target, factor).sum(axis=-1, dtype=np.uint16)


def compress_channels(x: np.ndarray, method: str, factor: int, *,
                      condition_or: int = 1,
                      rng: Optional[np.random.Generator] = None,
                      bernoulli_seed: int = 42) -> np.ndarray:
    """Compress the channel axis of ``x`` ([..., C]) by ``factor``.

    Methods (all reduce ONLY the last axis):
      * ``or_pool`` / ``conditional_or`` : binary, group fires if >= ``condition_or``
        spikes land in it (``or_pool`` == ``conditional_or`` with threshold 1).
      * ``graded``  : integer count per group (spike-count preserving).
      * ``bernoulli``: binary, group fires with prob count/factor (seeded).
    ``factor == 1`` is an identity copy. Output is uint8 (graded may exceed 1).
    """
    if factor == 1:
        return x.astype(np.uint8, copy=True)
    counts = _group_count(x, factor)  # [..., C//factor], uint16
    if method in ("or_pool", "conditional_or"):
        thr = max(1, int(condition_or))
        return (counts >= thr).astype(np.uint8)
    if method == "graded":
        cap = int(counts.max(initial=0))
        dtype = np.uint8 if cap <= 255 else np.uint16
        return counts.astype(dtype)
    if method == "bernoulli":
        if rng is None:
            rng = np.random.default_rng(bernoulli_seed)
        p = np.clip(counts.astype(np.float32) / float(factor), 0.0, 1.0)
        return (rng.random(p.shape, dtype=np.float32) < p).astype(np.uint8)
    raise ValueError(
        f"unknown channel_compression_method={method!r}; "
        f"available: {sorted(COMPRESSION_METHODS)}")


COMPRESSION_METHODS = ("or_pool", "conditional_or", "graded", "bernoulli")


def assert_compression_invariants(x_in: np.ndarray, x_out: np.ndarray,
                                  method: str, factor: int) -> None:
    """Cheap shape/value invariants that also guard against a T<->C transpose."""
    assert x_out.shape[-2] == x_in.shape[-2], (
        f"{method}: time axis changed {x_in.shape} -> {x_out.shape}")
    assert x_out.shape[-1] * factor == x_in.shape[-1], (
        f"{method}: channel axis not reduced by factor {factor}")
    assert x_out.shape[:-1] == x_in.shape[:-1], f"{method}: leading dims changed"
    assert int(x_out.min()) >= 0, f"{method}: negative values"
    if method in ("or_pool", "conditional_or", "bernoulli"):
        assert set(np.unique(x_out)).issubset({0, 1}), f"{method} must be binary"
    if method == "graded":
        assert int(x_out.sum()) == int(x_in.sum()), "graded must preserve spike count"
        assert int(x_out.max(initial=0)) <= factor, "graded value cannot exceed factor"


# =============================================================================
# .npz dataset IO (canonical format for the new scripts)
# =============================================================================

SPLIT_NAMES = (
    "pretrain_train", "pretrain_val", "pretrain_test",
    "continual_train", "continual_val", "continual_test",
)


def save_npz_split(path: str, X: np.ndarray, y: np.ndarray,
                   speaker: Optional[np.ndarray] = None) -> None:
    """Persist one split as ``X uint8 [N,T,C]``, ``y int64 [N]``, ``speaker int64``."""
    assert X.ndim == 3, f"expected X [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], f"X/y length mismatch {X.shape[0]} vs {y.shape[0]}"
    if speaker is None:
        speaker = np.full((X.shape[0],), -1, dtype=np.int64)
    np.savez_compressed(
        path,
        X=np.ascontiguousarray(X).astype(np.uint8, copy=False),
        y=np.ascontiguousarray(y).astype(np.int64, copy=False),
        speaker=np.ascontiguousarray(speaker).astype(np.int64, copy=False),
    )


def load_npz_split(path: str, limit: int = 0
                   ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load one ``.npz`` split -> (X float32 [N,T,C], y long [N], speaker int64 [N])."""
    if not os.path.isfile(path):
        raise SystemExit(f"dataset file not found: {path}")
    d = np.load(path)
    if "X" not in d or "y" not in d:
        raise SystemExit(f"{path} must contain 'X' and 'y' (found {list(d.keys())})")
    X = d["X"]
    y = d["y"]
    speaker = d["speaker"] if "speaker" in d else np.full((y.shape[0],), -1, np.int64)
    if limit and limit > 0:
        X, y, speaker = X[:limit], y[:limit], speaker[:limit]
    assert X.ndim == 3, f"expected X [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], f"X/y length mismatch in {path}"
    Xf = torch.from_numpy(np.ascontiguousarray(X)).float()
    yl = torch.from_numpy(np.ascontiguousarray(y)).long()
    return Xf, yl, np.ascontiguousarray(speaker).astype(np.int64)


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# =============================================================================
# Neuron decay constants
# =============================================================================


def derive_alpha_beta(dt_ms: float, tau_mem_ms: float, tau_syn_ms: float
                      ) -> Tuple[float, float]:
    """``alpha = exp(-dt/tau_syn)``, ``beta = exp(-dt/tau_mem)`` (all in ms).

    Same convention as ``train_snn_shd.py``.
    """
    alpha = math.exp(-dt_ms / tau_syn_ms)
    beta = math.exp(-dt_ms / tau_mem_ms)
    return alpha, beta


# =============================================================================
# Feature extraction, prediction and metrics (shared readout = mean-spike Phi)
# =============================================================================


@torch.no_grad()
def collect_features(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                     device: torch.device) -> torch.Tensor:
    """Phi = mean_t(hidden_spikes) for every sample -> ``[N, H]`` (on CPU)."""
    model.eval()
    feats = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        Phi, _ = model.hidden_spikes(xb)
        feats.append(Phi.cpu())
    if not feats:
        return torch.zeros((0, model.nb_hidden))
    return torch.cat(feats, 0)


@torch.no_grad()
def predict(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
            device: torch.device) -> np.ndarray:
    """20-way ``argmax(logits)`` predictions over a split -> int64 ``[N]``."""
    model.eval()
    preds = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu())
    if not preds:
        return np.zeros((0,), dtype=np.int64)
    return torch.cat(preds, 0).numpy().astype(np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy; empty -> NaN (so 'no samples' is not reported as 0)."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int = NUM_CLASSES
                       ) -> Dict[int, Optional[float]]:
    """Per-class top-1 accuracy over ``[0, num_classes)``.

    Classes with **no** samples are reported as ``None`` (JSON null / NaN), never
    silently as 0.0.
    """
    out: Dict[int, Optional[float]] = {}
    for c in range(num_classes):
        mask = (y_true == c)
        n = int(mask.sum())
        out[c] = float(np.mean(y_pred[mask] == c)) if n > 0 else None
    return out


def evaluate_split(model: ReservoirSNN, X: torch.Tensor, y: torch.Tensor,
                   batch_size: int, device: torch.device) -> Tuple[float, np.ndarray]:
    """Return ``(accuracy, y_pred)`` for a split via the 20-way readout."""
    y_true = y.cpu().numpy().astype(np.int64)
    y_pred = predict(model, X, batch_size, device)
    return accuracy(y_true, y_pred), y_pred


# =============================================================================
# Checkpoints
# =============================================================================


def build_checkpoint(model: ReservoirSNN, *, dt_ms: float, tau_mem_ms: float,
                     tau_syn_ms: float, threshold: float, weight_scale: float,
                     surrogate_slope: float, active_classes: Sequence[int],
                     removed_class: int, pretraining_mode: str,
                     pretraining_metrics: dict, dataset_dir: str, config: dict,
                     removed_class_init_policy: str = "zero") -> dict:
    """Assemble the self-describing checkpoint dict (Stage 1 -> Stage 2 contract)."""
    return {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_class": "ReservoirSNN",
        "architecture": {
            "nb_inputs": int(model.nb_inputs),
            "nb_hidden": int(model.nb_hidden),
            "nb_outputs": int(model.nb_outputs),
            "alpha": float(model.alpha),
            "beta": float(model.beta),
            "tau_mem_ms": float(tau_mem_ms),
            "tau_syn_ms": float(tau_syn_ms),
            "threshold": float(threshold),
            "weight_scale": float(weight_scale),
            "surrogate_slope": float(surrogate_slope),
            "dt_ms": float(dt_ms),
        },
        "active_classes": [int(c) for c in active_classes],
        "removed_class": int(removed_class),
        "removed_class_init_policy": removed_class_init_policy,
        "pretraining_mode": pretraining_mode,
        "pretraining_metrics": pretraining_metrics,
        "dataset_dir": dataset_dir,
        "config": config,
    }


def load_checkpoint_model(ckpt: dict, device: torch.device) -> ReservoirSNN:
    """Rebuild a ``ReservoirSNN`` from a checkpoint and load its weights exactly.

    Crucially this loads the *trained* (or frozen-but-rescaled) reservoir, never
    a fresh random one -- so fullbptt's learned W_in/W_rec and ridge's rescaled
    random W_in/W_rec are both reproduced bit-for-bit.
    """
    a = ckpt["architecture"]
    model = ReservoirSNN(
        nb_inputs=int(a["nb_inputs"]), nb_hidden=int(a["nb_hidden"]),
        nb_outputs=int(a["nb_outputs"]), alpha=float(a["alpha"]),
        beta=float(a["beta"]), threshold=float(a["threshold"]),
        weight_scale=float(a["weight_scale"]),
        surrogate_slope=float(a["surrogate_slope"]),
    ).to(device)
    state = {k: v.to(device) for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    return model


# =============================================================================
# Recursive Least Squares (online closed-form readout adaptation)
# =============================================================================


class RLS:
    """Recursive Least Squares update for the linear read-out ``W_out``.

    Operates on the SAME feature the ridge pretraining uses:
    ``Phi = mean_t(hidden_spikes)`` of shape ``[H]`` per sample. Updates only
    ``W`` (``[H, nb_outputs]``); the reservoir (``W_in``/``W_rec``) is frozen and
    never touched.

    Conventions
    -----------
    * ``P`` is the inverse-correlation matrix, ``[H, H]``, in **float64** for
      numerical stability; it is initialised to ``I / delta`` so that ``delta``
      plays exactly the role of the ridge regulariser ``lambda`` used during
      pretraining (delta = lambda => RLS with forgetting 1.0 converges to the
      same closed-form ridge solution incrementally).
    * ``lambda_forgetting`` in (0, 1] is the exponential forgetting factor; 1.0
      means "remember everything in the stream".

    Per-sample update (matches the requested algebra)::

        denom = lambda_forgetting + phi^T P phi
        k     = (P phi) / denom
        err   = target - W^T phi            # [nb_outputs]
        W     = W + outer(k, err)           # rank-1
        P     = (P - outer(k, phi^T P)) / lambda_forgetting
    """

    def __init__(self, W_init: torch.Tensor, delta: float = 1.0,
                 lambda_forgetting: float = 1.0):
        assert W_init.ndim == 2, f"W must be [H, nb_outputs], got {tuple(W_init.shape)}"
        if not delta > 0:
            raise ValueError(f"rls delta must be > 0, got {delta}")
        if not 0.0 < lambda_forgetting <= 1.0:
            raise ValueError(
                f"rls forgetting factor must be in (0, 1], got {lambda_forgetting}")
        self.H, self.nb_outputs = int(W_init.shape[0]), int(W_init.shape[1])
        self.delta = float(delta)
        self.lam = float(lambda_forgetting)
        # Work in float64 on CPU; copy back to the model dtype/device on demand.
        self.W = W_init.detach().to("cpu", torch.float64).clone()
        self.P = torch.eye(self.H, dtype=torch.float64) / self.delta

    def update(self, phi: torch.Tensor, target: torch.Tensor) -> None:
        """One rank-1 RLS step. ``phi`` is ``[H]``, ``target`` is ``[nb_outputs]``."""
        phi = phi.reshape(self.H, 1).to(torch.float64)            # [H, 1]
        target = target.reshape(self.nb_outputs, 1).to(torch.float64)  # [O, 1]
        Pphi = self.P @ phi                                       # [H, 1]
        denom = self.lam + float((phi.T @ Pphi).item())          # scalar
        if not math.isfinite(denom) or abs(denom) < 1e-12:
            raise FloatingPointError(f"RLS denominator unstable: {denom}")
        k = Pphi / denom                                         # [H, 1]
        err = target - (self.W.T @ phi)                          # [O, 1]
        self.W = self.W + k @ err.T                              # [H, O] rank-1
        self.P = (self.P - k @ (phi.T @ self.P)) / self.lam      # [H, H]
        # Keep P symmetric (counter float drift) without changing the math.
        self.P = 0.5 * (self.P + self.P.T)
        if not torch.isfinite(self.W).all():
            raise FloatingPointError("RLS produced non-finite W_out")
        if not torch.isfinite(self.P).all():
            raise FloatingPointError("RLS produced non-finite P")

    def run_stream(self, Phi: torch.Tensor, Y_onehot: torch.Tensor) -> None:
        """Apply ``update`` over a stream of features ``[N,H]`` / targets ``[N,O]``."""
        assert Phi.shape[0] == Y_onehot.shape[0], "stream length mismatch"
        for i in range(Phi.shape[0]):
            self.update(Phi[i], Y_onehot[i])

    def copy_into(self, model: ReservoirSNN) -> None:
        """Write the learned ``W`` back into ``model.W_out`` (dtype/device safe)."""
        with torch.no_grad():
            model.W_out.copy_(self.W.to(model.W_out.device, model.W_out.dtype))


def one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Float64 one-hot targets ``[N, num_classes]`` over original labels."""
    Y = torch.zeros((len(y), num_classes), dtype=torch.float64)
    if len(y):
        Y[torch.arange(len(y)), torch.from_numpy(np.asarray(y)).long()] = 1.0
    return Y
