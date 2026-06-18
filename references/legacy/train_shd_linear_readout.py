#!/usr/bin/env python3
"""BPTT vs Ridge with a SHARED bias-free linear hidden-average readout.

Run ONE method per process (``METHOD=bptt`` or ``METHOD=ridge``); compare the two
runs in Weights & Biases. Both methods share the identical readout and the
identical evaluation function, so the difference between two runs reflects only
HOW ``W_out`` is obtained:

    logits = mean_t(hidden_feature) @ W_out          # NO bias, NO output neuron
    active_logits = logits[:, ACTIVE_CLASSES]        # drop the held-out column

* ``METHOD=bptt``  -> ``W_in/W_rec/W_out`` (whichever are trainable) learned by BPTT
  on cross-entropy of ``active_logits``.
* ``METHOD=ridge`` -> ``W_in/W_rec`` stay at their seeded random init (reservoir),
  ``W_out`` is the closed-form, intercept-free ridge solution of
  ``mean_t(hidden) @ W_out ~= one-hot``.

NO BIAS ANYWHERE. Second-order hidden neuron (snnTorch ``Synaptic``); no output
neuron in Linear Readout. Per-layer trainability via ``TRAIN_W_IN/REC/OUT``; for a pure
readout comparison set ``TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1`` so both methods
use the identical frozen reservoir.

Reproducibility: every knob is env-overridable and logged to W&B (full config +
resolved constants + dataset manifest + library versions + a config hash
``run_id``). The trained checkpoint and its metadata are uploaded as a W&B
Artifact. A run can therefore be reconstructed from its W&B page alone:
re-export the logged config as environment variables and re-run.

GPU efficiency (RTX 5080 / Blackwell): hoisted input projection, GPU-resident
``uint8`` split caching, bf16 autocast, optional ``torch.compile`` + cuDNN
autotuning. Toggles: ``AMP``, ``GPU_CACHE``, ``TORCH_COMPILE``, ``DETERMINISTIC``.

Examples::

    SHD_PREPROCESSED_ROOT=data/class_incremental/shd_removed10_dt14ms METHOD=ridge \
      TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_linear_readout.py

    # Readout-only BPTT (frozen reservoir):
    SHD_PREPROCESSED_ROOT=data/class_incremental/shd_removed10_dt14ms METHOD=bptt \
      READOUT_FEATURE=hidden_spike_avg TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_linear_readout.py

    # Full BPTT (default TRAIN_W_IN=1 TRAIN_W_REC=1 TRAIN_W_OUT=1):
    SHD_PREPROCESSED_ROOT=... METHOD=bptt READOUT_FEATURE=hidden_spike_avg \
      python3 train_shd_linear_readout.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate


# =============================================================================
# Configuration (env-driven; manifest supplies the data-defining constants)
# =============================================================================

SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREPROCESSED_ROOT = os.environ.get("SHD_PREPROCESSED_ROOT", "data/class_incremental/shd_removed10_dt14ms").strip()
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts_linear_readout").strip()


def _read_manifest(root: str) -> dict:
    try:
        return json.loads((Path(root) / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_MANIFEST = _read_manifest(PREPROCESSED_ROOT)

# Data-defining constants: env override > manifest > default.
REMOVED_CLASS = int(os.environ.get("REMOVED_CLASS", _MANIFEST.get("removed_class", 10)))
NUM_INPUTS = int(os.environ.get("NUM_INPUTS", _MANIFEST.get("nb_inputs", 700)))
NUM_OUTPUTS = 20
DT_MS = float(os.environ.get("DT_MS", _MANIFEST.get("dt_ms", 14.0)))
MAX_TIME_SECONDS = float(os.environ.get("MAX_TIME_SECONDS", _MANIFEST.get("max_time_seconds", 1.4)))
NB_STEPS = int(_MANIFEST.get("nb_steps", int(math.ceil(MAX_TIME_SECONDS / (DT_MS / 1000.0)))))
ACTIVE_CLASSES = [c for c in range(NUM_OUTPUTS) if c != REMOVED_CLASS]

# Model.
NUM_HIDDEN = int(os.environ.get("NUM_HIDDEN", "1000"))
TAU_SYN_MS = float(os.environ.get("TAU_SYN_MS", "5.0"))
TAU_MEM_MS = float(os.environ.get("TAU_MEM_MS", "10.0"))
THRESHOLD = float(os.environ.get("THRESHOLD", "1.0"))
RESET_MECHANISM = os.environ.get("RESET_MECHANISM", "zero")
SURROGATE_GRADIENT_TYPE = os.environ.get("SURROGATE_GRADIENT_TYPE", "fast_sigmoid")
SURROGATE_SLOPE = float(os.environ.get("SURROGATE_SLOPE", "25.0"))
W_IN_SCALE = float(os.environ.get("W_IN_SCALE", "1.0"))
W_REC_SCALE = float(os.environ.get("W_REC_SCALE", "1.0"))
W_OUT_SCALE = float(os.environ.get("W_OUT_SCALE", "1.0"))
W_IN_INIT_STD = float(os.environ.get("W_IN_INIT_STD", str(W_IN_SCALE / math.sqrt(NUM_INPUTS))))
W_REC_INIT_STD = float(os.environ.get("W_REC_INIT_STD", str(W_REC_SCALE / math.sqrt(NUM_HIDDEN))))
W_OUT_INIT_STD = float(os.environ.get("W_OUT_INIT_STD", str(W_OUT_SCALE / math.sqrt(NUM_HIDDEN))))

ALPHA = math.exp(-(DT_MS / TAU_SYN_MS))
BETA = math.exp(-(DT_MS / TAU_MEM_MS))

TRAIN_W_IN = bool(int(os.environ.get("TRAIN_W_IN", "1")))
TRAIN_W_REC = bool(int(os.environ.get("TRAIN_W_REC", "1")))
TRAIN_W_OUT = bool(int(os.environ.get("TRAIN_W_OUT", "1")))

# Readout feature: which hidden trace is time-averaged before W_out.
READOUT_FEATURE = os.environ.get("READOUT_FEATURE", "hidden_spike_avg")
assert READOUT_FEATURE in ("hidden_spike_avg", "hidden_mem_avg")
_FEATURE_KEY = "hidden_spike_trace" if READOUT_FEATURE == "hidden_spike_avg" else "hidden_mem_trace"

# Training.
METHOD = os.environ.get("METHOD", "bptt").lower()           # bptt | ridge (one per run)
assert METHOD in ("bptt", "ridge")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
OPTIMIZER_TYPE = os.environ.get("OPTIMIZER_TYPE", "adamax").lower()
GRAD_CLIP_MAX_NORM = float(os.environ.get("GRAD_CLIP_MAX_NORM", "5.0"))
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1e-3"))

# Efficiency / reproducibility toggles.
DETERMINISTIC = bool(int(os.environ.get("DETERMINISTIC", "0")))
AMP = bool(int(os.environ.get("AMP", "1")))
AMP_DTYPE = torch.bfloat16
GPU_CACHE = bool(int(os.environ.get("GPU_CACHE", "1")))
TORCH_COMPILE = bool(int(os.environ.get("TORCH_COMPILE", "0")))
# CPU intra-op threads for this process (0 = leave PyTorch default). Set this when
# running several trainings in parallel so the per-process pools sum to <= #cores.
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))

# W&B.
WANDB_MODE = os.environ.get("WANDB_MODE", "online")          # online | offline | disabled
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "shd-linear-readout")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "") or None
READOUT_NAME = f"linear_mean_{READOUT_FEATURE}_no_bias"
RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    f"linear_readout_{METHOD}_{READOUT_FEATURE}_rm{REMOVED_CLASS}_dt{int(round(DT_MS))}ms",
)

# Offline JSON backup. Per-run ``result.json`` (full) + a shared, append-only,
# comparison-friendly JSONL (flock-guarded so parallel runs append safely).
RESULTS_JSONL = os.environ.get("RESULTS_JSONL", str(Path(ARTIFACTS_ROOT) / "results.jsonl"))


# =============================================================================
# Seeding & surrogate
# =============================================================================


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def make_surrogate(kind: str, slope: float):
    kind = (kind or "").lower()
    if kind == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=slope)
    if kind == "atan":
        return surrogate.atan(alpha=slope)
    if kind == "sigmoid":
        return surrogate.sigmoid(slope=slope)
    if kind in ("ste", "straight_through_estimator"):
        return surrogate.straight_through_estimator()
    raise ValueError(f"Unknown surrogate {kind!r}")


# =============================================================================
# Data: preprocessed pickles -> one GPU-resident uint8 tensor per split
# =============================================================================


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class SplitData:
    """Whole split as one dense ``uint8`` ``[N, T, C]`` tensor (+ labels), optionally on GPU."""

    def __init__(self, root: Path, split_name: str, device: torch.device, cache_on_gpu: bool) -> None:
        xb = _load_pickle(root / f"{split_name}_x_class_{REMOVED_CLASS}.pkl")
        yb = _load_pickle(root / f"{split_name}_y_class_{REMOVED_CLASS}.pkl")
        if len(xb) != len(yb):
            raise ValueError(f"{split_name}: x/y length mismatch")
        x_list, y_list = [], []
        for x, y in zip(xb, yb):
            x = x.to_dense() if x.is_sparse else x
            x = (x > 0).to(torch.uint8)
            if x.ndim != 3 or x.shape[1:] != (NB_STEPS, NUM_INPUTS):
                raise ValueError(f"{split_name}: bad shape {tuple(x.shape)}")
            x_list.append(x)
            y_list.append(torch.as_tensor(y).reshape(-1).long())
        self.x = torch.cat(x_list, dim=0).contiguous()
        self.y = torch.cat(y_list, dim=0).contiguous()
        if (self.y == REMOVED_CLASS).any() and not split_name.startswith("continual"):
            raise ValueError(f"{split_name}: removed class leaked into a pretrain split")
        self._store = device if cache_on_gpu else torch.device("cpu")
        self.x = self.x.to(self._store)
        self.y = self.y.to(self._store)
        self._device = device

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def batches(self, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None):
        n = len(self)
        idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        idx = idx.to(self.x.device)
        for s in range(0, n, batch_size):
            j = idx[s : s + batch_size]
            yield (self.x.index_select(0, j).to(self._device, non_blocking=True),
                   self.y.index_select(0, j).to(self._device, non_blocking=True))


# =============================================================================
# Model: reservoir SNN with NO output neuron (Linear readout from hidden neurons)
# =============================================================================


class ReservoirSNN(nn.Module):
    def __init__(self, collect_mem: bool) -> None:
        super().__init__()
        self.H = NUM_HIDDEN
        self.collect_mem = bool(collect_mem)
        W_in = torch.empty(NUM_INPUTS, NUM_HIDDEN)
        W_rec = torch.empty(NUM_HIDDEN, NUM_HIDDEN)
        W_out = torch.empty(NUM_HIDDEN, NUM_OUTPUTS)
        nn.init.normal_(W_in, 0.0, W_IN_INIT_STD)
        nn.init.normal_(W_rec, 0.0, W_REC_INIT_STD)
        nn.init.normal_(W_out, 0.0, W_OUT_INIT_STD)
        self.W_in = nn.Parameter(W_in, requires_grad=TRAIN_W_IN)
        self.W_rec = nn.Parameter(W_rec, requires_grad=TRAIN_W_REC)
        self.W_out = nn.Parameter(W_out, requires_grad=TRAIN_W_OUT)
        spike_grad = make_surrogate(SURROGATE_GRADIENT_TYPE, SURROGATE_SLOPE)
        self.hidden_neuron = snn.Synaptic(
            alpha=ALPHA, beta=BETA, threshold=THRESHOLD,
            spike_grad=spike_grad, reset_mechanism=RESET_MECHANISM,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.float()
        B, T, C = x.shape
        h_in = (x.reshape(B * T, C) @ self.W_in).reshape(B, T, self.H)   # hoisted projection
        syn = torch.zeros(B, self.H, device=x.device, dtype=h_in.dtype)
        mem = torch.zeros_like(syn)
        spk_prev = torch.zeros_like(syn)
        spikes: List[torch.Tensor] = []
        mems: List[torch.Tensor] = []
        for t in range(T):
            cur = h_in[:, t] + spk_prev @ self.W_rec
            spk, syn, mem = self.hidden_neuron(cur, syn, mem)
            spikes.append(spk)
            if self.collect_mem:
                mems.append(mem)
            spk_prev = spk
        out = {"hidden_spike_trace": torch.stack(spikes, dim=1)}
        if self.collect_mem:
            out["hidden_mem_trace"] = torch.stack(mems, dim=1)
        return out


# =============================================================================
# Shared readout / evaluation / label helpers
# =============================================================================

_ACTIVE_T: Optional[torch.Tensor] = None


def active_tensor(device: torch.device) -> torch.Tensor:
    global _ACTIVE_T
    if _ACTIVE_T is None or _ACTIVE_T.device != device:
        _ACTIVE_T = torch.as_tensor(ACTIVE_CLASSES, dtype=torch.long, device=device)
    return _ACTIVE_T


def readout_logits(model: ReservoirSNN, traces: Dict[str, torch.Tensor]) -> torch.Tensor:
    feat = traces[_FEATURE_KEY].mean(dim=1)        # [B, H]
    return feat @ model.W_out                      # [B, 20]; NO bias


def select_active(logits: torch.Tensor) -> torch.Tensor:
    return logits.index_select(1, active_tensor(logits.device))


def remap_to_active(labels: torch.Tensor) -> torch.Tensor:
    lm = torch.full((NUM_OUTPUTS,), -1, dtype=torch.long, device=labels.device)
    lm[active_tensor(labels.device)] = torch.arange(NUM_OUTPUTS - 1, device=labels.device)
    mapped = lm[labels.long()]
    if torch.any(mapped < 0):
        raise AssertionError("label outside active set (removed class leaked?)")
    return mapped


@torch.no_grad()
def evaluate(model: ReservoirSNN, split: SplitData, device: torch.device) -> Dict[str, float]:
    """The SINGLE evaluation function used for BOTH bptt and ridge."""
    model.eval()
    correct = total = 0
    spikes = 0.0
    sites = 0
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            traces = model(x)
            active = select_active(readout_logits(model, traces))
        pred = active_tensor(device)[active.float().argmax(dim=1)]
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        st = traces["hidden_spike_trace"]
        spikes += float(st.sum().item())
        sites += int(st.numel())
    return {"acc": correct / max(total, 1), "hidden_firing_rate": spikes / max(sites, 1)}


# =============================================================================
# BPTT training (shared readout)
# =============================================================================


def build_optimizer(model: ReservoirSNN) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters; set TRAIN_W_IN/REC/OUT.")
    if OPTIMIZER_TYPE == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE)
    if OPTIMIZER_TYPE == "adamax":
        return torch.optim.Adamax(params, lr=LEARNING_RATE)
    if OPTIMIZER_TYPE == "sgd":
        return torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
    raise ValueError(f"Unknown OPTIMIZER_TYPE={OPTIMIZER_TYPE!r}")


def _grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(sq)


def train_bptt(model: ReservoirSNN, train: SplitData, val: SplitData, device: torch.device,
               wandb_run) -> Dict[str, object]:
    optimizer = build_optimizer(model)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    trainable = [p for p in model.parameters() if p.requires_grad]
    best = {"val_acc": -1.0, "epoch": -1, "state": None}

    for epoch in range(NUM_EPOCHS):
        model.train()
        t0 = time.time()
        run_loss = run_acc = 0.0
        seen = 0
        gnorm_sum = 0.0
        nbatch = 0
        in_spk = in_sites = 0.0
        hid_spk = hid_sites = 0.0
        for x, y in train.batches(BATCH_SIZE, shuffle=True, generator=gen):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
                traces = model(x)
                active = select_active(readout_logits(model, traces))
                loss = loss_fn(active.float(), remap_to_active(y))
            loss.backward()
            gnorm_sum += _grad_norm(trainable)
            if GRAD_CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_MAX_NORM)
            optimizer.step()
            n = int(y.numel())
            run_loss += float(loss.item()) * n
            run_acc += float((active_tensor(device)[active.float().argmax(1)] == y).float().sum().item())
            seen += n
            nbatch += 1
            st = traces["hidden_spike_trace"]
            hid_spk += float(st.detach().sum().item()); hid_sites += int(st.numel())
            in_spk += float(x.float().sum().item()); in_sites += int(x.numel())

        val_m = evaluate(model, val, device)
        if val_m["acc"] > best["val_acc"]:
            best = {"val_acc": val_m["acc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        payload = {
            "epoch": epoch,
            "train/loss": run_loss / max(seen, 1),
            "train/acc": run_acc / max(seen, 1),
            "train/grad_norm_mean": gnorm_sum / max(nbatch, 1),
            "train/input_firing_rate": in_spk / max(in_sites, 1),
            "train/hidden_firing_rate": hid_spk / max(hid_sites, 1),
            "val/acc": val_m["acc"],
            "val/hidden_firing_rate": val_m["hidden_firing_rate"],
            "val/best_acc": best["val_acc"],
            "val/best_epoch": best["epoch"],
            "runtime/epoch_seconds": time.time() - t0,
        }
        if wandb_run is not None:
            wandb_run.log(payload)
        print(f"[bptt] epoch={epoch:03d} loss={payload['train/loss']:.4f} "
              f"train_acc={payload['train/acc']:.4f} val_acc={val_m['acc']:.4f} "
              f"best={best['val_acc']:.4f}@{best['epoch']} hz={val_m['hidden_firing_rate']:.3f} "
              f"({payload['runtime/epoch_seconds']:.1f}s)")
    if best["state"] is not None:
        model.load_state_dict(best["state"])  # restore best-val weights
    return {"best_val_acc": best["val_acc"], "best_epoch": best["epoch"]}


# =============================================================================
# Ridge training (closed form, NO bias) on the SAME readout feature
# =============================================================================


@torch.no_grad()
def extract_features(model: ReservoirSNN, split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        traces = model(x)
        feats.append(traces[_FEATURE_KEY].mean(dim=1).float())
        labels.append(y.to(device))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def solve_ridge_no_bias(X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    N = int(labels.numel())
    Y = torch.zeros((N, NUM_OUTPUTS - 1), dtype=torch.float64, device=X.device)
    Y[torch.arange(N, device=X.device), remap_to_active(labels)] = 1.0
    X64 = X.to(torch.float64)
    H = X64.shape[1]
    A = X64.T @ X64 + RIDGE_ALPHA * torch.eye(H, dtype=torch.float64, device=X.device)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(X64.T @ Y, L).to(X.dtype)          # [H,19]


def train_ridge(model: ReservoirSNN, train: SplitData, val: SplitData, device: torch.device,
                wandb_run) -> Dict[str, object]:
    model.requires_grad_(False)
    t0 = time.time()
    X, y = extract_features(model, train, device)
    W_active = solve_ridge_no_bias(X, y)
    W_full = torch.zeros((NUM_HIDDEN, NUM_OUTPUTS), dtype=W_active.dtype, device=W_active.device)
    W_full[:, active_tensor(W_active.device)] = W_active
    with torch.no_grad():
        model.W_out.copy_(W_full.to(model.W_out.device, model.W_out.dtype))
    train_m = evaluate(model, train, device)
    val_m = evaluate(model, val, device)
    info = {
        "ridge/alpha": RIDGE_ALPHA,
        "ridge/n_fit": int(y.numel()),
        "ridge/train_acc": train_m["acc"],
        "ridge/val_acc": val_m["acc"],
        "runtime/ridge_seconds": time.time() - t0,
    }
    if wandb_run is not None:
        wandb_run.log({**info, "epoch": 0})
    return info


# =============================================================================
# W&B config / init / artifacts
# =============================================================================


def _versions() -> Dict[str, object]:
    try:
        snn_ver = snn.__version__  # type: ignore[attr-defined]
    except Exception:
        snn_ver = "unknown"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "snntorch": snn_ver,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def build_config(train_n: int, val_n: int, test_n: int) -> Dict[str, object]:
    """Everything needed to reconstruct the run. Logged verbatim to W&B."""
    cfg = {
        "output_type": "linear_readout",
        "method": METHOD,
        "readout": READOUT_NAME,
        "readout_feature": READOUT_FEATURE,
        "seed": SEED,
        "device": DEVICE,
        # data
        "preprocessed_root": str(Path(PREPROCESSED_ROOT).resolve()),
        "removed_class": REMOVED_CLASS,
        "active_classes": ACTIVE_CLASSES,
        "num_inputs": NUM_INPUTS,
        "num_outputs": NUM_OUTPUTS,
        "dt_ms": DT_MS,
        "max_time_seconds": MAX_TIME_SECONDS,
        "nb_steps": NB_STEPS,
        "data_manifest": _MANIFEST,
        "n_train": train_n, "n_val": val_n, "n_test": test_n,
        # model
        "num_hidden": NUM_HIDDEN,
        "tau_syn_ms": TAU_SYN_MS, "tau_mem_ms": TAU_MEM_MS, "alpha": ALPHA, "beta": BETA,
        "threshold": THRESHOLD, "reset_mechanism": RESET_MECHANISM,
        "surrogate_gradient_type": SURROGATE_GRADIENT_TYPE, "surrogate_slope": SURROGATE_SLOPE,
        "w_in_init_std": W_IN_INIT_STD, "w_rec_init_std": W_REC_INIT_STD, "w_out_init_std": W_OUT_INIT_STD,
        "train_w_in": TRAIN_W_IN, "train_w_rec": TRAIN_W_REC, "train_w_out": TRAIN_W_OUT,
        # training
        "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimizer_type": OPTIMIZER_TYPE, "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "ridge_alpha": RIDGE_ALPHA,
        # efficiency / reproducibility
        "deterministic": DETERMINISTIC, "amp": AMP, "amp_dtype": "bfloat16",
        "gpu_cache": GPU_CACHE, "torch_compile": TORCH_COMPILE,
        "versions": _versions(),
    }
    cfg["run_id"] = "run_" + hashlib.sha1(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    # A copy-paste command that reproduces this run from the logged config alone.
    env_keys = [
        ("SHD_PREPROCESSED_ROOT", cfg["preprocessed_root"]), ("METHOD", METHOD),
        ("READOUT_FEATURE", READOUT_FEATURE), ("SEED", SEED),
        ("REMOVED_CLASS", REMOVED_CLASS), ("DT_MS", DT_MS), ("NUM_INPUTS", NUM_INPUTS),
        ("NUM_HIDDEN", NUM_HIDDEN), ("TAU_SYN_MS", TAU_SYN_MS), ("TAU_MEM_MS", TAU_MEM_MS),
        ("THRESHOLD", THRESHOLD), ("RESET_MECHANISM", RESET_MECHANISM),
        ("SURROGATE_GRADIENT_TYPE", SURROGATE_GRADIENT_TYPE), ("SURROGATE_SLOPE", SURROGATE_SLOPE),
        ("W_IN_INIT_STD", W_IN_INIT_STD), ("W_REC_INIT_STD", W_REC_INIT_STD), ("W_OUT_INIT_STD", W_OUT_INIT_STD),
        ("TRAIN_W_IN", int(TRAIN_W_IN)), ("TRAIN_W_REC", int(TRAIN_W_REC)), ("TRAIN_W_OUT", int(TRAIN_W_OUT)),
        ("NUM_EPOCHS", NUM_EPOCHS), ("BATCH_SIZE", BATCH_SIZE), ("LEARNING_RATE", LEARNING_RATE),
        ("OPTIMIZER_TYPE", OPTIMIZER_TYPE), ("GRAD_CLIP_MAX_NORM", GRAD_CLIP_MAX_NORM),
        ("RIDGE_ALPHA", RIDGE_ALPHA), ("DETERMINISTIC", int(DETERMINISTIC)),
    ]
    cfg["reproduce_command"] = " ".join(f"{k}={v}" for k, v in env_keys) + \
        " python3 train_shd_linear_readout.py"
    return cfg


def init_wandb(cfg: Dict[str, object]):
    if WANDB_MODE == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME,
                     config=cfg, mode=WANDB_MODE,
                     tags=["shd", "linear-readout", METHOD, READOUT_FEATURE])
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("ridge/*", step_metric="epoch")
        run.define_metric("runtime/*", step_metric="epoch")
    return run


def dump_results_offline(cfg: Dict[str, object], final: Dict[str, object], ckpt_path: Path) -> None:
    """Offline backup independent of W&B: a full per-run ``result.json`` plus a
    compact, append-only shared JSONL for easy cross-run comparison."""
    rec = {
        "run_id": cfg["run_id"], "run_name": RUN_NAME, "output_type": cfg["output_type"],
        "method": METHOD, "readout": cfg["readout"], "readout_feature": READOUT_FEATURE,
        "seed": SEED, "removed_class": REMOVED_CLASS, "dt_ms": DT_MS, "nb_steps": NB_STEPS,
        "num_hidden": NUM_HIDDEN,
        "train_w_in": TRAIN_W_IN, "train_w_rec": TRAIN_W_REC, "train_w_out": TRAIN_W_OUT,
        "val_acc": final.get("val/acc"), "test_acc": final.get("test/acc"),
        "best_val_acc": final.get("best_val_acc"), "best_epoch": final.get("best_epoch"),
        "val_hidden_firing_rate": final.get("val/hidden_firing_rate"),
        "test_hidden_firing_rate": final.get("test/hidden_firing_rate"),
        "preprocessed_root": cfg["preprocessed_root"], "checkpoint": str(ckpt_path),
        "reproduce_command": cfg["reproduce_command"],
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ckpt_path.parent / "result.json").write_text(
        json.dumps({**rec, "config": cfg, "metrics": final}, indent=2, default=str))
    p = Path(RESULTS_JSONL)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    with open(p, "a") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line); f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            f.write(line)
    print(f"results appended to {p}")


def save_and_log_checkpoint(model: ReservoirSNN, cfg: Dict[str, object],
                            metrics: Dict[str, object], wandb_run) -> Path:
    out = Path(ARTIFACTS_ROOT) / str(cfg["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    meta = out / "metadata.json"
    torch.save({"state_dict": model.state_dict(), "config": cfg, "metrics": metrics,
                "saved_at_utc": datetime.now(timezone.utc).isoformat()}, ckpt)
    meta.write_text(json.dumps({"config": cfg, "metrics": metrics}, indent=2, default=str))
    if wandb_run is not None:
        import wandb
        art = wandb.Artifact(f"{RUN_NAME}-{cfg['run_id']}", type="model",
                             metadata={"metrics": metrics, "readout": READOUT_NAME, "method": METHOD})
        art.add_file(str(ckpt)); art.add_file(str(meta))
        man = Path(PREPROCESSED_ROOT) / "preprocessing_manifest.json"
        if man.is_file():
            art.add_file(str(man))
        wandb_run.log_artifact(art)
    return ckpt


# =============================================================================
# Entry point (single method per run)
# =============================================================================


def main() -> int:
    set_seeds(SEED)
    if TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)
    device = torch.device(DEVICE)
    print(f"device={device} method={METHOD} feature={READOUT_FEATURE} "
          f"train(in,rec,out)=({TRAIN_W_IN},{TRAIN_W_REC},{TRAIN_W_OUT}) "
          f"AMP={AMP} GPU_CACHE={GPU_CACHE} compile={TORCH_COMPILE} "
          f"nb_steps={NB_STEPS} alpha={ALPHA:.4f} beta={BETA:.4f}")

    root = Path(PREPROCESSED_ROOT)
    train = SplitData(root, "pretrain_train", device, GPU_CACHE)
    val = SplitData(root, "pretrain_val", device, GPU_CACHE)
    test = SplitData(root, "pretrain_test", device, GPU_CACHE)
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    cfg = build_config(len(train), len(val), len(test))
    wandb_run = init_wandb(cfg)
    print(f"run_id={cfg['run_id']} run_name={RUN_NAME}")

    model = ReservoirSNN(collect_mem=(_FEATURE_KEY == "hidden_mem_trace")).to(device)
    if TORCH_COMPILE:
        model = torch.compile(model)
    real = model._orig_mod if TORCH_COMPILE else model

    with torch.no_grad():
        x0, _ = next(iter(train.batches(min(BATCH_SIZE, len(train)), shuffle=False)))
        hz = float(real(x0.to(device))["hidden_spike_trace"].float().mean().item())
    print(f"[smoke] initial hidden firing rate: {hz:.4f}")
    if not (0.001 <= hz <= 0.6):
        print(f"WARNING: hidden firing rate {hz:.4f} is extreme; consider tuning "
              f"THRESHOLD / W_IN_INIT_STD (continuing anyway).")
    if wandb_run is not None:
        wandb_run.summary["initial_hidden_firing_rate"] = hz

    start = time.time()
    if METHOD == "bptt":
        info = train_bptt(model, train, val, device, wandb_run)
    else:
        info = train_ridge(real, train, val, device, wandb_run)

    val_m = evaluate(model, val, device)
    test_m = evaluate(model, test, device)
    final = {**info,
             "val/acc": val_m["acc"], "val/hidden_firing_rate": val_m["hidden_firing_rate"],
             "test/acc": test_m["acc"], "test/hidden_firing_rate": test_m["hidden_firing_rate"],
             "runtime/total_seconds": time.time() - start}
    print(f"\n=== {METHOD.upper()} ===  val_acc={val_m['acc']:.4f}  test_acc={test_m['acc']:.4f}  "
          f"hidden_hz(test)={test_m['hidden_firing_rate']:.3f}")

    ckpt = save_and_log_checkpoint(real, cfg, final, wandb_run)
    print(f"checkpoint: {ckpt}")
    dump_results_offline(cfg, final, ckpt)
    if wandb_run is not None:
        for k, v in final.items():
            wandb_run.summary[k] = v
        wandb_run.summary["artifact_dir"] = str(ckpt.parent.resolve())
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
