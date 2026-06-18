#!/usr/bin/env python3
"""BPTT vs Ridge through the SECOND-ORDER OUTPUT LIF head.

Run ONE method per process (``METHOD=bptt`` or ``METHOD=ridge``); compare runs in
Weights & Biases. Both methods keep the spiking output head: hidden spikes are
projected by ``W_out`` and integrated by a non-spiking second-order ``Synaptic``
output neuron. Each run is scored by ITS OWN readout reduction and labelled
accordingly — a ``ce_max_mem`` run is evaluated/selected by the MAX of the output
membrane and is never relabelled as the mean metric.

    logits = REDUCE_t(output_mem_trace)              # REDUCE in {mean, max}; NO bias
    active_logits = logits[:, ACTIVE_CLASSES]

Readout label per run (config ``readout`` and the run name):
  * ``METHOD=bptt`` + ``BPTT_REDUCE=mean`` -> ``ce_mean_mem``  (eval/select: mean)
  * ``METHOD=bptt`` + ``BPTT_REDUCE=max``  -> ``ce_max_mem``   (eval/select: max)
  * ``METHOD=ridge``                       -> ``ridge_output_mean`` (eval: mean)

Ridge fits ``W_out`` with NO intercept in one closed-form solve:
``||X W - Y||^2 + lambda||W||^2``, where ``X`` is built from cached hidden traces and the
output LIF temporal response (equivalent to mean output membrane with ``h_t @ W_out``).
Compare ridge against ``ce_mean_mem`` BPTT (same ``READOUT_FEATURE``); ``ce_max_mem``
is a different objective.

NO BIAS ANYWHERE. Per-layer trainability via ``TRAIN_W_IN/REC/OUT``. Full config,
dataset manifest, versions and the checkpoint are logged to W&B for reproduction.

GPU efficiency (RTX 5080 / Blackwell): hoisted input projection, GPU-resident
``uint8`` split caching, bf16 autocast, optional ``torch.compile`` + cuDNN
autotuning. Toggles: ``AMP``, ``GPU_CACHE``, ``TORCH_COMPILE``, ``DETERMINISTIC``.

Examples::

    SHD_PREPROCESSED_ROOT=/data/class_incremental/shd_removed10_dt14ms METHOD=ridge \
      READOUT_FEATURE=hidden_spike_avg TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_output_lif.py

    SHD_PREPROCESSED_ROOT=/data/class_incremental/shd_removed10_dt14ms METHOD=ridge \
      READOUT_FEATURE=hidden_mem_avg TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_output_lif.py

    SHD_PREPROCESSED_ROOT=/data/class_incremental/shd_removed10_dt14ms METHOD=bptt BPTT_REDUCE=max \
      TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_output_lif.py        # readout BPTT on max output membrane
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
# Configuration
# =============================================================================

SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREPROCESSED_ROOT = os.environ.get("SHD_PREPROCESSED_ROOT", "data/class_incremental/shd_removed10_dt14ms").strip()
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts_output_lif").strip()


def _read_manifest(root: str) -> dict:
    try:
        return json.loads((Path(root) / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_MANIFEST = _read_manifest(PREPROCESSED_ROOT)

REMOVED_CLASS = int(os.environ.get("REMOVED_CLASS", _MANIFEST.get("removed_class", 10)))
NUM_INPUTS = int(os.environ.get("NUM_INPUTS", _MANIFEST.get("nb_inputs", 700)))
NUM_OUTPUTS = 20
DT_MS = float(os.environ.get("DT_MS", _MANIFEST.get("dt_ms", 14.0)))
MAX_TIME_SECONDS = float(os.environ.get("MAX_TIME_SECONDS", _MANIFEST.get("max_time_seconds", 1.4)))
NB_STEPS = int(_MANIFEST.get("nb_steps", int(math.ceil(MAX_TIME_SECONDS / (DT_MS / 1000.0)))))
ACTIVE_CLASSES = [c for c in range(NUM_OUTPUTS) if c != REMOVED_CLASS]

NUM_HIDDEN = int(os.environ.get("NUM_HIDDEN", "1000"))
TAU_SYN_MS = float(os.environ.get("TAU_SYN_MS", "5.0"))
TAU_MEM_MS = float(os.environ.get("TAU_MEM_MS", "10.0"))
THRESHOLD = float(os.environ.get("THRESHOLD", "1.0"))
RESET_MECHANISM = os.environ.get("RESET_MECHANISM", "zero")
OUTPUT_THRESHOLD = float(os.environ.get("OUTPUT_THRESHOLD", "1e9"))      # non-spiking integrator
OUTPUT_RESET_MECHANISM = os.environ.get("OUTPUT_RESET_MECHANISM", "none")
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

# Hidden signal fed into W_out each timestep (before the output LIF integrator).
READOUT_FEATURE = os.environ.get("READOUT_FEATURE", "hidden_spike_avg")
assert READOUT_FEATURE in ("hidden_spike_avg", "hidden_mem_avg")
_FEATURE_KEY = "hidden_spike_trace" if READOUT_FEATURE == "hidden_spike_avg" else "hidden_mem_trace"
_PROJECT_SPIKES = READOUT_FEATURE == "hidden_spike_avg"

METHOD = os.environ.get("METHOD", "bptt").lower()           # bptt | ridge (one per run)
assert METHOD in ("bptt", "ridge")
BPTT_REDUCE = os.environ.get("BPTT_REDUCE", "mean").lower()  # mean | max (BPTT objective)
assert BPTT_REDUCE in ("mean", "max")

# Each run is scored by its own output-membrane reduction; ridge is always mean (optimal for ridge).
EVAL_REDUCE = "mean" if METHOD == "ridge" else BPTT_REDUCE
if METHOD == "ridge":
    READOUT_NAME = f"ridge_output_mean_{READOUT_FEATURE}_no_bias"
else:
    READOUT_NAME = "ce_max_mem" if BPTT_REDUCE == "max" else "ce_mean_mem"

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
OPTIMIZER_TYPE = os.environ.get("OPTIMIZER_TYPE", "adamax").lower()
GRAD_CLIP_MAX_NORM = float(os.environ.get("GRAD_CLIP_MAX_NORM", "5.0"))
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1e-3"))

DETERMINISTIC = bool(int(os.environ.get("DETERMINISTIC", "0")))
AMP = bool(int(os.environ.get("AMP", "1")))
AMP_DTYPE = torch.bfloat16
GPU_CACHE = bool(int(os.environ.get("GPU_CACHE", "1")))
TORCH_COMPILE = bool(int(os.environ.get("TORCH_COMPILE", "0")))
# CPU intra-op threads for this process (0 = leave PyTorch default). Set this when
# running several trainings in parallel so the per-process pools sum to <= #cores.
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))

WANDB_MODE = os.environ.get("WANDB_MODE", "online")          # online | offline | disabled
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "shd-output-lif")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "") or None
RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    f"output_lif_{METHOD}_{READOUT_NAME}_{READOUT_FEATURE}_rm{REMOVED_CLASS}_dt{int(round(DT_MS))}ms",
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
# Data
# =============================================================================


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class SplitData:
    """Whole split as one ``uint8`` ``[N,T,C]`` tensor (+ labels), optionally on GPU."""

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
        self.x = torch.cat(x_list, 0).contiguous()
        self.y = torch.cat(y_list, 0).contiguous()
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
# Model: reservoir SNN WITH a second-order non-spiking output neuron
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
        self.output_neuron = snn.Synaptic(
            alpha=ALPHA, beta=BETA, threshold=OUTPUT_THRESHOLD,
            spike_grad=spike_grad, reset_mechanism=OUTPUT_RESET_MECHANISM,
        )

    def forward(self, x: torch.Tensor, need_output: bool = True) -> Dict[str, torch.Tensor]:
        x = x.float()
        B, T, C = x.shape
        h_in = (x.reshape(B * T, C) @ self.W_in).reshape(B, T, self.H)   # hoisted
        syn_h = torch.zeros(B, self.H, device=x.device, dtype=h_in.dtype)
        mem_h = torch.zeros_like(syn_h)
        spk_prev = torch.zeros_like(syn_h)
        syn_o = torch.zeros(B, NUM_OUTPUTS, device=x.device, dtype=h_in.dtype)
        mem_o = torch.zeros_like(syn_o)
        spikes: List[torch.Tensor] = []
        mems: List[torch.Tensor] = []
        out_mems: List[torch.Tensor] = []
        for t in range(T):
            cur = h_in[:, t] + spk_prev @ self.W_rec
            spk, syn_h, mem_h = self.hidden_neuron(cur, syn_h, mem_h)
            spikes.append(spk)
            if self.collect_mem:
                mems.append(mem_h)
            if need_output:
                h_proj = spk if _PROJECT_SPIKES else mem_h
                _, syn_o, mem_o = self.output_neuron(h_proj @ self.W_out, syn_o, mem_o)
                out_mems.append(mem_o)
            spk_prev = spk
        res = {"hidden_spike_trace": torch.stack(spikes, dim=1)}
        if self.collect_mem:
            res["hidden_mem_trace"] = torch.stack(mems, dim=1)
        if need_output:
            res["output_mem_trace"] = torch.stack(out_mems, dim=1)       # [B,T,20]
        return res


# =============================================================================
# Shared readout / evaluation / label helpers
# =============================================================================

_ACTIVE_T: Optional[torch.Tensor] = None


def active_tensor(device: torch.device) -> torch.Tensor:
    global _ACTIVE_T
    if _ACTIVE_T is None or _ACTIVE_T.device != device:
        _ACTIVE_T = torch.as_tensor(ACTIVE_CLASSES, dtype=torch.long, device=device)
    return _ACTIVE_T


def reduce_output_membrane(output_mem_trace: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "mean":
        return output_mem_trace.mean(dim=1)
    if reduce == "max":
        return output_mem_trace.max(dim=1).values
    raise ValueError(reduce)


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
def evaluate(model: ReservoirSNN, split: SplitData, device: torch.device, reduce: str) -> Dict[str, float]:
    """Evaluate with the run's native output-membrane reduction (``reduce``)."""
    model.eval()
    correct = total = 0
    spikes = 0.0
    sites = 0
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            traces = model(x, need_output=True)
            active = select_active(reduce_output_membrane(traces["output_mem_trace"], reduce))
        pred = active_tensor(device)[active.float().argmax(dim=1)]
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        st = traces["hidden_spike_trace"]
        spikes += float(st.sum().item())
        sites += int(st.numel())
    return {"acc": correct / max(total, 1), "hidden_firing_rate": spikes / max(sites, 1)}


# =============================================================================
# BPTT training (output-LIF readout; reduction = BPTT_REDUCE for both train & eval)
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
        hid_spk = hid_sites = 0.0
        for x, y in train.batches(BATCH_SIZE, shuffle=True, generator=gen):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
                traces = model(x, need_output=True)
                logits = select_active(reduce_output_membrane(traces["output_mem_trace"], BPTT_REDUCE))
                loss = loss_fn(logits.float(), remap_to_active(y))
            loss.backward()
            gnorm_sum += _grad_norm(trainable)
            if GRAD_CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_MAX_NORM)
            optimizer.step()
            n = int(y.numel())
            run_loss += float(loss.item()) * n
            run_acc += float((active_tensor(device)[logits.float().argmax(1)] == y).float().sum().item())
            seen += n
            nbatch += 1
            st = traces["hidden_spike_trace"]
            hid_spk += float(st.detach().sum().item()); hid_sites += int(st.numel())

        # Model selection on the run's OWN reduction (max for ce_max_mem, mean for ce_mean_mem).
        val_m = evaluate(model, val, device, reduce=EVAL_REDUCE)
        if val_m["acc"] > best["val_acc"]:
            best = {"val_acc": val_m["acc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        payload = {
            "epoch": epoch,
            "train/loss": run_loss / max(seen, 1),
            "train/acc": run_acc / max(seen, 1),
            "train/grad_norm_mean": gnorm_sum / max(nbatch, 1),
            "train/hidden_firing_rate": hid_spk / max(hid_sites, 1),
            "val/acc": val_m["acc"],
            "val/hidden_firing_rate": val_m["hidden_firing_rate"],
            "val/best_acc": best["val_acc"],
            "val/best_epoch": best["epoch"],
            "runtime/epoch_seconds": time.time() - t0,
        }
        if wandb_run is not None:
            wandb_run.log(payload)
        print(f"[{READOUT_NAME}] epoch={epoch:03d} loss={payload['train/loss']:.4f} "
              f"train_acc={payload['train/acc']:.4f} val_acc({EVAL_REDUCE})={val_m['acc']:.4f} "
              f"best={best['val_acc']:.4f}@{best['epoch']} hz={val_m['hidden_firing_rate']:.3f} "
              f"({payload['runtime/epoch_seconds']:.1f}s)")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"best_val_acc": best["val_acc"], "best_epoch": best["epoch"]}


# =============================================================================
# Ridge training (closed form, NO bias): mean output-membrane readout
# =============================================================================


def _install_ridge_w_out(model: ReservoirSNN, W_active: torch.Tensor) -> None:
    W_full = torch.zeros((NUM_HIDDEN, NUM_OUTPUTS), dtype=W_active.dtype, device=W_active.device)
    W_full[:, active_tensor(W_active.device)] = W_active
    with torch.no_grad():
        model.W_out.copy_(W_full.to(model.W_out.device, model.W_out.dtype))


@torch.no_grad()
def collect_hidden_traces(model: ReservoirSNN, split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reservoir forward only; cache ``[N, T, H]`` hidden traces and labels."""
    model.eval()
    hids: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        traces = model(x, need_output=False)
        hids.append(traces[_FEATURE_KEY].float())
        labels.append(y.to(device))
    return torch.cat(hids, 0), torch.cat(labels, 0)


@torch.no_grad()
def _ridge_output_time_weights(output_neuron: nn.Module, device: torch.device, T: int) -> torch.Tensor:
    """Weights ``c[t]`` (length ``T``) for feature ``X[i,h] = sum_t c[t] * H[i,t,h]``.

    Matches the mean output membrane of ``output_neuron`` driven by scalar input
    ``u_t = H[i, t, h]`` (same dynamics as one column of ``h_t @ W_out``).
    """
    eye = torch.eye(T, device=device, dtype=torch.float32)
    syn = torch.zeros(T, 1, device=device, dtype=torch.float32)
    mem = torch.zeros_like(syn)
    mem_trace: List[torch.Tensor] = []
    for t in range(T):
        _, syn, mem = output_neuron(eye[t].unsqueeze(1), syn, mem)
        mem_trace.append(mem)
    return torch.stack(mem_trace, dim=0).mean(dim=0).squeeze(1)   # [T]


@torch.no_grad()
def extract_ridge_features(model: ReservoirSNN, split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design matrix ``X`` ``[N, H]`` for ridge on mean output-membrane logits."""
    hid, labels = collect_hidden_traces(model, split, device)
    T = int(hid.shape[1])
    c = _ridge_output_time_weights(model.output_neuron, device, T)
    X = torch.einsum("nth,t->nh", hid, c.to(hid.dtype))
    return X, labels


def solve_ridge_no_bias(X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``W = (X^T X + lambda I)^{-1} X^T Y`` with one-hot ``Y`` on active classes."""
    N = int(labels.numel())
    Y = torch.zeros((N, NUM_OUTPUTS - 1), dtype=torch.float64, device=X.device)
    Y[torch.arange(N, device=X.device), remap_to_active(labels)] = 1.0
    X64 = X.to(torch.float64)
    H = X64.shape[1]
    A = X64.T @ X64 + RIDGE_ALPHA * torch.eye(H, dtype=torch.float64, device=X.device)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(X64.T @ Y, L).to(X.dtype)          # [H, 19]


@torch.no_grad()
def _ridge_train_mse(X: torch.Tensor, labels: torch.Tensor, W_active: torch.Tensor) -> float:
    N = int(labels.numel())
    Y = torch.zeros((N, NUM_OUTPUTS - 1), dtype=torch.float64, device=X.device)
    Y[torch.arange(N, device=X.device), remap_to_active(labels)] = 1.0
    pred = X.to(torch.float64) @ W_active.to(torch.float64)
    return float((pred - Y).pow(2).mean().item())


def train_ridge(model: ReservoirSNN, train: SplitData, val: SplitData, device: torch.device,
                wandb_run) -> Dict[str, object]:
    """Closed-form ridge on ``W_out`` (frozen reservoir)."""
    model.requires_grad_(False)
    t0 = time.time()
    X, y = extract_ridge_features(model, train, device)
    W_active = solve_ridge_no_bias(X, y)
    _install_ridge_w_out(model, W_active)
    final_mse = _ridge_train_mse(X, y, W_active)

    train_m = evaluate(model, train, device, reduce=EVAL_REDUCE)
    val_m = evaluate(model, val, device, reduce=EVAL_REDUCE)
    info = {
        "ridge/alpha": RIDGE_ALPHA,
        "ridge/n_fit": int(y.numel()),
        "ridge/final_mse": final_mse,
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
    cfg = {
        "output_type": "output_lif",
        "method": METHOD,
        "readout": READOUT_NAME,
        "readout_feature": READOUT_FEATURE,
        "bptt_reduce": BPTT_REDUCE,
        "eval_reduce": EVAL_REDUCE,
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
        "output_threshold": OUTPUT_THRESHOLD, "output_reset_mechanism": OUTPUT_RESET_MECHANISM,
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
    env_keys = [
        ("SHD_PREPROCESSED_ROOT", cfg["preprocessed_root"]), ("METHOD", METHOD),
        ("READOUT_FEATURE", READOUT_FEATURE), ("BPTT_REDUCE", BPTT_REDUCE), ("SEED", SEED),
        ("REMOVED_CLASS", REMOVED_CLASS), ("DT_MS", DT_MS), ("NUM_INPUTS", NUM_INPUTS),
        ("NUM_HIDDEN", NUM_HIDDEN), ("TAU_SYN_MS", TAU_SYN_MS), ("TAU_MEM_MS", TAU_MEM_MS),
        ("THRESHOLD", THRESHOLD), ("RESET_MECHANISM", RESET_MECHANISM),
        ("OUTPUT_THRESHOLD", OUTPUT_THRESHOLD), ("OUTPUT_RESET_MECHANISM", OUTPUT_RESET_MECHANISM),
        ("SURROGATE_GRADIENT_TYPE", SURROGATE_GRADIENT_TYPE), ("SURROGATE_SLOPE", SURROGATE_SLOPE),
        ("W_IN_INIT_STD", W_IN_INIT_STD), ("W_REC_INIT_STD", W_REC_INIT_STD), ("W_OUT_INIT_STD", W_OUT_INIT_STD),
        ("TRAIN_W_IN", int(TRAIN_W_IN)), ("TRAIN_W_REC", int(TRAIN_W_REC)), ("TRAIN_W_OUT", int(TRAIN_W_OUT)),
        ("NUM_EPOCHS", NUM_EPOCHS), ("BATCH_SIZE", BATCH_SIZE), ("LEARNING_RATE", LEARNING_RATE),
        ("OPTIMIZER_TYPE", OPTIMIZER_TYPE), ("GRAD_CLIP_MAX_NORM", GRAD_CLIP_MAX_NORM),
        ("RIDGE_ALPHA", RIDGE_ALPHA), ("DETERMINISTIC", int(DETERMINISTIC)),
    ]
    cfg["reproduce_command"] = " ".join(f"{k}={v}" for k, v in env_keys) + \
        " python3 train_shd_output_lif.py"
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
                     tags=["shd", "output-lif", METHOD, READOUT_NAME, READOUT_FEATURE])
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
        "bptt_reduce": BPTT_REDUCE, "eval_reduce": EVAL_REDUCE,
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
    print(f"device={device} method={METHOD} feature={READOUT_FEATURE} readout={READOUT_NAME} "
          f"eval_reduce={EVAL_REDUCE} train(in,rec,out)=({TRAIN_W_IN},{TRAIN_W_REC},{TRAIN_W_OUT}) "
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
        hz = float(real(x0.to(device), need_output=False)["hidden_spike_trace"].float().mean().item())
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

    val_m = evaluate(model, val, device, reduce=EVAL_REDUCE)
    test_m = evaluate(model, test, device, reduce=EVAL_REDUCE)
    final = {**info,
             "val/acc": val_m["acc"], "val/hidden_firing_rate": val_m["hidden_firing_rate"],
             "test/acc": test_m["acc"], "test/hidden_firing_rate": test_m["hidden_firing_rate"],
             "eval_reduce": EVAL_REDUCE, "runtime/total_seconds": time.time() - start}
    print(f"\n=== {READOUT_NAME} ({METHOD}) ===  val_acc({EVAL_REDUCE})={val_m['acc']:.4f}  "
          f"test_acc({EVAL_REDUCE})={test_m['acc']:.4f}  hidden_hz(test)={test_m['hidden_firing_rate']:.3f}")

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
