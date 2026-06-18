#!/usr/bin/env python3
"""Pittorino-style reservoir SNN on the FULL 20-class SHD (output-LIF head).

This is ``train_shd_output_lif.py`` re-pointed to faithfully reproduce the
Dequino/Pittorino setup, keeping the original code style (env-driven config,
``SplitData.batches`` iteration, W&B logging and offline artifacts) but with
HAND-ROLLED Zenke/Pittorino neurons (no snnTorch). The substantive changes are:

  * HAND-ROLLED ZENKE HIDDEN NEURON (no snnTorch). Second-order LIF where the spike
    is read from the membrane at the START of the step and reset is multiplicative
    (``mem *= 1-spk``). At alpha=0.82/beta=0.90 this fires ~6x more than snnTorch
    ``Synaptic`` with identical weights, so the readout gets a far richer spike code.
  * ALL 20 CLASSES. No removed/continual class; plain 20-way cross-entropy.
  * RAW SHD. Loads ``shd_train.h5`` + ``shd_test.h5`` directly, MERGES them and
    re-splits (Pittorino's protocol) instead of reading class-incremental pickles.
  * INPUT ENCODING switch (``INPUT_ENCODING=binary|count``). ``binary`` (default): a
    (time-bin, channel) cell is 0/1. ``count``: number of events in the cell (Pittorino's
    summed-sparse encoding) -- at 14 ms bins ~39% of cells collide, ~33% more spike mass.
  * DECOUPLED dt. Events are binned at ``DT_MS`` (=14 ms) but the LIF decays use a
    separate ``SIM_DT_MS`` (=1 ms) -> ALPHA=exp(-1/5)=0.819, BETA=exp(-1/10)=0.905.
    Long membrane memory across the utterance (the thing that makes SHD learnable).
  * SPECTRAL-RADIUS rescale of the recurrent matrix to ``SPECTRAL_RADIUS`` (0.8).
  * Reservoir init std ``WEIGHT_SCALE/sqrt(fan_in)`` with ``WEIGHT_SCALE=0.2``;
    surrogate slope 100.
  * BPTT readout = per-step ``W_out`` projection -> non-spiking output LIF ->
    ``max`` over time (Pittorino ``filtered_lif_max``); default ``BPTT_REDUCE=max``.
  * RIDGE = closed form on the UNIFORM ``mean_t(hidden spikes)`` feature (NO output
    LIF, NO kernel), one-hot over 20 classes, solved with ``torch.linalg.solve``;
    ridge is evaluated with the SAME firing-rate readout it was fit on.

Examples::

    METHOD=bptt  BPTT_REDUCE=max  WANDB_MODE=online  python3 train_shd_output_lif_Fabrizio.py
    METHOD=ridge RIDGE_ALPHA=1.0  WANDB_MODE=online  python3 train_shd_output_lif_Fabrizio.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Configuration
# =============================================================================

SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Raw SHD HDF5 location (Pittorino-style: load + merge + resplit).
DATA_DIR = os.environ.get("SHD_DATA_DIR", "data").strip()
TRAIN_H5 = os.environ.get("SHD_TRAIN_H5", str(Path(DATA_DIR) / "shd_train.h5")).strip()
TEST_H5 = os.environ.get("SHD_TEST_H5", str(Path(DATA_DIR) / "shd_test.h5")).strip()
MERGE_TRAIN_TEST = bool(int(os.environ.get("MERGE_TRAIN_TEST", "1")))   # merge then resplit
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.15"))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.15"))
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts_output_lif_fabrizio").strip()

NUM_INPUTS = int(os.environ.get("NUM_INPUTS", "700"))
NUM_OUTPUTS = 20                                                        # FULL SHD: all 20 classes
MAX_TIME_SECONDS = float(os.environ.get("MAX_TIME_SECONDS", "1.4"))
DT_MS = float(os.environ.get("DT_MS", "14.0"))                          # BINNING width (ms)
DT_SECONDS = DT_MS / 1000.0
NB_STEPS = int(os.environ.get("NB_STEPS", str(int(math.ceil(MAX_TIME_SECONDS / DT_SECONDS)))))
# Input encoding per (time-bin, channel) cell: "binary" -> 0/1; "count" -> number of
# events in the cell (Pittorino's summed-sparse encoding). At 14 ms bins ~39% of cells
# collide, so count carries ~33% more spike mass than binary.
INPUT_ENCODING = os.environ.get("INPUT_ENCODING", "binary").lower()
assert INPUT_ENCODING in ("binary", "count")

# Neuron dynamics: SIMULATED at SIM_DT_MS, decoupled from the 14 ms binning.
SIM_DT_MS = float(os.environ.get("SIM_DT_MS", "1.0"))
NUM_HIDDEN = int(os.environ.get("NUM_HIDDEN", "1000"))
TAU_SYN_MS = float(os.environ.get("TAU_SYN_MS", "5.0"))
TAU_MEM_MS = float(os.environ.get("TAU_MEM_MS", "10.0"))
THRESHOLD = float(os.environ.get("THRESHOLD", "1.0"))
RESET_MECHANISM = "zero"   # hand-rolled hidden neuron always resets to zero (Pittorino)
SURROGATE_GRADIENT_TYPE = os.environ.get("SURROGATE_GRADIENT_TYPE", "fast_sigmoid")
SURROGATE_SLOPE = float(os.environ.get("SURROGATE_SLOPE", "100.0"))     # Pittorino scale=100
WEIGHT_SCALE = float(os.environ.get("WEIGHT_SCALE", "0.2"))            # Pittorino weight_scale
W_IN_SCALE = float(os.environ.get("W_IN_SCALE", str(WEIGHT_SCALE)))
W_REC_SCALE = float(os.environ.get("W_REC_SCALE", str(WEIGHT_SCALE)))
W_OUT_SCALE = float(os.environ.get("W_OUT_SCALE", str(WEIGHT_SCALE)))
W_IN_INIT_STD = float(os.environ.get("W_IN_INIT_STD", str(W_IN_SCALE / math.sqrt(NUM_INPUTS))))
W_REC_INIT_STD = float(os.environ.get("W_REC_INIT_STD", str(W_REC_SCALE / math.sqrt(NUM_HIDDEN))))
W_OUT_INIT_STD = float(os.environ.get("W_OUT_INIT_STD", str(W_OUT_SCALE / math.sqrt(NUM_HIDDEN))))
SPECTRAL_RADIUS = float(os.environ.get("SPECTRAL_RADIUS", "0.8"))      # <=0 disables rescale

# Decays are derived from SIM_DT_MS (NOT the binning DT_MS).
ALPHA = math.exp(-(SIM_DT_MS / TAU_SYN_MS))
BETA = math.exp(-(SIM_DT_MS / TAU_MEM_MS))

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
BPTT_REDUCE = os.environ.get("BPTT_REDUCE", "max").lower()  # max | mean (Pittorino default max)
assert BPTT_REDUCE in ("mean", "max")

# Readout mode used at eval. BPTT runs through the output LIF (mean|max); ridge is
# scored by the SAME uniform firing-rate readout it was fit on.
EVAL_MODE = "firing_rate" if METHOD == "ridge" else BPTT_REDUCE
if METHOD == "ridge":
    READOUT_NAME = "ridge_firing_rate_no_bias"
else:
    READOUT_NAME = "ce_max_mem" if BPTT_REDUCE == "max" else "ce_mean_mem"

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
OPTIMIZER_TYPE = os.environ.get("OPTIMIZER_TYPE", "adamax").lower()
# Pittorino uses NO clipping, but its stability comes from a hand-rolled neuron;
# with the snnTorch Synaptic neuron kept here, full-BPTT through 100 steps diverges
# without it. Default to 5.0 for stability; set GRAD_CLIP_MAX_NORM=0 to match Pittorino.
GRAD_CLIP_MAX_NORM = float(os.environ.get("GRAD_CLIP_MAX_NORM", "5.0"))
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1.0"))                 # Pittorino ridge_lambda

DETERMINISTIC = bool(int(os.environ.get("DETERMINISTIC", "0")))
AMP = bool(int(os.environ.get("AMP", "1")))
AMP_DTYPE = torch.bfloat16
GPU_CACHE = bool(int(os.environ.get("GPU_CACHE", "1")))
TORCH_COMPILE = bool(int(os.environ.get("TORCH_COMPILE", "0")))
# CPU intra-op threads for this process (0 = leave PyTorch default).
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))

WANDB_MODE = os.environ.get("WANDB_MODE", "online")          # online | offline | disabled
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "shd-output-lif-fabrizio-replica")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "") or None
RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    f"fabrizio_{METHOD}_{READOUT_NAME}_{INPUT_ENCODING}_20class_dt{int(round(DT_MS))}ms",
)

# Offline JSON backup (full per-run result.json + shared append-only JSONL).
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


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, surrogate-gradient backward (pure torch; no snnTorch).

    ``fast_sigmoid`` reproduces Pittorino's ``SurrGradSpike`` (``grad/(slope*|x|+1)^2``)
    and snnTorch's ``fast_sigmoid(slope)``; other kinds provided for parity with the
    original config.
    """

    @staticmethod
    def forward(ctx, x, slope, kind):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        ctx.kind = kind
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        s, k = ctx.slope, ctx.kind
        if k == "fast_sigmoid":
            grad = grad_output / (s * x.abs() + 1.0) ** 2
        elif k == "sigmoid":
            sig = torch.sigmoid(s * x)
            grad = grad_output * s * sig * (1.0 - sig)
        elif k == "atan":
            grad = grad_output * s / (1.0 + (math.pi * s * x) ** 2)
        elif k in ("ste", "straight_through_estimator"):
            grad = grad_output
        else:
            raise ValueError(f"Unknown surrogate {k!r}")
        return grad, None, None


def make_surrogate(kind: str, slope: float):
    """Return a spike function ``f(x) -> Heaviside(x)`` with a surrogate backward."""
    kind = (kind or "").lower()
    if kind not in ("fast_sigmoid", "sigmoid", "atan", "ste", "straight_through_estimator"):
        raise ValueError(f"Unknown surrogate {kind!r}")
    return lambda x: SurrGradSpike.apply(x, slope, kind)


# =============================================================================
# Data: raw SHD HDF5 -> merge -> bin (floor) -> stratified resplit -> SplitData
# =============================================================================


def _read_h5_pool(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Load one SHD HDF5 file as ragged per-sample event arrays + labels."""
    with h5py.File(path, "r") as f:
        times = [np.asarray(t, dtype=np.float64) for t in f["spikes"]["times"]]
        units = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
        labels = np.asarray(f["labels"], dtype=np.int64)
    return times, units, labels


def _bin_sample(times: np.ndarray, units: np.ndarray) -> torch.Tensor:
    """One event sample -> dense uint8 ``[NB_STEPS, NUM_INPUTS]`` via floor binning."""
    x = np.zeros((NB_STEPS, NUM_INPUTS), dtype=np.uint8)
    keep = times < MAX_TIME_SECONDS
    t, u = times[keep], units[keep]
    if t.size:
        b = np.clip((t / DT_SECONDS).astype(np.int64), 0, NB_STEPS - 1)
        if INPUT_ENCODING == "count":
            np.add.at(x, (b, u), 1)            # accumulate event counts per cell (max ~8)
        else:
            x[b, u] = 1                         # binary: 0/1 regardless of collisions
    return torch.from_numpy(x)


def _stratified_three_way(labels: np.ndarray, val_frac: float, test_frac: float,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-stratified train/val/test index split (per-class proportions held)."""
    tr, va, te = [], [], []
    for c in range(NUM_OUTPUTS):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        te.extend(idx[:n_test].tolist())
        va.extend(idx[n_test:n_test + n_val].tolist())
        tr.extend(idx[n_test + n_val:].tolist())
    return (np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te)))


def load_raw_shd() -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Merge SHD train(+test), bin every sample, stratified-resplit into 3 sets."""
    pools = [_read_h5_pool(TRAIN_H5)]
    if MERGE_TRAIN_TEST:
        pools.append(_read_h5_pool(TEST_H5))
    times: List[np.ndarray] = []
    units: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for t, u, y in pools:
        times.extend(t)
        units.extend(u)
        labels_list.append(y)
    labels = np.concatenate(labels_list, axis=0)
    n = len(labels)
    print(f"loaded {n} samples from {len(pools)} file(s); binning -> [{NB_STEPS},{NUM_INPUTS}] uint8")

    X = torch.zeros((n, NB_STEPS, NUM_INPUTS), dtype=torch.uint8)
    for i in range(n):
        X[i] = _bin_sample(times[i], units[i])
    y = torch.from_numpy(labels).long()

    rng = np.random.default_rng(SEED)
    tr, va, te = _stratified_three_way(labels, VAL_FRACTION, TEST_FRACTION, rng)
    return {
        "train": (X[tr].contiguous(), y[tr].contiguous()),
        "val": (X[va].contiguous(), y[va].contiguous()),
        "test": (X[te].contiguous(), y[te].contiguous()),
    }


class SplitData:
    """Whole split as one ``uint8`` ``[N,T,C]`` tensor (+ labels), optionally on GPU."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: torch.device, cache_on_gpu: bool) -> None:
        if x.shape[1:] != (NB_STEPS, NUM_INPUTS):
            raise ValueError(f"bad shape {tuple(x.shape)}")
        self.x = x.to(torch.uint8).contiguous()
        self.y = y.long().contiguous()
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
        if SPECTRAL_RADIUS > 0:                          # edge-of-chaos reservoir rescale (CPU)
            r = torch.linalg.eigvals(W_rec).abs().max()
            W_rec.mul_(SPECTRAL_RADIUS / r)
            self._spectral_radius_before = float(r)
        else:
            self._spectral_radius_before = None
        self.W_in = nn.Parameter(W_in, requires_grad=TRAIN_W_IN)
        self.W_rec = nn.Parameter(W_rec, requires_grad=TRAIN_W_REC)
        self.W_out = nn.Parameter(W_out, requires_grad=TRAIN_W_OUT)
        self.spike_fn = make_surrogate(SURROGATE_GRADIENT_TYPE, SURROGATE_SLOPE)
        # Both neurons are HAND-ROLLED (no snnTorch), matching Pittorino:
        #   HIDDEN  second-order LIF; spike read from the membrane at the START of the
        #           step; multiplicative reset-to-zero (mem *= 1-spk).
        #   OUTPUT  non-spiking second-order LIF (filtered_lif_max): syn=ALPHA*syn+in;
        #           mem=BETA*mem+syn; bounded since alpha,beta < 1.

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
            # --- hand-rolled Zenke hidden neuron (spike read BEFORE the membrane update) ---
            cur = h_in[:, t] + spk_prev @ self.W_rec        # recurrent uses previous-step spike
            spk = self.spike_fn(mem_h - THRESHOLD)          # spike from membrane at step start
            rst = spk.detach()
            new_syn_h = ALPHA * syn_h + cur
            mem_h = (BETA * mem_h + syn_h) * (1.0 - rst)    # integrate OLD syn, then reset-to-zero
            syn_h = new_syn_h
            spikes.append(spk)
            if self.collect_mem:
                mems.append(mem_h)
            if need_output:
                h_proj = spk if _PROJECT_SPIKES else mem_h
                syn_o = ALPHA * syn_o + h_proj @ self.W_out
                mem_o = BETA * mem_o + syn_o                 # non-spiking; never resets
                out_mems.append(mem_o)
            spk_prev = spk
        res = {"hidden_spike_trace": torch.stack(spikes, dim=1)}
        if self.collect_mem:
            res["hidden_mem_trace"] = torch.stack(mems, dim=1)
        if need_output:
            res["output_mem_trace"] = torch.stack(out_mems, dim=1)       # [B,T,20]
        return res


# =============================================================================
# Readout / evaluation helpers (plain 20-class; no active-class remapping)
# =============================================================================


def reduce_output_membrane(output_mem_trace: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "mean":
        return output_mem_trace.mean(dim=1)
    if reduce == "max":
        return output_mem_trace.max(dim=1).values
    raise ValueError(reduce)


def compute_logits(model: ReservoirSNN, traces: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    """``firing_rate``: mean_t(spikes) @ W_out (no output LIF). ``mean``/``max``: output membrane."""
    if mode == "firing_rate":
        return traces["hidden_spike_trace"].float().mean(dim=1) @ model.W_out
    return reduce_output_membrane(traces["output_mem_trace"], mode)


@torch.no_grad()
def evaluate(model: ReservoirSNN, split: SplitData, device: torch.device, mode: str) -> Dict[str, float]:
    """Single evaluation used for BOTH methods (BPTT mean/max, ridge firing_rate)."""
    model.eval()
    correct = total = 0
    spikes = 0.0
    sites = 0
    need_output = mode != "firing_rate"
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            traces = model(x, need_output=need_output)
            logits = compute_logits(model, traces, mode)
        pred = logits.float().argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        st = traces["hidden_spike_trace"]
        spikes += float(st.sum().item())
        sites += int(st.numel())
    return {"acc": correct / max(total, 1), "hidden_firing_rate": spikes / max(sites, 1)}


# =============================================================================
# BPTT training (per-step W_out -> output LIF -> reduce over time; default max)
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
                logits = reduce_output_membrane(traces["output_mem_trace"], BPTT_REDUCE)
                loss = loss_fn(logits.float(), y)
            loss.backward()
            gnorm_sum += _grad_norm(trainable)
            if GRAD_CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_MAX_NORM)
            optimizer.step()
            n = int(y.numel())
            run_loss += float(loss.item()) * n
            run_acc += float((logits.float().argmax(1) == y).float().sum().item())
            seen += n
            nbatch += 1
            st = traces["hidden_spike_trace"]
            hid_spk += float(st.detach().sum().item()); hid_sites += int(st.numel())

        val_m = evaluate(model, val, device, mode=EVAL_MODE)
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
              f"train_acc={payload['train/acc']:.4f} val_acc({EVAL_MODE})={val_m['acc']:.4f} "
              f"best={best['val_acc']:.4f}@{best['epoch']} hz={val_m['hidden_firing_rate']:.3f} "
              f"({payload['runtime/epoch_seconds']:.1f}s)")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"best_val_acc": best["val_acc"], "best_epoch": best["epoch"]}


# =============================================================================
# Ridge training (closed form, NO bias): uniform mean_t(spikes) feature, 20 classes
# =============================================================================


@torch.no_grad()
def extract_mean_spike_features(model: ReservoirSNN, split: SplitData,
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design matrix ``X`` ``[N, H]`` = uniform time-mean of hidden spikes (NO output LIF)."""
    model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        traces = model(x, need_output=False)
        feats.append(traces["hidden_spike_trace"].float().mean(dim=1))
        labels.append(y.to(device))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def solve_ridge_no_bias(X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``W = (X^T X + lambda I)^{-1} X^T Y`` via ``torch.linalg.solve`` (fp64, CPU), 20-class one-hot."""
    N = int(labels.numel())
    X64 = X.detach().to("cpu", torch.float64)
    Y = torch.zeros((N, NUM_OUTPUTS), dtype=torch.float64)
    Y[torch.arange(N), labels.detach().cpu().long()] = 1.0
    H = X64.shape[1]
    A = X64.T @ X64 + RIDGE_ALPHA * torch.eye(H, dtype=torch.float64)
    return torch.linalg.solve(A, X64.T @ Y)                            # [H, 20]


@torch.no_grad()
def _ridge_train_mse(X: torch.Tensor, labels: torch.Tensor, W: torch.Tensor) -> float:
    N = int(labels.numel())
    Y = torch.zeros((N, NUM_OUTPUTS), dtype=torch.float64)
    Y[torch.arange(N), labels.detach().cpu().long()] = 1.0
    pred = X.detach().to("cpu", torch.float64) @ W.to("cpu", torch.float64)
    return float((pred - Y).pow(2).mean().item())


def train_ridge(model: ReservoirSNN, train: SplitData, val: SplitData, device: torch.device,
                wandb_run) -> Dict[str, object]:
    """Closed-form ridge on ``W_out`` (frozen reservoir); fit + scored on firing-rate readout."""
    model.requires_grad_(False)
    t0 = time.time()
    X, y = extract_mean_spike_features(model, train, device)
    W = solve_ridge_no_bias(X, y)
    with torch.no_grad():
        model.W_out.copy_(W.to(model.W_out.device, model.W_out.dtype))
    final_mse = _ridge_train_mse(X, y, W)

    train_m = evaluate(model, train, device, mode=EVAL_MODE)
    val_m = evaluate(model, val, device, mode=EVAL_MODE)
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
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "neuron": "hand_rolled_zenke",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def build_config(train_n: int, val_n: int, test_n: int) -> Dict[str, object]:
    cfg = {
        "output_type": "output_lif_fabrizio",
        "method": METHOD,
        "readout": READOUT_NAME,
        "readout_feature": READOUT_FEATURE,
        "bptt_reduce": BPTT_REDUCE,
        "eval_mode": EVAL_MODE,
        "seed": SEED,
        "device": DEVICE,
        # data
        "train_h5": str(Path(TRAIN_H5).resolve()),
        "test_h5": str(Path(TEST_H5).resolve()),
        "merge_train_test": MERGE_TRAIN_TEST,
        "val_fraction": VAL_FRACTION, "test_fraction": TEST_FRACTION,
        "num_inputs": NUM_INPUTS,
        "num_outputs": NUM_OUTPUTS,
        "dt_ms": DT_MS, "sim_dt_ms": SIM_DT_MS,
        "max_time_seconds": MAX_TIME_SECONDS,
        "nb_steps": NB_STEPS,
        "n_train": train_n, "n_val": val_n, "n_test": test_n,
        # model
        "num_hidden": NUM_HIDDEN,
        "tau_syn_ms": TAU_SYN_MS, "tau_mem_ms": TAU_MEM_MS, "alpha": ALPHA, "beta": BETA,
        "threshold": THRESHOLD, "reset_mechanism": RESET_MECHANISM, "neuron": "hand_rolled_zenke",
        "input_encoding": INPUT_ENCODING,
        "surrogate_gradient_type": SURROGATE_GRADIENT_TYPE, "surrogate_slope": SURROGATE_SLOPE,
        "weight_scale": WEIGHT_SCALE, "spectral_radius": SPECTRAL_RADIUS,
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
        ("SHD_DATA_DIR", DATA_DIR), ("METHOD", METHOD), ("INPUT_ENCODING", INPUT_ENCODING),
        ("READOUT_FEATURE", READOUT_FEATURE), ("BPTT_REDUCE", BPTT_REDUCE), ("SEED", SEED),
        ("DT_MS", DT_MS), ("SIM_DT_MS", SIM_DT_MS), ("NUM_INPUTS", NUM_INPUTS),
        ("NUM_HIDDEN", NUM_HIDDEN), ("TAU_SYN_MS", TAU_SYN_MS), ("TAU_MEM_MS", TAU_MEM_MS),
        ("THRESHOLD", THRESHOLD),
        ("SURROGATE_GRADIENT_TYPE", SURROGATE_GRADIENT_TYPE), ("SURROGATE_SLOPE", SURROGATE_SLOPE),
        ("WEIGHT_SCALE", WEIGHT_SCALE), ("SPECTRAL_RADIUS", SPECTRAL_RADIUS),
        ("TRAIN_W_IN", int(TRAIN_W_IN)), ("TRAIN_W_REC", int(TRAIN_W_REC)), ("TRAIN_W_OUT", int(TRAIN_W_OUT)),
        ("NUM_EPOCHS", NUM_EPOCHS), ("BATCH_SIZE", BATCH_SIZE), ("LEARNING_RATE", LEARNING_RATE),
        ("OPTIMIZER_TYPE", OPTIMIZER_TYPE), ("GRAD_CLIP_MAX_NORM", GRAD_CLIP_MAX_NORM),
        ("RIDGE_ALPHA", RIDGE_ALPHA), ("DETERMINISTIC", int(DETERMINISTIC)),
    ]
    cfg["reproduce_command"] = " ".join(f"{k}={v}" for k, v in env_keys) + \
        " python3 train_shd_output_lif_Fabrizio.py"
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
                     tags=["shd", "output-lif", "fabrizio", "20class", METHOD, READOUT_NAME, INPUT_ENCODING])
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("ridge/*", step_metric="epoch")
        run.define_metric("runtime/*", step_metric="epoch")
    return run


def dump_results_offline(cfg: Dict[str, object], final: Dict[str, object], ckpt_path: Path) -> None:
    rec = {
        "run_id": cfg["run_id"], "run_name": RUN_NAME, "output_type": cfg["output_type"],
        "method": METHOD, "readout": cfg["readout"], "readout_feature": READOUT_FEATURE,
        "bptt_reduce": BPTT_REDUCE, "eval_mode": EVAL_MODE,
        "seed": SEED, "dt_ms": DT_MS, "sim_dt_ms": SIM_DT_MS, "nb_steps": NB_STEPS,
        "num_hidden": NUM_HIDDEN, "spectral_radius": SPECTRAL_RADIUS,
        "train_w_in": TRAIN_W_IN, "train_w_rec": TRAIN_W_REC, "train_w_out": TRAIN_W_OUT,
        "val_acc": final.get("val/acc"), "test_acc": final.get("test/acc"),
        "best_val_acc": final.get("best_val_acc"), "best_epoch": final.get("best_epoch"),
        "val_hidden_firing_rate": final.get("val/hidden_firing_rate"),
        "test_hidden_firing_rate": final.get("test/hidden_firing_rate"),
        "checkpoint": str(ckpt_path),
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
          f"eval_mode={EVAL_MODE} train(in,rec,out)=({TRAIN_W_IN},{TRAIN_W_REC},{TRAIN_W_OUT}) "
          f"AMP={AMP} GPU_CACHE={GPU_CACHE} compile={TORCH_COMPILE} "
          f"nb_steps={NB_STEPS} bin_dt={DT_MS}ms sim_dt={SIM_DT_MS}ms alpha={ALPHA:.4f} beta={BETA:.4f} "
          f"spectral_radius={SPECTRAL_RADIUS}")

    splits = load_raw_shd()
    train = SplitData(*splits["train"], device, GPU_CACHE)
    val = SplitData(*splits["val"], device, GPU_CACHE)
    test = SplitData(*splits["test"], device, GPU_CACHE)
    print(f"splits (20-class): train={len(train)} val={len(val)} test={len(test)}")

    cfg = build_config(len(train), len(val), len(test))
    wandb_run = init_wandb(cfg)
    print(f"run_id={cfg['run_id']} run_name={RUN_NAME}")

    model = ReservoirSNN(collect_mem=(_FEATURE_KEY == "hidden_mem_trace")).to(device)
    if model._spectral_radius_before is not None:
        print(f"[reservoir] W_rec spectral radius {model._spectral_radius_before:.3f} -> {SPECTRAL_RADIUS}")
    if TORCH_COMPILE:
        model = torch.compile(model)
    real = model._orig_mod if TORCH_COMPILE else model

    with torch.no_grad():
        x0, _ = next(iter(train.batches(min(BATCH_SIZE, len(train)), shuffle=False)))
        hz = float(real(x0.to(device), need_output=False)["hidden_spike_trace"].float().mean().item())
    print(f"[smoke] initial hidden firing rate: {hz:.4f}")
    if not (0.001 <= hz <= 0.6):
        print(f"WARNING: hidden firing rate {hz:.4f} is extreme; consider tuning "
              f"THRESHOLD / WEIGHT_SCALE / SPECTRAL_RADIUS (continuing anyway).")
    if wandb_run is not None:
        wandb_run.summary["initial_hidden_firing_rate"] = hz

    start = time.time()
    if METHOD == "bptt":
        info = train_bptt(model, train, val, device, wandb_run)
    else:
        info = train_ridge(real, train, val, device, wandb_run)

    val_m = evaluate(model, val, device, mode=EVAL_MODE)
    test_m = evaluate(model, test, device, mode=EVAL_MODE)
    final = {**info,
             "val/acc": val_m["acc"], "val/hidden_firing_rate": val_m["hidden_firing_rate"],
             "test/acc": test_m["acc"], "test/hidden_firing_rate": test_m["hidden_firing_rate"],
             "eval_mode": EVAL_MODE, "runtime/total_seconds": time.time() - start}
    print(f"\n=== {READOUT_NAME} ({METHOD}) ===  val_acc({EVAL_MODE})={val_m['acc']:.4f}  "
          f"test_acc({EVAL_MODE})={test_m['acc']:.4f}  hidden_hz(test)={test_m['hidden_firing_rate']:.3f}")

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
