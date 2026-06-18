#!/usr/bin/env python3
"""BPTT vs Ridge through the SECOND-ORDER OUTPUT LIF head, on the COMPRESSED
35-channel SHD dataset at dt = 1 ms, with rich W&B SNN diagnostics.

This is a variant of ``train_shd_output_lif.py`` specialised for the experiment
"how is the recurrent SNN training going, where does it get stuck, and why":

  * INPUT  : OR-pooled 35-channel SHD (``nb_inputs = 35``), produced by
             ``build_shd_compressed_dataset.py --n-compressed-channels 35``.
  * TIMEBASE: ``dt = 1 ms`` -> ``nb_steps = ceil(max_time / 1e-3)`` (1400 for the
             1.4 s window). ALPHA/BETA are derived from this same dt, so the
             neuron dynamics match the data resolution.

All data-defining constants (``nb_inputs``, ``dt_ms``, ``nb_steps``,
``max_time_seconds``, ``removed_class``) are read from the preprocessing manifest
and can be overridden by environment variables; the defaults below assume a
``shd_removed{R}_dt1ms_or35`` preprocessed root.

W&B diagnostics added relative to the base trainer (BPTT-centric; Ridge gets the
final media + confusion only):
  * EVERY epoch:   ``train/input_firing_rate``, ``val/loss``,
                   ``val/input_firing_rate``, per-parameter grad-norm time series
                   (``train/grad_norm_w_in|w_rec|w_out``).
  * EVERY N epochs (``WANDB_DIAGNOSTIC_INTERVAL``): hidden-activity health on the
                   FULL validation split (silent / saturated fractions, spikes
                   per active neuron), weight + gradient histograms, and a single
                   multi-series "grad norms by parameter" chart.
  * FINAL (after best-val weights are restored): input/hidden spike rasters,
                   hidden- and output-membrane heatmaps for a FIXED train sample,
                   and a validation confusion matrix.

Every new logging path no-ops when ``wandb_run is None`` (``WANDB_MODE=disabled``
or wandb not installed). Training math, readout logic and checkpoint selection are
unchanged from the base trainer.

Examples::

    SHD_PREPROCESSED_ROOT=data/class_incremental/shd_removed10_dt1ms_or35 \
      METHOD=bptt BPTT_REDUCE=max WANDB_MODE=offline \
      WANDB_DIAGNOSTIC_INTERVAL=10 python3 train_shd_output_lif_c35_dt1ms.py

    SHD_PREPROCESSED_ROOT=data/class_incremental/shd_removed10_dt1ms_or35 \
      METHOD=ridge TRAIN_W_IN=0 TRAIN_W_REC=0 TRAIN_W_OUT=1 \
      python3 train_shd_output_lif_c35_dt1ms.py
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

# Headless plotting for W&B media (no display needed). MUST precede pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import snntorch as snn
from snntorch import surrogate


# =============================================================================
# Configuration
# =============================================================================

SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default root assumes the 35-channel / dt=1ms preprocessed dataset.
PREPROCESSED_ROOT = os.environ.get(
    "SHD_PREPROCESSED_ROOT", "data/class_incremental/shd_removed10_dt1ms_or35"
).strip()
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts_output_lif_c35_dt1ms").strip()


def _read_manifest(root: str) -> dict:
    try:
        return json.loads((Path(root) / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_MANIFEST = _read_manifest(PREPROCESSED_ROOT)

# Data-defining constants: env override > manifest > 35ch/dt1ms default.
REMOVED_CLASS = int(os.environ.get("REMOVED_CLASS", _MANIFEST.get("removed_class", 10)))
NUM_INPUTS = int(os.environ.get("NUM_INPUTS", _MANIFEST.get("nb_inputs", 35)))
NUM_OUTPUTS = 20
DT_MS = float(os.environ.get("DT_MS", _MANIFEST.get("dt_ms", 1.0)))
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

METHOD = os.environ.get("METHOD", "bptt").lower()           # bptt | ridge (one per run)
assert METHOD in ("bptt", "ridge")
BPTT_REDUCE = os.environ.get("BPTT_REDUCE", "mean").lower()  # mean | max (BPTT objective)
assert BPTT_REDUCE in ("mean", "max")

EVAL_REDUCE = "mean" if METHOD == "ridge" else BPTT_REDUCE
if METHOD == "ridge":
    READOUT_NAME = "ridge_output_mean_no_bias"
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
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))

# --- W&B SNN-diagnostic knobs -------------------------------------------------
WANDB_MODE = os.environ.get("WANDB_MODE", "online")          # online | offline | disabled
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "shd-output-lif-c35-dt1ms")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "") or None
# Periodic diagnostics every N epochs (epoch % N == 0, incl. epoch 0).
WANDB_DIAGNOSTIC_INTERVAL = int(os.environ.get("WANDB_DIAGNOSTIC_INTERVAL", "10"))
# Fixed index into the TRAIN split for raster / membrane media (same sample always).
WANDB_DIAGNOSTIC_SAMPLE_INDEX = int(os.environ.get("WANDB_DIAGNOSTIC_SAMPLE_INDEX", "0"))
# A neuron is "saturated" if its spike count over the diagnostic (val) pass is
# >= ceil(fraction * total_spike_sites_per_neuron). 1.0 => fired at every site.
WANDB_SATURATED_FRACTION = float(os.environ.get("WANDB_SATURATED_FRACTION", "1.0"))

RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    f"output_lif_c{NUM_INPUTS}_dt{DT_MS:g}ms_{METHOD}_{READOUT_NAME}_rm{REMOVED_CLASS}",
)

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
                raise ValueError(
                    f"{split_name}: bad shape {tuple(x.shape)}, expected (*, {NB_STEPS}, {NUM_INPUTS})"
                )
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

    def sample(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a single ``[1, T, C]`` sample (+ int label) at a fixed index."""
        if not 0 <= index < len(self):
            raise IndexError(f"sample index {index} out of range [0, {len(self)})")
        x = self.x[index : index + 1].to(self._device)   # [1, T, C]
        y = int(self.y[index].item())
        return x, y

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
    def __init__(self) -> None:
        super().__init__()
        self.H = NUM_HIDDEN
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

    def forward(
        self, x: torch.Tensor, need_output: bool = True, collect_hidden_mem: bool = False
    ) -> Dict[str, torch.Tensor]:
        """x: [B, T, C] (batch, time, channel).

        Returns ``hidden_spike_trace`` [B,T,H] always; ``output_mem_trace``
        [B,T,20] when ``need_output``; ``hidden_mem_trace`` [B,T,H] when
        ``collect_hidden_mem`` (diagnostics only — does NOT change training math).
        """
        x = x.float()
        B, T, C = x.shape
        h_in = (x.reshape(B * T, C) @ self.W_in).reshape(B, T, self.H)   # hoisted input proj
        syn_h = torch.zeros(B, self.H, device=x.device, dtype=h_in.dtype)
        mem_h = torch.zeros_like(syn_h)
        spk_prev = torch.zeros_like(syn_h)
        syn_o = torch.zeros(B, NUM_OUTPUTS, device=x.device, dtype=h_in.dtype)
        mem_o = torch.zeros_like(syn_o)
        spikes: List[torch.Tensor] = []
        hidden_mems: List[torch.Tensor] = []
        out_mems: List[torch.Tensor] = []
        for t in range(T):
            cur = h_in[:, t] + spk_prev @ self.W_rec
            spk, syn_h, mem_h = self.hidden_neuron(cur, syn_h, mem_h)
            spikes.append(spk)
            if collect_hidden_mem:
                hidden_mems.append(mem_h)
            if need_output:
                _, syn_o, mem_o = self.output_neuron(spk @ self.W_out, syn_o, mem_o)
                out_mems.append(mem_o)
            spk_prev = spk
        res = {"hidden_spike_trace": torch.stack(spikes, dim=1)}          # [B,T,H]
        if collect_hidden_mem:
            res["hidden_mem_trace"] = torch.stack(hidden_mems, dim=1)     # [B,T,H]
        if need_output:
            res["output_mem_trace"] = torch.stack(out_mems, dim=1)        # [B,T,20]
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
        return output_mem_trace.mean(dim=1)          # reduce over TIME -> [B,20]
    if reduce == "max":
        return output_mem_trace.max(dim=1).values    # reduce over TIME -> [B,20]
    raise ValueError(reduce)


def select_active(logits: torch.Tensor) -> torch.Tensor:
    return logits.index_select(1, active_tensor(logits.device))   # [B,19]


def remap_to_active(labels: torch.Tensor) -> torch.Tensor:
    lm = torch.full((NUM_OUTPUTS,), -1, dtype=torch.long, device=labels.device)
    lm[active_tensor(labels.device)] = torch.arange(NUM_OUTPUTS - 1, device=labels.device)
    mapped = lm[labels.long()]
    if torch.any(mapped < 0):
        raise AssertionError("label outside active set (removed class leaked?)")
    return mapped


@torch.no_grad()
def evaluate(model: "ReservoirSNN", split: SplitData, device: torch.device, reduce: str) -> Dict[str, float]:
    """Evaluate with the run's native output-membrane reduction (``reduce``).

    Extended vs the base trainer: also returns ``loss`` (CE on the SAME readout
    path used in training) and ``input_firing_rate`` so the caller can log
    ``val/loss`` and ``val/input_firing_rate`` every epoch.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct = total = 0
    loss_sum = 0.0
    spikes = 0.0
    sites = 0
    in_spk = 0.0
    in_sites = 0
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            traces = model(x, need_output=True)
            logits = select_active(reduce_output_membrane(traces["output_mem_trace"], reduce))
        loss_sum += float(loss_fn(logits.float(), remap_to_active(y)).item())
        pred = active_tensor(device)[logits.float().argmax(dim=1)]
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        st = traces["hidden_spike_trace"]
        spikes += float(st.sum().item())
        sites += int(st.numel())
        in_spk += float(x.float().sum().item())
        in_sites += int(x.numel())
    return {
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "hidden_firing_rate": spikes / max(sites, 1),
        "input_firing_rate": in_spk / max(in_sites, 1),
    }


# =============================================================================
# SNN diagnostics (activity health + confusion) — pure, wandb-agnostic
# =============================================================================


@torch.no_grad()
def hidden_activity_health(
    model: "ReservoirSNN", split: SplitData, device: torch.device, saturated_fraction: float
) -> Dict[str, float]:
    """Per-hidden-neuron spike statistics over the WHOLE ``split`` (one no-grad pass).

    Aggregates spike counts per hidden unit over all (sample, timestep) sites:
      * hidden_fraction_silent    : units that never spiked / H
      * hidden_fraction_saturated : units spiking at >= ceil(frac * sites) / H
      * hidden_spikes_per_active_neuron : total spikes / #units with >=1 spike
    """
    model.eval()
    spike_count = torch.zeros(NUM_HIDDEN, dtype=torch.float64, device=device)
    total_sites_per_neuron = 0  # = N_samples * T
    for x, _y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            st = model(x, need_output=False)["hidden_spike_trace"]   # [B,T,H]
        spike_count += st.float().sum(dim=(0, 1)).double()           # -> [H]
        total_sites_per_neuron += int(st.shape[0]) * int(st.shape[1])

    sat_threshold = math.ceil(max(0.0, saturated_fraction) * max(total_sites_per_neuron, 1))
    n_silent = int((spike_count == 0).sum().item())
    n_saturated = int((spike_count >= sat_threshold).sum().item())
    n_active = int((spike_count > 0).sum().item())
    total_spikes = float(spike_count.sum().item())
    return {
        "diagnostics/hidden_fraction_silent": n_silent / NUM_HIDDEN,
        "diagnostics/hidden_fraction_saturated": n_saturated / NUM_HIDDEN,
        "diagnostics/hidden_spikes_per_active_neuron": total_spikes / max(n_active, 1),
        "diagnostics/hidden_active_neuron_count": float(n_active),
        "diagnostics/hidden_saturation_threshold_spikes": float(sat_threshold),
    }


@torch.no_grad()
def compute_confusion(
    model: "ReservoirSNN", split: SplitData, device: torch.device, reduce: str
) -> Tuple[np.ndarray, List[int]]:
    """Confusion matrix over the active classes (rows=true, cols=pred)."""
    model.eval()
    a = ACTIVE_CLASSES
    pos = {c: i for i, c in enumerate(a)}
    cm = np.zeros((len(a), len(a)), dtype=np.int64)
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            logits = select_active(reduce_output_membrane(model(x)["output_mem_trace"], reduce))
        pred = active_tensor(device)[logits.float().argmax(dim=1)].cpu().tolist()
        true = y.cpu().tolist()
        for t_, p_ in zip(true, pred):
            if t_ in pos and p_ in pos:   # all should be active; guard defensively
                cm[pos[t_], pos[p_]] += 1
    return cm, a


# =============================================================================
# Plotting helpers (self-contained; no external pipeline.plotting dependency)
# =============================================================================


def fig_to_rgb_array(fig: "plt.Figure") -> np.ndarray:
    """Render a Matplotlib figure to an ``[H, W, 3]`` uint8 RGB array (Agg)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def make_input_hidden_raster_figure(
    x_sample: torch.Tensor, spk_hidden: torch.Tensor, *, epoch: int,
    sample_index: int, title_prefix: str, max_hidden_rows: int = 200,
) -> "plt.Figure":
    """Two stacked spike rasters (time on x): input [C,T] and hidden [H,T].

    ``x_sample``: [1,T,C] or [T,C]; ``spk_hidden``: [1,T,H] or [T,H].
    Hidden raster is capped to ``max_hidden_rows`` neurons for legibility.
    """
    xin = x_sample.detach().float().cpu()
    if xin.ndim == 3:
        xin = xin[0]
    sph = spk_hidden.detach().float().cpu()
    if sph.ndim == 3:
        sph = sph[0]
    xin_ct = xin.t().numpy()                       # [C, T]
    H = sph.shape[1]
    rows = min(H, max_hidden_rows)
    sph_ht = sph[:, :rows].t().numpy()             # [rows, T]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].imshow(xin_ct, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
    axes[0].set_ylabel("input channel")
    axes[0].set_title(f"{title_prefix} | input spikes (sample {sample_index}, epoch {epoch})")
    axes[1].imshow(sph_ht, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
    axes[1].set_ylabel(f"hidden neuron (0..{rows - 1})")
    axes[1].set_xlabel("timestep")
    axes[1].set_title(f"hidden spikes (showing {rows}/{H} neurons)")
    fig.tight_layout()
    return fig


def make_membrane_heatmap_figure(
    mem_trace: torch.Tensor, *, title: str, max_rows: int = 200, ylabel: str = "unit",
) -> "plt.Figure":
    """Heatmap of a membrane trace. ``mem_trace``: [1,T,U] or [T,U] -> imshow [U,T]."""
    m = mem_trace.detach().float().cpu()
    if m.ndim == 3:
        m = m[0]
    U = m.shape[1]
    rows = min(U, max_rows)
    arr = m[:, :rows].t().numpy()                  # [rows, T]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"{ylabel} (0..{rows - 1})")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def make_confusion_figure(cm: np.ndarray, class_labels: List[int], *, title: str) -> "plt.Figure":
    """Row-normalised confusion-matrix heatmap with raw counts annotated."""
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.clip(row_sums, 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(class_labels, fontsize=7)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    if len(class_labels) <= 20:
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                if cm[i, j]:
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                            fontsize=6, color="black" if norm[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised rate")
    fig.tight_layout()
    return fig


# =============================================================================
# W&B media logging (all guarded by `wandb_run is not None`)
# =============================================================================


def _unwrap(model: "ReservoirSNN") -> "ReservoirSNN":
    """Return the real module under a possible torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


def log_grad_norm_chart(wandb_run, gn_in: float, gn_rec: float, gn_out: float, epoch: int) -> None:
    """Single multi-series chart of per-parameter grad L2 norms (diagnostic epochs)."""
    if wandb_run is None:
        return
    try:
        import wandb
        chart = wandb.plot.line_series(
            xs=[0, 1, 2],
            ys=[[gn_in, gn_rec, gn_out]],
            keys=["grad_norm"],
            title="Per-parameter grad L2 norms (W_in=0, W_rec=1, W_out=2)",
            xname="parameter index",
        )
        wandb_run.log({"diagnostics/grad_norms_by_param": chart, "epoch": epoch})
    except Exception as e:  # never let logging break training
        print(f"WARNING: grad-norm chart logging failed: {e}")


def log_weight_and_grad_histograms(wandb_run, model: "ReservoirSNN", epoch: int) -> None:
    """Weight (+ gradient when present) histograms for W_in/W_rec/W_out."""
    if wandb_run is None:
        return
    try:
        import wandb
        core = _unwrap(model)
        payload: Dict[str, object] = {"epoch": epoch}
        for name, p in (("w_in", core.W_in), ("w_rec", core.W_rec), ("w_out", core.W_out)):
            payload[f"hist/{name}"] = wandb.Histogram(p.detach().float().cpu().numpy())
            if p.grad is not None:   # grads exist only after a backward (BPTT)
                payload[f"hist/grad_{name}"] = wandb.Histogram(p.grad.detach().float().cpu().numpy())
        wandb_run.log(payload)
    except Exception as e:
        print(f"WARNING: histogram logging failed: {e}")


def log_final_media(
    wandb_run, model: "ReservoirSNN", train: SplitData, val: SplitData,
    device: torch.device, epoch: int,
) -> None:
    """Final rich media: spike rasters + membrane heatmaps (fixed train sample) +
    validation confusion matrix. Uses the model's CURRENT weights (caller ensures
    best-val weights are loaded). No-op without wandb."""
    if wandb_run is None:
        return
    try:
        import wandb
    except Exception as e:
        print(f"WARNING: wandb unavailable for final media: {e}")
        return

    core = _unwrap(model)
    core.eval()
    idx = min(max(WANDB_DIAGNOSTIC_SAMPLE_INDEX, 0), len(train) - 1)
    x_sample, y_sample = train.sample(idx)            # [1,T,C]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP and device.type == "cuda"):
            traces = core(x_sample.to(device), need_output=True, collect_hidden_mem=True)

    figs: Dict[str, "plt.Figure"] = {}
    figs["media/raster_input_hidden"] = make_input_hidden_raster_figure(
        x_sample, traces["hidden_spike_trace"], epoch=epoch,
        sample_index=idx, title_prefix=READOUT_NAME,
    )
    figs["media/hidden_membrane_heatmap"] = make_membrane_heatmap_figure(
        traces["hidden_mem_trace"], title=f"hidden membrane (sample {idx}, true={y_sample})",
        ylabel="hidden neuron",
    )
    figs["media/output_membrane_heatmap"] = make_membrane_heatmap_figure(
        traces["output_mem_trace"], title=f"output membrane (sample {idx}, true={y_sample})",
        max_rows=NUM_OUTPUTS, ylabel="output class",
    )
    cm, labels = compute_confusion(core, val, device, EVAL_REDUCE)
    figs["media/val_confusion_matrix"] = make_confusion_figure(
        cm, labels, title=f"val confusion ({EVAL_REDUCE}) | epoch {epoch}",
    )

    payload: Dict[str, object] = {"epoch": epoch}
    for key, fig in figs.items():
        payload[key] = wandb.Image(fig_to_rgb_array(fig), caption=f"{key} @ epoch {epoch}")
        plt.close(fig)
    try:
        wandb_run.log(payload)
    except Exception as e:
        print(f"WARNING: final media logging failed: {e}")
        for fig in figs.values():
            plt.close(fig)


# =============================================================================
# BPTT training
# =============================================================================


def build_optimizer(model: "ReservoirSNN") -> torch.optim.Optimizer:
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


def _param_grad_norm(p: Optional[torch.Tensor]) -> float:
    """L2 norm of one parameter's grad; 0.0 if frozen or grad is None."""
    if p is None or p.grad is None:   # frozen (requires_grad=False) or no backward yet
        return 0.0
    return float(p.grad.detach().pow(2).sum().sqrt().item())


def train_bptt(model: "ReservoirSNN", train: SplitData, val: SplitData, device: torch.device,
               wandb_run) -> Dict[str, object]:
    optimizer = build_optimizer(model)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    trainable = [p for p in model.parameters() if p.requires_grad]
    core = _unwrap(model)
    best = {"val_acc": -1.0, "epoch": -1, "state": None}

    def is_diag_epoch(e: int) -> bool:
        return WANDB_DIAGNOSTIC_INTERVAL > 0 and e % WANDB_DIAGNOSTIC_INTERVAL == 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        t0 = time.time()
        run_loss = run_acc = 0.0
        seen = 0
        gnorm_sum = 0.0
        nbatch = 0
        hid_spk = hid_sites = 0.0
        in_spk = in_sites = 0.0
        # per-parameter grad-norm accumulators (mean over batches, like grad_norm_mean)
        gn_in_sum = gn_rec_sum = gn_out_sum = 0.0
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
            # per-parameter grad norms captured BEFORE clipping, like the global one
            gn_in_sum += _param_grad_norm(core.W_in)
            gn_rec_sum += _param_grad_norm(core.W_rec)
            gn_out_sum += _param_grad_norm(core.W_out)
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
            in_spk += float(x.float().sum().item()); in_sites += int(x.numel())

        # Model selection on the run's OWN reduction.
        val_m = evaluate(model, val, device, reduce=EVAL_REDUCE)
        if val_m["acc"] > best["val_acc"]:
            best = {"val_acc": val_m["acc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

        gn_in = gn_in_sum / max(nbatch, 1)
        gn_rec = gn_rec_sum / max(nbatch, 1)
        gn_out = gn_out_sum / max(nbatch, 1)
        payload = {
            "epoch": epoch,
            "train/loss": run_loss / max(seen, 1),
            "train/acc": run_acc / max(seen, 1),
            "train/grad_norm_mean": gnorm_sum / max(nbatch, 1),
            "train/grad_norm_w_in": gn_in,      # per-parameter time series (every epoch)
            "train/grad_norm_w_rec": gn_rec,
            "train/grad_norm_w_out": gn_out,
            "train/hidden_firing_rate": hid_spk / max(hid_sites, 1),
            "train/input_firing_rate": in_spk / max(in_sites, 1),
            "val/acc": val_m["acc"],
            "val/loss": val_m["loss"],
            "val/hidden_firing_rate": val_m["hidden_firing_rate"],
            "val/input_firing_rate": val_m["input_firing_rate"],
            "val/best_acc": best["val_acc"],
            "val/best_epoch": best["epoch"],
            "runtime/epoch_seconds": time.time() - t0,
        }
        if wandb_run is not None:
            wandb_run.log(payload)

        # Periodic heavier diagnostics (gated to keep cost down).
        if wandb_run is not None and is_diag_epoch(epoch):
            health = hidden_activity_health(model, val, device, WANDB_SATURATED_FRACTION)
            wandb_run.log({**health, "epoch": epoch})
            log_grad_norm_chart(wandb_run, gn_in, gn_rec, gn_out, epoch)
            log_weight_and_grad_histograms(wandb_run, model, epoch)

        print(f"[{READOUT_NAME}] epoch={epoch:03d} loss={payload['train/loss']:.4f} "
              f"train_acc={payload['train/acc']:.4f} val_acc({EVAL_REDUCE})={val_m['acc']:.4f} "
              f"val_loss={val_m['loss']:.4f} best={best['val_acc']:.4f}@{best['epoch']} "
              f"gn(in,rec,out)=({gn_in:.2e},{gn_rec:.2e},{gn_out:.2e}) "
              f"hz={val_m['hidden_firing_rate']:.3f} ({payload['runtime/epoch_seconds']:.1f}s)")

    if best["state"] is not None:
        model.load_state_dict(best["state"])  # restore best-val weights for final media
    # Final rich media is logged from main() once the model is finalised (best-val).
    return {"best_val_acc": best["val_acc"], "best_epoch": best["epoch"]}


# =============================================================================
# Ridge training (closed form, NO bias) matched to the mean-membrane readout
# =============================================================================


@torch.no_grad()
def output_mean_kernel(model: "ReservoirSNN", device: torch.device) -> torch.Tensor:
    """Empirical time-mean impulse kernel ``w`` [T] of the output neuron (fp32)."""
    T = NB_STEPS
    eye = torch.eye(T, device=device, dtype=torch.float32)
    syn = torch.zeros(T, 1, device=device, dtype=torch.float32)
    mem = torch.zeros(T, 1, device=device, dtype=torch.float32)
    mem_trace = []
    for t in range(T):
        inp = eye[t].unsqueeze(1)
        _, syn, mem = model.output_neuron(inp, syn, mem)
        mem_trace.append(mem)
    M = torch.stack(mem_trace, dim=0)
    return M.mean(dim=0).squeeze(1)


@torch.no_grad()
def extract_kernel_features(model: "ReservoirSNN", split: SplitData, w: torch.Tensor,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    for x, y in split.batches(BATCH_SIZE, shuffle=False):
        x = x.to(device, non_blocking=True)
        traces = model(x, need_output=False)
        spk = traces["hidden_spike_trace"].float()
        feats.append(torch.einsum("bth,t->bh", spk, w.to(spk.dtype)))
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
    return torch.cholesky_solve(X64.T @ Y, L).to(X.dtype)


def train_ridge(model: "ReservoirSNN", train: SplitData, device: torch.device,
                wandb_run) -> Dict[str, object]:
    model.requires_grad_(False)
    t0 = time.time()
    w = output_mean_kernel(model, device)
    X, y = extract_kernel_features(model, train, w, device)
    W_active = solve_ridge_no_bias(X, y)
    W_full = torch.zeros((NUM_HIDDEN, NUM_OUTPUTS), dtype=W_active.dtype, device=W_active.device)
    W_full[:, active_tensor(W_active.device)] = W_active
    with torch.no_grad():
        model.W_out.copy_(W_full.to(model.W_out.device, model.W_out.dtype))

    info = {
        "alpha": RIDGE_ALPHA,
        "n_fit": int(y.numel()),
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
        "variant": "compressed35_dt1ms",
        "method": METHOD,
        "readout": READOUT_NAME,
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
        # diagnostics
        "wandb_diagnostic_interval": WANDB_DIAGNOSTIC_INTERVAL,
        "wandb_diagnostic_sample_index": WANDB_DIAGNOSTIC_SAMPLE_INDEX,
        "wandb_saturated_fraction": WANDB_SATURATED_FRACTION,
        # efficiency / reproducibility
        "deterministic": DETERMINISTIC, "amp": AMP, "amp_dtype": "bfloat16",
        "gpu_cache": GPU_CACHE, "torch_compile": TORCH_COMPILE,
        "versions": _versions(),
    }
    cfg["run_id"] = "run_" + hashlib.sha1(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    env_keys = [
        ("SHD_PREPROCESSED_ROOT", cfg["preprocessed_root"]), ("METHOD", METHOD),
        ("BPTT_REDUCE", BPTT_REDUCE), ("SEED", SEED),
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
        ("WANDB_DIAGNOSTIC_INTERVAL", WANDB_DIAGNOSTIC_INTERVAL),
        ("WANDB_DIAGNOSTIC_SAMPLE_INDEX", WANDB_DIAGNOSTIC_SAMPLE_INDEX),
        ("WANDB_SATURATED_FRACTION", WANDB_SATURATED_FRACTION),
    ]
    cfg["reproduce_command"] = " ".join(f"{k}={v}" for k, v in env_keys) + \
        " python3 train_shd_output_lif_c35_dt1ms.py"
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
                     tags=["shd", "output-lif", "c35", "dt1ms", METHOD, READOUT_NAME])
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("ridge/*", step_metric="epoch")
        run.define_metric("runtime/*", step_metric="epoch")
        run.define_metric("diagnostics/*", step_metric="epoch")
        run.define_metric("hist/*", step_metric="epoch")
        run.define_metric("media/*", step_metric="epoch")
    return run


def dump_results_offline(cfg: Dict[str, object], final: Dict[str, object], ckpt_path: Path) -> None:
    rec = {
        "run_id": cfg["run_id"], "run_name": RUN_NAME, "output_type": cfg["output_type"],
        "method": METHOD, "readout": cfg["readout"],
        "bptt_reduce": BPTT_REDUCE, "eval_reduce": EVAL_REDUCE,
        "seed": SEED, "removed_class": REMOVED_CLASS, "dt_ms": DT_MS, "nb_steps": NB_STEPS,
        "num_inputs": NUM_INPUTS, "num_hidden": NUM_HIDDEN,
        "train_w_in": TRAIN_W_IN, "train_w_rec": TRAIN_W_REC, "train_w_out": TRAIN_W_OUT,
        "val_acc": final.get("val/acc"), "val_loss": final.get("val/loss"),
        "test_acc": final.get("test/acc"), "test_loss": final.get("test/loss"),
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


def save_and_log_checkpoint(model: "ReservoirSNN", cfg: Dict[str, object],
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
    print(f"device={device} method={METHOD} readout={READOUT_NAME} eval_reduce={EVAL_REDUCE} "
          f"inputs={NUM_INPUTS} dt={DT_MS}ms nb_steps={NB_STEPS} "
          f"train(in,rec,out)=({TRAIN_W_IN},{TRAIN_W_REC},{TRAIN_W_OUT}) "
          f"AMP={AMP} GPU_CACHE={GPU_CACHE} compile={TORCH_COMPILE} "
          f"alpha={ALPHA:.4f} beta={BETA:.4f} diag_every={WANDB_DIAGNOSTIC_INTERVAL}")

    root = Path(PREPROCESSED_ROOT)
    train = SplitData(root, "pretrain_train", device, GPU_CACHE)
    val = SplitData(root, "pretrain_val", device, GPU_CACHE)
    test = SplitData(root, "pretrain_test", device, GPU_CACHE)
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    cfg = build_config(len(train), len(val), len(test))
    wandb_run = init_wandb(cfg)
    print(f"run_id={cfg['run_id']} run_name={RUN_NAME}")

    model = ReservoirSNN().to(device)
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
        # Init-weight histograms (epoch 0) for the init-vs-trained comparison.
        log_weight_and_grad_histograms(wandb_run, real, epoch=0)

    start = time.time()
    final_epoch = max(NUM_EPOCHS - 1, 0)
    if METHOD == "bptt":
        info = train_bptt(model, train, val, device, wandb_run)   # restores best-val weights
    else:
        info = train_ridge(real, train, device, wandb_run)

    val_m = evaluate(model, val, device, reduce=EVAL_REDUCE)
    test_m = evaluate(model, test, device, reduce=EVAL_REDUCE)
    final = {**info,
             "val/acc": val_m["acc"], "val/loss": val_m["loss"],
             "val/hidden_firing_rate": val_m["hidden_firing_rate"],
             "test/acc": test_m["acc"], "test/loss": test_m["loss"],
             "test/hidden_firing_rate": test_m["hidden_firing_rate"],
             "eval_reduce": EVAL_REDUCE, "runtime/total_seconds": time.time() - start}
    print(f"\n=== {READOUT_NAME} ({METHOD}) ===  val_acc({EVAL_REDUCE})={val_m['acc']:.4f}  "
          f"val_loss={val_m['loss']:.4f}  test_acc({EVAL_REDUCE})={test_m['acc']:.4f}  "
          f"hidden_hz(test)={test_m['hidden_firing_rate']:.3f}")

    # Final rich media on the finalised model (best-val for BPTT; fitted W_out for ridge).
    log_final_media(wandb_run, real, train, val, device, epoch=final_epoch)

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
