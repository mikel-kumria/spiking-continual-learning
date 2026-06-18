#!/usr/bin/env python3
"""Minimal SHD pretraining baseline for a 19-class class-incremental setup.

Loads already-preprocessed dense uint8 pickles produced by
``build_shd_class_incremental_from_h5.py`` (see ``PREPROCESSED_ROOT``) and
trains a single-layer recurrent reservoir SNN on the 19 pretraining classes.

Two training workflows are provided in clearly separated sections:

* **BPTT** (cross-entropy on the leaky output membrane), via
  ``run_bptt_training`` -- trains ``W_in/W_rec/W_out`` end-to-end.
* **Ridge** (closed-form one-vs-rest readout on time-averaged hidden state),
  via ``run_ridge_training`` -- freezes ``W_in/W_rec`` and replaces ``W_out``
  with the analytic solution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import snntorch as snn
from snntorch import surrogate


# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME = "shd-pretraining-baseline"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["shd", "pretraining", "class-incremental", "baseline"]

PREPROCESSED_ROOT = os.environ.get(
    "SHD_PREPROCESSED_ROOT",
    "/data/class_incremental/removed_class_10_dt14ms_bs1_dense",
).strip()
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts").strip()


def _read_preprocessing_manifest(root: str) -> dict:
    """Best-effort read of the dataset's ``preprocessing_manifest.json``.

    The preprocessor (``build_shd_snn_dataset.py`` /
    ``build_shd_class_incremental_from_h5.py``) records the held-out class,
    timestep, window length and channel count it built the data with. Reading
    them here makes the *preprocessor command the single source of truth*: the
    trainer no longer needs hand-edited constants kept in sync with the data.

    Returns ``{}`` if the file is absent or unreadable (e.g. the root does not
    exist yet); the hardcoded defaults below then apply, and the integrity guard
    ``_verify_preprocessing_manifest`` still runs at load time.
    """
    if not root:
        return {}
    try:
        text = (Path(root) / "preprocessing_manifest.json").read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_MANIFEST = _read_preprocessing_manifest(PREPROCESSED_ROOT)

# Resolution order for the data-defining constants: explicit env override >
# dataset manifest > hardcoded default. Env override is kept so a value can still
# be forced for debugging; in normal use the manifest (i.e. the preprocessor
# command) drives these, so sweeping a held-out class or timestep means changing
# it in ONE place. A genuine train/data mismatch is still caught downstream by
# ``_verify_preprocessing_manifest``.
REMOVED_CLASS = int(os.environ.get("REMOVED_CLASS", _MANIFEST.get("removed_class", 10)))
NUM_INPUTS = int(os.environ.get("NUM_INPUTS", _MANIFEST.get("nb_inputs", 700)))
NUM_OUTPUTS = 20
DT_MS = float(os.environ.get("DT_MS", _MANIFEST.get("dt_ms", 14.0)))
MAX_TIME_SECONDS = float(os.environ.get("MAX_TIME_SECONDS", _MANIFEST.get("max_time_seconds", 1.4)))
BATCH_SIZE = 64
NUM_WORKERS = 0

NUM_HIDDEN = 1000
TAU_SYN_MS = 5.0
TAU_MEM_MS = 10.0
THRESHOLD = 1.0
RESET_MECHANISM = "zero"
OUTPUT_THRESHOLD = 1.0e9  # effectively non-spiking output neuron
OUTPUT_RESET_MECHANISM = "none"
SURROGATE_GRADIENT_TYPE = "fast_sigmoid"
SURROGATE_SLOPE = 25.0
SUPPORTED_SURROGATES = ("fast_sigmoid", "atan", "sigmoid", "straight_through_estimator")
W_IN_SCALE = 1.0
W_REC_SCALE = 1.0
W_OUT_SCALE = 1.0
# sqrt-N (fan-in) initialization keeps the hidden current variance O(1) so the
# membrane actually reaches THRESHOLD and the network spikes from the start. The
# previous ``scale / N`` made the initial hidden current ~sqrt(N) too small,
# leaving the membrane ~50x below threshold (a silent, non-learning network).
# Exposed via env vars so the init scale can be swept.
W_IN_INIT_STD = float(os.environ.get("W_IN_INIT_STD", str(W_IN_SCALE / math.sqrt(NUM_INPUTS))))
W_REC_INIT_STD = float(os.environ.get("W_REC_INIT_STD", str(W_REC_SCALE / math.sqrt(NUM_HIDDEN))))
W_OUT_INIT_STD = float(os.environ.get("W_OUT_INIT_STD", str(W_OUT_SCALE / math.sqrt(NUM_HIDDEN))))

# 1=True, 0=False. Clamp hidden- and output-membrane traces to >=0 every timestep
# (deviates from a vanilla Synaptic-LIF; kept as a feature flag for ablations).
# Default OFF: clamping feeds a rectified state back into the recurrence, has zero
# gradient where mem<0 (blocking inhibitory pathways), and is not in the paper.
CLAMP_VMEM = bool(int(os.environ.get("CLAMP_VMEM", "0")))

# When 1=True, configure cuDNN/CUBLAS for bitwise-reproducible runs at the cost
# of some speed and possibly failing if an op lacks a deterministic kernel.
DETERMINISTIC = bool(int(os.environ.get("DETERMINISTIC", "1")))

TRAINING_MODES = (
    "ce_max_mem",
    "ce_mean_mem",
    "ridge_hidden_spike_avg",
    "ridge_hidden_mem_avg",
)
TRAINING_MODE = os.environ.get("TRAINING_MODE", "ce_max_mem")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "200"))
FIRING_RATE_DIAGNOSTIC_INTERVAL = int(os.environ.get("FIRING_RATE_DIAGNOSTIC_INTERVAL", "10"))
LEARNING_RATE = 2.0e-4
OPTIMIZER_TYPE = os.environ.get("OPTIMIZER_TYPE", "adamax").strip().lower()
RIDGE_ALPHA = 1.0e-3
# Max global grad norm for BPTT (recurrent SNNs over many timesteps are prone to
# gradient explosion). Set to 0 to disable clipping.
GRAD_CLIP_MAX_NORM = float(os.environ.get("GRAD_CLIP_MAX_NORM", "5.0"))
TRAIN_W_IN = bool(int(os.environ.get("TRAIN_W_IN", "1")))
TRAIN_W_REC = bool(int(os.environ.get("TRAIN_W_REC", "1")))
TRAIN_W_OUT = bool(int(os.environ.get("TRAIN_W_OUT", "1")))

WANDB_MODE = os.environ.get("WANDB_MODE", "online")

NB_STEPS = int(math.ceil(MAX_TIME_SECONDS / (DT_MS / 1000.0)))
ALPHA = math.exp(-(DT_MS / TAU_SYN_MS))
BETA = math.exp(-(DT_MS / TAU_MEM_MS))
ACTIVE_CLASSES = [c for c in range(NUM_OUTPUTS) if c != REMOVED_CLASS]

# Mode-aware human-readable run name. The full immutable identifier is the
# hash-derived RUN_ID (see ``_compute_run_id`` below).
RUN_NAME = f"{TRAINING_MODE}_shd-pretrain-removed{REMOVED_CLASS}_bin{int(round(DT_MS))}ms"


# =============================================================================
# Surrogate gradients & seeding
# =============================================================================


def make_surrogate_grad(surrogate_type: str, slope: float):
    """Dispatch an snnTorch surrogate-gradient by name."""
    name = (surrogate_type or "").lower()
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=float(slope))
    if name == "atan":
        return surrogate.atan(alpha=float(slope))
    if name == "sigmoid":
        return surrogate.sigmoid(slope=float(slope))
    if name in ("straight_through_estimator", "ste", "straight_through"):
        return surrogate.straight_through_estimator()
    raise ValueError(
        f"Unknown SURROGATE_GRADIENT_TYPE={surrogate_type!r}; supported: {SUPPORTED_SURROGATES}"
    )


def set_random_seeds(seed: int, deterministic: bool = DETERMINISTIC) -> None:
    """Seed RNGs and (optionally) put PyTorch in deterministic-algorithm mode.

    Bit reproducibility requires:
      * matching CUBLAS workspace config (env var, must be set *before* the
        first CUBLAS call);
      * cuDNN deterministic kernels + benchmark disabled;
      * ``torch.use_deterministic_algorithms`` enabled.
    Some snnTorch ops have no deterministic kernel; we therefore use
    ``warn_only=True`` so the run does not crash but warnings surface the issue.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Generator used to seed the DataLoader shuffle order reproducibly."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed numpy/random inside DataLoader workers so transforms are reproducible."""
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# =============================================================================
# Data: preprocessed pickles -> Dataset -> DataLoader
# =============================================================================


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _to_dense_uint8(x: torch.Tensor) -> torch.Tensor:
    if x.is_sparse:
        x = x.to_dense()
    # Binarize: dtype conversion alone would let a stray value of 2 (e.g. two
    # spikes in one bin from a future storage change) pass through and corrupt
    # the binary-spike assumption.
    x = (x > 0).to(torch.uint8)
    return x.cpu()


def _preprocessed_root() -> Path:
    if not PREPROCESSED_ROOT:
        raise ValueError("PREPROCESSED_ROOT is empty; set SHD_PREPROCESSED_ROOT.")
    root = Path(PREPROCESSED_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PREPROCESSED_ROOT does not exist: {root}. "
            "Run build_shd_class_incremental_from_h5.py to create it."
        )
    return root


def _verify_preprocessing_manifest(manifest: dict) -> None:
    if manifest.get("storage_format") != "dense_uint8":
        raise ValueError(f"Expected storage_format='dense_uint8', got {manifest.get('storage_format')!r}")
    if int(manifest["removed_class"]) != REMOVED_CLASS:
        raise ValueError(f"manifest removed_class={manifest['removed_class']} != {REMOVED_CLASS}")
    if int(manifest["nb_steps"]) != NB_STEPS:
        raise ValueError(f"manifest nb_steps={manifest['nb_steps']} != {NB_STEPS}")
    if int(manifest["nb_inputs"]) != NUM_INPUTS:
        raise ValueError(f"manifest nb_inputs={manifest['nb_inputs']} != {NUM_INPUTS}")
    if abs(float(manifest["dt_ms"]) - DT_MS) > 1e-9:
        raise ValueError(f"manifest dt_ms={manifest['dt_ms']} != {DT_MS}")
    splits = manifest.get("splits", {})
    if splits:
        n_train = int(splits.get("pretrain_train_n", -1))
        if n_train <= 0:
            raise ValueError(f"manifest pretrain_train_n={n_train} -- empty split?")
        n_val = int(splits.get("pretrain_val_n", -1))
        if n_val <= 0:
            raise ValueError(f"manifest pretrain_val_n={n_val} -- empty split?")


class ShdPickleDataset(Dataset):
    """One preprocessed split: list of dense uint8 batches on disk (no upfront concat).

    Each pickle list entry has shape ``[B, T, C]`` (often ``B=1``). ``DataLoader``
    stacks samples into training batches ``[B_train, T, C]``.
    """

    def __init__(self, root: Path, split_name: str) -> None:
        x_path = root / f"{split_name}_x_class_{REMOVED_CLASS}.pkl"
        y_path = root / f"{split_name}_y_class_{REMOVED_CLASS}.pkl"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Missing split pickles for {split_name!r}: {x_path}, {y_path}")

        x_batches: List[torch.Tensor] = _load_pickle(x_path)
        y_batches: List[torch.Tensor] = _load_pickle(y_path)
        if len(x_batches) != len(y_batches):
            raise ValueError(f"{split_name}: x/y batch list length mismatch")

        self._x_batches = [_to_dense_uint8(xb) for xb in x_batches]
        self._y_batches = [torch.as_tensor(yb).reshape(-1).cpu().long() for yb in y_batches]

        sizes = [int(xb.shape[0]) for xb in self._x_batches]
        self._cum_sizes = np.cumsum([0] + sizes, dtype=np.int64)

        for xb, yb in zip(self._x_batches, self._y_batches):
            assert xb.ndim == 3 and xb.shape[1:] == (NB_STEPS, NUM_INPUTS), tuple(xb.shape)
            assert int(yb.numel()) == int(xb.shape[0]), "x/y batch sample count mismatch"
            assert not torch.any(yb == REMOVED_CLASS).item(), f"{split_name} contains removed class"

    def __len__(self) -> int:
        return int(self._cum_sizes[-1])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_i = int(np.searchsorted(self._cum_sizes[1:], index, side="right"))
        local_i = int(index - self._cum_sizes[batch_i])
        return self._x_batches[batch_i][local_i], self._y_batches[batch_i][local_i]


def load_pretrain_datasets() -> Tuple[ShdPickleDataset, ShdPickleDataset]:
    """Load pretrain train/val ``Dataset``s from ``PREPROCESSED_ROOT``.

    The pretrain test split is intentionally not loaded here; it is reserved for
    a later stage.
    """
    root = _preprocessed_root()
    manifest_path = root / "preprocessing_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing manifest: {manifest_path}")
    _verify_preprocessing_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    print(f"Loading preprocessed SHD from {root.resolve()}")
    return (
        ShdPickleDataset(root, "pretrain_train"),
        ShdPickleDataset(root, "pretrain_val"),
    )


def make_loader(dataset: Dataset, shuffle: bool, generator: Optional[torch.Generator] = None) -> DataLoader:
    """``DataLoader`` yielding ``x [B,T,C]`` and ``y [B]``.

    A ``generator`` should be supplied for shuffled loaders to keep the order
    bitwise-reproducible across runs (see ``make_dataloader_generator``).
    """
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=False,
        generator=generator if shuffle else None,
        worker_init_fn=_seed_dataloader_worker if NUM_WORKERS > 0 else None,
    )


# =============================================================================
# Model: ReservoirSNN
# =============================================================================


class ReservoirSNN(nn.Module):
    """Recurrent hidden SNN with a non-spiking 20-output membrane readout.

    Same architecture and dynamics as the original baseline:
      * ``W_in`` [700, 1000], ``W_rec`` [1000, 1000], ``W_out`` [1000, 20]
      * Hidden layer: snnTorch ``Synaptic`` (alpha, beta) with surrogate grad.
      * Output layer: snnTorch ``Synaptic`` with effectively infinite threshold
        so it integrates without spiking.
      * If ``clamp_vmem`` is True the hidden/output membranes are clipped to
        ``>=0`` at every timestep (non-standard LIF dynamics; kept behind a
        flag so it can be ablated explicitly).
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        num_hidden: int = NUM_HIDDEN,
        num_outputs: int = NUM_OUTPUTS,
        train_w_in: bool = TRAIN_W_IN,
        train_w_rec: bool = TRAIN_W_REC,
        train_w_out: bool = TRAIN_W_OUT,
        clamp_vmem: bool = CLAMP_VMEM,
    ) -> None:
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.num_hidden = int(num_hidden)
        self.num_outputs = int(num_outputs)
        self.clamp_vmem = bool(clamp_vmem)

        W_in = torch.empty(self.num_inputs, self.num_hidden)
        W_rec = torch.empty(self.num_hidden, self.num_hidden)
        W_out = torch.empty(self.num_hidden, self.num_outputs)
        nn.init.normal_(W_in, mean=0.0, std=float(W_IN_INIT_STD))
        nn.init.normal_(W_rec, mean=0.0, std=float(W_REC_INIT_STD))
        nn.init.normal_(W_out, mean=0.0, std=float(W_OUT_INIT_STD))

        self.W_in = nn.Parameter(W_in, requires_grad=bool(train_w_in))
        self.W_rec = nn.Parameter(W_rec, requires_grad=bool(train_w_rec))
        self.W_out = nn.Parameter(W_out, requires_grad=bool(train_w_out))

        # Non-trainable ridge readout bias. Stays zero (and is unused) for BPTT;
        # populated by ``_install_ridge_readout`` for ridge runs. Registering it
        # here keeps the state_dict shape stable so any checkpoint reloads cleanly.
        self.register_buffer("ridge_bias", torch.zeros(self.num_outputs))

        spike_grad = make_surrogate_grad(SURROGATE_GRADIENT_TYPE, SURROGATE_SLOPE)
        self.hidden_neuron = snn.Synaptic(
            alpha=ALPHA,
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET_MECHANISM,
        )
        self.output_neuron = snn.Synaptic(
            alpha=ALPHA,
            beta=BETA,
            threshold=OUTPUT_THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=OUTPUT_RESET_MECHANISM,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the recurrent SNN over one dense spike batch ``[B,T,C]``.

        Returns a dict with:
          * ``output_mem_trace`` [B,T,20]   -- non-spiking output membrane
          * ``hidden_spike_trace`` [B,T,H]  -- hidden spikes
          * ``hidden_mem_trace`` [B,T,H]    -- hidden membrane (non-negative)
        """
        x = x.float()
        batch_size, nb_steps, channels = x.shape
        assert channels == self.num_inputs and nb_steps == NB_STEPS

        device, dtype = x.device, x.dtype
        syn_h = torch.zeros(batch_size, self.num_hidden, device=device, dtype=dtype)
        mem_h = torch.zeros_like(syn_h)
        spk_h_prev = torch.zeros_like(syn_h)
        syn_o = torch.zeros(batch_size, self.num_outputs, device=device, dtype=dtype)
        mem_o = torch.zeros_like(syn_o)

        hidden_spikes: List[torch.Tensor] = []
        hidden_mems: List[torch.Tensor] = []
        output_mems: List[torch.Tensor] = []
        for t in range(nb_steps):
            hidden_current = x[:, t, :] @ self.W_in + spk_h_prev @ self.W_rec
            spk_h, syn_h, mem_h = self.hidden_neuron(hidden_current, syn_h, mem_h)
            if self.clamp_vmem:
                mem_h = mem_h.clamp(min=0)
            _spk_o, syn_o, mem_o = self.output_neuron(spk_h @ self.W_out, syn_o, mem_o)
            if self.clamp_vmem:
                mem_o = mem_o.clamp(min=0)
            hidden_spikes.append(spk_h)
            hidden_mems.append(mem_h)
            output_mems.append(mem_o)
            spk_h_prev = spk_h

        return {
            "output_mem_trace": torch.stack(output_mems, dim=1),
            "hidden_spike_trace": torch.stack(hidden_spikes, dim=1),
            "hidden_mem_trace": torch.stack(hidden_mems, dim=1),
        }


# =============================================================================
# Active-class helpers (shared by BPTT and Ridge sections)
# =============================================================================


def _active_classes_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(ACTIVE_CLASSES, dtype=torch.long, device=device)


def select_active_logits(logits: torch.Tensor) -> torch.Tensor:
    """Drop the held-out output column: [B,20] -> [B,19]."""
    return logits[:, _active_classes_tensor(logits.device)]


def remap_labels_to_active_indices(labels: torch.Tensor) -> torch.Tensor:
    """Map original SHD labels (excluding REMOVED_CLASS) to indices in 0..18.

    Raises if any label equals ``REMOVED_CLASS`` or lies outside ``[0, NUM_OUTPUTS)``;
    this turns a silent ``-1`` (which would corrupt CrossEntropyLoss) into a
    loud, attributable failure.
    """
    labels_long = labels.long()
    if torch.any(labels_long == REMOVED_CLASS).item():
        raise AssertionError(
            f"remap_labels_to_active_indices: received REMOVED_CLASS={REMOVED_CLASS} in labels"
        )
    if torch.any((labels_long < 0) | (labels_long >= NUM_OUTPUTS)).item():
        raise AssertionError(
            f"remap_labels_to_active_indices: labels outside [0, {NUM_OUTPUTS})"
        )
    label_map = torch.full((NUM_OUTPUTS,), -1, dtype=torch.long, device=labels.device)
    label_map[_active_classes_tensor(labels.device)] = torch.arange(NUM_OUTPUTS - 1, device=labels.device)
    mapped = label_map[labels_long]
    assert torch.all(mapped >= 0).item(), "internal: remap produced -1 (REMOVED_CLASS slipped through)"
    return mapped


def accuracy_from_active_logits(active_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Accuracy in original SHD label space from active-class logits."""
    pred_active = active_logits.argmax(dim=1)
    pred_original = _active_classes_tensor(active_logits.device)[pred_active]
    return (pred_original == labels.long()).float().mean()


def _cl_reinit_stats(model: ReservoirSNN, family: str) -> Dict[str, object]:
    """Old-class ``W_out`` weight statistics for class-incremental re-init.

    The paper (Dequino et al. 2024, sec. V-B) shows the held-out neuron should be
    re-initialised from a Normal matched to the *existing* (old-class) readout
    weight distribution -- random/Xavier init collapses accuracy. We record those
    stats so the continual-learning stage can sample the new column correctly.
    """
    with torch.no_grad():
        active_cols = model.W_out[:, _active_classes_tensor(model.W_out.device)]  # [H, 19]
        old_class_w_mean = float(active_cols.mean().item())
        old_class_w_std = float(active_cols.std().item())
    return {
        "cl_reinit": {
            "old_class_w_mean": old_class_w_mean,
            "old_class_w_std": old_class_w_std,
            "held_out_col_policy": "random_init" if family == "bptt" else "zero",
            "paper_recommendation": (
                "Re-initialize removed-class W_out column from "
                "Normal(old_class_w_mean, old_class_w_std) before CL training."
            ),
        }
    }


# =============================================================================
# BPTT training section (cross-entropy on output membrane)
# =============================================================================


def _reduce_output_membrane(output_mem_trace: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce ``[B,T,20]`` output membrane to CE logits ``[B,20]``."""
    if mode == "ce_max_mem":
        return output_mem_trace.max(dim=1).values
    if mode == "ce_mean_mem":
        return output_mem_trace.mean(dim=1)
    raise ValueError(f"mode {mode!r} is not a CE membrane readout mode")


def _build_optimizer(model: ReservoirSNN) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters; check TRAIN_W_IN/TRAIN_W_REC/TRAIN_W_OUT")
    opt = OPTIMIZER_TYPE  # already normalized to lowercase at config time
    if opt == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE)
    if opt == "adamax":
        return torch.optim.Adamax(params, lr=LEARNING_RATE)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=LEARNING_RATE)
    raise ValueError(f"Unknown OPTIMIZER_TYPE={OPTIMIZER_TYPE!r}")


def _global_grad_norm(parameters) -> float:
    """L2 norm of concatenated gradients across ``parameters`` (no clipping)."""
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_sq += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total_sq)


def _should_log_firing_rate_diagnostics(epoch: int) -> bool:
    return FIRING_RATE_DIAGNOSTIC_INTERVAL > 0 and epoch % FIRING_RATE_DIAGNOSTIC_INTERVAL == 0


def _first_sample_firing_by_timestep(x: torch.Tensor, hidden_spike_trace: torch.Tensor) -> Dict[str, List[float]]:
    """Return per-timestep firing rates for the first sample in a batch."""
    return {
        "diagnostics/input_firing_rate_by_timestep": (
            x[0].detach().float().mean(dim=1).cpu().tolist()
        ),
        "diagnostics/hidden_firing_rate_by_timestep": (
            hidden_spike_trace[0].detach().float().mean(dim=1).cpu().tolist()
        ),
    }


def _train_one_epoch_ce(
    model: ReservoirSNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    mode: str,
    device: torch.device,
    epoch: int,
) -> Dict[str, object]:
    """Train one epoch. Returns aggregated loss/acc/grad-norm; does not log to W&B."""
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    grad_norm_sum, grad_norm_max, n_batches = 0.0, 0.0, 0
    input_spikes, input_sites = 0.0, 0
    hidden_spikes, hidden_sites = 0.0, 0
    diagnostic_payload: Dict[str, List[float]] = {}
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        traces = model(x)
        hidden_spike_trace = traces["hidden_spike_trace"]
        if not diagnostic_payload and _should_log_firing_rate_diagnostics(epoch):
            diagnostic_payload = _first_sample_firing_by_timestep(x, hidden_spike_trace)
        active_logits = select_active_logits(_reduce_output_membrane(traces["output_mem_trace"], mode))
        targets = remap_labels_to_active_indices(y)
        loss = loss_fn(active_logits, targets)
        loss.backward()
        batch_grad_norm = _global_grad_norm(trainable_params)  # pre-clip, for logging
        if GRAD_CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        acc = accuracy_from_active_logits(active_logits.detach(), y)
        n = int(y.numel())
        total_loss += float(loss.item()) * n
        total_acc += float(acc.item()) * n
        total_n += n
        grad_norm_sum += batch_grad_norm
        grad_norm_max = max(grad_norm_max, batch_grad_norm)
        n_batches += 1
        input_spikes += float(x.detach().sum().item())
        input_sites += int(x.numel())
        hidden_spikes += float(hidden_spike_trace.detach().sum().item())
        hidden_sites += int(hidden_spike_trace.numel())

    metrics: Dict[str, object] = {
        "train/loss": total_loss / max(total_n, 1),
        "train/acc": total_acc / max(total_n, 1),
        "train/grad_norm_mean": grad_norm_sum / max(n_batches, 1),
        "train/grad_norm_max": grad_norm_max,
        "train/input_firing_rate": input_spikes / max(input_sites, 1),
        "train/hidden_firing_rate": hidden_spikes / max(hidden_sites, 1),
    }
    metrics.update(diagnostic_payload)
    return metrics


@torch.no_grad()
def _evaluate_ce(model: ReservoirSNN, loader: DataLoader, loss_fn: nn.Module, mode: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    input_spikes, input_sites = 0.0, 0
    hidden_spikes, hidden_sites = 0.0, 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        traces = model(x)
        hidden_spike_trace = traces["hidden_spike_trace"]
        active_logits = select_active_logits(_reduce_output_membrane(traces["output_mem_trace"], mode))
        targets = remap_labels_to_active_indices(y)
        loss = loss_fn(active_logits, targets)
        acc = accuracy_from_active_logits(active_logits, y)
        n = int(y.numel())
        total_loss += float(loss.item()) * n
        total_acc += float(acc.item()) * n
        total_n += n
        input_spikes += float(x.sum().item())
        input_sites += int(x.numel())
        hidden_spikes += float(hidden_spike_trace.sum().item())
        hidden_sites += int(hidden_spike_trace.numel())
    return {
        "val/loss": total_loss / max(total_n, 1),
        "val/acc": total_acc / max(total_n, 1),
        "val/input_firing_rate": input_spikes / max(input_sites, 1),
        "val/hidden_firing_rate": hidden_spikes / max(hidden_sites, 1),
    }


def run_bptt_training(
    model: ReservoirSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    device: torch.device,
    wandb_run,
    artifact_dir: Path,
    config_snapshot: Dict[str, object],
) -> Dict[str, object]:
    """End-to-end BPTT pretraining with CE on the output membrane.

    Logs to W&B once per epoch (no per-batch logs). Saves a checkpoint and
    metadata under ``artifact_dir``.
    """
    optimizer = _build_optimizer(model)
    loss_fn = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []

    inference_path = f"output_mem_{mode}"
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_metrics = _train_one_epoch_ce(model, train_loader, optimizer, loss_fn, mode, device, epoch)
        val_metrics = _evaluate_ce(model, val_loader, loss_fn, mode, device)

        payload: Dict[str, object] = {}
        payload.update(train_metrics)
        payload.update(val_metrics)
        payload["runtime/epoch_seconds"] = time.time() - t0
        payload["epoch"] = epoch

        if wandb_run is not None:
            wandb_run.log(payload)
        history.append({
            "epoch": epoch,
            **{
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in payload.items()
            },
        })

        # Best-validation checkpointing: the last epoch is often not the best.
        val_acc = float(val_metrics["val/acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            _save_checkpoint(
                model=model,
                artifact_dir=artifact_dir,
                config_snapshot=config_snapshot,
                history=history,
                final_metrics={"best_val_acc": best_val_acc, "best_epoch": best_epoch},
                family="bptt",
                filename="checkpoint_best.pt",
                extra={"inference_path": inference_path, **_cl_reinit_stats(model, "bptt")},
            )

        print(
            f"epoch={epoch:03d} train_loss={payload['train/loss']:.4f} "
            f"train_acc={payload['train/acc']:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f}@{best_epoch} "
            f"grad_norm_mean={payload['train/grad_norm_mean']:.3e}"
        )

    final: Dict[str, float] = {"best_val_acc": best_val_acc, "best_epoch": best_epoch}

    ckpt_path = _save_checkpoint(
        model=model,
        artifact_dir=artifact_dir,
        config_snapshot=config_snapshot,
        history=history,
        final_metrics=final,
        family="bptt",
        filename="checkpoint_last.pt",
        extra={"inference_path": inference_path, **_cl_reinit_stats(model, "bptt")},
    )
    if wandb_run is not None:
        wandb_run.summary.update({
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "checkpoint_path": str(ckpt_path),
        })
    return {"history": history, "final_metrics": final, "checkpoint_path": str(ckpt_path)}


# =============================================================================
# Ridge training section (closed-form one-vs-rest readout on hidden state)
# =============================================================================


def _ridge_feature_from_traces(traces: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    """Time-averaged hidden feature ``[B, NUM_HIDDEN]`` for the given ridge mode."""
    if mode == "ridge_hidden_spike_avg":
        return traces["hidden_spike_trace"].mean(dim=1)
    if mode == "ridge_hidden_mem_avg":
        return traces["hidden_mem_trace"].mean(dim=1)
    raise ValueError(f"{mode!r} is not a ridge feature mode")


@torch.no_grad()
def _extract_ridge_features(
    model: ReservoirSNN, loader: DataLoader, mode: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward the model and time-average a hidden trace per sample."""
    model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        traces = model(x)
        feats.append(_ridge_feature_from_traces(traces, mode).detach())
        labels.append(y.detach())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _solve_ridge_active(X: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge solve on active classes with an explicit bias term.

    Returns ``(W_active [D,19], bias [19])``. The feature matrix is augmented with
    a column of ones and solved in float64 via a Cholesky factorisation:
      * the bias matters because hidden spike rates / membrane values are
        non-negative, so a bias-free hyperplane is forced through the origin;
      * float64 + Cholesky is far better conditioned than float32 normal
        equations (which square the condition number of a 1000-unit Gram matrix).
    """
    N = labels.numel()
    Y = torch.zeros((N, NUM_OUTPUTS - 1), dtype=torch.float64, device=X.device)
    Y[torch.arange(N, device=X.device), remap_labels_to_active_indices(labels)] = 1.0

    X64 = X.to(torch.float64)
    ones = torch.ones((N, 1), dtype=torch.float64, device=X.device)
    Xb = torch.cat([X64, ones], dim=1)  # [N, D+1]

    D1 = Xb.shape[1]
    # Regularize the weights but NOT the bias (standard ridge with intercept;
    # matches sklearn's ``fit_intercept=True``). The bias diagonal entry is 0.
    reg = torch.full((D1,), float(RIDGE_ALPHA), dtype=torch.float64, device=X.device)
    reg[-1] = 0.0
    A = Xb.T @ Xb + torch.diag(reg)
    L = torch.linalg.cholesky(A)  # A is SPD
    sol = torch.cholesky_solve(Xb.T @ Y, L)  # [D+1, 19]

    W_active = sol[:-1].to(X.dtype)  # [D, 19]
    bias = sol[-1].to(X.dtype)  # [19]
    return W_active, bias


def _install_ridge_readout(model: ReservoirSNN, W_active: torch.Tensor, bias: torch.Tensor) -> None:
    """Embed ridge weights + bias into ``model.W_out`` and ``model.ridge_bias``.

    The held-out output column stays zero; the per-class bias is stored as a
    non-trainable buffer so a reloaded checkpoint reproduces predictions.
    """
    D = int(W_active.shape[0])
    assert D == NUM_HIDDEN, f"expected feature dim == NUM_HIDDEN, got {D}"
    W_full = torch.zeros((D, NUM_OUTPUTS), dtype=W_active.dtype, device=W_active.device)
    W_full[:, _active_classes_tensor(W_active.device)] = W_active
    with torch.no_grad():
        model.W_out.copy_(W_full.to(device=model.W_out.device, dtype=model.W_out.dtype))

    bias_full = torch.zeros(NUM_OUTPUTS, dtype=bias.dtype, device=bias.device)
    bias_full[_active_classes_tensor(bias.device)] = bias
    bias_full = bias_full.to(device=model.W_out.device, dtype=model.W_out.dtype)
    with torch.no_grad():
        model.ridge_bias.copy_(bias_full)


@torch.no_grad()
def _evaluate_ridge_via_inference(
    model: ReservoirSNN, loader: DataLoader, mode: str, device: torch.device
) -> Dict[str, float]:
    """Run a full forward pass on ``model`` (using the installed ``W_out``) and
    measure accuracy.

    NOTE: Ridge prediction does NOT use the output neuron's membrane dynamics.
    Prediction is ``logits = mean_t(hidden_trace) @ W_out [+ ridge_bias]``.
    This differs from the BPTT path (``output_mem_trace`` -> reduce -> logits).
    A reloaded checkpoint reproduces these numbers ONLY via this same path; use
    the ``predict()`` helper, which dispatches on the recorded ``inference_path``.

    Returns a dict with ``loss`` (mean-squared error to one-hot targets, the
    objective that ridge minimizes) and ``acc``.
    """
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    input_spikes, input_sites = 0.0, 0
    hidden_spikes, hidden_sites = 0.0, 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        traces = model(x)
        hidden_spike_trace = traces["hidden_spike_trace"]
        feat = _ridge_feature_from_traces(traces, mode)
        logits = feat @ model.W_out  # [B, NUM_OUTPUTS]; held-out column is zero
        if getattr(model, "ridge_bias", None) is not None:
            logits = logits + model.ridge_bias
        active_logits = select_active_logits(logits)  # [B, 19]

        # MSE against one-hot targets in the active-class space (the ridge loss).
        targets = remap_labels_to_active_indices(y)
        one_hot = torch.zeros_like(active_logits)
        one_hot[torch.arange(active_logits.shape[0], device=active_logits.device), targets] = 1.0
        n = int(y.numel())
        total_loss += float(((active_logits - one_hot) ** 2).sum().item())
        total_n += n

        pred_active = active_logits.argmax(dim=1)
        pred_original = _active_classes_tensor(active_logits.device)[pred_active]
        total_correct += int((pred_original == y.long()).sum().item())
        input_spikes += float(x.sum().item())
        input_sites += int(x.numel())
        hidden_spikes += float(hidden_spike_trace.sum().item())
        hidden_sites += int(hidden_spike_trace.numel())

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
        "input_firing_rate": input_spikes / max(input_sites, 1),
        "hidden_firing_rate": hidden_spikes / max(hidden_sites, 1),
    }


@torch.no_grad()
def predict(model: ReservoirSNN, x: torch.Tensor, inference_path: str) -> torch.Tensor:
    """Return active-class logits ``[B, 19]`` using the recorded inference path.

    ``inference_path`` values (as stored in the checkpoint ``extra`` field):
      * ``"output_mem_ce_max_mem"`` / ``"output_mem_ce_mean_mem"`` -- BPTT readout
        on the output-neuron membrane.
      * ``"ridge_hidden_spike_avg"`` / ``"ridge_hidden_mem_avg"`` -- ridge readout
        on the time-averaged hidden trace (+ ``model.ridge_bias`` if present).
    """
    traces = model(x)
    if inference_path.startswith("output_mem_"):
        mode = inference_path[len("output_mem_"):]
        return select_active_logits(_reduce_output_membrane(traces["output_mem_trace"], mode))
    if inference_path.startswith("ridge_hidden_"):
        feat = _ridge_feature_from_traces(traces, inference_path)
        logits = feat @ model.W_out
        if getattr(model, "ridge_bias", None) is not None:
            logits = logits + model.ridge_bias
        return select_active_logits(logits)
    raise ValueError(f"Unknown inference_path: {inference_path!r}")


def run_ridge_training(
    model: ReservoirSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    device: torch.device,
    wandb_run,
    artifact_dir: Path,
    config_snapshot: Dict[str, object],
) -> Dict[str, object]:
    """One-shot ridge pretraining.

    1. Freeze the model.
    2. Extract time-averaged hidden features on the training split.
    3. Solve the (bias-free) ridge problem on the active classes.
    4. Install the result into ``model.W_out``.
    5. Evaluate train/val by running full model inference -- this uses the
       installed ``W_out`` so the numbers match what a re-loaded checkpoint
       would produce.
    """
    model.requires_grad_(False)
    t0 = time.time()

    # Step 1-2: features needed only for the fit (training set).
    X_train, y_train = _extract_ridge_features(model, train_loader, mode, device)

    # Step 3-4: solve and install. After this point, all evaluation goes
    # through model inference using ``model.W_out`` (+ ``model.ridge_bias``).
    W_active, bias = _solve_ridge_active(X_train, y_train)
    _install_ridge_readout(model, W_active, bias)

    # Step 5: evaluate via inference (uses the installed W_out).
    train_eval = _evaluate_ridge_via_inference(model, train_loader, mode, device)
    val_eval = _evaluate_ridge_via_inference(model, val_loader, mode, device)
    payload: Dict[str, float] = {
        "ridge/train_loss": train_eval["loss"],
        "ridge/train_acc": train_eval["acc"],
        "ridge/val_loss": val_eval["loss"],
        "ridge/val_acc": val_eval["acc"],
        "train/input_firing_rate": train_eval["input_firing_rate"],
        "train/hidden_firing_rate": train_eval["hidden_firing_rate"],
        "val/input_firing_rate": val_eval["input_firing_rate"],
        "val/hidden_firing_rate": val_eval["hidden_firing_rate"],
        "runtime/ridge_seconds": time.time() - t0,
        "epoch": 0,
    }
    final: Dict[str, float] = {"val_acc": float(val_eval["acc"]), "train_acc": float(train_eval["acc"])}

    print(payload)
    if wandb_run is not None:
        wandb_run.log(payload)

    # inference_path encodes how predictions are produced (ridge on the time-
    # averaged hidden trace). ``mode`` is already one of the ridge_hidden_* modes,
    # so a reloaded checkpoint can dispatch on it directly via ``predict``.
    inference_path = mode
    ckpt_path = _save_checkpoint(
        model=model,
        artifact_dir=artifact_dir,
        config_snapshot=config_snapshot,
        history=[payload],
        final_metrics=final,
        family="ridge",
        extra={"inference_path": inference_path, **_cl_reinit_stats(model, "ridge")},
    )
    if wandb_run is not None:
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
    return {"metrics": payload, "final_metrics": final, "checkpoint_path": str(ckpt_path)}


# =============================================================================
# Config snapshot, run id, artifact dir, checkpointing
# =============================================================================


def build_full_config() -> Dict[str, object]:
    """Return a dict capturing *every* value required to reconstruct the run.

    Anything that influences the model, the data, the optimizer, the loss, the
    reduction, the seeding, or the artifact layout MUST be in here. This is
    used as the W&B config dict, written to ``metadata.json`` next to the
    checkpoint, and hashed to produce the immutable ``RUN_ID``.
    """
    return {
        # Identity
        "project_name": PROJECT_NAME,
        "run_name": RUN_NAME,
        "training_mode": TRAINING_MODE,
        "tags": list(TAGS),
        # Reproducibility
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        # Data
        "preprocessed_root": PREPROCESSED_ROOT,
        "removed_class": REMOVED_CLASS,
        "active_classes": list(ACTIVE_CLASSES),
        "num_inputs": NUM_INPUTS,
        "num_outputs": NUM_OUTPUTS,
        "dt_ms": DT_MS,
        "max_time_seconds": MAX_TIME_SECONDS,
        "nb_steps": NB_STEPS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        # Model
        "num_hidden": NUM_HIDDEN,
        "tau_syn_ms": TAU_SYN_MS,
        "tau_mem_ms": TAU_MEM_MS,
        "alpha": ALPHA,
        "beta": BETA,
        "threshold": THRESHOLD,
        "reset_mechanism": RESET_MECHANISM,
        "output_threshold": OUTPUT_THRESHOLD,
        "output_reset_mechanism": OUTPUT_RESET_MECHANISM,
        "surrogate_gradient_type": SURROGATE_GRADIENT_TYPE,
        "surrogate_slope": SURROGATE_SLOPE,
        "w_in_scale": W_IN_SCALE,
        "w_rec_scale": W_REC_SCALE,
        "w_out_scale": W_OUT_SCALE,
        "w_in_init_std": W_IN_INIT_STD,
        "w_rec_init_std": W_REC_INIT_STD,
        "w_out_init_std": W_OUT_INIT_STD,
        "clamp_vmem": CLAMP_VMEM,
        # Training
        "num_epochs": NUM_EPOCHS,
        "firing_rate_diagnostic_interval": FIRING_RATE_DIAGNOSTIC_INTERVAL,
        "learning_rate": LEARNING_RATE,
        "optimizer_type": OPTIMIZER_TYPE,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "ridge_alpha": RIDGE_ALPHA,
        "train_w_in": TRAIN_W_IN,
        "train_w_rec": TRAIN_W_REC,
        "train_w_out": TRAIN_W_OUT,
        # Logging / artifacts
        "wandb_mode": WANDB_MODE,
        "artifacts_root": ARTIFACTS_ROOT,
    }


def _compute_run_id(cfg: Dict[str, object]) -> str:
    """Stable 16-hex-char run id derived from a JSON-serialised config dict."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return "run_" + hashlib.sha1(payload).hexdigest()[:16]


def _resolve_artifact_dir(cfg: Dict[str, object], run_id: str) -> Path:
    """Artifact directory: ``<ARTIFACTS_ROOT>/<TRAINING_MODE>/<run_id>``."""
    root = Path(str(cfg["artifacts_root"]))
    mode_subdir = str(cfg["training_mode"])
    return (root / mode_subdir / run_id).resolve()


def _save_checkpoint(
    model: ReservoirSNN,
    artifact_dir: Path,
    config_snapshot: Dict[str, object],
    history: List[Dict[str, object]],
    final_metrics: Dict[str, float],
    family: str,
    filename: str = "checkpoint.pt",
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Persist model weights + metadata sufficient to fully reconstruct the run.

    Writes:
      * ``<filename>`` -- ``state_dict``, the immutable config snapshot, the
        W&B-style metrics history, and any ``extra`` fields (e.g. the recorded
        ``inference_path`` and CL re-init stats).
      * ``<filename>_metadata.json`` -- the same content in human-readable form.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifact_dir / filename
    # Keep the canonical "metadata.json" name for the default checkpoint (for
    # backward compatibility with downstream readers); suffix the variants.
    if filename == "checkpoint.pt":
        meta_name = "metadata.json"
    elif filename.endswith(".pt"):
        meta_name = filename[:-3] + "_metadata.json"
    else:
        meta_name = filename + "_metadata.json"
    meta_path = artifact_dir / meta_name

    saved_at = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, object] = {
        "state_dict": model.state_dict(),
        "config": config_snapshot,
        "family": family,
        "history": history,
        "final_metrics": final_metrics,
        "saved_at_utc": saved_at,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)

    metadata: Dict[str, object] = {
        "family": family,
        "config": config_snapshot,
        "history": history,
        "final_metrics": final_metrics,
        "checkpoint_file": ckpt_path.name,
        "saved_at_utc": saved_at,
    }
    if extra:
        # state_dict is not JSON-serialisable, but ``extra`` carries only metadata.
        metadata.update(extra)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"saved checkpoint to {ckpt_path}")
    return ckpt_path


# =============================================================================
# Entry point
# =============================================================================


def _init_wandb(config_snapshot: Dict[str, object], run_id: str):
    """Initialize a W&B run with the full reproducibility-relevant config.

    The local immutable ``run_id`` (config hash) is recorded in the W&B config
    and summary so it can be cross-referenced to the artifact directory, but
    it is NOT used as the W&B run id -- doing so would prevent the user from
    re-running the same configuration without manually resuming.
    """
    import wandb

    run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config=config_snapshot,
        tags=TAGS,
        mode=WANDB_MODE,
    )
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("ridge/*", step_metric="epoch")
        run.define_metric("diagnostics/*", step_metric="epoch")
        run.define_metric("runtime/epoch_seconds", step_metric="epoch")
    return run


def main() -> int:
    if TRAINING_MODE not in TRAINING_MODES:
        raise ValueError(f"TRAINING_MODE={TRAINING_MODE!r}; valid modes are {TRAINING_MODES}")

    set_random_seeds(SEED, deterministic=DETERMINISTIC)
    device = torch.device(DEVICE)

    config_snapshot = build_full_config()
    run_id = _compute_run_id(config_snapshot)
    config_snapshot["run_id"] = run_id  # carry the id inside the snapshot too
    artifact_dir = _resolve_artifact_dir(config_snapshot, run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} artifact_dir={artifact_dir}")

    wandb_run = _init_wandb(config_snapshot, run_id)

    train_ds, val_ds = load_pretrain_datasets()
    print(f"splits: train={len(train_ds)} val={len(val_ds)}")
    train_generator = make_dataloader_generator(SEED)
    train_loader = make_loader(train_ds, shuffle=True, generator=train_generator)
    val_loader = make_loader(val_ds, shuffle=False)

    model = ReservoirSNN().to(device)

    # Smoke test: a freshly-initialised reservoir must actually spike. A near-zero
    # rate means the init scale / threshold are wrong and no learning will happen.
    with torch.no_grad():
        x_smoke, _ = next(iter(train_loader))
        x_smoke = x_smoke.to(device=device, dtype=torch.float32)
        out_smoke = model(x_smoke)
        hidden_rate = float(out_smoke["hidden_spike_trace"].mean().item())
    print(f"[smoke] initial hidden firing rate: {hidden_rate:.4f}")
    if not (0.005 <= hidden_rate <= 0.50):
        raise RuntimeError(
            f"Initial hidden firing rate {hidden_rate:.4f} is outside [0.5%, 50%]. "
            f"Adjust THRESHOLD or W_IN_INIT_STD before training."
        )

    start = time.time()
    if TRAINING_MODE in ("ce_max_mem", "ce_mean_mem"):
        run_bptt_training(
            model, train_loader, val_loader, TRAINING_MODE, device,
            wandb_run, artifact_dir, config_snapshot,
        )
    else:
        run_ridge_training(
            model, train_loader, val_loader, TRAINING_MODE, device,
            wandb_run, artifact_dir, config_snapshot,
        )
    total = time.time() - start

    if wandb_run is not None:
        wandb_run.log({"runtime/total_seconds": total})
        wandb_run.summary["artifact_dir"] = str(artifact_dir)
        wandb_run.summary["run_id"] = run_id
        wandb_run.finish()
    print(f"done in {total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
