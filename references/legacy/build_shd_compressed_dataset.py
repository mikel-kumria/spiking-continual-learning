#!/usr/bin/env python3
"""Preprocess SHD into SNN-ready splits WITH channel compression.

This script combines:
  * the dataset-builder logic of ``build_shd_snn_dataset.py`` (HDF5 loading,
    optional train+test merge, class-incremental partition, fixed-width temporal
    binning, label-stratified train/val/test splits, trainer-compatible pickle
    layout), and
  * the channel-compression logic of ``shd_channel_compression.ipynb`` (adjacent
    channel grouping + OR pooling on the channel axis only).

Pipeline
--------
1.  Read official SHD ``shd_train.h5`` and ``shd_test.h5`` event files.
2.  Optionally MERGE train+test into one pool, then carve every split from it.
3.  Partition the pool by class:
      * pretrain : every class EXCEPT ``--removed-class`` (the base task)
      * continual: ONLY ``--removed-class``              (the new task)
4.  Bin each event sample into a dense ``[nb_steps, nb_inputs]`` uint8 spike
    matrix with fixed-width ``dt = dt_ms/1000`` bins. ``nb_steps =
    ceil(max_time / dt)``.
5.  **FINAL STEP — compress the channel axis**: ``[nb_steps, 700]`` ->
    ``[nb_steps, N_compressed_channels]`` via adjacent-channel OR pooling. Time is
    NEVER touched; output stays binary {0,1}.
6.  Label-stratified train/val/test split of pretrain and continual.
7.  Persist each split as a list of ``[1, nb_steps, N_compressed_channels]`` uint8
    tensors (+ aligned label/speaker pickles) and a JSON manifest.

Tensor-shape contract (the single highest-risk area — read this)
----------------------------------------------------------------
Every stage keeps the convention ``[..., TIME, CHANNEL]`` with CHANNEL last:

    raw events                 -> dense bin   : [T, 700]        (uint8, binary)
    OR-pool (channel axis only): [T, 700] -> [T, C_comp]        (uint8, binary)
    add batch dim for saving   : [T, C_comp] -> [1, T, C_comp]  (uint8, binary)
    trainer concatenates       : N x [1,T,C] -> [N, T, C_comp]

OR pooling reshapes ONLY the last axis ``700 -> (C_comp, factor)`` and reduces
``dim=-1``. It can therefore never pool over time, never transpose T<->C, and
never emit ``[C, T]``. The manifest's ``nb_inputs`` is set to ``C_comp`` so the
downstream trainer (which does ``NUM_INPUTS = manifest["nb_inputs"]`` and asserts
``x.shape[1:] == (nb_steps, NUM_INPUTS)``) agrees with the saved channel count.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

NUM_CLASSES = 20  # SHD: spoken digits 0-9 in English + German
DEFAULT_NB_INPUTS = 700  # native SHD cochlea channel count


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------


@dataclass
class EventPool:
    """All samples held in memory as ragged per-sample arrays."""

    times: List[np.ndarray]   # firing times in seconds, one array per sample
    units: List[np.ndarray]   # channel index per spike, aligned with ``times``
    labels: np.ndarray        # int label per sample          [N]
    speakers: np.ndarray      # int speaker id per sample      [N]

    def __len__(self) -> int:
        return len(self.labels)


def read_h5_pool(path: Path) -> EventPool:
    """Load one SHD HDF5 file into an in-memory :class:`EventPool`."""
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        times = [np.asarray(t, dtype=np.float64) for t in f["spikes"]["times"]]
        units = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
        labels = np.asarray(f["labels"], dtype=np.int64)
        if "extra" in f and "speaker" in f["extra"]:
            speakers = np.asarray(f["extra"]["speaker"], dtype=np.int64)
        else:
            # Speaker metadata is optional; sentinel -1 makes speaker-stratified
            # experiments fail loudly later rather than silently mislabel.
            speakers = np.full(len(labels), -1, dtype=np.int64)
    if not (len(times) == len(units) == len(labels) == len(speakers)):
        raise ValueError(f"{path}: ragged per-sample arrays have mismatched lengths")
    return EventPool(times=times, units=units, labels=labels, speakers=speakers)


def merge_pools(pools: Sequence[EventPool]) -> EventPool:
    """Concatenate several pools along the sample axis."""
    times: List[np.ndarray] = []
    units: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    speakers: List[np.ndarray] = []
    for p in pools:
        times.extend(p.times)
        units.extend(p.units)
        labels.append(p.labels)
        speakers.append(p.speakers)
    return EventPool(
        times=times,
        units=units,
        labels=np.concatenate(labels, axis=0),
        speakers=np.concatenate(speakers, axis=0),
    )


# ---------------------------------------------------------------------------
# Temporal binning
# ---------------------------------------------------------------------------


def compute_nb_steps(max_time_s: float, dt_ms: float) -> int:
    """Number of temporal bins implied by a fixed timestep over ``[0, max_time)``."""
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    if max_time_s <= 0:
        raise ValueError("max_time_s must be positive")
    dt_s = dt_ms / 1000.0
    return max(1, int(math.ceil(max_time_s / dt_s)))


def event_to_dense(
    times: np.ndarray,
    units: np.ndarray,
    *,
    nb_steps: int,
    nb_inputs: int,
    max_time_s: float,
    dt_s: float,
) -> torch.Tensor:
    """Bin one event sample into a dense ``[nb_steps, nb_inputs]`` uint8 matrix.

    Channel is the LAST axis. Spikes at ``t >= max_time_s`` are discarded;
    remaining spikes go to bin ``floor(t / dt_s)`` (clamped to ``nb_steps-1`` to
    absorb floating-point edge cases). Values are assigned (not incremented) so
    the dense matrix is already binary {0,1}.
    """
    x = torch.zeros((nb_steps, nb_inputs), dtype=torch.uint8)
    if len(times) == 0:
        return x
    t = np.asarray(times, dtype=np.float64)
    u = np.asarray(units, dtype=np.int64)
    if t.shape != u.shape:
        raise ValueError("times and units length mismatch")
    if np.any(u < 0) or np.any(u >= nb_inputs):
        raise ValueError(f"unit index outside [0, {nb_inputs})")
    if np.any(t < 0):
        raise ValueError("negative spike time encountered")

    keep = t < float(max_time_s)
    t, u = t[keep], u[keep]
    if t.size == 0:
        return x

    bins = np.floor(t / dt_s).astype(np.int64)
    np.clip(bins, 0, nb_steps - 1, out=bins)
    x[torch.from_numpy(bins), torch.from_numpy(u)] = 1
    return x


# ---------------------------------------------------------------------------
# Channel compression (final preprocessing step)
# ---------------------------------------------------------------------------


def validate_compression_factor(nb_inputs: int, n_compressed: int) -> int:
    """Return the integer compression factor or raise a clear ``ValueError``.

    For OR pooling the original channels are grouped into contiguous, EQUAL-sized
    blocks, so ``n_compressed`` must divide ``nb_inputs`` exactly. We never
    silently truncate or pad.
    """
    if n_compressed <= 0:
        raise ValueError(f"n_compressed_channels must be positive, got {n_compressed}")
    if n_compressed > nb_inputs:
        raise ValueError(
            f"n_compressed_channels={n_compressed} cannot exceed nb_inputs={nb_inputs}"
        )
    if nb_inputs % n_compressed != 0:
        raise ValueError(
            f"n_compressed_channels={n_compressed} must divide nb_inputs={nb_inputs} "
            f"exactly for OR pooling (got remainder {nb_inputs % n_compressed}); "
            f"choose a divisor of {nb_inputs} (e.g. "
            f"{[d for d in range(1, nb_inputs + 1) if nb_inputs % d == 0 and d <= 100]} ...)."
        )
    return nb_inputs // n_compressed


def or_pool_compress(x: torch.Tensor, factor: int) -> torch.Tensor:
    """OR-pool adjacent channels. Supports ``[T, C]`` and ``[B, T, C]``.

    Reshapes ONLY the last (channel) axis ``C -> (C//factor, factor)`` and ORs
    over the trailing ``factor`` axis. Time (axis ``-2``) is preserved exactly.
    Output is binary uint8 {0,1}. For ``factor == 1`` this is an identity copy.

    Shape: ``[..., T, C] -> [..., T, C // factor]``.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected [T, C] or [B, T, C], got shape {tuple(x.shape)}")
    c = x.shape[-1]
    if c % factor != 0:
        raise ValueError(f"channel count {c} not divisible by factor {factor}")
    target = c // factor
    grouped = x.reshape(*x.shape[:-1], target, factor)   # split ONLY the last axis
    compressed = (grouped > 0).any(dim=-1)               # OR over the factor axis
    return compressed.to(dtype=torch.uint8)


# Registry / dispatch so other methods can be added later without touching callers.
CompressionFn = Callable[[torch.Tensor, int], torch.Tensor]
COMPRESSION_METHODS: Dict[str, CompressionFn] = {
    "or_pool": or_pool_compress,
}


def compress_channels(x: torch.Tensor, method: str, factor: int) -> torch.Tensor:
    """Dispatch to a registered channel-compression method."""
    if method not in COMPRESSION_METHODS:
        raise ValueError(
            f"Unknown compression_method={method!r}; available: {sorted(COMPRESSION_METHODS)}"
        )
    return COMPRESSION_METHODS[method](x, factor)


def _assert_or_pool_invariant(original: torch.Tensor, compressed: torch.Tensor, factor: int) -> None:
    """Per-sample OR-pool invariants (cheap; run during materialisation).

    Checks, for a single ``[T, C]`` original and its ``[T, C_comp]`` compression:
      * binary output (max <= 1, min >= 0);
      * time axis preserved;
      * channel axis reduced by exactly ``factor``;
      * per-timestep compressed activity <= original activity over channels (OR
        can only merge coincident spikes, never create them).
    """
    assert compressed.shape[-2] == original.shape[-2], "OR pooling altered the time axis"
    assert compressed.shape[-1] * factor == original.shape[-1], "channel axis not reduced by factor"
    assert int(compressed.max()) <= 1 and int(compressed.min()) >= 0, "OR pooling broke binary range"
    # Coincident spikes inside a group collapse to one, so per-timestep sums shrink-or-equal.
    assert torch.all(compressed.sum(dim=-1) <= original.sum(dim=-1)), \
        "compressed per-timestep activity exceeds original (impossible for OR pooling)"


# ---------------------------------------------------------------------------
# Class filtering & dense materialisation (+ compression)
# ---------------------------------------------------------------------------


def select_indices_by_class(labels: np.ndarray, removed_class: int, keep_removed: bool) -> np.ndarray:
    """Sample indices for the pretrain (``keep_removed=False``) or continual
    (``keep_removed=True``) partition."""
    is_removed = labels == removed_class
    mask = is_removed if keep_removed else ~is_removed
    return np.nonzero(mask)[0]


def materialise_dense(
    pool: EventPool,
    indices: np.ndarray,
    *,
    nb_steps: int,
    nb_inputs: int,
    max_time_s: float,
    dt_s: float,
    compression_method: str,
    compression_factor: int,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    """Bin selected samples to dense uint8 then OR-pool the channel axis.

    Returns ``(x_list, y, speaker)`` where ``x_list[i]`` is a COMPRESSED
    ``[nb_steps, nb_inputs // factor]`` uint8 tensor aligned with ``y[i]`` /
    ``speaker[i]``. Only one full-resolution ``[T, 700]`` sample exists at a time
    (it is compressed then discarded), keeping memory bounded.
    """
    x_list: List[torch.Tensor] = []
    for idx in indices:
        dense = event_to_dense(
            pool.times[idx], pool.units[idx],
            nb_steps=nb_steps, nb_inputs=nb_inputs,
            max_time_s=max_time_s, dt_s=dt_s,
        )  # [T, 700], uint8, binary
        compressed = compress_channels(dense, compression_method, compression_factor)  # [T, C_comp]
        _assert_or_pool_invariant(dense, compressed, compression_factor)
        x_list.append(compressed)
    y = pool.labels[indices].astype(np.int64)
    spk = pool.speakers[indices].astype(np.int64)
    return x_list, y, spk


# ---------------------------------------------------------------------------
# Stratified three-way split
# ---------------------------------------------------------------------------


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    for name, v in (("train", train_frac), ("val", val_frac), ("test", test_frac)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}_fraction={v} must be in [0, 1]")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0 (got {total:.6f})")


def stratified_three_way(
    labels: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-stratified split returning ``(train_idx, val_idx, test_idx)``.

    Proportions are enforced within each class so val/test are not starved of
    rare classes. Each present class is guaranteed >= 1 training sample.
    """
    _validate_fractions(train_frac, val_frac, test_frac)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, l in enumerate(labels.tolist()):
        by_class[int(l)].append(i)

    for cls in sorted(by_class):
        idxs = np.array(by_class[cls], dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        if val_frac > 0 and n_val == 0 and n >= 2:
            n_val = 1
        if test_frac > 0 and n_test == 0 and n >= 3:
            n_test = 1
        n_train = n - n_val - n_test
        if n_train < 1:  # never let a class vanish from the train split
            n_train = 1
            over = (n_val + n_test) - (n - 1)
            while over > 0 and n_test > 0:
                n_test -= 1
                over -= 1
            while over > 0 and n_val > 0:
                n_val -= 1
                over -= 1
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _as_batch_list(x_list: List[torch.Tensor], idx: np.ndarray) -> List[torch.Tensor]:
    """Gather selected COMPRESSED samples as a list of ``[1, T, C_comp]`` tensors."""
    return [x_list[int(i)].unsqueeze(0) for i in idx]


def save_split(
    out_dir: Path,
    split_name: str,
    removed_class: int,
    x_list: List[torch.Tensor],
    y: np.ndarray,
    spk: np.ndarray,
    idx: np.ndarray,
) -> Dict[int, int]:
    """Persist one split's X / y / speaker pickles. Returns a class histogram."""
    c = removed_class
    x_batches = _as_batch_list(x_list, idx)
    y_batches = [torch.tensor([int(y[int(i)])], dtype=torch.long) for i in idx]
    spk_batches = [torch.tensor([int(spk[int(i)])], dtype=torch.long) for i in idx]

    artifacts = {
        f"{split_name}_x_class_{c}.pkl": x_batches,
        f"{split_name}_y_class_{c}.pkl": y_batches,
        f"{split_name}_speaker_class_{c}.pkl": spk_batches,
    }
    for name, obj in artifacts.items():
        with (out_dir / name).open("wb") as f:
            pickle.dump(obj, f)

    hist: Dict[int, int] = {}
    for i in idx:
        k = int(y[int(i)])
        hist[k] = hist.get(k, 0) + 1
    return hist


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_check(
    splits: Dict[str, tuple],
    removed_class: int,
    *,
    nb_steps: int,
    n_compressed: int,
) -> None:
    """Fail loudly on leakage, empty mandatory splits, or shape/binary violations.

    Validates, across every split:
      * no removed class leaks into pretrain; continual is ONLY the removed class;
      * mandatory train splits are non-empty;
      * each compressed sample (pre-batch) has shape ``[T, n_compressed]``;
      * each saved tensor (post batch dim) has shape ``[1, T, n_compressed]``;
      * all tensors are binary (max <= 1, min >= 0).
    """
    pre_labels: List[int] = []
    con_labels: List[int] = []
    for name, (xl, y, _spk, idx) in splits.items():
        lab = [int(y[int(i)]) for i in idx]
        (con_labels if name.startswith("continual") else pre_labels).extend(lab)
        for i in idx:
            sample = xl[int(i)]
            # pre-batch shape: [T, C_comp], channel last, never [C, T]
            if sample.ndim != 2 or tuple(sample.shape) != (nb_steps, n_compressed):
                raise RuntimeError(
                    f"{name}: compressed sample has shape {tuple(sample.shape)}, "
                    f"expected ({nb_steps}, {n_compressed})"
                )
            if sample.numel() and (int(sample.max()) > 1 or int(sample.min()) < 0):
                raise RuntimeError(f"{name}: compressed sample is not binary")
            # saved shape: [1, T, C_comp]
            batched = sample.unsqueeze(0)
            if tuple(batched.shape) != (1, nb_steps, n_compressed):
                raise RuntimeError(
                    f"{name}: saved tensor would be {tuple(batched.shape)}, "
                    f"expected (1, {nb_steps}, {n_compressed})"
                )

    if removed_class in pre_labels:
        raise RuntimeError("Leakage: pretrain split contains the removed class.")
    if any(l != removed_class for l in con_labels):
        raise RuntimeError("Leakage: continual split contains a non-removed class.")
    for mandatory in ("pretrain_train", "continual_train"):
        if splits[mandatory][3].size == 0:
            raise RuntimeError(f"{mandatory} is empty; adjust fractions or data.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    train_h5: Path
    test_h5: Optional[Path]
    output_dir: Path
    removed_class: int
    dt_ms: float
    max_time_s: float
    nb_inputs: int
    n_compressed_channels: int
    compression_method: str
    pretrain_train_fraction: float
    pretrain_val_fraction: float
    pretrain_test_fraction: float
    continual_train_fraction: float
    continual_val_fraction: float
    continual_test_fraction: float
    seed: int
    merge: bool


def run(cfg: Config) -> None:
    if not 0 <= cfg.removed_class < NUM_CLASSES:
        raise ValueError(f"removed_class must be in [0, {NUM_CLASSES})")
    _validate_fractions(
        cfg.pretrain_train_fraction, cfg.pretrain_val_fraction, cfg.pretrain_test_fraction
    )
    _validate_fractions(
        cfg.continual_train_fraction, cfg.continual_val_fraction, cfg.continual_test_fraction
    )
    factor = validate_compression_factor(cfg.nb_inputs, cfg.n_compressed_channels)
    if cfg.compression_method not in COMPRESSION_METHODS:
        raise ValueError(
            f"Unknown compression_method={cfg.compression_method!r}; "
            f"available: {sorted(COMPRESSION_METHODS)}"
        )

    rng = np.random.default_rng(cfg.seed)
    dt_s = cfg.dt_ms / 1000.0
    nb_steps = compute_nb_steps(cfg.max_time_s, cfg.dt_ms)
    print(
        f"Temporal binning: dt={cfg.dt_ms:g} ms, max_time={cfg.max_time_s:g} s "
        f"-> nb_steps={nb_steps} bins (dt={dt_s:g} s)"
    )
    print(
        f"Channel compression: method={cfg.compression_method} "
        f"{cfg.nb_inputs} -> {cfg.n_compressed_channels} (factor {factor})"
    )

    # 1-2. Load and (optionally) merge into the pool every split is carved from.
    pools = [read_h5_pool(cfg.train_h5)]
    if cfg.merge:
        if cfg.test_h5 is None:
            raise ValueError("--test-h5 is required unless --no-merge is set")
        pools.append(read_h5_pool(cfg.test_h5))
    pool = merge_pools(pools) if len(pools) > 1 else pools[0]
    print(f"Pool: {len(pool)} samples from {len(pools)} file(s)")

    # 3. Partition by class.
    pre_idx = select_indices_by_class(pool.labels, cfg.removed_class, keep_removed=False)
    con_idx = select_indices_by_class(pool.labels, cfg.removed_class, keep_removed=True)
    print(f"pretrain pool: {len(pre_idx)} samples ({NUM_CLASSES - 1} classes)")
    print(f"continual pool: {len(con_idx)} samples (class {cfg.removed_class})")
    if len(con_idx) == 0:
        raise ValueError(
            f"No samples for removed_class={cfg.removed_class}; nothing to learn continually."
        )

    # 4-5. Materialise dense uint8 then OR-pool the channel axis (final step).
    pre_x, pre_y, pre_spk = materialise_dense(
        pool, pre_idx, nb_steps=nb_steps, nb_inputs=cfg.nb_inputs,
        max_time_s=cfg.max_time_s, dt_s=dt_s,
        compression_method=cfg.compression_method, compression_factor=factor,
    )
    con_x, con_y, con_spk = materialise_dense(
        pool, con_idx, nb_steps=nb_steps, nb_inputs=cfg.nb_inputs,
        max_time_s=cfg.max_time_s, dt_s=dt_s,
        compression_method=cfg.compression_method, compression_factor=factor,
    )

    # 6. Stratified three-way split (indices are local to each materialised list).
    pre_tr, pre_va, pre_te = stratified_three_way(
        pre_y, train_frac=cfg.pretrain_train_fraction,
        val_frac=cfg.pretrain_val_fraction, test_frac=cfg.pretrain_test_fraction, rng=rng,
    )
    con_tr, con_va, con_te = stratified_three_way(
        con_y, train_frac=cfg.continual_train_fraction,
        val_frac=cfg.continual_val_fraction, test_frac=cfg.continual_test_fraction, rng=rng,
    )

    splits = {
        "pretrain_train": (pre_x, pre_y, pre_spk, pre_tr),
        "pretrain_val": (pre_x, pre_y, pre_spk, pre_va),
        "pretrain_test": (pre_x, pre_y, pre_spk, pre_te),
        "continual_train": (con_x, con_y, con_spk, con_tr),
        "continual_val": (con_x, con_y, con_spk, con_va),
        "continual_test": (con_x, con_y, con_spk, con_te),
    }

    # Validate BEFORE writing anything to disk.
    _sanity_check(splits, cfg.removed_class, nb_steps=nb_steps,
                  n_compressed=cfg.n_compressed_channels)

    # 7. Save.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    histograms: Dict[str, Dict[int, int]] = {}
    counts: Dict[str, int] = {}
    for name, (xl, y, spk, idx) in splits.items():
        histograms[name] = save_split(cfg.output_dir, name, cfg.removed_class, xl, y, spk, idx)
        counts[name] = int(len(idx))

    write_manifest(cfg, nb_steps=nb_steps, dt_s=dt_s, factor=factor,
                   counts=counts, histograms=histograms)

    print("\nPreprocessing complete.")
    for name in splits:
        print(f"  {name:<16}: {counts[name]:>6} samples")
    print(f"  saved to        : {cfg.output_dir.resolve()}")
    print(f"  sample shape    : [1, {nb_steps}, {cfg.n_compressed_channels}] (uint8, binary)")


def write_manifest(
    cfg: Config,
    *,
    nb_steps: int,
    dt_s: float,
    factor: int,
    counts: Dict[str, int],
    histograms: Dict[str, Dict[int, int]],
) -> None:
    manifest = {
        "dataset": "SHD",
        "storage_format": "dense_uint8",
        "binning": "fixed_width_floor_seconds",
        "removed_class": cfg.removed_class,
        "active_classes": [c for c in range(NUM_CLASSES) if c != cfg.removed_class],
        "dt_ms": cfg.dt_ms,
        "dt_seconds": dt_s,
        "max_time_seconds": cfg.max_time_s,
        "nb_steps": nb_steps,
        # Downstream trainers read nb_inputs as THE channel count they expect, so
        # it must be the COMPRESSED count; the native count is kept separately.
        "nb_inputs": cfg.n_compressed_channels,
        "original_nb_inputs": cfg.nb_inputs,
        "n_compressed_channels": cfg.n_compressed_channels,
        "compression_method": cfg.compression_method,
        "compression_factor": factor,
        "binary_output": True,
        "dense_shape_per_sample": [1, nb_steps, cfg.n_compressed_channels],
        "merged_train_test": cfg.merge,
        "seed": cfg.seed,
        "train_h5": str(cfg.train_h5.resolve()),
        "test_h5": str(cfg.test_h5.resolve()) if (cfg.merge and cfg.test_h5) else None,
        "fractions": {
            "pretrain_train": cfg.pretrain_train_fraction,
            "pretrain_val": cfg.pretrain_val_fraction,
            "pretrain_test": cfg.pretrain_test_fraction,
            "continual_train": cfg.continual_train_fraction,
            "continual_val": cfg.continual_val_fraction,
            "continual_test": cfg.continual_test_fraction,
        },
        "splits": {
            **{f"{k}_n": v for k, v in counts.items()},
            **{f"{k}_class_hist": histograms[k] for k in histograms},
        },
        "benchmark_note": (
            "Merged official train+test then re-split randomly; NOT the SHD "
            "benchmark test set. Accuracies are not comparable to the paper."
        ) if cfg.merge else "Preprocessed --train-h5 only; official test set untouched.",
    }
    (cfg.output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-h5", type=Path, required=True, help="Path to shd_train.h5.")
    p.add_argument("--test-h5", type=Path, default=None,
                   help="Path to shd_test.h5 (merged with train unless --no-merge).")
    p.add_argument("--no-merge", action="store_true",
                   help="Use --train-h5 only; do NOT merge in the test set.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for pickles + manifest.")
    p.add_argument("--removed-class", type=int, required=True,
                   help="Class held out of pretrain and used as the continual task (0-19).")
    # --dt-ms is primary; --timestep-ms is a backwards-compatible alias.
    p.add_argument("--dt-ms", "--timestep-ms", dest="dt_ms", type=float, required=True,
                   help="Temporal bin width in ms; sets nb_steps = ceil(max_time/dt).")
    p.add_argument("--max-time", type=float, default=1.4,
                   help="Window length in seconds (default 1.4).")
    p.add_argument("--nb-inputs", type=int, default=DEFAULT_NB_INPUTS,
                   help=f"Native input channels before compression (default {DEFAULT_NB_INPUTS}).")
    p.add_argument("--n-compressed-channels", type=int, required=True,
                   help="Channel count AFTER compression; must divide --nb-inputs exactly.")
    p.add_argument("--compression-method", type=str, default="or_pool",
                   choices=sorted(COMPRESSION_METHODS),
                   help="Channel-compression method (default or_pool).")
    p.add_argument("--pretrain-train-fraction", type=float, default=0.70)
    p.add_argument("--pretrain-val-fraction", type=float, default=0.15)
    p.add_argument("--pretrain-test-fraction", type=float, default=0.15)
    p.add_argument("--continual-train-fraction", type=float, default=0.70)
    p.add_argument("--continual-val-fraction", type=float, default=0.15)
    p.add_argument("--continual-test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling and splitting.")
    a = p.parse_args(argv)

    return Config(
        train_h5=a.train_h5,
        test_h5=a.test_h5,
        output_dir=a.output_dir,
        removed_class=a.removed_class,
        dt_ms=a.dt_ms,
        max_time_s=a.max_time,
        nb_inputs=a.nb_inputs,
        n_compressed_channels=a.n_compressed_channels,
        compression_method=a.compression_method,
        pretrain_train_fraction=a.pretrain_train_fraction,
        pretrain_val_fraction=a.pretrain_val_fraction,
        pretrain_test_fraction=a.pretrain_test_fraction,
        continual_train_fraction=a.continual_train_fraction,
        continual_val_fraction=a.continual_val_fraction,
        continual_test_fraction=a.continual_test_fraction,
        seed=a.seed,
        merge=not a.no_merge,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        run(cfg)
    except Exception as e:  # noqa: BLE001 - surface a clean message to the CLI
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Self-test (run with: python build_shd_compressed_dataset.py --selftest)
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Tiny shape/unit tests for OR pooling. No HDF5 / no disk required."""
    print("Running OR-pool self-tests...")

    # 1) Known-answer correctness on a hand-built [T=2, C=10] -> [T=2, C=2] (factor 5).
    x = torch.zeros((2, 10), dtype=torch.uint8)
    x[0, 0] = 1          # group 0 (ch 0..4) fires at t=0
    x[0, 3] = 1          # also group 0 at t=0 (coincident -> collapses to one)
    x[1, 9] = 1          # group 1 (ch 5..9) fires at t=1
    out = or_pool_compress(x, factor=5)
    assert tuple(out.shape) == (2, 2), f"shape {tuple(out.shape)} != (2, 2)"
    expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
    assert torch.equal(out, expected), f"OR-pool value mismatch:\n{out}"
    # multiplicity collapse: original had 2 spikes at t=0, compressed has 1.
    assert int(x[0].sum()) == 2 and int(out[0].sum()) == 1

    # 2) 700 -> 70 (factor 10) keeps time, makes 70 channels, stays binary.
    T = 13
    x700 = (torch.rand(T, 700) > 0.7).to(torch.uint8)
    out70 = or_pool_compress(x700, factor=10)
    assert tuple(out70.shape) == (T, 70), tuple(out70.shape)
    assert int(out70.max()) <= 1 and int(out70.min()) >= 0
    # per-timestep activity can only shrink-or-equal under OR pooling.
    assert torch.all(out70.sum(dim=-1) <= x700.sum(dim=-1))
    _assert_or_pool_invariant(x700, out70, 10)

    # 3) Batched [B, T, C] support, time axis untouched.
    xb = (torch.rand(4, T, 700) > 0.7).to(torch.uint8)
    outb = or_pool_compress(xb, factor=10)
    assert tuple(outb.shape) == (4, T, 70), tuple(outb.shape)

    # 4) ANTI-TRANSPOSE guard: the compressed time dim must equal the input time
    #    dim, never the channel dim. If T != C this catches a [C, T] swap.
    assert out70.shape[0] == T and out70.shape[0] != 70

    # 5) factor must divide channels exactly.
    try:
        or_pool_compress(torch.zeros(2, 700, dtype=torch.uint8), factor=3)
        raise AssertionError("expected ValueError for non-divisor factor")
    except ValueError:
        pass
    try:
        validate_compression_factor(700, 33)  # 700 % 33 != 0
        raise AssertionError("expected ValueError for non-divisor n_compressed")
    except ValueError:
        pass
    assert validate_compression_factor(700, 70) == 10
    assert validate_compression_factor(700, 700) == 1  # identity is allowed

    # 6) factor==1 is identity.
    x_id = (torch.rand(3, 700) > 0.5).to(torch.uint8)
    assert torch.equal(or_pool_compress(x_id, 1), x_id)

    print("All OR-pool self-tests passed.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    raise SystemExit(main())
