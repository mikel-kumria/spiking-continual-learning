"""Preprocessing orchestration: events -> dense bins -> compression -> splits.

Two regimes are supported:

* ``baseline_20_class`` : all 20 classes -> ``train`` / ``test`` splits.
* ``pretrain_19_class`` : hold out ``removed_class`` -> ``pretrain_{train,test}``
  (19 old classes) + ``continual_{train,test}`` (the removed class only).

Everything is deterministic given ``seed``. Labels are kept as ORIGINAL SHD ids
(0..19) in every split; there is no global remap. The channel axis is compressed;
the time axis is never touched. A manifest with full provenance is returned.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import NATIVE_SHD_CHANNELS, NUM_CLASSES
from . import BASELINE_SPLIT_NAMES, SPLIT_NAMES
from .compression import (assert_compression_invariants, compress_channels,
                          validate_compression_factor, COMPRESSION_METHODS)
from .io import save_npz_split, write_json
from .shd_events import (EventPool, merge_pools, read_h5_pool, synthetic_pool)
from .splits import stratified_two_way, validate_two_fractions


# =============================================================================
# Temporal binning
# =============================================================================


def compute_nb_steps(max_time_s: float, dt_ms: float) -> int:
    """``nb_steps = ceil(max_time_s / (dt_ms / 1000))`` (>= 1)."""
    if dt_ms <= 0:
        raise ValueError("dataset_binning_ms must be positive")
    if max_time_s <= 0:
        raise ValueError("dataset_max_seconds must be positive")
    return max(1, int(math.ceil(max_time_s / (dt_ms / 1000.0))))


def event_to_dense(times: np.ndarray, units: np.ndarray, *, nb_steps: int,
                   nb_inputs: int, max_time_s: float, dt_s: float) -> np.ndarray:
    """Bin one event sample into a dense ``[nb_steps, nb_inputs]`` uint8 raster.

    Channel is the LAST axis. Spikes at ``t >= max_time_s`` are DROPPED; the rest
    map to ``floor(t / dt_s)`` (clamped to the last bin). Values are ASSIGNED, not
    accumulated -> the raster is binary ({0,1}).
    """
    x = np.zeros((nb_steps, nb_inputs), dtype=np.uint8)
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
    x[bins, u] = 1  # binary assignment, not spike-count accumulation
    return x


# =============================================================================
# Binner config + materialisation
# =============================================================================


@dataclass
class Binner:
    """Everything needed to turn selected event samples into dense compressed X."""
    nb_steps: int
    nb_inputs: int          # native channel count (before compression), e.g. 700
    max_time_s: float
    dt_s: float
    method: str
    factor: int
    condition_or: int
    bernoulli_seed: int

    @property
    def n_compressed(self) -> int:
        return self.nb_inputs // self.factor


def materialise(pool: EventPool, indices: np.ndarray, binner: Binner
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin then channel-compress the selected samples -> ``(X [N,T,Ccomp], y, spk)``.

    Only one full-resolution ``[T, nb_inputs]`` raster exists at a time (compressed
    then discarded), so peak memory stays bounded even at fine ``dt``.
    """
    rng = (np.random.default_rng(binner.bernoulli_seed)
           if binner.method == "bernoulli" else None)
    xs: List[np.ndarray] = []
    for idx in indices:
        dense = event_to_dense(
            pool.times[idx], pool.units[idx], nb_steps=binner.nb_steps,
            nb_inputs=binner.nb_inputs, max_time_s=binner.max_time_s,
            dt_s=binner.dt_s)                                    # [T, nb_inputs]
        comp = compress_channels(dense, binner.method, binner.factor,
                                 condition_or=binner.condition_or, rng=rng)
        assert_compression_invariants(dense, comp, binner.method, binner.factor)
        xs.append(comp)
    if xs:
        X = np.stack(xs, 0).astype(np.uint8, copy=False)
    else:
        X = np.zeros((0, binner.nb_steps, binner.n_compressed), dtype=np.uint8)
    y = pool.labels[indices].astype(np.int64)
    spk = pool.speakers[indices].astype(np.int64)
    return X, y, spk


# =============================================================================
# Event-pool loading (real HDF5 / merged / synthetic)
# =============================================================================


def load_event_pool(*, train_h5: Optional[str], test_h5: Optional[str],
                    merge_train_test: bool, synthetic_samples_per_class: int,
                    nb_inputs: int, max_time_s: float, seed: int
                    ) -> Tuple[EventPool, bool]:
    """Return ``(pool, merged_flag)`` from synthetic, single-file, or merged SHD."""
    if synthetic_samples_per_class > 0:
        pool = synthetic_pool(synthetic_samples_per_class, nb_inputs, max_time_s, seed)
        return pool, False
    if not train_h5:
        raise ValueError("train_h5 is required unless synthetic_samples_per_class > 0")
    pools = [read_h5_pool(train_h5)]
    merged = bool(merge_train_test)
    if merged:
        if not test_h5:
            raise ValueError("test_h5 is required when merge_train_test is True")
        pools.append(read_h5_pool(test_h5))
    pool = merge_pools(pools) if len(pools) > 1 else pools[0]
    return pool, merged


# =============================================================================
# Config + orchestration
# =============================================================================


@dataclass
class PreprocessConfig:
    experiment_regime: str = "pretrain_19_class"   # or "baseline_20_class"
    removed_class: int = 10
    dataset_binning_ms: float = 14.0
    dataset_max_seconds: float = 1.4
    nb_inputs: int = NATIVE_SHD_CHANNELS
    n_compressed_channels: int = 70
    channel_compression_method: str = "or_pool"
    condition_or: int = 1
    train_fraction: float = 0.80
    test_fraction: float = 0.20
    merge_train_test: bool = True
    seed: int = 42
    synthetic_samples_per_class: int = 0
    train_h5: Optional[str] = None
    test_h5: Optional[str] = None


def _histogram(y: np.ndarray) -> Dict[int, int]:
    return {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}


def _validate_cfg(cfg: PreprocessConfig) -> int:
    if cfg.experiment_regime not in ("baseline_20_class", "pretrain_19_class"):
        raise ValueError(f"unknown experiment_regime {cfg.experiment_regime!r}")
    if cfg.channel_compression_method not in COMPRESSION_METHODS:
        raise ValueError(
            f"unknown channel_compression_method {cfg.channel_compression_method!r}")
    validate_two_fractions(cfg.train_fraction, cfg.test_fraction)
    if cfg.experiment_regime == "pretrain_19_class":
        if not 0 <= cfg.removed_class < NUM_CLASSES:
            raise ValueError(f"removed_class must be in [0, {NUM_CLASSES})")
    return validate_compression_factor(cfg.nb_inputs, cfg.n_compressed_channels)


def preprocess(cfg: PreprocessConfig, dataset_dir: str) -> dict:
    """Build + save the split ``.npz`` files and a manifest. Returns the manifest."""
    factor = _validate_cfg(cfg)
    dt_s = cfg.dataset_binning_ms / 1000.0
    nb_steps = compute_nb_steps(cfg.dataset_max_seconds, cfg.dataset_binning_ms)
    binner = Binner(nb_steps=nb_steps, nb_inputs=cfg.nb_inputs, max_time_s=cfg.dataset_max_seconds,
                    dt_s=dt_s, method=cfg.channel_compression_method, factor=factor,
                    condition_or=cfg.condition_or, bernoulli_seed=cfg.seed)

    print(f"Temporal binning: dt={cfg.dataset_binning_ms:g} ms, "
          f"max_time={cfg.dataset_max_seconds:g} s -> nb_steps={nb_steps}")
    print(f"Channel compression: {cfg.channel_compression_method} "
          f"{cfg.nb_inputs} -> {cfg.n_compressed_channels} (factor {factor})")

    pool, merged = load_event_pool(
        train_h5=cfg.train_h5, test_h5=cfg.test_h5, merge_train_test=cfg.merge_train_test,
        synthetic_samples_per_class=cfg.synthetic_samples_per_class,
        nb_inputs=cfg.nb_inputs, max_time_s=cfg.dataset_max_seconds, seed=cfg.seed)
    print(f"Pool: {len(pool)} samples "
          f"({'synthetic' if cfg.synthetic_samples_per_class > 0 else 'SHD'})")

    rng = np.random.default_rng(cfg.seed)
    if cfg.experiment_regime == "baseline_20_class":
        splits, active_classes = _build_baseline(pool, binner, cfg, rng)
    else:
        splits, active_classes = _build_class_incremental(pool, binner, cfg, rng)

    # ---- sanity checks BEFORE writing anything ----
    _check_splits(cfg.experiment_regime, splits, cfg.removed_class, nb_steps,
                  cfg.n_compressed_channels)

    # ---- write splits + collect stats ----
    os.makedirs(dataset_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    histograms: Dict[str, Dict[int, int]] = {}
    for name, (X, y, spk) in splits.items():
        save_npz_split(os.path.join(dataset_dir, f"{name}.npz"), X, y, spk)
        counts[name] = int(len(y))
        histograms[name] = _histogram(y)

    manifest = _build_manifest(cfg, binner, merged, active_classes, counts, histograms)
    write_json(os.path.join(os.path.dirname(os.path.abspath(dataset_dir)),
                            "preprocessing_manifest.json"), manifest)
    print("Preprocessing complete:")
    for name in splits:
        print(f"  {name:<16}: {counts[name]:>6} samples")
    return manifest


def _build_baseline(pool, binner, cfg, rng):
    idx = np.arange(len(pool), dtype=np.int64)
    X, y, spk = materialise(pool, idx, binner)
    tr, te = stratified_two_way(y, train_frac=cfg.train_fraction,
                                test_frac=cfg.test_fraction, rng=rng)
    splits = {
        "train": (X[tr], y[tr], spk[tr]),
        "test": (X[te], y[te], spk[te]),
    }
    active_classes = list(range(NUM_CLASSES))
    return splits, active_classes


def _build_class_incremental(pool, binner, cfg, rng):
    is_removed = pool.labels == cfg.removed_class
    pre_idx = np.nonzero(~is_removed)[0]
    con_idx = np.nonzero(is_removed)[0]
    if len(con_idx) == 0:
        raise ValueError(f"no samples for removed_class={cfg.removed_class}")
    if len(pre_idx) == 0:
        raise ValueError("pretrain pool is empty")
    print(f"pretrain pool: {len(pre_idx)} ({NUM_CLASSES - 1} classes) | "
          f"continual pool: {len(con_idx)} (class {cfg.removed_class})")

    pre_X, pre_y, pre_spk = materialise(pool, pre_idx, binner)
    con_X, con_y, con_spk = materialise(pool, con_idx, binner)

    pre_tr, pre_te = stratified_two_way(
        pre_y, train_frac=cfg.train_fraction, test_frac=cfg.test_fraction, rng=rng)
    con_tr, con_te = stratified_two_way(
        con_y, train_frac=cfg.train_fraction, test_frac=cfg.test_fraction, rng=rng)

    splits = {
        "pretrain_train": (pre_X[pre_tr], pre_y[pre_tr], pre_spk[pre_tr]),
        "pretrain_test": (pre_X[pre_te], pre_y[pre_te], pre_spk[pre_te]),
        "continual_train": (con_X[con_tr], con_y[con_tr], con_spk[con_tr]),
        "continual_test": (con_X[con_te], con_y[con_te], con_spk[con_te]),
    }
    active_classes = [c for c in range(NUM_CLASSES) if c != cfg.removed_class]
    return splits, active_classes


def _check_splits(regime, splits, removed_class, nb_steps, n_compressed) -> None:
    """Leakage + shape (anti-transpose) + label-range + non-empty checks."""
    for name, (X, y, _spk) in splits.items():
        if len(y):
            s = X[0]
            if s.ndim != 2 or s.shape != (nb_steps, n_compressed):
                raise RuntimeError(
                    f"{name}: sample shape {s.shape}, expected "
                    f"({nb_steps}, {n_compressed}) -- possible T<->C transpose")
            if int(y.min()) < 0 or int(y.max()) >= NUM_CLASSES:
                raise RuntimeError(f"{name}: labels outside [0, {NUM_CLASSES})")
    if regime == "pretrain_19_class":
        for name, (_X, y, _spk) in splits.items():
            labs = set(int(v) for v in y.tolist())
            if name.startswith("pretrain") and removed_class in labs:
                raise RuntimeError("Leakage: pretrain split contains the removed class.")
            if name.startswith("continual") and labs - {removed_class}:
                raise RuntimeError("Leakage: continual split contains a non-removed class.")
        mandatory = ("pretrain_train", "continual_train")
    else:
        mandatory = ("train", "test")
    for m in mandatory:
        if len(splits[m][1]) == 0:
            raise RuntimeError(f"{m} is empty; adjust fractions or data.")


def _build_manifest(cfg, binner, merged, active_classes, counts, histograms) -> dict:
    return {
        "dataset": "SHD",
        "experiment_regime": cfg.experiment_regime,
        "storage_format": "dense_uint8_npz",
        "binning": "fixed_width_floor_seconds_binary",
        "removed_class": (int(cfg.removed_class)
                          if cfg.experiment_regime == "pretrain_19_class" else None),
        "active_classes": active_classes,
        "dataset_binning_ms": float(cfg.dataset_binning_ms),
        "dt_ms": float(cfg.dataset_binning_ms),        # alias used by model builders
        "dt_seconds": binner.dt_s,
        "dataset_max_seconds": float(cfg.dataset_max_seconds),
        "nb_steps": int(binner.nb_steps),
        "nb_inputs": int(cfg.n_compressed_channels),   # compressed count == model input dim
        "original_nb_inputs": int(cfg.nb_inputs),
        "n_compressed_channels": int(cfg.n_compressed_channels),
        "channel_compression_method": cfg.channel_compression_method,
        "compression_factor": int(binner.factor),
        "condition_or": int(cfg.condition_or),
        "binary_output": cfg.channel_compression_method != "graded",
        "dense_shape_per_sample": [int(binner.nb_steps), int(cfg.n_compressed_channels)],
        "merged_train_test": bool(merged),
        "synthetic": bool(cfg.synthetic_samples_per_class > 0),
        "seed": int(cfg.seed),
        "train_h5": (os.path.abspath(cfg.train_h5) if cfg.train_h5 else None),
        "test_h5": (os.path.abspath(cfg.test_h5) if (merged and cfg.test_h5) else None),
        "fractions": {"train": cfg.train_fraction, "test": cfg.test_fraction},
        "splits": {**{f"{k}_n": v for k, v in counts.items()},
                   **{f"{k}_class_hist": histograms[k] for k in histograms}},
        "benchmark_note": (
            "Merged official train+test then re-split; NOT the SHD benchmark test "
            "set. Accuracies are not comparable to the paper." if merged else
            "Preprocessed train_h5 only (or synthetic)."),
    }
