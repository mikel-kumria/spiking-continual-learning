#!/usr/bin/env python3
"""Preprocess the Spiking Heidelberg Digits (SHD) dataset into SNN-ready splits.

This is a single, self-contained replacement for the original
``heidelberg_statedict_generator.py`` *preprocessing* stage (it does NOT train a
network) and a corrected/cleaned-up version of
``build_shd_class_incremental_from_h5.py``.

Pipeline
--------
1.  Read the official SHD ``shd_train.h5`` and ``shd_test.h5`` event files.
2.  **Merge** them into a single in-memory pool (the "merged dataset"). All
    splits below are carved out of this pool.
3.  Partition the pool by class into:
      * **pretrain** : every class EXCEPT ``--removed-class``  (the base task)
      * **continual**: ONLY ``--removed-class``               (the new task)
4.  Bin each event sample into a dense ``uint8`` ``[nb_steps, nb_inputs]`` spike
    matrix using **fixed-width** time bins of ``--timestep-ms`` milliseconds.
    ``nb_steps = ceil(max_time / dt)`` is therefore *derived from* the timestep.
5.  Split pretrain and continual each into train / val / test with the six
    user-supplied fractions, **label-stratified** (per-class proportions held).
6.  Persist every split as a list of ``[1, nb_steps, nb_inputs]`` uint8 tensors
    (one sample per list entry) so a trainer can choose any batch size later,
    plus aligned label and speaker pickles, plus a JSON manifest.

Why fixed-width ``floor`` binning (and not ``linspace`` + ``digitize``)
----------------------------------------------------------------------
The original generator built ``time_bins = np.linspace(0, max_time, nb_steps)``
and assigned spikes with ``np.digitize``. That has three problems:
  * ``digitize`` returns indices in ``[1, nb_steps]`` (1-indexed): bin 0 is never
    used and a spike at ``t >= max_time`` yields index ``nb_steps``, which is
    OUT OF RANGE for a length-``nb_steps`` axis (latent crash / silent corruption
    on ``to_dense``);
  * the *effective* bin width is ``max_time/(nb_steps-1)`` (~14 ms for the paper
    defaults), yet the LIF time constants in the trainer were derived from
    ``time_step=1e-3`` (1 ms) -- the data resolution and the neuron dynamics were
    inconsistent;
  * the bin edges depend on ``nb_steps`` rather than on a physical timestep.
Here a spike at time ``t`` (seconds) maps to ``floor(t / dt)`` with
``dt = timestep_ms/1000``; spikes at ``t >= max_time`` are discarded. Bins are
0-indexed, uniform, and the timestep is the single source of truth (the trainer
should derive ``alpha``/``beta`` from this same ``dt``).

Benchmark caveat
----------------
Merging the official train and test sets and re-splitting randomly (which both
the original pipeline and this script do) produces splits that are NOT the SHD
benchmark. Reported accuracies are not directly comparable to the Dequino et al.
paper. Pass ``--no-merge`` to preprocess ``--train-h5`` alone if you need to keep
the official test set untouched for separate evaluation.
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
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

NUM_CLASSES = 20  # SHD: spoken digits 0-9 in English + German


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


def _read_h5_pool(path: Path) -> EventPool:
    """Load one SHD HDF5 file into an in-memory :class:`EventPool`."""
    with h5py.File(path, "r") as f:
        times = [np.asarray(t, dtype=np.float64) for t in f["spikes"]["times"]]
        units = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
        labels = np.asarray(f["labels"], dtype=np.int64)
        if "extra" in f and "speaker" in f["extra"]:
            speakers = np.asarray(f["extra"]["speaker"], dtype=np.int64)
        else:
            # Speaker metadata is optional; fall back to a sentinel so downstream
            # speaker-stratified experiments fail loudly rather than silently.
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


def compute_nb_steps(max_time_s: float, timestep_ms: float) -> int:
    """Number of temporal bins implied by a fixed timestep over ``[0, max_time)``."""
    if timestep_ms <= 0:
        raise ValueError("timestep_ms must be positive")
    if max_time_s <= 0:
        raise ValueError("max_time_s must be positive")
    dt_s = timestep_ms / 1000.0
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

    Spikes at ``t >= max_time_s`` are discarded; remaining spikes go to bin
    ``floor(t / dt_s)``. The bin index is clamped to ``nb_steps - 1`` to absorb
    floating-point edge cases (e.g. ``t`` a hair below ``max_time`` rounding up).
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
# Class filtering & dense materialisation
# ---------------------------------------------------------------------------


def select_indices_by_class(labels: np.ndarray, removed_class: int, keep_removed: bool) -> np.ndarray:
    """Return sample indices for either the pretrain or continual partition.

    ``keep_removed=False`` -> every class except ``removed_class`` (pretrain).
    ``keep_removed=True``  -> only ``removed_class``                (continual).
    """
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
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    """Bin the selected samples into dense uint8 tensors.

    Returns ``(x_list, y, speaker)`` where ``x_list[i]`` is a
    ``[nb_steps, nb_inputs]`` tensor aligned with ``y[i]`` and ``speaker[i]``.
    """
    x_list: List[torch.Tensor] = []
    for idx in indices:
        x_list.append(
            event_to_dense(
                pool.times[idx], pool.units[idx],
                nb_steps=nb_steps, nb_inputs=nb_inputs,
                max_time_s=max_time_s, dt_s=dt_s,
            )
        )
    y = pool.labels[indices].astype(np.int64)
    spk = pool.speakers[indices].astype(np.int64)
    return x_list, y, spk


# ---------------------------------------------------------------------------
# Stratified three-way split
# ---------------------------------------------------------------------------


def stratified_three_way(
    labels: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-stratified split returning (train_idx, val_idx, test_idx).

    Proportions are enforced *within each class* so val/test are not starved of
    rare classes. Each present class is guaranteed >=1 training sample.
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


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    for name, v in (("train", train_frac), ("val", val_frac), ("test", test_frac)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}_fraction={v} must be in [0, 1]")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0 (got {total:.6f})")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _as_batch_list(x_list: List[torch.Tensor], idx: np.ndarray) -> List[torch.Tensor]:
    """Gather selected samples as a list of ``[1, T, C]`` tensors (batch dim added)."""
    return [x_list[int(i)].unsqueeze(0) for i in idx]


def save_split(
    out_dir: Path,
    split_name: str,
    removed_class: int,
    x_list: List[torch.Tensor],
    y: np.ndarray,
    spk: np.ndarray,
    idx: np.ndarray,
) -> Dict[str, int]:
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
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    train_h5: Path
    test_h5: Optional[Path]
    output_dir: Path
    removed_class: int
    timestep_ms: float
    max_time_s: float
    nb_inputs: int
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

    rng = np.random.default_rng(cfg.seed)
    dt_s = cfg.timestep_ms / 1000.0
    nb_steps = compute_nb_steps(cfg.max_time_s, cfg.timestep_ms)
    print(
        f"Temporal binning: timestep={cfg.timestep_ms:g} ms, max_time={cfg.max_time_s:g} s "
        f"-> nb_steps={nb_steps} bins (dt={dt_s:g} s)"
    )

    # 1-2. Load and merge into the pool every split is carved from.
    pools = [_read_h5_pool(cfg.train_h5)]
    if cfg.merge:
        if cfg.test_h5 is None:
            raise ValueError("--test-h5 is required unless --no-merge is set")
        pools.append(_read_h5_pool(cfg.test_h5))
    pool = merge_pools(pools) if len(pools) > 1 else pools[0]
    print(f"Merged pool: {len(pool)} samples from {len(pools)} file(s)")

    # 3. Partition by class.
    pre_idx = select_indices_by_class(pool.labels, cfg.removed_class, keep_removed=False)
    con_idx = select_indices_by_class(pool.labels, cfg.removed_class, keep_removed=True)
    print(f"pretrain pool: {len(pre_idx)} samples (19 classes)")
    print(f"continual pool: {len(con_idx)} samples (class {cfg.removed_class})")
    if len(con_idx) == 0:
        raise ValueError(f"No samples for removed_class={cfg.removed_class}; nothing to learn continually.")

    # 4. Materialise dense uint8 tensors.
    pre_x, pre_y, pre_spk = materialise_dense(
        pool, pre_idx, nb_steps=nb_steps, nb_inputs=cfg.nb_inputs,
        max_time_s=cfg.max_time_s, dt_s=dt_s,
    )
    con_x, con_y, con_spk = materialise_dense(
        pool, con_idx, nb_steps=nb_steps, nb_inputs=cfg.nb_inputs,
        max_time_s=cfg.max_time_s, dt_s=dt_s,
    )

    # 5. Stratified three-way split (indices are local to each materialised list).
    pre_tr, pre_va, pre_te = stratified_three_way(
        pre_y, train_frac=cfg.pretrain_train_fraction,
        val_frac=cfg.pretrain_val_fraction, test_frac=cfg.pretrain_test_fraction, rng=rng,
    )
    con_tr, con_va, con_te = stratified_three_way(
        con_y, train_frac=cfg.continual_train_fraction,
        val_frac=cfg.continual_val_fraction, test_frac=cfg.continual_test_fraction, rng=rng,
    )

    # 6. Save.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "pretrain_train": (pre_x, pre_y, pre_spk, pre_tr),
        "pretrain_val": (pre_x, pre_y, pre_spk, pre_va),
        "pretrain_test": (pre_x, pre_y, pre_spk, pre_te),
        "continual_train": (con_x, con_y, con_spk, con_tr),
        "continual_val": (con_x, con_y, con_spk, con_va),
        "continual_test": (con_x, con_y, con_spk, con_te),
    }
    histograms: Dict[str, Dict[int, int]] = {}
    counts: Dict[str, int] = {}
    for name, (xl, y, spk, idx) in splits.items():
        histograms[name] = save_split(cfg.output_dir, name, cfg.removed_class, xl, y, spk, idx)
        counts[name] = int(len(idx))

    _sanity_check(splits, cfg.removed_class)

    manifest = {
        "dataset": "SHD",
        "storage_format": "dense_uint8",
        "binning": "fixed_width_floor_seconds",
        "removed_class": cfg.removed_class,
        "active_classes": [c for c in range(NUM_CLASSES) if c != cfg.removed_class],
        "timestep_ms": cfg.timestep_ms,
        "dt_seconds": dt_s,
        "max_time_seconds": cfg.max_time_s,
        "nb_steps": nb_steps,
        "nb_inputs": cfg.nb_inputs,
        "dt_ms": cfg.timestep_ms,  # alias for trainers that read 'dt_ms'
        "merged_train_test": cfg.merge,
        "seed": cfg.seed,
        "train_h5": str(cfg.train_h5.resolve()),
        "test_h5": str(cfg.test_h5.resolve()) if (cfg.merge and cfg.test_h5) else None,
        "dense_shape_per_sample": [1, nb_steps, cfg.nb_inputs],
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

    print("\nPreprocessing complete.")
    for name in splits:
        print(f"  {name:<16}: {counts[name]:>6} samples")
    print(f"  saved to        : {cfg.output_dir.resolve()}")


def _sanity_check(splits: Dict[str, tuple], removed_class: int) -> None:
    """Fail loudly if class leakage or empty mandatory splits are detected."""
    pre_labels: List[int] = []
    con_labels: List[int] = []
    for name, (_xl, y, _spk, idx) in splits.items():
        lab = [int(y[int(i)]) for i in idx]
        (con_labels if name.startswith("continual") else pre_labels).extend(lab)
    if removed_class in pre_labels:
        raise RuntimeError("Leakage: pretrain split contains the removed class.")
    if any(l != removed_class for l in con_labels):
        raise RuntimeError("Leakage: continual split contains a non-removed class.")
    for mandatory in ("pretrain_train", "continual_train"):
        if splits[mandatory][3].size == 0:
            raise RuntimeError(f"{mandatory} is empty; adjust fractions or data.")


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
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for pickles + manifest.")
    p.add_argument("--removed-class", type=int, required=True,
                   help="Class held out of pretrain and used as the continual task (0-19).")
    p.add_argument("--timestep-ms", type=float, required=True,
                   help="Temporal bin width in ms; sets nb_steps = ceil(max_time/dt).")
    p.add_argument("--max-time", type=float, default=1.4, help="Window length in seconds (default 1.4).")
    p.add_argument("--nb-inputs", type=int, default=700, help="Input channels (default 700).")
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
        timestep_ms=a.timestep_ms,
        max_time_s=a.max_time,
        nb_inputs=a.nb_inputs,
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


if __name__ == "__main__":
    raise SystemExit(main())
