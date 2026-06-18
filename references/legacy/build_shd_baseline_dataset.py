#!/usr/bin/env python3
"""Preprocess raw SHD into dense, time-binned train / val / test splits (20 classes).


1. Load the official ``shd_train.h5`` and ``shd_test.h5`` event files (ragged spikes).
2. Bin each sample into a dense ``uint8`` tensor ``[nb_steps, nb_inputs]`` over the
   first ``max_time`` seconds (default 1.4 s, 100 bins -> 14 ms each).
3. Carve a stratified validation set from the official train split (per-class
   proportions preserved; default 15 %).
4. Persist train / val / test as pickle lists of ``[1, T, C]`` tensors plus a JSON
   manifest for downstream SNN training scripts.

The official SHD test set is kept untouched for final evaluation.

Output layout
-------------
::

    output_dir/
      train_x.pkl          # list of [1, T, C] uint8 tensors
      train_y.pkl          # list of [1] long label tensors
      val_x.pkl / val_y.pkl
      test_x.pkl / test_y.pkl
      split_indices.npz    # train_idx, val_idx, seed, validation_fraction
      preprocessing_manifest.json

Load in a training script::

    from build_shd_baseline_dataset import load_split, read_manifest

    x_train, y_train = load_split("data/shd_baseline_20class_dt14ms", "train")
    x_val,   y_val   = load_split("data/shd_baseline_20class_dt14ms", "val")


Run it:
    python3 build_shd_baseline_dataset.py \
        --train-h5 data/SHD_raw/shd_train.h5 \
        --test-h5 data/SHD_raw/shd_test.h5 \
        --output-dir data/preprocessed/20class/shd_baseline_20class_dt14ms
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Defaults (match SHD_baseline.ipynb configuration cell)
# ---------------------------------------------------------------------------

NUM_CLASSES = 20
DEFAULT_MAX_TIME = 1.4          # seconds
DEFAULT_NB_STEPS = 100          # number_of_dataset_binnings
DEFAULT_NB_INPUTS = 700         # input_neurons
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_shd(path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Load one SHD .h5 file: ragged spike events + one label per sample."""
    with h5py.File(path, "r") as f:
        spike_times = [np.asarray(t, dtype=np.float64) for t in f["spikes"]["times"]]
        spike_channel = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
        labels = f["labels"][:].astype(np.int64)
    return spike_times, spike_channel, labels


# ---------------------------------------------------------------------------
# Temporal binning (events -> dense spike tensor)
# ---------------------------------------------------------------------------


def shd_temporal_binning(
    spike_times: Sequence[np.ndarray],
    spike_channels: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    max_time_dataset: float,
    number_of_dataset_binnings: int,
    input_neurons: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn ragged SHD events into dense uint8 tensors ``[N, T, C]`` and labels ``[N]``."""
    dt = max_time_dataset / number_of_dataset_binnings
    n_samples = len(labels)

    x = torch.zeros((n_samples, number_of_dataset_binnings, input_neurons), dtype=torch.uint8)
    # NOTE: uint8 keeps the full dataset compact on CPU; trainers cast to float per batch

    for n in range(n_samples):
        t = spike_times[n]
        c = spike_channels[n]

        keep = t < max_time_dataset
        t, c = t[keep], c[keep]
        if t.size == 0:
            continue

        time_bin = np.clip((t / dt).astype(np.int64), 0, number_of_dataset_binnings - 1)
        x[n, torch.from_numpy(time_bin), torch.from_numpy(c)] = 1

    y = torch.from_numpy(labels).long()
    return x, y


# ---------------------------------------------------------------------------
# Stratified train / validation split
# ---------------------------------------------------------------------------


def stratified_validation_split(
    y: torch.Tensor,
    *,
    n_classes: int,
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class stratified split of the official train labels into train / val indices."""
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    train_idx: List[int] = []
    validation_idx: List[int] = []

    for class_index in range(n_classes):
        class_idx = np.where(y_np == class_index)[0]
        class_idx = np.array(class_idx, copy=True)
        rng.shuffle(class_idx)

        n_validation = int(round(len(class_idx) * validation_fraction))
        validation_idx.extend(class_idx[:n_validation].tolist())
        train_idx.extend(class_idx[n_validation:].tolist())

    return np.array(sorted(train_idx)), np.array(sorted(validation_idx))


# ---------------------------------------------------------------------------
# Loading (for downstream training scripts)
# ---------------------------------------------------------------------------


def load_split(root: Path | str, split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one split as stacked tensors ``[N, T, C]`` and labels ``[N]``."""
    root = Path(root)
    with (root / f"{split_name}_x.pkl").open("rb") as f:
        x_batches = pickle.load(f)
    with (root / f"{split_name}_y.pkl").open("rb") as f:
        y_batches = pickle.load(f)
    x = torch.cat(x_batches, dim=0)
    y = torch.cat([yb.reshape(-1) for yb in y_batches], dim=0).long()
    return x, y


def read_manifest(root: Path | str) -> dict:
    """Read ``preprocessing_manifest.json`` from a preprocessed root."""
    return json.loads((Path(root) / "preprocessing_manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _class_histogram(labels: torch.Tensor, indices: np.ndarray, n_classes: int) -> Dict[str, int]:
    y_np = labels[indices].numpy() if hasattr(labels, "numpy") else np.asarray(labels[indices])
    counts = np.bincount(y_np, minlength=n_classes)
    return {str(c): int(counts[c]) for c in range(n_classes)}


def save_split(
    output_dir: Path,
    split_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
) -> Dict[str, int]:
    """Persist one split as ``{split}_x.pkl`` / ``{split}_y.pkl`` pickle lists."""
    x_batches = [x[int(i)].unsqueeze(0).cpu() for i in indices]
    y_batches = [y[int(i)].reshape(1).cpu().long() for i in indices]

    with (output_dir / f"{split_name}_x.pkl").open("wb") as f:
        pickle.dump(x_batches, f)
    with (output_dir / f"{split_name}_y.pkl").open("wb") as f:
        pickle.dump(y_batches, f)

    hist = _class_histogram(y, indices, NUM_CLASSES)
    return {int(k): v for k, v in hist.items()}


def write_manifest(
    *,
    output_dir: Path,
    cfg: "Config",
    dt: float,
    counts: Dict[str, int],
    histograms: Dict[str, Dict[int, int]],
) -> None:
    timestep_ms = dt * 1000.0
    manifest = {
        "dataset": "SHD",
        "task": "20_class_baseline",
        "storage_format": "dense_uint8",
        "binning": "fixed_width_floor_seconds",
        "num_classes": NUM_CLASSES,
        "active_classes": list(range(NUM_CLASSES)),
        "removed_class": None,
        "timestep_ms": timestep_ms,
        "dt_ms": timestep_ms,
        "dt_seconds": dt,
        "max_time_seconds": cfg.max_time,
        "nb_steps": cfg.nb_steps,
        "nb_inputs": cfg.nb_inputs,
        "dense_shape_per_sample": [1, cfg.nb_steps, cfg.nb_inputs],
        "validation_fraction": cfg.validation_fraction,
        "seed": cfg.seed,
        "train_h5": str(cfg.train_h5.resolve()),
        "test_h5": str(cfg.test_h5.resolve()),
        "merged_train_test": False,
        "benchmark_note": (
            "Official SHD train/test splits preserved; "
            "validation carved from train only (stratified per class)."
        ),
        "splits": {
            **{f"{k}_n": v for k, v in counts.items()},
            **{f"{k}_class_hist": {str(c): n for c, n in h.items()} for k, h in histograms.items()},
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    train_h5: Path
    test_h5: Path
    output_dir: Path
    max_time: float
    nb_steps: int
    nb_inputs: int
    validation_fraction: float
    seed: int


def run(cfg: Config) -> None:
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {cfg.validation_fraction}")

    rng = np.random.default_rng(cfg.seed)
    dt = cfg.max_time / cfg.nb_steps
    print(
        f"Temporal binning: max_time={cfg.max_time:g} s, nb_steps={cfg.nb_steps} "
        f"-> dt={dt * 1000:g} ms"
    )

    # 1. Load official train and test event files.
    train_times, train_channel, train_labels = load_shd(cfg.train_h5)
    test_times, test_channel, test_labels = load_shd(cfg.test_h5)
    print(f"Raw train samples: {len(train_labels)}  |  raw test samples: {len(test_labels)}")

    # 2. Bin ragged events into dense uint8 tensors.
    x_train, y_train = shd_temporal_binning(
        train_times, train_channel, train_labels,
        max_time_dataset=cfg.max_time,
        number_of_dataset_binnings=cfg.nb_steps,
        input_neurons=cfg.nb_inputs,
    )
    x_test, y_test = shd_temporal_binning(
        test_times, test_channel, test_labels,
        max_time_dataset=cfg.max_time,
        number_of_dataset_binnings=cfg.nb_steps,
        input_neurons=cfg.nb_inputs,
    )
    print(
        f"Time binned train shape: {tuple(x_train.shape)}  |  "
        f"time binned test shape: {tuple(x_test.shape)}"
    )

    # 3. Stratified train / val split inside the official train set.
    train_idx, val_idx = stratified_validation_split(
        y_train,
        n_classes=NUM_CLASSES,
        validation_fraction=cfg.validation_fraction,
        rng=rng,
    )
    test_idx = np.arange(len(y_test))

    print(
        f"Train samples (after stratified split): {len(train_idx)}  |  "
        f"validation samples: {len(val_idx)}  |  "
        f"test samples: {len(test_idx)}"
    )

    # 4. Save pickles + manifest.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": (x_train, y_train, train_idx),
        "val": (x_train, y_train, val_idx),
        "test": (x_test, y_test, test_idx),
    }
    counts: Dict[str, int] = {}
    histograms: Dict[str, Dict[int, int]] = {}
    for name, (x, y, idx) in splits.items():
        histograms[name] = save_split(cfg.output_dir, name, x, y, idx)
        counts[name] = int(len(idx))

    np.savez(
        cfg.output_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
    )

    write_manifest(
        output_dir=cfg.output_dir,
        cfg=cfg,
        dt=dt,
        counts=counts,
        histograms=histograms,
    )

    print("\nPreprocessing complete.")
    for name in ("train", "val", "test"):
        print(f"  {name:<5}: {counts[name]:>6} samples")
    print(f"  saved to: {cfg.output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--train-h5", type=Path, required=True,
        help="Path to shd_train.h5.",
    )
    p.add_argument(
        "--test-h5", type=Path, required=True,
        help="Path to shd_test.h5 (kept as the held-out test split).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for pickles, split indices, and manifest.",
    )
    p.add_argument(
        "--max-time", type=float, default=DEFAULT_MAX_TIME,
        help=f"Window length in seconds (default {DEFAULT_MAX_TIME}).",
    )
    p.add_argument(
        "--nb-steps", type=int, default=DEFAULT_NB_STEPS,
        help=f"Number of time bins (default {DEFAULT_NB_STEPS}).",
    )
    p.add_argument(
        "--nb-inputs", type=int, default=DEFAULT_NB_INPUTS,
        help=f"Input channels (default {DEFAULT_NB_INPUTS}).",
    )
    p.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help=f"Fraction of each train class held out for validation (default {DEFAULT_VAL_FRACTION}).",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for the stratified split (default {DEFAULT_SEED}).",
    )
    args = p.parse_args(argv)
    return Config(
        train_h5=args.train_h5,
        test_h5=args.test_h5,
        output_dir=args.output_dir,
        max_time=args.max_time,
        nb_steps=args.nb_steps,
        nb_inputs=args.nb_inputs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
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
