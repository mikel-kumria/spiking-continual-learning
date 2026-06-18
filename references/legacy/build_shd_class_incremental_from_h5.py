#!/usr/bin/env python3
"""
Build class-incremental SHD pickles from a raw HDF5 file (same artifacts as
`class_incremental_naive_shd_preprocessing_refactored.ipynb`).

Each split (pretrain 19-class and continual held-out class) is written as
train / val / test **lists of dense uint8 batches**. Use ``--batch-size 1`` so
each list entry is one sample ``[1, nb_steps, nb_inputs]``; trainers can then
set any training batch size without re-preprocessing. The training pipeline
loads train + val only; ``*_test_*`` pickles are held out and must not be used
during training or model selection.

Storage: dense ``torch.uint8`` tensors (not sparse COO). Binning matches
``train_shd_pretraining_baseline.event_sample_to_dense``.

Time axis: uniform bins of width ``dt_ms`` milliseconds over ``[0, max_time)``
seconds, with ``nb_steps = ceil(max_time / (dt_ms/1000))``. Spike times in the
file are assumed to be in **seconds** (standard SHD); each spike at
``t >= max_time`` is discarded; remaining spikes are mapped to bin
``floor(t / dt_s)`` in ``[0, nb_steps - 1]``.

WARNING: When ``--test-h5`` is supplied, the official SHD train and test sets are
merged and then randomly re-split. The resulting splits are NOT equivalent to the
SHD benchmark. Reported accuracies will differ from (and typically exceed) paper
results. If benchmark comparability is required, use ``--h5 shd_train.h5`` only,
and evaluate separately on the official ``shd_test.h5``.

Example (train HDF5 only; one sample per saved batch entry)::

    python build_shd_class_incremental_from_h5.py \
      --h5 /data/SHD_raw/shd_train.h5 \
      --output-dir /data/SHD_raw/removed_class_10_dt14ms_bs1 \
      --removed-class 10 \
      --dt-ms 14 \
      --batch-size 1 \
      --seed 42 \
      --save-on-cpu
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


def _collect_dataset_paths(group: h5py.Group, prefix: str = "/") -> List[str]:
    paths: List[str] = []
    for name, item in group.items():
        path = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            paths.append(path)
        else:
            paths.extend(_collect_dataset_paths(item, prefix=f"{path}/"))
    return paths


def merge_train_test(train_path: Path, test_path: Path, merged_path: Path, rebuild: bool = False) -> Path:
    """Concatenate the official SHD train and test HDF5 files into one file.

    Per-sample datasets (spikes/times, spikes/units, labels, extra/speaker) are
    concatenated along axis 0; metadata datasets (e.g. extra/keys) are copied
    once from the train file and NOT duplicated.

    The previous implementation used ``Dataset.resize`` element-by-element, which
    raises ``TypeError`` on the fixed-extent (contiguous) datasets that standard
    SHD files use, and was O(N) in HDF5 API calls. This version creates chunked,
    growable datasets and writes data in bulk.
    """
    if merged_path.exists() and not rebuild:
        print(f"Using existing merged file: {merged_path}")
        return merged_path
    if merged_path.exists():
        merged_path.unlink()
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    # Datasets that are per-sample (concatenate along axis 0).
    SAMPLE_DATASETS = {"spikes/times", "spikes/units", "labels", "extra/speaker"}
    # Datasets that are metadata (copy once from train, do NOT duplicate).
    META_DATASETS = {"extra/keys"}  # add others if your SHD version has them

    with h5py.File(train_path, "r") as tr, \
         h5py.File(test_path, "r") as te, \
         h5py.File(merged_path, "w") as out:

        # Validate both files have the same structure. ``_collect_dataset_paths``
        # returns paths relative to the visited group with no leading slash.
        train_paths = set(_collect_dataset_paths(tr, prefix=""))
        test_paths = set(_collect_dataset_paths(te, prefix=""))
        if train_paths != test_paths:
            raise ValueError(
                f"Train and test HDF5 layouts differ.\n"
                f"  train-only: {train_paths - test_paths}\n"
                f"  test-only:  {test_paths - train_paths}"
            )

        # Create required groups (parents of every dataset path).
        for path in sorted(train_paths):
            parent = path.rsplit("/", 1)[0]
            if "/" in path and parent:
                out.require_group(parent)

        for path in sorted(train_paths):
            tr_ds = tr[path]
            parent = path.rsplit("/", 1)[0]
            parent_group = out[parent] if ("/" in path and parent) else out

            if path in META_DATASETS:
                # Copy metadata once; do not concatenate.
                tr.copy(path, parent_group, name=path.rsplit("/", 1)[-1])
                continue

            if path not in SAMPLE_DATASETS:
                # Unknown dataset: copy from train with a warning.
                import warnings
                warnings.warn(
                    f"merge_train_test: unknown dataset {path!r}; copying from train only.",
                    stacklevel=2,
                )
                tr.copy(path, parent_group, name=path.rsplit("/", 1)[-1])
                continue

            # Load both arrays into memory and concatenate.
            # SHD vlen arrays are small enough to fit in RAM.
            tr_data = tr_ds[()]
            te_data = te[path][()]
            n_total = len(tr_data) + len(te_data)

            vlen_base = h5py.check_vlen_dtype(tr_ds.dtype)
            if vlen_base is not None:
                # Variable-length dtype: create vlen dataset and assign per-row.
                vlen_dt = h5py.vlen_dtype(vlen_base)
                merged_ds = out.create_dataset(
                    path, shape=(n_total,), maxshape=(None,), dtype=vlen_dt,
                    chunks=(min(n_total, 512),),
                )
                for i, arr in enumerate(list(tr_data) + list(te_data)):
                    merged_ds[i] = arr
            else:
                # Fixed-dtype: stack and write in one shot.
                out.create_dataset(
                    path,
                    data=np.concatenate([tr_data, te_data], axis=0),
                    maxshape=(None,) + tr_ds.shape[1:],
                    chunks=True,
                )

        # Sanity checks.
        n_tr = tr["labels"].shape[0]
        n_te = te["labels"].shape[0]
        assert out["labels"].shape[0] == n_tr + n_te, "label count mismatch"
        assert out["spikes/times"].shape[0] == n_tr + n_te, "spike count mismatch"
        if "extra/keys" in out and "extra/keys" in tr:
            assert out["extra/keys"].shape[0] == tr["extra/keys"].shape[0], \
                "extra/keys doubled -- copy logic error"

        print(
            f"Created merged file: {merged_path} "
            f"({n_tr} train + {n_te} test = {n_tr + n_te} total samples)"
        )
    return merged_path


def filter_by_class(
    firing_times: Sequence[np.ndarray],
    units_fired: Sequence[np.ndarray],
    labels: np.ndarray,
    speakers: Sequence[int],
    class_id: int,
    keep_class: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    selected_times: List[np.ndarray] = []
    selected_units: List[np.ndarray] = []
    selected_labels: List[int] = []
    selected_speakers: List[int] = []

    for t, u, l, s in zip(firing_times, units_fired, labels, speakers):
        condition = l == class_id
        if condition == keep_class:
            selected_times.append(t)
            selected_units.append(u)
            selected_labels.append(int(l))
            selected_speakers.append(int(s))

    return selected_times, selected_units, selected_labels, selected_speakers


def compute_nb_steps(max_time_seconds: float, dt_ms: float) -> int:
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be positive")
    dt_s = dt_ms / 1000.0
    return max(1, int(math.ceil(max_time_seconds / dt_s)))


def event_sample_to_dense(
    times: np.ndarray,
    units: np.ndarray,
    nb_steps: int,
    num_inputs: int,
    max_time_seconds: float,
    dt_ms: float,
) -> torch.Tensor:
    """Bin one event sample into a dense ``[nb_steps, num_inputs]`` uint8 matrix.

    Matches ``train_shd_pretraining_baseline.event_sample_to_dense``.
    """
    if len(times) != len(units):
        raise ValueError("times and units length mismatch")
    x = torch.zeros((nb_steps, num_inputs), dtype=torch.uint8)
    if len(times) == 0:
        return x
    times_arr = np.asarray(times, dtype=np.float64)
    unit_arr = np.asarray(units, dtype=np.int64)
    if np.any(unit_arr < 0) or np.any(unit_arr >= num_inputs):
        raise ValueError(f"unit index outside [0, {num_inputs})")
    if np.any(times_arr < 0):
        raise ValueError("negative spike time encountered")
    keep_mask = times_arr < float(max_time_seconds)
    if not np.all(keep_mask):
        times_arr = times_arr[keep_mask]
        unit_arr = unit_arr[keep_mask]
    if times_arr.size == 0:
        return x
    dt_s = dt_ms / 1000.0
    time_bins = np.floor(times_arr / dt_s).astype(np.int64)
    if np.any(time_bins < 0) or np.any(time_bins >= nb_steps):
        raise ValueError(f"time bin index outside [0, {nb_steps})")
    x[torch.from_numpy(time_bins), torch.from_numpy(unit_arr)] = 1
    return x


def build_dense_batches(
    spikes_group: h5py.Group,
    labels_ds: h5py.Dataset,
    speakers_ds: h5py.Dataset,
    *,
    class_id: int,
    keep_class: bool,
    batch_size: int,
    nb_steps: int,
    nb_units: int,
    max_time_seconds: float,
    dt_ms: float,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    firing_times = spikes_group["times"]
    units_fired = spikes_group["units"]
    labels = np.array(labels_ds, dtype=int)

    times, units, labels_filtered, speakers_filtered = filter_by_class(
        firing_times=firing_times,
        units_fired=units_fired,
        labels=labels,
        speakers=speakers_ds,
        class_id=class_id,
        keep_class=keep_class,
    )

    labels_arr = np.array(labels_filtered, dtype=int)
    times_arr = np.array(times, dtype=object)
    units_arr = np.array(units, dtype=object)
    speakers_arr = np.array(speakers_filtered, dtype=int)

    n_samples = len(labels_arr)
    n_batches = n_samples // batch_size
    n_dropped = n_samples - n_batches * batch_size
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"build_dense_batches: dropping {n_dropped}/{n_samples} samples "
            f"(batch_size={batch_size} does not divide N evenly). "
            f"Use batch_size=1 to retain all samples.",
            stacklevel=2,
        )
    sample_index = np.arange(n_samples)

    if shuffle:
        if seed is None:
            np.random.shuffle(sample_index)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(sample_index)

    dt_s = dt_ms / 1000.0
    expected_bins = compute_nb_steps(max_time_seconds, dt_ms)
    if expected_bins != nb_steps:
        raise RuntimeError(f"internal nb_steps mismatch: {nb_steps} vs {expected_bins}")

    x_batches: List[torch.Tensor] = []
    y_batches: List[torch.Tensor] = []
    speaker_batches: List[torch.Tensor] = []

    for b in range(n_batches):
        batch_idx = sample_index[batch_size * b : batch_size * (b + 1)]
        rows = [
            event_sample_to_dense(
                times_arr[int(sample_idx)],
                units_arr[int(sample_idx)],
                nb_steps,
                nb_units,
                max_time_seconds,
                dt_ms,
            )
            for sample_idx in batch_idx
        ]
        x_batch = torch.stack(rows, dim=0)
        y_batch = torch.tensor(labels_arr[batch_idx], dtype=torch.long)
        speaker_batch = torch.tensor(speakers_arr[batch_idx], dtype=torch.long)
        assert x_batch.shape == (len(batch_idx), nb_steps, nb_units)
        assert x_batch.dtype == torch.uint8
        x_batches.append(x_batch)
        y_batches.append(y_batch)
        speaker_batches.append(speaker_batch)

    return x_batches, y_batches, speaker_batches


def _allocate_three_way_split(
    n_batches: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) batch counts that sum to ``n_batches``."""
    if n_batches == 0:
        return 0, 0, 0
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    n_val = int(round(n_batches * val_fraction))
    n_test = int(round(n_batches * test_fraction))
    if val_fraction > 0 and n_val == 0:
        n_val = 1
    if test_fraction > 0 and n_test == 0:
        n_test = 1
    n_train = n_batches - n_val - n_test
    if n_train < 1:
        deficit = 1 - n_train
        if n_val >= n_test and n_val > 0:
            take = min(deficit, n_val - (1 if val_fraction > 0 else 0))
            n_val -= take
            deficit -= take
        if deficit > 0 and n_test > 0:
            n_test = max(0, n_test - deficit)
        n_train = n_batches - n_val - n_test
    if n_train < 0:
        raise RuntimeError("Could not allocate a non-empty train split")
    return n_train, n_val, n_test


def split_batches_three_way(
    x_batches: List[torch.Tensor],
    y_batches: List[torch.Tensor],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Split shuffled batches into train | val | test (contiguous slices)."""
    n_batches = len(x_batches)
    if len(y_batches) != n_batches:
        raise ValueError("x_batches and y_batches length mismatch")

    n_train, n_val, n_test = _allocate_three_way_split(n_batches, val_fraction, test_fraction)
    i_val = n_train
    i_test = n_train + n_val

    x_train = x_batches[:n_train]
    y_train = y_batches[:n_train]
    x_val = x_batches[i_val:i_test]
    y_val = y_batches[i_val:i_test]
    x_test = x_batches[i_test:]
    y_test = y_batches[i_test:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def stratified_split_indices(
    y_batches: List[torch.Tensor],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Label-stratified three-way split returning (train, val, test) index lists.

    Proportions are held within each class so per-class metrics are not starved
    of samples in val/test. Assumes one label per batch entry (``batch_size=1``);
    the class key is taken from the first sample in each batch.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, yb in enumerate(y_batches):
        label = int(torch.as_tensor(yb).reshape(-1)[0].item())
        by_class[label].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for label in sorted(by_class.keys()):
        idxs = np.array(by_class[label])
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
        n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
        n_train = n - n_val - n_test
        if n_train < 1:
            # Guarantee at least one training sample per class.
            n_train = 1
            n_val = max(0, n - 1 - n_test)
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    return train_idx, val_idx, test_idx


def _gather_by_indices(items: List[torch.Tensor], idxs: List[int]) -> List[torch.Tensor]:
    return [items[i] for i in idxs]


def maybe_to_cpu_tensor_list(items: List[torch.Tensor], save_on_cpu: bool) -> List[torch.Tensor]:
    if not save_on_cpu:
        return items
    return [t.cpu() for t in items]


def save_outputs(
    output_dir: Path,
    removed_class: int,
    x_pretrain_train: List[torch.Tensor],
    y_pretrain_train: List[torch.Tensor],
    x_pretrain_val: List[torch.Tensor],
    y_pretrain_val: List[torch.Tensor],
    x_pretrain_test: List[torch.Tensor],
    y_pretrain_test: List[torch.Tensor],
    x_continual_train: List[torch.Tensor],
    y_continual_train: List[torch.Tensor],
    x_continual_val: List[torch.Tensor],
    y_continual_val: List[torch.Tensor],
    x_continual_test: List[torch.Tensor],
    y_continual_test: List[torch.Tensor],
    save_on_cpu: bool = False,
    speaker_splits: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    c = removed_class

    artifacts = {
        f"pretrain_train_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_pretrain_train, save_on_cpu),
        f"pretrain_train_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_pretrain_train, save_on_cpu),
        f"pretrain_val_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_pretrain_val, save_on_cpu),
        f"pretrain_val_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_pretrain_val, save_on_cpu),
        f"pretrain_test_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_pretrain_test, save_on_cpu),
        f"pretrain_test_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_pretrain_test, save_on_cpu),
        f"continual_train_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_continual_train, save_on_cpu),
        f"continual_train_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_continual_train, save_on_cpu),
        f"continual_val_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_continual_val, save_on_cpu),
        f"continual_val_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_continual_val, save_on_cpu),
        f"continual_test_x_class_{c}.pkl": maybe_to_cpu_tensor_list(x_continual_test, save_on_cpu),
        f"continual_test_y_class_{c}.pkl": maybe_to_cpu_tensor_list(y_continual_test, save_on_cpu),
    }

    # Persist speaker arrays alongside labels so speaker-stratified splits and the
    # paper's Sample-Incremental scenario can be reproduced later.
    if speaker_splits:
        for split_name, spk_batches in speaker_splits.items():
            artifacts[f"{split_name}_speaker_class_{c}.pkl"] = maybe_to_cpu_tensor_list(
                spk_batches, save_on_cpu
            )

    paths: Dict[str, Path] = {}
    for name, obj in artifacts.items():
        out_path = output_dir / name
        with out_path.open("wb") as f:
            pickle.dump(obj, f)
        paths[name] = out_path

    return paths


def validate_outputs(paths: Dict[str, Path], removed_class: int) -> None:
    required = [
        f"pretrain_train_x_class_{removed_class}.pkl",
        f"pretrain_train_y_class_{removed_class}.pkl",
        f"pretrain_val_x_class_{removed_class}.pkl",
        f"pretrain_val_y_class_{removed_class}.pkl",
        f"pretrain_test_x_class_{removed_class}.pkl",
        f"pretrain_test_y_class_{removed_class}.pkl",
        f"continual_train_x_class_{removed_class}.pkl",
        f"continual_train_y_class_{removed_class}.pkl",
        f"continual_val_x_class_{removed_class}.pkl",
        f"continual_val_y_class_{removed_class}.pkl",
        f"continual_test_x_class_{removed_class}.pkl",
        f"continual_test_y_class_{removed_class}.pkl",
    ]

    for key in required:
        if key not in paths or not paths[key].exists():
            raise RuntimeError(f"Missing expected output file: {key}")

    with (paths[f"pretrain_train_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_pretrain_train = pickle.load(f)
    with (paths[f"pretrain_val_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_pretrain_val = pickle.load(f)
    with (paths[f"pretrain_test_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_pretrain_test = pickle.load(f)
    with (paths[f"continual_train_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_continual_train = pickle.load(f)
    with (paths[f"continual_val_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_continual_val = pickle.load(f)
    with (paths[f"continual_test_y_class_{removed_class}.pkl"]).open("rb") as f:
        y_continual_test = pickle.load(f)

    pretrain_labels = torch.cat(
        [b.reshape(-1) for b in (y_pretrain_train + y_pretrain_val + y_pretrain_test)],
        dim=0,
    )
    continual_labels = torch.cat(
        [b.reshape(-1) for b in (y_continual_train + y_continual_val + y_continual_test)],
        dim=0,
    )

    if torch.any(pretrain_labels == removed_class):
        raise RuntimeError("Validation failed: pretrain labels contain removed_class")
    if torch.any(continual_labels != removed_class):
        raise RuntimeError("Validation failed: continual split labels contain classes other than removed_class")

    with (paths[f"pretrain_train_x_class_{removed_class}.pkl"]).open("rb") as f:
        _validate_dense_batch_lists(pickle.load(f), "pretrain_train_x")
    with (paths[f"pretrain_val_x_class_{removed_class}.pkl"]).open("rb") as f:
        _validate_dense_batch_lists(pickle.load(f), "pretrain_val_x")
    with (paths[f"continual_train_x_class_{removed_class}.pkl"]).open("rb") as f:
        _validate_dense_batch_lists(pickle.load(f), "continual_train_x")


def _validate_dense_batch_lists(x_batches: List[torch.Tensor], label: str) -> None:
    if not x_batches:
        return
    x0 = x_batches[0]
    if x0.is_sparse:
        raise RuntimeError(f"{label}: expected dense uint8 tensors, got sparse")
    if x0.dtype != torch.uint8:
        raise RuntimeError(f"{label}: expected uint8 dense tensors, got dtype={x0.dtype}")
    if x0.ndim != 3:
        raise RuntimeError(f"{label}: expected shape [B,T,C], got {tuple(x0.shape)}")
    if int(x0.max().item()) > 1:
        raise RuntimeError(
            f"{label}: spike tensor is not binary; max value = {int(x0.max().item())}"
        )


@dataclass(frozen=True)
class RunConfig:
    h5_path: Path
    test_h5_path: Optional[Path]
    merged_out: Optional[Path]
    rebuild_merged: bool
    output_dir: Path
    removed_class: int
    dt_ms: float
    max_time_seconds: float
    nb_inputs: int
    batch_size: int
    pretrain_val_fraction: float
    pretrain_test_fraction: float
    continual_val_fraction: float
    continual_test_fraction: float
    save_on_cpu: bool
    run_validation: bool
    seed: Optional[int]


def run_preprocessing(cfg: RunConfig) -> Dict[str, Path]:
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    print("Dense preprocessing on CPU (uint8 tensors).")

    if cfg.test_h5_path is not None:
        merged_path = cfg.merged_out or (cfg.output_dir / "_scratch_shd_merged.h5")
        h5_used = merge_train_test(
            train_path=cfg.h5_path,
            test_path=cfg.test_h5_path,
            merged_path=merged_path,
            rebuild=cfg.rebuild_merged,
        )
    else:
        h5_used = cfg.h5_path
        if not h5_used.is_file():
            raise FileNotFoundError(h5_used)

    nb_steps = compute_nb_steps(cfg.max_time_seconds, cfg.dt_ms)
    dt_s = cfg.dt_ms / 1000.0
    print(f"Temporal binning: dt_ms={cfg.dt_ms:g}, max_time_s={cfg.max_time_seconds:g} -> nb_steps={nb_steps} (dt_s={dt_s:g}s)")

    with h5py.File(h5_used, "r") as dataset:
        spikes = dataset["spikes"]
        labels = dataset["labels"]
        if "speaker" not in dataset.get("extra", {}):
            raise KeyError("Expected dataset group extra/speaker (standard SHD layout).")
        speakers = dataset["extra"]["speaker"]

        x_pretrain, y_pretrain, spk_pretrain = build_dense_batches(
            spikes_group=spikes,
            labels_ds=labels,
            speakers_ds=speakers,
            class_id=cfg.removed_class,
            keep_class=False,
            batch_size=cfg.batch_size,
            nb_steps=nb_steps,
            nb_units=cfg.nb_inputs,
            max_time_seconds=cfg.max_time_seconds,
            dt_ms=cfg.dt_ms,
            shuffle=True,
            seed=cfg.seed,
        )

        x_continual, y_continual, spk_continual = build_dense_batches(
            spikes_group=spikes,
            labels_ds=labels,
            speakers_ds=speakers,
            class_id=cfg.removed_class,
            keep_class=True,
            batch_size=cfg.batch_size,
            nb_steps=nb_steps,
            nb_units=cfg.nb_inputs,
            max_time_seconds=cfg.max_time_seconds,
            dt_ms=cfg.dt_ms,
            shuffle=True,
            seed=cfg.seed,
        )

    # Label-stratified three-way split. Computing indices once keeps the x/y/
    # speaker lists aligned across all three splits.
    pre_train_i, pre_val_i, pre_test_i = stratified_split_indices(
        y_pretrain,
        val_fraction=cfg.pretrain_val_fraction,
        test_fraction=cfg.pretrain_test_fraction,
        seed=cfg.seed,
    )
    x_pretrain_train = _gather_by_indices(x_pretrain, pre_train_i)
    y_pretrain_train = _gather_by_indices(y_pretrain, pre_train_i)
    x_pretrain_val = _gather_by_indices(x_pretrain, pre_val_i)
    y_pretrain_val = _gather_by_indices(y_pretrain, pre_val_i)
    x_pretrain_test = _gather_by_indices(x_pretrain, pre_test_i)
    y_pretrain_test = _gather_by_indices(y_pretrain, pre_test_i)
    spk_pretrain_train = _gather_by_indices(spk_pretrain, pre_train_i)
    spk_pretrain_val = _gather_by_indices(spk_pretrain, pre_val_i)
    spk_pretrain_test = _gather_by_indices(spk_pretrain, pre_test_i)

    con_train_i, con_val_i, con_test_i = stratified_split_indices(
        y_continual,
        val_fraction=cfg.continual_val_fraction,
        test_fraction=cfg.continual_test_fraction,
        seed=cfg.seed,
    )
    x_continual_train = _gather_by_indices(x_continual, con_train_i)
    y_continual_train = _gather_by_indices(y_continual, con_train_i)
    x_continual_val = _gather_by_indices(x_continual, con_val_i)
    y_continual_val = _gather_by_indices(y_continual, con_val_i)
    x_continual_test = _gather_by_indices(x_continual, con_test_i)
    y_continual_test = _gather_by_indices(y_continual, con_test_i)
    spk_continual_train = _gather_by_indices(spk_continual, con_train_i)
    spk_continual_val = _gather_by_indices(spk_continual, con_val_i)
    spk_continual_test = _gather_by_indices(spk_continual, con_test_i)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    speaker_splits = {
        "pretrain_train": spk_pretrain_train,
        "pretrain_val": spk_pretrain_val,
        "pretrain_test": spk_pretrain_test,
        "continual_train": spk_continual_train,
        "continual_val": spk_continual_val,
        "continual_test": spk_continual_test,
    }

    paths = save_outputs(
        output_dir=cfg.output_dir,
        removed_class=cfg.removed_class,
        x_pretrain_train=x_pretrain_train,
        y_pretrain_train=y_pretrain_train,
        x_pretrain_val=x_pretrain_val,
        y_pretrain_val=y_pretrain_val,
        x_pretrain_test=x_pretrain_test,
        y_pretrain_test=y_pretrain_test,
        x_continual_train=x_continual_train,
        y_continual_train=y_continual_train,
        x_continual_val=x_continual_val,
        y_continual_val=y_continual_val,
        x_continual_test=x_continual_test,
        y_continual_test=y_continual_test,
        save_on_cpu=cfg.save_on_cpu,
        speaker_splits=speaker_splits,
    )

    def _split_n(x_batches: List[torch.Tensor]) -> int:
        return int(sum(int(b.shape[0]) for b in x_batches))

    def _class_hist(y_batches: List[torch.Tensor]) -> Dict[int, int]:
        if not y_batches:
            return {}
        flat = torch.cat([torch.as_tensor(b).reshape(-1) for b in y_batches])
        counts: Dict[int, int] = {}
        for c in flat.tolist():
            counts[int(c)] = counts.get(int(c), 0) + 1
        return counts

    manifest = {
        "h5_used": str(h5_used.resolve()),
        "train_h5_arg": str(cfg.h5_path.resolve()),
        "test_h5_arg": str(cfg.test_h5_path.resolve()) if cfg.test_h5_path else None,
        "removed_class": cfg.removed_class,
        "dt_ms": cfg.dt_ms,
        "max_time_seconds": cfg.max_time_seconds,
        "nb_steps": nb_steps,
        "nb_inputs": cfg.nb_inputs,
        "batch_size": cfg.batch_size,
        "pretrain_train_fraction": 1.0 - cfg.pretrain_val_fraction - cfg.pretrain_test_fraction,
        "pretrain_val_fraction": cfg.pretrain_val_fraction,
        "pretrain_test_fraction": cfg.pretrain_test_fraction,
        "continual_train_fraction": 1.0 - cfg.continual_val_fraction - cfg.continual_test_fraction,
        "continual_val_fraction": cfg.continual_val_fraction,
        "continual_test_fraction": cfg.continual_test_fraction,
        "split_order": "label-stratified train|val|test (per-class proportions)",
        "split_strategy": "stratified",
        "binning": "uniform_dt_seconds_floor",
        "storage_format": "dense_uint8",
        "preprocess_batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "dense_shape_per_batch": [cfg.batch_size, nb_steps, cfg.nb_inputs],
        "splits": {
            "pretrain_train_n": _split_n(x_pretrain_train),
            "pretrain_val_n": _split_n(x_pretrain_val),
            "pretrain_test_n": _split_n(x_pretrain_test),
            "continual_train_n": _split_n(x_continual_train),
            "continual_val_n": _split_n(x_continual_val),
            "continual_test_n": _split_n(x_continual_test),
            "pretrain_train_class_hist": _class_hist(y_pretrain_train),
            "pretrain_val_class_hist": _class_hist(y_pretrain_val),
            "pretrain_test_class_hist": _class_hist(y_pretrain_test),
            "continual_train_class_hist": _class_hist(y_continual_train),
        },
    }
    if cfg.test_h5_path is not None:
        manifest["test_set_note"] = (
            "Custom random re-split of merged train+test. "
            "NOT the official SHD benchmark test set. "
            "Accuracy numbers are not directly comparable to the Dequino et al. paper."
        )
    manifest_path = cfg.output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if cfg.run_validation:
        validate_outputs(paths, cfg.removed_class)

    print("\nPreprocessing complete.")
    print(f"removed_class           : {cfg.removed_class}")
    print(f"pretrain_train batches  : {len(x_pretrain_train)}")
    print(f"pretrain_val batches    : {len(x_pretrain_val)}")
    print(f"pretrain_test batches   : {len(x_pretrain_test)}")
    print(f"continual_train batches : {len(x_continual_train)}")
    print(f"continual_val batches   : {len(x_continual_val)}")
    print(f"continual_test batches  : {len(x_continual_test)}")
    print(f"saved to                : {cfg.output_dir.resolve()}")
    print("\nOutput files:")
    for name, pth in paths.items():
        print(f"- {name}: {pth}")

    return paths


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True, help="Main SHD HDF5 (typically shd_train.h5 or an already-merged file).")
    p.add_argument(
        "--test-h5",
        type=Path,
        default=None,
        help="Optional official test HDF5; if set, train and test are merged like the refactored notebook.",
    )
    p.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help="When --test-h5 is set, path for the merged HDF5 (default: OUTPUT_DIR/_scratch_shd_merged.h5).",
    )
    p.add_argument("--rebuild-merged", action="store_true", help="Rebuild merged file when --test-h5 is used.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for pickles and manifest.")
    p.add_argument("--removed-class", type=int, required=True, help="Digit class held out for continual split (e.g. 10).")
    p.add_argument("--dt-ms", type=float, required=True, help="Uniform temporal bin width in milliseconds.")
    p.add_argument(
        "--max-time",
        type=float,
        default=1.4,
        help="Clip/window length in seconds (default 1.4, same spirit as the notebook).",
    )
    p.add_argument("--nb-inputs", type=int, default=700, help="Number of input channels (default 700).")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per saved batch entry (default 1: one [1,T,C] tensor per list item; "
        "trainers can use any training batch size without re-preprocessing).",
    )
    p.add_argument(
        "--pretrain-val-fraction",
        type=float,
        default=0.15,
        help="Fraction of pretrain batches for validation (default 0.15).",
    )
    p.add_argument(
        "--pretrain-test-fraction",
        type=float,
        default=0.15,
        help="Fraction of pretrain batches for test (default 0.15; train gets the remainder).",
    )
    p.add_argument(
        "--continual-val-fraction",
        type=float,
        default=0.15,
        help="Fraction of continual batches for validation (default 0.15).",
    )
    p.add_argument(
        "--continual-test-fraction",
        type=float,
        default=0.15,
        help="Fraction of continual batches for test (default 0.15; train gets the remainder).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for shuffling sample order into batches.")
    p.add_argument("--save-on-cpu", action="store_true", help="Move tensors to CPU before pickling.")
    p.add_argument("--no-validation", action="store_true", help="Skip label integrity checks.")
    args = p.parse_args(argv)

    return RunConfig(
        h5_path=args.h5,
        test_h5_path=args.test_h5,
        merged_out=args.merged_out,
        rebuild_merged=args.rebuild_merged,
        output_dir=args.output_dir,
        removed_class=args.removed_class,
        dt_ms=args.dt_ms,
        max_time_seconds=args.max_time,
        nb_inputs=args.nb_inputs,
        batch_size=args.batch_size,
        pretrain_val_fraction=args.pretrain_val_fraction,
        pretrain_test_fraction=args.pretrain_test_fraction,
        continual_val_fraction=args.continual_val_fraction,
        continual_test_fraction=args.continual_test_fraction,
        save_on_cpu=args.save_on_cpu,
        run_validation=not args.no_validation,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        run_preprocessing(cfg)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
