#!/usr/bin/env python3
"""Stage 1 of the SHD continual-learning pipeline: preprocess + pretrain.

This script does two things and writes a single deterministic, self-contained
run folder that Stage 2 (``class_incremental_snn_shd.py``) consumes:

1. PREPROCESS the raw Spiking Heidelberg Digits (SHD) HDF5 files into a
   class-incremental setup:
     * fixed-width temporal binning (``--dataset-binning-ms``);
     * a held-out ``--removed-class`` that is EXCLUDED from every pretrain split
       and saved on its own in the continual splits;
     * optional merge of the official train/test sets before re-splitting;
     * channel-axis-only compression (700 -> ``--n-compressed-channels``);
     * label-stratified train/val/test splits for both partitions;
     * leakage / binary / shape (anti-transpose) sanity checks before writing;
     * canonical ``.npz`` splits (``X uint8 [N,T,C]``, ``y int64 [N]``,
       ``speaker int64 [N]``) + a ``preprocessing_manifest.json``.

2. PRETRAIN the recurrent SNN from ``train_snn_shd.py`` (single recurrent hidden
   layer, ``Phi = mean_t(hidden_spikes)``, ``logits = Phi @ W_out``, no bias) on
   the 19 active old classes only:
     * ``fullbptt`` : train W_in + W_rec + W_out; cross-entropy computed over the
       19 ACTIVE classes (logits masked, labels remapped to [0,18] only inside
       the loss -- the saved labels stay original [0,19]).
     * ``ridge``    : freeze W_in/W_rec; closed-form solve of W_out on the active
       old-class targets from the mean-spike feature.
   The removed-class output column is initialised DETERMINISTICALLY to zero (so
   new-class accuracy before incremental learning is ~0 by construction -- this
   is expected and logged, not a bug). The policy is recorded in the checkpoint
   and manifest.

Output folder (``--output-root/--run-name``)::

    outputs/shd_pretraining/<run_name>/
      config.json
      preprocessing_manifest.json
      metrics.json
      dataset/{pretrain,continual}_{train,val,test}.npz
      checkpoints/{pretrained_model.pt,best_pretrained_model.pt}
      logs/

Example::

    python pretrain_snn_shd.py \
      --train-h5 data/SHD_raw/shd_train.h5 \
      --test-h5  data/SHD_raw/shd_test.h5 \
      --output-root outputs/shd_pretraining \
      --run-name rc10_dt14_or70_removed10_ridge \
      --removed-class 10 --dataset-binning-ms 14 \
      --n-compressed-channels 70 --channel-compression-method or_pool \
      --mode ridge --nb-hidden 1000 --batch-size 64 --wandb-mode disabled
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import snn_shd_common as C
import train_snn_shd as T  # reuse spectral_radius_sanity (exact same logic)

NUM_CLASSES = C.NUM_CLASSES
DEFAULT_NB_INPUTS = 700  # native SHD cochlea channel count


# =============================================================================
# Event pool (raw SHD samples held in memory) + loading / merging / synthetic
# =============================================================================


@dataclass
class EventPool:
    times: List[np.ndarray]   # firing times in seconds, one array per sample
    units: List[np.ndarray]   # channel index per spike, aligned with ``times``
    labels: np.ndarray        # int label per sample      [N]
    speakers: np.ndarray      # int speaker id per sample [N]

    def __len__(self) -> int:
        return len(self.labels)


def read_h5_pool(path: str) -> EventPool:
    """Load one SHD HDF5 file into memory."""
    import h5py
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        times = [np.asarray(t, dtype=np.float64) for t in f["spikes"]["times"]]
        units = [np.asarray(u, dtype=np.int64) for u in f["spikes"]["units"]]
        labels = np.asarray(f["labels"], dtype=np.int64)
        if "extra" in f and "speaker" in f["extra"]:
            speakers = np.asarray(f["extra"]["speaker"], dtype=np.int64)
        else:
            speakers = np.full(len(labels), -1, dtype=np.int64)
    if not (len(times) == len(units) == len(labels) == len(speakers)):
        raise ValueError(f"{path}: ragged arrays have mismatched lengths")
    return EventPool(times, units, labels, speakers)


def merge_pools(pools) -> EventPool:
    times: List[np.ndarray] = []
    units: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    speakers: List[np.ndarray] = []
    for p in pools:
        times.extend(p.times)
        units.extend(p.units)
        labels.append(p.labels)
        speakers.append(p.speakers)
    return EventPool(times, units,
                     np.concatenate(labels, 0), np.concatenate(speakers, 0))


def synthetic_pool(samples_per_class: int, nb_inputs: int, max_time_s: float,
                   seed: int, num_classes: int = NUM_CLASSES,
                   spikes_per_channel: int = 8) -> EventPool:
    """Fabricate a tiny event pool (all ``num_classes`` classes) for smoke tests.

    Each class fires in a distinct, wide channel band, densely enough in time
    that the binned feed-forward drive actually crosses the LIF threshold (so the
    reservoir spikes and a classifier beats chance). No SHD download required.
    """
    rng = np.random.default_rng(seed)
    times: List[np.ndarray] = []
    units: List[np.ndarray] = []
    labels: List[int] = []
    speakers: List[int] = []
    band = max(2, nb_inputs // 2)  # wide, overlapping bands -> dense per-step drive
    for c in range(num_classes):
        lo = int(round(c * (nb_inputs - band) / max(1, num_classes - 1)))
        for _ in range(samples_per_class):
            active = np.arange(lo, lo + band) % nb_inputs
            # several spikes per active channel, spread over time -> dense bins.
            u = np.repeat(active, spikes_per_channel)
            t = rng.uniform(0.0, max_time_s * 0.95, size=u.shape[0])
            order = np.argsort(t)
            times.append(t[order].astype(np.float64))
            units.append(u[order].astype(np.int64))
            labels.append(c)
            speakers.append(int(rng.integers(0, 5)))
    return EventPool(times, units,
                     np.asarray(labels, np.int64), np.asarray(speakers, np.int64))


# =============================================================================
# Temporal binning + per-partition materialisation (+ channel compression)
# =============================================================================


def compute_nb_steps(max_time_s: float, dt_ms: float) -> int:
    if dt_ms <= 0:
        raise ValueError("dataset-binning-ms must be positive")
    if max_time_s <= 0:
        raise ValueError("max-time must be positive")
    return max(1, int(math.ceil(max_time_s / (dt_ms / 1000.0))))


def event_to_dense_np(times: np.ndarray, units: np.ndarray, *, nb_steps: int,
                      nb_inputs: int, max_time_s: float, dt_s: float) -> np.ndarray:
    """Bin one event sample into a dense ``[nb_steps, nb_inputs]`` uint8 matrix.

    Channel is the LAST axis. Spikes at ``t >= max_time_s`` are dropped; the rest
    map to ``floor(t/dt_s)`` (clamped). Values assigned (not incremented) -> {0,1}.
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
    x[bins, u] = 1
    return x


def select_indices_by_class(labels: np.ndarray, removed_class: int,
                            keep_removed: bool) -> np.ndarray:
    """Indices for the pretrain (``keep_removed=False``) or continual partition."""
    is_removed = labels == removed_class
    mask = is_removed if keep_removed else ~is_removed
    return np.nonzero(mask)[0]


def materialise(pool: EventPool, indices: np.ndarray, *, nb_steps: int,
                nb_inputs: int, max_time_s: float, dt_s: float, method: str,
                factor: int, condition_or: int, bernoulli_seed: int
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin then channel-compress selected samples -> ``(X [N,T,Ccomp], y, spk)``.

    Only one full-resolution ``[T, 700]`` sample exists at a time (compressed
    then discarded), so peak memory stays bounded even at small dt.
    """
    rng = np.random.default_rng(bernoulli_seed) if method == "bernoulli" else None
    ccomp = nb_inputs // factor
    xs: List[np.ndarray] = []
    for idx in indices:
        dense = event_to_dense_np(
            pool.times[idx], pool.units[idx], nb_steps=nb_steps,
            nb_inputs=nb_inputs, max_time_s=max_time_s, dt_s=dt_s)  # [T, 700]
        comp = C.compress_channels(dense, method, factor,
                                   condition_or=condition_or, rng=rng)  # [T, Ccomp]
        C.assert_compression_invariants(dense, comp, method, factor)
        xs.append(comp)
    if xs:
        X = np.stack(xs, 0).astype(np.uint8, copy=False)
    else:
        X = np.zeros((0, nb_steps, ccomp), dtype=np.uint8)
    y = pool.labels[indices].astype(np.int64)
    spk = pool.speakers[indices].astype(np.int64)
    return X, y, spk


# =============================================================================
# Label-stratified three-way split (sample-level, deterministic)
# =============================================================================


def validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    for name, v in (("train", train_frac), ("val", val_frac), ("test", test_frac)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}_fraction={v} must be in [0, 1]")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0 (got {total:.6f})")


def stratified_three_way(labels: np.ndarray, *, train_frac: float, val_frac: float,
                         test_frac: float, rng: np.random.Generator
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class proportional split; every present class keeps >= 1 train sample."""
    validate_fractions(train_frac, val_frac, test_frac)
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
        if n_train < 1:
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
    return (np.array(train_idx, np.int64), np.array(val_idx, np.int64),
            np.array(test_idx, np.int64))


# =============================================================================
# Preprocessing orchestration
# =============================================================================


def run_preprocessing(args, dataset_dir: str) -> dict:
    """Build + save the six ``.npz`` splits and the manifest. Returns the manifest."""
    if not 0 <= args.removed_class < NUM_CLASSES:
        raise ValueError(f"removed-class must be in [0, {NUM_CLASSES})")
    validate_fractions(args.pretrain_train_fraction, args.pretrain_val_fraction,
                       args.pretrain_test_fraction)
    validate_fractions(args.continual_train_fraction, args.continual_val_fraction,
                       args.continual_test_fraction)
    factor = C.validate_compression_factor(args.nb_inputs, args.n_compressed_channels)
    if args.channel_compression_method not in C.COMPRESSION_METHODS:
        raise ValueError(f"unknown channel-compression-method "
                         f"{args.channel_compression_method!r}")

    rng = np.random.default_rng(args.seed)
    dt_s = args.dataset_binning_ms / 1000.0
    nb_steps = compute_nb_steps(args.max_time, args.dataset_binning_ms)
    print(f"Temporal binning: dt={args.dataset_binning_ms:g} ms, "
          f"max_time={args.max_time:g} s -> nb_steps={nb_steps} (dt={dt_s:g} s)")
    print(f"Channel compression: {args.channel_compression_method} "
          f"{args.nb_inputs} -> {args.n_compressed_channels} (factor {factor})")

    # ---- load the event pool ----
    if args.synthetic_samples_per_class > 0:
        print(f"SYNTHETIC mode: {args.synthetic_samples_per_class} samples/class "
              f"({NUM_CLASSES} classes)")
        pool = synthetic_pool(args.synthetic_samples_per_class, args.nb_inputs,
                              args.max_time, args.seed)
        merged = False
    else:
        pools = [read_h5_pool(args.train_h5)]
        merged = bool(args.merge_train_test)
        if merged:
            if not args.test_h5:
                raise ValueError("--test-h5 is required unless --no-merge-train-test")
            pools.append(read_h5_pool(args.test_h5))
        pool = merge_pools(pools) if len(pools) > 1 else pools[0]
    print(f"Pool: {len(pool)} samples")

    # ---- partition by class ----
    pre_idx = select_indices_by_class(pool.labels, args.removed_class, keep_removed=False)
    con_idx = select_indices_by_class(pool.labels, args.removed_class, keep_removed=True)
    print(f"pretrain pool: {len(pre_idx)} ({NUM_CLASSES - 1} classes) | "
          f"continual pool: {len(con_idx)} (class {args.removed_class})")
    if len(con_idx) == 0:
        raise ValueError(f"no samples for removed-class={args.removed_class}")
    if len(pre_idx) == 0:
        raise ValueError("pretrain pool is empty")

    mat_kw = dict(nb_steps=nb_steps, nb_inputs=args.nb_inputs, max_time_s=args.max_time,
                  dt_s=dt_s, method=args.channel_compression_method, factor=factor,
                  condition_or=args.condition_or, bernoulli_seed=args.seed)
    pre_X, pre_y, pre_spk = materialise(pool, pre_idx, **mat_kw)
    con_X, con_y, con_spk = materialise(pool, con_idx, **mat_kw)

    # ---- stratified splits (indices local to each materialised partition) ----
    pre_tr, pre_va, pre_te = stratified_three_way(
        pre_y, train_frac=args.pretrain_train_fraction,
        val_frac=args.pretrain_val_fraction, test_frac=args.pretrain_test_fraction, rng=rng)
    con_tr, con_va, con_te = stratified_three_way(
        con_y, train_frac=args.continual_train_fraction,
        val_frac=args.continual_val_fraction, test_frac=args.continual_test_fraction, rng=rng)

    splits = {
        "pretrain_train": (pre_X, pre_y, pre_spk, pre_tr),
        "pretrain_val": (pre_X, pre_y, pre_spk, pre_va),
        "pretrain_test": (pre_X, pre_y, pre_spk, pre_te),
        "continual_train": (con_X, con_y, con_spk, con_tr),
        "continual_val": (con_X, con_y, con_spk, con_va),
        "continual_test": (con_X, con_y, con_spk, con_te),
    }

    # ---- sanity checks BEFORE writing anything ----
    _check_splits(splits, args.removed_class, nb_steps, args.n_compressed_channels)

    # ---- write ----
    os.makedirs(dataset_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    histograms: Dict[str, Dict[int, int]] = {}
    for name, (X, y, spk, idx) in splits.items():
        Xs, ys, spks = X[idx], y[idx], spk[idx]
        C.save_npz_split(os.path.join(dataset_dir, f"{name}.npz"), Xs, ys, spks)
        counts[name] = int(len(idx))
        histograms[name] = {int(k): int(v) for k, v in
                            zip(*np.unique(ys, return_counts=True))}

    active_classes = [c for c in range(NUM_CLASSES) if c != args.removed_class]
    manifest = {
        "dataset": "SHD",
        "storage_format": "dense_uint8_npz",
        "binning": "fixed_width_floor_seconds",
        "removed_class": int(args.removed_class),
        "active_classes": active_classes,
        "dt_ms": float(args.dataset_binning_ms),
        "dt_seconds": dt_s,
        "max_time_seconds": float(args.max_time),
        "nb_steps": int(nb_steps),
        "nb_inputs": int(args.n_compressed_channels),       # compressed count (model input)
        "original_nb_inputs": int(args.nb_inputs),
        "n_compressed_channels": int(args.n_compressed_channels),
        "channel_compression_method": args.channel_compression_method,
        "compression_method": args.channel_compression_method,  # alias for train_snn_shd
        "compression_factor": int(factor),
        "condition_or": int(args.condition_or),
        "binary_output": args.channel_compression_method != "graded",
        "dense_shape_per_sample": [nb_steps, int(args.n_compressed_channels)],
        "merged_train_test": bool(merged),
        "synthetic": bool(args.synthetic_samples_per_class > 0),
        "seed": int(args.seed),
        "train_h5": os.path.abspath(args.train_h5) if args.train_h5 else None,
        "test_h5": (os.path.abspath(args.test_h5)
                    if (merged and args.test_h5) else None),
        "fractions": {
            "pretrain_train": args.pretrain_train_fraction,
            "pretrain_val": args.pretrain_val_fraction,
            "pretrain_test": args.pretrain_test_fraction,
            "continual_train": args.continual_train_fraction,
            "continual_val": args.continual_val_fraction,
            "continual_test": args.continual_test_fraction,
        },
        "splits": {**{f"{k}_n": v for k, v in counts.items()},
                   **{f"{k}_class_hist": histograms[k] for k in histograms}},
        "benchmark_note": (
            "Merged official train+test then re-split; NOT the SHD benchmark test "
            "set. Accuracies are not comparable to the paper." if merged else
            "Preprocessed --train-h5 only."),
    }
    C.write_json(os.path.join(os.path.dirname(dataset_dir),
                              "preprocessing_manifest.json"), manifest)
    print("Preprocessing complete:")
    for name in C.SPLIT_NAMES:
        print(f"  {name:<16}: {counts[name]:>6} samples")
    return manifest


def _check_splits(splits, removed_class, nb_steps, n_compressed) -> None:
    """Leakage + binary + shape (anti-transpose) checks across all splits."""
    pre_labels: List[int] = []
    con_labels: List[int] = []
    for name, (X, y, _spk, idx) in splits.items():
        labs = y[idx].tolist()
        (con_labels if name.startswith("continual") else pre_labels).extend(labs)
        if len(idx):
            sample = X[idx[0]]
            if sample.ndim != 2 or sample.shape != (nb_steps, n_compressed):
                raise RuntimeError(
                    f"{name}: sample shape {sample.shape}, expected "
                    f"({nb_steps}, {n_compressed}) -- possible T<->C transpose")
    if removed_class in pre_labels:
        raise RuntimeError("Leakage: pretrain split contains the removed class.")
    if any(l != removed_class for l in con_labels):
        raise RuntimeError("Leakage: continual split contains a non-removed class.")
    # labels stay original 0..19
    if pre_labels and (min(pre_labels) < 0 or max(pre_labels) >= NUM_CLASSES):
        raise RuntimeError("pretrain labels outside [0, 20)")
    for mandatory in ("pretrain_train", "continual_train"):
        if splits[mandatory][3].size == 0:
            raise RuntimeError(f"{mandatory} is empty; adjust fractions or data.")


# =============================================================================
# Pretraining (19 active classes only; removed column held at zero)
# =============================================================================


@torch.no_grad()
def evaluate_active(model, X, y, active_idx, batch_size, device) -> float:
    """19-way accuracy: argmax over the ACTIVE columns, mapped back to orig labels."""
    model.eval()
    if X.shape[0] == 0:
        return float("nan")
    correct = total = 0
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        logits = model(xb)                                  # [B, nb_outputs]
        sub = logits.index_select(1, active_idx)            # [B, 19]
        pred = active_idx[sub.argmax(1)]                    # back to orig labels
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def _zero_removed_column(model, removed_class: int) -> None:
    with torch.no_grad():
        model.W_out[:, removed_class] = 0.0


def train_ridge_active(model, X_tr, y_tr, active_classes, removed_class,
                       ridge_lambda, batch_size, device) -> None:
    """Closed-form (float64) ridge solve of W_out on ACTIVE-class targets only."""
    model.requires_grad_(False)
    model.eval()
    Phi = C.collect_features(model, X_tr, batch_size, device).double()  # [N, H]
    H = model.nb_hidden
    pos = {int(c): i for i, c in enumerate(active_classes)}
    remapped = np.array([pos[int(c)] for c in y_tr.numpy()], dtype=np.int64)
    Yk = torch.zeros((Phi.shape[0], len(active_classes)), dtype=torch.float64)
    Yk[torch.arange(Phi.shape[0]), torch.from_numpy(remapped)] = 1.0
    A = Phi.T @ Phi + ridge_lambda * torch.eye(H, dtype=torch.float64)
    B = Phi.T @ Yk
    try:
        L = torch.linalg.cholesky(A)
        Wk = torch.cholesky_solve(B, L)
    except RuntimeError:
        Wk = torch.linalg.solve(A, B)
    assert Wk.shape == (H, len(active_classes))
    Wfull = torch.zeros((H, model.nb_outputs), dtype=torch.float64)
    active_idx = torch.tensor(active_classes, dtype=torch.long)
    Wfull[:, active_idx] = Wk
    with torch.no_grad():
        model.W_out.copy_(Wfull.to(model.W_out.device, model.W_out.dtype))
    _zero_removed_column(model, removed_class)  # explicit + deterministic


def train_bptt_active(model, X_tr, y_tr, X_va, y_va, active_classes, removed_class,
                      args, device, wandb_run) -> dict:
    """fullbptt over all weights; cross-entropy on the 19 ACTIVE classes only."""
    model.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr > 0 else 2e-4
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adamax(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    active_idx = torch.tensor(active_classes, dtype=torch.long, device=device)
    pos = torch.full((NUM_CLASSES,), -1, dtype=torch.long, device=device)
    pos[active_idx] = torch.arange(len(active_classes), device=device)  # orig -> [0,18]

    gen = torch.Generator().manual_seed(args.seed)
    N = X_tr.shape[0]
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(args.nb_epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, generator=gen)
        run_loss = run_correct = seen = 0.0
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            yk = pos[yb]                                    # remapped [0,18]
            logits = model(xb).index_select(1, active_idx)  # mask to active
            loss = loss_fn(logits, yk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            n = int(yb.numel())
            run_loss += float(loss.item()) * n
            run_correct += int((active_idx[logits.argmax(1)] == yb).sum().item())
            seen += n
        train_loss = run_loss / max(seen, 1)
        train_acc = run_correct / max(seen, 1)
        val_acc = evaluate_active(model, X_va, y_va, active_idx, args.batch_size, device)
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss,
                           "train_acc_active": train_acc, "val_acc_active": val_acc,
                           "best_val_acc_active": best["val_acc"], "lr": lr,
                           "epoch_seconds": time.time() - t0})
        print(f"[fullbptt] epoch={epoch:03d} loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"best={best['val_acc']:.4f}@{best['epoch']} ({time.time()-t0:.1f}s)")
    return best


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()
    C.set_determinism(args.seed)
    device = C.resolve_device(args.device)
    print(f"device: {device}  mode: {args.mode}")

    run_dir = os.path.join(args.output_root, args.run_name)
    dataset_dir = os.path.join(run_dir, "dataset")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    for d in (run_dir, dataset_dir, ckpt_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    # ---- Stage 1a: preprocessing ----
    manifest = run_preprocessing(args, dataset_dir)
    active_classes = manifest["active_classes"]
    assert len(active_classes) == NUM_CLASSES - 1

    # ---- reload the saved splits (validates the npz round-trip) ----
    X_tr, y_tr, _ = C.load_npz_split(os.path.join(dataset_dir, "pretrain_train.npz"), args.limit)
    X_va, y_va, _ = C.load_npz_split(os.path.join(dataset_dir, "pretrain_val.npz"), args.limit)
    X_te, y_te, _ = C.load_npz_split(os.path.join(dataset_dir, "pretrain_test.npz"), args.limit)
    Xc_te, yc_te, _ = C.load_npz_split(os.path.join(dataset_dir, "continual_test.npz"), args.limit)
    nb_inputs = X_tr.shape[2]
    assert nb_inputs == args.n_compressed_channels, (
        f"dataset channels {nb_inputs} != n_compressed_channels {args.n_compressed_channels}")
    assert args.nb_outputs >= NUM_CLASSES, f"nb_outputs must be >= {NUM_CLASSES}"
    assert int(y_tr.max().item()) < NUM_CLASSES and args.removed_class not in set(y_tr.tolist())
    print(f"pretrain_train X={tuple(X_tr.shape)}  nb_inputs={nb_inputs}")

    # ---- neuron decays from the dataset binning dt ----
    dt_ms = float(args.dataset_binning_ms)
    alpha, beta = C.derive_alpha_beta(dt_ms, args.tau_mem_ms, args.tau_syn_ms)
    print(f"dt_ms={dt_ms} alpha={alpha:.4f} beta={beta:.4f}")

    # ---- model + spectral-radius sanity (reuse train_snn_shd's exact logic) ----
    model = C.ReservoirSNN(nb_inputs, args.nb_hidden, args.nb_outputs, alpha, beta,
                           args.threshold, args.weight_scale, args.surrogate_slope).to(device)
    sr_ns = SimpleNamespace(
        init_spectral_radius=args.init_spectral_radius, firing_low=args.firing_low,
        firing_high=args.firing_high, sr_scale_up=1.2, sr_scale_down=0.8, sr_max_iters=20)
    sane_n = min(args.batch_size, X_tr.shape[0])
    sr_info = T.spectral_radius_sanity(model, X_tr[:sane_n].to(device), sr_ns)
    print(f"[sanity] spectral_radius -> {sr_info['final_spectral_radius']:.4f}  "
          f"firing={sr_info['init_hidden_firing_rate']:.4f}")

    wandb_run = init_wandb(args, manifest, nb_inputs, dt_ms, alpha, beta, sr_info)

    # ---- Stage 1b: pretrain on the 19 active classes ----
    active_idx_dev = torch.tensor(active_classes, dtype=torch.long, device=device)
    t0 = time.time()
    best_state = None
    if args.mode == "ridge":
        train_ridge_active(model, X_tr, y_tr, active_classes, args.removed_class,
                           args.ridge_lambda, args.batch_size, device)
    else:  # fullbptt
        best = train_bptt_active(model, X_tr, y_tr, X_va, y_va, active_classes,
                                 args.removed_class, args, device, wandb_run)
        if best["state"] is not None:
            model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        _zero_removed_column(model, args.removed_class)  # deterministic removed col
        best_state = best["state"]
    train_seconds = time.time() - t0

    # ---- metrics (19-way active accuracy is the meaningful pretraining metric) ----
    train_acc = evaluate_active(model, X_tr, y_tr, active_idx_dev, args.batch_size, device)
    val_acc = evaluate_active(model, X_va, y_va, active_idx_dev, args.batch_size, device)
    test_acc = evaluate_active(model, X_te, y_te, active_idx_dev, args.batch_size, device)
    test_acc_full20, _ = C.evaluate_split(model, X_te, y_te, args.batch_size, device)
    # new-class accuracy BEFORE incremental learning -- expected ~0 (zero column).
    new_before, _ = C.evaluate_split(model, Xc_te, yc_te, args.batch_size, device)
    pretraining_metrics = {
        "mode": args.mode,
        "pretrain_train_acc_active19": train_acc,
        "pretrain_val_acc_active19": val_acc,
        "pretrain_test_acc_active19": test_acc,
        "pretrain_test_acc_full20": test_acc_full20,
        "continual_test_acc_before_incremental": new_before,
        "train_seconds": train_seconds,
        **{f"init_{k}": v for k, v in sr_info.items()},
    }
    print(f"=== {args.mode} ===  train19={train_acc:.4f} val19={val_acc:.4f} "
          f"test19={test_acc:.4f} test20={test_acc_full20:.4f} "
          f"continual_test(before)={new_before:.4f}")

    # ---- save checkpoints + config + metrics ----
    config = {k: (v if not isinstance(v, np.generic) else v.item())
              for k, v in vars(args).items()}
    ckpt = C.build_checkpoint(
        model, dt_ms=dt_ms, tau_mem_ms=args.tau_mem_ms, tau_syn_ms=args.tau_syn_ms,
        threshold=args.threshold, weight_scale=args.weight_scale,
        surrogate_slope=args.surrogate_slope, active_classes=active_classes,
        removed_class=args.removed_class, pretraining_mode=args.mode,
        pretraining_metrics=pretraining_metrics, dataset_dir=os.path.abspath(dataset_dir),
        config=config, removed_class_init_policy="zero")
    torch.save(ckpt, os.path.join(ckpt_dir, "pretrained_model.pt"))
    # best checkpoint: for fullbptt the best-val state (removed col already zeroed
    # on the loaded model); for ridge the closed-form solution == final.
    best_ckpt = dict(ckpt)
    if best_state is not None:
        bs = {k: v.clone() for k, v in best_state.items()}
        bs["W_out"][:, args.removed_class] = 0.0
        best_ckpt["model_state_dict"] = bs
    torch.save(best_ckpt, os.path.join(ckpt_dir, "best_pretrained_model.pt"))

    C.write_json(os.path.join(run_dir, "config.json"), config)
    C.write_json(os.path.join(run_dir, "metrics.json"), pretraining_metrics)
    print(f"Saved run to {os.path.abspath(run_dir)}")

    if wandb_run is not None:
        for k, v in pretraining_metrics.items():
            wandb_run.summary[k] = v
        wandb_run.finish()
    return 0


def init_wandb(args, manifest, nb_inputs, dt_ms, alpha, beta, sr_info):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    config = {**{f"arg/{k}": v for k, v in vars(args).items()},
              "nb_inputs": nb_inputs, "dt_ms": dt_ms, "alpha": alpha, "beta": beta,
              "removed_class": manifest["removed_class"],
              "active_classes": manifest["active_classes"],
              "channel_compression_method": manifest["channel_compression_method"],
              "compression_factor": manifest["compression_factor"],
              "nb_steps": manifest["nb_steps"],
              **{f"init/{k}": v for k, v in sr_info.items()}}
    return wandb.init(project=args.wandb_project, name=args.wandb_name or args.run_name,
                      entity=args.wandb_entity, mode=args.wandb_mode, config=config,
                      tags=["shd", "pretrain", args.mode,
                            f"removed{args.removed_class}", f"C{nb_inputs}"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / preprocessing
    p.add_argument("--train-h5", type=str, default=None, help="Path to shd_train.h5")
    p.add_argument("--test-h5", type=str, default=None, help="Path to shd_test.h5")
    p.add_argument("--output-root", type=str, default="outputs/shd_pretraining")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--removed-class", type=int, required=True, help="Held-out class 0-19")
    p.add_argument("--dataset-binning-ms", type=float, required=True,
                   help="Temporal bin width in ms (also sets alpha/beta dt)")
    p.add_argument("--max-time", type=float, default=1.4, help="Window length (s)")
    p.add_argument("--nb-inputs", type=int, default=DEFAULT_NB_INPUTS,
                   help="Native input channels before compression")
    p.add_argument("--n-compressed-channels", type=int, default=70,
                   help="Channel count AFTER compression; must divide --nb-inputs")
    p.add_argument("--channel-compression-method", type=str, default="or_pool",
                   choices=sorted(C.COMPRESSION_METHODS))
    p.add_argument("--condition-or", type=int, default=1,
                   help="Threshold for or_pool/conditional_or (group fires if >= this)")
    p.add_argument("--merge-train-test", action=argparse.BooleanOptionalAction, default=True,
                   help="Merge official train+test before re-splitting")
    p.add_argument("--pretrain-train-fraction", type=float, default=0.70)
    p.add_argument("--pretrain-val-fraction", type=float, default=0.15)
    p.add_argument("--pretrain-test-fraction", type=float, default=0.15)
    p.add_argument("--continual-train-fraction", type=float, default=0.70)
    p.add_argument("--continual-val-fraction", type=float, default=0.15)
    p.add_argument("--continual-test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-samples-per-class", type=int, default=0,
                   help="If >0, skip HDF5 and fabricate a tiny pool (smoke tests)")
    # model / training
    p.add_argument("--mode", type=str, default="ridge", choices=["fullbptt", "ridge"])
    p.add_argument("--nb-hidden", type=int, default=1000)
    p.add_argument("--nb-outputs", type=int, default=NUM_CLASSES)
    p.add_argument("--tau-mem-ms", type=float, default=10.0)
    p.add_argument("--tau-syn-ms", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--weight-scale", type=float, default=0.2)
    p.add_argument("--surrogate-slope", type=float, default=100.0)
    p.add_argument("--init-spectral-radius", type=float, default=1.0)
    p.add_argument("--firing-low", type=float, default=0.02)
    p.add_argument("--firing-high", type=float, default=0.20)
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--nb-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0, help="<=0 -> 2e-4 (fullbptt)")
    p.add_argument("--optimizer", type=str, default="adamax",
                   choices=["adamax", "adam", "sgd"])
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="Use first N of each split")
    # runtime / logging
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--wandb-mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", type=str, default="shd-snn-pretrain")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
