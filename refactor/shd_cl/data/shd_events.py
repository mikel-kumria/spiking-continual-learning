"""Raw SHD event pool: load HDF5, merge pools, or fabricate synthetic events.

An ``EventPool`` holds the raw (variable-length) spike streams in memory. Times
are in seconds; units are channel indices in ``[0, 700)``. Dense binning happens
later in :mod:`shd_cl.data.preprocessing`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from .. import NATIVE_SHD_CHANNELS, NUM_CLASSES


@dataclass
class EventPool:
    times: List[np.ndarray]   # firing times in seconds, one float64 array per sample
    units: List[np.ndarray]   # channel index per spike, aligned with ``times``
    labels: np.ndarray        # int64 label per sample      [N]
    speakers: np.ndarray      # int64 speaker id per sample  [N]

    def __len__(self) -> int:
        return len(self.labels)


def read_h5_pool(path: str) -> EventPool:
    """Load one SHD HDF5 file (``shd_train.h5`` / ``shd_test.h5``) into memory."""
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


def merge_pools(pools: List[EventPool]) -> EventPool:
    """Concatenate several event pools into one (used to merge train+test)."""
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


def synthetic_pool(samples_per_class: int, nb_inputs: int = NATIVE_SHD_CHANNELS,
                   max_time_s: float = 1.4, seed: int = 0,
                   num_classes: int = NUM_CLASSES,
                   spikes_per_channel: int = 8) -> EventPool:
    """Fabricate a tiny event pool (all ``num_classes`` classes) for smoke tests.

    Each class fires in a distinct, wide channel band, densely enough in time that
    the binned drive actually crosses the LIF threshold (so the reservoir spikes
    and a classifier beats chance). No SHD download required.
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
            u = np.repeat(active, spikes_per_channel)
            t = rng.uniform(0.0, max_time_s * 0.95, size=u.shape[0])
            order = np.argsort(t)
            times.append(t[order].astype(np.float64))
            units.append(u[order].astype(np.int64))
            labels.append(c)
            speakers.append(int(rng.integers(0, 5)))
    return EventPool(times, units,
                     np.asarray(labels, np.int64), np.asarray(speakers, np.int64))
