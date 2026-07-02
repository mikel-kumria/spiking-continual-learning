"""Deterministic, label-stratified two-way (train/test) splitting.

Splits are per-class proportional: every present class contributes roughly
``test_fraction`` of its samples to test (and keeps at least one train sample),
so no class is accidentally dropped from the train side.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def validate_two_fractions(train_frac: float, test_frac: float) -> None:
    for name, v in (("train", train_frac), ("test", test_frac)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}_fraction={v} must be in [0, 1]")
    total = train_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0 (got {total:.6f})")


def stratified_two_way(labels: np.ndarray, *, train_frac: float, test_frac: float,
                       rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class proportional train/test split; every present class keeps >= 1 train.

    Returns ``(train_idx, test_idx)`` as int64 arrays indexing into ``labels``.
    """
    validate_two_fractions(train_frac, test_frac)
    train_idx: List[int] = []
    test_idx: List[int] = []
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, l in enumerate(labels.tolist()):
        by_class[int(l)].append(i)
    for cls in sorted(by_class):
        idxs = np.array(by_class[cls], dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_frac))
        if test_frac > 0 and n_test == 0 and n >= 2:
            n_test = 1
        n_train = n - n_test
        if n_train < 1:
            n_train = 1
            n_test = n - 1
        train_idx.extend(idxs[:n_train].tolist())
        test_idx.extend(idxs[n_train:].tolist())
    return np.array(train_idx, np.int64), np.array(test_idx, np.int64)
