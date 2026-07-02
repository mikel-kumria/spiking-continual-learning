"""Accuracy metrics. Empty inputs report NaN (never a misleading 0.0)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .. import NUM_CLASSES


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy; empty -> NaN."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int = NUM_CLASSES
                       ) -> Dict[int, Optional[float]]:
    """Per-class top-1 accuracy over ``[0, num_classes)``; absent class -> None."""
    out: Dict[int, Optional[float]] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in range(num_classes):
        mask = (y_true == c)
        n = int(mask.sum())
        out[c] = float(np.mean(y_pred[mask] == c)) if n > 0 else None
    return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      num_classes: int = NUM_CLASSES) -> float:
    """Mean of per-class recalls over the classes actually present -> [0,1] or NaN.

    This is the right headline metric when the test set is class-imbalanced (e.g.
    a combined old+new CIL test set with ~19x more old samples than new).
    """
    pc = per_class_accuracy(y_true, y_pred, num_classes)
    vals = [v for v in pc.values() if v is not None]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def two_group_balanced_accuracy(old_acc: float, new_acc: float) -> float:
    """Mean of old-group and new-group accuracy (the CIL stability/plasticity mean)."""
    vals = [v for v in (old_acc, new_acc) if v == v]  # drop NaN
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def forgetting_old(old_before: float, old_after: float) -> float:
    return old_before - old_after


def learning_new(new_before: float, new_after: float) -> float:
    return new_after - new_before


def total_delta(total_before: float, total_after: float) -> float:
    return total_after - total_before


def average_old_perclass_forgetting(pc_before: Dict[int, Optional[float]],
                                    pc_after: Dict[int, Optional[float]],
                                    removed_class: int) -> Optional[float]:
    """Mean over OLD classes of ``pc_before - pc_after`` (skip removed/absent)."""
    diffs = []
    for c, vb in pc_before.items():
        if c == removed_class:
            continue
        va = pc_after.get(c)
        if vb is not None and va is not None:
            diffs.append(vb - va)
    if not diffs:
        return None
    return float(np.mean(diffs))
