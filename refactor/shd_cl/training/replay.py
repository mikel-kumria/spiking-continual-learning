"""Per-old-class replay sampler for class-incremental learning.

Semantics (spec §8) -- replay is defined PER OLD CLASS, not as a total budget::

    n       = number of new-class training samples used
    r       = replay ratio in [0, 1]  (= replay_percent / 100)
    m_old_per_class = round(r * n)          # SAME count for every old class
    total_old_replay = sum over old classes of the sampled count
    total_cil_train  = n + total_old_replay

At ``r = 1.0`` this gives ~``n`` samples per old class plus ``n`` new-class
samples -> class-balanced joint training on ~n samples/class (given enough old
data). If an old class has fewer than ``m_old_per_class`` samples, the
``replay_replacement_policy`` decides what happens.

The sampler returns index arrays (into the new pool and the old pool) plus a log
dict, so the caller can gather either raw ``X`` (for fullbptt) or precomputed
features (for ridge / lastbptt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

REPLAY_REPLACEMENT_POLICIES = ("with_replacement_if_needed", "cap_at_available", "error")


def replay_percentages(max_replay_percent: float, min_replay_percent: float,
                       num_replays: int) -> List[float]:
    """Descending percentage grid, e.g. (100, 0, 11) -> 100,90,...,10,0."""
    if num_replays < 1:
        raise ValueError("num_replays must be >= 1")
    if num_replays == 1:
        return [float(max_replay_percent)]
    return list(np.linspace(max_replay_percent, min_replay_percent, num_replays))


@dataclass
class ReplayPlan:
    new_indices: np.ndarray            # indices into the new-class pool
    replay_indices: np.ndarray         # indices into the old-class replay pool
    log: Dict = field(default_factory=dict)


def sample_replay(new_y: np.ndarray, old_y: np.ndarray, *, replay_ratio: float,
                  rng: np.random.Generator, n_new_cap: int = 0,
                  policy: str = "with_replacement_if_needed") -> ReplayPlan:
    """Build a per-old-class replay plan for one replay ratio.

    ``n_new_cap`` <= 0 uses ALL new-class samples; otherwise at most that many.
    """
    if policy not in REPLAY_REPLACEMENT_POLICIES:
        raise ValueError(f"unknown replay_replacement_policy {policy!r}")
    new_y = np.asarray(new_y)
    old_y = np.asarray(old_y)
    N_new_avail = int(len(new_y))
    if N_new_avail == 0:
        raise ValueError("new-class pool is empty")

    # ---- pick new-class samples ----
    n_new = N_new_avail if n_new_cap <= 0 else min(int(n_new_cap), N_new_avail)
    if n_new >= N_new_avail:
        new_indices = np.arange(N_new_avail, dtype=np.int64)
    else:
        new_indices = np.sort(rng.choice(N_new_avail, size=n_new, replace=False))

    # ---- per-old-class replay ----
    m_old_per_class = int(round(replay_ratio * n_new))
    old_classes = sorted(int(c) for c in np.unique(old_y))
    by_class = {c: np.nonzero(old_y == c)[0] for c in old_classes}

    replay_indices: List[int] = []
    per_class_counts: Dict[int, int] = {}
    replacement_used = False
    for c in old_classes:
        pool = by_class[c]
        avail = int(len(pool))
        want = m_old_per_class
        if want == 0 or avail == 0:
            taken = np.zeros((0,), dtype=np.int64)
        elif want <= avail:
            taken = rng.choice(pool, size=want, replace=False)
        else:  # not enough samples in this class
            if policy == "error":
                raise ValueError(
                    f"old class {c} has {avail} samples but {want} requested; "
                    f"policy=error")
            if policy == "cap_at_available":
                taken = rng.choice(pool, size=avail, replace=False)
            else:  # with_replacement_if_needed
                taken = rng.choice(pool, size=want, replace=True)
                replacement_used = True
        replay_indices.extend(taken.tolist())
        per_class_counts[c] = int(len(taken))

    replay_arr = np.array(replay_indices, dtype=np.int64)
    log = {
        "replay_ratio": float(replay_ratio),
        "replay_percent": float(replay_ratio * 100.0),
        "n_new_samples": int(n_new),
        "m_old_per_class": int(m_old_per_class),
        "total_old_replay": int(len(replay_arr)),
        "total_cil_train": int(n_new + len(replay_arr)),
        "replacement_used": bool(replacement_used),
        "replay_replacement_policy": policy,
        "per_class_replay_counts": per_class_counts,
        "n_old_classes": int(len(old_classes)),
    }
    return ReplayPlan(new_indices=new_indices, replay_indices=replay_arr, log=log)
