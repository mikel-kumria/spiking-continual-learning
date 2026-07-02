"""Channel-axis-only compression (700 -> n_compressed_channels).

Native SHD has 700 cochlea channels. We reduce them to ``n_compressed_channels``
by grouping ``factor = 700 / n_compressed_channels`` *adjacent* channels and
pooling each group. The compression touches ONLY the last (channel) axis; the
time axis (axis -2) is never modified. ``n_compressed_channels`` must divide the
native channel count exactly (no silent truncation/padding).

Methods
-------
* ``or_pool``        : binary, group fires if >= ``condition_or`` spikes land in it
                       (``condition_or=1`` is a logical OR over the group).
* ``conditional_or`` : same rule as ``or_pool`` (alias; tune with ``condition_or``).
* ``graded``         : integer spike-count per group (count-preserving).
* ``bernoulli``      : binary, group fires with prob count/factor (seeded RNG).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

COMPRESSION_METHODS = ("or_pool", "conditional_or", "graded", "bernoulli")


def validate_compression_factor(nb_inputs: int, n_compressed: int) -> int:
    """Return the integer compression factor or raise a clear ``ValueError``."""
    if n_compressed <= 0:
        raise ValueError(f"n_compressed_channels must be positive, got {n_compressed}")
    if n_compressed > nb_inputs:
        raise ValueError(
            f"n_compressed_channels={n_compressed} cannot exceed nb_inputs={nb_inputs}")
    if nb_inputs % n_compressed != 0:
        divisors = [d for d in range(1, nb_inputs + 1)
                    if nb_inputs % d == 0 and d <= 100]
        raise ValueError(
            f"n_compressed_channels={n_compressed} must divide nb_inputs={nb_inputs} "
            f"exactly (remainder {nb_inputs % n_compressed}); choose a divisor of "
            f"{nb_inputs} (e.g. {divisors} ...).")
    return nb_inputs // n_compressed


def _group_count(x: np.ndarray, factor: int) -> np.ndarray:
    """``[..., C] -> [..., C//factor]`` spike count per adjacent channel group."""
    *lead, c = x.shape
    if c % factor != 0:
        raise ValueError(f"channel count {c} not divisible by factor {factor}")
    target = c // factor
    return x.reshape(*lead, target, factor).sum(axis=-1, dtype=np.uint16)


def compress_channels(x: np.ndarray, method: str, factor: int, *,
                      condition_or: int = 1,
                      rng: Optional[np.random.Generator] = None,
                      bernoulli_seed: int = 42) -> np.ndarray:
    """Compress the channel (last) axis of ``x`` ([..., C]) by ``factor``.

    ``factor == 1`` is an identity copy. Output dtype is uint8 (graded may exceed
    1 but is capped by ``factor`` which is small in practice).
    """
    if factor == 1:
        return x.astype(np.uint8, copy=True)
    counts = _group_count(x, factor)  # [..., C//factor], uint16
    if method in ("or_pool", "conditional_or"):
        thr = max(1, int(condition_or))
        return (counts >= thr).astype(np.uint8)
    if method == "graded":
        cap = int(counts.max(initial=0))
        dtype = np.uint8 if cap <= 255 else np.uint16
        return counts.astype(dtype)
    if method == "bernoulli":
        if rng is None:
            rng = np.random.default_rng(bernoulli_seed)
        p = np.clip(counts.astype(np.float32) / float(factor), 0.0, 1.0)
        return (rng.random(p.shape, dtype=np.float32) < p).astype(np.uint8)
    raise ValueError(
        f"unknown channel_compression_method={method!r}; "
        f"available: {sorted(COMPRESSION_METHODS)}")


def assert_compression_invariants(x_in: np.ndarray, x_out: np.ndarray,
                                  method: str, factor: int) -> None:
    """Cheap shape/value invariants; doubles as a T<->C transpose tripwire."""
    assert x_out.shape[-2] == x_in.shape[-2], (
        f"{method}: time axis changed {x_in.shape} -> {x_out.shape}")
    assert x_out.shape[-1] * factor == x_in.shape[-1], (
        f"{method}: channel axis not reduced by factor {factor}")
    assert x_out.shape[:-1] == x_in.shape[:-1], f"{method}: leading dims changed"
    assert int(x_out.min(initial=0)) >= 0, f"{method}: negative values"
    if method in ("or_pool", "conditional_or", "bernoulli"):
        assert set(np.unique(x_out)).issubset({0, 1}), f"{method} must be binary"
    if method == "graded":
        assert int(x_out.sum()) == int(x_in.sum()), "graded must preserve spike count"
        assert int(x_out.max(initial=0)) <= factor, "graded value cannot exceed factor"
