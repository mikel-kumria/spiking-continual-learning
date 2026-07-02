"""Seeding helpers for reproducible splits, weight init and batch order."""
from __future__ import annotations

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Seed Python-NumPy, Torch and CUDA RNGs.

    This makes preprocessing and ridge fully reproducible. It intentionally does
    NOT enable ``torch.use_deterministic_algorithms(True)`` (which can error on
    some ops and slows BPTT); ``fullbptt`` on GPU may therefore not be
    bit-identical across hardware. Documented in REPORT.md.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int) -> torch.Generator:
    """A seeded CPU ``torch.Generator`` for deterministic shuffles."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
