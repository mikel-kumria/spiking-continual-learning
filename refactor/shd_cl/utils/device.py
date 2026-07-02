"""Device resolution (cpu / cuda / mps) with an ``auto`` policy."""
from __future__ import annotations

import torch


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available())


def resolve_device(choice: str) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``mps``.

    ``auto`` prefers cuda, then mps, then cpu. Note: the closed-form ridge solve
    runs in float64 on the CPU regardless of this device (MPS has no float64);
    only the SNN forward/backward uses the accelerator, so MPS is safe here.
    """
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not _mps_available():
            raise SystemExit("--device mps requested but MPS is not available")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
