"""Canonical ``.npz`` split IO and JSON helpers.

One split == one ``.npz`` holding ``X uint8 [N,T,C]``, ``y int64 [N]``,
``speaker int64 [N]``. Loaders return ``X`` as float32 (model input) and ``y`` as
long, never transposing the axes.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import torch


def save_npz_split(path: str, X: np.ndarray, y: np.ndarray,
                   speaker: np.ndarray | None = None) -> None:
    """Persist one split as compressed ``.npz`` (X uint8 [N,T,C], y/speaker int64)."""
    assert X.ndim == 3, f"expected X [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], f"X/y length mismatch {X.shape[0]} vs {y.shape[0]}"
    if speaker is None:
        speaker = np.full((X.shape[0],), -1, dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        X=np.ascontiguousarray(X).astype(np.uint8, copy=False),
        y=np.ascontiguousarray(y).astype(np.int64, copy=False),
        speaker=np.ascontiguousarray(speaker).astype(np.int64, copy=False),
    )


def load_npz_split(path: str, limit: int = 0
                   ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load one ``.npz`` split -> (X float32 [N,T,C], y long [N], speaker int64 [N])."""
    if not os.path.isfile(path):
        raise SystemExit(f"dataset file not found: {path}")
    d = np.load(path)
    if "X" not in d or "y" not in d:
        raise SystemExit(f"{path} must contain 'X' and 'y' (found {list(d.keys())})")
    X = d["X"]
    y = d["y"]
    speaker = d["speaker"] if "speaker" in d else np.full((y.shape[0],), -1, np.int64)
    if limit and limit > 0:
        X, y, speaker = X[:limit], y[:limit], speaker[:limit]
    assert X.ndim == 3, f"expected X [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], f"X/y length mismatch in {path}"
    Xf = torch.from_numpy(np.ascontiguousarray(X)).float()
    yl = torch.from_numpy(np.ascontiguousarray(y)).long()
    return Xf, yl, np.ascontiguousarray(speaker).astype(np.int64)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
