"""Tiny config layer: a flat YAML file plus explicit CLI overrides.

We deliberately avoid a heavy config framework. A config is just a flat ``dict``
(loaded from YAML) that scripts read by key. CLI flags override YAML values only
when the user actually passed them (argparse dest defaults to ``None`` for the
overridable flags, and ``None`` means "keep the YAML value").

Access pattern in scripts::

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args, [
        "training_method", "removed_class", "dataset_binning_ms", ...])
    value = cfg["dataset_binning_ms"]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

import yaml


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a flat YAML config into a dict. ``None``/missing -> empty dict."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise SystemExit(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config {path} must be a flat mapping, got {type(data)}")
    return dict(data)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace,
                    keys: Iterable[str]) -> Dict[str, Any]:
    """Return a copy of ``cfg`` with any non-``None`` ``args.<key>`` overriding it.

    Only the listed keys are considered, and only when the parsed value is not
    ``None`` (i.e. the user actually supplied the flag). This keeps the "CLI wins,
    else YAML, else script default" precedence explicit and predictable.
    """
    out = dict(cfg)
    for key in keys:
        if not hasattr(args, key):
            continue
        val = getattr(args, key)
        if val is not None:
            out[key] = val
    return out


def with_defaults(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys in ``cfg`` from ``defaults`` (cfg values take priority)."""
    out = dict(defaults)
    out.update({k: v for k, v in cfg.items() if v is not None})
    return out


def require(cfg: Dict[str, Any], key: str) -> Any:
    """Fetch ``key`` or raise a clear error naming the missing config value."""
    if key not in cfg or cfg[key] is None:
        raise SystemExit(f"missing required config value: {key!r}")
    return cfg[key]


def dump_config(cfg: Dict[str, Any]) -> str:
    """Human-readable one-line-per-key dump for logs."""
    return json.dumps(cfg, indent=2, default=str, sort_keys=True)
