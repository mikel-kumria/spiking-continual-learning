"""Self-describing checkpoints (Stage 1 -> Stage 2 contract).

A checkpoint carries everything needed to rebuild the exact model and understand
how it was produced: weights, architecture, preprocessing manifest, training
config, class bookkeeping, metrics, git commit and timestamp.
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

import torch

from ..models.snn import ReservoirSNN, derive_beta_out, derive_decays


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def architecture_dict(model: ReservoirSNN, *, tau_mem_ms: float, tau_syn_ms: float,
                      tau_out_mem_ms: float, weight_scale: float, dt_ms: float) -> dict:
    return {
        "nb_inputs": int(model.nb_inputs),
        "nb_hidden": int(model.nb_hidden),
        "nb_outputs": int(model.nb_outputs),
        "alpha": float(model.alpha),
        "beta": float(model.beta),
        "beta_out": float(model.beta_out),
        "tau_mem_ms": float(tau_mem_ms),
        "tau_syn_ms": float(tau_syn_ms),
        "tau_out_mem_ms": float(tau_out_mem_ms),
        "threshold": float(model.threshold),
        "weight_scale": float(weight_scale),
        "surrogate_slope": float(model.surrogate_slope),
        "dt_ms": float(dt_ms),
        "output_layer_type": model.output_layer_type,
        "output_threshold": float(model.output_threshold),
        "logit_source": model.logit_source,
        "leaky_readout": model.leaky_readout,
        "initial_spectral_radius": float(model.reservoir.initial_spectral_radius),
        "renormalized_spectral_radius": float(model.reservoir.renormalized_spectral_radius),
    }


def build_checkpoint(model: ReservoirSNN, *, arch: dict, manifest: dict, config: dict,
                     active_classes: Sequence[int], removed_class: Optional[int],
                     training_method: str, metrics: dict,
                     ridge_lambda: Optional[float] = None,
                     ridge_weighting: Optional[str] = None,
                     removed_class_init_policy: str = "zero",
                     extra: Optional[dict] = None) -> dict:
    ckpt = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_class": type(model).__name__,
        "architecture": arch,
        "preprocessing_manifest": manifest,
        "config": config,
        "active_classes": [int(c) for c in active_classes],
        "removed_class": (int(removed_class) if removed_class is not None else None),
        "removed_class_init_policy": removed_class_init_policy,
        "training_method": training_method,
        "output_layer_type": model.output_layer_type,
        "logit_source": model.logit_source,
        "ridge_lambda": ridge_lambda,
        "ridge_weighting": ridge_weighting,
        "metrics": metrics,
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def load_checkpoint_model(ckpt: dict, device: torch.device) -> ReservoirSNN:
    """Rebuild the exact :class:`ReservoirSNN` from a checkpoint and load weights.

    ``seed_spectral=False`` so the loaded (already-renormalised, possibly trained)
    ``W_rec`` is used as-is instead of being renormalised again.
    """
    a = ckpt["architecture"]
    model = ReservoirSNN(
        nb_inputs=int(a["nb_inputs"]), nb_hidden=int(a["nb_hidden"]),
        nb_outputs=int(a["nb_outputs"]), alpha=float(a["alpha"]), beta=float(a["beta"]),
        threshold=float(a["threshold"]), weight_scale=float(a["weight_scale"]),
        surrogate_slope=float(a["surrogate_slope"]),
        output_layer_type=a.get("output_layer_type", "linear_integrator"),
        output_threshold=float(a.get("output_threshold", 1.0)),
        beta_out=float(a.get("beta_out", 1.0)),
        logit_source=a.get("logit_source", "spike_sum"),
        leaky_readout=a.get("leaky_readout", "last_mem"),
        seed_spectral=False,
    ).to(device)
    state = {k: v.to(device) for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    return model
