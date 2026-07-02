"""Reusable, W&B-independent plots (hidden raster + firing-rate histogram).

Uses the non-interactive Agg backend so plots render headless. Functions return a
matplotlib ``Figure``; ``fig_to_wandb_image`` converts one for logging.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_hidden_raster(hidden_spike_trace: torch.Tensor, *, max_neurons: int = 200,
                       title: str = "hidden raster", most_active: bool = True):
    """Raster of hidden spikes (time on x, neuron index on y).

    ``hidden_spike_trace`` is ``[T, H]`` for a single sample (or ``[B,T,H]`` -> the
    first sample is used). Only ``max_neurons`` neurons are drawn (default: the most
    active) so the image stays readable for large H.
    """
    trace = hidden_spike_trace
    if torch.is_tensor(trace):
        trace = trace.detach().cpu().float().numpy()
    trace = np.asarray(trace)
    if trace.ndim == 3:
        trace = trace[0]
    assert trace.ndim == 2, f"expected [T,H] or [B,T,H], got {trace.shape}"
    T, H = trace.shape

    if H > max_neurons:
        if most_active:
            order = np.argsort(trace.sum(axis=0))[::-1][:max_neurons]
            order = np.sort(order)
        else:
            order = np.arange(max_neurons)
        sub = trace[:, order]
        ylabel = f"hidden neuron (top {max_neurons} of {H} by activity)"
    else:
        sub = trace
        ylabel = f"hidden neuron (of {H})"

    ts, ns = np.nonzero(sub)  # (time_idx, neuron_idx)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ts, ns, s=2, marker="|", color="black")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, sub.shape[1] - 0.5)
    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_firing_rate_histogram(per_neuron_rate: torch.Tensor, *, bins: int = 40,
                               title: str = "hidden firing-rate histogram"):
    """Histogram of per-neuron mean firing rates ``[H]``."""
    r = per_neuron_rate
    if torch.is_tensor(r):
        r = r.detach().cpu().float().numpy()
    r = np.asarray(r).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("per-neuron mean firing rate")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_to_wandb_image(fig):
    """Wrap a matplotlib figure as ``wandb.Image`` (or return None if wandb absent)."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb.Image(fig)


def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=120)
    plt.close(fig)
