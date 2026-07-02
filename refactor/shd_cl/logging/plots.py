"""Reusable, W&B-independent plots (hidden raster + firing-rate histogram).

Uses the non-interactive Agg backend so plots render headless. Functions return a
matplotlib ``Figure``; ``fig_to_wandb_image`` converts one for logging.

Publication exports (PDF/SVG) use TrueType fonts (``pdf.fonttype=42``) so labels
stay editable in Illustrator/Inkscape.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PUBLICATION_FIGSIZE = (6.5, 4.0)   # inches, ~ double-column width
PUBLICATION_RCPARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def _apply_publication_style() -> None:
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid", context="paper", font_scale=1.1)
    except ImportError:
        plt.style.use("ggplot")
    plt.rcParams.update(PUBLICATION_RCPARAMS)


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


def plot_noise_robustness(curves: dict, *, title: str = "W_rec noise robustness",
                          xlabel: str = "noise μ (absolute std)",
                          ylabel: str = "test accuracy"):
    """Line plot of test accuracy vs injected ``W_rec`` noise level.

    ``curves`` maps a series label to a list of ``{"mu", "test_acc"}`` rows.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in curves.items():
        xs = [r["mu"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_noise_acc_delta(rows: list, *, title: str = "fullbptt − ridge test accuracy",
                         xlabel: str = "noise μ (absolute std)",
                         ylabel: str = "Δ test accuracy (fullbptt − ridge)"):
    """Line plot of accuracy gap vs injected ``W_rec`` noise level."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["mu"] for r in rows]
    ys = [r["test_acc_delta"] for r in rows]
    ax.plot(xs, ys, marker="o", linewidth=1.8, color="tab:green")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_noise_robustness_band(stats: dict, *, title: str = "W_rec noise robustness",
                               xlabel: str = "noise μ (absolute std)",
                               ylabel: str = "test accuracy",
                               colors: Optional[dict] = None):
    """Mean line with ±1 std shaded band (seaborn-style), one series per method."""
    _apply_publication_style()

    colors = colors or {}
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    for i, (label, rows) in enumerate(stats.items()):
        xs = np.asarray([r["mu"] for r in rows], dtype=float)
        mean = np.asarray([r["mean"] for r in rows], dtype=float)
        std = np.asarray([r["std"] for r in rows], dtype=float)
        color = colors.get(label, default_colors[i % len(default_colors)])
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.30,
                        linewidth=0, zorder=1)
        ax.plot(xs, mean, color=color, linewidth=2.2, label=label, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.legend(title="method")
    fig.tight_layout()
    return fig


def plot_noise_acc_delta_band(rows: list, *, title: str = "fullbptt − ridge test accuracy",
                              xlabel: str = "noise μ (absolute std)",
                              ylabel: str = "Δ test accuracy (fullbptt − ridge)",
                              color: str = "#2ca02c"):
    """Mean line with ±1 std shaded band for the fullbptt−ridge accuracy gap."""
    _apply_publication_style()

    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    xs = np.asarray([r["mu"] for r in rows], dtype=float)
    mean = np.asarray([r["mean"] for r in rows], dtype=float)
    std = np.asarray([r["std"] for r in rows], dtype=float)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.30,
                    linewidth=0, zorder=1)
    ax.plot(xs, mean, color=color, linewidth=2.2, label="fullbptt − ridge", zorder=2)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def band_plot_arrays(rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xs, mean, lower, upper)`` for validating band plots."""
    xs = np.asarray([r["mu"] for r in rows], dtype=float)
    mean = np.asarray([r["mean"] for r in rows], dtype=float)
    std = np.asarray([r["std"] for r in rows], dtype=float)
    return xs, mean, mean - std, mean + std


def plotly_noise_robustness_band(stats: dict, *, title: str = "W_rec noise robustness",
                                 xlabel: str = "noise μ (absolute std)",
                                 ylabel: str = "test accuracy",
                                 colors: Optional[dict] = None):
    """Interactive mean ± std band plot for W&B (Plotly)."""
    import plotly.graph_objects as go

    colors = colors or {}
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig = go.Figure()
    for i, (label, rows) in enumerate(stats.items()):
        xs = [r["mu"] for r in rows]
        mean = np.asarray([r["mean"] for r in rows], dtype=float)
        std = np.asarray([r["std"] for r in rows], dtype=float)
        color = colors.get(label, default_colors[i % len(default_colors)])
        upper = (mean + std).tolist()
        lower = (mean - std).tolist()
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=upper + lower[::-1], fill="toself",
            fillcolor=_hex_to_rgba(color if color.startswith("#") else f"#{color}", 0.22),
            line=dict(width=0), hoverinfo="skip", showlegend=False, name=f"{label} ±1σ"))
        fig.add_trace(go.Scatter(
            x=xs, y=mean.tolist(), mode="lines", name=label,
            line=dict(width=3.5, color=color)))
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel,
        yaxis=dict(range=[0, 1.02]), template="plotly_white")
    return fig


def plotly_noise_acc_delta_band(rows: list, *, title: str = "fullbptt − ridge test accuracy",
                                xlabel: str = "noise μ (absolute std)",
                                ylabel: str = "Δ test accuracy (fullbptt − ridge)",
                                color: str = "#2ca02c"):
    """Interactive mean ± std band plot for the accuracy gap (Plotly)."""
    import plotly.graph_objects as go

    xs = [r["mu"] for r in rows]
    mean = np.asarray([r["mean"] for r in rows], dtype=float)
    std = np.asarray([r["std"] for r in rows], dtype=float)
    upper = (mean + std).tolist()
    lower = (mean - std).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1], y=upper + lower[::-1], fill="toself",
        fillcolor=_hex_to_rgba(color, 0.22), line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="±1σ"))
    fig.add_trace(go.Scatter(
        x=xs, y=mean.tolist(), mode="lines", name="fullbptt − ridge",
        line=dict(width=3.5, color=color)))
    fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
    return fig


def save_fig(fig, path: str, *, dpi: int = 600) -> None:
    """Save a figure; vector formats (``.pdf``/``.svg``/``.eps``) ignore ``dpi``."""
    ext = os.path.splitext(path)[1].lower()
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.05, "facecolor": "white"}
    if ext in (".pdf", ".svg", ".eps"):
        fig.savefig(path, format=ext.lstrip("."), **kwargs)
    else:
        fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)


def save_fig_publication(fig, base_path: str, *, dpi: int = 600,
                         formats: Sequence[str] = (".pdf", ".svg", ".png")) -> list[str]:
    """Save publication-ready figures: vector PDF/SVG plus optional high-res PNG.

    ``base_path`` should not include a file extension.
    """
    saved = []
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.05, "facecolor": "white"}
    for ext in formats:
        path = base_path + ext
        if ext in (".pdf", ".svg", ".eps"):
            fig.savefig(path, format=ext.lstrip("."), **kwargs)
        else:
            fig.savefig(path, dpi=dpi, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved
