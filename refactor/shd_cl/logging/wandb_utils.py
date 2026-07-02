"""Thin W&B wrappers that degrade gracefully to a no-op.

Every function accepts a ``run`` that may be ``None`` (W&B disabled or not
installed). ``init_wandb`` returns ``None`` for ``mode="disabled"`` or when the
``wandb`` package is missing, so the rest of the pipeline runs unchanged offline.
"""
from __future__ import annotations

from typing import Optional, Sequence


def init_wandb(*, mode: str, project: str, name: Optional[str], entity: Optional[str],
               config: dict, tags: Optional[Sequence[str]] = None):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    return wandb.init(project=project, name=name, entity=entity, mode=mode,
                      config=config, tags=list(tags) if tags else None)


def log(run, payload: dict) -> None:
    if run is not None:
        run.log(payload)


def define_metric(run, name: str, *, step_metric: Optional[str] = None) -> None:
    """Register a custom x-axis for W&B line charts (``step_metric`` = x key)."""
    if run is None:
        return
    import wandb
    if step_metric is None:
        wandb.define_metric(name)
    else:
        wandb.define_metric(name, step_metric=step_metric)


def log_image(run, key: str, fig) -> None:
    """Log a matplotlib figure under ``key`` (no-op if run is None)."""
    if run is None:
        return
    from .plots import fig_to_wandb_image
    img = fig_to_wandb_image(fig)
    if img is not None:
        run.log({key: img})


def set_summary(run, metrics: dict) -> None:
    if run is None:
        return
    for k, v in metrics.items():
        run.summary[k] = v


def finish(run) -> None:
    if run is not None:
        run.finish()


def log_plotly(run, key: str, fig) -> None:
    """Log a Plotly figure under ``key`` (no-op if run is None)."""
    if run is not None:
        run.log({key: fig})
