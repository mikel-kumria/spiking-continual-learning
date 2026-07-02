"""Feature extraction, logits and argmax prediction.

Two readouts share the same ``W_out``:

* PRIMARY linear readout: ``argmax((hidden_spike_sum @ W_out))`` -- the readout
  ridge fits; used for ridge accuracy and the "*_linear" diagnostics.
* Output-layer readout: ``argmax(output_layer(drive))`` -- used for BPTT and as a
  secondary diagnostic for ridge when the layer is nonlinear/spiking.

For ``output_layer_type="linear_integrator"`` the two coincide exactly.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..models.snn import ReservoirSNN


@torch.no_grad()
def collect_spike_sums(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                       device: torch.device) -> torch.Tensor:
    """``hidden_spike_sum`` for every sample -> ``[N, H]`` (float64, on CPU)."""
    model.eval()
    feats = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        feats.append(model.hidden_spike_sum(xb).cpu())
    if not feats:
        return torch.zeros((0, model.nb_hidden), dtype=torch.float64)
    return torch.cat(feats, 0).double()


@torch.no_grad()
def collect_traces(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                   device: torch.device) -> torch.Tensor:
    """Per-timestep hidden spikes for every sample -> ``[N, T, H]`` (CPU float32)."""
    model.eval()
    traces = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        traces.append(model.hidden_trace(xb).cpu())
    if not traces:
        T = X.shape[1] if X.ndim == 3 else 0
        return torch.zeros((0, T, model.nb_hidden))
    return torch.cat(traces, 0)


@torch.no_grad()
def output_logits(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    """Output-layer logits ``[N, O]`` (CPU) through the configured output layer."""
    model.eval()
    outs = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        outs.append(model(xb).cpu())
    if not outs:
        return torch.zeros((0, model.nb_outputs))
    return torch.cat(outs, 0)


@torch.no_grad()
def linear_logits(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    """Primary linear readout logits ``[N, O]`` = ``hidden_spike_sum @ W_out``."""
    sums = collect_spike_sums(model, X, batch_size, device).to(device).float()
    return model.linear_logits_from_sum(sums).cpu()


@torch.no_grad()
def collect_both_logits(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                        device: torch.device):
    """Single reservoir pass per batch -> ``(output_layer_logits, linear_logits)``.

    Returns two ``[N, O]`` CPU tensors: the configured output-layer decode and the
    primary linear readout (``hidden_spike_sum @ W_out``). Computing both from one
    trace avoids re-running the reservoir twice during evaluation.
    """
    model.eval()
    out_all, lin_all = [], []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        trace = model.hidden_trace(xb)
        out_all.append(model.logits_from_trace(trace).cpu())
        spike_sum = trace.sum(dim=1)
        lin_all.append(model.linear_logits_from_sum(spike_sum).cpu())
    if not out_all:
        z = torch.zeros((0, model.nb_outputs))
        return z, z.clone()
    return torch.cat(out_all, 0), torch.cat(lin_all, 0)


def argmax_predict(logits: torch.Tensor,
                   active_classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Argmax predictions -> int64 ``[N]``.

    If ``active_classes`` is given, argmax is restricted to those columns and the
    result is mapped back to ORIGINAL labels (used for 19-way pretraining eval).
    """
    if logits.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if active_classes is None:
        return logits.argmax(1).cpu().numpy().astype(np.int64)
    aidx = torch.as_tensor(list(active_classes), dtype=torch.long, device=logits.device)
    sub = logits.index_select(1, aidx)
    return aidx[sub.argmax(1)].cpu().numpy().astype(np.int64)
