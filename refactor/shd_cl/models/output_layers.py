"""Configurable output layers. Each maps a per-timestep drive ``[B,T,O]`` to
logits ``[B,O]``. The drive is always ``hidden_spikes_t @ W_out`` (times an
optional ``output_gain``); the parent SNN owns ``W_out``, the output layer only
implements the neuron dynamics. There is NO bias anywhere.

Three types (spec §4):

* ``linear_integrator`` : pure accumulator, no leak/reset/spiking.
      out_mem_t = out_mem_{t-1} + drive_t   =>   logits = out_mem_T = sum_t drive_t
  which equals ``hidden_spike_sum @ W_out``. This is the PRIMARY, correct readout
  for ridge (and the one ridge literally fits).

* ``leaky_integrator`` : leaky accumulator, no spiking/reset.
      out_mem_t = beta_out * out_mem_{t-1} + drive_t
  logits from ``last_mem`` (default), ``mean_mem`` or ``max_mem``.

* ``lif_no_reset`` : leaky membrane with spikes but NO reset after firing.
      out_mem_t = beta_out * out_mem_{t-1} + drive_t
      out_spk_t = spike_fn(out_mem_t - output_threshold)
  logits from ``spike_sum`` (default), ``spike_mean``, ``last_mem`` or ``max_mem``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .surrogate import SurrGradSpike

OUTPUT_LAYER_TYPES = ("linear_integrator", "leaky_integrator", "lif_no_reset")
LOGIT_SOURCES = ("spike_sum", "spike_mean", "last_mem", "max_mem", "mean_mem")


class LinearIntegrator(nn.Module):
    """logits = sum_t drive_t (== hidden_spike_sum @ W_out). No parameters."""

    output_layer_type = "linear_integrator"

    def forward(self, drive: torch.Tensor) -> torch.Tensor:  # drive [B,T,O]
        # An explicit cumulative sum over time equals the terminal membrane of a
        # no-leak, no-reset integrator; we take the direct sum for clarity/speed.
        return drive.sum(dim=1)


class LeakyIntegrator(nn.Module):
    """Leaky accumulator, no spiking. logit_readout in {last_mem, mean_mem, max_mem}."""

    output_layer_type = "leaky_integrator"

    def __init__(self, beta_out: float, logit_readout: str = "last_mem"):
        super().__init__()
        if logit_readout not in ("last_mem", "mean_mem", "max_mem"):
            raise ValueError(f"leaky_integrator logit_readout {logit_readout!r} invalid")
        self.beta_out = float(beta_out)
        self.logit_readout = logit_readout

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        B, T, O = drive.shape
        mem = torch.zeros(B, O, device=drive.device, dtype=drive.dtype)
        mems = []
        for t in range(T):
            mem = self.beta_out * mem + drive[:, t]
            mems.append(mem)
        mem_stack = torch.stack(mems, dim=1)                    # [B,T,O]
        if self.logit_readout == "last_mem":
            return mem_stack[:, -1]
        if self.logit_readout == "mean_mem":
            return mem_stack.mean(dim=1)
        return mem_stack.max(dim=1).values                     # max_mem


class LIFNoReset(nn.Module):
    """Leaky spiking membrane with NO reset. logit_source drives what CE sees."""

    output_layer_type = "lif_no_reset"

    def __init__(self, beta_out: float, output_threshold: float,
                 surrogate_slope: float, logit_source: str = "spike_sum"):
        super().__init__()
        if logit_source not in ("spike_sum", "spike_mean", "last_mem", "max_mem"):
            raise ValueError(f"lif_no_reset logit_source {logit_source!r} invalid")
        self.beta_out = float(beta_out)
        self.output_threshold = float(output_threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.logit_source = logit_source

    def spike_fn(self, x: torch.Tensor) -> torch.Tensor:
        return SurrGradSpike.apply(x, self.surrogate_slope)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        B, T, O = drive.shape
        mem = torch.zeros(B, O, device=drive.device, dtype=drive.dtype)
        spk_sum = torch.zeros_like(mem)
        mems = []
        spikes = []
        for t in range(T):
            mem = self.beta_out * mem + drive[:, t]            # leak, NO reset
            spk = self.spike_fn(mem - self.output_threshold)
            spk_sum = spk_sum + spk
            if self.logit_source in ("last_mem", "max_mem"):
                mems.append(mem)
            if self.logit_source == "spike_mean":
                spikes.append(spk)
        if self.logit_source == "spike_sum":
            return spk_sum
        if self.logit_source == "spike_mean":
            return spk_sum / T
        if self.logit_source == "last_mem":
            return mems[-1]
        return torch.stack(mems, dim=1).max(dim=1).values      # max_mem

    def output_firing_rate(self, drive: torch.Tensor) -> float:
        """Mean output spike rate over the window (for logging spiking layers)."""
        B, T, O = drive.shape
        mem = torch.zeros(B, O, device=drive.device, dtype=drive.dtype)
        total = 0.0
        for t in range(T):
            mem = self.beta_out * mem + drive[:, t]
            total += float((mem > self.output_threshold).float().mean().item())
        return total / max(T, 1)


def make_output_layer(output_layer_type: str, *, beta_out: float,
                      output_threshold: float, surrogate_slope: float,
                      logit_source: str = "spike_sum",
                      leaky_readout: str = "last_mem") -> nn.Module:
    """Factory for the three output-layer types."""
    if output_layer_type == "linear_integrator":
        return LinearIntegrator()
    if output_layer_type == "leaky_integrator":
        return LeakyIntegrator(beta_out=beta_out, logit_readout=leaky_readout)
    if output_layer_type == "lif_no_reset":
        return LIFNoReset(beta_out=beta_out, output_threshold=output_threshold,
                          surrogate_slope=surrogate_slope, logit_source=logit_source)
    raise ValueError(
        f"unknown output_layer_type {output_layer_type!r}; choose one of "
        f"{OUTPUT_LAYER_TYPES}")
