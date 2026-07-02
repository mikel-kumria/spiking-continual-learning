"""Recurrent hidden reservoir of second-order LIF neurons.

Dynamics (per timestep ``t``), preserving the reference update ordering exactly::

    drive_t   = x_t @ W_in + spk_prev @ W_rec        # recurrence uses PREVIOUS spikes
    spk_t     = spike_fn(mem_t - threshold)          # spike from membrane at step start
    new_syn   = alpha * syn_t + drive_t
    mem_next  = (beta * mem_t + syn_t) * (1 - stop_grad(spk_t))   # integrate OLD syn; hard reset
    syn_next  = new_syn
    spk_prev  = spk_t

Notes for faithful replication:
* The membrane integrates ``syn_t`` (the synaptic state BEFORE ``drive_t`` is added),
  so there is a built-in latency ``drive_t -> syn_{t+1} -> mem_{t+2} -> spk_{t+2}``.
* Reset is a hard reset-to-zero (``mem *= 1 - spk``), not subtract-threshold.
* The reset mask uses ``spk.detach()`` so no gradient flows through the reset path.

There is NO bias anywhere. ``W_rec`` is renormalised ONCE (deterministically) to a
target spectral radius; there is no iterative firing-rate sanity loop.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .surrogate import SurrGradSpike


class RecurrentReservoir(nn.Module):
    """W_in -> recurrent hidden LIF -> per-timestep hidden spike trace [B,T,H]."""

    def __init__(self, nb_inputs: int, nb_hidden: int, *, alpha: float, beta: float,
                 threshold: float, weight_scale: float, surrogate_slope: float,
                 target_spectral_radius: float = 1.0, seed_spectral: bool = True):
        super().__init__()
        self.nb_inputs = int(nb_inputs)
        self.nb_hidden = int(nb_hidden)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.target_spectral_radius = float(target_spectral_radius)

        W_in = torch.empty(nb_inputs, nb_hidden)
        W_rec = torch.empty(nb_hidden, nb_hidden)
        nn.init.normal_(W_in, 0.0, weight_scale / math.sqrt(nb_inputs))
        nn.init.normal_(W_rec, 0.0, weight_scale / math.sqrt(nb_hidden))
        self.W_in = nn.Parameter(W_in)
        self.W_rec = nn.Parameter(W_rec)

        # Provenance: measure the spectral radius BEFORE and AFTER the one-time
        # deterministic renormalisation to the target (default rho = 1.0).
        self.initial_spectral_radius = self.measured_spectral_radius()
        if seed_spectral:
            self.renormalize_spectral_radius(self.target_spectral_radius)
        self.renormalized_spectral_radius = self.measured_spectral_radius()

    # -- helpers ---------------------------------------------------------------
    def spike_fn(self, x: torch.Tensor) -> torch.Tensor:
        return SurrGradSpike.apply(x, self.surrogate_slope)

    def measured_spectral_radius(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.eigvals(
                self.W_rec.detach().to("cpu", torch.float32)).abs().max().item())

    def renormalize_spectral_radius(self, target: float) -> float:
        """Scale ``W_rec`` once so its spectral radius equals ``target`` exactly."""
        rho = self.measured_spectral_radius()
        if rho <= 0:
            raise RuntimeError("measured spectral radius is 0; cannot renormalise W_rec")
        with torch.no_grad():
            self.W_rec.mul_(target / rho)
        return target

    # -- forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reservoir -> hidden spike trace ``[B, T, H]`` (values in {0,1})."""
        x = x.float()
        B, T, C = x.shape
        assert C == self.nb_inputs, (
            f"input channels {C} != reservoir nb_inputs {self.nb_inputs}")
        H = self.nb_hidden
        h_in = (x.reshape(B * T, C) @ self.W_in).reshape(B, T, H)  # feed-forward drive

        syn = torch.zeros(B, H, device=x.device, dtype=h_in.dtype)
        mem = torch.zeros_like(syn)
        spk_prev = torch.zeros_like(syn)
        trace = []
        for t in range(T):
            drive = h_in[:, t] + spk_prev @ self.W_rec
            spk = self.spike_fn(mem - self.threshold)
            rst = spk.detach()
            new_syn = self.alpha * syn + drive
            mem = (self.beta * mem + syn) * (1.0 - rst)
            syn = new_syn
            spk_prev = spk
            trace.append(spk)
        return torch.stack(trace, dim=1)  # [B, T, H]

    # -- reduced features ------------------------------------------------------
    @staticmethod
    def spike_sum(trace: torch.Tensor) -> torch.Tensor:
        """Sum of hidden spikes over time -> ``[B, H]`` (the ridge feature)."""
        return trace.sum(dim=1)

    @staticmethod
    def spike_mean(trace: torch.Tensor) -> torch.Tensor:
        """Mean of hidden spikes over time -> ``[B, H]``."""
        return trace.mean(dim=1)

    def firing_diagnostics(self, trace: torch.Tensor) -> dict:
        """Firing-rate diagnostics on a trace ``[B,T,H]`` (logged, not acted on)."""
        active_bt = trace > 0                                    # [B,T,H]
        never = (~active_bt.any(dim=(0, 1)))                     # [H]
        always = active_bt.all(dim=(0, 1))                       # [H]
        H = self.nb_hidden
        return {
            "mean_hidden_firing_rate": float(trace.mean().item()),
            "frac_silent_hidden": float(never.sum().item()) / H,
            "frac_always_firing_hidden": float(always.sum().item()) / H,
        }

    def per_neuron_firing_rate(self, trace: torch.Tensor) -> torch.Tensor:
        """Per-neuron mean firing rate over batch+time -> ``[H]`` (for histograms)."""
        return trace.float().mean(dim=(0, 1)).detach().cpu()
