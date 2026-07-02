"""The full recurrent SNN: reservoir + linear ``W_out`` + configurable output layer.

    input raster [B,T,C]
        -> W_in [C,H]
        -> recurrent hidden reservoir (H second-order LIF neurons, W_rec)
        -> hidden spike trace [B,T,H]
        -> W_out [H, nb_outputs]                     (NO bias)
        -> configurable output layer -> logits [B, nb_outputs]

Feature contract for ridge / lastbptt: ``hidden_spike_sum = trace.sum(dim=1)``.
For ``output_layer_type="linear_integrator"`` (the default readout), the output
logits are EXACTLY ``hidden_spike_sum @ W_out``, so ridge and BPTT
are evaluated on equivalent logits. For leaky/lif output layers the output-layer
decode differs from the linear readout; that is logged as a secondary diagnostic.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .output_layers import make_output_layer
from .reservoir import RecurrentReservoir


def derive_decays(dt_ms: float, tau_mem_ms: float, tau_syn_ms: float
                  ) -> Tuple[float, float]:
    """``alpha = exp(-dt/tau_syn)``, ``beta = exp(-dt/tau_mem)`` (ms units)."""
    alpha = math.exp(-dt_ms / tau_syn_ms)
    beta = math.exp(-dt_ms / tau_mem_ms)
    return alpha, beta


def derive_beta_out(dt_ms: float, tau_out_mem_ms: float) -> float:
    """``beta_out = exp(-dt/tau_out_mem)`` for leaky/lif output layers."""
    return math.exp(-dt_ms / tau_out_mem_ms)


class ReservoirSNN(nn.Module):
    """Recurrent reservoir + linear read-out + configurable output-layer decode."""

    def __init__(self, nb_inputs: int, nb_hidden: int, nb_outputs: int, *,
                 alpha: float, beta: float, threshold: float, weight_scale: float,
                 surrogate_slope: float,
                 output_layer_type: str = "linear_integrator",
                 output_threshold: float = 1.0,
                 beta_out: float = 1.0, logit_source: str = "spike_sum",
                 leaky_readout: str = "last_mem",
                 target_spectral_radius: float = 1.0,
                 seed_spectral: bool = True):
        super().__init__()
        self.nb_inputs = int(nb_inputs)
        self.nb_hidden = int(nb_hidden)
        self.nb_outputs = int(nb_outputs)
        self.output_layer_type = output_layer_type
        self.output_threshold = float(output_threshold)
        self.beta_out = float(beta_out)
        self.logit_source = logit_source
        self.leaky_readout = leaky_readout

        self.reservoir = RecurrentReservoir(
            nb_inputs, nb_hidden, alpha=alpha, beta=beta, threshold=threshold,
            weight_scale=weight_scale, surrogate_slope=surrogate_slope,
            target_spectral_radius=target_spectral_radius, seed_spectral=seed_spectral)

        W_out = torch.empty(nb_hidden, nb_outputs)
        nn.init.normal_(W_out, 0.0, weight_scale / math.sqrt(nb_hidden))
        self.W_out = nn.Parameter(W_out)

        self.output_layer = make_output_layer(
            output_layer_type, beta_out=self.beta_out,
            output_threshold=self.output_threshold, surrogate_slope=surrogate_slope,
            logit_source=logit_source, leaky_readout=leaky_readout)

    # -- convenience passthroughs ---------------------------------------------
    @property
    def alpha(self) -> float:
        return self.reservoir.alpha

    @property
    def beta(self) -> float:
        return self.reservoir.beta

    @property
    def threshold(self) -> float:
        return self.reservoir.threshold

    @property
    def surrogate_slope(self) -> float:
        return self.reservoir.surrogate_slope

    @property
    def W_in(self) -> nn.Parameter:
        return self.reservoir.W_in

    @property
    def W_rec(self) -> nn.Parameter:
        return self.reservoir.W_rec

    # -- forward paths ---------------------------------------------------------
    def hidden_trace(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden spike trace ``[B,T,H]`` (values in {0,1})."""
        return self.reservoir(x)

    def hidden_spike_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Ridge feature: sum of hidden spikes over time ``[B,H]``."""
        return self.reservoir.spike_sum(self.reservoir(x))

    def output_drive(self, trace: torch.Tensor,
                     W_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-timestep output drive ``[B,T,O] = trace @ W_out``."""
        W = self.W_out if W_out is None else W_out.to(trace.dtype)
        B, T, H = trace.shape
        return (trace.reshape(B * T, H) @ W).reshape(B, T, self.nb_outputs)

    def logits_from_trace(self, trace: torch.Tensor,
                          W_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Output-layer logits ``[B,O]`` from a hidden trace."""
        return self.output_layer(self.output_drive(trace, W_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: input -> reservoir -> output layer -> logits ``[B,O]``."""
        return self.logits_from_trace(self.reservoir(x))

    def linear_logits_from_sum(self, spike_sum: torch.Tensor,
                               W_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PRIMARY linear readout: ``hidden_spike_sum @ W_out``.

        This bypasses any nonlinear output layer and is the correct ridge readout.
        """
        W = self.W_out if W_out is None else W_out.to(spike_sum.dtype)
        return spike_sum @ W


def build_model_from_manifest(manifest: dict, *, nb_outputs: int,
                              tau_mem_ms: float, tau_syn_ms: float,
                              tau_out_mem_ms: float, threshold: float,
                              weight_scale: float, surrogate_slope: float,
                              nb_hidden: int, output_layer_type: str,
                              output_threshold: float,
                              logit_source: str, leaky_readout: str,
                              target_spectral_radius: float = 1.0,
                              seed_spectral: bool = True) -> ReservoirSNN:
    """Construct a :class:`ReservoirSNN` from a preprocessing manifest.

    Neuron decays are derived from the dataset binning ``dt_ms`` (the manifest's
    ``dataset_binning_ms``). ``nb_inputs`` is the compressed channel count.
    """
    dt_ms = float(manifest["dataset_binning_ms"])
    nb_inputs = int(manifest["nb_inputs"])
    alpha, beta = derive_decays(dt_ms, tau_mem_ms, tau_syn_ms)
    beta_out = derive_beta_out(dt_ms, tau_out_mem_ms)
    return ReservoirSNN(
        nb_inputs, nb_hidden, nb_outputs, alpha=alpha, beta=beta, threshold=threshold,
        weight_scale=weight_scale, surrogate_slope=surrogate_slope,
        output_layer_type=output_layer_type,
        output_threshold=output_threshold, beta_out=beta_out,
        logit_source=logit_source, leaky_readout=leaky_readout,
        target_spectral_radius=target_spectral_radius, seed_spectral=seed_spectral)
