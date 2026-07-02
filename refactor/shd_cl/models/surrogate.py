"""Surrogate-gradient spike function (Heaviside forward, fast-sigmoid backward)."""
from __future__ import annotations

import torch


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward (Zenke/Pittorino).

    forward:  ``s = (x > 0)``  (exact step)
    backward: ``grad_in = grad_out / (slope * |x| + 1)**2``

    The surrogate gradient is largest at ``x = 0`` (membrane at threshold) and
    small far from threshold, which is why the reservoir must fire in a usable
    range for gradients to flow in BPTT.
    """

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output / (ctx.slope * x.abs() + 1.0) ** 2
        return grad, None


def spike(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Convenience wrapper around :class:`SurrGradSpike`."""
    return SurrGradSpike.apply(x, slope)
