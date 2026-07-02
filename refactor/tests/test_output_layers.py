"""Model / output-layer tests: input dim, trace shape, logits shape, per-layer math."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import torch  # noqa: E402

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.models.output_layers import (LeakyIntegrator, LIFNoReset,  # noqa: E402
                                         LinearIntegrator)
from shd_cl.models.snn import ReservoirSNN  # noqa: E402


def _model(output_layer_type="linear_integrator", C=14, H=20, **kw):
    torch.manual_seed(0)
    return ReservoirSNN(C, H, NUM_CLASSES, alpha=0.2, beta=0.3, threshold=1.0,
                        weight_scale=0.5, surrogate_slope=100.0,
                        output_layer_type=output_layer_type, **kw)


def test_model_input_dim_equals_channels():
    m = _model(C=14)
    assert m.nb_inputs == 14
    X = (torch.rand(3, 8, 14) < 0.3).float()
    _ = m(X)                                    # must not raise on C==nb_inputs
    bad = (torch.rand(3, 8, 13) < 0.3).float()
    try:
        m(bad)
        raise AssertionError("expected channel-mismatch assertion")
    except AssertionError as e:
        assert "input channels" in str(e) or "13" in str(e)


def test_hidden_trace_shape_B_T_H():
    m = _model(H=20)
    X = (torch.rand(5, 9, m.nb_inputs) < 0.3).float()
    trace = m.hidden_trace(X)
    assert trace.shape == (5, 9, 20), trace.shape
    assert m.hidden_spike_sum(X).shape == (5, 20)


def test_output_logits_shape_B_20():
    for olt in ("linear_integrator", "leaky_integrator", "lif_no_reset"):
        m = _model(output_layer_type=olt)
        X = (torch.rand(4, 7, m.nb_inputs) < 0.3).float()
        logits = m(X)
        assert logits.shape == (4, NUM_CLASSES), f"{olt}: {logits.shape}"


def test_linear_integrator_is_sum_over_time():
    drive = torch.randn(3, 11, NUM_CLASSES)
    out = LinearIntegrator()(drive)
    assert torch.allclose(out, drive.sum(dim=1), atol=1e-6)


def test_leaky_integrator_recurrence():
    beta = 0.7
    drive = torch.randn(2, 5, 4)
    layer = LeakyIntegrator(beta_out=beta, logit_readout="last_mem")
    # reference: mem_t = beta*mem_{t-1} + drive_t, take last
    mem = torch.zeros(2, 4)
    for t in range(5):
        mem = beta * mem + drive[:, t]
    assert torch.allclose(layer(drive), mem, atol=1e-6)


def test_lif_no_reset_spike_sum_no_reset():
    beta, thr = 0.8, 0.5
    drive = torch.rand(2, 6, 3)
    layer = LIFNoReset(beta_out=beta, output_threshold=thr, surrogate_slope=100.0,
                       logit_source="spike_sum")
    # reference (forward Heaviside, NO reset)
    mem = torch.zeros(2, 3)
    ssum = torch.zeros(2, 3)
    for t in range(6):
        mem = beta * mem + drive[:, t]
        ssum = ssum + (mem > thr).float()
    assert torch.allclose(layer(drive), ssum, atol=1e-6)


def test_spectral_radius_renormalized_to_one():
    m = _model()
    assert abs(m.reservoir.renormalized_spectral_radius - 1.0) < 1e-3


if __name__ == "__main__":
    print("test_output_layers")
    raise SystemExit(run_tests(globals()))
