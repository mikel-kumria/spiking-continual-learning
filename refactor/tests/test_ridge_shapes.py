"""Ridge tests: feature/W_out shapes, removed-column zero, weighted sqrt(w), and
the primary linear readout == argmax(hidden_spike_sum @ W_out)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.evaluation.predict import collect_spike_sums  # noqa: E402
from shd_cl.models.snn import ReservoirSNN  # noqa: E402
from shd_cl.training.ridge import compute_sample_weights, fit_readout, solve_ridge  # noqa: E402


def _model(C=12, H=16):
    torch.manual_seed(0)
    return ReservoirSNN(C, H, NUM_CLASSES, alpha=0.2, beta=0.3, threshold=1.0,
                        weight_scale=0.5, surrogate_slope=100.0,
                        output_layer_type="linear_integrator")


def test_ridge_feature_shape_N_H():
    m = _model()
    X = (torch.rand(9, 10, m.nb_inputs) < 0.3).float()
    feats = collect_spike_sums(m, X, batch_size=4, device=torch.device("cpu"))
    assert feats.shape == (9, m.nb_hidden), feats.shape


def test_ridge_Wout_shape_H_20():
    m = _model()
    X = (torch.rand(30, 10, m.nb_inputs) < 0.3).float()
    feats = collect_spike_sums(m, X, 8, torch.device("cpu"))
    y = np.array([i % NUM_CLASSES for i in range(30)])
    W, _ = fit_readout(feats, y, columns=list(range(NUM_CLASSES)),
                       nb_outputs=NUM_CLASSES, lam=1.0, weighting="none")
    assert W.shape == (m.nb_hidden, NUM_CLASSES), W.shape


def test_ridge_pretrain_removed_column_zero():
    m = _model()
    removed = 10
    active = [c for c in range(NUM_CLASSES) if c != removed]
    X = (torch.rand(38, 10, m.nb_inputs) < 0.3).float()
    feats = collect_spike_sums(m, X, 8, torch.device("cpu"))
    y = np.array([active[i % len(active)] for i in range(38)])  # only active labels
    W, info = fit_readout(feats, y, columns=active, nb_outputs=NUM_CLASSES, lam=1.0,
                          weighting="none", zero_columns=[removed])
    assert W.shape == (m.nb_hidden, NUM_CLASSES)
    assert torch.count_nonzero(W[:, removed]).item() == 0, "removed column must be 0"
    assert removed in info["zeroed_columns"]


def test_weighted_ridge_uses_sqrt_weight():
    """solve_ridge with weights must equal the closed form (X^T D X + lam I)^-1 X^T D Y."""
    g = torch.Generator().manual_seed(3)
    N, H, K, lam = 40, 7, 3, 0.5
    X = torch.randn(N, H, generator=g, dtype=torch.float64)
    Y = torch.randn(N, K, generator=g, dtype=torch.float64)
    w = torch.rand(N, generator=g, dtype=torch.float64) + 0.1
    W, _ = solve_ridge(X, Y, lam, sample_weight=w)
    D = torch.diag(w)
    A = X.T @ D @ X + lam * torch.eye(H, dtype=torch.float64)
    W_ref = torch.linalg.solve(A, X.T @ D @ Y)
    assert torch.allclose(W, W_ref, atol=1e-8), (W - W_ref).abs().max().item()


def test_normalized_inverse_class_count_weights():
    y = np.array([0, 0, 0, 1])           # class 0 has 3, class 1 has 1
    w, C = compute_sample_weights(y, "normalized_inverse_class_count")
    # w_i = N / (C * n_class[y_i]); N=4, C=2 -> class0: 4/(2*3)=0.667, class1: 4/(2*1)=2.0
    assert C == 2
    assert np.allclose(w, [4 / 6, 4 / 6, 4 / 6, 4 / 2])
    # each class contributes equal total weight
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())


def test_primary_linear_readout_matches_spike_sum_matmul():
    m = _model()
    with torch.no_grad():
        m.W_out.normal_(0, 0.5)
    X = (torch.rand(6, 10, m.nb_inputs) < 0.3).float()
    trace = m.hidden_trace(X)
    spike_sum = trace.sum(dim=1)
    manual = spike_sum @ m.W_out
    via_helper = m.linear_logits_from_sum(spike_sum)
    via_forward = m(X)                                  # linear_integrator forward
    assert torch.allclose(manual, via_helper, atol=1e-5)
    assert torch.allclose(manual, via_forward, atol=1e-5), \
        "linear_integrator forward must equal spike_sum @ W_out"


if __name__ == "__main__":
    print("test_ridge_shapes")
    raise SystemExit(run_tests(globals()))
