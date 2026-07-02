"""W_rec noise injection: absolute mu, restore semantics, shared noise."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import torch  # noqa: E402
import numpy as np  # noqa: E402

from shd_cl.evaluation.noise_robustness import (  # noqa: E402
    apply_w_rec_noise, inject_w_rec_noise, make_w_rec_noise, restore_w_rec,
    sweep_w_rec_noise)
from shd_cl.models.snn import ReservoirSNN  # noqa: E402
from shd_cl.logging import plots  # noqa: E402


def _model():
    torch.manual_seed(0)
    return ReservoirSNN(8, 12, 4, alpha=0.2, beta=0.3, threshold=1.0,
                        weight_scale=0.5, surrogate_slope=100.0,
                        output_layer_type="linear_integrator", seed_spectral=False)


def test_mu_zero_leaves_w_rec_unchanged():
    m = _model()
    backup = m.W_rec.data.clone()
    mu = inject_w_rec_noise(m, 0.0)
    assert mu == 0.0
    assert torch.allclose(m.W_rec.data, backup)


def test_absolute_mu_sets_noise_std():
    m = _model()
    backup = m.W_rec.data.clone()
    mu = 0.05
    inject_w_rec_noise(m, mu, rng=torch.Generator().manual_seed(1))
    delta = (m.W_rec.data - backup).flatten()
    assert abs(delta.std(unbiased=False).item() - mu) < 0.02


def test_shared_noise_tensor_is_identical():
    m1, m2 = _model(), _model()
    m2.load_state_dict(m1.state_dict())
    noise = make_w_rec_noise(m1.W_rec.data, 0.03, rng=torch.Generator().manual_seed(2))
    apply_w_rec_noise(m1, noise)
    apply_w_rec_noise(m2, noise)
    assert torch.allclose(m1.W_rec.data, m2.W_rec.data)


def test_restore_reverts_noise():
    m = _model()
    backup = m.W_rec.data.clone()
    inject_w_rec_noise(m, 0.05, rng=torch.Generator().manual_seed(1))
    assert not torch.allclose(m.W_rec.data, backup)
    restore_w_rec(m, backup)
    assert torch.allclose(m.W_rec.data, backup)


def test_sweep_restores_original_weights():
    m = _model()
    backup = m.W_rec.data.clone()
    X = (torch.rand(6, 5, m.nb_inputs) < 0.3).float()
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    rows = sweep_w_rec_noise(
        m, X, y, mu_values=[0.0, 0.05, 0.1], batch_size=3,
        device=torch.device("cpu"), use_linear_readout=True, seed=0)
    assert len(rows) == 3
    assert torch.allclose(m.W_rec.data, backup)


def test_aggregate_mean_std():
    mu_values = [0.0, 0.1]
    raw = [[0.9, 0.91, 0.89], [0.5, 0.55, 0.45]]
    rows = [
        {"mu": mu, "mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=0))}
        for mu, accs in zip(mu_values, raw)
    ]
    assert rows[0]["mean"] == 0.9
    assert abs(rows[1]["mean"] - 0.5) < 1e-9
    assert rows[1]["std"] > 0
    _, mean, lower, upper = plots.band_plot_arrays(rows)
    assert np.allclose(lower, mean - np.asarray([r["std"] for r in rows]))
    assert np.allclose(upper, mean + np.asarray([r["std"] for r in rows]))


if __name__ == "__main__":
    run_tests(globals())
