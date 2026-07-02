"""Preprocessing shape + compression-axis tests (synthetic data, no HDF5)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import numpy as np  # noqa: E402

from shd_cl.data.compression import compress_channels, assert_compression_invariants  # noqa: E402
from shd_cl.data.io import load_npz_split  # noqa: E402
from shd_cl.data.preprocessing import (PreprocessConfig, compute_nb_steps,  # noqa: E402
                                       preprocess)


def _synth_cfg(regime="pretrain_19_class", **kw):
    base = dict(experiment_regime=regime, removed_class=10, dataset_binning_ms=20.0,
                dataset_max_seconds=1.0, nb_inputs=100, n_compressed_channels=20,
                channel_compression_method="or_pool", synthetic_samples_per_class=6,
                seed=0)
    base.update(kw)
    return PreprocessConfig(**base)


def test_nb_steps_formula():
    # ceil(1.4 / 0.014) = 100 ; ceil(1.0 / 0.02) = 50
    assert compute_nb_steps(1.4, 14.0) == 100
    assert compute_nb_steps(1.0, 20.0) == 50


def test_preprocessing_produces_N_T_C():
    with tempfile.TemporaryDirectory() as d:
        cfg = _synth_cfg()
        man = preprocess(cfg, os.path.join(d, "dataset"))
        nb_steps = man["nb_steps"]
        assert nb_steps == compute_nb_steps(cfg.dataset_max_seconds, cfg.dataset_binning_ms)
        for name in ("pretrain_train", "pretrain_test", "continual_train", "continual_test"):
            X, y, _ = load_npz_split(os.path.join(d, "dataset", f"{name}.npz"))
            assert X.ndim == 3, f"{name} not [N,T,C]: {X.shape}"
            assert X.shape[1] == nb_steps, f"{name} time axis wrong: {X.shape}"
            assert X.shape[2] == cfg.n_compressed_channels, f"{name} channel axis wrong"


def test_baseline_regime_shapes():
    with tempfile.TemporaryDirectory() as d:
        cfg = _synth_cfg(regime="baseline_20_class")
        man = preprocess(cfg, os.path.join(d, "dataset"))
        for name in ("train", "test"):
            X, y, _ = load_npz_split(os.path.join(d, "dataset", f"{name}.npz"))
            assert X.shape[1:] == (man["nb_steps"], cfg.n_compressed_channels)
        # baseline has all 20 classes in train
        Xtr, ytr, _ = load_npz_split(os.path.join(d, "dataset", "train.npz"))
        assert set(int(v) for v in np.unique(ytr.numpy())) == set(range(20))


def test_compression_changes_only_channel_axis():
    rng = np.random.default_rng(0)
    x = (rng.random((37, 100)) < 0.2).astype(np.uint8)   # [T=37, C=100]
    for method in ("or_pool", "conditional_or", "graded", "bernoulli"):
        out = compress_channels(x, method, factor=5, condition_or=1,
                                rng=np.random.default_rng(1))
        assert out.shape == (37, 20), f"{method}: {out.shape}"   # T unchanged, C 100->20
        assert_compression_invariants(x, out, method, 5)
    # graded preserves total spike count
    g = compress_channels(x, "graded", factor=5)
    assert int(g.sum()) == int(x.sum())


if __name__ == "__main__":
    print("test_preprocessing_shapes")
    raise SystemExit(run_tests(globals()))
