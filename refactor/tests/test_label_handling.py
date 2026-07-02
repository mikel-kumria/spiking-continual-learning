"""Label-handling tests: original ids kept, leakage, BPTT removed-column freeze."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.data.io import load_npz_split  # noqa: E402
from shd_cl.data.preprocessing import PreprocessConfig, preprocess  # noqa: E402
from shd_cl.models.snn import ReservoirSNN  # noqa: E402
from shd_cl.training.bptt import BpttConfig, train_bptt  # noqa: E402


def _make_dataset(d, removed=10):
    cfg = PreprocessConfig(
        experiment_regime="pretrain_19_class", removed_class=removed,
        dataset_binning_ms=25.0, dataset_max_seconds=1.0, nb_inputs=100,
        n_compressed_channels=20, channel_compression_method="or_pool",
        synthetic_samples_per_class=6, seed=0)
    return preprocess(cfg, os.path.join(d, "dataset")), cfg


def test_labels_remain_original_0_19():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d)
        for name in ("pretrain_train", "continual_train"):
            _, y, _ = load_npz_split(os.path.join(d, "dataset", f"{name}.npz"))
            u = y.numpy()
            assert u.min() >= 0 and u.max() < NUM_CLASSES


def test_pretrain_split_excludes_removed_class():
    with tempfile.TemporaryDirectory() as d:
        _, cfg = _make_dataset(d, removed=7)
        for name in ("pretrain_train", "pretrain_test"):
            _, y, _ = load_npz_split(os.path.join(d, "dataset", f"{name}.npz"))
            assert cfg.removed_class not in set(y.tolist())


def test_continual_split_only_removed_class():
    with tempfile.TemporaryDirectory() as d:
        _, cfg = _make_dataset(d, removed=3)
        for name in ("continual_train", "continual_test"):
            _, y, _ = load_npz_split(os.path.join(d, "dataset", f"{name}.npz"))
            assert set(y.tolist()) <= {cfg.removed_class}


def test_bptt_pretraining_keeps_removed_column_zero():
    """19-class BPTT must never train the removed output column."""
    torch.manual_seed(0)
    removed = 5
    active = [c for c in range(NUM_CLASSES) if c != removed]
    B, T, C, H = 8, 10, 12, 16
    model = ReservoirSNN(C, H, NUM_CLASSES, alpha=0.2, beta=0.3, threshold=1.0,
                         weight_scale=0.5, surrogate_slope=100.0,
                         output_layer_type="linear_integrator")
    X = (torch.rand(B, T, C) < 0.3).float()
    y = torch.tensor([active[i % len(active)] for i in range(B)])  # only active labels
    cfg = BpttConfig(method="fullbptt", nb_epochs=3, batch_size=4, lr=1e-2, seed=0)
    train_bptt(model, X, y, active_classes=active, num_classes=NUM_CLASSES, cfg=cfg,
               device=torch.device("cpu"), removed_class=removed)
    col = model.W_out.detach()[:, removed]
    assert int(torch.count_nonzero(col).item()) == 0, "removed column must stay exactly 0"


if __name__ == "__main__":
    print("test_label_handling")
    raise SystemExit(run_tests(globals()))
