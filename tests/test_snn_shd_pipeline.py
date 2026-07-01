#!/usr/bin/env python3
"""Smoke / unit tests for the two-stage SHD continual-learning pipeline.

Runs without the real SHD dataset: the end-to-end test drives both scripts in
``--synthetic-samples-per-class`` mode. Runnable two ways::

    pytest tests/test_snn_shd_pipeline.py
    python3 tests/test_snn_shd_pipeline.py        # plain runner, no pytest needed

Covers (mirrors the audit checklist):
  1. channel compression  2. split leakage  3. model feature/logit shapes
  4. pretraining label masking  5. RLS update  6. tiny synthetic end-to-end
  7. static py_compile of all new modules   (+ per-class-null reporting)
"""
from __future__ import annotations

import csv
import json
import os
import py_compile
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PY = sys.executable

import snn_shd_common as C          # noqa: E402
import pretrain_snn_shd as P        # noqa: E402

NEW_MODULES = ["snn_shd_common.py", "pretrain_snn_shd.py",
               "class_incremental_snn_shd.py"]

# Firing-friendly tiny synthetic config (reservoir actually spikes).
_SYNTH = dict(removed_class=7, dataset_binning_ms=10, max_time=0.5,
              tau_mem_ms=20, tau_syn_ms=10, nb_inputs=70, n_compressed=35,
              nb_hidden=100, weight_scale=0.6, samples_per_class=20)

_PIPELINE = {}  # memoized (pre_dir, cil_dir, tmpdir)


def _run_pipeline():
    """Run Stage 1 + Stage 2 once (subprocess) into a temp dir; memoized."""
    if _PIPELINE:
        return _PIPELINE["pre"], _PIPELINE["cil"]
    tmp = tempfile.mkdtemp(prefix="snn_shd_e2e_")
    pre_root = os.path.join(tmp, "shd_pretraining")
    pre_dir = os.path.join(pre_root, "rc7_ridge")
    cil_dir = os.path.join(tmp, "shd_class_incremental", "rc7_rls")
    s = _SYNTH
    r1 = subprocess.run([
        PY, "pretrain_snn_shd.py", "--output-root", pre_root, "--run-name", "rc7_ridge",
        "--removed-class", str(s["removed_class"]),
        "--dataset-binning-ms", str(s["dataset_binning_ms"]),
        "--max-time", str(s["max_time"]), "--tau-mem-ms", str(s["tau_mem_ms"]),
        "--tau-syn-ms", str(s["tau_syn_ms"]), "--nb-inputs", str(s["nb_inputs"]),
        "--n-compressed-channels", str(s["n_compressed"]),
        "--channel-compression-method", "or_pool", "--mode", "ridge",
        "--nb-hidden", str(s["nb_hidden"]), "--batch-size", "32",
        "--weight-scale", str(s["weight_scale"]),
        "--synthetic-samples-per-class", str(s["samples_per_class"]),
        "--seed", "1", "--device", "cpu", "--wandb-mode", "disabled",
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r1.returncode == 0, f"pretrain failed:\n{r1.stdout}\n{r1.stderr}"
    r2 = subprocess.run([
        PY, "class_incremental_snn_shd.py", "--pretrain-run-dir", pre_dir,
        "--output-dir", cil_dir, "--rls-delta", "1.0",
        "--rls-forgetting-factor", "1.0", "--replay-runs", "3",
        "--replay-start-percent", "0", "--replay-end-percent", "100",
        "--replay-ratio-mode", "additive", "--n-seeds", "2", "--batch-size", "16",
        "--seed", "0", "--device", "cpu", "--wandb-mode", "disabled",
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r2.returncode == 0, f"class-incremental failed:\n{r2.stdout}\n{r2.stderr}"
    _PIPELINE.update(pre=pre_dir, cil=cil_dir, tmp=tmp)
    return pre_dir, cil_dir


# ---------------------------------------------------------------------------
# 1. Channel compression
# ---------------------------------------------------------------------------


def test_channel_compression():
    rng = np.random.default_rng(0)
    x = (rng.random((4, 13, 700)) > 0.7).astype(np.uint8)   # [N, T, C]
    out = C.compress_channels(x, "or_pool", 10)
    assert out.shape == (4, 13, 70), out.shape          # [T,700]->[T,70]
    assert out.shape[1] == x.shape[1]                   # time axis unchanged
    assert set(np.unique(out)).issubset({0, 1})         # binary
    C.assert_compression_invariants(x, out, "or_pool", 10)
    # no compression when factor == 1 (identity)
    assert np.array_equal(C.compress_channels(x, "or_pool", 1), x)
    # graded preserves the total spike count
    g = C.compress_channels(x, "graded", 10)
    assert int(g.sum()) == int(x.sum())
    # invalid divisor fails loudly
    for bad in (lambda: C.validate_compression_factor(700, 33),
                lambda: C.compress_channels(x, "or_pool", 3)):
        try:
            bad(); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert C.validate_compression_factor(700, 70) == 10


# ---------------------------------------------------------------------------
# 2. Split leakage
# ---------------------------------------------------------------------------


def test_split_leakage():
    pre_dir, _ = _run_pipeline()
    dd = os.path.join(pre_dir, "dataset")
    removed = _SYNTH["removed_class"]
    for split in ("pretrain_train", "pretrain_test"):
        y = np.load(os.path.join(dd, f"{split}.npz"))["y"]
        assert removed not in set(y.tolist()), f"{split} leaks removed class"
    for split in ("continual_train", "continual_test"):
        y = np.load(os.path.join(dd, f"{split}.npz"))["y"]
        assert set(y.tolist()) <= {removed}, f"{split} has non-removed classes"
    # labels stay original 0..19 (never permanently remapped)
    y = np.load(os.path.join(dd, "pretrain_train.npz"))["y"]
    assert y.min() >= 0 and y.max() < C.NUM_CLASSES


# ---------------------------------------------------------------------------
# 3. Model feature / logit shapes
# ---------------------------------------------------------------------------


def test_model_feature_shape():
    B, T, Cin, H = 5, 9, 35, 40
    model = C.ReservoirSNN(Cin, H, 20, alpha=0.37, beta=0.61, threshold=1.0,
                           weight_scale=0.6, surrogate_slope=100.0)
    x = (torch.rand(B, T, Cin) > 0.6).float()
    Phi, _ = model.hidden_spikes(x)
    assert tuple(Phi.shape) == (B, H), Phi.shape           # Phi = [B, H]
    logits = model.logits_from_features(Phi)
    assert tuple(logits.shape) == (B, 20), logits.shape    # logits = [B, 20]
    assert tuple(model(x).shape) == (B, 20)


# ---------------------------------------------------------------------------
# 4. Pretraining label masking
# ---------------------------------------------------------------------------


def test_pretrain_label_masking():
    removed = _SYNTH["removed_class"]
    active = [c for c in range(C.NUM_CLASSES) if c != removed]
    assert len(active) == 19                                # 19 active classes
    pre_dir, _ = _run_pipeline()
    ck = torch.load(os.path.join(pre_dir, "checkpoints", "pretrained_model.pt"),
                    map_location="cpu", weights_only=False)
    assert ck["active_classes"] == active
    assert ck["removed_class"] == removed
    # removed class has no pretraining samples
    y = np.load(os.path.join(pre_dir, "dataset", "pretrain_train.npz"))["y"]
    assert (y == removed).sum() == 0
    # temporary remap active labels -> [0, 18], invertible, never out of range
    pos = {c: i for i, c in enumerate(active)}
    remapped = np.array([pos[int(c)] for c in y])
    assert remapped.min() >= 0 and remapped.max() <= 18
    inv = np.array(active)[remapped]
    assert np.array_equal(inv, y)                          # remap is invertible
    # removed output column was deterministically zeroed
    assert float(ck["model_state_dict"]["W_out"][:, removed].abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# 5. RLS update
# ---------------------------------------------------------------------------


def test_rls_update():
    H = 16
    model = C.ReservoirSNN(35, H, 20, alpha=0.37, beta=0.61, threshold=1.0,
                           weight_scale=0.6, surrogate_slope=100.0)
    W_in0 = model.W_in.detach().clone()
    W_rec0 = model.W_rec.detach().clone()
    W_out0 = model.W_out.detach().clone()

    rls = C.RLS(model.W_out.detach(), delta=1.0, lambda_forgetting=1.0)
    assert tuple(rls.P.shape) == (H, H)                    # P is [H, H]
    assert tuple(rls.W.shape) == (H, 20)                   # W_out is [H, 20]
    rng = np.random.default_rng(0)
    for _ in range(8):
        phi = torch.from_numpy(rng.standard_normal(H))
        target = C.one_hot(np.array([int(rng.integers(0, 20))]))[0]
        rls.update(phi, target)
    assert torch.isfinite(rls.P).all() and torch.isfinite(rls.W).all()  # no NaN/Inf
    rls.copy_into(model)
    # only W_out changed; reservoir frozen
    assert torch.allclose(model.W_in, W_in0)
    assert torch.allclose(model.W_rec, W_rec0)
    assert not torch.allclose(model.W_out, W_out0)
    assert torch.isfinite(model(torch.rand(3, 9, 35)).float()).all()    # logits finite
    # bad hyperparameters fail loudly
    for bad in (dict(delta=0.0), dict(lambda_forgetting=0.0),
                dict(lambda_forgetting=1.5)):
        try:
            C.RLS(W_out0, **bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_spiking_output_layer():
    B, T, Cin, H = 4, 12, 35, 30
    m = C.SpikingReadoutReservoirSNN(
        Cin, H, 20, alpha=0.37, beta=0.61, threshold=1.0, weight_scale=0.6,
        surrogate_slope=100.0, output_gain=1.0, output_threshold=1.0,
        output_alpha=C.OUTPUT_IF_ALPHA, output_beta=C.OUTPUT_IF_BETA)
    x = (torch.rand(B, T, Cin) > 0.5).float()
    Psi = m.output_rates(x)
    assert tuple(Psi.shape) == (B, 20)                  # mean output spikes [B,20]
    assert float(Psi.min()) >= 0.0 and float(Psi.max()) <= 1.0  # rates in [0,1]
    assert tuple(m(x).shape) == (B, 20)                 # forward == output rates
    # output_rates_from_trace must match output_rates (same dynamics, reused trace)
    _, trace = m.hidden_spikes(x, return_trace=True)
    Psi2 = m.output_rates_from_trace(trace)
    assert torch.allclose(Psi, Psi2, atol=1e-6)
    # a zeroed read-out column produces ZERO output rate (the minimum) -> it can
    # never spuriously win the argmax (this removes the CE zero-column artifact).
    with torch.no_grad():
        m.W_out[:, 10] = 0.0
    Psi3 = m.output_rates(x)
    assert float(Psi3[:, 10].abs().sum()) == 0.0


def test_per_class_accuracy_null():
    y_true = np.array([0, 0, 1, 3, 3])     # class 2 absent
    y_pred = np.array([0, 1, 1, 3, 0])
    pc = C.per_class_accuracy(y_true, y_pred)
    assert pc[2] is None                   # absent -> null, not 0.0
    assert pc[0] == 0.5 and pc[1] == 1.0
    assert pc[19] is None


# ---------------------------------------------------------------------------
# 6. Tiny synthetic end-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_files_and_learning():
    pre_dir, cil_dir = _run_pipeline()
    # pretrain artifacts
    for rel in ("config.json", "preprocessing_manifest.json", "metrics.json",
                "checkpoints/pretrained_model.pt"):
        assert os.path.isfile(os.path.join(pre_dir, rel)), rel
    for split in C.SPLIT_NAMES:
        assert os.path.isfile(os.path.join(pre_dir, "dataset", f"{split}.npz")), split
    # class-incremental artifacts
    for rel in ("config.json", "inherited_pretrain_config.json",
                "inherited_preprocessing_manifest.json", "replay_sweep_results.csv",
                "replay_sweep_results.jsonl", "replay_sweep_summary.json"):
        assert os.path.isfile(os.path.join(cil_dir, rel)), rel
    for ppp in ("000", "050", "100"):
        for sss in ("000", "001"):
            f = os.path.join(cil_dir, "checkpoints", f"final_ratio_{ppp}_seed_{sss}.pt")
            assert os.path.isfile(f), f
    # rows: 3 ratios x 2 seeds
    rows = list(csv.DictReader(open(os.path.join(cil_dir, "replay_sweep_results.csv"))))
    assert len(rows) == 6, len(rows)
    # the new class is actually learned by RLS (it was ~0 before)
    for r in rows:
        assert float(r["new_acc_after"]) >= float(r["new_acc_before"])
    # more replay -> less forgetting (monotone non-increasing across mean forgetting)
    summ = json.load(open(os.path.join(cil_dir, "replay_sweep_summary.json")))
    forget = [e["forgetting_old_mean"] for e in summ["per_ratio"]]
    assert forget[0] >= forget[-1] - 1e-9, forget   # replay reduces forgetting
    assert summ["is_rehearsal"] is True


# ---------------------------------------------------------------------------
# 7. Static checks
# ---------------------------------------------------------------------------


def test_py_compile_all_modules():
    for mod in NEW_MODULES:
        py_compile.compile(os.path.join(REPO_ROOT, mod), doraise=True)


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
