#!/usr/bin/env python3
"""End-to-end smoke test on SYNTHETIC data (no SHD download needed).

Runs the unit tests, then a tiny full pipeline: preprocess -> pretrain (ridge,
fullbptt, lastbptt) -> class-incremental replay sweep (ridge + fullbptt) ->
audits. Exercises all three output layers. Exits non-zero on any failure.

    python refactor/scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from _common import add_package_to_path

add_package_to_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.data.io import load_npz_split  # noqa: E402
from shd_cl.data.preprocessing import PreprocessConfig, preprocess  # noqa: E402
from shd_cl.evaluation.predict import argmax_predict, collect_both_logits  # noqa: E402
from shd_cl.models.snn import build_model_from_manifest  # noqa: E402
from shd_cl.training.bptt import BpttConfig, train_bptt  # noqa: E402
from shd_cl.training.cil import CILConfig, evaluate_cil, precompute_pool_sums, run_one_cil  # noqa: E402
from shd_cl.training.ridge import apply_readout, fit_readout  # noqa: E402
from shd_cl.evaluation.predict import collect_spike_sums  # noqa: E402
from shd_cl.logging import plots  # noqa: E402
from shd_cl.utils.audit import (audit_active_classes, audit_dataset_dir,  # noqa: E402
                                audit_model_shapes, audit_removed_column_zero, print_audit)
from shd_cl.utils.determinism import set_determinism  # noqa: E402
from shd_cl.utils.device import resolve_device  # noqa: E402

TESTS_DIR = os.path.join(add_package_to_path(), "tests")


def run_unit_tests() -> bool:
    """Import and run each tests/test_*.py in-process."""
    sys.path.insert(0, TESTS_DIR)
    from _bootstrap import run_tests
    ok = True
    for mod in ("test_preprocessing_shapes", "test_label_handling", "test_ridge_shapes",
                "test_replay_sampler", "test_output_layers"):
        print(f"\n=== unit: {mod} ===")
        m = __import__(mod)
        ok &= (run_tests(vars(m)) == 0)
    return ok


def _small_model(manifest, output_layer_type="linear_integrator"):
    # weight_scale is deliberately larger than the SHD default (0.2): with no
    # firing-rate calibration loop (removed per spec), the tiny synthetic reservoir
    # needs stronger input drive to cross threshold and fire (~10% here).
    return build_model_from_manifest(
        manifest, nb_outputs=NUM_CLASSES, tau_mem_ms=10.0, tau_syn_ms=5.0,
        tau_out_mem_ms=20.0, threshold=1.0, weight_scale=2.0, surrogate_slope=100.0,
        nb_hidden=64, output_layer_type=output_layer_type,
        output_threshold=1.0, logit_source="spike_sum", leaky_readout="last_mem")


def _acc(logits, y, active=None):
    pred = argmax_predict(logits, active)
    yt = y.cpu().numpy().astype(np.int64)
    return float(np.mean(pred == yt)) if len(yt) else float("nan")


def run_pipeline() -> bool:
    set_determinism(0)
    device = resolve_device("cpu")
    ok = True
    with tempfile.TemporaryDirectory() as d:
        dataset_dir = os.path.join(d, "dataset")
        cfg = PreprocessConfig(
            experiment_regime="pretrain_19_class", removed_class=10,
            dataset_binning_ms=25.0, dataset_max_seconds=1.0, nb_inputs=40,
            n_compressed_channels=20, channel_compression_method="or_pool",
            synthetic_samples_per_class=12, seed=0)
        manifest = preprocess(cfg, dataset_dir)
        ok &= print_audit("dataset", audit_dataset_dir(dataset_dir, manifest))

        X_tr, y_tr, _ = load_npz_split(os.path.join(dataset_dir, "pretrain_train.npz"))
        X_te, y_te, _ = load_npz_split(os.path.join(dataset_dir, "pretrain_test.npz"))
        active = [int(c) for c in manifest["active_classes"]]

        # -- output-layer forward smoke (all three) --
        for olt in ("linear_integrator", "leaky_integrator", "lif_no_reset"):
            m = _small_model(manifest, olt)
            logits = m(X_tr[:6].to(device))
            assert logits.shape == (6, NUM_CLASSES), f"{olt} logits {logits.shape}"
            print(f"[output-layer] {olt}: logits {tuple(logits.shape)} OK")

        # -- ridge pretrain (19 classes) --
        model = _small_model(manifest).to(device)
        feats = collect_spike_sums(model, X_tr, 32, device)
        W, info = fit_readout(feats, y_tr.numpy(), columns=active, nb_outputs=NUM_CLASSES,
                              lam=1.0, weighting="none", zero_columns=[cfg.removed_class])
        apply_readout(model, W)
        _, lin_te = collect_both_logits(model, X_te, 32, device)
        ridge_acc = _acc(lin_te, y_te, active)
        ok &= print_audit("ridge model", audit_model_shapes(model, manifest))
        ok &= print_audit("ridge active", audit_active_classes(active, cfg.removed_class))
        ok &= print_audit("ridge removed_col", audit_removed_column_zero(model, cfg.removed_class))
        print(f"[ridge] pretrain_test active19 (linear) = {ridge_acc:.3f}")
        ok &= ridge_acc > 0.30  # synthetic classes are separable; must beat chance (~1/19)

        # -- fullbptt + lastbptt pretrain (few epochs) --
        for method in ("fullbptt", "lastbptt"):
            m = _small_model(manifest).to(device)
            res = train_bptt(m, X_tr, y_tr, active_classes=active, num_classes=NUM_CLASSES,
                             cfg=BpttConfig(method=method, nb_epochs=4, batch_size=16,
                                            lr=1e-2, seed=0),
                             device=device, removed_class=cfg.removed_class)
            col = m.W_out.detach()[:, cfg.removed_class]
            ok &= (int(torch.count_nonzero(col).item()) == 0)
            print(f"[{method}] final train_acc={res['final_train_acc']:.3f} "
                  f"(removed col zero: {int(torch.count_nonzero(col).item()) == 0})")

        # -- raster plot renders --
        trace = model.hidden_trace(X_tr[:1].to(device))
        fig = plots.plot_hidden_raster(trace[0], max_neurons=32, title="smoke raster")
        plots.save_fig(fig, os.path.join(d, "raster.png"))
        ok &= os.path.isfile(os.path.join(d, "raster.png"))
        print(f"[raster] saved: {os.path.isfile(os.path.join(d, 'raster.png'))}")

        # -- CIL replay sweep (ridge + fullbptt), a few ratios --
        replay_X, replay_y, _ = load_npz_split(os.path.join(dataset_dir, "pretrain_train.npz"))
        new_X, new_y, _ = load_npz_split(os.path.join(dataset_dir, "continual_train.npz"))
        newt_X, newt_y, _ = load_npz_split(os.path.join(dataset_dir, "continual_test.npz"))
        for method in ("ridge", "fullbptt"):
            m = _small_model(manifest).to(device)
            # give it a pretrained readout first (ridge) so 'before' is meaningful
            feats = collect_spike_sums(m, X_tr, 32, device)
            Wp, _ = fit_readout(feats, y_tr.numpy(), columns=active, nb_outputs=NUM_CLASSES,
                                lam=1.0, weighting="none", zero_columns=[cfg.removed_class])
            apply_readout(m, Wp)
            for p in m.parameters():
                p.requires_grad_(method == "fullbptt")
            state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            cil_cfg = CILConfig(cil_training_method=method, ridge_lambda=1.0,
                                ridge_weighting="normalized_inverse_class_count",
                                batch_size=32,
                                bptt=BpttConfig(method="fullbptt", nb_epochs=3,
                                                batch_size=16, lr=1e-2, seed=0))
            nps = rps = None
            if method == "ridge":
                nps = precompute_pool_sums(m, new_X, 32, device)
                rps = precompute_pool_sums(m, replay_X, 32, device)
            before = evaluate_cil(m, X_te, y_te, newt_X, newt_y, method=method,
                                  removed_class=cfg.removed_class, batch_size=32, device=device)
            row = run_one_cil(m, state, cil_cfg, replay_ratio=1.0,
                              rng=np.random.default_rng(0), new_X=new_X, new_y=new_y.numpy(),
                              replay_X=replay_X, replay_y=replay_y.numpy(), old_test_X=X_te,
                              old_test_y=y_te, new_test_X=newt_X, new_test_y=newt_y,
                              removed_class=cfg.removed_class, active_classes=active,
                              new_pool_sums=nps, replay_pool_sums=rps, before_metrics=before,
                              device=device)
            print(f"[cil-{method}] r=1.0 old {row['old_acc_before']:.2f}->{row['old_acc_after']:.2f} "
                  f"new {row['new_acc_before']:.2f}->{row['new_acc_after']:.2f} "
                  f"bal={row['two_group_balanced_acc_after']:.2f} "
                  f"m/cls={row['m_old_per_class']} n_train={row['total_cil_train']}")
            ok &= (row["new_acc_after"] >= row["new_acc_before"])  # replay must help learn new
    return ok


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST: unit tests")
    print("=" * 70)
    ok = run_unit_tests()
    print("\n" + "=" * 70)
    print("SMOKE TEST: synthetic end-to-end pipeline")
    print("=" * 70)
    ok &= run_pipeline()
    print("\n" + "=" * 70)
    print(f"SMOKE TEST: {'ALL PASSED' if ok else 'FAILURES DETECTED'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
