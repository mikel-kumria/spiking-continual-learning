#!/usr/bin/env python3
"""Pretrain the recurrent SNN on the 19 OLD classes (removed class held out).

Preprocesses (into ``<output_root>/<run_name>/dataset`` unless already present),
then trains W_out (ridge / lastbptt) or all weights (fullbptt) on the 19 active
classes with the removed-class output column held at exactly 0. Saves a
self-describing checkpoint that ``cil_sweep.py`` consumes.

Example::

    python refactor/scripts/pretrain.py \
      --config refactor/configs/pretrain_default.yaml \
      --training-method ridge --removed-class 10 --dataset-binning-ms 14 \
      --dataset-max-seconds 1.4 --n-compressed-channels 70 \
      --channel-compression-method or_pool
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from _common import add_package_to_path, ensure_dataset

add_package_to_path()

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.data.io import load_npz_split, write_json  # noqa: E402
from shd_cl.evaluation.predict import argmax_predict, collect_both_logits  # noqa: E402
from shd_cl.models.snn import build_model_from_manifest  # noqa: E402
from shd_cl.training.bptt import BpttConfig, train_bptt  # noqa: E402
from shd_cl.training.ridge import apply_readout, fit_readout  # noqa: E402
from shd_cl.logging import plots, wandb_utils  # noqa: E402
from shd_cl.utils.audit import (audit_active_classes, audit_model_shapes,  # noqa: E402
                                audit_removed_column_zero, print_audit)
from shd_cl.utils.checkpointing import (architecture_dict, build_checkpoint)  # noqa: E402
from shd_cl.utils.config import apply_overrides, load_config  # noqa: E402
from shd_cl.utils.determinism import set_determinism  # noqa: E402
from shd_cl.utils.device import resolve_device  # noqa: E402


OVERRIDABLE = [
    "run_name", "output_root", "removed_class", "dataset_binning_ms",
    "dataset_max_seconds", "n_compressed_channels", "channel_compression_method",
    "training_method", "ridge_lambda", "ridge_weighting", "nb_hidden", "nb_epochs",
    "batch_size", "lr", "optimizer", "grad_clip", "output_layer_type", "output_gain",
    "output_threshold", "logit_source", "leaky_readout", "device", "wandb_mode",
    "wandb_name", "limit", "seed",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    for key in OVERRIDABLE:
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, default=None)
    return p.parse_args()


def _acc(logits, y, active_classes=None):
    pred = argmax_predict(logits, active_classes)
    yt = y.cpu().numpy().astype(np.int64)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(pred == yt))


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args, OVERRIDABLE)
    cfg["experiment_regime"] = "pretrain_19_class"
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = resolve_device(cfg["device"])
    method = cfg["training_method"]
    removed_class = int(cfg["removed_class"])
    limit = int(cfg.get("limit", 0) or 0)
    print(f"device={device}  method={method}  removed_class={removed_class}  "
          f"output_layer={cfg['output_layer_type']}")

    run_dir = os.path.join(cfg["output_root"], cfg["run_name"])
    dataset_dir = os.path.join(run_dir, "dataset")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Stage 1a: dataset ----
    manifest = ensure_dataset(cfg, run_dir, "pretrain_19_class", removed_class)
    active_classes = [int(c) for c in manifest["active_classes"]]
    assert len(active_classes) == NUM_CLASSES - 1

    X_tr, y_tr, _ = load_npz_split(os.path.join(dataset_dir, "pretrain_train.npz"), limit)
    X_te, y_te, _ = load_npz_split(os.path.join(dataset_dir, "pretrain_test.npz"), limit)
    Xc_te, yc_te, _ = load_npz_split(os.path.join(dataset_dir, "continual_test.npz"), limit)
    print(f"pretrain_train X={tuple(X_tr.shape)}  test X={tuple(X_te.shape)}")

    # ---- model ----
    model = build_model_from_manifest(
        manifest, nb_outputs=int(cfg["nb_outputs"]), tau_mem_ms=float(cfg["tau_mem_ms"]),
        tau_syn_ms=float(cfg["tau_syn_ms"]), tau_out_mem_ms=float(cfg["tau_out_mem_ms"]),
        threshold=float(cfg["threshold"]), weight_scale=float(cfg["weight_scale"]),
        surrogate_slope=float(cfg["surrogate_slope"]), nb_hidden=int(cfg["nb_hidden"]),
        output_layer_type=cfg["output_layer_type"], output_gain=float(cfg["output_gain"]),
        output_threshold=float(cfg["output_threshold"]), logit_source=cfg["logit_source"],
        leaky_readout=cfg["leaky_readout"],
        target_spectral_radius=float(cfg["target_spectral_radius"])).to(device)

    # ---- reservoir diagnostics (one-time renorm, no firing-rate loop) ----
    sane_n = min(int(cfg["batch_size"]), X_tr.shape[0])
    with torch.no_grad():
        diag_trace = model.hidden_trace(X_tr[:sane_n].to(device))
    fire = model.reservoir.firing_diagnostics(diag_trace)
    reservoir_diag = {
        "initial_spectral_radius": model.reservoir.initial_spectral_radius,
        "target_spectral_radius": float(cfg["target_spectral_radius"]),
        "renormalized_spectral_radius": model.reservoir.renormalized_spectral_radius,
        **fire}
    print(f"[reservoir] rho_init={reservoir_diag['initial_spectral_radius']:.3f} "
          f"-> rho={reservoir_diag['renormalized_spectral_radius']:.3f} | "
          f"firing={fire['mean_hidden_firing_rate']:.4f} "
          f"silent={fire['frac_silent_hidden']:.3f} "
          f"always={fire['frac_always_firing_hidden']:.3f}")

    wandb_run = wandb_utils.init_wandb(
        mode=cfg["wandb_mode"], project=cfg["wandb_project"],
        name=cfg.get("wandb_name") or cfg["run_name"], entity=cfg.get("wandb_entity"),
        config={**cfg, **{f"manifest/{k}": manifest[k] for k in
                          ("nb_steps", "nb_inputs", "compression_factor")},
                **{f"reservoir/{k}": v for k, v in reservoir_diag.items()}},
        tags=["shd", "pretrain", method, f"removed{removed_class}",
              cfg["output_layer_type"]])

    # ---- Stage 1b: train on the 19 active classes ----
    t0 = time.time()
    ridge_info = {}
    if method == "ridge":
        from shd_cl.evaluation.predict import collect_spike_sums
        feats = collect_spike_sums(model, X_tr, int(cfg["batch_size"]), device)
        W_full, ridge_info = fit_readout(
            feats, y_tr.numpy(), columns=active_classes, nb_outputs=NUM_CLASSES,
            lam=float(cfg["ridge_lambda"]), weighting=cfg["ridge_weighting"],
            zero_columns=[removed_class])
        apply_readout(model, W_full)
    else:  # fullbptt / lastbptt
        bptt_cfg = BpttConfig(method=method, nb_epochs=int(cfg["nb_epochs"]),
                              batch_size=int(cfg["batch_size"]), lr=float(cfg["lr"]),
                              optimizer=cfg["optimizer"], grad_clip=float(cfg["grad_clip"]),
                              seed=seed)
        train_bptt(model, X_tr, y_tr, active_classes=active_classes,
                   num_classes=NUM_CLASSES, cfg=bptt_cfg, device=device,
                   removed_class=removed_class, X_te=X_te, y_te=y_te,
                   on_epoch=lambda rec: wandb_utils.log(wandb_run, rec))
    train_seconds = time.time() - t0

    # ---- metrics (primary = per method; linear diagnostic always logged) ----
    out_tr, lin_tr = collect_both_logits(model, X_tr, int(cfg["batch_size"]), device)
    out_te, lin_te = collect_both_logits(model, X_te, int(cfg["batch_size"]), device)
    out_ct, lin_ct = collect_both_logits(model, Xc_te, int(cfg["batch_size"]), device)
    primary_out = (method != "ridge")  # ridge -> linear primary; bptt -> output primary
    P_tr, P_te, P_ct = (out_tr, out_te, out_ct) if primary_out else (lin_tr, lin_te, lin_ct)

    metrics = {
        "training_method": method,
        "primary_readout": "output" if primary_out else "linear",
        "train_seconds": train_seconds,
        "pretrain_train_acc_active19": _acc(P_tr, y_tr, active_classes),
        "pretrain_test_acc_active19": _acc(P_te, y_te, active_classes),
        "pretrain_test_acc_full20_diagnostic": _acc(P_te, y_te, None),
        "continual_test_acc_before_cil": _acc(P_ct, yc_te, None),
        # linear (primary-ridge) readout diagnostics
        "pretrain_train_acc_active19_linear": _acc(lin_tr, y_tr, active_classes),
        "pretrain_test_acc_active19_linear": _acc(lin_te, y_te, active_classes),
        "pretrain_test_acc_full20_linear": _acc(lin_te, y_te, None),
        "continual_test_acc_before_cil_linear": _acc(lin_ct, yc_te, None),
        # output-layer decode diagnostics
        "pretrain_test_acc_active19_output": _acc(out_te, y_te, active_classes),
        **{f"reservoir_{k}": v for k, v in reservoir_diag.items()},
        **{f"ridge_{k}": v for k, v in ridge_info.items()
           if isinstance(v, (int, float, str, bool)) or v is None},
    }
    print(f"=== {method} ===  train19={metrics['pretrain_train_acc_active19']:.4f} "
          f"test19={metrics['pretrain_test_acc_active19']:.4f} "
          f"test20_diag={metrics['pretrain_test_acc_full20_diagnostic']:.4f} "
          f"continual_before={metrics['continual_test_acc_before_cil']:.4f}")
    print(f"    [linear] test19={metrics['pretrain_test_acc_active19_linear']:.4f} "
          f"continual_before={metrics['continual_test_acc_before_cil_linear']:.4f}")

    # ---- raster + firing histogram ----
    if wandb_run is not None:
        wandb_utils.log_image(wandb_run, "hidden_raster",
                              plots.plot_hidden_raster(
                                  diag_trace[0], max_neurons=int(cfg["raster_max_neurons"]),
                                  title=f"hidden raster ({method})"))
        wandb_utils.log_image(wandb_run, "hidden_firing_hist",
                              plots.plot_firing_rate_histogram(
                                  model.reservoir.per_neuron_firing_rate(diag_trace)))

    # ---- audits ----
    ok = True
    ok &= print_audit("model", audit_model_shapes(model, manifest))
    ok &= print_audit("active_classes", audit_active_classes(active_classes, removed_class))
    ok &= print_audit("removed_column", audit_removed_column_zero(model, removed_class))

    # ---- save checkpoint / config / metrics ----
    arch = architecture_dict(model, tau_mem_ms=float(cfg["tau_mem_ms"]),
                             tau_syn_ms=float(cfg["tau_syn_ms"]),
                             tau_out_mem_ms=float(cfg["tau_out_mem_ms"]),
                             weight_scale=float(cfg["weight_scale"]),
                             dt_ms=float(manifest["dataset_binning_ms"]))
    ckpt = build_checkpoint(
        model, arch=arch, manifest=manifest, config=cfg, active_classes=active_classes,
        removed_class=removed_class, training_method=method, metrics=metrics,
        ridge_lambda=float(cfg["ridge_lambda"]) if method == "ridge" else None,
        ridge_weighting=cfg["ridge_weighting"] if method == "ridge" else None,
        removed_class_init_policy="zero")
    torch.save(ckpt, os.path.join(ckpt_dir, "pretrained_model.pt"))
    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    if wandb_run is not None:
        loggable = {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
        wandb_utils.log(wandb_run, loggable)
        wandb_utils.set_summary(wandb_run, metrics)
        wandb_utils.finish(wandb_run)

    print(f"\nSaved run to {os.path.abspath(run_dir)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
