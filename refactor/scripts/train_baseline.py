#!/usr/bin/env python3
"""Baseline: preprocess all 20 classes, then train + evaluate a 20-way classifier.

Example::

    python refactor/scripts/train_baseline.py \
      --config refactor/configs/baseline_default.yaml
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
from shd_cl.evaluation.predict import (argmax_predict, collect_both_logits,  # noqa: E402
                                       collect_spike_sums)
from shd_cl.models.snn import build_model_from_manifest  # noqa: E402
from shd_cl.training.bptt import BpttConfig, train_bptt  # noqa: E402
from shd_cl.training.ridge import apply_readout, fit_readout  # noqa: E402
from shd_cl.logging import plots, wandb_utils  # noqa: E402
from shd_cl.utils.audit import audit_model_shapes, print_audit  # noqa: E402
from shd_cl.utils.checkpointing import architecture_dict, build_checkpoint  # noqa: E402
from shd_cl.utils.config import apply_overrides, load_config  # noqa: E402
from shd_cl.utils.determinism import set_determinism  # noqa: E402
from shd_cl.utils.device import resolve_device  # noqa: E402


OVERRIDABLE = [
    "run_name", "output_root", "dataset_binning_ms", "dataset_max_seconds",
    "n_compressed_channels", "channel_compression_method", "training_method",
    "ridge_lambda", "ridge_weighting", "nb_hidden", "nb_epochs", "batch_size", "lr",
    "optimizer", "grad_clip", "output_layer_type", "output_threshold",
    "logit_source", "leaky_readout", "device", "wandb_mode", "wandb_name", "limit",
    "seed",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    for key in OVERRIDABLE:
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, default=None)
    return p.parse_args()


def _acc(logits, y):
    pred = argmax_predict(logits, None)
    yt = y.cpu().numpy().astype(np.int64)
    return float("nan") if len(yt) == 0 else float(np.mean(pred == yt))


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args, OVERRIDABLE)
    cfg["experiment_regime"] = "baseline_20_class"
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = resolve_device(cfg["device"])
    method = cfg["training_method"]
    limit = int(cfg.get("limit", 0) or 0)
    all_classes = list(range(NUM_CLASSES))
    print(f"device={device}  method={method}  regime=baseline_20_class")

    run_dir = os.path.join(cfg["output_root"], cfg["run_name"])
    dataset_dir = os.path.join(run_dir, "dataset")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    manifest = ensure_dataset(cfg, run_dir, "baseline_20_class", None)
    X_tr, y_tr, _ = load_npz_split(os.path.join(dataset_dir, "train.npz"), limit)
    X_te, y_te, _ = load_npz_split(os.path.join(dataset_dir, "test.npz"), limit)
    print(f"train X={tuple(X_tr.shape)}  test X={tuple(X_te.shape)}")

    model = build_model_from_manifest(
        manifest, nb_outputs=int(cfg["nb_outputs"]), tau_mem_ms=float(cfg["tau_mem_ms"]),
        tau_syn_ms=float(cfg["tau_syn_ms"]), tau_out_mem_ms=float(cfg["tau_out_mem_ms"]),
        threshold=float(cfg["threshold"]), weight_scale=float(cfg["weight_scale"]),
        surrogate_slope=float(cfg["surrogate_slope"]), nb_hidden=int(cfg["nb_hidden"]),
        output_layer_type=cfg["output_layer_type"],
        output_threshold=float(cfg["output_threshold"]), logit_source=cfg["logit_source"],
        leaky_readout=cfg["leaky_readout"],
        target_spectral_radius=float(cfg["target_spectral_radius"])).to(device)

    sane_n = min(int(cfg["batch_size"]), X_tr.shape[0])
    with torch.no_grad():
        diag_trace = model.hidden_trace(X_tr[:sane_n].to(device))
    fire = model.reservoir.firing_diagnostics(diag_trace)
    print(f"[reservoir] rho={model.reservoir.renormalized_spectral_radius:.3f} "
          f"firing={fire['mean_hidden_firing_rate']:.4f}")

    wandb_run = wandb_utils.init_wandb(
        mode=cfg["wandb_mode"], project=cfg["wandb_project"],
        name=cfg.get("wandb_name") or cfg["run_name"], entity=cfg.get("wandb_entity"),
        config=cfg, tags=["shd", "baseline", method, cfg["output_layer_type"]])

    t0 = time.time()
    ridge_info = {}
    if method == "ridge":
        feats = collect_spike_sums(model, X_tr, int(cfg["batch_size"]), device)
        W_full, ridge_info = fit_readout(
            feats, y_tr.numpy(), columns=all_classes, nb_outputs=NUM_CLASSES,
            lam=float(cfg["ridge_lambda"]), weighting=cfg["ridge_weighting"])
        apply_readout(model, W_full)
    else:
        bptt_cfg = BpttConfig(method=method, nb_epochs=int(cfg["nb_epochs"]),
                              batch_size=int(cfg["batch_size"]), lr=float(cfg["lr"]),
                              optimizer=cfg["optimizer"], grad_clip=float(cfg["grad_clip"]),
                              seed=seed)
        train_bptt(model, X_tr, y_tr, active_classes=all_classes, num_classes=NUM_CLASSES,
                   cfg=bptt_cfg, device=device, removed_class=None, X_te=X_te, y_te=y_te,
                   on_epoch=lambda rec: wandb_utils.log(wandb_run, rec))
    train_seconds = time.time() - t0

    out_tr, lin_tr = collect_both_logits(model, X_tr, int(cfg["batch_size"]), device)
    out_te, lin_te = collect_both_logits(model, X_te, int(cfg["batch_size"]), device)
    primary_out = (method != "ridge")
    P_tr, P_te = (out_tr, out_te) if primary_out else (lin_tr, lin_te)
    metrics = {
        "training_method": method,
        "primary_readout": "output" if primary_out else "linear",
        "train_seconds": train_seconds,
        "train_acc": _acc(P_tr, y_tr),
        "test_acc": _acc(P_te, y_te),
        "train_acc_linear": _acc(lin_tr, y_tr),
        "test_acc_linear": _acc(lin_te, y_te),
        "test_acc_output": _acc(out_te, y_te),
        "reservoir_renormalized_spectral_radius": model.reservoir.renormalized_spectral_radius,
        **{f"reservoir_{k}": v for k, v in fire.items()},
        **{f"ridge_{k}": v for k, v in ridge_info.items()
           if isinstance(v, (int, float, str, bool)) or v is None},
    }
    print(f"=== {method} ===  train={metrics['train_acc']:.4f} test={metrics['test_acc']:.4f} "
          f"(linear test={metrics['test_acc_linear']:.4f})")

    if wandb_run is not None:
        wandb_utils.log_image(wandb_run, "hidden_raster",
                              plots.plot_hidden_raster(diag_trace[0],
                                  max_neurons=int(cfg["raster_max_neurons"]),
                                  title=f"hidden raster ({method})"))

    ok = print_audit("model", audit_model_shapes(model, manifest))

    arch = architecture_dict(model, tau_mem_ms=float(cfg["tau_mem_ms"]),
                             tau_syn_ms=float(cfg["tau_syn_ms"]),
                             tau_out_mem_ms=float(cfg["tau_out_mem_ms"]),
                             weight_scale=float(cfg["weight_scale"]),
                             dt_ms=float(manifest["dataset_binning_ms"]))
    ckpt = build_checkpoint(model, arch=arch, manifest=manifest, config=cfg,
                            active_classes=all_classes, removed_class=None,
                            training_method=method, metrics=metrics,
                            ridge_lambda=float(cfg["ridge_lambda"]) if method == "ridge" else None,
                            ridge_weighting=cfg["ridge_weighting"] if method == "ridge" else None)
    torch.save(ckpt, os.path.join(ckpt_dir, "baseline_model.pt"))
    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    if wandb_run is not None:
        wandb_utils.log(wandb_run, {k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
        wandb_utils.set_summary(wandb_run, metrics)
        wandb_utils.finish(wandb_run)
    print(f"\nSaved run to {os.path.abspath(run_dir)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
