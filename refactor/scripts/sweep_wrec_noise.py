#!/usr/bin/env python3
"""Sweep absolute Gaussian noise on ``W_rec`` and measure test accuracy.

At each noise level the same perturbation tensor is applied to every checkpoint
so comparisons use identical injected noise. Repeats across multiple seeds and
reports mean ± std.

Example::

    python refactor/scripts/sweep_wrec_noise.py \
      --config refactor/configs/baseline_default.yaml \
      --checkpoint outputs/refactor_baseline/baseline_dt14_graded35_ridge/checkpoints/baseline_model.pt \
      --checkpoint outputs/refactor_baseline/baseline_dt14_graded35_fullbptt/checkpoints/baseline_model.pt \
      --num-seeds 10 --wandb-mode online
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from _common import add_package_to_path

add_package_to_path()

from shd_cl.data.io import load_npz_split, write_json  # noqa: E402
from shd_cl.evaluation.noise_robustness import (  # noqa: E402
    apply_w_rec_noise, evaluate_test_accuracy, make_w_rec_noise, restore_w_rec,
    w_rec_std)
from shd_cl.logging import plots, wandb_utils  # noqa: E402
from shd_cl.utils.checkpointing import load_checkpoint_model  # noqa: E402
from shd_cl.utils.config import apply_overrides, load_config  # noqa: E402
from shd_cl.utils.determinism import set_determinism  # noqa: E402
from shd_cl.utils.device import resolve_device  # noqa: E402


METHOD_COLORS = {"ridge": "#1f77b4", "fullbptt": "#ff7f0e"}

OVERRIDABLE = [
    "output_dir", "batch_size", "device",
    "seed", "wandb_mode", "wandb_project", "wandb_entity", "wandb_name", "limit",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", action="append", dest="checkpoints", required=True)
    p.add_argument("--num-levels", type=int, default=50)
    p.add_argument("--max-mu", type=float, default=0.1,
                   help="max absolute noise std mu (default 0.1)")
    p.add_argument("--num-seeds", type=int, default=10,
                   help="number of noise seeds to average over (default 10)")
    for key in OVERRIDABLE:
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, default=None)
    return p.parse_args()


def _run_dir_from_checkpoint(path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(path)))


def _label_from_ckpt(ckpt: dict, path: str) -> str:
    method = ckpt.get("training_method", "unknown")
    run_name = os.path.basename(_run_dir_from_checkpoint(path))
    return f"{method} ({run_name})"


def _series_key(label: str) -> str:
    return label.split()[0]


def _aggregate(mu_values: list[float], raw: list[list[float]]) -> list[dict]:
    return [
        {
            "mu": float(mu),
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=0)),
        }
        for mu, accs in zip(mu_values, raw)
    ]


def _validate_summary(summary: dict) -> None:
    """Recompute mean/std from per-seed raw accs and assert they match aggregates."""
    for label, block in summary["results"].items():
        raw = block["per_seed_accs"]
        agg = block["aggregate"]
        for mu_i, row in enumerate(agg):
            accs = raw[mu_i]
            want_mean = float(np.mean(accs))
            want_std = float(np.std(accs, ddof=0))
            assert abs(row["mean"] - want_mean) < 1e-9, (label, mu_i, row["mean"], want_mean)
            assert abs(row["std"] - want_std) < 1e-9, (label, mu_i, row["std"], want_std)
            _, mean, lower, upper = plots.band_plot_arrays([row])
            assert abs(lower[0] - (want_mean - want_std)) < 1e-9
            assert abs(upper[0] - (want_mean + want_std)) < 1e-9
    print("[validate] aggregate mean/std and band bounds match per-seed raw accs")


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args, OVERRIDABLE)
    base_seed = int(cfg.get("seed", 42))
    set_determinism(base_seed)
    device = resolve_device(cfg.get("device", "auto"))
    batch_size = int(cfg.get("batch_size", 64))
    limit = int(cfg.get("limit", 0) or 0)
    num_levels = int(args.num_levels)
    max_mu = float(args.max_mu)
    num_seeds = int(args.num_seeds)
    mu_values = np.linspace(0.0, max_mu, num_levels).tolist()
    output_dir = cfg.get("output_dir") or os.path.join(
        cfg["output_root"], "wrec_noise_sweep")
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for ckpt_path in args.checkpoints:
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"checkpoint not found: {ckpt_path!r}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run_dir = _run_dir_from_checkpoint(ckpt_path)
        test_npz = os.path.join(run_dir, "dataset", "test.npz")
        if not os.path.isfile(test_npz):
            raise SystemExit(f"test split not found: {test_npz!r}")
        label = _label_from_ckpt(ckpt, ckpt_path)
        method = ckpt.get("training_method", "ridge")
        model = load_checkpoint_model(ckpt, device)
        X_te, y_te, _ = load_npz_split(test_npz, limit)
        entries.append({
            "label": label,
            "series_key": _series_key(label),
            "checkpoint": ckpt_path,
            "training_method": method,
            "use_linear": method == "ridge",
            "model": model,
            "backup": model.W_rec.data.clone(),
            "X_te": X_te,
            "y_te": y_te,
            "w_rec_std": w_rec_std(model),
        })
        print(f"loaded {label}  w_rec_std={entries[-1]['w_rec_std']:.6f}")

    ref_shape = entries[0]["backup"].shape
    for entry in entries[1:]:
        if entry["backup"].shape != ref_shape:
            raise SystemExit("all checkpoints must share the same W_rec shape "
                             f"(got {ref_shape} vs {entry['backup'].shape})")

    wandb_run = wandb_utils.init_wandb(
        mode=cfg.get("wandb_mode", "disabled"),
        project=cfg.get("wandb_project", "shd-refactor-baseline"),
        name=cfg.get("wandb_name") or "wrec_noise_sweep_graded35",
        entity=cfg.get("wandb_entity"),
        config={**cfg, "mu_values": mu_values, "max_mu": max_mu,
                "num_levels": num_levels, "num_seeds": num_seeds,
                "checkpoints": args.checkpoints},
        tags=["shd", "baseline", "wrec_noise", "robustness"])

    if wandb_run is not None:
        wandb_utils.define_metric(wandb_run, "noise_mu")
        for entry in entries:
            key = entry["series_key"]
            wandb_utils.define_metric(wandb_run, f"{key}/test_acc_mean", step_metric="noise_mu")
            wandb_utils.define_metric(wandb_run, f"{key}/test_acc_std", step_metric="noise_mu")
        wandb_utils.define_metric(
            wandb_run, "fullbptt_minus_ridge/test_acc_delta_mean", step_metric="noise_mu")
        wandb_utils.define_metric(
            wandb_run, "fullbptt_minus_ridge/test_acc_delta_std", step_metric="noise_mu")

    raw_by_method = {entry["series_key"]: [[] for _ in mu_values] for entry in entries}
    raw_deltas = [[] for _ in mu_values]
    per_seed_results = []

    for seed_idx in range(num_seeds):
        noise_seed = base_seed + seed_idx
        rng = torch.Generator(device=entries[0]["backup"].device)
        rng.manual_seed(noise_seed)
        seed_rows = {entry["series_key"]: [] for entry in entries}
        print(f"\n=== seed {seed_idx + 1}/{num_seeds} (noise_seed={noise_seed}) ===")

        for mu_i, mu in enumerate(mu_values):
            noise = make_w_rec_noise(entries[0]["backup"], float(mu), rng=rng)
            acc_by_method = {}
            for entry in entries:
                restore_w_rec(entry["model"], entry["backup"])
                if mu > 0:
                    apply_w_rec_noise(entry["model"], noise)
                acc = evaluate_test_accuracy(
                    entry["model"], entry["X_te"], entry["y_te"], batch_size, device,
                    use_linear_readout=entry["use_linear"])
                acc_by_method[entry["series_key"]] = acc
                raw_by_method[entry["series_key"]][mu_i].append(acc)
                seed_rows[entry["series_key"]].append({"mu": float(mu), "test_acc": acc})

            delta = None
            if "ridge" in acc_by_method and "fullbptt" in acc_by_method:
                delta = acc_by_method["fullbptt"] - acc_by_method["ridge"]
                raw_deltas[mu_i].append(delta)

        per_seed_results.append({"noise_seed": noise_seed, "by_method": seed_rows})

    for entry in entries:
        restore_w_rec(entry["model"], entry["backup"])

    stats_by_label = {
        entry["label"]: _aggregate(mu_values, raw_by_method[entry["series_key"]])
        for entry in entries
    }
    stats_by_method = {
        entry["series_key"]: _aggregate(mu_values, raw_by_method[entry["series_key"]])
        for entry in entries
    }
    delta_stats = _aggregate(mu_values, raw_deltas) if raw_deltas[0] else []

    print("\n=== mean ± std over seeds ===")
    for mu_i, mu in enumerate(mu_values):
        parts = []
        for entry in entries:
            key = entry["series_key"]
            s = stats_by_method[key][mu_i]
            parts.append(f"{key}={s['mean']:.4f}±{s['std']:.4f}")
        if delta_stats:
            d = delta_stats[mu_i]
            parts.append(f"delta={d['mean']:+.4f}±{d['std']:.4f}")
        print(f"  mu={mu:.4f}  " + "  ".join(parts))

        if wandb_run is not None:
            payload = {"noise_mu": float(mu)}
            for entry in entries:
                key = entry["series_key"]
                s = stats_by_method[key][mu_i]
                payload[f"{key}/test_acc_mean"] = s["mean"]
                payload[f"{key}/test_acc_std"] = s["std"]
            if delta_stats:
                payload["fullbptt_minus_ridge/test_acc_delta_mean"] = delta_stats[mu_i]["mean"]
                payload["fullbptt_minus_ridge/test_acc_delta_std"] = delta_stats[mu_i]["std"]
            wandb_utils.log(wandb_run, payload)

    label_colors = {
        entry["label"]: METHOD_COLORS.get(entry["series_key"], "#333333")
        for entry in entries
    }

    mpl_robust = plots.plot_noise_robustness_band(
        stats_by_label, title="W_rec noise robustness (mean ± std, graded-35)",
        colors=label_colors)
    if wandb_run is not None:
        wandb_utils.log_image(wandb_run, "wrec_noise_robustness", mpl_robust)
    robust_paths = plots.save_fig_publication(
        mpl_robust, os.path.join(output_dir, "wrec_noise_robustness"))
    print(f"[plots] saved {', '.join(os.path.basename(p) for p in robust_paths)}")

    if delta_stats:
        mpl_delta = plots.plot_noise_acc_delta_band(
            delta_stats, title="fullbptt − ridge (mean ± std, graded-35)")
        if wandb_run is not None:
            wandb_utils.log_image(wandb_run, "wrec_noise_acc_delta", mpl_delta)
        delta_paths = plots.save_fig_publication(
            mpl_delta, os.path.join(output_dir, "wrec_noise_acc_delta"))
        print(f"[plots] saved {', '.join(os.path.basename(p) for p in delta_paths)}")

    all_results = {}
    for entry in entries:
        key = entry["series_key"]
        all_results[entry["label"]] = {
            "checkpoint": entry["checkpoint"],
            "training_method": entry["training_method"],
            "w_rec_std": entry["w_rec_std"],
            "aggregate": stats_by_method[key],
            "per_seed_accs": raw_by_method[key],
        }

    summary = {
        "mu_values": mu_values,
        "max_mu": max_mu,
        "num_levels": num_levels,
        "num_seeds": num_seeds,
        "base_seed": base_seed,
        "shared_noise": True,
        "results": all_results,
        "delta_fullbptt_minus_ridge": delta_stats,
        "per_seed_results": per_seed_results,
    }
    write_json(os.path.join(output_dir, "wrec_noise_sweep.json"), summary)
    _validate_summary(summary)

    if wandb_run is not None:
        for entry in entries:
            key = entry["series_key"]
            rows = stats_by_method[key]
            wandb_utils.set_summary(wandb_run, {
                f"{key}/baseline_test_acc_mean": rows[0]["mean"],
                f"{key}/noisy_test_acc_mean_at_max_mu": rows[-1]["mean"],
            })
        if delta_stats:
            wandb_utils.set_summary(wandb_run, {
                "fullbptt_minus_ridge/baseline_test_acc_delta_mean": delta_stats[0]["mean"],
                "fullbptt_minus_ridge/noisy_test_acc_delta_mean_at_max_mu": delta_stats[-1]["mean"],
            })
        wandb_utils.finish(wandb_run)

    print(f"\nSaved sweep to {os.path.abspath(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
