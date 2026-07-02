#!/usr/bin/env python3
"""Class-incremental replay sweep: learn the removed class on top of a pretrained
model across a grid of PER-OLD-CLASS replay ratios (and seeds).

Example::

    python refactor/scripts/cil_sweep.py \
      --config refactor/configs/cil_default.yaml \
      --pretrained-checkpoint outputs/.../checkpoints/pretrained_model.pt \
      --cil-training-method ridge --max-replay-percent 100 --min-replay-percent 0 \
      --num-replays 11 --ridge-weighting normalized_inverse_class_count
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from _common import add_package_to_path

add_package_to_path()

from shd_cl import NUM_CLASSES  # noqa: E402
from shd_cl.data.io import load_npz_split, write_json, _json_default  # noqa: E402
from shd_cl.training.bptt import BpttConfig  # noqa: E402
from shd_cl.training.cil import (CILConfig, evaluate_cil, precompute_pool_sums,  # noqa: E402
                                 run_one_cil)
from shd_cl.training.replay import replay_percentages  # noqa: E402
from shd_cl.logging import wandb_utils  # noqa: E402
from shd_cl.utils.checkpointing import load_checkpoint_model  # noqa: E402
from shd_cl.utils.config import apply_overrides, load_config  # noqa: E402
from shd_cl.utils.determinism import set_determinism  # noqa: E402
from shd_cl.utils.device import resolve_device  # noqa: E402


OVERRIDABLE = [
    "pretrained_checkpoint", "output_dir", "cil_training_method", "ridge_lambda",
    "ridge_weighting", "max_replay_percent", "min_replay_percent", "num_replays",
    "n_new_cap", "replay_replacement_policy", "replay_source", "new_class_source",
    "seed", "n_seeds", "nb_epochs", "batch_size", "lr", "optimizer", "grad_clip",
    "device", "limit", "wandb_mode", "wandb_name",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    for key in OVERRIDABLE:
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, default=None)
    return p.parse_args()


def _scalar_items(row: dict):
    for k, v in row.items():
        if isinstance(v, dict):
            continue
        yield k, v


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args, OVERRIDABLE)
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = resolve_device(cfg["device"])
    method = cfg["cil_training_method"]
    limit = int(cfg.get("limit", 0) or 0)

    ckpt_path = cfg["pretrained_checkpoint"]
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise SystemExit(f"pretrained_checkpoint not found: {ckpt_path!r}")
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    dataset_dir = os.path.join(run_dir, "dataset")
    output_dir = cfg["output_dir"]
    ckpt_out_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    removed_class = int(ckpt["removed_class"])
    active_classes = [int(c) for c in ckpt["active_classes"]]
    model = load_checkpoint_model(ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded {ckpt.get('training_method')} checkpoint | removed_class={removed_class} "
          f"| output_layer={model.output_layer_type} | CIL method={method}")

    # ---- data splits ----
    replay_X, replay_y, _ = load_npz_split(
        os.path.join(dataset_dir, f"{cfg['replay_source']}.npz"), limit)
    new_X, new_y, _ = load_npz_split(
        os.path.join(dataset_dir, f"{cfg['new_class_source']}.npz"), limit)
    old_test_X, old_test_y, _ = load_npz_split(
        os.path.join(dataset_dir, "pretrain_test.npz"), limit)
    new_test_X, new_test_y, _ = load_npz_split(
        os.path.join(dataset_dir, "continual_test.npz"), limit)

    # ---- leakage assertions ----
    assert removed_class not in set(replay_y.tolist()), "replay pool leaks removed class"
    assert set(new_y.tolist()) <= {removed_class}, "new pool is not purely the removed class"
    assert removed_class not in set(old_test_y.tolist()), "pretrain_test leaks removed class"
    assert set(new_test_y.tolist()) <= {removed_class}, "continual_test has non-removed class"
    print(f"replay_pool={len(replay_y)} new_train={len(new_y)} "
          f"old_test={len(old_test_y)} new_test={len(new_test_y)}")

    # ---- CIL config ----
    cil_cfg = CILConfig(
        cil_training_method=method, ridge_lambda=float(cfg["ridge_lambda"]),
        ridge_weighting=cfg["ridge_weighting"], n_new_cap=int(cfg["n_new_cap"]),
        replay_replacement_policy=cfg["replay_replacement_policy"],
        batch_size=int(cfg["batch_size"]),
        bptt=BpttConfig(method=method if method in ("fullbptt", "lastbptt") else "fullbptt",
                        nb_epochs=int(cfg["nb_epochs"]), batch_size=int(cfg["batch_size"]),
                        lr=float(cfg["lr"]), optimizer=cfg["optimizer"],
                        grad_clip=float(cfg["grad_clip"]), seed=seed))

    # ---- precompute frozen-reservoir feature caches (ridge only) ----
    new_pool_sums = replay_pool_sums = None
    if method == "ridge":
        new_pool_sums = precompute_pool_sums(model, new_X, int(cfg["batch_size"]), device)
        replay_pool_sums = precompute_pool_sums(model, replay_X, int(cfg["batch_size"]), device)

    # ---- pretrained snapshot + constant before-metrics ----
    pretrained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before_metrics = evaluate_cil(model, old_test_X, old_test_y, new_test_X, new_test_y,
                                  method=method, removed_class=removed_class,
                                  batch_size=int(cfg["batch_size"]), device=device)
    print(f"[before CIL] old={before_metrics['old_acc']:.3f} "
          f"new={before_metrics['new_acc']:.3f} total={before_metrics['total_acc']:.3f} "
          f"balanced(old,new)={before_metrics['two_group_balanced_acc']:.3f}")

    percents = replay_percentages(float(cfg["max_replay_percent"]),
                                  float(cfg["min_replay_percent"]), int(cfg["num_replays"]))
    ratios = [p / 100.0 for p in percents]
    seeds = [seed + i for i in range(int(cfg["n_seeds"]))]
    print(f"ratios(%)={[round(p, 1) for p in percents]}  seeds={seeds}")

    wandb_run = wandb_utils.init_wandb(
        mode=cfg["wandb_mode"], project=cfg["wandb_project"],
        name=cfg.get("wandb_name"), entity=cfg.get("wandb_entity"),
        config={**cfg, "removed_class": removed_class, "active_classes": active_classes},
        tags=["shd", "class-incremental", method, f"removed{removed_class}"])

    # ---- sweep ----
    csv_path = os.path.join(output_dir, "replay_sweep_results.csv")
    jsonl_path = os.path.join(output_dir, "replay_sweep_results.jsonl")
    rows = []
    fieldnames = None
    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fjsonl:
        writer = None
        for ri, ratio in enumerate(ratios):
            for si, sd in enumerate(seeds):
                rng = np.random.default_rng(np.random.SeedSequence([seed, ri, si]))
                row = run_one_cil(
                    model, pretrained_state, cil_cfg, replay_ratio=float(ratio), rng=rng,
                    new_X=new_X, new_y=new_y.numpy(), replay_X=replay_X,
                    replay_y=replay_y.numpy(), old_test_X=old_test_X, old_test_y=old_test_y,
                    new_test_X=new_test_X, new_test_y=new_test_y, removed_class=removed_class,
                    active_classes=active_classes, new_pool_sums=new_pool_sums,
                    replay_pool_sums=replay_pool_sums, before_metrics=before_metrics,
                    device=device)
                row = {"ratio": float(ratio), "ratio_percent": float(ratio * 100.0),
                       "ratio_index": ri, "seed": int(sd), "seed_index": si,
                       "cil_method": method, "removed_class": removed_class, **row}

                scalar = dict(_scalar_items(row))
                pc_after = row.get("per_class_acc_after", {})
                for c in range(NUM_CLASSES):
                    v = pc_after.get(c)
                    scalar[f"pc_after_{c:02d}"] = "" if v is None else v
                if writer is None:
                    fieldnames = list(scalar.keys())
                    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow({k: scalar.get(k, "") for k in fieldnames})
                fjsonl.write(json.dumps(row, default=_json_default) + "\n")
                rows.append(row)

                _save_run_checkpoint(ckpt_out_dir, ckpt, model, ratio, sd, cfg, method)
                if wandb_run is not None:
                    wandb_utils.log(wandb_run, {k: v for k, v in scalar.items()
                                                if isinstance(v, (int, float))})
                print(f"r={ratio:4.2f} seed={sd} | old {row['old_acc_before']:.3f}"
                      f"->{row['old_acc_after']:.3f} new {row['new_acc_before']:.3f}"
                      f"->{row['new_acc_after']:.3f} bal(old,new)="
                      f"{row['two_group_balanced_acc_after']:.3f} "
                      f"forget={row['forgetting_old']:+.3f} "
                      f"(n_new={row['n_new_samples']} m/cls={row['m_old_per_class']})")

    summary = _summarize(rows, ratios, cfg, before_metrics)
    write_json(os.path.join(output_dir, "replay_sweep_summary.json"), summary)
    write_json(os.path.join(output_dir, "config.json"), cfg)
    print(f"\nSaved sweep to {os.path.abspath(output_dir)} ({len(rows)} rows)")
    wandb_utils.finish(wandb_run)
    return 0


def _save_run_checkpoint(ckpt_out_dir, base_ckpt, model, ratio, seed, cfg, method):
    out = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_class": type(model).__name__,
        "architecture": base_ckpt["architecture"],
        "active_classes": base_ckpt["active_classes"],
        "removed_class": base_ckpt["removed_class"],
        "pretrained_checkpoint": cfg["pretrained_checkpoint"],
        "cil_training_method": method,
        "replay_ratio": float(ratio),
        "ridge_lambda": float(cfg["ridge_lambda"]),
        "ridge_weighting": cfg["ridge_weighting"],
        "seed": int(seed),
    }
    fname = f"final_ratio_{round(float(ratio) * 100):03d}_seed_{int(seed):03d}.pt"
    torch.save(out, os.path.join(ckpt_out_dir, fname))


def _summarize(rows, ratios, cfg, before_metrics) -> dict:
    metrics = ["old_acc_after", "new_acc_after", "total_acc_after",
               "balanced_acc_after", "two_group_balanced_acc_after",
               "forgetting_old", "learning_new", "total_delta",
               "old_acc_after_linear", "new_acc_after_linear"]
    per_ratio = []
    for ri, ratio in enumerate(ratios):
        sel = [r for r in rows if r["ratio_index"] == ri]
        entry = {"ratio": float(ratio), "ratio_percent": float(ratio * 100.0),
                 "ratio_index": ri, "n_seeds": len(sel),
                 "n_new_samples": sel[0]["n_new_samples"] if sel else None,
                 "m_old_per_class": sel[0]["m_old_per_class"] if sel else None,
                 "total_cil_train": sel[0]["total_cil_train"] if sel else None}
        for m in metrics:
            vals = np.array([r[m] for r in sel if m in r], dtype=np.float64)
            entry[f"{m}_mean"] = float(np.nanmean(vals)) if len(vals) else None
            entry[f"{m}_std"] = float(np.nanstd(vals)) if len(vals) else None
        per_ratio.append(entry)
    return {
        "cil_training_method": cfg["cil_training_method"],
        "ridge_lambda": float(cfg["ridge_lambda"]),
        "ridge_weighting": cfg["ridge_weighting"],
        "replay_semantics": "per_old_class: m_old_per_class = round(r * n_new)",
        "replay_replacement_policy": cfg["replay_replacement_policy"],
        "before_cil": {k: v for k, v in before_metrics.items() if not isinstance(v, dict)},
        "ratios": [float(r) for r in ratios],
        "per_ratio": per_ratio,
    }


if __name__ == "__main__":
    raise SystemExit(main())
