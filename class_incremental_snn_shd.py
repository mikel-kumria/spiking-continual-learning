#!/usr/bin/env python3
"""Stage 2 of the SHD continual-learning pipeline: class-incremental RLS sweep.

Loads a run folder produced by ``pretrain_snn_shd.py`` and incrementally learns
the held-out (``removed_class``) NEW class with an online closed-form readout
update -- Recursive Least Squares (RLS) -- while the reservoir (``W_in``,
``W_rec``) stays FROZEN. Only ``W_out`` is adapted, on the same feature the ridge
pretraining used::

    Phi = mean_t(hidden_spikes)   # [B, H]
    logits = Phi @ W_out          # [B, 20]

Because the reservoir is frozen, every sample's ``Phi`` is fixed: we extract it
ONCE per split and run the whole sweep in feature space (``logits = Phi @ W``),
which is exact and fast. "Reload the pretrained checkpoint fresh" for each
(ratio, seed) therefore means: reset ``W = pretrained W_out`` and ``P = I/delta``.

Replay (rehearsal, NOT rehearsal-free)
--------------------------------------
Replay uses RAW old-class samples (their frozen features) mixed into the new
stream -- this is *rehearsal*. Each row records ``is_rehearsal=True`` and the
replay source. Two ratio semantics (``--replay-ratio-mode``):

* ``additive`` (default): all ``N`` new-class train samples are always used;
  ``round(r*N)`` old-class replay samples are added on top. Replay never shrinks
  the new-class data.
* ``fixed_budget``: total stream size is fixed at ``N``; ``round(r*N)`` old replay
  and ``round((1-r)*N)`` new samples. (At ``r=1`` no new samples are seen -- the
  new class is not learned; this is the documented downside.)

Output (``--output-dir``)::

    outputs/shd_class_incremental/<run_name>/
      config.json
      inherited_pretrain_config.json
      inherited_preprocessing_manifest.json
      replay_sweep_results.csv
      replay_sweep_results.jsonl
      replay_sweep_summary.json
      checkpoints/final_ratio_<ppp>_seed_<sss>.pt

Example::

    python class_incremental_snn_shd.py \
      --pretrain-run-dir outputs/shd_pretraining/rc10_dt14_or70_removed10_ridge \
      --output-dir outputs/shd_class_incremental/rc10_..._rls_sweep \
      --rls-delta 1.0 --rls-forgetting-factor 1.0 \
      --replay-runs 11 --replay-start-percent 0 --replay-end-percent 100 \
      --replay-ratio-mode fixed_budget --batch-size 1 --wandb-mode disabled
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import snn_shd_common as C

NUM_CLASSES = C.NUM_CLASSES


# =============================================================================
# Feature-space helpers
# =============================================================================


def feat_logits(Phi: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """``Phi [N,H] @ W [H,20] -> [N,20]`` in float64 (matches RLS algebra)."""
    return Phi.to(torch.float64) @ W.to(torch.float64)


def feat_predict(Phi: torch.Tensor, W: torch.Tensor) -> np.ndarray:
    if Phi.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return feat_logits(Phi, W).argmax(1).cpu().numpy().astype(np.int64)


def feat_accuracy(Phi: torch.Tensor, y: np.ndarray, W: torch.Tensor) -> float:
    return C.accuracy(y, feat_predict(Phi, W))


# =============================================================================
# Stream construction (deterministic, per (ratio, seed))
# =============================================================================


def build_stream(Phi_new: torch.Tensor, y_new: np.ndarray,
                 Phi_replay_pool: torch.Tensor, y_replay_pool: np.ndarray,
                 ratio: float, mode: str, rng: np.random.Generator
                 ) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    """Return ``(Phi_stream, y_stream, n_new, n_replay)`` for one replay ratio.

    The new-class pool size ``N`` is the reference for both modes.
    """
    N = Phi_new.shape[0]
    if N == 0:
        raise ValueError("new-class stream is empty (continual_train has no samples)")
    n_replay = int(round(ratio * N))
    if mode == "additive":
        n_new = N
    elif mode == "fixed_budget":
        n_new = int(round((1.0 - ratio) * N))
    else:
        raise ValueError(f"unknown replay-ratio-mode {mode!r}")

    # new-class samples
    if n_new >= N:
        new_idx = np.arange(N)
    else:
        new_idx = rng.choice(N, size=n_new, replace=False)
    # old-class replay samples
    M = Phi_replay_pool.shape[0]
    if n_replay == 0 or M == 0:
        rep_idx = np.zeros((0,), dtype=np.int64)
        n_replay = 0 if M == 0 else n_replay
    elif n_replay <= M:
        rep_idx = rng.choice(M, size=n_replay, replace=False)
    else:  # need more replay than available -> sample with replacement
        rep_idx = rng.choice(M, size=n_replay, replace=True)

    Phi_stream = torch.cat([Phi_new[torch.from_numpy(new_idx).long()],
                            Phi_replay_pool[torch.from_numpy(rep_idx).long()]], dim=0)
    y_stream = np.concatenate([y_new[new_idx], y_replay_pool[rep_idx]]).astype(np.int64)

    # deterministic shuffle of the interleaved stream order
    order = rng.permutation(Phi_stream.shape[0])
    return Phi_stream[torch.from_numpy(order).long()], y_stream[order], len(new_idx), len(rep_idx)


# =============================================================================
# One (ratio, seed) incremental run
# =============================================================================


def run_one(W_pre: torch.Tensor, *, Phi_old_test, y_old_test, Phi_new_test,
            y_new_test, Phi_replay_pool, y_replay_pool, Phi_new_train, y_new_train,
            ratio: float, mode: str, delta: float, lam: float,
            rng: np.random.Generator) -> dict:
    """Reset W from the pretrained read-out, run RLS over the stream, score."""
    Phi_comb_test = torch.cat([Phi_old_test, Phi_new_test], dim=0)
    y_comb_test = np.concatenate([y_old_test, y_new_test]).astype(np.int64)

    # ---- evaluate BEFORE incremental learning ----
    old_before = feat_accuracy(Phi_old_test, y_old_test, W_pre)
    new_before = feat_accuracy(Phi_new_test, y_new_test, W_pre)
    total_before = feat_accuracy(Phi_comb_test, y_comb_test, W_pre)
    pc_before = C.per_class_accuracy(y_comb_test, feat_predict(Phi_comb_test, W_pre))

    # ---- build stream + RLS (only W_out changes) ----
    Phi_stream, y_stream, n_new, n_replay = build_stream(
        Phi_new_train, y_new_train, Phi_replay_pool, y_replay_pool, ratio, mode, rng)
    rls = C.RLS(W_pre.clone(), delta=delta, lambda_forgetting=lam)
    rls.run_stream(Phi_stream, C.one_hot(y_stream))
    W_post = rls.W  # float64 [H, 20]

    # ---- evaluate AFTER ----
    old_after = feat_accuracy(Phi_old_test, y_old_test, W_post)
    new_after = feat_accuracy(Phi_new_test, y_new_test, W_post)
    total_after = feat_accuracy(Phi_comb_test, y_comb_test, W_post)
    pc_after = C.per_class_accuracy(y_comb_test, feat_predict(Phi_comb_test, W_post))

    return {
        "n_new": int(n_new), "n_replay": int(n_replay),
        "n_stream": int(Phi_stream.shape[0]),
        "old_acc_before": old_before, "old_acc_after": old_after,
        "new_acc_before": new_before, "new_acc_after": new_after,
        "total_acc_before": total_before, "total_acc_after": total_after,
        "forgetting_old": old_before - old_after,
        "old_acc_delta": old_after - old_before,
        "new_acc_delta": new_after - new_before,
        "per_class_acc_before": pc_before,
        "per_class_acc_after": pc_after,
        "W_post": W_post,
    }


# =============================================================================
# Orchestration
# =============================================================================


def load_split_features(model, dataset_dir, name, batch_size, device, limit):
    X, y, _ = C.load_npz_split(os.path.join(dataset_dir, f"{name}.npz"), limit)
    Phi = C.collect_features(model, X, batch_size, device)
    return Phi, y.numpy().astype(np.int64)


def main() -> int:
    args = parse_args()
    C.set_determinism(args.seed)
    device = C.resolve_device(args.device)

    run_dir = args.pretrain_run_dir
    dataset_dir = os.path.join(run_dir, "dataset")
    ckpt_path = os.path.join(run_dir, "checkpoints", args.checkpoint_name)
    for p in (dataset_dir, ckpt_path):
        if not os.path.exists(p):
            raise SystemExit(f"expected path not found in pretrain run dir: {p}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_out_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_out_dir, exist_ok=True)

    # ---- inherit pretrain provenance ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    removed_class = int(ckpt["removed_class"])
    active_classes = [int(c) for c in ckpt["active_classes"]]
    model = C.load_checkpoint_model(ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # reservoir + readout frozen (RLS works on a copy)
    W_pre = model.W_out.detach().cpu().double().clone()  # pretrained read-out [H,20]

    pre_cfg_path = os.path.join(run_dir, "config.json")
    pre_man_path = os.path.join(run_dir, "preprocessing_manifest.json")
    if os.path.isfile(pre_cfg_path):
        C.write_json(os.path.join(args.output_dir, "inherited_pretrain_config.json"),
                     C.read_json(pre_cfg_path))
    manifest = C.read_json(pre_man_path) if os.path.isfile(pre_man_path) else {}
    if manifest:
        C.write_json(os.path.join(args.output_dir, "inherited_preprocessing_manifest.json"),
                     manifest)

    # ---- features (extracted once; reservoir is frozen) ----
    Phi_old_test, y_old_test = load_split_features(
        model, dataset_dir, "pretrain_test", args.batch_size, device, args.limit)
    Phi_new_test, y_new_test = load_split_features(
        model, dataset_dir, "continual_test", args.batch_size, device, args.limit)
    Phi_replay_pool, y_replay_pool = load_split_features(
        model, dataset_dir, args.replay_source, args.batch_size, device, args.limit)
    Phi_new_train, y_new_train = load_split_features(
        model, dataset_dir, args.new_class_source, args.batch_size, device, args.limit)

    # ---- assertions (data/model contracts) ----
    # Channel count is also re-checked inside hidden_spikes during feature
    # extraction above; the manifest gives a cheap, early, explicit error.
    if manifest.get("nb_inputs") is not None:
        assert model.nb_inputs == int(manifest["nb_inputs"]), \
            f"model nb_inputs {model.nb_inputs} != dataset nb_inputs {manifest['nb_inputs']}"
    assert model.nb_outputs >= NUM_CLASSES, f"nb_outputs must be >= {NUM_CLASSES}"
    assert removed_class not in set(y_old_test.tolist()), \
        "pretrain_test leaks the removed class"
    assert set(y_new_test.tolist()) <= {removed_class}, \
        "continual_test contains a non-removed class"
    assert set(y_replay_pool.tolist()).isdisjoint({removed_class}), \
        f"replay source {args.replay_source} leaks the removed class"
    assert set(y_new_train.tolist()) <= {removed_class}, \
        f"new-class source {args.new_class_source} is not purely the removed class"
    print(f"removed_class={removed_class}  H={model.nb_hidden}  "
          f"old_test={len(y_old_test)} new_test={len(y_new_test)} "
          f"replay_pool={len(y_replay_pool)} new_train={len(y_new_train)}")

    # ---- ratios (exact floats) + seeds ----
    ratios = (np.linspace(args.replay_start_percent, args.replay_end_percent,
                          args.replay_runs) / 100.0)
    seeds = [args.seed + i for i in range(args.n_seeds)]
    print(f"ratios={[round(float(r), 4) for r in ratios]}  seeds={seeds}  "
          f"mode={args.replay_ratio_mode}")

    wandb_run = init_wandb(args, removed_class, active_classes)

    # ---- sweep ----
    csv_path = os.path.join(args.output_dir, "replay_sweep_results.csv")
    jsonl_path = os.path.join(args.output_dir, "replay_sweep_results.jsonl")
    fieldnames = [
        "ratio", "ratio_percent", "ratio_index", "seed", "seed_index",
        "replay_mode", "replay_source", "new_class_source", "removed_class",
        "rls_delta", "rls_forgetting_factor", "is_rehearsal",
        "n_new", "n_replay", "n_stream",
        "old_acc_before", "old_acc_after", "new_acc_before", "new_acc_after",
        "total_acc_before", "total_acc_after",
        "forgetting_old", "old_acc_delta", "new_acc_delta",
    ] + [f"pc_after_{c:02d}" for c in range(NUM_CLASSES)]

    rows: List[dict] = []
    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for ri, ratio in enumerate(ratios):
            for si, seed in enumerate(seeds):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, ri, si]))
                res = run_one(
                    W_pre, Phi_old_test=Phi_old_test, y_old_test=y_old_test,
                    Phi_new_test=Phi_new_test, y_new_test=y_new_test,
                    Phi_replay_pool=Phi_replay_pool, y_replay_pool=y_replay_pool,
                    Phi_new_train=Phi_new_train, y_new_train=y_new_train,
                    ratio=float(ratio), mode=args.replay_ratio_mode,
                    delta=args.rls_delta, lam=args.rls_forgetting_factor, rng=rng)

                base = {
                    "ratio": float(ratio), "ratio_percent": float(ratio * 100.0),
                    "ratio_index": ri, "seed": int(seed), "seed_index": si,
                    "replay_mode": args.replay_ratio_mode,
                    "replay_source": args.replay_source,
                    "new_class_source": args.new_class_source,
                    "removed_class": removed_class,
                    "rls_delta": args.rls_delta,
                    "rls_forgetting_factor": args.rls_forgetting_factor,
                    "is_rehearsal": True,
                }
                scalar = {k: v for k, v in res.items()
                          if k not in ("per_class_acc_before", "per_class_acc_after", "W_post")}
                csv_row = {**base, **scalar}
                for c in range(NUM_CLASSES):
                    v = res["per_class_acc_after"][c]
                    csv_row[f"pc_after_{c:02d}"] = "" if v is None else v
                writer.writerow(csv_row)

                jrow = {**base, **scalar,
                        "per_class_acc_before": res["per_class_acc_before"],
                        "per_class_acc_after": res["per_class_acc_after"]}
                fjsonl.write(json.dumps(jrow, default=C._json_default) + "\n")
                rows.append(jrow)

                # save per (ratio, seed) checkpoint with the updated read-out
                _save_final_checkpoint(ckpt_out_dir, ckpt, model, res["W_post"],
                                       ratio, seed, args)
                if wandb_run is not None:
                    wandb_run.log({"ratio": float(ratio), "seed": int(seed), **scalar})
                print(f"r={ratio:5.2f} seed={seed} | old {res['old_acc_before']:.3f}"
                      f"->{res['old_acc_after']:.3f} new {res['new_acc_before']:.3f}"
                      f"->{res['new_acc_after']:.3f} tot {res['total_acc_before']:.3f}"
                      f"->{res['total_acc_after']:.3f} forget={res['forgetting_old']:+.3f}"
                      f" (n_new={res['n_new']} n_replay={res['n_replay']})")

    # ---- per-ratio summary (mean/std across seeds) ----
    summary = _summarize(rows, ratios, args)
    C.write_json(os.path.join(args.output_dir, "replay_sweep_summary.json"), summary)
    C.write_json(os.path.join(args.output_dir, "config.json"),
                 {k: (v.item() if isinstance(v, np.generic) else v)
                  for k, v in vars(args).items()})
    print(f"\nSaved sweep to {os.path.abspath(args.output_dir)}")
    print(f"  rows: {len(rows)}  csv: {os.path.basename(csv_path)}  "
          f"jsonl: {os.path.basename(jsonl_path)}")
    if wandb_run is not None:
        wandb_run.finish()
    return 0


def _save_final_checkpoint(ckpt_out_dir, base_ckpt, model, W_post, ratio, seed, args):
    with torch.no_grad():
        model.W_out.copy_(W_post.to(model.W_out.device, model.W_out.dtype))
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out = {
        "model_state_dict": state,
        "model_class": "ReservoirSNN",
        "architecture": base_ckpt["architecture"],
        "active_classes": base_ckpt["active_classes"],
        "removed_class": base_ckpt["removed_class"],
        "incremental_method": "rls",
        "replay_ratio": float(ratio),
        "replay_ratio_mode": args.replay_ratio_mode,
        "rls_delta": args.rls_delta,
        "rls_forgetting_factor": args.rls_forgetting_factor,
        "seed": int(seed),
        "is_rehearsal": True,
    }
    fname = f"final_ratio_{round(float(ratio) * 100):03d}_seed_{int(seed):03d}.pt"
    torch.save(out, os.path.join(ckpt_out_dir, fname))


def _summarize(rows: List[dict], ratios, args) -> dict:
    metrics = ["old_acc_after", "new_acc_after", "total_acc_after",
               "forgetting_old", "old_acc_delta", "new_acc_delta",
               "old_acc_before", "new_acc_before", "total_acc_before"]
    per_ratio = []
    for ri, ratio in enumerate(ratios):
        sel = [r for r in rows if r["ratio_index"] == ri]
        entry = {"ratio": float(ratio), "ratio_percent": float(ratio * 100.0),
                 "ratio_index": ri, "n_seeds": len(sel),
                 "n_new": sel[0]["n_new"] if sel else None,
                 "n_replay": sel[0]["n_replay"] if sel else None}
        for m in metrics:
            vals = np.array([r[m] for r in sel], dtype=np.float64)
            entry[f"{m}_mean"] = float(np.nanmean(vals)) if len(vals) else None
            entry[f"{m}_std"] = float(np.nanstd(vals)) if len(vals) else None
        per_ratio.append(entry)
    return {
        "replay_ratio_mode": args.replay_ratio_mode,
        "replay_source": args.replay_source,
        "new_class_source": args.new_class_source,
        "rls_delta": args.rls_delta,
        "rls_forgetting_factor": args.rls_forgetting_factor,
        "ratios": [float(r) for r in ratios],
        "seeds": [args.seed + i for i in range(args.n_seeds)],
        "is_rehearsal": True,
        "note": "Replay uses raw old-class samples (frozen features) -> rehearsal.",
        "per_ratio": per_ratio,
    }


def init_wandb(args, removed_class, active_classes):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    return wandb.init(project=args.wandb_project, name=args.wandb_name,
                      entity=args.wandb_entity, mode=args.wandb_mode,
                      config={**vars(args), "removed_class": removed_class,
                              "active_classes": active_classes},
                      tags=["shd", "class-incremental", "rls", args.replay_ratio_mode,
                            f"removed{removed_class}"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrain-run-dir", type=str, required=True,
                   help="Run folder from pretrain_snn_shd.py")
    p.add_argument("--checkpoint-name", type=str, default="pretrained_model.pt")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--rls-delta", type=float, default=1.0,
                   help="RLS init P = I/delta; delta plays the ridge-lambda role")
    p.add_argument("--rls-forgetting-factor", type=float, default=1.0,
                   help="Exponential forgetting in (0, 1]; 1.0 = remember all")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batches the feature/eval passes only; RLS updates are "
                        "always per-sample (the canonical algorithm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Independent repeats per ratio (seeds = seed..seed+n-1)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--replay-runs", type=int, default=11,
                   help="Number of replay-ratio points (linspace start..end)")
    p.add_argument("--replay-start-percent", type=float, default=0.0)
    p.add_argument("--replay-end-percent", type=float, default=100.0)
    p.add_argument("--replay-ratio-mode", type=str, default="additive",
                   choices=["additive", "fixed_budget"])
    p.add_argument("--replay-source", type=str, default="pretrain_train",
                   choices=["pretrain_train", "pretrain_val", "pretrain_test"])
    p.add_argument("--new-class-source", type=str, default="continual_train",
                   choices=["continual_train"])
    p.add_argument("--eval-splits", type=str, nargs="+",
                   default=["pretrain_test", "continual_test", "combined_test"],
                   choices=["pretrain_test", "continual_test", "combined_test"],
                   help="Recorded for provenance; old/new/total are always computed")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--wandb-mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", type=str, default="shd-snn-class-incremental")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
