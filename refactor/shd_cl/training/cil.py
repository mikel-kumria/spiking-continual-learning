"""Class-incremental learning (CIL): learn the removed class on top of a
pretrained 19-class model, with per-old-class replay.

Pipeline per (replay_ratio, seed):
  1. Reset the model to the pretrained checkpoint (independent runs).
  2. Build a CIL train set = all/sampled new-class data + per-old-class replay.
  3. Adapt the read-out (and, for fullbptt, the reservoir) with the chosen method:
       * ``ridge``    : class-balanced weighted ridge on ``hidden_spike_sum`` -> W_out.
       * ``lastbptt`` : BPTT of W_out only (reservoir frozen), through the output layer.
       * ``fullbptt`` : BPTT of all weights.
  4. Evaluate old / new / combined test accuracy over all 20 outputs.

Accuracy is ALWAYS argmax over the 20 output neurons. For ridge the PRIMARY
readout is linear (``argmax(hidden_spike_sum @ W_out)``); for BPTT it is the
output-layer decode consistent with training. Both are logged either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .. import NUM_CLASSES
from ..evaluation.metrics import (average_old_perclass_forgetting, balanced_accuracy,
                                  per_class_accuracy, two_group_balanced_accuracy)
from ..evaluation.predict import collect_both_logits, collect_spike_sums
from ..models.snn import ReservoirSNN
from .bptt import BpttConfig, train_bptt
from .ridge import fit_readout, apply_readout
from .replay import ReplayPlan, sample_replay


@dataclass
class CILConfig:
    cil_training_method: str = "ridge"          # ridge | lastbptt | fullbptt
    ridge_lambda: float = 1.0
    ridge_weighting: str = "normalized_inverse_class_count"
    n_new_cap: int = 0                           # 0 -> use all new-class train samples
    replay_replacement_policy: str = "with_replacement_if_needed"
    bptt: BpttConfig = field(default_factory=lambda: BpttConfig(method="fullbptt"))
    batch_size: int = 64


# =============================================================================
# Evaluation
# =============================================================================


def _primary_readout(method: str) -> str:
    """ridge -> linear readout; bptt -> output-layer decode."""
    return "linear" if method == "ridge" else "output"


def evaluate_cil(model: ReservoirSNN, old_X, old_y, new_X, new_y, *, method: str,
                 removed_class: int, batch_size: int, device) -> dict:
    """Old / new / combined accuracy (primary + linear diagnostic) over 20 outputs."""
    out_old, lin_old = collect_both_logits(model, old_X, batch_size, device)
    out_new, lin_new = collect_both_logits(model, new_X, batch_size, device)
    yo = old_y.cpu().numpy().astype(np.int64) if torch.is_tensor(old_y) else np.asarray(old_y)
    yn = new_y.cpu().numpy().astype(np.int64) if torch.is_tensor(new_y) else np.asarray(new_y)
    y_comb = np.concatenate([yo, yn])

    def summarize(out_logits_old, out_logits_new, tag):
        pred_old = out_logits_old.argmax(1).numpy().astype(np.int64) if len(yo) else np.zeros((0,), np.int64)
        pred_new = out_logits_new.argmax(1).numpy().astype(np.int64) if len(yn) else np.zeros((0,), np.int64)
        pred_comb = np.concatenate([pred_old, pred_new])
        old_acc = float(np.mean(pred_old == yo)) if len(yo) else float("nan")
        new_acc = float(np.mean(pred_new == yn)) if len(yn) else float("nan")
        total_acc = float(np.mean(pred_comb == y_comb)) if len(y_comb) else float("nan")
        pc = per_class_accuracy(y_comb, pred_comb)
        return {
            f"old_acc{tag}": old_acc,
            f"new_acc{tag}": new_acc,
            f"total_acc{tag}": total_acc,
            f"balanced_acc{tag}": balanced_accuracy(y_comb, pred_comb),
            f"two_group_balanced_acc{tag}": two_group_balanced_accuracy(old_acc, new_acc),
            f"per_class_acc{tag}": pc,
        }

    linear = summarize(lin_old, lin_new, "_linear")
    output = summarize(out_old, out_new, "_output")
    # promote the method's primary readout to the unsuffixed keys
    primary = linear if _primary_readout(method) == "linear" else output
    src = "_linear" if _primary_readout(method) == "linear" else "_output"
    promoted = {k[:-len(src)] if k.endswith(src) else k: v for k, v in primary.items()}
    return {**promoted, **linear, **output, "primary_readout": _primary_readout(method)}


# =============================================================================
# One CIL run (single replay ratio + seed)
# =============================================================================


def _gather(X: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    return X[torch.from_numpy(np.asarray(idx)).long()]


def train_cil_once(model: ReservoirSNN, cfg: CILConfig, *, new_X, new_y,
                   replay_X, replay_y, plan: ReplayPlan, active_classes,
                   new_pool_sums: Optional[torch.Tensor] = None,
                   replay_pool_sums: Optional[torch.Tensor] = None,
                   device=torch.device("cpu")) -> dict:
    """Adapt the model in place on the CIL train set. Returns a small train-info dict."""
    method = cfg.cil_training_method
    new_y = np.asarray(new_y.cpu()) if torch.is_tensor(new_y) else np.asarray(new_y)
    replay_y = np.asarray(replay_y.cpu()) if torch.is_tensor(replay_y) else np.asarray(replay_y)
    y_train = np.concatenate([new_y[plan.new_indices], replay_y[plan.replay_indices]])

    if method == "ridge":
        # feature space (frozen reservoir): reuse precomputed pool spike-sums
        assert new_pool_sums is not None and replay_pool_sums is not None
        feats = torch.cat([new_pool_sums[torch.from_numpy(plan.new_indices).long()],
                           replay_pool_sums[torch.from_numpy(plan.replay_indices).long()]], 0)
        W_full, info = fit_readout(
            feats, y_train, columns=list(range(NUM_CLASSES)), nb_outputs=NUM_CLASSES,
            lam=cfg.ridge_lambda, weighting=cfg.ridge_weighting, zero_columns=None)
        apply_readout(model, W_full)
        return {"train_method": "ridge", **info, "n_train": int(len(y_train))}

    # BPTT (fullbptt / lastbptt): raw-X trainset, continue from pretrained weights
    X_train = torch.cat([_gather(new_X, plan.new_indices),
                         _gather(replay_X, plan.replay_indices)], 0)
    y_train_t = torch.from_numpy(y_train).long()
    bptt_cfg = BpttConfig(method=method, nb_epochs=cfg.bptt.nb_epochs,
                          batch_size=cfg.bptt.batch_size, lr=cfg.bptt.lr,
                          optimizer=cfg.bptt.optimizer, grad_clip=cfg.bptt.grad_clip,
                          seed=cfg.bptt.seed)
    res = train_bptt(model, X_train, y_train_t, active_classes=list(range(NUM_CLASSES)),
                     num_classes=NUM_CLASSES, cfg=bptt_cfg, device=device,
                     removed_class=None)
    return {"train_method": method, "n_train": int(len(y_train)),
            "final_train_acc": res.get("final_train_acc")}


def run_one_cil(model: ReservoirSNN, pretrained_state: Dict[str, torch.Tensor],
                cfg: CILConfig, *, replay_ratio: float, rng: np.random.Generator,
                new_X, new_y, replay_X, replay_y, old_test_X, old_test_y,
                new_test_X, new_test_y, removed_class: int, active_classes,
                new_pool_sums=None, replay_pool_sums=None,
                before_metrics: Optional[dict] = None,
                device=torch.device("cpu")) -> dict:
    """Full single CIL run: reset -> replay -> train -> eval; return a flat row."""
    # 1. reset to pretrained state (independent runs)
    model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})

    method = cfg.cil_training_method
    # 2. before-CIL metrics (constant across runs -> may be passed in)
    if before_metrics is None:
        before_metrics = evaluate_cil(
            model, old_test_X, old_test_y, new_test_X, new_test_y, method=method,
            removed_class=removed_class, batch_size=cfg.batch_size, device=device)

    # 3. replay plan + training
    plan = sample_replay(np.asarray(new_y), np.asarray(replay_y),
                         replay_ratio=replay_ratio, rng=rng, n_new_cap=cfg.n_new_cap,
                         policy=cfg.replay_replacement_policy)
    train_info = train_cil_once(
        model, cfg, new_X=new_X, new_y=new_y, replay_X=replay_X, replay_y=replay_y,
        plan=plan, active_classes=active_classes, new_pool_sums=new_pool_sums,
        replay_pool_sums=replay_pool_sums, device=device)

    # 4. after-CIL metrics
    after = evaluate_cil(model, old_test_X, old_test_y, new_test_X, new_test_y,
                         method=method, removed_class=removed_class,
                         batch_size=cfg.batch_size, device=device)

    row = _compose_row(before_metrics, after, plan, train_info, removed_class)
    return row


def _compose_row(before: dict, after: dict, plan: ReplayPlan, train_info: dict,
                 removed_class: int) -> dict:
    row = {**plan.log}
    row.update({f"train_{k}": v for k, v in train_info.items()})
    # scalar before/after for both readouts
    for tag in ("", "_linear", "_output"):
        for grp in ("old_acc", "new_acc", "total_acc", "balanced_acc",
                    "two_group_balanced_acc"):
            key = grp + tag
            if key in before:
                row[f"{key}_before"] = before[key]
            if key in after:
                row[f"{key}_after"] = after[key]
    # deltas on the primary readout
    row["forgetting_old"] = before.get("old_acc", float("nan")) - after.get("old_acc", float("nan"))
    row["learning_new"] = after.get("new_acc", float("nan")) - before.get("new_acc", float("nan"))
    row["total_delta"] = after.get("total_acc", float("nan")) - before.get("total_acc", float("nan"))
    row["avg_old_perclass_forgetting"] = average_old_perclass_forgetting(
        before.get("per_class_acc", {}), after.get("per_class_acc", {}), removed_class)
    row["primary_readout"] = after.get("primary_readout")
    # per-class before/after (primary readout)
    row["per_class_acc_before"] = before.get("per_class_acc", {})
    row["per_class_acc_after"] = after.get("per_class_acc", {})
    return row


# =============================================================================
# Feature precomputation for ridge (frozen reservoir)
# =============================================================================


def precompute_pool_sums(model: ReservoirSNN, X: torch.Tensor, batch_size: int,
                         device) -> torch.Tensor:
    """Hidden spike-sums for a whole pool -> ``[N,H]`` float64 (ridge feature cache)."""
    return collect_spike_sums(model, X, batch_size, device)
