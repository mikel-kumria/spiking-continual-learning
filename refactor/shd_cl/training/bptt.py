"""BPTT training: ``fullbptt`` (all weights) and ``lastbptt`` (``W_out`` only).

Both run the reservoir over time and backprop through the chosen output layer with
surrogate gradients. They differ only in which parameters receive gradients:

* ``fullbptt`` : train ``W_in`` + ``W_rec`` + ``W_out``.
* ``lastbptt`` : freeze ``W_in`` + ``W_rec``; train only ``W_out`` (still BPTT
  through the output-layer dynamics -- this is NOT ridge).

Label handling: when ``active_classes`` is a strict subset (19-class pretraining),
cross-entropy is computed on the ACTIVE logit columns only and targets are remapped
to contiguous indices; the removed column is held at exactly 0 (init + after each
step) so it never contributes to the loss or wins an argmax.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..models.snn import ReservoirSNN


@dataclass
class BpttConfig:
    method: str = "fullbptt"           # "fullbptt" | "lastbptt"
    nb_epochs: int = 200
    batch_size: int = 64
    lr: float = 0.0                    # <=0 -> per-method default
    optimizer: str = "adamax"          # adamax | adam | sgd
    grad_clip: float = 0.0
    seed: int = 0


def _build_optimizer(params, cfg: BpttConfig):
    lr = cfg.lr if cfg.lr > 0 else (2e-4 if cfg.method == "fullbptt" else 1e-3)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr), lr
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9), lr
    return torch.optim.Adamax(params, lr=lr), lr


def _set_trainable(model: ReservoirSNN, method: str) -> List[nn.Parameter]:
    if method == "lastbptt":
        model.W_in.requires_grad_(False)
        model.W_rec.requires_grad_(False)
        model.W_out.requires_grad_(True)
    elif method == "fullbptt":
        model.requires_grad_(True)
    else:
        raise ValueError(f"unknown bptt method {method!r}")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters for BPTT")
    return params


@torch.no_grad()
def _grad_global_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    return total ** 0.5


@torch.no_grad()
def _weight_norms(model: ReservoirSNN) -> dict:
    return {"W_in_norm": float(model.W_in.detach().norm().item()),
            "W_rec_norm": float(model.W_rec.detach().norm().item()),
            "W_out_norm": float(model.W_out.detach().norm().item())}


@torch.no_grad()
def _accuracy_active(model, X, y, active_idx, batch_size, device) -> float:
    model.eval()
    if X.shape[0] == 0:
        return float("nan")
    correct = total = 0
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        logits = model(xb)
        if active_idx is not None:
            sub = logits.index_select(1, active_idx)
            pred = active_idx[sub.argmax(1)]
        else:
            pred = logits.argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def train_bptt(model: ReservoirSNN, X_tr: torch.Tensor, y_tr: torch.Tensor, *,
               active_classes: Sequence[int], num_classes: int, cfg: BpttConfig,
               device: torch.device, removed_class: Optional[int] = None,
               X_te: Optional[torch.Tensor] = None, y_te: Optional[torch.Tensor] = None,
               on_epoch: Optional[Callable[[dict], None]] = None) -> dict:
    """Train by BPTT. Returns a metrics dict incl. per-epoch history.

    ``active_classes`` == all classes -> baseline (CE over all logits). A strict
    subset -> 19-class pretraining (CE over active columns, removed col frozen at 0).
    """
    params = _set_trainable(model, cfg.method)
    optimizer, lr = _build_optimizer(params, cfg)
    loss_fn = nn.CrossEntropyLoss()

    active_idx = torch.tensor(list(active_classes), dtype=torch.long, device=device)
    is_subset = len(active_classes) < num_classes
    # original label -> contiguous active index, for CE targets
    pos = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    pos[active_idx] = torch.arange(len(active_classes), device=device)

    if is_subset and removed_class is not None:
        with torch.no_grad():
            model.W_out[:, removed_class] = 0.0

    gen = torch.Generator().manual_seed(cfg.seed)
    N = X_tr.shape[0]
    history: List[dict] = []
    for epoch in range(cfg.nb_epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, generator=gen)
        run_loss = run_correct = seen = 0.0
        last_grad_norm = 0.0
        for s in range(0, N, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            logits = model(xb)                              # [B, num_classes]
            if is_subset:
                active_logits = logits.index_select(1, active_idx)  # [B, |active|]
                target = pos[yb]
                loss = loss_fn(active_logits, target)
                pred = active_idx[active_logits.argmax(1)]
            else:
                loss = loss_fn(logits, yb)
                pred = logits.argmax(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if is_subset and removed_class is not None and model.W_out.grad is not None:
                model.W_out.grad[:, removed_class] = 0.0    # freeze removed-col gradient
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            last_grad_norm = _grad_global_norm(params)
            optimizer.step()
            if is_subset and removed_class is not None:
                with torch.no_grad():
                    model.W_out[:, removed_class] = 0.0     # keep removed col at 0
            n = int(yb.numel())
            run_loss += float(loss.item()) * n
            run_correct += int((pred == yb).sum().item())
            seen += n
        train_loss = run_loss / max(seen, 1)
        train_acc = run_correct / max(seen, 1)
        rec = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "lr": lr, "grad_norm": last_grad_norm,
               "epoch_seconds": time.time() - t0, **_weight_norms(model)}
        if X_te is not None and y_te is not None:
            rec["test_acc"] = _accuracy_active(
                model, X_te, y_te, active_idx if is_subset else None,
                cfg.batch_size, device)
        history.append(rec)
        if on_epoch is not None:
            on_epoch(rec)
        msg = (f"[{cfg.method}] epoch={epoch:03d} loss={train_loss:.4f} "
               f"train_acc={train_acc:.4f}")
        if "test_acc" in rec:
            msg += f" test_acc={rec['test_acc']:.4f}"
        print(msg + f" ({rec['epoch_seconds']:.1f}s)")

    if is_subset and removed_class is not None:
        with torch.no_grad():
            model.W_out[:, removed_class] = 0.0
    return {"method": cfg.method, "lr": lr, "history": history,
            "final_train_acc": history[-1]["train_acc"] if history else float("nan")}
