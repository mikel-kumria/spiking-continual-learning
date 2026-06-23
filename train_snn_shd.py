"""Reservoir/recurrent SNN trainer for preprocessed SHD datasets (3 modes).

Architecture
------------
    input (C channels, inferred from the data)  ->  W_in
      -> single recurrent hidden layer of ``--nb_hidden`` (default 1000) neurons
      -> W_out -> ``--nb_outputs`` (default 20) output units

The hidden layer is a hand-rolled second-order LIF (synaptic current + membrane)
with a recurrent matrix ``W_rec``. The read-out is shared across all training
modes: ``Phi = mean_t(hidden_spikes)`` then ``logits = Phi @ W_out`` (NO bias),
so ``W_out`` means exactly the same thing whether it is solved in closed form or
learned by gradient descent.

Training modes (``--mode``)
---------------------------
* ``ridge``    : closed-form linear regression of ``W_out`` on the mean-spike
                 feature (no bias). ``W_in`` / ``W_rec`` frozen. Logs once.
* ``fullbptt`` : BPTT through everything (``W_in`` + ``W_rec`` + ``W_out``) with a
                 surrogate-gradient spike. Logs train/test accuracy per epoch.
* ``lastbptt`` : BPTT on ``W_out`` only (``W_in`` / ``W_rec`` frozen). Per epoch.

Dataset
-------
A directory holding ``train.npz`` and ``test.npz`` (dense ``X [N, T, C]``, ``y
[N]``, ``speaker [N]``) as produced by ``dataset_preprocessing/`` notebooks. The
neuron time-constants derive ``alpha``/``beta`` from the dataset's ``dt_ms``
(read from ``compression_manifest.json`` / ``preprocessing_manifest.json``).

Init sanity check
-----------------
Before training, ``W_rec`` is rescaled so the hidden layer fires within a sane
window (2-20% of spikes) on one batch, by iteratively scaling its spectral
radius by 0.8 (too active) or 1.2 (too quiet). The final spectral radius and the
fraction of always-/never-spiking hidden neurons are logged.

Example
-------
    python train_snn_shd.py --data_dir /path/SHD_350_conditional_or \
        --mode ridge --wandb_mode online
    python train_snn_shd.py --data_dir /path/SHD_700_uncompressed/... \
        --mode fullbptt --nb_epochs 200
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--data_dir", type=str, required=True,
                   help="directory with train.npz / test.npz (+ manifest json)")
    p.add_argument("--train_file", type=str, default="train.npz")
    p.add_argument("--test_file", type=str, default="test.npz")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, use only the first N samples of each split (smoke test)")
    # architecture
    p.add_argument("--nb_hidden", type=int, default=1000)
    p.add_argument("--nb_outputs", type=int, default=20)
    # neuron model
    p.add_argument("--tau_mem_ms", type=float, default=10.0)
    p.add_argument("--tau_syn_ms", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--surrogate_slope", type=float, default=100.0)
    p.add_argument("--sim_dt_ms", type=float, default=0.0,
                   help="simulation dt in ms for alpha/beta; <=0 uses the "
                        "dataset dt_ms from the manifest (recommended)")
    # reservoir / spectral-radius sanity check
    p.add_argument("--init_spectral_radius", type=float, default=1.0,
                   help="W_rec is first rescaled to this spectral radius")
    p.add_argument("--firing_low", type=float, default=0.02,
                   help="lower bound of the sane hidden firing-rate window")
    p.add_argument("--firing_high", type=float, default=0.20,
                   help="upper bound of the sane hidden firing-rate window")
    p.add_argument("--sr_scale_up", type=float, default=1.2,
                   help="multiply spectral radius by this when firing is too low")
    p.add_argument("--sr_scale_down", type=float, default=0.8,
                   help="multiply spectral radius by this when firing is too high")
    p.add_argument("--sr_max_iters", type=int, default=20)
    # ridge
    p.add_argument("--ridge_lambda", type=float, default=1.0)
    # bptt
    p.add_argument("--nb_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0,
                   help="learning rate; <=0 picks a per-mode default "
                        "(fullbptt 2e-4, lastbptt 1e-3)")
    p.add_argument("--optimizer", type=str, default="adamax",
                   choices=["adamax", "adam", "sgd"])
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="max grad norm for BPTT; <=0 disables clipping")
    # modes / runtime
    p.add_argument("--mode", type=str, default="ridge",
                   choices=["ridge", "fullbptt", "lastbptt"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    # logging
    p.add_argument("--wandb_project", type=str, default="shd-snn")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def resolve_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Dataset + manifest
# =============================================================================


def load_split(path, limit=0):
    """Load one ``.npz`` split -> (X float32 [N,T,C], y long [N])."""
    if not os.path.isfile(path):
        raise SystemExit(f"dataset file not found: {path}")
    d = np.load(path)
    if "X" not in d or "y" not in d:
        raise SystemExit(
            f"{path} must contain arrays 'X' and 'y' (found {list(d.keys())})")
    X = d["X"]
    y = d["y"]
    if limit and limit > 0:
        X, y = X[:limit], y[:limit]
    assert X.ndim == 3, f"expected X with shape [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], (
        f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    X = torch.from_numpy(np.ascontiguousarray(X)).float()
    y = torch.from_numpy(np.ascontiguousarray(y)).long()
    return X, y


def load_manifest(data_dir):
    """Read dataset metadata from compression_/preprocessing_ manifest.

    Returns a dict with normalized keys: ``compression_method``,
    ``compression_factor``, ``target_channels``, ``nb_steps``, ``condition_or``,
    ``dt_ms``, ``source_manifest`` (the nested provenance dict) and ``raw``.
    Missing files yield an all-unknown dict (``dt_ms`` None)."""
    p_comp = os.path.join(data_dir, "compression_manifest.json")
    p_pre = os.path.join(data_dir, "preprocessing_manifest.json")
    if os.path.isfile(p_comp):
        with open(p_comp) as f:
            m = json.load(f)
        src = m.get("source_manifest", {}) or {}
        return {
            "manifest_kind": "compression",
            "compression_method": m.get("compression_method"),
            "compression_factor": m.get("compression_factor"),
            "target_channels": m.get("target_channels"),
            "nb_steps": m.get("nb_steps"),
            "condition_or": m.get("condition_or"),
            "dt_ms": src.get("dt_ms"),
            "source_manifest": src,
            "raw": m,
        }
    if os.path.isfile(p_pre):
        with open(p_pre) as f:
            m = json.load(f)
        return {
            "manifest_kind": "preprocessing",
            "compression_method": "uncompressed",
            "compression_factor": 1,
            "target_channels": m.get("nb_inputs"),
            "nb_steps": m.get("nb_steps"),
            "condition_or": None,
            "dt_ms": m.get("dt_ms"),
            "source_manifest": m,
            "raw": m,
        }
    return {
        "manifest_kind": "none",
        "compression_method": None,
        "compression_factor": None,
        "target_channels": None,
        "nb_steps": None,
        "condition_or": None,
        "dt_ms": None,
        "source_manifest": {},
        "raw": {},
    }


# =============================================================================
# Neuron model
# =============================================================================


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward (Zenke/Pittorino)."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output / (ctx.slope * x.abs() + 1.0) ** 2
        return grad, None


class ReservoirSNN(nn.Module):
    """Variable-input recurrent SNN with a shared mean-spike linear read-out."""

    def __init__(self, nb_inputs, nb_hidden, nb_outputs, alpha, beta, threshold,
                 weight_scale, surrogate_slope):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)

        W_in = torch.empty(nb_inputs, nb_hidden)
        W_rec = torch.empty(nb_hidden, nb_hidden)
        W_out = torch.empty(nb_hidden, nb_outputs)
        nn.init.normal_(W_in, 0.0, weight_scale / math.sqrt(nb_inputs))
        nn.init.normal_(W_rec, 0.0, weight_scale / math.sqrt(nb_hidden))
        nn.init.normal_(W_out, 0.0, weight_scale / math.sqrt(nb_hidden))
        self.W_in = nn.Parameter(W_in)
        self.W_rec = nn.Parameter(W_rec)
        self.W_out = nn.Parameter(W_out)

    def spike_fn(self, x):
        return SurrGradSpike.apply(x, self.surrogate_slope)

    def measured_spectral_radius(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(
                self.W_rec.detach().to("cpu", torch.float32)).abs().max().item())

    def scale_recurrent(self, factor):
        """Multiply ``W_rec`` (and hence its spectral radius) by ``factor``."""
        with torch.no_grad():
            self.W_rec.mul_(factor)

    def hidden_spikes(self, x, return_trace=False):
        """Run the LIF reservoir; return (Phi [B,H], trace [B,T,H] or None).

        Phi is the time-mean of hidden spikes (the shared read-out feature)."""
        x = x.float()
        B, T, C = x.shape
        assert C == self.nb_inputs, (
            f"input channels {C} != model nb_inputs {self.nb_inputs}")
        H = self.nb_hidden
        # feed-forward drive for every timestep: [B, T, H]
        h_in = (x.reshape(B * T, C) @ self.W_in).reshape(B, T, H)

        syn = torch.zeros(B, H, device=x.device, dtype=h_in.dtype)
        mem = torch.zeros_like(syn)
        spk_prev = torch.zeros_like(syn)
        spk_sum = torch.zeros_like(syn)
        trace = [] if return_trace else None

        for t in range(T):
            drive = h_in[:, t] + spk_prev @ self.W_rec      # recurrent uses prev spike
            spk = self.spike_fn(mem - self.threshold)        # spike from membrane at step start
            rst = spk.detach()
            new_syn = self.alpha * syn + drive
            mem = (self.beta * mem + syn) * (1.0 - rst)      # integrate old syn, reset-to-zero
            syn = new_syn
            spk_sum = spk_sum + spk
            spk_prev = spk
            if return_trace:
                trace.append(spk)

        Phi = spk_sum / T
        trace_t = torch.stack(trace, dim=1) if return_trace else None
        return Phi, trace_t

    def logits_from_features(self, Phi):
        return Phi @ self.W_out

    def forward(self, x):
        Phi, _ = self.hidden_spikes(x)
        return self.logits_from_features(Phi)


# =============================================================================
# Spectral-radius sanity check
# =============================================================================


def spectral_radius_sanity(model, x_batch, args):
    """Rescale W_rec so the hidden firing rate lands in [firing_low, firing_high].

    Returns a dict of init-time diagnostics (final spectral radius, firing rate,
    and the always-/never-spiking neuron fractions on this batch)."""
    sr_measured = model.measured_spectral_radius()
    if sr_measured <= 0:
        raise SystemExit("measured spectral radius is 0; cannot rescale W_rec")
    # rescale to the requested starting spectral radius
    model.scale_recurrent(args.init_spectral_radius / sr_measured)
    sr_current = float(args.init_spectral_radius)

    rate = float("nan")
    trace = None
    iters_used = 0
    for it in range(args.sr_max_iters + 1):
        with torch.no_grad():
            _, trace = model.hidden_spikes(x_batch, return_trace=True)
        rate = float(trace.mean().item())
        iters_used = it
        if rate > args.firing_high:
            model.scale_recurrent(args.sr_scale_down)
            sr_current *= args.sr_scale_down
        elif rate < args.firing_low:
            model.scale_recurrent(args.sr_scale_up)
            sr_current *= args.sr_scale_up
        else:
            break

    in_window = args.firing_low <= rate <= args.firing_high
    if not in_window:
        print(f"WARNING: hidden firing rate {rate:.4f} still outside "
              f"[{args.firing_low}, {args.firing_high}] after {iters_used} "
              f"iters; proceeding with spectral_radius={sr_current:.4f}")

    # per-neuron activity on this batch: trace is [B, T, H]
    spk_any_bt = (trace > 0)
    never_spiked = (~spk_any_bt.any(dim=(0, 1)))              # [H] never fired
    always_spiked = spk_any_bt.all(dim=(0, 1))               # [H] fired every sample x step
    H = model.nb_hidden
    pct_never = 100.0 * float(never_spiked.sum().item()) / H
    pct_always = 100.0 * float(always_spiked.sum().item()) / H

    # recompute the actual spectral radius for an accurate log
    sr_final = model.measured_spectral_radius()
    return {
        "init_spectral_radius": float(args.init_spectral_radius),
        "spectral_radius_tracked": sr_current,
        "final_spectral_radius": sr_final,
        "init_hidden_firing_rate": rate,
        "init_firing_in_window": bool(in_window),
        "sr_iters": iters_used,
        "pct_always_spiked": pct_always,
        "pct_never_spiked": pct_never,
    }


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(model, X, y, batch_size, device):
    """Top-1 accuracy of the shared mean-spike read-out over a split."""
    model.eval()
    correct, total = 0, 0
    N = X.shape[0]
    for s in range(0, N, batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        logits = model(xb)
        correct += int((logits.argmax(1) == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


# =============================================================================
# Training modes
# =============================================================================


def train_ridge(model, X_tr, y_tr, X_te, y_te, args, device):
    """Closed-form (float64, no bias) ridge solve of W_out on mean-spike feats."""
    model.requires_grad_(False)
    model.eval()
    t0 = time.time()

    def collect(X):
        feats = []
        with torch.no_grad():
            for s in range(0, X.shape[0], args.batch_size):
                xb = X[s:s + args.batch_size].to(device)
                Phi, _ = model.hidden_spikes(xb)
                feats.append(Phi.cpu())
        return torch.cat(feats, 0)

    Phi_tr = collect(X_tr)                                    # [N, H]
    H = model.nb_hidden
    Phi64 = Phi_tr.double()
    Y = torch.zeros((Phi_tr.shape[0], model.nb_outputs), dtype=torch.float64)
    Y[torch.arange(Phi_tr.shape[0]), y_tr.long()] = 1.0

    A = Phi64.T @ Phi64 + args.ridge_lambda * torch.eye(H, dtype=torch.float64)
    B = Phi64.T @ Y
    try:
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(B, L)
    except RuntimeError:
        W = torch.linalg.solve(A, B)
    assert W.shape == (H, model.nb_outputs), f"W_out shape {tuple(W.shape)}"

    with torch.no_grad():
        model.W_out.copy_(W.to(model.W_out.device, model.W_out.dtype))

    train_mse = float(((Phi64 @ W) - Y).pow(2).mean().item())
    train_acc = evaluate(model, X_tr, y_tr, args.batch_size, device)
    test_acc = evaluate(model, X_te, y_te, args.batch_size, device)
    seconds = time.time() - t0
    print(f"[ridge] train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
          f"train_mse={train_mse:.4f} ({seconds:.1f}s)")
    return {"train_acc": train_acc, "test_acc": test_acc,
            "train_mse": train_mse, "ridge_seconds": seconds}


def build_optimizer(params, args):
    lr = args.lr if args.lr > 0 else (2e-4 if args.mode == "fullbptt" else 1e-3)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr), lr
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9), lr
    return torch.optim.Adamax(params, lr=lr), lr


def train_bptt(model, X_tr, y_tr, X_te, y_te, args, device, wandb_run):
    """fullbptt (all weights) or lastbptt (W_out only) with surrogate gradients."""
    if args.mode == "lastbptt":
        model.W_in.requires_grad_(False)
        model.W_rec.requires_grad_(False)
        model.W_out.requires_grad_(True)
    else:  # fullbptt
        model.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters for BPTT")
    optimizer, lr = build_optimizer(params, args)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(args.seed)
    N = X_tr.shape[0]

    best = {"test_acc": -1.0, "epoch": -1}
    for epoch in range(args.nb_epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, generator=gen)
        run_loss = run_correct = seen = 0.0
        spk_sum = spk_sites = 0.0
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            Phi, _ = model.hidden_spikes(xb)
            logits = model.logits_from_features(Phi)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            n = int(yb.numel())
            run_loss += float(loss.item()) * n
            run_correct += float((logits.argmax(1) == yb).sum().item())
            seen += n
            spk_sum += float(Phi.detach().sum().item())
            spk_sites += float(Phi.numel())

        train_loss = run_loss / max(seen, 1)
        train_acc = run_correct / max(seen, 1)
        test_acc = evaluate(model, X_te, y_te, args.batch_size, device)
        if test_acc > best["test_acc"]:
            best = {"test_acc": test_acc, "epoch": epoch}
        train_hz = spk_sum / max(spk_sites, 1)
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
            "train_hidden_firing_rate": train_hz,
            "lr": lr,
            "epoch_seconds": time.time() - t0,
        }
        if wandb_run is not None:
            wandb_run.log(payload)
        print(f"[{args.mode}] epoch={epoch:03d} loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
              f"best={best['test_acc']:.4f}@{best['epoch']} hz={train_hz:.3f} "
              f"({payload['epoch_seconds']:.1f}s)")

    final_train_acc = evaluate(model, X_tr, y_tr, args.batch_size, device)
    final_test_acc = evaluate(model, X_te, y_te, args.batch_size, device)
    return {"train_acc": final_train_acc, "test_acc": final_test_acc,
            "best_test_acc": best["test_acc"], "best_epoch": best["epoch"]}


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}  mode: {args.mode}")

    data_dir = os.path.expanduser(args.data_dir)
    manifest = load_manifest(data_dir)

    # ---- data ----
    X_tr, y_tr = load_split(os.path.join(data_dir, args.train_file), args.limit)
    X_te, y_te = load_split(os.path.join(data_dir, args.test_file), args.limit)
    nb_inputs = X_tr.shape[2]
    nb_steps = X_tr.shape[1]
    assert X_te.shape[2] == nb_inputs, (
        f"train/test channel mismatch: {nb_inputs} vs {X_te.shape[2]}")
    assert X_te.shape[1] == nb_steps, (
        f"train/test timestep mismatch: {nb_steps} vs {X_te.shape[1]}")
    n_classes_seen = int(max(y_tr.max().item(), y_te.max().item())) + 1
    assert n_classes_seen <= args.nb_outputs, (
        f"label {n_classes_seen - 1} >= nb_outputs {args.nb_outputs}")
    if manifest["target_channels"] is not None and \
            int(manifest["target_channels"]) != nb_inputs:
        print(f"WARNING: manifest target_channels={manifest['target_channels']} "
              f"!= data channels {nb_inputs}; trusting the data.")
    print(f"train X: {tuple(X_tr.shape)}  test X: {tuple(X_te.shape)}  "
          f"nb_inputs={nb_inputs} nb_steps={nb_steps}")

    # ---- dt / neuron decays (from dataset manifest unless overridden) ----
    if args.sim_dt_ms > 0:
        dt_ms = float(args.sim_dt_ms)
        dt_source = "cli"
    elif manifest["dt_ms"] is not None:
        dt_ms = float(manifest["dt_ms"])
        dt_source = "manifest"
    else:
        raise SystemExit(
            "dt_ms not found in manifest; pass --sim_dt_ms explicitly")
    alpha = math.exp(-dt_ms / args.tau_syn_ms)
    beta = math.exp(-dt_ms / args.tau_mem_ms)
    print(f"dt_ms={dt_ms} ({dt_source})  alpha={alpha:.4f} beta={beta:.4f}  "
          f"compression={manifest['compression_method']}")

    # ---- model ----
    model = ReservoirSNN(
        nb_inputs, args.nb_hidden, args.nb_outputs, alpha, beta,
        args.threshold, args.weight_scale, args.surrogate_slope).to(device)

    # ---- spectral-radius sanity check (one train batch) ----
    sane_n = min(args.batch_size, X_tr.shape[0])
    x_batch = X_tr[:sane_n].to(device)
    sr_info = spectral_radius_sanity(model, x_batch, args)
    print(f"[sanity] spectral_radius {sr_info['init_spectral_radius']:.3f} -> "
          f"{sr_info['final_spectral_radius']:.4f}  "
          f"firing_rate={sr_info['init_hidden_firing_rate']:.4f}  "
          f"always_spiked={sr_info['pct_always_spiked']:.2f}%  "
          f"never_spiked={sr_info['pct_never_spiked']:.2f}%")

    # ---- W&B ----
    wandb_run = init_wandb(args, manifest, nb_inputs, nb_steps, dt_ms, dt_source,
                           alpha, beta, sr_info)

    # ---- train ----
    if args.mode == "ridge":
        result = train_ridge(model, X_tr, y_tr, X_te, y_te, args, device)
        if wandb_run is not None:
            wandb_run.log({**result, **sr_info})
    else:
        result = train_bptt(model, X_tr, y_tr, X_te, y_te, args, device, wandb_run)

    print(f"=== {args.mode} ===  train_acc={result['train_acc']:.4f}  "
          f"test_acc={result['test_acc']:.4f}")

    if wandb_run is not None:
        for k, v in {**result, **sr_info}.items():
            wandb_run.summary[k] = v
        wandb_run.finish()


def init_wandb(args, manifest, nb_inputs, nb_steps, dt_ms, dt_source, alpha,
               beta, sr_info):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    config = {
        "mode": args.mode,
        "nb_inputs": nb_inputs,
        "input_channels": nb_inputs,
        "nb_hidden": args.nb_hidden,
        "nb_outputs": args.nb_outputs,
        "nb_steps": nb_steps,
        "dt_ms": dt_ms,
        "dt_source": dt_source,
        "sim_dt_ms": dt_ms,
        "alpha": alpha,
        "beta": beta,
        "tau_mem_ms": args.tau_mem_ms,
        "tau_syn_ms": args.tau_syn_ms,
        "threshold": args.threshold,
        "weight_scale": args.weight_scale,
        "surrogate_slope": args.surrogate_slope,
        "ridge_lambda": args.ridge_lambda,
        "nb_epochs": args.nb_epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        # dataset provenance / temporal binning
        "compression_method": manifest["compression_method"],
        "compression_factor": manifest["compression_factor"],
        "target_channels": manifest["target_channels"],
        "manifest_nb_steps": manifest["nb_steps"],
        "condition_or": manifest["condition_or"],
        "manifest_kind": manifest["manifest_kind"],
        "source_manifest": manifest["source_manifest"],
        # init spectral-radius sanity diagnostics
        **{f"init/{k}": v for k, v in sr_info.items()},
    }
    return wandb.init(project=args.wandb_project, name=args.wandb_name,
                      entity=args.wandb_entity, mode=args.wandb_mode,
                      config=config,
                      tags=["shd", args.mode, str(manifest["compression_method"]),
                            f"C{nb_inputs}", f"dt{dt_ms}ms"])


if __name__ == "__main__":
    main()
