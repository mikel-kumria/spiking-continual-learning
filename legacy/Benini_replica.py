"""Benini-replica SNN trainer for preprocessed SHD datasets.

This is ``train_snn_shd.py`` with three things swapped to match the
Spiking-Compressed-Continual-Learning ("Benini") pretraining model
(``references/.../statedicts_class_incremental/heidelberg_statedict_generator.py``):

  1. ARCHITECTURE  -- three recurrent LIF hidden layers
       ``nb_inputs -> 200 -> 100 -> 50 -> nb_outputs``
     (``nb_hidden_1 = --nb_hidden``, ``nb_hidden_2 = //2``, ``nb_hidden_3 = //2``),
     feed-forward ``w1,w2,w3,w_out`` plus a recurrent matrix ``v1,v2,v3`` on every
     hidden layer. The input dimension is still inferred from the dataset, so a
     compressed dataset feeds a smaller ``w1`` than Benini's fixed 700.
  2. READOUT       -- a NON-spiking leaky-integrator read-out on the last hidden
     layer's spikes (same alpha/beta filters, no reset), recorded over time and
     seeded with a zero, exactly like Benini's ``run_snn``.
  3. LOSS          -- ``LogSoftmax + NLLLoss`` on the MAX-over-time read-out, with
     Benini's optional spike regularizers ``reg_1`` (L1) / ``reg_2`` (L2), both 0
     by default. Accuracy = argmax of the max-over-time output.

Everything else is kept from ``train_snn_shd.py`` on purpose:
  * dataset + split   -- preprocessed dense ``train.npz`` / ``test.npz`` and their
                         own train/test split (NOT Benini's merged 1/5-batch split);
  * neuron model      -- same 2nd-order LIF, surrogate slope, tau_mem/tau_syn,
                         threshold, weight_scale, manifest-driven ``dt`` for
                         alpha/beta (pass ``--sim_dt_ms 1.0`` to match Benini's
                         fixed-1ms decays exactly);
  * optimizer/sched   -- Adamax @ 2e-4 (fullbptt), 200 epochs, batch 64;
  * W&B logging       -- defaults to the ``Benini_replica`` project.

Two ``train_snn_shd.py`` features are intentionally DROPPED because they are tied
to the single-``W_rec`` reservoir and have no Benini counterpart:
  * the ``ridge`` closed-form mode (it solves a linear mean-rate read-out);
  * the spectral-radius / firing-rate sanity rescale of ``W_rec``.

Example
-------
    python Benini_replica.py --data_dir /path/SHD_700_uncompressed/... \
        --nb_epochs 200 --wandb_project Benini_replica
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
    # architecture (Benini: nb_hidden -> //2 -> //2)
    p.add_argument("--nb_hidden", type=int, default=200,
                   help="size of the FIRST hidden layer; the next two are //2 and //4")
    p.add_argument("--nb_outputs", type=int, default=20)
    # neuron model
    p.add_argument("--tau_mem_ms", type=float, default=10.0)
    p.add_argument("--tau_syn_ms", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--surrogate_slope", type=float, default=100.0)
    p.add_argument("--sim_dt_ms", type=float, default=0.0,
                   help="simulation dt in ms for alpha/beta; <=0 uses the "
                        "dataset dt_ms from the manifest. Pass 1.0 to match "
                        "Benini's fixed-1ms decays exactly.")
    # Benini spike regularizers (0 = off, the published 'best result' setting)
    p.add_argument("--reg_1", type=float, default=0.0,
                   help="L1 weight on total last-hidden-layer spikes")
    p.add_argument("--reg_2", type=float, default=0.0,
                   help="L2 weight on per-neuron spike counts of the last hidden layer")
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
    # modes / runtime  (ridge dropped: it solves a linear mean-rate read-out)
    p.add_argument("--mode", type=str, default="fullbptt",
                   choices=["fullbptt", "lastbptt"],
                   help="fullbptt trains all 7 weight tensors (Benini); "
                        "lastbptt trains only the read-out w_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    # logging
    p.add_argument("--wandb_project", type=str, default="Benini_replica")
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
# Dataset + manifest   (unchanged from train_snn_shd.py)
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
    """Read dataset metadata from compression_/preprocessing_ manifest."""
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
# Neuron model   (surrogate unchanged from train_snn_shd.py)
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


class BeniniSNN(nn.Module):
    """Three-layer recurrent SNN with a leaky-integrator read-out.

    Mirrors Benini's ``run_snn``: feed-forward spikes propagate through all three
    hidden layers within the same timestep, while each recurrent connection
    (``v1,v2,v3``) feeds back the PREVIOUS timestep's spikes. The read-out is a
    non-spiking 2nd-order leaky integrator over the last hidden layer's spikes.
    """

    def __init__(self, nb_inputs, nb_hidden_1, nb_hidden_2, nb_hidden_3,
                 nb_outputs, alpha, beta, threshold, weight_scale,
                 surrogate_slope):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden_1 = nb_hidden_1
        self.nb_hidden_2 = nb_hidden_2
        self.nb_hidden_3 = nb_hidden_3
        self.nb_outputs = nb_outputs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)

        def param(rows, cols, fan_in):
            w = torch.empty(rows, cols)
            nn.init.normal_(w, 0.0, weight_scale / math.sqrt(fan_in))
            return nn.Parameter(w)

        # feed-forward weights
        self.w1 = param(nb_inputs, nb_hidden_1, nb_inputs)
        self.w2 = param(nb_hidden_1, nb_hidden_2, nb_hidden_1)
        self.w3 = param(nb_hidden_2, nb_hidden_3, nb_hidden_2)
        self.w_out = param(nb_hidden_3, nb_outputs, nb_hidden_3)
        # recurrent weights (one per hidden layer)
        self.v1 = param(nb_hidden_1, nb_hidden_1, nb_hidden_1)
        self.v2 = param(nb_hidden_2, nb_hidden_2, nb_hidden_2)
        self.v3 = param(nb_hidden_3, nb_hidden_3, nb_hidden_3)

    def spike_fn(self, x):
        return SurrGradSpike.apply(x, self.surrogate_slope)

    def hidden_spikes(self, x):
        """Run the 3 recurrent LIF layers; return last-layer spikes ``spk_rec`` [B,T,H3]."""
        x = x.float()
        B, T, C = x.shape
        assert C == self.nb_inputs, (
            f"input channels {C} != model nb_inputs {self.nb_inputs}")
        a, b, thr = self.alpha, self.beta, self.threshold

        # feed-forward drive into layer 1 for every timestep: [B, T, H1]
        h1_from_input = torch.einsum("abc,cd->abd", (x, self.w1))

        z = lambda H: torch.zeros(B, H, device=x.device, dtype=h1_from_input.dtype)
        syn1, mem1, out1 = z(self.nb_hidden_1), z(self.nb_hidden_1), z(self.nb_hidden_1)
        syn2, mem2, out2 = z(self.nb_hidden_2), z(self.nb_hidden_2), z(self.nb_hidden_2)
        syn3, mem3, out3 = z(self.nb_hidden_3), z(self.nb_hidden_3), z(self.nb_hidden_3)

        spk_rec = []
        for t in range(T):
            # layer 1: recurrence uses PREVIOUS out1
            h1 = h1_from_input[:, t] + out1 @ self.v1
            out1 = self.spike_fn(mem1 - thr)
            rst1 = out1.detach()
            new_syn1 = a * syn1 + h1
            new_mem1 = (b * mem1 + syn1) * (1.0 - rst1)

            # layer 2: feed-forward uses NEW out1, recurrence uses PREVIOUS out2
            h2 = out1 @ self.w2 + out2 @ self.v2
            out2 = self.spike_fn(mem2 - thr)
            rst2 = out2.detach()
            new_syn2 = a * syn2 + h2
            new_mem2 = (b * mem2 + syn2) * (1.0 - rst2)

            # layer 3: feed-forward uses NEW out2, recurrence uses PREVIOUS out3
            h3 = out2 @ self.w3 + out3 @ self.v3
            out3 = self.spike_fn(mem3 - thr)
            rst3 = out3.detach()
            new_syn3 = a * syn3 + h3
            new_mem3 = (b * mem3 + syn3) * (1.0 - rst3)

            spk_rec.append(out3)

            mem1, mem2, mem3 = new_mem1, new_mem2, new_mem3
            syn1, syn2, syn3 = new_syn1, new_syn2, new_syn3

        return torch.stack(spk_rec, dim=1)            # [B, T, H3]

    def readout(self, spk_rec):
        """Non-spiking leaky-integrator read-out; return ``out_rec`` [B, T+1, nb_outputs]."""
        B, T, _ = spk_rec.shape
        a, b = self.alpha, self.beta
        h4 = torch.einsum("abc,cd->abd", (spk_rec, self.w_out))   # [B, T, out]
        flt = torch.zeros(B, self.nb_outputs, device=spk_rec.device, dtype=h4.dtype)
        out = torch.zeros_like(flt)
        out_rec = [out]                                # seeded with a zero (Benini)
        for t in range(T):
            new_flt = a * flt + h4[:, t]
            new_out = b * out + flt                    # membrane integrates OLD flt
            flt = new_flt
            out = new_out
            out_rec.append(out)
        return torch.stack(out_rec, dim=1)             # [B, T+1, out]

    def forward(self, x):
        spk_rec = self.hidden_spikes(x)
        return self.readout(spk_rec), spk_rec


# =============================================================================
# Evaluation   (max-over-time decode, like Benini)
# =============================================================================


@torch.no_grad()
def evaluate(model, X, y, batch_size, device):
    """Top-1 accuracy of the max-over-time read-out over a split."""
    model.eval()
    correct, total = 0, 0
    N = X.shape[0]
    for s in range(0, N, batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        out_rec, _ = model(xb)
        m, _ = torch.max(out_rec, dim=1)               # max over time
        pred = m.argmax(1)                             # argmax over output units
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


# =============================================================================
# Training   (BPTT; LogSoftmax + NLLLoss on the max-over-time read-out)
# =============================================================================


def build_optimizer(params, args):
    lr = args.lr if args.lr > 0 else (2e-4 if args.mode == "fullbptt" else 1e-3)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr), lr
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9), lr
    return torch.optim.Adamax(params, lr=lr), lr     # betas default (0.9,0.999)


def train_bptt(model, X_tr, y_tr, X_te, y_te, args, device, wandb_run):
    """fullbptt (all weights) or lastbptt (w_out only) with surrogate gradients."""
    if args.mode == "lastbptt":
        model.requires_grad_(False)
        model.w_out.requires_grad_(True)
    else:  # fullbptt
        model.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters for BPTT")
    optimizer, lr = build_optimizer(params, args)

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
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
            out_rec, spk_rec = model(xb)
            m, _ = torch.max(out_rec, dim=1)           # max over time -> [B, out]
            log_p_y = log_softmax_fn(m)
            # Benini spike regularizers on the last hidden layer (default 0)
            reg_loss = args.reg_1 * torch.sum(spk_rec)
            reg_loss = reg_loss + args.reg_2 * torch.mean(
                torch.sum(torch.sum(spk_rec, dim=0), dim=0) ** 2)
            loss = loss_fn(log_p_y, yb) + reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            n = int(yb.numel())
            run_loss += float(loss.item()) * n
            run_correct += float((m.argmax(1) == yb).sum().item())
            seen += n
            spk_sum += float(spk_rec.detach().sum().item())
            spk_sites += float(spk_rec.numel())

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

    # ---- data (same split as train_snn_shd.py: the dataset's own files) ----
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
          f"nb_inputs={nb_inputs} nb_steps={nb_steps} classes_seen={n_classes_seen}")

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

    # ---- Benini architecture: nb_hidden -> //2 -> //2 ----
    nb_hidden_1 = args.nb_hidden
    nb_hidden_2 = nb_hidden_1 // 2
    nb_hidden_3 = nb_hidden_2 // 2
    print(f"architecture: {nb_inputs} -> {nb_hidden_1} -> {nb_hidden_2} -> "
          f"{nb_hidden_3} -> {args.nb_outputs}")

    model = BeniniSNN(
        nb_inputs, nb_hidden_1, nb_hidden_2, nb_hidden_3, args.nb_outputs,
        alpha, beta, args.threshold, args.weight_scale,
        args.surrogate_slope).to(device)

    # ---- W&B ----
    wandb_run = init_wandb(args, manifest, nb_inputs, nb_steps, dt_ms, dt_source,
                           alpha, beta, nb_hidden_1, nb_hidden_2, nb_hidden_3)

    # ---- train ----
    result = train_bptt(model, X_tr, y_tr, X_te, y_te, args, device, wandb_run)

    print(f"=== {args.mode} ===  train_acc={result['train_acc']:.4f}  "
          f"test_acc={result['test_acc']:.4f}  "
          f"best_test_acc={result['best_test_acc']:.4f}@{result['best_epoch']}")

    if wandb_run is not None:
        for k, v in result.items():
            wandb_run.summary[k] = v
        wandb_run.finish()


def init_wandb(args, manifest, nb_inputs, nb_steps, dt_ms, dt_source, alpha,
               beta, nb_hidden_1, nb_hidden_2, nb_hidden_3):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; running without logging.")
        return None
    config = {
        "model": "benini_replica_3layer",
        "mode": args.mode,
        "nb_inputs": nb_inputs,
        "input_channels": nb_inputs,
        "nb_hidden_1": nb_hidden_1,
        "nb_hidden_2": nb_hidden_2,
        "nb_hidden_3": nb_hidden_3,
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
        "reg_1": args.reg_1,
        "reg_2": args.reg_2,
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
    }
    return wandb.init(project=args.wandb_project, name=args.wandb_name,
                      entity=args.wandb_entity, mode=args.wandb_mode,
                      config=config,
                      tags=["shd", "benini_replica", args.mode,
                            str(manifest["compression_method"]),
                            f"C{nb_inputs}", f"dt{dt_ms}ms"])


if __name__ == "__main__":
    main()
