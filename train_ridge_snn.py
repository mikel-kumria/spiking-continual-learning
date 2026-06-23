"""Reservoir SNN + closed-form Ridge read-out on preprocessed SHD datasets.

Pipeline
--------
1. Load a preprocessed SHD dataset directory containing ``train.npz`` and
   ``test.npz`` (as produced by ``dataset_preprocessing/shd_build_dataset.ipynb``
   or ``dataset_preprocessing/shd_channel_compression.ipynb``). Each ``.npz`` holds
   a dense array ``X`` of shape ``[N, T, C]`` (binary/count spikes), ``y`` ``[N]``
   and ``speaker`` ``[N]``.
2. Build a fixed random reservoir: an input projection ``W_in`` ``[C, H]`` and a
   recurrent matrix ``W_rec`` ``[H, H]`` rescaled to a chosen spectral radius.
3. Run a hand-written second-order LIF (synaptic current + membrane) over the
   ``T`` timesteps; pool hidden spikes by averaging over time to get a feature
   vector ``Phi`` ``[N, H]``.
4. Fit a linear read-out ``Phi @ w_out ~= one_hot(y)`` in closed form (Ridge),
   accumulating the Gram matrices in float64.
5. Report train/test accuracy and MSE; log everything to Weights & Biases.

The recurrent reservoir is never trained; only the linear read-out is solved.

Example
-------
    python train_ridge_snn.py --data_dir /path/to/SHD_PREPROCESSED/... \
        --nb_hidden 1000 --spectral_radius 1.0 --ridge_lambda 1.0 \
        --wandb_project shd-reservoir
"""
import argparse
import os
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--data_dir", type=str, required=True,
                   help="directory containing train.npz and test.npz")
    p.add_argument("--train_file", type=str, default="train.npz")
    p.add_argument("--test_file", type=str, default="test.npz")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, use only the first N samples of each split (smoke test)")
    # architecture
    p.add_argument("--nb_hidden", type=int, default=1000)
    p.add_argument("--nb_outputs", type=int, default=20)
    # reservoir / neuron model
    p.add_argument("--spectral_radius", type=float, default=1.0,
                   help="rescale W_rec to this spectral radius; <=0 disables rescaling")
    p.add_argument("--tau_mem", type=float, default=10e-3)
    p.add_argument("--tau_syn", type=float, default=5e-3)
    p.add_argument("--time_step", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--tau_mode", choices=["homogeneous", "heterogeneous"],
                   default="homogeneous",
                   help="heterogeneous: per-neuron tau sampled log-uniform "
                        "(tau_mem in [5,100]ms, tau_syn in [5,50]ms)")
    # ridge read-out
    p.add_argument("--ridge_lambda", type=float, default=1.0)
    p.add_argument("--bias", dest="bias", action="store_true", default=True,
                   help="append a bias/intercept column to the features (default on)")
    p.add_argument("--no-bias", dest="bias", action="store_false")
    p.add_argument("--standardize", dest="standardize", action="store_true",
                   default=False, help="z-score features using train statistics")
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    # runtime
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    # logging
    p.add_argument("--wandb_project", type=str, default="shd-reservoir")
    p.add_argument("--wandb_name", type=str, default=None)
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


def load_split(path, limit=0):
    """Load one ``.npz`` split -> (X float32 [N,T,C], y long [N])."""
    if not os.path.isfile(path):
        raise SystemExit(f"dataset file not found: {path}")
    d = np.load(path)
    if "X" not in d or "y" not in d:
        raise SystemExit(f"{path} must contain arrays 'X' and 'y' (found {list(d.keys())})")
    X = d["X"]
    y = d["y"]
    if limit and limit > 0:
        X, y = X[:limit], y[:limit]
    assert X.ndim == 3, f"expected X with shape [N,T,C], got {X.shape}"
    assert X.shape[0] == y.shape[0], f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}"
    X = torch.from_numpy(np.ascontiguousarray(X)).float()
    y = torch.from_numpy(np.ascontiguousarray(y)).long()
    return X, y


def build_reservoir(nb_inputs, nb_hidden, weight_scale, spectral_radius,
                    device, generator):
    """Create fixed input and recurrent matrices.

    Returns (W_in [C,H], W_rec [H,H], measured_spectral_radius_pre)."""
    W_in = torch.empty((nb_inputs, nb_hidden), device=device)
    torch.nn.init.normal_(W_in, mean=0.0, std=weight_scale / np.sqrt(nb_inputs),
                          generator=generator)
    W_rec = torch.empty((nb_hidden, nb_hidden), device=device)
    torch.nn.init.normal_(W_rec, mean=0.0, std=weight_scale / np.sqrt(nb_hidden),
                          generator=generator)

    sr_pre = torch.linalg.eigvals(W_rec.detach().cpu()).abs().max().item()
    if spectral_radius > 0:
        if sr_pre <= 0:
            raise SystemExit("measured spectral radius is 0; cannot rescale")
        W_rec.mul_(spectral_radius / sr_pre)
    return W_in, W_rec, sr_pre


def make_tau_vectors(nb_hidden, tau_mode, tau_mem, tau_syn, time_step, device, generator):
    """Per-neuron membrane/synaptic decay constants -> (beta [H], alpha [H])."""
    if tau_mode == "heterogeneous":
        tm = torch.empty(nb_hidden, device=device).uniform_(
            float(np.log(5e-3)), float(np.log(100e-3)), generator=generator).exp()
        ts = torch.empty(nb_hidden, device=device).uniform_(
            float(np.log(5e-3)), float(np.log(50e-3)), generator=generator).exp()
    else:
        tm = torch.full((nb_hidden,), tau_mem, device=device)
        ts = torch.full((nb_hidden,), tau_syn, device=device)
    beta = torch.exp(-time_step / tm)
    alpha = torch.exp(-time_step / ts)
    return beta, alpha


def hidden_features(X, W_in, W_rec, beta, alpha, threshold):
    """Run the second-order LIF reservoir on a batch and return time-averaged spikes.

    X: [B, T, C] -> Phi: [B, H] (mean over time of hidden spikes)."""
    B, T, _ = X.shape
    H = W_in.shape[1]
    device = W_in.device
    # pre-compute feed-forward drive for all timesteps: [B, T, H]
    h_in = torch.einsum("btc,ch->bth", X, W_in)

    syn = torch.zeros((B, H), device=device)
    mem = torch.zeros((B, H), device=device)
    out = torch.zeros((B, H), device=device)
    spk_sum = torch.zeros((B, H), device=device)

    for t in range(T):
        drive = h_in[:, t] + out @ W_rec          # feed-forward + recurrent
        out = (mem > threshold).float()           # spike (forward Heaviside)
        rst = out                                 # reset signal (no grad needed)
        new_syn = alpha * syn + drive
        mem = (beta * mem + syn) * (1.0 - rst)
        syn = new_syn
        spk_sum = spk_sum + out

    return spk_sum / T                            # [B, H]


def collect_features(X, y, W_in, W_rec, beta, alpha, threshold, batch_size):
    """Run the reservoir over a full split -> (Phi [N,H], y [N])."""
    feats = []
    N = X.shape[0]
    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = X[start:start + batch_size].to(W_in.device)
            feats.append(hidden_features(xb, W_in, W_rec, beta, alpha, threshold))
    Phi = torch.cat(feats, dim=0)
    assert Phi.shape == (N, W_in.shape[1]), f"Phi shape {Phi.shape} != {(N, W_in.shape[1])}"
    return Phi


def ridge_closed_form(Phi_tr, Y_tr, ridge_lambda, use_bias):
    """Solve (Phi^T Phi + lambda I) w = Phi^T Y in float64.

    Returns w_out [F, C_out] where F = H (+1 if bias). The bias row is not
    penalised. Uses a Cholesky solve with a generic-solve fallback."""
    Phi = Phi_tr.double()
    Y = Y_tr.double()
    N, H = Phi.shape

    if use_bias:
        ones = torch.ones((N, 1), dtype=Phi.dtype, device=Phi.device)
        Phi = torch.cat([Phi, ones], dim=1)          # [N, H+1]
    F = Phi.shape[1]

    A = Phi.T @ Phi                                  # [F, F]
    B = Phi.T @ Y                                    # [F, C_out]

    reg = torch.full((F,), ridge_lambda, dtype=Phi.dtype, device=Phi.device)
    if use_bias:
        reg[-1] = 0.0                                # do not regularise the intercept
    A = A + torch.diag(reg)

    try:
        L = torch.linalg.cholesky(A)
        w_out = torch.cholesky_solve(B, L)
    except RuntimeError:
        w_out = torch.linalg.solve(A, B)

    assert w_out.shape == (F, Y.shape[1]), f"w_out shape {w_out.shape}"
    return w_out


def apply_features(Phi, mu, sigma, use_bias):
    """Standardise (optional) and append bias column to match the solver layout."""
    out = Phi.double()
    if mu is not None:
        out = (out - mu) / sigma
    if use_bias:
        ones = torch.ones((out.shape[0], 1), dtype=out.dtype, device=out.device)
        out = torch.cat([out, ones], dim=1)
    return out


def accuracy_and_mse(Phi, y, w_out, mu, sigma, use_bias, nb_outputs):
    feats = apply_features(Phi, mu, sigma, use_bias)          # [N, F]
    logits = feats @ w_out                                    # [N, C_out]
    assert logits.shape == (Phi.shape[0], nb_outputs), f"logits shape {logits.shape}"
    Y = torch.nn.functional.one_hot(y, nb_outputs).double().to(logits.device)
    mse = ((logits - Y) ** 2).mean().item()
    acc = (logits.argmax(1) == y.to(logits.device)).float().mean().item()
    return acc, mse


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"device: {device}")

    # ---- data ----
    train_path = os.path.join(os.path.expanduser(args.data_dir), args.train_file)
    test_path = os.path.join(os.path.expanduser(args.data_dir), args.test_file)
    X_tr, y_tr = load_split(train_path, args.limit)
    X_te, y_te = load_split(test_path, args.limit)

    nb_inputs = X_tr.shape[2]
    nb_steps = X_tr.shape[1]
    assert X_te.shape[2] == nb_inputs, (
        f"train/test channel mismatch: {nb_inputs} vs {X_te.shape[2]}")
    assert X_te.shape[1] == nb_steps, (
        f"train/test timestep mismatch: {nb_steps} vs {X_te.shape[1]}")
    n_classes_seen = int(max(y_tr.max().item(), y_te.max().item())) + 1
    assert n_classes_seen <= args.nb_outputs, (
        f"label {n_classes_seen - 1} >= nb_outputs {args.nb_outputs}")
    print(f"train X: {tuple(X_tr.shape)}  test X: {tuple(X_te.shape)}  "
          f"nb_inputs={nb_inputs} nb_steps={nb_steps}")

    # ---- reservoir ----
    W_in, W_rec, sr_pre = build_reservoir(
        nb_inputs, args.nb_hidden, args.weight_scale, args.spectral_radius,
        device, generator)
    beta, alpha = make_tau_vectors(
        args.nb_hidden, args.tau_mode, args.tau_mem, args.tau_syn,
        args.time_step, device, generator)
    print(f"W_rec spectral radius: {sr_pre:.3f} -> "
          f"{args.spectral_radius if args.spectral_radius > 0 else sr_pre:.3f}")

    # ---- features ----
    t0 = time.time()
    Phi_tr = collect_features(X_tr, y_tr, W_in, W_rec, beta, alpha,
                              args.threshold, args.batch_size)
    Phi_te = collect_features(X_te, y_te, W_in, W_rec, beta, alpha,
                              args.threshold, args.batch_size)
    feat_seconds = time.time() - t0
    mean_rate = Phi_tr.mean().item()
    print(f"features: Phi_tr={tuple(Phi_tr.shape)} Phi_te={tuple(Phi_te.shape)} "
          f"mean_rate={mean_rate:.4f} ({feat_seconds:.1f}s)")

    # ---- optional standardisation (train statistics) ----
    if args.standardize:
        mu = Phi_tr.double().mean(dim=0, keepdim=True)
        sigma = Phi_tr.double().std(dim=0, keepdim=True)
        sigma = torch.where(sigma > 1e-8, sigma, torch.ones_like(sigma))
    else:
        mu, sigma = None, None

    # ---- closed-form ridge ----
    t1 = time.time()
    Phi_tr_std = Phi_tr.double()
    if args.standardize:
        Phi_tr_std = (Phi_tr_std - mu) / sigma
    Y_tr = torch.nn.functional.one_hot(y_tr, args.nb_outputs).double().to(device)
    w_out = ridge_closed_form(Phi_tr_std, Y_tr, args.ridge_lambda, args.bias)
    fit_seconds = time.time() - t1

    # ---- evaluation ----
    train_acc, train_loss = accuracy_and_mse(
        Phi_tr, y_tr, w_out, mu, sigma, args.bias, args.nb_outputs)
    test_acc, test_loss = accuracy_and_mse(
        Phi_te, y_te, w_out, mu, sigma, args.bias, args.nb_outputs)

    print(f"closed_form: train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
          f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
          f"(fit {fit_seconds:.2f}s)")

    # ---- logging ----
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name,
               mode=args.wandb_mode, config=vars(args))
    wandb.log({
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "spectral_radius_pre": sr_pre,
        "spectral_radius": args.spectral_radius,
        "nb_inputs": nb_inputs,
        "nb_steps": nb_steps,
        "nb_train": int(X_tr.shape[0]),
        "nb_test": int(X_te.shape[0]),
        "mean_hidden_rate": mean_rate,
        "feat_seconds": feat_seconds,
        "fit_seconds": fit_seconds,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
