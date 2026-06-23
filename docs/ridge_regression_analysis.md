# Closed-form Ridge regression in the reference, and how to adapt it

This document explains, step by step, how the reference project
`references/Fabrizio-Spiking-Compressed-CL` implements Ridge regression as the
read-out of a recurrent spiking neural network (SNN), and then how that
implementation is adapted and improved in `train_ridge_snn.py` for the task of
training a second-order LIF reservoir on the preprocessed SHD `.npz` datasets
produced by `dataset_preprocessing/shd_channel_compression.ipynb` and
`dataset_preprocessing/shd_build_dataset.ipynb`.

All reference line numbers below point to
`references/Fabrizio-Spiking-Compressed-CL/Spiking-Compressed-Continual-Learning copy/train.py`.


## 1. The big picture

A reservoir / echo-state-style SNN classifier has three layers:

```
input spikes  -->  recurrent hidden layer (fixed, random)  -->  linear read-out
   [C]                       [H]                                     [20]
```

The hidden layer is a *fixed* random recurrent network of spiking neurons. It is
never trained. Its only job is to project the input spike trains into a
high-dimensional dynamical feature space. The *only* trainable part is the
linear read-out `w_out`, and because the read-out is linear and the loss is the
squared error, its optimal weights have a **closed-form solution** (Ridge
regression). No gradient descent, no epochs, no back-propagation through time.

This is attractive because:

- It is fast: one forward pass over the data to collect features, then one
  linear solve.
- It is deterministic given the random seed and the regularisation strength.
- It isolates the contribution of the reservoir dynamics (the spectral radius,
  the time constants) from the optimisation procedure.


## 2. The neuron model: hand-written second-order LIF

The hidden layer uses a **second-order leaky integrate-and-fire (LIF)** neuron,
written by hand (no `snnTorch`). "Second order" means each neuron carries *two*
state variables that form a cascade of two leaky integrators:

- a **synaptic current** `syn` (fast), and
- a **membrane potential** `mem` (slow), driven by `syn`.

The recurrence (reference `hidden_pass`, lines ~192-210) is, per timestep `t`:

```python
up   = h_in0[:, t] if i == 0 else outs[i - 1] @ Ws[i]   # feed-forward drive
h    = up + outs[i] @ Vs[i]                             # + recurrent drive
outs[i] = spike_fn(mems[i] - 1.0)                       # spike if mem > threshold (=1)
rst  = outs[i].detach()                                 # reset signal
new_syn  = alphas[i] * syns[i] + h                      # synapse leaks with alpha
mems[i]  = (betas[i] * mems[i] + syns[i]) * (1.0 - rst) # membrane leaks with beta, reset
syns[i]  = new_syn
```

with decay constants derived from the time constants and the simulation step:

```
alpha = exp(-time_step / tau_syn)   # synaptic decay  (tau_syn ~ 5 ms)
beta  = exp(-time_step / tau_mem)   # membrane decay   (tau_mem ~ 10 ms)
```

Key points:

- The threshold is fixed at `1.0`; a spike is emitted when `mem > 1`.
- On a spike, the membrane is multiplicatively reset to 0 (`* (1 - rst)`), while
  the synaptic current is *not* reset.
- `Ws[0]` (shape `[C, H]`) is the input projection; `Vs[0]` (shape `[H, H]`) is
  the recurrent matrix. Both are random and frozen.
- `spike_fn` is a surrogate-gradient Heaviside. In closed-form mode the gradient
  is never used (everything runs under `torch.no_grad()`), so the surrogate is
  irrelevant here; only the forward `(x > 0)` matters.

### Spectral-radius rescaling of the recurrent matrix

A reservoir only produces useful, stable-yet-rich dynamics when the recurrent
matrix sits near the "edge of chaos". The reference enforces this by rescaling
the recurrent matrix `V` so that its largest-magnitude eigenvalue (its spectral
radius) equals a chosen value (reference lines ~167-172):

```python
current = torch.linalg.eigvals(v.detach().cpu()).abs().max().item()
v.mul_(args.spectral_radius / current)
```

The default in the sweep scripts is `0.8`-`0.95`; values near (or slightly
above) `1.0` push the reservoir toward the edge of chaos.


## 3. Turning spike trains into a static feature vector

Ridge regression needs one fixed-length feature vector per sample. The reference
pools the hidden spikes over time (reference `aggregate`, lines ~213-216):

```python
def aggregate(spk_rec, mem_rec):
    if args.aggregation == "mean_t_spk":
        return spk_rec.mean(dim=1)   # average each hidden neuron's spikes over time
    return mem_rec.mean(dim=1)
```

So a sample whose hidden spike tensor is `spk_rec` of shape `[B, T, H]` becomes a
feature vector of shape `[B, H]`: **the per-neuron average firing rate over the
`T` timesteps**. This is exactly the "average over timesteps of each hidden
neuron's spikes" the task calls for.


## 4. The closed-form Ridge solve, step by step

The whole closed-form path is reference lines ~284-320. Walk through it with
shapes (`N` = number of samples, `H` = hidden size, `C_out` = 20 classes).

### Step 4.1 - Freeze the reservoir

```python
for w in Ws + Vs:
    w.requires_grad_(False)
```

### Step 4.2 - Collect features for the whole split

```python
def collect_features(batches):
    feats, ys = [], []
    with torch.no_grad():
        for X, y in batches:
            spk_rec, mem_rec = hidden_pass(X.to_dense().to(device))
            feats.append(aggregate(spk_rec, mem_rec))   # [B, H]
            ys.append(y.to(device))
    return torch.cat(feats), torch.cat(ys)

Phi_tr, y_tr = collect_features(train_batches)   # Phi_tr: [N_tr, H]
Phi_te, y_te = collect_features(test_batches)    # Phi_te: [N_te, H]
```

### Step 4.3 - One-hot the targets

```python
Y_tr = nn.functional.one_hot(y_tr, NB_OUTPUTS).float()  # [N_tr, 20]
Y_te = nn.functional.one_hot(y_te, NB_OUTPUTS).float()  # [N_te, 20]
```

The classification problem is cast as a **multi-output linear regression**: fit
`Phi @ w_out ~= Y`, where each column of `Y` is the indicator of one class.

### Step 4.4 - Build and solve the normal equations

```python
H_last = layer_sizes[-1]                                  # H
A = (Phi_tr.T @ Phi_tr) + args.ridge_lambda * torch.eye(H_last)  # [H, H]
B_rhs = (Phi_tr.T @ Y_tr)                                 # [H, 20]
w_out_solved = torch.linalg.solve(A, B_rhs)               # [H, 20]
```

This is the Ridge (Tikhonov-regularised) least-squares solution. The objective is

```
min_w  || Phi w - Y ||_F^2  +  lambda * || w ||_F^2
```

whose closed-form minimiser is

```
w = (Phi^T Phi + lambda I)^{-1} Phi^T Y
```

The `lambda * I` term (the "ridge") makes `A` invertible even when `Phi^T Phi` is
rank-deficient (e.g. dead neurons, more features than samples) and shrinks the
weights to reduce over-fitting.

### Step 4.5 - Evaluate

```python
logits_tr = Phi_tr @ w_out_solved          # [N_tr, 20]
logits_te = Phi_te @ w_out_solved          # [N_te, 20]
train_acc = (logits_tr.argmax(1) == y_tr).float().mean().item()
test_acc  = (logits_te.argmax(1) == y_te).float().mean().item()
train_loss = ((logits_tr - Y_tr) ** 2).mean().item()   # MSE
test_loss  = ((logits_te - Y_te) ** 2).mean().item()
```

Prediction is `argmax` over the 20 regression outputs. Both train and test
accuracy plus the MSE losses are logged to Weights & Biases (lines ~318-319).

### Shape summary

| object      | shape        | meaning                                  |
|-------------|--------------|------------------------------------------|
| `X`         | `[B, T, C]`  | input spikes per batch                   |
| `spk_rec`   | `[B, T, H]`  | hidden spikes over time                  |
| `Phi`       | `[N, H]`     | time-averaged hidden firing rates        |
| `Y`         | `[N, 20]`    | one-hot targets                          |
| `A`         | `[H, H]`     | Gram matrix `Phi^T Phi + lambda I`       |
| `B_rhs`     | `[H, 20]`    | `Phi^T Y`                                |
| `w_out`     | `[H, 20]`    | read-out weights                         |
| `logits`    | `[N, 20]`    | `Phi @ w_out`                            |


## 5. How the new task differs from the reference

The reference loads raw SHD from `.h5.gz`, re-bins spikes on the fly into a
sparse COO tensor with a fixed `NB_INPUTS = 700`, and supports three training
modes. The new task is narrower and uses a different data source:

- **Input** is a preprocessed `.npz` dataset (`train.npz` / `test.npz`) with a
  dense array `X` of shape `[N, T, C]` (`uint8`/`uint16`), `y` `[N]`, and
  `speaker` `[N]`. The number of input neurons `C` and timesteps `T` are read
  directly from `X.shape`, **not** hard-coded to 700/100, because channel
  compression changes `C` (e.g. 350, 175, ...).
- **Architecture** is fixed to the single recurrent hidden layer of 1000
  neurons, output of 20, closed-form Ridge only.
- **Spectral radius** defaults to `1.0` and is the primary sweepable knob.
- **Device** is auto-selected (CUDA if available), since a hand-written LIF loop
  over `T` steps with `H = 1000` recurrent neurons is far faster on GPU.
- Everything is driven from the command line so it can be scripted across many
  dataset paths.


## 6. Improvements implemented in `train_ridge_snn.py`

The new script keeps the same mathematical core but addresses several
weaknesses of the reference closed-form path.

### 6.1 Bias / intercept term

The reference fits `Phi @ w_out` with no intercept. Because the features are
non-negative firing rates, the best-fit hyperplane generally does not pass
through the origin. The new script optionally augments `Phi` with a constant
column of ones, `Phi -> [Phi | 1]` of shape `[N, H + 1]`, so `w_out` has shape
`[H + 1, 20]` and the last row is the per-class bias. The ridge penalty is *not*
applied to the bias row (its diagonal entry in `lambda I` is set to 0), which is
the standard, correct way to regularise an intercept.

### 6.2 Optional feature standardisation

Ridge is not scale-invariant: the `lambda || w ||^2` penalty treats every feature
on the same footing, so features with larger variance are implicitly penalised
less. Spike-rate features can have very different variances (some neurons fire a
lot, many barely fire). The script can z-score each feature column using the
*training* mean and standard deviation (with a small epsilon for dead neurons)
and apply the same transform to the test set, which often improves conditioning
and accuracy.

### 6.3 Numerical stability

`torch.linalg.solve(Phi^T Phi + lambda I, ...)` squares the condition number of
`Phi`. The new script:

- Accumulates and solves in **float64** even when the SNN runs in float32.
- Uses a Cholesky-based solver (`torch.linalg.cholesky` + `cholesky_solve`) on
  the symmetric positive-definite system, with an automatic fallback to
  `torch.linalg.solve` if the Cholesky factorisation fails. This is both faster
  and more stable than a generic solve.

### 6.4 Gram accumulation (scalability)

The reference concatenates every per-batch feature vector into one big `[N, H]`
matrix before the solve. For large `N` and `H = 1000` this can be wasteful. The
new script accumulates the Gram matrix incrementally:

```
A += Phi_b^T @ Phi_b      # [H+1, H+1]
B += Phi_b^T @ Y_b        # [H+1, 20]
```

so peak memory for the solve is `O(H^2)` regardless of `N`. (The feature matrix
is still kept for computing accuracy/MSE; if needed it could be streamed too.)

### 6.5 Spectral radius as a first-class, swept hyperparameter

The rescaling is generalised to the single recurrent matrix `W_rec`, defaults to
`1.0`, and both the *measured* spectral radius before rescaling and the target
after rescaling are logged to W&B (`spectral_radius_pre`, `spectral_radius`).
Setting `--spectral_radius <= 0` disables rescaling.

### 6.6 Richer, fully-logged metrics

The full config plus `train_acc`, `test_acc`, `train_loss`, `test_loss`,
`spectral_radius_pre`, `nb_inputs`, `nb_steps`, `fit_seconds`, and the mean
hidden firing rate are logged to W&B, so a terminal sweep over datasets and
spectral radii produces directly comparable runs.

### 6.7 Possible further work (not implemented by default)

- **`lambda` selection by efficient LOOCV.** For ridge there is a closed-form
  leave-one-out error using the hat matrix `Phi (Phi^T Phi + lambda I)^{-1} Phi^T`,
  so the regularisation strength could be tuned without a separate validation
  split. A simple `lambda`-grid hook is the pragmatic alternative.
- **Richer temporal features** (e.g. early/late time windows, or membrane mean)
  would add information the single time-average discards, at the cost of leaving
  the strict "mean over timesteps of spikes" definition.


## 7. End-to-end logic and dimension check

For a dataset with `C` input channels, `T` timesteps, `N` samples, `H = 1000`
hidden neurons and 20 classes:

1. Load `X` `[N, T, C]`, `y` `[N]`; assert `y in [0, 20)`.
2. `W_in` `[C, H]`, `W_rec` `[H, H]`; rescale `W_rec` to spectral radius `rho`.
3. Per batch `[B, T, C]` -> second-order LIF -> hidden spikes `[B, T, H]` ->
   mean over time -> `Phi_b` `[B, H]`.
4. (optional standardise) -> (optional bias) `Phi` `[N, H(+1)]`.
5. `A` `[H(+1), H(+1)]`, `B` `[H(+1), 20]`, solve `w_out` `[H(+1), 20]`.
6. `logits = Phi @ w_out` `[N, 20]`; `argmax` -> prediction; compare to `y`.
7. Log train/test accuracy and MSE to W&B.

The script asserts these shapes at runtime and includes a `--limit` smoke-test
flag to validate the whole pipeline on a small subset.
