# Technical report — `pretrain_snn_shd.py` (SHD continual-learning, Stage 1)

This report explains, step by step, everything `pretrain_snn_shd.py` does, with
tensor/vector dimensions, the mathematics behind each step, and the exact data
each operation touches. It is written so the experiment can be reproduced from
this document alone. It is grounded strictly on the code in
`pretrain_snn_shd.py`, `snn_shd_common.py` (imported as `C`) and
`train_snn_shd.py` (imported as `T`). Places where the code is logically or
mathematically questionable are flagged in **§10 Issues & improvements**.

Throughout, the **worked example** is the command from the script docstring:

```
--removed-class 10 --dataset-binning-ms 14 --n-compressed-channels 70 \
--channel-compression-method or_pool --mode ridge --nb-hidden 1000 \
--batch-size 64 --max-time 1.4 (default)
```

---

## 0. What the script produces (one-paragraph summary)

It is **Stage 1** of a two-stage class-incremental SHD pipeline. It (1)
**preprocesses** the raw Spiking Heidelberg Digits (SHD) HDF5 files into a
class-incremental dataset where one class (`--removed-class`) is held out, and
(2) **pretrains** a recurrent spiking neural network (SNN) on the remaining 19
classes. The output is a single self-contained run folder containing the four
`.npz` data splits, a preprocessing manifest, a model checkpoint, a config dump,
and a metrics dump. Stage 2 (`class_incremental_snn_shd.py`) later consumes that
folder to learn the held-out class incrementally.

```
outputs/shd_pretraining/<run_name>/
  config.json                       # the argparse namespace
  preprocessing_manifest.json       # full data provenance
  metrics.json                      # pretraining accuracies + init diagnostics
  dataset/
    pretrain_train.npz              # 19 active classes, 80%
    pretrain_test.npz               # 19 active classes, 20%
    continual_train.npz             # removed class only, 80%  (used by Stage 2)
    continual_test.npz              # removed class only, 20%
  checkpoints/pretrained_model.pt   # weights + architecture + provenance
  logs/                             # created, not written to by this script
```

---

## 1. Dataset and the key parameters

**SHD** = Spiking Heidelberg Digits: spoken digits 0–9 in English and German,
**`NUM_CLASSES = 20`** classes (`snn_shd_common.py:41`). Each utterance was
passed through an artificial cochlea producing **700 input channels**
(`DEFAULT_NB_INPUTS = 700`). The raw data is event-based: per utterance, a list
of spike **times** (seconds) and the **unit** (channel index 0–699) of each
spike.

Parameters that define the experiment (defaults from `parse_args`,
`pretrain_snn_shd.py:772`):

| Group | Arg | Default | Meaning |
|---|---|---|---|
| Data | `--train-h5` / `--test-h5` | `../../data/SHD_RAW/shd_{train,test}.h5` | Raw HDF5 inputs |
| Data | `--removed-class` | **required** | Held-out class 0–19 (excluded from pretraining) |
| Data | `--dataset-binning-ms` | **required** | Temporal bin width Δt (ms); also sets neuron α/β |
| Data | `--max-time` | `1.4` | Window length (s); spikes at t ≥ this are dropped |
| Data | `--nb-inputs` | `700` | Native channels before compression |
| Data | `--n-compressed-channels` | `70` | Channels after compression (must divide 700) |
| Data | `--channel-compression-method` | `or_pool` | `or_pool`/`conditional_or`/`graded`/`bernoulli` |
| Data | `--condition-or` | `1` | Group fires if ≥ this many channel-spikes land in it |
| Data | `--merge-train-test` | `True` | Merge official train+test, then re-split |
| Data | `--pretrain-{train,test}-fraction` | `0.80 / 0.20` | Split of the 19-class pool |
| Data | `--continual-{train,test}-fraction` | `0.80 / 0.20` | Split of the removed-class pool |
| Data | `--seed` | `42` | Seeds splits, init, batch order |
| Model | `--mode` | `ridge` | `ridge` (closed-form) or `fullbptt` |
| Model | `--nb-hidden` | `1000` | Recurrent hidden neurons H |
| Model | `--nb-outputs` | `20` | Output neurons (≥ NUM_CLASSES) |
| Model | `--tau-mem-ms` / `--tau-syn-ms` | `10.0 / 5.0` | Hidden-neuron membrane / synaptic time constants |
| Model | `--threshold` | `1.0` | Hidden spike threshold |
| Model | `--output-threshold` / `--output-gain` | `1.0 / 1.0` | IF output threshold / input gain |
| Model | `--weight-scale` | `0.2` | Std scaling of weight init |
| Model | `--surrogate-slope` | `100.0` | Fast-sigmoid surrogate-gradient slope |
| Model | `--init-spectral-radius` | `1.0` | Target spectral radius of W_rec before firing-rate tuning |
| Model | `--firing-low` / `--firing-high` | `0.02 / 0.20` | Sane firing-rate window (hidden, and output for bptt) |
| Train | `--ridge-lambda` | `1.0` | Ridge regulariser λ |
| Train | `--nb-epochs` | `200` | BPTT epochs |
| Train | `--batch-size` | `64` | Batch size (also used for chunked eval) |
| Train | `--lr` | `0.0`→`2e-4` | BPTT learning rate (≤0 ⇒ 2e-4) |
| Train | `--optimizer` | `adamax` | `adamax`/`adam`/`sgd(momentum 0.9)` |
| Train | `--grad-clip` | `0.0` | Max grad-norm (0 = off) |
| Train | `--limit` | `0` | Use first N of each split (smoke test) |
| Runtime | `--device` | `auto` | `auto`(cuda→mps→cpu)/`cpu`/`cuda`/`mps` |
| Logging | `--wandb-mode` | `disabled` | `online`/`offline`/`disabled` |

---

## 2. Stage 1a — Preprocessing (`run_preprocessing`, `pretrain_snn_shd.py:284`)

### 2.1 Load the raw event pool (`read_h5_pool`, line 98)

Each HDF5 file is read into an `EventPool` dataclass (line 87):

- `times`: Python list of length N; element i is a `float64` array `[n_spikes_i]`
  of firing times (seconds) for utterance i.
- `units`: list of length N; element i is an `int64` array `[n_spikes_i]` of
  channel indices (0–699), aligned 1:1 with `times[i]`.
- `labels`: `int64 [N]` class label per utterance.
- `speakers`: `int64 [N]` speaker id (or all `-1` if `extra/speaker` absent).

HDF5 schema assumed: `f["spikes"]["times"]`, `f["spikes"]["units"]`,
`f["labels"]`, optional `f["extra"]["speaker"]`. Lengths must all match (line
111) or it raises.

### 2.2 Optionally merge official train + test (`merge_pools`, line 116)

With `--merge-train-test` (default **True**), the official train and test pools
are concatenated into one pool, **then** re-split later (§2.6). The manifest
records this and warns: *“Merged official train+test … NOT the SHD benchmark
test set. Accuracies are not comparable to the paper.”* (line 399). This is a
deliberate trade-off (more data per class) at the cost of benchmark
comparability — see §10.

> A `synthetic_pool` (line 130) can fabricate a tiny all-class pool for smoke
> tests when `--synthetic-samples-per-class > 0`; the default 0 means real SHD.

After loading, the pool has `len(pool) = N_total` utterances (≈ all of SHD when
merged).

### 2.3 Temporal binning (`compute_nb_steps` line 166, `event_to_dense_np` line 174)

The continuous event stream is discretised onto a regular time grid of width
Δt = `dataset_binning_ms / 1000` seconds.

Number of time steps:

```
nb_steps = max(1, ceil(max_time_s / Δt))
```

*Worked example:* Δt = 14 ms = 0.014 s, max_time = 1.4 s ⇒ `nb_steps =
ceil(1.4/0.014) = 100`.

Per utterance, `event_to_dense_np` builds a dense binary raster
`x ∈ {0,1}^{nb_steps × nb_inputs}` (uint8, **channel is the last axis**):

1. Drop spikes with `t ≥ max_time_s` (`keep = t < max_time_s`) — late spikes are
   **truncated**, not wrapped.
2. Bin index `b = floor(t / Δt)`, clamped to `[0, nb_steps-1]`.
3. `x[b, u] = 1` — **assignment**, not increment: multiple spikes in the same
   (bin, channel) collapse to a single 1. The raster is therefore **binary**.
4. Guards: units must be in `[0, nb_inputs)`; times non-negative.

Per-sample dense shape: **`[T=nb_steps, C=nb_inputs] = [100, 700]`** (worked
example, pre-compression).

Mathematical note: this is a fixed-width, floor-binned, OR-over-time rasterizer.
Spike-count information within a bin is discarded (binary). Time axis is never
touched again after this.

### 2.4 Partition by class (`select_indices_by_class`, line 202)

The pool is split into two **disjoint class groups**:

- **pretrain pool** = indices with `label ≠ removed_class` → 19 classes.
- **continual pool** = indices with `label == removed_class` → 1 class.

This is the class-incremental setup: the model is pretrained on 19 “old” classes
and the 20th “new” class is reserved for Stage 2. The code asserts both pools are
non-empty (lines 325–328).

### 2.5 Materialise + channel compression (`materialise` line 210, `compress_channels` in `C`)

For each selected utterance, `materialise`:

1. bins it to dense `[T, 700]` (§2.3),
2. compresses the **channel axis only** by an integer `factor = nb_inputs /
   n_compressed_channels` (`validate_compression_factor`, must divide exactly;
   *worked example:* 700/70 = **10**),
3. checks invariants, and
4. stacks results into `X ∈ uint8^{N × T × C_comp}`, with `y int64 [N]` and
   `speaker int64 [N]`.

Only one full-resolution `[T,700]` sample exists at a time (compressed then
discarded) so peak memory stays bounded even at small Δt.

**Compression maths** (`compress_channels`, `snn_shd_common.py:211`): adjacent
channels are grouped into `C_comp` contiguous groups of `factor` channels.
`_group_count` reshapes `[…, C] → […, C_comp, factor]` and sums over the last
axis to get per-group spike counts `[…, C_comp]`. Then per method:

- **`or_pool` / `conditional_or`** (default): binary, `group = (count ≥
  condition_or)`. With `condition_or = 1` this is a logical OR over the `factor`
  native channels in the group. Output `{0,1}`.
- **`graded`**: integer `count` per group (spike-count preserving; value ≤ factor).
- **`bernoulli`**: binary, group fires with probability `count/factor` (seeded RNG).
- `factor == 1`: identity copy.

`assert_compression_invariants` (`snn_shd_common.py:247`) guards shape (time axis
unchanged, channel axis reduced by exactly `factor`, leading dims unchanged) and
value range (binary for or/bernoulli; sum-preserving for graded). This doubles as
a **transpose tripwire** (catches an accidental T↔C swap).

Per-sample compressed shape: **`[T, C_comp] = [100, 70]`** (worked example). The
model’s input dimension is therefore `C_comp = 70`, **not** 700.

Why: reduce input dimensionality (and W_in size) while preserving coarse
cochlear frequency-band activity, keeping the spike-train semantics binary.

### 2.6 Label-stratified 80/20 split (`stratified_two_way`, line 253)

Each of the two materialised pools is split independently into train/test using
per-class proportional sampling, seeded by a single `np.random.default_rng(seed)`
(line 295) shared across both splits (so the split is deterministic given the
seed; pretrain is split first, then continual consumes the advanced RNG state):

- For each class present, shuffle its indices, set `n_test = round(n ·
  test_frac)`; if `test_frac > 0` rounded to 0 and `n ≥ 2`, force `n_test = 1`;
  guarantee `n_train ≥ 1`.
- Fractions are validated to lie in [0,1] and sum to 1 (`validate_two_fractions`,
  line 244).

This yields four index sets, producing four logical splits (note pretrain_train
and pretrain_test **share the same materialised array `pre_X`** but with disjoint
index sets; likewise continual):

```
pretrain_train  = (pre_X, pre_y, pre_spk, pre_tr)   # 19 classes, ~80%
pretrain_test   = (pre_X, pre_y, pre_spk, pre_te)   # 19 classes, ~20%
continual_train = (con_X, con_y, con_spk, con_tr)   # removed class, ~80%
continual_test  = (con_X, con_y, con_spk, con_te)   # removed class, ~20%
```

### 2.7 Sanity checks before writing (`_check_splits`, line 412)

- **Shape / anti-transpose**: the first sample of each non-empty split must be
  `[nb_steps, n_compressed]`.
- **No leakage**: the removed class must not appear in any pretrain split; every
  continual split must contain only the removed class.
- **Label range**: pretrain labels stay in `[0, 20)` (labels are kept as
  **original** SHD ids, not re-indexed).
- **Non-empty**: `pretrain_train` and `continual_train` must be non-empty.

### 2.8 Write the dataset and manifest (lines 354–409)

- Each split is saved with `C.save_npz_split` (`snn_shd_common.py:273`) as a
  compressed `.npz` with `X uint8 [N,T,C]`, `y int64 [N]`, `speaker int64 [N]`.
- Per-split sample counts and per-class histograms are recorded.
- `preprocessing_manifest.json` is written to the **run dir** (parent of
  `dataset/`) with full provenance: dataset, storage format, binning rule,
  `removed_class`, `active_classes`, Δt (ms and s), `max_time`, `nb_steps`,
  `nb_inputs` (= **compressed** count, the model input dim), `original_nb_inputs`
  (700), compression method/factor/`condition_or`, `binary_output`, per-sample
  dense shape, merge flag, synthetic flag, seed, absolute HDF5 paths, the four
  fractions, all split counts + class histograms, and the benchmark note.

Note the manifest key `nb_inputs = n_compressed_channels` (70) is the
model-facing input dimension; `original_nb_inputs = 700` is provenance only. This
matters because `train_snn_shd.load_manifest` reads `nb_inputs` as the channel
count.

---

## 3. Stage 1b — The SNN architecture

The classifier is `C.SpikingReadoutReservoirSNN` (`snn_shd_common.py:67`), a
subclass of `train_snn_shd.ReservoirSNN` (`train_snn_shd.py:229`) with an added
spiking output layer. Pipeline:

```
X [B,T,C]
   │  W_in  [C,H]
   ▼
recurrent hidden layer  (H = 1000 second-order LIF neurons, recurrent W_rec [H,H])
   │  hidden spikes  spk_t ∈ {0,1}^{B×H}  for t=0..T-1   (trace [B,T,H])
   ├──────────────────────────────► Phi = mean_t(spk)  ∈ ℝ^{B×H}   (shared feature)
   │  W_out [H,O]                                    │ W_out [H,O]
   ▼ (per timestep)                                 ▼ (linear)
IF output layer (O=20 neurons)                 logits_lin = Phi @ W_out  [B,O]
   │  mean output spike rate
   ▼
Psi ∈ [0,1]^{B×O}   ← model(x) returns this (the spiking readout)
```

There are therefore **two readouts that share the same `W_out`**:

- **Spiking readout** `Psi = model(x)` — hidden spikes are projected per timestep
  and run through IF output neurons; the decode is the mean output spike rate.
  This is what `forward`/`model(x)` returns and what `evaluate_active` /
  `evaluate_split` use.
- **Linear readout** `Phi @ W_out` — the mean-spike feature projected directly.
  This is what `ridge` actually fits, and what the `*_linear` metrics report.

### 3.1 Input “neurons” (there are none)

The input is **not** a layer of spiking neurons. `X[b]` is the precomputed binary
spike raster `[T, C]`; it is injected directly as input current through `W_in`.
So the 70 “input units” are just the compressed cochlear channels presenting
binary spike trains. No input-side dynamics, threshold, or state.

### 3.2 Hidden layer — second-order recurrent LIF (`ReservoirSNN.hidden_spikes`, `train_snn_shd.py:266`)

Parameters (all `nn.Parameter`, normal init, `train_snn_shd.py:243`):

| Weight | Shape | Init std |
|---|---|---|
| `W_in` | `[C, H] = [70, 1000]` | `weight_scale / √C = 0.2/√70 ≈ 0.0239` |
| `W_rec` | `[H, H] = [1000, 1000]` | `weight_scale / √H = 0.2/√1000 ≈ 0.00632` |
| `W_out` | `[H, O] = [1000, 20]` | `weight_scale / √H ≈ 0.00632` |

Neuron decay constants are derived from the **dataset binning Δt**
(`C.derive_alpha_beta`, `snn_shd_common.py:334`):

```
alpha (synaptic) = exp(-Δt_ms / tau_syn_ms)
beta  (membrane) = exp(-Δt_ms / tau_mem_ms)
```

*Worked example* (Δt=14, τ_syn=5, τ_mem=10): `α = exp(-2.8) ≈ 0.0608`,
`β = exp(-1.4) ≈ 0.2466`.

**Feed-forward drive** for all timesteps at once:
`h_in = (X.reshape(B·T, C) @ W_in).reshape(B, T, H)` — shape `[B, T, H]`.

**Per-step dynamics.** State vectors `syn, mem, spk_prev, spk_sum ∈ ℝ^{B×H}`,
all initialised to 0. For `t = 0 … T-1`, with `Θ` the (surrogate) Heaviside:

```
drive_t   = h_in[:,t] + spk_prev @ W_rec          # recurrent uses PREVIOUS spikes
spk_t     = Θ(mem_t − threshold)                  # spike from membrane at step start
mem_{t+1} = (β · mem_t + syn_t) · (1 − spk_t)     # integrate OLD syn; hard reset-to-0
syn_{t+1} = α · syn_t + drive_t                   # leaky integrator of the drive
spk_sum  += spk_t ;  spk_prev = spk_t
```

Key mathematical points for faithful replication:

- It is a **second-order LIF**: a synaptic current `syn` (leak α) low-pass-filters
  the drive, and the membrane `mem` (leak β) integrates `syn`.
- **Update ordering matters.** The membrane uses `syn_t` (the value *before*
  `drive_t` is added), and the spike uses `mem_t` (the value *before* this step’s
  update). Consequently there is a built-in latency: `drive_t → syn_{t+1} →
  mem_{t+2} → spk_{t+2}` (earliest). A naive re-implementation using
  `mem = β·mem + syn_new` would change the dynamics.
- **Recurrence** is via `spk_prev @ W_rec` (the previous step’s spikes), i.e. a
  1-step recurrent delay.
- **Reset is hard reset-to-zero** (`mem ·= (1 − spk)`), not subtract-threshold;
  super-threshold residual is discarded. The reset mask `spk.detach()` blocks
  gradient through the reset path.
- **Spike function** `SurrGradSpike` (`train_snn_shd.py:213`): forward is exact
  Heaviside `(x>0)`; backward is the fast-sigmoid surrogate
  `grad_in = grad_out / (slope·|x| + 1)²` with `slope = 100`. The gradient is
  largest at `x=0` (membrane at threshold) and small far from threshold — this is
  why the firing-rate sanity checks matter (§3.5–3.6).

**Shared feature** `Phi = spk_sum / T ∈ ℝ^{B×H}` — the per-neuron mean spike rate
over the window (each entry in `[0,1]`). Optional `trace ∈ {0,1}^{B×T×H}` is the
full per-step spike tensor (needed by the output layer and by Stage 2).

### 3.3 Output layer — integrate-and-fire (`SpikingReadoutReservoirSNN._run_output_if`, `snn_shd_common.py:85`)

The output neurons are **IF**, with **different dynamics from the hidden layer**:
no synaptic filter (`OUTPUT_IF_ALPHA = 0`) and **no membrane leak**
(`OUTPUT_IF_BETA = 1`), no recurrence.

Per-step drive into the output layer (`output_rates_from_trace`, line 102):
`drive = (trace.reshape(B·T,H) @ W_out).reshape(B,T,O) · output_gain` — shape
`[B,T,O]`. Then IF integration (`mem ∈ ℝ^{B×O}`, `spk_sum=0`):

```
for t = 0 … T-1:
    spk_t     = Θ(mem_t − output_threshold)
    mem_{t+1} = (mem_t + drive_t) · (1 − spk_t)     # add current directly, hard reset
return Psi = spk_sum / T                             # mean output spike rate [B,O]
```

`forward(x)` (line 125) returns `Psi`. So `model(x) ∈ [0,1]^{B×O}` is a vector of
**mean output spike rates**, not unbounded logits.

**Why the IF output ≈ the linear readout (the crucial identity).** In the
sub-threshold regime (no reset), the membrane simply accumulates:
`mem_T = Σ_t drive_t = output_gain · (Σ_t spk_t) @ W_out = output_gain · T ·
(Phi @ W_out)`. Hence the spike count ≈ `output_gain·T·(Phi@W_out)/output_threshold`
and the mean rate

```
Psi ≈ clamp( output_gain · (Phi @ W_out) / output_threshold , 0, 1 ).
```

So the spiking decode is a **non-negative, saturating, scaled** version of the
linear readout. Two consequences that drive the dual-metric design (§5):

1. `Psi ≥ 0` — negative linear logits cannot be represented; only the positive
   part matters. Spiking argmax matches linear argmax only when the winning
   linear logit is positive and unsaturated.
2. Calibration depends on `output_gain`, `output_threshold` and the **scale of
   `W_out`**. For `ridge`, `W_out` comes from the regression at whatever
   magnitude the data dictates — it is **not** tuned to the IF threshold — so the
   spiking decode can be badly mis-calibrated (rates mostly 0 or mostly 1). This
   is exactly why the report records the **linear** metrics as the meaningful
   ones for ridge.

### 3.4 Spectral-radius sanity (`T.spectral_radius_sanity`, `train_snn_shd.py:313`)

Reused verbatim from the base trainer. On one batch `X_tr[:batch_size]`:

1. Measure `ρ = max |eig(W_rec)|`. Rescale `W_rec ·= init_spectral_radius / ρ`
   so it starts at `ρ = init_spectral_radius = 1.0`.
2. Iterate (≤20 times): run the reservoir, compute mean hidden firing rate
   `r = trace.mean()`; if `r > firing_high (0.20)` scale `W_rec ·= 0.8`; if
   `r < firing_low (0.02)` scale `W_rec ·= 1.2`; else stop.
3. Log diagnostics: final spectral radius, init firing rate, in-window flag, and
   the fractions of always-/never-spiking hidden neurons.

Purpose: the spectral radius of `W_rec` governs the reservoir’s memory/stability
(echo-state / edge-of-chaos). The loop nudges it so the reservoir produces a
usable spike density (2–20% activity), neither silent nor saturated, before any
training. The rescaled `W_rec` is what gets used (and, for ridge, frozen).

### 3.5 Output-firing sanity — **fullbptt only** (`output_firing_sanity`, line 503)

Analogue of the above, for the spiking output. On one batch it rescales `W_out`
uniformly (`·0.8` if too active, `·1.25` if too quiet, ≤25 iters) until the
**initial** mean output spike rate lies in `[firing_low, firing_high]`.

Rationale: at BPTT init `W_out` is tiny, so the output membrane sits far below
threshold; the surrogate gradient there is `≈ 1/(100·1+1)² ≈ 1e-4` — effectively
vanishing. Scaling `W_out` up puts the output near threshold so gradients flow.
`ridge` skips this because it **overwrites** `W_out` with the closed-form solve.

---

## 4. Training (Stage 1b core)

Both modes train **only on `pretrain_train`** (the 19 active classes) and never
look at any test split or at `continual_train` (Stage 2 data). Data is held
fully in memory as float32 tensors and iterated by manual slicing (no DataLoader).

Class bookkeeping: `active_classes = [0..19] \ {removed_class}` (length 19). A
map `pos` sends original labels → contiguous `[0,18]`; predictions are mapped
back via `active_idx`.

### 4.1 `ridge` (default, `train_ridge_active`, line 526)

Closed-form regularised least-squares of `W_out` on the **active** classes only.
Reservoir (`W_in`, `W_rec`) frozen.

1. `model.requires_grad_(False)`, eval mode.
2. `Phi = C.collect_features(model, X_tr, …)` → `[N, H]` (mean-spike features),
   cast to **float64 on CPU**.
3. Build one-hot targets over the 19 active classes: `Yk ∈ {0,1}^{N×19}` with
   `Yk[i, pos[y_i]] = 1`.
4. Solve the ridge normal equations (`H = 1000`):

   ```
   A = Phiᵀ Phi + λ·I_H      ∈ ℝ^{H×H}      (λ = ridge_lambda = 1.0)
   B = Phiᵀ Yk               ∈ ℝ^{H×19}
   Wk = A⁻¹ B                ∈ ℝ^{H×19}      (Cholesky; fallback to general solve)
   ```

   This minimises `‖Phi·Wk − Yk‖² + λ‖Wk‖²` — standard ridge regression, **no
   bias term**.
5. Scatter into the full output matrix: `Wfull ∈ ℝ^{H×20}` zeros, `Wfull[:,
   active_idx] = Wk`; copy into `model.W_out`. The **removed-class column stays
   exactly 0**.
6. `_zero_removed_column` is called again (redundant but explicit/deterministic).

float64-on-CPU is deliberate for numerical stability of the solve (MPS has no
float64; `resolve_device` keeps the SNN forward on the accelerator and the linear
algebra on CPU).

### 4.2 `fullbptt` (`train_bptt_active`, line 553)

Backprop-through-time over **all** weights (`W_in + W_rec + W_out`) with surrogate
gradients, cross-entropy on the 19 active classes.

- `model.requires_grad_(True)`; optimizer `adamax` (default) / `adam` /
  `sgd(momentum 0.9)`; `lr = args.lr if >0 else 2e-4`. Loss = `CrossEntropyLoss`.
- For each of `nb_epochs` (default 200) epochs, shuffle (seeded
  `torch.Generator`), and for each batch:

  ```
  counts = model(xb) · nb_steps           # Psi·T = output spike COUNTS, [B,20]
  logits = counts[:, active_idx]          # [B,19]
  loss   = CE(logits, pos[yb])            # targets remapped to [0,18]
  loss.backward();  (optional grad-clip);  optimizer.step()
  ```

  Note the logits fed to CE are **output spike counts** (`Psi·T ∈ [0, nb_steps]`),
  not unbounded scores — see §10.
- Running training loss and **active-19 training accuracy** are accumulated over
  the epoch and (if W&B is on) logged each epoch as `train_acc_active`.
- **Checkpoint policy:** the model is left at the **last epoch**. There is **no
  validation split and no best-epoch selection** (unlike `train_snn_shd.train_bptt`,
  which tracks best test acc). After the loop, `main` calls `_zero_removed_column`
  (line 684) to force the removed column back to 0.

### 4.3 Removed-class column init policy

In both modes the removed-class output column is set to **exactly 0**
(`removed_class_init_policy = "zero"`, recorded in the checkpoint and manifest).
By construction, new-class (removed-class) accuracy before incremental learning is
≈ 0 — this is **expected and logged, not a bug**, because the removed column
produces zero drive → zero output spikes → it never wins the argmax.

---

## 5. Metrics — how train/test accuracy are computed (lines 687–718)

After training, eight accuracies are computed (all in `@torch.no_grad`, eval
mode, chunked by `batch_size`). There are two orthogonal axes:

- **Decode**: *spiking* (`Psi = model(x)`) vs *linear* (`Phi @ W_out`, bypassing
  the IF output entirely).
- **Label space**: *19-way active* (argmax restricted to active columns, mapped
  back to original labels) vs *20-way full* (argmax over all 20 columns).

| Metric key | Function | Split | Decode | Label space |
|---|---|---|---|---|
| `pretrain_train_acc_active19` | `evaluate_active` (l.442) | pretrain_train | spiking | 19-way |
| `pretrain_test_acc_active19` | `evaluate_active` | pretrain_test | spiking | 19-way |
| `pretrain_test_acc_full20` | `C.evaluate_split` | pretrain_test | spiking | 20-way |
| `continual_test_acc_before_incremental` | `C.evaluate_split` | continual_test | spiking | 20-way |
| `pretrain_train_acc_active19_linear` | `evaluate_active_linear` (l.478) | pretrain_train | linear | 19-way |
| `pretrain_test_acc_active19_linear` | `evaluate_active_linear` | pretrain_test | linear | 19-way |
| `pretrain_test_acc_full20_linear` | `evaluate_split_linear` (l.489) | pretrain_test | linear | 20-way |
| `continual_test_acc_before_incremental_linear` | `evaluate_split_linear` | continual_test | linear | 20-way |

Mechanics:

- **`evaluate_active`**: `logits = model(xb)` `[B,20]`; restrict to active columns
  `[B,19]`; `pred = active_idx[argmax]` (back to original labels); compare to
  `yb`. This is the **meaningful pretraining metric** (the model can only predict
  classes it was trained on).
- **`evaluate_active_linear`**: identical but `logits = Phi @ W_out` (the readout
  `ridge` literally fits). For `ridge` this is the faithful, IF-calibration-free
  number; the spiking version may understate true separability (§3.3).
- **`C.evaluate_split`** (`snn_shd_common.py:426`): 20-way `argmax(model(x))`.
  Empty split → NaN (so “no samples” is never reported as 0).
- On `pretrain_test`, the 19-way and 20-way numbers are usually nearly equal,
  because that split contains **only active classes** and the removed column is 0
  (it rarely wins). `*_full20` here does **not** measure removed-class accuracy.
- On `continual_test` (only removed-class samples), accuracy ≈ 0 by construction
  (zero column) — the headline “new-class before incremental learning” number.

`train_seconds` and the init diagnostics (`init_*` from spectral-radius and, for
fullbptt, output-firing sanity) are merged into `pretraining_metrics`.

---

## 6. Weights & Biases logging (lines 601–604, 736–769)

- **Default `--wandb-mode disabled` ⇒ `init_wandb` returns `None` and nothing is
  logged.** You must pass `--wandb-mode online` (or `offline`) to get any plots.
- `init_wandb` (line 750) sets `project="shd-snn-pretrain"`,
  `name = wandb_name or run_name`, tags `["shd","pretrain",<mode>,"removed<k>",
  "C<channels>"]`, and a config dict containing every CLI arg (`arg/*`),
  `nb_inputs`, `dt_ms`, `alpha`, `beta`, removed/active classes, compression
  method/factor, `nb_steps`, and the init spectral-radius diagnostics (`init/*`).

**What actually becomes plottable:**

- **fullbptt**: each epoch logs `{epoch, train_loss, train_acc_active, lr,
  epoch_seconds}` → you get a **per-epoch training-accuracy/loss curve**. The
  per-epoch `train_acc_active` is a *running* training accuracy (accumulated as
  weights update within the epoch), not a clean eval pass. **Test accuracy is NOT
  logged per epoch** — only the final-eval scalars are logged once at the end.
- **ridge**: nothing is logged per step; only the final scalars.
- **End of run (both modes)**: every scalar metric in `pretraining_metrics` is
  logged once via `wandb_run.log(loggable)` (so it renders as a panel, not just a
  summary number) **and** written to `wandb_run.summary[...]`. Then `finish()`.

Net effect in the W&B UI: fullbptt shows training curves over epochs plus a
single final point per test/linear metric; ridge shows only the single final
points. (See §10 — there is no per-epoch test curve to diagnose overfitting,
which compounds the “last-epoch, no-validation” checkpoint policy.)

---

## 7. What is saved (lines 720–734) and the checkpoint schema

To `outputs/shd_pretraining/<run_name>/`:

1. **`dataset/{pretrain,continual}_{train,test}.npz`** — the four splits
   (`X uint8 [N,T,C]`, `y int64 [N]`, `speaker int64 [N]`).
2. **`preprocessing_manifest.json`** — full data provenance (§2.8).
3. **`checkpoints/pretrained_model.pt`** — `torch.save` of the dict from
   `C.build_checkpoint` (`snn_shd_common.py:439`):
   - `model_state_dict` — `W_in`, `W_rec`, `W_out` (CPU tensors).
   - `model_class` = `"SpikingReadoutReservoirSNN"`.
   - `architecture` — `nb_inputs`(=70), `nb_hidden`(=1000), `nb_outputs`(=20),
     `alpha`, `beta`, `tau_mem_ms`, `tau_syn_ms`, `threshold`, `weight_scale`,
     `surrogate_slope`, `dt_ms`, output-layer params (`output_gain`,
     `output_threshold`, `output_alpha=0`, `output_beta=1`, `output_layer="if"`).
   - `active_classes`, `removed_class`, `removed_class_init_policy="zero"`,
     `pretraining_mode`, `pretraining_metrics`, `dataset_dir` (abs path), `config`.
   This is the Stage 1 → Stage 2 contract; `C.load_checkpoint_model`
   (`snn_shd_common.py:478`) rebuilds the exact architecture and loads weights.
4. **`config.json`** — the full argparse namespace (numpy scalars coerced).
5. **`metrics.json`** — `pretraining_metrics` (the table in §5 plus timings/init
   diagnostics).
6. **`logs/`** — directory is created but this script writes nothing into it.

---

## 8. Determinism / reproducibility (`C.set_determinism`, `snn_shd_common.py:165`)

`set_determinism(seed)` seeds Python-NumPy (`np.random.seed`), torch
(`torch.manual_seed`) and CUDA (`torch.cuda.manual_seed_all`). Splits and weight
init use explicit generators (`np.random.default_rng(seed)`, a seeded
`torch.Generator`). This makes preprocessing and `ridge` fully reproducible.

It does **not** set `torch.use_deterministic_algorithms(True)` or cuDNN
deterministic flags, so `fullbptt` on a GPU may not be bit-reproducible across
runs/hardware (see §10).

---

## 9. End-to-end algorithm (replication checklist)

1. Read `shd_train.h5` (+ `shd_test.h5` if merging) → event pool (times, units,
   labels, speakers).
2. `nb_steps = ceil(max_time / Δt)`. For each utterance, bin to dense binary
   `[nb_steps, 700]` (drop `t ≥ max_time`, `b=floor(t/Δt)`, assign 1).
3. Split utterances into pretrain (label ≠ removed) and continual (label =
   removed).
4. Compress channels 700 → `C_comp` by integer factor (default `or_pool`,
   factor 10 → 70), channel axis only.
5. Per pool, stratified 80/20 split (seeded) → 4 splits; sanity-check (no
   leakage, correct shape) and save as `.npz` + manifest.
6. Reload `pretrain_train`/`pretrain_test`/`continual_test` as float32.
7. `α = exp(-Δt/τ_syn)`, `β = exp(-Δt/τ_mem)`. Build
   `SpikingReadoutReservoirSNN(C_comp, 1000, 20, α, β, threshold=1, weight_scale=0.2,
   surrogate_slope=100, output_gain=1, output_threshold=1, output_alpha=0,
   output_beta=1)`.
8. Spectral-radius sanity on one batch → rescale `W_rec` to firing in
   [0.02, 0.20]. For fullbptt only, output-firing sanity → rescale `W_out`.
9. Train on `pretrain_train` (19 classes):
   - **ridge**: `Phi = mean_t spikes`; `W_out[:,active] = (PhiᵀPhi + λI)⁻¹ Phiᵀ
     Yk` (float64); removed column 0.
   - **fullbptt**: 200 epochs, CE on `(Psi·nb_steps)[:,active]` vs remapped
     labels, adamax lr 2e-4; keep last-epoch weights; zero removed column.
10. Compute the 8 accuracies (spiking/linear × 19-way/20-way) on
    train/test/continual; save checkpoint, `config.json`, `metrics.json`;
    optionally log to W&B.

**Exact reproduction command** (worked example):

```bash
python pretrain_snn_shd.py \
  --output-root outputs/shd_pretraining \
  --run-name rc10_dt14_or70_removed10_ridge \
  --removed-class 10 --dataset-binning-ms 14 \
  --n-compressed-channels 70 --channel-compression-method or_pool \
  --mode ridge --nb-hidden 1000 --batch-size 64 --wandb-mode disabled
```

---

## 10. Issues, caveats and suggested improvements

Grounded in the code; ordered roughly by impact.

1. **No per-epoch test curve + last-epoch checkpoint + no validation (fullbptt).**
   `train_bptt_active` keeps the final-epoch weights, logs only training accuracy
   per epoch, and never evaluates the test set during training (contrast
   `train_snn_shd.train_bptt`, which tracks best test acc). With 200 epochs and no
   early stopping you cannot detect or avoid overfitting from the logs.
   *Fix:* log `evaluate_active(test)` per epoch and either checkpoint the best
   epoch on a held-out validation slice of `pretrain_train`, or document that
   last-epoch is intentional. (At minimum, a validation split would let
   early-stopping work.)

2. **Spiking metrics are mis-calibrated for `ridge`.** The IF output decode
   `Psi ≈ clamp(gain·(Phi@W_out)/threshold, 0, 1)` depends on the magnitude of
   the ridge-solved `W_out`, which is set by the regression, not by `gain`/
   `threshold` (both default 1, and ridge skips output-firing sanity). So
   `pretrain_*_acc_active19` (spiking) can be far below the true separability,
   while `*_active19_linear` is faithful. The code already flags this in comments,
   but the **headline** `metrics.json` keys are the spiking ones.
   *Fix:* for `ridge`, either run an output-firing/`W_out`-scale calibration
   before the spiking eval, or promote the `*_linear` metrics as primary and label
   the spiking ones as “uncalibrated”.

3. **Bounded-logit cross-entropy in fullbptt.** Logits = output spike counts
   `Psi·nb_steps ∈ [0, nb_steps]`. Once a neuron spikes every step its logit
   saturates at `nb_steps` and the surrogate gradient there collapses, capping the
   softmax confidence and putting a floor on CE loss. Training dynamics are
   therefore coupled to `nb_steps`, `output_gain` and `output_threshold` in a
   non-obvious way.
   *Fix:* consider a learnable output scale/temperature, or train CE on the linear
   readout `Phi@W_out` and use the IF output only at inference, or document the
   coupling.

4. **Neuron dynamics collapse when Δt ≫ τ.** Because `α=exp(-Δt/τ_syn)`,
   `β=exp(-Δt/τ_mem)`, the worked-example Δt=14 ms with τ_syn=5, τ_mem=10 gives
   `α≈0.06`, `β≈0.25` — i.e. the synaptic/membrane memory is almost gone and the
   “second-order LIF” is effectively near-memoryless per step. This is a
   consequence of tying the simulation step to the data binning. It is a modelling
   choice, but worth being explicit about: at coarse Δt the temporal dynamics
   contribute little and the model leans on the spatial (channel) pattern.
   *Fix:* if temporal dynamics are desired, decouple simulation dt from binning dt,
   or scale τ with Δt, or sweep Δt and report the dependence.

5. **`*_full20` on `pretrain_test` is nearly meaningless / mis-named.** That split
   has no removed-class samples and the removed column is 0, so 20-way ≈ 19-way
   there. It does not test the removed class (that is `continual_test`).
   *Fix:* drop it or rename to clarify it measures “does the zero column ever
   wrongly win on old-class data”.

6. **No bias in the readout.** `W_out` has no intercept and `Phi ∈ [0,1]^H` is
   non-negative (uncentered). Ridge without a bias on uncentered, non-negative
   features can lose a bit of accuracy. Consistent across the pipeline, but a bias
   (or feature centering) typically helps the linear readout.

7. **Reproducibility of `fullbptt` on GPU is not guaranteed.** `set_determinism`
   omits `torch.use_deterministic_algorithms(True)` / cuDNN-deterministic, so the
   BPTT path may vary run-to-run. Preprocessing and `ridge` are deterministic.
   *Fix:* add the deterministic flags (accepting the speed cost) for exact replays.

8. **Merged train+test breaks benchmark comparability.** Default
   `--merge-train-test=True` re-splits the union, so the test split shares
   speakers/conditions with train and is not the official SHD test set. The
   manifest warns about this; just don’t compare these accuracies to published SHD
   numbers. Pass `--no-merge-train-test` for a (more) comparable setup.

9. **Whole dataset held in memory, manual batching.** `load_npz_split` loads each
   split as one float32 tensor and training slices it directly. For SHD-scale data
   this is fine; for larger T/C or more data it can be memory-heavy and there is no
   `DataLoader`/prefetch. Minor.

10. **Binary-over-time binning discards intra-bin spike counts** (`x[b,u]=1`
    assignment). At small Δt this is negligible; at coarse Δt it can lose
    information that `graded` channel compression would otherwise have preserved.
    Intentional (keeps the raster binary), but worth noting when choosing Δt.

None of items 4, 8, 10 are bugs — they are documented design choices. Items 1, 2,
3, 7 are the ones most likely to affect conclusions drawn from the run.
