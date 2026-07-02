# `refactor/` — audit report

This report documents the modular re-implementation of the SHD spiking
continual-learning pipeline under `refactor/`: what was built, how it maps to the
old monolithic scripts, what was preserved vs. changed, the mathematics of each
component, the W&B metric definitions, known limitations, and suggested
improvements. It is grounded strictly in the code under `refactor/shd_cl` and
`refactor/scripts`, plus empirical checks run during development.

---

## 1. What was implemented

A package `shd_cl` with single-responsibility modules and thin CLI wrappers:

- **Data** (`shd_cl/data`): raw SHD event loading / merging / synthetic
  (`shd_events.py`), fixed-width binary temporal binning + channel compression
  materialisation (`preprocessing.py`), channel-axis compression with 4 methods
  (`compression.py`), deterministic stratified splitting (`splits.py`), canonical
  `.npz`/JSON IO (`io.py`). Supports both `baseline_20_class` and
  `pretrain_19_class` regimes and writes a full provenance manifest.
- **Models** (`shd_cl/models`): surrogate spike (`surrogate.py`), recurrent
  second-order-LIF reservoir with one-time spectral-radius renormalisation
  (`reservoir.py`), three configurable output layers (`output_layers.py`), and the
  assembled `ReservoirSNN` (`snn.py`). No bias anywhere.
- **Training** (`shd_cl/training`): closed-form (optionally class-balanced
  weighted) ridge (`ridge.py`), `fullbptt`/`lastbptt` (`bptt.py`), per-old-class
  replay sampler (`replay.py`), and the CIL adaptation/eval logic (`cil.py`).
- **Evaluation** (`shd_cl/evaluation`): prediction/feature/logit helpers
  (`predict.py`) and accuracy metrics incl. balanced accuracy and CIL
  forgetting/learning deltas (`metrics.py`).
- **Logging** (`shd_cl/logging`): graceful W&B wrappers (`wandb_utils.py`) and
  reusable, W&B-independent plots — the hidden raster and firing-rate histogram
  (`plots.py`).
- **Utils** (`shd_cl/utils`): flat-YAML config with explicit CLI overrides
  (`config.py`), determinism (`determinism.py`), device resolution (`device.py`),
  self-describing checkpoints (`checkpointing.py`), and runtime consistency audits
  (`audit.py`).
- **Scripts**: `preprocess_shd.py`, `train_baseline.py`, `pretrain.py`,
  `cil_sweep.py`, `smoke_test.py`.
- **Tests**: 5 plain-python files (run without pytest) covering preprocessing
  shapes, label handling, ridge shapes/weighting, the replay sampler, and the
  output layers. The smoke test runs them plus a synthetic end-to-end pipeline.

---

## 2. How this maps to the old scripts

| Old | New |
|---|---|
| `pretrain_snn_shd.py` (preprocess + pretrain) | `scripts/preprocess_shd.py` + `scripts/pretrain.py`, backed by `data/preprocessing.py`, `training/{ridge,bptt}.py` |
| `train_snn_shd.py` (`ReservoirSNN`, `SurrGradSpike`, ridge/bptt) | `models/{surrogate,reservoir,output_layers,snn}.py`, `training/{ridge,bptt}.py` |
| `snn_shd_common.py` (`SpikingReadoutReservoirSNN`, compression, IO, checkpoints, RLS) | split across `models/`, `data/`, `utils/checkpointing.py`; RLS replaced by ridge/bptt CIL |
| `class_incremental_snn_shd.py` (RLS replay sweep) | `scripts/cil_sweep.py` + `training/{cil,replay}.py` |
| `train_baseline` (implicit) | `scripts/train_baseline.py` (explicit `baseline_20_class` regime) |

---

## 3. Behavior intentionally preserved

- **Data contract**: `X [N,T,C]` uint8, `y int64 [N]` original labels `0..19`,
  `speaker int64 [N]`. Time = axis 1, channel = last axis, never transposed.
- **Temporal binning**: `nb_steps = ceil(max_seconds / dt_s)`; drop `t >=
  max_seconds`; `bin = floor(t / dt_s)`; **binary assignment** `X[bin,unit]=1`
  (not spike-count accumulation).
- **Channel compression**: channel-axis only, `factor = 700 / n_compressed` must
  divide exactly; `or_pool`, `conditional_or`, `graded`, `bernoulli`; the
  transpose-tripwire invariant checks.
- **Hidden dynamics ordering** (verbatim): recurrence uses the *previous* step's
  spikes; the spike is read from the membrane at the start of the step; the
  membrane integrates the *old* synaptic current; hard reset-to-zero with a
  detached reset mask. (`models/reservoir.py`.)
- **Weight init**: `Normal(0, weight_scale/sqrt(fan_in))` for `W_in`, `W_rec`,
  `W_out`; no bias.
- **Determinism**: seeded NumPy/Torch RNGs and explicit generators for splits,
  init and batch order (ridge + preprocessing fully reproducible).
- **Removed-class column held at 0** during 19-class pretraining, recorded as
  `removed_class_init_policy="zero"`.
- **float64-on-CPU** closed-form solves (MPS has no float64).

## 4. Behavior intentionally changed

1. **Ridge feature is `hidden_spike_sum` (sum over time), not the mean.** The old
   code used `Phi = mean_t`. The sum is the natural `linear_integrator` readout and
   keeps ridge/BPTT on identical logits. This rescales features by `T`, so an
   equivalent `ridge_lambda` differs from the old code by a factor of ~`T^2`.
2. **Three explicit output layers.** The old code only had a hard-reset IF output
   ("mean output spike rate"). The new `linear_integrator` is the primary readout;
   `leaky_integrator` and `lif_no_reset` are alternatives. The old reset-IF layer
   is intentionally NOT reproduced (its rate decode was mis-calibrated for ridge).
3. **One-time `W_rec` renormalisation to spectral radius 1.0; no firing-rate
   sanity loop.** The old `spectral_radius_sanity` iteratively scaled `W_rec` to
   land firing in `[0.02, 0.20]`. Removed per spec. **Consequence (measured):** at
   the reference `weight_scale=0.2` the reservoir is completely silent even on real
   SHD (firing = 0.0000, 100% silent neurons), because initial firing is driven by
   the feed-forward `W_in` scale, which the spectral radius of `W_rec` does not
   fix. The default `weight_scale` is therefore `1.0` (self-fires at ~3% on SHD),
   with firing diagnostics logged, not acted on. See §13.
4. **CIL uses ridge/lastbptt/fullbptt, not RLS.** The old Stage 2 used online RLS.
   The new CIL supports closed-form (class-balanced weighted) ridge and BPTT.
5. **Per-old-class replay** (`m_old_per_class = round(r * n_new)`), replacing the
   old "total replay = r·N_new" semantics (§10).
6. **Two explicit regimes** (`baseline_20_class`, `pretrain_19_class`) in one
   preprocessing entry point.

---

## 5. The three output layers (`models/output_layers.py`)

All receive a per-timestep drive `drive_t = (hidden_spikes_t @ W_out)` and return
logits `[B, O]`. No bias.

- **`linear_integrator`** — pure accumulator, no leak/reset/spiking:
  `out_mem_t = out_mem_{t-1} + drive_t`, so `logits = out_mem_T = sum_t drive_t =
  hidden_spike_sum @ W_out`. This is the **primary, correct** readout
  and exactly the target ridge fits. Verified in `tests/test_ridge_shapes.py`
  (`linear_integrator` forward == `spike_sum @ W_out`).
- **`leaky_integrator`** — leaky accumulator, no spiking:
  `out_mem_t = beta_out * out_mem_{t-1} + drive_t`, `beta_out = exp(-dt/tau_out)`.
  Logit readout `last_mem` (default), `mean_mem`, or `max_mem`.
- **`lif_no_reset`** — leaky spiking membrane with **no** reset after firing:
  `out_mem_t = beta_out * out_mem_{t-1} + drive_t`; `out_spk_t =
  spike_fn(out_mem_t - output_threshold)`. Logit source `spike_sum` (default),
  `spike_mean`, `last_mem`, or `max_mem`.

**Comparability caveat.** Only `linear_integrator` gives logits that equal the
ridge target. For `leaky_integrator`/`lif_no_reset`, the output-layer decode is a
different (leaky / thresholded) function of the same `W_out`, so a ridge `W_out`
fit on `hidden_spike_sum` is not calibrated to that decode — hence the output-layer
accuracy is reported as a **secondary diagnostic** for ridge (§7).

---

## 6. `fullbptt`, `lastbptt`, `ridge` (`training/bptt.py`, `training/ridge.py`)

- **`fullbptt`** — BPTT through the reservoir and the chosen output layer; trains
  `W_in + W_rec + W_out` with surrogate gradients. Cross-entropy on the active
  logit columns (all 20 for baseline, 19 for pretraining).
- **`lastbptt`** — same forward/loss, but `W_in`/`W_rec` are frozen and only
  `W_out` gets gradients. It still backprops through the output-layer dynamics, so
  it is **not** ridge — it is gradient training of the readout through the selected
  output decode.
- **`ridge`** — freeze the reservoir; collect `hidden_spike_sum` `[N,H]`; solve
  `W = (XᵀX + λI)⁻¹ XᵀY` in float64 (Cholesky, general-solve fallback), no bias.
  For 19-class pretraining the solve is over the 19 active columns and scattered
  into `[H,20]` with the removed column zeroed.

---

## 7. Ridge evaluation (the important one)

**Primary ridge accuracy is `argmax(hidden_spike_sum @ W_out)`** — the linear
readout ridge literally fits (`evaluation/predict.py::linear_logits`, and the
`_linear` metrics). When `output_layer_type="linear_integrator"` the output-layer
decode is identical to this, so `*_output` == `*_linear`.

If a nonlinear/spiking output layer is configured, the output-layer accuracy is
**also** logged but clearly named a secondary diagnostic (`*_output` keys /
`primary_readout` field). It must not be confused with the primary ridge readout,
because a spiking/leaky decode of a ridge-solved `W_out` can be badly mis-calibrated
(its scale is set by the regression, not by the output threshold).

`cil.py::evaluate_cil` promotes the method's primary readout (ridge→linear,
bptt→output) to the unsuffixed metric keys and keeps both `_linear` and `_output`
variants for every run.

---

## 8. 19-class pretraining with 20 output neurons

`W_out` always has 20 columns because the removed class's column is needed at CIL
time. During pretraining only the 19 active classes are used:

- **BPTT**: compute full logits `[B,20]`, slice `active_logits = logits[:,
  active_classes]`, remap targets to contiguous `[0,18]` for cross-entropy. The
  removed column never enters the loss.
- **Ridge**: build one-hot targets over the 19 active columns, solve, scatter into
  `[H,20]`.

Three distinct pretraining metrics (avoiding the classic mislabel):
`pretrain_*_acc_active19` (argmax over active columns only — the meaningful
number); `pretrain_test_acc_full20_diagnostic` (20-way argmax on old-class test —
only checks whether the zero column ever wrongly wins; it does **not** test the
removed class); `continual_test_acc_before_cil` (removed-class accuracy before CIL,
~0 by construction for ridge).

---

## 9. Removed-class output handling

`removed_class_init_policy="zero"`. In pretraining the removed `W_out` column is:
(a) initialised to 0 before BPTT; (b) its gradient is zeroed each step
(`model.W_out.grad[:, removed] = 0`); (c) the column is re-zeroed after every
optimizer step and once more before saving; (d) for ridge it is simply never
solved and left 0. Verified in `tests/test_label_handling.py` (BPTT) and
`tests/test_ridge_shapes.py` (ridge), and audited at save time
(`utils/audit.py::audit_removed_column_zero`). Consequence: new-class accuracy
before CIL is ~0 by construction — expected and logged, not a bug.

---

## 10. Per-old-class replay (`training/replay.py`)

Replay is defined **per old class**, not as a shared budget. With `n` new-class
train samples and ratio `r = replay_percent / 100`:

```
m_old_per_class  = round(r * n)          # SAME for every old class
total_old_replay = sum over old classes of the sampled count   (= 19 * m if all full)
total_cil_train  = n + total_old_replay
```

At `r = 1.0` this is `m = n` per old class plus `n` new samples ⇒ class-balanced
joint training on ~`n` samples/class (given enough old data), verified in
`tests/test_replay_sampler.py`. Under-full classes are handled by
`replay_replacement_policy ∈ {with_replacement_if_needed (default),
cap_at_available, error}`. The sampler logs the ratio/percent, `n_new`,
`m_old_per_class`, `total_old_replay`, `total_cil_train`, whether replacement was
used, and per-class counts.

This differs from the old "total replay = r·N_new" semantics, which starved
per-class replay at high class counts (see the project memory note: additive/total
replay made ridge look far worse than balanced replay did).

---

## 11. Weighted ridge and why `sqrt(sample_weight)` (`training/ridge.py`)

For class-imbalanced CIL training sets, weighted ridge solves
`min_W ||√D (XW − Y)||² + λ||W||²` with `D = diag(w)`, closed form `W = (XᵀDX +
λI)⁻¹ XᵀDY`. It is implemented in the numerically stable **row-scaling** form:
`Xw = X * sqrt(w)[:,None]`, `Yw = Y * sqrt(w)[:,None]`, then solve the ordinary
normal equations `Xwᵀ Xw` / `Xwᵀ Yw`. Scaling rows by `√w` makes `XwᵀXw = XᵀDX`
and `XwᵀYw = XᵀDY` exactly, while avoiding forming the `N×N` matrix `D` and keeping
the Gram matrix symmetric positive-definite for Cholesky. Verified to match the
explicit `(XᵀDX+λI)⁻¹XᵀDY` in `tests/test_ridge_shapes.py`.

Weighting modes: `none` (w=1); `inverse_class_count` (w=1/n_class, each class total
weight 1 — changes the loss scale, hence the effective λ); and
`normalized_inverse_class_count` (`w = N/(C·n_class)`, each class equal total
weight with mean sample weight ≈ 1, keeping λ comparable across replay ratios).
CIL defaults to `normalized_inverse_class_count`; baseline/pretraining to `none`.

---

## 12. W&B metric definitions

Logging is optional (`wandb_mode=disabled` by default; wrappers no-op when off or
when wandb is missing).

**Preprocessing / manifest:** per-split counts and class histograms,
`dataset_binning_ms`, `dataset_max_seconds`, `nb_steps`, `n_compressed_channels`,
compression method/factor, merge flag, removed/active classes.

**Reservoir diagnostics** (logged, not acted on): `initial_spectral_radius`,
`target_spectral_radius`, `renormalized_spectral_radius`, `mean_hidden_firing_rate`,
`frac_silent_hidden`, `frac_always_firing_hidden`, per-neuron firing histogram, and
a hidden **raster** image (`plot_hidden_raster`, first diagnostic sample, top-N most
active neurons).

**Pretraining/baseline:** BPTT per-epoch `train_loss`, `train_acc`, `test_acc`,
`grad_norm`, `W_{in,rec,out}_norm`, `lr`, `epoch_seconds`. Final scalars:
`pretrain_*_acc_active19` (primary), `*_full20_diagnostic`,
`continual_test_acc_before_cil`, all `_linear` and `_output` variants, ridge λ /
weighting / solve status / condition number.

**CIL (per ratio+seed):** replay ratio/percent, `n_new_samples`,
`m_old_per_class`, `total_old_replay`, `total_cil_train`, per-class replay counts,
`old/new/total/balanced/two_group_balanced` accuracy **before and after** (primary
+ `_linear` + `_output`), `forgetting_old`, `learning_new`, `total_delta`,
`avg_old_perclass_forgetting`, and per-class before/after. Written to CSV + JSONL,
with a per-ratio mean/std summary JSON.

Note on totals: the combined test set is ~19× old-weighted, so `total_acc` is
old-dominated; **`two_group_balanced_acc` = mean(old, new)** is the honest headline
for the stability/plasticity tradeoff (consistent with the project memory finding).

---

## 13. Known limitations (be especially critical)

1. **Silent reservoir at low `weight_scale` (most important).** Removing the
   firing-rate calibration loop means initial firing is set by the feed-forward
   `W_in` scale, which the `W_rec` spectral-radius renormalisation does **not**
   control. Measured: `weight_scale=0.2` → 0% firing on real SHD (dead reservoir);
   `~1.0` → ~3% firing. The default was raised to `1.0` and firing diagnostics are
   logged. If you sweep `dataset_binning_ms`, `threshold`, or compression, re-check
   `mean_hidden_firing_rate` — there is no auto-tuner anymore. (Documented change,
   not a bug, but it can silently zero your accuracy if ignored.)
2. **Output-layer comparability across methods.** ridge/`linear_integrator` and
   BPTT/`linear_integrator` are on identical logits, but a ridge `W_out` decoded
   through `leaky_integrator`/`lif_no_reset` is uncalibrated (its magnitude comes
   from the regression, not the threshold). Compare like-for-like; the `_linear`
   metric is the invariant one.
3. **`dt` tied to dataset binning ⇒ near-memoryless LIF at coarse bins.** With
   `α = exp(-dt/τ_syn)`, `β = exp(-dt/τ_mem)` and `dt=14 ms`, `τ_syn=5`, `τ_mem=10`
   give `α≈0.06`, `β≈0.25` — little temporal memory, so the model leans on the
   spatial (channel) pattern. Decouple sim-dt from binning-dt or scale τ with dt if
   temporal dynamics matter.
4. **No-bias ridge on non-negative spike-sum features.** `hidden_spike_sum ≥ 0` is
   uncentered and there is no intercept; a bias / feature centering usually helps.
   Kept bias-free for faithfulness and pipeline consistency.
5. **Spike sums are not time-normalised.** Comparing runs with different
   `dataset_max_seconds`/`nb_steps` compares sums over different horizons; divide by
   `T` (or fix the window) for cross-`max_seconds` comparisons.
6. **Merged train+test breaks benchmark comparability.** Default
   `merge_train_test=true` re-splits the union, so the test split shares
   speakers/conditions with train and is not the official SHD test set — the
   manifest records `benchmark_note`. Set `merge_train_test: false` for comparable
   numbers.
7. **Binary-over-time binning discards intra-bin spike counts** (`X[b,u]=1`
   assignment). Negligible at fine `dt`, lossy at coarse `dt` (where `graded`
   channel compression could otherwise retain counts).
8. **BPTT with spiking output layers can vanish at init.** `lif_no_reset` +
   `spike_sum` with a tiny init `W_out` starts near-silent (small surrogate
   gradient). No output-firing calibration is implemented (spec removed it); use
   `linear_integrator` for BPTT, or raise init scale.
9. **BPTT GPU reproducibility not guaranteed** (`use_deterministic_algorithms` is
   not enabled). Preprocessing and ridge are deterministic.
10. **Whole splits held in memory, manual batching.** Fine at SHD scale; no
    DataLoader/prefetch.

---

## 14. Suggested improvements for class-incremental learning

- **Report both the old/new tradeoff and total.** Always headline
  `two_group_balanced_acc` (mean of old, new) alongside `total_acc`; total alone is
  old-dominated (~19:1) and hides forgetting/plasticity.
- **True class-balanced joint training** is `r=1.0` here (per-old-class replay =
  `n_new` per class). Distinguish it explicitly from "balanced relative to the
  new-class count" when interpreting the sweep.
- **δ / λ (stability–plasticity) sweep.** For ridge CIL, sweep `ridge_lambda` with
  `normalized_inverse_class_count` weighting so λ is comparable across ratios; larger
  λ keeps `W_out` near the pretrained readout (retains old, learns less new).
- **Rehearsal-free baselines.** Add `r=0` regularisation-only and EWC/feature-
  distillation baselines to quantify how much replay is doing (the project memory
  finding: r=0 cannot learn the new class without forgetting here).
- **Calibrate the readout before spiking eval**, or standardise on the linear
  readout for cross-method CIL comparisons, to remove the output-decode
  mis-calibration confound.
- **Bias / feature centering** in the readout, and **time-normalised features**,
  to make ridge stronger and comparisons across `max_seconds` fair.
- **Restore an optional firing-rate init check** (as a warning, not the old
  iterative loop) so a silent reservoir is caught before a whole sweep runs.

---

## 15. Tests run / not run

**Run and passing** (26 unit assertions + synthetic end-to-end):
`python refactor/scripts/smoke_test.py` runs all five test files
(`test_preprocessing_shapes`, `test_label_handling`, `test_ridge_shapes`,
`test_replay_sampler`, `test_output_layers`) plus a full synthetic pipeline
(preprocess → ridge/fullbptt/lastbptt pretrain → CIL ridge+fullbptt sweep → raster
render → audits). The CLI scripts were exercised end-to-end on synthetic configs
(`preprocess_shd`, `pretrain`, `train_baseline`, `cil_sweep`) and the real SHD
event/binning/reservoir path was validated on a subset (ridge beats chance,
`weight_scale` firing behavior above).

**Not run:** a full-scale real-SHD pretraining + 11-point × multi-seed CIL sweep
(long-running; the reference numbers in the project memory apply). No GPU/MPS run
was performed here (CPU only); `use_deterministic_algorithms` is not enabled, so
full-scale BPTT-on-GPU reproducibility is untested.
