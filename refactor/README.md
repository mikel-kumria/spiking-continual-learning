# `refactor/` — modular SHD spiking continual-learning

A clean, modular re-implementation of the monolithic SHD pipeline
(`pretrain_snn_shd.py`, `class_incremental_snn_shd.py`, `train_snn_shd.py`,
`snn_shd_common.py`). Same experimental family, split into small importable
modules with thin CLI wrappers. See [`REPORT.md`](REPORT.md) for the full audit,
design rationale, and how each piece maps to the old scripts.

## Layout

```
refactor/
  configs/                 flat YAML defaults (CLI flags override any key)
  shd_cl/
    data/        shd_events, compression, splits, preprocessing, io
    models/      surrogate, reservoir, output_layers, snn
    training/    ridge, bptt, replay, cil
    evaluation/  metrics, predict
    logging/     wandb_utils, plots (hidden raster)
    utils/       config, determinism, device, checkpointing, audit
  scripts/       preprocess_shd, train_baseline, pretrain, cil_sweep, smoke_test
  tests/         5 plain-python test files (run without pytest)
```

## Key design points

- **Data contract preserved.** `X` is always `[N, T, C]` (time = axis 1, channel =
  last axis); labels stay ORIGINAL SHD ids `0..19` with no global remap. Subset
  remapping happens only inside cross-entropy for active-class training.
- **Ridge feature = `hidden_spike_sum = trace.sum(dim=1)`** (sum over time, not
  mean). With `output_layer_type="linear_integrator"` the output logits are
  exactly `hidden_spike_sum @ W_out`, so ridge and BPTT are evaluated on
  equivalent logits.
- **Three output layers:** `linear_integrator` (primary/correct ridge readout),
  `leaky_integrator`, `lif_no_reset` (leak, no reset).
- **Three training methods:** `fullbptt`, `lastbptt`, `ridge` (weighted ridge for
  CIL with the numerically-stable `sqrt(sample_weight)` form).
- **One-time spectral-radius renormalisation** of `W_rec` to 1.0 — no firing-rate
  calibration loop (see REPORT "Known limitations": this makes initial firing
  depend on `weight_scale`, so the default is `1.0`, not the reference's `0.2`).
- **Per-old-class replay:** `m_old_per_class = round(r * n_new)` for every old
  class (not a shared total budget).

## Quick start

Smoke test (synthetic data, no SHD needed — runs unit tests + full pipeline):

```bash
python refactor/scripts/smoke_test.py
```

Run the unit tests individually (no pytest required):

```bash
for t in refactor/tests/test_*.py; do python "$t"; done
```

## End-to-end commands (real SHD)

Raw SHD HDF5 is expected at `datasets/SHD_raw/shd_{train,test}.h5` (configurable).

Preprocess only:

```bash
python refactor/scripts/preprocess_shd.py --config refactor/configs/pretrain_default.yaml
```

Baseline (all 20 classes):

```bash
python refactor/scripts/train_baseline.py --config refactor/configs/baseline_default.yaml
```

Pretrain on 19 classes (removed class held out):

```bash
python refactor/scripts/pretrain.py \
  --config refactor/configs/pretrain_default.yaml \
  --training-method ridge --removed-class 10 --dataset-binning-ms 14 \
  --dataset-max-seconds 1.4 --n-compressed-channels 70 \
  --channel-compression-method or_pool
```

Class-incremental replay sweep (per-old-class replay):

```bash
python refactor/scripts/cil_sweep.py \
  --config refactor/configs/cil_default.yaml \
  --pretrained-checkpoint outputs/refactor_pretraining/<run>/checkpoints/pretrained_model.pt \
  --cil-training-method ridge --max-replay-percent 100 --min-replay-percent 0 \
  --num-replays 11 --ridge-weighting normalized_inverse_class_count
```

Enable Weights & Biases (hidden raster, firing histogram, metrics) with
`--wandb-mode online` (or `offline`).

## Outputs

- Pretrain/baseline run dir: `config.json`, `preprocessing_manifest.json`,
  `metrics.json`, `dataset/*.npz`, `checkpoints/*.pt`.
- CIL sweep dir: `replay_sweep_results.csv`, `replay_sweep_results.jsonl`,
  `replay_sweep_summary.json`, `checkpoints/final_ratio_<ppp>_seed_<sss>.pt`.
