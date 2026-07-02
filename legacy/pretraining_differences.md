# Differences: `train_snn_shd.py` vs. SCCL `heidelberg_statedict_generator.py` (pretraining)

This document lists, step by step, the differences between your trainer

- **YOURS:** `train_snn_shd.py` (mode `fullbptt` is the closest match)
- **THEIRS:** `references/Spiking-Compressed-Continual-Learning-main/statedicts_class_incremental/heidelberg_statedict_generator.py`
  (the script that pretrains the SNN on **19 classes** and dumps the state dict; the model/data code lives inline in that file, the dataset download helper is in the sibling `utils.py`)

Goal context: you want to reproduce **their pretraining**, but on **20 classes instead of 19**. The single change that takes them from 19→20 classes is trivial (Section 6); the differences that actually affect whether you reproduce their *results* are the **architecture** (Section 1), the **readout/decoding** (Section 3), the **loss** (Section 4), and the **dataset/split** (Section 6). Read those first.

---

## 0. TL;DR comparison table

| Aspect | YOURS (`train_snn_shd.py`) | THEIRS (`heidelberg_statedict_generator.py`) |
|---|---|---|
| Hidden topology | **1** recurrent hidden layer, `nb_hidden` default **1000** | **3** recurrent hidden layers **200 → 100 → 50** (`nb_hidden`, `//2`, `//4`) |
| Full stack | `C → 1000 → 20` | `700 → 200 → 100 → 50 → 20` |
| Recurrence | one matrix `W_rec` (1000×1000) | three matrices `v1` (200²), `v2` (100²), `v3` (50²) |
| Input channels | inferred from data (`X.shape[2]`), may be compressed | fixed **700** (raw SHD) |
| Timesteps | from data (`X.shape[1]`) | fixed **100** |
| Readout | mean-rate of hidden spikes → linear `W_out`, **no bias** | leaky-integrator readout neurons, **max-over-time** |
| Loss | `CrossEntropyLoss` on one logit vector | `LogSoftmax + NLLLoss` on max-over-time output |
| Regularization | none | `reg_1` (L1 spikes) + `reg_2` (L2 spikes), **both default 0** |
| Dataset | preprocessed dense `train.npz` / `test.npz` | raw merged `shd_merged.h5`, sparse, built on the fly |
| Classes | all present (20) | **19** (one class removed for continual step) |
| Train/test split | the dataset's own train/test files | first ~1/5 of (shuffled) batches = test, rest = train |
| `dt` for `alpha`/`beta` | from manifest `dt_ms` (or `--sim_dt_ms`) | fixed `time_step = 1e-3 s` (1 ms), **regardless of bin width** |
| Init firing control | spectral-radius sanity rescale of `W_rec` | none |
| Training | 3 modes: `ridge`, `fullbptt`, `lastbptt` | single end-to-end BPTT over all 7 weight tensors |
| Optimizer | Adamax, lr `2e-4` (fullbptt default) | Adamax, lr `2e-4`, `betas=(0.9,0.999)` ✅ same |
| Epochs / batch | 200 / 64 (defaults) | 200 / 64 (defaults) ✅ same |
| `tau_mem`/`tau_syn`/`weight_scale`/`threshold`/surrogate slope | 10 ms / 5 ms / 0.2 / 1.0 / 100 | 10 ms / 5 ms / 0.2 / 1.0 / 100 ✅ same |
| Saved artifacts | none (W&B logging only) | pickles state dict + pretrain/removed data splits |

Items marked ✅ already match. Everything else is a real difference.

---

## 1. Network architecture — the biggest difference

**THEIRS** is a **3-layer deep recurrent SNN** (`heidelberg_statedict_generator.py:67-71`):

```
nb_hidden_1 = nb_hidden            # default 200
nb_hidden_2 = nb_hidden_1 // 2     # 100
nb_hidden_3 = nb_hidden_2 // 2     # 50
```

Weights (`:117-137`):
- Feedforward: `w1` (700×200), `w2` (200×100), `w3` (100×50), `w_out` (50×20)
- Recurrent (one per hidden layer): `v1` (200×200), `v2` (100×100), `v3` (50×50)

Each hidden layer is a second-order LIF with its **own** recurrent self-connection; spikes flow `input → L1 → L2 → L3 → readout`. All 7 tensors are trained.

**YOURS** (`ReservoirSNN`, `train_snn_shd.py:229-251`) is a **single** recurrent hidden layer:
- `W_in` (C×H), `W_rec` (H×H), `W_out` (H×nb_outputs), default `H = 1000`.

➡️ **To reproduce them you must add two more hidden layers** (200→100→50 with their own recurrent matrices), not just change `nb_hidden`. Your reservoir-style single layer cannot reproduce their numbers as-is. This is the dominant difference.

---

## 2. Neuron model — essentially identical (good news)

Both use the Zenke/Pittorino second-order LIF with the exact same update order, reset-to-zero, and fast-sigmoid surrogate (slope 100):

- recurrence uses the **previous** timestep's spikes,
- the spike is emitted from the membrane **before** its update (`spike_fn(mem - threshold)`),
- `new_syn = alpha*syn + drive`, `mem = (beta*mem + syn)*(1 - rst)`.

Compare YOURS `train_snn_shd.py:284-296` with THEIRS `:394-448`. The per-neuron dynamics are the same; the only difference is that theirs chains three such layers. Constants also match: `tau_mem=10 ms`, `tau_syn=5 ms`, `threshold=1.0`, `weight_scale=0.2`, surrogate slope `100`.

---

## 3. Readout layer & decoding — different mechanism

**THEIRS** (`:453-472`) feeds the **last hidden layer's spikes** through `w_out` and then through a **non-spiking leaky-integrator** readout (same `alpha`/`beta` filters, **no reset**), recording the output over time. Decision is **max over the time dimension**, then argmax over the 20 units (`:503-504`, `:542-544`). Note their `out_rec` is seeded with a zero (`out_rec=[out]`, `:459`) so it has `nb_steps+1` entries.

**YOURS** (`:296-305`) computes `Phi = mean_t(hidden_spikes)` (mean spike **rate** of the hidden layer) and a plain linear `logits = Phi @ W_out` with **no bias and no temporal filtering**. Decision is argmax of those logits.

➡️ Different read-out philosophy: **max-over-time of leaky-integrated membranes** (theirs) vs. **mean firing rate → linear** (yours). This changes gradients and final accuracy. To reproduce them, implement the leaky-integrator readout + max-over-time decode.

---

## 4. Loss function & regularization

**THEIRS** (`:489-513`):
```python
log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()
m, _ = torch.max(output, 1)            # max over time
log_p_y = log_softmax_fn(m)
reg_loss  = reg_1 * sum(spks)          # L1 on total spikes
reg_loss += reg_2 * mean(sum_t sum_b spks ** 2)  # L2 per-neuron
loss = loss_fn(log_p_y, y) + reg_loss
```
`reg_1 = reg_2 = 0` by default (the argparse help even says "Leave to zero for best result"), so in practice it is plain NLL on the max-over-time output. The spike counts used for regularization come from the **last hidden layer** (`spk_rec`).

**YOURS** (`:460`, `:477`): `nn.CrossEntropyLoss()` applied to the single rate-coded `logits`. No spike regularization term.

➡️ `LogSoftmax+NLLLoss` vs `CrossEntropyLoss` are numerically equivalent *given the same input*, but the **input differs** (max-over-time leaky-integrator output vs. mean-rate linear logits), so the losses are not equivalent in practice. The regularizers are off by default, so you can ignore them for a baseline reproduction.

---

## 5. Time discretization / `dt` quirk

**THEIRS**: `alpha`/`beta` are computed from a **fixed** `time_step = 1e-3 s` (`:54`, `:111-112`):
`alpha = exp(-1e-3/5e-3) ≈ 0.8187`, `beta = exp(-1e-3/10e-3) ≈ 0.9048`.
But the spikes are binned with `time_bins = linspace(0, max_time=1.4, num=nb_steps=100)` (`:281`), i.e. each of the 100 steps actually spans **~14 ms** of real time. So the neuron decays assume 1 ms steps while the data is binned at ~14 ms — a known quirk inherited from the Zenke SHD tutorial. They run exactly **`nb_steps = 100`** steps.

**YOURS**: `dt_ms` comes from the dataset manifest (or `--sim_dt_ms`), and `alpha/beta = exp(-dt_ms/tau)` use that **same** `dt` consistently (`:555-565`). `nb_steps` is whatever the preprocessed data has.

➡️ To match their effective decays, run with `--sim_dt_ms 1.0` (giving `alpha≈0.8187`, `beta≈0.9048`) **and** make sure the data is binned into **100 steps**. Don't let your manifest's real `dt_ms` (e.g. ~14 ms) drive `alpha/beta` if you want their exact dynamics.

---

## 6. Dataset, number of classes, and train/test split

This is where 19 vs 20 lives.

**THEIRS:**
- Loads the **merged** SHD file `shd_merged.h5` (train+test combined) — `:97-101`. (Created by `heidelberg_unite_datasets.py`.)
- Builds **sparse** spike tensors on the fly via `class_sparse_data_generator_from_hdf5_spikes` (`:253-319`); input is raw **700** channels, value 1.0 per spike (duplicate spikes in a bin **sum** on `to_dense()`).
- **Class removal:** `filter_class` drops every sample whose label `== removed_class` (default `0`) → the **pretrain set has 19 classes**; `filter_class_inverse` keeps only that class → the **"continual"/removed set** held out for the later incremental step (`:202-232`, `:587-588`).
- **Split:** number of batches `n // 64` (the remainder is **dropped** — every batch is exactly 64). After shuffling, the **first ~1/5 of batches** (`n_batches//5`, minus one) is the **test** set; the rest is **train** (`:609-630`). So train and test come from the *same merged pool*, split by batch, not the official SHD split.
- Reports `"Pre-Training accuracy on the 19 classes"` (`:639`).

**YOURS:**
- Loads **preprocessed dense** `train.npz` / `test.npz` (`X [N,T,C]`), already split by the preprocessing step — `:536-537`. Channels `C` may be **compressed** (e.g. 350) and `T` set by preprocessing.
- Uses **all** classes present; `nb_outputs` default 20; no class is removed; the asserts only check labels fit in `nb_outputs` (`:544-546`).
- Variable batch size allowed (last batch can be < `batch_size`).

➡️ **For 20 classes:** in their script, set `--removed_class` to a value that is **not a valid label** (e.g. `-1` or `20`) so `filter_class` removes nothing → all **20** classes remain in pretraining; the "continual"/removed set will then be empty. (Or delete the `filter_class` call entirely and just use the whole merged set.) In your script, you already train on all classes, so "20 classes" is the default — the catch is matching their **merged-dataset + 1/5 batch split**, not the official SHD split, if you want comparable numbers.

➡️ **Other data caveats for reproduction:** use **uncompressed 700-channel** data and **100 timesteps**; their input is raw SHD with per-bin spike **counts** (can be >1), whereas your dense preprocessing may binarize/normalize differently.

---

## 7. Weight initialization

Same scheme in both: `normal(mean=0, std = weight_scale / sqrt(fan_in))` with `weight_scale=0.2`.
- THEIRS per layer (`:118-137`): `w1` std `0.2/√700`, `w2` `0.2/√200`, `w3` `0.2/√100`, `v1` `0.2/√200`, `v2` `0.2/√100`, `v3` `0.2/√50`, `w_out` `0.2/√50`.
- YOURS (`:246-248`): `W_in` `0.2/√C`, `W_rec` `0.2/√H`, `W_out` `0.2/√H`.

Same formula; the per-tensor values differ only because of the different topology. No re-scaling of recurrent weights in theirs.

---

## 8. Spectral-radius / firing sanity check — only in YOURS

**YOURS** (`spectral_radius_sanity`, `:313-367`) rescales `W_rec` before training so the hidden firing rate lands in `[2%, 20%]`, and logs always-/never-spiking fractions. **THEIRS has nothing like this** — it just initializes and trains. If you want a faithful reproduction, **disable / skip this step** (their `v1,v2,v3` are used as-initialized), since it changes the effective recurrent gain.

---

## 9. Training procedure & code structure

| | YOURS | THEIRS |
|---|---|---|
| Modes | `ridge` (closed-form), `fullbptt`, `lastbptt` | single end-to-end BPTT only |
| What is trained | depends on mode; `fullbptt` trains `W_in,W_rec,W_out` | always all 7: `w1,w2,w3,w_out,v1,v2,v3` (`:486`) |
| Optimizer | Adamax (default), lr `2e-4` for fullbptt | Adamax, lr `2e-4`, `betas=(0.9,0.999)` (`:487`, `:635`) |
| Epochs | 200 | 200 |
| Batch size | 64, variable last batch | 64, fixed (remainder dropped) |
| Grad clip | optional (`--grad_clip`, off by default) | none |
| Seed control | `--seed` seeds torch/numpy/permutation | no explicit seeding |

➡️ The closest mode to reproduce them is **`fullbptt`** (it trains all weights with Adamax @ `2e-4`, matching theirs). `ridge`/`lastbptt` freeze the recurrent/input weights and are **not** comparable to their pretraining.

➡️ Note theirs has **no seeding**, so runs aren't deterministic; your `--seed` makes yours reproducible.

---

## 10. Saved artifacts & logging

**THEIRS** (`:483-664`) pickles:
- `state_dict_class_{c}.pkl` — the list `[w1,w2,w3,w_out,v1,v2,v3]` (raw tensors via `pickle`, **not** a `torch` state-dict),
- `pretrain_x/y_class_{c}.pkl`, `pretrain_class_x/y_class_{c}.pkl` — the train/test batch splits,
- `removed_x/y_class_{c}.pkl` — the held-out class data for the continual step.

It prints per-epoch mean loss and the final 19-class accuracy; **no W&B**.

**YOURS** logs to **W&B** (per-epoch train/test accuracy, firing rate, init diagnostics) and **saves no checkpoint**. If you need their downstream continual-learning step, you'll have to add weight saving.

---

## 11. Minimal checklist to reproduce their pretraining on 20 classes

1. **Architecture:** replace the single hidden layer with **3 recurrent layers 200→100→50** (`w1,w2,w3` + `v1,v2,v3`), `700→200→100→50→20`.
2. **Readout:** swap mean-rate linear readout for the **leaky-integrator readout + max-over-time** decode (NLL/CE on the time-max).
3. **Data:** use **raw/uncompressed 700-channel** SHD, **100 timesteps**, merged train+test, with the **first ~1/5 of shuffled batches as test**, rest as train (drop the remainder so all batches are exactly 64).
4. **20 classes:** do **not** remove any class (set their `--removed_class` to an invalid label, or skip `filter_class`). In your code this is already the default.
5. **Decays:** `--sim_dt_ms 1.0` so `alpha≈0.8187`, `beta≈0.9048` (their fixed 1 ms), with `tau_syn=5 ms`, `tau_mem=10 ms`.
6. **Disable** the spectral-radius sanity rescale.
7. **Optimizer/schedule:** `fullbptt`, Adamax, lr `2e-4`, `betas=(0.9,0.999)`, 200 epochs, batch 64 — already matching.
8. Keep `weight_scale=0.2`, `threshold=1.0`, surrogate slope `100` — already matching.

Items 1–3 are the ones that actually move the accuracy; 4 is your 19→20 change; 5–6 are second-order but needed for a faithful match.
