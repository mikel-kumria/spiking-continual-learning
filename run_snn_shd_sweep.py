"""Sweep runner for ``train_snn_shd.py`` over many preprocessed SHD datasets.

Edit the two blocks below and run ``python run_snn_shd_sweep.py``:

* ``CONFIG``        -- the shared training settings applied to *every* run
                       (mirror of the ``train_snn_shd.py`` CLI flags).
* ``DATASET_PATHS`` -- the datasets to iterate over, given as paths *relative*
                       to ``SHD_PREPROCESSED`` (absolute paths also accepted).

Each entry of ``DATASET_PATHS`` may point either directly at a dataset
directory (one holding ``train.npz``) or at a parent folder, in which case
every descendant directory containing ``train.npz`` is run. The nested
``SHD_700_uncompressed/SHD_preprocessed_dtXms_allclasses`` layout is handled
automatically.

Parallelism is memory-aware and tuned for a single RTX 5080 (16 GB VRAM,
~61 GB RAM): both ridge and BPTT load the whole split into CPU RAM, so the
scheduler caps the *total in-flight CPU RAM* (``RAM_BUDGET_GB``) as well as the
*number of concurrent processes* (``MAX_PARALLEL``). Light ``dt14ms`` datasets
run many-at-once; the heavy T=700 ``dt1ms`` datasets run only a couple at a
time. Each run logs to W&B as its own run (grouped together), with the dataset
provenance -- compression method, input channels, dt/time-binning, nb_steps --
loaded from the manifest by ``train_snn_shd.py`` itself.
"""
import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

import numpy.lib.format as npy_format


# =============================================================================
# 1. Shared training settings (edit once; applied to every run)
# =============================================================================
# Keys map 1:1 onto train_snn_shd.py CLI flags (argparse dest names). A value
# of ``None`` means "leave the flag out / use the script default". ``wandb_name``
# is set automatically per dataset and should stay out of here.
CONFIG = {
    # mode for the whole sweep
    "mode": "ridge",                 # ridge | fullbptt | lastbptt
    # architecture
    "nb_hidden": 1000,
    "nb_outputs": 20,
    # neuron model
    "tau_mem_ms": 140.0,
    "tau_syn_ms": 70.0,
    "threshold": 1.0,
    "weight_scale": 0.2,
    "surrogate_slope": 100.0,
    "sim_dt_ms": 0.0,                # <=0 -> use dataset dt_ms from manifest
    # reservoir / spectral-radius sanity check
    "init_spectral_radius": 1.0,
    "firing_low": 0.02,
    "firing_high": 0.20,
    "sr_scale_up": 1.2,
    "sr_scale_down": 0.8,
    "sr_max_iters": 20,
    # ridge
    "ridge_lambda": 1.0,
    # bptt
    "nb_epochs": 200,
    "batch_size": 64,
    "lr": 0.0,                       # <=0 -> per-mode default
    "optimizer": "adamax",           # adamax | adam | sgd
    "grad_clip": 0.0,
    # runtime
    "seed": 0,
    "device": "auto",                # auto | cpu | cuda
    "limit": 0,                      # >0 -> smoke test on first N samples
    # logging
    "wandb_project": "shd-snn",
    "wandb_entity": None,            # set to your team/entity, or leave None
    "wandb_mode": "online",          # online | offline | disabled
}


# =============================================================================
# 2. Datasets to run (paths relative to SHD_PREPROCESSED; absolute also OK)
# =============================================================================
# Paste one path per line. Point at a dataset dir (has train.npz) or a parent
# folder to run every dataset beneath it.
DATASET_PATHS = [
    "SHD_dt14ms_1400ms",
]


# =============================================================================
# 3. Parallelism knobs (tuned for one RTX 5080)
# =============================================================================
MAX_PARALLEL = 4        # max concurrent train_snn_shd.py processes (GPU cap)
RAM_BUDGET_GB = 40.0    # max total in-flight CPU RAM (of ~61 GB) for loaded splits
POLL_SECONDS = 2.0      # how often the scheduler polls running jobs
# VRAM guard: GPU activations scale with nb_steps (T). A job is "heavy" when its
# T >= HEAVY_NB_STEPS, and at most MAX_PARALLEL_HEAVY heavy jobs run at once
# (on top of MAX_PARALLEL). This prevents the T=700 dt1ms datasets from
# over-committing the 16 GB GPU. A heavy job may still run alone if nothing else
# is running, regardless of this cap.
MAX_PARALLEL_HEAVY = 2  # max concurrent heavy (large-T) jobs
HEAVY_NB_STEPS = 400    # T threshold above which a job counts as heavy


# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_snn_shd.py"
# WSL-independent: SHD_PREPROCESSED sits two levels up under research/data.
DATA_ROOT = (SCRIPT_DIR / ".." / ".." / "data" / "SHD_PREPROCESSED").resolve()


# =============================================================================
# Dataset discovery
# =============================================================================
def resolve_entry(entry):
    """Map a (possibly relative) DATASET_PATHS entry to an absolute Path.

    Tolerant of Windows-style paste: backslashes are normalized to ``/`` and a
    leading separator is treated as relative to ``DATA_ROOT`` (so a pasted
    ``\\SHD_.../...`` works the same as ``SHD_.../...``).
    """
    s = os.path.expanduser(str(entry).strip()).replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        s = s.lstrip("/")          # leading-slash "relative" paste -> under DATA_ROOT
    return (DATA_ROOT / s).resolve()


def discover_datasets(entry_path):
    """Return dataset dirs (those holding train.npz) reachable from a path.

    If ``entry_path`` itself holds train.npz it is returned as-is; otherwise we
    recurse into it and collect every descendant directory holding train.npz.
    """
    if not entry_path.exists():
        return []
    if (entry_path / "train.npz").is_file():
        return [entry_path]
    return sorted({f.parent for f in entry_path.rglob("train.npz")})


def run_name_for(data_dir):
    """Readable run name from the dataset dir relative to DATA_ROOT."""
    try:
        rel = data_dir.relative_to(DATA_ROOT)
    except ValueError:
        rel = Path(data_dir.name)
    return str(rel).replace(os.sep, "__")


# =============================================================================
# CPU-RAM estimation (manifest shapes first, npz header as fallback)
# =============================================================================
def _shapes_from_manifest(data_dir):
    import json
    for fname in ("compression_manifest.json", "preprocessing_manifest.json"):
        p = data_dir / fname
        if p.is_file():
            with open(p) as f:
                m = json.load(f)
            shapes = m.get("shapes") or {}
            if "train" in shapes and "test" in shapes:
                return shapes["train"], shapes["test"]
    return None, None


def _npz_array_shape(npz_path, name="X"):
    """Read an array's shape from a .npz without loading the data."""
    with zipfile.ZipFile(npz_path) as z:
        member = name + ".npy"
        if member not in z.namelist():
            return None
        with z.open(member) as f:
            version = npy_format.read_magic(f)
            shape, _fortran, _dtype = npy_format._read_array_header(f, version)
    return shape


def _prod(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def estimate_ram_gb(data_dir):
    """Estimate CPU RAM (GB) to hold train+test X as float32 ([N,T,C]*4)."""
    tr, te = _shapes_from_manifest(data_dir)
    if tr is None or te is None:
        try:
            tr = _npz_array_shape(data_dir / "train.npz")
            te = _npz_array_shape(data_dir / "test.npz")
        except Exception:
            tr = te = None
    if tr is None or te is None:
        return None
    total_bytes = (_prod(tr) + _prod(te)) * 4
    return total_bytes / (1024 ** 3)


def estimate_nb_steps(data_dir):
    """Estimate nb_steps (T), the time dimension of X with layout [N, T, C]."""
    tr, _ = _shapes_from_manifest(data_dir)
    if tr is None:
        try:
            tr = _npz_array_shape(data_dir / "train.npz")
        except Exception:
            tr = None
    if tr is None or len(tr) < 2:
        return None
    return int(tr[1])


# =============================================================================
# Subprocess command
# =============================================================================
def build_command(data_dir, run_name):
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--data_dir", str(data_dir)]
    for key, value in CONFIG.items():
        if value is None or key == "wandb_name":
            continue
        cmd += [f"--{key}", str(value)]
    # suffix the run name with the mode so runs of different modes for the same
    # dataset are distinguishable when sharing a W&B project.
    mode = CONFIG.get("mode")
    wandb_name = f"{run_name}__{mode}" if mode else run_name
    cmd += ["--wandb_name", wandb_name]
    return cmd


# =============================================================================
# Memory-aware scheduler
# =============================================================================
def run_sweep(jobs, log_dir, run_group):
    """jobs: list of dicts {name, data_dir, est_ram}. Returns result records."""
    env = dict(os.environ)
    env["WANDB_RUN_GROUP"] = run_group

    pending = list(jobs)
    running = []   # dicts: name, proc, log_fp, est_ram, t0
    results = []   # dicts: name, returncode, seconds, log_path

    def in_flight_ram():
        return sum(j["est_ram"] or 0.0 for j in running)

    def running_heavy():
        return sum(1 for j in running if j["is_heavy"])

    def launch(job):
        log_path = log_dir / f"{job['name']}.log"
        log_fp = open(log_path, "w")
        cmd = build_command(job["data_dir"], job["name"])
        log_fp.write("$ " + " ".join(cmd) + "\n\n")
        log_fp.flush()
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                                 env=env, cwd=str(SCRIPT_DIR))
        ram_txt = f"{job['est_ram']:.1f} GB" if job["est_ram"] else "unknown RAM"
        heavy_txt = " heavy" if job["is_heavy"] else ""
        print(f"[launch] {job['name']}  (~{ram_txt}{heavy_txt})  -> {log_path.name}")
        running.append({"name": job["name"], "proc": proc, "log_fp": log_fp,
                        "est_ram": job["est_ram"], "is_heavy": job["is_heavy"],
                        "t0": time.time(), "log_path": log_path})

    while pending or running:
        # 1) launch as many pending jobs as the budgets allow
        made_progress = True
        while made_progress and pending and len(running) < MAX_PARALLEL:
            made_progress = False
            for i, job in enumerate(pending):
                fits_ram = (in_flight_ram() + (job["est_ram"] or 0.0)
                            <= RAM_BUDGET_GB)
                fits_heavy = (not job["is_heavy"]
                              or running_heavy() < MAX_PARALLEL_HEAVY)
                # always allow one job to run alone even if it exceeds a budget
                if (fits_ram and fits_heavy) or not running:
                    launch(pending.pop(i))
                    made_progress = True
                    break
                if len(running) >= MAX_PARALLEL:
                    break

        # 2) reap finished jobs
        time.sleep(POLL_SECONDS)
        still = []
        for j in running:
            rc = j["proc"].poll()
            if rc is None:
                still.append(j)
                continue
            j["log_fp"].close()
            secs = time.time() - j["t0"]
            status = "ok" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[done]   {j['name']}  {status}  ({secs:.1f}s)")
            results.append({"name": j["name"], "returncode": rc,
                            "seconds": secs, "log_path": j["log_path"]})
        running = still

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    if not TRAIN_SCRIPT.is_file():
        raise SystemExit(f"train script not found: {TRAIN_SCRIPT}")
    if not DATA_ROOT.exists():
        print(f"WARNING: data root does not exist: {DATA_ROOT}")

    # ---- expand DATASET_PATHS into concrete dataset dirs (dedup, keep order) ----
    seen = set()
    jobs = []
    for entry in DATASET_PATHS:
        resolved = resolve_entry(entry)
        found = discover_datasets(resolved)
        if not found:
            print(f"WARNING: no train.npz found for entry '{entry}' "
                  f"(resolved: {resolved})")
            continue
        for d in found:
            key = str(d)
            if key in seen:
                continue
            seen.add(key)
            steps = estimate_nb_steps(d)
            jobs.append({"name": run_name_for(d), "data_dir": d,
                         "est_ram": estimate_ram_gb(d), "est_steps": steps,
                         "is_heavy": steps is not None and steps >= HEAVY_NB_STEPS})

    if not jobs:
        raise SystemExit("no datasets to run; check DATASET_PATHS / DATA_ROOT")

    # ---- sweep bookkeeping ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = f"sweep_{stamp}"
    log_dir = SCRIPT_DIR / "sweep_logs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SHD sweep ===  mode={CONFIG['mode']}  group={run_group}")
    print(f"data root : {DATA_ROOT}")
    print(f"datasets  : {len(jobs)}")
    n_heavy = sum(1 for j in jobs if j["is_heavy"])
    print(f"limits    : MAX_PARALLEL={MAX_PARALLEL}  "
          f"RAM_BUDGET_GB={RAM_BUDGET_GB}  "
          f"MAX_PARALLEL_HEAVY={MAX_PARALLEL_HEAVY} (T>={HEAVY_NB_STEPS})")
    print(f"logs      : {log_dir}")
    print(f"heavy     : {n_heavy}/{len(jobs)} datasets")
    for j in jobs:
        ram_txt = f"{j['est_ram']:.1f} GB" if j["est_ram"] else "unknown"
        steps_txt = f"T={j['est_steps']}" if j["est_steps"] else "T=?"
        heavy_txt = " heavy" if j["is_heavy"] else ""
        print(f"  - {j['name']}  (~{ram_txt} RAM, {steps_txt}{heavy_txt})")
    print()

    t0 = time.time()
    results = run_sweep(jobs, log_dir, run_group)
    total = time.time() - t0

    # ---- summary ----
    print("\n=== summary ===")
    name_w = max((len(r["name"]) for r in results), default=4)
    n_ok = 0
    for r in sorted(results, key=lambda x: x["name"]):
        status = "ok" if r["returncode"] == 0 else f"FAILED(rc={r['returncode']})"
        n_ok += int(r["returncode"] == 0)
        print(f"  {r['name']:<{name_w}}  {status:<14}  {r['seconds']:7.1f}s")
    print(f"\n{n_ok}/{len(results)} runs ok in {total:.1f}s")
    print(f"logs: {log_dir}")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
