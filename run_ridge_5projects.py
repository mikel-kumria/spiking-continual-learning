"""One-off driver: ridge sweep over five SHD parent folders, sequentially.

Each parent folder is run as its own W&B *project* (5 projects total). Folders
are processed strictly one after another (all runs of folder 1 finish before
folder 2 starts); within a folder the existing memory-aware scheduler may run
several ridge jobs in parallel. A final report verifies every run returned 0.

Run:  .venv/bin/python run_ridge_5projects.py
"""
import sys
from datetime import datetime

import run_snn_shd_sweep as sweep


# Parent folders (relative to SHD_PREPROCESSED), processed in this order.
# Each becomes its own W&B project (project name == folder name).
FOLDERS = [
    "SHD_dt14ms_1400ms",
    "SHD_dt14ms_700ms",
    "SHD_dt7ms_700ms",
    "SHD_dt3ms_700ms",
    "SHD_dt1ms_700ms",
]


def build_jobs(folder):
    """Discover dataset dirs under one parent folder (dedup, keep order)."""
    seen = set()
    jobs = []
    resolved = sweep.resolve_entry(folder)
    for d in sweep.discover_datasets(resolved):
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        steps = sweep.estimate_nb_steps(d)
        jobs.append({
            "name": sweep.run_name_for(d),
            "data_dir": d,
            "est_ram": sweep.estimate_ram_gb(d),
            "est_steps": steps,
            "is_heavy": steps is not None and steps >= sweep.HEAVY_NB_STEPS,
        })
    return jobs


def main():
    sweep.CONFIG["mode"] = "ridge"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log = sweep.SCRIPT_DIR / "sweep_logs" / f"ridge_5proj_{stamp}"
    base_log.mkdir(parents=True, exist_ok=True)

    print(f"=== RIDGE 5-project sweep ===  stamp={stamp}")
    print(f"data root : {sweep.DATA_ROOT}")
    print(f"folders   : {len(FOLDERS)} (each -> its own W&B project)")
    print(f"logs      : {base_log}\n")

    overall = []  # (folder, results)
    for folder in FOLDERS:
        jobs = build_jobs(folder)
        if not jobs:
            print(f"WARNING: no datasets found for '{folder}' -- skipping\n")
            overall.append((folder, []))
            continue

        # one project per folder
        sweep.CONFIG["wandb_project"] = folder
        run_group = f"ridge_{folder}_{stamp}"
        log_dir = base_log / folder
        log_dir.mkdir(parents=True, exist_ok=True)

        n_heavy = sum(1 for j in jobs if j["is_heavy"])
        print(f"########## {folder} ##########")
        print(f"project   : {folder}")
        print(f"datasets  : {len(jobs)}  (heavy={n_heavy})")
        for j in jobs:
            ram = f"{j['est_ram']:.1f}GB" if j["est_ram"] else "?GB"
            t = f"T={j['est_steps']}" if j["est_steps"] else "T=?"
            print(f"  - {j['name']}  (~{ram}, {t}{' heavy' if j['is_heavy'] else ''})")
        print(flush=True)

        results = sweep.run_sweep(jobs, log_dir, run_group)
        overall.append((folder, results))

        n_ok = sum(1 for r in results if r["returncode"] == 0)
        print(f"\n[{folder}] {n_ok}/{len(results)} runs ok\n", flush=True)

    # ---- final verification report ----
    print("\n================= FINAL VERIFICATION =================")
    grand_total = grand_ok = 0
    all_good = True
    for folder, results in overall:
        n = len(results)
        n_ok = sum(1 for r in results if r["returncode"] == 0)
        grand_total += n
        grand_ok += n_ok
        status = "OK" if n_ok == n and n > 0 else "INCOMPLETE/FAIL"
        if n_ok != n or n == 0:
            all_good = False
        print(f"  {folder:<22} {n_ok}/{n}  [{status}]")
        for r in results:
            if r["returncode"] != 0:
                print(f"      FAILED rc={r['returncode']}: {r['name']}  "
                      f"(log: {r['log_path']})")
    print(f"\n  TOTAL: {grand_ok}/{grand_total} runs completed successfully")
    print("=====================================================")

    sys.exit(0 if all_good else 1)


if __name__ == "__main__":
    main()
