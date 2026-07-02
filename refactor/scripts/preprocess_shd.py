#!/usr/bin/env python3
"""Preprocess raw SHD into the split ``.npz`` files + manifest (no training).

Example::

    python refactor/scripts/preprocess_shd.py \
      --config refactor/configs/pretrain_default.yaml

Writes to ``<output_root>/<run_name>/dataset/`` and
``<output_root>/<run_name>/preprocessing_manifest.json``.
"""
from __future__ import annotations

import argparse
import os

from _common import add_package_to_path

add_package_to_path()

from shd_cl.data.preprocessing import PreprocessConfig, preprocess  # noqa: E402
from shd_cl.utils.audit import audit_dataset_dir, print_audit  # noqa: E402
from shd_cl.utils.config import apply_overrides, load_config  # noqa: E402
from shd_cl.utils.determinism import set_determinism  # noqa: E402


OVERRIDABLE = [
    "experiment_regime", "run_name", "output_root", "train_h5", "test_h5",
    "merge_train_test", "synthetic_samples_per_class", "removed_class",
    "dataset_binning_ms", "dataset_max_seconds", "nb_inputs", "n_compressed_channels",
    "channel_compression_method", "condition_or", "train_fraction", "test_fraction",
    "seed",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    for key in OVERRIDABLE:
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args, OVERRIDABLE)
    set_determinism(int(cfg["seed"]))

    regime = cfg["experiment_regime"]
    removed = cfg.get("removed_class")
    removed = int(removed) if (removed is not None and regime == "pretrain_19_class") else None

    run_dir = os.path.join(cfg["output_root"], cfg["run_name"])
    dataset_dir = os.path.join(run_dir, "dataset")

    pcfg = PreprocessConfig(
        experiment_regime=regime,
        removed_class=int(removed) if removed is not None else 10,
        dataset_binning_ms=float(cfg["dataset_binning_ms"]),
        dataset_max_seconds=float(cfg["dataset_max_seconds"]),
        nb_inputs=int(cfg["nb_inputs"]),
        n_compressed_channels=int(cfg["n_compressed_channels"]),
        channel_compression_method=cfg["channel_compression_method"],
        condition_or=int(cfg["condition_or"]),
        train_fraction=float(cfg["train_fraction"]),
        test_fraction=float(cfg["test_fraction"]),
        merge_train_test=bool(cfg["merge_train_test"]),
        seed=int(cfg["seed"]),
        synthetic_samples_per_class=int(cfg.get("synthetic_samples_per_class", 0) or 0),
        train_h5=cfg.get("train_h5"),
        test_h5=cfg.get("test_h5"),
    )
    manifest = preprocess(pcfg, dataset_dir)
    ok = print_audit("dataset", audit_dataset_dir(dataset_dir, manifest))
    print(f"\nSaved dataset to {os.path.abspath(dataset_dir)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
