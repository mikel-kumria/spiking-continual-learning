"""Shared helpers for the thin CLI scripts (path bootstrap + dataset ensure)."""
from __future__ import annotations

import os
import sys
from typing import Optional


def add_package_to_path() -> str:
    """Put ``refactor/`` on ``sys.path`` so ``import shd_cl`` works from anywhere.

    Returns the ``refactor/`` directory (parent of this ``scripts/`` folder).
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    refactor_dir = os.path.dirname(scripts_dir)
    if refactor_dir not in sys.path:
        sys.path.insert(0, refactor_dir)
    return refactor_dir


def ensure_dataset(cfg: dict, run_dir: str, regime: str,
                   removed_class: Optional[int]) -> dict:
    """Preprocess into ``run_dir/dataset`` if no manifest exists yet; else reuse.

    Returns the preprocessing manifest. Reusing an existing dataset makes it cheap
    to run ``preprocess_shd.py`` then ``pretrain.py`` on the same run folder.
    """
    from shd_cl.data.io import read_json
    from shd_cl.data.preprocessing import PreprocessConfig, preprocess

    dataset_dir = os.path.join(run_dir, "dataset")
    manifest_path = os.path.join(run_dir, "preprocessing_manifest.json")
    if os.path.isfile(manifest_path):
        print(f"[data] reusing existing dataset at {dataset_dir}")
        return read_json(manifest_path)

    pcfg = PreprocessConfig(
        experiment_regime=regime,
        removed_class=int(removed_class) if removed_class is not None else 10,
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
        synthetic_samples_per_class=int(cfg.get("synthetic_samples_per_class", 0)),
        train_h5=cfg.get("train_h5"),
        test_h5=cfg.get("test_h5"),
    )
    return preprocess(pcfg, dataset_dir)
