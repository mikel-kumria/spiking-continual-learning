"""Runtime consistency audits shared by scripts and the smoke test.

Each ``audit_*`` returns a list of ``(check_name, ok, detail)`` tuples so callers
can print a report and optionally fail hard. These catch the classic mistakes:
transposed axes, global label remaps, leaked classes, wrong W_out shape and a
non-zero removed-class column after 19-class pretraining.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import NUM_CLASSES
from ..data.io import load_npz_split

Check = Tuple[str, bool, str]


def _c(name: str, ok: bool, detail: str = "") -> Check:
    return (name, bool(ok), detail)


def audit_dataset_dir(dataset_dir: str, manifest: dict) -> List[Check]:
    """Shape / label-range / leakage checks on the saved splits."""
    checks: List[Check] = []
    nb_steps = int(manifest["nb_steps"])
    n_ch = int(manifest["n_compressed_channels"])
    removed = manifest.get("removed_class")
    regime = manifest.get("experiment_regime", "pretrain_19_class")
    split_names = (("train", "test") if regime == "baseline_20_class"
                   else ("pretrain_train", "pretrain_test",
                         "continual_train", "continual_test"))
    for name in split_names:
        path = os.path.join(dataset_dir, f"{name}.npz")
        if not os.path.isfile(path):
            checks.append(_c(f"{name}:exists", False, f"missing {path}"))
            continue
        X, y, _ = load_npz_split(path)
        checks.append(_c(f"{name}:shape[N,T,C]",
                         X.ndim == 3 and X.shape[1] == nb_steps and X.shape[2] == n_ch,
                         f"{tuple(X.shape)} vs (*, {nb_steps}, {n_ch})"))
        labs = y.numpy()
        checks.append(_c(f"{name}:labels_in_0_19",
                         labs.size == 0 or (labs.min() >= 0 and labs.max() < NUM_CLASSES),
                         f"min={int(labs.min()) if labs.size else '-'} "
                         f"max={int(labs.max()) if labs.size else '-'}"))
        if regime == "pretrain_19_class" and removed is not None:
            uniq = set(int(v) for v in np.unique(labs))
            if name.startswith("pretrain"):
                checks.append(_c(f"{name}:no_removed_leak", removed not in uniq,
                                 f"removed={removed} present={removed in uniq}"))
            if name.startswith("continual"):
                checks.append(_c(f"{name}:only_removed", uniq <= {removed},
                                 f"labels={sorted(uniq)}"))
    return checks


def audit_model_shapes(model, manifest: dict) -> List[Check]:
    checks = []
    checks.append(_c("model.nb_inputs==manifest",
                     model.nb_inputs == int(manifest["nb_inputs"]),
                     f"{model.nb_inputs} vs {manifest['nb_inputs']}"))
    checks.append(_c("W_out.shape==[H,nb_outputs]",
                     tuple(model.W_out.shape) == (model.nb_hidden, model.nb_outputs),
                     f"{tuple(model.W_out.shape)}"))
    checks.append(_c("nb_outputs>=NUM_CLASSES", model.nb_outputs >= NUM_CLASSES,
                     f"{model.nb_outputs}"))
    return checks


def audit_removed_column_zero(model, removed_class: Optional[int]) -> List[Check]:
    if removed_class is None:
        return [_c("removed_column_zero", True, "n/a (baseline)")]
    col = model.W_out.detach()[:, int(removed_class)]
    return [_c("removed_column_zero", bool(torch.count_nonzero(col).item() == 0),
               f"nnz={int(torch.count_nonzero(col).item())}")]


def audit_active_classes(active_classes: Sequence[int], removed_class: Optional[int]
                         ) -> List[Check]:
    checks = []
    if removed_class is None:
        checks.append(_c("active==20", len(active_classes) == NUM_CLASSES,
                         f"{len(active_classes)}"))
    else:
        checks.append(_c("active==19", len(active_classes) == NUM_CLASSES - 1,
                         f"{len(active_classes)}"))
        checks.append(_c("removed_not_in_active", removed_class not in active_classes,
                         f"removed={removed_class}"))
    return checks


def print_audit(title: str, checks: List[Check]) -> bool:
    """Print a section of checks; return True iff all passed."""
    print(f"\n[audit] {title}")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return all_ok
