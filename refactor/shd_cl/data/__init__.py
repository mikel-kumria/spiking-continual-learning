"""Data pipeline: raw SHD events -> dense binning -> channel compression -> splits.

The canonical on-disk contract (NEVER transposed) is::

    X : uint8 array  [N, T, C]   (time = axis 1, channel = last axis)
    y : int64 array  [N]         (ORIGINAL SHD labels in [0, 19]; no global remap)
    speaker : int64 array [N]    (or all -1 if unavailable)
"""

SPLIT_NAMES = (
    "pretrain_train", "pretrain_test",
    "continual_train", "continual_test",
)
BASELINE_SPLIT_NAMES = ("train", "test")
