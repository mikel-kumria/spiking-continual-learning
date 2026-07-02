"""shd_cl: modular SHD spiking continual-learning library.

A clean re-implementation of the monolithic ``pretrain_snn_shd.py`` /
``class_incremental_snn_shd.py`` scripts. The package is split into small,
single-responsibility modules so the same building blocks are reused across the
baseline / pretraining / class-incremental experiments:

    data/        raw SHD events -> dense binned -> channel-compressed -> splits
    models/      surrogate spike, recurrent reservoir, output layers, SNN
    training/    ridge (closed form), bptt (full/last), replay sampler, CIL
    evaluation/  metrics + prediction helpers
    logging/     W&B helpers + hidden raster plot
    utils/       config, determinism, device, checkpointing, audit

The scripts under ``refactor/scripts`` are thin wrappers around these modules.
"""

NUM_CLASSES = 20  # SHD: spoken digits 0-9 in English + German -> labels 0..19
NATIVE_SHD_CHANNELS = 700  # native cochlea channel count before compression
