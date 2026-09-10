"""
Step 1 orchestration: data acquisition + train/val split.

This module is the bridge between the low-level building blocks
(data/acquisition.py, data/train_val_split.py - which know nothing about
config, file layout, or caching) and a future top-level pipeline script -
it knows about all three: it reads paths and parameters from a
PipelineConfig, decides what's already done vs. what needs (re)running,
and calls the low-level functions in the right order with the right
arguments.

Renamed from pipeline.py's run_phase2_data_acquisition() and
run_phase3_splitting() - descriptive names, no phase numbers. Behavior is
unchanged from the original except where noted.

No registry needed in this module: data acquisition + splitting doesn't
currently have alternative strategies to select between. Once a pluggable
step needs one (starting with preprocessing, in the next modularization
step), the pattern will be: this module's config gains a `.variant`
field, and it calls `SOME_REGISTRY.get(cfg.some_step.variant)(...)`
instead of a single hardcoded function - same shape as this file, just
with a lookup added.
"""

from pathlib import Path

import h5py
import numpy as np

from src.config import PipelineConfig
from src.data.acquisition import acquire_split
from src.data.train_val_split import (
    find_session_overlap, assign_block_splits,
    check_split_has_no_block_overlap, materialize_split,
)


def _exists(path: Path) -> bool:
    return path.exists()


def run_data_acquisition(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Streams the train pool and the held-out test set from Hugging Face
    into local HDF5 files, sized per cfg.data (dataset_variant,
    subsample_fraction).

    Idempotent by default: if both output files already exist, does
    nothing (streaming the full dataset is slow - no reason to repeat it
    on every run). Pass force=True to re-acquire from scratch.

    Outputs:
        cfg.data.interim_dir / "train_pool.h5"  - not yet split into train/val
        cfg.data.splits_dir / "test.h5"           - held out, never touched again after this
    """
    cfg.data.interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.splits_dir.mkdir(parents=True, exist_ok=True)

    train_pool_path = cfg.data.interim_dir / "train_pool.h5"
    if force or not _exists(train_pool_path):
        print(f"Data acquisition [{cfg.data.variant_tag}]: streaming train pool "
              f"(target_per_class={cfg.data.target_per_class})...")
        counts = acquire_split(
            output_path=train_pool_path,
            hf_dataset_name=cfg.data.hf_dataset_name,
            hf_split="train",
            target_per_class=cfg.data.target_per_class,
            n_channels=cfg.data.n_channels_nominal,
            n_samples=cfg.data.n_samples,
            max_stream_multiplier=cfg.data.max_stream_multiplier,
        )
        print(f"Train pool counts: {counts}")
    else:
        print(f"Data acquisition: train pool already exists at {train_pool_path}, skipping.")

    test_path = cfg.data.splits_dir / "test.h5"
    if force or not _exists(test_path):
        print(f"Data acquisition [{cfg.data.variant_tag}]: streaming test set "
              f"(target_per_class={cfg.data.test_target_per_class})...")
        counts = acquire_split(
            output_path=test_path,
            hf_dataset_name=cfg.data.hf_dataset_name,
            hf_split="test",
            target_per_class=cfg.data.test_target_per_class,
            n_channels=cfg.data.n_channels_nominal,
            n_samples=cfg.data.n_samples,
            max_stream_multiplier=cfg.data.max_stream_multiplier,
        )
        print(f"Test set counts: {counts}")
    else:
        print(f"Data acquisition: test set already exists at {test_path}, skipping.")


def run_train_val_split(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Splits the acquired train pool into leakage-safe train/val HDF5 files,
    using block-aware assignment (see data/train_val_split.py). Also
    checks for session overlap against the held-out test set and excludes
    any overlapping sessions' trials from train/val (test itself is never
    modified).

    Idempotent by default: if both output files already exist, does
    nothing. Pass force=True to re-split from scratch. Requires
    run_data_acquisition() to have completed first.

    Outputs:
        cfg.data.splits_dir / "train.h5"
        cfg.data.splits_dir / "val.h5"
    """
    train_path = cfg.data.splits_dir / "train.h5"
    val_path = cfg.data.splits_dir / "val.h5"

    if not force and _exists(train_path) and _exists(val_path):
        print("Train/val split: outputs already exist, skipping.")
        return

    train_pool_path = cfg.data.interim_dir / "train_pool.h5"
    test_path = cfg.data.splits_dir / "test.h5"
    if not _exists(train_pool_path) or not _exists(test_path):
        raise RuntimeError(
            f"Train/val split requires run_data_acquisition() to have completed first - "
            f"missing {train_pool_path if not _exists(train_pool_path) else test_path}."
        )

    overlap = find_session_overlap(train_pool_path, test_path)
    with h5py.File(train_pool_path, "r") as f:
        sessionnum, blocknum = f["sessionnum"][:], f["blocknum"][:]

    total_sessions = len(set(sessionnum.tolist()))

    # A single overlapping session (e.g. session 207 at 100% scale) is
    # treated as an isolated data-source quirk, not a systemic leakage
    # problem - excluded from train/val rather than blocking the run
    # entirely. If overlap is large, that DOES indicate a systemic
    # problem, so the 5% threshold below still hard-errors in that case.
    if overlap:
        overlap_fraction = len(overlap) / total_sessions
        if overlap_fraction > 0.05:
            raise RuntimeError(
                f"Session overlap found: {overlap} ({overlap_fraction:.1%} of train pool sessions). "
                "Too large to treat as an isolated edge case - HF's train/test boundary may not "
                "be leakage-safe overall. Resolve before proceeding (e.g. a custom 3-way split)."
            )
        print(f"Train/val split: WARNING - {len(overlap)} session(s) overlap between train pool "
              f"and test ({overlap_fraction:.2%} of train sessions): {overlap}. Excluding these "
              "sessions' trials from train/val - test remains untouched as the official held-out set.")
    else:
        print("Train/val split: session overlap check passed - zero overlap.")

    trial_splits = assign_block_splits(sessionnum, blocknum, cfg.split.val_fraction, cfg.split.seed)

    if overlap:
        exclude_mask = np.isin(sessionnum, list(overlap))
        trial_splits = trial_splits.copy()
        trial_splits[exclude_mask] = "excluded"
        print(f"Train/val split: excluded {exclude_mask.sum()} trials from {len(overlap)} overlapping session(s).")

    validation = check_split_has_no_block_overlap(sessionnum, blocknum, trial_splits)
    if validation["block_overlap"]:
        raise RuntimeError(f"Block overlap detected: {validation['block_overlap']}")
    print(f"Train/val split: train={validation['n_train']}, val={validation['n_val']}, zero overlap confirmed.")

    materialize_split(train_pool_path, trial_splits, {"train": train_path, "val": val_path})
    print(f"Train/val split: wrote {train_path} and {val_path}.")


def run_data_preparation(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Full Step 1: acquisition, then train/val split. This is the single
    function later pipeline code (and scripts/run_data_preparation.py)
    should call for this step - it doesn't need to know acquisition and
    splitting are two separate functions internally.
    """
    run_data_acquisition(cfg, force=force)
    run_train_val_split(cfg, force=force)
