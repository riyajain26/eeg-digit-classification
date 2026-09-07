"""
Leakage-safe, block-aware train/val splitting.

Renamed from splitting.py - "train_val_split" says what this produces;
"splitting" was ambiguous with e.g. train/test splitting (handled
separately, at acquisition time - see acquisition.py). No behavior
changes from the original.

Core rule: an entire (sessionnum, blocknum) block is assigned to ONE
split - train or val - and never split across that boundary. This keeps
paired blank/digit trials and temporally-adjacent trials together,
preventing the kind of near-duplicate-across-splits leakage that would
make a val score look better than the model actually generalizes.

No variant registry here: there's one splitting strategy in use
(block-aware random assignment) and no current need for alternatives.
If a second splitting strategy is ever wanted (e.g. stratified by
session recency, or a fixed hold-out block list), this is the file to
add a src.utils.registry.Registry to.
"""

from pathlib import Path

import h5py
import numpy as np


def find_session_overlap(path_a: Path, path_b: Path) -> set:
    """
    Returns the set of `sessionnum` values present in BOTH HDF5 files at
    path_a and path_b. Used to check whether the train pool and the
    held-out test set (acquired independently from HF's train/test
    split) share any recording sessions - which would be a leakage risk
    if left unaddressed. An empty set is the expected, healthy result.
    """
    with h5py.File(path_a, "r") as f:
        sessions_a = set(f["sessionnum"][:])
    with h5py.File(path_b, "r") as f:
        sessions_b = set(f["sessionnum"][:])
    return sessions_a & sessions_b


def assign_block_splits(
    sessionnum: np.ndarray,
    blocknum: np.ndarray,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Randomly assigns each unique (sessionnum, blocknum) block to "train"
    or "val", then maps every trial to its own block's assignment - so
    two trials from the same block always land in the same split.

    Args:
        sessionnum: (n_trials,) session id per trial.
        blocknum: (n_trials,) block id per trial (unique only within a
            session - the (sessionnum, blocknum) PAIR is the real block key).
        val_fraction: target fraction of BLOCKS (not trials) assigned to val.
        seed: RNG seed - same seed always produces the same split, given
            the same input arrays.

    Returns:
        (n_trials,) array of "train"/"val" strings, one per input trial,
        in the same order as sessionnum/blocknum.
    """
    rng = np.random.default_rng(seed)
    block_ids = list(zip(sessionnum.tolist(), blocknum.tolist()))
    unique_blocks = sorted(set(block_ids))

    block_to_split = {
        block: ("val" if rng.random() < val_fraction else "train")
        for block in unique_blocks
    }

    return np.array([block_to_split[b] for b in block_ids])


def check_split_has_no_block_overlap(sessionnum, blocknum, trial_splits: np.ndarray) -> dict:
    """
    Confirms zero (sessionnum, blocknum) block appears in both train and
    val - a hard, direct check rather than trusting assign_block_splits's
    logic by inspection alone.

    Returns:
        dict with "block_overlap" (should be an empty set), "n_train",
        and "n_val" trial counts.
    """
    block_ids = list(zip(sessionnum.tolist(), blocknum.tolist()))
    train_blocks = {b for b, s in zip(block_ids, trial_splits) if s == "train"}
    val_blocks = {b for b, s in zip(block_ids, trial_splits) if s == "val"}
    overlap = train_blocks & val_blocks

    return {
        "block_overlap": overlap,
        "n_train": int((trial_splits == "train").sum()),
        "n_val": int((trial_splits == "val").sum()),
    }


def materialize_split(
    source_path: Path,
    trial_splits: np.ndarray,
    output_paths: dict[str, Path],
) -> None:
    """
    Writes each split ("train", "val", ...) to its own HDF5 file, rather
    than one shared file plus a boolean mask - faster to read later, and
    makes accidentally mixing splits structurally impossible.

    Reads ONLY the rows belonging to each split from disk (h5py fancy
    indexing) rather than loading the full source dataset into memory
    first - the naive `src[key][:][mask]` pattern reads the ENTIRE array
    from disk once per split, which at 100% scale is ~15.7GB per read
    and was the direct cause of an earlier out-of-memory crash.

    Args:
        source_path: HDF5 file containing every trial (e.g. the raw
            train pool, before splitting).
        trial_splits: (n_trials,) array of split-name strings, same
            order as the rows in source_path, e.g. from assign_block_splits().
        output_paths: maps each split name ("train", "val") to the
            output file path to write that split's rows into.
    """
    with h5py.File(source_path, "r") as src:
        keys = list(src.keys())

        for split_name, out_path in output_paths.items():
            idx = np.where(trial_splits == split_name)[0]   # ascending order - required for h5py fancy indexing
            with h5py.File(out_path, "w") as dst:
                for key in keys:
                    dst.create_dataset(
                        key, data=src[key][idx],
                        compression="gzip" if key == "eeg" else None,
                    )
