"""
Leakage-safe, block-aware train/val splitting (Notebook 03 logic).

An entire (sessionnum, blocknum) block is assigned to one split — never
split across boundaries — to avoid separating paired blank/digit trials or
temporally adjacent trials.
"""

from pathlib import Path

import h5py
import numpy as np


def check_session_overlap(path_a: Path, path_b: Path) -> set:
    """Returns the set of sessionnum values present in both files (should be empty)."""
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
    Randomly assigns each unique (sessionnum, blocknum) block to 'train' or
    'val', then maps every trial to its block's assignment. Returns an array
    of split labels, one per trial, same order as the input arrays.
    """
    rng = np.random.default_rng(seed)
    block_ids = list(zip(sessionnum.tolist(), blocknum.tolist()))
    unique_blocks = sorted(set(block_ids))

    block_to_split = {
        block: ("val" if rng.random() < val_fraction else "train")
        for block in unique_blocks
    }

    return np.array([block_to_split[b] for b in block_ids])


def validate_split(sessionnum, blocknum, trial_splits: np.ndarray) -> dict:
    """Confirms zero block overlap between train/val — hard assert, not eyeballing."""
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
    Writes separate output files per split (not a shared file + mask) — for
    faster reads and to make mixing splits structurally impossible later.
    """
    with h5py.File(source_path, "r") as src:
        keys = list(src.keys())

        for split_name, out_path in output_paths.items():
            mask = trial_splits == split_name
            with h5py.File(out_path, "w") as dst:
                for key in keys:
                    dst.create_dataset(
                        key, data=src[key][:][mask],
                        compression="gzip" if key == "eeg" else None,
                    )
