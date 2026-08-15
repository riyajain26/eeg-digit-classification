"""
Streaming extraction of MindBigData2023 from Hugging Face into local HDF5,
with class-balanced blank/digit pairing (Notebook 02 logic).
"""

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset


def extract_eeg(row: dict, channel_names: list[str], n_samples: int) -> np.ndarray:
    """Reshape one streamed row's flat EEG columns into (n_channels, n_samples)."""
    return np.array(
        [[row[f"{ch}_{i}"] for i in range(n_samples)] for ch in channel_names],
        dtype=np.float32,
    )


def detect_channel_names(row: dict) -> list[str]:
    return sorted({
        k.rsplit("_", 1)[0] for k in row.keys()
        if "_" in k and not k.startswith("label")
    })


def _append_row(dset: h5py.Dataset, value) -> None:
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = value


def extract_split(
    output_path: Path,          # where to write the output HDF5 file
    hf_dataset_name: str,       # HF repo id, e.g. CFG.data.hf_dataset_name
    hf_split: str,               # which HF split to stream: "train" or "test"
    target_per_class: int | None,  # digit trials per class to collect; None = no cap (full dataset)
    n_channels: int,              # expected channel count (n_channels_nominal from config)
    n_samples: int,                # expected samples per channel (n_samples from config)
    max_stream_multiplier: int = 20,   # safety cap multiplier - see DataConfig docstring
    verbose: bool = True,               # whether to print progress as it streams
) -> dict:
    """
    Stream `hf_split` from `hf_dataset_name`, extracting a class-balanced
    subsample: each accepted digit trial (0-9) is paired with its
    immediately-preceding blank (-1) trial, so the binary classes end up
    balanced by construction rather than sampled independently.

    target_per_class=None means "collect everything the stream has" (used
    for subsample_fraction=1.0 / full-dataset runs) - in this mode there is
    no per-class stopping condition; the max_rows safety cap (set very high)
    is the only thing that eventually ends the loop.

    Returns a dict mapping digit label -> final count actually collected,
    for the caller to verify every class hit its target (or diagnose which
    one didn't, if the stream ran out first).
    """
    ds = load_dataset(hf_dataset_name, split=hf_split, streaming=True)

    digit_counts: dict[int, int] = defaultdict(int)   # progress tracker per digit class
    channel_names: list[str] | None = None             # filled in on first row
    pending_blank = None                                 # most recent blank, awaiting its digit pair
    trial_idx = 0                                          # total trials written so far

    if target_per_class is not None:
        max_rows = target_per_class * 10 * max_stream_multiplier
    else:
        # No target to base a cap on - use a very high safety ceiling purely
        # as a runaway guard, not a real stopping condition. The stream
        # itself running out is the expected way this loop ends.
        max_rows = 50_000_000

    with h5py.File(output_path, "a") as f:
        # Resizable datasets - shape starts at 0 and grows one row at a time
        # via _append_row(), since we don't know the final trial count upfront.
        eeg_ds = f.create_dataset("eeg", shape=(0, n_channels, n_samples),
                                   maxshape=(None, n_channels, n_samples),
                                   chunks=(1, n_channels, n_samples),
                                   compression="gzip", dtype="float32")
        label_binary_ds = f.create_dataset("label_binary", shape=(0,), maxshape=(None,), dtype="int8")
        label_digit_ds = f.create_dataset("label_digit", shape=(0,), maxshape=(None,), dtype="int8")
        sessionnum_ds = f.create_dataset("sessionnum", shape=(0,), maxshape=(None,), dtype="int64")
        blocknum_ds = f.create_dataset("blocknum", shape=(0,), maxshape=(None,), dtype="int64")
        blockpos_ds = f.create_dataset("blockpos", shape=(0,), maxshape=(None,), dtype="int64")
        timestamp_ds = f.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype="int64")

        def write_trial(row, label_digit, label_binary):
            """Writes one trial's EEG + labels + metadata across all datasets above."""
            nonlocal trial_idx
            eeg = extract_eeg(row, channel_names, n_samples)
            _append_row(eeg_ds, eeg)
            _append_row(label_binary_ds, label_binary)   # 0=blank, 1=digit - for Stage 1
            _append_row(label_digit_ds, label_digit)       # -1=blank, 0-9=digit - for Stage 2
            _append_row(sessionnum_ds, row["sessionnum"])   # needed later for leakage-safe splitting
            _append_row(blocknum_ds, row["blocknum"])         # needed later for leakage-safe splitting
            _append_row(blockpos_ds, row["blockpos"])           # position within block (metadata)
            _append_row(timestamp_ds, row["timestamp"])           # capture time (metadata)
            trial_idx += 1

        for row_num, row in enumerate(ds):
            if row_num >= max_rows:
                if verbose:
                    print("Hit streaming cap before all classes filled.")
                break

            label = row["label"]   # -1 = blank screen, 0-9 = digit shown

            if channel_names is None:
                # Only needs to happen once - every row has the same columns.
                channel_names = detect_channel_names(row)
                if verbose:
                    print(f"Detected {len(channel_names)} channels")

            if label == -1:
                # Don't write it yet - just remember it in case the NEXT
                # row is a digit trial that still needs pairing.
                pending_blank = row
                continue

            # When target_per_class is None, never skip on count - always accept.
            if target_per_class is not None and digit_counts[label] >= target_per_class:
                # Already have enough of this digit - skip it, and drop the
                # pending blank too (its digit partner wasn't wanted).
                pending_blank = None
                continue

            if pending_blank is None:
                # Shouldn't normally happen given the dataset's alternating
                # blank/digit pattern - flag it rather than silently skip.
                if verbose:
                    print(f"WARNING: digit {label} at row {row_num} has no preceding blank - skipping")
                continue

            # Write both halves of the pair together, keeping binary classes balanced.
            write_trial(row, label_digit=label, label_binary=1)
            write_trial(pending_blank, label_digit=-1, label_binary=0)

            digit_counts[label] += 1
            pending_blank = None

            if verbose and trial_idx % 500 == 0:
                print(f"{trial_idx} trials written...")

            # Only check the "all classes reached target" stop condition
            # when there IS a target - otherwise let the stream run out naturally.
            if target_per_class is not None and all(digit_counts[d] >= target_per_class for d in range(10)):
                if verbose:
                    print(f"All digit classes reached target after {trial_idx} trials written.")
                break

    return dict(digit_counts)
