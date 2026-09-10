"""
Data acquisition: streams MindBigData2023 from Hugging Face into local
HDF5 files, one call per split ("train" or "test"), with class-balanced
blank/digit pairing.

Renamed from conversion.py - "acquisition" matches what this step
actually does (get raw data from the source) and matches the pipeline
stage name it's called from. No behavior changes from the original.

No variant registry here (unlike preprocessing/features/models, covered
in later modules): there is exactly one way to acquire this dataset -
stream it from its one Hugging Face source with this pairing logic - so
a plug-in mechanism would add indirection without adding flexibility.
If a second acquisition strategy is ever needed (e.g. loading from a
local mirror instead of streaming), THIS is the file to add a
src.utils.registry.Registry to.
"""

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset


def extract_eeg_trial(row: dict, channel_names: list[str], n_samples: int) -> np.ndarray:
    """
    Reshapes one streamed HF row's flat per-sample columns (e.g.
    "Fp1_0", "Fp1_1", ..., "Fp1_249") into a proper (n_channels,
    n_samples) array for a single trial.

    Args:
        row: one row dict as yielded by the HF streaming dataset.
        channel_names: ordered channel names to extract, e.g. from
            detect_channel_names(). Determines both the row order of
            the output array and which columns are read.
        n_samples: samples per channel for this trial (differs by
            dataset variant: 256 for the 2B release, 500 for 8B).

    Returns:
        (n_channels, n_samples) float32 array.
    """
    return np.array(
        [[row[f"{ch}_{i}"] for i in range(n_samples)] for ch in channel_names],
        dtype=np.float32,
    )


def detect_channel_names(row: dict) -> list[str]:
    """
    Infers the sorted list of EEG channel names from one row's column
    names, by stripping the trailing "_<sample_index>" suffix from every
    non-label column and deduplicating. Only needs to run once per
    stream (every row has the same columns), so callers should cache
    the result rather than calling this per-row.
    """
    return sorted({
        k.rsplit("_", 1)[0] for k in row.keys()
        if "_" in k and not k.startswith("label")
    })


def _append_row(dset: h5py.Dataset, value) -> None:
    """Grows a resizable HDF5 dataset by one row and writes `value` into it."""
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = value


def acquire_split(
    output_path: Path,
    hf_dataset_name: str,
    hf_split: str,
    target_per_class: int | None,
    n_channels: int,
    n_samples: int,
    max_stream_multiplier: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Streams one Hugging Face split into a local HDF5 file, keeping only
    class-balanced (blank, digit) trial pairs.

    Pairing logic: MindBigData alternates a blank/rest trial immediately
    before each digit trial. Every accepted digit trial (0-9) is written
    together with its immediately-preceding blank trial, so binary
    (blank vs. digit) classes end up balanced by construction rather
    than by independent sampling of each class after the fact.

    Args:
        output_path: where to write the output HDF5 file. Opened in
            append mode, so calling this twice on the same path resumes
            writing into the same file rather than overwriting it -
            callers that want a fresh file should delete it first.
        hf_dataset_name: Hugging Face repo id, e.g. cfg.data.hf_dataset_name.
        hf_split: which HF split to stream - "train" or "test".
        target_per_class: digit trials to collect per class (0-9).
            None means "collect everything the stream has" (used for
            subsample_fraction=1.0 / full-dataset runs) - in this mode
            there's no per-class stopping condition, and the stream
            running out is the expected way the loop ends.
        n_channels: expected channel count (cfg.data.n_channels_nominal).
        n_samples: expected samples per channel for this dataset variant
            (cfg.data.n_samples - 256 for 2B, 500 for 8B).
        max_stream_multiplier: safety-cap multiplier on rows SCANNED
            (not rows kept) when target_per_class is set - guards
            against an unexpectedly-imbalanced stream looping forever
            waiting for one rare class to fill.
        verbose: whether to print progress as trials are written.

    Returns:
        dict mapping digit label (0-9) -> final count actually
        collected, so the caller can confirm every class hit its target
        or diagnose which one didn't if the stream ran out first.
    """
    ds = load_dataset(hf_dataset_name, split=hf_split, streaming=True)

    digit_counts: dict[int, int] = defaultdict(int)   # progress tracker per digit class
    channel_names: list[str] | None = None             # filled in from the first row
    pending_blank = None                                 # most recent blank, awaiting its digit pair
    trial_idx = 0                                          # total trials written so far (both halves of each pair)

    if target_per_class is not None:
        # Cap on rows SCANNED, not rows kept - deliberately generous
        # (x10 classes x max_stream_multiplier) since most scanned rows
        # are skipped (already-full classes, or blanks awaiting pairing).
        max_rows_scanned = target_per_class * 10 * max_stream_multiplier
    else:
        # No target to size a cap from - use a very high safety ceiling
        # purely as a runaway guard, not a real stopping condition. The
        # stream running out is the expected way this loop ends.
        max_rows_scanned = 50_000_000

    with h5py.File(output_path, "a") as f:
        # Resizable datasets: start at 0 rows, grow one row at a time via
        # _append_row(), since the final trial count isn't known upfront.
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

        def write_trial(row: dict, label_digit: int, label_binary: int) -> None:
            """Writes one trial's EEG + labels + split-relevant metadata
            (sessionnum/blocknum, needed later for leakage-safe
            splitting) across every dataset opened above."""
            nonlocal trial_idx
            eeg = extract_eeg_trial(row, channel_names, n_samples)
            _append_row(eeg_ds, eeg)
            _append_row(label_binary_ds, label_binary)   # 0=blank, 1=digit - Stage 1 target
            _append_row(label_digit_ds, label_digit)       # -1=blank, 0-9=digit - Stage 2 target
            _append_row(sessionnum_ds, row["sessionnum"])
            _append_row(blocknum_ds, row["blocknum"])
            _append_row(blockpos_ds, row["blockpos"])       # position within block (metadata only)
            _append_row(timestamp_ds, row["timestamp"])       # capture time (metadata only)
            trial_idx += 1

        for row_num, row in enumerate(ds):
            if row_num >= max_rows_scanned:
                if verbose:
                    print("acquire_split: hit the scan cap before every class reached its target.")
                break

            label = row["label"]   # -1 = blank screen, 0-9 = digit shown

            if channel_names is None:
                # Only needs to happen once - every row has the same columns.
                channel_names = detect_channel_names(row)
                if verbose:
                    print(f"acquire_split: detected {len(channel_names)} channels")

            if label == -1:
                # Hold onto it - don't write yet - in case the NEXT row
                # is the digit trial it should be paired with.
                pending_blank = row
                continue

            # target_per_class=None means never skip on count - always accept.
            if target_per_class is not None and digit_counts[label] >= target_per_class:
                # Already have enough of this digit - skip it, and drop
                # the pending blank too (its digit partner wasn't wanted).
                pending_blank = None
                continue

            if pending_blank is None:
                # Shouldn't normally happen given the dataset's alternating
                # blank/digit pattern - flagged rather than silently skipped.
                if verbose:
                    print(f"acquire_split: WARNING - digit {label} at row {row_num} "
                          "has no preceding blank, skipping.")
                continue

            # Write both halves of the pair together, keeping binary classes balanced.
            write_trial(row, label_digit=label, label_binary=1)
            write_trial(pending_blank, label_digit=-1, label_binary=0)

            digit_counts[label] += 1
            pending_blank = None

            if verbose and trial_idx % 500 == 0:
                print(f"acquire_split: {trial_idx} trials written...")

            # Only check the "every class reached target" stop condition
            # when there IS a target - otherwise let the stream run out naturally.
            if target_per_class is not None and all(digit_counts[d] >= target_per_class for d in range(10)):
                if verbose:
                    print(f"acquire_split: every digit class reached target after {trial_idx} trials written.")
                break

    return dict(digit_counts)
