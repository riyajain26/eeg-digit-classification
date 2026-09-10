"""
Artifact-detection strategies for EEG preprocessing.

Every variant registered here bundles four functions that must be used
together (see ArtifactStrategy): detecting bad channels, computing raw
diagnostic arrays, deriving flagging thresholds from train data, and
applying those thresholds to any split - because thresholds derived by
one variant's logic are meaningless to a different variant's flagging
function.

The one variant currently registered, percentile_multi_criteria, runs two
distinct, complementary checks:
1. Bad-channel detection: a *global*, per-channel quality check (dead/
   faulty electrode signature - near-zero MAD despite normal std). These
   channels are excluded entirely, not left for per-trial flagging.
2. Artifact flagging: a *per-trial* check on the remaining good channels,
   using thresholds derived from the actual data's percentiles (not fixed
   Gaussian-style assumptions, which badly over-flag on heavy-tailed EEG
   once many channels per trial are checked).

To add a new variant (e.g. an ICA-based approach): write functions
matching ArtifactStrategy's four-function contract, bundle them, register
under a new name, and set cfg.artifact.variant to select it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import kurtosis as kurt_fn

from src.utils.registry import Registry

ARTIFACT_REGISTRY = Registry(step_name="artifact")


@dataclass
class ArtifactStrategy:
    """
    Args/returns for each slot match the percentile_multi_criteria
    functions below exactly - see their docstrings for the authoritative
    contract, since a different variant's internals (e.g. an ICA-based
    approach) may differ substantially while still fitting these same
    four call sites.
    """
    detect_bad_channels: Callable
    compute_arrays: Callable
    derive_thresholds: Callable
    flag: Callable


def _detect_bad_channels_percentile(scale: np.ndarray, bad_channel_scale_floor: float = 1e-7) -> list[int]:
    """
    Flags channels showing a dead/faulty-electrode signature: normalization
    scale (MAD-derived) at or below `bad_channel_scale_floor` - i.e.
    essentially zero variation, indicating a dead lead or short rather
    than real EEG signal.

    Args:
        scale: the `scale` array from a normalization strategy's fit() -
            (1, n_channels, 1).
        bad_channel_scale_floor: threshold below which a channel is
            considered dead (cfg.artifact.bad_channel_scale_floor).
    Returns:
        Sorted list of bad channel indices (into the nominal channel axis).
    """
    return [int(c) for c in np.where(scale.squeeze() <= bad_channel_scale_floor)[0]]


def exclude_channels(eeg: np.ndarray, bad_channels: list[int], n_channels_nominal: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Drops bad_channels from the channel axis. Not part of ArtifactStrategy
    (every variant needs this exact operation once it has a bad-channel
    list, regardless of how that list was derived), so it's a plain shared
    utility rather than a per-variant function.

    Returns:
        (reduced eeg array, array of retained ORIGINAL channel indices -
        needed later to know which physical channel each remaining column
        corresponds to).
    """
    good_channels = np.array([c for c in range(n_channels_nominal) if c not in bad_channels])
    return eeg[:, good_channels, :], good_channels


def _compute_arrays_percentile(eeg_normalized: np.ndarray) -> dict:
    """Raw per-channel-per-trial diagnostic arrays (not yet flagged) -
    amplitude, sample-to-sample jump size, standard deviation, and
    kurtosis, each (n_trials, n_channels)."""
    amp_max = np.abs(eeg_normalized).max(axis=2)
    jump_max = np.abs(np.diff(eeg_normalized, axis=2)).max(axis=2)
    trial_std = eeg_normalized.std(axis=2)

    n_trials, n_channels, _ = eeg_normalized.shape
    kurt_vals = np.zeros((n_trials, n_channels), dtype=np.float64)
    for c in range(n_channels):
        raw_kurt = kurt_fn(eeg_normalized[:, c, :].astype(np.float64), axis=1)
        # Undefined kurtosis on near-zero-variance segments -> never
        # flagged, rather than silently propagating NaN downstream.
        kurt_vals[:, c] = np.nan_to_num(raw_kurt, nan=-np.inf)

    return {"amplitude": amp_max, "jump": jump_max, "std": trial_std, "kurtosis": kurt_vals}


def _derive_thresholds_percentile(train_arrays: dict, artifact_percentile: float, flatline_percentile: float) -> dict:
    """
    Thresholds derived from the TRAIN data's own percentile distribution -
    NOT fixed Gaussian-style values, which badly over-flag on heavy-tailed
    EEG once many channels per trial are checked (multiple-comparisons
    effect). Always fit on train; reused unchanged on val/test.
    """
    kurt_finite = np.where(np.isinf(train_arrays["kurtosis"]), np.nan, train_arrays["kurtosis"])
    return {
        "amplitude": float(np.percentile(train_arrays["amplitude"], artifact_percentile)),
        "jump": float(np.percentile(train_arrays["jump"], artifact_percentile)),
        "kurtosis": float(np.nanpercentile(kurt_finite, artifact_percentile)),
        "flatline": float(np.percentile(train_arrays["std"], flatline_percentile)),
    }


def _flag_artifacts_percentile(arrays: dict, thresholds: dict, trial_concern_min_channels: int) -> dict:
    """
    Applies derive_thresholds' thresholds to ANY split's diagnostic arrays
    (train, val, or test) to produce per-trial/per-channel flags.

    Args:
        arrays: output of compute_arrays() for the split being flagged.
        thresholds: output of derive_thresholds() - always fit on TRAIN.
        trial_concern_min_channels: a trial is flagged "of concern" only
            if at least this many channels show SOME flag - a single
            noisy channel isn't enough on its own to discard a trial.
    """
    amp_flag = arrays["amplitude"] > thresholds["amplitude"]
    jump_flag = arrays["jump"] > thresholds["jump"]
    kurt_flag = arrays["kurtosis"] > thresholds["kurtosis"]
    flatline_flag = arrays["std"] <= thresholds["flatline"]  # <=, not < - deliberate (catches exact-zero flatlines)

    any_flag = amp_flag | jump_flag | kurt_flag | flatline_flag
    n_flagged = any_flag.sum(axis=1)
    trial_concern = n_flagged >= trial_concern_min_channels

    return {
        "amplitude": amp_flag, "jump": jump_flag,
        "kurtosis": kurt_flag, "flatline": flatline_flag,
        "any": any_flag, "n_flagged_channels_per_trial": n_flagged,
        "trial_concern": trial_concern,
    }


ARTIFACT_REGISTRY.register("percentile_multi_criteria")(
    ArtifactStrategy(
        detect_bad_channels=_detect_bad_channels_percentile,
        compute_arrays=_compute_arrays_percentile,
        derive_thresholds=_derive_thresholds_percentile,
        flag=_flag_artifacts_percentile,
    )
)


def save_artifact_params(path: Path, bad_channels: list[int], thresholds: dict) -> None:
    with open(path, "w") as f:
        json.dump({"bad_channels": bad_channels, "thresholds": thresholds}, f, indent=2)


def load_artifact_params(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
