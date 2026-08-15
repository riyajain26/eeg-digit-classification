"""
Bad-channel detection and per-channel, multi-criteria artifact flagging
(Notebook 04, Sections 5-7 final revision).

Two distinct, complementary checks:
1. Bad-channel detection: a *global*, per-channel quality check (dead/
   faulty electrode signature - near-zero MAD despite normal std). These
   channels are excluded entirely, not left for per-trial flagging.
2. Artifact flagging: a *per-trial* check on the remaining good channels,
   using thresholds derived from the actual data's percentiles (not fixed
   Gaussian-style assumptions, which badly over-flag on heavy-tailed EEG).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis as kurt_fn


def detect_bad_channels(scale: np.ndarray, floor: float = 1e-7) -> list[int]:
    """
    ...
    Returns indices of channels showing the dead/faulty-electrode signature...
    """
    return [int(c) for c in np.where(scale.squeeze() <= floor)[0]]


def exclude_channels(eeg: np.ndarray, bad_channels: list[int], n_channels_nominal: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (reduced eeg array, array of retained original channel indices)."""
    good_channels = np.array([c for c in range(n_channels_nominal) if c not in bad_channels])
    return eeg[:, good_channels, :], good_channels


def compute_artifact_arrays(eeg_normalized: np.ndarray) -> dict:
    """Raw per-channel-per-trial diagnostic arrays (not yet flagged)."""
    amp_max = np.abs(eeg_normalized).max(axis=2)
    jump_max = np.abs(np.diff(eeg_normalized, axis=2)).max(axis=2)
    trial_std = eeg_normalized.std(axis=2)

    n_trials, n_channels, _ = eeg_normalized.shape
    kurt_vals = np.zeros((n_trials, n_channels), dtype=np.float64)
    for c in range(n_channels):
        raw_kurt = kurt_fn(eeg_normalized[:, c, :].astype(np.float64), axis=1)
        # Undefined kurtosis on near-zero-variance segments -> never flagged,
        # rather than silently propagating NaN downstream.
        kurt_vals[:, c] = np.nan_to_num(raw_kurt, nan=-np.inf)

    return {"amplitude": amp_max, "jump": jump_max, "std": trial_std, "kurtosis": kurt_vals}


def derive_thresholds(train_arrays: dict, artifact_percentile: float, flatline_percentile: float) -> dict:
    """
    Thresholds derived from the TRAIN data's own percentile distribution -
    NOT fixed Gaussian-style values, which badly over-flag on heavy-tailed
    EEG once you check many channels per trial (multiple-comparisons effect).
    """
    kurt_finite = np.where(np.isinf(train_arrays["kurtosis"]), np.nan, train_arrays["kurtosis"])
    return {
        "amplitude": float(np.percentile(train_arrays["amplitude"], artifact_percentile)),
        "jump": float(np.percentile(train_arrays["jump"], artifact_percentile)),
        "kurtosis": float(np.nanpercentile(kurt_finite, artifact_percentile)),
        "flatline": float(np.percentile(train_arrays["std"], flatline_percentile)),
    }


def flag_artifacts(arrays: dict, thresholds: dict, trial_concern_min_channels: int) -> dict:
    amp_flag = arrays["amplitude"] > thresholds["amplitude"]
    jump_flag = arrays["jump"] > thresholds["jump"]
    kurt_flag = arrays["kurtosis"] > thresholds["kurtosis"]
    flatline_flag = arrays["std"] <= thresholds["flatline"]  # <=, not < (see Notebook 04 fix note)

    any_flag = amp_flag | jump_flag | kurt_flag | flatline_flag
    n_flagged = any_flag.sum(axis=1)
    trial_concern = n_flagged >= trial_concern_min_channels

    return {
        "amplitude": amp_flag, "jump": jump_flag,
        "kurtosis": kurt_flag, "flatline": flatline_flag,
        "any": any_flag, "n_flagged_channels_per_trial": n_flagged,
        "trial_concern": trial_concern,
    }


def save_artifact_params(path: Path, bad_channels: list[int], thresholds: dict) -> None:
    with open(path, "w") as f:
        json.dump({"bad_channels": bad_channels, "thresholds": thresholds}, f, indent=2)


def load_artifact_params(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
