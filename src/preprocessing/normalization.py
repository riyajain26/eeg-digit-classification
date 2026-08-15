"""
Robust (median/MAD) normalization, with fitted-parameter persistence.

Critical rule enforced by this module's design: normalization must be FIT
on train only, and the exact same fitted center/scale reused (never
re-fit) on val and test. Saving fitted params to disk is what makes this
safe to apply later, e.g. to the test set in a separate notebook/session,
without risk of accidentally re-fitting.
"""

import json
from pathlib import Path

import numpy as np


def fit_normalization_robust(eeg_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    eeg_train: (n_trials, n_channels, n_samples).
    Median/MAD instead of mean/std - resistant to outlier trials (the very
    artifacts artifact detection looks for) skewing the scale.
    """
    center = np.median(eeg_train, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(eeg_train - center), axis=(0, 2), keepdims=True)
    scale = mad * 1.4826  # scales MAD to be std-equivalent under a Gaussian assumption
    scale = np.where(scale < 1e-8, 1e-8, scale)
    return center, scale


def apply_normalization(eeg: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (eeg - center) / scale


def save_normalization_params(path: Path, center: np.ndarray, scale: np.ndarray) -> None:
    np.savez(path, center=center, scale=scale)


def load_normalization_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["center"], data["scale"]
