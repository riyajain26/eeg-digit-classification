"""
Normalization strategies for EEG preprocessing.

Critical rule enforced by every variant registered here: normalization
must be FIT on train only, and the exact same fitted center/scale reused
(never re-fit) on val and test. save_normalization_params/
load_normalization_params exist specifically to make that safe - a fitted
result can be persisted and reloaded exactly, rather than risking an
accidental re-fit when normalizing a different split later.

To add a new normalization variant: write a `fit` function and an `apply`
function matching NormalizationStrategy's contract, bundle them into a
NormalizationStrategy, and register it under a new name. Set
cfg.normalization.variant to select it - nothing else needs to change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.utils.registry import Registry

NORMALIZATION_REGISTRY = Registry(step_name="normalization")


@dataclass
class NormalizationStrategy:
    """
    A normalization variant's fit/apply pair, kept bundled together
    (rather than as two separately-selectable registry entries) because
    fit and apply for a given variant must always be used as a matched
    pair - the (center, scale) one variant's fit() produces would be
    meaningless to a different variant's apply().

    fit(eeg_train) -> (center, scale), each broadcastable against an
        (n_trials, n_channels, n_samples) array.
    apply(eeg, center, scale) -> normalized eeg, same shape as input.
    """
    fit: Callable
    apply: Callable


def _fit_robust_median_mad(eeg_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Median/MAD instead of mean/std - resistant to outlier trials (the
    very artifacts artifact detection looks for) skewing the fitted scale.

    Args:
        eeg_train: (n_trials, n_channels, n_samples) - TRAIN split only.
    Returns:
        (center, scale), each (1, n_channels, 1) for broadcasting.
    """
    center = np.median(eeg_train, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(eeg_train - center), axis=(0, 2), keepdims=True)
    scale = mad * 1.4826  # scales MAD to be std-equivalent under a Gaussian assumption
    scale = np.where(scale < 1e-8, 1e-8, scale)
    return center, scale


def _apply_robust_median_mad(eeg: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (eeg - center) / scale


NORMALIZATION_REGISTRY.register("robust_median_mad")(
    NormalizationStrategy(fit=_fit_robust_median_mad, apply=_apply_robust_median_mad)
)


def save_normalization_params(path: Path, center: np.ndarray, scale: np.ndarray) -> None:
    """Persists fitted (center, scale) so they can be reloaded and reused
    exactly - never re-fit - when normalizing val/test/future data."""
    np.savez(path, center=center, scale=scale)


def load_normalization_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["center"], data["scale"]
