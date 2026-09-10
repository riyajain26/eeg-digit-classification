"""
Feature extraction strategies for EEG.

Reorganized from features/extraction.py - the extraction pipeline is now
resolved by name via FEATURE_REGISTRY. Math/logic is unchanged.

IMPORTANT, carried over unchanged from the original: this operates on
RAW (bad-channel-excluded, but NOT filtered/normalized) EEG, not on
Step 2's filtered output. Band power features (including high_gamma,
30-80Hz) are only meaningful computed on the unfiltered wide-band signal -
Step 2's default bandpass (1-40Hz) would zero out everything above 40Hz
before these features ever saw it. This is why src/steps/features.py
reads from cfg.data.splits_dir (Step 1's raw output), not
cfg.data.filtered_dir (Step 2's output) - see that module's docstring.

To add a new feature-extraction variant: write a function matching the
contract below, decorate it with `@FEATURE_REGISTRY.register("your_name")`,
and set cfg.feature.variant = "your_name".

Contract every registered variant must satisfy:
    fn(eeg_batch_raw, fs, **variant_params) -> np.ndarray
    eeg_batch_raw: (n_trials, n_channels, n_samples), RAW signal.
    Returns: (n_trials, n_features) float32 array.
    variant_params are that variant's own FeatureConfig fields.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis as kurt_fn

from src.utils.registry import Registry

FEATURE_REGISTRY = Registry(step_name="feature")

# np.trapz was removed in NumPy 2.x in favor of np.trapezoid - support both
# so this module works regardless of which NumPy version is installed.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def compute_band_power(sig: np.ndarray, fs: float, bands: dict, nperseg: int = 128) -> dict:
    """Integrates power spectral density within each named frequency band."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), nperseg))
    powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = float(_trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def compute_statistical_features(sig: np.ndarray) -> dict:
    """Time-domain shape statistics for one channel's signal."""
    var = float(np.var(sig))
    if var < 1e-12:   # near-zero variance: skew/kurtosis are undefined, not just small
        skewness, kurt = 0.0, 0.0
    else:
        skewness, kurt = float(skew(sig)), float(kurt_fn(sig))
    return {
        "mean": float(np.mean(sig)), "variance": var,
        "skewness": skewness, "kurtosis": kurt,
        "zero_crossing_rate": float(np.mean(np.diff(np.sign(sig)) != 0)),
    }


def compute_frequency_features(sig: np.ndarray, fs: float, nperseg: int = 128) -> dict:
    """Coarse frequency-domain summary beyond per-band power."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), nperseg))
    return {
        "dominant_freq": float(freqs[np.argmax(psd)]),
        "total_power": float(_trapezoid(psd, freqs)),
    }


def extract_features_for_trial(eeg_trial_raw: np.ndarray, fs: float, bands: dict, welch_nperseg: int = 128) -> np.ndarray:
    """
    Args:
        eeg_trial_raw: (n_channels, n_samples) - RAW unfiltered signal
            (see module docstring for why raw, not filtered).
        fs: sample rate in Hz.
        bands: named frequency bands, e.g. cfg.feature.eeg_bands.
        welch_nperseg: Welch method window length.

    Returns:
        1D feature vector - band powers, then statistical features, then
        frequency features, concatenated per channel in that order. See
        feature_names() for the matching ordered name list.
    """
    features = []
    for c in range(eeg_trial_raw.shape[0]):
        sig = eeg_trial_raw[c]
        features.extend(compute_band_power(sig, fs, bands, welch_nperseg).values())
        features.extend(compute_statistical_features(sig).values())
        features.extend(compute_frequency_features(sig, fs, welch_nperseg).values())
    return np.array(features, dtype=np.float32)


@FEATURE_REGISTRY.register("band_power_stat_freq")
def extract_features_batch(eeg_batch_raw: np.ndarray, fs: float, bands: dict, welch_nperseg: int = 128) -> np.ndarray:
    """Path A feature set: per-channel band power + statistical +
    frequency-domain features, concatenated across channels. See
    extract_features_for_trial for the per-trial contract this applies
    to every trial in the batch."""
    return np.array([
        extract_features_for_trial(trial, fs, bands, welch_nperseg) for trial in eeg_batch_raw
    ])


def feature_names(bands: dict) -> list[str]:
    """Ordered feature names per channel for the band_power_stat_freq
    variant - must match the order features are appended in
    extract_features_for_trial above. A different registered variant
    would need its own matching name function if callers need names for it."""
    stat_names = ["mean", "variance", "skewness", "kurtosis", "zero_crossing_rate"]
    freq_names = ["dominant_freq", "total_power"]
    return list(bands.keys()) + stat_names + freq_names
