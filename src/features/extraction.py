"""
Path A feature extraction: band power, statistical, and frequency-domain
features per channel, per trial (Notebook 04, Section 8).

NaN handling: skewness/kurtosis are undefined on near-zero-variance
segments (a real mathematical edge case, not a bug) - mapped to 0.0 rather
than left as NaN, since a perfectly flat signal has no meaningful shape.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis as kurt_fn

# np.trapz was removed in NumPy 2.x in favor of np.trapezoid - support both
# so this module works regardless of which NumPy version is installed.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def compute_band_power(sig: np.ndarray, fs: float, bands: dict, nperseg: int = 128) -> dict:
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), nperseg))
    powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = float(_trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def compute_statistical_features(sig: np.ndarray) -> dict:
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
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), nperseg))
    return {
        "dominant_freq": float(freqs[np.argmax(psd)]),
        "total_power": float(_trapezoid(psd, freqs)),
    }


FEATURE_NAMES_PER_CHANNEL = None  # set dynamically below, depends on `bands` config


def extract_features_for_trial(eeg_trial_raw: np.ndarray, fs: float, bands: dict, welch_nperseg: int = 128) -> np.ndarray:
    """eeg_trial_raw: (n_channels, n_samples), RAW unfiltered signal (band
    power legitimately includes high_gamma etc. only if computed on
    unfiltered/wide-band signal, not the Path B bandpass-limited version)."""
    features = []
    for c in range(eeg_trial_raw.shape[0]):
        sig = eeg_trial_raw[c]
        features.extend(compute_band_power(sig, fs, bands, welch_nperseg).values())
        features.extend(compute_statistical_features(sig).values())
        features.extend(compute_frequency_features(sig, fs, welch_nperseg).values())
    return np.array(features, dtype=np.float32)


def extract_features_batch(eeg_batch_raw: np.ndarray, fs: float, bands: dict, welch_nperseg: int = 128) -> np.ndarray:
    return np.array([
        extract_features_for_trial(trial, fs, bands, welch_nperseg) for trial in eeg_batch_raw
    ])


def feature_names(bands: dict) -> list[str]:
    """Ordered feature names per channel - must match the order features are
    appended in extract_features_for_trial above."""
    stat_names = ["mean", "variance", "skewness", "kurtosis", "zero_crossing_rate"]
    freq_names = ["dominant_freq", "total_power"]
    return list(bands.keys()) + stat_names + freq_names
