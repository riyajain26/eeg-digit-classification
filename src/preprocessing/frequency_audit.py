"""
Empirical frequency-content audit (Notebook 04, Section 2).

Kept as a standalone, reusable diagnostic tool rather than folded silently
into the filter pipeline - useful any time a bandpass cutoff choice needs
re-validating (new dataset subset, different task, wider/narrower band
under consideration). Not run automatically as part of the main pipeline;
call explicitly when you want to re-check a cutoff decision.
"""

import numpy as np
from scipy import signal


def compute_class_psd(eeg_a: np.ndarray, eeg_b: np.ndarray, fs: float, nperseg: int = 128):
    """
    eeg_a, eeg_b: (n_trials, n_channels, n_samples) for two classes to compare
    (e.g. digit vs. blank). Returns (freqs, log_diff) where log_diff has
    shape (n_channels, n_freqs) - positive means eeg_a has more power than
    eeg_b at that channel/frequency.
    """
    freqs, psd_a = signal.welch(eeg_a, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_b = signal.welch(eeg_b, fs=fs, nperseg=nperseg, axis=-1)

    mean_psd_a = psd_a.mean(axis=0)
    mean_psd_b = psd_b.mean(axis=0)
    log_diff = np.log10(mean_psd_a + 1e-12) - np.log10(mean_psd_b + 1e-12)
    return freqs, log_diff


def audit_cutoff(freqs: np.ndarray, log_diff: np.ndarray, cutoff_hz: float) -> dict:
    """
    Quantitative check: is the region above cutoff_hz actually quieter
    (less class-divergent) than the region at/below it? Returns overall
    ratio and any per-channel flags where the discarded region diverges
    MORE than the kept region - candidates for re-examining the cutoff.

    This is a proxy check (average power divergence), not a formal
    significance test - treat borderline results as worth a closer look,
    not as definitive either way.
    """
    below_mask = freqs <= cutoff_hz
    above_mask = freqs > cutoff_hz

    overall_below = np.abs(log_diff[:, below_mask]).mean()
    overall_above = np.abs(log_diff[:, above_mask]).mean()

    diff_below_per_channel = np.abs(log_diff[:, below_mask]).mean(axis=1)
    diff_above_per_channel = np.abs(log_diff[:, above_mask]).mean(axis=1)
    flagged_channels = list(np.where(diff_above_per_channel > diff_below_per_channel)[0])

    return {
        "overall_below": float(overall_below),
        "overall_above": float(overall_above),
        "ratio": float(overall_above / overall_below) if overall_below > 0 else float("nan"),
        "flagged_channels": flagged_channels,
        "n_channels": log_diff.shape[0],
    }
