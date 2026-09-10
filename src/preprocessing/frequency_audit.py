"""
Empirical frequency-content audit - a standalone diagnostic tool for
validating a bandpass cutoff choice. Not part of the main preprocessing
run.

Kept separate from filters.py rather than folded into it: this answers a
different question ("is my chosen cutoff still appropriate for this
data?") than filters.py answers ("apply this already-chosen cutoff").
Not called automatically by src/steps/preprocessing.py - run it explicitly
whenever a cutoff decision needs re-validating (new dataset subset,
different task, wider/narrower band under consideration). This is the
tool docs/roadmap_step3.md's planned "frequency audit re-run at 100%"
diagnostic refers to.

No registry here: there's one audit method, and "audit whether my filter
choice is still right" isn't itself a pluggable pipeline step the way
filtering/normalization/artifact-detection are - it's a tool you run
alongside the pipeline, not a stage within it.
"""

import numpy as np
from scipy import signal


def compute_class_psd(eeg_a: np.ndarray, eeg_b: np.ndarray, fs: float, nperseg: int = 128):
    """
    Compares power spectral density between two classes (e.g. digit vs.
    blank) to see where in the frequency spectrum they actually differ.

    Args:
        eeg_a, eeg_b: (n_trials, n_channels, n_samples) for the two
            classes being compared.
        fs: sample rate in Hz.
        nperseg: Welch method window length.

    Returns:
        (freqs, log_diff) - log_diff has shape (n_channels, n_freqs);
        positive means eeg_a has more power than eeg_b at that
        channel/frequency.
    """
    freqs, psd_a = signal.welch(eeg_a, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_b = signal.welch(eeg_b, fs=fs, nperseg=nperseg, axis=-1)

    mean_psd_a = psd_a.mean(axis=0)
    mean_psd_b = psd_b.mean(axis=0)
    log_diff = np.log10(mean_psd_a + 1e-12) - np.log10(mean_psd_b + 1e-12)
    return freqs, log_diff


def audit_cutoff(freqs: np.ndarray, log_diff: np.ndarray, cutoff_hz: float) -> dict:
    """
    Quantitative check: is the region ABOVE cutoff_hz (what a bandpass
    filter with this cutoff would DISCARD) actually quieter/less
    class-divergent than the region being KEPT? If not, the cutoff may be
    discarding real signal.

    This is a proxy check (average power divergence), not a formal
    significance test - treat borderline results as worth a closer look,
    not as definitive either way.

    Returns:
        dict with overall_below/overall_above (mean |log_diff| in each
        region), ratio (above/below - well below 1.0 supports the cutoff;
        near/above 1.0 suggests re-examining it), flagged_channels
        (channels where the discarded region diverges MORE than the kept
        region), and n_channels.
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
