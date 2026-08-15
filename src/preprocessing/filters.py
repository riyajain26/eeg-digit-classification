"""
Bandpass/notch filtering (Notebook 04, Section 3).

Uses second-order-sections (sos) form for numerical stability at this
filter order, and zero-phase filtfilt to avoid introducing time-shift
distortion.
"""

import numpy as np
from scipy import signal


def design_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyquist = fs / 2
    return signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")


def design_notch_sos(freq_hz: float, fs: float, quality: float = 30):
    b, a = signal.iirnotch(freq_hz, quality, fs)
    return signal.tf2sos(b, a)


def apply_filters(
    eeg: np.ndarray,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    order: int = 4,
    apply_notch: bool = False,
    notch_freq: float = 60.0,
) -> np.ndarray:
    """
    eeg: (n_trials, n_channels, n_samples). Returns filtered array, same shape.
    """
    sos_band = design_bandpass_sos(bandpass_low, bandpass_high, fs, order)
    sos_notch = design_notch_sos(notch_freq, fs) if apply_notch else None

    filtered = np.empty_like(eeg)
    for i in range(eeg.shape[0]):
        for c in range(eeg.shape[1]):
            sig = signal.sosfiltfilt(sos_band, eeg[i, c])
            if apply_notch:
                sig = signal.sosfiltfilt(sos_notch, sig)
            filtered[i, c] = sig
    return filtered


def check_filter_stability(low_hz: float, high_hz: float, fs: float, order: int = 4) -> bool:
    """Returns True if filter coefficients are finite (stable) for this config."""
    sos = design_bandpass_sos(low_hz, high_hz, fs, order)
    return bool(np.all(np.isfinite(sos)))
