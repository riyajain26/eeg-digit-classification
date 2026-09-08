"""
Filtering strategies for EEG preprocessing.

Reorganized from filters.py - one real change from the original: the
filter applied is now resolved by NAME via FILTER_REGISTRY instead of
being the only function in the module. The filtering math itself is
unchanged.

To add a new filter variant (e.g. wavelet denoising): write a function
matching the contract below, decorate it with
`@FILTER_REGISTRY.register("your_name")`, and set
`cfg.filter.variant = "your_name"`. Nothing else needs to change -
src/steps/preprocessing.py only ever calls
`FILTER_REGISTRY.get(cfg.filter.variant)`.

Contract every registered filter variant must satisfy:
    fn(eeg, fs, **variant_params) -> np.ndarray
    eeg: (n_trials, n_channels, n_samples) in, same shape out.
    variant_params are that variant's own config fields, passed as
    keyword arguments by whoever calls the registry (src/steps/preprocessing.py).
"""

import numpy as np
from scipy import signal

from src.utils.registry import Registry

FILTER_REGISTRY = Registry(step_name="filter")


def design_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4):
    """Second-order-sections bandpass design - sos form used (rather than
    b/a polynomial form) for numerical stability at this filter order."""
    nyquist = fs / 2
    return signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")


def design_notch_sos(freq_hz: float, fs: float, quality: float = 30):
    """Notch filter design (e.g. for mains hum), converted to sos form so
    it can be applied with the same sosfiltfilt call as the bandpass filter."""
    b, a = signal.iirnotch(freq_hz, quality, fs)
    return signal.tf2sos(b, a)


@FILTER_REGISTRY.register("butterworth_bandpass")
def apply_butterworth_bandpass(
    eeg: np.ndarray,
    fs: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    filter_order: int = 4,
    apply_notch: bool = False,
    notch_freq_hz: float = 60.0,
) -> np.ndarray:
    """
    Zero-phase (filtfilt) Butterworth bandpass, with an optional notch
    filter layered on top. Zero-phase avoids introducing a time shift
    that would misalign the EEG relative to trial-onset timing.

    Args:
        eeg: (n_trials, n_channels, n_samples).
        fs: sample rate in Hz (cfg.data.sample_rate_hz).
        bandpass_low_hz, bandpass_high_hz: passband edges in Hz.
        filter_order: Butterworth filter order.
        apply_notch: whether to additionally apply a notch filter - off by default.
        notch_freq_hz: notch center frequency, only used if apply_notch=True.

    Returns:
        Filtered array, same shape as `eeg`.
    """
    sos_band = design_bandpass_sos(bandpass_low_hz, bandpass_high_hz, fs, filter_order)
    sos_notch = design_notch_sos(notch_freq_hz, fs) if apply_notch else None

    filtered = np.empty_like(eeg)
    for i in range(eeg.shape[0]):
        for c in range(eeg.shape[1]):
            sig = signal.sosfiltfilt(sos_band, eeg[i, c])
            if apply_notch:
                sig = signal.sosfiltfilt(sos_notch, sig)
            filtered[i, c] = sig
    return filtered


def check_filter_stability(low_hz: float, high_hz: float, fs: float, order: int = 4) -> bool:
    """
    Returns True if butterworth_bandpass's filter coefficients are finite
    (numerically stable) for this cutoff/order/sample-rate combination.
    A design-time diagnostic, not itself a registered variant - call this
    when choosing cutoffs/order, not from the main pipeline.
    """
    sos = design_bandpass_sos(low_hz, high_hz, fs, order)
    return bool(np.all(np.isfinite(sos)))
