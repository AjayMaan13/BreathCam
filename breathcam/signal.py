"""Respiratory-rate estimation from a 1D breathing signal."""

import numpy as np
from scipy import signal as sps

# Default breathing band in Hz (6-42 breaths/min); widen for children.
LOW_HZ, HIGH_HZ = 0.1, 0.7


def detrend(x):
    """Remove slow linear drift so it doesn't dominate the filter.

    Inputs:
        x: 1D array-like, the raw signal.
    Returns:
        1D float array, same length as x, with the best-fit straight line subtracted.
    """
    return sps.detrend(np.asarray(x, dtype=float))


def bandpass(x, low, high, fs, order=4):
    """Keep only frequencies between low and high (where breathing lives).

    Uses a Butterworth filter in second-order sections applied forwards and
    backwards (sosfiltfilt), so the output has zero phase shift: breaths are
    not delayed relative to the input.

    Inputs:
        x: 1D array, the signal.
        low, high: band edges in Hz.
        fs: sampling rate in Hz.
        order: filter order (default 4).
    Returns:
        1D array, same length as x, the filtered signal.
    """
    sos = sps.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _spectrum(x, fs, low, high, nfft_mult=8):
    """Welch power spectrum of x, restricted to the band [low, high].

    Inputs:
        x: 1D array, one window of signal (used as a single Welch segment).
        fs: sampling rate in Hz.
        low, high: band edges in Hz.
        nfft_mult: zero-padding factor; larger gives a finer frequency grid
            (it interpolates the peak position, it does not add real resolution).
    Returns:
        (f, p): frequencies in Hz and power at each, in-band only.
    """
    nperseg = len(x)
    f, p = sps.welch(x, fs=fs, nperseg=nperseg, nfft=nperseg * nfft_mult)
    band = (f >= low) & (f <= high)
    return f[band], p[band]


def quality_score(x, fs, low=LOW_HZ, high=HIGH_HZ):
    """Score how clean and periodic a window is, from 0 (junk) to 1 (clean).

    A clean breathing signal puts most of its in-band power in one narrow
    spectral peak; noise and motion spread it out. The score is the share of
    in-band power within +/-0.05 Hz of the dominant peak. Use it to reject
    windows instead of reporting a rate you can't trust.

    Inputs:
        x: 1D array, one window of signal (ideally already detrended/filtered).
        fs: sampling rate in Hz.
        low, high: band edges in Hz (default: the module breathing band).
    Returns:
        float in [0, 1].
    """
    f, p = _spectrum(x, fs, low, high)
    if p.sum() == 0:
        return 0.0
    peak = f[np.argmax(p)]
    return float(p[np.abs(f - peak) <= 0.05].sum() / p.sum())


def welch_rate(x, fs, window_s=30, step_s=5, low=LOW_HZ, high=HIGH_HZ):
    """Estimate breathing rate over time with a sliding-window spectral peak.

    Detrends and bandpasses the whole signal, then for each window finds the
    strongest in-band frequency (Welch) and converts it: rate_bpm = peak_Hz * 60.

    Inputs:
        x: 1D array, the raw breathing signal.
        fs: sampling rate in Hz.
        window_s: window length in seconds (longer = finer frequency resolution,
            slower to react).
        step_s: seconds between window starts.
        low, high: band edges in Hz.
    Returns:
        (t_start, rate_bpm, quality): three 1D arrays with one entry per window:
        window start time in s, estimated breaths/min, and quality_score.
    """
    x = bandpass(detrend(x), low, high, fs)
    win, step = int(window_s * fs), int(step_s * fs)
    starts, rates, quals = [], [], []
    for i in range(0, len(x) - win + 1, step):
        seg = x[i:i + win]
        f, p = _spectrum(seg, fs, low, high)
        starts.append(i / fs)
        rates.append(f[np.argmax(p)] * 60)
        quals.append(quality_score(seg, fs, low, high))
    return np.array(starts), np.array(rates), np.array(quals)


def peak_rate(x, fs, low=LOW_HZ, high=HIGH_HZ):
    """Estimate breathing rate by detecting individual breaths (signal peaks).

    Detrends and bandpasses x, finds peaks at least one fastest-allowed-breath
    apart with a minimum prominence, and converts the median gap between
    peaks to breaths/min. Works on the whole input as one window.

    Inputs:
        x: 1D array, the raw breathing signal (one window or a whole clip).
        fs: sampling rate in Hz.
        low, high: band edges in Hz (high also sets the minimum peak spacing).
    Returns:
        (rate_bpm, peak_indices): breaths/min (NaN if fewer than 2 peaks found)
        and the sample indices of the detected breaths.
    """
    x = bandpass(detrend(x), low, high, fs)
    min_gap = int(fs / high)  # no two breaths closer than the fastest allowed rate
    peaks, _ = sps.find_peaks(x, distance=min_gap, prominence=0.3 * np.std(x))
    if len(peaks) < 2:
        return np.nan, peaks
    return 60 * fs / np.median(np.diff(peaks)), peaks
