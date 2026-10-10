import numpy as np
import pytest

from breathcam.signal import peak_rate, quality_score, welch_rate

FS = 30
RNG = np.random.default_rng(0)


def synth(bpm, dur=60, noise=0.0, drift=0.0):
    t = np.arange(0, dur, 1 / FS)
    x = np.sin(2 * np.pi * bpm / 60 * t)
    x = x + noise * RNG.standard_normal(len(t)) + drift * t / dur
    return t, x


@pytest.mark.parametrize("bpm", [10, 15, 24, 36])
def test_clean_rate_recovered(bpm):
    _, x = synth(bpm)
    assert np.median(welch_rate(x, FS)[1]) == pytest.approx(bpm, abs=1.5)
    assert peak_rate(x, FS)[0] == pytest.approx(bpm, abs=1.5)


def test_noise_and_drift():
    _, x = synth(15, noise=0.5, drift=5.0)
    assert np.median(welch_rate(x, FS)[1]) == pytest.approx(15, abs=2)
    assert peak_rate(x, FS)[0] == pytest.approx(15, abs=2)


def test_motion_spikes_lower_quality():
    _, clean = synth(15, dur=30)
    spiky = clean.copy()
    spiky[::37] += 15 * RNG.standard_normal(len(spiky[::37]))
    assert quality_score(spiky, FS) < quality_score(clean, FS)


def test_noise_only_has_low_quality():
    noise = RNG.standard_normal(30 * FS)
    _, clean = synth(15, dur=30)
    assert quality_score(noise, FS) < 0.5 < quality_score(clean, FS)


def test_short_window_is_coarser():
    # 10 s window: frequency resolution is 6 bpm, so error can reach ~3 bpm
    _, x = synth(17, dur=60)
    err10 = np.abs(welch_rate(x, FS, window_s=10, step_s=5)[1] - 17).max()
    err30 = np.abs(welch_rate(x, FS, window_s=30, step_s=5)[1] - 17).max()
    assert err30 <= err10
