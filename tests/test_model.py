from pathlib import Path

import numpy as np
import pytest
import torch

from breathcam.dataset import load_bidmc, subject_folds
from breathcam.models import BreathCNN
from breathcam.train import fit, predict

BIDMC = Path(__file__).resolve().parent.parent / "data" / "bidmc"


def synth(n, seed):
    """30 s windows at 25 Hz of a sine at a random 8-30 bpm plus noise, z-scored."""
    rng = np.random.default_rng(seed)
    bpm = rng.uniform(8, 30, n)
    t = np.arange(750) / 25
    X = np.sin(2 * np.pi * bpm[:, None] / 60 * t + rng.uniform(0, 6.28, (n, 1))) + 0.3 * rng.standard_normal((n, 750))
    return ((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)).astype(np.float32), bpm


def test_forward_shape():
    assert BreathCNN()(torch.zeros(4, 750)).shape == (4,)


def test_cnn_learns_rate_better_than_mean_baseline():
    Xtr, ytr = synth(600, 0)
    Xva, yva = synth(100, 1)
    Xte, yte = synth(200, 2)
    model, hist = fit(Xtr, ytr, Xva, yva, epochs=25, seed=0)
    mae = np.abs(predict(model, Xte) - yte).mean()
    baseline = np.abs(ytr.mean() - yte).mean()
    assert mae < 0.5 * baseline


def test_subject_folds_never_share_a_patient():
    groups = np.repeat(np.arange(20), 10)
    for tr, te in subject_folds(groups, 5):
        assert not set(groups[tr]) & set(groups[te])


@pytest.mark.skipif(not BIDMC.exists(), reason="BIDMC data not downloaded")
def test_bidmc_windows():
    d = load_bidmc(BIDMC)
    assert d["X"].shape[1] == 750 and len(d["X"]) == len(d["y"]) == len(d["rec"]) == len(d["t0"])
    assert abs(d["X"].mean(axis=1)).max() < 1e-3 and set(d["rec"]) == set(range(1, 54))
