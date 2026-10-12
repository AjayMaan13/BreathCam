"""BIDMC windows for learning, and patient-level cross-validation splits."""

import numpy as np
import wfdb
from scipy import signal as sps
from sklearn.model_selection import GroupKFold


def load_bidmc(data_dir="data/bidmc", window_s=30, step_s=5, fs_out=25, min_breaths=3):
    """Cut every BIDMC recording into labelled windows for a neural network.

    Uses the same windows (30 s, 5 s step) and the same reference rule as the
    Phase 1 notebook, so classical and learned methods are scored on identical
    windows. The respiration signal is low-pass filtered and downsampled from
    125 Hz to fs_out, then each window is linearly detrended and z-scored.

    Inputs:
        data_dir: folder with the downloaded BIDMC files.
        window_s, step_s: window length and hop in seconds.
        fs_out: sampling rate (Hz) of the network input; must divide 125.
        min_breaths: a window needs at least this many annotated breaths from an
            annotator for that annotator to contribute to the reference.
    Returns:
        dict of arrays, one entry per window:
          X: (n, window_s*fs_out) float32 network input,
          y: reference rate in breaths/min (mean of the annotators' 60/median
             inter-breath interval),
          rec: recording number 1..53 (one recording = one patient),
          t0: window start in seconds.
    """
    X, y, rec, t0 = [], [], [], []
    for i in range(1, 54):
        name = f"{data_dir}/bidmc{i:02d}"
        r = wfdb.rdrecord(name, channels=[0])  # RESP (impedance)
        fs = r.fs
        down = int(fs // fs_out)
        xd = sps.decimate(r.p_signal[:, 0], down, zero_phase=True)
        ann = wfdb.rdann(name, "breath")
        t_ann = [ann.sample[np.array(ann.aux_note) == a] / fs for a in ("ann1", "ann2")]
        win, step = int(window_s * fs), int(step_s * fs)
        for s in range(0, len(r.p_signal) - win + 1, step):
            t = s / fs
            refs = []
            for ta in t_ann:
                b = ta[(ta >= t) & (ta < t + window_s)]
                if len(b) >= min_breaths:
                    refs.append(60 / np.median(np.diff(b)))
            if not refs:
                continue
            X.append(xd[s // down:(s + win) // down])
            y.append(np.mean(refs))
            rec.append(i)
            t0.append(t)
    X = sps.detrend(np.array(X), axis=1)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return dict(X=X.astype(np.float32), y=np.array(y), rec=np.array(rec), t0=np.array(t0))


def subject_folds(groups, n_splits=5):
    """Cross-validation splits where no patient appears in both train and test.

    Inputs:
        groups: array with one patient/recording id per window.
        n_splits: number of folds.
    Returns:
        list of (train_indices, test_indices) pairs over the windows.
    """
    groups = np.asarray(groups)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
