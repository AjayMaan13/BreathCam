"""Training and patient-level cross-validation for BreathCNN."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from .dataset import subject_folds
from .models import BreathCNN


def predict(model, X):
    """Predict rates for a batch of windows.

    Inputs:
        model: a fitted BreathCNN from fit().
        X: (n, time) float array of standardized windows.
    Returns:
        (n,) numpy array of predicted rates in breaths/min.
    """
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(torch.from_numpy(X[i:i + 512])) for i in range(0, len(X), 512)])
    return out.numpy() * model.y_sd + model.y_mu


def fit(Xtr, ytr, Xva, yva, epochs=40, lr=1e-3, bs=64, patience=8, seed=0, verbose=False):
    """Train a BreathCNN with early stopping on a validation set.

    The target is standardized with the training mean and std. The loss is L1
    (mean absolute error), matching the metric we report. The weights with the
    best validation MAE are restored at the end.

    Inputs:
        Xtr, ytr: training windows (n, time) and rates in breaths/min.
        Xva, yva: validation windows and rates (used only to pick the epoch).
        epochs: maximum epochs; patience: stop after this many epochs with no
            validation improvement; lr, bs: Adam learning rate and batch size.
        seed: random seed (weights and batch order).
        verbose: print one line per epoch.
    Returns:
        (model, history): the fitted model (carries y_mu / y_sd) and a list of
        dicts with epoch, train_loss and val_mae.
    """
    torch.manual_seed(seed)
    mu, sd = float(ytr.mean()), float(ytr.std())
    model = BreathCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(((ytr - mu) / sd).astype(np.float32))),
                        batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.y_mu, model.y_sd = mu, sd
    best, best_state, wait, history = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.l1_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val = float(np.abs(predict(model, Xva) - yva).mean())
        history.append(dict(epoch=ep + 1, train_loss=total / len(Xtr) * sd, val_mae=val))
        if verbose:
            print(f"  epoch {ep + 1:2d}  train L1 {total / len(Xtr) * sd:.2f}  val MAE {val:.2f}")
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def cross_val_predict(X, y, groups, n_splits=5, seed=0, ckpt_dir=None, split="patient", **fit_kw):
    """Out-of-fold predictions.

    split="patient" (correct): each window is predicted by a model that never saw
    that window's patient. Inside each training set, 10% of the patients are held
    out to choose the stopping epoch (so the test fold is never used for tuning).

    split="random" (the leakage mistake, for the Phase 6 experiment): windows are
    split at random, so windows of the same patient (which overlap in time) land
    in both train and test; the validation set is 10% of random training windows.

    Inputs:
        X, y: windows and reference rates; groups: patient id per window.
        n_splits: folds; seed: base random seed.
        ckpt_dir: if given, save each fold's weights there as cnn1d_fold{k}.pt.
        split: "patient" or "random" (see above).
        **fit_kw: passed to fit (epochs, patience, ...).
    Returns:
        (pred, mean_pred, log): pred is the CNN's out-of-fold prediction per
        window; mean_pred is the trivial baseline (each fold's training-set mean
        rate); log is a list of per-fold dicts (fold, epochs, val_mae, test_mae).
    """
    pred, mean_pred, log = np.full(len(y), np.nan), np.full(len(y), np.nan), []
    folds = (subject_folds(groups, n_splits) if split == "patient"
             else list(KFold(n_splits, shuffle=True, random_state=seed).split(X)))
    for k, (tr, te) in enumerate(folds):
        rng = np.random.default_rng(seed + k)
        if split == "patient":
            patients = np.unique(groups[tr])
            val_patients = rng.choice(patients, max(1, len(patients) // 10), replace=False)
            is_val = np.isin(groups[tr], val_patients)
        else:
            is_val = rng.random(len(tr)) < 0.1
        tr2, va = tr[~is_val], tr[is_val]
        model, hist = fit(X[tr2], y[tr2], X[va], y[va], seed=seed + k, **fit_kw)
        pred[te] = predict(model, X[te])
        mean_pred[te] = y[tr].mean()
        if ckpt_dir:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(ckpt_dir) / f"cnn1d_fold{k}.pt")
        log.append(dict(fold=k + 1, epochs=len(hist), val_mae=min(h["val_mae"] for h in hist),
                        test_mae=float(np.abs(pred[te] - y[te]).mean())))
        print(f"fold {k + 1}/{n_splits}: {len(hist)} epochs, val MAE {log[-1]['val_mae']:.2f}, "
              f"test MAE {log[-1]['test_mae']:.2f}")
    return pred, mean_pred, log
