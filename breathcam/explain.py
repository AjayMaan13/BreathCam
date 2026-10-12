"""Grad-CAM for the 1D breathing CNN: which part of the window drives the prediction."""

import numpy as np
import torch


def grad_cam_1d(model, X):
    """Grad-CAM heatmap over time for each window.

    Takes the activations of the last conv block, weights each channel by the
    average gradient of the predicted rate with respect to it, and sums over
    channels. Because the output is a regression (a rate, not a class score),
    both increases and decreases matter, so the absolute value is used. The map
    is interpolated up to the input length and scaled to a 0..1 maximum per window.

    Inputs:
        model: a fitted BreathCNN (eval mode is set here).
        X: (n, time) float32 array of standardized windows.
    Returns:
        (n, time) array of attention in [0, 1]; larger = more influence on the output.
    """
    model.eval()
    x = torch.from_numpy(X).unsqueeze(1)
    a = model.features[:-1](x)                    # activations before the last max-pool
    a.retain_grad()
    out = model.head(model.features[-1](a).mean(dim=-1)).squeeze(-1)
    out.sum().backward()                          # eval-mode BN: samples are independent
    cam = (a.grad.mean(dim=-1, keepdim=True) * a).sum(dim=1).abs().detach().numpy()
    t_in = np.linspace(0, 1, X.shape[1])
    t_cam = np.linspace(0, 1, cam.shape[1])
    cam = np.stack([np.interp(t_in, t_cam, c) for c in cam])
    return cam / (cam.max(axis=1, keepdims=True) + 1e-12)
