"""Neural network models. Model A: a small 1D CNN on a breathing signal."""

import torch
from torch import nn


class BreathCNN(nn.Module):
    """1D CNN that maps a window of a breathing signal to a rate.

    Five conv blocks (conv, batch norm, ReLU, max-pool by 2) widen the receptive
    field to about 10 s of signal at 25 Hz, which covers a full slow breath.
    Global average pooling over time makes the output independent of where in
    the window the breaths fall, and a linear layer gives one number.
    """

    def __init__(self, channels=(16, 32, 64, 64, 64), kernel=9):
        """Inputs:
            channels: output channels of each conv block.
            kernel: convolution kernel size (odd).
        """
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.BatchNorm1d(c),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        """Inputs:
            x: (batch, time) float tensor, a standardized signal window.
        Returns:
            (batch,) tensor, the predicted rate in standardized units
            (train.predict converts it back to breaths/min).
        """
        h = self.features(x.unsqueeze(1))
        return self.head(h.mean(dim=-1)).squeeze(-1)
