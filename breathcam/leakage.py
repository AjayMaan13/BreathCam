"""Leakage experiment: the same CNN, trained with random-window splits.

    python -m breathcam.leakage

Runs the Phase 5 cross-validation again with split="random" and the same seed,
epochs and model, so the only difference from the patient-level run is the split.
Saves data/leakage_random.npz (out-of-fold predictions); notebook 04 compares it
with the patient-level predictions stored in data/bidmc.db.
"""

import numpy as np

from .dataset import load_bidmc
from .train import cross_val_predict

if __name__ == "__main__":
    d = load_bidmc()
    pred, mean, log = cross_val_predict(d["X"], d["y"], d["rec"], split="random", epochs=40, patience=8)
    np.savez("data/leakage_random.npz", pred=pred, mean=mean, y=d["y"], rec=d["rec"], t0=d["t0"])
    print("wrote data/leakage_random.npz")
