"""Compare classical methods and the 1D CNN on BIDMC, with patient-level splits.

    python -m breathcam.bidmc_eval

Writes data/bidmc.db using the same schema as the video database, so the same
queries and eval.py functions work on it. Each BIDMC recording is a "subject"
(B01..B53) and one session; the reference is the manual breath annotation
(ref_source = 'annotation'). Methods stored per 30 s window:
  welch, peaks   classical, from the Phase 1 code
  cnn1d          out-of-fold predictions of the 1D CNN (the patient was never in training)
  meanrate       trivial baseline: mean rate of the training patients
quality_score is the Welch quality of the window for every method, and no
quality gate is applied (filter on quality_score in SQL to gate).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb

from . import db
from .dataset import load_bidmc
from .signal import peak_rate, welch_rate
from .train import cross_val_predict

WINDOW_S, STEP_S = 30, 5


def classical_estimates(d, data_dir):
    """Welch rate, peak-detection rate and Welch quality for each window in d.

    Inputs:
        d: dict from load_bidmc.
        data_dir: BIDMC folder.
    Returns:
        (welch, peaks, quality): three arrays aligned with d's windows.
    """
    welch, peaks, qual = (np.full(len(d["y"]), np.nan) for _ in range(3))
    for rec in np.unique(d["rec"]):
        r = wfdb.rdrecord(f"{data_dir}/bidmc{rec:02d}", channels=[0])
        x, fs = r.p_signal[:, 0], r.fs
        _, w_rate, w_q = welch_rate(x, fs, WINDOW_S, STEP_S)
        for j in np.flatnonzero(d["rec"] == rec):
            k = int(round(d["t0"][j] / STEP_S))
            welch[j], qual[j] = w_rate[k], w_q[k]
            s = int(d["t0"][j] * fs)
            peaks[j] = peak_rate(x[s:s + int(WINDOW_S * fs)], fs)[0]
    return welch, peaks, qual


def build(db_path="data/bidmc.db", data_dir="data/bidmc", seed=0, epochs=40, patience=8):
    """Run all methods on BIDMC and write the database.

    Inputs:
        db_path: output database (replaced if it exists).
        data_dir: BIDMC folder.
        seed: random seed for the CNN.
        epochs, patience: CNN training limits.
    Returns:
        list of per-fold training logs (also saved next to the database as bidmc_cv_log.json).
    """
    d = load_bidmc(data_dir, WINDOW_S, STEP_S)
    print(f"{len(d['y'])} windows from {len(np.unique(d['rec']))} patients")
    welch, peaks, qual = classical_estimates(d, data_dir)
    cnn, mean, log = cross_val_predict(d["X"], d["y"], d["rec"], seed=seed, epochs=epochs, patience=patience,
                                       ckpt_dir=Path(db_path).parent / "models")

    db_path = Path(db_path)
    db_path.unlink(missing_ok=True)
    conn = db.connect(db_path)
    db.init(conn)
    estimates = {"welch": welch, "peaks": peaks, "cnn1d": cnn, "meanrate": mean}
    for rec in np.unique(d["rec"]):
        sid_label = f"B{rec:02d}"
        db.add_subject(conn, sid_label, "n/a (public PhysioNet dataset)", "BIDMC ICU patient")
        sid = db.add_session(conn, sid_label, f"{data_dir}/bidmc{rec:02d}", notes="BIDMC RESP signal (no video)")
        idx = np.flatnonzero(d["rec"] == rec)
        for j in idx:
            db.add_reference(conn, sid, float(d["t0"][j]), float(d["t0"][j]) + STEP_S, float(d["y"][j]), "annotation")
        for method, est in estimates.items():
            db.add_predictions(conn, sid, method, d["t0"][idx], est[idx], qual[idx])
    conn.commit()
    conn.close()
    Path(str(db_path.with_suffix("")) + "_cv_log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {db_path}")
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/bidmc.db")
    p.add_argument("--data", default="data/bidmc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args()
    build(a.db, a.data, a.seed, a.epochs)
