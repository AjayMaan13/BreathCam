"""Build data/breathcam.db from the recorded clips.

    python -m breathcam.build_db

Registers the subject and clips, stores the metronome target as a provisional
reference for the paced clips, runs the pipeline three ways per clip (welch,
peaks, framediff) and stores one prediction per window. The database is derived
data, so it is rebuilt from scratch each run.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from . import db
from .run import estimate
from .signal import peak_rate

# Manual chest box for the current clips (320-px-wide frame coordinates);
# all clips share one fixed camera setup. Replace with MediaPipe when conditions vary.
ROI = (115, 78, 205, 135)

# (file, metronome bpm or None, notes). Only conditions known to be true are filled in.
CLIPS = [
    ("01_12bpm.mov", 12, "paced at 12 breaths/min"),
    ("02_20bpm.mov", 20, "paced at 20 breaths/min"),
    ("03_hold.mov", None, "breath-hold in the middle (~17-32 s)"),
    ("04_wave.mov", None, "hand-wave artifact"),
    ("05_normal.mov", None, "natural unpaced breathing"),
]
COMMON = dict(clothing="fitted")  # dark fitted polo; lighting, distance and mount were not recorded


def window_peak_rates(sig, fs, starts, window_s, low, high):
    """Peak-detection rate for each window, to compare with welch_rate's windows.

    Inputs:
        sig: 1D signal; fs: sampling rate (Hz).
        starts: window start times (s); window_s: window length (s).
        low, high: breathing band (Hz).
    Returns:
        1D array of breaths/min (NaN where fewer than 2 breaths were found).
    """
    return np.array([peak_rate(sig[int(t * fs):int((t + window_s) * fs)], fs, low, high)[0]
                     for t in starts])


def build(db_path, raw_dir, min_quality=0.6):
    """Create the database and fill it from the clips in raw_dir.

    Inputs:
        db_path: output database file (replaced if it exists).
        raw_dir: folder holding the clips listed in CLIPS.
        min_quality: quality gate; windows below it are stored with NULL est_bpm.
    Returns:
        None.
    """
    db_path, raw_dir = Path(db_path), Path(raw_dir)
    db_path.unlink(missing_ok=True)
    conn = db.connect(db_path)
    db.init(conn)
    db.add_subject(conn, "S01", date.today().isoformat(), "researcher (self)")

    for name, bpm, notes in CLIPS:
        video = raw_dir / name
        sid = db.add_session(conn, "S01", video, paced_bpm=bpm, notes=notes, **COMMON)
        flow = estimate(video, "flow", ROI, min_quality=min_quality)
        diff = estimate(video, "diff", ROI, min_quality=min_quality)
        if bpm is not None:
            duration = len(flow["signal"]) / flow["fs"]
            db.add_reference(conn, sid, 0.0, duration + 1, bpm, "metronome")

        gate = lambda est, ok: np.where(ok, est, np.nan)  # rejected windows -> NULL
        peaks = window_peak_rates(flow["signal"], flow["fs"], flow["t"], flow["window_s"], flow["low"], flow["high"])
        db.add_predictions(conn, sid, "welch", flow["t"], gate(flow["rate"], flow["accepted"]), flow["quality"])
        db.add_predictions(conn, sid, "peaks", flow["t"], gate(peaks, flow["accepted"]), flow["quality"])
        db.add_predictions(conn, sid, "framediff", diff["t"], gate(diff["rate"], diff["accepted"]), diff["quality"])
        print(f"{name}: {len(flow['t'])} windows")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"wrote {db_path} ({n} predictions)")
    conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/breathcam.db")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--min-quality", type=float, default=0.6)
    a = p.parse_args()
    build(a.db, a.raw, a.min_quality)
