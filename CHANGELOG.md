# Changelog

## Unreleased

### Phase 2: video pipeline v1 (code)
- `breathcam/roi.py` (MediaPipe chest ROI), `breathcam/motion.py` (Farneback and frame-differencing signals), `breathcam/run.py` (CLI + diagnostic figure)
- Synthetic-video tests in `tests/test_video.py`; real-clip tests (paced rates, breath-hold, hand-wave) still to do

### Phase 1: signal foundations on BIDMC
- `breathcam/signal.py` (detrend, bandpass, Welch and peak rate, quality score), `breathcam/eval.py` (MAE, RMSE, within-tolerance, Bland-Altman)
- Synthetic-signal tests in `tests/test_signal.py`
- `notebooks/01_bidmc_signal.ipynb`: 53 recordings, 4809 windows; Welch MAE 1.16 breaths/min, peaks MAE 1.62

### Phase 0: setup, ethics, lit review kickoff
- Repo skeleton, `.gitignore` (data and identifiable files excluded), pinned Python 3.12 environment
- Draft consent form and ethics statement skeleton in `docs/`
- Initial reference list (`docs/references.md`)
