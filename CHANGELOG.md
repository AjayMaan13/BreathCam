# Changelog

## Unreleased

### Phase 6 (on BIDMC, 1D CNN): explainability and leakage
- `breathcam/explain.py` (1D Grad-CAM), `breathcam/leakage.py`, `split="random"` option in `train.py`, `notebooks/04_explain_and_leakage.ipynb`, figures in `docs/figures/`
- Leakage: same CNN, random-window split MAE 0.46 vs patient-split 0.98 (honest error 2.1x higher)
- Explainability: attention is not concentrated at annotated breaths (1.01x vs 1.01x control); passes the randomised-weights check; occlusion effect weak (1.28 vs 1.23 vs 1.05 bpm)
- Not done: temporal Grad-CAM on a 3D video model and chest-vs-background occlusion (need video data)

### Phase 5 (partial): Model A, 1D CNN on BIDMC
- `breathcam/dataset.py` (BIDMC windows, patient-level `GroupKFold`), `breathcam/models.py` (`BreathCNN`), `breathcam/train.py` (early stopping, out-of-fold predictions), `breathcam/bidmc_eval.py` (writes `data/bidmc.db`, same schema as the video database)
- `notebooks/03_cnn_vs_classical.ipynb`: CNN MAE 0.98 vs Welch 1.16 vs peaks 1.62 vs mean baseline 2.31 (53 patients); paired difference CNN-Welch -0.18, 95% CI -0.57 to +0.18, so not distinguishable; CNN regresses to the mean at slow and fast rates
- Not done: video models (r3d_18, CNN+LSTM) and 16-frame video clips; they need more video subjects

### Phase 4: evaluation (on existing data; metronome reference only)
- `breathcam/eval.py`: `load_windows`, `summarize`, `rejection_rate`, `bootstrap_ci` (resamples subjects; refuses with fewer than 5), `tests/test_eval.py`
- `notebooks/02_evaluation.ipynb` and figures in `docs/figures/`: method comparison (MAE 0.11-0.18 bpm vs metronome on 26 windows), Bland-Altman, failure analysis
- Limits: 1 subject, 1 setup, no independent reference; CI not estimable. Per-condition conclusions need new recordings

### Phase 3: SQL layer (on the 5 existing clips; no accelerometer yet)
- `sql/schema.sql` (plan schema + `ref_source` on references, `notes` on sessions), `sql/eval_queries.sql`, `breathcam/db.py`, `breathcam/build_db.py`
- 99 predictions (welch / peaks / framediff) over 33 windows; paced clips agree with the metronome target (MAE about 0.1 bpm). Reference is the metronome target, not an independent measurement
- Still to do: accelerometer recordings, sync script, other participants (needs signed consent)

### Phase 2: video pipeline v1 (code)
- `breathcam/roi.py` (MediaPipe chest ROI), `breathcam/motion.py` (Farneback and frame-differencing signals), `breathcam/run.py` (CLI + diagnostic figure)
- Synthetic-video tests in `tests/test_video.py`
- Real-clip checks (self-recorded, manual ROI): 12 bpm clip -> 12.0, 20 bpm clip -> 20.0, breath-hold windows rejected, hand-wave clip rejected at quality threshold 0.6 (0.53 scored; 0.5 let it through)

### Phase 1: signal foundations on BIDMC
- `breathcam/signal.py` (detrend, bandpass, Welch and peak rate, quality score), `breathcam/eval.py` (MAE, RMSE, within-tolerance, Bland-Altman)
- Synthetic-signal tests in `tests/test_signal.py`
- `notebooks/01_bidmc_signal.ipynb`: 53 recordings, 4809 windows; Welch MAE 1.16 breaths/min, peaks MAE 1.62

### Phase 0: setup, ethics, lit review kickoff
- Repo skeleton, `.gitignore` (data and identifiable files excluded), pinned Python 3.12 environment
- Draft consent form and ethics statement skeleton in `docs/`
- Initial reference list (`docs/references.md`)
