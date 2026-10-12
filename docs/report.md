# BreathCam: camera-based respiratory-rate estimation, evaluated like a measurement study

*Research prototype. Not a medical device. Healthy adult volunteer (n = 1) and adult ICU patients (BIDMC, n = 53); no pediatric data.*

## 1. Background
Respiratory rate is a basic vital sign that is often counted by eye. Contactless methods (camera, radar, thermal) could make it easier to monitor, particularly in children, for whom normal rates are higher and change with age (about 44 breaths/min at birth falling to about 26 at age 2 [Fleming 2011]). This project builds a small camera-based estimator and, more importantly, evaluates it the way a clinical measurement study would: against a reference, with agreement statistics, with subject-level validation, and with explicit limitations. Related work is summarised in `related_work.md` and references in `references.md`.

## 2. Methods
**Signal estimator (`signal.py`).** A one-dimensional breathing signal is detrended, band-passed (zero-phase Butterworth; 0.1-0.7 Hz for BIDMC, 0.1-1.0 Hz for video), and cut into 30 s windows every 5 s. Rate is the strongest in-band spectral peak (Welch) times 60; peak detection is a second estimator. A quality score (share of in-band power in the main peak, 0-1) lets the system refuse to answer: windows below 0.6 are rejected.

**Video pipeline (`roi.py`, `motion.py`, `run.py`).** Frames are downscaled to 320 px, a chest region is placed (MediaPipe Pose, or a manual box), and the mean vertical Farneback optical flow in that region is accumulated into a chest-position signal. A cheaper frame-differencing estimator is the baseline.

**Learned model (`models.py`, `train.py`).** A 1D CNN (five conv blocks, global pooling, about 100k parameters) regresses the rate from a standardised 30 s window of the BIDMC respiration signal, trained with patient-level 5-fold cross-validation (early stopping on held-out training patients). A mean-rate predictor is the trivial baseline.

**Data and reference.** (a) Five clips (about 24-93 s each) of the author recorded with a phone at 30 fps: paced at 12 and 20 breaths/min with a metronome, a breath-hold, a hand-wave, and natural breathing. Reference for the paced clips is the metronome rate, which is a *target*, not a measurement. (b) BIDMC: 53 ICU patients, 8 min each, 125 Hz, with breath annotations from two annotators; reference per window is the mean of the annotators' interval-based rates (4,809 windows).

**Evaluation.** MAE, RMSE, share within +/-2 breaths/min, Bland-Altman bias and limits of agreement, and bootstrap confidence intervals that resample patients, not windows. Results are stored in SQLite and queried with SQL.

**De-identification (`deid.py`).** Faces are blurred on every frame, audio is dropped and source metadata (including GPS) is stripped.

## 3. Results
**Video clips (metronome reference, one person, one setup).** 26 scored windows from the paced clips: all three methods were within +/-2 breaths/min 100% of the time; MAE 0.11 (frame differencing), 0.12 (Welch on optical flow), 0.18 (peak detection); bias within 0.04; limits of agreement about +/-0.4. The quality gate rejected the windows containing the breath-hold (2 of 5) and the hand-wave clip's only window (quality 0.53); the natural-breathing clip gave 16.8 breaths/min (no reference). Face blurring did not change the estimate (12.0 vs 12.0 breaths/min; signal correlation 0.988).

**BIDMC (53 patients, patient-level splits).**

| Method | MAE (breaths/min) | 95% CI (resampling patients) |
|---|---|---|
| Mean-rate baseline | 2.31 | 1.82 - 2.85 |
| Peak detection | 1.62 | 1.03 - 2.26 |
| Welch | 1.16 | 0.81 - 1.55 |
| 1D CNN | 0.98 | 0.69 - 1.38 |

The CNN's advantage over Welch is not distinguishable from zero (paired difference -0.18, 95% CI -0.57 to +0.18; lower for 34 of 53 patients). It is better in the common 12-20 range (MAE 0.70 vs 1.17) and worse at the extremes (below 12: 5.24 vs 2.75), where it regresses towards the mean. With the quality gate (windows scored >= 0.6 by Welch) Welch scores 0.55 and the CNN 0.68; this favours Welch because the gate is Welch's own confidence.

**Leakage.** Same CNN, only the split changed: random-window splits gave MAE 0.46 (0.40-0.54) versus 0.98 (0.69-1.38) for patient-level splits, so a leaky evaluation would have halved the error and made the CNN look 2.5x better than Welch.

**Explainability.** Grad-CAM attention was not concentrated at annotated breaths (1.01x average, same as a shifted control); the randomised-weights check passed (heatmap correlation 0.00); zeroing the highest-attention 5 s moved predictions by 1.28 bpm versus 1.23 (random) and 1.05 (lowest attention), a weak effect. The model appears to judge the rhythm of the whole window.

## 4. Limitations
- **Reference:** the video results compare to a metronome target, not an independent measurement; no accelerometer data were collected.
- **Sample:** one person, one camera setup, five clips, no variation in lighting, clothing, distance or camera mount, so the per-condition analysis is empty. Confidence intervals by subject are not estimable for the video data.
- **Population:** healthy adult at rest and adult ICU patients. No children, no abnormal breathing, no claims about asthma, wheeze or retractions.
- **Learned model:** one architecture, one seed, no tuning; operates on a clinical respiration signal, not video; regresses towards the mean at slow and fast rates; no confidence output.
- **Annotation:** BIDMC annotations are made on the same impedance signal the methods read.
- **De-identification:** verified on one clip; a missed face would not be blurred.

## 5. Next steps
Record a varied dataset with an independent reference (phone accelerometer) and several consenting participants; re-run the evaluation with subject-level confidence intervals; train and explain a video model (pretrained 3D CNN) only once there are enough subjects; add a confidence output to the learned model; evaluate in conditions closer to the paediatric target.
