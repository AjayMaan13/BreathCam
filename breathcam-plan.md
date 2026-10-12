# BreathCam: Full Project Plan

**Goal:** Build a camera-based respiratory-rate estimator and evaluate it the way a clinician-facing research team would: independent ground truth, agreement statistics, subject-level validation, and honest limitations.

**Why this project:** It targets the Seneca / CHU Sainte-Justine RespiSentinel research assistant posting: image processing, computer vision, PyTorch, deep learning, annotation, SQL, Git, literature review, technical writing, and research ethics.

**Pace:** 10-day sprint at roughly 4-6 focused hours per day. If you need it faster, the "Cut list" at the bottom says what to drop first.

**Compute (MacBook Air M1, 8GB):** Do signal work, MediaPipe, and optical flow locally at low resolution (downscale to 640x480 or smaller). Do all deep-learning training on Colab or Kaggle GPUs. Don't train locally; it will run hot and slow.

---

## Timeline at a glance

| Day | Phase | Output |
|---|---|---|
| 1 | 0. Setup, ethics, lit review start | Repo, TCPS 2 started, Zotero library |
| 1-2 | 1. Signal foundations (BIDMC) | Notebook: respiratory rate from a clinical signal, scored against annotations |
| 2-4 | 2. Video pipeline v1 | Working video-to-breaths/min pipeline |
| 4-5 | 3. Data collection and SQL layer | Recorded dataset, SQLite database, consent log |
| 5-6 | 4. Evaluation | MAE, Bland-Altman, per-condition tables, failure analysis |
| 6-8 | 5. Deep learning v2 | Fine-tuned video model vs. v1, subject-level splits |
| 8-9 | 6. Explainability and leakage experiment | Temporal Grad-CAM, leakage figure |
| 9-10 | 7. Demo, de-ID, write-up, resume | Streamlit demo, README, 2-page report, bullets |

The lit review runs in the background across all phases (about 30 minutes a day).

---

## Repo structure

```
breathcam/
├── data/                # raw clips (gitignored), metadata, consent log
├── sql/
│   ├── schema.sql
│   └── eval_queries.sql
├── breathcam/
│   ├── roi.py           # MediaPipe torso ROI
│   ├── motion.py        # optical flow / frame differencing
│   ├── signal.py        # filtering, Welch, peaks, quality score
│   ├── db.py            # SQLite helpers
│   ├── dataset.py       # PyTorch Dataset + subject-level splits
│   ├── models.py        # 1D CNN, r3d_18 head, CNN+LSTM
│   ├── explain.py       # temporal Grad-CAM
│   ├── deid.py          # face blur + metadata strip
│   └── eval.py          # MAE, Bland-Altman, per-condition tables
├── notebooks/
├── tests/               # unit tests on synthetic signals
├── app.py               # Streamlit demo
├── README.md
└── docs/                # report, ethics statement, protocol
```

**Git habits to practice (the posting asks for collaboration):** one issue per phase, a branch per phase, a pull request into `main` (self-review it), meaningful commit messages, and a short `CHANGELOG.md`.

---

# Phase 0: Setup, Ethics, and Lit Review Kickoff (Day 1)

## Learning before Phase 0

**What you're learning:** How research projects are scoped, how human-participant research is governed, and how to read papers efficiently.

**Concepts**
- **TCPS 2 (Tri-Council Policy Statement)**: the Canadian ethics framework for research involving humans. The CORE-2022 tutorial is the standard training and gives you a certificate.
- **Informed consent**: what participants must know, and that they can withdraw at any time.
- **Identifiable data**: face video is identifiable; a breathing signal alone is much less so. This distinction drives your privacy design.
- **How to read a paper fast**: abstract, then figures, then methods, then limitations. Don't read linearly.

**Resources**
- The TCPS 2 CORE-2022 online tutorial (free; search "TCPS 2 CORE-2022").
- Zotero (free reference manager) plus its browser connector.
- Search terms for papers: "camera-based respiratory rate estimation", "non-contact respiration video optical flow", "remote photoplethysmography review", "pediatric respiratory rate normal ranges".

**Check yourself**
- Can you explain in two sentences why a face video is more sensitive than a breathing waveform?
- What would you tell a friend who asks "what happens to my video?"

## Tasks
- [x] Create the GitHub repo with the structure above, a `.gitignore` that excludes `data/`, a README stub, and a Python environment (`uv` or `venv`). Pin versions in `requirements.txt`.
- [x] Start TCPS 2 CORE-2022 (finish by Day 3).
- [x] Set up Zotero; add the first 5 papers; make a "BreathCam" collection.
- [x] Draft a one-page consent form: purpose, what's recorded, who sees it, storage, retention, withdrawal, contact. Save as `docs/consent_form.md`.
- [x] Create `docs/ethics_statement.md` (fill in later).

**Done when:** repo is pushed, consent draft exists, TCPS 2 is in progress.

---

# Phase 1: Signal Foundations on Clinical Data (Day 1-2)

## Learning before Phase 1

**What you're learning:** How a breathing waveform becomes a rate, using a real clinical dataset with real annotations before you touch video.

**Concepts**
- **Sampling rate and Nyquist**: you can only see frequencies below half the sampling rate. Why 30 fps video is plenty for breathing (under 1 Hz).
- **Detrending**: removing slow drift so it doesn't dominate the filter.
- **Bandpass filtering**: keeping only the frequency band where breathing lives. Butterworth filters, and why `sosfiltfilt` (zero-phase, second-order sections) is preferred over a plain `lfilter`.
- **Spectral estimation**: FFT vs. Welch's method (averaged, smoother, better for noisy short windows). What the dominant peak means, and why `rate_bpm = peak_Hz x 60`.
- **Peak detection**: `find_peaks` with `distance` and `prominence`; how breath-by-breath intervals give variability.
- **Pediatric context**: normal respiratory rates are much higher in young children than adults, so your band must be wider than adult defaults. Look up age-based reference ranges and cite them.
- **Annotation as ground truth**: the BIDMC dataset's breaths were manually annotated; you will score your algorithm against them.

**Resources**
- `scipy.signal` documentation: `butter`, `sosfiltfilt`, `welch`, `find_peaks`.
- PhysioNet: "BIDMC PPG and Respiration Dataset" (check the page for the exact signals and annotation details). Load with the `wfdb` Python package.
- A short DSP primer on the Nyquist theorem and filter design (any university lecture notes will do).

**Check yourself**
- Generate a synthetic 0.25 Hz sine plus noise and drift. Can you recover 15 breaths/min?
- What happens to your estimate if the window is only 10 s long? Why?
- Why would peak detection and Welch disagree on a noisy signal?

## Tasks
- [x] Write `signal.py` with: `detrend`, `bandpass(low, high, fs)`, `welch_rate(x, fs, window_s, step_s)`, `peak_rate(x, fs)`, and `quality_score` (peak prominence relative to in-band power).
- [x] Write unit tests on synthetic signals (known rate, added noise, added drift, added motion spikes).
- [x] Load BIDMC with `wfdb`; extract respiratory rate from the respiration signal; compare against the annotated breaths.
- [x] Compute MAE, RMSE, and a first Bland-Altman plot on this data.
- [x] Save the notebook as `notebooks/01_bidmc_signal.ipynb`.

**Done when:** your synthetic tests pass and you can state a BIDMC MAE in breaths/min. This notebook is also your "worked with a clinically annotated dataset" line.

---

# Phase 2: Video Pipeline v1 (Day 2-4)

## Learning before Phase 2

**What you're learning:** How to turn pixels into a 1D physiological signal.

**Concepts**
- **Video basics**: frame rate, resolution, color spaces, why you convert to grayscale for motion work, and how to read frames with `cv2.VideoCapture`.
- **Pose/landmark detection (MediaPipe Pose)**: it returns normalized landmark coordinates (shoulders, hips). You turn them into a pixel ROI. Understand visibility scores and why you recompute the ROI periodically.
- **Optical flow**: estimating per-pixel motion between consecutive frames. Dense methods (Farneback) vs. sparse (Lucas-Kanade). Why mean vertical flow in the chest ROI approximates breathing motion.
- **Frame differencing**: the simple baseline. Faster, noisier. You'll compare it against optical flow.
- **Denoising**: Gaussian blur before flow; temporal smoothing after; outlier rejection for sudden motion (talking, fidgeting).
- **Signal vs. artifact**: global camera shake, lighting flicker (50/60 Hz aliasing), and body sway all create fake "breathing". Learn to recognize each in the raw signal.
- **Quality gating**: refusing to answer is a feature. Clinical tools report confidence.

**Resources**
- OpenCV optical flow tutorial (`calcOpticalFlowFarneback`).
- MediaPipe Pose documentation (Python solutions).
- Search terms: "video-based respiratory rate optical flow", "chest ROI motion respiration".

**Check yourself**
- Hold your breath for 10 s on camera. Does your signal go flat? That's your sanity test.
- Wave your hand in the ROI. Can your quality score detect the artifact?
- Why use vertical flow rather than total flow magnitude?

## Tasks
- [x] `roi.py`: MediaPipe Pose, build a chest ROI from shoulder and hip landmarks, smooth it over time, and fall back to the last good ROI when visibility drops.
- [x] `motion.py`: Farneback mean vertical flow per frame in the ROI; also a frame-differencing baseline. Downscale frames (e.g., 320x240) for speed.
- [x] Wire it together: video file to raw signal to bandpass (about 0.1-1.0 Hz) to sliding-window Welch (30 s window, 5 s step) to rate plus quality score.
- [x] Record a few quick test clips of yourself (paced at 12 and 20 breaths/min with a metronome) and iterate until the estimate is in the right neighbourhood.
- [x] Plot the raw signal, filtered signal, spectrum, and detected peaks in one diagnostic figure. You'll reuse it in the write-up.
- [x] Add a CLI: `python -m breathcam.run --video clip.mp4`.

**Done when:** the pipeline outputs a rate and a confidence for any clip, and your sanity tests (breath-hold, hand-wave) behave sensibly.

---

# Phase 3: Data Collection and SQL Layer (Day 4-5)

## Learning before Phase 3

**What you're learning:** How to design an experiment so results mean something, and how to store and query research data.

**Concepts**
- **Ground truth vs. a target**: a metronome gives the intended rate, not the true rate. Use an independent sensor (phone accelerometer) as the reference and report against it.
- **Time synchronization**: a shared clap-and-tap event at the start of each recording lets you align video and sensor streams.
- **Experimental design**: change one factor at a time (rate, lighting, clothing, distance, camera mount); record the conditions as data, not as file names.
- **Subject-level thinking**: even 3-5 participants lets you do subject-wise splits later. Plan for it now.
- **Relational modelling**: subjects, sessions, reference readings, predictions; primary keys, foreign keys, why separate tables prevent mistakes.
- **SQL for analysis**: `JOIN`, `GROUP BY`, aggregate functions, `CASE WHEN`, and CTEs (`WITH`). Computing MAE in SQL.
- **Data hygiene**: a consent log, pseudonymous subject IDs (S01, S02), and keeping the key-to-name mapping off the repo.

**Resources**
- SQLite documentation (built into Python via `sqlite3`).
- Any "SQL JOINs and GROUP BY" tutorial; practice on your own tables.
- A free accelerometer logger app for the phone (e.g., Physics Toolbox) that exports CSV.

**Check yourself**
- Write a query returning MAE per lighting condition without using pandas.
- Why is it a problem if the same subject appears in both train and test?

## Data collection protocol

- **Setup:** fixed camera on a tripod, 30 fps, 1-2 m away, subject seated, facing the camera, phone taped to the chest (or abdomen) logging accelerometer.
- **Sync:** at the start of each recording, clap once in view and tap the phone once.
- **Clip length:** 60-90 s.
- **Participants:** you plus 2-4 consenting adults. Get written or recorded consent first (use your consent form). Assign IDs S01, S02, ...
- **Conditions (vary one factor at a time, 1-2 clips per cell):**

| Factor | Levels |
|---|---|
| Paced rate | 8, 12, 20, 30 breaths/min (metronome) |
| Lighting | bright, dim, backlit |
| Clothing | fitted, loose, hoodie |
| Distance | near (1 m), far (2 m) |
| Camera mount | tripod, handheld (expect failure; report it) |
| Natural breathing | after brief light exercise |

Aim for about 40-60 clips total across subjects. Don't try to fill every combination.

## SQL schema (`sql/schema.sql`)

```sql
CREATE TABLE subjects (
  subject_id TEXT PRIMARY KEY,
  consent_date TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE sessions (
  session_id INTEGER PRIMARY KEY,
  subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
  paced_bpm REAL,
  lighting TEXT,
  clothing TEXT,
  distance_m REAL,
  camera_mount TEXT,
  video_path TEXT NOT NULL,
  sensor_path TEXT
);

CREATE TABLE reference_readings (
  session_id INTEGER NOT NULL REFERENCES sessions(session_id),
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  ref_bpm REAL NOT NULL
);

CREATE TABLE predictions (
  session_id INTEGER NOT NULL REFERENCES sessions(session_id),
  window_start REAL NOT NULL,
  method TEXT NOT NULL,          -- welch | peaks | framediff | cnn1d | r3d18
  est_bpm REAL,
  quality_score REAL
);
```

## Example evaluation queries (`sql/eval_queries.sql`)

```sql
-- MAE by lighting, per method
SELECT s.lighting, p.method,
       ROUND(AVG(ABS(p.est_bpm - r.ref_bpm)), 2) AS mae
FROM predictions p
JOIN sessions s ON s.session_id = p.session_id
JOIN reference_readings r
  ON r.session_id = p.session_id
 AND p.window_start >= r.t_start AND p.window_start < r.t_end
WHERE p.est_bpm IS NOT NULL
GROUP BY s.lighting, p.method;

-- Share of windows within +/-2 bpm, per method
SELECT p.method,
       ROUND(100.0 * AVG(CASE WHEN ABS(p.est_bpm - r.ref_bpm) <= 2 THEN 1 ELSE 0 END), 1) AS pct_within_2
FROM predictions p
JOIN reference_readings r
  ON r.session_id = p.session_id
 AND p.window_start >= r.t_start AND p.window_start < r.t_end
WHERE p.est_bpm IS NOT NULL
GROUP BY p.method;

-- Windows rejected by the quality gate, by condition
SELECT s.lighting, s.camera_mount,
       COUNT(*) AS windows,
       SUM(CASE WHEN p.est_bpm IS NULL THEN 1 ELSE 0 END) AS rejected
FROM predictions p
JOIN sessions s ON s.session_id = p.session_id
GROUP BY s.lighting, s.camera_mount;
```

## Tasks
- [x] Finish TCPS 2; save the certificate PDF in `docs/` (don't commit if it shows personal info you'd rather keep private).
- [ ] Collect consent, then record the dataset per the protocol; export accelerometer CSVs.
- [ ] Write a script that aligns sensor and video via the sync event and derives `ref_bpm` per window using the same filter pipeline on the accelerometer Z axis.
- [x] Create the database; load subjects, sessions, and reference readings with `db.py`.
- [x] Run the pipeline on every clip and write predictions into `predictions`.

**Done when:** `SELECT COUNT(*) FROM predictions` returns thousands of rows and your three queries run.

---

# Phase 4: Evaluation (Day 5-6)

## Learning before Phase 4

**What you're learning:** How clinical measurement studies judge a new method against a reference.

**Concepts**
- **Error metrics**: MAE and RMSE, and why RMSE punishes outliers more. Report in breaths/min.
- **Agreement vs. correlation**: two methods can correlate strongly and still disagree systematically. Correlation alone is not agreement.
- **Bland-Altman analysis**: plot the difference against the mean of the two methods; report **bias** (mean difference) and **limits of agreement** (bias plus or minus 1.96 SD). Read the original Bland and Altman 1986 paper in *The Lancet*.
- **Accuracy within a tolerance**: "percent of readings within +/-2 bpm" is a common reporting convention.
- **Confidence intervals**: with few subjects, uncertainty is large; bootstrap by subject, not by window.
- **Failure analysis**: characterizing where it breaks matters as much as headline accuracy.
- **Sensitivity and specificity** (for the gate or the labored-vs-normal classifier): what each means, ROC curves, AUC.

**Resources**
- Bland JM, Altman DG, 1986, "Statistical methods for assessing agreement between two methods of clinical measurement."
- `scikit-learn` metrics (`roc_curve`, `roc_auc_score`, `cohen_kappa_score`).

**Check yourself**
- If bias is +3 bpm and limits are -5 to +11, would a clinician trust it? Why or why not?
- Why must the bootstrap resample subjects rather than windows?

## Tasks
- [x] `eval.py`: MAE, RMSE, Bland-Altman plot, within-2-bpm accuracy, and per-condition tables (pull from SQL).
- [x] Bootstrap confidence intervals by subject.
- [x] Compare Welch vs. peaks vs. frame differencing vs. optical flow.
- [x] Failure analysis: pick 3-5 worst clips, plot their signals, and write one line each on the cause (motion, low light, loose clothing, handheld camera).
- [x] Save headline figures to `docs/figures/`.

**Done when:** you can state, with numbers, where v1 works and where it fails.

---

# Phase 5: Deep Learning v2 (Day 6-8)

## Learning before Phase 5

**What you're learning:** How to apply pretrained video models to a small dataset without fooling yourself.

**Concepts**
- **3D convolutions**: convolution over time as well as space; what `r3d_18` and `r2plus1d_18` do; Kinetics-pretrained weights and why transfer learning matters on tiny datasets.
- **CNN+LSTM**: per-frame spatial features feeding a recurrent model for temporal context. Trade-offs vs. 3D CNNs.
- **1D CNNs on signals**: a small model on the optical-flow trace; a useful middle step that asks "does learning beat hand-tuned filtering?"
- **PyTorch data pipeline**: `Dataset`, `DataLoader`, collate functions, windowed clip sampling, frame sampling strategies (uniform vs. strided), and augmentation (brightness, small crops, temporal jitter).
- **Training loop**: forward, loss, backward, optimizer step, learning-rate schedule, early stopping, checkpointing, seeding for reproducibility.
- **Regression vs. classification framing**: predicting rate bins (e.g., slow/normal/fast) is easier and more robust on small data than regressing the exact rate.
- **Overfitting and data leakage**: overlapping windows from the same clip across train and test inflate accuracy. **Always split by subject** (or at minimum by session).
- **Class imbalance and baselines**: always compare against a trivial baseline (predict the majority class or the mean rate).

**Resources**
- PyTorch tutorials: "Datasets & DataLoaders" and "Transfer Learning".
- `torchvision.models.video` documentation (`r3d_18`, `r2plus1d_18`).
- Colab or Kaggle notebooks with a GPU runtime.

**Check yourself**
- Why can a model get 99% accuracy with random window splits and 60% with subject splits?
- What's your baseline, and does your model beat it?

## Tasks
- [ ] `dataset.py`: build clips of 16 frames at 112x112; `GroupKFold` or leave-one-subject-out splits using `subject_id` from the database. *(Done for BIDMC signal windows with patient-level `GroupKFold`; 16-frame video clips wait for more video data.)*
- [x] Model A: 1D CNN on the flow signal. *(Trained on the BIDMC respiration signal; the video flow signal needs more subjects.)*
- [ ] Model B: `r3d_18` (or `r2plus1d_18`) pretrained; freeze early layers; train the head; predict rate bin.
- [ ] Model C (optional): ResNet18 features plus LSTM.
- [x] Train on Colab/Kaggle; log metrics; save checkpoints; export per-window predictions into the `predictions` table. *(Model A is tiny and was trained locally on CPU; predictions are in `data/bidmc.db`.)*
- [x] Compare all methods in one table using your existing SQL queries. *(BIDMC: welch, peaks, cnn1d, mean baseline.)*

**Done when:** you have a fair, subject-split comparison of classical vs. deep methods, even if the classical method wins (say so).

---

# Phase 6: Explainability and the Leakage Experiment (Day 8-9)

## Learning before Phase 6

**What you're learning:** How to see what a video model is looking at, and how to quantify a common research mistake.

**Concepts**
- **Grad-CAM**: weight a convolutional layer's feature maps by the gradients of the target score to produce a heatmap. You've done this in 2D on GTSRB.
- **Temporal Grad-CAM**: apply it to a 3D CNN's last convolutional block to get a heatmap per time step; overlay on frames and on the timeline.
- **Sanity checks for explanations**: a heatmap that looks plausible isn't proof; run a randomized-weights check, or occlude the highlighted region and see whether the prediction drops.
- **Data leakage**: information from the test set leaking into training; in video, usually through shared subjects, sessions, or overlapping windows.
- **Experimental comparison**: same model, same data, only the split changes.

**Resources**
- The original Grad-CAM paper (Selvaraju et al.).
- Your own GTSRB Grad-CAM code, which is the starting point.

**Check yourself**
- If your heatmap lights up the face or background, what does that tell you about the model?
- What's the minimal change you need to make to demonstrate leakage?

## Tasks
- [ ] `explain.py`: temporal Grad-CAM for the 3D model; export overlay videos or frame strips.
- [ ] Occlusion test: mask the chest ROI vs. a background patch and compare the effect on predictions.
- [ ] Leakage experiment: train the same model with (a) random window splits and (b) subject-level splits; report both accuracies in one bar chart.
- [ ] Write a half-page note: "What the model attends to" and "How much leakage inflates results".

**Done when:** you have one explanation figure and one leakage figure you could put in a slide.

---

# Phase 7: Demo, De-ID, Write-up, and Resume (Day 9-10)

## Learning before Phase 7

**What you're learning:** How to communicate research clearly and handle health-adjacent data responsibly.

**Concepts**
- **De-identification**: face blurring, stripping metadata (EXIF/container tags), file naming without names, and checking whether your signal survives the blur.
- **Canadian health privacy context**: PHIPA (Ontario's personal health information law) in plain terms; identifiable vs. de-identified data; consent, minimum necessary use, secure storage. You're working with volunteers, not patient data, but you should know the vocabulary.
- **Technical writing for researchers**: the structure (problem, method, data, results, limitations, next steps), figures that stand alone, and honest limitations.
- **Reproducibility**: pinned dependencies, seeds, a one-command run, and a README that someone else can follow.
- **Demo design**: a 30-second screen recording that shows input, signal, rate, and confidence.

**Resources**
- Ontario IPC guidance on PHIPA (search "PHIPA overview Ontario IPC").
- MediaPipe face detection or OpenCV's DNN face detector for blurring.
- Streamlit documentation.

**Check yourself**
- Could a stranger reproduce your headline result from the README alone?
- What does your ethics statement say about what you did not do?

## Tasks
- [ ] `deid.py`: detect and blur faces, strip metadata, and verify the breathing estimate still works on blurred video.
- [ ] `app.py` (Streamlit): upload or webcam, live signal plot, rate and confidence, a clear "research prototype" banner.
- [ ] Record a 30-second demo clip and add a GIF to the README.
- [ ] README sections: overview, install and run, data protocol, results table, figures, limitations, ethics statement, future work.
- [ ] Two-page technical summary in `docs/report.md` (or PDF): background, methods, results, limitations.
- [ ] Finish the lit review: 8-10 papers in Zotero, a half-page "related work" paragraph, and a one-page glossary (respiratory rate, tachypnea, retractions, wheeze, pediatric normal ranges).
- [ ] Tag a release (`v1.0`), clean the repo, and make it public (data excluded).

## Resume bullets (fill with real numbers only)

> Built BreathCam, a video-based respiratory-rate estimator (MediaPipe, OpenCV optical flow, bandpass/Welch), validated against a chest-worn accelerometer on N volunteers with a SQL-backed evaluation pipeline; MAE of X breaths/min.

> Fine-tuned a Kinetics-pretrained 3D CNN in PyTorch with subject-level validation, quantified leakage inflation (X% vs. Y%), and applied temporal Grad-CAM to verify chest-focused attention.

> Completed TCPS 2 CORE-2022; implemented face de-identification and consent protocol for human-participant video data.

---

## Limitations to state in the README

- Healthy adult volunteers, not children; no pediatric performance claims.
- Small sample, single camera type, controlled conditions.
- Paced and resting breathing only; this does not detect asthma, wheeze, or retractions.
- Research prototype, not a medical device.

## Optional: link it to the posting's goal (exacerbation prediction)

If time allows after Day 10: compute rolling features of rate and quality per subject (mean, variability, trend), define a personal baseline, and flag sustained deviations with a simple model (logistic regression or gradient boosting). Label it as a toy "early warning" demonstration of multimodal, time-series thinking, not a validated predictor.

## Cut list (if you need to finish sooner)

Drop in this order:
1. CNN+LSTM model (keep 1D CNN and r3d_18).
2. Handheld-camera and natural-after-exercise conditions.
3. Streamlit live webcam mode (keep upload only).
4. Occlusion sanity test (keep Grad-CAM and leakage figure).
5. The optional exacerbation-prediction add-on.

**Do not cut:** the independent ground truth, subject-level splits, SQL layer, ethics steps, and the limitations section. They are what make the project credible.

## Daily checklist

- [ ] Commit and push
- [ ] 30 minutes of reading (one paper section) into Zotero notes
- [ ] Update the README results table with anything new
- [ ] Write down tomorrow's first task
