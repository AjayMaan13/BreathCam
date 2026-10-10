# BreathCam

Camera-based respiratory-rate estimation, evaluated the way a clinical measurement study would be: independent ground truth, agreement statistics (Bland-Altman), subject-level validation, and honest limitations.

> **Research prototype. Not a medical device.** Do not use it to make clinical decisions.

**Status:** Phase 1 done (signal estimation on BIDMC). Phase 2 code done; needs real test clips.

Estimate a rate from a video (one-time pose-model download first, see the error message or `breathcam/roi.py`):

```bash
python -m breathcam.run --video data/raw/self_12bpm.mp4 --plot out.png
```

```bash
python -m pytest                                 # unit tests on synthetic signals
python -c "import wfdb; wfdb.dl_database('bidmc','data/bidmc'); wfdb.dl_files('bidmc','data/bidmc',[f'bidmc{i:02d}.breath' for i in range(1,54)])"
jupyter notebook notebooks/01_bidmc_signal.ipynb  # needs the BIDMC download above
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`).

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Direct dependencies live in `requirements.in`; `requirements.txt` is the fully pinned resolution (`uv pip compile requirements.in --python-version 3.12 -o requirements.txt`).

Signal work, MediaPipe and optical flow run locally at low resolution. Deep-learning training is intended for Colab or Kaggle GPUs.

## Repository layout

```
breathcam/   # package (modules added phase by phase)
sql/         # schema and evaluation queries
notebooks/   # analysis notebooks
tests/       # unit tests on synthetic signals
docs/        # consent form, ethics statement, references, report
data/        # gitignored: raw clips, database, consent log
```

## Documentation

- [Project plan](breathcam-plan.md)
- [References](docs/references.md)
- [Consent form (draft)](docs/consent_form.md)
- [Ethics statement (in progress)](docs/ethics_statement.md)

## Limitations

- Healthy adult volunteers, not children; no pediatric performance claims.
- Small sample, single camera type, controlled conditions.
- Paced and resting breathing only; this does not detect asthma, wheeze, or retractions.
- Research prototype, not a medical device.

(Expanded as results arrive.)
