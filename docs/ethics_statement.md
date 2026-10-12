# Ethics Statement

Status: final for v1.0, except one item marked **[researcher to fill in]** (the retention date), which only the project owner can set.

## 1. Scope and oversight
- Personal skills-development project. It is not affiliated with any institution or clinical service.
- **No research ethics board (REB) reviewed this project.** It was designed to follow TCPS 2 principles, but it has not been assessed by anyone independent.
- The researcher completed the TCPS 2 CORE-2022 tutorial. The certificate is kept locally and is not committed to the repository (it shows a personal name).
- No patient data was collected, and no children were recorded.

## 2. Participants and consent
- The only person recorded on video is the researcher (subject code S01). No other person has been recorded.
- Consent from oneself is not a substitute for ethics review of recordings of others. A participant information and consent form ([`consent_form.md`](consent_form.md)) was drafted for the day other volunteers are recorded; it is still a draft with placeholders and must be finalised before any recording of another person. Everything described below as "not uploaded to any cloud service" corresponds to its Option B.
- The learned-model experiments use the public BIDMC PPG and Respiration Dataset (PhysioNet; Pimentel et al. 2017), de-identified by its publishers. It was used under its own licence terms and is not redistributed in this repository.

## 3. Data protection
- Participants are identified only by pseudonymous codes (S01, ...). Any code-to-name key would be stored apart from the data and never in this repository.
- Raw video, the SQLite database of results, model checkpoints and downloaded datasets are excluded from version control (`.gitignore`). Nothing identifiable is committed.
- **No video has been uploaded to any cloud service or notebook.** All processing and training ran on the researcher's own computer. The demo app processes uploads locally and deletes its temporary copy immediately after each step.
- **De-identification (`breathcam/deid.py`).** It blurs faces on every frame (BlazeFace detector, a padded box, and detections carried over to neighbouring frames when one is missed), drops audio, and writes a new file with no source metadata. On the researcher's 93 s test clip the detector found the face in 2,777 of 2,782 frames, the remaining 5 were covered from neighbouring frames, and 0 frames were left unblurred. The breathing estimate was unchanged (12.0 breaths/min before and after; correlation of the raw signals 0.988). The metadata removed included precise GPS coordinates, the device model and the recording time, all present in the original phone file. Limits: detection was checked on one person and one setup, and a missed face would not be blurred, so output should still be looked at before sharing.
- The README demo GIF shows only a crop of the shirt and the signal, never a face.
- Retention: raw video and any key will be deleted by **[researcher to fill in: date]**.

## 4. Intended use and misuse
- Research prototype, not a medical device. It makes no clinical claims and no pediatric claims. The app and README say so.
- Foreseeable misuse: presenting a camera estimate as a clinical measurement, using it to decide care, or monitoring someone without their knowledge or consent. The tool should only be used with the knowledge and agreement of the person on camera.
- The method refuses to answer when signal quality is low, to reduce the risk of a confident wrong number, but it has been tested on very few conditions.

## 5. What I did not do
- No ethics board review.
- No recording of anyone other than the researcher; no patient, pediatric or clinical data of my own.
- No independent reference sensor for the video clips: they were scored against a metronome target, which is not a measurement of actual breathing.
- No validation across lighting, clothing, distance, camera type or people, and no test of unpaced or abnormal breathing beyond one natural-breathing clip.
- No detection of wheeze, retractions, asthma or any other condition.
- No video deep-learning models: the learned model works on a clinical respiration signal, not on video.
- No security audit of the demo app; it is meant to be run locally.

## 6. Applicable framework
- TCPS 2 (Canada) for research involving humans.
- PHIPA (Ontario) used as a vocabulary reference only. No legal determination is made here about whether it applies; the same principles (consent, minimum necessary use, secure storage) were followed regardless.
