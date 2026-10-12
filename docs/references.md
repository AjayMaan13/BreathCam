# References

Working bibliography for BreathCam. Mirrors the "BreathCam" Zotero collection. Citation metadata (title, authors, volume, pages, DOI) was checked against Crossref, or against the arXiv / publisher page for the two arXiv-only papers. The "Why it matters" notes are mine, written for this project.

## Respiratory rate: ranges, datasets, reviews

1. Fleming S, Thompson M, Stevens R, Heneghan C, Plüddemann A, Maconochie I, Tarassenko L, Mant D. Normal ranges of heart rate and respiratory rate in children from birth to 18 years of age: a systematic review of observational studies. *The Lancet*. 2011;377(9770):1011-1018. doi:[10.1016/S0140-6736(10)62226-X](https://doi.org/10.1016/S0140-6736(10)62226-X)
   - **Why it matters:** Evidence-based pediatric respiratory-rate ranges. Median falls from about 44 breaths/min at birth to 26 at age 2, and the paper reports large disagreement with existing published reference ranges. It is the reason the filter band should be wider than adult defaults for children (no pediatric claims are made in this project).
   - **Used in:** glossary, limitations.

2. Pimentel MAF, Johnson AEW, Charlton PH, Birrenkott D, Watkinson PJ, Tarassenko L, Clifton DA. Toward a robust estimation of respiratory rate from pulse oximeters. *IEEE Transactions on Biomedical Engineering*. 2017;64(8):1914-1923. doi:[10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124)
   - **Why it matters:** Introduces the BIDMC dataset used for the signal and CNN experiments. Breath annotations come from two annotators working on the impedance respiratory signal.
   - **Used in:** Phases 1, 5, 6.

3. Charlton PH, Birrenkott DA, Bonnici T, Pimentel MAF, Johnson AEW, Alastruey J, Tarassenko L, Watkinson PJ, Beale R, Clifton DA. Breathing rate estimation from the electrocardiogram and photoplethysmogram: a review. *IEEE Reviews in Biomedical Engineering*. 2018;11:2-20. doi:[10.1109/RBME.2017.2763681](https://doi.org/10.1109/RBME.2017.2763681)
   - **Why it matters:** Review of how breathing rate is estimated from the ECG and pulse-oximeter signals; background for the clinical-signal side of the project.
   - **Used in:** related work.

## Contactless and camera-based sensing

4. Massaroni C, Nicolò A, Sacchetti M, Schena E. Contactless methods for measuring respiratory rate: a review. *IEEE Sensors Journal*. 2021;21(11):12821-12839. doi:[10.1109/JSEN.2020.3023486](https://doi.org/10.1109/JSEN.2020.3023486)
   - **Why it matters:** Survey of non-contact approaches (camera, radar, thermal and others). Gives the landscape BreathCam sits in.
   - **Used in:** related work.

5. Rehouma H, Noumeir R, Essouri S, Jouvet P. Advancements in methods and camera-based sensors for the quantification of respiration. *Sensors*. 2020;20(24):7252. doi:[10.3390/s20247252](https://doi.org/10.3390/s20247252) (open access, [PMC7766256](https://pmc.ncbi.nlm.nih.gov/articles/PMC7766256/))
   - **Why it matters:** Review focused on camera-based systems for respiratory function assessment, which matches the video pipeline. Two of the authors (Noumeir, Jouvet) are at CHU Sainte-Justine.
   - **Used in:** related work.

## Methods used

6. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*. 1967;15(2):70-73. doi:[10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   - **Why it matters:** The spectral estimator behind `welch_rate`.
   - **Used in:** Phases 1-2.

7. Farnebäck G. Two-frame motion estimation based on polynomial expansion. In: *Image Analysis (SCIA 2003), Lecture Notes in Computer Science*. 2003:363-370. doi:[10.1007/3-540-45103-X_50](https://doi.org/10.1007/3-540-45103-X_50)
   - **Why it matters:** The dense optical-flow method (`cv2.calcOpticalFlowFarneback`) that turns video into the breathing signal.
   - **Used in:** Phase 2.

8. Bazarevsky V, Grishchenko I, Raveendran K, Zhu T, Zhang F, Grundmann M. BlazePose: on-device real-time body pose tracking. arXiv:[2006.10204](https://arxiv.org/abs/2006.10204). 2020.
   - **Why it matters:** The pose model family behind MediaPipe Pose, used to locate the shoulders and hips for the chest region.
   - **Used in:** Phase 2 (`roi.py`).

## Evaluation and explanation

9. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*. 1986;327(8476):307-310. doi:[10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8)
   - **Why it matters:** Basis for the Bland-Altman plots (bias and 95% limits of agreement). Explains why correlation is not agreement.
   - **Used in:** Phases 1, 4.

10. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. *International Journal of Computer Vision*. 2020;128(2):336-359. doi:[10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7)
    - **Why it matters:** The explanation method applied (in 1D) to the CNN in Phase 6.
    - **Used in:** Phase 6.

11. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. arXiv:[1810.03292](https://arxiv.org/abs/1810.03292).
    - **Why it matters:** Argues that a plausible-looking heatmap is not evidence; proposes randomisation tests. Source of the randomised-weights check in Phase 6.
    - **Used in:** Phase 6.

## Not a paper, still to cite
- Information and Privacy Commissioner of Ontario, guidance on the Personal Health Information Protection Act (PHIPA). Used only as a vocabulary reference in the ethics statement; not yet read in full.
- Tri-Council Policy Statement: Ethical Conduct for Research Involving Humans (TCPS 2, 2022) and its CORE-2022 tutorial.
