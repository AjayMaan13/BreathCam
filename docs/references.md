# References

Working bibliography for BreathCam. Mirrors the "BreathCam" Zotero collection. Citation metadata (title, authors, volume, pages, DOI) was checked against Crossref. The "Why it matters" notes are mine, written for this project.

Target by the end of Phase 7: 8-10 papers.

## Core references

1. Fleming S, Thompson M, Stevens R, Heneghan C, Plüddemann A, Maconochie I, Tarassenko L, Mant D. Normal ranges of heart rate and respiratory rate in children from birth to 18 years of age: a systematic review of observational studies. *The Lancet*. 2011;377(9770):1011-1018. doi:[10.1016/S0140-6736(10)62226-X](https://doi.org/10.1016/S0140-6736(10)62226-X)
   - **Why it matters:** Evidence-based pediatric respiratory-rate ranges. Median falls from about 44 breaths/min at birth to 26 at age 2, and the paper reports large disagreement with existing published reference ranges. Cited for the filter band in Phase 1 (wider than adult defaults) and in the glossary entry on pediatric normal ranges.
   - **Used in:** Phase 1, Phase 7 glossary.

2. Pimentel MAF, Johnson AEW, Charlton PH, Birrenkott D, Watkinson PJ, Tarassenko L, Clifton DA. Toward a robust estimation of respiratory rate from pulse oximeters. *IEEE Transactions on Biomedical Engineering*. 2017;64(8):1914-1923. doi:[10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124)
   - **Why it matters:** Introduces the BIDMC dataset used in Phase 1. Breath annotations come from two annotators working on the impedance respiratory signal. Read for how the ground truth was built and how algorithms were scored against it.
   - **Used in:** Phase 1.

3. Massaroni C, Nicolò A, Sacchetti M, Schena E. Contactless methods for measuring respiratory rate: a review. *IEEE Sensors Journal*. 2021;21(11):12821-12839. doi:[10.1109/JSEN.2020.3023486](https://doi.org/10.1109/JSEN.2020.3023486)
   - **Why it matters:** Survey of non-contact approaches (camera, radar, thermal and others). Gives the landscape BreathCam sits in.
   - **Used in:** Related work (Phase 7).

4. Rehouma H, Noumeir R, Essouri S, Jouvet P. Advancements in methods and camera-based sensors for the quantification of respiration. *Sensors*. 2020;20(24):7252. doi:[10.3390/s20247252](https://doi.org/10.3390/s20247252) (open access, [PMC7766256](https://pmc.ncbi.nlm.nih.gov/articles/PMC7766256/))
   - **Why it matters:** Review focused on camera-based systems for respiratory function assessment, which matches the video pipeline. Two of the authors (Noumeir, Jouvet) are at CHU Sainte-Justine, so it is worth knowing well.
   - **Used in:** Phase 2, related work (Phase 7).

5. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*. 1986;327(8476):307-310. doi:[10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8)
   - **Why it matters:** Basis for the Bland-Altman plots (bias and 95% limits of agreement). Explains why correlation is not agreement. Read before collecting data so the evaluation design is right.
   - **Used in:** Phase 1 (first plot), Phase 4.

## Planned (not yet added)

- Selvaraju et al., Grad-CAM (Phase 6)
- Ontario IPC guidance on PHIPA (Phase 7)
