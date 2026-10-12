# Glossary

Plain-language definitions for this project. Educational only; nothing here is medical advice.

## Clinical terms
- **Respiratory rate (RR):** the number of breaths per minute (one breath = one inhale plus one exhale). In this project it is written breaths/min (or bpm). It is measured here at rest only.
- **Tachypnea:** abnormally fast breathing. What counts as "fast" depends on age, so there is no single cut-off.
- **Retractions:** visible pulling-in of the skin between or below the ribs, or at the base of the neck, when someone is working hard to breathe. Seen especially in children. BreathCam cannot detect them.
- **Wheeze:** a high-pitched whistling sound during breathing, caused by narrowed airways (for example in asthma). It is a sound, so a camera-motion method cannot detect it.
- **Pediatric normal ranges:** typical respiratory rates change a lot with age. In the systematic review by Fleming et al. (2011), the median rate falls from about 44 breaths/min at birth to about 26 at age 2, and published reference ranges often disagreed with the evidence. This is why adult filter settings and adult results must not be applied to children.

## Signal and method terms
- **Breathing signal:** a one-dimensional trace that rises and falls with each breath (here: chest motion from video, or the impedance sensor in BIDMC).
- **Band-pass filter:** keeps only frequencies where breathing lives (here 0.1-1.0 Hz, i.e. 6-60 breaths/min for video) and removes slow drift and fast noise.
- **Welch's method:** estimates how strongly each frequency is present by averaging spectra of short pieces. The strongest peak (in Hz) times 60 is the rate in breaths/min.
- **Optical flow:** an estimate of how every pixel moved between two frames. The average vertical motion inside the chest region follows the chest rising and falling.
- **Region of interest (ROI):** the rectangle of the image that is analysed (the chest).
- **Quality score (0-1):** how much of the signal's breathing-band power sits in one narrow peak. Clean breathing scores near 1; noise or motion scores low. Windows below 0.6 are rejected.
- **Window:** a 30 s stretch of signal analysed at once (a new one every 5 s).

## Evaluation terms
- **Reference / ground truth:** the value an estimate is compared with. For BIDMC: manual breath annotations. For the video clips so far: the metronome rate (a target, not a measurement).
- **MAE / RMSE:** average absolute error / root-mean-square error, in breaths/min. RMSE punishes large errors more.
- **Bland-Altman plot, bias, limits of agreement:** plots the difference between two methods against their average. The bias is the average difference; the limits of agreement (bias +/- 1.96 standard deviations) show the range containing about 95% of differences.
- **Bootstrap confidence interval:** an uncertainty range from re-sampling the data many times. Here whole patients are re-sampled, because windows from one patient are not independent.
- **Data leakage:** information from the test set getting into training, for example windows of the same patient in both. It makes results look better than they are.
- **Grad-CAM:** a heatmap showing which part of the input most influenced a neural network's output.
- **De-identification:** removing what could identify a person: blurring faces, dropping audio and stripping metadata such as GPS location and device model.
