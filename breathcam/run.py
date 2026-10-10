"""Command line: video file -> breathing rate and confidence.

    python -m breathcam.run --video data/raw/self_12bpm.mp4 --plot out.png
"""

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .motion import video_signal
from .roi import DEFAULT_MODEL
from .signal import bandpass, detrend, peak_rate, quality_score, welch_rate


def estimate(video, method="flow", roi=None, model=None, low=0.1, high=1.0,
             window_s=30, step_s=5, min_quality=0.6):
    """Run the whole pipeline on one video.

    Pipeline: video -> chest-ROI motion signal -> bandpass -> sliding-window
    Welch -> rate and quality per window. Windows below min_quality are rejected
    (the tool refuses to answer rather than guess).

    Inputs:
        video: video file path.
        method: "flow" or "diff" (see motion.py).
        roi: optional fixed (x0, y0, x1, y1) in 320-px-wide frame coordinates;
            skips MediaPipe.
        model: pose_landmarker .task path (default data/models/...).
        low, high: breathing band in Hz (0.1-1.0 Hz = 6-60 breaths/min).
        window_s, step_s: Welch window length and step in seconds (window is
            shortened to the clip length for short clips).
        min_quality: quality_score threshold (0..1) for accepting a window.
    Returns:
        dict: signal, fs, t (window starts, s), rate (bpm per window),
        quality (per window), accepted (bool per window), rate_bpm (median of
        accepted windows, NaN if none), quality_overall (whole-clip quality),
        peak_bpm (peak-detection rate on the whole clip), plus first_frame,
        rois and n_no_roi from video_signal.
    """
    v = video_signal(video, method=method, roi=roi, model=model)
    sig, fs = v["signal"], v["fs"]
    if len(sig) < 5 * fs:
        raise ValueError(f"clip too short ({len(sig) / fs:.1f} s); need at least 5 s")
    window_s = min(window_s, len(sig) / fs)
    t, rate, q = welch_rate(sig, fs, window_s, step_s, low, high)
    accepted = q >= min_quality
    filtered = bandpass(detrend(sig), low, high, fs)
    return dict(v, t=t, rate=rate, quality=q, accepted=accepted, window_s=window_s,
                rate_bpm=float(np.median(rate[accepted])) if accepted.any() else float("nan"),
                quality_overall=quality_score(filtered, fs, low, high),
                peak_bpm=float(peak_rate(sig, fs, low, high)[0]), low=low, high=high)


def diagnostic_figure(r, path):
    """Save a one-page diagnostic plot for an estimate() result.

    Panels: first frame with the ROI box, raw signal, filtered signal with
    detected breaths, whole-clip spectrum with the breathing band, and
    rate / quality per window.

    Inputs:
        r: dict returned by estimate().
        path: output image path (png).
    Returns:
        None (writes the file).
    """
    sig, fs, low, high = r["signal"], r["fs"], r["low"], r["high"]
    t = np.arange(len(sig)) / fs
    filtered = bandpass(detrend(sig), low, high, fs)
    _, peaks = peak_rate(sig, fs, low, high)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9))
    ax = ax.ravel()

    frame = r["first_frame"].copy()
    roi = next((b for b in r["rois"] if b is not None), None)
    if roi is not None:
        cv2.rectangle(frame, roi[:2], roi[2:], (0, 255, 0), 1)
    ax[0].imshow(frame[..., ::-1])
    ax[0].set_title("first frame + chest ROI" if roi else "first frame (no ROI found)")
    ax[0].axis("off")

    ax[1].plot(t, sig, lw=0.8)
    ax[1].set(title="raw signal (cumulative vertical motion)", xlabel="s", ylabel="px")

    ax[2].plot(t, filtered, lw=0.8)
    ax[2].plot(t[peaks], filtered[peaks], "rv", ms=4)
    ax[2].set(title=f"filtered, {len(peaks)} breaths detected (peak rate {r['peak_bpm']:.1f} bpm)",
              xlabel="s")

    freqs = np.fft.rfftfreq(len(filtered), 1 / fs)
    power = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered)))) ** 2
    ax[3].plot(freqs * 60, power, lw=0.8)
    ax[3].axvspan(low * 60, high * 60, color="g", alpha=0.1)
    ax[3].set(title="spectrum (green = breathing band)", xlabel="breaths/min", xlim=(0, 80))

    ax[4].plot(r["t"], r["rate"], "o-", ms=4)
    ax[4].plot(r["t"][~r["accepted"]], r["rate"][~r["accepted"]], "rx", ms=8, label="rejected")
    ax[4].set(title=f"Welch rate per {r['window_s']:.0f} s window", xlabel="window start (s)",
              ylabel="breaths/min")
    ax[4].legend()

    ax[5].plot(r["t"], r["quality"], "o-", ms=4)
    ax[5].set(title="quality per window", xlabel="window start (s)", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--method", choices=["flow", "diff"], default="flow")
    p.add_argument("--roi", help="x0,y0,x1,y1 in 320-px-wide frame coordinates (skips MediaPipe)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pose_landmarker .task file")
    p.add_argument("--plot", help="save diagnostic figure to this png")
    p.add_argument("--min-quality", type=float, default=0.6)
    p.add_argument("--window", type=float, default=30)
    p.add_argument("--step", type=float, default=5)
    a = p.parse_args()

    roi = tuple(int(s) for s in a.roi.split(",")) if a.roi else None
    r = estimate(a.video, a.method, roi, a.model, window_s=a.window, step_s=a.step,
                 min_quality=a.min_quality)
    print(f"{len(r['signal'])} frames at {r['fs']:.1f} fps, {r['n_no_roi']} without ROI")
    for t0, rate, q, ok in zip(r["t"], r["rate"], r["quality"], r["accepted"]):
        print(f"  window @{t0:5.1f}s  {rate:5.1f} bpm  quality {q:.2f}  {'' if ok else 'REJECTED'}")
    if np.isnan(r["rate_bpm"]):
        print(f"RESULT: no reliable estimate (whole-clip quality {r['quality_overall']:.2f})")
    else:
        print(f"RESULT: {r['rate_bpm']:.1f} breaths/min "
              f"(quality {np.median(r['quality'][r['accepted']]):.2f}; peak-detection says {r['peak_bpm']:.1f})")
    if a.plot:
        diagnostic_figure(r, a.plot)
        print("figure:", a.plot)


if __name__ == "__main__":
    main()
