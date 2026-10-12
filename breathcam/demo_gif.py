"""Make the README demo GIF: chest crop + live breathing signal (no face is ever shown).

    python -m breathcam.demo_gif --video data/deid/01_12bpm.mp4 --roi 115,78,205,135 --out docs/demo.gif
"""

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .run import estimate


def make_gif(video, roi, out, start_s=10, length_s=30, fps=6, trail_s=15):
    """Render a GIF of the chest region next to the breathing signal.

    Inputs:
        video: video file (a de-identified copy is recommended).
        roi: chest box (x0, y0, x1, y1) in 320-px-wide frame coordinates.
        out: output .gif path.
        start_s, length_s: which part of the clip to show.
        fps: GIF frames per second (real time).
        trail_s: seconds of signal shown behind the cursor.
    Returns:
        None (writes the file).
    """
    r = estimate(video, "flow", roi)
    sig, vfps = r["signal"], r["fs"]
    cap = cv2.VideoCapture(str(video))
    scale = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 320
    x0, y0, x1, y1 = (int(v * scale) for v in roi)
    frames = []
    for k in range(int(length_s * fps)):
        t = start_s + k / fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * vfps))
        ok, img = cap.read()
        if not ok:
            break
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw=dict(width_ratios=[1, 2.2]), dpi=80)
        a.imshow(img[y0:y1, x0:x1, ::-1]); a.axis("off"); a.set_title("chest region (only this is analysed)", fontsize=9)
        i = int(t * vfps); lo = max(0, i - int(trail_s * vfps))
        b.plot(np.arange(lo, i) / vfps, sig[lo:i], "C0", lw=1.2)
        b.plot(t, sig[i], "ro", ms=4)
        b.set(xlim=(t - trail_s, t + 0.5), xlabel="seconds", yticks=[])
        b.set_title(f"breathing signal -> {r['rate_bpm']:.1f} breaths/min (clip quality {r['quality_overall']:.2f})", fontsize=9)
        fig.text(0.5, 0.01, "Research prototype. Not a medical device.", ha="center", fontsize=7, color="gray")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(64))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--roi", required=True)
    p.add_argument("--out", default="docs/demo.gif")
    a = p.parse_args()
    make_gif(a.video, tuple(int(v) for v in a.roi.split(",")), a.out)
    print("wrote", a.out)
