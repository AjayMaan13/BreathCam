"""De-identification: blur faces, drop audio and strip metadata from a video.

    python -m breathcam.deid --video data/raw/01_12bpm.mov --out data/deid/01_12bpm.mp4 \\
        --verify --roi 115,78,205,135

The output is a new file written frame by frame, so none of the source's
container metadata (location, device, creation time) is carried over, and audio
(which could contain a voice) is dropped. If ffmpeg is installed the file is
re-encoded to H.264 with all remaining tags removed.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

FACE_MODEL = "data/models/blaze_face_short_range.tflite"


def face_detector(model_path=FACE_MODEL, min_confidence=0.4):
    """Build a face detector.

    Inputs:
        model_path: path to the MediaPipe BlazeFace .tflite model.
        min_confidence: detection threshold (low on purpose: a missed face leaks identity,
            a false alarm only blurs something extra).
    Returns:
        function detect(frame_bgr) -> list of (x0, y0, x1, y1) pixel boxes.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"face model not found: {model_path}\nDownload it with:\n"
            "  mkdir -p data/models && curl -L -o data/models/blaze_face_short_range.tflite "
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/"
            "float16/latest/blaze_face_short_range.tflite")
    opts = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_confidence)
    det = mp.tasks.vision.FaceDetector.create_from_options(opts)

    def detect(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return [(b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height)
                for b in (d.bounding_box for d in result.detections)]
    return detect


def fill_boxes(per_frame, hold=15):
    """Fill frames where no face was found with boxes from nearby frames.

    A detector misses faces on some frames (turned head, motion blur). Leaving
    those frames unblurred would leak identity, so the boxes of the nearest
    detection within `hold` frames (looking forward in time first, then back)
    are reused.

    Inputs:
        per_frame: list with, for each frame, a list of boxes (empty = none found).
        hold: how many frames a detection may be carried forward or backward.
    Returns:
        list of the same length with gaps filled where possible; frames with no
        detection within `hold` frames stay empty.
    """
    out = [list(b) for b in per_frame]
    for order in (range(len(out)), reversed(range(len(out)))):
        last, age = [], hold + 1
        for i in order:
            if per_frame[i]:
                last, age = per_frame[i], 0
            else:
                age += 1
                if not out[i] and age <= hold:
                    out[i] = list(last)
    return out


def blur_boxes(frame, boxes, pad=0.4):
    """Blur rectangles of a frame in place.

    Inputs:
        frame: HxWx3 uint8 image (modified).
        boxes: list of (x0, y0, x1, y1) pixel boxes.
        pad: fraction of the box size added on every side (covers hair, ears, chin).
    Returns:
        the frame.
    """
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in boxes:
        px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
        x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
        if x1 <= x0 or y1 <= y0:
            continue
        k = max(3, ((x1 - x0) // 2) | 1)  # odd kernel, about half the face width: unrecognisable
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (k, k), 0)
    return frame


def _scaled_frames(path, width):
    """Yield (frame, fps) pairs downscaled to `width`; fps is repeated for every frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA), fps
    cap.release()


def deidentify(src, dst, width=640, detect=None, hold=15, pad=0.4):
    """Write a de-identified copy of a video.

    Pass 1 detects faces on every frame; gaps are filled with nearby detections
    (fill_boxes); pass 2 blurs the boxes and writes the frames. The output has the
    same frame rate, no audio and no source metadata.

    Inputs:
        src: input video; dst: output .mp4 path (parent folders are created).
        width: output width in pixels (smaller files; the pipeline downsamples anyway).
        detect: face detector function (default: BlazeFace from face_detector()).
        hold, pad: see fill_boxes and blur_boxes.
    Returns:
        dict with frames, detected (frames with a real detection), filled (frames
        protected by a neighbouring detection), unprotected (frames with no blur at
        all: these may show a face) and fps.
    """
    detect = detect or face_detector()
    raw = [detect(f) for f, _ in _scaled_frames(src, width)]
    boxes = fill_boxes(raw, hold)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    writer, fps = None, 30.0
    for (frame, fps), b in zip(_scaled_frames(src, width), boxes):
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(blur_boxes(frame, b, pad))
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-map_metadata", "-1", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                        str(dst)], check=True)
    else:
        shutil.copy(tmp, dst)
    shutil.rmtree(tmp.parent)
    return dict(frames=len(raw), detected=sum(bool(r) for r in raw), fps=fps,
                filled=sum(1 for r, b in zip(raw, boxes) if b and not r),
                unprotected=sum(1 for b in boxes if not b))


def compare_estimates(src, dst, roi, method="flow"):
    """Check that the breathing estimate survives de-identification.

    Inputs:
        src, dst: original and de-identified videos.
        roi: chest box (x0, y0, x1, y1) in 320-px-wide frame coordinates.
        method: "flow" or "diff".
    Returns:
        dict with the rate and quality of both versions and the correlation of
        their raw breathing signals.
    """
    from .run import estimate
    a, b = estimate(src, method, roi), estimate(dst, method, roi)
    n = min(len(a["signal"]), len(b["signal"]))
    return dict(rate_original=a["rate_bpm"], rate_deid=b["rate_bpm"],
                quality_original=a["quality_overall"], quality_deid=b["quality_overall"],
                signal_corr=float(np.corrcoef(a["signal"][:n], b["signal"][:n])[0, 1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--verify", action="store_true", help="compare the breathing estimate before and after")
    p.add_argument("--roi", help="x0,y0,x1,y1 in 320-px-wide frame coordinates (needed for --verify)")
    a = p.parse_args()
    s = deidentify(a.video, a.out, a.width)
    print(f"{s['frames']} frames: {s['detected']} with a detected face, {s['filled']} covered by a neighbouring "
          f"detection, {s['unprotected']} unprotected -> {a.out}")
    if s["unprotected"]:
        print("WARNING: some frames have no blur and may show a face. Inspect the output before sharing.")
    if a.verify:
        if not a.roi:
            p.error("--verify needs --roi")
        r = compare_estimates(a.video, a.out, tuple(int(v) for v in a.roi.split(",")))
        print(f"original: {r['rate_original']:.1f} bpm (quality {r['quality_original']:.2f})   "
              f"de-identified: {r['rate_deid']:.1f} bpm (quality {r['quality_deid']:.2f})   "
              f"signal correlation {r['signal_corr']:.3f}")


if __name__ == "__main__":
    main()
