"""Turn video into a 1D breathing signal from motion in the chest ROI."""

import cv2
import numpy as np

from .roi import DEFAULT_MODEL, ChestROI


def vertical_flow(prev, curr):
    """Mean vertical motion between two grayscale crops, via Farneback optical flow.

    Inputs:
        prev, curr: 2D uint8 arrays of the same shape (consecutive frames, cropped to the ROI).
    Returns:
        float, mean vertical displacement in pixels (positive = moving down).
    """
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(flow[..., 1].mean())


def vertical_diff(prev, curr):
    """Cheaper baseline: vertical motion from frame differencing.

    Plain |frame difference| loses direction (it would peak on both inhale and
    exhale and double the apparent rate), so this uses the signed brightness-
    constancy estimate v = -sum(I_t * I_y) / sum(I_y^2) over the whole crop:
    one global vertical velocity per frame pair, no per-pixel flow field.

    Inputs:
        prev, curr: 2D uint8 arrays of the same shape (consecutive ROI crops).
    Returns:
        float, estimated vertical displacement in pixels (positive = moving down).
    """
    prev, curr = prev.astype(np.float32), curr.astype(np.float32)
    iy = np.gradient((prev + curr) / 2, axis=0)
    den = float((iy ** 2).sum())
    return float(-((curr - prev) * iy).sum() / den) if den > 0 else 0.0


METHODS = {"flow": vertical_flow, "diff": vertical_diff}


def video_signal(path, method="flow", roi=None, model=None, width=320, roi_every=10):
    """Read a video and produce one breathing-signal sample per frame.

    Each frame is downscaled to `width` pixels, converted to grayscale and
    blurred. The vertical motion between consecutive frames inside the chest ROI
    is accumulated (cumulative sum) so the output looks like chest position
    over time (breathing waveform) rather than velocity.

    Inputs:
        path: video file path.
        method: "flow" (Farneback optical flow) or "diff" (frame-differencing baseline).
        roi: fixed (x0, y0, x1, y1) in downscaled-pixel coordinates; if given,
            MediaPipe is not used.
        model: pose_landmarker .task path used when roi is None (default: DEFAULT_MODEL).
        width: downscaled frame width in pixels.
        roi_every: re-detect the ROI every this many frames.
    Returns:
        dict with
          signal: 1D array, one value per frame after the first,
          fs: frame rate in Hz,
          rois: list of the ROI used for each frame (None where not yet found),
          first_frame: the first downscaled BGR frame (for drawing the ROI),
          n_no_roi: number of frames with no ROI (their motion is counted as 0).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fs = cap.get(cv2.CAP_PROP_FPS) or 30.0
    motion = METHODS[method]
    tracker = ChestROI(model or DEFAULT_MODEL) if roi is None else None
    cur_roi, prev, first = roi, None, None
    vals, rois, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
        if first is None:
            first = small
        if tracker is not None and i % roi_every == 0:
            cur_roi = tracker.update(small)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            if cur_roi is None:
                vals.append(0.0)
            else:
                x0, y0, x1, y1 = cur_roi
                vals.append(motion(prev[y0:y1, x0:x1], gray[y0:y1, x0:x1]))
            rois.append(cur_roi)
        prev, i = gray, i + 1
    cap.release()
    return dict(signal=np.cumsum(vals), fs=float(fs), rois=rois, first_frame=first,
                n_no_roi=sum(r is None for r in rois))
