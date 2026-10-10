"""Chest region of interest (ROI) from MediaPipe Pose landmarks."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

DEFAULT_MODEL = "data/models/pose_landmarker_lite.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


class ChestROI:
    """Tracks a chest rectangle across frames.

    Call update(frame) every few frames. It runs MediaPipe Pose, builds a
    rectangle from the shoulders down towards the hips, smooths it over time
    (exponential moving average), and keeps the last good rectangle whenever
    the landmarks are not reliably visible.
    """

    def __init__(self, model_path=DEFAULT_MODEL, min_visibility=0.5, smoothing=0.3):
        """Inputs:
            model_path: path to the pose_landmarker .task model file.
            min_visibility: landmark visibility (0..1) required to trust a detection.
            smoothing: weight of the newest detection in the moving average (0..1).
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"pose model not found: {model_path}\nDownload it with:\n"
                f"  mkdir -p data/models && curl -L -o {model_path} {MODEL_URL}\n"
                "or skip pose detection with --roi x0,y0,x1,y1")
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        self.min_visibility = min_visibility
        self.smoothing = smoothing
        self.box = None  # smoothed [x0, y0, x1, y1] in pixels

    def update(self, frame_bgr):
        """Detect the pose in one frame and update the smoothed ROI.

        Inputs:
            frame_bgr: HxWx3 uint8 image (OpenCV BGR order).
        Returns:
            (x0, y0, x1, y1) ints in pixels, or None if no good detection has
            been seen yet.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.pose_landmarks:
            new = self._box_from_landmarks(result.pose_landmarks[0], w, h)
            if new is not None:
                a = self.smoothing
                self.box = new if self.box is None else (1 - a) * self.box + a * new
        if self.box is None:
            return None
        x0, y0, x1, y1 = self.box.round().astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    def _box_from_landmarks(self, lm, w, h):
        """Chest box from pose landmarks, or None if the shoulders aren't visible."""
        sh = [lm[L_SHOULDER], lm[R_SHOULDER]]
        if min(p.visibility for p in sh) < self.min_visibility:
            return None
        xs = np.array([p.x for p in sh]) * w
        y_sh = np.mean([p.y for p in sh]) * h
        width = abs(xs[0] - xs[1])
        hips = [lm[L_HIP], lm[R_HIP]]
        if min(p.visibility for p in hips) >= self.min_visibility:
            torso = np.mean([p.y for p in hips]) * h - y_sh
        else:
            torso = 1.2 * width  # hips hidden (seated behind a table): estimate from shoulder width
        if torso <= 0:
            return None
        # chest = from the shoulder line down 60% of the torso, trimmed 10% at each side
        x0, x1 = xs.min() + 0.1 * width, xs.max() - 0.1 * width
        y0, y1 = y_sh, y_sh + 0.6 * torso
        box = np.clip([x0, y0, x1, y1], 0, [w, h, w, h])
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        return box
