"""BreathCam demo (Streamlit): upload a video, get a breathing rate and a confidence.

    streamlit run app.py
"""

import io
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from breathcam.roi import DEFAULT_MODEL
from breathcam.run import diagnostic_figure, estimate

st.set_page_config(page_title="BreathCam", layout="wide")
st.warning("**Research prototype. Not a medical device.** Do not use it to make clinical decisions.")
st.title("BreathCam")
st.write("Estimates breathing rate (breaths/min) from the chest motion of a person sitting still and facing a fixed "
         "camera. Upload a 30-90 s clip. Prefer a face-blurred copy made with `python -m breathcam.deid`. "
         "Everything runs on this computer; the temporary copy of your video is deleted right after each step.")


@contextmanager
def temp_copy(upload):
    """Write the upload to a temporary file for OpenCV and delete it when the block ends."""
    f = tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False)
    try:
        f.write(upload.getvalue())
        f.close()
        yield Path(f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)


upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
if upload is None:
    st.info("Upload a video to start.")
    st.stop()

with temp_copy(upload) as path:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
if not ok:
    st.error("Could not read this video.")
    st.stop()
frame = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
h = frame.shape[0]

left, right = st.columns(2)
with left:
    st.subheader("1. Chest region")
    auto_ok = Path(DEFAULT_MODEL).exists()
    auto = st.checkbox("Find the chest automatically (MediaPipe)", value=auto_ok, disabled=not auto_ok)
    if not auto_ok:
        st.caption(f"Automatic mode needs the pose model at `{DEFAULT_MODEL}` (see README). Using a manual box.")
    x0, x1 = st.slider("Left / right edge (px)", 0, 320, (112, 208))
    y0, y1 = st.slider("Top / bottom edge (px)", 0, h, (int(0.43 * h), int(0.75 * h)))
    preview = frame.copy()
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 1)
    st.image(preview[..., ::-1], caption="Green box = region used when not automatic. Cover the chest, not the face or arms.",
             width=480)
with right:
    st.subheader("2. Settings")
    method = st.selectbox("Motion method", ["flow", "diff"], format_func=lambda m: {
        "flow": "Optical flow (default)", "diff": "Frame differencing (faster baseline)"}[m])
    min_quality = st.slider("Minimum quality to accept a window", 0.0, 1.0, 0.6, 0.05)
    go = st.button("Estimate breathing rate", type="primary")

if go:
    with st.spinner("Processing video..."), temp_copy(upload) as path:
        st.session_state.result = estimate(path, method, None if auto else (x0, y0, x1, y1),
                                           min_quality=min_quality)

r = st.session_state.get("result")
if r is not None:
    st.subheader("3. Result")
    c1, c2, c3 = st.columns(3)
    if np.isnan(r["rate_bpm"]):
        c1.metric("Breathing rate", "no reliable estimate")
        st.error("The signal was too noisy or too irregular to answer (motion, no visible breathing, or a poor "
                 "chest region). Refusing to guess is intended behaviour.")
    else:
        c1.metric("Breathing rate", f"{r['rate_bpm']:.1f} breaths/min")
    c2.metric("Whole-clip quality (0-1)", f"{r['quality_overall']:.2f}")
    c3.metric("Windows accepted", f"{int(r['accepted'].sum())} of {len(r['accepted'])}")
    buf = io.BytesIO()
    diagnostic_figure(r, buf)
    st.image(buf.getvalue(), caption="Diagnostic: signal, detected breaths, spectrum, per-window rate and quality")
    st.dataframe(pd.DataFrame({"window start (s)": r["t"].round(1), "breaths/min": r["rate"].round(1),
                               "quality": r["quality"].round(2), "accepted": r["accepted"]}), hide_index=True)
