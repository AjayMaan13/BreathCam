import cv2
import numpy as np
import pytest

from breathcam.run import estimate

FS, W, H = 30, 320, 240
ROI = (60, 60, 260, 180)


def write_video(path, bpm, dur=40, amp=1.5, hold=None, noise=2.0):
    """Textured scene whose vertical position oscillates at `bpm`.
    hold=(t0, t1): motion frozen in that interval (breath-hold)."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3)
    tex = (255 * (tex - tex.min()) / np.ptp(tex)).astype(np.float32)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FS, (W, H))
    phase = 0.0
    for i in range(dur * FS):
        t = i / FS
        if not (hold and hold[0] <= t < hold[1]):
            phase += 2 * np.pi * bpm / 60 / FS
        dy = amp * np.sin(phase)
        frame = cv2.warpAffine(tex, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        frame = np.clip(frame + noise * rng.standard_normal(frame.shape), 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()


@pytest.mark.parametrize("method", ["flow", "diff"])
@pytest.mark.parametrize("bpm", [12, 20])
def test_rate_recovered(tmp_path, method, bpm):
    write_video(tmp_path / "v.avi", bpm)
    r = estimate(tmp_path / "v.avi", method=method, roi=ROI)
    assert r["rate_bpm"] == pytest.approx(bpm, abs=1.5)


def test_no_motion_is_rejected(tmp_path):
    write_video(tmp_path / "v.avi", 15, amp=0.0)
    r = estimate(tmp_path / "v.avi", roi=ROI)
    assert np.isnan(r["rate_bpm"])


def test_breath_hold_lowers_quality(tmp_path):
    write_video(tmp_path / "a.avi", 15)
    write_video(tmp_path / "b.avi", 15, hold=(5, 35))
    assert estimate(tmp_path / "b.avi", roi=ROI)["quality"].max() < estimate(tmp_path / "a.avi", roi=ROI)["quality"].min()
