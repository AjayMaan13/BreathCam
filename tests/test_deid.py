import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from breathcam.deid import blur_boxes, deidentify, fill_boxes

BOX = (40, 30, 100, 90)


def test_fill_boxes_carries_detections_both_ways_within_hold():
    per = [[], [], [BOX], [], [], [], [], [], []]
    out = fill_boxes(per, hold=2)
    assert [bool(b) for b in out] == [True, True, True, True, True, False, False, False, False]


def test_fill_boxes_leaves_real_detections_alone():
    a, b = [(0, 0, 10, 10)], [(50, 50, 60, 60)]
    assert fill_boxes([a, [], b], hold=5) == [a, a, b]


def test_blur_boxes_blurs_only_inside_padded_box():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out = blur_boxes(img.copy(), [BOX])
    inside, outside = (slice(40, 80), slice(50, 90)), (slice(0, 5), slice(130, 160))  # padded box spans rows 6-114, cols 16-124
    assert out[inside].std() < 0.5 * img[inside].std()
    assert np.array_equal(out[outside], img[outside])


def _write_input(path, n=60):
    rng = np.random.default_rng(1)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for _ in range(n):
        w.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    w.release()


def test_deidentify_blurs_reports_coverage(tmp_path):
    _write_input(tmp_path / "in.avi")
    detect = lambda f: [BOX] if detect.i % 60 < 20 else []   # face visible only in the first 20 frames
    detect.i = -1
    def stub(f):
        detect.i += 1
        return detect(f)
    s = deidentify(tmp_path / "in.avi", tmp_path / "out.mp4", width=160, detect=stub, hold=10)
    assert (s["frames"], s["detected"], s["filled"], s["unprotected"]) == (60, 20, 10, 30)
    c = cv2.VideoCapture(str(tmp_path / "out.mp4")); c.read(); ok, frame = c.read()
    assert ok and frame[40:80, 50:90].std() < 40   # noise std is about 74 unblurred


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe")
def test_deidentify_strips_container_metadata(tmp_path):
    _write_input(tmp_path / "raw.avi")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "raw.avi"), "-c:v", "libx264",
                    "-metadata", "title=SECRETNAME", "-metadata", "location=+12.3456-065.4321/",
                    "-metadata", "creation_time=2026-10-10T17:30:18Z", str(tmp_path / "in.mp4")], check=True)
    before = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(tmp_path / "in.mp4")],
                            capture_output=True, text=True).stdout
    assert "SECRETNAME" in before and "+12.3456" in before
    deidentify(tmp_path / "in.mp4", tmp_path / "out.mp4", width=160, detect=lambda f: [])
    after = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json",
                            str(tmp_path / "out.mp4")], capture_output=True, text=True).stdout
    assert "SECRETNAME" not in after and "12.3456" not in after and "creation_time" not in after
    tags = json.loads(after)
    assert not any(k.lower() in ("title", "location", "creation_time", "comment")
                   for k in tags.get("format", {}).get("tags", {}))
