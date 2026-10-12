from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_with_disclaimer_and_waits_for_upload():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Not a medical device" in w.value for w in at.warning)
    assert any("Upload a video" in i.value for i in at.info)
