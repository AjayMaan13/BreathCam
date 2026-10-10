from pathlib import Path

import pytest

from breathcam import db

QUERIES = (Path(__file__).resolve().parent.parent / "sql" / "eval_queries.sql").read_text()


def run_query(conn, n):
    """Run the n-th (1-based) statement of eval_queries.sql and return its rows."""
    stmts = [s for s in QUERIES.split(";") if "SELECT" in s]
    return conn.execute(stmts[n - 1]).fetchall()


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init(c)
    db.add_subject(c, "S01", "2026-10-10")
    sid = db.add_session(c, "S01", "clip.mov", paced_bpm=12, lighting="bright", camera_mount="tripod")
    db.add_reference(c, sid, 0, 100, 12, "metronome")
    # est 11 and 14 are 1 and 2 bpm off, 20 is 8 off, one window rejected (NaN)
    db.add_predictions(c, sid, "welch", [0, 5, 10, 15], [11, 14, 20, float("nan")], [0.9, 0.9, 0.8, 0.2])
    return c


def test_mae_by_lighting(conn):
    (lighting, method, source, n, mae), = run_query(conn, 1)
    assert (lighting, method, source, n) == ("bright", "welch", "metronome", 3)
    assert mae == pytest.approx((1 + 2 + 8) / 3, abs=0.01)


def test_within_2(conn):
    (_, _, n, pct), = run_query(conn, 2)
    assert n == 3 and pct == pytest.approx(66.7, abs=0.1)


def test_rejected_windows(conn):
    (_, _, windows, rejected), = run_query(conn, 3)
    assert (windows, rejected) == (4, 1)


def test_foreign_keys_enforced(conn):
    with pytest.raises(Exception):
        db.add_session(conn, "S99", "x.mov")
