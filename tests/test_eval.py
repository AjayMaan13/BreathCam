import numpy as np
import pandas as pd
import pytest

from breathcam import db
from breathcam.eval import bootstrap_ci, load_windows, mae, rejection_rate, summarize


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init(c)
    db.add_subject(c, "S01", "2026-10-10")
    a = db.add_session(c, "S01", "a.mov", paced_bpm=12, lighting="bright")
    b = db.add_session(c, "S01", "b.mov")  # no reference
    db.add_reference(c, a, 0, 100, 12, "metronome")
    nan = float("nan")
    db.add_predictions(c, a, "welch", [0, 5, 10], [11, 14, nan], [0.9, 0.9, 0.2])
    db.add_predictions(c, b, "welch", [0, 5], [15, 16], [0.9, 0.9])
    return c


def test_load_windows_joins_reference(conn):
    df = load_windows(conn)
    assert len(df) == 5
    assert df.ref_bpm.notna().sum() == 3 and df[df.session_id == 2].ref_bpm.isna().all()


def test_summarize_scores_only_accepted_with_reference(conn):
    t = summarize(load_windows(conn), ["method"])
    row = t.loc["welch"]
    assert row.n == 2 and row.mae == pytest.approx(1.5) and row.within_2 == 1.0 and row.bias == pytest.approx(0.5)


def test_summarize_keeps_missing_group_values(conn):
    t = summarize(load_windows(conn), ["lighting", "method"])
    assert len(t) == 1  # session b has no reference, so only the 'bright' group is scored


def test_rejection_rate_counts_windows_without_reference(conn):
    r = rejection_rate(load_windows(conn), ["method"]).loc["welch"]
    assert (r.n, r.rejected) == (5, 1)


def _clustered(n_subjects, rng):
    rows = []
    for s in range(n_subjects):
        bias = rng.normal(0, 2)  # each subject has its own offset
        for _ in range(20):
            ref = 15.0
            rows.append(dict(subject_id=f"S{s:02d}", est_bpm=ref + bias + rng.normal(0, 0.5), ref_bpm=ref))
    return pd.DataFrame(rows)


def test_bootstrap_ci_refuses_with_few_clusters():
    point, lo, hi = bootstrap_ci(_clustered(3, np.random.default_rng(0)))
    assert not np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_brackets_point_estimate():
    df = _clustered(12, np.random.default_rng(1))
    point, lo, hi = bootstrap_ci(df, mae)
    assert lo <= point <= hi and hi > lo


def test_bootstrap_by_subject_wider_than_by_window():
    # between-subject offsets dominate, so resampling subjects must give a wider interval
    df = _clustered(12, np.random.default_rng(2)).assign(win=lambda d: range(len(d)))
    _, lo_s, hi_s = bootstrap_ci(df, mae, cluster="subject_id", min_clusters=5)
    _, lo_w, hi_w = bootstrap_ci(df, mae, cluster="win", min_clusters=5)
    assert (hi_s - lo_s) > (hi_w - lo_w)


def test_summarize_group_with_missing_key_keeps_values():
    df = pd.DataFrame(dict(lighting=[None, None, "dim"], method="welch",
                           est_bpm=[11.0, 13.0, 12.0], ref_bpm=12.0))
    t = summarize(df, ["lighting", "method"])
    assert len(t) == 2 and t.n.tolist().count(2) == 1 and t.mae.notna().all()
