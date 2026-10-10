"""Agreement metrics between an estimated and a reference rate."""

import numpy as np
import pandas as pd


def _clean(est, ref):
    """Drop pairs where either value is NaN.

    Inputs:
        est, ref: array-likes of equal length (estimated and reference rates).
    Returns:
        (est, ref): float arrays containing only the pairs with no NaN.
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = ~(np.isnan(est) | np.isnan(ref))
    return est[ok], ref[ok]


def mae(est, ref):
    """Mean absolute error: the average size of the miss, in the input units.

    Inputs:
        est, ref: equal-length arrays of estimated and reference rates (breaths/min).
            NaN pairs are ignored.
    Returns:
        float, mean(|est - ref|).
    """
    est, ref = _clean(est, ref)
    return float(np.mean(np.abs(est - ref)))


def rmse(est, ref):
    """Root mean squared error; like MAE but punishes large misses more.

    Inputs:
        est, ref: equal-length arrays of estimated and reference rates (breaths/min).
            NaN pairs are ignored.
    Returns:
        float, sqrt(mean((est - ref)^2)).
    """
    est, ref = _clean(est, ref)
    return float(np.sqrt(np.mean((est - ref) ** 2)))


def within(est, ref, tol=2.0):
    """Share of readings that are close enough to the reference.

    Inputs:
        est, ref: equal-length arrays of estimated and reference rates (breaths/min).
            NaN pairs are ignored.
        tol: allowed absolute error (default +/-2 breaths/min).
    Returns:
        float in [0, 1], the fraction of pairs with |est - ref| <= tol.
    """
    est, ref = _clean(est, ref)
    return float(np.mean(np.abs(est - ref) <= tol))


def bland_altman(est, ref):
    """Compute Bland-Altman agreement statistics between two methods.

    Inputs:
        est, ref: equal-length arrays of the new method's and the reference
            method's rates (breaths/min). NaN pairs are ignored.
    Returns:
        (mean, diff, bias, lower_loa, upper_loa):
        mean: per-pair average of the two methods (x-axis of the plot),
        diff: per-pair est - ref (y-axis of the plot),
        bias: average difference (systematic offset),
        lower_loa, upper_loa: limits of agreement, bias -/+ 1.96 SD of the
            differences (the range holding ~95% of differences).
    """
    est, ref = _clean(est, ref)
    diff = est - ref
    bias, sd = diff.mean(), diff.std(ddof=1)
    return (est + ref) / 2, diff, float(bias), float(bias - 1.96 * sd), float(bias + 1.96 * sd)


def plot_bland_altman(est, ref, ax, title=""):
    """Draw a Bland-Altman plot onto an existing matplotlib axis.

    Plots est - ref against the mean of the two, with horizontal lines at the
    bias (solid) and the limits of agreement (dashed), each labelled with its value.

    Inputs:
        est, ref: equal-length arrays of estimated and reference rates (breaths/min).
        ax: matplotlib Axes to draw on.
        title: optional plot title.
    Returns:
        None (draws on ax).
    """
    mean, diff, bias, lo, hi = bland_altman(est, ref)
    ax.scatter(mean, diff, s=6, alpha=0.4)
    for y, ls in [(bias, "-"), (lo, "--"), (hi, "--")]:
        ax.axhline(y, color="k", ls=ls, lw=1)
        ax.text(ax.get_xlim()[1], y, f" {y:+.1f}", va="center", fontsize=8)
    ax.set_xlabel("mean of estimate and reference (breaths/min)")
    ax.set_ylabel("estimate - reference (breaths/min)")
    ax.set_title(title)


# ---- evaluation from the SQLite database ----

def load_windows(conn):
    """Read every prediction joined to its session conditions and reference.

    A prediction is matched to the reference reading covering its window start.
    Windows with no reference keep ref_bpm = NaN (they still count for the
    rejection rate). If a window has references from two sources it appears
    once per source, so always group by ref_source when both exist.

    Inputs:
        conn: sqlite3.Connection to the BreathCam database.
    Returns:
        DataFrame, one row per (window, method[, reference]), with columns
        session_id, subject_id, video_path, paced_bpm, lighting, clothing,
        distance_m, camera_mount, window_start, method, est_bpm (NaN =
        rejected), quality_score, ref_bpm, ref_source.
    """
    return pd.read_sql_query(
        """SELECT p.session_id, s.subject_id, s.video_path, s.paced_bpm, s.lighting,
                  s.clothing, s.distance_m, s.camera_mount,
                  p.window_start, p.method, p.est_bpm, p.quality_score,
                  r.ref_bpm, r.ref_source
           FROM predictions p
           JOIN sessions s ON s.session_id = p.session_id
           LEFT JOIN reference_readings r
             ON r.session_id = p.session_id
            AND p.window_start >= r.t_start AND p.window_start < r.t_end""", conn)


def summarize(df, by):
    """Accuracy table: one row per group of `by` columns.

    Only windows that were accepted by the quality gate and have a reference
    are scored. Groups with a missing value (e.g. unknown lighting) are kept.

    Inputs:
        df: DataFrame from load_windows.
        by: list of column names to group by, e.g. ["method", "ref_source"].
    Returns:
        DataFrame indexed by `by` with n (scored windows), mae, rmse,
        within_2 (fraction within +/-2 bpm) and bias (mean est - ref).
    """
    scored = df[df.est_bpm.notna() & df.ref_bpm.notna()]
    keys, rows = [], []
    for key, g in scored.groupby(by, dropna=False):
        keys.append(key if len(by) > 1 else (key[0] if isinstance(key, tuple) else key))
        rows.append(dict(n=len(g), mae=mae(g.est_bpm, g.ref_bpm), rmse=rmse(g.est_bpm, g.ref_bpm),
                         within_2=within(g.est_bpm, g.ref_bpm), bias=float((g.est_bpm - g.ref_bpm).mean())))
    if not rows:
        return pd.DataFrame(columns=["n", "mae", "rmse", "within_2", "bias"])
    index = pd.MultiIndex.from_tuples(keys, names=by) if len(by) > 1 else pd.Index(keys, name=by[0])
    out = pd.DataFrame(rows, index=index)
    return out


def rejection_rate(df, by):
    """Share of windows rejected by the quality gate, per group.

    Inputs:
        df: DataFrame from load_windows (windows without a reference count too).
        by: list of column names to group by.
    Returns:
        DataFrame with n (windows), rejected (count) and rejected_frac.
    """
    g = df.groupby(by, dropna=False).est_bpm
    return pd.DataFrame({"n": g.size(), "rejected": g.apply(lambda s: int(s.isna().sum()))}).assign(
        rejected_frac=lambda t: t.rejected / t.n)


def bootstrap_ci(df, metric=mae, cluster="subject_id", n_boot=2000, seed=0, min_clusters=5):
    """Bootstrap confidence interval for a metric, resampling whole clusters.

    Windows from one subject (or one clip) are strongly correlated, so
    resampling windows would make the interval far too narrow. Instead whole
    clusters (default: subjects) are resampled with replacement. With fewer
    than `min_clusters` clusters the interval is not meaningful, so NaN is
    returned instead of a misleading number.

    Inputs:
        df: DataFrame from load_windows, already filtered to one method and
            one ref_source.
        metric: function (est, ref) -> float, e.g. mae or rmse.
        cluster: column to resample by ("subject_id", or "session_id" as a weaker fallback).
        n_boot: number of resamples.
        seed: random seed (reproducible).
        min_clusters: minimum number of clusters needed to report an interval.
    Returns:
        (point_estimate, ci_low, ci_high); the CI is (nan, nan) if too few clusters.
    """
    scored = df[df.est_bpm.notna() & df.ref_bpm.notna()]
    groups = [(g.est_bpm.to_numpy(), g.ref_bpm.to_numpy()) for _, g in scored.groupby(cluster)]
    point = metric(scored.est_bpm, scored.ref_bpm)
    if len(groups) < min_clusters:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        vals.append(metric(np.concatenate([e for e, _ in pick]), np.concatenate([r for _, r in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)
