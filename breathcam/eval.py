"""Agreement metrics between an estimated and a reference rate."""

import numpy as np


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
