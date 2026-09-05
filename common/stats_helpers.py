"""Statistics used by the reproduction notebooks (numpy / scipy only)."""
import numpy as np
from scipy import stats


def crawford_howell(case, control, tail="lower"):
    """Crawford & Howell (1998) modified t for a single case against a small control sample.

    Returns (t, p, d_cc) with d_cc = (case - mean) / sd = t * sqrt((n + 1) / n).
    tail: 'lower' (deficit), 'upper' (elevation) or 'two'.
    """
    control = np.asarray(control, float)
    control = control[~np.isnan(control)]
    n = len(control)
    m, sd = control.mean(), control.std(ddof=1)
    t = (case - m) / (sd * np.sqrt((n + 1) / n))
    if tail == "lower":
        p = stats.t.cdf(t, n - 1)
    elif tail == "upper":
        p = stats.t.sf(t, n - 1)
    else:
        p = 2 * stats.t.sf(abs(t), n - 1)
    return float(t), float(p), float((case - m) / sd)


def hedges_g(a, b):
    """Hedges' g for mean(a) - mean(b) with the small-sample correction J."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    g = (a.mean() - b.mean()) / sp
    return float(g * (1 - 3 / (4 * (n1 + n2) - 9)))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q) for a vector of p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def wilson_interval(k, n, z=1.959964):
    """Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)
