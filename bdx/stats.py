"""Statistical inference used in the paper.

Stream observations are *not* independent (autocorrelated signals, overlapping windows), so:
* for per-run summary statistics (one number per independent run/seed) we use classical paired tests
  (Wilcoxon signed-rank, sign test) and percentile bootstrap CIs — runs with different seeds ARE independent;
* for statistics computed *inside* one stream (cross-correlation on Elec2) we use the moving-block
  bootstrap (Kuensch, 1989), which keeps the dependence structure inside blocks;
* results from several datasets are pooled with a DerSimonian-Laird random-effects meta-analysis,
  reporting the between-dataset heterogeneity tau^2 and I^2 (Higgins & Thompson, 2002).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sst


def bootstrap_ci(x, stat=np.median, B=5000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([stat(x[rng.integers(len(x), size=len(x))]) for _ in range(B)])
    return float(stat(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilcoxon_greater(x, mu=0.0):
    """One-sided H1: median(x) > mu. Returns (statistic, p). Zeros are dropped (Wilcox)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)] - mu
    x = x[x != 0]
    if len(x) < 3:
        return np.nan, np.nan
    r = sst.wilcoxon(x, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def paired_wilcoxon(a, b, alternative="less"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    d = d[d != 0]
    if len(d) < 3:
        return np.nan, np.nan, int(m.sum())
    r = sst.wilcoxon(d, alternative=alternative)
    return float(r.statistic), float(r.pvalue), int(m.sum())


def binomial_greater(successes, n, p0):
    if n == 0:
        return np.nan
    return float(sst.binomtest(int(successes), int(n), p0, alternative="greater").pvalue)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def dersimonian_laird(effects, ses):
    """Random-effects pooled estimate. effects/ses: one per study (dataset)."""
    y, s = np.asarray(effects, float), np.asarray(ses, float)
    m = ~(np.isnan(y) | np.isnan(s) | (s <= 0))
    y, s = y[m], s[m]
    k = len(y)
    if k == 0:
        return {}
    w = 1 / s ** 2
    fe = (w * y).sum() / w.sum()
    Q = (w * (y - fe) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (s ** 2 + tau2)
    re = (ws * y).sum() / ws.sum()
    se = np.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 and k > 1 else 0.0
    z = re / se
    return {"pooled": float(re), "se": float(se), "ci_low": float(re - 1.96 * se), "ci_high": float(re + 1.96 * se),
            "z": float(z), "p_two_sided": float(2 * sst.norm.sf(abs(z))), "tau2": float(tau2), "I2": float(I2),
            "Q": float(Q), "k": int(k), "weights": (ws / ws.sum()).tolist()}


def moving_block_bootstrap_indices(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def lagged_corr(x, y, lag):
    """corr(x_t, y_{t+lag}); positive lag = x leads y."""
    if lag > 0:
        a, b = x[:-lag], y[lag:]
    elif lag < 0:
        a, b = x[-lag:], y[:lag]
    else:
        a, b = x, y
    return float(np.corrcoef(a, b)[0, 1])


def cross_correlation_block_bootstrap(x, y, lags, block=20, B=1000, seed=0):
    """Lead-lag cross-correlation with moving-block bootstrap CIs (pairs resampled jointly)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    est = np.array([lagged_corr(x, y, l) for l in lags])
    n = len(x)
    boots = np.empty((B, len(lags)))
    for bi in range(B):
        # resample blocks of the *lag-aligned* pairs to keep cross-dependence
        for j, l in enumerate(lags):
            if l >= 0:
                a, b = x[: n - l], y[l:]
            else:
                a, b = x[-l:], y[: n + l]
            idx = moving_block_bootstrap_indices(len(a), block, rng)
            boots[bi, j] = np.corrcoef(a[idx], b[idx])[0, 1]
    return est, np.quantile(boots, 0.025, 0), np.quantile(boots, 0.975, 0)


def granger(x, y, maxlag=5):
    """Does x Granger-cause y? Returns min p-value over lags (SSR F-test) and per-lag p-values."""
    import contextlib
    import io
    import warnings

    from statsmodels.tsa.stattools import grangercausalitytests
    data = np.column_stack([y, x])  # statsmodels tests whether column 2 causes column 1
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        except TypeError:          # statsmodels >= 0.15 removed the `verbose` argument
            res = grangercausalitytests(data, maxlag=maxlag)
    ps = {lag: float(r[0]["ssr_ftest"][1]) for lag, r in res.items()}
    return min(ps.values()), ps
