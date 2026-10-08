"""Drift detectors.

Supervised (need the true label of every stream sample — in production labels often arrive late):
    DDM (Gama et al., 2004), ADWIN on the 0/1 error (Bifet & Gavalda, 2007), Page-Hinkley on the error
    (Page, 1954) — all from River.

Label-free (only need x):
    PH / ADWIN on a per-sample uncertainty signal (the UDD idea of Baier et al., 2021, generalised to
    every component of the decomposition), and two detectors implemented here from first principles:

    * BOCPD — Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with a Normal-Inverse-Gamma
      conjugate prior on batch means of the signal. The posterior over the *run length* r_t (time since
      the last change) is updated exactly; an alarm is raised when P(r_t < r_min | data) is high.
    * EM-GMM — a Gaussian mixture fitted by Expectation-Maximisation (written from scratch, K chosen by
      BIC) on the joint (log AU, log EU) of the calibration data. Drift score = average negative
      log-likelihood of a batch under that reference density.
    * KS — feature-wise two-sample Kolmogorov-Smirnov tests on raw inputs (Bonferroni corrected):
      the usual input-space baseline (Rabanser et al., 2019).

Fair comparison: every signal is z-scored with calibration statistics and each detector's threshold is
calibrated on bootstrap streams drawn from the calibration set so that the probability of *any* false
alarm on a stationary stream of the same length is <= ``target_far`` (default 10 %).
"""
from __future__ import annotations

import numpy as np
from river import drift as rdrift
from scipy import stats as sst
from scipy.special import gammaln, logsumexp


# ----------------------------------------------------------------------------- utilities
def _first_after(alarms, t0):
    if t0 is None:
        return None
    a = [t for t in alarms if t >= t0]
    return a[0] if a else None


def run_river(det_factory, xs):
    det = det_factory()
    alarms = []
    for t, v in enumerate(xs):
        det.update(float(v))
        if det.drift_detected:
            alarms.append(t)
    return alarms


def batch_means(x, b):
    n = len(x) // b
    return x[: n * b].reshape(n, b).mean(1)


# ----------------------------------------------------------------------------- BOCPD (own implementation)
class BOCPD:
    """Adams & MacKay (2007). Gaussian observations with unknown mean and variance.

    Prior: mu | sigma^2 ~ N(mu0, sigma^2/kappa0),  sigma^2 ~ InvGamma(alpha0, beta0).
    Predictive for a run of length r: Student-t(2 alpha_r, mu_r, beta_r (kappa_r+1) / (alpha_r kappa_r)).
    Hazard H = 1/lambda (memoryless prior on segment length).
    """

    def __init__(self, mu0=0.0, kappa0=1.0, alpha0=5.0, beta0=5.0, hazard=1 / 100, r_max=400):
        self.p0 = (mu0, kappa0, alpha0, beta0)
        self.H, self.r_max = hazard, r_max
        self.logR = np.array([0.0])
        self.mu, self.kappa = np.array([mu0]), np.array([kappa0])
        self.alpha, self.beta = np.array([alpha0]), np.array([beta0])

    @staticmethod
    def _log_student(x, mu, kappa, alpha, beta):
        nu = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        z = (x - mu) ** 2 / scale2
        return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
                - (nu + 1) / 2 * np.log1p(z / nu))

    def update(self, x):
        lp = self._log_student(x, self.mu, self.kappa, self.alpha, self.beta)
        log_growth = self.logR + lp + np.log1p(-self.H)
        log_cp = logsumexp(self.logR + lp + np.log(self.H))
        logR = np.append(log_cp, log_growth)
        logR -= logsumexp(logR)
        mu0, k0, a0, b0 = self.p0
        mu = np.append(mu0, (self.kappa * self.mu + x) / (self.kappa + 1))
        beta = np.append(b0, self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1)))
        kappa = np.append(k0, self.kappa + 1)
        alpha = np.append(a0, self.alpha + 0.5)
        if len(logR) > self.r_max:  # truncate, folding the tail mass into the last cell
            logR[self.r_max - 1] = logsumexp(logR[self.r_max - 1:])
            logR, mu, kappa = logR[: self.r_max], mu[: self.r_max], kappa[: self.r_max]
            alpha, beta = alpha[: self.r_max], beta[: self.r_max]
        self.logR, self.mu, self.kappa, self.alpha, self.beta = logR, mu, kappa, alpha, beta
        return np.exp(logR)

    def run(self, xs, r_min=4, thr=0.5, refractory=10, warmup=5):
        """Return (alarm indices, P(r_t < r_min) trace)."""
        alarms, trace, last = [], [], -10 ** 9
        for t, x in enumerate(xs):
            R = self.update(x)
            pc = R[:r_min].sum()
            trace.append(pc)
            if t >= warmup and pc > thr and t - last >= refractory:
                alarms.append(t)
                last = t
        return alarms, np.array(trace)


def bocpd_alarms(z_signal, b, prior_var, thr, hazard=1 / 100):
    bm = batch_means(z_signal, b)
    det = BOCPD(mu0=0.0, kappa0=1.0, alpha0=5.0, beta0=5.0 * prior_var, hazard=hazard)
    al, tr = det.run(bm, thr=thr)
    return [(a + 1) * b - 1 for a in al], tr


# ----------------------------------------------------------------------------- EM for Gaussian mixtures (own)
class GaussianMixtureEM:
    """Maximum-likelihood Gaussian mixture via Expectation-Maximisation (Dempster, Laird & Rubin, 1977).

    E-step: responsibilities  r_ik = pi_k N(x_i|mu_k,S_k) / sum_j pi_j N(x_i|mu_j,S_j)
    M-step: pi_k = mean_i r_ik ; mu_k = sum r_ik x_i / sum r_ik ; S_k = weighted covariance (+ ridge)
    The log-likelihood is guaranteed not to decrease between iterations.
    """

    def __init__(self, K=2, max_iter=300, tol=1e-6, reg=1e-4, seed=0):
        self.K, self.max_iter, self.tol, self.reg, self.seed = K, max_iter, tol, reg, seed

    def _logpdf(self, X):
        n, d = X.shape
        out = np.empty((n, self.K))
        for k in range(self.K):
            L = np.linalg.cholesky(self.S[k])
            diff = np.linalg.solve(L, (X - self.mu[k]).T)
            out[:, k] = (-0.5 * (diff ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)
                         + np.log(self.pi[k]))
        return out

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        # k-means++ style initialisation
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.K):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], 0)
            centers.append(X[rng.choice(n, p=d2 / d2.sum())])
        self.mu = np.array(centers)
        self.S = np.array([np.cov(X.T) + self.reg * np.eye(d)] * self.K)
        self.pi = np.full(self.K, 1 / self.K)
        self.history = []
        prev = -np.inf
        for _ in range(self.max_iter):
            lp = self._logpdf(X)                                   # E-step
            ll = logsumexp(lp, axis=1)
            r = np.exp(lp - ll[:, None])
            total = ll.sum()
            self.history.append(total)
            if total - prev < self.tol * abs(total):
                break
            prev = total
            Nk = r.sum(0) + 1e-12                                   # M-step
            self.pi = Nk / n
            self.mu = (r.T @ X) / Nk[:, None]
            for k in range(self.K):
                diff = X - self.mu[k]
                self.S[k] = (r[:, k, None] * diff).T @ diff / Nk[k] + self.reg * np.eye(d)
        self.loglik = self.history[-1]
        p = self.K * (d + d * (d + 1) / 2) + self.K - 1
        self.bic = -2 * self.loglik + p * np.log(n)
        return self

    def score_samples(self, X):
        return logsumexp(self._logpdf(X), axis=1)


def fit_gmm_bic(X, Ks=(1, 2, 3, 4), seed=0):
    models = [GaussianMixtureEM(K=k, seed=seed).fit(X) for k in Ks]
    return min(models, key=lambda m: m.bic)


def em_alarms(F_cal, F_stream, b, seed, q=0.99, consecutive=2):
    half = len(F_cal) // 2
    gmm = fit_gmm_bic(F_cal[:half], seed=seed)
    rng = np.random.default_rng(seed)
    nll_cal = -gmm.score_samples(F_cal[half:])
    boot = np.array([nll_cal[rng.integers(len(nll_cal), size=b)].mean() for _ in range(2000)])
    thr = np.quantile(boot, q)
    scores = batch_means(-gmm.score_samples(F_stream), b)
    alarms, run = [], 0
    for i, s in enumerate(scores):
        run = run + 1 if s > thr else 0
        if run == consecutive:
            alarms.append((i + 1) * b - 1)
            run = 0
    return alarms, scores, thr, gmm


# ----------------------------------------------------------------------------- KS input-space baseline
def ks_alarms(X_ref, X_stream, window=200, step=50, alpha=0.01, consecutive=2):
    d = X_ref.shape[1]
    alarms, run, pmins = [], 0, []
    for end in range(window, len(X_stream) + 1, step):
        W = X_stream[end - window:end]
        pv = min(sst.ks_2samp(X_ref[:, j], W[:, j]).pvalue for j in range(d)
                 if X_ref[:, j].std() > 0 or W[:, j].std() > 0) if d else 1.0
        pmins.append(pv)
        run = run + 1 if pv < alpha / d else 0
        if run == consecutive:
            alarms.append(end - 1)
            run = 0
    return alarms, np.array(pmins)


# ----------------------------------------------------------------------------- calibration
def calibrate(run_fn, grid, cal_values, length, rng, reps=10, target_far=0.1):
    """Smallest-sensitivity-loss threshold: first value in ``grid`` (ordered from most to least
    sensitive) whose false-alarm probability on bootstrap stationary streams is <= target_far."""
    streams = [cal_values[rng.integers(len(cal_values), size=length)] for _ in range(reps)]
    for g in grid:
        fa = np.mean([len(run_fn(g, s)) > 0 for s in streams])
        if fa <= target_far:
            return g
    return grid[-1]


PH_GRID = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]
ADWIN_GRID = [2e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14, 1e-20]
BOCPD_GRID = [0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 0.999]


def ph_up(lam):
    return lambda: rdrift.PageHinkley(min_instances=30, delta=0.05, threshold=lam, mode="up")


def adwin(delta):
    return lambda: rdrift.ADWIN(delta=delta)


# ----------------------------------------------------------------------------- run everything
def run_all_detectors(sig_stream: dict, sig_cal: dict, err_stream, err_cal, X_cal, X_stream, cfg, seed,
                      signals=("conf", "tu", "au", "eu", "lr")):
    """Return {detector_name: sorted list of alarm indices} and auxiliary traces."""
    rng = np.random.default_rng(seed + 12345)
    dc = cfg.get("detectors", {})
    b = dc.get("batch", 50)
    target = dc.get("target_far", 0.1)
    calib_len = dc.get("calib_len") or len(err_stream)   # default: same length as the monitored stream
    reps = dc.get("calib_reps", 10)
    label_delay = dc.get("label_delay", 0)
    out, aux = {}, {}

    # ---- supervised detectors on the 0/1 error stream
    err_stream = np.asarray(err_stream, float)
    shift = lambda al: [a + label_delay for a in al if a + label_delay < len(err_stream)]
    out["DDM[err]"] = shift(run_river(lambda: rdrift.binary.DDM(), err_stream))
    out["ADWIN[err]"] = shift(run_river(lambda: rdrift.ADWIN(delta=0.002), err_stream))
    lam = calibrate(lambda g, s: run_river(ph_up(g), s), PH_GRID, np.asarray(err_cal, float), calib_len, rng, reps,
                    target)
    out["PH[err]"] = shift(run_river(ph_up(lam), err_stream))
    aux["thr_PH[err]"] = lam

    # ---- label-free detectors on uncertainty signals
    for s in signals:
        mu, sd = sig_cal[s].mean(), sig_cal[s].std() + 1e-12
        zc, zs = (sig_cal[s] - mu) / sd, (sig_stream[s] - mu) / sd
        aux[f"z_{s}"] = zs
        lam = calibrate(lambda g, x: run_river(ph_up(g), x), PH_GRID, zc, calib_len, rng, reps, target)
        out[f"PH[{s}]"] = run_river(ph_up(lam), zs)
        dl = calibrate(lambda g, x: run_river(adwin(g), x), ADWIN_GRID, zc, calib_len, rng, reps, target)
        out[f"ADWIN[{s}]"] = run_river(adwin(dl), zs)
        bm_var = np.var([zc[rng.integers(len(zc), size=b)].mean() for _ in range(500)])
        thr = calibrate(lambda g, x: bocpd_alarms(x, b, bm_var, g)[0], BOCPD_GRID, zc, calib_len, rng,
                        max(3, reps // 2), target)
        out[f"BOCPD[{s}]"], aux[f"bocpd_trace_{s}"] = bocpd_alarms(zs, b, bm_var, thr)
        aux[f"thr_PH[{s}]"], aux[f"thr_ADWIN[{s}]"], aux[f"thr_BOCPD[{s}]"] = lam, dl, thr

    # ---- EM-GMM on the joint (log AU, log EU)
    feat = lambda S: np.column_stack([np.log(S["au"] + 1e-4), np.log(S["eu"] + 1e-4)])
    al, scores, thr, gmm = em_alarms(feat(sig_cal), feat(sig_stream), b, seed)
    out["EM-GMM[au,eu]"] = al
    aux["em_scores"], aux["em_thr"], aux["em_K"] = scores, thr, gmm.K

    # ---- input-space KS
    al, pm = ks_alarms(X_cal, X_stream, window=dc.get("ks_window", 200), step=b)
    out["KS[x]"] = al
    aux["ks_pmin"] = pm
    return out, aux


SUPERVISED = ["DDM[err]", "ADWIN[err]", "PH[err]"]


def detector_family(name):
    if name in SUPERVISED:
        return "supervised (labels)"
    if name.startswith("KS"):
        return "input space"
    return "uncertainty (label-free)"
