"""Unit tests for the parts implemented from first principles. Run: pytest -q"""
import numpy as np
import pytest

from bdx import detectors as DET
from bdx import stats as S
from bdx.uncertainty import decompose


def test_decomposition_identity_and_bounds():
    rng = np.random.default_rng(0)
    samples = rng.dirichlet([1, 1, 1], size=(20, 50))          # (S, N, C)
    d = decompose(samples)
    assert np.all(d["eu"] >= 0)                                 # Jensen
    assert np.allclose(d["tu"], d["au"] + d["eu"], atol=1e-9)   # TU = AU + EU
    assert np.all(d["tu"] <= np.log(3) + 1e-9)


def test_epistemic_zero_when_models_agree():
    p = np.tile(np.array([[0.3, 0.7]]), (10, 5, 1))
    d = decompose(p)
    assert np.allclose(d["eu"], 0, atol=1e-9)
    assert np.allclose(d["au"], -(0.3 * np.log(0.3) + 0.7 * np.log(0.7)))


def test_epistemic_max_when_models_disagree_confidently():
    p = np.array([[[1 - 1e-9, 1e-9]], [[1e-9, 1 - 1e-9]]])
    d = decompose(p)
    assert d["au"][0] < 1e-6 and abs(d["eu"][0] - np.log(2)) < 1e-6


def test_em_loglik_monotone_and_recovers_means():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(-3, 1, (400, 2)), rng.normal(3, 1, (400, 2))]
    g = DET.GaussianMixtureEM(K=2, seed=0).fit(X)
    h = np.array(g.history)
    assert np.all(np.diff(h) >= -1e-6)
    assert np.allclose(sorted(g.mu[:, 0]), [-3, 3], atol=0.3)
    assert DET.fit_gmm_bic(X).K == 2


def test_bocpd_detects_mean_shift():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    al, _ = DET.BOCPD(beta0=5.0).run(x)
    assert any(100 <= a <= 110 for a in al)
    assert not any(a < 95 for a in al)


def test_dersimonian_laird_homogeneous():
    r = S.dersimonian_laird([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])
    assert abs(r["pooled"] - 1.0) < 1e-9 and r["tau2"] == 0.0


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


@pytest.mark.slow
def test_end_to_end_small():
    import yaml
    from bdx.experiment import run_one
    cfg = yaml.safe_load(open("config/default.yaml"))
    cfg.update(n_train=500, n_cal=300, stream_len=1500, t0=600, width=300)
    cfg["model"]["epochs"] = 5
    out = run_one("digits", "covariate", "mcd", 0, cfg)
    assert len(out["rows"]) == 20 and len(out["signal_rows"]) == 5
