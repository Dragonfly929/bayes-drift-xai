"""Information-theoretic decomposition of predictive uncertainty (Depeweg et al., 2018; Houlsby et al., 2011).

Given S posterior draws p_s(y|x) = p(y|x, theta_s):

    total      TU(x) = H[ (1/S) sum_s p_s ]                     entropy of the predictive mean
    aleatoric  AU(x) = (1/S) sum_s H[ p_s ]                     expected entropy  (noise in the data)
    epistemic  EU(x) = TU(x) - AU(x) = I(y; theta | x, D)       mutual information (model ignorance)

EU >= 0 by Jensen's inequality (entropy is concave). EU is large when the sampled models *disagree*,
which happens far from the training data -> it reacts to changes in P(x). AU is large where every
model agrees the classes overlap. The drift signal proposed here is the log-ratio

    LR(x) = log(EU + eps) - log(AU + eps)

which rises when ignorance grows *relative to* irreducible noise; a pure shift in class overlap moves AU
and EU together and leaves LR roughly unchanged, so LR is less confounded than EU or TU alone.
"""
from __future__ import annotations

import numpy as np

EPS_LOG = 1e-12
SIGNALS = ["conf", "tu", "au", "eu", "lr"]
SIGNAL_LABELS = {
    "conf": "1 - max prob (MSP baseline)",
    "tu": "Total entropy TU",
    "au": "Aleatoric AU",
    "eu": "Epistemic EU (mutual information)",
    "lr": "log(EU/AU) ratio  [proposed]",
}


def entropy(p, axis=-1):
    return -(p * np.log(p + EPS_LOG)).sum(axis)


def decompose(samples: np.ndarray, ratio_eps: float = 1e-4) -> dict:
    """samples: (S, N, C) -> dict of per-observation arrays (N,)."""
    mean = samples.mean(0)
    tu = entropy(mean)
    au = entropy(samples).mean(0)
    eu = np.clip(tu - au, 0.0, None)
    return {
        "mean": mean,
        "pred": mean.argmax(-1),
        "conf": 1.0 - mean.max(-1),
        "tu": tu,
        "au": au,
        "eu": eu,
        "lr": np.log(eu + ratio_eps) - np.log(au + ratio_eps),
    }


def variance_decomposition(samples: np.ndarray) -> dict:
    """Alternative (law of total variance) decomposition for the positive class, for reference:
    Var[y] = E_theta[p(1-p)] (aleatoric) + Var_theta[p] (epistemic)."""
    p1 = samples[..., -1]
    return {"au_var": (p1 * (1 - p1)).mean(0), "eu_var": p1.var(0)}
