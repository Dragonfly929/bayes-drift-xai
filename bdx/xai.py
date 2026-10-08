"""Explainability as *validation* of the uncertainty alarm.

Two questions, asked on the samples that trigger the alarm:

1. Localisation — do attributions of the **epistemic uncertainty** EU(x) point at the features that
   actually drifted (known ground truth)? We explain the function x -> EU(x) itself (possible because the
   MC-Dropout masks are fixed, see models.py). Score: precision@k of the features whose mean |attribution|
   increased most from the pre-drift reference to the alarm window; chance level = k/d.
2. Stability — do the explanations of the **prediction** keep the same feature ranking after the alarm?
   Score: Spearman rho between global importance vectors (mean |attribution|), compared with the rho
   between two disjoint pre-drift windows (the no-drift noise floor).

Explainers (three families, as discussed in the course):
* KernelSHAP  (Lundberg & Lee, 2017) — model-agnostic, Shapley values estimated by weighted regression;
* GradientSHAP / expected gradients — model-specific ("diagnostic"), uses the network's gradients;
* LIME        (Ribeiro et al., 2016) — model-agnostic local linear surrogate.
"""
from __future__ import annotations

import warnings

import numpy as np
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from .uncertainty import decompose

warnings.filterwarnings("ignore")


def eu_fn(model):
    return lambda X: decompose(model.predict_samples(np.asarray(X, float)))["eu"]


def lr_fn(model):
    return lambda X: decompose(model.predict_samples(np.asarray(X, float)))["lr"]


def pred_fn(model, cls=1):
    return lambda X: model.predict_samples(np.asarray(X, float)).mean(0)[:, cls]


def kernel_shap(f, X_bg, X, nsamples=300, seed=0):
    np.random.seed(seed)
    bg = shap.kmeans(X_bg, min(len(X_bg), 40)) if len(X_bg) > 40 else X_bg
    ex = shap.KernelExplainer(f, bg)
    return np.asarray(ex.shap_values(X, nsamples=nsamples, silent=True))


def gradient_shap(model, X_bg, X, cls=1, seed=0):
    """Expected-gradients SHAP on the MC-averaged network (only for torch models)."""
    if not hasattr(model, "torch_mean_model"):
        return None
    torch.manual_seed(seed)
    net = model.torch_mean_model()
    bg = torch.as_tensor(X_bg[:100], dtype=torch.float32)
    ex = shap.GradientExplainer(net, bg)
    sv = ex.shap_values(torch.as_tensor(X, dtype=torch.float32), nsamples=200)
    sv = np.asarray(sv)
    # shap>=0.45 returns (n, d, C); older versions a list over classes
    if sv.ndim == 3 and sv.shape[0] == len(X):
        return sv[:, :, cls]
    return sv[cls]


def lime_matrix(f, X_train, X, mode, seed=0, num_samples=1000):
    d = X.shape[1]
    ex = LimeTabularExplainer(X_train, mode="regression", discretize_continuous=False,
                              random_state=seed, verbose=False)
    out = np.zeros((len(X), d))
    for i, x in enumerate(X):
        e = ex.explain_instance(x, f, num_features=d, num_samples=num_samples)
        for j, w in e.local_exp[next(iter(e.local_exp))]:   # regression surrogate: one coefficient list
            out[i, j] = w
    return out


def precision_at_k(delta, gt):
    k = len(gt)
    if k == 0:
        return np.nan
    top = np.argsort(delta)[::-1][:k]
    return len(set(top) & set(gt)) / k


def gt_rank_score(delta, gt):
    """Mean normalised rank of the ground-truth features in the Delta ordering (1 = top, 0 = bottom)."""
    if not gt:
        return np.nan
    order = np.argsort(delta)[::-1]
    d = len(delta)
    ranks = [int(np.where(order == g)[0][0]) for g in gt]
    return float(np.mean([1 - r / (d - 1) for r in ranks]))


def rho(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else np.nan


def select_windows(trace_signal, t0, alarm, n, rng, width=300, pre_len=1000):
    """Matched selection: top-n by the alarm signal in each window (two disjoint pre windows + alarm)."""
    def top(lo, hi):
        lo, hi = max(0, lo), min(len(trace_signal), hi)
        idx = np.arange(lo, hi)
        return idx[np.argsort(trace_signal[idx])[::-1][:n]]
    preA = top(t0 - pre_len, t0 - pre_len // 2)
    preB = top(t0 - pre_len // 2, t0)
    post = top(alarm, alarm + width)
    return preA, preB, post


def explain_run(model, st, signals, alarm, cfg, seed=0, use_lime=True, use_grad=True):
    """Compute all attributions for one stream and return a metrics dict + importance vectors."""
    xc = cfg.get("xai", {})
    n, ns = xc.get("n_explain", 40), xc.get("kernel_nsamples", 300)
    rng = np.random.default_rng(seed)
    preA, preB, post = select_windows(signals["lr"], st.t0, alarm, n, rng)
    Xs = st.X_stream
    bg = st.X_train[rng.choice(len(st.X_train), size=min(400, len(st.X_train)), replace=False)]
    gt = st.gt_features
    res, imps = {}, {}

    explainers = {
        "SHAP_EU": lambda X: kernel_shap(eu_fn(model), bg, X, ns, seed),
        "SHAP_pred": lambda X: kernel_shap(pred_fn(model), bg, X, ns, seed),
    }
    if use_grad and hasattr(model, "torch_mean_model"):
        explainers["GradSHAP_pred"] = lambda X: gradient_shap(model, bg, X, seed=seed)
    if use_lime:
        explainers["LIME_EU"] = lambda X: lime_matrix(eu_fn(model), st.X_train, X, "regression", seed)
        explainers["LIME_pred"] = lambda X: lime_matrix(pred_fn(model), st.X_train, X, "regression", seed)

    raw = {}
    for name, fn in explainers.items():
        A, B, P = fn(Xs[preA]), fn(Xs[preB]), fn(Xs[post])
        raw[name] = (A, B, P)
        iA, iB, iP = np.abs(A).mean(0), np.abs(B).mean(0), np.abs(P).mean(0)
        delta = iP - iB
        imps[name] = {"preA": iA, "preB": iB, "post": iP, "delta": delta}
        res[f"{name}_prec_at_k"] = precision_at_k(delta, gt)
        res[f"{name}_gt_rank"] = gt_rank_score(delta, gt)
        res[f"{name}_rho_null"] = rho(iA, iB)
        res[f"{name}_rho_drift"] = rho(iB, iP)
    # agreement between agnostic SHAP and LIME (per sample) before / after
    if "LIME_pred" in raw:
        for k, (S, Lm) in {"pre": (raw["SHAP_pred"][1], raw["LIME_pred"][1]),
                           "post": (raw["SHAP_pred"][2], raw["LIME_pred"][2])}.items():
            res[f"agree_SHAP_LIME_{k}"] = float(np.nanmean([rho(np.abs(a), np.abs(b)) for a, b in zip(S, Lm)]))
    if "GradSHAP_pred" in raw:
        for k, j in {"pre": 1, "post": 2}.items():
            res[f"agree_SHAP_Grad_{k}"] = float(np.nanmean([rho(np.abs(a), np.abs(b))
                                                             for a, b in zip(raw["SHAP_pred"][j], raw["GradSHAP_pred"][j])]))
    res["chance_prec_at_k"] = len(gt) / Xs.shape[1] if gt else np.nan
    res["d"] = Xs.shape[1]
    res["k"] = len(gt)
    return res, imps
