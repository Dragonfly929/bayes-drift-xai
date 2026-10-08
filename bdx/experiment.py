"""One end-to-end experimental unit:

    data -> drifted stream -> probabilistic model -> uncertainty decomposition -> detectors -> metrics

Key quantities per run
----------------------
t0         true drift onset (known, injected)
t_acc      "visible accuracy drop": first t >= t0 where the trailing accuracy over W samples falls below
           acc_ref - z * sqrt(acc_ref (1-acc_ref) / W)   (acc_ref measured on the calibration set)
delay_d    first alarm of detector d at or after t0, minus t0
lead_d     t_acc - first alarm of d   (> 0  ==> early warning, the alarm came *before* the drop was visible)
FA_d       alarms raised before t0 (false alarms); on 'none' streams every alarm is false
AUROC_s    how well signal s separates fully-drifted samples from pre-drift samples (label-free separability)
d_s        Cohen's d of signal s, post vs pre
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import detectors as DET
from . import drift as DR
from .models import make_model
from .uncertainty import SIGNALS, decompose


def rolling_mean(x, w):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def visible_drop_time(correct, acc_ref, t0, W=300, z=3.0):
    if t0 is None:
        return None, None
    roll = rolling_mean(correct, W)
    thr = acc_ref - z * np.sqrt(max(acc_ref * (1 - acc_ref), 1e-6) / W)
    idx = np.where((np.arange(len(roll)) >= t0) & (roll < thr))[0]
    return (int(idx[0]) if len(idx) else None), thr


def cohen_d(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    return float((b.mean() - a.mean()) / s)


def run_one(dataset, drift_type, model_name, seed, cfg, tab=None, keep_traces=False, stream=None, model=None):
    t_start = time.time()
    st = stream if stream is not None else DR.make_stream(dataset, drift_type, seed, cfg, tab)
    m = model if model is not None else make_model(model_name, st.n_classes, seed, cfg).fit(st.X_train, st.y_train)
    S_cal = decompose(m.predict_samples(st.X_cal), cfg.get("ratio_eps", 1e-4))
    S_str = decompose(m.predict_samples(st.X_stream), cfg.get("ratio_eps", 1e-4))
    err_cal = (S_cal["pred"] != st.y_cal).astype(float)
    err = (S_str["pred"] != st.y_stream).astype(float)
    acc_ref = 1 - err_cal.mean()
    ev = cfg.get("evaluation", {})
    # the visible drop is judged against the accuracy the dashboard showed before the drift
    acc_op = 1 - err[: st.t0].mean() if st.t0 else acc_ref
    t_acc, acc_thr = visible_drop_time(1 - err, acc_op, st.t0, ev.get("W", 300), ev.get("z", 3.0))

    alarms, aux = DET.run_all_detectors(S_str, S_cal, err, err_cal, st.X_cal, st.X_stream, cfg, seed)

    L = len(err)
    base = dict(dataset=dataset, drift=drift_type, model=model_name, seed=seed, t0=st.t0, width=st.width,
                L=L, acc_ref=float(acc_ref), t_acc=t_acc, acc_drop_detected=t_acc is not None,
                gt_features=";".join(st.feature_names[i] for i in st.gt_features))
    if st.t0 is not None:
        pre = slice(0, st.t0)
        post_mask = st.p >= 1.0
        base["acc_pre"] = float(1 - err[pre].mean())
        base["acc_post"] = float(1 - err[post_mask].mean()) if post_mask.any() else np.nan
    else:
        base["acc_pre"] = float(1 - err.mean())
        base["acc_post"] = np.nan

    rows = []
    for name, al in alarms.items():
        r = dict(base)
        r["detector"] = name
        r["family"] = DET.detector_family(name)
        if st.t0 is None:
            # stationary control: every alarm is false. 'detected' = a spurious alarm inside the same
            # post-onset window [t0, L) that drifted streams are scored on -> null detection rate.
            vt0 = None if drift_type == "natural" else cfg["t0"]
            first = DET._first_after(al, vt0) if vt0 is not None else None
            r["n_false_alarms"] = len(al)
            r["first_alarm"] = al[0] if al else None
            r["delay"] = (first - vt0) if first is not None else None
            r["lead"] = None
            r["detected"] = first is not None
        else:
            fa = [a for a in al if a < st.t0]
            first = DET._first_after(al, st.t0)
            r["n_false_alarms"] = len(fa)
            r["first_alarm"] = first
            r["detected"] = first is not None
            r["delay"] = (first - st.t0) if first is not None else None
            r["lead"] = (t_acc - first) if (first is not None and t_acc is not None) else None
        r["far_per_1000"] = 1000 * r["n_false_alarms"] / (st.t0 if st.t0 else L)
        rows.append(r)

    sig_rows = []
    if st.t0 is not None:
        pre_idx = np.arange(st.t0)
        post_idx = np.where(st.p >= 1.0)[0]
        for s in SIGNALS:
            a, b = S_str[s][pre_idx], S_str[s][post_idx]
            yy = np.r_[np.zeros(len(a)), np.ones(len(b))]
            sig_rows.append(dict(base, signal=s, auroc=float(roc_auc_score(yy, np.r_[a, b])),
                                 cohen_d=cohen_d(a, b), mean_pre=float(a.mean()), mean_post=float(b.mean())))
    out = {"rows": rows, "signal_rows": sig_rows, "seconds": time.time() - t_start}
    if keep_traces:
        out["traces"] = dict(signals={k: S_str[k] for k in SIGNALS}, cal={k: S_cal[k] for k in SIGNALS},
                             err=err, p=st.p, y=st.y_stream, pred=S_str["pred"], alarms=alarms, aux=aux,
                             acc_ref=acc_ref, acc_thr=acc_thr, t_acc=t_acc, t0=st.t0, stream=st, model=m)
    return out
