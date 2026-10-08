"""Turn raw results/ into the paper's tables, hypothesis tests and figures.

    python -m bdx.analyze [--res results]

Writes results/summary/*.csv, results/summary/hypotheses.md and figures/*.png.
All decisions below were fixed *before* looking at the results (pre-registered in README.md):
primary detector = PH on the log-ratio signal (PH[lr]); alpha = 0.05 with Holm correction inside each
hypothesis family; missed detections are censored at the end of the stream.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst

from . import stats as S
from .data import ROOT

# reference palette (validated categorical order) + neutral inks
C = {"lr": "#2a78d6", "eu": "#eb6834", "au": "#1baf7a", "tu": "#eda100", "conf": "#e87ba4",
     "sup": "#52514e", "ks": "#4a3aa7", "em": "#008300", "ink": "#0b0b0b", "muted": "#8a8984",
     "grid": "#e4e3df", "red": "#e34948"}
SIG_ORDER = ["lr", "eu", "au", "tu", "conf"]
SIG_NAME = {"lr": "log(EU/AU)", "eu": "EU", "au": "AU", "tu": "TU", "conf": "1-maxprob"}
PX_DRIFTS = ["covariate", "sensor"]          # drifts that change P(x)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["muted"], "axes.linewidth": 0.6, "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold",
})


def scenario(row):
    if row["dataset"] == "digits":
        return "digits rotation" if row["drift"] == "covariate" else "digits none"
    if row["dataset"] == "sea":
        return "SEA real" if row["drift"] == "real" else "SEA none"
    return row["drift"]


def drift_kind(drift):
    return {"covariate": "P(x)", "sensor": "P(x)+P(y|x)", "real": "P(y|x)", "none": "none"}.get(drift, drift)


def load(res):
    runs = pd.read_csv(os.path.join(res, "runs.csv"))
    sig = pd.read_csv(os.path.join(res, "signals.csv"))
    runs["scenario"] = runs.apply(scenario, axis=1)
    sig["scenario"] = sig.apply(scenario, axis=1)
    horizon = runs["L"] - runs["t0"].fillna(runs["L"] - 3500)
    runs["delay_c"] = runs["delay"].where(runs["detected"], horizon)   # censored delay
    runs["early"] = runs["detected"] & runs["acc_drop_detected"] & (runs["lead"] >= 0)
    # abrupt-drift ablation runs are analysed separately (width == 0)
    abrupt = runs[(runs.width == 0) & (runs.drift != "none")].copy()
    runs = runs[~runs.index.isin(abrupt.index)].copy()
    sig = sig[sig.width != 0]
    runs.attrs["abrupt"] = abrupt
    return runs, sig


# ----------------------------------------------------------------------------- tables
def detector_table(runs):
    drifted = runs[runs.drift != "none"]
    null = runs[runs.drift == "none"]
    g = drifted.groupby(["drift", "detector"])
    t = pd.DataFrame({
        "detection_rate": g["detected"].mean(),
        "median_delay": g["delay"].median(),
        "false_alarms_pre_per_1000": g["far_per_1000"].mean(),
        "early_warning_rate": g.apply(lambda d: d.loc[d.acc_drop_detected, "early"].mean(), include_groups=False),
        "median_lead": g["lead"].median(),
        "n": g.size(),
    }).reset_index()
    nr = null.groupby("detector")["detected"].mean().rename("null_detection_rate")
    fa = null.groupby("detector")["far_per_1000"].mean().rename("null_far_per_1000")
    return t.merge(nr, on="detector").merge(fa, on="detector")


def h1_early_warning(runs, primary=("PH[lr]", "BOCPD[lr]", "ADWIN[lr]", "PH[eu]")):
    """H1: label-free alarm precedes the visible accuracy drop when P(x) changes."""
    out = []
    sub = runs[(runs.drift.isin(PX_DRIFTS)) & runs.acc_drop_detected]
    comps = ["DDM[err]", "ADWIN[err]", "PH[err]"]
    for det in list(primary):
        d = sub[sub.detector == det]
        lead = d["lead"].dropna()
        stat, p = S.wilcoxon_greater(lead)
        med, lo, hi = S.bootstrap_ci(lead)
        row = dict(detector=det, n_runs_with_drop=len(d), n_detected=int(d.detected.sum()),
                   early_warning_rate=float(d.early.mean()), median_lead=med, lead_ci_low=lo, lead_ci_high=hi,
                   wilcoxon_p_lead_gt_0=p)
        for c in comps:
            a = d.set_index(["dataset", "drift", "model", "seed"])["delay_c"]
            b = sub[sub.detector == c].set_index(["dataset", "drift", "model", "seed"])["delay_c"]
            a, b = a.align(b, join="inner")
            _, pp, n = S.paired_wilcoxon(a.values, b.values, alternative="less")
            row[f"p_delay_lt_{c}"] = pp
            row[f"median_delay_diff_vs_{c}"] = float(np.median(a.values - b.values))
        out.append(row)
    df = pd.DataFrame(out)
    df["holm_p_lead"] = S.holm(df["wilcoxon_p_lead_gt_0"].fillna(1).values)
    return df


def h2_ratio(sig, runs):
    """H2: the log-ratio separates drifted from pre-drift data better than EU, TU, AU, 1-maxprob."""
    px = sig[sig.drift.isin(PX_DRIFTS)]
    keys = ["dataset", "drift", "model", "seed"]
    piv = px.pivot_table(index=keys, columns="signal", values="auroc")
    rows = []
    for s in ["eu", "tu", "au", "conf"]:
        _, p, n = S.paired_wilcoxon(piv["lr"].values, piv[s].values, alternative="greater")
        rows.append(dict(comparison=f"AUROC lr > {s}", n=n, median_lr=float(piv["lr"].median()),
                         median_other=float(piv[s].median()), median_diff=float((piv["lr"] - piv[s]).median()),
                         p=p))
    dsub = runs[runs.drift.isin(PX_DRIFTS)]
    for s in ["eu", "tu", "au", "conf"]:
        a = dsub[dsub.detector == "PH[lr]"].set_index(keys)["delay_c"]
        b = dsub[dsub.detector == f"PH[{s}]"].set_index(keys)["delay_c"]
        a, b = a.align(b, join="inner")
        _, p, n = S.paired_wilcoxon(a.values, b.values, alternative="less")
        rows.append(dict(comparison=f"delay PH[lr] < PH[{s}]", n=n, median_lr=float(np.median(a)),
                         median_other=float(np.median(b)), median_diff=float(np.median(a.values - b.values)), p=p))
    df = pd.DataFrame(rows)
    df["holm_p"] = S.holm(df["p"].fillna(1).values)
    by_model = px.groupby(["model", "signal"])["auroc"].median().unstack()
    return df, by_model


def h2_by_model(sig):
    """H2/H5: is the ratio's advantage over EU specific to the model family? Paired Wilcoxon per model,
    two-sided, Holm-corrected across models."""
    px = sig[sig.drift.isin(PX_DRIFTS)]
    keys = ["dataset", "drift", "model", "seed"]
    rows = []
    for m, d in px.groupby("model"):
        piv = d.pivot_table(index=keys, columns="signal", values="auroc")
        for s in ["eu", "tu"]:
            _, p, n = S.paired_wilcoxon(piv["lr"].values, piv[s].values, alternative="two-sided")
            rows.append(dict(model=m, comparison=f"lr vs {s}", n=n, median_lr=float(piv["lr"].median()),
                             median_other=float(piv[s].median()),
                             share_lr_better=float((piv["lr"] > piv[s]).mean()), p_two_sided=p))
    df = pd.DataFrame(rows)
    df["holm_p"] = S.holm(df["p_two_sided"].fillna(1).values)
    return df


def h3_real_drift(sig, runs, margin=0.05):
    """H3: under pure P(y|x) drift label-free signals carry no information (AUROC ~ 0.5) while supervised
    detectors fire. Equivalence tested with TOST (two one-sided Wilcoxon tests, margin +-0.05)."""
    real = sig[sig.drift == "real"]
    rows = []
    for s in SIG_ORDER:
        x = real[real.signal == s]["auroc"].values - 0.5
        p_low = sst.wilcoxon(x + margin, alternative="greater").pvalue
        p_high = sst.wilcoxon(x - margin, alternative="less").pvalue
        rows.append(dict(signal=s, n=len(x), median_auroc=float(np.median(x + 0.5)),
                         tost_p=float(max(p_low, p_high)), equivalent_to_chance=bool(max(p_low, p_high) < 0.05)))
    eq = pd.DataFrame(rows)
    # detection rate on real drift vs null detection rate (Fisher exact), per detector
    drows = []
    for det, d in runs[runs.drift.isin(["real", "none"])].groupby("detector"):
        r, n0 = d[d.drift == "real"], d[d.drift == "none"]
        table = [[int(r.detected.sum()), int((~r.detected).sum())], [int(n0.detected.sum()), int((~n0.detected).sum())]]
        p = sst.fisher_exact(table, alternative="greater").pvalue
        drows.append(dict(detector=det, detection_rate_real=float(r.detected.mean()),
                          null_detection_rate=float(n0.detected.mean()), fisher_p_real_gt_null=float(p)))
    return eq, pd.DataFrame(drows)


def h4_xai(xai):
    rows = []
    expl = sorted({c.rsplit("_prec_at_k", 1)[0] for c in xai.columns
                   if c.endswith("_prec_at_k") and not c.startswith("chance")})
    for drift, d in xai.groupby("drift"):
        for e in expl:
            gr = d[f"{e}_gt_rank"].dropna()
            pk = d[f"{e}_prec_at_k"].dropna()
            _, p_rank = S.wilcoxon_greater(gr, 0.5)
            hits = (pk * d.loc[pk.index, "k"]).sum()
            trials = d.loc[pk.index, "k"].sum()
            p_binom = S.binomial_greater(hits, trials, float(d["chance_prec_at_k"].mean()))
            rn, rd = d[f"{e}_rho_null"], d[f"{e}_rho_drift"]
            _, p_stab, _ = S.paired_wilcoxon(rd.values, rn.values, alternative="less")
            rows.append(dict(drift=drift, explainer=e, n=len(d), mean_prec_at_k=float(pk.mean()),
                             chance=float(d["chance_prec_at_k"].mean()), binom_p=p_binom,
                             median_gt_rank=float(gr.median()), wilcoxon_p_rank_gt_half=p_rank,
                             median_rho_null=float(rn.median()), median_rho_drift=float(rd.median()),
                             p_rho_drift_lt_null=p_stab))
    df = pd.DataFrame(rows)
    df["holm_p_rank"] = S.holm(df["wilcoxon_p_rank_gt_half"].fillna(1).values)
    agree = xai.groupby("drift")[[c for c in xai.columns if c.startswith("agree_")]].median()
    return df, agree


def meta_analysis(runs, det="PH[lr]", ref="PH[err]"):
    """Per dataset 'study': mean paired delay advantage (ref - det, positive = label-free earlier)
    on P(x)-drift scenarios; pooled with DerSimonian-Laird random effects."""
    keys = ["dataset", "drift", "model", "seed"]
    sub = runs[runs.drift.isin(PX_DRIFTS)]
    a = sub[sub.detector == det].set_index(keys)["delay_c"]
    b = sub[sub.detector == ref].set_index(keys)["delay_c"]
    a, b = a.align(b, join="inner")
    diff = (b - a).reset_index(name="adv")
    diff["study"] = diff["dataset"] + " / " + diff["drift"]
    st = diff.groupby("study")["adv"].agg(["mean", "std", "count"]).reset_index()
    st["se"] = st["std"] / np.sqrt(st["count"])
    ma = S.dersimonian_laird(st["mean"], st["se"])
    return st, ma


def label_delay_ablation(runs, delays=(0, 250, 500, 1000)):
    """Supervised detectors need labels; delaying labels shifts their alarms. Early-warning rate of the
    best supervised detector vs PH[lr] as a function of label delay (no re-run needed)."""
    sub = runs[runs.drift.isin(PX_DRIFTS) & runs.acc_drop_detected].copy()
    rows = []
    for ld in delays:
        for det in ["PH[lr]", "DDM[err]", "ADWIN[err]", "PH[err]"]:
            d = sub[sub.detector == det]
            first = d["first_alarm"] + (ld if det.endswith("[err]") else 0)
            ok = d["detected"] & (first <= d["t_acc"]) & (first < d["L"])
            rows.append(dict(label_delay=ld, detector=det, early_warning_rate=float(ok.mean())))
    return pd.DataFrame(rows)


def elec2_summary(el):
    el = el.copy()
    # stats.granger returns the minimum p over lags 1..3 -> Bonferroni-adjust for the 3 lags tried
    for c in ["granger_p_signal_to_error", "granger_p_error_to_signal"]:
        el[c] = (3 * el[c]).clip(upper=1.0)
    g = el.groupby(["signal", "lag"])
    t = g[["corr", "ci_low", "ci_high"]].median().reset_index()
    gr = el.groupby("signal")[["granger_p_signal_to_error", "granger_p_error_to_signal"]].median()
    frac = el.drop_duplicates(["model", "seed", "signal"]).groupby("signal").apply(
        lambda d: pd.Series({"frac_runs_signal_GC_error": float((d.granger_p_signal_to_error < 0.05).mean()),
                             "frac_runs_error_GC_signal": float((d.granger_p_error_to_signal < 0.05).mean())}),
        include_groups=False)
    return t, gr.join(frac)


# ----------------------------------------------------------------------------- figures
def fig_example(res, out):
    p = os.path.join(res, "example_trace.npz")
    if not os.path.exists(p):
        return
    z = np.load(p, allow_pickle=True)
    al = json.loads(str(z["alarms"]))
    t0, t_acc = int(z["t0"]), int(z["t_acc"])
    L = len(z["err"])
    roll = lambda x, w=200: pd.Series(x).rolling(w, min_periods=20).mean().values
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw={"hspace": 0.25})
    ramp_end = t0 + int(np.argmax(z["p"][t0:] >= 1)) if (z["p"] >= 1).any() else L
    for ax in axes:
        ax.axvspan(t0, ramp_end, color="#f3efe6", lw=0)
        ax.axvline(t0, color=C["muted"], lw=0.8, ls="--")
        if t_acc > 0:
            ax.axvline(t_acc, color=C["red"], lw=0.8)
    ax = axes[0]
    ax.plot(roll(1 - z["err"], 300), color=C["ink"], lw=1.4)
    ax.axhline(float(z["acc_thr"]), color=C["red"], lw=0.8, ls=":")
    ax.set_ylabel("rolling accuracy\n(needs labels)")
    ax.set_title("Credit default, covariate drift, MC-Dropout (seed 0)", loc="left")
    ax.text(t0 + 20, ax.get_ylim()[0] + 0.02, "drift ramp", color=C["muted"], fontsize=8)
    if t_acc > 0:
        ax.text(t_acc + 30, float(z["acc_thr"]) + 0.03, "visible accuracy drop", color=C["red"], fontsize=8)
    for ax, keys, lab in [(axes[1], ["eu", "au"], "uncertainty (nats)"), (axes[2], ["lr"], "log(EU/AU)")]:
        for k in keys:
            ax.plot(roll(z[f"sig_{k}"]), color=C[k], lw=1.4, label=SIG_NAME[k])
        ax.set_ylabel(lab)
        if len(keys) > 1:
            ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.6), ncol=2, facecolor="white", framealpha=1)
    ax = axes[3]
    dets = [("PH[lr]", C["lr"]), ("BOCPD[lr]", C["lr"]), ("PH[eu]", C["eu"]), ("KS[x]", C["ks"]),
            ("PH[err]", C["sup"]), ("ADWIN[err]", C["sup"]), ("DDM[err]", C["sup"])]
    for i, (d, col) in enumerate(dets):
        ax.scatter(al.get(d, []), [i] * len(al.get(d, [])), marker="v", s=36, color=col, zorder=3)
    ax.set_yticks(range(len(dets)), [d for d, _ in dets])
    ax.set_ylim(-0.7, len(dets) - 0.3)
    ax.set_xlabel("stream position (samples)")
    ax.set_ylabel("alarms")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(out, "fig1_example_trace.png"), bbox_inches="tight")
    plt.close(fig)


def fig_auroc(sig, out):
    order = ["covariate", "sensor", "digits rotation", "real", "SEA real"]
    sig = sig[sig.scenario.isin(order)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.16
    for j, s in enumerate(SIG_ORDER):
        for i, sc in enumerate(order):
            v = sig[(sig.scenario == sc) & (sig.signal == s)]["auroc"].values
            if len(v) == 0:
                continue
            med, lo, hi = S.bootstrap_ci(v, B=1000)
            x = i + (j - 2) * w
            ax.bar(x, med - 0.5, bottom=0.5, width=w * 0.86, color=C[s], label=SIG_NAME[s] if i == 0 else None)
            ax.plot([x, x], [lo, hi], color=C["ink"], lw=0.8)
    ax.axhline(0.5, color=C["ink"], lw=0.8)
    for i in (3, 4):
        ax.text(i, 0.53, "all signals ≈ 0.5\n(invisible without labels)", ha="center", va="bottom", fontsize=7.5,
                color=C["muted"])
    ax.set_xticks(range(len(order)), ["covariate\nP(x)", "sensor\nP(x)+P(y|x)", "digits rotation\nP(x)",
                                      "real\nP(y|x)", "SEA\nP(y|x)"])
    ax.set_ylabel("AUROC drifted vs pre-drift\n(0.5 = no information)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(out, "fig2_signal_auroc.png"), bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(tab, out):
    t = tab[tab.drift.isin(PX_DRIFTS)].groupby("detector").agg(
        delay=("median_delay", "median"), det=("detection_rate", "mean"), null=("null_detection_rate", "first"),
        ewr=("early_warning_rate", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for _, r in t.iterrows():
        d = r.detector
        sigkey = d[d.find("[") + 1:-1]
        col = C["sup"] if d.endswith("[err]") else C["ks"] if d.startswith("KS") else C["em"] if d.startswith("EM") \
            else C.get(sigkey, C["muted"])
        mk = {"PH": "o", "ADWIN": "s", "BOCPD": "D", "DDM": "^", "KS": "P", "EM-GMM": "X"}[d.split("[")[0]]
        ax.scatter(r.null, r.ewr, color=col, marker=mk, s=46, edgecolor="white", linewidth=0.8, zorder=3)
        if d in ("PH[lr]", "BOCPD[lr]", "PH[err]", "DDM[err]", "ADWIN[err]", "KS[x]", "PH[eu]", "EM-GMM[au,eu]"):
            off = (-34, 7) if d == "PH[eu]" else (5, 3)
            ax.annotate(d, (r.null, r.ewr), xytext=off, textcoords="offset points", fontsize=7.5, color=C["ink"])
    ax.set_xlabel("spurious detection rate on stationary streams (lower is better)")
    ax.set_ylabel("early-warning rate\n(alarm before visible accuracy drop)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C[s], label=SIG_NAME[s]) for s in SIG_ORDER] + \
        [Line2D([], [], marker="o", ls="", color=C["sup"], label="error (labels)"),
         Line2D([], [], marker="P", ls="", color=C["ks"], label="KS on inputs"),
         Line2D([], [], marker="X", ls="", color=C["em"], label="EM-GMM")]
    h += [Line2D([], [], ls="", label=" "),
          Line2D([], [], marker="o", ls="", color=C["muted"], label="Page-Hinkley"),
          Line2D([], [], marker="s", ls="", color=C["muted"], label="ADWIN"),
          Line2D([], [], marker="D", ls="", color=C["muted"], label="BOCPD"),
          Line2D([], [], marker="^", ls="", color=C["muted"], label="DDM")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.06, 0.5), title="colour = signal\nmarker = detector",
              title_fontsize=7.5, fontsize=7.5)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.02)
    ax.set_title("Early warning vs. false alarms (P(x)-changing drifts)", loc="left")
    fig.savefig(os.path.join(out, "fig3_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def fig_forest(st, ma, out):
    if not ma:
        return
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * (len(st) + 2) + 0.8))
    y = np.arange(len(st))[::-1] + 1
    ax.errorbar(st["mean"], y, xerr=1.96 * st["se"], fmt="s", color=C["lr"], ecolor=C["ink"], elinewidth=0.8,
                capsize=2, ms=6)
    ax.errorbar([ma["pooled"]], [0], xerr=[[ma["pooled"] - ma["ci_low"]], [ma["ci_high"] - ma["pooled"]]],
                fmt="D", color=C["ink"], ms=7, capsize=2)
    ax.axvline(0, color=C["muted"], lw=0.8)
    ax.set_yticks(list(y) + [0], list(st["study"]) + [f"Random effects (I²={ma['I2']:.0%})"])
    ax.set_xlabel("delay advantage of PH[log(EU/AU)] over PH[error]  (samples; >0 = earlier)")
    ax.set_title("Meta-analysis across datasets", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(out, "fig4_meta_forest.png"), bbox_inches="tight")
    plt.close(fig)


def fig_xai(xai, out):
    expl = ["SHAP_EU", "LIME_EU", "SHAP_pred", "GradSHAP_pred", "LIME_pred"]
    expl = [e for e in expl if f"{e}_gt_rank" in xai.columns]
    drifts = [d for d in ["covariate", "sensor", "real"] if d in xai.drift.unique()]
    cols = [C["lr"], C["eu"], C["au"], C["tu"], C["conf"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.35})
    w = 0.16
    for j, e in enumerate(expl):
        for i, dr in enumerate(drifts):
            v = xai[xai.drift == dr][f"{e}_gt_rank"].dropna().values
            med, lo, hi = S.bootstrap_ci(v, np.mean, B=1000)
            x = i + (j - 2) * w
            axes[0].bar(x, med - 0.5, bottom=0.5, width=w * 0.86, color=cols[j], label=e if i == 0 else None)
            axes[0].plot([x, x], [lo, hi], color=C["ink"], lw=0.8)
    axes[0].axhline(0.5, color=C["ink"], lw=0.8)
    axes[0].set_xticks(range(len(drifts)), drifts)
    axes[0].set_ylabel("rank of drifted features in Δ|attribution|\n(1 = top, 0.5 = chance)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Localisation", loc="left")
    axes[0].grid(axis="x", visible=False)
    for j, e in enumerate(expl):
        for i, dr in enumerate(drifts):
            d = xai[xai.drift == dr]
            x = i + (j - 2) * w
            axes[1].plot([x, x], [d[f"{e}_rho_null"].median(), d[f"{e}_rho_drift"].median()], color=cols[j], lw=1.5)
            axes[1].scatter([x], [d[f"{e}_rho_null"].median()], color="white", edgecolor=cols[j], s=22, zorder=3)
            axes[1].scatter([x], [d[f"{e}_rho_drift"].median()], color=cols[j], s=22, zorder=3)
    axes[1].set_xticks(range(len(drifts)), drifts)
    axes[1].set_ylabel("Spearman ρ of global importance\n(○ pre vs pre, ● pre vs alarm)")
    axes[1].set_title("Stability", loc="left")
    axes[1].grid(axis="x", visible=False)
    axes[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.25), fontsize=7.5)
    fig.savefig(os.path.join(out, "fig5_xai.png"), bbox_inches="tight")
    plt.close(fig)


def fig_elec(el_t, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for s in ["lr", "eu", "conf"]:
        d = el_t[el_t.signal == s]
        ax.fill_between(d.lag, d.ci_low, d.ci_high, color=C[s], alpha=0.15, lw=0)
        ax.plot(d.lag, d["corr"], color=C[s], marker="o", ms=4, lw=1.6, label=SIG_NAME[s])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.axvline(0, color=C["muted"], lw=0.8, ls="--")
    ax.set_xlabel("lag k (blocks of 200 samples); k > 0: signal leads the error rate")
    ax.set_ylabel("corr(signal_t, error_{t+k})")
    ax.set_title("Elec2 natural drift: lead-lag with moving-block bootstrap 95% CI", loc="left")
    ax.legend()
    fig.savefig(os.path.join(out, "fig6_elec2_leadlag.png"), bbox_inches="tight")
    plt.close(fig)


def fig_models(by_model, out):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    models = [m for m in ["mcd", "ens", "bsvm"] if m in by_model.index]
    names = {"mcd": "MC-Dropout\n(Bayesian approx.)", "ens": "Deep Ensemble\n(Bayesian approx.)",
             "bsvm": "Bootstrap SVM\n(frequentist)"}
    w = 0.16
    for j, s in enumerate(SIG_ORDER):
        for i, m in enumerate(models):
            ax.bar(i + (j - 2) * w, by_model.loc[m, s] - 0.5, bottom=0.5, width=w * 0.86, color=C[s],
                   label=SIG_NAME[s] if i == 0 else None)
    ax.axhline(0.5, color=C["ink"], lw=0.8)
    ax.set_xticks(range(len(models)), [names[m] for m in models])
    ax.set_ylabel("median AUROC (P(x) drifts)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(out, "fig7_bayes_vs_freq.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def fmt_p(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--fig", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args(argv)
    out = os.path.join(a.res, "summary")
    os.makedirs(out, exist_ok=True)
    os.makedirs(a.fig, exist_ok=True)
    runs, sig = load(a.res)

    tab = detector_table(runs)
    tab.to_csv(os.path.join(out, "T1_detectors.csv"), index=False)
    abrupt = runs.attrs.get("abrupt")
    ab = None
    if abrupt is not None and len(abrupt):
        abrupt = abrupt.copy()
        abrupt["early"] = abrupt["early"].where(abrupt["acc_drop_detected"])   # among runs with a visible drop
        ab = abrupt.groupby(["detector"]).agg(detection_rate=("detected", "mean"), median_delay=("delay", "median"),
                                              early_warning_rate=("early", "mean")).reset_index()
        grad = runs[runs.dataset.isin(abrupt.dataset.unique()) & runs.drift.isin(abrupt.drift.unique())
                    & ~((runs.dataset == "digits") & (runs.drift != "covariate"))]
        gr = grad.groupby("detector").agg(gradual_detection_rate=("detected", "mean"),
                                          gradual_median_delay=("delay", "median")).reset_index()
        ab = ab.merge(gr, on="detector")
        ab.to_csv(os.path.join(out, "ablation_abrupt_vs_gradual.csv"), index=False)
    h1 = h1_early_warning(runs)
    h1.to_csv(os.path.join(out, "H1_early_warning.csv"), index=False)
    h2, by_model = h2_ratio(sig, runs)
    h2.to_csv(os.path.join(out, "H2_ratio.csv"), index=False)
    by_model.to_csv(os.path.join(out, "H5_auroc_by_model.csv"))
    h5 = h2_by_model(sig)
    h5.to_csv(os.path.join(out, "H5_ratio_by_model.csv"), index=False)
    # H1 broken down by scenario (where does the early warning work?)
    sub = runs[runs.drift.isin(PX_DRIFTS)].copy()
    sub["early"] = sub["early"].where(sub["acc_drop_detected"])     # rate among runs that have a visible drop
    h1s = sub[sub.detector.isin(["PH[lr]", "PH[eu]", "KS[x]", "PH[err]", "DDM[err]"])].groupby(
        ["dataset", "drift", "detector"]).agg(detection_rate=("detected", "mean"), median_delay=("delay", "median"),
                                              runs_with_drop=("acc_drop_detected", "sum"),
                                              early_warning_rate=("early", "mean"),
                                              median_lead=("lead", "median")).reset_index()
    h1s.to_csv(os.path.join(out, "H1_by_scenario.csv"), index=False)
    eq, det_real = h3_real_drift(sig, runs)
    eq.to_csv(os.path.join(out, "H3_equivalence.csv"), index=False)
    det_real.to_csv(os.path.join(out, "H3_detection_real_vs_null.csv"), index=False)
    st, ma = meta_analysis(runs)
    st.to_csv(os.path.join(out, "meta_studies.csv"), index=False)
    json.dump(ma, open(os.path.join(out, "meta_pooled.json"), "w"), indent=2)
    abl = label_delay_ablation(runs)
    abl.to_csv(os.path.join(out, "ablation_label_delay.csv"), index=False)
    sig_summary = sig.groupby(["scenario", "model", "signal"])[["auroc", "cohen_d"]].median().reset_index()
    sig_summary.to_csv(os.path.join(out, "T2_signals.csv"), index=False)
    acc = runs.drop_duplicates(["dataset", "drift", "model", "seed"]).groupby(["scenario", "model"])[
        ["acc_pre", "acc_post", "acc_drop_detected"]].mean().reset_index()
    acc.to_csv(os.path.join(out, "T0_accuracy.csv"), index=False)

    fig_example(a.res, a.fig)
    fig_auroc(sig, a.fig)
    fig_tradeoff(tab, a.fig)
    fig_forest(st, ma, a.fig)
    fig_models(by_model, a.fig)

    xai_ok = os.path.exists(os.path.join(a.res, "xai.csv"))
    if xai_ok:
        xai = pd.read_csv(os.path.join(a.res, "xai.csv"))
        h4, agree = h4_xai(xai)
        h4.to_csv(os.path.join(out, "H4_xai.csv"), index=False)
        agree.to_csv(os.path.join(out, "H4_agreement.csv"))
        fig_xai(xai, a.fig)
    el_ok = os.path.exists(os.path.join(a.res, "elec2.csv"))
    if el_ok:
        el = pd.read_csv(os.path.join(a.res, "elec2.csv"))
        el_t, el_g = elec2_summary(el)
        el_t.to_csv(os.path.join(out, "E_elec2_crosscorr.csv"), index=False)
        el_g.to_csv(os.path.join(out, "E_elec2_granger.csv"))
        fig_elec(el_t, a.fig)

    # ---------------- verdicts
    L = ["# Hypothesis verdicts (auto-generated by `python -m bdx.analyze`)", ""]
    r = h1.set_index("detector").loc["PH[lr]"]
    L += ["## H1 — early warning (P(x)-changing drifts, runs with a visible accuracy drop)",
          f"* PH[log(EU/AU)]: early-warning rate = {r.early_warning_rate:.2f} over {int(r.n_runs_with_drop)} runs; "
          f"median lead = {r.median_lead:.0f} samples (95% CI {r.lead_ci_low:.0f} … {r.lead_ci_high:.0f}); "
          f"Wilcoxon lead>0 p = {fmt_p(r.wilcoxon_p_lead_gt_0)} (Holm {fmt_p(r.holm_p_lead)}).",
          f"* Paired delay vs DDM p = {fmt_p(r['p_delay_lt_DDM[err]'])}, vs ADWIN[err] p = {fmt_p(r['p_delay_lt_ADWIN[err]'])}, "
          f"vs PH[err] p = {fmt_p(r['p_delay_lt_PH[err]'])} (one-sided, label-free earlier).",
          f"* Verdict: **{'CONFIRMED' if r.holm_p_lead < 0.05 else 'NOT CONFIRMED'}**", ""]
    L += ["## H2 — the ratio beats its components"] + [
        f"* {x.comparison}: median {x.median_lr:.3f} vs {x.median_other:.3f}, Holm p = {fmt_p(x.holm_p)}"
        for x in h2.itertuples()]
    n_win = int((h2.holm_p < 0.05).sum())
    L += [f"* Verdict: **{'CONFIRMED' if n_win == len(h2) else 'PARTIALLY CONFIRMED' if n_win else 'NOT CONFIRMED'}** "
          f"({n_win}/{len(h2)} comparisons significant)", ""]
    L += ["## H3 — boundary: pure P(y|x) drift is invisible without labels"] + [
        f"* {x.signal}: median AUROC {x.median_auroc:.3f}, TOST(±0.05) p = {fmt_p(x.tost_p)} -> "
        f"{'equivalent to chance' if x.equivalent_to_chance else 'not shown equivalent'}" for x in eq.itertuples()]
    dr = det_real.set_index("detector")
    L += [f"* Detection rate on real drift vs stationary: PH[lr] {dr.loc['PH[lr]', 'detection_rate_real']:.2f} vs "
          f"{dr.loc['PH[lr]', 'null_detection_rate']:.2f}; DDM[err] {dr.loc['DDM[err]', 'detection_rate_real']:.2f} vs "
          f"{dr.loc['DDM[err]', 'null_detection_rate']:.2f}",
          f"* Verdict: early-warning claim is **{'REFUTED for P(y|x) drift (as predicted)' if eq.equivalent_to_chance.all() else 'partly refuted'}**", ""]
    if xai_ok:
        L += ["## H4 — XAI validation"]
        for x in h4.itertuples():
            L += [f"* {x.drift:9s} {x.explainer:14s} gt-rank {x.median_gt_rank:.2f} (p {fmt_p(x.wilcoxon_p_rank_gt_half)}, "
                  f"Holm {fmt_p(x.holm_p_rank)}); prec@k {x.mean_prec_at_k:.2f} vs chance {x.chance:.2f}; "
                  f"rho null {x.median_rho_null:.2f} -> drift {x.median_rho_drift:.2f} (p {fmt_p(x.p_rho_drift_lt_null)})"]
        L += [""]
    if ma:
        L += ["## Meta-analysis (DerSimonian-Laird, studies = dataset × drift)",
              f"* pooled delay advantage of PH[lr] over PH[err] = {ma['pooled']:.0f} samples "
              f"(95% CI {ma['ci_low']:.0f} … {ma['ci_high']:.0f}), p = {fmt_p(ma['p_two_sided'])}, "
              f"I² = {ma['I2']:.0%}, tau² = {ma['tau2']:.0f}", ""]
    L += ["## H5 — Bayesian vs frequentist uncertainty (median AUROC on P(x) drifts)", "",
          by_model.round(3).to_markdown(), "", "Paired test of the ratio vs its components, per model (two-sided, Holm):",
          "", h5.round(4).to_markdown(index=False), ""]
    if el_ok:
        L += ["## Elec2 (natural drift, no ground truth)", "", el_g.round(3).to_markdown(), ""]
    L += ["## Label-delay ablation (early-warning rate)", "",
          abl.pivot(index="label_delay", columns="detector", values="early_warning_rate").round(2).to_markdown()]
    if ab is not None:
        keep = ["PH[lr]", "ADWIN[lr]", "BOCPD[lr]", "BOCPD[eu]", "PH[eu]", "EM-GMM[au,eu]", "KS[x]", "DDM[err]",
                "PH[err]", "ADWIN[err]"]
        L += ["", "## Ablation: abrupt (width 0) vs gradual drift (credit covariate/sensor, digits rotation)", "",
              ab[ab.detector.isin(keep)].round(2).to_markdown(index=False)]
    open(os.path.join(out, "hypotheses.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
