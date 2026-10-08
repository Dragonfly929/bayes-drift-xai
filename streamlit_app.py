"""Streamlit interface — Bayesian uncertainty decomposition for early concept-drift warning, validated by XAI.

Run locally:   streamlit run streamlit_app.py
Deploy:        push the repo to GitHub -> share.streamlit.io -> New app -> main file: streamlit_app.py
"""
from __future__ import annotations

import copy
import json
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

st.set_page_config(page_title="Bayesian Drift Early-Warning", page_icon="📉", layout="wide")

from bdx import data as D  # noqa: E402
from bdx import drift as DR  # noqa: E402
from bdx.models import MODEL_LABELS  # noqa: E402
from bdx.uncertainty import SIGNAL_LABELS  # noqa: E402

COL = {"lr": "#2a78d6", "eu": "#eb6834", "au": "#1baf7a", "tu": "#eda100", "conf": "#e87ba4",
       "sup": "#52514e", "ks": "#4a3aa7", "em": "#008300", "red": "#e34948", "ink": "#0b0b0b"}


@st.cache_data(show_spinner=False)
def base_cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))


@st.cache_resource(show_spinner=False)
def cached_tab(key):
    return D.load_dataset(key)


@st.cache_resource(show_spinner="Training the model and running all detectors …", max_entries=8)
def run_cached(dataset, drift, model, seed, cfg_json, upload_bytes=None, target=None, drop_json="[]"):
    from bdx.experiment import run_one
    cfg = json.loads(cfg_json)
    tab = None
    if dataset == "upload":
        import io
        tab = D.load_csv(io.BytesIO(upload_bytes), target, drop=json.loads(drop_json), name="upload")
        if not tab.numeric_idx:
            tab.numeric_idx = list(range(tab.X.shape[1]))
    elif dataset in D.SPECS:
        tab = cached_tab(dataset)
    return run_one(dataset if dataset != "upload" else "upload", drift, model, seed, cfg, tab=tab, keep_traces=True,
                   stream=DR.build_stream(tab, drift, seed, cfg) if dataset == "upload" else None)


# ============================================================================ sidebar
cfg0 = base_cfg()
st.sidebar.title("⚙️ Experiment")
ds_names = {"credit": "Credit default (finance, UCI)", "telco": "Telco churn (customers, IBM)",
            "digits": "Digits 8×8 rotation (MNIST-like)", "sea": "SEA synthetic (real drift)",
            "elec2": "Elec2 electricity (natural drift)", "upload": "📤 Upload your own CSV"}
dataset = st.sidebar.selectbox("Dataset", list(ds_names), format_func=ds_names.get)
upload_bytes, target, drop_cols, upload_ok = None, None, [], True
if dataset == "upload":
    up = st.sidebar.file_uploader("CSV file (one row per record, one label column)", type=["csv"])
    if up is not None:
        upload_bytes = up.getvalue()
        import io
        df_up = pd.read_csv(io.BytesIO(upload_bytes))
        cols = list(df_up.columns)
        target = st.sidebar.selectbox("Label column (what the model predicts)", cols, index=len(cols) - 1)
        # columns that would only add noise: identifiers and free text (almost one value per row)
        guess = [c for c in cols if c != target and (c.lower() in ("id", "index", "unnamed: 0") or c.lower().endswith("id")
                 or (not pd.api.types.is_numeric_dtype(df_up[c]) and df_up[c].nunique() > 0.5 * len(df_up)))]
        drop_cols = st.sidebar.multiselect("Columns to ignore (IDs, dates, free text)",
                                           [c for c in cols if c != target], default=guess)
        n_cls = df_up[target].nunique()
        st.sidebar.caption(f"{len(df_up):,} rows · {len(cols) - 1 - len(drop_cols)} input columns · "
                           f"{n_cls} label classes")
        if n_cls < 2 or n_cls > 20:
            upload_ok = False
            st.sidebar.error("The label column must have 2–20 classes (classification). "
                             "For a numeric target, bin it into classes first.")
        elif len(df_up) < 1000:
            st.sidebar.warning("Fewer than 1 000 rows: the run works, but results will be noisy.")

allowed = {"digits": ["covariate", "none"], "sea": ["real", "none"], "elec2": ["natural"]}.get(
    dataset, ["covariate", "sensor", "real", "none"])
drift_help = {"covariate": "P(x) moves to an unseen segment (labels genuine)",
              "sensor": "selected features shift by δ·σ (fault / pipeline change)",
              "real": "P(y|x) changes in a region (labels flip), P(x) untouched",
              "none": "stationary control stream", "natural": "time-ordered real stream, unknown drift"}
drift = st.sidebar.selectbox("Drift type", allowed, format_func=lambda d: f"{d} — {drift_help[d]}")
model = st.sidebar.selectbox("Probabilistic model", ["mcd", "ens", "bsvm"], format_func=MODEL_LABELS.get)
seed = st.sidebar.number_input("Seed", 0, 999, 0)

with st.sidebar.expander("Stream & model settings"):
    cfg = copy.deepcopy(cfg0)
    cfg["stream_len"] = st.slider("Stream length", 2000, 10000, cfg0["stream_len"], 500)
    cfg["t0"] = st.slider("Drift onset t0", 500, cfg["stream_len"] - 500, min(cfg0["t0"], cfg["stream_len"] - 500), 100)
    cfg["width"] = st.slider("Ramp width (0 = abrupt)", 0, 3000, cfg0["width"], 100)
    cfg["sensor_delta"] = st.slider("Sensor shift δ (σ units)", 0.5, 5.0, cfg0["sensor_delta"], 0.5)
    cfg["flip_rho"] = st.slider("Real drift: flip probability ρ", 0.1, 1.0, cfg0["flip_rho"], 0.1)
    cfg["model"]["T"] = st.slider("MC-Dropout samples T", 5, 100, cfg0["model"]["T"], 5)
    cfg["model"]["dropout"] = st.slider("Dropout rate p", 0.05, 0.5, cfg0["model"]["dropout"], 0.05)
    cfg["detectors"]["target_far"] = st.slider("Target false-alarm probability", 0.01, 0.3,
                                               cfg0["detectors"]["target_far"], 0.01)
    cfg["detectors"]["label_delay"] = st.slider("Label delay for supervised detectors", 0, 2000, 0, 50)
run_btn = st.sidebar.button("▶ Run experiment", type="primary", width="stretch")

# ============================================================================ header
st.title("📉 Bayesian uncertainty decomposition for early concept-drift warning")
st.caption("MC-Dropout / ensembles / bootstrap-SVM → aleatoric vs epistemic uncertainty → label-free drift "
           "alarms → validated with SHAP & LIME")

tabs = st.tabs(["🧭 Overview", "▶ Live experiment", "🔍 XAI validation", "📊 Research results",
                "📚 Theory", "♻ Reproducibility"])

# ============================================================================ overview
with tabs[0]:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("""
**Problem.** Deployed models drift. Classic detectors (DDM, ADWIN, Page-Hinkley) watch the *error rate*,
so they need labels — which in fraud, maintenance or churn arrive days to months late — and they fire only
*after* accuracy has fallen.

**Idea.** A Bayesian network knows *what it doesn't know*. Split predictive uncertainty into
**aleatoric** (noise in the data, AU) and **epistemic** (model ignorance, EU). When inputs move away from
the training data, EU grows *relative to* AU. Watch the ratio **log(EU/AU)** — no labels needed.

**Hypotheses** (each can be confirmed *or refuted*):
1. **H1 Early warning** — under drifts that change P(x), the label-free alarm precedes the visible accuracy drop.
2. **H2 Decomposition matters** — log(EU/AU) separates drifted data better than EU, AU, total entropy or 1−max p.
3. **H3 Boundary** — under *pure* P(y|x) drift, no label-free signal can work (expected refutation of H1).
4. **H4 XAI validation** — SHAP/LIME attributions of EU point at the features that really drifted;
   prediction explanations lose stability after the alarm.
5. **H5 Bayesian vs frequentist** — posterior (MC-Dropout, ensembles) vs bootstrap-SVM uncertainty.
""")
    with c2:
        st.graphviz_chart("""
digraph G { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#f5f4f0", color="#c3c2b7",
fontname="Helvetica", fontsize=11]; edge [color="#8a8984"];
D [label="Real dataset (UCI / IBM / Elec2)\\n+ SHA-256 provenance"];
I [label="Drift injection with ground truth\\ncovariate · sensor · real · none"];
M [label="Probabilistic model\\nMC-Dropout | Ensemble | Bootstrap-SVM"];
U [label="Decomposition  TU = AU + EU\\nsignal  log(EU/AU)"];
A [label="Label-free detectors\\nPH · ADWIN · BOCPD · EM-GMM"];
S [label="Supervised baselines\\nDDM · ADWIN · PH on errors"];
X [label="XAI on alarm samples\\nKernelSHAP · GradSHAP · LIME"];
R [label="Statistics\\nWilcoxon · TOST · block bootstrap\\nmeta-analysis (DerSimonian-Laird)"];
D -> I -> M -> U -> A -> X -> R; M -> S -> R; A -> R; }
""")
    st.info("Use the sidebar to run a live experiment on any dataset — including your own CSV — then open "
            "**XAI validation** to explain the samples that triggered the alarm.")

# ============================================================================ live experiment
with tabs[1]:
    if dataset == "upload" and (upload_bytes is None or not upload_ok):
        st.warning("Upload a CSV in the sidebar first." if upload_bytes is None else
                   "Choose a label column with 2–20 classes in the sidebar.")
    elif run_btn or st.session_state.get("last_key"):
        key = (dataset, drift, model, int(seed), json.dumps(cfg, sort_keys=True), target, json.dumps(sorted(drop_cols)))
        if run_btn:
            st.session_state["last_key"] = key
        key = st.session_state["last_key"]
        out = run_cached(key[0], key[1], key[2], key[3], key[4], upload_bytes if key[0] == "upload" else None,
                         key[5], key[6])
        st.session_state["last_out"] = out
        tr = out["traces"]
        stt = tr["stream"]
        rows = pd.DataFrame(out["rows"])
        t0, t_acc = tr["t0"], tr["t_acc"]
        L = len(tr["err"])
        roll = lambda x, w=200: pd.Series(x).rolling(w, min_periods=20).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reference accuracy", f"{tr['acc_ref']:.3f}")
        if t0 is not None:
            post = stt.p >= 1
            m2.metric("Accuracy after drift", f"{1 - tr['err'][post].mean():.3f}" if post.any() else "—",
                      delta=f"{(1 - tr['err'][post].mean()) - tr['acc_ref']:+.3f}" if post.any() else None)
            m3.metric("Visible accuracy drop at", f"t = {t_acc}" if t_acc is not None else "never")
            fa = rows.set_index("detector")
            first_lr = fa.loc["PH[lr]", "first_alarm"]
            m4.metric("PH[log(EU/AU)] alarm at", f"t = {int(first_lr)}" if pd.notna(first_lr) else "none",
                      delta=(f"{t_acc - int(first_lr):+d} samples before drop"
                             if pd.notna(first_lr) and t_acc is not None else None))
        else:
            m2.metric("Stream accuracy", f"{1 - tr['err'].mean():.3f}")
            m3.metric("Stream length", L)
            m4.metric("Alarms PH[lr]", len(tr["alarms"]["PH[lr]"]))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.24, 0.24, 0.2, 0.32],
                            subplot_titles=("Rolling accuracy (needs labels)", "Aleatoric vs epistemic uncertainty",
                                            "log(EU/AU) — proposed signal", "Alarms"))
        fig.add_trace(go.Scatter(y=roll(1 - tr["err"], 300), name="accuracy", line=dict(color=COL["ink"], width=1.6)), 1, 1)
        if tr["acc_thr"] is not None:
            fig.add_hline(y=tr["acc_thr"], line=dict(color=COL["red"], dash="dot", width=1), row=1, col=1)
        for k in ["eu", "au"]:
            fig.add_trace(go.Scatter(y=roll(tr["signals"][k]), name=k.upper(), line=dict(color=COL[k], width=1.6)), 2, 1)
        fig.add_trace(go.Scatter(y=roll(tr["signals"]["lr"]), name="log(EU/AU)", line=dict(color=COL["lr"], width=1.6)), 3, 1)
        dets = ["PH[lr]", "BOCPD[lr]", "ADWIN[lr]", "PH[eu]", "PH[tu]", "PH[conf]", "EM-GMM[au,eu]", "KS[x]",
                "PH[err]", "ADWIN[err]", "DDM[err]"]
        for i, d in enumerate(dets):
            sk = d[d.find("[") + 1:-1]
            c = COL["sup"] if d.endswith("[err]") else COL["ks"] if d.startswith("KS") else COL["em"] if d.startswith("EM") else COL.get(sk, "#888")
            a = tr["alarms"].get(d, [])
            fig.add_trace(go.Scatter(x=a, y=[d] * len(a), mode="markers", marker=dict(symbol="triangle-down", size=10, color=c),
                                     name=d, showlegend=False, hovertemplate=f"{d}<br>t=%{{x}}<extra></extra>"), 4, 1)
        for r in range(1, 5):
            if t0 is not None:
                fig.add_vrect(x0=t0, x1=t0 + max(stt.width, 1), fillcolor="#f3efe6", line_width=0, layer="below",
                              row=r, col=1)
                fig.add_vline(x=t0, line=dict(color="#8a8984", dash="dash", width=1), row=r, col=1)
            if t_acc is not None:
                fig.add_vline(x=t_acc, line=dict(color=COL["red"], width=1), row=r, col=1)
        fig.update_layout(height=820, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.06),
                          plot_bgcolor="white", hovermode="x unified")
        fig.update_xaxes(gridcolor="#eeeeea")
        fig.update_yaxes(gridcolor="#eeeeea")
        st.plotly_chart(fig, width="stretch")
        st.caption("Grey dashed line: true drift onset t0 · shaded: ramp · red line: visible accuracy drop "
                   "(trailing accuracy below reference − 3 s.e.) · ▼ alarms. Label-free detectors only use x.")

        show = rows[["detector", "family", "first_alarm", "delay", "lead", "n_false_alarms"]].copy()
        show = show.rename(columns={"first_alarm": "first alarm ≥ t0", "lead": "lead (>0 = before drop)",
                                    "n_false_alarms": "false alarms"})
        st.subheader("Detector scorecard")
        st.dataframe(show, width="stretch", hide_index=True)
        if out["signal_rows"]:
            sr = pd.DataFrame(out["signal_rows"])[["signal", "auroc", "cohen_d", "mean_pre", "mean_post"]]
            sr["signal"] = sr["signal"].map(SIGNAL_LABELS)
            st.subheader("Label-free separability of each signal (post-drift vs pre-drift)")
            st.dataframe(sr.round(3), width="stretch", hide_index=True)
        st.download_button("⬇ Download scorecard (CSV)", rows.to_csv(index=False), "scorecard.csv")
    else:
        st.info("Choose settings in the sidebar and press **▶ Run experiment**.")

# ============================================================================ XAI
with tabs[2]:
    out = st.session_state.get("last_out")
    if not out or out["traces"]["t0"] is None:
        st.info("Run a live experiment with an injected drift (not 'none'/'natural') first.")
    else:
        from bdx import xai as XAI
        tr = out["traces"]
        stt = tr["stream"]
        cand = [d for d in tr["alarms"] if not d.endswith("[err]")]
        c1, c2, c3 = st.columns(3)
        det = c1.selectbox("Alarm that defines the explained window", cand, index=cand.index("PH[lr]"))
        n_exp = c2.slider("Samples per window", 10, 60, 25, 5)
        use_lime = c3.checkbox("Include LIME (slower)", value=True)
        al = [a for a in tr["alarms"][det] if a >= stt.t0]
        alarm = al[0] if al else stt.t0 + stt.width // 2
        st.write(f"Explaining the {n_exp} highest-log(EU/AU) samples in **[{alarm}, {alarm + 300})** "
                 f"({'first alarm of ' + det if al else 'no alarm — mid-ramp fallback'}) against two pre-drift windows.")
        if st.button("🔍 Compute explanations", type="primary"):
            xcfg = copy.deepcopy(base_cfg())
            xcfg["xai"]["n_explain"] = n_exp
            with st.spinner("KernelSHAP / GradientSHAP / LIME …"):
                res, imps = XAI.explain_run(tr["model"], stt, tr["signals"], alarm, xcfg, use_lime=use_lime)
            st.session_state["xai"] = (res, imps)
        if "xai" in st.session_state:
            res, imps = st.session_state["xai"]
            names = stt.feature_names
            gt = set(stt.gt_features)
            ex = st.selectbox("Explainer", list(imps), format_func=lambda e: {
                "SHAP_EU": "KernelSHAP of epistemic uncertainty (agnostic)", "LIME_EU": "LIME of epistemic uncertainty",
                "SHAP_pred": "KernelSHAP of prediction (agnostic)", "GradSHAP_pred": "Gradient SHAP of prediction (model-specific)",
                "LIME_pred": "LIME of prediction"}.get(e, e))
            v = imps[ex]
            order = np.argsort(v["delta"])[::-1][:15]
            fig = go.Figure(go.Bar(x=v["delta"][order][::-1], y=[names[i] + ("  ★" if i in gt else "") for i in order][::-1],
                                   orientation="h", marker_color=[COL["red"] if i in gt else COL["lr"] for i in order][::-1]))
            fig.update_layout(height=460, title="Δ mean |attribution| (alarm window − pre-drift)  ·  ★ = truly drifted feature",
                              plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            k1, k2, k3 = st.columns(3)
            k1.metric("Precision@k", f"{res[ex + '_prec_at_k']:.2f}", help=f"chance = {res['chance_prec_at_k']:.2f}")
            k2.metric("Ground-truth rank score", f"{res[ex + '_gt_rank']:.2f}", help="1 = drifted features on top, 0.5 = chance")
            k3.metric("Rank stability ρ (pre→alarm)", f"{res[ex + '_rho_drift']:.2f}",
                      delta=f"{res[ex + '_rho_drift'] - res[ex + '_rho_null']:+.2f} vs pre→pre")
            with st.expander("All XAI metrics"):
                st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

# ============================================================================ research results
with tabs[3]:
    hyp = os.path.join(RES, "summary", "hypotheses.md")
    if not os.path.exists(hyp):
        st.warning("No precomputed results found. Run `python -m bdx.run_all && python -m bdx.analyze`.")
    else:
        st.markdown("Results of the full pre-registered grid (all datasets × drift types × 3 models × 5 seeds). "
                    "Regenerate with `python -m bdx.run_all && python -m bdx.analyze`.")
        figs = [("fig1_example_trace.png", "Example: a label-free alarm before the visible accuracy drop"),
                ("fig2_signal_auroc.png", "H2/H3 — separability of each signal by drift type"),
                ("fig3_tradeoff.png", "H1 — early warning vs false alarms"),
                ("fig4_meta_forest.png", "Meta-analysis across datasets"),
                ("fig5_xai.png", "H4 — XAI localisation and stability"),
                ("fig7_bayes_vs_freq.png", "H5 — Bayesian vs frequentist uncertainty"),
                ("fig6_elec2_leadlag.png", "Natural drift (Elec2) — does uncertainty lead the error?")]
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for c, (f, cap) in zip(cols, figs[i:i + 2]):
                p = os.path.join(FIG, f)
                if os.path.exists(p):
                    c.image(p, caption=cap, width="stretch")
        st.markdown(open(hyp).read())
        sdir = os.path.join(RES, "summary")
        with st.expander("All summary tables"):
            for f in sorted(os.listdir(sdir)):
                if f.endswith(".csv"):
                    st.markdown(f"**{f}**")
                    st.dataframe(pd.read_csv(os.path.join(sdir, f)), width="stretch", hide_index=True)

# ============================================================================ theory
with tabs[4]:
    st.markdown(r"""
### 1 · From likelihood to posterior
A classifier defines a likelihood $p(y\mid x,\theta)$. **Maximum likelihood** picks
$\hat\theta_{MLE}=\arg\max_\theta\sum_i\log p(y_i\mid x_i,\theta)$ — minimising cross-entropy *is* MLE.
Adding L2 weight decay $\lambda\|\theta\|^2$ is **MAP** with a Gaussian prior $\theta\sim\mathcal N(0,\sigma^2 I)$.
The **Bayesian** answer keeps the whole posterior $p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)p(\theta)$
and predicts with $p(y\mid x,\mathcal D)=\int p(y\mid x,\theta)\,p(\theta\mid\mathcal D)\,d\theta\approx\frac1T\sum_t p(y\mid x,\theta_t)$.

### 2 · MC-Dropout = approximate variational inference
Gal & Ghahramani (2016): training with dropout minimises a KL divergence to the posterior for a
Bernoulli variational family $q(\theta)$. Keeping dropout *on* at test time samples $\theta_t\sim q(\theta)$.

### 3 · Decomposition
$$\underbrace{\mathcal H\Big[\tfrac1T\sum_t p_t\Big]}_{\text{total}}=\underbrace{\tfrac1T\sum_t\mathcal H[p_t]}_{\text{aleatoric}}+\underbrace{\mathcal I(y;\theta\mid x,\mathcal D)}_{\text{epistemic}}$$
EU (mutual information) is large when sampled models **disagree** — typical of inputs unlike the training data.
AU is large when every model agrees the classes **overlap**. Proposed signal: $\log(EU+\epsilon)-\log(AU+\epsilon)$.

### 4 · Fisher (frequentist) vs Bayes
Frequentist uncertainty comes from the *sampling distribution* of an estimator — here, the bootstrap:
refit an SVM on resampled data and look at the spread. Bayesian uncertainty is a *posterior over parameters*.
Both produce a set of predictive distributions, so the same decomposition applies, but they mean different things.

### 5 · Why this cannot see every drift
EU and AU are functions of $x$ only. If $P(x)$ stays fixed and only $P(y\mid x)$ changes (real concept drift),
the distribution of any label-free signal is **unchanged** — no detector on it can do better than chance.
That is a falsifiable boundary, tested as H3.

### 6 · Detectors
* **Page-Hinkley**: cumulative sum $m_t=\sum(x_i-\bar x_i-\delta)$; alarm when $m_t-\min m>\lambda$.
* **ADWIN**: adaptive window; cuts when two sub-window means differ beyond a Hoeffding bound.
* **BOCPD** (Bayesian): exact recursive posterior over the run length $r_t$ with a Normal-Inverse-Gamma prior;
  alarm when $P(r_t<4\mid x_{1:t})$ is high.
* **EM-GMM**: mixture density of (log AU, log EU) fitted by Expectation-Maximisation; alarm when a batch is unlikely.

### 7 · XAI
SHAP gives each feature its Shapley value — its average marginal contribution over all coalitions.
LIME fits a local linear surrogate around one sample. We explain **EU itself**: if the alarm is about a real
input shift, the features whose attribution grows most should be the ones that drifted.

### 8 · Statistics on dependent data
Samples in a stream are autocorrelated, so tests use **independent runs (seeds)** as units, the **moving-block
bootstrap** inside a stream, **TOST** to show equivalence to chance, Holm correction, and a **DerSimonian-Laird**
random-effects meta-analysis across datasets.
""")

# ============================================================================ reproducibility
with tabs[5]:
    st.markdown("""
```bash
git clone <your-repo-url> && cd bayes-drift-xai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q                                   # unit tests of the from-scratch components
python -m bdx.run_all                       # full grid -> results/*.csv   (python -m bdx.run_all --quick for a 5-min check)
python -m bdx.analyze                       # tables, hypothesis tests, figures
streamlit run streamlit_app.py
```
Every number in the paper comes from `config/default.yaml` + the seeds in it. Data files are verified by SHA-256.
""")
    rows = [dict(key=k, dataset=v.name, file=v.file, sha256=v.sha256, licence=v.licence, citation=v.citation)
            for k, v in D.SPECS.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    mp = os.path.join(RES, "manifest.json")
    if os.path.exists(mp):
        with st.expander("Run manifest (versions, config hash)"):
            st.json(json.load(open(mp)))
    st.download_button("⬇ config/default.yaml", open(os.path.join(ROOT, "config", "default.yaml")).read(),
                       "default.yaml")
