# Bayesian uncertainty decomposition for early concept-drift warning, validated with XAI

> Does **epistemic** uncertainty rise *before* a deployed model's accuracy visibly falls — and can SHAP/LIME
> confirm that the alarm is about the features that really drifted?

A reproducible research pipeline + Streamlit app. A probabilistic classifier (MC-Dropout network, deep ensemble,
or bootstrap-SVM) splits its predictive uncertainty into **aleatoric** (AU, data noise) and **epistemic**
(EU, model ignorance). The ratio **log(EU/AU)** is monitored *without labels* by Page-Hinkley, ADWIN, a
from-scratch **Bayesian online changepoint detector** and a from-scratch **EM Gaussian-mixture** detector, and
compared with label-based detectors (DDM, ADWIN, Page-Hinkley on the error stream). Alarms are validated by
explaining **EU itself** with KernelSHAP, GradientSHAP and LIME against the known drifted features.

**For whom:** MLOps teams running models on non-stationary data (fraud/credit, predictive maintenance,
customer churn) where labels arrive late.

---

## 1. Research question, hypotheses (pre-registered) and objectives

| | Hypothesis | Test | Can be refuted? |
|---|---|---|---|
| **H1** | Under drift that changes P(x), a label-free alarm on log(EU/AU) fires **before** the visible accuracy drop | Wilcoxon lead > 0; paired delay vs DDM/ADWIN/PH on errors; meta-analysis | yes |
| **H2** | log(EU/AU) separates drifted from pre-drift data better than EU, AU, total entropy, 1−max p | paired Wilcoxon on AUROC and on detection delay, Holm | yes |
| **H3** | Under **pure P(y\|x)** drift, *no* label-free uncertainty signal carries information (boundary of H1) | TOST equivalence of AUROC to 0.5 (±0.05); Fisher test vs stationary streams | yes — predicts a *refutation* of H1 here |
| **H4** | SHAP/LIME attributions of EU on alarm samples rank the truly drifted features on top; prediction explanations lose rank stability after the alarm | Wilcoxon of ground-truth rank > 0.5, binomial precision@k vs chance; ρ(pre,alarm) < ρ(pre,pre) | yes |
| **H5** | Bayesian (posterior) and frequentist (bootstrap) uncertainty behave differently as drift monitors | AUROC by model family | exploratory |

**Objectives (5):** (O1) build a reusable uncertainty-monitoring tool; (O2) create semi-synthetic drift streams
with ground truth from real data; (O3) test H1–H3 statistically across datasets; (O4) validate alarms with XAI
(H4); (O5) compare Bayesian vs frequentist uncertainty (H5).

## 2. Originality — what is new and the concrete source it builds on

* **Closest prior work:** *Uncertainty Drift Detection* (UDD) — Baier, Schlör, Schöffer & Kühl, HICSS 2023,
  [arXiv:2107.01873](https://arxiv.org/abs/2107.01873): MC-Dropout *total* uncertainty + ADWIN, no labels.
* **Decomposition theory:** Depeweg et al. (ICML 2018); Kendall & Gal (NeurIPS 2017); Hüllermeier &
  Waegeman (Machine Learning, 2021).
* **What this project adds** (we did not find these combined in the literature we checked):
  1. the **decomposed ratio log(EU/AU)** as the monitored statistic, compared head-to-head with its components;
  2. a **falsifiable boundary** (H3): a proof sketch + experiment that label-free signals cannot see pure P(y|x) drift;
  3. **XAI applied to the epistemic uncertainty function** (not only to predictions) and scored against
     **ground-truth drifted features**;
  4. a **Bayesian vs frequentist** comparison in this setting, with two detectors written from first principles
     (BOCPD with a Normal-Inverse-Gamma prior, EM-fitted Gaussian mixture with BIC);
  5. **semi-synthetic drift streams** generated from real records, giving known onset times so *lead time* can be measured.

## 2b. Results at a glance (full grid: 225 runs × 20 detectors, 30 XAI runs, 15 Elec2 runs)

| | Verdict | Key numbers |
|---|---|---|
| **H1** early warning | **not confirmed** (strong form) | PH[log(EU/AU)] fires 99–268 samples *earlier than every label-based detector* (paired p ≤ 0.007), but precedes the visible accuracy drop in only 51 % of runs (median lead 30, 95 % CI −25…151, Holm p = 0.24) |
| **H2** ratio beats components | **confirmed** | AUROC 0.80 vs EU 0.67, TU 0.53, AU 0.50 (Holm p < 0.001, 8/8 tests) |
| **H3** P(y\|x) boundary | **confirmed** (H1 refuted there, as predicted) | all signals AUROC ≈ 0.50, TOST p < 0.001; supervised detectors catch 100 % |
| **H4** XAI validation | **confirmed for SHAP, refuted for LIME** | SHAP rank of drifted features 0.78–0.92 (Holm p ≤ 0.03); LIME ≈ 0.5 |
| **H5** Bayes vs frequentist | **ratio advantage only for Bayesian NNs** | MC-Dropout/ensemble: LR > EU; bootstrap-SVM: EU 0.78 > LR 0.61 — ReLU nets get *over-confident* off-data (AU↓, EU↑), RBF-SVMs revert to the prior (AU↑, EU↑) |
| Elec2 (natural) | co-movement, no lead | corr(LR, error) = 0.30 at lag 0, symmetric; KS on inputs raises 148 alarms vs 7 for PH[LR] |

Honest caveat: on the clean synthetic drifts a plain feature-wise KS test is a better early-warning detector;
on the natural stream it is unusable. Full report: `results/summary/hypotheses.md`; article: `docs/article/`.

## 3. Repository layout
```
bdx/                 the tool (importable package)
  data.py            loaders, SHA-256 provenance, CSV upload support, SEA & digits
  drift.py           drift injection with ground truth (none / covariate / sensor / real)
  models.py          MC-Dropout MLP, Deep Ensemble, Bootstrap-SVM — common predict_samples() API
  uncertainty.py     TU = AU + EU decomposition, log-ratio signal
  detectors.py       River baselines + own BOCPD, own EM-GMM, KS; threshold calibration
  xai.py             KernelSHAP / GradientSHAP / LIME on EU and on predictions; localisation & stability
  stats.py           Wilcoxon, bootstrap & moving-block bootstrap, Holm, TOST, DerSimonian-Laird, Granger
  experiment.py      one end-to-end run
  run_all.py         the full experimental grid  -> results/*.csv
  analyze.py         hypothesis tests, tables, figures -> results/summary/, figures/
config/default.yaml  every parameter of the paper (single source of truth)
data/raw/            the exact data files (verified by checksum); data/DATASETS.md
results/             raw results + summary tables + hypotheses.md (auto-generated)
figures/             paper figures (auto-generated)
docs/GUIDE.md        study guide: theory -> implementation, line by line
docs/article/        the paper (Romanian): articol.md source, articol.docx, articol.pdf, build.py
tests/               unit tests of the from-scratch components
streamlit_app.py     interactive interface
```

## 4. Run it (VS Code)

1. Install **Python 3.13** (the version all results were produced with) and VS Code with the *Python* extension. Open the folder in VS Code.
2. Terminal (`Ctrl+\``):
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   pytest -q                         # 8 tests, ~20 s
   python -m bdx.run_all --quick     # 5-minute sanity run (2 seeds, MC-Dropout only) into results/
   python -m bdx.run_all             # full grid (≈ 45-60 min on a 2-core laptop)
   python -m bdx.analyze             # tables, tests, figures
   python docs/article/build.py      # rebuild the article (needs pandoc; PDF needs xelatex)
   streamlit run streamlit_app.py    # opens http://localhost:8501
   ```
   In VS Code pick the interpreter `.venv` (`Ctrl+Shift+P` → *Python: Select Interpreter*).

### Using other data (reproducibility check)
In the app choose **📤 Upload your own CSV**, pick the label column, a drift type, and run. In code:
```python
from bdx import data as D, drift as DR
from bdx.experiment import run_one
import yaml
cfg = yaml.safe_load(open("config/default.yaml"))
tab = D.load_csv("my.csv", target="label")
st = DR.build_stream(tab, "covariate", seed=0, cfg=cfg)
out = run_one("my", "covariate", "mcd", 0, cfg, stream=st)
```

## 5. Deploy on Streamlit Community Cloud
1. Create a GitHub repository and push this folder (VS Code: *Source Control* → *Publish to GitHub*, or
   `git init && git add . && git commit -m "initial" && git branch -M main && git remote add origin <url> && git push -u origin main`).
2. Go to <https://share.streamlit.io> → **Create app** → choose the repo, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings → Python 3.13.** Deploy. The first build takes a few minutes (CPU-only PyTorch is
   pulled through the extra index in `requirements.txt`).
4. The *Research results* tab reads the committed `results/` and `figures/`, so run the grid locally and commit them first.

## 6. Reproducibility checklist (Sandve et al., 2013)
* every result produced by code (`run_all` → `analyze`), no manual steps;
* all parameters in one YAML file; `results/manifest.json` stores the config hash, package versions and data checksums;
* all randomness seeded (NumPy, PyTorch, River, SHAP, LIME, scikit-learn);
* data files version-controlled and checksum-verified; exact package versions pinned;
* raw results kept (`results/*.csv`), so every figure can be regenerated without re-training.

## 7. Licence
Code: MIT. Data: see `data/DATASETS.md` for each dataset's licence and citation.
