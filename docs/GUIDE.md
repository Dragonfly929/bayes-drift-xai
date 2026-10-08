# Study guide — from theory to implementation

This guide walks you through the project in the order you will need to *defend* it: the problem,
the theory, each design decision and the file that implements it, the statistics, and the questions you
are likely to get at the exam. Read it with the code open in VS Code.

---

## Part A — The problem in one page

A model is trained once on historical data and then deployed. The world keeps changing:
new customer segments appear, a sensor is recalibrated, fraudsters change tactics. This is **concept drift**:
the joint distribution P_t(x, y) changes over time t.

Write P(x, y) = P(x) · P(y | x). Two very different things can change:

| change | name | example | can you see it without labels? |
|---|---|---|---|
| P(x) | *virtual* / covariate drift | a new region of customers starts applying | **yes**, inputs look different |
| P(y \| x) | *real* concept drift | same customers, but the economy changes who defaults | **no**, inputs look the same |
| both | mixed | a sensor drifts *and* the relation to failure changes | partly |

Classic detectors (DDM, ADWIN, Page-Hinkley) monitor the **error rate**. They need the true label of every
prediction. In credit, labels arrive after months; in fraud after a chargeback; in maintenance after a failure.
And they react only *after* accuracy has dropped.

**Our question:** can the model's own *uncertainty* — computed from x alone — warn us earlier? And which
*kind* of uncertainty?

---

## Part B — Theory

### B1. Likelihood, MLE, MAP, Bayes (the course's "verosimilitate" thread)

A classifier with parameters θ says: given x, y follows a categorical distribution with probabilities
p(y | x, θ) = softmax(f_θ(x)). For data D = {(x_i, y_i)} the **likelihood** is

    L(θ) = Π_i p(y_i | x_i, θ),      log L(θ) = Σ_i log p(y_i | x_i, θ).

* **Maximum likelihood (MLE, Fisher):** θ̂ = argmax log L(θ). Minimising cross-entropy *is* exactly this —
  `nn.CrossEntropyLoss` in `bdx/models.py::_train` is the negative log-likelihood.
* **MAP:** add a prior p(θ) = N(0, σ²I). Then argmax [log L(θ) + log p(θ)] = argmin [NLL + λ‖θ‖²].
  That λ is `weight_decay` in `config/default.yaml`. So our "ordinary" training is already a Bayesian point estimate.
* **Full Bayes:** don't pick one θ; keep the posterior p(θ | D) ∝ L(θ) p(θ), and predict by averaging:

      p(y | x, D) = ∫ p(y | x, θ) p(θ | D) dθ ≈ (1/T) Σ_t p(y | x, θ_t),   θ_t ~ p(θ | D).

The integral is intractable for a neural network, so we approximate the posterior.

### B2. MC-Dropout as approximate Bayesian inference

Dropout switches off each hidden unit with probability p during training. Gal & Ghahramani (2016) showed that
this training objective is a variational approximation: it minimises KL(q(θ) ‖ p(θ | D)) for a family q in
which each weight row is either kept or zeroed. Keeping dropout **on at test time** samples θ_t ~ q(θ).

Implementation detail that matters (`MCDropoutMLP.fit`): we draw **T fixed masks** once. Each mask defines one
network θ_t. Every input is pushed through the same T networks. Consequences:
1. the predictive distribution is a proper mixture over T networks;
2. EU(x) becomes a deterministic function of x — necessary to explain it with SHAP/LIME.

Alternatives in the code: **Deep Ensemble** (M networks from different random initialisations,
Lakshminarayanan et al. 2017) and the **frequentist** counterpart **Bootstrap-SVM** (B SVMs on bootstrap
resamples; spread = sampling variability of the estimator, *not* a posterior). Same API: `predict_samples(X)`
returns an array (S, N, C).

### B3. Splitting uncertainty: aleatoric vs epistemic

With S sampled predictive vectors p_1..p_S for an input x (`bdx/uncertainty.py::decompose`):

    TU(x) = H[ mean_s p_s ]            total uncertainty (entropy of the average)
    AU(x) = mean_s H[ p_s ]            aleatoric: average entropy of each model
    EU(x) = TU − AU = I(y; θ | x, D)   epistemic: mutual information between label and parameters

Intuition with two models and a binary label:
* both say (0.5, 0.5) → each model is unsure, they agree → **AU high, EU = 0** (genuinely noisy region);
* one says (1, 0), the other (0, 1) → each is confident, they disagree → **AU = 0, EU = log 2** (we don't know).

EU ≥ 0 always (entropy is concave → Jensen). The unit tests in `tests/test_core.py` check exactly these
three facts.

**Why the ratio?** When inputs move into a region the model never saw, models disagree → EU rises. But AU can
move too (e.g. a neural network extrapolates *over-confidently*, so AU and 1−max p may *fall*). Watching
log(EU/AU) asks "is ignorance growing *relative to* noise?" — it rises when EU goes up, and also when AU goes
down while EU stays put, which is the signature of confident extrapolation. ε = 1e-4 avoids log(0).

### B4. Why no label-free signal can detect pure P(y|x) drift (the boundary, H3)

Any signal s_t = g(x_t) computed from inputs only has distribution P_t(s) = ∫ 1[g(x)=s] P_t(x) dx.
If P_t(x) does not change, P_t(s) does not change, whatever g is — so *every* detector on s is at chance.
This is a short proof, and we test it experimentally (credit/telco "real" drift and SEA). Showing that a
hypothesis **fails where theory says it must** is exactly the "confirm or refute" your professor asked for.

### B5. Detectors

* **Page-Hinkley** (Page 1954): m_t = Σ (x_i − mean_i − δ); alarm when m_t − min_{i≤t} m_i > λ.
* **ADWIN** (Bifet & Gavaldà 2007): keeps a variable-length window; cuts it when two sub-windows have
  significantly different means (Hoeffding bound with confidence δ).
* **DDM** (Gama et al. 2004): treats errors as Bernoulli; alarm when p_t + s_t > p_min + 3 s_min.
* **BOCPD** (Adams & MacKay 2007) — *Bayesian*, written from scratch in `detectors.py::BOCPD`.
  Let r_t be the run length (time since the last change). With a constant hazard H, the posterior is updated
  recursively:

      P(r_t = r+1, x_1:t) = P(r_{t-1} = r, x_1:t-1) · π(x_t | r) · (1 − H)       growth
      P(r_t = 0,  x_1:t)  = Σ_r P(r_{t-1} = r, x_1:t-1) · π(x_t | r) · H          changepoint

  π is the posterior predictive of the segment's data: with a **Normal-Inverse-Gamma** conjugate prior on
  (mean, variance) it is a Student-t, and the parameters update in closed form (μ, κ, α, β). We alarm when
  P(r_t < 4 | data) > threshold. It runs on batch means (50 samples) so the Gaussian assumption is
  reasonable (central limit theorem).
* **EM-GMM** — *Expectation-Maximisation*, from scratch in `detectors.py::GaussianMixtureEM`.
  Fit a K-component Gaussian mixture to (log AU, log EU) of calibration data:
  E-step computes responsibilities r_ik; M-step re-estimates weights, means, covariances;
  the log-likelihood never decreases (tested). K is chosen by **BIC** = −2 log L + p log n.
  Drift score of a batch = its average negative log-likelihood under this reference density.
* **KS on inputs** — the standard input-space baseline (Rabanser et al. 2019): feature-wise two-sample
  Kolmogorov-Smirnov tests with Bonferroni correction.

**Fair calibration.** Each detector's threshold is chosen on bootstrap streams drawn from the calibration set so
that P(any false alarm on 2500 stationary samples) ≤ 10 % (`detectors.py::calibrate`). Without that, a
"faster" detector might just be a more trigger-happy one.

### B6. Explainability (XAI) as validation

* **Shapley values / SHAP** (Lundberg & Lee 2017): attribution φ_j = average marginal contribution of
  feature j over all subsets of other features. **KernelSHAP** estimates it by weighted linear regression
  on random coalitions — *model-agnostic*. **GradientSHAP / expected gradients** uses the network's gradients
  — *model-specific* ("diagnostic"). 
* **LIME** (Ribeiro et al. 2016): sample perturbations around x, fit a weighted linear model; coefficients = local importance.

**The twist:** we explain the function x ↦ EU(x), not only x ↦ p(y | x). If the alarm is caused by a genuine
input shift, the features whose EU-attribution grows most between the pre-drift window and the alarm window
should be the features we shifted. We know them (ground truth), so we can *score* the explanation:
precision@k and a normalised rank (1 = top, 0.5 = random). Stability: Spearman ρ between importance vectors
before vs after the alarm, compared with ρ between two pre-drift windows (the noise floor).

### B7. Statistics on dependent data

Your professor stressed that classical tests assume **random and independent** observations. Samples in a
stream are neither (overlapping windows, autocorrelated signals). We handle it in three ways:
1. the **unit of analysis is an independent run** (a seed draws new train/stream samples and a new model),
   so tests across runs are valid — Wilcoxon signed-rank (paired, non-parametric), Fisher exact, binomial;
2. inside a single time series (Elec2) we use the **moving-block bootstrap** (Künsch 1989): resample blocks
   of consecutive observations so the dependence inside blocks is preserved;
3. results from several datasets are pooled by a **random-effects meta-analysis** (DerSimonian & Laird 1986):
   each dataset is a "study", the pooled effect accounts for between-study variance τ², and I² tells how
   heterogeneous the datasets are.
Multiple comparisons inside a hypothesis are corrected with **Holm**. To *support absence of an effect*
(H3) we use **TOST equivalence testing** instead of "p > 0.05" (absence of evidence ≠ evidence of absence).

---

## Part C — The experimental design, decision by decision

| decision | value | why |
|---|---|---|
| training size | 3000 | realistic deployed model; keeps CPU time low |
| calibration set | 1000 | z-scoring, thresholds and reference accuracy, never used for training |
| stream | 6000, onset t0 = 2500, linear ramp 1500 | gradual drift is where "early" matters |
| covariate drift | top-25 % of a 2-feature score becomes a new segment | genuine labels; P(y\|x) untouched |
| sensor drift | +2σ shift on 2 informative features | instrument fault / pipeline change |
| real drift | labels flip with prob. 0.7 in half of one feature's range | P(x) untouched |
| models | MC-Dropout (T=30, p=0.2), Ensemble (M=5), Bootstrap-SVM (B=10) | Bayesian vs frequentist |
| seeds | 5 | independent replications |
| visible accuracy drop | trailing 300-sample accuracy < reference − 3 s.e. | what a dashboard would show |
| lead time | t_drop − t_alarm | > 0 means warning came first |

---

## Part D — Reading the results

After `python -m bdx.analyze` open `results/summary/hypotheses.md`. For each hypothesis it prints the
effect size, the p-value (Holm-corrected) and a verdict. The figures in `figures/`:

1. **fig1** — one stream: the accuracy curve, EU/AU, log-ratio and every detector's alarms. Your best
   single picture for the presentation.
2. **fig2** — AUROC of every signal by drift type. Bars above 0.5 = the signal moves after drift.
   Real-drift bars sit on 0.5 (H3).
3. **fig3** — early-warning rate against spurious detections on stationary streams: the trade-off.
4. **fig4** — meta-analysis forest plot: one line per dataset×drift, the pooled diamond at the bottom.
5. **fig5** — XAI localisation (left) and stability (right).
6. **fig6** — Elec2: correlation between the signal now and the error k blocks later.
7. **fig7** — the same signals for Bayesian vs frequentist models.

### D2. What we actually found — and how to explain each result

**H2 confirmed: the ratio is the best uncertainty signal (for neural nets).** Median AUROC on P(x) drifts:
log(EU/AU) 0.80, EU 0.67, TU 0.53, AU 0.50, 1−max p 0.53. *Why?* Look at fig1: when credit customers from an
unseen segment arrive, **AU goes down** and EU goes up. The log-ratio is log EU − log AU, so it *adds* the two
movements. TU = AU + EU barely moves because the two cancel — that's why plain entropy and 1−max p are useless here.

**H5: but only for Bayesian neural nets — the most interesting finding.** For the bootstrap SVM, EU (0.78)
beats the ratio (0.61). The mechanism (say this at the exam, it shows understanding):
* a ReLU network is piecewise *linear*: far from the data its logits keep growing → it becomes **over-confident**
  (Hein et al., CVPR 2019) → low entropy for each sampled network → **AU ↓**; different networks extrapolate in
  different directions → **EU ↑**;
* an RBF kernel decays to 0 far from the support vectors → the SVM decision tends to its bias → Platt
  probability tends to a moderate constant, *the same for all bootstrap members* → **AU ↑** and EU ↑ together →
  the ratio cancels them.
Lesson: *what "aleatoric" and "epistemic" mean in practice depends on how the model extrapolates.*

**H1 not confirmed (strong form) — and that's fine.** PH[log(EU/AU)] fires significantly *earlier than all three
label-based detectors* (median 99–268 samples), but it beats the *visible accuracy drop* in only 51 % of runs
(median lead 30, CI −25…151, Holm p = 0.24). A pre-registered hypothesis that fails is a legitimate scientific
result; the professor asked for "confirmare *sau infirmare*". Two honest extra findings:
* a plain KS test on the inputs does better on our synthetic drifts (early in 57–100 % of cases);
* "harmless drift": on telco covariate the model never loses accuracy, yet every label-free detector fires.

**H3 confirmed: the boundary.** Under pure P(y|x) drift every signal has AUROC 0.50 (TOST equivalence p < 0.001)
while DDM/ADWIN/PH on errors catch 100 %. This is the proposition in the paper: if P(x) doesn't change, nothing
computed from x can change. *Practical message:* label-free monitoring cannot fully replace labels.

**H4: SHAP yes, LIME no.** KernelSHAP and GradientSHAP put the truly drifted features near the top
(normalised rank 0.78–0.92, precision@k up to 0.65 vs 0.08 by chance) and their rankings become unstable after
the alarm. LIME stays at chance and its rankings barely change (ρ ≈ 0.99): with default settings it samples
perturbations from the *training* distribution with a wide kernel, so its "local" surrogate is effectively global.

**Elec2 (real stream).** The ratio co-moves with the error rate (r = 0.30) but doesn't lead it (symmetric
cross-correlation, Granger significant in only 20 % of runs). The KS test raises 148 alarms on 15 000 samples
vs 7 for PH[LR] — on real, seasonal data input tests drown you in alarms. This is the best argument *for*
monitoring uncertainty instead of raw inputs.

**Meta-analysis.** Pooled delay advantage over PH[error] = 1 083 samples but CI [−538; 2 703], I² = 99.7 %: the
effect is real in some domains and reversed in others (digits). Report it as heterogeneous, not as a win.

---

## Part E — Likely exam questions and short answers

**"What is the originality, and what is the concrete source?"**
The closest work is Baier et al. (HICSS 2023, *Uncertainty Drift Detection*): MC-Dropout uncertainty + ADWIN.
They monitor *total* uncertainty. We (1) decompose it and monitor log(EU/AU), (2) prove and test where any
label-free method must fail, (3) validate alarms by explaining the epistemic-uncertainty function against known
drifted features, (4) compare Bayesian and frequentist uncertainty and add two first-principles detectors
(BOCPD, EM-GMM). The data contribution: semi-synthetic drift streams with ground truth generated from real records.

**"Why not just use a library?"** We use libraries for standard pieces (PyTorch, River, SHAP), but the
decomposition-based monitor, MC-Dropout with fixed masks for explainable EU, BOCPD, EM-GMM, threshold
calibration, drift injection with ground truth, and the statistical layer are implemented in this repository.

**"Is your data legal and how did you get it?"** See `data/DATASETS.md` — public licences (CC BY 4.0,
Apache-2.0, research use), checksums, citations.

**"Your observations aren't independent."** Correct — that's why runs, not samples, are the unit of
inference, and why Elec2 uses a block bootstrap.

**"Is MC-Dropout really Bayesian?"** It is a variational approximation with a restrictive family; it
underestimates uncertainty. That's why we also report deep ensembles and a frequentist bootstrap.

**"Is the entropy decomposition sound?"** It is the standard one (Depeweg 2018) but has known critiques
(Wimmer et al., UAI 2023): mutual information can be small for strongly disagreeing models in some cases.
We mention this as a limitation.

**"Your early-warning hypothesis was not confirmed — so what is the contribution?"**
Three things hold up: (1) the ratio is the strongest uncertainty signal for neural nets and we explain *why*
(opposite movement of AU and EU); (2) we show — and prove — where any label-free monitor must fail; (3) SHAP on EU
correctly locates the drifted features, LIME does not. Plus an honest negative result with pre-registered tests,
which is exactly what prevents the field from over-claiming.

**"Where are the EM algorithm, maximum likelihood, Bayes vs Fisher, SVM, deep nets, XAI in your project?"**
MLE/MAP = training loss + weight decay (`models.py`); Bayesian posterior = MC-Dropout & ensembles; Fisher /
frequentist = bootstrap-SVM; EM = Gaussian mixture on (log AU, log EU) in `detectors.py`; Bayesian inference
again in BOCPD; deep model-specific ("diagnostic") XAI = GradientSHAP, model-agnostic = KernelSHAP & LIME.

**"What would make it fail?"** Pure P(y|x) drift (shown), drifts inside the training support,
overconfident models whose EU stays low, and very small label noise where AU ≈ 0 (ratio unstable → ε).

---

## Part F — Glossary (EN → RO)

| English | Română |
|---|---|
| concept drift | deviere conceptuală |
| aleatoric / epistemic uncertainty | incertitudine aleatorică / epistemică |
| likelihood, maximum likelihood | verosimilitate, verosimilitate maximă |
| prior / posterior | distribuție a priori / a posteriori |
| mutual information | informație mutuală |
| false alarm, detection delay, lead time | alarmă falsă, întârziere de detecție, avans (timp de avertizare) |
| changepoint | punct de schimbare |
| bootstrap, block bootstrap | bootstrap, bootstrap pe blocuri |
| equivalence test (TOST) | test de echivalență (TOST) |
| random-effects meta-analysis | meta-analiză cu efecte aleatoare |
| feature attribution | atribuirea importanței caracteristicilor |
