"""bdx — Bayesian uncertainty Decomposition for concept-drift detection with XAI validation.

Modules
-------
data         dataset loading, provenance (SHA-256), preprocessing, synthetic streams
drift        controlled drift injection (covariate / sensor / real / none) with ground truth
models       MC-Dropout MLP, Deep Ensemble, Bootstrap-SVM (frequentist counterpart)
uncertainty  entropy decomposition: total = aleatoric + epistemic (mutual information)
detectors    River baselines (DDM, ADWIN, Page-Hinkley) + own BOCPD, EM-GMM, KS detectors
xai          SHAP / LIME on predictions and on epistemic uncertainty; stability & localization
stats        Wilcoxon, (block) bootstrap CIs, DerSimonian-Laird meta-analysis, lead-lag tests
experiment   one end-to-end run: data -> stream -> model -> signals -> detectors -> metrics
"""

__version__ = "1.0.0"
