"""Probabilistic classifiers that return *samples* of the predictive distribution.

All models expose ``predict_samples(X) -> array (S, N, C)``: S draws of class-probability vectors.
That common interface is what lets the same decomposition (``uncertainty.py``) be applied to

* ``MCDropoutMLP``   approximate Bayesian NN. Dropout at test time = sampling weights from a variational
                     posterior q(theta) (Gal & Ghahramani, 2016). Training minimises cross-entropy + L2,
                     i.e. a MAP estimate under a Gaussian prior (the MLE when weight decay = 0).
* ``DeepEnsemble``   M independently initialised MLPs (Lakshminarayanan et al., 2017) — a non-parametric
                     approximation of the posterior's modes.
* ``BootstrapSVM``   the *frequentist* counterpart: B RBF-SVMs fitted on bootstrap resamples, each with
                     Platt-scaled probabilities. Spread across members approximates the sampling
                     distribution of the estimator, not a posterior.

Masks for MC-Dropout are drawn explicitly and are shared across the batch inside one pass, so every pass
is one fixed network theta_s ~ q(theta). This makes epistemic uncertainty a deterministic function of x,
which is required to explain it with SHAP/LIME.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC

torch.set_num_threads(max(1, torch.get_num_threads()))


class _Net(nn.Module):
    def __init__(self, d_in, n_classes, hidden=(64, 64)):
        super().__init__()
        dims = (d_in,) + tuple(hidden)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(dims[-1], n_classes)

    def forward(self, x, masks=None, p=0.0):
        """masks: list (per hidden layer) of tensors (h,) with values in {0, 1/(1-p)} or None."""
        for i, lin in enumerate(self.layers):
            x = torch.relu(lin(x))
            if masks is not None:
                x = x * masks[i]
            elif p > 0 and self.training:
                x = nn.functional.dropout(x, p, training=True)
        return self.out(x)


def _train(net, X, y, p, epochs, lr, weight_decay, batch, seed):
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()           # negative log-likelihood of a categorical model
    net.train()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[b], p=p), yt[b])
            loss.backward()
            opt.step()
    net.eval()
    return net


class MCDropoutMLP:
    kind = "bayesian"

    def __init__(self, n_classes, hidden=(64, 64), p=0.2, epochs=60, lr=1e-3, weight_decay=1e-4,
                 batch=128, T=30, seed=0):
        self.C, self.hidden, self.p, self.epochs = n_classes, hidden, p, epochs
        self.lr, self.wd, self.batch, self.T, self.seed = lr, weight_decay, batch, T, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.net = _train(_Net(X.shape[1], self.C, self.hidden), X, y, self.p, self.epochs,
                          self.lr, self.wd, self.batch, self.seed)
        g = torch.Generator().manual_seed(self.seed + 7)
        keep = 1.0 - self.p
        # T fixed sub-networks theta_1..theta_T ~ q(theta)
        self.masks = [[(torch.rand(h, generator=g) < keep).float() / keep for h in self.hidden]
                      for _ in range(self.T)]
        return self

    @torch.no_grad()
    def predict_samples(self, X):
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        out = [torch.softmax(self.net(Xt, masks=m), -1) for m in self.masks]
        return torch.stack(out).numpy()

    @torch.no_grad()
    def predict_deterministic(self, X):
        """Standard (weight-scaled) prediction without dropout — the MAP network."""
        return torch.softmax(self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32)), -1).numpy()

    def torch_mean_model(self):
        """A differentiable nn.Module returning the MC-averaged probabilities (for GradientSHAP)."""
        parent = self

        class Mean(nn.Module):
            def forward(self, x):
                return torch.stack([torch.softmax(parent.net(x, masks=m), -1) for m in parent.masks]).mean(0)
        return Mean()


class DeepEnsemble:
    kind = "bayesian"

    def __init__(self, n_classes, M=5, hidden=(64, 64), epochs=60, lr=1e-3, weight_decay=1e-4,
                 batch=128, seed=0):
        self.C, self.M, self.hidden, self.epochs = n_classes, M, hidden, epochs
        self.lr, self.wd, self.batch, self.seed = lr, weight_decay, batch, seed

    def fit(self, X, y):
        self.nets = []
        for m in range(self.M):
            torch.manual_seed(self.seed * 100 + m)
            self.nets.append(_train(_Net(X.shape[1], self.C, self.hidden), X, y, 0.0, self.epochs,
                                    self.lr, self.wd, self.batch, self.seed * 100 + m))
        return self

    @torch.no_grad()
    def predict_samples(self, X):
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return torch.stack([torch.softmax(n(Xt), -1) for n in self.nets]).numpy()


class BootstrapSVM:
    kind = "frequentist"

    def __init__(self, n_classes, B=10, C=1.0, max_train=1500, seed=0):
        self.nc, self.B, self.Cr, self.max_train, self.seed = n_classes, B, C, max_train, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.members = []
        n = min(len(X), self.max_train)
        for b in range(self.B):
            idx = rng.choice(len(X), size=n, replace=True)
            # ensure every class is present in the resample
            for c in range(self.nc):
                if not (y[idx] == c).any():
                    idx[rng.integers(n)] = rng.choice(np.where(y == c)[0])
            m = SVC(C=self.Cr, kernel="rbf", gamma="scale", probability=True, random_state=self.seed + b)
            self.members.append(m.fit(X[idx], y[idx]))
        return self

    def predict_samples(self, X):
        out = []
        for m in self.members:
            P = np.zeros((len(X), self.nc))
            P[:, m.classes_] = m.predict_proba(X)
            out.append(P)
        return np.stack(out)


def make_model(name: str, n_classes: int, seed: int, cfg: dict):
    mc = cfg.get("model", {})
    if name == "mcd":
        return MCDropoutMLP(n_classes, hidden=tuple(mc.get("hidden", (64, 64))), p=mc.get("dropout", 0.2),
                            epochs=mc.get("epochs", 60), weight_decay=mc.get("weight_decay", 1e-4),
                            T=mc.get("T", 30), seed=seed)
    if name == "ens":
        return DeepEnsemble(n_classes, M=mc.get("M", 5), hidden=tuple(mc.get("hidden", (64, 64))),
                            epochs=mc.get("epochs", 60), weight_decay=mc.get("weight_decay", 1e-4), seed=seed)
    if name == "bsvm":
        return BootstrapSVM(n_classes, B=mc.get("B", 10), max_train=mc.get("svm_max_train", 1500), seed=seed)
    raise ValueError(name)


MODEL_LABELS = {"mcd": "MC-Dropout MLP (Bayesian approx.)",
                "ens": "Deep Ensemble (Bayesian approx.)",
                "bsvm": "Bootstrap RBF-SVM (frequentist)"}
