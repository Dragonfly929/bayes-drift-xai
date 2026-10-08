"""Dataset loading with provenance checks and preprocessing.

Every real dataset is described by a ``DatasetSpec``: where it comes from, its licence, how to cite it
and the SHA-256 of the exact file used in the paper. ``load_dataset`` verifies the checksum so a
reader can confirm that they are working on byte-identical data (reproducibility requirement).

Synthetic streams (SEA, rotated digits) are generated deterministically from a seed.
Any user CSV can also be loaded with ``load_csv`` — this is what makes the pipeline reusable on
other data ("if the professor brings other data, the same pipeline runs end-to-end").
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")


@dataclass
class DatasetSpec:
    name: str
    file: str
    urls: list  # official source first, mirror(s) after
    sha256: Optional[str]
    target: str
    domain: str
    licence: str
    citation: str
    drop: list = field(default_factory=list)
    natural_order: bool = False  # True -> rows are in time order (natural stream)


SPECS = {
    "credit": DatasetSpec(
        name="Default of Credit Card Clients (Taiwan, 2005)",
        file="credit_default.csv",
        urls=[
            "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/main/dataset/default_of_credit_card_clients.csv",
        ],
        sha256="4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb",
        target="default payment next month",
        domain="finance / credit risk (fraud-like)",
        licence="CC BY 4.0 (UCI Machine Learning Repository, id 350)",
        citation="Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive "
                 "accuracy of probability of default of credit card clients. Expert Systems with Applications, "
                 "36(2), 2473-2480. https://doi.org/10.1016/j.eswa.2007.12.020 ; UCI ML Repository, "
                 "https://doi.org/10.24432/C55S3H",
        drop=["ID"],
    ),
    "telco": DatasetSpec(
        name="IBM Telco Customer Churn",
        file="telco_churn.csv",
        urls=[
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        ],
        sha256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        target="Churn",
        domain="customer behaviour (churn)",
        licence="Apache-2.0 (IBM sample data, github.com/IBM/telco-customer-churn-on-icp4d)",
        citation="IBM (2019). Telco Customer Churn sample dataset. "
                 "https://github.com/IBM/telco-customer-churn-on-icp4d",
        drop=["customerID"],
    ),
    "elec2": DatasetSpec(
        name="Electricity (Elec2, NSW market 1996-1998)",
        file="elec2.csv",
        urls=[
            "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv",
        ],
        sha256="cdf901433885f29eca6911f70c0eeafb50d90596c879c30c5b99f5a2e8e734ff",
        target="class",
        domain="energy market (natural, real-world non-stationary stream)",
        licence="Freely available for research (Harries 1999; distributed with MOA / scikit-multiflow, BSD-3)",
        citation="Harries, M. (1999). SPLICE-2 Comparative Evaluation: Electricity Pricing. Technical report, "
                 "University of New South Wales. ; Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004). "
                 "Learning with drift detection. SBIA 2004, LNCS 3171, 286-295.",
        natural_order=True,
    ),
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, verbose: bool = True) -> str:
    """Return a local path to the raw file, downloading it if needed, and verify the checksum."""
    spec = SPECS[key]
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, spec.file)
    if not os.path.exists(path):
        last = None
        for url in spec.urls:
            try:
                if verbose:
                    print(f"[data] downloading {key} from {url}")
                urllib.request.urlretrieve(url, path)
                break
            except Exception as e:  # pragma: no cover - network dependent
                last = e
        else:
            raise RuntimeError(f"Could not download {key}: {last}")
    if spec.sha256:
        got = sha256_of(path)
        if got != spec.sha256:
            raise ValueError(f"Checksum mismatch for {key}: expected {spec.sha256}, got {got}. "
                             "The file differs from the one used in the paper.")
    return path


@dataclass
class Tabular:
    """Encoded dataset: numeric matrix, binary/multiclass labels and metadata."""
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    numeric_idx: list          # columns that are genuinely continuous (eligible as drift features)
    n_classes: int
    name: str
    natural_order: bool = False


def encode_frame(df: pd.DataFrame, target: str, drop=None, name="custom", natural_order=False) -> Tabular:
    """One-hot encode categoricals, coerce numerics, map target to 0..C-1."""
    df = df.copy()
    for c in (drop or []):
        if c in df.columns:
            df = df.drop(columns=c)
    df = df[df[target].notna()].reset_index(drop=True)     # records without a label cannot be scored
    y_raw = df.pop(target)
    # coerce numeric-looking strings (e.g. Telco TotalCharges contains blanks)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c].astype(str).str.strip().replace("", np.nan), errors="coerce")
            if conv.notna().mean() > 0.95:
                df[c] = conv
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    Xdf = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    feats = list(Xdf.columns)
    numeric_idx = [feats.index(c) for c in num_cols if df[c].nunique() > 10]
    classes = sorted(pd.Series(y_raw).astype(str).unique())
    y = pd.Series(y_raw).astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return Tabular(Xdf.to_numpy(float), y.astype(int), feats, numeric_idx, len(classes), name, natural_order)


def load_dataset(key: str) -> Tabular:
    spec = SPECS[key]
    df = pd.read_csv(fetch(key, verbose=False))
    return encode_frame(df, spec.target, spec.drop, name=key, natural_order=spec.natural_order)


def load_csv(path_or_buffer, target: str, drop=None, name="custom") -> Tabular:
    df = pd.read_csv(path_or_buffer)
    return encode_frame(df, target, drop, name=name)


# ----------------------------------------------------------------------------- synthetic data
def load_digits_tab() -> Tabular:
    """8x8 handwritten digits (sklearn bundled copy of UCI Optical Recognition of Handwritten Digits)."""
    from sklearn.datasets import load_digits
    d = load_digits()
    return Tabular(d.data.astype(float), d.target.astype(int), [f"px{i}" for i in range(64)],
                   [], 10, "digits")


def rotate_digits(X: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rotate each 8x8 image by its own angle (covariate drift that keeps the label)."""
    from scipy.ndimage import rotate
    out = X.copy()
    for i, a in enumerate(angles_deg):
        if a != 0:
            out[i] = rotate(X[i].reshape(8, 8), a, reshape=False, order=1, mode="grid-constant", cval=0.0).ravel()
    return out


def sea_stream(n: int, variant_before: int, variant_after: int, t0: int, width: int, seed: int,
               noise: float = 0.1):
    """SEA concepts (Street & Kim, 2001) via River, with a sigmoid transition of given width at t0.
    Only P(y|x) changes -> pure *real* concept drift; P(x) is uniform on [0,10]^3 throughout."""
    from river.datasets import synth
    if variant_before == variant_after:
        s = synth.SEA(variant=variant_before, noise=noise, seed=seed)
    else:
        s = synth.ConceptDriftStream(
            stream=synth.SEA(variant=variant_before, noise=noise, seed=seed),
            drift_stream=synth.SEA(variant=variant_after, noise=noise, seed=seed + 1),
            position=t0, width=max(width, 1), seed=seed)
    X, y = [], []
    for x, label in s.take(n):
        X.append([x[0], x[1], x[2]])
        y.append(int(label))
    return np.asarray(X, float), np.asarray(y, int)
