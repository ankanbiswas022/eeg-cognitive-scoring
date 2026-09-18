"""Classical feature-based models (scikit-learn API).

The feature-based model is the production default: it is fast, robust on small cohorts,
explainable with SHAP, and its inputs (band powers, ratios, 1/f slope) are quantities
neuroscientists already reason about. The deep model is kept as a challenger.
"""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classical(cfg: dict, seed: int = 7) -> Pipeline:
    mcfg = cfg["models"]["classical"]
    kind = mcfg.get("type", "lightgbm")
    if kind == "logreg":
        clf = LogisticRegression(C=mcfg.get("C", 0.1), max_iter=2000, class_weight="balanced")
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=mcfg.get("n_estimators", 300),
            learning_rate=mcfg.get("learning_rate", 0.03),
            num_leaves=mcfg.get("num_leaves", 15),
            min_child_samples=mcfg.get("min_child_samples", 20),
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, verbose=-1,
        )
    else:
        raise ValueError(f"unknown classical model type {kind!r}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
