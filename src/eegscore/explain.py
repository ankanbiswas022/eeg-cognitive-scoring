"""Explainability for the feature model and the deep model.

* :func:`permutation_importance_by_family` - model-agnostic; groups the ~300 individual
  features into neuroscientific families (band power, ratios, 1/f, connectivity ...)
  and permutes a whole family at once. Answers "which *kind* of signal does the model use?"
* :func:`shap_summary` - TreeSHAP values for the LightGBM model, per feature and per family.
* :func:`channel_importance_topomap` - projects per-channel importance onto a 10-20
  scalp map so a neuroscientist can sanity-check the model (frontal theta / parietal
  alpha for arithmetic load is the expected picture).
* :func:`deep_saliency_map` - gradient x input for the PyTorch model, averaged into a
  (channel x frequency band) matrix so it is comparable with the feature model.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import feature_family

log = logging.getLogger(__name__)


def permutation_importance_by_family(model, F: pd.DataFrame, y: np.ndarray, n_repeats: int = 5,
                                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(F)[:, 1])
    fams = pd.Series({c: feature_family(c) for c in F.columns})
    rows = []
    for fam in sorted(fams.unique()):
        cols = fams.index[fams == fam]
        drops = []
        for _ in range(n_repeats):
            Fp = F.copy()
            Fp[cols] = Fp[cols].to_numpy()[rng.permutation(len(Fp))]
            drops.append(base - roc_auc_score(y, model.predict_proba(Fp)[:, 1]))
        rows.append({"family": fam, "n_features": len(cols), "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def shap_summary(pipeline, F: pd.DataFrame, max_rows: int = 2000, seed: int = 0):
    """Return (shap_values DataFrame, per-feature mean|SHAP|, per-family mean|SHAP|)."""
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=min(max_rows, len(F)), replace=False)
    Fs = F.iloc[idx]
    Xs = pipeline[:-1].transform(Fs)                 # apply imputer + scaler
    explainer = shap.TreeExplainer(pipeline[-1])
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):                         # older shap returns per-class list
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[..., 1]
    S = pd.DataFrame(sv, columns=F.columns, index=Fs.index)
    per_feat = S.abs().mean().sort_values(ascending=False)
    fam = per_feat.groupby(per_feat.index.map(feature_family)).sum().sort_values(ascending=False)
    return S, per_feat, fam


def channel_importance(per_feature_importance: pd.Series, ch_names: list[str],
                       pattern: str | None = None) -> pd.Series:
    """Sum importance over all features belonging to each channel (optionally filtered by
    a regex on the feature name, e.g. 'theta_logpow' or 'alpha_relpow')."""
    imp = pd.Series(0.0, index=ch_names)
    for name, v in per_feature_importance.items():
        if pattern and not re.search(pattern, name):
            continue
        for ch in ch_names:
            if name.endswith(f"_{ch}"):
                imp[ch] += v
                break
    return imp


def plot_topomap(values: pd.Series, title: str, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    info = mne.create_info(list(values.index), sfreq=250.0, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(values.to_numpy(), info, axes=ax, show=False, cmap="Reds",
                         contours=4)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def deep_saliency_map(clf, X: np.ndarray, sfreq: float, ch_names: list[str],
                      bands: dict, n_windows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Gradient x input saliency -> (channel x band) table of mean |saliency| power."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_windows, len(X)), replace=False)
    sal = clf.saliency(X[idx])                        # (n, ch, t)
    spec = np.abs(np.fft.rfft(sal, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(sal.shape[-1], 1 / sfreq)
    rows = {}
    for band, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        rows[band] = spec[..., m].mean((0, 2))
    return pd.DataFrame(rows, index=ch_names)
