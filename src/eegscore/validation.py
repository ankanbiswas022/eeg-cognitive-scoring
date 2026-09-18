"""Subject-wise validation and statistical evaluation.

Why subject-wise? Adjacent EEG windows from the same person are highly correlated.
A random window-level split leaks subject identity into the test fold and inflates
accuracy by 10-30 points - the single most common mistake in published EEG-ML work.
All splits here use ``GroupKFold`` on subject id, and the headline number is the
*out-of-fold* (OOF) AUC computed over all windows plus a *subject-level bootstrap CI*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)


@dataclass
class CVResult:
    oof_prob: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    fold_metrics: list[dict] = field(default_factory=list)

    def metrics(self) -> dict:
        pred = (self.oof_prob >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(self.y, self.oof_prob)),
            "balanced_accuracy": float(balanced_accuracy_score(self.y, pred)),
            "f1": float(f1_score(self.y, pred)),
            "brier": float(brier_score_loss(self.y, self.oof_prob)),
            "n_windows": int(len(self.y)),
            "n_subjects": int(len(np.unique(self.groups))),
        }

    def per_subject_accuracy(self) -> dict[int, float]:
        pred = (self.oof_prob >= 0.5).astype(int)
        return {int(g): float((pred[self.groups == g] == self.y[self.groups == g]).mean())
                for g in np.unique(self.groups)}

    def session_level_metrics(self) -> dict:
        """Aggregate window probabilities per (subject, label) recording -> recording AUC.

        This is the deployment-relevant number: a scoring system reports one score
        per recording, not per 4-s window.
        """
        keys, probs, ys = [], [], []
        for g in np.unique(self.groups):
            for lab in (0, 1):
                m = (self.groups == g) & (self.y == lab)
                if m.any():
                    keys.append((int(g), lab)); probs.append(float(np.median(self.oof_prob[m]))); ys.append(lab)
        probs, ys = np.array(probs), np.array(ys)
        return {"recording_auc": float(roc_auc_score(ys, probs)),
                "recording_accuracy": float(((probs >= 0.5) == ys).mean()),
                "n_recordings": int(len(ys))}


def subject_cv(fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
               X, y: np.ndarray, groups: np.ndarray, n_splits: int = 6) -> CVResult:
    """Generic subject-wise CV. ``fit_predict(X_train, y_train, X_test) -> p(test)``.

    ``X`` may be a NumPy array (deep models) or a DataFrame (feature models).
    """
    y, groups = np.asarray(y), np.asarray(groups)
    oof = np.full(len(y), np.nan)
    res = CVResult(oof, y, groups)
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups)):
        Xtr, Xte = (X.iloc[tr], X.iloc[te]) if hasattr(X, "iloc") else (X[tr], X[te])
        p = fit_predict(Xtr, y[tr], Xte)
        oof[te] = p
        fm = {"fold": k, "auc": float(roc_auc_score(y[te], p)),
              "balanced_accuracy": float(balanced_accuracy_score(y[te], (p >= 0.5).astype(int))),
              "test_subjects": sorted(map(int, np.unique(groups[te])))}
        res.fold_metrics.append(fm)
        log.info("fold %d AUC %.3f  bAcc %.3f  (subjects %s)", k, fm["auc"],
                 fm["balanced_accuracy"], fm["test_subjects"])
    return res


def subject_bootstrap_auc(res: CVResult, n_boot: int = 1000, seed: int = 0
                          ) -> tuple[float, float, float]:
    """Cluster bootstrap over subjects -> (auc, ci_low, ci_high). Resampling subjects
    (not windows) respects the within-subject correlation structure."""
    rng = np.random.default_rng(seed)
    subj = np.unique(res.groups)
    idx_by_subj = {g: np.where(res.groups == g)[0] for g in subj}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(subj, size=len(subj), replace=True)
        idx = np.concatenate([idx_by_subj[g] for g in pick])
        if len(np.unique(res.y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(res.y[idx], res.oof_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(res.y, res.oof_prob)), float(lo), float(hi)


def permutation_test_auc(fit_predict, X, y, groups, n_splits=6, n_perm=20, seed=0
                         ) -> dict:
    """Recording-level label permutation: shuffle which recording of each subject is
    called 'task'. Keeps within-recording correlation intact, so the null is honest.
    Slow (re-runs CV n_perm times); use with the fast classical model."""
    rng = np.random.default_rng(seed)
    observed = subject_cv(fit_predict, X, y, groups, n_splits).metrics()["auc"]
    null = []
    for _ in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            if rng.random() < 0.5:
                m = groups == g
                y_perm[m] = 1 - y[m]
        null.append(subject_cv(fit_predict, X, y_perm, groups, n_splits).metrics()["auc"])
    null = np.array(null)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {"observed_auc": observed, "null_mean": float(null.mean()),
            "null_max": float(null.max()), "p_value": float(p), "n_perm": n_perm}
