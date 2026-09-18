"""End-to-end training entry point.

    python -m eegscore.train --config configs/default.yaml [--skip-deep] [--n-subjects 12]

Steps
-----
1. Load + preprocess + window all subjects (subject ids kept as groups).
2. Extract interpretable features.
3. Subject-wise CV for the feature model (LightGBM) and the deep model (EEGNet/TCN/GRU).
4. Bootstrap CIs, recording-level metrics, permutation test (feature model).
5. Explainability: family permutation importance, SHAP, channel topomaps, deep saliency.
6. Fit the calibrated Scorer on out-of-fold probabilities.
7. Fit the final feature model on all data, save every artifact the service needs:
   model.joblib, scorer.json, feature_reference.json, feature_names.json,
   deep_model.pt, metrics.json, plus figures under reports/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import build_dataset, list_subjects
from .explain import (channel_importance, deep_saliency_map, permutation_importance_by_family,
                      plot_topomap, shap_summary)
from .features import FeatureExtractor
from .models import TorchClassifier, build_classical
from .monitoring import FeatureReference
from .scoring import Scorer
from .validation import permutation_test_auc, subject_bootstrap_auc, subject_cv

log = logging.getLogger("eegscore.train")


def make_extractor(cfg: dict, sfreq: float, ch_names: list[str]) -> FeatureExtractor:
    fc = cfg["features"]
    return FeatureExtractor(sfreq=sfreq, ch_names=ch_names,
                            bands={k: tuple(v) for k, v in fc["bands"].items()},
                            multitaper_bandwidth=fc["multitaper_bandwidth"],
                            aperiodic_fit_range=tuple(fc["aperiodic_fit_range"]),
                            connectivity_band=fc["connectivity_band"],
                            fmax=cfg["preprocess"]["h_freq"])


def run(cfg: dict, skip_deep: bool = False, n_subjects: int | None = None,
        n_perm: int = 20) -> dict:
    t0 = time.time()
    art = resolve(cfg["artifacts_dir"]); art.mkdir(parents=True, exist_ok=True)
    rep = resolve("reports"); rep.mkdir(exist_ok=True)
    raw_dir = resolve(cfg["data"]["raw_dir"])
    subjects = list_subjects(raw_dir)[:n_subjects] if n_subjects else None

    # 1-2. data + features ----------------------------------------------------------
    ds = build_dataset(cfg, subjects=subjects, raw_dir=raw_dir)
    fx = make_extractor(cfg, ds.sfreq, ds.ch_names)
    F = fx.transform(ds.X)
    log.info("features: %s", F.shape)
    metrics: dict = {"dataset": ds.summary(), "n_features": int(F.shape[1]),
                     "config": cfg["_config_path"]}

    # 3. subject-wise CV: feature model ---------------------------------------------
    n_splits = cfg["validation"]["n_splits"]
    seed = cfg["validation"]["seed"]

    def fp_classical(Xtr, ytr, Xte):
        return build_classical(cfg, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    log.info("== feature model (%s), %d-fold subject-wise CV", cfg["models"]["classical"]["type"], n_splits)
    res_c = subject_cv(fp_classical, F, ds.y, ds.groups, n_splits)
    auc, lo, hi = subject_bootstrap_auc(res_c)
    metrics["feature_model"] = {**res_c.metrics(), "auc_ci95": [lo, hi],
                                **res_c.session_level_metrics(), "folds": res_c.fold_metrics,
                                "per_subject_accuracy": res_c.per_subject_accuracy()}
    log.info("feature model OOF AUC %.3f [%.3f, %.3f]  recording AUC %.3f", auc, lo, hi,
             metrics["feature_model"]["recording_auc"])
    if n_perm:
        metrics["feature_model"]["permutation_test"] = permutation_test_auc(
            fp_classical, F, ds.y, ds.groups, n_splits, n_perm=n_perm, seed=seed)
        log.info("permutation p = %.3f (null max %.3f)", metrics["feature_model"]["permutation_test"]["p_value"],
                 metrics["feature_model"]["permutation_test"]["null_max"])

    # 3b. deep model ----------------------------------------------------------------
    if not skip_deep:
        dcfg = cfg["models"]["deep"]

        def fp_deep(Xtr, ytr, Xte):
            clf = TorchClassifier(arch=dcfg["arch"], epochs=dcfg["epochs"], batch_size=dcfg["batch_size"],
                                  lr=dcfg["lr"], weight_decay=dcfg["weight_decay"], seed=dcfg["seed"])
            return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

        log.info("== deep model (%s), %d-fold subject-wise CV", dcfg["arch"], n_splits)
        res_d = subject_cv(fp_deep, ds.X, ds.y, ds.groups, n_splits)
        dauc, dlo, dhi = subject_bootstrap_auc(res_d)
        metrics["deep_model"] = {"arch": dcfg["arch"], **res_d.metrics(), "auc_ci95": [dlo, dhi],
                                 **res_d.session_level_metrics(), "folds": res_d.fold_metrics}
        log.info("deep model OOF AUC %.3f [%.3f, %.3f]", dauc, dlo, dhi)
        # simple late fusion of the two OOF probabilities
        fused = 0.5 * (res_c.oof_prob + res_d.oof_prob)
        from sklearn.metrics import roc_auc_score
        metrics["fusion_auc"] = float(roc_auc_score(ds.y, fused))

    # 4-5. final feature model + explainability ------------------------------------
    model = build_classical(cfg, seed).fit(F, ds.y)
    imp_fam = permutation_importance_by_family(model, F, ds.y)
    imp_fam.to_csv(rep / "importance_by_family.csv", index=False)
    metrics["importance_by_family"] = imp_fam.to_dict("records")
    try:
        _, per_feat, fam = shap_summary(model, F)
        per_feat.head(40).to_csv(rep / "shap_top_features.csv", header=["mean_abs_shap"])
        fam.to_csv(rep / "shap_by_family.csv", header=["mean_abs_shap"])
        metrics["shap_top10"] = per_feat.head(10).round(4).to_dict()
        for pat, title in (("theta_logpow", "theta power"), ("alpha_relpow", "relative alpha"),
                           (None, "all features")):
            ci = channel_importance(per_feat, ds.ch_names, pat)
            plot_topomap(ci, f"SHAP importance: {title}", rep / f"topomap_{(pat or 'all')}.png")
    except Exception as e:  # shap is optional at runtime
        log.warning("SHAP step skipped: %s", e)

    if not skip_deep:
        clf = TorchClassifier(arch=dcfg["arch"], epochs=dcfg["epochs"], batch_size=dcfg["batch_size"],
                              lr=dcfg["lr"], weight_decay=dcfg["weight_decay"], seed=dcfg["seed"])
        clf.fit(ds.X, ds.y)
        clf.save(art / "deep_model.pt")
        sal = deep_saliency_map(clf, ds.X, ds.sfreq, ds.ch_names, fx.bands)
        sal.to_csv(rep / "deep_saliency_channel_band.csv")
        metrics["deep_saliency_band_share"] = (sal.sum() / sal.sum().sum()).round(3).to_dict()

    # 6. scorer on OOF probabilities -----------------------------------------------
    scorer = Scorer(cfg["scoring"]["score_min"], cfg["scoring"]["score_max"],
                    cfg["scoring"]["low_confidence_margin"], cfg["scoring"]["calibration"])
    scorer.fit(res_c.oof_prob, ds.y)
    scorer.to_json(art / "scorer.json")
    from sklearn.metrics import brier_score_loss
    metrics["feature_model"]["brier_after_calibration"] = float(
        brier_score_loss(ds.y, scorer.calibrate(res_c.oof_prob)))

    # 7. artifacts -------------------------------------------------------------------
    joblib.dump(model, art / "model.joblib")
    FeatureReference.fit(F).to_json(art / "feature_reference.json")
    (art / "feature_names.json").write_text(json.dumps(list(F.columns)))
    (art / "data_spec.json").write_text(json.dumps({
        "sfreq": ds.sfreq, "ch_names": ds.ch_names, "window_sec": cfg["data"]["window_sec"],
        "n_samples": int(ds.X.shape[2]), "preprocess": cfg["preprocess"],
        "resample_to": cfg["data"].get("resample_to"), "features": cfg["features"]}, indent=2))
    pd.DataFrame({"subject": ds.groups, "y": ds.y, "oof_prob": res_c.oof_prob,
                  "score": scorer.window_scores(res_c.oof_prob)}).to_csv(rep / "oof_predictions.csv", index=False)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("done in %.0fs -> %s", metrics["train_seconds"], art)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--n-subjects", type=int, default=None, help="debug: use only the first N subjects")
    ap.add_argument("--n-perm", type=int, default=20, help="label permutations (0 to skip)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    m = run(load_config(a.config), skip_deep=a.skip_deep, n_subjects=a.n_subjects, n_perm=a.n_perm)
    print(json.dumps({k: v for k, v in m.items() if k in ("feature_model", "deep_model", "fusion_auc")},
                     indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
