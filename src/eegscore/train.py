"""End-to-end training entry point.

    python -m eegscore.train --config configs/default.yaml [--skip-deep] [--n-subjects 12]

Steps
-----
1. Load + preprocess + window all subjects (subject ids kept as groups; the first
   ``baseline_seconds`` of each rest recording are calibration windows, never scored).
2. Extract interpretable features; z-score them against each subject's own baseline
   (``features.baseline_relative``). The absolute-feature model is also evaluated as an
   ablation so the gain from calibration is documented.
3. Subject-wise CV for the feature model (LightGBM) and the deep model (EEGNet/TCN/GRU).
4. Bootstrap CIs, recording-level metrics, permutation test (feature model).
5. Explainability: family permutation importance, SHAP, channel topomaps, deep saliency.
6. Fit the calibrated Scorer on out-of-fold probabilities.
7. Fit the final feature model on all data, save every artifact the service needs:
   model.joblib, scorer.json, feature_reference.json, feature_names.json, data_spec.json,
   deep_model.pt, metrics.json, plus figures/tables under reports/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from .config import load_config, resolve
from .data import WindowedDataset, build_dataset, list_subjects
from .explain import (
    channel_importance,
    deep_saliency_map,
    permutation_importance_by_family,
    plot_topomap,
    shap_summary,
)
from .features import FeatureExtractor, baseline_normalise
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


def baseline_scale_raw(ds: WindowedDataset) -> np.ndarray:
    """Divide each subject's raw windows by that subject's per-channel baseline std.

    Raw-domain analogue of feature baseline-normalisation for the deep model.
    """
    X = ds.X.copy()
    for g in np.unique(ds.groups):
        m = ds.groups == g
        b = m & ds.baseline
        sd = ds.X[b].std(axis=(0, 2), keepdims=True) + 1e-12      # (1, n_ch, 1)
        X[m] = ds.X[m] / sd
    return X


def _cv_block(name, fit_predict, X, y, groups, n_splits):
    res = subject_cv(fit_predict, X, y, groups, n_splits)
    auc, lo, hi = subject_bootstrap_auc(res)
    m = {**res.metrics(), "auc_ci95": [lo, hi], **res.session_level_metrics(),
         "folds": res.fold_metrics, "per_subject_accuracy": res.per_subject_accuracy()}
    log.info("%s: OOF AUC %.3f [%.3f, %.3f]  bAcc %.3f  recording AUC %.3f  recording acc %.3f",
             name, auc, lo, hi, m["balanced_accuracy"], m["recording_auc"], m["recording_accuracy"])
    return res, m


def run(cfg: dict, skip_deep: bool = False, n_subjects: int | None = None,
        n_perm: int = 20) -> dict:
    t0 = time.time()
    art = resolve(cfg["artifacts_dir"]); art.mkdir(parents=True, exist_ok=True)
    rep = resolve("reports"); rep.mkdir(exist_ok=True)
    raw_dir = resolve(cfg["data"]["raw_dir"])
    subjects = list_subjects(raw_dir)[:n_subjects] if n_subjects else None
    baseline_relative = bool(cfg["features"].get("baseline_relative", False))
    if baseline_relative and not cfg["data"].get("baseline_seconds"):
        raise ValueError("features.baseline_relative requires data.baseline_seconds > 0")

    # 1-2. data + features ----------------------------------------------------------
    ds = build_dataset(cfg, subjects=subjects, raw_dir=raw_dir)
    fx = make_extractor(cfg, ds.sfreq, ds.ch_names)
    F_all = fx.transform(ds.X)
    s = ds.scored
    y, groups = ds.y[s], ds.groups[s]
    F_abs = F_all[s].reset_index(drop=True)
    if baseline_relative:
        F = baseline_normalise(F_all, ds.groups, ds.baseline)[s].reset_index(drop=True)
        X_raw = baseline_scale_raw(ds)[s]
    else:
        F, X_raw = F_abs, ds.X[s]
    log.info("features: %s (baseline_relative=%s)", F.shape, baseline_relative)
    metrics: dict = {"dataset": ds.summary(), "n_features": int(F.shape[1]),
                     "baseline_relative": baseline_relative, "config": cfg["_config_path"]}

    # 3. subject-wise CV: feature model ---------------------------------------------
    n_splits = cfg["validation"]["n_splits"]
    seed = cfg["validation"]["seed"]

    def fp_classical(Xtr, ytr, Xte):
        return build_classical(cfg, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    kind = cfg["models"]["classical"]["type"]
    log.info("== feature model (%s), %d-fold subject-wise CV", kind, n_splits)
    res_c, metrics["feature_model"] = _cv_block("feature model", fp_classical, F, y, groups, n_splits)
    if baseline_relative:
        # ablation: same model on absolute (un-calibrated) features
        _, metrics["feature_model_absolute_ablation"] = _cv_block(
            "ablation (absolute features)", fp_classical, F_abs, y, groups, n_splits)
    if n_perm:
        metrics["feature_model"]["permutation_test"] = permutation_test_auc(
            fp_classical, F, y, groups, n_splits, n_perm=n_perm, seed=seed)
        pt = metrics["feature_model"]["permutation_test"]
        log.info("permutation p = %.3f (null mean %.3f, max %.3f)", pt["p_value"], pt["null_mean"], pt["null_max"])

    # 3b. deep model ----------------------------------------------------------------
    if not skip_deep:
        dcfg = cfg["models"]["deep"]

        def make_deep():
            return TorchClassifier(arch=dcfg["arch"], epochs=dcfg["epochs"], batch_size=dcfg["batch_size"],
                                   lr=dcfg["lr"], weight_decay=dcfg["weight_decay"], seed=dcfg["seed"])

        def fp_deep(Xtr, ytr, Xte):
            return make_deep().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

        log.info("== deep model (%s), %d-fold subject-wise CV", dcfg["arch"], n_splits)
        res_d, metrics["deep_model"] = _cv_block(f"deep model ({dcfg['arch']})", fp_deep, X_raw, y, groups, n_splits)
        metrics["deep_model"]["arch"] = dcfg["arch"]
        fused = 0.5 * (res_c.oof_prob + res_d.oof_prob)
        metrics["fusion_auc"] = float(roc_auc_score(y, fused))
        log.info("late fusion AUC %.3f", metrics["fusion_auc"])

    # 4-5. final feature model + explainability ------------------------------------
    model = build_classical(cfg, seed).fit(F, y)
    imp_fam = permutation_importance_by_family(model, F, y)
    imp_fam.to_csv(rep / "importance_by_family.csv", index=False)
    metrics["importance_by_family"] = imp_fam.to_dict("records")
    try:
        _, per_feat, fam = shap_summary(model, F)
        per_feat.head(40).to_csv(rep / "shap_top_features.csv", header=["mean_abs_shap"])
        fam.to_csv(rep / "shap_by_family.csv", header=["mean_abs_shap"])
        metrics["shap_top10"] = per_feat.head(10).round(4).to_dict()
        metrics["shap_by_family"] = fam.round(4).to_dict()
        for pat, title in (("theta_logpow", "theta power"), ("alpha_relpow", "relative alpha"),
                           (None, "all features")):
            ci = channel_importance(per_feat, ds.ch_names, pat)
            plot_topomap(ci, f"SHAP importance: {title}", rep / f"topomap_{(pat or 'all')}.png")
    except Exception as e:  # noqa: BLE001 - shap is optional at runtime
        log.warning("SHAP step skipped: %s", e)

    if not skip_deep:
        clf = make_deep().fit(X_raw, y)
        clf.save(art / "deep_model.pt")
        sal = deep_saliency_map(clf, X_raw, ds.sfreq, ds.ch_names, fx.bands)
        sal.to_csv(rep / "deep_saliency_channel_band.csv")
        metrics["deep_saliency_band_share"] = (sal.sum() / sal.sum().sum()).round(3).to_dict()

    # 6. scorer on OOF probabilities -----------------------------------------------
    scorer = Scorer(cfg["scoring"]["score_min"], cfg["scoring"]["score_max"],
                    cfg["scoring"]["low_confidence_margin"], cfg["scoring"]["calibration"])
    scorer.fit(res_c.oof_prob, y)
    scorer.to_json(art / "scorer.json")
    p_cal = scorer.calibrate(res_c.oof_prob)
    metrics["feature_model"]["brier_after_calibration"] = float(brier_score_loss(y, p_cal))
    metrics["feature_model"]["balanced_accuracy_after_calibration"] = float(
        balanced_accuracy_score(y, (p_cal >= 0.5).astype(int)))
    # recording-level decision after calibration (the product's unit of output)
    rec = pd.DataFrame({"g": groups, "y": y, "p": p_cal}).groupby(["g", "y"]).p.median().reset_index()
    metrics["feature_model"]["recording_accuracy_after_calibration"] = float(
        ((rec.p >= 0.5).astype(int) == rec.y).mean())
    log.info("after calibration: bAcc %.3f  recording acc %.3f",
             metrics["feature_model"]["balanced_accuracy_after_calibration"],
             metrics["feature_model"]["recording_accuracy_after_calibration"])

    # 7. artifacts -------------------------------------------------------------------
    joblib.dump(model, art / "model.joblib")
    FeatureReference.fit(F).to_json(art / "feature_reference.json")
    (art / "feature_names.json").write_text(json.dumps(list(F.columns)))
    (art / "data_spec.json").write_text(json.dumps({
        "sfreq": ds.sfreq, "ch_names": ds.ch_names, "window_sec": cfg["data"]["window_sec"],
        "step_sec": cfg["data"]["step_sec"], "n_samples": int(ds.X.shape[2]),
        "preprocess": cfg["preprocess"], "resample_to": cfg["data"].get("resample_to"),
        "features": cfg["features"], "baseline_relative": baseline_relative,
        "baseline_seconds": cfg["data"].get("baseline_seconds")}, indent=2))
    pd.DataFrame({"subject": groups, "y": y, "oof_prob": res_c.oof_prob,
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
    keep = ("auc", "auc_ci95", "balanced_accuracy", "recording_auc", "recording_accuracy", "brier",
            "brier_after_calibration", "permutation_test", "arch")
    summary = {k: ({kk: vv for kk, vv in v.items() if kk in keep} if isinstance(v, dict) else v)
               for k, v in m.items() if k in ("feature_model", "feature_model_absolute_ablation",
                                              "deep_model", "fusion_auc")}
    summary["shap_by_family"] = m.get("shap_by_family")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
