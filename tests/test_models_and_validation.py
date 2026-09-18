import numpy as np

from eegscore.config import load_config
from eegscore.features import FeatureExtractor
from eegscore.models import TorchClassifier, build_classical
from eegscore.scoring import Scorer
from eegscore.validation import subject_bootstrap_auc, subject_cv

from .conftest import CH, SFREQ


def test_subject_cv_never_mixes_subjects(small_dataset):
    X, y, groups = small_dataset
    # encode the subject id into the "prediction" so we can recover which subjects were
    # in each test fold and check they never appear in the corresponding train fold
    def fp(Xtr, ytr, Xte):
        tr_ids = {int(round(v)) for v in Xtr[:, 0, 0]}
        te_ids = {int(round(v)) for v in Xte[:, 0, 0]}
        assert tr_ids.isdisjoint(te_ids)
        return np.full(len(Xte), 0.5)

    Xid = X.copy()
    Xid[:, 0, 0] = groups                      # stamp subject id into one sample
    res = subject_cv(fp, Xid, y, groups, n_splits=4)
    assert len(res.fold_metrics) == 4
    assert np.isfinite(res.oof_prob).all()


def test_classical_pipeline_learns_alpha_difference(small_dataset):
    X, y, groups = small_dataset
    cfg = load_config()
    cfg["models"]["classical"] = {"type": "logreg", "C": 0.5}
    F = FeatureExtractor(sfreq=SFREQ, ch_names=CH).transform(X)

    def fp(Xtr, ytr, Xte):
        return build_classical(cfg).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    res = subject_cv(fp, F, y, groups, n_splits=4)
    assert res.metrics()["auc"] > 0.9
    auc, lo, hi = subject_bootstrap_auc(res, n_boot=50)
    assert lo <= auc <= hi


def test_torch_classifier_fit_predict_save_load(small_dataset, tmp_path):
    X, y, _ = small_dataset
    clf = TorchClassifier(arch="eegnet", epochs=3, batch_size=16, device="cpu").fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(1), 1)
    clf.save(tmp_path / "m.pt")
    clf2 = TorchClassifier.load(tmp_path / "m.pt", device="cpu")
    assert np.allclose(clf2.predict_proba(X), p, atol=1e-5)
    sal = clf.saliency(X[:2])
    assert sal.shape == X[:2].shape


def test_all_deep_archs_forward(small_dataset):
    X, y, _ = small_dataset
    for arch in ("eegnet", "tcn", "gru"):
        clf = TorchClassifier(arch=arch, epochs=1, batch_size=16, device="cpu").fit(X[:16], y[:16])
        assert clf.predict_proba(X[:4]).shape == (4, 2)


def test_scorer_calibration_and_summary(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + 0.35 + 0.15 * rng.standard_normal(500), 0, 1)
    sc = Scorer().fit(p, y)
    s = sc.window_scores(p)
    assert s.min() >= 0 and s.max() <= 100
    summ = sc.summarise(p[y == 1])
    assert summ["label"] == "high_load" and 0 <= summ["score"] <= 100
    sc.to_json(tmp_path / "s.json")
    sc2 = Scorer.from_json(tmp_path / "s.json")
    assert np.allclose(sc2.window_scores(p), s)
