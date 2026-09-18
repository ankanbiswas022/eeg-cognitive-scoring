
import numpy as np
import pandas as pd

from eegscore.monitoring import FeatureReference, psi, signal_quality

from .conftest import SFREQ, synth_eeg


def test_signal_quality_flags():
    X = synth_eeg(1)[0].astype(np.float64)
    assert signal_quality(X, SFREQ, expected_n_ch=X.shape[0]) == []
    bad = X.copy()
    bad[0] = 0.0                                  # flat channel
    bad[1] *= 100                                  # saturated
    flags = signal_quality(bad, SFREQ, expected_n_ch=X.shape[0])
    assert any(f.startswith("flat_ch") for f in flags)
    assert any(f.startswith("high_amplitude") for f in flags)
    assert "channel_count" in signal_quality(X[:5], SFREQ, expected_n_ch=19)[0]


def test_mains_contamination_detected():
    X = synth_eeg(1)[0].astype(np.float64)
    t = np.arange(X.shape[1]) / SFREQ
    X += 30e-6 * np.sin(2 * np.pi * 50 * t)
    assert "mains_contamination" in signal_quality(X, SFREQ)


def test_psi_zero_for_identical_and_large_for_shift():
    ref = np.array([10, 20, 30, 20, 10])
    assert psi(ref, ref) < 1e-6
    assert psi(ref, np.array([40, 30, 10, 5, 5])) > 0.25


def test_feature_reference_drift_report(tmp_path):
    rng = np.random.default_rng(0)
    F = pd.DataFrame({"a": rng.standard_normal(1000), "b": rng.standard_normal(1000)})
    ref = FeatureReference.fit(F)
    ref.to_json(tmp_path / "ref.json")
    ref = FeatureReference.from_json(tmp_path / "ref.json")
    same = ref.drift_report(F)
    assert (same["status"] == "ok").all()
    shifted = F.copy()
    shifted["a"] += 3.0
    rep = ref.drift_report(shifted).set_index("feature")
    assert rep.loc["a", "status"] == "alert" and rep.loc["b", "status"] == "ok"


def test_api_health_without_model(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import eegscore.serve.app as appmod
    monkeypatch.setattr(appmod, "ART", tmp_path)
    monkeypatch.setattr(appmod, "_bundle", None)
    c = TestClient(appmod.app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "no_model"
    r = c.post("/score", json={"sfreq": 250, "ch_names": ["Fp1"], "data": [[0.0] * 10]})
    assert r.status_code == 503
