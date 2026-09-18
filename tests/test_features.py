import numpy as np

from eegscore.features import FeatureExtractor, feature_family

from .conftest import CH, SFREQ, synth_eeg


def test_feature_shapes_and_finite():
    fx = FeatureExtractor(sfreq=SFREQ, ch_names=CH)
    X = synth_eeg(6)
    F = fx.transform(X)
    assert F.shape[0] == 6
    assert F.shape[1] > 100
    assert np.isfinite(F.to_numpy()).all()
    assert list(F.columns) == fx.feature_names


def test_alpha_feature_tracks_alpha_amplitude():
    fx = FeatureExtractor(sfreq=SFREQ, ch_names=CH)
    strong = fx.transform(synth_eeg(10, alpha_uv=15.0, seed=3))
    weak = fx.transform(synth_eeg(10, alpha_uv=2.0, seed=3))
    assert strong["alpha_relpow_O1"].mean() > weak["alpha_relpow_O1"].mean()
    assert strong["theta_alpha_ratio_all"].mean() < weak["theta_alpha_ratio_all"].mean()


def test_single_window_input_is_promoted():
    fx = FeatureExtractor(sfreq=SFREQ, ch_names=CH)
    F = fx.transform(synth_eeg(1)[0])
    assert F.shape[0] == 1


def test_aperiodic_exponent_is_positive_for_pink_noise():
    fx = FeatureExtractor(sfreq=SFREQ, ch_names=CH)
    F = fx.transform(synth_eeg(5, alpha_uv=0.0))
    assert F["aperiodic_exponent_mean"].mean() > 0.5


def test_plv_bounded():
    fx = FeatureExtractor(sfreq=SFREQ, ch_names=CH)
    F = fx.transform(synth_eeg(5))
    assert ((F["plv_alpha_global"] >= 0) & (F["plv_alpha_global"] <= 1)).all()


def test_feature_family_mapping():
    assert feature_family("alpha_logpow_O1") == "logpow"
    assert feature_family("theta_alpha_ratio_all") == "ratio"
    assert feature_family("aperiodic_exponent_mean") == "aperiodic"
