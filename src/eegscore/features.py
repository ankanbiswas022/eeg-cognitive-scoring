"""Interpretable feature engineering for windowed EEG.

Feature families (all computed per 4-s window, from an (n_ch, n_samples) array in Volts):

1. **Band power** (multitaper PSD, log10 absolute and relative) for delta/theta/alpha/beta/gamma
   per channel. Multitaper is used because it gives a low-variance spectral estimate for
   short windows (Thomson 1982) and is what I used in my EEG papers.
2. **Cognitive-load ratios**: theta/alpha, beta/alpha and the engagement index
   beta/(alpha+theta) averaged over frontal and over all channels. These are the classic
   workload / stress markers in the applied EEG literature.
3. **Frontal alpha asymmetry** log(alpha_R) - log(alpha_L) for homologous frontal pairs.
4. **Aperiodic (1/f) parameters**: slope (exponent) and offset of a linear fit to the
   log-log PSD over 2-40 Hz. Proxy for cortical excitation/inhibition balance and arousal.
5. **Spectral entropy** per channel (flatness of the normalised PSD).
6. **Hjorth activity / mobility / complexity** (cheap time-domain descriptors).
7. **Alpha-band connectivity**: mean phase-locking value (PLV) over all channel pairs and
   over inter-hemispheric pairs.

The extractor is stateless: the same call is used for training and for serving.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, hilbert, sosfiltfilt

DEFAULT_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
                 "gamma": (30, 45)}
FRONTAL = ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")
LEFT_RIGHT_PAIRS = (("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("T3", "T4"),
                    ("P3", "P4"), ("T5", "T6"), ("O1", "O2"))
FRONTAL_PAIRS = (("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"))


@dataclass
class FeatureExtractor:
    sfreq: float
    ch_names: list[str]
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    multitaper_bandwidth: float = 2.0
    aperiodic_fit_range: tuple[float, float] = (2.0, 40.0)
    connectivity_band: str = "alpha"
    fmax: float = 45.0

    # ------------------------------------------------------------------ public API
    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """X: (n_windows, n_ch, n_samples) Volts -> DataFrame (n_windows, n_features)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        psd, freqs = psd_array_multitaper(X, self.sfreq, fmin=0.5, fmax=self.fmax,
                                          bandwidth=self.multitaper_bandwidth,
                                          adaptive=False, normalization="full",
                                          verbose="ERROR")
        psd = psd * 1e12                              # V^2/Hz -> uV^2/Hz (nicer numbers)
        feats: dict[str, np.ndarray] = {}
        feats.update(self._band_power(psd, freqs))
        feats.update(self._ratios(feats))
        feats.update(self._asymmetry(feats))
        feats.update(self._aperiodic(psd, freqs))
        feats.update(self._spectral_entropy(psd, freqs))
        feats.update(self._hjorth(X))
        feats.update(self._connectivity(X))
        return pd.DataFrame(feats)

    @property
    def feature_names(self) -> list[str]:
        dummy = np.random.default_rng(0).standard_normal((1, len(self.ch_names),
                                                          int(self.sfreq * 2))) * 1e-5
        return list(self.transform(dummy).columns)

    # ------------------------------------------------------------------ families
    def _band_power(self, psd, freqs):
        out = {}
        total = np.trapezoid(psd, freqs, axis=-1)                # (n_win, n_ch)
        for band, (lo, hi) in self.bands.items():
            m = (freqs >= lo) & (freqs < hi)
            bp = np.trapezoid(psd[..., m], freqs[m], axis=-1)
            for i, ch in enumerate(self.ch_names):
                out[f"{band}_logpow_{ch}"] = np.log10(bp[:, i] + 1e-12)
                out[f"{band}_relpow_{ch}"] = bp[:, i] / (total[:, i] + 1e-12)
        return out

    def _ratios(self, f):
        out = {}
        chs = self.ch_names
        frontal = [c for c in chs if c in FRONTAL] or chs

        def mean_pow(band, subset):
            return np.mean([10 ** f[f"{band}_logpow_{c}"] for c in subset], axis=0)

        for name, subset in (("all", chs), ("frontal", frontal)):
            th, al, be = (mean_pow(b, subset) for b in ("theta", "alpha", "beta"))
            out[f"theta_alpha_ratio_{name}"] = np.log10((th + 1e-12) / (al + 1e-12))
            out[f"beta_alpha_ratio_{name}"] = np.log10((be + 1e-12) / (al + 1e-12))
            out[f"engagement_index_{name}"] = np.log10((be + 1e-12) / (al + th + 1e-12))
        return out

    def _asymmetry(self, f):
        out = {}
        for left, right in FRONTAL_PAIRS:
            if left in self.ch_names and right in self.ch_names:
                out[f"alpha_asym_{right}_{left}"] = (f[f"alpha_logpow_{right}"]
                                                     - f[f"alpha_logpow_{left}"])
        return out

    def _aperiodic(self, psd, freqs):
        lo, hi = self.aperiodic_fit_range
        m = (freqs >= lo) & (freqs <= hi)
        # exclude the alpha peak region from the fit so the slope is not biased by it
        m &= ~((freqs >= 7) & (freqs <= 14))
        lf = np.log10(freqs[m])
        A = np.vstack([lf, np.ones_like(lf)]).T                # (n_f, 2)
        lp = np.log10(psd[..., m] + 1e-12)                     # (n_win, n_ch, n_f)
        coef, *_ = np.linalg.lstsq(A, lp.reshape(-1, lp.shape[-1]).T, rcond=None)
        coef = coef.T.reshape(psd.shape[0], psd.shape[1], 2)
        out = {}
        for i, ch in enumerate(self.ch_names):
            out[f"aperiodic_exponent_{ch}"] = -coef[:, i, 0]   # positive = steeper 1/f
            out[f"aperiodic_offset_{ch}"] = coef[:, i, 1]
        out["aperiodic_exponent_mean"] = -coef[:, :, 0].mean(1)
        return out

    def _spectral_entropy(self, psd, freqs):
        m = (freqs >= 1) & (freqs <= self.fmax)
        p = psd[..., m] / (psd[..., m].sum(-1, keepdims=True) + 1e-20)
        H = -(p * np.log2(p + 1e-20)).sum(-1) / np.log2(p.shape[-1])
        return {f"spectral_entropy_{ch}": H[:, i] for i, ch in enumerate(self.ch_names)}

    def _hjorth(self, X):
        d1 = np.diff(X, axis=-1)
        d2 = np.diff(d1, axis=-1)
        act = X.var(-1)
        mob = np.sqrt(d1.var(-1) / (act + 1e-30))
        comp = np.sqrt(d2.var(-1) / (d1.var(-1) + 1e-30)) / (mob + 1e-30)
        out = {}
        for i, ch in enumerate(self.ch_names):
            out[f"hjorth_mobility_{ch}"] = mob[:, i]
            out[f"hjorth_complexity_{ch}"] = comp[:, i]
        return out

    def _connectivity(self, X):
        lo, hi = self.bands[self.connectivity_band]
        sos = butter(4, [lo, hi], btype="band", fs=self.sfreq, output="sos")
        phase = np.angle(hilbert(sosfiltfilt(sos, X, axis=-1), axis=-1))   # (n_win, n_ch, n_t)
        z = np.exp(1j * phase)
        n_t = z.shape[-1]
        plv = np.abs(z @ np.conj(z).transpose(0, 2, 1)) / n_t                # (n_win, n_ch, n_ch)
        iu = np.triu_indices(len(self.ch_names), k=1)
        out = {f"plv_{self.connectivity_band}_global": plv[:, iu[0], iu[1]].mean(1)}
        idx = {c: i for i, c in enumerate(self.ch_names)}
        pairs = [(idx[a], idx[b]) for a, b in LEFT_RIGHT_PAIRS if a in idx and b in idx]
        if pairs:
            out[f"plv_{self.connectivity_band}_interhemispheric"] = np.mean(
                [plv[:, a, b] for a, b in pairs], axis=0)
        return out


def feature_family(name: str) -> str:
    """Map a feature column name to its family (used for grouped explainability)."""
    for fam in ("logpow", "relpow", "ratio", "engagement", "asym", "aperiodic",
                "spectral_entropy", "hjorth", "plv"):
        if fam in name:
            return fam
    return "other"
