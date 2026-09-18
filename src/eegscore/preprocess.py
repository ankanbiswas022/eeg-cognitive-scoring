"""Signal conditioning and window-level artifact gating.

Design notes
------------
* Everything here is deterministic and stateless so the same code path is used in
  training and in the online scoring service (train/serve skew is a classic failure
  mode for biosignal ML).
* Window rejection is done by simple amplitude and flat-line rules. In production these
  become *quality flags* attached to the score rather than silent drops (see
  :mod:`eegscore.monitoring`).
"""
from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class PreprocessParams:
    l_freq: float = 1.0
    h_freq: float = 45.0
    notch: float | None = 50.0
    reference: str = "average"
    resample_to: float | None = 250.0
    amplitude_reject_uv: float = 150.0
    flat_threshold_uv: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "PreprocessParams":
        pcfg, dcfg = cfg["preprocess"], cfg["data"]
        return cls(
            l_freq=pcfg["l_freq"], h_freq=pcfg["h_freq"], notch=pcfg.get("notch"),
            reference=pcfg.get("reference", "average"), resample_to=dcfg.get("resample_to"),
            amplitude_reject_uv=pcfg["amplitude_reject_uv"],
            flat_threshold_uv=pcfg["flat_threshold_uv"],
        )


def preprocess_raw(raw: mne.io.BaseRaw, p: PreprocessParams) -> mne.io.BaseRaw:
    raw = raw.copy()
    if p.notch:
        raw.notch_filter(p.notch, verbose="ERROR")
    raw.filter(p.l_freq, p.h_freq, fir_design="firwin", verbose="ERROR")
    if p.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if p.resample_to and abs(raw.info["sfreq"] - p.resample_to) > 1e-6:
        raw.resample(p.resample_to, verbose="ERROR")
    return raw


def window_quality_mask(X: np.ndarray, amplitude_reject_uv: float, flat_threshold_uv: float
                        ) -> np.ndarray:
    """Return boolean mask of windows that pass amplitude/flat-line gates. X in Volts."""
    Xuv = X * 1e6
    ptp = Xuv.max(-1) - Xuv.min(-1)          # (n_win, n_ch)
    std = Xuv.std(-1)
    ok_amp = (ptp < amplitude_reject_uv).all(-1)
    ok_flat = (std > flat_threshold_uv).all(-1)
    return ok_amp & ok_flat


def window_raw(raw: mne.io.BaseRaw, window_sec: float, step_sec: float, p: PreprocessParams
               ) -> tuple[np.ndarray, np.ndarray]:
    """Slice a continuous recording into overlapping windows and apply quality gates.

    Returns ``(X_kept, kept_mask)`` where ``kept_mask`` covers all candidate windows so
    callers can report rejection rates.
    """
    sf = raw.info["sfreq"]
    n_win, n_step = int(round(window_sec * sf)), int(round(step_sec * sf))
    data = raw.get_data()                     # (n_ch, n_samples) Volts
    starts = np.arange(0, data.shape[1] - n_win + 1, n_step)
    if len(starts) == 0:
        return np.empty((0, data.shape[0], n_win)), np.zeros(0, bool)
    X = np.stack([data[:, s:s + n_win] for s in starts])
    kept = window_quality_mask(X, p.amplitude_reject_uv, p.flat_threshold_uv)
    return X[kept], kept


def preprocess_array(X: np.ndarray, sfreq: float, p: PreprocessParams) -> np.ndarray:
    """Array-only version used by the inference service (no MNE Raw object needed).

    X: (n_channels, n_samples) in Volts at ``sfreq``. Returns filtered, re-referenced,
    resampled array. Kept separate from :func:`preprocess_raw` so serving has no
    dependency on file readers.
    """
    x = np.ascontiguousarray(X, dtype=np.float64)
    if p.notch:
        x = mne.filter.notch_filter(x, sfreq, p.notch, verbose="ERROR")
    x = mne.filter.filter_data(x, sfreq, p.l_freq, p.h_freq, verbose="ERROR")
    if p.reference == "average":
        x = x - x.mean(0, keepdims=True)
    if p.resample_to and abs(sfreq - p.resample_to) > 1e-6:
        x = mne.filter.resample(x, up=p.resample_to, down=sfreq, verbose="ERROR")
    return x
