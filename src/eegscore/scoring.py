"""Turn model probabilities into a calibrated, bounded cognitive-load score.

A raw classifier probability is not a score a clinician or product can consume:
* it is not calibrated (LightGBM/EEGNet probabilities are over-confident),
* it is per-window (noisy), whereas the product needs one number per recording,
* it carries no notion of *confidence* or *data quality*.

:class:`Scorer` fixes all three. It is fitted on **out-of-fold** probabilities from
cross-validation so calibration is not learned on training predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Scorer:
    score_min: float = 0.0
    score_max: float = 100.0
    low_confidence_margin: float = 0.15
    calibration: str = "isotonic"
    _iso: IsotonicRegression | None = field(default=None, repr=False)
    _x: np.ndarray | None = field(default=None, repr=False)
    _y: np.ndarray | None = field(default=None, repr=False)

    def fit(self, oof_prob: np.ndarray, y: np.ndarray) -> "Scorer":
        if self.calibration == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(oof_prob, y)
            self._x, self._y = self._iso.X_thresholds_, self._iso.y_thresholds_
        return self

    def calibrate(self, prob: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(prob, dtype=float), 0, 1)
        if self._x is None:
            return prob
        return np.interp(prob, self._x, self._y)

    def window_scores(self, prob: np.ndarray) -> np.ndarray:
        p = self.calibrate(prob)
        return self.score_min + p * (self.score_max - self.score_min)

    def summarise(self, prob: np.ndarray, quality_flags: list[str] | None = None) -> dict:
        """Recording-level summary from window probabilities."""
        p = self.calibrate(prob)
        s = self.window_scores(prob)
        conf = float(np.mean(np.abs(p - 0.5) >= self.low_confidence_margin))
        return {
            "score": float(np.median(s)),
            "score_iqr": [float(np.percentile(s, 25)), float(np.percentile(s, 75))],
            "n_windows": int(len(s)),
            "fraction_confident_windows": conf,
            "label": "high_load" if np.median(p) >= 0.5 else "rest",
            "confidence": "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low",
            "quality_flags": list(quality_flags or []),
        }

    # ---- persistence -------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        d = {"score_min": self.score_min, "score_max": self.score_max,
             "low_confidence_margin": self.low_confidence_margin, "calibration": self.calibration,
             "x": None if self._x is None else self._x.tolist(),
             "y": None if self._y is None else self._y.tolist()}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Scorer":
        d = json.loads(Path(path).read_text())
        obj = cls(d["score_min"], d["score_max"], d["low_confidence_margin"], d["calibration"])
        if d["x"] is not None:
            obj._x, obj._y = np.array(d["x"]), np.array(d["y"])
        return obj
