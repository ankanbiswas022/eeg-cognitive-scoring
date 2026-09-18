"""Production monitoring: signal-quality gates and feature drift.

Two questions a deployed EEG scorer must answer on every request:

1. **Is this input trustworthy?**  :func:`signal_quality` returns human-readable flags
   (flat channel, saturated channel, mains contamination, wrong channel count) that
   are attached to the score instead of silently producing a number from garbage.
2. **Does this input look like the training distribution?**  :class:`FeatureReference`
   stores per-feature histograms from the training set; :func:`psi` computes the
   Population Stability Index of new features against them. PSI > 0.25 on the key
   features is the standard trigger for a retraining / investigation ticket.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# --------------------------------------------------------------------- signal quality
def signal_quality(X: np.ndarray, sfreq: float, expected_n_ch: int | None = None,
                   amp_uv: float = 150.0, flat_uv: float = 0.5, line_freq: float = 50.0
                   ) -> list[str]:
    """X: (n_ch, n_samples) Volts, one window. Returns a list of flag strings (empty = ok)."""
    flags = []
    if expected_n_ch is not None and X.shape[0] != expected_n_ch:
        flags.append(f"channel_count_{X.shape[0]}_expected_{expected_n_ch}")
    Xuv = X * 1e6
    ptp = Xuv.max(-1) - Xuv.min(-1)
    std = Xuv.std(-1)
    if (ptp > amp_uv).any():
        flags.append(f"high_amplitude_ch_{np.where(ptp > amp_uv)[0].tolist()}")
    if (std < flat_uv).any():
        flags.append(f"flat_ch_{np.where(std < flat_uv)[0].tolist()}")
    if not np.isfinite(X).all():
        flags.append("non_finite_values")
    # mains contamination: power at line frequency vs. neighbouring bins
    f, p = welch(Xuv, sfreq, nperseg=min(int(sfreq), X.shape[-1]), axis=-1)
    if f.max() > line_freq + 3:
        at = p[:, np.abs(f - line_freq) <= 1].mean()
        near = p[:, (np.abs(f - line_freq) > 3) & (np.abs(f - line_freq) <= 8)].mean()
        if at > 5 * near:
            flags.append("mains_contamination")
    return flags


# --------------------------------------------------------------------- drift (PSI)
def psi(ref_counts: np.ndarray, new_counts: np.ndarray, eps: float = 1e-4) -> float:
    r = ref_counts / max(ref_counts.sum(), 1) + eps
    n = new_counts / max(new_counts.sum(), 1) + eps
    return float(np.sum((n - r) * np.log(n / r)))


@dataclass
class FeatureReference:
    """Per-feature histogram edges + counts from the training set."""
    edges: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, list[int]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, F: pd.DataFrame, n_bins: int = 10) -> "FeatureReference":
        ref = cls()
        for c in F.columns:
            v = F[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
            qs[0], qs[-1] = -np.inf, np.inf
            qs = np.unique(qs)
            ref.edges[c] = qs.tolist()
            ref.counts[c] = np.histogram(v, qs)[0].tolist()
            ref.means[c], ref.stds[c] = float(v.mean()), float(v.std() + 1e-12)
        return ref

    def drift_report(self, F: pd.DataFrame, warn: float = 0.1, alert: float = 0.25
                     ) -> pd.DataFrame:
        rows = []
        for c, edges in self.edges.items():
            if c not in F:
                continue
            v = F[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            new = np.histogram(v, np.array(edges))[0]
            s = psi(np.array(self.counts[c]), new)
            z = (v.mean() - self.means[c]) / self.stds[c] if len(v) else np.nan
            rows.append({"feature": c, "psi": s, "mean_shift_z": float(z),
                         "status": "alert" if s >= alert else "warn" if s >= warn else "ok"})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__))

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureReference":
        return cls(**json.loads(Path(path).read_text()))
