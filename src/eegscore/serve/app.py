"""FastAPI inference service.

    eegscore-serve            # or: uvicorn eegscore.serve.app:app --port 8000

Endpoints
---------
GET  /health        liveness + model version
GET  /model         metadata: channels, sampling rate, metrics from training
POST /score         body: {"sfreq": 500, "ch_names": [...], "data": [[...], ...]}  (Volts or uV)
                    -> calibrated 0-100 score, confidence, quality flags, drift status,
                       per-window scores, top contributing feature families
POST /score/edf     multipart upload of an EDF file (convenience for batch/QA use)

The service reuses the *exact* training code path (preprocess_array -> FeatureExtractor
-> model -> Scorer), so there is no train/serve skew. Every response also carries the
signal-quality flags and a PSI drift status so downstream systems can decide whether
to trust the number.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import __version__
from ..config import resolve
from ..features import FeatureExtractor
from ..monitoring import FeatureReference, signal_quality
from ..preprocess import PreprocessParams, preprocess_array, window_quality_mask
from ..scoring import Scorer

log = logging.getLogger("eegscore.serve")
ART = Path(os.environ.get("EEGSCORE_ARTIFACTS", resolve("artifacts")))


class ScoreRequest(BaseModel):
    sfreq: float = Field(..., description="sampling rate of `data` in Hz")
    ch_names: list[str] = Field(..., description="channel names in 10-20 notation (e.g. 'Fp1')")
    data: list[list[float]] = Field(..., description="(n_channels, n_samples) EEG samples")
    units: str = Field("uV", description="'uV' or 'V'")


class Bundle:
    """Loads all artifacts once at startup."""

    def __init__(self, art: Path):
        self.art = art
        self.model = joblib.load(art / "model.joblib")
        self.scorer = Scorer.from_json(art / "scorer.json")
        self.ref = FeatureReference.from_json(art / "feature_reference.json")
        self.feature_names = json.loads((art / "feature_names.json").read_text())
        self.spec = json.loads((art / "data_spec.json").read_text())
        self.metrics = json.loads((art / "metrics.json").read_text()) if (art / "metrics.json").exists() else {}
        p = self.spec["preprocess"]
        self.pp = PreprocessParams(l_freq=p["l_freq"], h_freq=p["h_freq"], notch=p.get("notch"),
                                   reference=p.get("reference", "average"),
                                   resample_to=self.spec.get("resample_to"),
                                   amplitude_reject_uv=p["amplitude_reject_uv"],
                                   flat_threshold_uv=p["flat_threshold_uv"])
        f = self.spec["features"]
        self.fx = FeatureExtractor(sfreq=self.spec["sfreq"], ch_names=self.spec["ch_names"],
                                   bands={k: tuple(v) for k, v in f["bands"].items()},
                                   multitaper_bandwidth=f["multitaper_bandwidth"],
                                   aperiodic_fit_range=tuple(f["aperiodic_fit_range"]),
                                   connectivity_band=f["connectivity_band"], fmax=p["h_freq"])
        self.n_samples = int(self.spec["n_samples"])
        self.loaded_at = time.time()

    # ------------------------------------------------------------------ core path
    def score_array(self, X: np.ndarray, sfreq: float, ch_names: list[str]) -> dict[str, Any]:
        t0 = time.time()
        # channel alignment (order + presence)
        want = self.spec["ch_names"]
        norm = {c.replace("EEG ", "").strip(): i for i, c in enumerate(ch_names)}
        missing = [c for c in want if c not in norm]
        if missing:
            raise HTTPException(422, f"missing channels {missing}; expected {want}")
        X = X[[norm[c] for c in want]]
        flags = signal_quality(X, sfreq, expected_n_ch=len(want),
                               amp_uv=self.pp.amplitude_reject_uv, flat_uv=self.pp.flat_threshold_uv,
                               line_freq=self.pp.notch or 50.0)
        x = preprocess_array(X, sfreq, self.pp)
        step = self.n_samples // 2
        starts = np.arange(0, x.shape[1] - self.n_samples + 1, step)
        if len(starts) == 0:
            raise HTTPException(422, f"need at least {self.n_samples / self.spec['sfreq']:.1f} s of data")
        W = np.stack([x[:, s:s + self.n_samples] for s in starts])
        ok = window_quality_mask(W, self.pp.amplitude_reject_uv, self.pp.flat_threshold_uv)
        if ok.sum() == 0:
            flags.append("all_windows_rejected")
            ok[:] = True                                   # score anyway but flag it
        F = self.fx.transform(W[ok])[self.feature_names]
        prob = self.model.predict_proba(F)[:, 1]
        drift = self.ref.drift_report(F)
        top_drift = drift.head(5)[["feature", "psi", "status"]].to_dict("records")
        drift_status = "alert" if (drift["status"] == "alert").mean() > 0.2 else \
                       "warn" if (drift["status"] != "ok").mean() > 0.2 else "ok"
        out = self.scorer.summarise(prob, flags)
        out.update({
            "window_scores": [round(float(s), 1) for s in self.scorer.window_scores(prob)],
            "windows_rejected": int((~ok).sum()),
            "drift": {"status": drift_status, "top": top_drift},
            "model_version": __version__,
            "latency_ms": round(1000 * (time.time() - t0), 1),
        })
        return out


app = FastAPI(title="EEG cognitive-load scoring", version=__version__)
_bundle: Bundle | None = None


def bundle() -> Bundle:
    global _bundle
    if _bundle is None:
        if not (ART / "model.joblib").exists():
            raise HTTPException(503, f"no trained artifacts in {ART}; run eegscore-train first")
        _bundle = Bundle(ART)
        log.info("loaded artifacts from %s", ART)
    return _bundle


@app.get("/health")
def health():
    ok = (ART / "model.joblib").exists()
    return {"status": "ok" if ok else "no_model", "version": __version__, "artifacts": str(ART)}


@app.get("/model")
def model_info():
    b = bundle()
    fm = b.metrics.get("feature_model", {})
    return {"ch_names": b.spec["ch_names"], "sfreq": b.spec["sfreq"], "window_sec": b.spec["window_sec"],
            "n_features": len(b.feature_names), "cv_auc": fm.get("auc"), "cv_auc_ci95": fm.get("auc_ci95"),
            "recording_auc": fm.get("recording_auc"), "trained_config": b.metrics.get("config")}


@app.post("/score")
def score(req: ScoreRequest):
    X = np.asarray(req.data, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(req.ch_names):
        raise HTTPException(422, "data must be (n_channels, n_samples) matching ch_names")
    if req.units.lower() == "uv":
        X = X * 1e-6
    return bundle().score_array(X, req.sfreq, req.ch_names)


@app.post("/score/edf")
async def score_edf(file: UploadFile = File(...)):
    import mne
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eeg = [c for c in raw.ch_names if c.startswith("EEG ") and "A2-A1" not in c]
        raw.pick(eeg)
        return bundle().score_array(raw.get_data(), raw.info["sfreq"], raw.ch_names)
    finally:
        os.unlink(path)


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("eegscore.serve.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
