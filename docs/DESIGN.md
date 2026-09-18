# Production design notes

This document is the "how would this run for real" companion to the code. It is written
the way I would write an internal design doc for an EEG scoring service.

## 1. Problem framing

Input: a multichannel EEG recording (19 ch, 10-20, 250-500 Hz, 1-5 min) from a consumer or
clinical headset. Output: a bounded score (0-100) for cognitive load / stress state, a
confidence level, and quality flags, delivered within a few hundred ms of the recording
ending (or streamed every 2 s for live use).

The ML task underneath is binary (rest vs. high-load) at the window level, but the
*product* contract is at the recording level. That distinction drives most decisions
below: validation, calibration, aggregation, and monitoring all happen at the recording
level.

## 2. Pipeline

```
EDF / stream ──> preprocess_array ──> windows (4 s, 50 % overlap)
                     │                      │
                     │                      ├─> signal_quality flags (per window)
                     │                      │
                     │                      └─> FeatureExtractor (~300 features)
                     │                              │
                     │                              ├─> FeatureReference.drift_report (PSI)
                     │                              │
                     │                              └─> LightGBM ──> p(window)
                     │                                                 │
                     └── (challenger) EEGNet/TCN on raw windows ───────┤ (late fusion, optional)
                                                                       │
                                                        Scorer (isotonic calibration,
                                                        median + IQR over windows,
                                                        confidence, flags)
                                                                       │
                                                                  JSON response
```

One code path for training and serving: `preprocess_array` and `FeatureExtractor` are
the same objects used by `train.py` and by the FastAPI app. The `data_spec.json` artifact
pins channel order, sampling rate, window length and filter settings so the service
refuses inputs that do not match.

## 3. Why a feature model is the champion and the deep model the challenger

| | Feature model (LightGBM) | Deep model (EEGNet / TCN / GRU) |
|---|---|---|
| Data needed | tens of subjects | hundreds+ |
| Cross-subject generalisation on small data | good | fragile; over-fits subject identity |
| Explainability | SHAP on neuroscience-meaningful features, scalp maps | saliency only |
| Inference cost | ms on CPU | ms on CPU (small nets) |
| Regulatory / scientific review | easy to defend | harder |

With 36 subjects, the feature model wins or ties; the deep model becomes attractive once
the cohort is 10x larger or when the input is high-density / multimodal. The wrapper keeps
the interface identical so swapping is a config change and a re-run of the same
validation.

## 4. Validation contract

* **Subject-wise GroupKFold** only. A random window split leaks subject identity and
  reports inflated numbers.
* Headline metrics: OOF AUC with a **subject-level cluster bootstrap CI**, and
  **recording-level AUC** (median of window probabilities per recording).
* **Recording-level label permutation test** to check the pipeline cannot learn from
  nothing.
* Calibration (Brier before/after isotonic) reported because the product consumes
  probabilities, not labels.

## 5. Robustness to noisy real-world data

* Quality gates (amplitude, flat line, mains contamination, channel count) run *before*
  scoring and are returned as flags. A score is never silently produced from bad data.
* Features are robust by construction (log powers, relative powers, ratios) and the
  scaler/imputer are fitted inside the CV loop.
* Training-time augmentation for the deep model (Gaussian noise); MAD-based scaling.
* Next steps for a real deployment: per-device calibration sessions (headset-specific
  reference), ICA/ASR artefact handling for eye blinks, and subject-specific baselines
  (score relative to a person's own rest).

## 6. Monitoring and continuous improvement

* **Feature drift**: PSI of every feature against the training histogram, returned with
  each response and logged. Alert if >20 % of features exceed PSI 0.25.
* **Prediction drift**: distribution of scores and of the "confidence" field per device
  / site / week.
* **Quality-flag rate**: fraction of windows rejected per device; a rising rate usually
  means hardware or electrode-gel problems, not model problems.
* **Ground truth loop**: where task labels or self-report exist, append to a labelled
  store; retraining is a scheduled job that re-runs `train.py` and promotes the model
  only if recording-level AUC on a frozen hold-out cohort does not regress.
* **Versioning**: every artifact directory carries `metrics.json` and the config path;
  the service exposes them at `/model`.

## 7. Serving

FastAPI + uvicorn, stateless, artifacts loaded once at startup. Docker image is CPU-only.
Latency on a laptop CPU for a 60 s recording (29 windows): tens of ms for features and
inference. Batch scoring of EDF files is the same code behind `/score/edf`.

## 8. What I would do next with more time

1. Multimodal fusion: ECG (present in this dataset) for HRV features, PPG on wearables.
2. Sequence model over the *window feature time series* (GRU on 2 s hops) to capture
   load dynamics rather than treating windows as i.i.d.
3. Domain adaptation across headsets (Euclidean alignment / covariance re-centering).
4. Conformal prediction for per-recording uncertainty instead of the margin heuristic.
