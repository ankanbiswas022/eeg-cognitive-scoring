# Model card: EEG cognitive-load scorer v0.1

## Intended use
Score the level of cognitive load (rest vs. demanding mental task) from 1–5 minutes of
multichannel scalp EEG, as a bounded 0–100 number with confidence and quality flags.
Intended as a *screening / monitoring* signal for research and product prototyping.
**Not a medical device; not validated for clinical decisions.**

## Training data
PhysioNet EEGMAT v1.0.0 (Zyma et al. 2019): 36 healthy university students (ages 17–26),
19-channel 10-20 EEG at 500 Hz, eyes-closed rest (3 min, last 60 s used) and eyes-closed
serial subtraction (60 s). Recorded in a laboratory with a research amplifier.

## Model
Champion: LightGBM on ~300 interpretable spectral / aperiodic / asymmetry / connectivity
features computed on 4-s windows (50 % overlap). Isotonic calibration on out-of-fold
probabilities. Challenger: EEGNet on raw windows (PyTorch).

## Evaluation
Subject-wise 6-fold cross-validation; subject-level bootstrap CI; recording-level metrics;
recording-level label permutation test. See `artifacts/metrics.json`.

## Known limitations
* Small, homogeneous cohort (students, one site, one amplifier). Generalisation to
  consumer headsets, other age groups, or field recordings is **not established**.
* "Cognitive load" is operationalised as mental arithmetic; stress was not independently
  measured (no cortisol / self-report / HRV in the labels).
* Both conditions are eyes-closed; eyes-open use requires re-validation.
* Rest windows are drawn from the final minute of a 3-min rest to balance classes; the
  first two minutes are unused.

## Monitoring in deployment
Quality flags per request; PSI feature drift against the training reference; score and
confidence distribution per device. Retraining gated on hold-out recording-level AUC.

## Ethical considerations
EEG-derived "stress" or "cognitive" scores can be misused for surveillance or employment
decisions. Any deployment should be consented, transparent about accuracy limits, and
should never be the sole basis for a decision about a person.
