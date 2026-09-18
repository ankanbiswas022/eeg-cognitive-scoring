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
features computed on 4-s windows (50 % overlap), each z-scored against the person's own
60-s eyes-closed calibration segment (baseline-relative). Isotonic calibration on out-of-fold
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
* A 60-s eyes-closed calibration segment is **required** at inference; without it the
  service refuses to score.
* Rest windows are drawn from the final minute of a 3-min rest to balance classes; the
  first two minutes are unused.

## Monitoring in deployment
Quality flags per request; PSI feature drift against the training reference; score and
confidence distribution per device. Retraining gated on hold-out recording-level AUC.

## Ethical considerations
EEG-derived "stress" or "cognitive" scores can be misused for surveillance or employment
decisions. Any deployment should be consented, transparent about accuracy limits, and
should never be the sole basis for a decision about a person.

## Results (v0.1, 36 subjects, subject-wise 6-fold CV)

| model | OOF AUC [95 % CI] | balanced acc. (calibrated) | recording AUC | recording acc. (calibrated) |
|---|---|---|---|---|
| LightGBM, baseline-relative features (champion) | 0.915 [0.887, 0.941] | 0.83 | 0.982 | 0.917 |
| LightGBM, absolute features (ablation) | 0.745 [0.666, 0.816] | 0.70 | 0.781 | 0.750 |
| EEGNet, raw windows (challenger) | 0.858 [0.804, 0.906] | 0.77 | 0.913 | 0.833 |
| late fusion (mean of probabilities) | 0.929 | | | |

Recording-level label permutation (20 permutations): null mean 0.46, null max 0.58,
p = 1/21. Brier 0.135 → 0.115 after isotonic calibration.

## What the model uses (and a caveat)

SHAP importance is dominated by absolute band power (log-power family 3.05, relative
power 1.49, Hjorth 0.47, aperiodic 0.47). The top single features are gamma power at
Fp1/T4/Cz and delta power at Fp1/F8. In an eyes-closed mental-arithmetic task, frontal and
temporal gamma/delta are plausibly **muscle (jaw, forehead) and ocular activity associated
with effort**, not only cortical oscillations. The score therefore captures physiological
effort/arousal, which is legitimate for a load/stress product, but it should not be
described as a pure cortical marker. Planned mitigations: restrict features to < 30 Hz and
re-validate; EOG/EMG regression or ICA; an independent stress reference (HRV from the ECG
channel in this dataset). The EEGNet saliency, by contrast, concentrates below 13 Hz
(delta 34 %, theta 33 %, alpha 21 %).
