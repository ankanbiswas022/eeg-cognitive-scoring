# eeg-cognitive-scoring

**From raw EEG to a calibrated, monitored cognitive-load score, served over an API.**

A production-style pipeline that turns multichannel EEG into a 0–100 cognitive-load /
stress-state score. Built on the public PhysioNet *EEG During Mental Arithmetic Tasks*
dataset (36 subjects, 19 channels, rest vs. serial-subtraction). The design decisions come
from eight years of EEG work (neurofeedback, >100-subject cohorts, source localisation) and
are documented in [docs/DESIGN.md](docs/DESIGN.md).

```
EDF ──> MNE preprocessing ──> 4-s windows ──> ~300 interpretable features ──> LightGBM ──┐
                                   │                                                     ├─> calibrated score
                                   └──> EEGNet / TCN / GRU (PyTorch, challenger) ────────┘   + confidence
                                                                                             + quality flags
   subject-wise CV · bootstrap CI · permutation test · SHAP · PSI drift · FastAPI · Docker    + drift status
```

## Results (subject-wise 6-fold CV, 36 subjects)

RESULTS_TABLE

Numbers are out-of-fold over all windows; CI is a subject-level cluster bootstrap. See
[artifacts/metrics.json](artifacts/metrics.json) and [reports/](reports/) for the full
breakdown, SHAP tables and scalp maps.

TOPOMAPS

## Quick start

```bash
git clone https://github.com/ankanbiswas022/eeg-cognitive-scoring && cd eeg-cognitive-scoring
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e ".[dev]"

pytest -q                                              # unit tests on synthetic EEG (no download)
bash scripts/download_data.sh                          # ~330 MB from PhysioNet
eegscore-train                                         # full run: features, CV, deep model, explainability
eegscore-train --skip-deep --n-perm 0                  # fast run (~1 min)

eegscore-serve                                         # FastAPI on :8000
python scripts/score_example.py data/raw/eegmat/Subject05_2.edf
```

Example response from `POST /score`:

```json
EXAMPLE_RESPONSE
```

## What is in the box

| Module | Purpose |
|---|---|
| `eegscore.data` / `preprocess` | EDF loading, notch/band-pass, average reference, resampling, window quality gates. One code path for training and serving. |
| `eegscore.features` | Multitaper band powers (abs/rel), load ratios (θ/α, β/α, engagement), frontal alpha asymmetry, aperiodic 1/f slope & offset, spectral entropy, Hjorth, alpha-band PLV. |
| `eegscore.models.classical` | scikit-learn pipeline (impute → scale → LightGBM or logistic regression). Champion. |
| `eegscore.models.deep` | EEGNet, dilated TCN, BiGRU on raw windows with a scikit-learn-like wrapper. Challenger. |
| `eegscore.validation` | `GroupKFold` by subject, recording-level metrics, subject bootstrap CI, recording-level permutation test. |
| `eegscore.explain` | Family-level permutation importance, TreeSHAP, channel topomaps, deep saliency by band. |
| `eegscore.scoring` | Isotonic calibration on out-of-fold probabilities → 0–100 score, IQR, confidence, label. |
| `eegscore.monitoring` | Signal-quality flags (flat, saturated, mains, channel count) and PSI feature drift vs. training reference. |
| `eegscore.serve.app` | FastAPI: `/health`, `/model`, `/score` (JSON array), `/score/edf` (file upload). |
| `tests/` | 17 tests on synthetic EEG: feature sanity, leakage guard, model round-trip, calibration, drift, API. |

## Design choices worth arguing about

* **Subject-wise validation only.** Random window splits leak subject identity and inflate
  AUC by 0.1–0.3. Every number here is out-of-fold across subjects.
* **Feature model as champion, deep model as challenger.** With 36 subjects the
  interpretable model is at least as good and far easier to defend to a neuroscientist or a
  regulator; the deep model is wired into the same harness for when the cohort grows.
* **Scores, not probabilities.** Isotonic calibration fitted on out-of-fold predictions,
  recording-level median + IQR, an explicit confidence field, and quality flags that travel
  with the number.
* **Monitoring is part of the model.** Feature reference histograms are an artifact; every
  response carries a PSI drift status.
* **Eyes-closed in both conditions.** The dataset was chosen because rest and task are both
  eyes-closed, so the classifier cannot cheat on eye state via alpha.

## Data

Zyma I., Tukaev S., Seleznov I., Kiyono K., Popov A., Chernykh M., Shpenkov O.
*Electroencephalograms during Mental Arithmetic Task Performance.* Data 2019, 4(1):14.
PhysioNet: https://physionet.org/content/eegmat/1.0.0/ (ODC-BY 1.0).

## Author

Ankan Biswas — PhD (Neuroscience), IISc Bengaluru. EEG / LFP / closed-loop neurofeedback;
papers in *Imaging Neuroscience*, *EJN*, *eNeuro*. [Google Scholar](https://scholar.google.com/citations?user=oG28KRIAAAAJ) · [LinkedIn](https://www.linkedin.com/in/ankan-biswas-45357685/)
