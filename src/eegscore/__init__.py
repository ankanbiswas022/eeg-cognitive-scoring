"""eegscore: EEG cognitive-load / stress-state scoring pipeline.

Modules
-------
data        load PhysioNet EEGMAT EDF files -> windowed epochs with labels/groups
preprocess  filtering, referencing, window-level artifact rejection
features    interpretable spectral / aperiodic / asymmetry / connectivity features
models      classical (scikit-learn, LightGBM) and deep (PyTorch) models
validation  subject-wise cross-validation and statistical evaluation
explain     model explainability (permutation importance, SHAP, saliency)
scoring     calibrated 0-100 score with confidence and quality flags
monitoring  signal-quality gates and feature-drift (PSI) checks
serve       FastAPI inference service
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eegscore")
except PackageNotFoundError:  # editable/dev install
    __version__ = "0.1.0"
