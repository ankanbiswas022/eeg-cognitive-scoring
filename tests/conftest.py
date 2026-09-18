import numpy as np
import pytest

CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4",
      "O1", "O2", "Fz", "Cz", "Pz"]
SFREQ = 250.0


def synth_eeg(n_windows: int, n_samples: int = 1000, alpha_uv: float = 10.0, seed: int = 0,
              n_ch: int = len(CH)) -> np.ndarray:
    """Pink-ish noise + occipital alpha, in Volts, shape (n_windows, n_ch, n_samples)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SFREQ
    white = rng.standard_normal((n_windows, n_ch, n_samples))
    # crude 1/f: cumulative sum then detrend
    pink = np.cumsum(white, axis=-1)
    pink -= pink.mean(-1, keepdims=True)
    pink /= pink.std(-1, keepdims=True) + 1e-12
    x = 5.0 * pink + 2.0 * white
    alpha = alpha_uv * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi, (n_windows, 1, 1)))
    occ = [i for i, c in enumerate(CH[:n_ch]) if c in ("O1", "O2", "P3", "P4", "Pz")]
    x[:, occ, :] += alpha
    return (x * 1e-6).astype(np.float32)


@pytest.fixture
def small_dataset():
    """Two synthetic 'subjects' x two classes with a real spectral difference."""
    X0 = synth_eeg(40, alpha_uv=12.0, seed=1)      # rest: strong alpha
    X1 = synth_eeg(40, alpha_uv=3.0, seed=2)       # task: suppressed alpha
    X = np.concatenate([X0, X1])
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    groups = np.r_[np.repeat([1, 2, 3, 4], 10), np.repeat([1, 2, 3, 4], 10)]
    return X, y, groups
