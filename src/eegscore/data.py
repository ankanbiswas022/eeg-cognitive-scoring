"""Data loading for the PhysioNet 'EEG During Mental Arithmetic Tasks' dataset (EEGMAT).

Each subject has two EDF recordings: ``SubjectXX_1.edf`` (3 min eyes-closed rest) and
``SubjectXX_2.edf`` (1 min serial-subtraction mental arithmetic). We treat the task as
the *high cognitive load / stress* class (label 1) and rest as label 0.

The output of :func:`build_dataset` is a :class:`WindowedDataset` whose fields are
plain NumPy arrays so downstream code (features, models, tests) never depends on MNE
objects. Subject identifiers are kept as the ``groups`` array so that every
validation split is subject-wise (no leakage of a subject across train/test).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .preprocess import PreprocessParams, preprocess_raw, window_raw

log = logging.getLogger(__name__)
mne.set_log_level("WARNING")

REST_SUFFIX, TASK_SUFFIX = "_1", "_2"
_SUBJ_RE = re.compile(r"(Subject\d+)_[12]\.edf")


@dataclass
class WindowedDataset:
    X: np.ndarray                     # (n_windows, n_channels, n_samples), float32, Volts
    y: np.ndarray                     # (n_windows,) int  0=rest, 1=task
    groups: np.ndarray                # (n_windows,) int subject id
    ch_names: list[str]
    sfreq: float
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)   # per-window metadata

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        n_subj = len(np.unique(self.groups))
        return (f"{len(self)} windows | {self.X.shape[1]} ch x {self.X.shape[2]} samples "
                f"@ {self.sfreq:g} Hz | {n_subj} subjects | positive fraction "
                f"{np.mean(self.y):.2f}")


def list_subjects(raw_dir: Path) -> list[str]:
    ids = {m.group(1) for p in raw_dir.glob("Subject*_?.edf") if (m := _SUBJ_RE.match(p.name))}
    return sorted(ids)


def load_subject_info(path: Path) -> pd.DataFrame:
    """subject-info.csv columns: Subject, Age, Gender, Recording date,
    Number of subtractions, Count quality (1 = good counter, 0 = bad)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_edf(path: Path, drop_channels: list[str] | None = None,
             eeg_only: bool = True) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    to_drop = [c for c in (drop_channels or []) if c in raw.ch_names]
    if eeg_only:
        to_drop += [c for c in raw.ch_names if not c.startswith("EEG ")]
        to_drop += [c for c in raw.ch_names if "A2-A1" in c]   # reference derivation channel
    raw.drop_channels(sorted(set(to_drop)))
    # attach a standard 10-20 montage so topographic explainability plots work
    raw.rename_channels({c: c.replace("EEG ", "") for c in raw.ch_names})
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


def build_dataset(cfg: dict, subjects: list[str] | None = None,
                  raw_dir: Path | None = None) -> WindowedDataset:
    """Load, preprocess and window every subject in ``raw_dir``.

    Rest recordings are cropped to the final ``rest_minutes_used`` minutes so that both
    classes contribute a comparable number of windows per subject (class balance and
    the subject-wise validation both benefit from this).
    """
    dcfg = cfg["data"]
    raw_dir = Path(raw_dir or dcfg["raw_dir"])
    subjects = subjects or list_subjects(raw_dir)
    if not subjects:
        raise FileNotFoundError(f"No Subject*_?.edf files found in {raw_dir}")

    params = PreprocessParams.from_config(cfg)
    Xs, ys, gs, metas = [], [], [], []
    ch_names, sfreq = None, None
    for sid in subjects:
        subj_num = int(re.sub(r"\D", "", sid))
        for suffix, label in ((REST_SUFFIX, 0), (TASK_SUFFIX, 1)):
            f = raw_dir / f"{sid}{suffix}.edf"
            if not f.exists():
                log.warning("missing %s", f)
                continue
            raw = read_edf(f, dcfg.get("drop_channels"), dcfg.get("eeg_channels_only", True))
            if label == 0 and dcfg.get("rest_minutes_used"):
                keep = dcfg["rest_minutes_used"] * 60.0
                tmax = raw.times[-1]
                raw.crop(tmin=max(0.0, tmax - keep), tmax=tmax)
            raw = preprocess_raw(raw, params)
            X, kept = window_raw(raw, dcfg["window_sec"], dcfg["step_sec"], params)
            if ch_names is None:
                ch_names, sfreq = list(raw.ch_names), float(raw.info["sfreq"])
            Xs.append(X)
            ys.append(np.full(len(X), label))
            gs.append(np.full(len(X), subj_num))
            metas.append(pd.DataFrame({"subject": sid, "label": label,
                                       "window_idx": np.arange(len(X)),
                                       "n_rejected": int(len(kept) - kept.sum())}))
            log.info("%s label=%d -> %d windows (%d rejected)", sid, label, len(X),
                     len(kept) - kept.sum())
    X = np.concatenate(Xs).astype(np.float32)
    ds = WindowedDataset(X, np.concatenate(ys), np.concatenate(gs), ch_names, sfreq,
                         pd.concat(metas, ignore_index=True))
    log.info("dataset: %s", ds.summary())
    return ds
