"""Deep time-series models on raw EEG windows (PyTorch).

Three architectures share one training wrapper so they can be swapped from config:

* ``eegnet``  - compact EEGNet-style (temporal conv -> depthwise spatial conv -> separable
  conv), Lawhern et al. 2018. Strong baseline for small EEG datasets.
* ``tcn``     - dilated causal temporal CNN over a learned spatial projection.
* ``gru``     - bidirectional GRU over a learned spatial projection.

:class:`TorchClassifier` exposes a scikit-learn-like ``fit`` / ``predict_proba`` so the
same validation code handles both classical and deep models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- nets
class EEGNet(nn.Module):
    def __init__(self, n_ch: int, n_samples: int, n_classes: int = 2, F1: int = 8, D: int = 2,
                 F2: int = 16, kern: int = 64, drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),      # depthwise spatial
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )
        with torch.no_grad():
            n_flat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_samples))).numel()
        self.head = nn.Linear(n_flat, n_classes)

    def forward(self, x):                       # x: (B, C, T)
        x = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(x.flatten(1))


class _CausalBlock(nn.Module):
    def __init__(self, ch, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=dil)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = F.gelu(self.c1(F.pad(x, (self.pad, 0))))
        y = self.drop(F.gelu(self.c2(F.pad(y, (self.pad, 0)))))
        return self.norm(x + y)


class TCN(nn.Module):
    def __init__(self, n_ch, n_samples, n_classes=2, hidden=32, k=5, n_blocks=5, drop=0.3):
        super().__init__()
        self.spatial = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.Sequential(*[_CausalBlock(hidden, k, 2 ** i, drop) for i in range(n_blocks)])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.blocks(self.spatial(x))
        return self.head(h.mean(-1))


class GRUNet(nn.Module):
    def __init__(self, n_ch, n_samples, n_classes=2, hidden=32, pool=8, drop=0.3):
        super().__init__()
        self.spatial = nn.Conv1d(n_ch, hidden, pool, stride=pool)   # downsample + mix channels
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        h = F.gelu(self.spatial(x)).transpose(1, 2)      # (B, T', hidden)
        out, _ = self.gru(h)
        return self.head(self.drop(out.mean(1)))


def build_net(arch: str, n_ch: int, n_samples: int) -> nn.Module:
    return {"eegnet": EEGNet, "tcn": TCN, "gru": GRUNet}[arch](n_ch, n_samples)


# ------------------------------------------------------------------------- wrapper
@dataclass
class TorchClassifier:
    arch: str = "eegnet"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    device: str | None = None
    verbose: bool = False

    def __post_init__(self):
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: nn.Module | None = None
        self.scale_: float | None = None

    # per-channel robust scaling fitted on training data only (no test leakage)
    def _normalise(self, X):
        return (X / self.scale_).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "TorchClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scale_ = np.median(np.abs(X)) * 1.4826 + 1e-12          # MAD-based global scale
        self._input_shape = (int(X.shape[1]), int(X.shape[2]))
        Xt = torch.from_numpy(self._normalise(X))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.net = build_net(self.arch, X.shape[1], X.shape[2]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        w = torch.tensor(np.bincount(y, minlength=2), dtype=torch.float32)
        w = (w.sum() / (2 * w)).to(self.device)                       # class re-weighting
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        for ep in range(self.epochs):
            self.net.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb = xb + 0.05 * torch.randn_like(xb)                 # light noise augmentation
                loss = F.cross_entropy(self.net(xb), yb, weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            if self.verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                msg = f"[{self.arch}] epoch {ep+1}/{self.epochs} loss {tot/len(ds):.4f}"
                if X_val is not None:
                    from sklearn.metrics import roc_auc_score
                    msg += f" val AUC {roc_auc_score(y_val, self.predict_proba(X_val)[:, 1]):.3f}"
                log.info(msg)
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xt = torch.from_numpy(self._normalise(X))
        outs = []
        for i in range(0, len(Xt), 256):
            outs.append(F.softmax(self.net(Xt[i:i + 256].to(self.device)), -1).cpu())
        return torch.cat(outs).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    # ---- persistence -------------------------------------------------------------
    def save(self, path):
        torch.save({"arch": self.arch, "scale": self.scale_, "state": self.net.state_dict(),
                    "shape": tuple(self._input_shape)}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        obj = cls(arch=ck["arch"], device=device)
        obj.scale_ = ck["scale"]
        obj.net = build_net(ck["arch"], *ck["shape"]).to(obj.device)
        obj.net.load_state_dict(ck["state"])
        obj.net.eval()
        obj._input_shape = tuple(ck["shape"])
        return obj

    def saliency(self, X: np.ndarray, target: int = 1) -> np.ndarray:
        """Gradient x input saliency per (channel, time) for explainability."""
        self.net.eval()
        Xt = torch.from_numpy(self._normalise(X)).to(self.device).requires_grad_(True)
        logits = self.net(Xt)[:, target].sum()
        logits.backward()
        return (Xt.grad * Xt).detach().cpu().numpy()
