"""
1D CNN Classifier for Stress Detection
========================================
End-to-end deep learning model that classifies stress vs. non-stress
directly from raw (preprocessed) wrist PPG and ACC signals — no manual
feature engineering required.

Architecture
------------
Input: (batch, channels, time_steps)
  channels = 4  [BVP, ACC_x, ACC_y, ACC_z]  resampled to BVP rate (64 Hz)

Network:
  Conv1d block 1 : 32 filters, kernel 7  → BN → ReLU → MaxPool 2
  Conv1d block 2 : 64 filters, kernel 5  → BN → ReLU → MaxPool 2
  Conv1d block 3 : 128 filters, kernel 3 → BN → ReLU → AdaptiveAvgPool
  FC layers       : 256 → 64 → 2
  Dropout (0.4) before each FC layer

Design choices
--------------
- 1D convolutions over time are the right inductive bias for physiological
  signals (local temporal patterns = heartbeats, motion bursts).
- Batch normalization after each conv stabilises training with small datasets.
- Adaptive average pooling decouples architecture from exact input length —
  the same model handles any window size ≥ 64 samples.
- Dropout 0.4 is aggressive but appropriate: WESAD has only 14 subjects and
  overfitting is the primary risk.

Companion: RandomForest baseline on HRV features (see baseline_rf.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from typing import Optional


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StressCNN(nn.Module):
    """
    1D CNN for binary stress classification from multimodal wrist signals.

    Parameters
    ----------
    in_channels : number of input signal channels
                  4 = BVP + ACC xyz (original)
                  6 = BVP + ACC xyz + TEMP + EDA (default, recommended)
    num_classes : 2 (stress / non-stress)
    dropout     : dropout rate before FC layers
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels,  32, kernel=7, pool=2),
            ConvBlock(32,           64, kernel=5, pool=2),
            ConvBlock(64,          128, kernel=3, pool=2),
        )

        # AdaptiveAvgPool collapses time dimension → 128-d vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, channels, time)

        Returns
        -------
        logits : (batch, num_classes)
        """
        h = self.features(x)       # (B, 128, T')
        h = self.pool(h).squeeze(-1)  # (B, 128)
        return self.classifier(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (no grad)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def prepare_cnn_input(
    windows_bvp:  np.ndarray,            # (W, N_bvp)
    windows_acc:  np.ndarray,            # (W, N_acc, 3)
    labels:       np.ndarray,            # (W,)
    windows_temp: Optional[np.ndarray] = None,  # (W, N_temp)
    windows_eda:  Optional[np.ndarray] = None,  # (W, N_eda)
    fs_bvp:       int = 64,
    fs_acc:       int = 32,
) -> TensorDataset:
    """
    Build a TensorDataset suitable for the StressCNN.

    All signals are resampled to BVP rate (64 Hz) so all channels share
    the same time axis.

    Channels:
      0     : BVP  (64 Hz, no resampling needed)
      1–3   : ACC x/y/z (32 Hz → 64 Hz, linear interp)
      4     : TEMP (4 Hz → 64 Hz, linear interp)  — if provided
      5     : EDA  (4 Hz → 64 Hz, linear interp)  — if provided
    """
    from scipy.interpolate import interp1d

    W, N_bvp = windows_bvp.shape
    N_target = N_bvp

    def resample(arr2d, n_orig, n_target):
        """Resample (W, N_orig) → (W, N_target) via linear interpolation."""
        t_orig = np.linspace(0, 1, n_orig)
        t_new  = np.linspace(0, 1, n_target)
        out = np.zeros((W, n_target), dtype=np.float32)
        for i in range(W):
            out[i] = interp1d(t_orig, arr2d[i], kind="linear")(t_new)
        return out

    # ACC: (W, N_acc, 3) → resample each axis → (W, 3, N_target)
    acc_resampled = np.zeros((W, N_target, 3), dtype=np.float32)
    t_acc_orig = np.linspace(0, 1, windows_acc.shape[1])
    t_acc_new  = np.linspace(0, 1, N_target)
    for i in range(W):
        for ax in range(3):
            acc_resampled[i, :, ax] = interp1d(
                t_acc_orig, windows_acc[i, :, ax], kind="linear"
            )(t_acc_new)

    # Stack channels: start with BVP + ACC
    bvp_ch = windows_bvp[:, np.newaxis, :]          # (W, 1, N)
    acc_ch = acc_resampled.transpose(0, 2, 1)        # (W, 3, N)
    channels = [bvp_ch, acc_ch]

    # Optionally add TEMP and EDA
    if windows_temp is not None:
        temp_up = resample(windows_temp, windows_temp.shape[1], N_target)
        channels.append(temp_up[:, np.newaxis, :])   # (W, 1, N)

    if windows_eda is not None:
        eda_up = resample(windows_eda, windows_eda.shape[1], N_target)
        channels.append(eda_up[:, np.newaxis, :])    # (W, 1, N)

    X = np.concatenate(channels, axis=1)             # (W, 4 or 6, N)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(X_t, y_t)


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> dict[str, float]:
    """Evaluate accuracy and F1 on a DataLoader."""
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        logits = model(X.to(device))
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def train_model(
    model:        StressCNN,
    train_dataset: TensorDataset,
    val_dataset:   Optional[TensorDataset] = None,
    n_epochs:      int = 30,
    batch_size:    int = 32,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    device:        Optional[torch.device] = None,
    verbose:       bool = True,
) -> dict[str, list]:
    """
    Full training loop with optional validation.

    Returns history dict with 'train_loss', 'val_accuracy', 'val_f1'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class-weighted loss to handle stress/non-stress imbalance
    labels = np.array([y.item() for _, y in train_dataset])
    class_counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False) if val_dataset else None

    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}

    # Early stopping state
    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    patience = 10  # stop if no F1 improvement for 10 consecutive epochs

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        history["train_loss"].append(loss)

        if val_loader:
            metrics = evaluate(model, val_loader, device)
            history["val_accuracy"].append(metrics["accuracy"])
            history["val_f1"].append(metrics["f1"])

            # Track best model by validation F1
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d} | loss={loss:.4f} | "
                      f"val_acc={metrics['accuracy']:.3f} | val_f1={metrics['f1']:.3f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best F1={best_f1:.3f})")
                break

        elif verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | loss={loss:.4f}")

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ── Model summary ─────────────────────────────────────────────────────────────

def model_summary(model: StressCNN, input_shape: tuple = (4, 3840)) -> None:
    """Print parameter count and layer-by-layer shapes."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStressCNN — {total:,} total params ({trainable:,} trainable)")
    print(f"Input shape: (batch, {input_shape[0]}, {input_shape[1]})")

    x = torch.zeros(1, *input_shape)
    with torch.no_grad():
        h = x
        for i, layer in enumerate(model.features):
            h = layer(h)
            print(f"  features[{i}] → {tuple(h.shape)}")
        h = model.pool(h).squeeze(-1)
        print(f"  pool + squeeze → {tuple(h.shape)}")
    print()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Simulate one subject's windows: 100 windows, 60s at 64 Hz = 3840 samples
    W, N_bvp, N_acc = 100, 3840, 1920
    X_dummy  = torch.randn(W, 4, N_bvp)      # (W, channels, time)
    y_dummy  = torch.randint(0, 2, (W,))

    ds    = TensorDataset(X_dummy, y_dummy)
    model = StressCNN(in_channels=4)

    model_summary(model)

    history = train_model(model, ds, val_dataset=ds, n_epochs=10, batch_size=16, verbose=True)
    print("\nTraining complete.")
    print(f"Final val F1: {history['val_f1'][-1]:.3f}")
