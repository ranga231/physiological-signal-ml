"""
Transformer Classifier for Stress Detection
=============================================
Patch-based Vision-Transformer-style architecture adapted for 1D
multimodal physiological signals.

Architecture
------------
Input : (batch, channels, time)   e.g. (B, 6, 3840) for 60 s @ 64 Hz

1. Patch Embedding
   Split each channel's time axis into non-overlapping patches of
   `patch_size` samples.  Flatten all channels within a patch and
   project to d_model → (B, n_patches, d_model).

2. [CLS] token prepended → (B, 1 + n_patches, d_model).

3. Learnable positional encoding added.

4. Transformer Encoder  (n_layers × multi-head self-attention + FFN).

5. Classification Head  [CLS] → Dropout → Linear(d_model, 2).

Design rationale
----------------
- Patches of 64 samples = 1 second of BVP context — roughly one
  cardiac cycle.  Self-attention then learns which seconds co-vary
  (e.g. stress onset at second 5 correlates with sustained elevation
  at second 30).
- CLS-token classification is standard (BERT, ViT) and isolates the
  aggregation step from the sequence processing.
- Small model (d_model=64, 3 layers) to avoid overfitting on WESAD's
  ~1,300 training windows per fold.  Parameter count ≈ 180 K —
  comparable to the CNN baseline.
- Same input format as StressCNN so both share prepare_cnn_input()
  and run_loso_cnn() without modification.

References
----------
- Vaswani et al. (2017) "Attention Is All You Need"
- Dosovitskiy et al. (2020) "An Image Is Worth 16x16 Words" (ViT)
- Nie et al. (2023) "A Time Series Is Worth 64 Words" (PatchTST)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split (B, C, T) → (B, n_patches, d_model).

    Each patch covers `patch_size` time steps across ALL channels,
    so the token dimension is C × patch_size before projection.
    This lets the model attend to cross-channel patterns within a patch.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size:  int,
        d_model:     int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        tokens : (B, n_patches, d_model)
        """
        B, C, T = x.shape
        n_patches = T // self.patch_size
        # Trim to exact multiple of patch_size
        x = x[:, :, : n_patches * self.patch_size]          # (B, C, n_patches*P)
        x = x.reshape(B, C, n_patches, self.patch_size)     # (B, C, n_p, P)
        x = x.permute(0, 2, 1, 3)                           # (B, n_p, C, P)
        x = x.reshape(B, n_patches, C * self.patch_size)    # (B, n_p, C*P)
        return self.norm(self.proj(x))                       # (B, n_p, d_model)


# ── Positional Encoding ───────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings (preferred over sinusoidal for short
    sequences; the model can learn task-specific position biases).
    """

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, seq_len, d_model)"""
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ── Full Transformer Classifier ───────────────────────────────────────────────

class StressTransformer(nn.Module):
    """
    Patch-based Transformer for binary stress classification.

    Parameters
    ----------
    in_channels : number of input channels (default 6: BVP+ACC+TEMP+EDA)
    patch_size  : samples per patch (default 64 = 1 s @ 64 Hz)
    d_model     : token / embedding dimension
    n_heads     : number of attention heads (must divide d_model)
    n_layers    : number of TransformerEncoder layers
    d_ff        : feedforward hidden dimension
    dropout     : applied in attention, FFN, and classification head
    num_classes : 2 (stress / non-stress)
    max_patches : maximum sequence length incl. CLS token (>= n_patches + 1)
    """

    def __init__(
        self,
        in_channels: int = 6,
        patch_size:  int = 64,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 3,
        d_ff:        int = 128,
        dropout:     float = 0.3,
        num_classes: int = 2,
        max_patches: int = 128,   # 60 patches + CLS + headroom
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model)

        # CLS token — learnable, broadcast across batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # +1 for CLS token
        self.pos_enc = LearnablePositionalEncoding(max_patches + 1, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,    # (B, seq, d_model) convention
            norm_first=True,     # Pre-LN: more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.size(0)

        # Patch tokens
        tokens = self.patch_embed(x)                         # (B, n_p, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, 1+n_p, d_model)

        # Positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer encoder
        encoded = self.encoder(tokens)                       # (B, 1+n_p, d_model)

        # Classify from CLS token
        cls_out = encoded[:, 0, :]                           # (B, d_model)
        return self.classifier(cls_out)                      # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (no grad)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


# ── Training (reuses CNN training loop) ──────────────────────────────────────

def train_model(
    model:         StressTransformer,
    train_dataset: TensorDataset,
    val_dataset:   Optional[TensorDataset] = None,
    n_epochs:      int = 40,
    batch_size:    int = 32,
    lr:            float = 5e-4,
    weight_decay:  float = 1e-4,
    device:        Optional[torch.device] = None,
    verbose:       bool = True,
) -> dict[str, list]:
    """
    Training loop for StressTransformer.

    Identical structure to cnn_classifier.train_model — class-weighted
    cross-entropy, AdamW, cosine LR, early stopping on val F1.

    Transformers typically need more epochs than CNNs to converge
    (default 40 vs 30 for CNN) due to the attention warm-up phase.
    """
    from sklearn.metrics import f1_score, accuracy_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class-weighted loss
    labels = np.array([y.item() for _, y in train_dataset])
    class_counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False) if val_dataset else None

    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}

    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    patience = 12   # slightly more patience than CNN — transformers are slower to converge

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            # Gradient clipping — important for transformers
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        history["train_loss"].append(avg_loss)

        # ── Validate ───────────────────────────────────────────────────
        if val_loader:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    preds = model(X.to(device)).argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
            history["val_accuracy"].append(acc)
            history["val_f1"].append(f1)

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                      f"val_acc={acc:.3f} | val_f1={f1:.3f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best F1={best_f1:.3f})")
                break

        elif verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ── Model summary ─────────────────────────────────────────────────────────────

def model_summary(model: StressTransformer, in_channels: int = 6, T: int = 3840) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStressTransformer — {total:,} total params ({trainable:,} trainable)")
    print(f"Input shape : (batch, {in_channels}, {T})")

    x = torch.zeros(1, in_channels, T)
    with torch.no_grad():
        tokens = model.patch_embed(x)
        n_patches = tokens.shape[1]
    print(f"Patches     : {n_patches} × {model.patch_embed.patch_size} samples "
          f"(patch_size={model.patch_embed.patch_size})")
    print(f"d_model     : {model.cls_token.shape[-1]}")
    print()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    W, C, T = 100, 6, 3840    # 100 windows, 6 channels, 60 s @ 64 Hz
    X = torch.randn(W, C, T)
    y = torch.randint(0, 2, (W,))

    model = StressTransformer()
    model_summary(model)

    ds = TensorDataset(X, y)
    history = train_model(model, ds, val_dataset=ds, n_epochs=10, batch_size=16, verbose=True)
    print(f"\nFinal val F1: {history['val_f1'][-1]:.3f}")
