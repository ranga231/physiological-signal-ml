"""
Leave-One-Subject-Out Cross-Validation (LOSO-CV)
=================================================
LOSO-CV is the correct evaluation protocol for wearable physiological datasets:

  - Train on N-1 subjects, test on 1, repeat for all subjects.
  - Reports mean ± std across folds — accounts for inter-subject variability.
  - Prevents data leakage: a subject's windows appear in EITHER train OR test,
    never both.

This is the protocol used in the WESAD paper and is standard for 510k
analytical validation of SaMD classification algorithms.

Both the RF feature-based pipeline and the CNN end-to-end model are
evaluated here using the same LOSO split for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)


@dataclass
class LOSOResult:
    """Results container for one LOSO fold."""
    subject_id:   int
    y_true:       np.ndarray
    y_pred:       np.ndarray
    y_proba:      Optional[np.ndarray] = None

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def f1(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="binary", zero_division=0))

    @property
    def f1_macro(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    @property
    def auc_roc(self) -> Optional[float]:
        if self.y_proba is None or len(np.unique(self.y_true)) < 2:
            return None
        return float(roc_auc_score(self.y_true, self.y_proba[:, 1]))

    def to_dict(self) -> dict:
        d = {
            "subject_id": self.subject_id,
            "n_test":     len(self.y_true),
            "n_stress":   int((self.y_true == 1).sum()),
            "accuracy":   self.accuracy,
            "f1":         self.f1,
            "f1_macro":   self.f1_macro,
        }
        if self.auc_roc is not None:
            d["auc_roc"] = self.auc_roc
        return d


def loso_summary(results: list[LOSOResult]) -> pd.DataFrame:
    """
    Aggregate LOSO results into a summary DataFrame.
    Prints mean ± std for key metrics.
    """
    rows = [r.to_dict() for r in results]
    df   = pd.DataFrame(rows).set_index("subject_id")

    metric_cols = ["accuracy", "f1", "f1_macro"] + (["auc_roc"] if "auc_roc" in df.columns else [])

    print("\n── LOSO-CV Results ──────────────────────────────────────────────")
    print(df[metric_cols].to_string(float_format="%.4f"))
    print("─" * 60)
    for col in metric_cols:
        vals = df[col].dropna()
        print(f"  {col:<12} : {vals.mean():.4f} ± {vals.std():.4f}  "
              f"(min {vals.min():.4f}, max {vals.max():.4f})")
    print()

    return df


def run_loso_rf(
    all_windows: list,           # list of SubjectWindows
    feature_fn,                  # callable: SubjectWindows → (W, D) np.ndarray
    pipeline_fn,                 # callable: () → sklearn Pipeline
    feature_names: Optional[list[str]] = None,
    remove_artifacts: bool = True,
    verbose: bool = True,
) -> list[LOSOResult]:
    """
    Run LOSO-CV with a sklearn Pipeline on handcrafted features.

    Parameters
    ----------
    all_windows    : list of SubjectWindows (one per subject)
    feature_fn     : function(SubjectWindows) → (W, D) feature matrix
    pipeline_fn    : factory that returns a fresh sklearn Pipeline
    remove_artifacts: drop windows where artifact_mask is True
    """
    from tqdm import tqdm

    # Pre-compute features for all subjects
    if verbose:
        print("Extracting features for all subjects...")
    X_per_subject = []
    y_per_subject = []

    for sw in tqdm(all_windows, disable=not verbose):
        X = feature_fn(sw)
        y = sw.labels

        if remove_artifacts:
            keep = ~sw.artifact_mask
            X, y = X[keep], y[keep]

        # Drop windows with artifact or invalid labels
        valid = y >= 0
        X_per_subject.append(X[valid])
        y_per_subject.append(y[valid])

    results: list[LOSOResult] = []
    sids = [sw.subject_id for sw in all_windows]

    for i, (sid, X_test, y_test) in enumerate(zip(sids, X_per_subject, y_per_subject)):
        # Train on all other subjects
        X_train = np.concatenate([X_per_subject[j] for j in range(len(sids)) if j != i])
        y_train = np.concatenate([y_per_subject[j] for j in range(len(sids)) if j != i])

        pipe = pipeline_fn()
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

        result = LOSOResult(sid, y_test, y_pred, y_proba)
        results.append(result)

        if verbose:
            auc_str = f" AUC={result.auc_roc:.3f}" if result.auc_roc else ""
            print(f"  S{sid:2d} → acc={result.accuracy:.3f}  f1={result.f1:.3f}{auc_str}")

    return results


def run_loso_cnn(
    all_windows: list,           # list of SubjectWindows
    model_fn,                    # callable: () → StressCNN
    prepare_dataset_fn,          # callable: (SubjectWindows) → TensorDataset
    train_kwargs: Optional[dict] = None,
    remove_artifacts: bool = True,
    verbose: bool = True,
) -> list[LOSOResult]:
    """
    Run LOSO-CV with the StressCNN.

    A fresh model is trained from scratch for each fold to avoid data leakage.
    """
    import torch
    from torch.utils.data import ConcatDataset

    train_kwargs = train_kwargs or {"n_epochs": 30, "batch_size": 32}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-build TensorDatasets for all subjects
    if verbose:
        print("Building CNN datasets for all subjects...")

    datasets = []
    for sw in all_windows:
        if remove_artifacts:
            from dataclasses import replace as dc_replace
            keep = ~sw.artifact_mask
            sw_clean = type(sw)(
                subject_id       = sw.subject_id,
                windows_bvp      = sw.windows_bvp[keep],
                windows_bvp_filt = sw.windows_bvp_filt[keep],
                windows_acc      = sw.windows_acc[keep],
                windows_temp     = sw.windows_temp[keep],
                windows_eda      = sw.windows_eda[keep],
                labels           = sw.labels[keep],
                artifact_mask    = sw.artifact_mask[keep],
            )
        else:
            sw_clean = sw

        valid = sw_clean.labels >= 0
        datasets.append(prepare_dataset_fn(
            sw_clean.windows_bvp[valid],
            sw_clean.windows_acc[valid],
            sw_clean.labels[valid],
            windows_temp=sw_clean.windows_temp[valid],
            windows_eda=sw_clean.windows_eda[valid],
        ))

    results: list[LOSOResult] = []
    sids = [sw.subject_id for sw in all_windows]

    for i, (sid, test_ds) in enumerate(zip(sids, datasets)):
        train_ds = ConcatDataset([datasets[j] for j in range(len(sids)) if j != i])

        model = model_fn().to(device)
        if verbose:
            print(f"\n── S{sid} (fold {i+1}/{len(sids)}) ──────────")

        from ..models.cnn_classifier import train_model, evaluate
        from torch.utils.data import DataLoader

        train_model(model, train_ds, val_dataset=test_ds, device=device,
                    verbose=verbose, **train_kwargs)

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        all_preds, all_labels, all_proba = [], [], []

        model.eval()
        import torch.nn.functional as F

        with torch.no_grad():
            for X, y in test_loader:
                logits = model(X.to(device))
                proba  = F.softmax(logits, dim=-1).cpu().numpy()
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                all_proba.extend(proba)

        y_true  = np.array(all_labels)
        y_pred  = np.array(all_preds)
        y_proba = np.array(all_proba)

        result = LOSOResult(sid, y_true, y_pred, y_proba)
        results.append(result)
        if verbose:
            print(f"  → acc={result.accuracy:.3f}  f1={result.f1:.3f}  AUC={result.auc_roc:.3f}")

    return results
