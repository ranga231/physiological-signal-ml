"""
RandomForest Baseline on HRV + Multimodal Features
====================================================
Provides a strong, interpretable baseline to benchmark the 1D CNN against.

The RF baseline is important in regulated health AI because:
  - Explainability: feature importances map to clinical concepts (HRV, EDA, TEMP)
  - Data efficiency: performs well with small N (14 subjects, ~thousands of windows)
  - Predicate alignment: mirrors feature-based approaches used in cleared SaMD

Performance target: ~80–85% F1 on LOSO-CV (consistent with WESAD literature).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from typing import Optional


def build_rf_pipeline(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """
    Scikit-learn Pipeline: impute NaN → scale → RandomForest.

    Imputation handles windows where peak detection failed (NaN HRV features).
    Scaling is included for compatibility; RF itself doesn't need it, but the
    same pipeline can swap in SVM without changing downstream code.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            class_weight  = class_weight,
            n_jobs        = -1,
            random_state  = random_state,
        )),
    ])


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute accuracy, F1, and optionally AUC-ROC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    return metrics


def feature_importance_report(
    pipeline: Pipeline,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """Print top-N features by RF importance."""
    rf = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    print(f"\n── Top {top_n} Feature Importances ─────────────────────────────")
    for rank, i in enumerate(idx, 1):
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {rank:2d}. {name:<30} {importances[i]:.4f}")
