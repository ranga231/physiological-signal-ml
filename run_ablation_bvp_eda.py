"""
Single ablation config: BVP + EDA only (2 channels)
=====================================================
Runs LOSO-CV for BVP + EDA only to test whether ACC
is genuinely unnecessary for stress detection.

Usage:
    python run_ablation_bvp_eda.py
"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import TensorDataset
from scipy.interpolate import interp1d

from src.ingestion.wesad_loader import WESADDataset
from src.preprocessing.signal_processing import WindowConfig, segment_subject
from src.models.cnn_classifier import StressCNN
from src.evaluation.loso_cv import run_loso_cnn, loso_summary


def prepare_bvp_eda(windows_bvp, windows_acc, labels,
                    windows_temp=None, windows_eda=None):
    """BVP + EDA only — 2 channels, no ACC, no TEMP."""
    W, N_bvp = windows_bvp.shape
    N_target  = N_bvp

    # Upsample EDA: 4 Hz → 64 Hz
    t_orig = np.linspace(0, 1, windows_eda.shape[1])
    t_new  = np.linspace(0, 1, N_target)
    eda_up = np.zeros((W, N_target), dtype=np.float32)
    for i in range(W):
        eda_up[i] = interp1d(t_orig, windows_eda[i], kind="linear")(t_new)

    bvp_ch = windows_bvp[:, np.newaxis, :]   # (W, 1, N)
    eda_ch = eda_up[:, np.newaxis, :]         # (W, 1, N)
    X = np.concatenate([bvp_ch, eda_ch], axis=1)  # (W, 2, N)

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(labels.astype(np.int64))
    )


def main():
    print("\n── BVP + EDA Only Ablation (2 channels) ────────────────────")

    ds = WESADDataset("data/raw")
    subjects = ds.load_all(verbose=False)

    cfg = WindowConfig(window_s=60.0, step_s=30.0)
    all_windows = [segment_subject(s, cfg) for s in subjects]
    print(f"  {len(all_windows)} subjects windowed\n")

    results = run_loso_cnn(
        all_windows        = all_windows,
        model_fn           = lambda: StressCNN(in_channels=2),
        prepare_dataset_fn = prepare_bvp_eda,
        train_kwargs       = {"n_epochs": 30, "batch_size": 32},
        remove_artifacts   = True,
        verbose            = True,
    )

    print("\n── BVP + EDA Results ────────────────────────────────────────")
    df = loso_summary(results)
    df.to_csv("ablation_bvp_eda.csv")
    print("  Saved: ablation_bvp_eda.csv")

    f1s = [r.f1 for r in results]
    print(f"\n  Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
