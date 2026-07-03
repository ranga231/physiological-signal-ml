"""
Channel Ablation Study
======================
Systematically tests which input channels drive CNN stress detection
performance. Runs LOSO-CV for 5 channel configurations:

  1. BVP only          (1 ch)
  2. BVP + ACC         (4 ch)  ← prior baseline
  3. BVP + ACC + TEMP  (5 ch)
  4. BVP + ACC + EDA   (5 ch)
  5. BVP + ACC + TEMP + EDA (6 ch)  ← full model

Usage
-----
    python run_ablation.py
    python run_ablation.py --epochs 30 --subjects 2 3 4 5 6  # quick test

Output
------
    ablation_results.csv  — per-subject F1 for each config
    ablation_summary.csv  — mean ± std F1 across subjects per config
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ── Channel configurations ────────────────────────────────────────────────────

CONFIGS = [
    {
        "name":       "BVP only",
        "in_channels": 1,
        "use_temp":   False,
        "use_eda":    False,
        "bvp_only":   True,
    },
    {
        "name":       "BVP + ACC",
        "in_channels": 4,
        "use_temp":   False,
        "use_eda":    False,
        "bvp_only":   False,
    },
    {
        "name":       "BVP + ACC + TEMP",
        "in_channels": 5,
        "use_temp":   True,
        "use_eda":    False,
        "bvp_only":   False,
    },
    {
        "name":       "BVP + ACC + EDA",
        "in_channels": 5,
        "use_temp":   False,
        "use_eda":    True,
        "bvp_only":   False,
    },
    {
        "name":       "BVP + ACC + TEMP + EDA",
        "in_channels": 6,
        "use_temp":   True,
        "use_eda":    True,
        "bvp_only":   False,
    },
]


def make_prepare_fn(use_temp: bool, use_eda: bool, bvp_only: bool):
    """
    Return a prepare_cnn_input wrapper that passes only the specified channels.
    BVP-only mode passes a dummy single-axis ACC to satisfy the function signature.
    """
    from src.models.cnn_classifier import prepare_cnn_input
    import numpy as np

    def prepare_fn(windows_bvp, windows_acc, labels,
                   windows_temp=None, windows_eda=None):

        if bvp_only:
            # Pass minimal ACC (zeros) but only BVP channel will be used
            # We override by building the tensor directly
            import torch
            from torch.utils.data import TensorDataset
            W, N = windows_bvp.shape
            X = torch.from_numpy(windows_bvp[:, np.newaxis, :])   # (W, 1, N)
            y = torch.from_numpy(labels.astype(np.int64))
            return TensorDataset(X, y)

        return prepare_cnn_input(
            windows_bvp  = windows_bvp,
            windows_acc  = windows_acc,
            labels       = labels,
            windows_temp = windows_temp if use_temp else None,
            windows_eda  = windows_eda  if use_eda  else None,
        )

    return prepare_fn


def run(args):
    print("\n" + "=" * 62)
    print("  WESAD Channel Ablation Study")
    print("=" * 62)

    # ── Load and window data ───────────────────────────────────────────
    from src.ingestion.wesad_loader import WESADDataset
    from src.preprocessing.signal_processing import WindowConfig, segment_subject
    from src.evaluation.loso_cv import run_loso_cnn, loso_summary
    from src.models.cnn_classifier import StressCNN, train_model

    print(f"\nLoading subjects from: {args.data_root}")
    ds = WESADDataset(args.data_root, subject_ids=args.subjects)
    subjects = ds.load_all(verbose=False)

    cfg = WindowConfig(window_s=60.0, step_s=30.0)
    all_windows = []
    for s in subjects:
        sw = segment_subject(s, cfg)
        all_windows.append(sw)
    print(f"  {len(all_windows)} subjects windowed\n")

    # ── Run each config ────────────────────────────────────────────────
    all_results = {}   # config_name → list of LOSOResult

    for config in CONFIGS:
        print(f"\n{'─'*62}")
        print(f"  Config: {config['name']}  ({config['in_channels']} channels)")
        print(f"{'─'*62}")

        prepare_fn = make_prepare_fn(
            use_temp = config["use_temp"],
            use_eda  = config["use_eda"],
            bvp_only = config["bvp_only"],
        )

        in_ch = config["in_channels"]

        # Monkey-patch train_model in loso_cv to use CNN's train_model
        import src.models.cnn_classifier as _cm
        import src.evaluation.loso_cv as _loso
        # loso_cv imports train_model from cnn_classifier at call time, so no patch needed

        results = run_loso_cnn(
            all_windows       = all_windows,
            model_fn          = lambda ch=in_ch: StressCNN(in_channels=ch),
            prepare_dataset_fn= prepare_fn,
            train_kwargs      = {"n_epochs": args.epochs, "batch_size": 32},
            remove_artifacts  = True,
            verbose           = True,
        )

        all_results[config["name"]] = results

        f1s = [r.f1 for r in results]
        print(f"\n  → Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # ── Build results tables ───────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  ABLATION SUMMARY")
    print(f"{'='*62}\n")

    sids = [sw.subject_id for sw in all_windows]

    # Per-subject F1 table
    rows = {}
    for config in CONFIGS:
        name = config["name"]
        results = all_results[name]
        rows[name] = {r.subject_id: r.f1 for r in results}

    df_per_subject = pd.DataFrame(rows, index=sids)
    df_per_subject.index.name = "subject_id"
    print("Per-subject F1:")
    print(df_per_subject.to_string(float_format="%.4f"))

    # Summary table
    summary_rows = []
    for config in CONFIGS:
        name = config["name"]
        results = all_results[name]
        f1s   = [r.f1      for r in results]
        aucs  = [r.auc_roc for r in results if r.auc_roc is not None]
        accs  = [r.accuracy for r in results]
        summary_rows.append({
            "config":      name,
            "in_channels": config["in_channels"],
            "mean_f1":     np.mean(f1s),
            "std_f1":      np.std(f1s),
            "mean_auc":    np.mean(aucs) if aucs else np.nan,
            "mean_acc":    np.mean(accs),
        })

    df_summary = pd.DataFrame(summary_rows).set_index("config")

    print(f"\n{'─'*62}")
    print("Summary (mean ± std across subjects):")
    print(f"{'─'*62}")
    for _, row in df_summary.iterrows():
        print(f"  {row.name:<30}  F1={row['mean_f1']:.4f} ± {row['std_f1']:.4f}  "
              f"AUC={row['mean_auc']:.4f}  Acc={row['mean_acc']:.4f}")

    # Save
    df_per_subject.to_csv("ablation_per_subject.csv")
    df_summary.to_csv("ablation_summary.csv")
    print(f"\n  Saved: ablation_per_subject.csv, ablation_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="WESAD channel ablation study")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subset of subjects for quick test (e.g. --subjects 2 3 4 5 6)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
