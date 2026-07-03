"""
RF Feature Group Ablation Study
================================
Tests which handcrafted feature groups drive Random Forest
stress detection performance.

Feature groups
--------------
  hrv      : hr_mean, hr_std, sdnn, rmssd, pnn50,
              sd1, sd2, sd1_sd2,
              lf_power, hf_power, lf_hf, total_power,
              ppg_amplitude, ppg_rise_time_ms, ppg_pulse_width_ms
  acc      : acc_mag_mean, acc_mag_std, acc_sma,
              acc_x/y/z mean+std (9 features)
  eda      : eda_mean, eda_std, eda_min, eda_max, eda_slope
  temp     : temp_mean, temp_std, temp_min, temp_max, temp_slope

Configs tested
--------------
  1.  HRV only
  2.  ACC only
  3.  EDA only
  4.  TEMP only
  5.  HRV + ACC
  6.  HRV + EDA
  7.  HRV + TEMP
  8.  EDA + TEMP
  9.  HRV + EDA + TEMP
  10. All features  (HRV + ACC + EDA + TEMP)

Usage
-----
    python run_ablation_rf.py
    python run_ablation_rf.py --subjects 2 3 4 5 6   # quick test
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ── Feature group index maps ───────────────────────────────────────────────────

def get_feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """
    Map group names → list of column indices in the feature matrix.
    """
    hrv_prefixes  = ("hr_", "sdnn", "rmssd", "pnn50",
                     "sd1", "sd2", "lf_", "hf_", "total_", "ppg_")
    acc_prefix    = "acc_"
    eda_prefix    = "eda_"
    temp_prefix   = "temp_"

    groups = {"hrv": [], "acc": [], "eda": [], "temp": []}

    for i, name in enumerate(feature_names):
        if any(name.startswith(p) or name == p.rstrip("_")
               for p in hrv_prefixes):
            groups["hrv"].append(i)
        elif name.startswith(acc_prefix):
            groups["acc"].append(i)
        elif name.startswith(eda_prefix):
            groups["eda"].append(i)
        elif name.startswith(temp_prefix):
            groups["temp"].append(i)

    return groups


CONFIGS = [
    {"name": "HRV only",           "groups": ["hrv"]},
    {"name": "ACC only",           "groups": ["acc"]},
    {"name": "EDA only",           "groups": ["eda"]},
    {"name": "TEMP only",          "groups": ["temp"]},
    {"name": "HRV + ACC",          "groups": ["hrv", "acc"]},
    {"name": "HRV + EDA",          "groups": ["hrv", "eda"]},
    {"name": "HRV + TEMP",         "groups": ["hrv", "temp"]},
    {"name": "EDA + TEMP",         "groups": ["eda", "temp"]},
    {"name": "HRV + EDA + TEMP",   "groups": ["hrv", "eda", "temp"]},
    {"name": "All features",       "groups": ["hrv", "acc", "eda", "temp"]},
]


def run(args):
    print("\n" + "=" * 62)
    print("  WESAD RF Feature Group Ablation")
    print("=" * 62)

    from src.ingestion.wesad_loader import WESADDataset
    from src.preprocessing.signal_processing import WindowConfig, segment_subject
    from src.features.hrv_features import extract_subject_features, FEATURE_NAMES
    from src.models.baseline_rf import build_rf_pipeline
    from src.evaluation.loso_cv import run_loso_rf, loso_summary

    # ── Load and window ────────────────────────────────────────────────
    print(f"\nLoading subjects from: {args.data_root}")
    ds = WESADDataset(args.data_root, subject_ids=args.subjects)
    subjects = ds.load_all(verbose=False)

    cfg = WindowConfig(window_s=60.0, step_s=30.0)
    all_windows = [segment_subject(s, cfg) for s in subjects]
    print(f"  {len(all_windows)} subjects windowed")

    # ── Pre-extract ALL features once ─────────────────────────────────
    print("\nExtracting features for all subjects (done once, reused across configs)...")
    X_all = []
    y_all = []
    art_all = []

    for sw in tqdm(all_windows):
        X = extract_subject_features(sw)
        X_all.append(X)
        y_all.append(sw.labels)
        art_all.append(sw.artifact_mask)

    # ── Feature group indices ──────────────────────────────────────────
    groups = get_feature_groups(FEATURE_NAMES)
    print(f"\nFeature groups:")
    for g, idxs in groups.items():
        names = [FEATURE_NAMES[i] for i in idxs]
        print(f"  {g:<6} ({len(idxs):2d} features): {', '.join(names[:4])}{'...' if len(names)>4 else ''}")

    # ── Run each config ────────────────────────────────────────────────
    all_results = {}
    sids = [sw.subject_id for sw in all_windows]

    for config in CONFIGS:
        print(f"\n{'─'*62}")
        print(f"  Config: {config['name']}")
        print(f"{'─'*62}")

        # Build column selector for this config
        col_idx = []
        for g in config["groups"]:
            col_idx.extend(groups[g])
        col_idx = sorted(col_idx)

        # Slice feature matrices to selected columns
        X_sub = [X[:, col_idx] for X in X_all]

        # LOSO-CV manually (reuse logic from run_loso_rf but with pre-sliced features)
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
        from src.evaluation.loso_cv import LOSOResult

        results = []
        for i, (sid, X_test, y_test, art_test) in enumerate(
                zip(sids, X_sub, y_all, art_all)):

            # Apply artifact mask and valid labels to test subject
            keep_test = (~art_test) & (y_test >= 0)
            X_te = X_test[keep_test]
            y_te = y_test[keep_test]

            # Train on all other subjects
            X_tr_list, y_tr_list = [], []
            for j, (X_j, y_j, art_j) in enumerate(zip(X_sub, y_all, art_all)):
                if j == i:
                    continue
                keep = (~art_j) & (y_j >= 0)
                X_tr_list.append(X_j[keep])
                y_tr_list.append(y_j[keep])

            X_tr = np.concatenate(X_tr_list)
            y_tr = np.concatenate(y_tr_list)

            # Impute NaNs (some HRV features NaN for bad windows)
            col_means = np.nanmean(X_tr, axis=0)
            for col in range(X_tr.shape[1]):
                X_tr[np.isnan(X_tr[:, col]), col] = col_means[col]
                X_te[np.isnan(X_te[:, col]), col] = col_means[col]

            pipe = build_rf_pipeline()
            pipe.fit(X_tr, y_tr)

            y_pred  = pipe.predict(X_te)
            y_proba = pipe.predict_proba(X_te)

            result = LOSOResult(sid, y_te, y_pred, y_proba)
            results.append(result)

            f1  = result.f1
            auc = result.auc_roc or float("nan")
            print(f"  S{sid:2d} → f1={f1:.3f}  auc={auc:.3f}")

        all_results[config["name"]] = results
        f1s = [r.f1 for r in results]
        print(f"\n  → Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  RF ABLATION SUMMARY")
    print(f"{'='*62}\n")

    summary_rows = []
    for config in CONFIGS:
        name    = config["name"]
        results = all_results[name]
        f1s     = [r.f1 for r in results]
        aucs    = [r.auc_roc for r in results if r.auc_roc is not None]
        accs    = [r.accuracy for r in results]
        n_feats = sum(len(groups[g]) for g in config["groups"])
        summary_rows.append({
            "config":    name,
            "n_features": n_feats,
            "mean_f1":   np.mean(f1s),
            "std_f1":    np.std(f1s),
            "mean_auc":  np.mean(aucs) if aucs else np.nan,
            "mean_acc":  np.mean(accs),
        })

    df_summary = pd.DataFrame(summary_rows).set_index("config")

    print(f"{'Config':<25} {'N feats':>8} {'Mean F1':>10} {'Std F1':>8} {'AUC':>8}")
    print("─" * 62)
    for _, row in df_summary.iterrows():
        print(f"  {row.name:<23} {int(row['n_features']):>8} "
              f"{row['mean_f1']:>10.4f} {row['std_f1']:>8.4f} "
              f"{row['mean_auc']:>8.4f}")

    # Per-subject F1 table
    per_subj = {}
    for config in CONFIGS:
        name = config["name"]
        per_subj[name] = {r.subject_id: r.f1 for r in all_results[name]}

    df_per = pd.DataFrame(per_subj, index=sids)
    df_per.index.name = "subject_id"

    df_summary.to_csv("ablation_rf_summary.csv")
    df_per.to_csv("ablation_rf_per_subject.csv")
    print(f"\n  Saved: ablation_rf_summary.csv, ablation_rf_per_subject.csv")


def main():
    parser = argparse.ArgumentParser(description="RF feature group ablation")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subset of subjects (e.g. --subjects 2 3 4 5 6)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
