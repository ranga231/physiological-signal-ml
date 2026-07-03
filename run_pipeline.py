"""
End-to-End Pipeline Runner
===========================
Runs the full stress detection pipeline:
  1. Load WESAD subjects
  2. Preprocess and window signals
  3. Extract HRV + multimodal features (RF baseline)
  4. Run LOSO-CV — RandomForest and/or 1D CNN
  5. Print results

Usage
-----
    python run_pipeline.py --data_root data/raw --model rf
    python run_pipeline.py --data_root data/raw --model cnn
    python run_pipeline.py --data_root data/raw --model both

Data download
-------------
    1. Request access: https://uni-siegen.de/labs/sigproc/redmine/projects/wesad
    2. Download WESAD.zip (~500 MB)
    3. Extract so that data/raw/S2/S2.pkl etc. exist
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).parent))


def run(args):
    print("\n" + "=" * 62)
    print("  WESAD Physiological Stress Detection Pipeline")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────
    from src.ingestion.wesad_loader import WESADDataset
    from src.preprocessing.signal_processing import WindowConfig, segment_subject
    from src.features.hrv_features import extract_subject_features, FEATURE_NAMES
    from src.evaluation.loso_cv import run_loso_rf, run_loso_cnn, loso_summary

    print(f"\n[1/4] Loading WESAD subjects from: {args.data_root}")
    ds = WESADDataset(args.data_root, subject_ids=args.subjects)
    subjects = ds.load_all(verbose=True)

    if not subjects:
        print("No subjects loaded. Check data path.")
        sys.exit(1)

    # ── Preprocessing ─────────────────────────────────────────────────
    print(f"\n[2/4] Windowing (window={args.window_s}s, step={args.step_s}s)")
    cfg = WindowConfig(window_s=args.window_s, step_s=args.step_s)

    t0 = time.time()
    all_windows = []
    for s in subjects:
        sw = segment_subject(s, cfg)
        n_stress = (sw.labels == 1).sum()
        n_rest   = (sw.labels == 0).sum()
        n_art    = sw.artifact_mask.sum()
        print(f"  S{s.sid:2d}: {sw.windows_bvp.shape[0]} windows  "
              f"(stress={n_stress}, non-stress={n_rest}, artifact={n_art})")
        all_windows.append(sw)
    print(f"  Windowing time: {time.time() - t0:.1f}s")

    # ── Run selected model(s) ─────────────────────────────────────────
    if args.model in ("rf", "both"):
        print("\n[3/4] RandomForest LOSO-CV")
        from src.models.baseline_rf import build_rf_pipeline, feature_importance_report

        results_rf = run_loso_rf(
            all_windows  = all_windows,
            feature_fn   = extract_subject_features,
            pipeline_fn  = build_rf_pipeline,
            feature_names= FEATURE_NAMES,
            remove_artifacts= True,
            verbose      = True,
        )
        print("\n── RandomForest Results ─────────────────────────────────────")
        df_rf = loso_summary(results_rf)
        df_rf.to_csv("results_rf.csv")
        print("  Saved: results_rf.csv")

        # Show feature importances from the last fold's model as a sample
        # (for a real report, average importances across all folds)

    if args.model in ("cnn", "both", "all"):
        print("\n[3/4] 1D CNN LOSO-CV")
        import torch
        from src.models.cnn_classifier import StressCNN, prepare_cnn_input

        results_cnn = run_loso_cnn(
            all_windows       = all_windows,
            model_fn          = lambda: StressCNN(in_channels=6),
            prepare_dataset_fn= prepare_cnn_input,
            train_kwargs      = {"n_epochs": args.epochs, "batch_size": args.batch_size},
            remove_artifacts  = True,
            verbose           = True,
        )
        print("\n── CNN Results ──────────────────────────────────────────────")
        df_cnn = loso_summary(results_cnn)
        df_cnn.to_csv("results_cnn.csv")
        print("  Saved: results_cnn.csv")

    if args.model in ("transformer", "all"):
        print("\n[3/4] Transformer LOSO-CV")
        import torch
        from src.models.transformer_classifier import StressTransformer, train_model as transformer_train
        from src.models.cnn_classifier import prepare_cnn_input
        from src.evaluation.loso_cv import run_loso_cnn  # same LOSO loop works for transformer

        # Monkey-patch train_model so run_loso_cnn uses the transformer's training loop
        import src.models.transformer_classifier as _tm
        import src.models.cnn_classifier as _cm
        _orig_train = _cm.train_model
        _cm.train_model = _tm.train_model

        results_transformer = run_loso_cnn(
            all_windows       = all_windows,
            model_fn          = lambda: StressTransformer(in_channels=6),
            prepare_dataset_fn= prepare_cnn_input,
            train_kwargs      = {"n_epochs": args.epochs + 10, "batch_size": args.batch_size},
            remove_artifacts  = True,
            verbose           = True,
        )

        _cm.train_model = _orig_train   # restore

        print("\n── Transformer Results ───────────────────────────────────────")
        df_tr = loso_summary(results_transformer)
        df_tr.to_csv("results_transformer.csv")
        print("  Saved: results_transformer.csv")

    print("\n[4/4] Done.\n")


def main():
    parser = argparse.ArgumentParser(description="WESAD stress detection pipeline")
    parser.add_argument("--data_root",  default="data/raw",   help="Path to WESAD raw data")
    parser.add_argument("--model",      default="rf",         choices=["rf", "cnn", "transformer", "both", "all"])
    parser.add_argument("--window_s",   type=float, default=60.0,  help="Window length (seconds)")
    parser.add_argument("--step_s",     type=float, default=30.0,  help="Step size (seconds)")
    parser.add_argument("--epochs",     type=int,   default=30,    help="CNN training epochs per fold")
    parser.add_argument("--batch_size", type=int,   default=32,    help="CNN batch size")
    parser.add_argument(
        "--subjects", type=int, nargs="+",
        default=None,
        help="Subset of subject IDs to load (e.g. --subjects 2 3 4). Default: all."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
