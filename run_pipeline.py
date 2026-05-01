"""
run_pipeline.py
───────────────
Master script: runs the full beginning-to-preprocessing pipeline.

Steps executed:
  1. Load & validate data
  2. Exploratory Data Analysis (EDA) — saves all plots
  3. Feature Engineering      — derives new features
  4. Preprocessing            — imputation, scaling, encoding, train/test split

Usage:
  python run_pipeline.py                # full pipeline, expanded dataset
  python run_pipeline.py --raw          # use original 3,000-row dataset
  python run_pipeline.py --no-eda       # skip EDA plots (faster)
  python run_pipeline.py --smote        # apply SMOTE oversampling
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  DeepMind – Detection of Fraudulent Social Media Accounts   ║
║  Pipeline: Data Loading → EDA → Feature Engineering         ║
║            → Preprocessing                                  ║
║  University of Moratuwa · Faculty of IT · 2026              ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="DeepMind Preprocessing Pipeline")
    parser.add_argument("--raw",    action="store_true", help="Use original 3K dataset")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--smote",  action="store_true", help="Apply SMOTE oversampling")
    args = parser.parse_args()

    print(BANNER)
    t_total = time.time()

    # ── Step 1: Load & Validate ───────────────────────────────────────────────
    print("\n" + "▶ STEP 1: Data Loading & Validation")
    print("─" * 55)
    t0 = time.time()
    from src.data_loader import load_data, validate_data

    df = load_data(use_expanded=not args.raw)
    df = validate_data(df)
    print(f"\n  ✅ Completed in {time.time()-t0:.1f}s")

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    if not args.no_eda:
        print("\n" + "▶ STEP 2: Exploratory Data Analysis")
        print("─" * 55)
        t0 = time.time()
        from src.eda import run_full_eda
        summary = run_full_eda(df)
        print(f"\n  Statistical summary (first 5 features):")
        print(summary.head().to_string(index=False))
        print(f"\n  ✅ Completed in {time.time()-t0:.1f}s")
    else:
        print("\n  [EDA] Skipped (--no-eda flag)")

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("\n" + "▶ STEP 3: Feature Engineering")
    print("─" * 55)
    t0 = time.time()
    from src.feature_engineering import (
        run_all_feature_engineering,
        plot_engineered_features,
    )
    df = run_all_feature_engineering(df)

    if not args.no_eda:
        plot_engineered_features(df)

    print(f"\n  ✅ Completed in {time.time()-t0:.1f}s")

    # ── Step 4: Preprocessing ─────────────────────────────────────────────────
    print("\n" + "▶ STEP 4: Preprocessing")
    print("─" * 55)
    t0 = time.time()
    from src.preprocessing import run_preprocessing, plot_preprocessing_report

    result = run_preprocessing(df, apply_smote=args.smote)

    if not args.no_eda:
        plot_preprocessing_report(result)

    print(f"\n  ✅ Completed in {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_total
    print("\n\n" + "═" * 55)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("═" * 55)
    print(f"  Dataset used        : {'Expanded (10K)' if not args.raw else 'Original (3K)'}")
    print(f"  SMOTE applied       : {'Yes' if args.smote else 'No'}")
    print(f"  Total time          : {total:.1f}s")
    print(f"\n  Output shapes:")
    print(f"    X_train : {result['X_train'].shape}")
    print(f"    X_test  : {result['X_test'].shape}")
    print(f"    y_train : {result['y_train'].shape}  "
          f"(Real={sum(result['y_train']==0):,}  "
          f"Fake={sum(result['y_train']==1):,})")
    print(f"    y_test  : {result['y_test'].shape}   "
          f"(Real={sum(result['y_test']==0):,}   "
          f"Fake={sum(result['y_test']==1):,})")
    print(f"\n  Features after engineering + encoding:")
    print(f"    {result['X_train'].shape[1]} total features")
    print(f"\n  All artefacts saved to: outputs/")
    print(f"    preprocessor.pkl")
    print(f"    X_train.npy  X_test.npy")
    print(f"    y_train.npy  y_test.npy")
    print(f"    EDA & feature engineering plots (.png)")
    print("═" * 55)
    print("\n  ✅ Ready for model training (Modules 1–4)")

    return result


if __name__ == "__main__":
    result = main()
