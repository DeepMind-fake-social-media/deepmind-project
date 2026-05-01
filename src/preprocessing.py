"""
src/preprocessing.py
─────────────────────
Step 4 of the pipeline: Preprocessing.

Builds a production-ready sklearn preprocessing pipeline:
  1. Outlier capping  (IQR-based, numeric features only)
  2. Imputation       (median for numeric, mode for binary, constant for cat)
  3. Encoding         (OneHotEncoder for categorical)
  4. Scaling          (StandardScaler for numeric)
  5. SMOTE            (optional oversampling for class imbalance)
  6. Train/test split (stratified)

All steps are fitted ONLY on training data to prevent data leakage.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection      import train_test_split
from sklearn.pipeline             import Pipeline
from sklearn.compose              import ColumnTransformer
from sklearn.preprocessing        import StandardScaler, OneHotEncoder
from sklearn.impute               import SimpleImputer

from config import (
    TARGET, TEST_SIZE, RANDOM_STATE, OUTPUT_DIR,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    OUTLIER_IQR_MULTIPLIER, SCALE_NUMERIC, ENCODE_CATEGORICAL,
)


# ── Outlier capping ────────────────────────────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame,
                     numeric_cols: list,
                     multiplier: float = OUTLIER_IQR_MULTIPLIER,
                     fitted_bounds: dict = None) -> tuple:
    """
    Cap outliers using IQR method: clip values to
    [Q1 - k*IQR, Q3 + k*IQR] per column.

    Args:
        df            : DataFrame with features.
        numeric_cols  : Columns to apply capping to.
        multiplier    : IQR multiplier k (default 3.0).
        fitted_bounds : Pre-computed bounds dict for test-time use.
                        If None, bounds are computed from df (train mode).

    Returns:
        (capped DataFrame, bounds dict)
    """
    df     = df.copy()
    bounds = fitted_bounds or {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col not in bounds:
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            bounds[col] = {
                "lower": q1 - multiplier * iqr,
                "upper": q3 + multiplier * iqr,
            }
        df[col] = df[col].clip(
            lower=bounds[col]["lower"],
            upper=bounds[col]["upper"],
        )

    return df, bounds


# ── Build ColumnTransformer ────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construct a sklearn ColumnTransformer with three sub-pipelines:

      Numeric  → median imputation → StandardScaler
      Binary   → mode imputation   → pass-through
      Categorical → constant imputation → OneHotEncoder

    Args:
        X : Feature DataFrame (determines which columns are present).

    Returns:
        Unfitted ColumnTransformer.
    """
    # Determine available columns per group
    # Include engineered features in numeric group automatically
    num_cols = [c for c in X.columns
                if X[c].dtype in [np.float64, np.int64, float, int]
                and c not in BINARY_FEATURES
                and c not in CATEGORICAL_FEATURES]
    bin_cols = [c for c in BINARY_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    print(f"\n  [Preprocessing] ColumnTransformer groups:")
    print(f"    Numeric     : {len(num_cols)} cols")
    print(f"    Binary      : {len(bin_cols)} cols  {bin_cols}")
    print(f"    Categorical : {len(cat_cols)} cols  {cat_cols}")

    transformers = []

    if num_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler() if SCALE_NUMERIC else "passthrough"),
        ])
        transformers.append(("numeric", numeric_pipe, num_cols))

    if bin_cols:
        binary_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("binary", binary_pipe, bin_cols))

    if cat_cols and ENCODE_CATEGORICAL:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


# ── Get feature names after transform ─────────────────────────────────────────
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names from a fitted ColumnTransformer."""
    names = []
    for name, pipe, cols in preprocessor.transformers_:
        if name == "categorical":
            enc       = pipe.named_steps["encoder"]
            cat_names = enc.get_feature_names_out(cols)
            names.extend(cat_names.tolist())
        elif isinstance(cols, list):
            names.extend(cols)
    return names


# ── Main preprocessing entry point ────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame,
                      feature_cols: list = None,
                      apply_smote: bool = False,
                      verbose: bool = True) -> dict:
    """
    Full preprocessing pipeline:
      1. Feature/target split
      2. Train/test split (stratified)
      3. Outlier capping on train set
      4. Build and fit ColumnTransformer
      5. Transform train and test sets
      6. Optionally apply SMOTE to training set
      7. Save preprocessor and metadata

    Args:
        df           : Fully feature-engineered DataFrame.
        feature_cols : Columns to use as features. Defaults to all non-target.
        apply_smote  : Whether to apply SMOTE to the training set.
        verbose      : Print progress.

    Returns:
        dict with keys:
          X_train, X_test         : np.ndarray processed features
          y_train, y_test         : pd.Series labels
          X_train_raw, X_test_raw : pd.DataFrame before sklearn transform
          feature_names           : list of feature names after encoding
          preprocessor            : fitted ColumnTransformer
          outlier_bounds          : dict of IQR bounds
          (if SMOTE) X_train_sm, y_train_sm : SMOTE-augmented training set
    """
    # ── 1. Feature/target split ───────────────────────────────────────────────
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != TARGET]

    available = [c for c in feature_cols if c in df.columns]
    X         = df[available].copy()
    y         = df[TARGET].copy()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  [Preprocessing] Starting full pipeline")
        print(f"{'='*55}")
        print(f"\n  Features : {len(available)}")
        print(f"  Samples  : {len(X):,}")

    # ── 2. Train/test split ───────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if verbose:
        print(f"\n  Train/Test split (stratified, {int(TEST_SIZE*100)}% test):")
        print(f"    Train : {len(X_train_raw):,} samples  "
              f"(Real={sum(y_train==0):,}  Fake={sum(y_train==1):,})")
        print(f"    Test  : {len(X_test_raw):,}  samples  "
              f"(Real={sum(y_test==0):,}   Fake={sum(y_test==1):,})")

    # ── 3. Outlier capping ────────────────────────────────────────────────────
    num_cols = [c for c in X_train_raw.columns
                if X_train_raw[c].dtype in [np.float64, np.int64, float, int]
                and c not in BINARY_FEATURES and c not in CATEGORICAL_FEATURES]

    X_train_raw, outlier_bounds = cap_outliers_iqr(X_train_raw, num_cols)
    X_test_raw,  _              = cap_outliers_iqr(X_test_raw, num_cols,
                                                    fitted_bounds=outlier_bounds)

    n_capped = sum(1 for b in outlier_bounds.values()
                   if b["lower"] is not None)
    if verbose:
        print(f"\n  Outlier capping (IQR ×{OUTLIER_IQR_MULTIPLIER}):")
        print(f"    Columns capped : {n_capped}")

    # ── 4. Build and fit preprocessor ─────────────────────────────────────────
    preprocessor = build_preprocessor(X_train_raw)
    X_train      = preprocessor.fit_transform(X_train_raw)
    X_test       = preprocessor.transform(X_test_raw)

    feat_names = get_feature_names(preprocessor)

    if verbose:
        print(f"\n  Preprocessing pipeline fitted:")
        print(f"    Processed train shape : {X_train.shape}")
        print(f"    Processed test shape  : {X_test.shape}")
        print(f"    Feature names         : {len(feat_names)}")
        print(f"    Missing after impute  : {np.isnan(X_train).sum()}")

    # ── 5. Optional SMOTE ─────────────────────────────────────────────────────
    result = {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "X_train_raw":  X_train_raw,
        "X_test_raw":   X_test_raw,
        "feature_names": feat_names,
        "preprocessor":  preprocessor,
        "outlier_bounds": outlier_bounds,
    }

    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE)
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
            if verbose:
                print(f"\n  SMOTE oversampling:")
                print(f"    Before : Real={sum(y_train==0):,}  Fake={sum(y_train==1):,}")
                print(f"    After  : Real={sum(y_train_sm==0):,}  Fake={sum(y_train_sm==1):,}")
            result["X_train_sm"] = X_train_sm
            result["y_train_sm"] = y_train_sm
        except ImportError:
            print("  SMOTE skipped — install imbalanced-learn: pip install imbalanced-learn")

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(preprocessor,   os.path.join(OUTPUT_DIR, "preprocessor.pkl"))
    joblib.dump(outlier_bounds, os.path.join(OUTPUT_DIR, "outlier_bounds.pkl"))
    joblib.dump(feat_names,     os.path.join(OUTPUT_DIR, "feature_names.pkl"))
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"),  X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),   X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"),  y_train.values)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),   y_test.values)

    if verbose:
        print(f"\n  Artefacts saved to: {OUTPUT_DIR}/")
        print(f"    preprocessor.pkl")
        print(f"    outlier_bounds.pkl")
        print(f"    feature_names.pkl")
        print(f"    X_train.npy  X_test.npy  y_train.npy  y_test.npy")
        print(f"\n{'='*55}")
        print(f"  [Preprocessing] Pipeline complete ✅")
        print(f"{'='*55}")

    return result


# ── Visualise preprocessing results ───────────────────────────────────────────
def plot_preprocessing_report(result: dict, save: bool = True):
    """
    Generate preprocessing summary plots:
      - Class balance before/after SMOTE
      - Feature value range after scaling
      - Missing value rate before/after imputation
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Class balance
    y_train = result["y_train"]
    counts  = y_train.value_counts().sort_index()
    colors  = ["#0077B6", "#EF233C"]
    axes[0].bar(["Real (Train)", "Fake (Train)"], counts.values,
                color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Training Set Class Balance (after split)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")

    # Feature value range after scaling (sample 30 features)
    X_train     = result["X_train"]
    feat_names  = result["feature_names"][:30]
    X_sample    = X_train[:, :len(feat_names)]
    feat_means  = np.nanmean(X_sample, axis=0)
    feat_stds   = np.nanstd(X_sample, axis=0)

    axes[1].errorbar(range(len(feat_names)), feat_means, yerr=feat_stds,
                     fmt="o", markersize=4, color="#0077B6", alpha=0.7,
                     ecolor="#AED6F1", elinewidth=1, capsize=2)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(range(len(feat_names)))
    axes[1].set_xticklabels(feat_names, rotation=90, fontsize=7)
    axes[1].set_title("Feature Means ± Std After Scaling (first 30)")
    axes[1].set_ylabel("Scaled Value")

    plt.suptitle("Preprocessing Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "preprocess_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader        import load_data, validate_data
    from src.feature_engineering import run_all_feature_engineering

    df     = load_data()
    df     = validate_data(df)
    df     = run_all_feature_engineering(df)
    result = run_preprocessing(df, apply_smote=True)
    plot_preprocessing_report(result)
    print(f"\nFinal train shape : {result['X_train'].shape}")
    print(f"Final test shape  : {result['X_test'].shape}")
