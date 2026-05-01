"""
src/data_loader.py
──────────────────
Step 1 of the pipeline: Load and validate the raw dataset.

Responsibilities:
  - Load raw Excel file
  - Report shape, dtypes, class balance
  - Drop completely empty rows
  - Remove exact duplicate rows
  - Drop columns exceeding missing-value threshold
  - Verify target column exists and is binary
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from config import (
    RAW_DATA_PATH, EXPANDED_DATA_PATH,
    TARGET, MISSING_DROP_THRESHOLD, DROP_COLUMNS
)


# ──────────────────────────────────────────────────────────────────────────────
def load_data(use_expanded: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load the dataset from Excel file.

    Args:
        use_expanded : If True loads the 10,000-row expanded dataset.

        verbose      : Print loading summary.

    Returns:
        Raw pandas DataFrame.
    """
    path = EXPANDED_DATA_PATH if use_expanded else RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_excel(path)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  [DataLoader] Dataset loaded")
        print(f"{'='*55}")
        print(f"  File   : {os.path.basename(path)}")
        print(f"  Rows   : {len(df):,}")
        print(f"  Cols   : {df.shape[1]}")
        print(f"  Memory : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


# ──────────────────────────────────────────────────────────────────────────────
def validate_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Validate and clean the raw dataset.

    Steps:
      1. Verify target column exists and is binary (0/1)
      2. Report class distribution
      3. Drop columns listed in DROP_COLUMNS config
      4. Remove all-NaN rows
      5. Remove exact duplicate rows
      6. Drop columns with missing rate > MISSING_DROP_THRESHOLD
      7. Report final missing value summary

    Args:
        df      : Raw DataFrame from load_data().
        verbose : Print validation report.

    Returns:
        Validated and cleaned DataFrame.
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"  [DataLoader] Validation report")
        print(f"{'='*55}")

    # ── 1. Target column check ────────────────────────────────────────────────
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    unique_vals = set(df[TARGET].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET}' must be binary (0/1). "
            f"Found: {unique_vals}"
        )

    # ── 2. Class distribution ─────────────────────────────────────────────────
    counts = df[TARGET].value_counts().sort_index()
    total  = len(df)
    if verbose:
        print(f"\n  Class distribution:")
        print(f"    Real (0) : {counts.get(0, 0):>6,}  ({counts.get(0,0)/total*100:.1f}%)")
        print(f"    Fake (1) : {counts.get(1, 0):>6,}  ({counts.get(1,0)/total*100:.1f}%)")
        print(f"    Total    : {total:>6,}")

    # ── 3. Drop identifier / non-predictive columns ───────────────────────────
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        if verbose:
            print(f"\n  Dropped identifier columns: {to_drop}")

    # ── 4. Remove all-NaN rows ────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(how="all")
    n_empty  = n_before - len(df)
    if verbose and n_empty > 0:
        print(f"\n  Removed {n_empty} completely empty rows.")

    # ── 5. Remove duplicates ──────────────────────────────────────────────────
    n_before  = len(df)
    df        = df.drop_duplicates()
    n_dupes   = n_before - len(df)
    if verbose:
        print(f"\n  Duplicate rows removed : {n_dupes:,}")
        print(f"  Rows after dedup       : {len(df):,}")

    # ── 6. Drop high-missing columns ──────────────────────────────────────────
    missing_rate = df.isnull().mean()
    high_miss    = missing_rate[missing_rate > MISSING_DROP_THRESHOLD].index.tolist()
    if high_miss:
        df = df.drop(columns=high_miss)
        if verbose:
            print(f"\n  Columns dropped (>{MISSING_DROP_THRESHOLD*100:.0f}% missing):")
            for c in high_miss:
                print(f"    {c}: {missing_rate[c]*100:.1f}%")

    # ── 7. Missing value summary ──────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if verbose:
        print(f"\n  Missing value summary (top 10):")
        for col, n in missing.head(10).items():
            print(f"    {col:<35} {n:>6,} ({n/len(df)*100:.1f}%)")
        print(f"\n  Total missing cells : {df.isnull().sum().sum():,}")
        print(f"  Final shape         : {df.shape}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
def get_feature_target_split(df: pd.DataFrame,
                              feature_cols: list = None):
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df           : Validated DataFrame.
        feature_cols : Explicit list of feature column names.
                       If None, uses all columns except TARGET.

    Returns:
        X (DataFrame), y (Series)
    """
    from config import ALL_FEATURES
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f"  [DataLoader] Warning: {len(missing)} features not in data: {missing[:5]}...")

    X = df[available].copy()
    y = df[TARGET].copy()

    print(f"\n  [DataLoader] Feature/target split complete.")
    print(f"    X shape : {X.shape}")
    print(f"    y shape : {y.shape}")
    print(f"    Features: {available}")

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(use_expanded=True)
    df = validate_data(df)
    X, y = get_feature_target_split(df)
    print(f"\nReady for preprocessing.")
