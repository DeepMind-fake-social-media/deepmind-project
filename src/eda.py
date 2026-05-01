"""
src/eda.py
──────────
Step 2 of the pipeline: Exploratory Data Analysis.

Generates and saves:
  - Class distribution (count + pie)
  - Platform breakdown by class
  - Missing value heatmap
  - Feature distributions (Real vs Fake)
  - Correlation matrix
  - Statistical summary table
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from config import TARGET, OUTPUT_DIR, NUMERIC_FEATURES, BINARY_FEATURES

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
})
PALETTE = {"real": "#0077B6", "fake": "#EF233C"}


# ──────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """Plot class distribution as bar chart and pie chart."""
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Real (0)", "Fake (1)"]
    colors = [PALETTE["real"], PALETTE["fake"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Bar
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class Distribution – Count")
    axes[0].set_ylabel("Number of Profiles")
    axes[0].set_ylim(0, counts.max() * 1.18)
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + counts.max()*0.02,
                     f"{v:,}", ha="center", fontweight="bold", fontsize=12)

    # Pie
    axes[1].pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Class Distribution – Percentage")

    plt.suptitle(f"Target Variable: {TARGET}  |  Total: {len(df):,} profiles",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_01_class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_platform_breakdown(df: pd.DataFrame, save: bool = True):
    """Plot fake vs real breakdown per platform."""
    if "platform" not in df.columns:
        print("  [EDA] 'platform' column not found — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Stacked bar
    plat = df.groupby(["platform", TARGET]).size().unstack(fill_value=0)
    plat.columns = ["Real (0)", "Fake (1)"]
    plat.plot(kind="bar", ax=axes[0], color=[PALETTE["real"], PALETTE["fake"]],
              edgecolor="white", width=0.65)
    axes[0].set_title("Fake vs Real by Platform")
    axes[0].set_xlabel("Platform")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(loc="upper right")

    # Fake rate per platform
    fake_rate = df.groupby("platform")[TARGET].mean() * 100
    axes[1].bar(fake_rate.index, fake_rate.values,
                color="#FF9500", edgecolor="white", linewidth=1.5)
    axes[1].set_title("Fake Account Rate by Platform (%)")
    axes[1].set_ylabel("Fake Rate (%)")
    axes[1].set_ylim(0, 100)
    for i, (platform, rate) in enumerate(fake_rate.items()):
        axes[1].text(i, rate + 1, f"{rate:.1f}%", ha="center", fontweight="bold")

    plt.suptitle("Platform Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_02_platform_breakdown.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame, save: bool = True):
    """Plot missing value rates per column."""
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("  [EDA] No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(11, max(4, len(missing) * 0.4)))
    colors = ["#EF233C" if v > 0.3 else "#FF9500" if v > 0.1 else "#0077B6"
              for v in missing.values]
    bars = ax.barh(missing.index[::-1], missing.values[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Missing Values (%)")
    ax.set_title(f"Missing Value Rate per Column  ({df.isnull().sum().sum():,} total missing cells)")
    ax.axvline(10,  color="#FF9500", linestyle="--", alpha=0.7, linewidth=1.5, label="10% line")
    ax.axvline(30,  color="#EF233C", linestyle="--", alpha=0.7, linewidth=1.5, label="30% line")
    ax.axvline(60,  color="#A00020", linestyle="--", alpha=0.7, linewidth=1.5, label="60% drop threshold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, missing.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_03_missing_values.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame, save: bool = True):
    """Plot distributions of key numeric features by class (Real vs Fake)."""
    key_features = [
        "followers", "following", "follower_following_ratio",
        "account_age_days", "posts_per_day", "follow_unfollow_rate",
        "username_randomness", "digit_ratio", "bio_length",
        "spam_comments_rate",
    ]
    available = [f for f in key_features if f in df.columns]
    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        for lbl, color, label in [(0, PALETTE["real"], "Real"),
                                   (1, PALETTE["fake"], "Fake")]:
            data = df[df[TARGET] == lbl][feat].dropna()
            ax.hist(data, bins=40, alpha=0.6, color=color, density=True, label=label)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions: Real vs Fake Accounts",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_04_feature_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame, save: bool = True):
    """Plot correlation matrix of numeric features with target."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.3, linecolor="white",
    )
    ax.set_title("Correlation Matrix – Numeric Features", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_05_correlation_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_target_correlation(df: pd.DataFrame, save: bool = True):
    """Bar chart of Pearson correlation of each feature with the target."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET not in num_cols:
        return

    corr = df[num_cols].corr()[TARGET].drop(TARGET).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(5, len(corr) * 0.35)))
    colors = [PALETTE["fake"] if v > 0 else PALETTE["real"] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with is_fake")
    ax.set_title("Feature Correlation with Target (is_fake)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_06_target_correlation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_binary_features(df: pd.DataFrame, save: bool = True):
    """Plot binary feature distributions by class."""
    bin_cols = [c for c in BINARY_FEATURES if c in df.columns]
    if not bin_cols:
        return

    fig, axes = plt.subplots(1, len(bin_cols), figsize=(5 * len(bin_cols), 4))
    if len(bin_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, bin_cols):
        grp = df.groupby([col, TARGET]).size().unstack(fill_value=0)
        grp.columns = ["Real", "Fake"]
        grp.index = ["No (0)", "Yes (1)"]
        grp.plot(kind="bar", ax=ax, color=[PALETTE["real"], PALETTE["fake"]],
                 edgecolor="white", width=0.65)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=9)

    plt.suptitle("Binary Feature Distributions: Real vs Fake", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_07_binary_features.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def statistical_summary(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Generate per-class statistical summary for numeric features."""
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    rows = []
    for col in num_cols:
        real = df[df[TARGET] == 0][col].dropna()
        fake = df[df[TARGET] == 1][col].dropna()
        rows.append({
            "Feature":      col,
            "Real Mean":    round(real.mean(), 3),
            "Real Median":  round(real.median(), 3),
            "Real Std":     round(real.std(), 3),
            "Fake Mean":    round(fake.mean(), 3),
            "Fake Median":  round(fake.median(), 3),
            "Fake Std":     round(fake.std(), 3),
            "Missing %":    round(df[col].isnull().mean() * 100, 1),
        })

    summary_df = pd.DataFrame(rows)

    if save:
        path = os.path.join(OUTPUT_DIR, "eda_statistical_summary.xlsx")
        summary_df.to_excel(path, index=False)
        print(f"  Saved: {os.path.basename(path)}")

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame):
    """Run the complete EDA pipeline and save all plots."""
    print(f"\n{'='*55}")
    print(f"  [EDA] Running full exploratory data analysis")
    print(f"{'='*55}")
    print(f"\n  Saving all plots to: {OUTPUT_DIR}/\n")

    plot_class_distribution(df)
    plot_platform_breakdown(df)
    plot_missing_values(df)
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_target_correlation(df)
    plot_binary_features(df)
    summary = statistical_summary(df)

    print(f"\n  EDA complete. {len(os.listdir(OUTPUT_DIR))} files saved.")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_data, validate_data
    df  = load_data()
    df  = validate_data(df)
    run_full_eda(df)
