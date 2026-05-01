"""
src/feature_engineering.py
───────────────────────────
Step 3 of the pipeline: Feature Engineering.

Derives new discriminative features from raw profile data across
four modality groups:
  1. Metadata features  (log transforms, ratios, composite scores)
  2. NLP/text features  (lexical entropy, hashtag density, spam score)
  3. Image features     (realness score, stock photo proxy, mismatch)
  4. Behaviour features (anomaly flags, suspicion score)

All functions are pure transforms: they accept a DataFrame and
return a NEW DataFrame with additional columns, never modifying
the input in place.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    OUTPUT_DIR, TARGET,
    HIGH_DIGIT_RATIO_THRESH, VERY_NEW_ACCOUNT_DAYS,
    BURST_POSTING_PER_DAY, HIGH_SPAM_RATE_THRESH,
    HIGH_GENERIC_RATE_THRESH, COPY_PASTE_SIMILARITY_THRESH,
    EXTREME_FUF_THRESH, LOW_FF_RATIO_THRESH,
)


# ── 1. Metadata feature engineering ───────────────────────────────────────────
def engineer_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engineered metadata features.

    New columns:
      followers_log         : log1p(followers)
      following_log         : log1p(following)
      posts_log             : log1p(posts)
      account_age_log       : log1p(account_age_days)
      activity_index        : followers × posts_per_day
      follower_deficit      : max(0, following - followers)
      high_digit_username   : digit_ratio > HIGH_DIGIT_RATIO_THRESH
      no_profile_pic        : has_profile_pic == 0
      very_new_account      : account_age_days < VERY_NEW_ACCOUNT_DAYS
      suspicious_composite  : follow_unfollow_rate × no_profile_pic
    """
    df = df.copy()

    # Log transforms (reduce right-skew in count distributions)
    for col in ["followers", "following", "posts", "account_age_days"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0).fillna(0))

    # Activity index
    if {"followers", "posts_per_day"}.issubset(df.columns):
        df["activity_index"] = (
            df["followers"].fillna(0) * df["posts_per_day"].fillna(0)
        )

    # Follower deficit (following >> followers signals bot)
    if {"following", "followers"}.issubset(df.columns):
        df["follower_deficit"] = (
            df["following"].fillna(0) - df["followers"].fillna(0)
        ).clip(lower=0)

    # Binary anomaly flags
    if "digit_ratio" in df.columns:
        df["high_digit_username"] = (
            df["digit_ratio"].fillna(0) > HIGH_DIGIT_RATIO_THRESH
        ).astype(int)

    if "has_profile_pic" in df.columns:
        df["no_profile_pic"] = (
            df["has_profile_pic"].fillna(1) == 0
        ).astype(int)

    if "account_age_days" in df.columns:
        df["very_new_account"] = (
            df["account_age_days"].fillna(365) < VERY_NEW_ACCOUNT_DAYS
        ).astype(int)

    # Suspicious composite (no pic AND high follow-unfollow)
    if {"follow_unfollow_rate", "has_profile_pic"}.issubset(df.columns):
        df["suspicious_composite"] = (
            df["follow_unfollow_rate"].fillna(0) *
            (1 - df["has_profile_pic"].fillna(1))
        )

    return df


# ── 2. NLP / text feature engineering ─────────────────────────────────────────
def engineer_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive NLP proxy features from username and bio characteristics.

    New columns:
      lexical_entropy   : proxy for username randomness (entropy × 4.5)
      hashtag_density   : special_char_count / username_length
      repetition_ratio  : repeat_char_count / username_length
      bio_too_short     : bio_length < 20
      bio_log           : log1p(bio_length)
      bio_pic_match     : has profile pic AND bio_length > 50
      spam_score        : weighted sum of text anomaly signals
    """
    df = df.copy()

    # Lexical entropy proxy
    if "username_randomness" in df.columns:
        df["lexical_entropy"] = df["username_randomness"].fillna(0) * 4.5

    # Hashtag density (special characters as proxy)
    if "special_char_count" in df.columns:
        denom = df["username_length"].fillna(1).replace(0, 1) \
                if "username_length" in df.columns else 1
        df["hashtag_density"] = df["special_char_count"].fillna(0) / denom

    # Repetition ratio
    if "repeat_char_count" in df.columns:
        denom = df["username_length"].fillna(1).replace(0, 1) \
                if "username_length" in df.columns else 1
        df["repetition_ratio"] = df["repeat_char_count"].fillna(0) / denom

    # Bio signals
    if "bio_length" in df.columns:
        df["bio_too_short"] = (df["bio_length"].fillna(50) < 20).astype(int)
        df["bio_log"]       = np.log1p(df["bio_length"].fillna(0))

    # Bio + pic consistency (real users with pics usually have bios)
    if {"bio_length", "has_profile_pic"}.issubset(df.columns):
        df["bio_pic_match"] = (
            (df["has_profile_pic"].fillna(0) == 1) &
            (df["bio_length"].fillna(0) > 50)
        ).astype(int)

    # Composite spam score
    cols_for_spam = {
        "lexical_entropy":  0.30,
        "hashtag_density":  0.25,
        "repetition_ratio": 0.20,
        "digit_ratio":      0.15,
        "bio_too_short":    0.10,
    }
    spam_score = pd.Series(0.0, index=df.index)
    for col, weight in cols_for_spam.items():
        if col in df.columns:
            spam_score += df[col].fillna(0) * weight
    df["spam_score"] = spam_score

    return df


# ── 3. Image feature engineering ───────────────────────────────────────────────
def engineer_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive image proxy features (full OpenCV extraction is future work).

    New columns:
      face_detected_proxy   : has_pic weighted by verified status
      image_realness_score  : combined proxy of pic + age + followers
      stock_photo_proxy     : pic present but new account + high digit ratio
      bio_pic_mismatch      : has profile pic but very short bio
    """
    df = df.copy()

    # Core image signal
    df["has_pic"] = df["has_profile_pic"].fillna(0) if "has_profile_pic" in df.columns else 0

    # Face detection proxy
    if "verified" in df.columns:
        df["face_detected_proxy"] = (
            df["has_pic"] * (
                0.90 * df["verified"].fillna(0) +
                0.50 * (1 - df["verified"].fillna(0))
            )
        )
    else:
        df["face_detected_proxy"] = df["has_pic"] * 0.5

    # Image realness composite
    age_norm  = np.clip(df["account_age_log"] / 8.0, 0, 1) \
                if "account_age_log" in df.columns else 0
    fol_norm  = np.clip(df["followers_log"] / 10.0, 0, 1) \
                if "followers_log" in df.columns else 0

    df["image_realness_score"] = (
        df["has_pic"]              * 0.50 +
        df["face_detected_proxy"]  * 0.30 +
        age_norm                   * 0.10 +
        fol_norm                   * 0.10
    )

    # Stock photo proxy
    if {"account_age_days", "digit_ratio"}.issubset(df.columns):
        df["stock_photo_proxy"] = (
            df["has_pic"] *
            (df["account_age_days"].fillna(365) < 60).astype(float) *
            (df["digit_ratio"].fillna(0) > 0.35).astype(float)
        )
    else:
        df["stock_photo_proxy"] = 0.0

    # Bio-picture mismatch
    if "bio_length" in df.columns:
        df["bio_pic_mismatch"] = (
            (df["has_pic"] == 1) & (df["bio_length"].fillna(50) < 15)
        ).astype(int)
    else:
        df["bio_pic_mismatch"] = 0

    return df


# ── 4. Behaviour feature engineering ──────────────────────────────────────────
def engineer_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive behavioural anomaly flags and composite suspicion score.

    New columns:
      burst_posting        : posts_per_day > BURST threshold
      posts_day_log        : log1p(posts_per_day)
      high_spam            : spam_comments_rate > threshold
      high_generic         : generic_comment_rate > threshold
      avg_similarity       : mean of caption + content similarity
      copy_paste_flag      : avg_similarity > threshold
      fuf_log              : log1p(follow_unfollow_rate)
      extreme_fuf          : follow_unfollow_rate > EXTREME threshold
      low_ff_ratio         : follower_following_ratio < LOW threshold
      behaviour_suspicion  : normalised sum of all binary anomaly flags
    """
    df = df.copy()

    if "posts_per_day" in df.columns:
        df["burst_posting"] = (
            df["posts_per_day"].fillna(0) > BURST_POSTING_PER_DAY
        ).astype(float)
        df["posts_day_log"] = np.log1p(df["posts_per_day"].fillna(0))

    if "spam_comments_rate" in df.columns:
        df["high_spam"] = (
            df["spam_comments_rate"].fillna(0) > HIGH_SPAM_RATE_THRESH
        ).astype(float)

    if "generic_comment_rate" in df.columns:
        df["high_generic"] = (
            df["generic_comment_rate"].fillna(0) > HIGH_GENERIC_RATE_THRESH
        ).astype(float)

    if {"caption_similarity_score", "content_similarity_score"}.issubset(df.columns):
        df["avg_similarity"] = (
            df["caption_similarity_score"].fillna(0) +
            df["content_similarity_score"].fillna(0)
        ) / 2.0
        df["copy_paste_flag"] = (
            df["avg_similarity"] > COPY_PASTE_SIMILARITY_THRESH
        ).astype(float)

    if "follow_unfollow_rate" in df.columns:
        df["fuf_log"]    = np.log1p(df["follow_unfollow_rate"].fillna(0))
        df["extreme_fuf"] = (
            df["follow_unfollow_rate"].fillna(0) > EXTREME_FUF_THRESH
        ).astype(float)

    if "follower_following_ratio" in df.columns:
        df["low_ff_ratio"] = (
            df["follower_following_ratio"].fillna(1) < LOW_FF_RATIO_THRESH
        ).astype(float)

    # Composite behaviour suspicion score
    flag_cols = ["high_spam", "high_generic", "copy_paste_flag",
                 "extreme_fuf", "burst_posting", "low_ff_ratio"]
    present   = [c for c in flag_cols if c in df.columns]
    if present:
        df["behaviour_suspicion"] = df[present].sum(axis=1) / len(present)

    return df


# ── Run all engineering steps ──────────────────────────────────────────────────
def run_all_feature_engineering(df: pd.DataFrame,
                                 verbose: bool = True) -> pd.DataFrame:
    """
    Apply all four feature engineering steps in sequence.

    Returns enriched DataFrame with all derived features.
    """
    original_cols = set(df.columns)

    df = engineer_metadata_features(df)
    df = engineer_text_features(df)
    df = engineer_image_features(df)
    df = engineer_behaviour_features(df)

    new_cols = [c for c in df.columns if c not in original_cols]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  [FeatureEngineering] Complete")
        print(f"{'='*55}")
        print(f"  Original features : {len(original_cols)}")
        print(f"  New features added: {len(new_cols)}")
        print(f"  Total features    : {df.shape[1]}")
        print(f"\n  New columns:")
        for i, col in enumerate(new_cols, 1):
            print(f"    {i:2}. {col}")

    return df


# ── Visualise engineered features ─────────────────────────────────────────────
def plot_engineered_features(df: pd.DataFrame, save: bool = True):
    """Plot distributions of selected engineered features by class."""
    eng_features = [
        "followers_log", "activity_index", "follower_deficit",
        "suspicious_composite", "lexical_entropy", "spam_score",
        "image_realness_score", "behaviour_suspicion",
    ]
    available = [f for f in eng_features if f in df.columns]
    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        for lbl, color, label in [(0, "#0077B6", "Real"),
                                   (1, "#EF233C", "Fake")]:
            data = df[df[TARGET] == lbl][feat].dropna()
            ax.hist(data, bins=35, alpha=0.6, color=color,
                    density=True, label=label)
        ax.set_title(feat.replace("_", " ").title())
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Engineered Feature Distributions: Real vs Fake",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fe_engineered_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_data, validate_data
    df  = load_data()
    df  = validate_data(df)
    df  = run_all_feature_engineering(df)
    plot_engineered_features(df)
    print(f"\nFinal shape: {df.shape}")
