"""
config.py
─────────
Central configuration for the DeepMind project.
Detection of Fraudulent Social Media Accounts Using AI
University of Moratuwa · Faculty of IT · 2026
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
SRC_DIR    = os.path.join(BASE_DIR, "src")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH      = os.path.join(DATA_DIR, "fake_social_media_raw.xlsx")
EXPANDED_DATA_PATH = os.path.join(DATA_DIR, "fake_social_media_10000.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET       = "is_fake"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "followers",
    "following",
    "follower_following_ratio",
    "account_age_days",
    "posts",
    "posts_per_day",
    "bio_length",
    "username_length",
    "digits_count",
    "digit_ratio",
    "special_char_count",
    "repeat_char_count",
    "username_randomness",
    "follow_unfollow_rate",
    "spam_comments_rate",
    "generic_comment_rate",
    "caption_similarity_score",
    "content_similarity_score",
    "num_friends",
    "num_likes",
]

BINARY_FEATURES = [
    "has_profile_pic",
    "verified",
    "suspicious_links_in_bio",
]

CATEGORICAL_FEATURES = [
    "platform",
]

# Columns to DROP before modelling (non-predictive identifiers)
DROP_COLUMNS = [
    "username",
    "profile_pic_url",
    "num_posts",   # duplicate of posts
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ── Preprocessing thresholds ───────────────────────────────────────────────────
MISSING_DROP_THRESHOLD  = 0.60   # drop column if >60% missing
OUTLIER_IQR_MULTIPLIER  = 3.0    # IQR multiplier for outlier capping
SCALE_NUMERIC           = True
ENCODE_CATEGORICAL      = True

# ── Engineered feature thresholds ─────────────────────────────────────────────
HIGH_DIGIT_RATIO_THRESH     = 0.40
VERY_NEW_ACCOUNT_DAYS       = 30
BURST_POSTING_PER_DAY       = 10
HIGH_SPAM_RATE_THRESH       = 0.50
HIGH_GENERIC_RATE_THRESH    = 0.60
COPY_PASTE_SIMILARITY_THRESH= 0.70
EXTREME_FUF_THRESH          = 0.70
LOW_FF_RATIO_THRESH         = 0.10
