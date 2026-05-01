# DeepMind – Fake Account Detection: Data to Preprocessing
**Faculty of Information Technology · University of Moratuwa · 2026**

---

## What this package covers

This is **Stage 1** of the DeepMind project — everything from raw data to a clean, engineered, and preprocessed dataset ready for model training.

```
Raw Excel Data
    ↓
Step 1: Data Loading & Validation
    ↓
Step 2: Exploratory Data Analysis
    ↓
Step 3: Feature Engineering
    ↓
Step 4: Preprocessing (impute → scale → encode → split)
    ↓
X_train, X_test, y_train, y_test  (ready for model training)
```

---

## Project Structure

```
deepmind_preprocess/
│
├── config.py                      ← All settings & thresholds
├── run_pipeline.py                ← Master script (run this)
├── requirements.txt
│
├── data/
│   ├── fake_social_media_raw.xlsx         ← Original 3,000 rows
│   └── fake_social_media_10000.xlsx       ← Expanded 10,000 rows
│
├── src/
│   ├── data_loader.py             ← Step 1: Load & validate
│   ├── eda.py                     ← Step 2: EDA plots & stats
│   ├── feature_engineering.py     ← Step 3: Derive new features
│   └── preprocessing.py           ← Step 4: Impute/scale/encode/split
│
└── outputs/                       ← All plots & saved artefacts
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

### 3. Options
```bash
python run_pipeline.py --raw         # use original 3K dataset
python run_pipeline.py --no-eda      # skip EDA plots (faster)
python run_pipeline.py --smote       # apply SMOTE oversampling
python run_pipeline.py --smote --no-eda   # fast mode with SMOTE
```

### 4. Run individual steps
```bash
python src/data_loader.py           # Step 1 only
python src/eda.py                   # Step 2 only
python src/feature_engineering.py   # Step 3 only
python src/preprocessing.py         # Step 4 only
```

---

## What each step does

### Step 1 – Data Loading & Validation (`src/data_loader.py`)
- Loads raw Excel file (3K or 10K rows)
- Reports shape, class balance, missing values
- Drops identifier columns (username, profile_pic_url)
- Removes all-NaN rows and duplicate rows
- Drops columns with >60% missing values
- Returns clean DataFrame

### Step 2 – Exploratory Data Analysis (`src/eda.py`)
Saves 7 plots to `outputs/`:
- `eda_01_class_distribution.png`     — bar + pie chart
- `eda_02_platform_breakdown.png`     — per-platform fake rate
- `eda_03_missing_values.png`         — missing rate per column
- `eda_04_feature_distributions.png`  — Real vs Fake histograms
- `eda_05_correlation_matrix.png`     — feature correlation heatmap
- `eda_06_target_correlation.png`     — correlation with is_fake
- `eda_07_binary_features.png`        — binary feature breakdown
- `eda_statistical_summary.xlsx`      — per-class stats table

### Step 3 – Feature Engineering (`src/feature_engineering.py`)
Derives **~25 new features** across 4 groups:

| Group | New Features |
|---|---|
| Metadata | log transforms, activity_index, follower_deficit, anomaly flags |
| Text/NLP | lexical_entropy, hashtag_density, repetition_ratio, spam_score |
| Image | face_detected_proxy, image_realness_score, stock_photo_proxy |
| Behaviour | burst_posting, high_spam, copy_paste_flag, behaviour_suspicion |

### Step 4 – Preprocessing (`src/preprocessing.py`)
- **Outlier capping**: IQR × 3.0 (fitted on train only)
- **Train/test split**: 80/20 stratified by class
- **Imputation**: median (numeric), mode (binary), constant (categorical)
- **Scaling**: StandardScaler for numeric features
- **Encoding**: OneHotEncoder for platform column
- **Optional SMOTE**: synthetic minority oversampling
- Saves `preprocessor.pkl`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

---

## Outputs after running

```
outputs/
├── eda_01_class_distribution.png
├── eda_02_platform_breakdown.png
├── eda_03_missing_values.png
├── eda_04_feature_distributions.png
├── eda_05_correlation_matrix.png
├── eda_06_target_correlation.png
├── eda_07_binary_features.png
├── eda_statistical_summary.xlsx
├── fe_engineered_distributions.png
├── preprocess_summary.png
├── preprocessor.pkl            ← fitted sklearn ColumnTransformer
├── outlier_bounds.pkl          ← IQR bounds per column
├── feature_names.pkl           ← list of feature names after encoding
├── X_train.npy                 ← processed training features
├── X_test.npy                  ← processed test features
├── y_train.npy                 ← training labels
└── y_test.npy                  ← test labels
```

---

## Group Members
| Name | Index |
|---|---|
| Perera J.K.A.T | 204155M |
| Jayalath N.S | 204081G |
| Wijethunge L.S.K | 204235J |
| Sandeep A.Y | 204188P |

**Supervisors:** Dr. Thanuja A.L.A.R.R | Dr. T.M. Thanthriwatta
