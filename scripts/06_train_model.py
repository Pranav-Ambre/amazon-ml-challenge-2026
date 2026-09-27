#!/usr/bin/env python3

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

ROOT = Path.cwd()

S2_FEATURES = ROOT / "artifacts/features/train_features_s2.parquet"
S3_FEATURES = ROOT / "artifacts/features/train_features_s3.parquet"

MODEL_DIR = ROOT / "artifacts/models"
REPORT_DIR = ROOT / "artifacts/reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_pair_model.txt"
REPORT_PATH = REPORT_DIR / "model_training_report.json"


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "name_exact",
    "name_compact_exact",
    "name_alnum_exact",
    "name_sorted_exact",

    "address_exact",
    "address_compact_exact",
    "address_alnum_exact",
    "address_sorted_exact",

    "country_exact",
    "house_number_exact",

    "postal_overlap",
    "address_numeric_overlap",
    "name_token_overlap",
    "address_token_overlap",

    "block_support_count_feature",

    "s1_name_len",
    "s2_name_len",
    "s1_address_len",
    "s2_address_len",

    "name_length_diff",
    "address_length_diff",
]


TARGET_COLUMN = "label"


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("STAGE 06 - LIGHTGBM MODEL TRAINING")
print("=" * 70)

start_time = time.time()

print("\nLoading S2 features...")
s2 = pl.read_parquet(S2_FEATURES)

print(f"S2 rows: {s2.height:,}")
print(f"S2 columns: {s2.width}")

print("\nLoading S3 features...")
s3 = pl.read_parquet(S3_FEATURES)

print(f"S3 rows: {s3.height:,}")
print(f"S3 columns: {s3.width}")


# ============================================================
# COMBINE
# ============================================================

print("\nCombining S2 + S3...")

df = pl.concat(
    [
        s2,
        s3,
    ],
    how="vertical",
)

print(f"Total rows: {df.height:,}")


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise RuntimeError(
        f"Target column '{TARGET_COLUMN}' not found.\n"
        f"Available columns:\n{df.columns}"
    )

print("\nTarget distribution:")

target_counts = (
    df
    .group_by(TARGET_COLUMN)
    .len()
    .sort(TARGET_COLUMN)
)

print(target_counts)


# ============================================================
# CONVERT FEATURES
# ============================================================

print("\nPreparing feature matrix...")

X = df.select(FEATURE_COLUMNS).to_numpy()
y = df[TARGET_COLUMN].to_numpy().astype(np.int8)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ============================================================
# ENTITY-AWARE VALIDATION SPLIT
# ============================================================
#
# IMPORTANT:
# We split by source1_entity_id so the same entity does not
# appear in both training and validation.
#
# This is much safer than a random row split because each
# Source 1 entity can have many candidate pairs.
# ============================================================

print("\nCreating entity-aware validation split...")

source1_ids = df["source1_entity_id"].to_numpy()

unique_entities = np.unique(source1_ids)

rng = np.random.default_rng(SEED)

rng.shuffle(unique_entities)

validation_fraction = 0.20

n_validation_entities = int(
    len(unique_entities) * validation_fraction
)

validation_entities = set(
    unique_entities[:n_validation_entities]
)

validation_mask = np.array(
    [x in validation_entities for x in source1_ids],
    dtype=bool,
)

train_mask = ~validation_mask

X_train = X[train_mask]
y_train = y[train_mask]

X_valid = X[validation_mask]
y_valid = y[validation_mask]

print(f"Unique entities: {len(unique_entities):,}")
print(f"Training entities: {train_mask.sum():,}")
print(f"Validation entities: {validation_mask.sum():,}")

print(f"Training rows: {len(y_train):,}")
print(f"Validation rows: {len(y_valid):,}")


# ============================================================
# MODEL
# ============================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=700,

    learning_rate=0.05,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=100,

    subsample=0.85,

    colsample_bytree=0.90,

    reg_alpha=0.1,

    reg_lambda=1.0,

    random_state=SEED,

    n_jobs=8,

    verbosity=-1,
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid),
    ],

    eval_metric="binary_logloss",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            verbose=True,
        ),
        lgb.log_evaluation(
            period=25,
        ),
    ],
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

model.booster_.save_model(
    str(MODEL_PATH)
)

print(f"Model saved: {MODEL_PATH}")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

valid_probability = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nSearching probability threshold...")

threshold_results = []

thresholds = np.arange(
    0.05,
    0.951,
    0.01,
)

for threshold in thresholds:

    prediction = (
        valid_probability >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_valid,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        prediction,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_valid,
        prediction,
        beta=0.5,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f0.5": float(f05),
        }
    )


best_threshold_result = max(
    threshold_results,
    key=lambda x: x["f0.5"],
)

best_threshold = best_threshold_result["threshold"]


print("\nBest pair-level threshold:")
print(
    f"threshold={best_threshold:.2f} "
    f"precision={best_threshold_result['precision']:.4f} "
    f"recall={best_threshold_result['recall']:.4f} "
    f"F0.5={best_threshold_result['f0.5']:.4f}"
)


# ============================================================
# BEST THRESHOLD CLASSIFICATION REPORT
# ============================================================

best_prediction = (
    valid_probability >= best_threshold
).astype(np.int8)

print("\nClassification report:")

print(
    classification_report(
        y_valid,
        best_prediction,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# VALIDATION ENTITY STATISTICS
# ============================================================

valid_source1_ids = source1_ids[validation_mask]

predicted_positive_count = int(
    best_prediction.sum()
)

actual_positive_count = int(
    y_valid.sum()
)

print("\nValidation statistics:")
print(f"Actual positive pairs: {actual_positive_count:,}")
print(f"Predicted positive pairs: {predicted_positive_count:,}")


# ============================================================
# SAVE REPORT
# ============================================================

runtime = time.time() - start_time

report = {
    "stage": "06_model_training",

    "status": "complete",

    "seed": SEED,

    "model": {
        "algorithm": "LightGBM",
        "objective": "binary",
        "n_estimators": int(model.n_estimators_),
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    "dataset": {
        "s2_rows": int(s2.height),
        "s3_rows": int(s3.height),
        "total_rows": int(df.height),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    },

    "split": {
        "validation_fraction": validation_fraction,
        "unique_entities": int(len(unique_entities)),
        "training_entities": int(train_mask.sum()),
        "validation_entities": int(validation_mask.sum()),
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
    },

    "validation": {
        "actual_positive_pairs": actual_positive_count,
        "predicted_positive_pairs": predicted_positive_count,
        "best_threshold": best_threshold,
        "best_threshold_metrics": best_threshold_result,
        "threshold_results": threshold_results,
    },

    "model_path": str(MODEL_PATH),

    "runtime_seconds": runtime,
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("STAGE 06 COMPLETE")
print("=" * 70)

print(f"Model:  {MODEL_PATH}")
print(f"Report: {REPORT_PATH}")
print(f"Runtime: {runtime:.2f} sec")
print("=" * 70)