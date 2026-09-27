#!/usr/bin/env python3

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl


# ============================================================
# CONFIG
# ============================================================

SEED = 42
VALIDATION_FRACTION = 0.20
THRESHOLD = 0.80

ROOT = Path.cwd()

S2_FEATURES = ROOT / "artifacts/features/train_features_s2.parquet"
S3_FEATURES = ROOT / "artifacts/features/train_features_s3.parquet"

MODEL_PATH = ROOT / "artifacts/models/lightgbm_pair_model.txt"

REPORT_PATH = (
    ROOT / "artifacts/reports/entity_validation_report.json"
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

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


# ============================================================
# HELPERS
# ============================================================

def fbeta_score_single(
    precision,
    recall,
    beta=0.5,
):
    if precision == 0 and recall == 0:
        return 0.0

    beta2 = beta * beta

    denominator = (
        beta2 * precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta2)
        * precision
        * recall
        / denominator
    )


def entity_f05(
    predicted,
    actual,
):
    """
    Set-based F0.5 for one Source-1 entity.
    """

    predicted = set(predicted)
    actual = set(actual)

    tp = len(predicted & actual)

    fp = len(predicted - actual)

    fn = len(actual - predicted)

    if tp + fp == 0:
        precision = 1.0 if tp + fn == 0 else 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    f05 = fbeta_score_single(
        precision,
        recall,
        beta=0.5,
    )

    return precision, recall, f05, tp, fp, fn


# ============================================================
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("STAGE 07 - ENTITY LEVEL DECISION ENGINE")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature files...")

s2 = pl.read_parquet(S2_FEATURES)

s3 = pl.read_parquet(S3_FEATURES)

df = pl.concat(
    [s2, s3],
    how="vertical",
)

print(f"S2 rows: {s2.height:,}")
print(f"S3 rows: {s3.height:,}")
print(f"Total rows: {df.height:,}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LightGBM model...")

booster = lgb.Booster(
    model_file=str(MODEL_PATH)
)

print(
    f"Model loaded. "
    f"Trees: {booster.num_trees()}"
)


# ============================================================
# REPRODUCE EXACT ENTITY VALIDATION SPLIT
# ============================================================

print("\nReproducing validation split...")

source1_ids = df[
    "source1_entity_id"
].to_numpy()

unique_entities = np.unique(
    source1_ids
)

rng = np.random.default_rng(
    SEED
)

rng.shuffle(
    unique_entities
)

n_validation_entities = int(
    len(unique_entities)
    * VALIDATION_FRACTION
)

validation_entities = set(
    unique_entities[
        :n_validation_entities
    ]
)

validation_mask = np.array(
    [
        x in validation_entities
        for x in source1_ids
    ],
    dtype=bool,
)

valid_df = df.filter(
    pl.Series(
        "validation_mask",
        validation_mask,
    )
)

print(
    f"Validation entities: "
    f"{len(validation_entities):,}"
)

print(
    f"Validation rows: "
    f"{valid_df.height:,}"
)


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating pair probabilities...")

X_valid = valid_df.select(
    FEATURE_COLUMNS
).to_numpy()

probabilities = booster.predict(
    X_valid
)

print(
    f"Predictions generated: "
    f"{len(probabilities):,}"
)


# ============================================================
# BUILD PREDICTED MATCHES
# ============================================================

print(
    f"\nApplying entity threshold: "
    f"{THRESHOLD:.2f}"
)

valid_df = valid_df.with_columns(
    pl.Series(
        "match_probability",
        probabilities,
    )
)

predicted_df = valid_df.filter(
    pl.col("match_probability")
    >= THRESHOLD
)

print(
    f"Predicted positive pairs: "
    f"{predicted_df.height:,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

from business_entity_resolution.src.data_loader import (
    load_train_ground_truth,
)

ground_truth = load_train_ground_truth()

gt = pl.from_pandas(
    ground_truth
)

gt = gt.select(
    [
        pl.col(
            "source1_entity_id"
        )
        .cast(pl.Utf8),

        pl.col(
            "matched_entity_ids"
        )
        .cast(pl.Utf8),
    ]
)


# ============================================================
# EXPAND GROUND TRUTH
# ============================================================

gt_pairs = (
    gt
    .filter(
        pl.col(
            "matched_entity_ids"
        ).is_not_null()
        &
        (
            pl.col(
                "matched_entity_ids"
            )
            .str.strip_chars()
            != ""
        )
    )
    .with_columns(
        pl.col(
            "matched_entity_ids"
        )
        .str.split(",")
        .alias("matched_list")
    )
    .explode("matched_list")
    .with_columns(
        pl.col("matched_list")
        .str.strip_chars()
        .alias("candidate_entity_id")
    )
    .select(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )
    .filter(
        pl.col(
            "candidate_entity_id"
        ).is_not_null()
        &
        (
            pl.col(
                "candidate_entity_id"
            ) != ""
        )
    )
    .unique()
)


# ============================================================
# RESTRICT GT TO VALIDATION ENTITIES
# ============================================================

gt_pairs = gt_pairs.filter(
    pl.col("source1_entity_id")
    .is_in(
        pl.Series(
            list(validation_entities)
        )
    )
)


# ============================================================
# BUILD DICTIONARIES
# ============================================================

print("\nBuilding entity match sets...")

predicted_groups = (
    predicted_df
    .select(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )
    .group_by(
        "source1_entity_id"
    )
    .agg(
        pl.col(
            "candidate_entity_id"
        ).alias("predicted_matches")
    )
)

actual_groups = (
    gt_pairs
    .group_by(
        "source1_entity_id"
    )
    .agg(
        pl.col(
            "candidate_entity_id"
        ).alias("actual_matches")
    )
)

predicted_dict = {
    row["source1_entity_id"]:
        set(row["predicted_matches"])
    for row in predicted_groups.iter_rows(
        named=True
    )
}

actual_dict = {
    row["source1_entity_id"]:
        set(row["actual_matches"])
    for row in actual_groups.iter_rows(
        named=True
    )
}


# ============================================================
# ENTITY-LEVEL MACRO F0.5
# ============================================================

print(
    "\nCalculating entity-level Macro F0.5..."
)

entity_scores = []

total_tp = 0
total_fp = 0
total_fn = 0

predicted_nonzero = 0
actual_nonzero = 0

zero_correct = 0

for entity_id in validation_entities:

    predicted = predicted_dict.get(
        entity_id,
        set(),
    )

    actual = actual_dict.get(
        entity_id,
        set(),
    )

    precision, recall, f05, tp, fp, fn = (
        entity_f05(
            predicted,
            actual,
        )
    )

    entity_scores.append(
        f05
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    if len(predicted) > 0:
        predicted_nonzero += 1

    if len(actual) > 0:
        actual_nonzero += 1

    if (
        len(predicted) == 0
        and len(actual) == 0
    ):
        zero_correct += 1


macro_f05 = float(
    np.mean(entity_scores)
)

micro_precision = (
    total_tp
    / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0.0
)

micro_recall = (
    total_tp
    / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0.0
)

micro_f05 = fbeta_score_single(
    micro_precision,
    micro_recall,
    beta=0.5,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ENTITY-LEVEL VALIDATION RESULTS")
print("=" * 70)

print(
    f"Macro F0.5:          {macro_f05:.6f}"
)

print(
    f"Micro precision:     {micro_precision:.6f}"
)

print(
    f"Micro recall:        {micro_recall:.6f}"
)

print(
    f"Micro F0.5:          {micro_f05:.6f}"
)

print(
    f"TP:                  {total_tp:,}"
)

print(
    f"FP:                  {total_fp:,}"
)

print(
    f"FN:                  {total_fn:,}"
)

print(
    f"Actual nonzero:      {actual_nonzero:,}"
)

print(
    f"Predicted nonzero:   {predicted_nonzero:,}"
)

print(
    f"Correct zero-match:  {zero_correct:,}"
)

print(
    f"Total validation:    {len(validation_entities):,}"
)


# ============================================================
# REPORT
# ============================================================

runtime = time.time() - start_time

report = {
    "stage": "07_entity_decision",

    "status": "complete",

    "seed": SEED,

    "threshold": THRESHOLD,

    "validation": {
        "entities": int(
            len(validation_entities)
        ),

        "rows": int(
            valid_df.height
        ),

        "predicted_positive_pairs": int(
            predicted_df.height
        ),

        "actual_positive_pairs": int(
            sum(
                len(v)
                for v in actual_dict.values()
            )
        ),

        "macro_f0.5": macro_f05,

        "micro_precision": micro_precision,

        "micro_recall": micro_recall,

        "micro_f0.5": micro_f05,

        "tp": int(total_tp),

        "fp": int(total_fp),

        "fn": int(total_fn),

        "actual_nonzero_entities": int(
            actual_nonzero
        ),

        "predicted_nonzero_entities": int(
            predicted_nonzero
        ),

        "correct_zero_match_entities": int(
            zero_correct
        ),
    },

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


print("\nReport:")
print(REPORT_PATH)

print(
    f"\nRuntime: {runtime:.2f} seconds"
)

print("=" * 70)
print("STAGE 07 COMPLETE")
print("=" * 70)