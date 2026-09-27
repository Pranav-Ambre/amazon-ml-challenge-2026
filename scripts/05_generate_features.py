#!/usr/bin/env python3

"""
Stage 05 - Training Feature Generation

Deadline-oriented feature generation for Amazon ML Challenge 2026.

Strategy
--------
1. Load ground truth using the project's official loader.
2. Convert comma-separated matched_entity_ids into pair-level GT.
3. Load generated candidate pairs.
4. Keep every candidate that is a true positive.
5. Sample negatives from the remaining candidates.
6. Join normalized source attributes.
7. Generate lightweight pairwise features.
8. Write train_features_s2.parquet and train_features_s3.parquet.

Important
---------
We intentionally do NOT materialize all 444M candidate pairs into pandas.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import polars as pl

from business_entity_resolution.src.config import ARTIFACTS_DIR
from business_entity_resolution.src.data_loader import load_train_ground_truth


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

# Negative / positive ratio.
# 1.0 means approximately one negative per positive.
NEGATIVE_TO_POSITIVE_RATIO = 1.0

# Prevent accidental explosion if the positive count is unexpectedly huge.
MAX_NEGATIVES = 8_000_000

RNG = random.Random(SEED)

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
FEATURE_DIR = ARTIFACTS_DIR / "features"
REPORT_DIR = ARTIFACTS_DIR / "reports"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_S1 = NORMALIZED_DIR / "train_source1_normalized.parquet"
TRAIN_S2 = NORMALIZED_DIR / "train_source2_normalized.parquet"
TRAIN_S3 = NORMALIZED_DIR / "train_source3_normalized.parquet"

CAND_S2 = CANDIDATES_DIR / "train_s1_to_s2_candidates.parquet"
CAND_S3 = CANDIDATES_DIR / "train_s1_to_s3_candidates.parquet"

OUT_S2 = FEATURE_DIR / "train_features_s2.parquet"
OUT_S3 = FEATURE_DIR / "train_features_s3.parquet"

REPORT_PATH = REPORT_DIR / "feature_generation_report.json"


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger("stage05")


# =============================================================================
# HELPERS
# =============================================================================

def check_paths() -> None:

    paths = [
        TRAIN_S1,
        TRAIN_S2,
        TRAIN_S3,
        CAND_S2,
        CAND_S3,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def load_ground_truth_pairs() -> tuple[pl.DataFrame, int]:

    LOGGER.info("Loading training ground truth...")

    ground_truth = load_train_ground_truth()

    gt = pl.from_pandas(ground_truth)

    LOGGER.info(
        "Ground truth rows: %,d",
        gt.height,
    )

    gt = gt.select(
        [
            pl.col("source1_entity_id")
            .cast(pl.Utf8),

            pl.col("matched_entity_ids")
            .cast(pl.Utf8),
        ]
    )

    zero_match_count = (
        gt
        .filter(
            pl.col("matched_entity_ids").is_null()
            |
            (
                pl.col("matched_entity_ids")
                .str.strip_chars()
                == ""
            )
        )
        .height
    )

    pairs = (
        gt
        .filter(
            pl.col("matched_entity_ids").is_not_null()
            &
            (
                pl.col("matched_entity_ids")
                .str.strip_chars()
                != ""
            )
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("matched_entity_id_list")
        )
        .explode("matched_entity_id_list")
        .with_columns(
            pl.col("matched_entity_id_list")
            .str.strip_chars()
            .cast(pl.Utf8)
            .alias("candidate_entity_id")
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .filter(
            pl.col("candidate_entity_id").is_not_null()
            &
            (
                pl.col("candidate_entity_id") != ""
            )
        )
        .unique()
    )

    LOGGER.info(
        "True matched pairs: %,d",
        pairs.height,
    )

    LOGGER.info(
        "Zero-match Source-1 entities: %,d",
        zero_match_count,
    )

    return pairs, zero_match_count


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_features(
    pairs: pl.LazyFrame,
    s1_path: Path,
    candidate_path: Path,
    source_name: str,
) -> pl.LazyFrame:

    LOGGER.info(
        "Preparing features for %s",
        source_name,
    )

    s1 = (
        pl.scan_parquet(s1_path)
        .select(
            [
                "entity_id",
                "business_name_norm",
                "business_name_compact",
                "business_name_alnum",
                "business_name_tokens",
                "business_name_sorted_tokens",
                "business_name_numeric_tokens",
                "business_address_norm",
                "business_address_compact",
                "business_address_alnum",
                "business_address_tokens",
                "business_address_sorted_tokens",
                "business_address_alpha_tokens",
                "business_address_numeric_tokens",
                "business_address_house_number",
                "business_address_postal_tokens",
                "country_norm",
            ]
        )
        .rename(
            {
                "entity_id": "source1_entity_id",

                "business_name_norm": "s1_name_norm",
                "business_name_compact": "s1_name_compact",
                "business_name_alnum": "s1_name_alnum",
                "business_name_tokens": "s1_name_tokens",
                "business_name_sorted_tokens": "s1_name_sorted_tokens",
                "business_name_numeric_tokens": "s1_name_numeric_tokens",

                "business_address_norm": "s1_address_norm",
                "business_address_compact": "s1_address_compact",
                "business_address_alnum": "s1_address_alnum",
                "business_address_tokens": "s1_address_tokens",
                "business_address_sorted_tokens": "s1_address_sorted_tokens",
                "business_address_alpha_tokens": "s1_address_alpha_tokens",
                "business_address_numeric_tokens": "s1_address_numeric_tokens",
                "business_address_house_number": "s1_house_number",
                "business_address_postal_tokens": "s1_postal_tokens",

                "country_norm": "s1_country_norm",
            }
        )
    )

    candidate = (
        pl.scan_parquet(candidate_path)
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "block_support_count",
            ]
        )
    )

    source = (
        pl.scan_parquet(
            TRAIN_S2 if source_name == "s2" else TRAIN_S3
        )
        .select(
            [
                "entity_id",
                "business_name_norm",
                "business_name_compact",
                "business_name_alnum",
                "business_name_tokens",
                "business_name_sorted_tokens",
                "business_name_numeric_tokens",
                "business_address_norm",
                "business_address_compact",
                "business_address_alnum",
                "business_address_tokens",
                "business_address_sorted_tokens",
                "business_address_alpha_tokens",
                "business_address_numeric_tokens",
                "business_address_house_number",
                "business_address_postal_tokens",
                "country_norm",
            ]
        )
        .rename(
            {
                "entity_id": "candidate_entity_id",

                "business_name_norm": "s2_name_norm",
                "business_name_compact": "s2_name_compact",
                "business_name_alnum": "s2_name_alnum",
                "business_name_tokens": "s2_name_tokens",
                "business_name_sorted_tokens": "s2_name_sorted_tokens",
                "business_name_numeric_tokens": "s2_name_numeric_tokens",

                "business_address_norm": "s2_address_norm",
                "business_address_compact": "s2_address_compact",
                "business_address_alnum": "s2_address_alnum",
                "business_address_tokens": "s2_address_tokens",
                "business_address_sorted_tokens": "s2_address_sorted_tokens",
                "business_address_alpha_tokens": "s2_address_alpha_tokens",
                "business_address_numeric_tokens": "s2_address_numeric_tokens",
                "business_address_house_number": "s2_house_number",
                "business_address_postal_tokens": "s2_postal_tokens",

                "country_norm": "s2_country_norm",
            }
        )
    )

    joined = (
        pairs
        .join(
            candidate,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )
        .join(
            s1,
            on="source1_entity_id",
            how="left",
        )
        .join(
            source,
            on="candidate_entity_id",
            how="left",
        )
    )

    # -------------------------------------------------------------------------
    # Lightweight deterministic features
    # -------------------------------------------------------------------------

    joined = joined.with_columns(

        # Exact normalized name.
        (
            pl.col("s1_name_norm")
            == pl.col("s2_name_norm")
        )
        .cast(pl.Int8)
        .alias("name_exact"),

        # Exact compact name.
        (
            pl.col("s1_name_compact")
            == pl.col("s2_name_compact")
        )
        .cast(pl.Int8)
        .alias("name_compact_exact"),

        # Exact alphanumeric name.
        (
            pl.col("s1_name_alnum")
            == pl.col("s2_name_alnum")
        )
        .cast(pl.Int8)
        .alias("name_alnum_exact"),

        # Sorted-token name equality.
        (
            pl.col("s1_name_sorted_tokens")
            == pl.col("s2_name_sorted_tokens")
        )
        .cast(pl.Int8)
        .alias("name_sorted_exact"),

        # Exact address.
        (
            pl.col("s1_address_norm")
            == pl.col("s2_address_norm")
        )
        .cast(pl.Int8)
        .alias("address_exact"),

        # Compact address.
        (
            pl.col("s1_address_compact")
            == pl.col("s2_address_compact")
        )
        .cast(pl.Int8)
        .alias("address_compact_exact"),

        # Alphanumeric address.
        (
            pl.col("s1_address_alnum")
            == pl.col("s2_address_alnum")
        )
        .cast(pl.Int8)
        .alias("address_alnum_exact"),

        # Sorted address tokens.
        (
            pl.col("s1_address_sorted_tokens")
            == pl.col("s2_address_sorted_tokens")
        )
        .cast(pl.Int8)
        .alias("address_sorted_exact"),

        # Country.
        (
            pl.col("s1_country_norm")
            == pl.col("s2_country_norm")
        )
        .cast(pl.Int8)
        .alias("country_exact"),

        # House number.
        (
            pl.col("s1_house_number")
            == pl.col("s2_house_number")
        )
        .cast(pl.Int8)
        .alias("house_number_exact"),

        # Postal token overlap.
        (
            pl.col("s1_postal_tokens")
            .list.set_intersection(
                pl.col("s2_postal_tokens")
            )
            .list.len()
        )
        .cast(pl.Int16)
        .alias("postal_overlap"),

        # Numeric address overlap.
        (
            pl.col("s1_address_numeric_tokens")
            .list.set_intersection(
                pl.col("s2_address_numeric_tokens")
            )
            .list.len()
        )
        .cast(pl.Int16)
        .alias("address_numeric_overlap"),

        # Name token overlap.
        (
            pl.col("s1_name_tokens")
            .list.set_intersection(
                pl.col("s2_name_tokens")
            )
            .list.len()
        )
        .cast(pl.Int16)
        .alias("name_token_overlap"),

        # Address token overlap.
        (
            pl.col("s1_address_tokens")
            .list.set_intersection(
                pl.col("s2_address_tokens")
            )
            .list.len()
        )
        .cast(pl.Int16)
        .alias("address_token_overlap"),

        # Block support.
        pl.col("block_support_count")
        .cast(pl.Int16)
        .alias("block_support_count_feature"),
    )

    # -------------------------------------------------------------------------
    # Length features
    # -------------------------------------------------------------------------

    joined = joined.with_columns(

        pl.col("s1_name_norm")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias("s1_name_len"),

        pl.col("s2_name_norm")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias("s2_name_len"),

        pl.col("s1_address_norm")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias("s1_address_len"),

        pl.col("s2_address_norm")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias("s2_address_len"),
    )

    joined = joined.with_columns(

        (
            pl.col("s1_name_len")
            - pl.col("s2_name_len")
        )
        .abs()
        .cast(pl.Int16)
        .alias("name_length_diff"),

        (
            pl.col("s1_address_len")
            - pl.col("s2_address_len")
        )
        .abs()
        .cast(pl.Int16)
        .alias("address_length_diff"),
    )

    return joined


# =============================================================================
# BUILD POSITIVES
# =============================================================================

def retrieve_positive_pairs(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
) -> pl.DataFrame:

    LOGGER.info(
        "Retrieving positives from %s",
        candidate_path.name,
    )

    candidates = pl.scan_parquet(
        candidate_path
    ).select(
        [
            "source1_entity_id",
            "candidate_entity_id",
            "block_support_count",
        ]
    )

    positives = (
        gt_pairs.lazy()
        .join(
            candidates,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )
        .unique(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .collect(
            engine="streaming"
        )
    )

    LOGGER.info(
        "Retrieved positive candidates: %,d",
        positives.height,
    )

    return positives


# =============================================================================
# SAMPLE NEGATIVES
# =============================================================================

def sample_negatives(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
    positive_count: int,
) -> pl.DataFrame:

    target = min(
        int(positive_count * NEGATIVE_TO_POSITIVE_RATIO),
        MAX_NEGATIVES,
    )

    LOGGER.info(
        "Sampling up to %,d negatives from %s",
        target,
        candidate_path.name,
    )

    # Hash-based deterministic sampling.
    #
    # We avoid collecting all negatives. Instead we assign a deterministic
    # hash score and take the lowest-scoring rows after removing GT pairs.

    gt_lazy = gt_pairs.lazy()

    candidates = (
        pl.scan_parquet(candidate_path)
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "block_support_count",
            ]
        )
        .unique(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .join(
            gt_lazy,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="anti",
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("source1_entity_id"),
                    pl.col("candidate_entity_id"),
                ],
                separator="|",
            )
            .hash(seed=SEED)
            .alias("_sample_hash")
        )
        .sort("_sample_hash")
        .head(target)
        .drop("_sample_hash")
    )

    negatives = candidates.collect(
        engine="streaming"
    )

    LOGGER.info(
        "Selected negatives: %,d",
        negatives.height,
    )

    return negatives


# =============================================================================
# BUILD LABELLED DATASET
# =============================================================================

def build_dataset(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
    s1_path: Path,
    source_path: Path,
    source_name: str,
) -> tuple[Path, dict]:

    start = time.time()

    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "Building %s training features",
        source_name,
    )

    # -------------------------------------------------------------------------
    # Positives
    # -------------------------------------------------------------------------

    positives = retrieve_positive_pairs(
        gt_pairs,
        candidate_path,
    )

    positive_count = positives.height

    # -------------------------------------------------------------------------
    # Negatives
    # -------------------------------------------------------------------------

    negatives = sample_negatives(
        gt_pairs,
        candidate_path,
        positive_count,
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    positives = positives.with_columns(
        pl.lit(1, dtype=pl.Int8).alias("label")
    )

    negatives = negatives.with_columns(
        pl.lit(0, dtype=pl.Int8).alias("label")
    )

    pairs = (
        pl.concat(
            [
                positives,
                negatives,
            ],
            how="vertical",
        )
        .unique(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
    )

    LOGGER.info(
        "Labelled pairs: %,d",
        pairs.height,
    )

    LOGGER.info(
        "Positive: %,d | Negative: %,d",
        positives.height,
        negatives.height,
    )

    # -------------------------------------------------------------------------
    # Generate features
    # -------------------------------------------------------------------------

    feature_lf = add_features(
        pairs.lazy(),
        s1_path,
        source_path,
        source_name,
    )

    # -------------------------------------------------------------------------
    # Keep model features + identifiers
    # -------------------------------------------------------------------------

    feature_lf = feature_lf.select(
        [
            "source1_entity_id",
            "candidate_entity_id",

            "label",

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
    )

    LOGGER.info(
        "Collecting final %s feature dataset...",
        source_name,
    )

    features = feature_lf.collect(
        engine="streaming"
    )

    # Fill numeric nulls.
    numeric_cols = [
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

    features = features.with_columns(
        [
            pl.col(c)
            .fill_null(0)
            .cast(pl.Int32)
            for c in numeric_cols
        ]
    )

    # Shuffle deterministically.
    features = (
        features
        .with_columns(
            pl.int_range(
                pl.len()
            )
            .shuffle(seed=SEED)
            .alias("_shuffle")
        )
        .sort("_shuffle")
        .drop("_shuffle")
    )

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    output_path = (
        OUT_S2
        if source_name == "s2"
        else OUT_S3
    )

    features.write_parquet(
        output_path,
        compression="zstd",
        compression_level=3,
    )

    elapsed = time.time() - start

    report = {
        "source": source_name,
        "candidate_file": str(candidate_path),
        "output_file": str(output_path),
        "positive_pairs_retrieved": int(positive_count),
        "negative_pairs_selected": int(negatives.height),
        "total_training_rows": int(features.height),
        "positive_rows": int(
            features.filter(
                pl.col("label") == 1
            ).height
        ),
        "negative_rows": int(
            features.filter(
                pl.col("label") == 0
            ).height
        ),
        "runtime_seconds": round(elapsed, 2),
    }

    LOGGER.info(
        "%s complete: %,d rows | %.2f min",
        source_name,
        features.height,
        elapsed / 60,
    )

    return output_path, report


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    overall_start = time.time()

    LOGGER.info(
        "============================================================"
    )
    LOGGER.info(
        "STAGE 05 - TRAINING FEATURE GENERATION"
    )
    LOGGER.info(
        "============================================================"
    )

    check_paths()

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    gt_pairs, zero_match_count = (
        load_ground_truth_pairs()
    )

    # -------------------------------------------------------------------------
    # S2
    # -------------------------------------------------------------------------

    _, report_s2 = build_dataset(
        gt_pairs=gt_pairs,
        candidate_path=CAND_S2,
        s1_path=TRAIN_S1,
        source_path=TRAIN_S2,
        source_name="s2",
    )

    # -------------------------------------------------------------------------
    # S3
    # -------------------------------------------------------------------------

    _, report_s3 = build_dataset(
        gt_pairs=gt_pairs,
        candidate_path=CAND_S3,
        s1_path=TRAIN_S1,
        source_path=TRAIN_S3,
        source_name="s3",
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    total_runtime = time.time() - overall_start

    report = {
        "stage": "05_feature_generation",
        "status": "complete",
        "seed": SEED,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "max_negatives": MAX_NEGATIVES,
        "zero_match_entities": int(zero_match_count),
        "s2": report_s2,
        "s3": report_s3,
        "total_runtime_seconds": round(
            total_runtime,
            2,
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "============================================================"
    )
    LOGGER.info(
        "STAGE 05 COMPLETE"
    )
    LOGGER.info(
        "Total runtime: %.2f min",
        total_runtime / 60,
    )
    LOGGER.info(
        "Report: %s",
        REPORT_PATH,
    )
    LOGGER.info(
        "============================================================"
    )


if __name__ == "__main__":
    main()