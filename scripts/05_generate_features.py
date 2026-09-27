#!/usr/bin/env python3

"""
Stage 05 - Training Feature Generation
Amazon ML Challenge 2026 - Business Entity Resolution

Strategy
--------
1. Load official training ground truth.
2. Expand comma-separated matched_entity_ids.
3. Retrieve all ground-truth pairs present in candidate files.
4. Sample approximately 1 negative per positive.
5. Join normalized Source 1 + Source 2/3 data.
6. Generate lightweight deterministic pairwise features.
7. Write:
       artifacts/features/train_features_s2.parquet
       artifacts/features/train_features_s3.parquet

Important
---------
We DO NOT generate features for all 444M+ candidates.
Only retrieved positives + sampled negatives are feature-engineered.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import polars as pl

from business_entity_resolution.src.config import ARTIFACTS_DIR
from business_entity_resolution.src.data_loader import (
    load_train_ground_truth,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

NEGATIVE_TO_POSITIVE_RATIO = 1.0

MAX_NEGATIVES = 8_000_000


# =============================================================================
# DIRECTORIES
# =============================================================================

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
FEATURE_DIR = ARTIFACTS_DIR / "features"
REPORT_DIR = ARTIFACTS_DIR / "reports"


FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# NORMALIZED FILES
# =============================================================================

TRAIN_S1 = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

TRAIN_S2 = (
    NORMALIZED_DIR
    / "train_source2_normalized.parquet"
)

TRAIN_S3 = (
    NORMALIZED_DIR
    / "train_source3_normalized.parquet"
)


# =============================================================================
# CANDIDATE FILES
# =============================================================================

CAND_S2 = (
    CANDIDATES_DIR
    / "train_s1_s2_candidates.parquet"
)

CAND_S3 = (
    CANDIDATES_DIR
    / "train_s1_s3_candidates.parquet"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

OUT_S2 = (
    FEATURE_DIR
    / "train_features_s2.parquet"
)

OUT_S3 = (
    FEATURE_DIR
    / "train_features_s3.parquet"
)


REPORT_PATH = (
    REPORT_DIR
    / "feature_generation_report.json"
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

LOGGER = logging.getLogger(
    "stage05"
)


# =============================================================================
# PATH VALIDATION
# =============================================================================

def check_paths() -> None:

    required_paths = [
        TRAIN_S1,
        TRAIN_S2,
        TRAIN_S3,
        CAND_S2,
        CAND_S3,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        LOGGER.info(
            "Found: %s",
            path,
        )


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_ground_truth_pairs() -> tuple[
    pl.DataFrame,
    int,
]:
    """
    Load official training ground truth and convert:

        source1_entity_id | matched_entity_ids

    into:

        source1_entity_id | candidate_entity_id

    One row per true pair.
    """

    LOGGER.info(
        "Loading training ground truth..."
    )

    ground_truth = (
        load_train_ground_truth()
    )

    gt = pl.from_pandas(
        ground_truth
    )

    LOGGER.info(
        "Ground truth rows: %d",
        gt.height,
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    gt = gt.select(
        [
            pl.col(
                "source1_entity_id"
            )
            .cast(pl.Utf8)
            .alias(
                "source1_entity_id"
            ),

            pl.col(
                "matched_entity_ids"
            )
            .cast(pl.Utf8)
            .alias(
                "matched_entity_ids"
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Zero-match entities
    # -------------------------------------------------------------------------

    zero_match_count = (
        gt
        .filter(
            pl.col(
                "matched_entity_ids"
            ).is_null()
            |
            (
                pl.col(
                    "matched_entity_ids"
                )
                .str.strip_chars()
                == ""
            )
        )
        .height
    )

    LOGGER.info(
        "Zero-match Source-1 entities: %d",
        zero_match_count,
    )

    # -------------------------------------------------------------------------
    # Expand matched_entity_ids
    # -------------------------------------------------------------------------

    pairs = (
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
            .alias(
                "matched_entity_id_list"
            )
        )

        .explode(
            "matched_entity_id_list"
        )

        .with_columns(
            pl.col(
                "matched_entity_id_list"
            )
            .str.strip_chars()
            .cast(pl.Utf8)
            .alias(
                "candidate_entity_id"
            )
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
                )
                != ""
            )
        )

        .unique()
    )

    LOGGER.info(
        "True matched pairs: %d",
        pairs.height,
    )

    return (
        pairs,
        zero_match_count,
    )


# =============================================================================
# POSITIVE RETRIEVAL
# =============================================================================

def retrieve_positive_pairs(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
) -> pl.DataFrame:
    """
    Keep only GT pairs that exist in the candidate set.

    Blocking recall determines the maximum possible positive count.
    """

    LOGGER.info(
        "Retrieving positives from %s",
        candidate_path.name,
    )

    candidates = (
        pl.scan_parquet(
            candidate_path
        )

        .select(
            [
                pl.col(
                    "source1_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "candidate_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "block_support_count"
                ),
            ]
        )
    )

    positives = (
        gt_pairs
        .lazy()

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
        "Retrieved positive candidates: %d",
        positives.height,
    )

    return positives


# =============================================================================
# NEGATIVE SAMPLING
# =============================================================================

def sample_negatives(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
    positive_count: int,
) -> pl.DataFrame:
    """
    Sample deterministic negatives from the candidate set.

    All known ground-truth pairs are removed first.

    Approximately:
        negatives = positives * NEGATIVE_TO_POSITIVE_RATIO
    """

    target = min(
        int(
            positive_count
            * NEGATIVE_TO_POSITIVE_RATIO
        ),
        MAX_NEGATIVES,
    )

    LOGGER.info(
        "Sampling %d negatives from %s",
        target,
        candidate_path.name,
    )

    # -------------------------------------------------------------------------
    # Ground truth lookup
    # -------------------------------------------------------------------------

    gt_lazy = (
        gt_pairs.lazy()
    )

    # -------------------------------------------------------------------------
    # Candidate scan + anti join
    # -------------------------------------------------------------------------

    candidates = (
        pl.scan_parquet(
            candidate_path
        )

        .select(
            [
                pl.col(
                    "source1_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "candidate_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "block_support_count"
                ),
            ]
        )

        .unique(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

        # Remove true GT pairs.
        .join(
            gt_lazy,

            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],

            how="anti",
        )

        # Deterministic hash.
        .with_columns(
            pl.concat_str(
                [
                    pl.col(
                        "source1_entity_id"
                    ),

                    pl.col(
                        "candidate_entity_id"
                    ),
                ],

                separator="|",
            )
            .hash(
                seed=SEED
            )
            .alias(
                "_sample_hash"
            )
        )

        .sort(
            "_sample_hash"
        )

        .head(
            target
        )

        .drop(
            "_sample_hash"
        )
    )

    negatives = candidates.collect(
        engine="streaming"
    )

    LOGGER.info(
        "Selected negatives: %d",
        negatives.height,
    )

    return negatives


# =============================================================================
# TOKEN OVERLAP HELPER
# =============================================================================

def token_overlap_expr(
    left_column: str,
    right_column: str,
    output_name: str,
) -> pl.Expr:
    """
    Calculate token intersection for columns stored as STRING.

    Example:

        "abc xyz pvt"

    becomes:

        ["abc", "xyz", "pvt"]

    before list intersection.

    The normalized token columns in this project are strings,
    not native Polars List columns.
    """

    return (
        pl.col(
            left_column
        )
        .fill_null("")
        .str.strip_chars()
        .str.split(" ")

        .list.set_intersection(

            pl.col(
                right_column
            )
            .fill_null("")
            .str.strip_chars()
            .str.split(" ")
        )

        .list.len()

        .cast(pl.Int16)

        .alias(
            output_name
        )
    )


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def add_features(
    pairs: pl.LazyFrame,
    s1_path: Path,
    source_path: Path,
    source_name: str,
) -> pl.LazyFrame:

    LOGGER.info(
        "Preparing features for %s",
        source_name.upper(),
    )

    # =========================================================================
    # SOURCE 1
    # =========================================================================

    s1 = (
        pl.scan_parquet(
            s1_path
        )

        .select(
            [
                pl.col(
                    "entity_id"
                )
                .cast(pl.Utf8)
                .alias(
                    "source1_entity_id"
                ),

                pl.col(
                    "business_name_norm"
                )
                .alias(
                    "s1_name_norm"
                ),

                pl.col(
                    "business_name_compact"
                )
                .alias(
                    "s1_name_compact"
                ),

                pl.col(
                    "business_name_alnum"
                )
                .alias(
                    "s1_name_alnum"
                ),

                pl.col(
                    "business_name_tokens"
                )
                .alias(
                    "s1_name_tokens"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                )
                .alias(
                    "s1_name_sorted_tokens"
                ),

                pl.col(
                    "business_name_numeric_tokens"
                )
                .alias(
                    "s1_name_numeric_tokens"
                ),

                pl.col(
                    "business_address_norm"
                )
                .alias(
                    "s1_address_norm"
                ),

                pl.col(
                    "business_address_compact"
                )
                .alias(
                    "s1_address_compact"
                ),

                pl.col(
                    "business_address_alnum"
                )
                .alias(
                    "s1_address_alnum"
                ),

                pl.col(
                    "business_address_tokens"
                )
                .alias(
                    "s1_address_tokens"
                ),

                pl.col(
                    "business_address_sorted_tokens"
                )
                .alias(
                    "s1_address_sorted_tokens"
                ),

                pl.col(
                    "business_address_alpha_tokens"
                )
                .alias(
                    "s1_address_alpha_tokens"
                ),

                pl.col(
                    "business_address_numeric_tokens"
                )
                .alias(
                    "s1_address_numeric_tokens"
                ),

                pl.col(
                    "business_address_house_number"
                )
                .alias(
                    "s1_house_number"
                ),

                pl.col(
                    "business_address_postal_tokens"
                )
                .alias(
                    "s1_postal_tokens"
                ),

                pl.col(
                    "country_norm"
                )
                .alias(
                    "s1_country_norm"
                ),
            ]
        )
    )

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    candidate_path = (
        CAND_S2
        if source_name == "s2"
        else CAND_S3
    )

    candidates = (
        pl.scan_parquet(
            candidate_path
        )

        .select(
            [
                pl.col(
                    "source1_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "candidate_entity_id"
                )
                .cast(pl.Utf8),

                pl.col(
                    "block_support_count"
                ),
            ]
        )
    )

    # =========================================================================
    # SOURCE 2 / SOURCE 3
    # =========================================================================

    source = (
        pl.scan_parquet(
            source_path
        )

        .select(
            [
                pl.col(
                    "entity_id"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_entity_id"
                ),

                pl.col(
                    "business_name_norm"
                )
                .alias(
                    "s2_name_norm"
                ),

                pl.col(
                    "business_name_compact"
                )
                .alias(
                    "s2_name_compact"
                ),

                pl.col(
                    "business_name_alnum"
                )
                .alias(
                    "s2_name_alnum"
                ),

                pl.col(
                    "business_name_tokens"
                )
                .alias(
                    "s2_name_tokens"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                )
                .alias(
                    "s2_name_sorted_tokens"
                ),

                pl.col(
                    "business_name_numeric_tokens"
                )
                .alias(
                    "s2_name_numeric_tokens"
                ),

                pl.col(
                    "business_address_norm"
                )
                .alias(
                    "s2_address_norm"
                ),

                pl.col(
                    "business_address_compact"
                )
                .alias(
                    "s2_address_compact"
                ),

                pl.col(
                    "business_address_alnum"
                )
                .alias(
                    "s2_address_alnum"
                ),

                pl.col(
                    "business_address_tokens"
                )
                .alias(
                    "s2_address_tokens"
                ),

                pl.col(
                    "business_address_sorted_tokens"
                )
                .alias(
                    "s2_address_sorted_tokens"
                ),

                pl.col(
                    "business_address_alpha_tokens"
                )
                .alias(
                    "s2_address_alpha_tokens"
                ),

                pl.col(
                    "business_address_numeric_tokens"
                )
                .alias(
                    "s2_address_numeric_tokens"
                ),

                pl.col(
                    "business_address_house_number"
                )
                .alias(
                    "s2_house_number"
                ),

                pl.col(
                    "business_address_postal_tokens"
                )
                .alias(
                    "s2_postal_tokens"
                ),

                pl.col(
                    "country_norm"
                )
                .alias(
                    "s2_country_norm"
                ),
            ]
        )
    )

    # =========================================================================
    # JOIN
    # =========================================================================

    joined = (
        pairs

        .join(
            candidates,

            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],

            how="inner",
        )

        .join(
            s1,

            on=[
                "source1_entity_id",
            ],

            how="left",
        )

        .join(
            source,

            on=[
                "candidate_entity_id",
            ],

            how="left",
        )
    )

    # =========================================================================
    # EXACT FEATURES
    # =========================================================================

    joined = joined.with_columns(

        (
            pl.col(
                "s1_name_norm"
            )
            ==
            pl.col(
                "s2_name_norm"
            )
        )
        .cast(pl.Int8)
        .alias(
            "name_exact"
        ),

        (
            pl.col(
                "s1_name_compact"
            )
            ==
            pl.col(
                "s2_name_compact"
            )
        )
        .cast(pl.Int8)
        .alias(
            "name_compact_exact"
        ),

        (
            pl.col(
                "s1_name_alnum"
            )
            ==
            pl.col(
                "s2_name_alnum"
            )
        )
        .cast(pl.Int8)
        .alias(
            "name_alnum_exact"
        ),

        (
            pl.col(
                "s1_name_sorted_tokens"
            )
            ==
            pl.col(
                "s2_name_sorted_tokens"
            )
        )
        .cast(pl.Int8)
        .alias(
            "name_sorted_exact"
        ),

        (
            pl.col(
                "s1_address_norm"
            )
            ==
            pl.col(
                "s2_address_norm"
            )
        )
        .cast(pl.Int8)
        .alias(
            "address_exact"
        ),

        (
            pl.col(
                "s1_address_compact"
            )
            ==
            pl.col(
                "s2_address_compact"
            )
        )
        .cast(pl.Int8)
        .alias(
            "address_compact_exact"
        ),

        (
            pl.col(
                "s1_address_alnum"
            )
            ==
            pl.col(
                "s2_address_alnum"
            )
        )
        .cast(pl.Int8)
        .alias(
            "address_alnum_exact"
        ),

        (
            pl.col(
                "s1_address_sorted_tokens"
            )
            ==
            pl.col(
                "s2_address_sorted_tokens"
            )
        )
        .cast(pl.Int8)
        .alias(
            "address_sorted_exact"
        ),

        (
            pl.col(
                "s1_country_norm"
            )
            ==
            pl.col(
                "s2_country_norm"
            )
        )
        .cast(pl.Int8)
        .alias(
            "country_exact"
        ),

        (
            pl.col(
                "s1_house_number"
            )
            ==
            pl.col(
                "s2_house_number"
            )
        )
        .cast(pl.Int8)
        .alias(
            "house_number_exact"
        ),

        # ---------------------------------------------------------------------
        # Token overlaps
        # ---------------------------------------------------------------------

        token_overlap_expr(
            "s1_postal_tokens",
            "s2_postal_tokens",
            "postal_overlap",
        ),

        token_overlap_expr(
            "s1_address_numeric_tokens",
            "s2_address_numeric_tokens",
            "address_numeric_overlap",
        ),

        token_overlap_expr(
            "s1_name_tokens",
            "s2_name_tokens",
            "name_token_overlap",
        ),

        token_overlap_expr(
            "s1_address_tokens",
            "s2_address_tokens",
            "address_token_overlap",
        ),

        # ---------------------------------------------------------------------
        # Blocking support
        # ---------------------------------------------------------------------

        pl.col(
            "block_support_count"
        )
        .cast(pl.Int16)
        .alias(
            "block_support_count_feature"
        ),
    )

    # =========================================================================
    # LENGTH FEATURES
    # =========================================================================

    joined = joined.with_columns(

        pl.col(
            "s1_name_norm"
        )
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias(
            "s1_name_len"
        ),

        pl.col(
            "s2_name_norm"
        )
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias(
            "s2_name_len"
        ),

        pl.col(
            "s1_address_norm"
        )
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias(
            "s1_address_len"
        ),

        pl.col(
            "s2_address_norm"
        )
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int16)
        .alias(
            "s2_address_len"
        ),
    )

    # =========================================================================
    # LENGTH DIFFERENCE
    # =========================================================================

    joined = joined.with_columns(

        (
            pl.col(
                "s1_name_len"
            )
            -
            pl.col(
                "s2_name_len"
            )
        )
        .abs()
        .cast(pl.Int16)
        .alias(
            "name_length_diff"
        ),

        (
            pl.col(
                "s1_address_len"
            )
            -
            pl.col(
                "s2_address_len"
            )
        )
        .abs()
        .cast(pl.Int16)
        .alias(
            "address_length_diff"
        ),
    )

    return joined


# =============================================================================
# BUILD DATASET
# =============================================================================

def build_dataset(
    gt_pairs: pl.DataFrame,
    candidate_path: Path,
    s1_path: Path,
    source_path: Path,
    source_name: str,
) -> tuple[
    Path,
    dict,
]:

    start = time.time()

    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "Building %s training features",
        source_name.upper(),
    )

    # =========================================================================
    # POSITIVES
    # =========================================================================

    positives = retrieve_positive_pairs(
        gt_pairs=gt_pairs,
        candidate_path=candidate_path,
    )

    positive_count = (
        positives.height
    )

    # =========================================================================
    # NEGATIVES
    # =========================================================================

    negatives = sample_negatives(
        gt_pairs=gt_pairs,
        candidate_path=candidate_path,
        positive_count=positive_count,
    )

    # =========================================================================
    # LABELS
    # =========================================================================

    positives = positives.with_columns(
        pl.lit(
            1,
            dtype=pl.Int8,
        )
        .alias(
            "label"
        )
    )

    negatives = negatives.with_columns(
        pl.lit(
            0,
            dtype=pl.Int8,
        )
        .alias(
            "label"
        )
    )

    # =========================================================================
    # COMBINE
    # =========================================================================

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
        "Labelled pairs: %d",
        pairs.height,
    )

    LOGGER.info(
        "Positive: %d | Negative: %d",
        positives.height,
        negatives.height,
    )

    # =========================================================================
    # FEATURES
    # =========================================================================

    feature_lf = add_features(
        pairs=pairs.lazy(),
        s1_path=s1_path,
        source_path=source_path,
        source_name=source_name,
    )

    # =========================================================================
    # FINAL MODEL FEATURES
    # =========================================================================

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

    # =========================================================================
    # COLLECT
    # =========================================================================

    LOGGER.info(
        "Collecting final %s feature dataset...",
        source_name.upper(),
    )

    features = feature_lf.collect(
        engine="streaming"
    )

    LOGGER.info(
        "Feature collection complete: %d rows",
        features.height,
    )

    # =========================================================================
    # NULL HANDLING
    # =========================================================================

    numeric_columns = [
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
            pl.col(
                column
            )
            .fill_null(0)
            .cast(pl.Int32)

            for column in numeric_columns
        ]
    )

    # =========================================================================
    # DETERMINISTIC SHUFFLE
    # =========================================================================

    features = (
        features

        .with_columns(
            pl.int_range(
                pl.len()
            )
            .shuffle(
                seed=SEED
            )
            .alias(
                "_shuffle"
            )
        )

        .sort(
            "_shuffle"
        )

        .drop(
            "_shuffle"
        )
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    output_path = (
        OUT_S2
        if source_name == "s2"
        else OUT_S3
    )

    LOGGER.info(
        "Writing %s",
        output_path,
    )

    features.write_parquet(
        output_path,
        compression="zstd",
        compression_level=3,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    elapsed = (
        time.time()
        - start
    )

    positive_rows = (
        features
        .filter(
            pl.col("label") == 1
        )
        .height
    )

    negative_rows = (
        features
        .filter(
            pl.col("label") == 0
        )
        .height
    )

    report = {
        "source": source_name,

        "candidate_file": str(
            candidate_path
        ),

        "output_file": str(
            output_path
        ),

        "positive_pairs_retrieved": int(
            positive_count
        ),

        "negative_pairs_selected": int(
            negatives.height
        ),

        "total_training_rows": int(
            features.height
        ),

        "positive_rows": int(
            positive_rows
        ),

        "negative_rows": int(
            negative_rows
        ),

        "runtime_seconds": round(
            elapsed,
            2,
        ),
    }

    LOGGER.info(
        "%s complete: %d rows | %.2f minutes",
        source_name.upper(),
        features.height,
        elapsed / 60,
    )

    return (
        output_path,
        report,
    )


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

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    check_paths()

    # =========================================================================
    # GROUND TRUTH
    # =========================================================================

    (
        gt_pairs,
        zero_match_count,
    ) = load_ground_truth_pairs()

    # =========================================================================
    # SOURCE 2
    # =========================================================================

    (
        _,
        report_s2,
    ) = build_dataset(
        gt_pairs=gt_pairs,
        candidate_path=CAND_S2,
        s1_path=TRAIN_S1,
        source_path=TRAIN_S2,
        source_name="s2",
    )

    # =========================================================================
    # SOURCE 3
    # =========================================================================

    (
        _,
        report_s3,
    ) = build_dataset(
        gt_pairs=gt_pairs,
        candidate_path=CAND_S3,
        s1_path=TRAIN_S1,
        source_path=TRAIN_S3,
        source_name="s3",
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    total_runtime = (
        time.time()
        - overall_start
    )

    report = {
        "stage": "05_feature_generation",

        "status": "complete",

        "seed": SEED,

        "negative_to_positive_ratio": (
            NEGATIVE_TO_POSITIVE_RATIO
        ),

        "max_negatives": MAX_NEGATIVES,

        "zero_match_entities": int(
            zero_match_count
        ),

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
        "Total runtime: %.2f minutes",
        total_runtime / 60,
    )

    LOGGER.info(
        "Report: %s",
        REPORT_PATH,
    )

    LOGGER.info(
        "============================================================"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()