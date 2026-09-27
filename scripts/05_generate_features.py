"""
STAGE 05 — TRAINING FEATURE GENERATION

Purpose
-------
Create a manageable supervised training dataset from the very large
candidate sets.

Strategy
--------
1. Load candidate pairs from S1->S2 and S1->S3.
2. Load ground truth.
3. Keep ALL true positive candidate pairs.
4. Sample approximately 1 negative for every positive.
5. Join normalized Source 1 / Source 2 / Source 3 attributes.
6. Generate cheap, high-value matching features.
7. Write partitioned Parquet output.

IMPORTANT
---------
Do NOT calculate expensive fuzzy metrics over all 444M candidates.
This stage is deliberately deadline-oriented.

Output
------
artifacts/features/train_features_s2.parquet
artifacts/features/train_features_s3.parquet
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import polars as pl


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ============================================================
# DIRECTORIES
# ============================================================

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
FEATURES_DIR = ARTIFACTS_DIR / "features"


FEATURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

# 1 negative for approximately every positive.
NEGATIVE_TO_POSITIVE_RATIO = 1.0

# Deterministic hash sampling.
#
# Smaller values produce more negatives.
#
# We will use hash % 20 == 0 as the initial negative sample.
#
# This is deliberately conservative because the candidate set
# contains hundreds of millions of rows.
NEGATIVE_HASH_MOD = 20

RANDOM_SEED = 42


# ============================================================
# LOGGING
# ============================================================

LOGGER = logging.getLogger("stage05")


def configure_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Expected file but found:\n{path}"
        )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth() -> pl.LazyFrame:
    """
    Load the challenge ground truth.

    The actual filename used by the project is detected from
    the train data directory.
    """

    possible_paths = [
        ROOT_DIR / "data" / "train_ground_truth.tsv",
        ROOT_DIR / "data" / "train_ground_truth.csv",
        ROOT_DIR / "data" / "train_ground_truth.parquet",
        ROOT_DIR / "artifacts" / "train_ground_truth.parquet",
    ]

    gt_path = None

    for path in possible_paths:

        if path.exists():

            gt_path = path
            break

    if gt_path is None:

        raise FileNotFoundError(
            "Could not locate training ground truth.\n"
            "Expected one of:\n"
            + "\n".join(
                str(p)
                for p in possible_paths
            )
        )

    LOGGER.info(
        "Ground truth: %s",
        gt_path,
    )

    suffix = gt_path.suffix.lower()

    if suffix == ".parquet":

        return pl.scan_parquet(gt_path)

    if suffix == ".tsv":

        return pl.scan_csv(
            gt_path,
            separator="\t",
        )

    if suffix == ".csv":

        return pl.scan_csv(
            gt_path,
        )

    raise ValueError(
        f"Unsupported ground truth format: {suffix}"
    )


# ============================================================
# NORMALIZE GROUND TRUTH
# ============================================================

def prepare_ground_truth(
    gt: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Convert:

        source1_entity_id
        matched_entity_ids

    into one row per true pair.
    """

    columns = gt.collect_schema().names()

    LOGGER.info(
        "Ground truth columns: %s",
        columns,
    )

    if (
        "source1_entity_id" not in columns
        or "matched_entity_ids" not in columns
    ):

        raise ValueError(
            "Ground truth must contain:\n"
            "  source1_entity_id\n"
            "  matched_entity_ids"
        )

    return (
        gt
        .select(
            [
                pl.col(
                    "source1_entity_id"
                ).cast(pl.String),

                pl.col(
                    "matched_entity_ids"
                ),
            ]
        )
        .with_columns(
            pl.col(
                "matched_entity_ids"
            )
            .cast(pl.String)
            .str.split(",")
            .alias(
                "_matched_list"
            )
        )
        .explode(
            "_matched_list"
        )
        .with_columns(
            pl.col(
                "_matched_list"
            )
            .str.strip_chars()
            .alias(
                "candidate_entity_id"
            )
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
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .unique()
    )


# ============================================================
# LOAD NORMALIZED ATTRIBUTES
# ============================================================

def load_source1(
    path: Path,
) -> pl.LazyFrame:

    return (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("entity_id")
                .alias("source1_entity_id"),

                pl.col(
                    "business_name_norm"
                ).alias(
                    "s1_name"
                ),

                pl.col(
                    "business_name_compact"
                ).alias(
                    "s1_name_compact"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                ).alias(
                    "s1_name_sorted"
                ),

                pl.col(
                    "business_address_norm"
                ).alias(
                    "s1_address"
                ),

                pl.col(
                    "business_address_compact"
                ).alias(
                    "s1_address_compact"
                ),

                pl.col(
                    "business_address_house_number"
                ).alias(
                    "s1_house_number"
                ),

                pl.col(
                    "business_address_postal_tokens"
                ).alias(
                    "s1_postal"
                ),

                pl.col(
                    "country_norm"
                ).alias(
                    "s1_country"
                ),
            ]
        )
    )


def load_source2(
    path: Path,
) -> pl.LazyFrame:

    return (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("entity_id")
                .alias("candidate_entity_id"),

                pl.col(
                    "business_name_norm"
                ).alias(
                    "candidate_name"
                ),

                pl.col(
                    "business_name_compact"
                ).alias(
                    "candidate_name_compact"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                ).alias(
                    "candidate_name_sorted"
                ),

                pl.col(
                    "business_address_norm"
                ).alias(
                    "candidate_address"
                ),

                pl.col(
                    "business_address_compact"
                ).alias(
                    "candidate_address_compact"
                ),

                pl.col(
                    "business_address_house_number"
                ).alias(
                    "candidate_house_number"
                ),

                pl.col(
                    "business_address_postal_tokens"
                ).alias(
                    "candidate_postal"
                ),

                pl.col(
                    "country_norm"
                ).alias(
                    "candidate_country"
                ),
            ]
        )
    )


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(
    pairs: pl.LazyFrame,
    source1: pl.LazyFrame,
    source2: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Join source attributes and calculate cheap vectorized features.
    """

    result = (
        pairs

        # ----------------------------------------------------
        # Source 1
        # ----------------------------------------------------

        .join(
            source1,
            on="source1_entity_id",
            how="left",
        )

        # ----------------------------------------------------
        # Candidate source
        # ----------------------------------------------------

        .join(
            source2,
            on="candidate_entity_id",
            how="left",
        )

        # ----------------------------------------------------
        # Exact-match features
        # ----------------------------------------------------

        .with_columns(
            [

                (
                    pl.col("s1_name_compact")
                    ==
                    pl.col("candidate_name_compact")
                )
                .cast(pl.UInt8)
                .alias(
                    "name_exact"
                ),

                (
                    pl.col("s1_address_compact")
                    ==
                    pl.col("candidate_address_compact")
                )
                .cast(pl.UInt8)
                .alias(
                    "address_exact"
                ),

                (
                    pl.col("s1_name_sorted")
                    ==
                    pl.col("candidate_name_sorted")
                )
                .cast(pl.UInt8)
                .alias(
                    "name_sorted_exact"
                ),

                (
                    pl.col("s1_country")
                    ==
                    pl.col("candidate_country")
                )
                .cast(pl.UInt8)
                .alias(
                    "country_exact"
                ),

                (
                    pl.col("s1_house_number")
                    ==
                    pl.col("candidate_house_number")
                )
                .cast(pl.UInt8)
                .alias(
                    "house_number_exact"
                ),

                (
                    pl.col("s1_postal")
                    ==
                    pl.col("candidate_postal")
                )
                .cast(pl.UInt8)
                .alias(
                    "postal_exact"
                ),
            ]
        )

        # ----------------------------------------------------
        # String length features
        # ----------------------------------------------------

        .with_columns(
            [

                pl.col(
                    "s1_name_compact"
                )
                .fill_null("")
                .str.len_chars()
                .cast(pl.UInt16)
                .alias(
                    "s1_name_len"
                ),

                pl.col(
                    "candidate_name_compact"
                )
                .fill_null("")
                .str.len_chars()
                .cast(pl.UInt16)
                .alias(
                    "candidate_name_len"
                ),

                pl.col(
                    "s1_address_compact"
                )
                .fill_null("")
                .str.len_chars()
                .cast(pl.UInt16)
                .alias(
                    "s1_address_len"
                ),

                pl.col(
                    "candidate_address_compact"
                )
                .fill_null("")
                .str.len_chars()
                .cast(pl.UInt16)
                .alias(
                    "candidate_address_len"
                ),
            ]
        )

        # ----------------------------------------------------
        # Length difference
        # ----------------------------------------------------

        .with_columns(
            [

                (
                    pl.col("s1_name_len")
                    -
                    pl.col("candidate_name_len")
                )
                .abs()
                .cast(pl.UInt16)
                .alias(
                    "name_length_diff"
                ),

                (
                    pl.col("s1_address_len")
                    -
                    pl.col("candidate_address_len")
                )
                .abs()
                .cast(pl.UInt16)
                .alias(
                    "address_length_diff"
                ),
            ]
        )

        # ----------------------------------------------------
        # Prefix equality
        # ----------------------------------------------------

        .with_columns(
            [

                (
                    pl.col(
                        "s1_name_compact"
                    )
                    .fill_null("")
                    .str.slice(0, 4)
                    ==
                    pl.col(
                        "candidate_name_compact"
                    )
                    .fill_null("")
                    .str.slice(0, 4)
                )
                .cast(pl.UInt8)
                .alias(
                    "name_prefix4_exact"
                ),

                (
                    pl.col(
                        "s1_name_compact"
                    )
                    .fill_null("")
                    .str.slice(0, 6)
                    ==
                    pl.col(
                        "candidate_name_compact"
                    )
                    .fill_null("")
                    .str.slice(0, 6)
                )
                .cast(pl.UInt8)
                .alias(
                    "name_prefix6_exact"
                ),
            ]
        )

        # ----------------------------------------------------
        # Keep only model-ready features
        # ----------------------------------------------------

        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",

                "candidate_source",

                "block_support_count",

                "blocks",

                "label",

                "name_exact",
                "address_exact",
                "name_sorted_exact",
                "country_exact",
                "house_number_exact",
                "postal_exact",

                "name_prefix4_exact",
                "name_prefix6_exact",

                "s1_name_len",
                "candidate_name_len",

                "s1_address_len",
                "candidate_address_len",

                "name_length_diff",
                "address_length_diff",
            ]
        )
    )

    return result


# ============================================================
# BUILD TRAINING PAIRS
# ============================================================

def build_training_pairs(
    candidate_path: Path,
    source1_path: Path,
    source2_path: Path,
    gt: pl.LazyFrame,
    candidate_source: str,
    output_path: Path,
) -> dict:

    LOGGER.info(
        "=" * 70
    )

    LOGGER.info(
        "Processing candidate source: %s",
        candidate_source,
    )

    LOGGER.info(
        "Candidate file: %s",
        candidate_path,
    )

    # --------------------------------------------------------
    # Candidate scan
    # --------------------------------------------------------

    candidates = (
        pl.scan_parquet(
            candidate_path
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "block_support_count",
                "blocks",
            ]
        )
    )

    # --------------------------------------------------------
    # TRUE POSITIVES
    # --------------------------------------------------------

    positives = (
        candidates
        .join(
            gt,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )
        .with_columns(
            pl.lit(1)
            .cast(pl.UInt8)
            .alias("label")
        )
    )

    positive_count = (
        positives
        .select(pl.len())
        .collect()
        .item()
    )

    LOGGER.info(
        "Positive candidate pairs: %s",
        f"{positive_count:,}",
    )

    # --------------------------------------------------------
    # NEGATIVES
    #
    # Sample candidates that are NOT in GT.
    #
    # Deterministic hashing means repeated runs produce
    # reproducible samples.
    # --------------------------------------------------------

    negative_candidates = (
        candidates
        .join(
            gt,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="anti",
        )
        .with_columns(
            pl.struct(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            )
            .hash(seed=RANDOM_SEED)
            .alias("_hash")
        )
        .filter(
            (
                pl.col("_hash")
                % NEGATIVE_HASH_MOD
            )
            == 0
        )
        .drop("_hash")
        .with_columns(
            pl.lit(0)
            .cast(pl.UInt8)
            .alias("label")
        )
    )

    # --------------------------------------------------------
    # Limit negatives to desired ratio.
    #
    # Sort by deterministic hash first, then head.
    # --------------------------------------------------------

    desired_negative_count = int(
        positive_count
        * NEGATIVE_TO_POSITIVE_RATIO
    )

    negatives = (
        negative_candidates
        .with_columns(
            pl.struct(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            )
            .hash(seed=RANDOM_SEED + 1)
            .alias("_sample_hash")
        )
        .sort("_sample_hash")
        .head(desired_negative_count)
        .drop("_sample_hash")
    )

    negative_count = (
        negatives
        .select(pl.len())
        .collect()
        .item()
    )

    LOGGER.info(
        "Negative training pairs: %s",
        f"{negative_count:,}",
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    training_pairs = pl.concat(
        [
            positives,
            negatives,
        ],
        how="vertical_relaxed",
    )

    total_pairs = (
        training_pairs
        .select(pl.len())
        .collect()
        .item()
    )

    LOGGER.info(
        "Total training pairs: %s",
        f"{total_pairs:,}",
    )

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    source1 = load_source1(
        source1_path
    )

    source2 = load_source2(
        source2_path
    )

    features = create_features(
        training_pairs,
        source1,
        source2,
    )

    # --------------------------------------------------------
    # Collect + write
    # --------------------------------------------------------

    LOGGER.info(
        "Collecting feature dataset..."
    )

    feature_df = features.collect()

    LOGGER.info(
        "Feature rows: %s",
        f"{feature_df.height:,}",
    )

    LOGGER.info(
        "Feature columns: %s",
        feature_df.width,
    )

    feature_df.write_parquet(
        output_path,
        compression="zstd",
    )

    LOGGER.info(
        "Written: %s",
        output_path,
    )

    return {
        "candidate_source": candidate_source,
        "positive_pairs": int(
            positive_count
        ),
        "negative_pairs": int(
            negative_count
        ),
        "total_pairs": int(
            total_pairs
        ),
        "feature_columns": int(
            feature_df.width
        ),
        "output_path": str(
            output_path
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    configure_logging()

    start = time.time()

    LOGGER.info(
        "=" * 80
    )

    LOGGER.info(
        "STAGE 05 — TRAINING FEATURE GENERATION"
    )

    LOGGER.info(
        "=" * 80
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_s1 = (
        NORMALIZED_DIR
        / "train_source1_normalized.parquet"
    )

    train_s2 = (
        NORMALIZED_DIR
        / "train_source2_normalized.parquet"
    )

    train_s3 = (
        NORMALIZED_DIR
        / "train_source3_normalized.parquet"
    )

    candidate_s2 = (
        CANDIDATES_DIR
        / "train_s1_s2_candidates.parquet"
    )

    candidate_s3 = (
        CANDIDATES_DIR
        / "train_s1_s3_candidates.parquet"
    )

    output_s2 = (
        FEATURES_DIR
        / "train_features_s2.parquet"
    )

    output_s3 = (
        FEATURES_DIR
        / "train_features_s3.parquet"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    for path in [
        train_s1,
        train_s2,
        train_s3,
        candidate_s2,
        candidate_s3,
    ]:

        require_file(path)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    LOGGER.info(
        "Loading ground truth..."
    )

    gt = prepare_ground_truth(
        load_ground_truth()
    )

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    result_s2 = build_training_pairs(
        candidate_path=candidate_s2,
        source1_path=train_s1,
        source2_path=train_s2,
        gt=gt,
        candidate_source="S2",
        output_path=output_s2,
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    result_s3 = build_training_pairs(
        candidate_path=candidate_s3,
        source1_path=train_s1,
        source2_path=train_s3,
        gt=gt,
        candidate_source="S3",
        output_path=output_s3,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    elapsed = time.time() - start

    LOGGER.info(
        "=" * 80
    )

    LOGGER.info(
        "STAGE 05 COMPLETE"
    )

    LOGGER.info(
        "Runtime: %.2f minutes",
        elapsed / 60,
    )

    LOGGER.info(
        "S2 features: %s",
        output_s2,
    )

    LOGGER.info(
        "S3 features: %s",
        output_s3,
    )

    LOGGER.info(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()