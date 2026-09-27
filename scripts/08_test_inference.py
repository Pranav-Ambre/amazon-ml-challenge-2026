#!/usr/bin/env python3

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

MODEL_PATH = (
    ROOT / "artifacts/models/lightgbm_pair_model.txt"
)

NORM_DIR = ROOT / "artifacts/normalized"
CAND_DIR = ROOT / "artifacts/candidates"
OUT_DIR = ROOT / "artifacts/inference"
REPORT_DIR = ROOT / "artifacts/reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.80

# AWS instance has 64 GB RAM.
# 1M candidate rows per batch is a reasonable starting point.
BATCH_SIZE = 1_000_000


# ============================================================
# FILES
# ============================================================

S1_NORMALIZED = (
    NORM_DIR / "test_source1_normalized.parquet"
)

NORMALIZED_FILES = {
    "s2": NORM_DIR / "test_source2_normalized.parquet",
    "s3": NORM_DIR / "test_source3_normalized.parquet",
}

CANDIDATE_FILES = {
    "s2": CAND_DIR / "test_s1_s2_candidates.parquet",
    "s3": CAND_DIR / "test_s1_s3_candidates.parquet",
}

OUTPUT_FILES = {
    "s2": OUT_DIR / "test_scored_s2.parquet",
    "s3": OUT_DIR / "test_scored_s3.parquet",
}

REPORT_PATH = (
    REPORT_DIR / "test_inference_report.json"
)


# ============================================================
# MODEL FEATURES
# MUST MATCH STAGE 05 EXACTLY
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


# ============================================================
# TOKEN OVERLAP
# ============================================================

def token_overlap_expr(
    left_column,
    right_column,
    output_name,
):
    """
    Stage 05-compatible token overlap.

    Normalized token columns are stored as space-separated
    strings, not Polars List columns.
    """

    return (
        pl.col(left_column)
        .fill_null("")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.split(" ")
        .list.set_intersection(
            pl.col(right_column)
            .fill_null("")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.split(" ")
        )
        .list.len()
        .cast(pl.Int16)
        .alias(output_name)
    )


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_features(
    joined: pl.DataFrame,
    candidate_source: str,
) -> pl.DataFrame:

    return joined.select(
        [

            # ==================================================
            # IDs
            # ==================================================

            pl.col(
                "source1_entity_id"
            ).cast(pl.Utf8),

            pl.col(
                "candidate_entity_id"
            ).cast(pl.Utf8),

            pl.lit(
                candidate_source
            ).alias("candidate_source"),


            # ==================================================
            # NAME FEATURES
            # ==================================================

            (
                pl.col(
                    "s1_business_name_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_name_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("name_exact"),


            (
                pl.col(
                    "s1_business_name_compact"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_name_compact"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("name_compact_exact"),


            (
                pl.col(
                    "s1_business_name_alnum"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_name_alnum"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("name_alnum_exact"),


            (
                pl.col(
                    "s1_business_name_sorted_tokens"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_name_sorted_tokens"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("name_sorted_exact"),


            # ==================================================
            # ADDRESS FEATURES
            # ==================================================

            (
                pl.col(
                    "s1_business_address_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_address_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("address_exact"),


            (
                pl.col(
                    "s1_business_address_compact"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_address_compact"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("address_compact_exact"),


            (
                pl.col(
                    "s1_business_address_alnum"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_address_alnum"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("address_alnum_exact"),


            (
                pl.col(
                    "s1_business_address_sorted_tokens"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_address_sorted_tokens"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("address_sorted_exact"),


            # ==================================================
            # COUNTRY
            # ==================================================

            (
                pl.col(
                    "s1_country_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_country_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("country_exact"),


            # ==================================================
            # HOUSE NUMBER
            # ==================================================

            (
                pl.col(
                    "s1_business_address_house_number"
                )
                .fill_null("")
                .cast(pl.Utf8)
                ==
                pl.col(
                    "candidate_business_address_house_number"
                )
                .fill_null("")
                .cast(pl.Utf8)
            )
            .cast(pl.Int8)
            .alias("house_number_exact"),


            # ==================================================
            # TOKEN OVERLAPS
            # ==================================================

            token_overlap_expr(
                "s1_business_address_postal_tokens",
                "candidate_business_address_postal_tokens",
                "postal_overlap",
            ),

            token_overlap_expr(
                "s1_business_address_numeric_tokens",
                "candidate_business_address_numeric_tokens",
                "address_numeric_overlap",
            ),

            token_overlap_expr(
                "s1_business_name_tokens",
                "candidate_business_name_tokens",
                "name_token_overlap",
            ),

            token_overlap_expr(
                "s1_business_address_tokens",
                "candidate_business_address_tokens",
                "address_token_overlap",
            ),


            # ==================================================
            # BLOCK SUPPORT
            # ==================================================

            pl.col(
                "block_support_count"
            )
            .cast(pl.Int16)
            .alias(
                "block_support_count_feature"
            ),


            # ==================================================
            # NAME LENGTHS
            #
            # Explicit Int32 conversion BEFORE subtraction.
            # This prevents unsigned integer underflow.
            # ==================================================

            pl.col(
                "s1_business_name_norm"
            )
            .fill_null("")
            .cast(pl.Utf8)
            .str.len_chars()
            .cast(pl.Int32)
            .alias("s1_name_len"),


            pl.col(
                "candidate_business_name_norm"
            )
            .fill_null("")
            .cast(pl.Utf8)
            .str.len_chars()
            .cast(pl.Int32)
            .alias("s2_name_len"),


            # ==================================================
            # ADDRESS LENGTHS
            # ==================================================

            pl.col(
                "s1_business_address_norm"
            )
            .fill_null("")
            .cast(pl.Utf8)
            .str.len_chars()
            .cast(pl.Int32)
            .alias("s1_address_len"),


            pl.col(
                "candidate_business_address_norm"
            )
            .fill_null("")
            .cast(pl.Utf8)
            .str.len_chars()
            .cast(pl.Int32)
            .alias("s2_address_len"),


            # ==================================================
            # NAME LENGTH DIFFERENCE
            # ==================================================

            (
                pl.col(
                    "s1_business_name_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                .str.len_chars()
                .cast(pl.Int32)
                -
                pl.col(
                    "candidate_business_name_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                .str.len_chars()
                .cast(pl.Int32)
            )
            .abs()
            .cast(pl.Int16)
            .alias(
                "name_length_diff"
            ),


            # ==================================================
            # ADDRESS LENGTH DIFFERENCE
            # ==================================================

            (
                pl.col(
                    "s1_business_address_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                .str.len_chars()
                .cast(pl.Int32)
                -
                pl.col(
                    "candidate_business_address_norm"
                )
                .fill_null("")
                .cast(pl.Utf8)
                .str.len_chars()
                .cast(pl.Int32)
            )
            .abs()
            .cast(pl.Int16)
            .alias(
                "address_length_diff"
            ),
        ]
    )


# ============================================================
# LOAD SOURCE 1
# ============================================================

def load_source1():

    print("\nLoading normalized test Source 1...")

    s1 = (
        pl.read_parquet(
            S1_NORMALIZED
        )
        .select(
            [
                pl.col("entity_id")
                .cast(pl.Utf8)
                .alias(
                    "source1_entity_id"
                ),

                pl.col("business_name_norm")
                .cast(pl.Utf8)
                .alias(
                    "s1_business_name_norm"
                ),

                pl.col("business_name_compact")
                .cast(pl.Utf8)
                .alias(
                    "s1_business_name_compact"
                ),

                pl.col("business_name_alnum")
                .cast(pl.Utf8)
                .alias(
                    "s1_business_name_alnum"
                ),

                pl.col("business_name_tokens")
                .cast(pl.Utf8)
                .alias(
                    "s1_business_name_tokens"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_name_sorted_tokens"
                ),

                pl.col("business_address_norm")
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_norm"
                ),

                pl.col(
                    "business_address_compact"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_compact"
                ),

                pl.col(
                    "business_address_alnum"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_alnum"
                ),

                pl.col(
                    "business_address_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_tokens"
                ),

                pl.col(
                    "business_address_sorted_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_sorted_tokens"
                ),

                pl.col(
                    "business_address_house_number"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_house_number"
                ),

                pl.col(
                    "business_address_numeric_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_numeric_tokens"
                ),

                pl.col(
                    "business_address_postal_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "s1_business_address_postal_tokens"
                ),

                pl.col("country_norm")
                .cast(pl.Utf8)
                .alias(
                    "s1_country_norm"
                ),
            ]
        )
    )

    print(
        f"Source 1 rows: {s1.height:,}"
    )

    return s1


# ============================================================
# LOAD TARGET SOURCE
# ============================================================

def load_target_source(
    source_name,
):

    path = NORMALIZED_FILES[
        source_name
    ]

    print(
        f"\nLoading normalized "
        f"{source_name.upper()}..."
    )

    target = (
        pl.read_parquet(path)
        .select(
            [
                pl.col("entity_id")
                .cast(pl.Utf8)
                .alias(
                    "candidate_entity_id"
                ),

                pl.col("business_name_norm")
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_name_norm"
                ),

                pl.col("business_name_compact")
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_name_compact"
                ),

                pl.col("business_name_alnum")
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_name_alnum"
                ),

                pl.col("business_name_tokens")
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_name_tokens"
                ),

                pl.col(
                    "business_name_sorted_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_name_sorted_tokens"
                ),

                pl.col("business_address_norm")
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_norm"
                ),

                pl.col(
                    "business_address_compact"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_compact"
                ),

                pl.col(
                    "business_address_alnum"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_alnum"
                ),

                pl.col(
                    "business_address_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_tokens"
                ),

                pl.col(
                    "business_address_sorted_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_sorted_tokens"
                ),

                pl.col(
                    "business_address_house_number"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_house_number"
                ),

                pl.col(
                    "business_address_numeric_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_numeric_tokens"
                ),

                pl.col(
                    "business_address_postal_tokens"
                )
                .cast(pl.Utf8)
                .alias(
                    "candidate_business_address_postal_tokens"
                ),

                pl.col("country_norm")
                .cast(pl.Utf8)
                .alias(
                    "candidate_country_norm"
                ),
            ]
        )
    )

    print(
        f"{source_name.upper()} rows: "
        f"{target.height:,}"
    )

    return target


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    source_name,
    source1,
    model,
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING TEST "
        f"{source_name.upper()}"
    )
    print("=" * 70)

    candidate_path = (
        CANDIDATE_FILES[
            source_name
        ]
    )

    output_path = (
        OUTPUT_FILES[
            source_name
        ]
    )

    # --------------------------------------------------------
    # Target source
    # --------------------------------------------------------

    target = load_target_source(
        source_name
    )

    # --------------------------------------------------------
    # Candidate parquet
    # --------------------------------------------------------

    print(
        "\nOpening candidate parquet..."
    )

    parquet_file = pq.ParquetFile(
        candidate_path
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        f"Candidate rows: "
        f"{total_rows:,}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE:,}"
    )

    # --------------------------------------------------------
    # Delete old output
    # --------------------------------------------------------

    if output_path.exists():
        print(
            f"\nRemoving old output: "
            f"{output_path}"
        )
        output_path.unlink()

    writer = None

    total_processed = 0
    total_predicted = 0
    batch_number = 0

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for arrow_batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):

        batch_number += 1

        candidates = (
            pl.from_arrow(
                arrow_batch
            )
        )

        candidates = candidates.select(
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
                    "candidate_source"
                )
                .cast(pl.Utf8),

                pl.col(
                    "block_support_count"
                )
                .cast(pl.UInt32),

                pl.col("blocks"),
            ]
        )

        # ----------------------------------------------------
        # Join Source 1
        # ----------------------------------------------------

        joined = candidates.join(
            source1,
            on="source1_entity_id",
            how="inner",
        )

        # ----------------------------------------------------
        # Join target source
        # ----------------------------------------------------

        joined = joined.join(
            target,
            on="candidate_entity_id",
            how="inner",
        )

        # ----------------------------------------------------
        # Feature generation
        # ----------------------------------------------------

        feature_df = make_features(
            joined,
            source_name,
        )

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        if batch_number == 1:

            missing_features = [
                col
                for col in FEATURE_COLUMNS
                if col not in feature_df.columns
            ]

            if missing_features:
                raise RuntimeError(
                    "Missing feature columns: "
                    + str(
                        missing_features
                    )
                )

            print(
                "\nFeature schema verified."
            )

        # ----------------------------------------------------
        # Model matrix
        # ----------------------------------------------------

        X = (
            feature_df
            .select(
                FEATURE_COLUMNS
            )
            .to_numpy()
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        probabilities = (
            model.predict(X)
        )

        # ----------------------------------------------------
        # Apply threshold
        # ----------------------------------------------------

        keep = (
            probabilities
            >= THRESHOLD
        )

        if np.any(keep):

            positive = (
                feature_df
                .filter(
                    pl.Series(
                        "keep",
                        keep,
                    )
                )
                .select(
                    [
                        "source1_entity_id",
                        "candidate_entity_id",
                        "candidate_source",
                    ]
                )
            )

            positive = positive.with_columns(
                pl.Series(
                    "match_probability",
                    probabilities[keep],
                )
                .cast(pl.Float32)
            )

            # ----------------------------------------------
            # Write only positive matches
            # ----------------------------------------------

            arrow_table = (
                positive.to_arrow()
            )

            if writer is None:

                writer = (
                    pq.ParquetWriter(
                        output_path,
                        arrow_table.schema,
                        compression="zstd",
                    )
                )

            writer.write_table(
                arrow_table
            )

            total_predicted += (
                positive.height
            )

        total_processed += (
            candidates.height
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_number == 1
            or batch_number % 5 == 0
            or total_processed >= total_rows
        ):

            percentage = (
                total_processed
                / total_rows
                * 100
            )

            print(
                f"[{source_name}] "
                f"batch={batch_number:,} | "
                f"processed="
                f"{total_processed:,}/"
                f"{total_rows:,} "
                f"({percentage:.1f}%) | "
                f"predicted="
                f"{total_predicted:,}"
            )

    # --------------------------------------------------------
    # Close writer
    # --------------------------------------------------------

    if writer is not None:
        writer.close()

    print(
        f"\n{source_name.upper()} COMPLETE"
    )

    print(
        f"Candidates processed: "
        f"{total_processed:,}"
    )

    print(
        f"Predicted matches: "
        f"{total_predicted:,}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )

    return {
        "source": source_name,
        "candidate_rows": int(
            total_rows
        ),
        "processed_rows": int(
            total_processed
        ),
        "predicted_matches": int(
            total_predicted
        ),
        "output_file": str(
            output_path
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("STAGE 08 - TEST INFERENCE")
    print("=" * 70)

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Batch size: {BATCH_SIZE:,}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        S1_NORMALIZED,
        NORMALIZED_FILES["s2"],
        NORMALIZED_FILES["s3"],
        CANDIDATE_FILES["s2"],
        CANDIDATE_FILES["s3"],
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: "
                f"{path}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading LightGBM model..."
    )

    model = lgb.Booster(
        model_file=str(
            MODEL_PATH
        )
    )

    print(
        f"Model trees: "
        f"{model.num_trees():,}"
    )

    # --------------------------------------------------------
    # Load S1 once
    # --------------------------------------------------------

    source1 = load_source1()

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    result_s2 = process_source(
        "s2",
        source1,
        model,
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    result_s3 = process_source(
        "s3",
        source1,
        model,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    runtime = (
        time.time()
        - start_time
    )

    report = {
        "stage": "08_test_inference",
        "status": "complete",
        "threshold": THRESHOLD,
        "batch_size": BATCH_SIZE,
        "model": str(
            MODEL_PATH
        ),
        "results": [
            result_s2,
            result_s3,
        ],
        "total_predicted_matches": (
            result_s2[
                "predicted_matches"
            ]
            +
            result_s3[
                "predicted_matches"
            ]
        ),
        "total_runtime_seconds": runtime,
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

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 08 COMPLETE")
    print("=" * 70)

    print(
        f"S2 predicted matches: "
        f"{result_s2['predicted_matches']:,}"
    )

    print(
        f"S3 predicted matches: "
        f"{result_s3['predicted_matches']:,}"
    )

    print(
        f"Total predicted matches: "
        f"{report['total_predicted_matches']:,}"
    )

    print(
        f"Runtime: "
        f"{runtime:.2f} sec "
        f"({runtime / 60:.2f} min)"
    )

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()