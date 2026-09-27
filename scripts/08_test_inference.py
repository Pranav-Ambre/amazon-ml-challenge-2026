#!/usr/bin/env python3

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
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

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLD = 0.80

# Keep this reasonably small.
# 1M rows is a good starting point for 64 GB RAM.
BATCH_SIZE = 1_000_000


# ============================================================
# NORMALIZED FILES
# ============================================================

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


# ============================================================
# FEATURES
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
    return (
        pl.col(left_column)
        .fill_null("")
        .str.strip_chars()
        .str.split(" ")
        .list.set_intersection(
            pl.col(right_column)
            .fill_null("")
            .str.strip_chars()
            .str.split(" ")
        )
        .list.len()
        .cast(pl.Int16)
        .alias(output_name)
    )


# ============================================================
# FEATURE EXPRESSION
# ============================================================

def make_features(
    joined,
    candidate_source,
):
    return joined.select(
        [
            # IDs
            pl.col("source1_entity_id"),
            pl.col("candidate_entity_id"),

            # Keep source
            pl.lit(candidate_source)
            .alias("candidate_source"),

            # Exact name
            (
                pl.col("s1_business_name_norm")
                == pl.col("candidate_business_name_norm")
            )
            .cast(pl.Int8)
            .alias("name_exact"),

            (
                pl.col("s1_business_name_compact")
                == pl.col("candidate_business_name_compact")
            )
            .cast(pl.Int8)
            .alias("name_compact_exact"),

            (
                pl.col("s1_business_name_alnum")
                == pl.col("candidate_business_name_alnum")
            )
            .cast(pl.Int8)
            .alias("name_alnum_exact"),

            (
                pl.col("s1_business_name_sorted_tokens")
                == pl.col("candidate_business_name_sorted_tokens")
            )
            .cast(pl.Int8)
            .alias("name_sorted_exact"),

            # Exact address
            (
                pl.col("s1_business_address_norm")
                == pl.col("candidate_business_address_norm")
            )
            .cast(pl.Int8)
            .alias("address_exact"),

            (
                pl.col("s1_business_address_compact")
                == pl.col("candidate_business_address_compact")
            )
            .cast(pl.Int8)
            .alias("address_compact_exact"),

            (
                pl.col("s1_business_address_alnum")
                == pl.col("candidate_business_address_alnum")
            )
            .cast(pl.Int8)
            .alias("address_alnum_exact"),

            (
                pl.col("s1_business_address_sorted_tokens")
                == pl.col("candidate_business_address_sorted_tokens")
            )
            .cast(pl.Int8)
            .alias("address_sorted_exact"),

            # Country
            (
                pl.col("s1_country_norm")
                == pl.col("candidate_country_norm")
            )
            .cast(pl.Int8)
            .alias("country_exact"),

            # House number
            (
                pl.col("s1_business_address_house_number")
                == pl.col("candidate_business_address_house_number")
            )
            .cast(pl.Int8)
            .alias("house_number_exact"),

            # Token overlap
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

            # Blocking support
            pl.col("block_support_count")
            .cast(pl.Int16)
            .alias("block_support_count_feature"),

            # Lengths
            pl.col("s1_business_name_norm")
            .fill_null("")
            .str.len_chars()
            .cast(pl.Int16)
            .alias("s1_name_len"),

            pl.col("candidate_business_name_norm")
            .fill_null("")
            .str.len_chars()
            .cast(pl.Int16)
            .alias("s2_name_len"),

            pl.col("s1_business_address_norm")
            .fill_null("")
            .str.len_chars()
            .cast(pl.Int16)
            .alias("s1_address_len"),

            pl.col("candidate_business_address_norm")
            .fill_null("")
            .str.len_chars()
            .cast(pl.Int16)
            .alias("s2_address_len"),

            (
                pl.col("s1_business_name_norm")
                .fill_null("")
                .str.len_chars()
                -
                pl.col("candidate_business_name_norm")
                .fill_null("")
                .str.len_chars()
            )
            .abs()
            .cast(pl.Int16)
            .alias("name_length_diff"),

            (
                pl.col("s1_business_address_norm")
                .fill_null("")
                .str.len_chars()
                -
                pl.col("candidate_business_address_norm")
                .fill_null("")
                .str.len_chars()
            )
            .abs()
            .cast(pl.Int16)
            .alias("address_length_diff"),
        ]
    )


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 70)
print("STAGE 08 - TEST INFERENCE")
print("=" * 70)

start_time = time.time()

print("\nLoading normalized Source 1...")

s1_path = NORM_DIR / "test_source1_normalized.parquet"

s1 = (
    pl.read_parquet(s1_path)
    .select(
        [
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("business_name")
            .alias("s1_business_name"),

            pl.col("business_name_norm")
            .alias("s1_business_name_norm"),

            pl.col("business_name_compact")
            .alias("s1_business_name_compact"),

            pl.col("business_name_alnum")
            .alias("s1_business_name_alnum"),

            pl.col("business_name_tokens")
            .alias("s1_business_name_tokens"),

            pl.col("business_name_sorted_tokens")
            .alias("s1_business_name_sorted_tokens"),

            pl.col("business_address_norm")
            .alias("s1_business_address_norm"),

            pl.col("business_address_compact")
            .alias("s1_business_address_compact"),

            pl.col("business_address_alnum")
            .alias("s1_business_address_alnum"),

            pl.col("business_address_tokens")
            .alias("s1_business_address_tokens"),

            pl.col("business_address_sorted_tokens")
            .alias("s1_business_address_sorted_tokens"),

            pl.col("business_address_house_number")
            .alias("s1_business_address_house_number"),

            pl.col("business_address_numeric_tokens")
            .alias("s1_business_address_numeric_tokens"),

            pl.col("business_address_postal_tokens")
            .alias("s1_business_address_postal_tokens"),

            pl.col("country_norm")
            .alias("s1_country_norm"),
        ]
    )
)

print(
    f"Source 1 rows: {s1.height:,}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LightGBM model...")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

print(
    f"Model trees: {model.num_trees():,}"
)

print(
    f"Decision threshold: {THRESHOLD:.2f}"
)


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    source_name,
    source1,
    model,
):
    print("\n" + "=" * 70)
    print(f"PROCESSING TEST {source_name.upper()}")
    print("=" * 70)

    norm_path = NORMALIZED_FILES[source_name]
    cand_path = CANDIDATE_FILES[source_name]
    output_path = OUTPUT_FILES[source_name]

    print(f"Candidate file: {cand_path}")
    print(f"Normalized file: {norm_path}")
    print(f"Output file: {output_path}")

    # --------------------------------------------------------
    # Load target normalized source
    # --------------------------------------------------------

    print("\nLoading target normalized source...")

    target = pl.read_parquet(norm_path)

    target = target.select(
        [
            pl.col("entity_id")
            .alias("candidate_entity_id"),

            pl.col("business_name_norm")
            .alias("candidate_business_name_norm"),

            pl.col("business_name_compact")
            .alias("candidate_business_name_compact"),

            pl.col("business_name_alnum")
            .alias("candidate_business_name_alnum"),

            pl.col("business_name_tokens")
            .alias("candidate_business_name_tokens"),

            pl.col("business_name_sorted_tokens")
            .alias("candidate_business_name_sorted_tokens"),

            pl.col("business_address_norm")
            .alias("candidate_business_address_norm"),

            pl.col("business_address_compact")
            .alias("candidate_business_address_compact"),

            pl.col("business_address_alnum")
            .alias("candidate_business_address_alnum"),

            pl.col("business_address_tokens")
            .alias("candidate_business_address_tokens"),

            pl.col("business_address_sorted_tokens")
            .alias("candidate_business_address_sorted_tokens"),

            pl.col("business_address_house_number")
            .alias("candidate_business_address_house_number"),

            pl.col("business_address_numeric_tokens")
            .alias("candidate_business_address_numeric_tokens"),

            pl.col("business_address_postal_tokens")
            .alias("candidate_business_address_postal_tokens"),

            pl.col("country_norm")
            .alias("candidate_country_norm"),
        ]
    )

    print(
        f"Target rows: {target.height:,}"
    )

    # --------------------------------------------------------
    # Candidate parquet reader
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(
        cand_path
    )

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Candidate rows: {total_rows:,}"
    )

    print(
        f"Batch size: {BATCH_SIZE:,}"
    )

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    writer = None

    total_processed = 0
    total_predicted = 0
    batch_number = 0

    if output_path.exists():
        output_path.unlink()

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):

        batch_number += 1

        candidates = pl.from_arrow(batch)

        # Ensure expected types
        candidates = candidates.select(
            [
                pl.col("source1_entity_id")
                .cast(pl.Utf8),

                pl.col("candidate_entity_id")
                .cast(pl.Utf8),

                pl.col("candidate_source")
                .cast(pl.Utf8),

                pl.col("block_support_count")
                .cast(pl.UInt32),

                pl.col("blocks"),
            ]
        )

        # ----------------------------------------------------
        # Join S1
        # ----------------------------------------------------

        joined = candidates.join(
            source1,
            on="source1_entity_id",
            how="inner",
        )

        # ----------------------------------------------------
        # Join S2/S3
        # ----------------------------------------------------

        joined = joined.join(
            target,
            on="candidate_entity_id",
            how="inner",
        )

        # ----------------------------------------------------
        # Generate features
        # ----------------------------------------------------

        feature_df = make_features(
            joined,
            source_name,
        )

        # ----------------------------------------------------
        # Model matrix
        # ----------------------------------------------------

        X = feature_df.select(
            FEATURE_COLUMNS
        ).to_numpy()

        probabilities = model.predict(
            X
        )

        # ----------------------------------------------------
        # Keep only predicted matches
        # ----------------------------------------------------

        keep = (
            probabilities
            >= THRESHOLD
        )

        if keep.any():

            positive = feature_df.filter(
                pl.Series(
                    "keep",
                    keep,
                )
            )

            positive = positive.with_columns(
                pl.Series(
                    "match_probability",
                    probabilities[keep],
                )
            )

            positive = positive.select(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "candidate_source",
                    "match_probability",
                ]
            )

            arrow_table = positive.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    arrow_table.schema,
                    compression="zstd",
                )

            writer.write_table(
                arrow_table
            )

            total_predicted += positive.height

        total_processed += candidates.height

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or total_processed >= total_rows
        ):
            pct = (
                total_processed
                / total_rows
                * 100
            )

            print(
                f"[{source_name}] "
                f"batch={batch_number:,} "
                f"processed={total_processed:,}/"
                f"{total_rows:,} "
                f"({pct:.1f}%) "
                f"predicted={total_predicted:,}"
            )

    if writer is not None:
        writer.close()

    print("\nCompleted:", source_name)

    print(
        f"Processed candidates: "
        f"{total_processed:,}"
    )

    print(
        f"Predicted matches: "
        f"{total_predicted:,}"
    )

    print(
        f"Output: {output_path}"
    )

    return {
        "source": source_name,
        "candidate_rows": int(total_rows),
        "processed_rows": int(total_processed),
        "predicted_matches": int(total_predicted),
        "output": str(output_path),
    }


# ============================================================
# RUN S2 + S3
# ============================================================

results = []

results.append(
    process_source(
        "s2",
        s1,
        model,
    )
)

results.append(
    process_source(
        "s3",
        s1,
        model,
    )
)


# ============================================================
# FINAL REPORT
# ============================================================

runtime = time.time() - start_time

report = {
    "stage": "08_test_inference",
    "status": "complete",
    "threshold": THRESHOLD,
    "batch_size": BATCH_SIZE,
    "model": str(MODEL_PATH),
    "results": results,
    "total_runtime_seconds": runtime,
}

report_path = (
    ROOT
    / "artifacts/reports/test_inference_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:
    import json

    json.dump(
        report,
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("STAGE 08 COMPLETE")
print("=" * 70)

for result in results:
    print(
        f"{result['source'].upper()}: "
        f"{result['predicted_matches']:,} predicted matches"
    )

print(
    f"Runtime: {runtime:.2f} sec "
    f"({runtime / 60:.2f} min)"
)

print(
    f"Report: {report_path}"
)

print("=" * 70)