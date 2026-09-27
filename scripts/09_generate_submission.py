#!/usr/bin/env python3

import json
import time
from pathlib import Path

import polars as pl


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

S1_FILE = (
    ROOT
    / "data/raw/test_source1.tsv"
)

S2_FILE = (
    ROOT
    / "data/raw/test_source2.tsv"
)

S3_FILE = (
    ROOT
    / "data/raw/test_source3.tsv"
)

S2_SCORED = (
    ROOT
    / "artifacts/inference/test_scored_s2.parquet"
)

S3_SCORED = (
    ROOT
    / "artifacts/inference/test_scored_s3.parquet"
)

SUBMISSION_DIR = (
    ROOT / "artifacts/submission"
)

REPORT_DIR = (
    ROOT / "artifacts/reports"
)

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CANDIDATE_PAIRS_FILE = (
    SUBMISSION_DIR
    / "candidate_pairs.tsv"
)

MATCHING_RESULTS_FILE = (
    SUBMISSION_DIR
    / "matching_results.tsv"
)

REPORT_FILE = (
    REPORT_DIR
    / "submission_generation_report.json"
)


# ============================================================
# START
# ============================================================

start = time.time()

print("=" * 70)
print("STAGE 09/10 - SUBMISSION GENERATION")
print("=" * 70)


# ============================================================
# LOAD SCORED PREDICTIONS
# ============================================================

print("\nLoading S2 predictions...")

s2 = pl.read_parquet(
    S2_SCORED
)

print(
    f"S2 predictions: "
    f"{s2.height:,}"
)

print("\nLoading S3 predictions...")

s3 = pl.read_parquet(
    S3_SCORED
)

print(
    f"S3 predictions: "
    f"{s3.height:,}"
)


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

predictions = pl.concat(
    [
        s2,
        s3,
    ],
    how="vertical",
)

# Safety deduplication
predictions = (
    predictions
    .unique(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
        keep="first",
    )
)

print(
    f"\nTotal unique predicted pairs: "
    f"{predictions.height:,}"
)


# ============================================================
# CANDIDATE PAIRS
# ============================================================

print("\nGenerating candidate_pairs.tsv...")

candidate_pairs = (
    predictions
    .select(
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )
    .sort(
        [
            "source1_entity_id",
            "candidate_source",
            "candidate_entity_id",
        ]
    )
)

candidate_pairs.write_csv(
    CANDIDATE_PAIRS_FILE,
    separator="\t",
)

print(
    f"Written: "
    f"{CANDIDATE_PAIRS_FILE}"
)

print(
    f"Rows: "
    f"{candidate_pairs.height:,}"
)


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print(
    "\nLoading test Source 1 "
    "to preserve zero-match entities..."
)

s1 = pl.read_csv(
    S1_FILE,
    separator="\t",
    has_header=True,
    infer_schema_length=10000,
)

s1_ids = (
    s1
    .select(
        pl.col("entity_id")
        .cast(pl.Utf8)
        .alias("source1_entity_id")
    )
    .unique()
)

print(
    f"Test Source 1 entities: "
    f"{s1_ids.height:,}"
)


# ============================================================
# GROUP MATCHES BY SOURCE 1
# ============================================================

print(
    "\nBuilding matching_results..."
)

grouped = (
    predictions
    .select(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )
    .sort(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )
    .group_by(
        "source1_entity_id",
        maintain_order=True,
    )
    .agg(
        pl.col(
            "candidate_entity_id"
        ).alias(
            "matched_ids"
        )
    )
)


# ============================================================
# ADD ZERO-MATCH ENTITIES
# ============================================================

results = (
    s1_ids
    .join(
        grouped,
        on="source1_entity_id",
        how="left",
    )
    .with_columns(
        pl.when(
            pl.col("matched_ids").is_null()
        )
        .then(
            pl.lit("")
        )
        .otherwise(
            pl.col("matched_ids")
            .list.join(",")
        )
        .alias("matched_entity_ids")
    )
    .select(
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    )
)


# ============================================================
# SORT
# ============================================================

results = results.sort(
    "source1_entity_id"
)


# ============================================================
# WRITE FINAL MATCHING RESULTS
# ============================================================

results.write_csv(
    MATCHING_RESULTS_FILE,
    separator="\t",
)

print(
    f"\nWritten: "
    f"{MATCHING_RESULTS_FILE}"
)

print(
    f"Rows: "
    f"{results.height:,}"
)


# ============================================================
# STATISTICS
# ============================================================

zero_matches = results.filter(
    pl.col(
        "matched_entity_ids"
    ) == ""
).height

nonzero_matches = (
    results.height
    - zero_matches
)

multi_matches = results.filter(
    pl.col(
        "matched_entity_ids"
    )
    .str.contains(",")
).height

print("\nSubmission statistics:")
print(
    f"Total Source 1: "
    f"{results.height:,}"
)

print(
    f"Zero matches: "
    f"{zero_matches:,}"
)

print(
    f"Non-zero matches: "
    f"{nonzero_matches:,}"
)

print(
    f"Multiple matches: "
    f"{multi_matches:,}"
)


# ============================================================
# VALIDATE SOURCE 1 COVERAGE
# ============================================================

if results.height != s1_ids.height:

    raise RuntimeError(
        "ERROR: matching_results.tsv "
        "does not contain exactly one row "
        "for every test Source 1 entity."
    )

if (
    results["source1_entity_id"]
    .n_unique()
    != s1_ids.height
):

    raise RuntimeError(
        "ERROR: duplicate Source 1 IDs "
        "in matching_results.tsv."
    )


# ============================================================
# REPORT
# ============================================================

runtime = time.time() - start

report = {
    "stage": "09_10_submission_generation",
    "status": "complete",

    "predicted_pairs": int(
        predictions.height
    ),

    "candidate_pairs_rows": int(
        candidate_pairs.height
    ),

    "test_source1_entities": int(
        s1_ids.height
    ),

    "matching_results_rows": int(
        results.height
    ),

    "zero_match_entities": int(
        zero_matches
    ),

    "nonzero_match_entities": int(
        nonzero_matches
    ),

    "multiple_match_entities": int(
        multi_matches
    ),

    "candidate_pairs_file": str(
        CANDIDATE_PAIRS_FILE
    ),

    "matching_results_file": str(
        MATCHING_RESULTS_FILE
    ),

    "runtime_seconds": runtime,
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 09/10 COMPLETE")
print("=" * 70)

print(
    f"candidate_pairs.tsv: "
    f"{CANDIDATE_PAIRS_FILE}"
)

print(
    f"matching_results.tsv: "
    f"{MATCHING_RESULTS_FILE}"
)

print(
    f"Runtime: "
    f"{runtime:.2f} sec"
)

print("=" * 70)