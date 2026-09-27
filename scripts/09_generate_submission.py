#!/usr/bin/env python3

import json
import time
from pathlib import Path

import polars as pl


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

# Correct dataset path discovered from project config
S1_FILE = (
    ROOT / "dataset/test/test_source1.tsv"
)

S2_FILE = (
    ROOT / "dataset/test/test_source2.tsv"
)

S3_FILE = (
    ROOT / "dataset/test/test_source3.tsv"
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
# FILE CHECK
# ============================================================

print("\nChecking required files...")

required_files = [
    S1_FILE,
    S2_FILE,
    S3_FILE,
    S2_SCORED,
    S3_SCORED,
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"OK: {path}")


# ============================================================
# LOAD MODEL PREDICTIONS
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
# COMBINE
# ============================================================

print("\nCombining predictions...")

predictions = pl.concat(
    [
        s2,
        s3,
    ],
    how="vertical",
)

print(
    f"Raw predicted pairs: "
    f"{predictions.height:,}"
)


# ============================================================
# SAFETY DEDUPLICATION
# ============================================================

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
    f"Unique predicted pairs: "
    f"{predictions.height:,}"
)


# ============================================================
# VALIDATE SOURCE LABELS
# ============================================================

print("\nChecking candidate sources...")

source_counts = (
    predictions
    .group_by(
        "candidate_source"
    )
    .len()
    .sort("candidate_source")
)

print(source_counts)


# ============================================================
# STAGE 09
# candidate_pairs.tsv
# ============================================================

print(
    "\nGenerating candidate_pairs.tsv..."
)

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
    f"candidate_pairs rows: "
    f"{candidate_pairs.height:,}"
)

print(
    f"Written: "
    f"{CANDIDATE_PAIRS_FILE}"
)


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print(
    "\nLoading test Source 1..."
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
        .alias(
            "source1_entity_id"
        )
    )
    .unique()
)

print(
    f"Test Source 1 entities: "
    f"{s1_ids.height:,}"
)


# ============================================================
# CHECK EXPECTED TEST SIZE
# ============================================================

EXPECTED_S1 = 1_732_544

if s1_ids.height != EXPECTED_S1:

    print(
        "WARNING: Expected "
        f"{EXPECTED_S1:,} Source 1 entities "
        f"but found "
        f"{s1_ids.height:,}"
    )

else:

    print(
        "Source 1 count verified: "
        f"{EXPECTED_S1:,}"
    )


# ============================================================
# BUILD ENTITY MATCH SETS
# ============================================================

print(
    "\nGrouping predicted matches..."
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
# LEFT JOIN TO ALL SOURCE 1
# ============================================================

print(
    "Preserving zero-match entities..."
)

results = (
    s1_ids
    .join(
        grouped,
        on="source1_entity_id",
        how="left",
    )
    .with_columns(
        pl.when(
            pl.col(
                "matched_ids"
            ).is_null()
        )
        .then(
            pl.lit("")
        )
        .otherwise(
            pl.col(
                "matched_ids"
            )
            .list.join(",")
        )
        .alias(
            "matched_entity_ids"
        )
    )
    .select(
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    )
    .sort(
        "source1_entity_id"
    )
)


# ============================================================
# STAGE 10
# matching_results.tsv
# ============================================================

print(
    "\nGenerating matching_results.tsv..."
)

results.write_csv(
    MATCHING_RESULTS_FILE,
    separator="\t",
)

print(
    f"matching_results rows: "
    f"{results.height:,}"
)

print(
    f"Written: "
    f"{MATCHING_RESULTS_FILE}"
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

multiple_matches = results.filter(
    pl.col(
        "matched_entity_ids"
    )
    .str.contains(",")
).height


# ============================================================
# MATCH DISTRIBUTION
# ============================================================

match_counts = (
    results
    .with_columns(
        pl.when(
            pl.col(
                "matched_entity_ids"
            ) == ""
        )
        .then(
            pl.lit(0)
        )
        .otherwise(
            pl.col(
                "matched_entity_ids"
            )
            .str.split(",")
            .list.len()
        )
        .alias(
            "match_count"
        )
    )
    .group_by(
        "match_count"
    )
    .len()
    .sort(
        "match_count"
    )
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nRunning basic submission checks..."
)

# One row per Source 1
if results.height != s1_ids.height:

    raise RuntimeError(
        "FAIL: matching_results does not "
        "contain exactly one row per "
        "Source 1 entity."
    )

# Unique Source 1 IDs
if (
    results[
        "source1_entity_id"
    ].n_unique()
    != s1_ids.height
):

    raise RuntimeError(
        "FAIL: Duplicate Source 1 IDs "
        "in matching_results."
    )

# No null IDs
if results[
    "source1_entity_id"
].null_count() > 0:

    raise RuntimeError(
        "FAIL: Null Source 1 entity ID."
    )

# Candidate pair columns
expected_candidate_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]

if (
    candidate_pairs.columns
    != expected_candidate_columns
):

    raise RuntimeError(
        "FAIL: candidate_pairs "
        "columns are incorrect."
    )

# Matching result columns
expected_result_columns = [
    "source1_entity_id",
    "matched_entity_ids",
]

if (
    results.columns
    != expected_result_columns
):

    raise RuntimeError(
        "FAIL: matching_results "
        "columns are incorrect."
    )


print(
    "Basic validation: PASS"
)


# ============================================================
# REPORT
# ============================================================

runtime = (
    time.time() - start
)

report = {
    "stage":
        "09_10_submission_generation",

    "status":
        "complete",

    "predicted_pairs":
        int(predictions.height),

    "candidate_pairs_rows":
        int(candidate_pairs.height),

    "test_source1_entities":
        int(s1_ids.height),

    "matching_results_rows":
        int(results.height),

    "zero_match_entities":
        int(zero_matches),

    "nonzero_match_entities":
        int(nonzero_matches),

    "multiple_match_entities":
        int(multiple_matches),

    "source_counts":
        {
            row[
                "candidate_source"
            ]:
            int(row["len"])
            for row
            in source_counts.iter_rows(
                named=True
            )
        },

    "match_distribution":
        {
            str(row["match_count"]):
            int(row["len"])
            for row
            in match_counts.iter_rows(
                named=True
            )
        },

    "candidate_pairs_file":
        str(
            CANDIDATE_PAIRS_FILE
        ),

    "matching_results_file":
        str(
            MATCHING_RESULTS_FILE
        ),

    "runtime_seconds":
        runtime,
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
    f"Total predicted pairs: "
    f"{predictions.height:,}"
)

print(
    f"Candidate pairs: "
    f"{candidate_pairs.height:,}"
)

print(
    f"Test Source 1 entities: "
    f"{s1_ids.height:,}"
)

print(
    f"Zero-match entities: "
    f"{zero_matches:,}"
)

print(
    f"Non-zero entities: "
    f"{nonzero_matches:,}"
)

print(
    f"Multiple-match entities: "
    f"{multiple_matches:,}"
)

print(
    f"\nCandidate file:\n"
    f"{CANDIDATE_PAIRS_FILE}"
)

print(
    f"\nMatching file:\n"
    f"{MATCHING_RESULTS_FILE}"
)

print(
    f"\nRuntime: "
    f"{runtime:.2f} sec"
)

print("=" * 70)