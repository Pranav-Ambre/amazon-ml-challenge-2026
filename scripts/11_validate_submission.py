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
    ROOT / "dataset/test/test_source1.tsv"
)

S2_FILE = (
    ROOT / "dataset/test/test_source2.tsv"
)

S3_FILE = (
    ROOT / "dataset/test/test_source3.tsv"
)

CANDIDATE_FILE = (
    ROOT
    / "artifacts/submission/candidate_pairs.tsv"
)

MATCHING_FILE = (
    ROOT
    / "artifacts/submission/matching_results.tsv"
)

REPORT_FILE = (
    ROOT
    / "artifacts/reports/final_submission_validation.json"
)


# ============================================================
# START
# ============================================================

start = time.time()

print("=" * 70)
print("STAGE 11 - FINAL SUBMISSION VALIDATION")
print("=" * 70)


errors = []
warnings = []


def fail(message):
    errors.append(message)
    print(f"FAIL: {message}")


def warn(message):
    warnings.append(message)
    print(f"WARNING: {message}")


def ok(message):
    print(f"PASS: {message}")


# ============================================================
# FILE EXISTENCE
# ============================================================

print("\n[1] Checking files...")

required_files = [
    S1_FILE,
    S2_FILE,
    S3_FILE,
    CANDIDATE_FILE,
    MATCHING_FILE,
]

for path in required_files:

    if path.exists():
        size_mb = (
            path.stat().st_size
            / (1024 * 1024)
        )

        ok(
            f"{path.name} exists "
            f"({size_mb:.2f} MB)"
        )

    else:
        fail(
            f"Missing file: {path}"
        )


# ============================================================
# LOAD SOURCE IDS
# ============================================================

print("\n[2] Loading valid entity IDs...")

s1 = pl.read_csv(
    S1_FILE,
    separator="\t",
    has_header=True,
    columns=["entity_id"],
    infer_schema_length=10000,
)

s2 = pl.read_csv(
    S2_FILE,
    separator="\t",
    has_header=True,
    columns=["entity_id"],
    infer_schema_length=10000,
)

s3 = pl.read_csv(
    S3_FILE,
    separator="\t",
    has_header=True,
    columns=["entity_id"],
    infer_schema_length=10000,
)

s1_ids = set(
    s1["entity_id"]
    .cast(pl.Utf8)
    .to_list()
)

s2_ids = set(
    s2["entity_id"]
    .cast(pl.Utf8)
    .to_list()
)

s3_ids = set(
    s3["entity_id"]
    .cast(pl.Utf8)
    .to_list()
)

print(
    f"S1 IDs: {len(s1_ids):,}"
)

print(
    f"S2 IDs: {len(s2_ids):,}"
)

print(
    f"S3 IDs: {len(s3_ids):,}"
)


# ============================================================
# CHECK S2/S3 ID OVERLAP
# ============================================================

print(
    "\n[3] Checking S2/S3 ID overlap..."
)

overlap = (
    s2_ids
    & s3_ids
)

if overlap:

    fail(
        f"S2/S3 entity ID overlap: "
        f"{len(overlap):,}"
    )

else:

    ok(
        "No S2/S3 entity ID overlap."
    )


# ============================================================
# LOAD CANDIDATE PAIRS
# ============================================================

print(
    "\n[4] Loading candidate_pairs.tsv..."
)

candidate_pairs = pl.read_csv(
    CANDIDATE_FILE,
    separator="\t",
    has_header=True,
    infer_schema_length=10000,
)

print(
    f"Candidate rows: "
    f"{candidate_pairs.height:,}"
)

print(
    f"Columns: "
    f"{candidate_pairs.columns}"
)


# ============================================================
# CANDIDATE HEADER
# ============================================================

expected_candidate_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]

if (
    candidate_pairs.columns
    == expected_candidate_columns
):

    ok(
        "candidate_pairs columns correct."
    )

else:

    fail(
        "candidate_pairs columns are "
        f"{candidate_pairs.columns}, "
        f"expected "
        f"{expected_candidate_columns}"
    )


# ============================================================
# CANDIDATE NULL CHECK
# ============================================================

for column in [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]:

    null_count = (
        candidate_pairs[column]
        .null_count()
    )

    if null_count:

        fail(
            f"candidate_pairs has "
            f"{null_count:,} null values "
            f"in {column}"
        )

    else:

        ok(
            f"No nulls in "
            f"{column}."
        )


# ============================================================
# CANDIDATE SOURCE VALUES
# ============================================================

candidate_sources = set(
    candidate_pairs[
        "candidate_source"
    ]
    .cast(pl.Utf8)
    .unique()
    .to_list()
)

print(
    f"Candidate sources: "
    f"{candidate_sources}"
)

invalid_sources = (
    candidate_sources
    - {"s2", "s3"}
)

if invalid_sources:

    fail(
        f"Invalid candidate_source values: "
        f"{invalid_sources}"
    )

else:

    ok(
        "candidate_source values valid."
    )


# ============================================================
# VALIDATE SOURCE 1 IDS
# ============================================================

print(
    "\n[5] Validating candidate Source-1 IDs..."
)

candidate_s1 = set(
    candidate_pairs[
        "source1_entity_id"
    ]
    .cast(pl.Utf8)
    .unique()
    .to_list()
)

invalid_s1 = (
    candidate_s1
    - s1_ids
)

if invalid_s1:

    fail(
        f"Invalid S1 IDs in candidate_pairs: "
        f"{len(invalid_s1):,}"
    )

else:

    ok(
        "All candidate Source-1 IDs "
        "exist in test S1."
    )


# ============================================================
# VALIDATE CANDIDATE IDS
# ============================================================

print(
    "\n[6] Validating candidate entity IDs..."
)

s2_rows = candidate_pairs.filter(
    pl.col(
        "candidate_source"
    ) == "s2"
)

s3_rows = candidate_pairs.filter(
    pl.col(
        "candidate_source"
    ) == "s3"
)

invalid_s2 = (
    set(
        s2_rows[
            "candidate_entity_id"
        ]
        .cast(pl.Utf8)
        .unique()
        .to_list()
    )
    - s2_ids
)

invalid_s3 = (
    set(
        s3_rows[
            "candidate_entity_id"
        ]
        .cast(pl.Utf8)
        .unique()
        .to_list()
    )
    - s3_ids
)

if invalid_s2:

    fail(
        f"Invalid S2 candidate IDs: "
        f"{len(invalid_s2):,}"
    )

else:

    ok(
        "All S2 candidate IDs valid."
    )


if invalid_s3:

    fail(
        f"Invalid S3 candidate IDs: "
        f"{len(invalid_s3):,}"
    )

else:

    ok(
        "All S3 candidate IDs valid."
    )


# ============================================================
# DUPLICATE CANDIDATE PAIRS
# ============================================================

print(
    "\n[7] Checking duplicate candidate pairs..."
)

unique_candidate_pairs = (
    candidate_pairs
    .select(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )
    .unique()
)

duplicate_count = (
    candidate_pairs.height
    - unique_candidate_pairs.height
)

if duplicate_count:

    fail(
        f"Duplicate candidate pairs: "
        f"{duplicate_count:,}"
    )

else:

    ok(
        "No duplicate candidate pairs."
    )


# ============================================================
# LOAD MATCHING RESULTS
# ============================================================

print(
    "\n[8] Loading matching_results.tsv..."
)

results = pl.read_csv(
    MATCHING_FILE,
    separator="\t",
    has_header=True,
    infer_schema_length=10000,
    null_values=[],
)

print(
    f"Result rows: "
    f"{results.height:,}"
)

print(
    f"Columns: "
    f"{results.columns}"
)


# ============================================================
# RESULT HEADER
# ============================================================

expected_result_columns = [
    "source1_entity_id",
    "matched_entity_ids",
]

if (
    results.columns
    == expected_result_columns
):

    ok(
        "matching_results columns correct."
    )

else:

    fail(
        "matching_results columns are "
        f"{results.columns}, "
        f"expected "
        f"{expected_result_columns}"
    )


# ============================================================
# RESULT ROW COUNT
# ============================================================

if results.height == len(s1_ids):

    ok(
        "Exactly one result row per "
        "test Source-1 entity."
    )

else:

    fail(
        f"Expected {len(s1_ids):,} result "
        f"rows, found {results.height:,}."
    )


# ============================================================
# RESULT SOURCE 1 IDs
# ============================================================

result_s1_ids = set(
    results[
        "source1_entity_id"
    ]
    .cast(pl.Utf8)
    .to_list()
)

missing_s1 = (
    s1_ids
    - result_s1_ids
)

extra_s1 = (
    result_s1_ids
    - s1_ids
)

if missing_s1:

    fail(
        f"Missing S1 IDs in results: "
        f"{len(missing_s1):,}"
    )

else:

    ok(
        "No missing Source-1 entities."
    )


if extra_s1:

    fail(
        f"Invalid extra S1 IDs: "
        f"{len(extra_s1):,}"
    )

else:

    ok(
        "No invalid Source-1 entities."
    )


# ============================================================
# DUPLICATE RESULT SOURCE 1 IDS
# ============================================================

duplicate_result_s1 = (
    results.height
    - results[
        "source1_entity_id"
    ].n_unique()
)

if duplicate_result_s1:

    fail(
        f"Duplicate Source-1 result rows: "
        f"{duplicate_result_s1:,}"
    )

else:

    ok(
        "Exactly one row per Source-1 ID."
    )


# ============================================================
# VALIDATE MATCHED ENTITY IDS
# ============================================================

print(
    "\n[9] Validating matched entity IDs..."
)

invalid_match_count = 0
invalid_match_examples = []

for row in results.iter_rows(
    named=True
):

    source1_id = str(
        row["source1_entity_id"]
    )

    matched = row[
        "matched_entity_ids"
    ]

    if matched is None:
        matched = ""

    matched = str(matched).strip()

    if matched == "":
        continue

    ids = [
        x.strip()
        for x in matched.split(",")
        if x.strip()
    ]

    seen = set()

    for entity_id in ids:

        # Duplicate within same entity
        if entity_id in seen:

            invalid_match_count += 1

            if len(
                invalid_match_examples
            ) < 10:

                invalid_match_examples.append(
                    (
                        source1_id,
                        entity_id,
                        "duplicate",
                    )
                )

            continue

        seen.add(entity_id)

        # Candidate must belong to S2 or S3
        if (
            entity_id not in s2_ids
            and entity_id not in s3_ids
        ):

            invalid_match_count += 1

            if len(
                invalid_match_examples
            ) < 10:

                invalid_match_examples.append(
                    (
                        source1_id,
                        entity_id,
                        "invalid_entity_id",
                    )
                )


if invalid_match_count:

    fail(
        f"Invalid matched IDs: "
        f"{invalid_match_count:,}"
    )

    print(
        "Examples:",
        invalid_match_examples,
    )

else:

    ok(
        "All matched entity IDs "
        "are valid and unique per entity."
    )


# ============================================================
# MATCHING RESULTS VS CANDIDATE PAIRS
# ============================================================

print(
    "\n[10] Checking prediction consistency..."
)

candidate_pair_set = set(
    zip(
        candidate_pairs[
            "source1_entity_id"
        ]
        .cast(pl.Utf8)
        .to_list(),

        candidate_pairs[
            "candidate_entity_id"
        ]
        .cast(pl.Utf8)
        .to_list(),
    )
)

submission_pair_count = 0
missing_candidate_pairs = 0
examples = []

for row in results.iter_rows(
    named=True
):

    source1_id = str(
        row["source1_entity_id"]
    )

    matched = row[
        "matched_entity_ids"
    ]

    if matched is None:
        continue

    matched = str(
        matched
    ).strip()

    if matched == "":
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if not candidate_id:
            continue

        submission_pair_count += 1

        pair = (
            source1_id,
            candidate_id,
        )

        if pair not in candidate_pair_set:

            missing_candidate_pairs += 1

            if len(examples) < 10:
                examples.append(pair)


if missing_candidate_pairs:

    fail(
        f"{missing_candidate_pairs:,} "
        "submission pairs are not present "
        "in candidate_pairs.tsv."
    )

    print(
        "Examples:",
        examples,
    )

else:

    ok(
        "Every submitted match exists "
        "in candidate_pairs.tsv."
    )


# ============================================================
# ZERO MATCHES
# ============================================================

zero_match_entities = results.filter(
    pl.col(
        "matched_entity_ids"
    ) == ""
).height

nonzero_entities = (
    results.height
    - zero_match_entities
)

print(
    "\nZero-match entities: "
    f"{zero_match_entities:,}"
)

print(
    "Non-zero entities: "
    f"{nonzero_entities:,}"
)


# ============================================================
# FILE SIZE
# ============================================================

candidate_size_mb = (
    CANDIDATE_FILE.stat().st_size
    / (1024 * 1024)
)

matching_size_mb = (
    MATCHING_FILE.stat().st_size
    / (1024 * 1024)
)

print(
    "\nCandidate file size: "
    f"{candidate_size_mb:.2f} MB"
)

print(
    "Matching file size: "
    f"{matching_size_mb:.2f} MB"
)


# ============================================================
# FINAL STATUS
# ============================================================

runtime = (
    time.time()
    - start
)

status = (
    "PASS"
    if not errors
    else "FAIL"
)

report = {
    "stage":
        "11_final_submission_validation",

    "status":
        status,

    "errors":
        errors,

    "warnings":
        warnings,

    "statistics": {
        "s1_entities":
            len(s1_ids),

        "s2_entities":
            len(s2_ids),

        "s3_entities":
            len(s3_ids),

        "candidate_pairs":
            candidate_pairs.height,

        "unique_candidate_pairs":
            unique_candidate_pairs.height,

        "matching_result_rows":
            results.height,

        "zero_match_entities":
            zero_match_entities,

        "nonzero_entities":
            nonzero_entities,

        "submission_pairs":
            submission_pair_count,

        "candidate_file_mb":
            candidate_size_mb,

        "matching_file_mb":
            matching_size_mb,
    },

    "files": {
        "candidate_pairs":
            str(CANDIDATE_FILE),

        "matching_results":
            str(MATCHING_FILE),
    },

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
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUBMISSION VALIDATION")
print("=" * 70)

if errors:

    print(
        f"❌ VALIDATION FAILED "
        f"({len(errors)} errors)"
    )

    for error in errors:
        print(
            f"  - {error}"
        )

else:

    print(
        "✅ VALIDATION PASSED"
    )

    print(
        "\nYour submission files are ready."
    )

print(
    f"\nReport: {REPORT_FILE}"
)

print(
    f"Runtime: {runtime:.2f} sec"
)

print("=" * 70)