"""
STAGE 04 — CANDIDATE RECALL EVALUATION

Purpose
-------
Evaluate whether the generated blocking/candidate pipeline is capable of
retrieving the true matches from the training ground truth.

IMPORTANT
---------
The ground truth contains matched_entity_ids from BOTH Source 2 and Source 3.

Therefore we MUST:
    1. Load the Source 2 entity IDs.
    2. Load the Source 3 entity IDs.
    3. Split ground-truth pairs into S1 -> S2 and S1 -> S3.
    4. Evaluate each candidate file against its correct ground-truth subset.

This prevents the incorrect evaluation where all 7.6M ground-truth pairs
are compared against both S2 and S3 candidate files.
"""


# =============================================================================
# IMPORT PATH FIX
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

# Repository root:
# /home/ubuntu/github-aml2026
ROOT_DIR = Path(__file__).resolve().parents[1]

# Python package directory:
# /home/ubuntu/github-aml2026/code
CODE_DIR = ROOT_DIR / "code"

# Allow:
# from business_entity_resolution.src....
sys.path.insert(0, str(CODE_DIR))


# =============================================================================
# STANDARD IMPORTS
# =============================================================================

import json
import time

import polars as pl


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from business_entity_resolution.src.config import ARTIFACTS_DIR
from business_entity_resolution.src.data_loader import load_train_ground_truth


# =============================================================================
# PATHS
# =============================================================================

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"

CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"

REPORT_DIR = ARTIFACTS_DIR / "reports"

MISS_DIR = REPORT_DIR / "candidate_recall_misses"


# -----------------------------------------------------------------------------
# Normalized source files
# -----------------------------------------------------------------------------

S2_NORMALIZED = (
    NORMALIZED_DIR /
    "train_source2_normalized.parquet"
)

S3_NORMALIZED = (
    NORMALIZED_DIR /
    "train_source3_normalized.parquet"
)


# -----------------------------------------------------------------------------
# Candidate files
# -----------------------------------------------------------------------------

S2_CANDIDATES = (
    CANDIDATES_DIR /
    "train_s1_s2_candidates.parquet"
)

S3_CANDIDATES = (
    CANDIDATES_DIR /
    "train_s1_s3_candidates.parquet"
)


# -----------------------------------------------------------------------------
# Output report
# -----------------------------------------------------------------------------

REPORT_PATH = (
    REPORT_DIR /
    "candidate_recall_report.json"
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_file(path: Path, label: str) -> None:
    """
    Validate that an input file exists and display its size.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\n{label} does not exist:\n{path}\n"
        )

    size_gb = path.stat().st_size / (1024 ** 3)

    print(f"{label}:")
    print(f"  {path}")
    print(f"  Size: {size_gb:.2f} GB")
    print()


# =============================================================================
# LOAD SOURCE ENTITY IDs
# =============================================================================

def load_source_ids(
    path: Path,
    source_name: str,
) -> pl.LazyFrame:
    """
    Load only entity_id from a normalized source.

    Lazy execution is used so the complete normalized dataset does not need
    to be loaded into Python memory.
    """

    print(f"Loading {source_name} entity IDs:")
    print(f"  {path}")

    return (
        pl.scan_parquet(path)
        .select(
            pl.col("entity_id")
            .cast(pl.Utf8)
            .alias("matched_entity_id")
        )
        .unique()
    )


# =============================================================================
# PREPARE GROUND TRUTH
# =============================================================================

def prepare_ground_truth(ground_truth) -> tuple[pl.LazyFrame, int]:
    """
    Convert the ground-truth DataFrame into one row per true pair.

    Input:

        source1_entity_id
        matched_entity_ids

    Output:

        source1_entity_id
        matched_entity_id

    Returns:
        LazyFrame
        zero-match entity count
    """

    # -------------------------------------------------------------------------
    # Convert pandas -> Polars
    # -------------------------------------------------------------------------

    gt = pl.from_pandas(ground_truth)

    print(f"Ground truth rows: {gt.height:,}")

    # -------------------------------------------------------------------------
    # Keep only required columns
    # -------------------------------------------------------------------------

    gt = gt.select(
        [
            pl.col("source1_entity_id")
            .cast(pl.Utf8)
            .alias("source1_entity_id"),

            pl.col("matched_entity_ids")
            .cast(pl.Utf8)
            .alias("matched_entity_ids"),
        ]
    )

    # -------------------------------------------------------------------------
    # Count zero-match entities
    # -------------------------------------------------------------------------

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

    print(
        f"Zero-match S1 entities: "
        f"{zero_match_count:,}"
    )

    # -------------------------------------------------------------------------
    # Convert comma-separated matched IDs into individual rows
    # -------------------------------------------------------------------------

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

        # "id1,id2,id3" -> ["id1","id2","id3"]
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("matched_entity_id_list")
        )

        # One row per matched entity
        .explode("matched_entity_id_list")

        # Clean IDs
        .with_columns(
            pl.col("matched_entity_id_list")
            .str.strip_chars()
            .cast(pl.Utf8)
            .alias("matched_entity_id")
        )

        # Keep only the required columns
        .select(
            [
                "source1_entity_id",
                "matched_entity_id",
            ]
        )

        # Remove invalid IDs
        .filter(
            pl.col("matched_entity_id").is_not_null()
            &
            (
                pl.col("matched_entity_id")
                != ""
            )
        )

        # Remove duplicate GT pairs
        .unique()
    )

    return pairs.lazy(), zero_match_count


# =============================================================================
# SPLIT GROUND TRUTH BY SOURCE
# =============================================================================

def split_ground_truth_by_source(
    gt_pairs: pl.LazyFrame,
    s2_ids: pl.LazyFrame,
    s3_ids: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Split the complete ground truth into:

        S1 -> S2
        S1 -> S3

    using actual entity IDs present in each source.

    A semi-join keeps only GT pairs whose matched_entity_id exists in
    the corresponding source.
    """

    # -------------------------------------------------------------------------
    # S1 -> S2
    # -------------------------------------------------------------------------

    gt_s2 = (
        gt_pairs
        .join(
            s2_ids,
            on="matched_entity_id",
            how="semi",
        )
        .unique()
    )

    # -------------------------------------------------------------------------
    # S1 -> S3
    # -------------------------------------------------------------------------

    gt_s3 = (
        gt_pairs
        .join(
            s3_ids,
            on="matched_entity_id",
            how="semi",
        )
        .unique()
    )

    return gt_s2, gt_s3


# =============================================================================
# COUNT LAZYFRAME
# =============================================================================

def count_rows(lazy_frame: pl.LazyFrame) -> int:
    """
    Count rows without materializing the complete LazyFrame.
    """

    result = (
        lazy_frame
        .select(
            pl.len().alias("count")
        )
        .collect(
            engine="streaming"
        )
    )

    return int(result.item())


# =============================================================================
# EVALUATE ONE SOURCE
# =============================================================================

def evaluate_recall(
    gt_pairs: pl.LazyFrame,
    candidate_path: Path,
    source_name: str,
) -> dict:
    """
    Evaluate candidate recall for one source.
    """

    print_section(
        f"EVALUATING S1 -> {source_name}"
    )

    print("Candidate file:")
    print(candidate_path)
    print()

    # -------------------------------------------------------------------------
    # Load candidates lazily
    # -------------------------------------------------------------------------

    candidates = (
        pl.scan_parquet(candidate_path)

        .select(
            [
                pl.col("source1_entity_id")
                .cast(pl.Utf8)
                .alias("source1_entity_id"),

                pl.col("candidate_entity_id")
                .cast(pl.Utf8)
                .alias("matched_entity_id"),
            ]
        )

        # Protect against duplicate candidate rows
        .unique()
    )

    # -------------------------------------------------------------------------
    # Number of true pairs
    # -------------------------------------------------------------------------

    true_count = count_rows(gt_pairs)

    # -------------------------------------------------------------------------
    # Find retrieved true pairs
    #
    # A true pair is retrieved when the exact:
    #
    #     source1_entity_id
    #     matched_entity_id
    #
    # exists in the candidate file.
    # -------------------------------------------------------------------------

    retrieved = (
        gt_pairs
        .join(
            candidates,
            on=[
                "source1_entity_id",
                "matched_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    # Materialize only the retrieved true pairs.
    retrieved_df = retrieved.collect(
        engine="streaming"
    )

    retrieved_count = retrieved_df.height

    # -------------------------------------------------------------------------
    # Missed pairs
    # -------------------------------------------------------------------------

    missed_count = (
        true_count -
        retrieved_count
    )

    # -------------------------------------------------------------------------
    # Recall
    # -------------------------------------------------------------------------

    if true_count > 0:
        recall = (
            retrieved_count /
            true_count
        )
    else:
        recall = 0.0

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    print(
        f"True pairs       : "
        f"{true_count:,}"
    )

    print(
        f"Retrieved pairs  : "
        f"{retrieved_count:,}"
    )

    print(
        f"Missed pairs     : "
        f"{missed_count:,}"
    )

    print(
        f"Candidate recall : "
        f"{recall:.4%}"
    )

    # -------------------------------------------------------------------------
    # Save a sample of missed pairs
    #
    # Limit to 100,000 rows so the report file remains manageable.
    # -------------------------------------------------------------------------

    missed = (
        gt_pairs

        .join(
            retrieved_df.lazy(),

            on=[
                "source1_entity_id",
                "matched_entity_id",
            ],

            how="anti",
        )

        .limit(100_000)

        .collect(
            engine="streaming"
        )
    )

    # -------------------------------------------------------------------------
    # Save misses
    # -------------------------------------------------------------------------

    MISS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    miss_path = (
        MISS_DIR /
        f"s1_{source_name.lower()}_missed_examples.parquet"
    )

    missed.write_parquet(
        miss_path,
        compression="zstd",
    )

    print()
    print("Missed examples saved:")
    print(miss_path)

    # -------------------------------------------------------------------------
    # Return result
    # -------------------------------------------------------------------------

    return {
        "true_pairs": int(true_count),

        "retrieved_pairs": int(
            retrieved_count
        ),

        "missed_pairs": int(
            missed_count
        ),

        "candidate_recall": float(
            recall
        ),

        "missed_examples_path": str(
            miss_path
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = time.time()

    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("#" * 80)
    print("# STAGE 04 — CANDIDATE RECALL EVALUATION")
    print("#" * 80)

    # =========================================================================
    # VALIDATE INPUT FILES
    # =========================================================================

    print_section(
        "VALIDATING INPUT FILES"
    )

    validate_file(
        S2_NORMALIZED,
        "S2 normalized source",
    )

    validate_file(
        S3_NORMALIZED,
        "S3 normalized source",
    )

    validate_file(
        S2_CANDIDATES,
        "S2 candidates",
    )

    validate_file(
        S3_CANDIDATES,
        "S3 candidates",
    )

    # =========================================================================
    # LOAD GROUND TRUTH
    # =========================================================================

    print_section(
        "LOADING GROUND TRUTH"
    )

    ground_truth = load_train_ground_truth()

    gt_pairs, zero_match_count = (
        prepare_ground_truth(
            ground_truth
        )
    )

    total_pairs = count_rows(
        gt_pairs
    )

    print(
        f"True matched pairs: "
        f"{total_pairs:,}"
    )

    # =========================================================================
    # LOAD SOURCE ENTITY IDs
    # =========================================================================

    print_section(
        "BUILDING SOURCE ID INDEXES"
    )

    s2_ids = load_source_ids(
        S2_NORMALIZED,
        "Source 2",
    )

    s3_ids = load_source_ids(
        S3_NORMALIZED,
        "Source 3",
    )

    # =========================================================================
    # CHECK SOURCE ID OVERLAP
    # =========================================================================

    print_section(
        "CHECKING SOURCE ID OVERLAP"
    )

    overlapping_ids = (
        s2_ids
        .join(
            s3_ids,
            on="matched_entity_id",
            how="inner",
        )
        .select(
            pl.len().alias("count")
        )
        .collect(
            engine="streaming"
        )
        .item()
    )

    overlapping_ids = int(
        overlapping_ids
    )

    print(
        "Entity IDs appearing in BOTH "
        f"S2 and S3: {overlapping_ids:,}"
    )

    if overlapping_ids > 0:

        print()
        print(
            "WARNING:"
        )

        print(
            "Some entity IDs occur in both "
            "Source 2 and Source 3."
        )

        print(
            "Those IDs will be counted in both "
            "source-specific ground-truth sets."
        )

    # =========================================================================
    # SPLIT GROUND TRUTH
    # =========================================================================

    print_section(
        "SPLITTING GROUND TRUTH BY SOURCE"
    )

    gt_s2, gt_s3 = (
        split_ground_truth_by_source(
            gt_pairs,
            s2_ids,
            s3_ids,
        )
    )

    # -------------------------------------------------------------------------
    # Count S2 true pairs
    # -------------------------------------------------------------------------

    s2_true_count = count_rows(
        gt_s2
    )

    # -------------------------------------------------------------------------
    # Count S3 true pairs
    # -------------------------------------------------------------------------

    s3_true_count = count_rows(
        gt_s3
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    print(
        f"S1 -> S2 true pairs: "
        f"{s2_true_count:,}"
    )

    print(
        f"S1 -> S3 true pairs: "
        f"{s3_true_count:,}"
    )

    print(
        f"S2 + S3:              "
        f"{s2_true_count + s3_true_count:,}"
    )

    print(
        f"Original GT pairs:     "
        f"{total_pairs:,}"
    )

    # =========================================================================
    # SANITY CHECK
    # =========================================================================

    combined_source_pairs = (
        s2_true_count +
        s3_true_count
    )

    if overlapping_ids == 0:

        if combined_source_pairs != total_pairs:

            print()
            print(
                "WARNING:"
            )

            print(
                "S2 + S3 ground-truth pair count "
                "does not equal the original GT pair count."
            )

            print(
                f"Original: {total_pairs:,}"
            )

            print(
                f"S2 + S3: {combined_source_pairs:,}"
            )

            print(
                "This indicates that some GT matched IDs "
                "were not found in either source."
            )

    # =========================================================================
    # EVALUATE S1 -> S2
    # =========================================================================

    result_s2 = evaluate_recall(
        gt_pairs=gt_s2,
        candidate_path=S2_CANDIDATES,
        source_name="S2",
    )

    # =========================================================================
    # EVALUATE S1 -> S3
    # =========================================================================

    result_s3 = evaluate_recall(
        gt_pairs=gt_s3,
        candidate_path=S3_CANDIDATES,
        source_name="S3",
    )

    # =========================================================================
    # OVERALL RECALL
    # =========================================================================

    total_true = (
        result_s2["true_pairs"] +
        result_s3["true_pairs"]
    )

    total_retrieved = (
        result_s2["retrieved_pairs"] +
        result_s3["retrieved_pairs"]
    )

    total_missed = (
        total_true -
        total_retrieved
    )

    if total_true > 0:

        overall_recall = (
            total_retrieved /
            total_true
        )

    else:

        overall_recall = 0.0

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    print_section(
        "FINAL CANDIDATE RECALL"
    )

    print()
    print("S1 -> S2")

    print(
        f"  True pairs      : "
        f"{result_s2['true_pairs']:,}"
    )

    print(
        f"  Retrieved       : "
        f"{result_s2['retrieved_pairs']:,}"
    )

    print(
        f"  Missed          : "
        f"{result_s2['missed_pairs']:,}"
    )

    print(
        f"  Recall          : "
        f"{result_s2['candidate_recall']:.4%}"
    )

    print()
    print("S1 -> S3")

    print(
        f"  True pairs      : "
        f"{result_s3['true_pairs']:,}"
    )

    print(
        f"  Retrieved       : "
        f"{result_s3['retrieved_pairs']:,}"
    )

    print(
        f"  Missed          : "
        f"{result_s3['missed_pairs']:,}"
    )

    print(
        f"  Recall          : "
        f"{result_s3['candidate_recall']:.4%}"
    )

    print()
    print("OVERALL")

    print(
        f"  True pairs      : "
        f"{total_true:,}"
    )

    print(
        f"  Retrieved       : "
        f"{total_retrieved:,}"
    )

    print(
        f"  Missed          : "
        f"{total_missed:,}"
    )

    print(
        f"  Recall          : "
        f"{overall_recall:.4%}"
    )

    # =========================================================================
    # SAVE JSON REPORT
    # =========================================================================

    report = {

        "stage":
            "04_candidate_recall",

        "status":
            "complete",

        "ground_truth": {

            "total_rows":
                int(ground_truth.shape[0]),

            "zero_match_entities":
                int(zero_match_count),

            "total_matched_pairs":
                int(total_pairs),

            "s2_true_pairs":
                int(s2_true_count),

            "s3_true_pairs":
                int(s3_true_count),
        },

        "source_id_overlap": {

            "overlapping_ids":
                int(overlapping_ids),
        },

        "s1_to_s2":
            result_s2,

        "s1_to_s3":
            result_s3,

        "overall": {

            "true_pairs":
                int(total_true),

            "retrieved_pairs":
                int(total_retrieved),

            "missed_pairs":
                int(total_missed),

            "candidate_recall":
                float(overall_recall),
        },

        "runtime_seconds":
            float(time.time() - start_time),
    }

    # =========================================================================
    # WRITE REPORT
    # =========================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # =========================================================================
    # COMPLETION
    # =========================================================================

    print()
    print("Report:")
    print(REPORT_PATH)

    runtime_minutes = (
        time.time() -
        start_time
    ) / 60

    print()
    print(
        f"Runtime: "
        f"{runtime_minutes:.2f} minutes"
    )

    print()
    print("#" * 80)
    print("# STAGE 04 COMPLETE")
    print("#" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()