"""
STAGE 04 — CANDIDATE RECALL EVALUATION

Correctly evaluates candidate recall separately for:
    S1 -> S2
    S1 -> S3

Important:
Ground truth contains matched_entity_ids from both Source 2 and Source 3.
Therefore, we MUST first identify which matched IDs belong to S2/S3.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from business_entity_resolution.src.config import (
    ARTIFACTS_DIR,
)
from business_entity_resolution.src.data_loader import (
    load_train_ground_truth,
)


# ============================================================================
# PATHS
# ============================================================================

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
REPORT_DIR = ARTIFACTS_DIR / "reports"
MISS_DIR = REPORT_DIR / "candidate_recall_misses"

S2_NORMALIZED = NORMALIZED_DIR / "train_source2_normalized.parquet"
S3_NORMALIZED = NORMALIZED_DIR / "train_source3_normalized.parquet"

S2_CANDIDATES = CANDIDATES_DIR / "train_s1_s2_candidates.parquet"
S3_CANDIDATES = CANDIDATES_DIR / "train_s1_s3_candidates.parquet"

REPORT_PATH = REPORT_DIR / "candidate_recall_report.json"


# ============================================================================
# HELPERS
# ============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist:\n{path}"
        )

    size_gb = path.stat().st_size / (1024 ** 3)

    print(f"{label}:")
    print(f"  {path}")
    print(f"  Size: {size_gb:.2f} GB")
    print()


def load_source_ids(path: Path, source_name: str) -> pl.LazyFrame:
    """
    Load only entity_id from normalized source.

    Lazy execution avoids loading the entire normalized dataset.
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


def prepare_ground_truth(ground_truth) -> pl.LazyFrame:
    """
    Convert ground truth into:

        source1_entity_id
        matched_entity_id

    one row per true pair.
    """

    gt = pl.from_pandas(ground_truth)

    print(f"Ground truth rows: {gt.height:,}")

    # ------------------------------------------------------------------------
    # Normalize column names
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Zero-match entities
    # ------------------------------------------------------------------------

    zero_match_count = (
        gt.filter(
            pl.col("matched_entity_ids").is_null()
            | (pl.col("matched_entity_ids").str.strip_chars() == "")
        )
        .height
    )

    print(f"Zero-match S1 entities: {zero_match_count:,}")

    # ------------------------------------------------------------------------
    # Explode matched IDs
    # ------------------------------------------------------------------------

    pairs = (
        gt
        .filter(
            pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids").str.strip_chars() != "")
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
            .alias("matched_entity_id")
        )
        .select(
            [
                "source1_entity_id",
                "matched_entity_id",
            ]
        )
        .filter(
            pl.col("matched_entity_id").is_not_null()
            & (pl.col("matched_entity_id") != "")
        )
        .unique()
    )

    return pairs.lazy()


def split_ground_truth_by_source(
    gt_pairs: pl.LazyFrame,
    s2_ids: pl.LazyFrame,
    s3_ids: pl.LazyFrame,
):
    """
    Split ground-truth pairs according to source membership.

    S2 true pairs:
        GT matched_entity_id exists in Source 2.

    S3 true pairs:
        GT matched_entity_id exists in Source 3.
    """

    # ------------------------------------------------------------------------
    # S1 -> S2
    # ------------------------------------------------------------------------

    gt_s2 = (
        gt_pairs
        .join(
            s2_ids,
            on="matched_entity_id",
            how="semi",
        )
        .unique()
    )

    # ------------------------------------------------------------------------
    # S1 -> S3
    # ------------------------------------------------------------------------

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


def evaluate_recall(
    gt_pairs: pl.LazyFrame,
    candidate_path: Path,
    source_name: str,
):
    """
    Evaluate candidate recall for one source.
    """

    print_section(f"EVALUATING S1 -> {source_name}")

    print("Candidate file:")
    print(candidate_path)
    print()

    # ------------------------------------------------------------------------
    # Load candidate pairs lazily
    # ------------------------------------------------------------------------

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
        .unique()
    )

    # ------------------------------------------------------------------------
    # Count true pairs
    # ------------------------------------------------------------------------

    true_count = (
        gt_pairs
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )

    # ------------------------------------------------------------------------
    # Retrieved true pairs
    #
    # INNER JOIN:
    #   GT pair exists in candidate set.
    # ------------------------------------------------------------------------

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

    retrieved_df = retrieved.collect(engine="streaming")

    retrieved_count = retrieved_df.height

    missed_count = true_count - retrieved_count

    recall = (
        retrieved_count / true_count
        if true_count > 0
        else 0.0
    )

    print(f"True pairs       : {true_count:,}")
    print(f"Retrieved pairs  : {retrieved_count:,}")
    print(f"Missed pairs     : {missed_count:,}")
    print(f"Candidate recall : {recall:.4%}")

    # ------------------------------------------------------------------------
    # Save missed examples
    # ------------------------------------------------------------------------

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
        .collect(engine="streaming")
    )

    MISS_DIR.mkdir(parents=True, exist_ok=True)

    miss_path = MISS_DIR / f"s1_{source_name.lower()}_missed_examples.parquet"

    missed.write_parquet(
        miss_path,
        compression="zstd",
    )

    print(f"Missed examples saved:")
    print(miss_path)

    return {
        "true_pairs": int(true_count),
        "retrieved_pairs": int(retrieved_count),
        "missed_pairs": int(missed_count),
        "candidate_recall": float(recall),
        "missed_examples_path": str(miss_path),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print()
    print("#" * 80)
    print("# STAGE 04 — CORRECTED CANDIDATE RECALL EVALUATION")
    print("#" * 80)

    # ========================================================================
    # VALIDATE FILES
    # ========================================================================

    print_section("VALIDATING INPUT FILES")

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

    # ========================================================================
    # LOAD GROUND TRUTH
    # ========================================================================

    print_section("LOADING GROUND TRUTH")

    ground_truth = load_train_ground_truth()

    gt_pairs = prepare_ground_truth(ground_truth)

    # Materialize total pair count only.
    total_pairs = (
        gt_pairs
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )

    print(f"Total true matched pairs: {total_pairs:,}")

    # ========================================================================
    # LOAD SOURCE ID INDEXES
    # ========================================================================

    print_section("BUILDING SOURCE ID INDEXES")

    s2_ids = load_source_ids(
        S2_NORMALIZED,
        "Source 2",
    )

    s3_ids = load_source_ids(
        S3_NORMALIZED,
        "Source 3",
    )

    # ========================================================================
    # CHECK ID OVERLAP
    # ========================================================================

    print_section("CHECKING SOURCE ID OVERLAP")

    overlapping_ids = (
        s2_ids
        .join(
            s3_ids,
            on="matched_entity_id",
            how="inner",
        )
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )

    print(
        f"Entity IDs appearing in BOTH S2 and S3: "
        f"{overlapping_ids:,}"
    )

    if overlapping_ids > 0:
        print()
        print(
            "WARNING: Some entity IDs exist in both S2 and S3."
        )
        print(
            "Those IDs may belong to both source namespaces."
        )

    # ========================================================================
    # SPLIT GROUND TRUTH
    # ========================================================================

    print_section("SPLITTING GROUND TRUTH BY SOURCE")

    gt_s2, gt_s3 = split_ground_truth_by_source(
        gt_pairs,
        s2_ids,
        s3_ids,
    )

    s2_true_count = (
        gt_s2
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )

    s3_true_count = (
        gt_s3
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )

    print(f"S1 -> S2 true pairs: {s2_true_count:,}")
    print(f"S1 -> S3 true pairs: {s3_true_count:,}")
    print(
        f"S2 + S3:              "
        f"{s2_true_count + s3_true_count:,}"
    )
    print(
        f"Original GT pairs:     "
        f"{total_pairs:,}"
    )

    # ========================================================================
    # EVALUATE
    # ========================================================================

    result_s2 = evaluate_recall(
        gt_s2,
        S2_CANDIDATES,
        "S2",
    )

    result_s3 = evaluate_recall(
        gt_s3,
        S3_CANDIDATES,
        "S3",
    )

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    print_section("FINAL CANDIDATE RECALL")

    print()
    print("S1 -> S2")
    print(f"  True pairs      : {result_s2['true_pairs']:,}")
    print(f"  Retrieved       : {result_s2['retrieved_pairs']:,}")
    print(f"  Missed          : {result_s2['missed_pairs']:,}")
    print(f"  Recall          : {result_s2['candidate_recall']:.4%}")

    print()
    print("S1 -> S3")
    print(f"  True pairs      : {result_s3['true_pairs']:,}")
    print(f"  Retrieved       : {result_s3['retrieved_pairs']:,}")
    print(f"  Missed          : {result_s3['missed_pairs']:,}")
    print(f"  Recall          : {result_s3['candidate_recall']:.4%}")

    # ========================================================================
    # OVERALL
    # ========================================================================

    total_true = (
        result_s2["true_pairs"]
        + result_s3["true_pairs"]
    )

    total_retrieved = (
        result_s2["retrieved_pairs"]
        + result_s3["retrieved_pairs"]
    )

    total_missed = total_true - total_retrieved

    overall_recall = (
        total_retrieved / total_true
        if total_true > 0
        else 0.0
    )

    print()
    print("OVERALL")
    print(f"  True pairs      : {total_true:,}")
    print(f"  Retrieved       : {total_retrieved:,}")
    print(f"  Missed          : {total_missed:,}")
    print(f"  Recall          : {overall_recall:.4%}")

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    report = {
        "stage": "04_candidate_recall",
        "status": "complete",
        "ground_truth": {
            "total_pairs": int(total_pairs),
            "s2_true_pairs": int(s2_true_count),
            "s3_true_pairs": int(s3_true_count),
        },
        "source_id_overlap": {
            "overlapping_ids": int(overlapping_ids),
        },
        "s1_to_s2": result_s2,
        "s1_to_s3": result_s3,
        "overall": {
            "true_pairs": int(total_true),
            "retrieved_pairs": int(total_retrieved),
            "missed_pairs": int(total_missed),
            "candidate_recall": float(overall_recall),
        },
        "runtime_seconds": time.time() - start_time,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(f"Report:")
    print(REPORT_PATH)

    print()
    print(
        f"Runtime: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

    print()
    print("#" * 80)
    print("# STAGE 04 COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()