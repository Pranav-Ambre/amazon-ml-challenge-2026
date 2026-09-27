"""
Stage 04 — Candidate Recall Evaluation

Purpose
-------
Measure how many true ground-truth matches are present in the
candidate sets generated during Stage 03.

Evaluates:

    TRAIN S1 -> S2
    TRAIN S1 -> S3

Ground truth format:

    source1_entity_id
    matched_entity_ids

where matched_entity_ids may contain:

    ID1
    ID1,ID2
    ID1,ID2,ID3

Empty matched_entity_ids means zero matches.

Outputs
-------
artifacts/reports/candidate_recall_report.json

artifacts/reports/candidate_recall_misses/
    s1_s2_missed_examples.parquet
    s1_s3_missed_examples.parquet
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from code.business_entity_resolution.src.data_loader import (
    load_train_ground_truth,
)


# ============================================================
# PATHS
# ============================================================

CANDIDATE_DIR = (
    ROOT
    / "artifacts"
    / "candidates"
)

REPORT_DIR = (
    ROOT
    / "artifacts"
    / "reports"
)

MISS_DIR = (
    REPORT_DIR
    / "candidate_recall_misses"
)


# ============================================================
# Candidate files
# ============================================================

S2_CANDIDATES = (
    CANDIDATE_DIR
    / "train_s1_s2_candidates.parquet"
)

S3_CANDIDATES = (
    CANDIDATE_DIR
    / "train_s1_s3_candidates.parquet"
)


# ============================================================
# Helper: validate files
# ============================================================

def validate_input_files() -> None:

    print()
    print("=" * 80)
    print("VALIDATING INPUT FILES")
    print("=" * 80)

    files = {
        "S2 candidates": S2_CANDIDATES,
        "S3 candidates": S3_CANDIDATES,
    }

    for name, path in files.items():

        print()
        print(f"{name}:")
        print(f"  {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"\nRequired file does not exist:\n{path}"
            )

        size_gb = (
            path.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"  Size: {size_gb:.2f} GB"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MISS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Ground truth preparation
# ============================================================

def prepare_ground_truth():
    """
    Load and explode the training ground truth.

    Returns
    -------
    matched_gt:
        DataFrame containing one true pair per row.

    full_gt:
        Complete ground truth including zero-match entities.
    """

    print()
    print("=" * 80)
    print("LOADING GROUND TRUTH")
    print("=" * 80)

    gt = load_train_ground_truth()

    print(
        f"Ground truth rows: {len(gt):,}"
    )

    # --------------------------------------------------------
    # Validate expected columns.
    # --------------------------------------------------------

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = (
        required
        - set(gt.columns)
    )

    if missing:
        raise ValueError(
            "Ground truth is missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Keep required columns only.
    # --------------------------------------------------------

    gt = gt[
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    ].copy()

    # --------------------------------------------------------
    # Normalize IDs.
    # --------------------------------------------------------

    gt["source1_entity_id"] = (
        gt["source1_entity_id"]
        .astype("string")
        .str.strip()
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Count zero-match S1 entities.
    # --------------------------------------------------------

    zero_match_mask = (
        gt["matched_entity_ids"]
        == ""
    )

    zero_match_count = int(
        zero_match_mask.sum()
    )

    print(
        f"Zero-match S1 entities: "
        f"{zero_match_count:,}"
    )

    # --------------------------------------------------------
    # Keep only rows with matches.
    # --------------------------------------------------------

    matched = gt[
        ~zero_match_mask
    ].copy()

    # --------------------------------------------------------
    # Split comma-separated target IDs.
    # --------------------------------------------------------

    matched["candidate_entity_id"] = (
        matched["matched_entity_ids"]
        .str.split(",")
    )

    # --------------------------------------------------------
    # Explode to one true pair per row.
    # --------------------------------------------------------

    matched = matched.explode(
        "candidate_entity_id",
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Clean candidate IDs.
    # --------------------------------------------------------

    matched["candidate_entity_id"] = (
        matched["candidate_entity_id"]
        .astype("string")
        .str.strip()
    )

    # Remove accidental empty IDs.
    matched = matched[
        matched["candidate_entity_id"]
        != ""
    ]

    # --------------------------------------------------------
    # Keep only pair columns.
    # --------------------------------------------------------

    matched = matched[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ]

    # --------------------------------------------------------
    # Remove duplicate ground-truth pairs.
    # --------------------------------------------------------

    matched = matched.drop_duplicates(
        ignore_index=True
    )

    print(
        f"True matched pairs: "
        f"{len(matched):,}"
    )

    return matched, gt


# ============================================================
# Evaluate one candidate file
# ============================================================

def evaluate_candidate_file(
    candidate_path: Path,
    ground_truth: pd.DataFrame,
    source_name: str,
):
    """
    Compare ground-truth pairs against one candidate Parquet.

    Important:
        The candidate file is NOT loaded into pandas.

        Polars scans only the two columns required:

            source1_entity_id
            candidate_entity_id
    """

    print()
    print("=" * 80)
    print(f"EVALUATING {source_name}")
    print("=" * 80)

    print(
        f"Candidate file:\n"
        f"{candidate_path}"
    )

    # --------------------------------------------------------
    # Ground truth → Polars LazyFrame
    # --------------------------------------------------------

    gt_lazy = (
        pl.from_pandas(
            ground_truth
        )
        .lazy()
    )

    # --------------------------------------------------------
    # Candidate file → LazyFrame
    #
    # Only two columns are read.
    # --------------------------------------------------------

    candidates_lazy = (
        pl.scan_parquet(
            candidate_path,
            low_memory=True,
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .unique()
    )

    # --------------------------------------------------------
    # Retrieve true pairs that exist in candidates.
    #
    # Everything remains LazyFrame until collect().
    # --------------------------------------------------------

    retrieved_lazy = (
        gt_lazy
        .join(
            candidates_lazy,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    # --------------------------------------------------------
    # Materialize only retrieved TRUE pairs.
    #
    # This should be at most the number of true pairs,
    # not the entire candidate set.
    # --------------------------------------------------------

    retrieved_df = (
        retrieved_lazy
        .collect(
            engine="streaming"
        )
    )

    retrieved_count = len(
        retrieved_df
    )

    true_count = len(
        ground_truth
    )

    # --------------------------------------------------------
    # Candidate recall.
    # --------------------------------------------------------

    if true_count > 0:
        recall = (
            retrieved_count
            / true_count
        )
    else:
        recall = 0.0

    # --------------------------------------------------------
    # Find missed ground-truth pairs.
    #
    # Convert retrieved DataFrame back to LazyFrame so both
    # sides of the join are LazyFrames.
    # --------------------------------------------------------

    missed_lazy = (
        gt_lazy
        .join(
            retrieved_df.lazy(),
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="anti",
        )
    )

    missed_df = (
        missed_lazy
        .collect(
            engine="streaming"
        )
    )

    missed_count = len(
        missed_df
    )

    # --------------------------------------------------------
    # Results.
    # --------------------------------------------------------

    print()
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
        f"{recall * 100:.4f}%"
    )

    return {
        "source": source_name,
        "true_pairs": true_count,
        "retrieved_pairs": retrieved_count,
        "missed_pairs": missed_count,
        "candidate_recall": recall,
        "missed_examples": missed_df.head(100),
    }


# ============================================================
# Save missed examples
# ============================================================

def save_missed_examples(
    result,
    filename: str,
) -> None:

    path = (
        MISS_DIR
        / filename
    )

    result["missed_examples"].write_parquet(
        path,
        compression="zstd",
    )

    print(
        f"Missed examples saved:\n"
        f"{path}"
    )


# ============================================================
# Save JSON report
# ============================================================

def save_report(
    s2_result,
    s3_result,
) -> Path:

    report = {
        "stage": "04_candidate_recall",

        "s1_s2": {
            "true_pairs": s2_result[
                "true_pairs"
            ],

            "retrieved_pairs": s2_result[
                "retrieved_pairs"
            ],

            "missed_pairs": s2_result[
                "missed_pairs"
            ],

            "candidate_recall": s2_result[
                "candidate_recall"
            ],

            "candidate_recall_percent": (
                s2_result[
                    "candidate_recall"
                ]
                * 100
            ),
        },

        "s1_s3": {
            "true_pairs": s3_result[
                "true_pairs"
            ],

            "retrieved_pairs": s3_result[
                "retrieved_pairs"
            ],

            "missed_pairs": s3_result[
                "missed_pairs"
            ],

            "candidate_recall": s3_result[
                "candidate_recall"
            ],

            "candidate_recall_percent": (
                s3_result[
                    "candidate_recall"
                ]
                * 100
            ),
        },
    }

    path = (
        REPORT_DIR
        / "candidate_recall_report.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    return path


# ============================================================
# Main
# ============================================================

def main():

    start = time.perf_counter()

    print()
    print("#" * 80)
    print("# STAGE 04 — CANDIDATE RECALL EVALUATION")
    print("#" * 80)

    # --------------------------------------------------------
    # Validate candidate files.
    # --------------------------------------------------------

    validate_input_files()

    # --------------------------------------------------------
    # Load ground truth.
    # --------------------------------------------------------

    matched_gt, full_gt = (
        prepare_ground_truth()
    )

    # --------------------------------------------------------
    # Evaluate S1 -> S2.
    # --------------------------------------------------------

    s2_result = evaluate_candidate_file(
        candidate_path=S2_CANDIDATES,
        ground_truth=matched_gt,
        source_name="S1 -> S2",
    )

    # --------------------------------------------------------
    # Evaluate S1 -> S3.
    # --------------------------------------------------------

    s3_result = evaluate_candidate_file(
        candidate_path=S3_CANDIDATES,
        ground_truth=matched_gt,
        source_name="S1 -> S3",
    )

    # --------------------------------------------------------
    # Save missed examples.
    # --------------------------------------------------------

    save_missed_examples(
        s2_result,
        "s1_s2_missed_examples.parquet",
    )

    save_missed_examples(
        s3_result,
        "s1_s3_missed_examples.parquet",
    )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    report_path = save_report(
        s2_result,
        s3_result,
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("#" * 80)
    print("# FINAL CANDIDATE RECALL")
    print("#" * 80)

    print()
    print(
        "S1 -> S2"
    )

    print(
        f"  True pairs      : "
        f"{s2_result['true_pairs']:,}"
    )

    print(
        f"  Retrieved       : "
        f"{s2_result['retrieved_pairs']:,}"
    )

    print(
        f"  Missed          : "
        f"{s2_result['missed_pairs']:,}"
    )

    print(
        f"  Recall          : "
        f"{s2_result['candidate_recall'] * 100:.4f}%"
    )

    print()
    print(
        "S1 -> S3"
    )

    print(
        f"  True pairs      : "
        f"{s3_result['true_pairs']:,}"
    )

    print(
        f"  Retrieved       : "
        f"{s3_result['retrieved_pairs']:,}"
    )

    print(
        f"  Missed          : "
        f"{s3_result['missed_pairs']:,}"
    )

    print(
        f"  Recall          : "
        f"{s3_result['candidate_recall'] * 100:.4f}%"
    )

    print()
    print(
        f"Report:\n"
        f"{report_path}"
    )

    print()
    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("#" * 80)
    print("# STAGE 04 COMPLETE")
    print("#" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()