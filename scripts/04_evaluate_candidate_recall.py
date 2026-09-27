"""
Stage 04 — Candidate Recall Evaluation

Measures how many ground-truth matches are present inside
the generated candidate sets.

This answers:

    "Can our blocking stage retrieve the true matches?"

We evaluate S1->S2 and S1->S3 separately.

No candidate regeneration is performed.
"""

from __future__ import annotations

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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# Ground truth preparation
# ============================================================

def prepare_ground_truth():
    """
    Load and explode ground truth.

    Ground truth format:

        source1_entity_id
        matched_entity_ids

    matched_entity_ids can contain:

        ID1
        ID1,ID2
        ID1,ID2,ID3

    Empty values represent zero-match entities.
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
    # Normalize column names.
    # --------------------------------------------------------

    gt = gt[
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    ].copy()

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
    # Keep only entities having at least one match.
    #
    # Zero-match entities are evaluated separately.
    # --------------------------------------------------------

    matched = gt[
        gt["matched_entity_ids"]
        .notna()
        & (
            gt["matched_entity_ids"]
            != ""
        )
    ].copy()

    # --------------------------------------------------------
    # Explode comma-separated IDs.
    # --------------------------------------------------------

    matched["candidate_entity_id"] = (
        matched["matched_entity_ids"]
        .str.split(",")
    )

    matched = matched.explode(
        "candidate_entity_id",
        ignore_index=True,
    )

    matched["candidate_entity_id"] = (
        matched["candidate_entity_id"]
        .astype("string")
        .str.strip()
    )

    matched = matched[
        matched["candidate_entity_id"]
        != ""
    ]

    matched = matched[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].drop_duplicates()

    print(
        f"True matched pairs: "
        f"{len(matched):,}"
    )

    print(
        f"Zero-match S1 entities: "
        f"{(
            gt['matched_entity_ids'] == ''
        ).sum():,}"
    )

    return matched, gt


# ============================================================
# Candidate recall
# ============================================================

def evaluate_candidate_file(
    candidate_path: Path,
    ground_truth: pd.DataFrame,
    source_name: str,
):
    """
    Evaluate candidate recall.

    We only need:

        source1_entity_id
        candidate_entity_id

    from the candidate Parquet.
    """

    print()
    print("=" * 80)
    print(f"EVALUATING {source_name}")
    print("=" * 80)

    print(
        f"Candidate file:\n{candidate_path}"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{candidate_path}"
        )

    # --------------------------------------------------------
    # Convert ground truth to Polars.
    # --------------------------------------------------------

    gt_pl = pl.from_pandas(
        ground_truth,
    )

    # --------------------------------------------------------
    # Lazy candidate scan.
    #
    # We only read two columns.
    # --------------------------------------------------------

    candidates = (
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
    # True pairs that appear in candidates.
    # --------------------------------------------------------

    retrieved = (
        gt_pl.lazy()
        .join(
            candidates,
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="inner",
        )
        .unique()
        .collect(
            engine="streaming",
        )
    )

    retrieved_count = len(retrieved)

    true_count = len(
        ground_truth
    )

    recall = (
        retrieved_count / true_count
        if true_count
        else 0.0
    )

    missed = (
        gt_pl
        .join(
            retrieved.lazy(),
            on=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            how="anti",
        )
        .collect(
            engine="streaming",
        )
    )

    missed_count = len(missed)

    print()
    print(
        f"True pairs       : {true_count:,}"
    )

    print(
        f"Retrieved pairs  : {retrieved_count:,}"
    )

    print(
        f"Missed pairs     : {missed_count:,}"
    )

    print(
        f"Candidate recall : {recall * 100:.4f}%"
    )

    return {
        "source": source_name,
        "true_pairs": true_count,
        "retrieved_pairs": retrieved_count,
        "missed_pairs": missed_count,
        "candidate_recall": recall,
        "missed_examples": missed.head(100),
    }


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
    # Ground truth
    # --------------------------------------------------------

    matched_gt, full_gt = (
        prepare_ground_truth()
    )

    # --------------------------------------------------------
    # S1 -> S2
    # --------------------------------------------------------

    s2_result = evaluate_candidate_file(
        candidate_path=S2_CANDIDATES,
        ground_truth=matched_gt,
        source_name="S1 -> S2",
    )

    # --------------------------------------------------------
    # S1 -> S3
    # --------------------------------------------------------

    s3_result = evaluate_candidate_file(
        candidate_path=S3_CANDIDATES,
        ground_truth=matched_gt,
        source_name="S1 -> S3",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    results = {
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
        },
    }

    # --------------------------------------------------------
    # Save JSON-compatible report.
    # --------------------------------------------------------

    import json

    report_path = (
        REPORT_DIR
        / "candidate_recall_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save missed examples.
    # --------------------------------------------------------

    missed_dir = (
        REPORT_DIR
        / "candidate_recall_misses"
    )

    missed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    s2_result[
        "missed_examples"
    ].write_parquet(
        missed_dir
        / "s1_s2_missed_examples.parquet"
    )

    s3_result[
        "missed_examples"
    ].write_parquet(
        missed_dir
        / "s1_s3_missed_examples.parquet"
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("#" * 80)
    print("# FINAL RECALL REPORT")
    print("#" * 80)

    print(
        f"S1 -> S2 recall: "
        f"{s2_result['candidate_recall'] * 100:.4f}%"
    )

    print(
        f"S1 -> S3 recall: "
        f"{s3_result['candidate_recall'] * 100:.4f}%"
    )

    print()
    print(
        f"Report saved to:\n{report_path}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print("#" * 80)


if __name__ == "__main__":
    main()