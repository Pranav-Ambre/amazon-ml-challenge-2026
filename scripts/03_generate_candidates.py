"""
Full-dataset candidate generation.

Processes:

    TRAIN:
        S1 → S2
        S1 → S3

    TEST:
        S1 → S2
        S1 → S3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from code.business_entity_resolution.src.blocking import (
    generate_candidates,
)


# ============================================================
# Paths
# ============================================================

NORMALIZED_DIR = (
    ROOT
    / "artifacts"
    / "normalized"
)

CANDIDATE_DIR = (
    ROOT
    / "artifacts"
    / "candidates"
)

CANDIDATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Complete dataset
# ============================================================

DATASETS = [

    # --------------------------------------------------------
    # TRAIN S1 → S2
    # --------------------------------------------------------

    {
        "name": "TRAIN S1 -> S2",

        "source1": (
            NORMALIZED_DIR
            / "train_source1_normalized.parquet"
        ),

        "source2": (
            NORMALIZED_DIR
            / "train_source2_normalized.parquet"
        ),

        "output": (
            CANDIDATE_DIR
            / "train_s1_s2_candidates.parquet"
        ),
    },

    # --------------------------------------------------------
    # TRAIN S1 → S3
    # --------------------------------------------------------

    {
        "name": "TRAIN S1 -> S3",

        "source1": (
            NORMALIZED_DIR
            / "train_source1_normalized.parquet"
        ),

        "source2": (
            NORMALIZED_DIR
            / "train_source3_normalized.parquet"
        ),

        "output": (
            CANDIDATE_DIR
            / "train_s1_s3_candidates.parquet"
        ),
    },

    # --------------------------------------------------------
    # TEST S1 → S2
    # --------------------------------------------------------

    {
        "name": "TEST S1 -> S2",

        "source1": (
            NORMALIZED_DIR
            / "test_source1_normalized.parquet"
        ),

        "source2": (
            NORMALIZED_DIR
            / "test_source2_normalized.parquet"
        ),

        "output": (
            CANDIDATE_DIR
            / "test_s1_s2_candidates.parquet"
        ),
    },

    # --------------------------------------------------------
    # TEST S1 → S3
    # --------------------------------------------------------

    {
        "name": "TEST S1 -> S3",

        "source1": (
            NORMALIZED_DIR
            / "test_source1_normalized.parquet"
        ),

        "source2": (
            NORMALIZED_DIR
            / "test_source3_normalized.parquet"
        ),

        "output": (
            CANDIDATE_DIR
            / "test_s1_s3_candidates.parquet"
        ),
    },
]


# ============================================================
# Validation
# ============================================================

def validate_inputs():

    print("\nChecking normalized files...\n")

    for dataset in DATASETS:

        source1 = dataset["source1"]
        source2 = dataset["source2"]

        if not source1.exists():
            raise FileNotFoundError(
                f"Missing S1 file:\n{source1}"
            )

        if not source2.exists():
            raise FileNotFoundError(
                f"Missing candidate file:\n{source2}"
            )

        print(f"✓ {source1.name}")
        print(f"✓ {source2.name}")


# ============================================================
# Main
# ============================================================

def main():

    validate_inputs()

    overall_start = time.perf_counter()

    print()
    print("#" * 80)
    print("# FULL DATASET CANDIDATE GENERATION")
    print("#" * 80)

    for dataset in DATASETS:

        print()
        print("=" * 80)
        print(dataset["name"])
        print("=" * 80)

        start = time.perf_counter()

        generate_candidates(
            source1_path=dataset["source1"],
            source2_path=dataset["source2"],
            output_path=dataset["output"],
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"{dataset['name']} completed "
            f"in {elapsed / 60:.2f} minutes"
        )

    total_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("#" * 80)
    print(
        "ALL CANDIDATE GENERATION COMPLETED"
    )
    print(
        f"Total time: "
        f"{total_elapsed / 3600:.2f} hours"
    )
    print("#" * 80)


if __name__ == "__main__":
    main()