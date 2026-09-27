"""
Full-dataset memory-safe candidate generation.

Every blocking rule is written independently.
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
# Directories
# ============================================================

NORMALIZED_DIR = (
    ROOT
    / "artifacts"
    / "normalized"
)

CANDIDATES_DIR = (
    ROOT
    / "artifacts"
    / "candidates"
)


# ============================================================
# Dataset jobs
# ============================================================

JOBS = [

    (
        "train_s1_s2",

        NORMALIZED_DIR
        / "train_source1_normalized.parquet",

        NORMALIZED_DIR
        / "train_source2_normalized.parquet",
    ),

    (
        "train_s1_s3",

        NORMALIZED_DIR
        / "train_source1_normalized.parquet",

        NORMALIZED_DIR
        / "train_source3_normalized.parquet",
    ),

    (
        "test_s1_s2",

        NORMALIZED_DIR
        / "test_source1_normalized.parquet",

        NORMALIZED_DIR
        / "test_source2_normalized.parquet",
    ),

    (
        "test_s1_s3",

        NORMALIZED_DIR
        / "test_source1_normalized.parquet",

        NORMALIZED_DIR
        / "test_source3_normalized.parquet",
    ),
]


# ============================================================
# Validate
# ============================================================

def validate_files():

    print()
    print("=" * 80)
    print("CHECKING INPUT FILES")
    print("=" * 80)

    for name, s1, s2 in JOBS:

        if not s1.exists():
            raise FileNotFoundError(
                f"Missing:\n{s1}"
            )

        if not s2.exists():
            raise FileNotFoundError(
                f"Missing:\n{s2}"
            )

        print(f"✓ {name}")


# ============================================================
# Main
# ============================================================

def main():

    validate_files()

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_start = time.perf_counter()

    print()
    print("#" * 80)
    print("# MEMORY-SAFE FULL DATASET CANDIDATE GENERATION")
    print("#" * 80)

    for name, s1, s2 in JOBS:

        print()
        print("=" * 80)
        print(name.upper())
        print("=" * 80)

        start = time.perf_counter()

        output_dir = (
            CANDIDATES_DIR
            / name
        )

        generate_candidates(
            source1_path=s1,
            source2_path=s2,
            output_dir=output_dir,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"{name} completed in "
            f"{elapsed / 60:.2f} minutes"
        )

    total = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("#" * 80)
    print("FULL CANDIDATE GENERATION COMPLETED")
    print(
        f"Total time: {total / 3600:.2f} hours"
    )
    print("#" * 80)


if __name__ == "__main__":
    main()