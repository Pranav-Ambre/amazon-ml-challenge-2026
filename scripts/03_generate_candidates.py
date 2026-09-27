#!/usr/bin/env python3

"""
Amazon ML Challenge 2026
Stage 03 — Full Dataset Candidate Generation

Uses the production blocking implementation from:

code/business_entity_resolution/src/blocking.py

Current generate_candidates() API:

    generate_candidates(
        source1="S1",
        candidate_source="S2",
        output_path=...
    )

Training candidate generation:
    S1 -> S2
    S1 -> S3

Test candidate generation is intentionally NOT included here because
the current blocking implementation resolves normalized files using
the train_* naming convention. Test blocking should be added only
after the blocking module is made test-aware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT BLOCKING ENGINE
# ============================================================

from code.business_entity_resolution.src.blocking import (
    generate_candidates,
)


# ============================================================
# DIRECTORIES
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
# TRAINING JOBS
# ============================================================

JOBS = [
    {
        "name": "train_s1_s2",
        "source1": "S1",
        "candidate_source": "S2",
        "output": (
            CANDIDATES_DIR
            / "train_s1_s2_candidates.parquet"
        ),
    },
    {
        "name": "train_s1_s3",
        "source1": "S1",
        "candidate_source": "S3",
        "output": (
            CANDIDATES_DIR
            / "train_s1_s3_candidates.parquet"
        ),
    },
]


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs() -> None:

    print()
    print("=" * 80)
    print("CHECKING NORMALIZED INPUT FILES")
    print("=" * 80)

    required = [
        NORMALIZED_DIR / "train_source1_normalized.parquet",
        NORMALIZED_DIR / "train_source2_normalized.parquet",
        NORMALIZED_DIR / "train_source3_normalized.parquet",
    ]

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                f"\nMissing normalized input:\n{path}\n"
            )

        size_gb = (
            path.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"✓ {path.name}"
            f" ({size_gb:.2f} GB)"
        )


# ============================================================
# RUN ONE JOB
# ============================================================

def run_job(job: dict) -> dict:

    name = job["name"]
    source1 = job["source1"]
    candidate_source = job["candidate_source"]
    output_path = job["output"]

    print()
    print("#" * 80)
    print(f"# {name.upper()}")
    print("#" * 80)

    print()
    print(f"Source 1:        {source1}")
    print(f"Candidate source:{candidate_source}")
    print(f"Output:          {output_path}")

    start = time.perf_counter()

    result = generate_candidates(
        source1=source1,
        candidate_source=candidate_source,
        output_path=output_path,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("-" * 80)
    print(f"{name} COMPLETE")
    print("-" * 80)

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )

    if isinstance(result, dict):

        if "unique_candidate_pairs" in result:
            print(
                "Candidate pairs: "
                f"{result['unique_candidate_pairs']:,}"
            )

        if "output_size_mb" in result:
            print(
                "Output size: "
                f"{result['output_size_mb']:.2f} MB"
            )

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "result": result,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("STAGE 03 — FULL DATASET CANDIDATE GENERATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate normalized data
    # --------------------------------------------------------

    validate_inputs()

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall timer
    # --------------------------------------------------------

    overall_start = time.perf_counter()

    results = []

    # --------------------------------------------------------
    # Run jobs sequentially
    # --------------------------------------------------------

    for job in JOBS:

        result = run_job(job)

        results.append(result)

        print()
        print(
            "Waiting before next candidate-generation job..."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print()
    print("#" * 80)
    print("# STAGE 03 COMPLETE")
    print("#" * 80)

    print()

    for result in results:

        print(
            f"{result['name']:20s} "
            f"{result['elapsed_seconds'] / 60:8.2f} min"
        )

    print("-" * 80)

    print(
        f"{'TOTAL':20s} "
        f"{total_elapsed / 60:8.2f} min"
    )

    print()
    print("Generated outputs:")

    for job in JOBS:

        output = job["output"]

        metadata = output.with_suffix(
            ".metadata.json"
        )

        print(
            f"  Candidate: {output}"
        )

        print(
            f"  Metadata:  {metadata}"
        )

    print()
    print("=" * 80)
    print("NEXT STEP: INSPECT CANDIDATE COUNTS AND OUTPUT SIZES")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()