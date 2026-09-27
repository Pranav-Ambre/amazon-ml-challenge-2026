#!/usr/bin/env python3

"""
Amazon ML Challenge 2026
Stage 03 — Training Candidate Generation

Generates:

    Train Source 1 -> Train Source 2
    Train Source 1 -> Train Source 3

Input:
    artifacts/normalized/

Output:
    artifacts/candidates/

Large Parquet files stay on EC2.
Compact metadata/report files can be committed to GitHub.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

# Add repository code directory to Python import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from business_entity_resolution.src.blocking import (
    generate_candidates,
)


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DIRECTORIES
# ============================================================

NORMALIZED_DIR = (
    ROOT / "artifacts" / "normalized"
)

CANDIDATE_DIR = (
    ROOT / "artifacts" / "candidates"
)

REPORT_DIR = (
    ROOT / "artifacts" / "reports"
)


CANDIDATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NORMALIZED INPUT FILES
# ============================================================

TRAIN_SOURCE1 = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

TRAIN_SOURCE2 = (
    NORMALIZED_DIR
    / "train_source2_normalized.parquet"
)

TRAIN_SOURCE3 = (
    NORMALIZED_DIR
    / "train_source3_normalized.parquet"
)


# ============================================================
# JOBS
# ============================================================

JOBS = [
    {
        "name": "train_s1_s2",

        "source1":
            TRAIN_SOURCE1,

        "candidate_source":
            TRAIN_SOURCE2,

        "candidate_label":
            "S2",

        "output":
            CANDIDATE_DIR
            / "train_s1_s2_candidates.parquet",
    },

    {
        "name": "train_s1_s3",

        "source1":
            TRAIN_SOURCE1,

        "candidate_source":
            TRAIN_SOURCE3,

        "candidate_label":
            "S3",

        "output":
            CANDIDATE_DIR
            / "train_s1_s3_candidates.parquet",
    },
]


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs() -> None:

    print()
    print("=" * 80)
    print("VALIDATING NORMALIZED INPUTS")
    print("=" * 80)

    for path in [
        TRAIN_SOURCE1,
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print()
        print(
            f"Checking:\n"
            f"  {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired normalized file "
                f"not found:\n"
                f"  {path}\n"
            )

        size_gb = (
            path.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"Size: "
            f"{size_gb:.2f} GB"
        )

        if path.stat().st_size == 0:

            raise ValueError(
                f"Normalized file is empty:\n"
                f"  {path}"
            )

    print()
    print(
        "All normalized training "
        "files are available."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    validate_inputs()

    overall_start = time.time()

    summaries = []

    # ========================================================
    # RUN JOBS SEQUENTIALLY
    # ========================================================

    for job in JOBS:

        print()
        print()
        print("#" * 80)
        print(
            f"STARTING: {job['name']}"
        )
        print("#" * 80)

        start = time.time()

        metadata = generate_candidates(

            source1_path=
                job["source1"],

            candidate_source_path=
                job["candidate_source"],

            output_path=
                job["output"],

            candidate_source=
                job["candidate_label"],
        )

        elapsed = (
            time.time() - start
        )

        summaries.append(
            {
                "job":
                    job["name"],

                "candidate_source":
                    job["candidate_label"],

                "output":
                    str(job["output"]),

                "runtime_seconds":
                    elapsed,

                "metadata":
                    metadata,
            }
        )

        print()
        print(
            f"COMPLETED: "
            f"{job['name']}"
        )

        print(
            f"Runtime: "
            f"{elapsed:.2f} sec"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_elapsed = (
        time.time()
        - overall_start
    )

    report = {

        "stage":
            "03_candidate_generation",

        "jobs":
            summaries,

        "total_runtime_seconds":
            total_elapsed,
    }

    report_path = (
        REPORT_DIR
        / "candidate_generation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 03 COMPLETE")
    print("=" * 80)

    print(
        f"Total runtime: "
        f"{total_elapsed:.2f} sec"
    )

    print()
    print(
        f"Report:"
    )

    print(
        f"  {report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
