"""
Stage 03 - Candidate Generation

Generates candidate pairs for:

    Train:
        Source 1 -> Source 2
        Source 1 -> Source 3

    Test:
        Source 1 -> Source 2
        Source 1 -> Source 3

Candidate generation uses the memory-safe blocking implementation
from:

    code/business_entity_resolution/src/blocking.py

Pipeline:

    Normalized Parquet
            |
            v
    Blocking rules
            |
            v
    Candidate pairs
            |
            v
    Deduplicated candidate parquet
            |
            v
    Stage 04 candidate recall evaluation

IMPORTANT:
- No all-pairs Cartesian join.
- Candidate generation is performed block-by-block.
- Prefix6 and Prefix4 blocking use their corrected blocking keys.
- Candidate provenance is preserved.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List


# ============================================================
# PROJECT PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CODE_DIR = ROOT_DIR / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ============================================================
# IMPORTS
# ============================================================

from business_entity_resolution.src.blocking import (
    generate_candidates,
)


# ============================================================
# PATHS
# ============================================================

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"

CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"

REPORTS_DIR = ARTIFACTS_DIR / "reports"


# ============================================================
# INPUT / OUTPUT CONFIGURATION
# ============================================================

JOBS = [
    # --------------------------------------------------------
    # TRAIN S1 -> S2
    # --------------------------------------------------------
    {
        "name": "train_s1_s2",
        "source1": (
            NORMALIZED_DIR
            / "train_source1_normalized.parquet"
        ),
        "source2": (
            NORMALIZED_DIR
            / "train_source2_normalized.parquet"
        ),
        "output": (
            CANDIDATES_DIR
            / "train_s1_s2_candidates.parquet"
        ),
        "candidate_source": "S2",
    },

    # --------------------------------------------------------
    # TRAIN S1 -> S3
    # --------------------------------------------------------
    {
        "name": "train_s1_s3",
        "source1": (
            NORMALIZED_DIR
            / "train_source1_normalized.parquet"
        ),
        "source2": (
            NORMALIZED_DIR
            / "train_source3_normalized.parquet"
        ),
        "output": (
            CANDIDATES_DIR
            / "train_s1_s3_candidates.parquet"
        ),
        "candidate_source": "S3",
    },

    # --------------------------------------------------------
    # TEST S1 -> S2
    # --------------------------------------------------------
    {
        "name": "test_s1_s2",
        "source1": (
            NORMALIZED_DIR
            / "test_source1_normalized.parquet"
        ),
        "source2": (
            NORMALIZED_DIR
            / "test_source2_normalized.parquet"
        ),
        "output": (
            CANDIDATES_DIR
            / "test_s1_s2_candidates.parquet"
        ),
        "candidate_source": "S2",
    },

    # --------------------------------------------------------
    # TEST S1 -> S3
    # --------------------------------------------------------
    {
        "name": "test_s1_s3",
        "source1": (
            NORMALIZED_DIR
            / "test_source1_normalized.parquet"
        ),
        "source2": (
            NORMALIZED_DIR
            / "test_source3_normalized.parquet"
        ),
        "output": (
            CANDIDATES_DIR
            / "test_s1_s3_candidates.parquet"
        ),
        "candidate_source": "S3",
    },
]


# ============================================================
# LOGGING
# ============================================================

LOGGER = logging.getLogger("stage03")


def configure_logging() -> None:
    """
    Configure console logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(path: Path) -> None:
    """
    Validate that a required parquet file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but found something else:\n{path}"
        )


def file_size_gb(path: Path) -> float:
    """
    Return file size in GB.
    """

    return path.stat().st_size / (1024 ** 3)


# ============================================================
# NORMALIZED INPUT VALIDATION
# ============================================================

def validate_normalized_inputs() -> None:
    """
    Validate all normalized input files required by Stage 03.
    """

    print()
    print("=" * 80)
    print("VALIDATING NORMALIZED INPUTS")
    print("=" * 80)

    required_files = sorted(
        {
            job["source1"]
            for job in JOBS
        }
        |
        {
            job["source2"]
            for job in JOBS
        }
    )

    for path in required_files:

        print()
        print("Checking:")
        print(f"  {path}")

        validate_file(path)

        print(
            f"Size: {file_size_gb(path):.2f} GB"
        )

    print()
    print(
        "All normalized input files are available."
    )
    print()


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories() -> None:
    """
    Create Stage 03 output directories.
    """

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# SINGLE JOB EXECUTION
# ============================================================

def run_job(job: Dict) -> Dict:
    """
    Execute one candidate-generation job.

    This function intentionally calls the exact API exposed by
    the corrected blocking.py:

        generate_candidates(
            source1_path=...,
            source2_path=...,
            output_path=...,
            candidate_source=...,
        )
    """

    name = job["name"]

    source1 = Path(job["source1"])
    source2 = Path(job["source2"])
    output = Path(job["output"])
    candidate_source = job["candidate_source"]

    print()
    print("#" * 80)
    print(f"STARTING: {name}")
    print("#" * 80)

    print()
    print("Configuration:")
    print(f"  Source 1 : {source1}")
    print(f"  Source 2 : {source2}")
    print(f"  Source   : {candidate_source}")
    print(f"  Output   : {output}")
    print()

    validate_file(source1)
    validate_file(source2)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    metadata = generate_candidates(
        source1_path=source1,
        source2_path=source2,
        output_path=output,
        candidate_source=candidate_source,
    )

    elapsed = time.time() - start_time

    metadata = dict(metadata)

    metadata["job_name"] = name
    metadata["elapsed_seconds"] = round(
        elapsed,
        2,
    )

    metadata["source1_path"] = str(source1)
    metadata["source2_path"] = str(source2)
    metadata["output_path"] = str(output)

    print()
    print("-" * 80)
    print(f"COMPLETED: {name}")
    print("-" * 80)

    print(
        f"Raw candidates : "
        f"{metadata.get('raw_candidates', 0):,}"
    )

    print(
        f"Final candidates : "
        f"{metadata.get('final_candidates', 0):,}"
    )

    print(
        f"Duplicate reduction : "
        f"{metadata.get('duplicate_reduction_percent', 0.0):.2f}%"
    )

    print(
        f"Runtime : "
        f"{elapsed:.2f} seconds"
    )

    if output.exists():

        print(
            f"Output size : "
            f"{file_size_gb(output):.2f} GB"
        )

    return metadata


# ============================================================
# REPORT WRITER
# ============================================================

def write_report(
    results: List[Dict],
    total_runtime: float,
) -> Path:
    """
    Save Stage 03 metadata report.
    """

    report = {
        "stage": "03_candidate_generation",
        "status": "completed",
        "total_runtime_seconds": round(
            total_runtime,
            2,
        ),
        "jobs": results,
    }

    report_path = (
        REPORTS_DIR
        / "candidate_generation_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    return report_path


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: List[Dict],
    total_runtime: float,
) -> None:
    """
    Print final Stage 03 summary.
    """

    print()
    print()
    print("=" * 80)
    print("STAGE 03 CANDIDATE GENERATION SUMMARY")
    print("=" * 80)

    print()

    for result in results:

        print(
            f"{result['job_name']:<20}"
            f" | "
            f"raw={result.get('raw_candidates', 0):>15,}"
            f" | "
            f"final={result.get('final_candidates', 0):>15,}"
            f" | "
            f"time={result.get('elapsed_seconds', 0):>8.2f}s"
        )

    print()
    print(
        f"Total runtime: "
        f"{total_runtime:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )

    print()
    print(
        "Candidate files:"
    )

    for result in results:

        output = Path(
            result["output_path"]
        )

        if output.exists():

            print(
                f"  {output}"
            )

    print()
    print(
        "Report:"
    )

    print(
        f"  {REPORTS_DIR / 'candidate_generation_report.json'}"
    )

    print()
    print("=" * 80)
    print("STAGE 03 COMPLETE")
    print("=" * 80)
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run Stage 03 candidate generation.
    """

    configure_logging()

    stage_start = time.time()

    create_directories()

    validate_normalized_inputs()

    results: List[Dict] = []

    failed_jobs: List[str] = []

    # --------------------------------------------------------
    # Run jobs sequentially.
    #
    # DO NOT parallelize these jobs.
    #
    # Candidate generation is memory-intensive and the AWS
    # instance has finite RAM.
    # --------------------------------------------------------

    for job in JOBS:

        try:

            result = run_job(job)

            results.append(result)

        except Exception as exc:

            LOGGER.exception(
                "Candidate generation failed for %s",
                job["name"],
            )

            print()
            print(
                f"ERROR: {job['name']} failed."
            )

            print(
                f"Reason: {exc}"
            )

            failed_jobs.append(
                job["name"]
            )

            # ------------------------------------------------
            # Stop immediately.
            #
            # If training candidate generation fails, continuing
            # to test generation would waste resources and could
            # leave the pipeline in a partially valid state.
            # ------------------------------------------------

            raise

    total_runtime = (
        time.time() - stage_start
    )

    report_path = write_report(
        results,
        total_runtime,
    )

    print_summary(
        results,
        total_runtime,
    )

    if failed_jobs:

        raise RuntimeError(
            "Stage 03 failed jobs: "
            + ", ".join(failed_jobs)
        )

    print(
        f"Stage 03 report written to:\n"
        f"{report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()