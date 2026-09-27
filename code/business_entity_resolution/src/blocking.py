#!/usr/bin/env python3

"""
Amazon ML Challenge 2026
Stage 04 — Production Candidate Blocking

Pipeline:
    Normalized Parquet
        ↓
    Blocking keys
        ↓
    6 blocking rules
        ↓
    Candidate pair generation
        ↓
    Union
        ↓
    Pair deduplication
        ↓
    Block support count
        ↓
    Candidate Parquet

Designed for:
    8 vCPU
    64 GB RAM
    Polars
    Parquet

Output:
    artifacts/candidates/train_s1_s2_candidates.parquet
    artifacts/candidates/train_s1_s3_candidates.parquet
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import polars as pl
import psutil


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = PROJECT_ROOT / "artifacts" / "normalized"
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "candidates"

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

ENTITY_COL = "entity_id"

BLOCKS = [
    {
        "key": "_k_exact_name",
        "name": "exact_name",
        "maximum_bucket": 5000,
    },
    {
        "key": "_k_exact_address",
        "name": "exact_address",
        "maximum_bucket": 3000,
    },
    {
        "key": "_k_sorted_name",
        "name": "sorted_name",
        "maximum_bucket": 1000,
    },
    {
        "key": "_k_name_prefix6",
        "name": "name_prefix6",
        "maximum_bucket": 500,
    },
    {
        "key": "_k_name_prefix4",
        "name": "name_prefix4",
        "maximum_bucket": 250,
    },
    {
        "key": "_k_numeric_address",
        "name": "numeric_address",
        "maximum_bucket": 300,
    },
]


# ============================================================
# RESOURCE MONITORING
# ============================================================

PROCESS = psutil.Process(os.getpid())


def process_ram_gb() -> float:
    """Return current Python process RSS in GiB."""
    return PROCESS.memory_info().rss / (1024 ** 3)


def system_ram_gb() -> tuple[float, float]:
    """Return used and available system RAM in GiB."""
    mem = psutil.virtual_memory()

    used = mem.used / (1024 ** 3)
    available = mem.available / (1024 ** 3)

    return used, available


def print_resource_status(prefix: str = "") -> None:
    used, available = system_ram_gb()

    print(
        f"{prefix}"
        f"Process RAM: {process_ram_gb():.2f} GB | "
        f"System used: {used:.2f} GB | "
        f"Available: {available:.2f} GB"
    )


# ============================================================
# DATA LOADING
# ============================================================

def normalized_path(source: str) -> Path:
    path = NORMALIZED_DIR / f"train_{source}_normalized.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized file not found:\n{path}"
        )

    return path


def load_normalized(source: str) -> pl.LazyFrame:
    """
    Lazily load normalized dataset.

    Only columns required by blocking are selected.
    """

    path = normalized_path(source)

    required_columns = [
        ENTITY_COL,
        "_k_exact_name",
        "_k_exact_address",
        "_k_sorted_name",
        "_k_name_prefix6",
        "_k_name_prefix4",
        "_k_numeric_address",
    ]

    print(f"Loading normalized {source}:")
    print(f"  {path}")

    return (
        pl.scan_parquet(path)
        .select(required_columns)
    )


# ============================================================
# BLOCK GENERATION
# ============================================================

def make_block(
    s1: pl.LazyFrame,
    s2: pl.LazyFrame,
    key: str,
    block_name: str,
    maximum_bucket: int,
) -> pl.LazyFrame:
    """
    Generate candidate pairs for one blocking rule.

    Large buckets are discarded to prevent candidate explosion.
    """

    left = (
        s1
        .select(
            [
                pl.col(ENTITY_COL).alias(
                    "source1_entity_id"
                ),
                pl.col(key),
            ]
        )
        .filter(
            pl.col(key).is_not_null()
            & (pl.col(key) != "")
            & (pl.col(key).str.len_chars() > 2)
        )
    )

    right = (
        s2
        .select(
            [
                pl.col(ENTITY_COL).alias(
                    "candidate_entity_id"
                ),
                pl.col(key),
            ]
        )
        .filter(
            pl.col(key).is_not_null()
            & (pl.col(key) != "")
            & (pl.col(key).str.len_chars() > 2)
        )
    )

    # --------------------------------------------------------
    # Safe buckets on S1
    # --------------------------------------------------------

    left_keys = (
        left
        .group_by(key)
        .len(name="_left_count")
        .filter(
            pl.col("_left_count") <= maximum_bucket
        )
        .select(key)
    )

    # --------------------------------------------------------
    # Safe buckets on S2
    # --------------------------------------------------------

    right_keys = (
        right
        .group_by(key)
        .len(name="_right_count")
        .filter(
            pl.col("_right_count") <= maximum_bucket
        )
        .select(key)
    )

    # --------------------------------------------------------
    # Keys must exist on both sides
    # --------------------------------------------------------

    safe_keys = (
        left_keys
        .join(
            right_keys,
            on=key,
            how="inner",
        )
        .select(key)
    )

    # --------------------------------------------------------
    # Reduce both sides before many-to-many join
    # --------------------------------------------------------

    left = (
        left
        .join(
            safe_keys,
            on=key,
            how="inner",
        )
    )

    right = (
        right
        .join(
            safe_keys,
            on=key,
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    return (
        left
        .join(
            right,
            on=key,
            how="inner",
            validate="m:m",
            coalesce=True,
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .with_columns(
            pl.lit(block_name).alias("block")
        )
    )


# ============================================================
# ONE SOURCE PAIR
# ============================================================

def generate_candidates(
    source1: str,
    candidate_source: str,
    output_path: Path,
) -> dict:

    print()
    print("=" * 80)
    print(
        f"BLOCKING: S1 -> {candidate_source}"
    )
    print("=" * 80)

    overall_start = time.time()

    s1 = load_normalized(source1)
    s2 = load_normalized(candidate_source)

    results: list[pl.LazyFrame] = []
    block_stats = []

    # --------------------------------------------------------
    # Individual blocks
    # --------------------------------------------------------

    for config in BLOCKS:

        key = config["key"]
        block_name = config["name"]
        maximum_bucket = config["maximum_bucket"]

        print()
        print("-" * 80)
        print(f"Block: {block_name}")
        print(f"Key: {key}")
        print(f"Maximum bucket: {maximum_bucket:,}")

        print_resource_status("Before: ")

        start = time.time()

        block = make_block(
            s1,
            s2,
            key,
            block_name,
            maximum_bucket,
        )

        # Materialize the block before continuing.
        block_df = block.collect(
            engine="streaming"
        )

        elapsed = time.time() - start
        candidate_count = len(block_df)

        print(
            f"Candidates: {candidate_count:,}"
        )
        print(
            f"Time: {elapsed:.2f} sec"
        )

        print_resource_status("After:  ")

        block_stats.append(
            {
                "block": block_name,
                "key": key,
                "maximum_bucket": maximum_bucket,
                "candidate_count": candidate_count,
                "runtime_seconds": elapsed,
                "process_ram_gb": process_ram_gb(),
            }
        )

        results.append(
            block_df.lazy()
        )

        del block_df
        gc.collect()

    # --------------------------------------------------------
    # Union all blocks
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("UNION + DEDUPLICATION")
    print("=" * 80)

    union_start = time.time()

    all_candidates = pl.concat(
        results,
        how="vertical",
    )

    # --------------------------------------------------------
    # Deduplicate candidate pair
    # --------------------------------------------------------

    final_candidates = (
        all_candidates
        .group_by(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .agg(
            [
                pl.col("block")
                .n_unique()
                .alias("block_support_count"),

                pl.col("block")
                .unique()
                .sort()
                .alias("blocks"),
            ]
        )
        .with_columns(
            pl.lit(candidate_source)
            .alias("candidate_source")
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "block_support_count",
                "blocks",
            ]
        )
    )

    # --------------------------------------------------------
    # Materialize final candidates
    # --------------------------------------------------------

    final_df = final_candidates.collect(
        engine="streaming"
    )

    union_elapsed = time.time() - union_start

    print(
        f"Final unique candidates: "
        f"{len(final_df):,}"
    )

    print(
        f"Union/dedup time: "
        f"{union_elapsed:.2f} sec"
    )

    print_resource_status("After union: ")

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("WRITING CANDIDATE PARQUET")
    print("=" * 80)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(
            f"Removing existing output:\n"
            f"{output_path}"
        )
        output_path.unlink()

    write_start = time.time()

    final_df.write_parquet(
        output_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )

    write_elapsed = time.time() - write_start

    output_size_mb = (
        output_path.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Output size: "
        f"{output_size_mb:,.2f} MB"
    )

    print(
        f"Write time: "
        f"{write_elapsed:.2f} sec"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_elapsed = time.time() - overall_start

    source1_count = (
        s1.select(pl.len())
        .collect()
        .item()
    )

    candidate_count = len(final_df)

    average_candidates = (
        candidate_count / source1_count
        if source1_count
        else 0
    )

    reduction_ratio = (
        source1_count
        if candidate_count == 0
        else (
            source1_count * 1.0
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "stage": "04_blocking",
        "source1": source1,
        "candidate_source": candidate_source,
        "source1_rows": source1_count,
        "unique_candidate_pairs": candidate_count,
        "average_candidates_per_source1": average_candidates,
        "block_stats": block_stats,
        "union_runtime_seconds": union_elapsed,
        "write_runtime_seconds": write_elapsed,
        "total_runtime_seconds": total_elapsed,
        "output_path": str(output_path),
        "output_size_mb": output_size_mb,
        "blocking_rules": BLOCKS,
        "candidate_schema": {
            "source1_entity_id": "entity identifier from source1",
            "candidate_entity_id": "candidate entity identifier",
            "candidate_source": "S2 or S3",
            "block_support_count": "number of blocking rules supporting pair",
            "blocks": "blocking rules supporting pair",
        },
    }

    metadata_path = output_path.with_suffix(
        ".metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL BLOCKING RESULT")
    print("=" * 80)

    print(
        f"Source:                 {source1}"
    )

    print(
        f"Candidate source:       {candidate_source}"
    )

    print(
        f"Source1 rows:            "
        f"{source1_count:,}"
    )

    print(
        f"Unique candidate pairs:  "
        f"{candidate_count:,}"
    )

    print(
        f"Avg candidates/S1:       "
        f"{average_candidates:.2f}"
    )

    print(
        f"Union/dedup time:        "
        f"{union_elapsed:.2f} sec"
    )

    print(
        f"Write time:              "
        f"{write_elapsed:.2f} sec"
    )

    print(
        f"Total runtime:           "
        f"{total_elapsed / 60:.2f} min"
    )

    print(
        f"Output size:             "
        f"{output_size_mb:.2f} MB"
    )

    print()
    print(
        f"Metadata: {metadata_path}"
    )

    print(
        f"Candidates: {output_path}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del final_df
    del final_candidates
    del all_candidates
    gc.collect()

    return metadata


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Amazon ML Challenge 2026 "
            "Stage 04 candidate blocking"
        )
    )

    parser.add_argument(
        "--candidate-source",
        choices=["S2", "S3", "both"],
        default="S2",
        help="Candidate source to block against.",
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("STAGE 04 — BLOCKING")
    print("=" * 80)

    if args.candidate_source in ("S2", "both"):

        output = (
            CANDIDATE_DIR
            / "train_s1_s2_candidates.parquet"
        )

        generate_candidates(
            source1="S1",
            candidate_source="S2",
            output_path=output,
        )

    if args.candidate_source in ("S3", "both"):

        output = (
            CANDIDATE_DIR
            / "train_s1_s3_candidates.parquet"
        )

        generate_candidates(
            source1="S1",
            candidate_source="S3",
            output_path=output,
        )

    print()
    print("=" * 80)
    print("BLOCKING STAGE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()