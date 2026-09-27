#!/usr/bin/env python3

"""
Amazon ML Challenge 2026
Stage 03 — Candidate Blocking

Generates candidate pairs between:

    Source 1 -> Source 2
    Source 1 -> Source 3

Input:
    artifacts/normalized/*.parquet

Output:
    artifacts/candidates/*.parquet
    artifacts/candidates/*.metadata.json

Designed for:
    8 vCPU
    64 GB RAM
    Polars
    Parquet
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import polars as pl
import psutil


# ============================================================
# RESOURCE MONITOR
# ============================================================

PROCESS = psutil.Process(os.getpid())


def process_ram_gb() -> float:
    return PROCESS.memory_info().rss / (1024 ** 3)


def system_ram_gb() -> tuple[float, float]:
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
# BLOCKING CONFIGURATION
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
        "key": "_k_exact_name",
        "name": "name_prefix6",
        "maximum_bucket": 500,
    },
    {
        "key": "_k_exact_name",
        "name": "name_prefix4",
        "maximum_bucket": 250,
    },
    {
        "key": "_k_numeric_address",
        "name": "numeric_address",
        "maximum_bucket": 300,
    },
]


REQUIRED_COLUMNS = [
    ENTITY_COL,
    "business_name_compact",
    "business_address_compact",
    "business_name_sorted_tokens",
    "business_address_numeric_tokens",
]


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_normalized_file(path: Path) -> Path:
    """
    Verify that a normalized Parquet file exists and is non-empty.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nNormalized file not found:\n"
            f"  {path}\n"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"\nNormalized file is empty:\n"
            f"  {path}\n"
        )

    return path


# ============================================================
# DATA LOADING
# ============================================================

def load_normalized(path: Path) -> pl.LazyFrame:
    """
    Lazily load normalized Parquet and create the six
    blocking keys required by the candidate generator.
    """

    path = validate_normalized_file(path)

    print()
    print("Loading normalized file:")
    print(f"  {path}")

    return (
        pl.scan_parquet(path)
        .select(
            [
                ENTITY_COL,
                "business_name_compact",
                "business_address_compact",
                "business_name_sorted_tokens",
                "business_address_numeric_tokens",
            ]
        )
        .with_columns(
            [
                pl.col("business_name_compact")
                .str.slice(0, 6)
                .alias("_k_name_prefix6"),

                pl.col("business_name_compact")
                .str.slice(0, 4)
                .alias("_k_name_prefix4"),
            ]
        )
        .with_columns(
            [
                pl.col("business_name_compact")
                .alias("_k_exact_name"),

                pl.col("business_address_compact")
                .alias("_k_exact_address"),

                pl.col("business_name_sorted_tokens")
                .alias("_k_sorted_name"),

                pl.col("business_address_numeric_tokens")
                .alias("_k_numeric_address"),
            ]
        )
        .select(
            [
                ENTITY_COL,
                "_k_exact_name",
                "_k_exact_address",
                "_k_sorted_name",
                "_k_name_prefix6",
                "_k_name_prefix4",
                "_k_numeric_address",
            ]
        )
    )


# ============================================================
# SINGLE BLOCK
# ============================================================

def make_block(
    s1: pl.LazyFrame,
    candidate: pl.LazyFrame,
    key: str,
    block_name: str,
    maximum_bucket: int,
) -> pl.LazyFrame:
    """
    Generate candidate pairs for one blocking rule.

    Large buckets are discarded to prevent candidate explosion.
    """

    # --------------------------------------------------------
    # LEFT / SOURCE 1
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RIGHT / CANDIDATE SOURCE
    # --------------------------------------------------------

    right = (
        candidate
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
    # SAFE BUCKETS — SOURCE 1
    # --------------------------------------------------------

    left_keys = (
        left
        .group_by(key)
        .len(name="_left_count")
        .filter(
            pl.col("_left_count")
            <= maximum_bucket
        )
        .select(key)
    )

    # --------------------------------------------------------
    # SAFE BUCKETS — CANDIDATE SOURCE
    # --------------------------------------------------------

    right_keys = (
        right
        .group_by(key)
        .len(name="_right_count")
        .filter(
            pl.col("_right_count")
            <= maximum_bucket
        )
        .select(key)
    )

    # --------------------------------------------------------
    # ONLY KEYS PRESENT ON BOTH SIDES
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
    # REDUCE BOTH SIDES BEFORE MANY-TO-MANY JOIN
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
    # MANY-TO-MANY CANDIDATE JOIN
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
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    source1_path: Path,
    candidate_source_path: Path,
    output_path: Path,
    candidate_source: str,
) -> dict:
    """
    Generate candidate pairs.

    Parameters
    ----------
    source1_path:
        Normalized Source 1 Parquet.

    candidate_source_path:
        Normalized Source 2 or Source 3 Parquet.

    output_path:
        Candidate Parquet output.

    candidate_source:
        "S2" or "S3".
    """

    source1_path = validate_normalized_file(
        Path(source1_path)
    )

    candidate_source_path = validate_normalized_file(
        Path(candidate_source_path)
    )

    output_path = Path(output_path)

    print()
    print("=" * 80)
    print(
        f"BLOCKING: S1 -> {candidate_source}"
    )
    print("=" * 80)

    print(f"S1:")
    print(f"  {source1_path}")

    print(f"Candidate source:")
    print(f"  {candidate_source_path}")

    print(f"Output:")
    print(f"  {output_path}")

    overall_start = time.time()

    print_resource_status("Initial: ")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    s1 = load_normalized(
        source1_path
    )

    candidate = load_normalized(
        candidate_source_path
    )

    results = []
    block_stats = []

    # ========================================================
    # RUN SIX BLOCKS
    # ========================================================

    for config in BLOCKS:

        key = config["key"]
        block_name = config["name"]
        maximum_bucket = config["maximum_bucket"]

        print()
        print("-" * 80)
        print(f"BLOCK: {block_name}")
        print(f"KEY: {key}")
        print(
            f"MAX BUCKET: "
            f"{maximum_bucket:,}"
        )
        print("-" * 80)

        print_resource_status(
            "Before: "
        )

        start = time.time()

        block = make_block(
            s1=s1,
            candidate=candidate,
            key=key,
            block_name=block_name,
            maximum_bucket=maximum_bucket,
        )

        # Materialize this block.
        block_df = block.collect(
            engine="streaming"
        )

        elapsed = (
            time.time() - start
        )

        candidate_count = len(
            block_df
        )

        print(
            f"Candidates: "
            f"{candidate_count:,}"
        )

        print(
            f"Runtime: "
            f"{elapsed:.2f} sec"
        )

        print_resource_status(
            "After:  "
        )

        block_stats.append(
            {
                "block": block_name,
                "key": key,
                "maximum_bucket":
                    maximum_bucket,
                "candidate_count":
                    candidate_count,
                "runtime_seconds":
                    elapsed,
                "process_ram_gb":
                    process_ram_gb(),
            }
        )

        results.append(
            block_df.lazy()
        )

        del block_df
        gc.collect()

    # ========================================================
    # UNION
    # ========================================================

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
    # Deduplicate candidate pairs.
    # Keep how many blocking rules produced each pair.
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
                .alias(
                    "block_support_count"
                ),

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
    # Materialize final candidate set
    # --------------------------------------------------------

    final_df = final_candidates.collect(
        engine="streaming"
    )

    union_elapsed = (
        time.time() - union_start
    )

    final_count = len(
        final_df
    )

    raw_candidate_count = sum(
        x["candidate_count"]
        for x in block_stats
    )

    duplicate_reduction = (
        1.0
        - (
            final_count
            / raw_candidate_count
        )
        if raw_candidate_count > 0
        else 0.0
    )

    print(
        f"Raw block candidates: "
        f"{raw_candidate_count:,}"
    )

    print(
        f"Final unique candidates: "
        f"{final_count:,}"
    )

    print(
        f"Duplicate reduction: "
        f"{duplicate_reduction * 100:.2f}%"
    )

    print(
        f"Union runtime: "
        f"{union_elapsed:.2f} sec"
    )

    print_resource_status(
        "After union: "
    )

    # ========================================================
    # WRITE PARQUET
    # ========================================================

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
            f"Removing existing output:"
        )
        print(
            f"  {output_path}"
        )

        output_path.unlink()

    write_start = time.time()

    final_df.write_parquet(
        output_path,
        compression="zstd",
        compression_level=3,
    )

    write_elapsed = (
        time.time() - write_start
    )

    output_size_bytes = (
        output_path.stat().st_size
    )

    output_size_gb = (
        output_size_bytes
        / (1024 ** 3)
    )

    print(
        f"Output written:"
    )
    print(
        f"  {output_path}"
    )

    print(
        f"Output size: "
        f"{output_size_gb:.2f} GB"
    )

    print(
        f"Write runtime: "
        f"{write_elapsed:.2f} sec"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_elapsed = (
        time.time() - overall_start
    )

    print()
    print("=" * 80)
    print("BLOCKING COMPLETE")
    print("=" * 80)

    print(
        f"Candidate source: "
        f"{candidate_source}"
    )

    print(
        f"Raw block candidates: "
        f"{raw_candidate_count:,}"
    )

    print(
        f"Final unique candidates: "
        f"{final_count:,}"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed:.2f} sec"
    )

    print_resource_status(
        "Final: "
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "stage":
            "03_candidate_generation",

        "source1_path":
            str(source1_path),

        "candidate_source":
            candidate_source,

        "candidate_source_path":
            str(candidate_source_path),

        "output_path":
            str(output_path),

        "blocks":
            block_stats,

        "raw_block_candidate_total":
            raw_candidate_count,

        "final_unique_candidates":
            final_count,

        "duplicate_reduction":
            duplicate_reduction,

        "union_runtime_seconds":
            union_elapsed,

        "write_runtime_seconds":
            write_elapsed,

        "total_runtime_seconds":
            total_elapsed,

        "output_size_bytes":
            output_size_bytes,

        "output_size_gb":
            output_size_gb,

        "final_process_ram_gb":
            process_ram_gb(),
    }

    metadata_path = (
        output_path.with_suffix(
            ".metadata.json"
        )
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Metadata:"
    )
    print(
        f"  {metadata_path}"
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    del final_df
    del final_candidates
    del all_candidates
    del results
    del s1
    del candidate

    gc.collect()

    return metadata


# ============================================================
# COMMAND LINE SUPPORT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Amazon ML Challenge 2026 "
            "candidate generation"
        )
    )

    parser.add_argument(
        "--source1",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-label",
        required=True,
        choices=["S2", "S3"],
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    generate_candidates(
        source1_path=args.source1,
        candidate_source_path=(
            args.candidate_source
        ),
        output_path=args.output,
        candidate_source=args.candidate_label,
    )
