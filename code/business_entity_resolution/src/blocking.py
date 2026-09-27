"""
Memory-safe candidate generation for business entity resolution.

Pipeline:
    Normalized Source 1
        ↓
    Block-by-block joins
        ↓
    Candidate pairs with provenance
        ↓
    Deduplicated candidate pairs
        ↓
    Parquet output

Important:
- Uses Polars lazy scanning.
- Processes one blocking rule at a time.
- Does not build the full Cartesian product.
- Preserves block provenance.
- Uses source-specific candidate outputs.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import polars as pl


# ============================================================
# CONSTANTS
# ============================================================

ENTITY_COL = "entity_id"

LOGGER = logging.getLogger(__name__)


# ============================================================
# BLOCKING CONFIGURATION
# ============================================================

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
# REQUIRED NORMALIZED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    ENTITY_COL,
    "business_name_compact",
    "business_address_compact",
    "business_name_sorted_tokens",
    "business_address_numeric_tokens",
]


# ============================================================
# NORMALIZED DATA LOADER
# ============================================================

def load_normalized(path: str | Path) -> pl.LazyFrame:
    """
    Load normalized parquet data lazily and create blocking keys.

    Parameters
    ----------
    path:
        Path to normalized parquet file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing entity_id and all blocking keys.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized parquet file not found: {path}"
        )

    return (
        pl.scan_parquet(path)
        .select(REQUIRED_COLUMNS)
        .with_columns(
            [
                # ------------------------------------------------
                # Prefix blocking keys
                # ------------------------------------------------
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
                # ------------------------------------------------
                # Exact / derived blocking keys
                # ------------------------------------------------
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
# BLOCK KEY VALIDATION
# ============================================================

def validate_block_keys(lf: pl.LazyFrame) -> None:
    """
    Validate that all configured blocking keys exist.
    """

    schema = lf.collect_schema()
    available_columns = set(schema.names())

    missing = []

    for block in BLOCKS:
        key = block["key"]

        if key not in available_columns:
            missing.append(key)

    if missing:
        raise ValueError(
            "Missing blocking keys: "
            + ", ".join(sorted(set(missing)))
        )


# ============================================================
# BLOCK SIZE FILTER
# ============================================================

def filter_block_sizes(
    lf: pl.LazyFrame,
    key: str,
    maximum_bucket: int,
) -> pl.LazyFrame:
    """
    Remove oversized blocking buckets.

    Very common values such as empty strings or generic prefixes
    can create extremely large candidate sets. Such buckets are
    excluded before joining.
    """

    valid_keys = (
        lf
        .filter(
            pl.col(key).is_not_null()
            & (pl.col(key).str.len_chars() > 0)
        )
        .group_by(key)
        .agg(
            pl.len().alias("_block_size")
        )
        .filter(
            pl.col("_block_size") <= maximum_bucket
        )
        .select(key)
    )

    return lf.join(
        valid_keys,
        on=key,
        how="semi",
    )


# ============================================================
# SINGLE BLOCK GENERATION
# ============================================================

def generate_block_candidates(
    source1: pl.LazyFrame,
    source2: pl.LazyFrame,
    block: Dict,
) -> pl.LazyFrame:
    """
    Generate candidate pairs for one blocking rule.

    Output columns:
        source1_entity_id
        candidate_entity_id
        candidate_source
        block_support_count
        blocks
    """

    key = block["key"]
    block_name = block["name"]
    maximum_bucket = block["maximum_bucket"]

    LOGGER.info(
        "Generating block: %s | key=%s | max_bucket=%s",
        block_name,
        key,
        maximum_bucket,
    )

    # --------------------------------------------------------
    # Filter oversized buckets independently on both sides.
    # --------------------------------------------------------

    s1 = filter_block_sizes(
        source1,
        key,
        maximum_bucket,
    )

    s2 = filter_block_sizes(
        source2,
        key,
        maximum_bucket,
    )

    # --------------------------------------------------------
    # Rename entity IDs before join.
    # --------------------------------------------------------

    s1 = s1.select(
        [
            pl.col(ENTITY_COL).alias("source1_entity_id"),
            key,
        ]
    )

    s2 = s2.select(
        [
            pl.col(ENTITY_COL).alias("candidate_entity_id"),
            key,
        ]
    )

    # --------------------------------------------------------
    # Inner join on blocking key.
    # --------------------------------------------------------

    candidates = (
        s1.join(
            s2,
            on=key,
            how="inner",
        )
        .filter(
            pl.col("source1_entity_id")
            != pl.col("candidate_entity_id")
        )
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .with_columns(
            [
                pl.lit(block_name)
                .alias("block_name"),
            ]
        )
    )

    return candidates


# ============================================================
# BLOCK-BY-BLOCK CANDIDATE WRITER
# ============================================================

def generate_candidates(
    source1_path: str | Path,
    source2_path: str | Path,
    output_path: str | Path,
    candidate_source: str,
) -> Dict:
    """
    Generate memory-safe candidate pairs.

    Each blocking rule is processed independently and written
    to temporary parquet files. Final candidates are then
    deduplicated while preserving block provenance.

    Parameters
    ----------
    source1_path:
        Normalized Source 1 parquet.

    source2_path:
        Normalized Source 2 or Source 3 parquet.

    output_path:
        Final candidate parquet.

    candidate_source:
        Candidate source label, e.g. "S2" or "S3".

    Returns
    -------
    dict
        Generation statistics.
    """

    start_time = time.time()

    source1_path = Path(source1_path)
    source2_path = Path(source2_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = output_path.parent / (
        f".{output_path.stem}_blocks"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info("=" * 70)
    LOGGER.info("Candidate generation")
    LOGGER.info("=" * 70)
    LOGGER.info("Source 1: %s", source1_path)
    LOGGER.info("Source 2: %s", source2_path)
    LOGGER.info("Candidate source: %s", candidate_source)
    LOGGER.info("Output: %s", output_path)

    # --------------------------------------------------------
    # Load lazily
    # --------------------------------------------------------

    source1 = load_normalized(source1_path)
    source2 = load_normalized(source2_path)

    validate_block_keys(source1)
    validate_block_keys(source2)

    # --------------------------------------------------------
    # Generate every block independently
    # --------------------------------------------------------

    block_files: List[Path] = []
    block_statistics = {}

    for block_index, block in enumerate(BLOCKS):

        block_name = block["name"]

        block_start = time.time()

        LOGGER.info(
            "\n[%d/%d] BLOCK: %s",
            block_index + 1,
            len(BLOCKS),
            block_name,
        )

        candidates = generate_block_candidates(
            source1=source1,
            source2=source2,
            block=block,
        )

        temp_file = (
            temp_dir
            / f"block_{block_index:02d}_{block_name}.parquet"
        )

        # ----------------------------------------------------
        # Collect ONLY this block.
        # ----------------------------------------------------

        block_df = candidates.collect()

        row_count = block_df.height

        # ----------------------------------------------------
        # Save block
        # ----------------------------------------------------

        if row_count > 0:

            block_df.write_parquet(
                temp_file,
                compression="zstd",
            )

            block_files.append(temp_file)

        block_runtime = time.time() - block_start

        block_statistics[block_name] = {
            "rows": row_count,
            "runtime_seconds": round(
                block_runtime,
                2,
            ),
            "temporary_file": str(temp_file),
        }

        LOGGER.info(
            "Block %s: %,d candidates | %.2f sec",
            block_name,
            row_count,
            block_runtime,
        )

        del block_df

    # --------------------------------------------------------
    # No candidates
    # --------------------------------------------------------

    if not block_files:

        empty_df = pl.DataFrame(
            {
                "source1_entity_id": pl.Series(
                    [],
                    dtype=pl.String,
                ),
                "candidate_entity_id": pl.Series(
                    [],
                    dtype=pl.String,
                ),
                "candidate_source": pl.Series(
                    [],
                    dtype=pl.String,
                ),
                "block_support_count": pl.Series(
                    [],
                    dtype=pl.UInt32,
                ),
                "blocks": pl.Series(
                    [],
                    dtype=pl.List(pl.String),
                ),
            }
        )

        empty_df.write_parquet(
            output_path,
            compression="zstd",
        )

        return {
            "candidate_source": candidate_source,
            "raw_candidates": 0,
            "final_candidates": 0,
            "duplicate_reduction_percent": 0.0,
            "runtime_seconds": round(
                time.time() - start_time,
                2,
            ),
            "blocks": block_statistics,
        }

    # ========================================================
    # FINAL DEDUPLICATION
    # ========================================================

    LOGGER.info("\nCombining block files...")

    # --------------------------------------------------------
    # Scan all temporary block files lazily.
    # --------------------------------------------------------

    combined = pl.scan_parquet(
        [str(path) for path in block_files]
    )

    raw_candidates = (
        combined
        .select(pl.len())
        .collect()
        .item()
    )

    LOGGER.info(
        "Raw block candidates: %,d",
        raw_candidates,
    )

    # --------------------------------------------------------
    # Aggregate duplicate candidate pairs.
    #
    # One pair may be found through multiple blocking rules.
    # Preserve all block names as provenance.
    # --------------------------------------------------------

    final_candidates = (
        combined
        .group_by(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .agg(
            [
                pl.col("block_name")
                .n_unique()
                .cast(pl.UInt32)
                .alias("block_support_count"),

                pl.col("block_name")
                .unique()
                .sort()
                .alias("blocks"),
            ]
        )
        .with_columns(
            [
                pl.lit(candidate_source)
                .alias("candidate_source"),
            ]
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
    # Collect final result and write parquet.
    # --------------------------------------------------------

    final_df = final_candidates.collect()

    final_count = final_df.height

    final_df.write_parquet(
        output_path,
        compression="zstd",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    duplicate_reduction = (
        (
            raw_candidates - final_count
        )
        / raw_candidates
        * 100
        if raw_candidates > 0
        else 0.0
    )

    runtime = time.time() - start_time

    LOGGER.info(
        "\nFinal candidates: %,d",
        final_count,
    )

    LOGGER.info(
        "Duplicate reduction: %.2f%%",
        duplicate_reduction,
    )

    LOGGER.info(
        "Runtime: %.2f sec",
        runtime,
    )

    LOGGER.info(
        "Output: %s",
        output_path,
    )

    # --------------------------------------------------------
    # Cleanup temporary files
    # --------------------------------------------------------

    for block_file in block_files:

        try:
            block_file.unlink()
        except OSError as exc:
            LOGGER.warning(
                "Could not remove temporary file %s: %s",
                block_file,
                exc,
            )

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    return {
        "candidate_source": candidate_source,
        "raw_candidates": int(raw_candidates),
        "final_candidates": int(final_count),
        "duplicate_reduction_percent": round(
            duplicate_reduction,
            2,
        ),
        "runtime_seconds": round(
            runtime,
            2,
        ),
        "blocks": block_statistics,
        "output_path": str(output_path),
    }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def generate_train_s1_s2_candidates(
    train_s1_path: str | Path,
    train_s2_path: str | Path,
    output_path: str | Path,
) -> Dict:

    return generate_candidates(
        source1_path=train_s1_path,
        source2_path=train_s2_path,
        output_path=output_path,
        candidate_source="S2",
    )


def generate_train_s1_s3_candidates(
    train_s1_path: str | Path,
    train_s3_path: str | Path,
    output_path: str | Path,
) -> Dict:

    return generate_candidates(
        source1_path=train_s1_path,
        source2_path=train_s3_path,
        output_path=output_path,
        candidate_source="S3",
    )


def generate_test_s1_s2_candidates(
    test_s1_path: str | Path,
    test_s2_path: str | Path,
    output_path: str | Path,
) -> Dict:

    return generate_candidates(
        source1_path=test_s1_path,
        source2_path=test_s2_path,
        output_path=output_path,
        candidate_source="S2",
    )


def generate_test_s1_s3_candidates(
    test_s1_path: str | Path,
    test_s3_path: str | Path,
    output_path: str | Path,
) -> Dict:

    return generate_candidates(
        source1_path=test_s1_path,
        source2_path=test_s3_path,
        output_path=output_path,
        candidate_source="S3",
    )


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    """
    Configure console logging when this module is executed
    directly or imported by a script that has not configured
    logging.
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
# MAIN
# ============================================================

if __name__ == "__main__":

    configure_logging()

    LOGGER.info(
        "Blocking module loaded successfully."
    )

    LOGGER.info(
        "Configured blocking rules:"
    )

    for block in BLOCKS:
        LOGGER.info(
            "  %-20s -> %s",
            block["name"],
            block["key"],
        )