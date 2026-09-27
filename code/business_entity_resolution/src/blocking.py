"""
High-performance candidate generation for Amazon ML Challenge 2026.

Pipeline:
    Normalized Parquet
        ↓
    Multiple blocking keys
        ↓
    Native Polars joins
        ↓
    Union
        ↓
    Deduplication
        ↓
    block_support_count
        ↓
    Candidate Parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ============================================================
# Column names
# ============================================================

ENTITY_COL = "entity_id"

COUNTRY_COL = "country_norm"

NAME_COL = "business_name_compact"

ADDRESS_COL = "business_address_compact"

NUMERIC_ADDRESS_COL = "business_address_numeric_tokens"

SORTED_NAME_COL = "business_name_sorted_tokens"


# ============================================================
# Bucket limits
# ============================================================

# Prevent huge Cartesian products.
#
# These are deliberately generous for high recall while
# protecting runtime/memory.
MAX_BUCKET = {
    "exact_name": 5000,
    "exact_address": 3000,
    "sorted_name": 1000,
    "name_prefix6": 500,
    "name_prefix4": 250,
    "numeric_address": 300,
}


# ============================================================
# Read normalized Parquet
# ============================================================

def scan_normalized(path: str | Path) -> pl.LazyFrame:
    """
    Lazily scan one normalized Parquet file.

    IMPORTANT:
    Normalization produces one file per dataset, e.g.

        train_source1_normalized.parquet
        train_source2_normalized.parquet
        train_source3_normalized.parquet
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized file not found: {path}"
        )

    return (
        pl.scan_parquet(
            str(path),
            low_memory=True,
        )
        .select(
            [
                ENTITY_COL,
                COUNTRY_COL,
                NAME_COL,
                ADDRESS_COL,
                NUMERIC_ADDRESS_COL,
                SORTED_NAME_COL,
            ]
        )
    )


# ============================================================
# Blocking keys
# ============================================================

def add_blocking_keys(
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Generate blocking keys.

    Country is included in every key to prevent unnecessary
    cross-country comparisons.
    """

    country = (
        pl.col(COUNTRY_COL)
        .fill_null("")
        .cast(pl.String)
    )

    name = (
        pl.col(NAME_COL)
        .fill_null("")
        .cast(pl.String)
    )

    address = (
        pl.col(ADDRESS_COL)
        .fill_null("")
        .cast(pl.String)
    )

    numeric_address = (
        pl.col(NUMERIC_ADDRESS_COL)
        .fill_null("")
        .cast(pl.String)
    )

    sorted_name = (
        pl.col(SORTED_NAME_COL)
        .fill_null("")
        .cast(pl.String)
    )

    return lf.with_columns(

        # ----------------------------------------------------
        # B1 — exact normalized name
        # ----------------------------------------------------

        (
            country + "|" + name
        ).alias("_k_exact_name"),

        # ----------------------------------------------------
        # B2 — exact normalized address
        # ----------------------------------------------------

        (
            country + "|" + address
        ).alias("_k_exact_address"),

        # ----------------------------------------------------
        # B3 — order-independent name tokens
        # ----------------------------------------------------

        (
            country + "|" + sorted_name
        ).alias("_k_sorted_name"),

        # ----------------------------------------------------
        # B4 — name prefix 6
        # ----------------------------------------------------

        (
            country + "|" + name.str.slice(0, 6)
        ).alias("_k_name_prefix6"),

        # ----------------------------------------------------
        # B5 — name prefix 4
        # ----------------------------------------------------

        (
            country + "|" + name.str.slice(0, 4)
        ).alias("_k_name_prefix4"),

        # ----------------------------------------------------
        # B6 — numeric address
        # ----------------------------------------------------

        (
            country + "|" + numeric_address
        ).alias("_k_numeric_address"),
    )


# ============================================================
# Generate one blocking rule
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

    Uses Polars native joins rather than Python row loops.
    """

    left = (
        s1
        .select(
            [
                pl.col(ENTITY_COL)
                .alias("source1_entity_id"),

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
                pl.col(ENTITY_COL)
                .alias("candidate_entity_id"),

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
    # Find safe buckets on S1
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
    # Find safe buckets on S2
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
    # Only keep keys present on BOTH sides.
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
    # Reduce both sides before Cartesian join.
    # --------------------------------------------------------

    left = left.join(
        safe_keys,
        on=key,
        how="inner",
    )

    right = right.join(
        safe_keys,
        on=key,
        how="inner",
    )

    # --------------------------------------------------------
    # Native many-to-many join.
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
# Generate all candidates
# ============================================================

def generate_candidates(
    source1_path: str | Path,
    source2_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Generate candidates from S1 to S2/S3.

    The same function is used for:

        S1 → S2
        S1 → S3
    """

    print()
    print("=" * 80)
    print("Loading normalized datasets")
    print("=" * 80)

    print(f"S1: {source1_path}")
    print(f"S2: {source2_path}")

    s1 = add_blocking_keys(
        scan_normalized(source1_path)
    )

    s2 = add_blocking_keys(
        scan_normalized(source2_path)
    )

    # --------------------------------------------------------
    # Blocking rules
    # --------------------------------------------------------

    blocks = [
        (
            "_k_exact_name",
            "B1_EXACT_NAME",
            MAX_BUCKET["exact_name"],
        ),

        (
            "_k_exact_address",
            "B2_EXACT_ADDRESS",
            MAX_BUCKET["exact_address"],
        ),

        (
            "_k_sorted_name",
            "B3_SORTED_NAME",
            MAX_BUCKET["sorted_name"],
        ),

        (
            "_k_name_prefix6",
            "B4_NAME_PREFIX6",
            MAX_BUCKET["name_prefix6"],
        ),

        (
            "_k_name_prefix4",
            "B5_NAME_PREFIX4",
            MAX_BUCKET["name_prefix4"],
        ),

        (
            "_k_numeric_address",
            "B6_NUMERIC_ADDRESS",
            MAX_BUCKET["numeric_address"],
        ),
    ]

    block_results = []

    # --------------------------------------------------------
    # Generate every block.
    # --------------------------------------------------------

    for key, block_name, maximum_bucket in blocks:

        print()
        print(
            f"Running {block_name} "
            f"(max bucket = {maximum_bucket:,})"
        )

        block = make_block(
            s1=s1,
            s2=s2,
            key=key,
            block_name=block_name,
            maximum_bucket=maximum_bucket,
        )

        block_results.append(block)

    # --------------------------------------------------------
    # Union all blocks.
    # --------------------------------------------------------

    print()
    print("Combining blocking results...")

    candidates = pl.concat(
        block_results,
        how="vertical",
    )

    # --------------------------------------------------------
    # Deduplicate pairs and preserve block evidence.
    # --------------------------------------------------------

    candidates = (
        candidates
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
                .str.join("|")
                .alias("blocks"),
            ]
        )
        .with_columns(
            pl.lit("S2")
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
    # Write output.
    # --------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Writing candidates → {output_path}")

    candidates.sink_parquet(
        output_path,
        compression="zstd",
        engine="streaming",
    )

    print("Candidate generation finished.")