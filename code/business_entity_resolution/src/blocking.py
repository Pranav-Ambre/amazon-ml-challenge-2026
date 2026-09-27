"""
Memory-safe high-performance candidate generation.

Important:
    Each blocking rule is executed and written independently.
    We NEVER concatenate all candidate pairs in RAM.

Pipeline:

    Normalized Parquet
          ↓
    Blocking key
          ↓
    Safe bucket filtering
          ↓
    Native Polars join
          ↓
    Parquet block output

Later:
    block outputs
          ↓
    disk-based union/deduplication
          ↓
    final candidate set
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ============================================================
# Normalized columns
# ============================================================

ENTITY_COL = "entity_id"

COUNTRY_COL = "country_norm"

NAME_COL = "business_name_compact"

ADDRESS_COL = "business_address_compact"

NUMERIC_ADDRESS_COL = "business_address_numeric_tokens"

SORTED_NAME_COL = "business_name_sorted_tokens"

HOUSE_NUMBER_COL = "business_address_house_number"

POSTAL_COL = "business_address_postal_tokens"


# ============================================================
# Maximum candidate pairs allowed per blocking bucket
# ============================================================

# This is MUCH safer than allowing:
#
#       5000 x 5000 = 25,000,000
#
# candidates from a single name bucket.

MAX_PAIR_PRODUCT = {
    "exact_name": 100_000,
    "exact_address": 100_000,
    "sorted_name": 50_000,
    "name_house": 25_000,
    "name_postal": 25_000,
    "numeric_address": 50_000,
}


# ============================================================
# Read normalized file
# ============================================================

def scan_normalized(path: str | Path) -> pl.LazyFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized file not found:\n{path}"
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
                HOUSE_NUMBER_COL,
                POSTAL_COL,
            ]
        )
    )


# ============================================================
# Create blocking keys
# ============================================================

def add_blocking_keys(
    lf: pl.LazyFrame,
) -> pl.LazyFrame:

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

    house = (
        pl.col(HOUSE_NUMBER_COL)
        .fill_null("")
        .cast(pl.String)
    )

    postal = (
        pl.col(POSTAL_COL)
        .fill_null("")
        .cast(pl.String)
    )

    return lf.with_columns(

        # ----------------------------------------------------
        # B1
        # Exact normalized name
        # ----------------------------------------------------

        (
            country + "|" + name
        ).alias("_b1"),

        # ----------------------------------------------------
        # B2
        # Exact normalized address
        # ----------------------------------------------------

        (
            country + "|" + address
        ).alias("_b2"),

        # ----------------------------------------------------
        # B3
        # Order-independent name
        # ----------------------------------------------------

        (
            country + "|" + sorted_name
        ).alias("_b3"),

        # ----------------------------------------------------
        # B4
        # Name prefix + house number
        #
        # Much safer than pure name-prefix blocking.
        # ----------------------------------------------------

        (
            country
            + "|"
            + name.str.slice(0, 6)
            + "|"
            + house
        ).alias("_b4"),

        # ----------------------------------------------------
        # B5
        # Name prefix + postal information
        # ----------------------------------------------------

        (
            country
            + "|"
            + name.str.slice(0, 6)
            + "|"
            + postal
        ).alias("_b5"),

        # ----------------------------------------------------
        # B6
        # Numeric address
        # ----------------------------------------------------

        (
            country
            + "|"
            + numeric_address
        ).alias("_b6"),
    )


# ============================================================
# Create one blocking result
# ============================================================

def make_block(
    s1: pl.LazyFrame,
    s2: pl.LazyFrame,
    key: str,
    block_name: str,
    max_pairs: int,
) -> pl.LazyFrame:

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
    # Count occurrences of every key.
    # --------------------------------------------------------

    left_counts = (
        left
        .group_by(key)
        .len(name="_n_left")
    )

    right_counts = (
        right
        .group_by(key)
        .len(name="_n_right")
    )

    # --------------------------------------------------------
    # Keep only keys where:
    #
    #       left_count × right_count <= max_pairs
    #
    # This directly controls Cartesian explosion.
    # --------------------------------------------------------

    safe_keys = (
        left_counts
        .join(
            right_counts,
            on=key,
            how="inner",
        )
        .filter(
            (
                pl.col("_n_left")
                * pl.col("_n_right")
            )
            <= max_pairs
        )
        .select(key)
    )

    # --------------------------------------------------------
    # Reduce both datasets before the actual join.
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
    #
    # No Python loops.
    # --------------------------------------------------------

    result = (
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
        .unique(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .with_columns(
            pl.lit(block_name).alias("block")
        )
    )

    return result


# ============================================================
# Generate ONE block and immediately write it
# ============================================================

def run_block(
    s1: pl.LazyFrame,
    s2: pl.LazyFrame,
    key: str,
    block_name: str,
    max_pairs: int,
    output_path: Path,
) -> None:

    print()
    print("-" * 70)
    print(
        f"{block_name} | "
        f"max pairs/bucket = {max_pairs:,}"
    )
    print("-" * 70)

    result = make_block(
        s1=s1,
        s2=s2,
        key=key,
        block_name=block_name,
        max_pairs=max_pairs,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Writing → {output_path}"
    )

    # IMPORTANT:
    # Stream directly to disk.
    result.sink_parquet(
        output_path,
        compression="zstd",
        row_group_size=250_000,
        maintain_order=False,
        engine="streaming",
    )

    print(
        f"Finished {block_name}"
    )


# ============================================================
# Complete candidate generation
# ============================================================

def generate_candidates(
    source1_path: str | Path,
    source2_path: str | Path,
    output_dir: str | Path,
) -> None:

    source1_path = Path(source1_path)
    source2_path = Path(source2_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("LOADING NORMALIZED DATA")
    print("=" * 80)

    print(f"S1: {source1_path}")
    print(f"S2: {source2_path}")

    s1 = add_blocking_keys(
        scan_normalized(source1_path)
    )

    s2 = add_blocking_keys(
        scan_normalized(source2_path)
    )

    # ========================================================
    # Blocking configuration
    # ========================================================

    blocks = [
        (
            "_b1",
            "B1_EXACT_NAME",
            MAX_PAIR_PRODUCT["exact_name"],
        ),

        (
            "_b2",
            "B2_EXACT_ADDRESS",
            MAX_PAIR_PRODUCT["exact_address"],
        ),

        (
            "_b3",
            "B3_SORTED_NAME",
            MAX_PAIR_PRODUCT["sorted_name"],
        ),

        (
            "_b4",
            "B4_NAME_HOUSE",
            MAX_PAIR_PRODUCT["name_house"],
        ),

        (
            "_b5",
            "B5_NAME_POSTAL",
            MAX_PAIR_PRODUCT["name_postal"],
        ),

        (
            "_b6",
            "B6_NUMERIC_ADDRESS",
            MAX_PAIR_PRODUCT["numeric_address"],
        ),
    ]

    # ========================================================
    # Execute each block independently.
    # ========================================================

    for key, block_name, max_pairs in blocks:

        output_path = (
            output_dir
            / f"{block_name.lower()}.parquet"
        )

        run_block(
            s1=s1,
            s2=s2,
            key=key,
            block_name=block_name,
            max_pairs=max_pairs,
            output_path=output_path,
        )

    print()
    print("=" * 80)
    print("ALL BLOCKS WRITTEN SUCCESSFULLY")
    print("=" * 80)