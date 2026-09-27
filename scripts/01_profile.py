"""
Detailed dataset profiling for Amazon ML Challenge 2026.

This script analyzes:
- dataset sizes
- columns
- ID patterns
- country distributions
- name statistics
- address statistics
- missing values
- empty values
- duplicate IDs
- ground-truth match distribution
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODE_DIR = PROJECT_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from business_entity_resolution.src.config import (
    REPORTS_DIR,
)

from business_entity_resolution.src.data_loader import (
    load_train_source1,
    load_train_source2,
    load_train_source3,
    load_train_ground_truth,
    load_test_source1,
    load_test_source2,
    load_test_source3,
)


# ============================================================
# STRING PROFILE
# ============================================================

def profile_string_column(
    series: pd.Series,
) -> dict:
    """
    Generate statistics for a text column.
    """

    values = series.fillna("").astype(str)

    lengths = values.str.len()

    empty_mask = values.str.strip().eq("")

    return {
        "total": int(len(values)),
        "null_count": int(series.isna().sum()),
        "empty_count": int(empty_mask.sum()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
        "p25_length": float(lengths.quantile(0.25)),
        "p75_length": float(lengths.quantile(0.75)),
    }


# ============================================================
# ID PROFILE
# ============================================================

def profile_ids(series: pd.Series) -> dict:
    """
    Analyze entity IDs.
    """

    values = (
        series
        .dropna()
        .astype(str)
    )

    prefixes = {}

    for value in values:

        # Extract alphabetic prefix at the beginning.
        match = re.match(r"^[A-Za-z]+", value)

        prefix = (
            match.group(0)
            if match
            else "<NO_PREFIX>"
        )

        prefixes[prefix] = (
            prefixes.get(prefix, 0) + 1
        )

    return {
        "total": int(len(series)),
        "null_count": int(series.isna().sum()),
        "unique_count": int(series.nunique()),
        "duplicate_count": int(
            series.duplicated().sum()
        ),
        "prefix_counts": prefixes,
        "sample_ids": values.head(20).tolist(),
    }


# ============================================================
# COUNTRY PROFILE
# ============================================================

def profile_country(
    series: pd.Series,
) -> dict:
    """
    Analyze country values.
    """

    values = (
        series
        .fillna("<NULL>")
        .astype(str)
        .str.strip()
    )

    counts = values.value_counts()

    return {
        "unique_countries": int(
            values.nunique()
        ),
        "distribution": {
            str(country): int(count)
            for country, count in counts.items()
        },
    }


# ============================================================
# SOURCE PROFILE
# ============================================================

def profile_source(
    df: pd.DataFrame,
    name: str,
) -> dict:
    """
    Generate a detailed profile for a source dataframe.
    """

    print(f"\nProfiling {name}...")

    profile = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    if "entity_id" in df.columns:

        profile["entity_id"] = profile_ids(
            df["entity_id"]
        )

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    if "business_name" in df.columns:

        profile["business_name"] = (
            profile_string_column(
                df["business_name"]
            )
        )

    # --------------------------------------------------------
    # Business address
    # --------------------------------------------------------

    if "business_address" in df.columns:

        profile["business_address"] = (
            profile_string_column(
                df["business_address"]
            )
        )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    if "country" in df.columns:

        profile["country"] = (
            profile_country(
                df["country"]
            )
        )

    # --------------------------------------------------------
    # Full duplicate rows
    # --------------------------------------------------------

    profile["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    return profile


# ============================================================
# GROUND TRUTH PROFILE
# ============================================================

def profile_ground_truth(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze ground-truth match distribution.

    The exact parsing of matched_entity_ids depends on the
    challenge file format. This function first reports the
    raw values and basic structure.
    """

    result = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
    }

    if "source1_entity_id" in df.columns:

        result["source1_unique_ids"] = int(
            df["source1_entity_id"].nunique()
        )

        result["source1_duplicate_ids"] = int(
            df["source1_entity_id"].duplicated().sum()
        )

    if "matched_entity_ids" in df.columns:

        values = (
            df["matched_entity_ids"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        result["empty_match_rows"] = int(
            values.eq("").sum()
        )

        result["sample_values"] = (
            values.head(20).tolist()
        )

        # Character-level information.
        result["match_field_length"] = {
            "min": int(values.str.len().min()),
            "max": int(values.str.len().max()),
            "mean": float(values.str.len().mean()),
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Amazon ML Challenge 2026 - Detailed Data Profiling")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading training datasets...")

    train_source1 = load_train_source1()
    train_source2 = load_train_source2()
    train_source3 = load_train_source3()
    ground_truth = load_train_ground_truth()

    print("Loading test datasets...")

    test_source1 = load_test_source1()
    test_source2 = load_test_source2()
    test_source3 = load_test_source3()

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    report = {
        "train": {},
        "test": {},
        "ground_truth": {},
    }

    report["train"]["source1"] = profile_source(
        train_source1,
        "train_source1",
    )

    report["train"]["source2"] = profile_source(
        train_source2,
        "train_source2",
    )

    report["train"]["source3"] = profile_source(
        train_source3,
        "train_source3",
    )

    report["test"]["source1"] = profile_source(
        test_source1,
        "test_source1",
    )

    report["test"]["source2"] = profile_source(
        test_source2,
        "test_source2",
    )

    report["test"]["source3"] = profile_source(
        test_source3,
        "test_source3",
    )

    report["ground_truth"] = profile_ground_truth(
        ground_truth
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        REPORTS_DIR /
        "phase0_profile.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)

    print(
        f"\nReport saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()