from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code" / "business_entity_resolution")
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    REPORTS_DIR,
    create_project_directories,
)

from src.data_loader import (
    load_source1,
    load_source2,
    load_source3,
    load_ground_truth,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_COLUMNS = [
    "business_name",
    "business_address",
    "country",
]

ID_COLUMN = "entity_id"


# ============================================================
# TEXT COLUMN PROFILE
# ============================================================

def profile_text_column(df: pd.DataFrame, column: str) -> dict:
    """
    Generate statistics for a text column.
    """

    if column not in df.columns:
        return {
            "exists": False
        }

    series = df[column]

    string_series = (
        series
        .fillna("")
        .astype(str)
    )

    lengths = string_series.str.len()

    return {
        "exists": True,
        "null_count": int(
            series.isna().sum()
        ),
        "empty_count": int(
            (string_series.str.strip() == "").sum()
        ),
        "min_length": (
            int(lengths.min())
            if len(lengths)
            else 0
        ),
        "max_length": (
            int(lengths.max())
            if len(lengths)
            else 0
        ),
        "mean_length": (
            float(lengths.mean())
            if len(lengths)
            else 0
        ),
        "median_length": (
            float(lengths.median())
            if len(lengths)
            else 0
        ),
        "p25_length": (
            float(lengths.quantile(0.25))
            if len(lengths)
            else 0
        ),
        "p75_length": (
            float(lengths.quantile(0.75))
            if len(lengths)
            else 0
        ),
    }


# ============================================================
# ID PROFILE
# ============================================================

def profile_id_column(
    df: pd.DataFrame,
    column: str
) -> dict:
    """
    Generate statistics for entity IDs.
    """

    if column not in df.columns:
        return {
            "exists": False
        }

    series = df[column]

    return {
        "exists": True,
        "row_count": int(len(series)),
        "null_count": int(
            series.isna().sum()
        ),
        "unique_count": int(
            series.nunique(dropna=True)
        ),
        "duplicate_count": int(
            series.duplicated(keep=False).sum()
        ),
    }


# ============================================================
# COUNTRY PROFILE
# ============================================================

def profile_country(
    df: pd.DataFrame
) -> dict:
    """
    Return the most common country values.
    """

    if "country" not in df.columns:
        return {}

    country_series = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    counts = country_series.value_counts(
        dropna=False
    )

    return {
        str(country): int(count)
        for country, count in counts.head(50).items()
    }


# ============================================================
# DUPLICATE ROW PROFILE
# ============================================================

def profile_duplicate_rows(
    df: pd.DataFrame
) -> dict:
    """
    Count completely duplicated rows.
    """

    return {
        "duplicate_full_rows": int(
            df.duplicated().sum()
        )
    }


# ============================================================
# SOURCE DATASET PROFILE
# ============================================================

def profile_source(
    df: pd.DataFrame,
    dataset_name: str
) -> dict:
    """
    Generate complete profile for Source 1/2/3.
    """

    profile = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": list(df.columns),

        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "id_profile": profile_id_column(
            df,
            ID_COLUMN
        ),

        "text_profile": {},

        "country_distribution": profile_country(df),

        "duplicates": profile_duplicate_rows(df),
    }

    for column in TEXT_COLUMNS:

        profile["text_profile"][column] = (
            profile_text_column(
                df,
                column
            )
        )

    return profile


# ============================================================
# GROUND TRUTH PROFILE
# ============================================================

def profile_ground_truth(
    df: pd.DataFrame
) -> dict:
    """
    Generate profile for training ground truth.
    """

    profile = {
        "rows": int(len(df)),

        "columns": list(df.columns),

        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "null_counts": {
            column: int(
                df[column].isna().sum()
            )
            for column in df.columns
        },
    }

    # --------------------------------------------------------
    # SOURCE 1 ENTITY ID
    # --------------------------------------------------------

    if "source1_entity_id" in df.columns:

        profile["source1_entity_id"] = {

            "unique_count": int(
                df["source1_entity_id"]
                .nunique(dropna=True)
            ),

            "duplicate_count": int(
                df["source1_entity_id"]
                .duplicated(keep=False)
                .sum()
            ),
        }

    # --------------------------------------------------------
    # MATCHED ENTITY IDS
    # --------------------------------------------------------

    if "matched_entity_ids" in df.columns:

        matched = (
            df["matched_entity_ids"]
            .fillna("")
            .astype(str)
        )

        profile["matched_entity_ids"] = {

            "empty_count": int(
                (matched.str.strip() == "").sum()
            ),

            "sample_values": (
                matched
                .head(20)
                .tolist()
            ),

            "max_string_length": (
                int(matched.str.len().max())
                if len(matched)
                else 0
            ),
        }

    return profile


# ============================================================
# LOAD AND PROFILE SOURCE
# ============================================================

def load_and_profile_source(
    path: Path,
    loader,
    name: str
) -> dict:
    """
    Load a dataset using the project's data loader
    and generate its profile.

    IMPORTANT:
    The current data_loader.py functions such as
    load_source1(), load_source2(), and load_source3()
    do not accept a path argument.

    Therefore we call:
        loader()

    instead of:
        loader(path)
    """

    print()
    print("-" * 70)
    print(f"Loading: {name}")
    print(f"Expected path: {path}")
    print("-" * 70)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not path.exists():

        print("WARNING: File does not exist.")

        return {
            "dataset": name,
            "exists": False,
            "path": str(path),
        }

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = loader()

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    return profile_source(
        df,
        name
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("DATASET PROFILING")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE PROJECT DIRECTORIES
    # --------------------------------------------------------

    create_project_directories()

    # --------------------------------------------------------
    # REPORT STRUCTURE
    # --------------------------------------------------------

    report = {

        "project_root": str(
            PROJECT_ROOT
        ),

        "train": {},

        "test": {},

        "ground_truth": {},
    }

    # ========================================================
    # TRAIN SOURCE 1
    # ========================================================

    report["train"]["source1"] = (
        load_and_profile_source(
            TRAIN_SOURCE1,
            load_source1,
            "train_source1",
        )
    )

    # ========================================================
    # TRAIN SOURCE 2
    # ========================================================

    report["train"]["source2"] = (
        load_and_profile_source(
            TRAIN_SOURCE2,
            load_source2,
            "train_source2",
        )
    )

    # ========================================================
    # TRAIN SOURCE 3
    # ========================================================

    report["train"]["source3"] = (
        load_and_profile_source(
            TRAIN_SOURCE3,
            load_source3,
            "train_source3",
        )
    )

    # ========================================================
    # TEST SOURCE 1
    # ========================================================

    report["test"]["source1"] = (
        load_and_profile_source(
            TEST_SOURCE1,
            load_source1,
            "test_source1",
        )
    )

    # ========================================================
    # TEST SOURCE 2
    # ========================================================

    report["test"]["source2"] = (
        load_and_profile_source(
            TEST_SOURCE2,
            load_source2,
            "test_source2",
        )
    )

    # ========================================================
    # TEST SOURCE 3
    # ========================================================

    report["test"]["source3"] = (
        load_and_profile_source(
            TEST_SOURCE3,
            load_source3,
            "test_source3",
        )
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print()
    print("-" * 70)
    print("Loading: train_ground_truth")
    print(f"Expected path: {TRAIN_GROUND_TRUTH}")
    print("-" * 70)

    if TRAIN_GROUND_TRUTH.exists():

        ground_truth = load_ground_truth()

        report["ground_truth"] = (
            profile_ground_truth(
                ground_truth
            )
        )

        print(
            f"Rows: {len(ground_truth):,}"
        )

        print(
            f"Columns: {len(ground_truth.columns)}"
        )

    else:

        print(
            "WARNING: Ground truth file does not exist."
        )

        report["ground_truth"] = {
            "exists": False,
            "path": str(
                TRAIN_GROUND_TRUTH
            ),
        }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_file = (
        REPORTS_DIR /
        "phase0_profile.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Report saved to:\n{output_file}"
    )
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()