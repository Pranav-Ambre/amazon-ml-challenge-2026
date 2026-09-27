"""
Dataset audit for Amazon ML Challenge 2026.

Checks:
- file existence
- schema
- row counts
- dtypes
- nulls
- empty strings
- duplicate IDs
- ID prefixes
- country values
- name/address lengths
- ground-truth statistics
"""

import json
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
# IMPORTS
# ============================================================

from business_entity_resolution.src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    AUDIT_DIR,
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
# HELPERS
# ============================================================

def dataframe_audit(
    df: pd.DataFrame,
    name: str,
) -> dict:
    """
    Generate basic statistics for a dataframe.
    """

    result = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "null_counts": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Empty strings
    empty_strings = {}

    for column in df.columns:
        if df[column].dtype == "object":
            empty_strings[column] = int(
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

    result["empty_string_counts"] = empty_strings

    # Entity ID checks
    if "entity_id" in df.columns:

        result["unique_entity_ids"] = int(
            df["entity_id"].nunique(dropna=True)
        )

        result["duplicate_entity_ids"] = int(
            df["entity_id"].duplicated().sum()
        )

        result["sample_entity_ids"] = (
            df["entity_id"]
            .dropna()
            .astype(str)
            .head(10)
            .tolist()
        )

    # Country information
    if "country" in df.columns:

        result["country_counts"] = {
            str(country): int(count)
            for country, count in
            df["country"]
            .fillna("<NULL>")
            .astype(str)
            .value_counts()
            .items()
        }

    # Name length statistics
    if "business_name" in df.columns:

        lengths = (
            df["business_name"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        result["business_name_length"] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        }

    # Address length statistics
    if "business_address" in df.columns:

        lengths = (
            df["business_address"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        result["business_address_length"] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        }

    return result


# ============================================================
# GROUND TRUTH AUDIT
# ============================================================

def ground_truth_audit(
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Analyze ground-truth match counts.
    """

    match_counts = (
        ground_truth["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .apply(
            lambda value: 0
            if not value.strip()
            else len(
                [
                    x
                    for x in value.split(",")
                    if x.strip()
                ]
            )
        )
    )

    return {
        "rows": int(len(ground_truth)),
        "zero_match_entities": int(
            (match_counts == 0).sum()
        ),
        "singleton_entities": int(
            (match_counts == 1).sum()
        ),
        "multi_match_entities": int(
            (match_counts > 1).sum()
        ),
        "maximum_matches": int(
            match_counts.max()
        ),
        "average_matches": float(
            match_counts.mean()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Amazon ML Challenge 2026 - Dataset Audit")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    files = {
        "train_source1": TRAIN_SOURCE1,
        "train_source2": TRAIN_SOURCE2,
        "train_source3": TRAIN_SOURCE3,
        "train_ground_truth": TRAIN_GROUND_TRUTH,
        "test_source1": TEST_SOURCE1,
        "test_source2": TEST_SOURCE2,
        "test_source3": TEST_SOURCE3,
    }

    print("\nChecking dataset files...")

    missing_files = []

    for name, path in files.items():

        exists = path.exists()

        print(
            f"{name:25} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing_files.append(str(path))

    if missing_files:

        print("\nERROR: Missing files:")

        for path in missing_files:
            print(f"  {path}")

        raise FileNotFoundError(
            "Dataset audit cannot continue."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    datasets = {
        "train_source1": load_train_source1(),
        "train_source2": load_train_source2(),
        "train_source3": load_train_source3(),
        "train_ground_truth": load_train_ground_truth(),
        "test_source1": load_test_source1(),
        "test_source2": load_test_source2(),
        "test_source3": load_test_source3(),
    }

    print("All datasets loaded successfully.")

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit = {
        "datasets": {},
        "ground_truth": {},
    }

    for name, df in datasets.items():

        print(f"\nAuditing {name}...")

        audit["datasets"][name] = dataframe_audit(
            df,
            name,
        )

        print(
            f"Rows: "
            f"{audit['datasets'][name]['rows']:,}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\nAuditing ground truth...")

    audit["ground_truth"] = ground_truth_audit(
        datasets["train_ground_truth"]
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = AUDIT_DIR / "dataset_audit.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nAudit saved to:\n{output_file}"
    )

    print("\nDataset audit completed successfully.")


if __name__ == "__main__":
    main()