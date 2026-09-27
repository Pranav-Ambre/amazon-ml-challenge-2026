"""
Data loading utilities for the Amazon ML Challenge 2026.

All challenge data is expected to be provided as TSV files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    TSV_SEPARATOR,
    TEXT_ENCODING,
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================
# GENERIC TSV LOADER
# ============================================================

def load_tsv(
    path: Path,
    usecols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a TSV file.

    Parameters
    ----------
    path:
        Path to TSV file.

    usecols:
        Optional list of columns to load.

    Returns
    -------
    pandas.DataFrame
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=TEXT_ENCODING,
        usecols=usecols,
        low_memory=False,
    )


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source_columns(df: pd.DataFrame) -> None:
    """
    Validate the schema of a Source 1/2/3 dataframe.
    """

    actual_columns = list(df.columns)

    if actual_columns != SOURCE_COLUMNS:
        raise ValueError(
            "\nUnexpected source columns.\n"
            f"Expected: {SOURCE_COLUMNS}\n"
            f"Received: {actual_columns}"
        )


def validate_ground_truth_columns(df: pd.DataFrame) -> None:
    """
    Validate the schema of the ground-truth dataframe.
    """

    actual_columns = list(df.columns)

    if actual_columns != GROUND_TRUTH_COLUMNS:
        raise ValueError(
            "\nUnexpected ground-truth columns.\n"
            f"Expected: {GROUND_TRUTH_COLUMNS}\n"
            f"Received: {actual_columns}"
        )


# ============================================================
# TRAIN LOADERS
# ============================================================

def load_train_source1() -> pd.DataFrame:
    """Load training Source 1."""
    df = load_tsv(TRAIN_SOURCE1)
    validate_source_columns(df)
    return df


def load_train_source2() -> pd.DataFrame:
    """Load training Source 2."""
    df = load_tsv(TRAIN_SOURCE2)
    validate_source_columns(df)
    return df


def load_train_source3() -> pd.DataFrame:
    """Load training Source 3."""
    df = load_tsv(TRAIN_SOURCE3)
    validate_source_columns(df)
    return df


def load_train_ground_truth() -> pd.DataFrame:
    """Load training ground truth."""
    df = load_tsv(TRAIN_GROUND_TRUTH)
    validate_ground_truth_columns(df)
    return df


# ============================================================
# TEST LOADERS
# ============================================================

def load_test_source1() -> pd.DataFrame:
    """Load test Source 1."""
    df = load_tsv(TEST_SOURCE1)
    validate_source_columns(df)
    return df


def load_test_source2() -> pd.DataFrame:
    """Load test Source 2."""
    df = load_tsv(TEST_SOURCE2)
    validate_source_columns(df)
    return df


def load_test_source3() -> pd.DataFrame:
    """Load test Source 3."""
    df = load_tsv(TEST_SOURCE3)
    validate_source_columns(df)
    return df