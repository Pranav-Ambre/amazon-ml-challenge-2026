from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "code"
        / "business_entity_resolution"
    )
)


# ============================================================
# IMPORTS
# ============================================================

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    NORMALIZED_DIR,
    create_project_directories,
)

from src.data_loader import (
    load_train_source1,
    load_train_source2,
    load_train_source3,
    load_test_source1,
    load_test_source2,
    load_test_source3,
)

from src.normalization import (
    normalize_source_dataframe,
    NORMALIZATION_VERSION,
)


# ============================================================
# NORMALIZE ONE DATASET
# ============================================================

def normalize_dataset(
    loader,
    input_path: Path,
    output_path: Path,
    dataset_name: str,
):
    """
    Load one dataset, normalize it, and save the
    normalized result as Parquet.
    """

    print()
    print("=" * 70)
    print(f"NORMALIZING: {dataset_name}")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = loader()

    print(
        f"Loaded {len(df):,} rows"
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    print("\nApplying normalization...")

    normalized_df = normalize_source_dataframe(
        df
    )

    print(
        f"Normalized {len(normalized_df):,} rows"
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\nWriting Parquet...")

    normalized_df.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    print("\nVerifying output...")

    if not output_path.exists():
        raise RuntimeError(
            f"Output was not created: {output_path}"
        )

    print(
        f"Output size: "
        f"{output_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print(
        f"Columns: {len(normalized_df.columns)}"
    )

    print(
        f"Normalization version: "
        f"{NORMALIZATION_VERSION}"
    )

    print(
        f"Completed: {dataset_name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("NORMALIZATION PIPELINE")
    print("=" * 70)

    create_project_directories()

    # ========================================================
    # TRAIN DATA
    # ========================================================

    normalize_dataset(
        loader=load_train_source1,
        input_path=TRAIN_SOURCE1,
        output_path=(
            NORMALIZED_DIR
            / "train_source1_normalized.parquet"
        ),
        dataset_name="train_source1",
    )

    normalize_dataset(
        loader=load_train_source2,
        input_path=TRAIN_SOURCE2,
        output_path=(
            NORMALIZED_DIR
            / "train_source2_normalized.parquet"
        ),
        dataset_name="train_source2",
    )

    normalize_dataset(
        loader=load_train_source3,
        input_path=TRAIN_SOURCE3,
        output_path=(
            NORMALIZED_DIR
            / "train_source3_normalized.parquet"
        ),
        dataset_name="train_source3",
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    normalize_dataset(
        loader=load_test_source1,
        input_path=TEST_SOURCE1,
        output_path=(
            NORMALIZED_DIR
            / "test_source1_normalized.parquet"
        ),
        dataset_name="test_source1",
    )

    normalize_dataset(
        loader=load_test_source2,
        input_path=TEST_SOURCE2,
        output_path=(
            NORMALIZED_DIR
            / "test_source2_normalized.parquet"
        ),
        dataset_name="test_source2",
    )

    normalize_dataset(
        loader=load_test_source3,
        input_path=TEST_SOURCE3,
        output_path=(
            NORMALIZED_DIR
            / "test_source3_normalized.parquet"
        ),
        dataset_name="test_source3",
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Normalized files stored in:\n"
        f"{NORMALIZED_DIR}"
    )


if __name__ == "__main__":
    main()