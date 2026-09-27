"""
Environment validation for the Amazon ML Challenge 2026 pipeline.
"""

import sys
from pathlib import Path


# ============================================================
# ADD PROJECT CODE TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODE_DIR = PROJECT_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from business_entity_resolution.src.config import (
    PROJECT_ROOT,
    DATASET_DIR,
    TRAIN_DIR,
    TEST_DIR,
    ARTIFACTS_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
    RANDOM_SEED,
    create_project_directories,
)


def main():

    print("=" * 60)
    print("Amazon ML Challenge 2026 - Environment Check")
    print("=" * 60)

    print(f"\nPython version : {sys.version.split()[0]}")
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Dataset dir    : {DATASET_DIR}")
    print(f"Train dir      : {TRAIN_DIR}")
    print(f"Test dir       : {TEST_DIR}")
    print(f"Artifacts dir  : {ARTIFACTS_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Logs dir       : {LOGS_DIR}")
    print(f"Random seed    : {RANDOM_SEED}")

    print("\nCreating generated directories...")

    create_project_directories()

    print("Directories created successfully.")

    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()