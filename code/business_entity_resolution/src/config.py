"""
Central project configuration.

Contains project paths and common runtime settings used throughout
the Amazon ML Challenge 2026 entity-resolution pipeline.
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# config.py:
# project/code/business_entity_resolution/src/config.py
#
# parents[0] -> src
# parents[1] -> business_entity_resolution
# parents[2] -> code
# parents[3] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# TRAIN FILES
# ============================================================

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# TEST FILES
# ============================================================

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# ============================================================
# ARTIFACT DIRECTORIES
# ============================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

AUDIT_DIR = ARTIFACTS_DIR / "audit"
NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
BLOCKING_DIR = ARTIFACTS_DIR / "blocking"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
THRESHOLDS_DIR = ARTIFACTS_DIR / "thresholds"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "output"

MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# LOGGING
# ============================================================

LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# COMMON SETTINGS
# ============================================================

TSV_SEPARATOR = "\t"
TEXT_ENCODING = "utf-8"


# ============================================================
# DIRECTORY CREATION
# ============================================================

DIRECTORIES = [
    ARTIFACTS_DIR,
    AUDIT_DIR,
    NORMALIZED_DIR,
    BLOCKING_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    THRESHOLDS_DIR,
    REPORTS_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
]


def create_project_directories() -> None:
    """Create generated project directories when they do not exist."""

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)