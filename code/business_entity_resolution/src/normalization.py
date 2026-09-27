"""
Normalization utilities for Business Entity Resolution.

This module creates multiple deterministic representations of
business names, addresses, and countries.

Important:
- Raw columns are never overwritten.
- Normalization is deterministic.
- Aggressive semantic transformations are avoided.
- Numeric information is preserved.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


# ============================================================
# REGEX
# ============================================================

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")

DIGIT_RE = re.compile(r"\d+")

TOKEN_RE = re.compile(r"[a-z0-9]+")


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def unicode_normalize(value: object) -> str:
    """
    Apply Unicode NFKC normalization.

    Example:
        full-width characters -> standard characters
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    return unicodedata.normalize(
        "NFKC",
        text
    )


def normalize_case(value: object) -> str:
    """
    Unicode normalize + lowercase/casefold.
    """

    text = unicode_normalize(value)

    return text.casefold()


def normalize_whitespace(value: object) -> str:
    """
    Normalize repeated whitespace.
    """

    text = normalize_case(value)

    text = WHITESPACE_RE.sub(
        " ",
        text
    )

    return text.strip()


def normalize_punctuation(value: object) -> str:
    """
    Replace punctuation with spaces while preserving
    alphanumeric content.
    """

    text = normalize_whitespace(value)

    text = NON_ALNUM_SPACE_RE.sub(
        " ",
        text
    )

    text = WHITESPACE_RE.sub(
        " ",
        text
    )

    return text.strip()


def compact_text(value: object) -> str:
    """
    Remove spaces and non-alphanumeric characters.

    Example:
        'ABC Pvt. Ltd.' -> 'abcpvt ltd' -> 'abcpvtltd'
    """

    text = normalize_case(value)

    return NON_ALNUM_RE.sub(
        "",
        text
    )


def alphanumeric_text(value: object) -> str:
    """
    Keep only ASCII letters and digits.
    """

    text = normalize_case(value)

    return NON_ALNUM_RE.sub(
        "",
        text
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value: object) -> list[str]:
    """
    Extract normalized alphanumeric tokens.
    """

    text = normalize_punctuation(value)

    if not text:
        return []

    return TOKEN_RE.findall(text)


def sorted_tokens(value: object) -> list[str]:
    """
    Return alphabetically sorted tokens.
    """

    return sorted(
        tokenize(value)
    )


def token_string(value: object) -> str:
    """
    Return tokens joined by spaces.
    """

    return " ".join(
        tokenize(value)
    )


def sorted_token_string(value: object) -> str:
    """
    Return sorted tokens joined by spaces.
    """

    return " ".join(
        sorted_tokens(value)
    )


# ============================================================
# NUMERIC INFORMATION
# ============================================================

def numeric_tokens(value: object) -> list[str]:
    """
    Extract numeric sequences.

    Numeric information is intentionally preserved because
    house numbers, postal codes, unit numbers, etc. can be
    important for entity resolution.
    """

    text = normalize_case(value)

    if not text:
        return []

    return DIGIT_RE.findall(text)


def numeric_token_string(value: object) -> str:
    """
    Return numeric tokens as a space-separated string.
    """

    return " ".join(
        numeric_tokens(value)
    )


# ============================================================
# ALPHA TOKENS
# ============================================================

def alpha_tokens(value: object) -> list[str]:
    """
    Extract alphabetic tokens only.
    """

    tokens = tokenize(value)

    return [
        token
        for token in tokens
        if token.isalpha()
    ]


def alpha_token_string(value: object) -> str:
    """
    Return alphabetic tokens joined by spaces.
    """

    return " ".join(
        alpha_tokens(value)
    )


# ============================================================
# ADDRESS-SPECIFIC HELPERS
# ============================================================

def extract_house_number(value: object) -> str:
    """
    Extract the first numeric sequence from an address.

    This is intentionally conservative.
    """

    numbers = numeric_tokens(value)

    if not numbers:
        return ""

    return numbers[0]


def extract_postal_like_tokens(
    value: object
) -> list[str]:
    """
    Extract numeric tokens that may represent postal codes.

    This does NOT assume a specific country's postal format.
    """

    numbers = numeric_tokens(value)

    return [
        number
        for number in numbers
        if len(number) >= 4
    ]


def postal_token_string(value: object) -> str:
    """
    Return postal-like numeric tokens.
    """

    return " ".join(
        extract_postal_like_tokens(value)
    )


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(value: object) -> str:
    """
    Conservative country normalization.

    We deliberately do not map country names to ISO codes here
    because the profile shows that country values are short and
    can include multiple formats.

    Country-specific mappings can be added later after validation.
    """

    return normalize_whitespace(value)


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_business_name(
    value: object
) -> dict[str, object]:
    """
    Generate multiple normalized business-name representations.
    """

    return {
        "business_name_norm":
            normalize_punctuation(value),

        "business_name_compact":
            compact_text(value),

        "business_name_alnum":
            alphanumeric_text(value),

        "business_name_tokens":
            token_string(value),

        "business_name_sorted_tokens":
            sorted_token_string(value),

        "business_name_numeric_tokens":
            numeric_token_string(value),
    }


# ============================================================
# BUSINESS ADDRESS NORMALIZATION
# ============================================================

def normalize_business_address(
    value: object
) -> dict[str, object]:
    """
    Generate multiple normalized address representations.
    """

    return {
        "business_address_norm":
            normalize_punctuation(value),

        "business_address_compact":
            compact_text(value),

        "business_address_alnum":
            alphanumeric_text(value),

        "business_address_tokens":
            token_string(value),

        "business_address_sorted_tokens":
            sorted_token_string(value),

        "business_address_alpha_tokens":
            alpha_token_string(value),

        "business_address_numeric_tokens":
            numeric_token_string(value),

        "business_address_house_number":
            extract_house_number(value),

        "business_address_postal_tokens":
            postal_token_string(value),
    }


# ============================================================
# DATAFRAME NORMALIZATION
# ============================================================

def normalize_source_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add normalized representations to a source dataframe.

    Original columns are preserved.
    """

    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = df.copy()

    # --------------------------------------------------------
    # BUSINESS NAME
    # --------------------------------------------------------

    name_normalized = (
        result["business_name"]
        .apply(normalize_business_name)
        .apply(pd.Series)
    )

    # --------------------------------------------------------
    # BUSINESS ADDRESS
    # --------------------------------------------------------

    address_normalized = (
        result["business_address"]
        .apply(normalize_business_address)
        .apply(pd.Series)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    result["country_norm"] = (
        result["country"]
        .apply(normalize_country)
    )

    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    result = pd.concat(
        [
            result,
            name_normalized,
            address_normalized,
        ],
        axis=1
    )

    return result


# ============================================================
# NORMALIZE ITERABLE OF VALUES
# ============================================================

def normalize_values(
    values: Iterable[object]
) -> list[str]:
    """
    Normalize an iterable of values.
    """

    return [
        normalize_punctuation(value)
        for value in values
    ]


# ============================================================
# NORMALIZATION VERSION
# ============================================================

NORMALIZATION_VERSION = "v1.0.0"