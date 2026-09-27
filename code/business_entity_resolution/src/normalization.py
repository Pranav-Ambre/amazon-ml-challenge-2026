"""
Optimized deterministic normalization utilities
for Business Entity Resolution.

The public normalization helpers are preserved for compatibility.

The dataframe implementation is optimized to:
- avoid repeated Python-level normalization of the same values
- reuse intermediate normalized Series
- use pandas vectorized string operations where possible
- preserve the existing output schema
- preserve numeric information
- preserve deterministic behavior
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
    """Apply Unicode NFKC normalization."""

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return unicodedata.normalize("NFKC", str(value))


def normalize_case(value: object) -> str:
    """Unicode normalize + lowercase/casefold."""

    return unicode_normalize(value).casefold()


def normalize_whitespace(value: object) -> str:
    """Normalize repeated whitespace."""

    text = normalize_case(value)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_punctuation(value: object) -> str:
    """Replace punctuation with spaces while preserving alphanumeric content."""

    text = normalize_whitespace(value)
    text = NON_ALNUM_SPACE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def compact_text(value: object) -> str:
    """Remove spaces and non-alphanumeric characters."""

    text = normalize_case(value)
    return NON_ALNUM_RE.sub("", text)


def alphanumeric_text(value: object) -> str:
    """Keep only ASCII letters and digits."""

    text = normalize_case(value)
    return NON_ALNUM_RE.sub("", text)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value: object) -> list[str]:
    """Extract normalized alphanumeric tokens."""

    text = normalize_punctuation(value)

    if not text:
        return []

    return TOKEN_RE.findall(text)


def sorted_tokens(value: object) -> list[str]:
    """Return alphabetically sorted tokens."""

    return sorted(tokenize(value))


def token_string(value: object) -> str:
    """Return tokens joined by spaces."""

    return " ".join(tokenize(value))


def sorted_token_string(value: object) -> str:
    """Return sorted tokens joined by spaces."""

    return " ".join(sorted_tokens(value))


# ============================================================
# NUMERIC INFORMATION
# ============================================================

def numeric_tokens(value: object) -> list[str]:
    """Extract numeric sequences."""

    text = normalize_case(value)

    if not text:
        return []

    return DIGIT_RE.findall(text)


def numeric_token_string(value: object) -> str:
    """Return numeric tokens as a space-separated string."""

    return " ".join(numeric_tokens(value))


# ============================================================
# ALPHA TOKENS
# ============================================================

def alpha_tokens(value: object) -> list[str]:
    """Extract alphabetic tokens only."""

    return [
        token
        for token in tokenize(value)
        if token.isalpha()
    ]


def alpha_token_string(value: object) -> str:
    """Return alphabetic tokens joined by spaces."""

    return " ".join(alpha_tokens(value))


# ============================================================
# ADDRESS-SPECIFIC HELPERS
# ============================================================

def extract_house_number(value: object) -> str:
    """Extract the first numeric sequence from an address."""

    numbers = numeric_tokens(value)

    if not numbers:
        return ""

    return numbers[0]


def extract_postal_like_tokens(value: object) -> list[str]:
    """Extract numeric tokens with at least four digits."""

    numbers = numeric_tokens(value)

    return [
        number
        for number in numbers
        if len(number) >= 4
    ]


def postal_token_string(value: object) -> str:
    """Return postal-like numeric tokens."""

    return " ".join(extract_postal_like_tokens(value))


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(value: object) -> str:
    """
    Conservative country normalization.

    Country-specific mappings are intentionally not applied.
    """

    return normalize_whitespace(value)


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_business_name(
    value: object,
) -> dict[str, object]:
    """Generate multiple normalized business-name representations."""

    return {
        "business_name_norm": normalize_punctuation(value),
        "business_name_compact": compact_text(value),
        "business_name_alnum": alphanumeric_text(value),
        "business_name_tokens": token_string(value),
        "business_name_sorted_tokens": sorted_token_string(value),
        "business_name_numeric_tokens": numeric_token_string(value),
    }


# ============================================================
# BUSINESS ADDRESS NORMALIZATION
# ============================================================

def normalize_business_address(
    value: object,
) -> dict[str, object]:
    """Generate multiple normalized address representations."""

    return {
        "business_address_norm": normalize_punctuation(value),
        "business_address_compact": compact_text(value),
        "business_address_alnum": alphanumeric_text(value),
        "business_address_tokens": token_string(value),
        "business_address_sorted_tokens": sorted_token_string(value),
        "business_address_alpha_tokens": alpha_token_string(value),
        "business_address_numeric_tokens": numeric_token_string(value),
        "business_address_house_number": extract_house_number(value),
        "business_address_postal_tokens": postal_token_string(value),
    }


# ============================================================
# OPTIMIZED SERIES HELPERS
# ============================================================

def _prepare_text_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to strings while preserving the
    original normalization semantics for null values.
    """

    # fillna("") handles None, NaN and pandas missing values.
    return series.fillna("").astype(str)


def _casefold_series(series: pd.Series) -> pd.Series:
    """
    NFKC normalize + casefold a complete Series.

    This is the only unavoidable Python-level operation here
    because Python's unicodedata.normalize is not exposed as
    a native pandas string method.
    """

    return _prepare_text_series(series).map(
        lambda x: unicodedata.normalize("NFKC", x).casefold()
    )


def _punctuation_series(casefolded: pd.Series) -> pd.Series:
    """Normalize punctuation and whitespace from an already casefolded Series."""

    result = casefolded.str.replace(
        WHITESPACE_RE,
        " ",
        regex=True,
    )

    result = result.str.replace(
        NON_ALNUM_SPACE_RE,
        " ",
        regex=True,
    )

    result = result.str.replace(
        WHITESPACE_RE,
        " ",
        regex=True,
    )

    return result.str.strip()


def _compact_series(casefolded: pd.Series) -> pd.Series:
    """Remove all non-alphanumeric characters."""

    return casefolded.str.replace(
        NON_ALNUM_RE,
        "",
        regex=True,
    )


def _tokens_series(punctuation: pd.Series) -> pd.Series:
    """Extract normalized token strings."""

    return punctuation.str.findall(TOKEN_RE).str.join(" ")


def _sorted_tokens_series(punctuation: pd.Series) -> pd.Series:
    """Extract tokens and sort them alphabetically."""

    token_lists = punctuation.str.findall(TOKEN_RE)

    return token_lists.map(
        lambda tokens: " ".join(sorted(tokens))
    )


def _numeric_tokens_series(casefolded: pd.Series) -> pd.Series:
    """Extract numeric sequences."""

    return casefolded.str.findall(DIGIT_RE).str.join(" ")


def _alpha_tokens_series(punctuation: pd.Series) -> pd.Series:
    """Extract alphabetic tokens only."""

    token_lists = punctuation.str.findall(TOKEN_RE)

    return token_lists.map(
        lambda tokens: " ".join(
            token for token in tokens
            if token.isalpha()
        )
    )


def _house_number_series(casefolded: pd.Series) -> pd.Series:
    """Extract first numeric sequence."""

    extracted = casefolded.str.extract(
        r"(\d+)",
        expand=False,
    )

    return extracted.fillna("")


def _postal_tokens_series(casefolded: pd.Series) -> pd.Series:
    """
    Extract numeric sequences containing at least four digits.

    This preserves the original behavior of
    extract_postal_like_tokens().
    """

    number_lists = casefolded.str.findall(DIGIT_RE)

    return number_lists.map(
        lambda numbers: " ".join(
            number
            for number in numbers
            if len(number) >= 4
        )
    )


# ============================================================
# OPTIMIZED DATAFRAME NORMALIZATION
# ============================================================

def normalize_source_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized representations to a source dataframe.

    Original columns are preserved.

    Output columns are compatible with normalization v1.0.0.
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

    # --------------------------------------------------------
    # PRESERVE ORIGINAL DATA
    # --------------------------------------------------------

    result = df.copy()

    # --------------------------------------------------------
    # BUSINESS NAME
    # --------------------------------------------------------

    name_casefold = _casefold_series(
        result["business_name"]
    )

    name_norm = _punctuation_series(
        name_casefold
    )

    name_compact = _compact_series(
        name_casefold
    )

    name_tokens = _tokens_series(
        name_norm
    )

    name_sorted_tokens = _sorted_tokens_series(
        name_norm
    )

    name_numeric_tokens = _numeric_tokens_series(
        name_casefold
    )

    result["business_name_norm"] = name_norm
    result["business_name_compact"] = name_compact
    result["business_name_alnum"] = name_compact
    result["business_name_tokens"] = name_tokens
    result["business_name_sorted_tokens"] = name_sorted_tokens
    result["business_name_numeric_tokens"] = name_numeric_tokens

    # --------------------------------------------------------
    # BUSINESS ADDRESS
    # --------------------------------------------------------

    address_casefold = _casefold_series(
        result["business_address"]
    )

    address_norm = _punctuation_series(
        address_casefold
    )

    address_compact = _compact_series(
        address_casefold
    )

    address_tokens = _tokens_series(
        address_norm
    )

    address_sorted_tokens = _sorted_tokens_series(
        address_norm
    )

    address_alpha_tokens = _alpha_tokens_series(
        address_norm
    )

    address_numeric_tokens = _numeric_tokens_series(
        address_casefold
    )

    address_house_number = _house_number_series(
        address_casefold
    )

    address_postal_tokens = _postal_tokens_series(
        address_casefold
    )

    result["business_address_norm"] = address_norm
    result["business_address_compact"] = address_compact
    result["business_address_alnum"] = address_compact
    result["business_address_tokens"] = address_tokens
    result["business_address_sorted_tokens"] = address_sorted_tokens
    result["business_address_alpha_tokens"] = address_alpha_tokens
    result["business_address_numeric_tokens"] = address_numeric_tokens
    result["business_address_house_number"] = address_house_number
    result["business_address_postal_tokens"] = address_postal_tokens

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_casefold = _casefold_series(
        result["country"]
    )

    country_norm = (
        country_casefold
        .str.replace(WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )

    result["country_norm"] = country_norm

    return result


# ============================================================
# NORMALIZE ITERABLE OF VALUES
# ============================================================

def normalize_values(
    values: Iterable[object],
) -> list[str]:
    """Normalize an iterable of values."""

    return [
        normalize_punctuation(value)
        for value in values
    ]


# ============================================================
# NORMALIZATION VERSION
# ============================================================

NORMALIZATION_VERSION = "v1.1.0-optimized"