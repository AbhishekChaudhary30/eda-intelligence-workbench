"""Data validation and semantic type inference module.

Performs conservative type inference, identifies potential identifiers heuristically,
detects mixed-type columns, and validates structural integrity without mutating the underlying data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class InferredTypeInfo:
    """Type inference details for a specific column."""

    column_name: str
    pandas_dtype: str
    semantic_type: str  # "Numeric Continuous", "Numeric Discrete", "Categorical", "Text", "Datetime", "Boolean", "Mixed"
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_boolean: bool
    is_potential_identifier: bool
    identifier_reason: str | None = None
    is_mixed_type: bool = False
    detected_python_types: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ValidationReport:
    """Report summarizing dataset validation and inferred semantic types."""

    is_valid: bool
    total_rows: int
    total_cols: int
    inferred_types: dict[str, InferredTypeInfo]
    potential_identifiers: list[str]
    constant_columns: list[str]
    all_null_columns: list[str]
    type_counts: dict[str, int]
    warnings: list[str]


IDENTIFIER_NAME_PATTERNS = re.compile(
    r"(^|_|\b)(id|identifier|uuid|key|code|account|cust(omer)?_id|user_id|ssn|pk|index)(\b|_|$)",
    re.IGNORECASE,
)


def check_column_python_types(series: pd.Series, sample_size: int = 500) -> tuple[bool, list[str]]:
    """Check if an object series contains multiple conflicting Python types."""
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return False, []

    sample = non_nulls.iloc[:sample_size]
    types_found = {type(val).__name__ for val in sample}

    is_mixed = len(types_found) > 1
    return is_mixed, sorted(types_found)


def test_is_datetime_series(series: pd.Series, sample_size: int = 200) -> bool:
    """Conservatively check whether a string/object series represents dates/datetimes.

    Only flags as datetime if >= 85% of non-null sample parses successfully
    and values have date-like structural characteristics (slashes, hyphens, colons, or ISO formats).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_nulls = series.dropna()
    if len(non_nulls) < 5:
        return False

    sample = non_nulls.astype(str).str.strip().iloc[:sample_size]

    # Quick pre-filter: must contain typical date separators
    date_like_regex = re.compile(r"[\/\-:\sT]")
    if not any(date_like_regex.search(s) for s in sample):
        return False

    # Avoid purely numeric strings like '12345' being parsed as year/epoch
    if all(s.isdigit() for s in sample):
        return False

    # Attempt parsing with dateutil
    success_count = 0
    try:
        from dateutil.parser import ParserError, parse

        for s in sample:
            if len(s) < 6:  # too short for meaningful date
                continue
            try:
                parse(s, fuzzy=False)
                success_count += 1
            except (ParserError, ValueError, OverflowError):
                pass
    except ImportError:
        # Fallback to pandas
        parsed = pd.to_datetime(sample, errors="coerce")
        success_count = int(parsed.notna().sum())

    return (success_count / len(sample)) >= 0.85


def infer_column_type(series: pd.Series, column_name: str, total_rows: int) -> InferredTypeInfo:
    """Infer the semantic type of a single column conservatively."""
    pandas_dtype_str = str(series.dtype)
    non_null_count = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))

    is_potential_id = False
    id_reason: str | None = None
    is_mixed = False
    py_types: list[str] = []

    # Check for mixed python types in object columns
    if series.dtype == object or str(series.dtype) == "string":
        is_mixed, py_types = check_column_python_types(series)

    # 1. Check if potential identifier
    if total_rows > 0 and unique_count > 0:
        uniqueness_ratio = unique_count / max(1, non_null_count)
        name_match = bool(IDENTIFIER_NAME_PATTERNS.search(column_name))

        if name_match and uniqueness_ratio >= 0.85:
            is_potential_id = True
            id_reason = f"Column name matches ID pattern and uniqueness ratio is {uniqueness_ratio:.1%}."
        elif uniqueness_ratio == 1.0 and non_null_count >= 10:
            # Strictly unique
            if pd.api.types.is_integer_dtype(series) or series.dtype == object:
                is_potential_id = True
                id_reason = "100% of non-null values are unique (Candidate primary key/identifier)."

    # 2. Boolean check
    if pd.api.types.is_bool_dtype(series):
        return InferredTypeInfo(
            column_name=column_name,
            pandas_dtype=pandas_dtype_str,
            semantic_type="Boolean",
            is_numeric=False,
            is_categorical=True,
            is_datetime=False,
            is_boolean=True,
            is_potential_identifier=is_potential_id,
            identifier_reason=id_reason,
            confidence=1.0,
        )

    # Check boolean in object or int (e.g. {0, 1} or {'yes', 'no'}, {'true', 'false'})
    if unique_count == 2 and non_null_count > 0:
        unique_vals = set(series.dropna().unique())
        bool_sets = [
            {0, 1},
            {"0", "1"},
            {True, False},
            {"true", "false"},
            {"True", "False"},
            {"yes", "no"},
            {"Yes", "No"},
            {"Y", "N"},
            {"y", "n"},
        ]
        if any(unique_vals.issubset(b_set) for b_set in bool_sets):
            return InferredTypeInfo(
                column_name=column_name,
                pandas_dtype=pandas_dtype_str,
                semantic_type="Boolean",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                is_boolean=True,
                is_potential_identifier=False,
                confidence=0.95,
            )

    # 3. Datetime check
    if pd.api.types.is_datetime64_any_dtype(series):
        return InferredTypeInfo(
            column_name=column_name,
            pandas_dtype=pandas_dtype_str,
            semantic_type="Datetime",
            is_numeric=False,
            is_categorical=False,
            is_datetime=True,
            is_boolean=False,
            is_potential_identifier=False,
            confidence=1.0,
        )

    if (series.dtype == object or str(series.dtype) == "string") and test_is_datetime_series(series):
        return InferredTypeInfo(
            column_name=column_name,
            pandas_dtype=pandas_dtype_str,
            semantic_type="Datetime",
            is_numeric=False,
            is_categorical=False,
            is_datetime=True,
            is_boolean=False,
            is_potential_identifier=False,
            confidence=0.9,
        )

    # 4. Numeric check
    if pd.api.types.is_numeric_dtype(series):
        # Distinguish Discrete vs Continuous
        if pd.api.types.is_float_dtype(series):
            sem_type = "Numeric Continuous"
        elif unique_count <= 15 and not is_potential_id:
            sem_type = "Numeric Discrete"
        else:
            sem_type = "Numeric Continuous"

        return InferredTypeInfo(
            column_name=column_name,
            pandas_dtype=pandas_dtype_str,
            semantic_type=sem_type,
            is_numeric=True,
            is_categorical=False,
            is_datetime=False,
            is_boolean=False,
            is_potential_identifier=is_potential_id,
            identifier_reason=id_reason,
            confidence=1.0,
        )

    # 5. Mixed type
    if is_mixed:
        return InferredTypeInfo(
            column_name=column_name,
            pandas_dtype=pandas_dtype_str,
            semantic_type="Mixed",
            is_numeric=False,
            is_categorical=True,
            is_datetime=False,
            is_boolean=False,
            is_potential_identifier=is_potential_id,
            identifier_reason=id_reason,
            is_mixed_type=True,
            detected_python_types=py_types,
            confidence=0.85,
        )

    # 6. Categorical vs Free Text
    # If cardinality ratio is high (> 0.6) and average string length > 30, classify as Text
    if non_null_count > 0:
        sample_lens = series.dropna().astype(str).str.len()
        avg_len = sample_lens.mean() if len(sample_lens) > 0 else 0
        uniqueness_ratio = unique_count / max(1, non_null_count)

        if uniqueness_ratio > 0.5 and avg_len > 35:
            sem_type = "Text"
        else:
            sem_type = "Categorical"
    else:
        sem_type = "Categorical"

    return InferredTypeInfo(
        column_name=column_name,
        pandas_dtype=pandas_dtype_str,
        semantic_type=sem_type,
        is_numeric=False,
        is_categorical=True,
        is_datetime=False,
        is_boolean=False,
        is_potential_identifier=is_potential_id,
        identifier_reason=id_reason,
        is_mixed_type=False,
        confidence=0.9,
    )


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate DataFrame integrity and produce semantic type report."""
    warnings: list[str] = []
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0:
        warnings.append("Dataset has zero records (empty table).")
    if total_cols == 0:
        warnings.append("Dataset has zero columns.")

    inferred_types: dict[str, InferredTypeInfo] = {}
    potential_identifiers: list[str] = []
    constant_columns: list[str] = []
    all_null_columns: list[str] = []
    type_counts: dict[str, int] = {
        "Numeric Continuous": 0,
        "Numeric Discrete": 0,
        "Categorical": 0,
        "Text": 0,
        "Datetime": 0,
        "Boolean": 0,
        "Mixed": 0,
    }

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        if null_count == total_rows and total_rows > 0:
            all_null_columns.append(col)

        if unique_count <= 1 and total_rows > 1:
            constant_columns.append(col)

        type_info = infer_column_type(series, str(col), total_rows)
        inferred_types[str(col)] = type_info

        if type_info.is_potential_identifier:
            potential_identifiers.append(str(col))

        sem = type_info.semantic_type
        type_counts[sem] = type_counts.get(sem, 0) + 1

    return ValidationReport(
        is_valid=(total_rows > 0 and total_cols > 0),
        total_rows=total_rows,
        total_cols=total_cols,
        inferred_types=inferred_types,
        potential_identifiers=potential_identifiers,
        constant_columns=constant_columns,
        all_null_columns=all_null_columns,
        type_counts=type_counts,
        warnings=warnings,
    )
