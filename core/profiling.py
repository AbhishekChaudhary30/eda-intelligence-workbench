"""Dataset profiling engine for EDA Intelligence Workbench.

Calculates comprehensive structural, numeric, categorical, datetime, and boolean
profiles for whole datasets and individual columns with explicit edge-case handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.validation import InferredTypeInfo, validate_dataset
from utils.formatting import format_bytes


@dataclass
class ColumnProfile:
    """Comprehensive statistical and structural profile of a single column."""

    column_name: str
    dtype_str: str
    semantic_type: str
    total_count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    uniqueness_ratio: float
    memory_bytes: int
    is_constant: bool
    is_all_null: bool
    is_potential_identifier: bool
    identifier_reason: str | None = None

    # Numeric specific
    mean: float | None = None
    std: float | None = None
    variance: float | None = None
    min_val: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max_val: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    is_zero_variance: bool = False

    # Categorical specific
    top_values: list[tuple[str, int, float]] = field(default_factory=list)  # (val, count, pct)
    empty_string_count: int = 0
    whitespace_only_count: int = 0

    # Datetime specific
    min_date: str | None = None
    max_date: str | None = None
    temporal_span_days: float | None = None
    duplicate_timestamps: int = 0

    # Boolean specific
    true_count: int | None = None
    true_percentage: float | None = None
    false_count: int | None = None
    false_percentage: float | None = None


@dataclass
class DatasetProfile:
    """High-level structural profile of the entire dataset."""

    total_rows: int
    total_cols: int
    total_cells: int
    total_memory_bytes: int
    formatted_memory: str
    total_missing_cells: int
    missing_cells_percentage: float
    complete_rows_count: int
    complete_rows_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    numeric_columns_count: int
    categorical_columns_count: int
    datetime_columns_count: int
    boolean_columns_count: int
    constant_columns: list[str]
    all_null_columns: list[str]
    potential_identifiers: list[str]
    high_cardinality_columns: list[str]
    column_profiles: dict[str, ColumnProfile]

    @property
    def duplicate_row_count(self) -> int:
        """Alias for duplicate_rows_count."""
        return self.duplicate_rows_count


def profile_numeric_column(
    series: pd.Series,
    col_name: str,
    base_kwargs: dict[str, Any],
) -> ColumnProfile:
    """Calculate descriptive metrics for numeric series handling edge cases."""
    clean_series = series.dropna()
    n = len(clean_series)

    if n == 0:
        return ColumnProfile(
            is_zero_variance=True,
            **base_kwargs,
        )

    # Cast to float64 and filter to finite values for stable computation
    try:
        raw_vals = clean_series.astype(float).values
    except (ValueError, TypeError):
        raw_vals = pd.to_numeric(clean_series, errors="coerce").dropna().values

    finite_mask = np.isfinite(raw_vals)
    vals = raw_vals[finite_mask]
    n = len(vals)

    if n == 0:
        return ColumnProfile(
            is_zero_variance=True,
            **base_kwargs,
        )

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    var_val = float(np.var(vals, ddof=1)) if n > 1 else 0.0
    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    q25 = float(np.percentile(vals, 25))
    median_v = float(np.median(vals))
    q75 = float(np.percentile(vals, 75))
    iqr_val = float(q75 - q25)

    is_zero_var = (std_val == 0.0) or (min_v == max_v)

    # Skewness & kurtosis (valid if n >= 3 and std > 0)
    skew_val: float | None = None
    kurt_val: float | None = None
    if n >= 3 and not is_zero_var:
        try:
            sk = float(stats.skew(vals, bias=False, nan_policy="omit"))
            if not math.isnan(sk) and not math.isinf(sk):
                skew_val = sk
        except (ValueError, ZeroDivisionError, FloatingPointError):
            skew_val = None

        try:
            kt = float(stats.kurtosis(vals, bias=False, nan_policy="omit"))
            if not math.isnan(kt) and not math.isinf(kt):
                kurt_val = kt
        except (ValueError, ZeroDivisionError, FloatingPointError):
            kurt_val = None

    return ColumnProfile(
        mean=mean_val,
        std=std_val,
        variance=var_val,
        min_val=min_v,
        q25=q25,
        median=median_v,
        q75=q75,
        max_val=max_v,
        iqr=iqr_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        is_zero_variance=is_zero_var,
        **base_kwargs,
    )


def profile_categorical_column(
    series: pd.Series,
    base_kwargs: dict[str, Any],
) -> ColumnProfile:
    """Calculate frequency metrics for categorical / text series."""
    clean = series.dropna()
    non_null_n = len(clean)

    empty_str_count = 0
    whitespace_count = 0

    if non_null_n > 0:
        str_series = clean.astype(str)
        empty_str_count = int((str_series == "").sum())
        whitespace_count = int(
            ((str_series != "") & (str_series.str.strip() == "")).sum()
        )

        # Top 5 most frequent values
        val_counts = str_series.value_counts().head(5)
        top_vals: list[tuple[str, int, float]] = []
        for val, count in val_counts.items():
            pct = (count / non_null_n) * 100.0
            top_vals.append((str(val), int(count), round(pct, 2)))
    else:
        top_vals = []

    return ColumnProfile(
        top_values=top_vals,
        empty_string_count=empty_str_count,
        whitespace_only_count=whitespace_count,
        **base_kwargs,
    )


def profile_datetime_column(
    series: pd.Series,
    base_kwargs: dict[str, Any],
) -> ColumnProfile:
    """Calculate temporal metrics for datetime series."""
    dt_series = pd.to_datetime(series, errors="coerce").dropna()
    n = len(dt_series)

    if n == 0:
        return ColumnProfile(**base_kwargs)

    min_d = dt_series.min()
    max_d = dt_series.max()
    span_days = float((max_d - min_d).total_seconds() / 86400.0) if n > 1 else 0.0
    dup_timestamps = int(dt_series.duplicated().sum())

    return ColumnProfile(
        min_date=str(min_d),
        max_date=str(max_d),
        temporal_span_days=round(span_days, 2),
        duplicate_timestamps=dup_timestamps,
        **base_kwargs,
    )


def profile_boolean_column(
    series: pd.Series,
    base_kwargs: dict[str, Any],
) -> ColumnProfile:
    """Calculate true/false metrics for boolean series."""
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return ColumnProfile(
            true_count=0,
            true_percentage=0.0,
            false_count=0,
            false_percentage=0.0,
            **base_kwargs,
        )

    # Convert to standard boolean representation
    truthy_vals = {True, "1", "true", "True", "yes", "Yes", "Y", "y"}
    true_count = int(clean.map(lambda x: x in truthy_vals).sum())
    false_count = n - true_count

    return ColumnProfile(
        true_count=true_count,
        true_percentage=round((true_count / n) * 100.0, 1),
        false_count=false_count,
        false_percentage=round((false_count / n) * 100.0, 1),
        **base_kwargs,
    )


def compute_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """Generate high-level and detailed per-column profiles of the DataFrame."""
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    val_report = validate_dataset(df)

    # Memory usage
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except (ValueError, TypeError):
        mem_bytes = int(df.memory_usage().sum())

    # Missing & duplicate counts
    null_counts = df.isna().sum()
    total_missing = int(null_counts.sum())
    missing_pct = (total_missing / max(1, total_cells)) * 100.0

    complete_rows = int(df.dropna().shape[0]) if total_rows > 0 else 0
    complete_pct = (complete_rows / max(1, total_rows)) * 100.0

    dup_rows = int(df.duplicated().sum()) if total_rows > 0 else 0
    dup_pct = (dup_rows / max(1, total_rows)) * 100.0

    high_card_cols: list[str] = []
    column_profiles: dict[str, ColumnProfile] = {}

    num_count = 0
    cat_count = 0
    dt_count = 0
    bool_count = 0

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        type_info: InferredTypeInfo = val_report.inferred_types[col_str]

        non_null_n = int(series.notna().sum())
        null_n = total_rows - non_null_n
        null_col_pct = (null_n / max(1, total_rows)) * 100.0
        distinct_n = int(series.nunique(dropna=True))
        uniq_ratio = distinct_n / max(1, non_null_n)

        try:
            col_mem = int(series.memory_usage(deep=True))
        except (ValueError, TypeError):
            col_mem = int(series.memory_usage())

        is_const = distinct_n <= 1 and total_rows > 1
        is_all_null = non_null_n == 0 and total_rows > 0

        # Flag high cardinality for categorical
        if type_info.is_categorical and distinct_n > 50 and uniq_ratio > 0.4:
            high_card_cols.append(col_str)

        base_kwargs = {
            "column_name": col_str,
            "dtype_str": type_info.pandas_dtype,
            "semantic_type": type_info.semantic_type,
            "total_count": total_rows,
            "non_null_count": non_null_n,
            "null_count": null_n,
            "null_percentage": round(null_col_pct, 2),
            "distinct_count": distinct_n,
            "uniqueness_ratio": round(uniq_ratio, 4),
            "memory_bytes": col_mem,
            "is_constant": is_const,
            "is_all_null": is_all_null,
            "is_potential_identifier": type_info.is_potential_identifier,
            "identifier_reason": type_info.identifier_reason,
        }

        if type_info.is_boolean:
            bool_count += 1
            col_prof = profile_boolean_column(series, base_kwargs)
        elif type_info.is_datetime:
            dt_count += 1
            col_prof = profile_datetime_column(series, base_kwargs)
        elif type_info.is_numeric:
            num_count += 1
            col_prof = profile_numeric_column(series, col_str, base_kwargs)
        else:
            cat_count += 1
            col_prof = profile_categorical_column(series, base_kwargs)

        column_profiles[col_str] = col_prof

    return DatasetProfile(
        total_rows=total_rows,
        total_cols=total_cols,
        total_cells=total_cells,
        total_memory_bytes=mem_bytes,
        formatted_memory=format_bytes(mem_bytes),
        total_missing_cells=total_missing,
        missing_cells_percentage=round(missing_pct, 2),
        complete_rows_count=complete_rows,
        complete_rows_percentage=round(complete_pct, 1),
        duplicate_rows_count=dup_rows,
        duplicate_rows_percentage=round(dup_pct, 1),
        numeric_columns_count=num_count,
        categorical_columns_count=cat_count,
        datetime_columns_count=dt_count,
        boolean_columns_count=bool_count,
        constant_columns=val_report.constant_columns,
        all_null_columns=val_report.all_null_columns,
        potential_identifiers=val_report.potential_identifiers,
        high_cardinality_columns=high_card_cols,
        column_profiles=column_profiles,
    )
