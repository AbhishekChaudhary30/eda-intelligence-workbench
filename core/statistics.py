"""Reusable statistical analysis engine for EDA Intelligence Workbench.

Provides robust descriptive statistics, Pearson and Spearman correlation matrices,
top correlation pair extraction with degenerate input handling, categorical frequency distributions,
and group-wise aggregation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.profiling import DatasetProfile


@dataclass
class CorrelatedPair:
    """Pair of columns with calculated linear or monotonic correlation."""

    column_a: str
    column_b: str
    correlation: float
    abs_correlation: float
    method: str
    sample_size: int
    strength: str  # "Very Strong", "Strong", "Moderate", "Weak"
    direction: str  # "Positive", "Negative", "None"


@dataclass
class CorrelationAnalysisResult:
    """Structured correlation analysis output."""

    method: str
    matrix: pd.DataFrame
    top_pairs: list[CorrelatedPair]
    excluded_columns: list[str]
    warnings: list[str]
    disclaimer: str = (
        "Correlation measures mathematical association, NOT causation. A high correlation "
        "does not establish that changes in one variable cause changes in the other."
    )


def compute_descriptive_stats_table(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Generate a clean, unified descriptive statistics table for all numeric columns."""
    numeric_cols = [
        col for col, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null
    ]

    if not numeric_cols:
        return pd.DataFrame()

    records = []
    for col in numeric_cols:
        p = profile.column_profiles[col]
        records.append({
            "Column": col,
            "Type": p.semantic_type,
            "Count": p.non_null_count,
            "Missing": p.null_count,
            "Missing %": f"{p.null_percentage:.1f}%",
            "Mean": round(p.mean, 3) if p.mean is not None else np.nan,
            "Std Dev": round(p.std, 3) if p.std is not None else np.nan,
            "Min": round(p.min_val, 3) if p.min_val is not None else np.nan,
            "25% (Q1)": round(p.q25, 3) if p.q25 is not None else np.nan,
            "Median": round(p.median, 3) if p.median is not None else np.nan,
            "75% (Q3)": round(p.q75, 3) if p.q75 is not None else np.nan,
            "Max": round(p.max_val, 3) if p.max_val is not None else np.nan,
            "IQR": round(p.iqr, 3) if p.iqr is not None else np.nan,
            "Skewness": round(p.skewness, 2) if p.skewness is not None else np.nan,
            "Kurtosis": round(p.kurtosis, 2) if p.kurtosis is not None else np.nan,
            "Zero Variance": "Yes" if p.is_zero_variance else "No",
        })

    return pd.DataFrame(records)


def compute_correlations(
    df: pd.DataFrame,
    profile: DatasetProfile,
    method: str = "pearson",
    max_columns: int = 50,
) -> CorrelationAnalysisResult:
    """Compute Pearson or Spearman correlation matrix with degenerate input guards.

    Args:
        df: Input DataFrame.
        profile: DatasetProfile.
        method: "pearson" or "spearman".
        max_columns: Guardrail limit on maximum columns to correlate simultaneously.

    Returns:
        CorrelationAnalysisResult containing matrix, top pairs, and warnings.
    """
    warnings: list[str] = []
    excluded: list[str] = []

    # Filter strictly to numeric columns that have non-zero variance and are not all-null
    valid_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_str = str(col)
        col_prof = profile.column_profiles.get(col_str)
        if col_prof:
            if col_prof.is_all_null:
                excluded.append(f"{col_str} (All null)")
                continue
            if col_prof.is_constant or col_prof.is_zero_variance:
                excluded.append(f"{col_str} (Zero variance / constant)")
                continue
            if col_prof.non_null_count < 3:
                excluded.append(f"{col_str} (Fewer than 3 valid observations)")
                continue
        valid_cols.append(col)

    if len(valid_cols) > max_columns:
        warnings.append(
            f"Dataset has {len(valid_cols)} numeric columns. Truncated to first {max_columns} "
            "to prevent browser performance degradation."
        )
        valid_cols = valid_cols[:max_columns]

    if len(valid_cols) < 2:
        return CorrelationAnalysisResult(
            method=method,
            matrix=pd.DataFrame(),
            top_pairs=[],
            excluded_columns=excluded,
            warnings=["At least two non-constant numeric columns with >= 3 observations are required for correlation."],
        )

    # Compute correlation with pandas (pairwise complete)
    sub_df = df[valid_cols].astype(float)
    corr_matrix = sub_df.corr(method=method).round(4)

    # Extract top unique pairs
    top_pairs: list[CorrelatedPair] = []
    cols_list = list(corr_matrix.columns)
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            col_a = cols_list[i]
            col_b = cols_list[j]
            r = corr_matrix.loc[col_a, col_b]

            if math.isnan(r):
                continue

            abs_r = abs(r)
            if abs_r >= 0.8:
                strength = "Very Strong"
            elif abs_r >= 0.6:
                strength = "Strong"
            elif abs_r >= 0.35:
                strength = "Moderate"
            else:
                strength = "Weak"

            direction = "Positive" if r > 0.05 else ("Negative" if r < -0.05 else "None")
            sample_n = len(sub_df[[col_a, col_b]].dropna())

            top_pairs.append(
                CorrelatedPair(
                    column_a=str(col_a),
                    column_b=str(col_b),
                    correlation=float(r),
                    abs_correlation=float(abs_r),
                    method=method.capitalize(),
                    sample_size=sample_n,
                    strength=strength,
                    direction=direction,
                )
            )

    # Sort descending by absolute correlation
    top_pairs.sort(key=lambda x: x.abs_correlation, reverse=True)

    return CorrelationAnalysisResult(
        method=method.capitalize(),
        matrix=corr_matrix,
        top_pairs=top_pairs,
        excluded_columns=excluded,
        warnings=warnings,
    )


def compute_frequency_distribution(
    series: pd.Series,
    max_categories: int = 20,
) -> tuple[pd.DataFrame, bool, int]:
    """Compute frequency and percentage distribution for a categorical column.

    Returns:
        tuple of (distribution_df, is_truncated, other_count)
    """
    clean = series.dropna().astype(str)
    total_n = len(clean)

    if total_n == 0:
        return pd.DataFrame(), False, 0

    counts = clean.value_counts()
    distinct_count = len(counts)

    is_truncated = False
    other_count = 0

    if distinct_count > max_categories:
        is_truncated = True
        top_counts = counts.iloc[: max_categories - 1]
        other_count = int(counts.iloc[max_categories - 1 :].sum())
        # Append 'Other (Aggregated)'
        series_display = pd.concat([top_counts, pd.Series({"Other (Aggregated)": other_count})])
    else:
        series_display = counts

    dist_df = pd.DataFrame({
        "Category": series_display.index,
        "Count": series_display.values,
        "Percentage": (series_display.values / total_n * 100.0).round(2),
    })

    dist_df["Cumulative %"] = dist_df["Percentage"].cumsum().round(2)
    return dist_df, is_truncated, other_count


def compute_group_summary(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    max_groups: int = 15,
) -> pd.DataFrame:
    """Compute aggregated summary statistics for a numeric column grouped by a category."""
    if group_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()

    sub = df[[group_col, metric_col]].dropna()
    if sub.empty:
        return pd.DataFrame()

    try:
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
        sub = sub.dropna()
    except (ValueError, TypeError, KeyError):
        return pd.DataFrame()

    # Limit to top N groups by frequency
    top_groups = sub[group_col].value_counts().head(max_groups).index
    sub_filtered = sub[sub[group_col].isin(top_groups)]

    grouped = sub_filtered.groupby(group_col)[metric_col].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Median="median",
        Min="min",
        Max="max",
    ).reset_index()

    grouped["Mean"] = grouped["Mean"].round(2)
    grouped["Std"] = grouped["Std"].round(2)
    grouped["Median"] = grouped["Median"].round(2)
    grouped["Min"] = grouped["Min"].round(2)
    grouped["Max"] = grouped["Max"].round(2)

    return grouped.sort_values(by="Count", ascending=False)
