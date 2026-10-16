"""Tests for Reusable Statistical Analysis Engine."""

from __future__ import annotations

import pandas as pd
import pytest

from core.profiling import compute_dataset_profile
from core.statistics import (
    compute_correlations,
    compute_descriptive_stats_table,
    compute_frequency_distribution,
    compute_group_summary,
)


def test_descriptive_statistics_table(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify descriptive statistics output table contains proper columns and values."""
    profile = compute_dataset_profile(mixed_dataset_df)
    desc_df = compute_descriptive_stats_table(mixed_dataset_df, profile)

    assert not desc_df.empty
    assert "Column" in desc_df.columns
    assert "Mean" in desc_df.columns
    assert "Std Dev" in desc_df.columns
    assert "IQR" in desc_df.columns
    assert "Skewness" in desc_df.columns

    age_row = desc_df[desc_df["Column"] == "age"].iloc[0]
    assert age_row["Count"] == 20
    assert age_row["Min"] == 22.0
    assert age_row["Max"] == 62.0


def test_pearson_and_spearman_correlations(clean_numeric_df: pd.DataFrame) -> None:
    """Verify correlation values for perfectly inverse variables (var_a vs var_b)."""
    profile = compute_dataset_profile(clean_numeric_df)

    res_pearson = compute_correlations(clean_numeric_df, profile, method="pearson")
    assert not res_pearson.matrix.empty
    r_val = res_pearson.matrix.loc["var_a", "var_b"]
    assert pytest.approx(r_val, abs=1e-3) == -1.0

    res_spearman = compute_correlations(clean_numeric_df, profile, method="spearman")
    assert not res_spearman.matrix.empty
    rho_val = res_spearman.matrix.loc["var_a", "var_b"]
    assert pytest.approx(rho_val, abs=1e-3) == -1.0

    # Top pairs
    assert len(res_spearman.top_pairs) > 0
    top_pair = res_spearman.top_pairs[0]
    assert {top_pair.column_a, top_pair.column_b} == {"var_a", "var_b"}
    assert top_pair.strength == "Very Strong"
    assert top_pair.direction == "Negative"


def test_correlation_excludes_constant_columns(edge_case_df: pd.DataFrame) -> None:
    """Verify that zero-variance columns are excluded gracefully from correlation calculations."""
    profile = compute_dataset_profile(edge_case_df)
    res = compute_correlations(edge_case_df, profile, method="pearson")

    # constant_col and all_null_col should be in excluded_columns
    excluded_str = " ".join(res.excluded_columns)
    assert "constant_col" in excluded_str
    assert "all_null_col" in excluded_str
    assert "constant_col" not in res.matrix.columns


def test_frequency_distribution_aggregation() -> None:
    """Verify that frequency distributions with many categories aggregate into 'Other'."""
    # 30 distinct categories
    cats = [f"Category_{i}" for i in range(30)] * 3
    s = pd.Series(cats)

    dist_df, is_truncated, other_cnt = compute_frequency_distribution(s, max_categories=10)
    assert is_truncated is True
    assert "Other (Aggregated)" in dist_df["Category"].values
    assert other_cnt > 0
    assert len(dist_df) == 10


def test_group_summary_statistics(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify calculation of group-wise aggregations (salary by department)."""
    grp_df = compute_group_summary(mixed_dataset_df, group_col="department", metric_col="salary")
    assert not grp_df.empty
    assert "department" in grp_df.columns
    assert "Mean" in grp_df.columns
    assert "Median" in grp_df.columns
    assert "Count" in grp_df.columns
    assert "Engineering" in grp_df["department"].values
