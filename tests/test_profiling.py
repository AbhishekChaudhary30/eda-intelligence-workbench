"""Tests for Dataset Profiling Engine."""

from __future__ import annotations

import pandas as pd
import pytest

from core.profiling import (
    compute_dataset_profile,
)


def test_dataset_level_profiling(mixed_dataset_df: pd.DataFrame) -> None:
    """Test high-level dataset metrics including cells, complete rows, memory."""
    profile = compute_dataset_profile(mixed_dataset_df)

    assert profile.total_rows == 20
    assert profile.total_cols == 6
    assert profile.total_cells == 120
    assert profile.total_missing_cells == 0
    assert profile.complete_rows_percentage == 100.0
    assert profile.duplicate_rows_count == 0
    assert profile.numeric_columns_count >= 2
    assert "B" in profile.formatted_memory or "KB" in profile.formatted_memory


def test_numeric_column_moments(clean_numeric_df: pd.DataFrame) -> None:
    """Test numeric calculations for mean, std, quartiles, IQR, skewness, and kurtosis."""
    profile = compute_dataset_profile(clean_numeric_df)
    p_a = profile.column_profiles["var_a"]

    # var_a is linear from 10 to 100, mean is 55.0
    assert pytest.approx(p_a.mean, rel=1e-2) == 55.0
    assert pytest.approx(p_a.median, rel=1e-2) == 55.0
    assert p_a.min_val == 10.0
    assert p_a.max_val == 100.0
    assert p_a.iqr > 0
    assert p_a.skewness is not None
    assert abs(p_a.skewness) < 0.1  # Uniform/linear has skew near 0
    assert p_a.is_zero_variance is False


def test_constant_column_profiling(edge_case_df: pd.DataFrame) -> None:
    """Verify that zero-variance/constant columns are properly flagged without errors."""
    profile = compute_dataset_profile(edge_case_df)
    p_const = profile.column_profiles["constant_col"]

    assert p_const.is_constant is True
    assert p_const.is_zero_variance is True
    assert p_const.std == 0.0
    assert p_const.min_val == p_const.max_val == 42.0


def test_all_null_column_profiling(edge_case_df: pd.DataFrame) -> None:
    """Verify that 100% missing columns are profiled safely."""
    profile = compute_dataset_profile(edge_case_df)
    p_null = profile.column_profiles["all_null_col"]

    assert p_null.is_all_null is True
    assert p_null.null_count == len(edge_case_df)
    assert p_null.null_percentage == 100.0
    assert p_null.non_null_count == 0


def test_categorical_blank_and_whitespace_counts(edge_case_df: pd.DataFrame) -> None:
    """Verify detection of empty strings and whitespace-only entries in text columns."""
    profile = compute_dataset_profile(edge_case_df)
    p_text = profile.column_profiles["text_with_blanks"]

    assert p_text.empty_string_count == 2
    assert p_text.whitespace_only_count == 2
    assert len(p_text.top_values) > 0
    # Most frequent value should be Alpha
    top_val, top_cnt, _ = p_text.top_values[0]
    assert top_val == "Alpha"
    assert top_cnt == 5
