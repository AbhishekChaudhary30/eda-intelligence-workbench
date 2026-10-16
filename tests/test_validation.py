"""Tests for Data Validation and Semantic Type Inference."""

from __future__ import annotations

import pandas as pd

from core.validation import (
    check_column_python_types,
    validate_dataset,
)


def test_validation_on_mixed_dataset(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify validation detects numeric, categorical, boolean, and datetime columns accurately."""
    report = validate_dataset(mixed_dataset_df)
    assert report.is_valid is True
    assert report.total_rows == 20
    assert report.total_cols == 6

    # ID column
    assert report.inferred_types["id"].is_potential_identifier is True

    # Age and Salary
    assert report.inferred_types["age"].is_numeric is True
    assert report.inferred_types["salary"].is_numeric is True

    # Department
    assert report.inferred_types["department"].is_categorical is True

    # Boolean
    assert report.inferred_types["is_active"].is_boolean is True

    # Datetime
    assert report.inferred_types["join_date"].is_datetime is True


def test_mixed_python_type_detection() -> None:
    """Verify that columns with mixed Python types (e.g. str and int) are detected."""
    s = pd.Series([10, "Twenty", 30, "Forty", None])
    is_mixed, py_types = check_column_python_types(s)
    assert is_mixed is True
    assert "int" in py_types
    assert "str" in py_types


def test_constant_and_all_null_validation_detection(edge_case_df: pd.DataFrame) -> None:
    """Verify detection of degenerate constant and all-null columns."""
    report = validate_dataset(edge_case_df)
    assert "constant_col" in report.constant_columns
    assert "all_null_col" in report.all_null_columns


def test_empty_dataframe_validation() -> None:
    """Verify validation flags empty DataFrame with clear warnings."""
    empty_df = pd.DataFrame()
    report = validate_dataset(empty_df)
    assert report.is_valid is False
    assert len(report.warnings) > 0
