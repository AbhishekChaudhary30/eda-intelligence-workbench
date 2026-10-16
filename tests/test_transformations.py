"""Tests for controlled data transformations, previewing, undo, and immutability."""

from __future__ import annotations

import pandas as pd
import pytest

from core.transformations import (
    TransformationError,
    TransformationManager,
    apply_single_transformation,
    preview_transformation,
)


@pytest.fixture
def transform_df() -> pd.DataFrame:
    """Create test dataset for transformation operations."""
    return pd.DataFrame({
        "num": [10.0, 20.0, 30.0, None, 50.0],
        "cat": ["Apple", "Banana", "Apple", "Orange", "Apple"],
        "str_num": ["1", "2", "3", "4", "5"],
        "to_drop": [1, 2, 3, 4, 5],
    })


def test_drop_columns_and_rename(transform_df):
    """Test column dropping and column renaming."""
    # Drop
    df_dropped, desc = apply_single_transformation(transform_df, "drop_columns", {"columns": ["to_drop"]})
    assert "to_drop" not in df_dropped.columns
    assert "Dropped" in desc

    # Rename
    df_renamed, desc = apply_single_transformation(transform_df, "rename_column", {"old_name": "cat", "new_name": "fruit"})
    assert "fruit" in df_renamed.columns
    assert "cat" not in df_renamed.columns


def test_safe_row_filter(transform_df):
    """Test safe row filtering without eval or exec."""
    # Greater than operator
    df_filtered, _ = apply_single_transformation(transform_df, "filter_rows", {"column": "num", "operator": ">", "value": 25.0})
    assert len(df_filtered) == 2  # 30.0 and 50.0

    # Is null operator
    df_null, _ = apply_single_transformation(transform_df, "filter_rows", {"column": "num", "operator": "is_null"})
    assert len(df_null) == 1

    # Contains operator on text
    df_contains, _ = apply_single_transformation(transform_df, "filter_rows", {"column": "cat", "operator": "contains", "value": "app"})
    assert len(df_contains) == 3


def test_imputation(transform_df):
    """Test mean, median, mode, and constant imputation."""
    # Mean imputation
    df_imp, _ = apply_single_transformation(transform_df, "impute_missing", {"column": "num", "strategy": "mean"})
    assert df_imp["num"].isnull().sum() == 0
    # Mean of [10, 20, 30, 50] is 27.5
    assert df_imp.loc[3, "num"] == 27.5

    # Mode imputation on categorical
    df_mode, _ = apply_single_transformation(transform_df, "impute_missing", {"column": "cat", "strategy": "mode"})
    assert df_mode["cat"].isnull().sum() == 0


def test_scaling_and_log_transforms():
    """Test numeric scaling methods and log(1 + x) guardrails."""
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})

    # Min-Max scaling
    df_minmax, _ = apply_single_transformation(df, "scale_numeric", {"column": "val", "method": "minmax"})
    assert df_minmax["val"].min() == 0.0
    assert df_minmax["val"].max() == 1.0

    # Log transformation
    df_log, _ = apply_single_transformation(df, "log_transform", {"column": "val"})
    assert df_log["val"].iloc[0] > 0

    # Rejection of negative numbers for log
    df_neg = pd.DataFrame({"val": [-5.0, 10.0, 20.0]})
    with pytest.raises(TransformationError, match="negative numbers"):
        apply_single_transformation(df_neg, "log_transform", {"column": "val"})


def test_preview_transformation(transform_df):
    """Verify that transformation preview does not mutate the input dataset."""
    orig_rows = len(transform_df)
    prev = preview_transformation(transform_df, "filter_rows", {"column": "num", "operator": ">", "value": 20.0})
    assert prev.expected_rows_after == 2
    # Ensure original was not mutated
    assert len(transform_df) == orig_rows


def test_transformation_manager_lifecycle(transform_df):
    """Verify TransformationManager preserves original dataset and supports undo and reset."""
    manager = TransformationManager(transform_df)

    # Apply 1: drop column
    manager.apply("drop_columns", {"columns": ["to_drop"]})
    assert len(manager.history) == 1
    assert "to_drop" not in manager.working_df.columns
    assert "to_drop" in manager.original_df.columns  # Original strictly preserved!

    # Apply 2: filter rows
    manager.apply("filter_rows", {"column": "num", "operator": ">=", "value": 20.0})
    assert len(manager.history) == 2
    assert len(manager.working_df) == 3

    # Undo last step (filter rows)
    success = manager.undo()
    assert success is True
    assert len(manager.history) == 1
    assert len(manager.working_df) == 5  # Restored to 5 rows
    assert "to_drop" not in manager.working_df.columns  # Step 1 still in effect

    # Reset
    manager.reset()
    assert len(manager.history) == 0
    assert len(manager.working_df) == 5
    assert "to_drop" in manager.working_df.columns  # Completely restored
