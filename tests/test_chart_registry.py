"""Tests for centralized chart registry and compatibility evaluation system."""

from __future__ import annotations

import pandas as pd
import pytest

from core.chart_registry import (
    CHART_REGISTRY,
    evaluate_chart_compatibility,
    get_available_chart_families,
    get_charts_by_family,
)
from core.profiling import compute_dataset_profile


@pytest.fixture
def synthetic_chart_df() -> pd.DataFrame:
    """Create multi-type dataset for chart compatibility testing."""
    return pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num2": [10.0, 20.0, 15.0, 30.0, 25.0, 40.0, 35.0, 50.0, 45.0, 60.0],
        "num3": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0],
        "cat1": ["A", "B", "A", "B", "A", "C", "C", "A", "B", "C"],
        "cat2": ["X", "Y", "X", "Y", "Z", "X", "Y", "Z", "X", "Y"],
        "date_col": pd.date_range("2025-01-01", periods=10, freq="D"),
        "constant_num": [5.0] * 10,
    })


def test_registry_contains_all_seven_families():
    """Verify that all 7 required chart families are represented."""
    families = get_available_chart_families()
    assert len(families) == 7
    expected = {
        "Distributions",
        "Relationships",
        "Categoricals",
        "Time-Series",
        "Comparisons",
        "Multivariate",
        "3D Visualizations",
    }
    assert set(families) == expected
    # Ensure get_charts_by_family works for each family
    for fam in families:
        charts = get_charts_by_family(fam)
        assert len(charts) > 0


def test_registry_definitions_have_valid_structure():
    """Ensure every registered chart specifies required metadata and parameters."""
    for chart_id, chart_def in CHART_REGISTRY.items():
        assert chart_def.chart_id == chart_id
        assert chart_def.display_name != ""
        assert chart_def.purpose != ""
        assert chart_def.family in get_available_chart_families()
        assert len(chart_def.required_column_types) >= 1
        assert callable(chart_def.compatibility_check)


def test_distribution_charts_compatibility(synthetic_chart_df):
    """Test compatibility checks for distribution charts."""
    profile = compute_dataset_profile(synthetic_chart_df)

    # Histogram should be compatible with non-constant numeric
    is_compat, msg = evaluate_chart_compatibility("histogram_box", synthetic_chart_df, profile)
    assert is_compat is True
    assert msg == ""

    # Boxplot should be compatible
    is_compat, _ = evaluate_chart_compatibility("boxplot", synthetic_chart_df, profile)
    assert is_compat is True


def test_relationship_charts_compatibility(synthetic_chart_df):
    """Test compatibility checks for relationship and multivariate charts."""
    profile = compute_dataset_profile(synthetic_chart_df)

    # Scatter requires 2 non-constant numeric features
    is_compat, _ = evaluate_chart_compatibility("scatter_regression", synthetic_chart_df, profile)
    assert is_compat is True

    # Bubble chart requires 3 numeric features
    is_compat, _ = evaluate_chart_compatibility("bubble_chart", synthetic_chart_df, profile)
    assert is_compat is True

    # Incompatible test: DataFrame with only 1 numeric feature
    single_num_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["X", "Y", "Z"]})
    single_prof = compute_dataset_profile(single_num_df)
    is_compat, msg = evaluate_chart_compatibility("bubble_chart", single_num_df, single_prof)
    assert is_compat is False
    assert "Requires at least 3" in msg


def test_time_series_compatibility(synthetic_chart_df):
    """Test datetime compatibility guard for temporal charts."""
    profile = compute_dataset_profile(synthetic_chart_df)

    # Time series line with valid datetime column
    is_compat, _ = evaluate_chart_compatibility("time_series_line", synthetic_chart_df, profile)
    assert is_compat is True

    # Incompatible test: Dataset without any datetime column
    no_dt_df = pd.DataFrame({"val1": [1, 2, 3], "val2": [4, 5, 6]})
    no_dt_prof = compute_dataset_profile(no_dt_df)
    is_compat, msg = evaluate_chart_compatibility("time_series_line", no_dt_df, no_dt_prof)
    assert is_compat is False
    assert "Requires at least 1 Datetime" in msg


def test_3d_visualizations_compatibility(synthetic_chart_df):
    """Test 3D chart compatibility requirements."""
    profile = compute_dataset_profile(synthetic_chart_df)

    # 3D scatter requires 3 numeric dimensions
    is_compat, _ = evaluate_chart_compatibility("scatter_3d", synthetic_chart_df, profile)
    assert is_compat is True


def test_unknown_chart_compatibility(synthetic_chart_df):
    """Verify graceful handling for non-existent chart IDs."""
    profile = compute_dataset_profile(synthetic_chart_df)
    is_compat, msg = evaluate_chart_compatibility("non_existent_chart_xyz", synthetic_chart_df, profile)
    assert is_compat is False
    assert "Unknown chart identifier" in msg
