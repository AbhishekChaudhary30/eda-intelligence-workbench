"""Tests for statistical outlier analysis (IQR, Z-Score, and MAD)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.outlier_analysis import (
    audit_column_outliers_iqr,
    audit_column_outliers_mad,
    audit_column_outliers_zscore,
    run_comprehensive_outlier_analysis,
)


@pytest.fixture
def outlier_sample_series() -> pd.Series:
    """Create numeric series with a distinct extreme outlier at the end."""
    # 20 regular values around 50, and one extreme point at 1000
    vals = [50.0 + i for i in range(20)] + [1000.0]
    return pd.Series(vals, name="val_col")


def test_iqr_outlier_detection(outlier_sample_series):
    """Verify Tukey IQR detection identifies the extreme point."""
    res = audit_column_outliers_iqr(outlier_sample_series, "val_col", multiplier=1.5)
    assert res is not None
    assert res.outlier_count == 1
    assert 20 in res.flagged_indices
    assert res.upper_bound < 1000.0


def test_zscore_outlier_detection(outlier_sample_series):
    """Verify standard Z-Score detection flags the extreme point."""
    res = audit_column_outliers_zscore(outlier_sample_series, "val_col", threshold=3.0)
    assert res is not None
    assert res.outlier_count == 1
    assert 20 in res.flagged_indices


def test_mad_outlier_detection(outlier_sample_series):
    """Verify Median Absolute Deviation (MAD) modified Z-score flags the extreme point."""
    res = audit_column_outliers_mad(outlier_sample_series, "val_col", threshold=3.5)
    assert res is not None
    assert res.outlier_count == 1
    assert 20 in res.flagged_indices


def test_mad_zero_variance_handling():
    """Verify MAD handles degenerate zero-MAD series gracefully."""
    # When > 50% values are identical, MAD is 0.0, function safely returns None
    zero_mad_series = pd.Series([5.0] * 20 + [10.0], name="zero_mad")
    res = audit_column_outliers_mad(zero_mad_series, "zero_mad", threshold=3.5)
    assert res is None  # Handled safely without ZeroDivisionError


def test_comprehensive_outlier_analysis():
    """Verify batch outlier analysis generates complete summary table."""
    df = pd.DataFrame({
        "norm_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "outlier_col": [10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 10.0, 11.0, 500.0],
        "constant_col": [1.0] * 12,
    })
    report = run_comprehensive_outlier_analysis(df, method="iqr", threshold=1.5)
    assert report.total_numeric_cols_evaluated == 3
    assert not report.summary_table.empty
    assert "Outlier Count" in report.summary_table.columns
    assert "Outlier %" in report.summary_table.columns

    out_row = report.summary_table[report.summary_table["Feature"] == "outlier_col"].iloc[0]
    assert out_row["Outlier Count"] == 1
