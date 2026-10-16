"""Tests for the Data Quality Analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data_quality import (
    analyze_data_quality,
    detect_iqr_outliers,
    detect_zscore_outliers,
)
from core.profiling import compute_dataset_profile


def test_clean_dataset_quality_score(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify that a clean dataset receives a high health index (>= 85)."""
    profile = compute_dataset_profile(mixed_dataset_df)
    report = analyze_data_quality(mixed_dataset_df, profile)

    assert report.overall_health_score >= 85
    assert report.critical_issues_count == 0
    assert report.health_status in ("Excellent", "Good")


def test_edge_case_quality_issues(edge_case_df: pd.DataFrame) -> None:
    """Verify detection of constant, all-null, duplicate rows, whitespace, and outliers."""
    profile = compute_dataset_profile(edge_case_df)
    report = analyze_data_quality(edge_case_df, profile)

    issue_names = {i.name for i in report.issues}

    assert "All-Null Column" in issue_names
    assert "Constant Column" in issue_names
    assert "Blank / Whitespace Strings" in issue_names
    assert "Duplicate Rows Detected" in issue_names
    assert "Potential Outliers (IQR Method)" in issue_names

    # All-Null should be Critical severity
    all_null_issue = next(i for i in report.issues if i.name == "All-Null Column")
    assert all_null_issue.severity == "Critical"
    assert "all_null_col" in all_null_issue.affected_columns


def test_iqr_outlier_detection() -> None:
    """Verify IQR Tukey's fence outlier detection logic."""
    # 20 normal values around 10, plus two extreme outliers: 100, -50
    vals = [10.0, 10.2, 9.8, 10.5, 10.1, 9.9, 10.3, 10.0, 9.7, 10.4,
            10.1, 10.2, 9.9, 10.0, 10.3, 9.8, 10.2, 10.1, 100.0, -50.0]
    s = pd.Series(vals)
    res = detect_iqr_outliers(s, "test_outliers")

    assert res is not None
    assert res.outlier_count == 2
    assert res.min_outlier == -50.0
    assert res.max_outlier == 100.0
    assert "Tukey" in res.method


def test_zscore_outlier_detection() -> None:
    """Verify Z-score |z| > 3.0 outlier detection logic."""
    np.random.seed(42)
    # 100 standard normal points + one massive outlier at 10.0
    vals = list(np.random.normal(0, 1, 100))
    vals.append(25.0)  # Massive outlier
    s = pd.Series(vals)

    res = detect_zscore_outliers(s, "z_test", threshold=3.0)
    assert res is not None
    assert res.outlier_count >= 1
    assert res.max_outlier == 25.0


def test_infinite_values_detection() -> None:
    """Verify that infinite (+Inf, -Inf) values in numeric columns are caught."""
    df = pd.DataFrame({"ratio": [1.0, 2.0, np.inf, 4.0, -np.inf, 6.0, 7.0, 8.0, 9.0, 10.0]})
    profile = compute_dataset_profile(df)
    report = analyze_data_quality(df, profile)

    inf_issue = next((i for i in report.issues if i.name == "Non-Finite Numeric Values"), None)
    assert inf_issue is not None
    assert inf_issue.affected_count == 2
    assert inf_issue.severity == "Warning"
