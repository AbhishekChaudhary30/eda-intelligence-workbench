"""Tests for missing-data investigation, nullity correlation, and token audits."""

from __future__ import annotations

import pandas as pd
import pytest

from core.missingness import (
    analyze_missingness,
    compute_group_missingness,
)


@pytest.fixture
def missingness_df() -> pd.DataFrame:
    """Create dataset with known missingness patterns and pseudo-tokens."""
    return pd.DataFrame({
        "clean_col": [1, 2, 3, 4, 5, 6],
        "col_a": [10.0, None, 30.0, None, 50.0, None],  # 50% missing
        "col_b": [100.0, None, 300.0, None, 500.0, None],  # Perfect co-missing with col_a (phi=1.0)
        "group_col": ["TeamA", "TeamA", "TeamA", "TeamB", "TeamB", "TeamB"],
        "text_with_tokens": ["Valid", "N/A", "Unknown", "Valid", "none", "?"],
    })


def test_nullity_correlation_matrix(missingness_df):
    """Verify nullity correlation identifies perfect co-occurrence of missingness."""
    report = analyze_missingness(missingness_df)
    null_corr = report.nullity_correlation_matrix
    assert not null_corr.empty
    assert "col_a" in null_corr.columns
    assert "col_b" in null_corr.columns
    # col_a and col_b always go missing together, so nullity correlation should be +1.0
    assert round(null_corr.loc["col_a", "col_b"], 2) == 1.0


def test_row_missingness_distribution(missingness_df):
    """Verify frequency breakdown of missing features per row."""
    report = analyze_missingness(missingness_df)
    row_dist = report.row_missing_distribution
    assert not row_dist.empty
    assert "Missing Columns Per Row" in row_dist.columns
    assert "Number of Rows" in row_dist.columns
    # Rows with index 1, 3, 5 have 2 missing fields (col_a, col_b)
    two_missing_row = row_dist[row_dist["Missing Columns Per Row"] == 2]
    assert not two_missing_row.empty
    assert two_missing_row.iloc[0]["Number of Rows"] == 3


def test_group_missingness(missingness_df):
    """Verify missingness rate calculation segmented across categorical groups."""
    grp_miss = compute_group_missingness(missingness_df, target_col="col_a", group_col="group_col")
    assert not grp_miss.empty
    assert "Missing_Pct" in grp_miss.columns
    assert "Missing_Count" in grp_miss.columns
    # Both TeamA and TeamB have missing values in col_a
    assert grp_miss["Missing_Count"].sum() == 3


def test_detect_pseudo_missing_tokens(missingness_df):
    """Verify detection of disguised missing text tokens like 'N/A', '?', 'none'."""
    report = analyze_missingness(
        missingness_df,
        custom_tokens=["N/A", "Unknown", "none", "?"],
    )
    detected = report.custom_token_occurrences
    assert len(detected) >= 3
    found_tokens = {item.token for item in detected}
    assert "N/A" in found_tokens
    assert "?" in found_tokens
