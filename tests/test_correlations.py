"""Tests for advanced correlation matrix, p-values, FDR adjustment, and ranked extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from core.correlations import (
    adjust_p_values_fdr,
    compute_advanced_correlations,
)
from core.profiling import compute_dataset_profile


@pytest.fixture
def correlation_sample_df() -> pd.DataFrame:
    """Create synthetic dataset with known linear, monotonic, and constant relationships."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return pd.DataFrame({
        "x": x,
        "y_linear": [2.0 * v + 1.0 for v in x],  # Perfect Pearson (r=1.0)
        "y_monotonic": [v**3 for v in x],  # Perfect Spearman (r=1.0)
        "constant": [42.0] * 10,  # Zero variance
        "y_with_nulls": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0],
    })


def test_pearson_perfect_correlation(correlation_sample_df):
    """Verify Pearson correlation detects exact linear dependency with r=1.0."""
    profile = compute_dataset_profile(correlation_sample_df)
    res = compute_advanced_correlations(correlation_sample_df, profile, method="pearson")

    assert "x" in res.corr_matrix.columns
    assert "y_linear" in res.corr_matrix.columns
    r_val = res.corr_matrix.loc["x", "y_linear"]
    assert round(r_val, 4) == 1.0

    p_val = res.p_value_matrix.loc["x", "y_linear"]
    assert p_val < 1e-5


def test_spearman_monotonic_correlation(correlation_sample_df):
    """Verify Spearman correlation detects rank monotonicity where relationship is non-linear."""
    profile = compute_dataset_profile(correlation_sample_df)

    res_spearman = compute_advanced_correlations(correlation_sample_df, profile, method="spearman")
    r_spearman = res_spearman.corr_matrix.loc["x", "y_monotonic"]
    assert round(r_spearman, 4) == 1.0

    # Kendall should also be 1.0 for monotonic concordance
    res_kendall = compute_advanced_correlations(correlation_sample_df, profile, method="kendall")
    r_kendall = res_kendall.corr_matrix.loc["x", "y_monotonic"]
    assert round(r_kendall, 4) == 1.0


def test_constant_column_graceful_handling(correlation_sample_df):
    """Ensure constant columns with zero variance are safely excluded without error."""
    profile = compute_dataset_profile(correlation_sample_df)
    res = compute_advanced_correlations(correlation_sample_df, profile, method="pearson")

    # 'constant' should be excluded from correlation matrix
    assert "constant" not in res.corr_matrix.columns
    assert any("constant" in excl for excl in res.excluded_columns)


def test_pairwise_sample_size_tracking(correlation_sample_df):
    """Ensure pairwise sample size matrix accurately reflects non-null overlaps."""
    profile = compute_dataset_profile(correlation_sample_df)
    res = compute_advanced_correlations(correlation_sample_df, profile, method="pearson")

    sample_size = res.sample_size_matrix.loc["x", "y_with_nulls"]
    assert sample_size == 8  # 10 - 2 nulls


def test_benjamini_hochberg_fdr_adjustment():
    """Verify False Discovery Rate (FDR) adjustment controls multiple testing inflation."""
    raw_p = [0.001, 0.01, 0.04, 0.08, 0.15]
    adj_p = adjust_p_values_fdr(raw_p)

    assert len(adj_p) == len(raw_p)
    # Adjusted p-values must be greater than or equal to raw p-values
    for r, a in zip(raw_p, adj_p, strict=True):
        assert a >= r
        assert a <= 1.0


def test_extract_ranked_correlations(correlation_sample_df):
    """Verify extraction and sorting of top pairwise correlations."""
    profile = compute_dataset_profile(correlation_sample_df)
    res = compute_advanced_correlations(
        correlation_sample_df,
        profile,
        method="spearman",
        min_threshold=0.5,
    )
    assert len(res.ranked_pairs) >= 1
    top_pair = res.ranked_pairs[0]
    assert top_pair.column_a != ""
    assert top_pair.column_b != ""
    assert top_pair.abs_correlation >= 0.5
    assert top_pair.strength in ("Very Strong", "Strong", "Moderate", "Weak")
