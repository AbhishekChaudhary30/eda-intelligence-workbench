"""Tests for advanced descriptive and inferential statistics engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.advanced_statistics import (
    compute_confidence_intervals,
    compute_extended_descriptive_stats,
    run_chi_square_test,
    run_group_comparison_test,
    run_normality_test,
)
from core.profiling import compute_dataset_profile


@pytest.fixture
def normal_sample() -> pd.Series:
    """Generate deterministic normally distributed sample (N=100)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=50.0, scale=10.0, size=100), name="normal_var")


@pytest.fixture
def skewed_sample() -> pd.Series:
    """Generate heavily skewed exponential sample (N=100)."""
    np.random.seed(42)
    return pd.Series(np.random.exponential(scale=5.0, size=100), name="skewed_var")


def test_extended_descriptive_stats_table():
    """Verify calculation of extended statistical moments, percentiles, and CV."""
    df = pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    profile = compute_dataset_profile(df)
    stats_df = compute_extended_descriptive_stats(df, profile)

    assert not stats_df.empty
    row = stats_df[stats_df["Feature"] == "score"].iloc[0]
    assert row["Mean"] == 30.0
    assert row["Median"] == 30.0
    assert row["Min"] == 10.0
    assert row["Max"] == 50.0
    assert row["Range"] == 40.0
    assert row["IQR"] == 20.0
    assert row["Std Dev"] > 0
    assert row["Variance"] > 0


def test_shapiro_wilk_normality(normal_sample, skewed_sample):
    """Test Shapiro-Wilk normality test on normal vs skewed distributions."""
    # Normal sample should not reject normality
    res_norm = run_normality_test(normal_sample, "normal_var", test_type="shapiro")
    assert res_norm.is_normal_at_alpha is True
    assert res_norm.p_value > 0.05
    assert res_norm.sample_size == 100

    # Exponential sample should reject normality
    res_skew = run_normality_test(skewed_sample, "skewed_var", test_type="shapiro")
    assert res_skew.is_normal_at_alpha is False
    assert res_skew.p_value < 0.05

    # Small sample (< 3) should report insufficient non-missing data gracefully
    res_small = run_normality_test(pd.Series([1.0, 2.0]), "small_var", test_type="shapiro")
    assert res_small.is_normal_at_alpha is False
    assert "Insufficient" in res_small.interpretation


def test_dagostino_pearson_normality(normal_sample):
    """Test D'Agostino-Pearson omnibus normality test."""
    res = run_normality_test(normal_sample, "normal_var", test_type="dagostino")
    assert res.is_normal_at_alpha is True
    assert res.p_value > 0.05

    # Under 20 observations should return informative limitation
    res_under20 = run_normality_test(normal_sample.iloc[:15], "short_var", test_type="dagostino")
    assert res_under20.is_normal_at_alpha is False
    assert "Requires at least 20" in res_under20.interpretation


def test_confidence_intervals(normal_sample):
    """Test Student's t and Bootstrap confidence intervals for the mean and median."""
    ci_results = compute_confidence_intervals(
        normal_sample, "normal_var", confidence_level=0.95, bootstrap_resamples=500, seed=42
    )
    assert len(ci_results) == 2

    t_res = ci_results[0]
    assert "Student's t" in t_res.statistic_name
    assert t_res.lower_bound < t_res.point_estimate < t_res.upper_bound
    assert round(t_res.point_estimate, 1) == 49.0

    boot_res = ci_results[1]
    assert "Bootstrap" in boot_res.statistic_name
    assert boot_res.lower_bound < boot_res.point_estimate < boot_res.upper_bound
    # Check that bootstrap and parametric estimates are reasonably close
    assert abs(boot_res.point_estimate - t_res.point_estimate) < 2.0


def test_welchs_ttest_and_cohens_d():
    """Test two-sample location difference and standardized effect size."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "score": np.concatenate([
            np.random.normal(loc=100.0, scale=15.0, size=n),
            np.random.normal(loc=120.0, scale=20.0, size=n),
        ]),
        "group": ["A"] * n + ["B"] * n,
    })

    res = run_group_comparison_test(df, metric_col="score", group_col="group", test_type="welch_t")
    assert res.is_statistically_significant is True
    assert res.p_value < 0.001
    assert res.effect_size_name == "Cohen's d"
    assert res.effect_size_value is not None
    assert res.effect_size_value < -0.8  # Large negative effect size


def test_mann_whitney_u_evaluation():
    """Test non-parametric Mann-Whitney rank sum comparison."""
    df = pd.DataFrame({
        "val": [1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70],
        "cat": ["Low"] * 7 + ["High"] * 7,
    })

    res = run_group_comparison_test(df, metric_col="val", group_col="cat", test_type="mann_whitney")
    assert res.is_statistically_significant is True
    assert res.p_value < 0.01


def test_one_way_anova_and_kruskal_wallis():
    """Test multi-group parametric ANOVA and Kruskal-Wallis H-test."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "metric": np.concatenate([
            np.random.normal(loc=10.0, scale=2.0, size=n),
            np.random.normal(loc=20.0, scale=2.0, size=n),
            np.random.normal(loc=30.0, scale=2.0, size=n),
        ]),
        "segment": ["S1"] * n + ["S2"] * n + ["S3"] * n,
    })

    anova_res = run_group_comparison_test(df, metric_col="metric", group_col="segment", test_type="anova")
    assert anova_res.is_statistically_significant is True
    assert anova_res.p_value < 0.001

    kw_res = run_group_comparison_test(df, metric_col="metric", group_col="segment", test_type="kruskal")
    assert kw_res.is_statistically_significant is True
    assert kw_res.p_value < 0.001


def test_chi_square_independence_evaluation():
    """Test Chi-Square test of independence on categorical columns."""
    df = pd.DataFrame({
        "treatment": ["Drug"] * 60 + ["Placebo"] * 60,
        "outcome": (["Cured"] * 50 + ["Sick"] * 10) + (["Cured"] * 15 + ["Sick"] * 45),
    })

    contingency, chi2_stat, p_val, dof, caution = run_chi_square_test(df, "treatment", "outcome")
    assert contingency.shape == (2, 2)
    assert chi2_stat > 30.0
    assert p_val < 0.001
    assert dof == 1
    assert caution == ""  # All cells > 5
