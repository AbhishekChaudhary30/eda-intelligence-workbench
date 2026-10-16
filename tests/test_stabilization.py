"""Regression and stabilization test suite for EDA Intelligence Workbench.

Verifies:
1. AnalysisSettings contract (alpha, default_correlation_method, default_outlier_method,
   boundary validation, backward-compatibility aliases, JSON serialization/deserialization).
2. Sample data loader fixes (employee analytics salary formula, correct types, no TypeError).
3. Plotly chart engine fixes (theme application, title_font vs titlefont, figure validation).
4. Statistical engine contracts (alpha propagation, warnings, ChiSquareTestResult interpretation).
5. Correlation Explorer contracts (FDR adjustment, ranked pairs aliases, method support).
6. Missingness analysis contracts (row distribution columns, edge case datasets).
7. Outlier analysis contracts (zero variance, constant columns, MAD/IQR/Z-score).
8. Consolidated export and executive HTML reporting with configured settings.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from components.chart_components import (
    _apply_workbench_theme,
    plot_histogram,
)
from core.advanced_statistics import (
    compute_confidence_intervals,
    compute_extended_descriptive_stats,
)
from core.advanced_statistics import (
    test_chi_square_independence as run_chi_square,
)
from core.advanced_statistics import (
    test_dagostino_pearson as run_dagostino_pearson,
)
from core.advanced_statistics import (
    test_kruskal_wallis as run_kruskal_wallis,
)
from core.advanced_statistics import (
    test_mann_whitney_u as run_mann_whitney_u,
)
from core.advanced_statistics import (
    test_one_way_anova as run_one_way_anova,
)
from core.advanced_statistics import (
    test_shapiro_wilk as run_shapiro_wilk,
)
from core.advanced_statistics import (
    test_welchs_ttest as run_welchs_ttest,
)
from core.correlations import compute_correlation_matrix, extract_ranked_correlations
from core.data_quality import analyze_data_quality
from core.export import (
    export_correlation_matrix_as_csv,
    export_dataset_as_csv,
    export_descriptive_stats_as_csv,
    export_outlier_summary_as_csv,
    export_profile_summary_as_csv,
    export_quality_issues_as_csv,
    export_ranked_pairs_as_csv,
)
from core.first_look import run_first_look_analysis
from core.insights import generate_deterministic_insights
from core.missingness import (
    compute_row_missingness_distribution,
    detect_pseudo_missing_tokens,
)
from core.outlier_analysis import (
    analyze_all_numeric_outliers,
    detect_iqr_outliers,
    detect_mad_outliers,
    detect_zscore_outliers,
)
from core.profiling import compute_dataset_profile
from core.report_generation import generate_consolidated_html_report
from core.settings import AnalysisSettings, ValidationResult
from sample_data.loader import (
    SAMPLE_DATASETS,
    create_customer_churn_sample,
    create_employee_analytics_sample,
)

# =============================================================================
# 1. ANALYSIS SETTINGS CONTRACT TESTS
# =============================================================================

def test_analysis_settings_default_initialization() -> None:
    """Verify AnalysisSettings default values and validation."""
    settings = AnalysisSettings()
    assert settings.alpha == 0.05
    assert settings.default_correlation_method == "spearman"
    assert settings.default_outlier_method == "iqr"
    assert settings.confidence_level == 0.95
    assert settings.correlation_threshold == 0.30
    assert settings.correlation_fdr_correction is True

    val_res = settings.validate()
    assert isinstance(val_res, ValidationResult)
    assert val_res.is_valid is True
    assert val_res.error_message == ""


def test_analysis_settings_valid_custom() -> None:
    """Verify custom valid settings pass validation."""
    settings = AnalysisSettings(
        alpha=0.01,
        default_correlation_method="pearson",
        default_outlier_method="mad",
        confidence_level=0.99,
        correlation_threshold=0.50,
        iqr_multiplier=2.0,
        zscore_threshold=3.5,
        mad_threshold=4.0,
    )
    val_res = settings.validate()
    assert val_res.is_valid is True
    assert val_res.error_message == ""


@pytest.mark.parametrize("invalid_alpha", [-0.05, 0.0, 1.0, 1.5])
def test_analysis_settings_invalid_alpha(invalid_alpha: float) -> None:
    """Verify non-boundary alpha values are rejected with informative message."""
    settings = AnalysisSettings(alpha=invalid_alpha)
    val_res = settings.validate()
    assert val_res.is_valid is False
    assert "alpha" in val_res.error_message.lower()


@pytest.mark.parametrize("invalid_corr", ["invalid", "spearman_rank", "ridge", ""])
def test_analysis_settings_invalid_correlation_method(invalid_corr: str) -> None:
    """Verify unsupported correlation methods are rejected."""
    settings = AnalysisSettings(default_correlation_method=invalid_corr)
    val_res = settings.validate()
    assert val_res.is_valid is False
    assert "correlation method" in val_res.error_message.lower()


@pytest.mark.parametrize("invalid_outlier", ["isolation_forest", "dbscan", "svm", ""])
def test_analysis_settings_invalid_outlier_method(invalid_outlier: str) -> None:
    """Verify unsupported outlier methods are rejected."""
    settings = AnalysisSettings(default_outlier_method=invalid_outlier)
    val_res = settings.validate()
    assert val_res.is_valid is False
    assert "outlier method" in val_res.error_message.lower()


def test_analysis_settings_property_aliases() -> None:
    """Verify backward-compatible aliases for correlation_method, outlier_method, and alpha."""
    settings = AnalysisSettings()

    # Getters
    assert settings.correlation_method == settings.default_correlation_method
    assert settings.outlier_method == settings.default_outlier_method
    assert settings.normality_test_alpha == settings.alpha
    assert settings.min_correlation_threshold == settings.correlation_threshold
    assert settings.max_categories_display == settings.max_categorical_levels

    # Setters
    settings.correlation_method = "kendall"
    assert settings.default_correlation_method == "kendall"

    settings.outlier_method = "zscore"
    assert settings.default_outlier_method == "zscore"

    settings.normality_test_alpha = 0.01
    assert settings.alpha == 0.01

    settings.min_correlation_threshold = 0.40
    assert settings.correlation_threshold == 0.40

    settings.max_categories_display = 25
    assert settings.max_categorical_levels == 25


def test_analysis_settings_legacy_dict_and_json_import() -> None:
    """Verify AnalysisSettings gracefully deserializes legacy configuration keys."""
    legacy_dict = {
        "correlation_method": "pearson",
        "outlier_method": "zscore",
        "normality_test_alpha": 0.01,
        "min_correlation_threshold": 0.45,
        "max_categories_display": 20,
    }
    settings = AnalysisSettings.from_dict(legacy_dict)
    assert settings.default_correlation_method == "pearson"
    assert settings.default_outlier_method == "zscore"
    assert settings.alpha == 0.01
    assert settings.correlation_threshold == 0.45
    assert settings.max_categorical_levels == 20

    # JSON round-trip
    json_str = settings.to_json()
    reloaded = AnalysisSettings.from_json(json_str)
    assert reloaded.default_correlation_method == "pearson"
    assert reloaded.default_outlier_method == "zscore"
    assert reloaded.alpha == 0.01
    assert reloaded.correlation_threshold == 0.45


def test_analysis_settings_malformed_json_fallback() -> None:
    """Verify corrupted or invalid JSON string falls back safely to default settings."""
    assert AnalysisSettings.from_json("{broken json").alpha == 0.05
    assert AnalysisSettings.from_json("[]").default_correlation_method == "spearman"
    # Invalid alpha falls back to default
    invalid_alpha_json = json.dumps({"alpha": 99.0})
    assert AnalysisSettings.from_json(invalid_alpha_json).alpha == 0.05


# =============================================================================
# 2. SAMPLE DATASET LOADER TESTS
# =============================================================================

def test_employee_analytics_sample_loader_no_type_error() -> None:
    """Verify create_employee_analytics_sample runs without TypeError: int + list."""
    df = create_employee_analytics_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 400
    assert "annual_salary" in df.columns
    assert "department" in df.columns
    assert "experience_years" in df.columns

    # Verify annual_salary is purely numeric with no list-valued cells
    assert pd.api.types.is_numeric_dtype(df["annual_salary"])
    assert not df["annual_salary"].isna().any()
    assert (df["annual_salary"] > 0).all()

    # Verify departments
    expected_depts = {"Engineering", "Product", "Sales", "Marketing", "Human Resources"}
    assert set(df["department"].unique()).issubset(expected_depts)


def test_customer_churn_sample_loader() -> None:
    """Verify create_customer_churn_sample runs and produces expected structure."""
    df = create_customer_churn_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 502  # 500 generated + 2 duplicate rows
    assert "monthly_charges" in df.columns
    assert "churn" in df.columns
    assert pd.api.types.is_numeric_dtype(df["monthly_charges"])


def test_sample_datasets_registry() -> None:
    """Verify SAMPLE_DATASETS registry contains both demo datasets and loaders work."""
    assert "customer_churn" in SAMPLE_DATASETS
    assert "employee_analytics" in SAMPLE_DATASETS
    for spec in SAMPLE_DATASETS.values():
        loader_fn = spec["loader"]
        data = loader_fn()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


# =============================================================================
# 3. PLOTLY THEME & AXIS PROPERTIES TESTS
# =============================================================================

def test_plotly_theme_no_titlefont_error() -> None:
    """Verify _apply_workbench_theme uses title_font rather than deprecated/invalid titlefont."""
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig.update_layout(
        title="Test Figure",
        xaxis_title="X Axis Label",
        yaxis_title="Y Axis Label",
    )

    # Applying the theme must NOT raise ValueError: 'titlefont'
    themed_fig = _apply_workbench_theme(fig)
    assert themed_fig is not None

    # Inspect layout dict to verify valid title_font property
    layout_dict = themed_fig.to_dict()["layout"]
    assert "titlefont" not in layout_dict.get("xaxis", {})
    assert "titlefont" not in layout_dict.get("yaxis", {})


def test_chart_component_figure_creation(clean_numeric_df: pd.DataFrame) -> None:
    """Verify chart generator creates valid dark-themed figure without axis errors."""
    fig = plot_histogram(clean_numeric_df, col="var_a", nbins=20)
    assert isinstance(fig, go.Figure)
    assert fig.layout.paper_bgcolor is not None


# =============================================================================
# 4. STATISTICAL ANALYSIS ENGINE & ALPHA PROPAGATION TESTS
# =============================================================================

def test_statistical_engine_alpha_propagation(clean_numeric_df: pd.DataFrame) -> None:
    """Verify significance testing correctly propagates alpha parameter."""
    s1 = clean_numeric_df["var_a"]
    s2 = clean_numeric_df["var_b"]

    # Test Normality with custom alpha
    res_sw = run_shapiro_wilk(s1, alpha=0.01)
    assert hasattr(res_sw, "warnings")
    assert isinstance(res_sw.warnings, list)

    res_dp = run_dagostino_pearson(s1, alpha=0.01)
    assert hasattr(res_dp, "warnings")
    assert isinstance(res_dp.warnings, list)

    # Test 2-sample Welch's t-test
    res_welch = run_welchs_ttest(s1, s2, alpha=0.01)
    assert hasattr(res_welch, "warnings")
    assert isinstance(res_welch.warnings, list)
    assert res_welch.is_significant == (res_welch.p_value < 0.01)

    # Test Mann-Whitney U
    res_mwu = run_mann_whitney_u(s1, s2, alpha=0.01)
    assert hasattr(res_mwu, "warnings")
    assert isinstance(res_mwu.warnings, list)
    assert res_mwu.is_significant == (res_mwu.p_value < 0.01)

    # Test ANOVA and Kruskal-Wallis
    res_anova = run_one_way_anova([s1, s2], alpha=0.01)
    assert hasattr(res_anova, "warnings")
    assert res_anova.is_significant == (res_anova.p_value < 0.01)

    res_kw = run_kruskal_wallis([s1, s2], alpha=0.01)
    assert hasattr(res_kw, "warnings")
    assert res_kw.is_significant == (res_kw.p_value < 0.01)


def test_chi_square_result_contracts() -> None:
    """Verify ChiSquareTestResult contract has warnings and interpretation attributes."""
    # 2x2 contingency matrix
    matrix = np.array([[50, 10], [15, 45]])
    res = run_chi_square(matrix, alpha=0.05)

    assert hasattr(res, "warnings")
    assert isinstance(res.warnings, list)
    assert hasattr(res, "interpretation")
    assert isinstance(res.interpretation, str)
    assert len(res.interpretation) > 0
    assert res.is_significant is True  # Strong dependence


def test_confidence_intervals_boundary_check(clean_numeric_df: pd.DataFrame) -> None:
    """Verify confidence interval calculation with custom confidence levels."""
    series = clean_numeric_df["var_a"]
    ci_res = compute_confidence_intervals(
        series, "var_a", confidence_level=0.99, bootstrap_resamples=300
    )
    assert len(ci_res) == 2
    t_res = ci_res[0]
    boot_res = ci_res[1]
    assert t_res.lower_bound < series.mean() < t_res.upper_bound
    assert boot_res.lower_bound < series.mean() < boot_res.upper_bound


# =============================================================================
# 5. CORRELATION EXPLORER CONTRACT TESTS
# =============================================================================

def test_correlation_matrix_fdr_and_methods(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify compute_correlation_matrix supports apply_fdr flag and all 3 methods."""
    for method in ["spearman", "pearson", "kendall"]:
        res_fdr = compute_correlation_matrix(mixed_dataset_df, method=method, apply_fdr=True)
        assert res_fdr.matrix is not None
        assert not res_fdr.matrix.empty

        res_no_fdr = compute_correlation_matrix(mixed_dataset_df, method=method, apply_fdr=False)
        assert res_no_fdr.matrix is not None

        # Ranked correlations contract check
        ranked = extract_ranked_correlations(res_fdr, top_n=5)
        for pair in ranked:
            assert "feature_a" in pair
            assert "feature_b" in pair
            assert "correlation" in pair
            assert "p_value" in pair
            assert "adj_p_value" in pair
            assert "significant" in pair


# =============================================================================
# 6. MISSINGNESS ANALYSIS CONTRACT TESTS
# =============================================================================

def test_row_missingness_percentage_column_present(edge_case_df: pd.DataFrame) -> None:
    """Verify compute_row_missingness_distribution provides 'Percentage' column."""
    row_dist = compute_row_missingness_distribution(edge_case_df)
    assert isinstance(row_dist, pd.DataFrame)
    # The Plotly express bar chart in missingness_panel expects text="Percentage"
    assert "Percentage" in row_dist.columns
    assert "Percentage of Dataset" in row_dist.columns
    assert "Missing_Fields_Count" in row_dist.columns
    assert "Row_Count" in row_dist.columns


def test_missingness_pseudo_missing_detection() -> None:
    """Verify detection of pseudo-missing string tokens."""
    df = pd.DataFrame({
        "codes": ["N/A", "valid", "?", "-999", "clean", "null", "None"],
        "vals": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })
    tokens = ["N/A", "?", "-999", "null", "None"]
    detected = detect_pseudo_missing_tokens(df, tokens=tokens)
    assert len(detected) >= 5
    assert all(d["column"] == "codes" for d in detected)


# =============================================================================
# 7. OUTLIER INVESTIGATION CONTRACT TESTS
# =============================================================================

def test_outlier_analysis_with_all_methods(edge_case_df: pd.DataFrame) -> None:
    """Verify analyze_all_numeric_outliers functions across all supported methods."""
    for method in ["iqr", "zscore", "mad"]:
        out_df = analyze_all_numeric_outliers(edge_case_df, method=method)
        assert isinstance(out_df, pd.DataFrame)
        assert not out_df.empty
        assert "Feature" in out_df.columns
        assert "Outlier Count" in out_df.columns
        assert "Outlier %" in out_df.columns


def test_outlier_zero_variance_and_constant_columns() -> None:
    """Verify constant columns (variance=0 or MAD=0) do not trigger division by zero."""
    const_series = pd.Series([100.0] * 50, name="constant_feature")

    # IQR
    res_iqr = detect_iqr_outliers(const_series)
    assert res_iqr.outlier_count == 0

    # Z-Score
    res_z = detect_zscore_outliers(const_series)
    assert res_z.outlier_count == 0

    # MAD
    res_mad = detect_mad_outliers(const_series)
    assert res_mad.outlier_count == 0


# =============================================================================
# 8. EXPORT AND EXECUTIVE REPORTING INTEGRATION TESTS
# =============================================================================

def test_export_and_executive_report_with_settings(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify export pipeline and HTML executive report with AnalysisSettings."""
    settings = AnalysisSettings()
    profile = compute_dataset_profile(mixed_dataset_df)
    quality = analyze_data_quality(mixed_dataset_df, profile)
    insights = generate_deterministic_insights(mixed_dataset_df, profile, quality, settings)
    stats_df = compute_extended_descriptive_stats(mixed_dataset_df, profile)
    outlier_df = analyze_all_numeric_outliers(mixed_dataset_df, method=settings.default_outlier_method)
    corr_res = compute_correlation_matrix(mixed_dataset_df, method=settings.default_correlation_method)
    ranked_pairs = extract_ranked_correlations(corr_res, top_n=5)

    # Test CSV exports
    assert isinstance(export_dataset_as_csv(mixed_dataset_df), bytes)
    assert isinstance(export_profile_summary_as_csv(profile), bytes)
    assert isinstance(export_quality_issues_as_csv(quality), bytes)
    assert isinstance(export_descriptive_stats_as_csv(mixed_dataset_df, profile), bytes)
    assert isinstance(export_correlation_matrix_as_csv(corr_res.matrix), bytes)
    assert isinstance(export_ranked_pairs_as_csv(ranked_pairs), bytes)
    assert isinstance(export_outlier_summary_as_csv(outlier_df), bytes)

    # Test HTML executive report generation
    from core.ingestion import DatasetMetadata

    meta = DatasetMetadata(
        filename="test_mixed_dataset.csv",
        file_format="CSV",
        file_size_bytes=1024,
        original_rows=len(mixed_dataset_df),
        original_cols=len(mixed_dataset_df.columns),
        loaded_rows=len(mixed_dataset_df),
        loaded_cols=len(mixed_dataset_df.columns),
        is_sampled=False,
    )

    html_report = generate_consolidated_html_report(
        metadata=meta,
        profile=profile,
        quality_report=quality,
        insights=insights,
        settings=settings,
        stats_df=stats_df,
        outlier_df=outlier_df,
        ranked_pairs=ranked_pairs,
    )

    assert isinstance(html_report, str)
    assert len(html_report) > 500
    assert "<!DOCTYPE html>" in html_report
    assert "EDA Intelligence Workbench" in html_report
    assert "Executive Audit Report" in html_report


def test_first_look_integration_with_samples() -> None:
    """Verify first-look analysis runs seamlessly on employee sample dataset."""
    emp_df = create_employee_analytics_sample()
    profile = compute_dataset_profile(emp_df)
    quality = analyze_data_quality(emp_df, profile)
    report = run_first_look_analysis(emp_df, profile, quality)
    assert len(report.facts) >= 2
    assert report.visual_candidates.primary_numeric_col is not None

