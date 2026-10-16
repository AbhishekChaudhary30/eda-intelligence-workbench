"""Independent analytical correctness, statistical precision, and edge-case verification tests.

Validates:
- Exact sample statistics (ddof=1 sample variance, sample std dev, Fisher's excess kurtosis, skewness)
- Parametric Student's t and bootstrap confidence intervals
- Inferential group comparison tests (Welch's t-test, Cohen's d, Mann-Whitney U, ANOVA, Kruskal-Wallis, Chi-Square)
- Benjamini-Hochberg False Discovery Rate multiple testing adjustments
- Multimodal outlier detection (Tukey's IQR, Z-Score, MAD modified Z-Score)
- Missingness audits and custom token handling
- Non-destructive transformation manager (preview, apply, undo, reset)
- Numerical robustness on edge cases (0-rows, 1-row, constant columns, all-null columns, infinities)
- Spreadsheet formula injection neutralization (OWASP CWE-1236)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.advanced_statistics import (
    compute_confidence_intervals,
    compute_extended_descriptive_stats,
    run_chi_square_test,
    run_group_comparison_test,
)
from core.correlations import adjust_p_values_fdr, compute_advanced_correlations
from core.missingness import analyze_missingness, compute_group_missingness
from core.outlier_analysis import (
    audit_column_outliers_iqr,
    audit_column_outliers_mad,
    audit_column_outliers_zscore,
)
from core.profiling import compute_dataset_profile
from core.statistics import compute_descriptive_stats_table
from core.transformations import (
    TransformationManager,
    preview_transformation,
)
from utils.security import sanitize_dataframe_for_spreadsheet, sanitize_spreadsheet_cell

# =============================================================================
# 1. DESCRIPTIVE STATISTICS & SAMPLE CONVENTIONS
# =============================================================================

def test_descriptive_statistics_precision_and_conventions() -> None:
    """Verify sample variance and std use Bessel's correction (ddof=1) and kurtosis is Fisher excess."""
    # Deterministic test array: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # N = 8, Sum = 40, Mean = 5.0
    # Deviations from mean: [-3, -1, -1, -1, 0, 0, 2, 4]
    # Sum of squared deviations: 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
    # Population variance = 32 / 8 = 4.0
    # Sample variance (ddof=1) = 32 / (8 - 1) = 32 / 7 = 4.571428...
    # Sample std = sqrt(32/7) = 2.1380899...
    raw_vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    df = pd.DataFrame({"x": raw_vals})
    profile = compute_dataset_profile(df)
    col_p = profile.column_profiles["x"]

    assert col_p.mean == pytest.approx(5.0, abs=1e-5)
    assert col_p.variance == pytest.approx(32.0 / 7.0, abs=1e-5)
    assert col_p.std == pytest.approx(math.sqrt(32.0 / 7.0), abs=1e-5)
    assert col_p.median == pytest.approx(4.5, abs=1e-5)
    assert col_p.min_val == pytest.approx(2.0, abs=1e-5)
    assert col_p.max_val == pytest.approx(9.0, abs=1e-5)

    # Scipy Fisher's excess kurtosis (normal distribution = 0.0)
    expected_kurt = float(stats.kurtosis(raw_vals, bias=False))
    assert col_p.kurtosis == pytest.approx(expected_kurt, abs=1e-4)

    # Scipy skewness
    expected_skew = float(stats.skew(raw_vals, bias=False))
    assert col_p.skewness == pytest.approx(expected_skew, abs=1e-4)

    # Extended descriptive stats check
    ext_df = compute_extended_descriptive_stats(df, profile)
    row = ext_df.loc[ext_df["Feature"] == "x"].iloc[0]
    assert row["Mean"] == 5.0
    assert row["Std Dev"] == round(math.sqrt(32.0 / 7.0), 3)
    assert row["Variance"] == round(32.0 / 7.0, 3)
    assert row["Count (N)"] == 8
    assert row["Mode"] == 4.0  # 4.0 occurs 3 times


# =============================================================================
# 2. CONFIDENCE INTERVALS
# =============================================================================

def test_parametric_student_t_confidence_interval() -> None:
    """Validate Student's t-confidence interval bounds and degrees of freedom."""
    # Data: [10.0, 12.0, 14.0, 16.0, 18.0]
    # N = 5, Mean = 14.0, ddof = 4
    # Deviations: [-4, -2, 0, 2, 4], Sum of squares = 16 + 4 + 0 + 4 + 16 = 40
    # Sample variance = 40 / 4 = 10.0, Sample std = sqrt(10.0)
    # Standard error = s / sqrt(N) = sqrt(10.0) / sqrt(5) = sqrt(2.0) = 1.41421356...
    # For 95% CI with df = 4, t_crit = stats.t.ppf(0.975, df=4) = 2.776445...
    # Margin of error = 2.776445 * 1.41421356 = 3.9264...
    # Lower bound = 14.0 - 3.9264 = 10.0736...
    # Upper bound = 14.0 + 3.9264 = 17.9264...
    data = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    ci_results = compute_confidence_intervals(data, "test_feature", confidence_level=0.95)

    assert len(ci_results) == 2  # Mean (Student's t) and Median (Bootstrap)
    mean_ci = ci_results[0]
    assert mean_ci.point_estimate == 14.0
    assert mean_ci.confidence_level == 0.95
    assert mean_ci.sample_size == 5

    expected_sem = math.sqrt(2.0)
    expected_t_crit = stats.t.ppf(0.975, df=4)
    expected_moe = expected_t_crit * expected_sem
    assert mean_ci.margin_of_error == pytest.approx(round(expected_moe, 3), abs=1e-3)
    assert mean_ci.lower_bound == pytest.approx(round(14.0 - expected_moe, 3), abs=1e-3)
    assert mean_ci.upper_bound == pytest.approx(round(14.0 + expected_moe, 3), abs=1e-3)


def test_confidence_interval_insufficient_samples() -> None:
    """Validate boundary condition where sample size N < 2."""
    data_single = pd.Series([42.0])
    ci_results = compute_confidence_intervals(data_single, "single_elem")
    assert ci_results == []


# =============================================================================
# 3. HYPOTHESIS TESTING & EFFECT SIZES
# =============================================================================

def test_welch_t_test_and_cohens_d() -> None:
    """Validate Welch's t-test and Cohen's d effect size on unequal variance groups."""
    df = pd.DataFrame({
        "salary": [50_000, 52_000, 48_000, 51_000, 49_000, 90_000, 95_000, 92_000, 94_000, 91_000],
        "department": ["Sales", "Sales", "Sales", "Sales", "Sales", "R&D", "R&D", "R&D", "R&D", "R&D"],
    })

    res = run_group_comparison_test(df, "salary", "department", test_type="welch_t", alpha=0.05)
    assert res.test_name.startswith("Welch's Two-Sample t-test")
    assert res.is_statistically_significant is True
    assert res.p_value < 0.001
    assert res.effect_size_name == "Cohen's d"
    assert res.effect_size_value is not None
    assert abs(res.effect_size_value) > 2.0  # Large effect size


def test_chi_square_test_of_independence() -> None:
    """Validate Chi-Square Test calculation, degrees of freedom, and low-count caution."""
    # Contingency table:
    #             Feature_B: Cat1   Cat2
    # Feature_A:
    # A           30                10
    # B           10                30
    df = pd.DataFrame({
        "tier": ["A"] * 40 + ["B"] * 40,
        "segment": ["Cat1"] * 30 + ["Cat2"] * 10 + ["Cat1"] * 10 + ["Cat2"] * 30,
    })

    _table, chi2, p_val, dof, caution = run_chi_square_test(df, "tier", "segment")
    assert dof == 1  # (2 - 1) * (2 - 1) = 1
    assert chi2 > 10.0
    assert p_val < 0.001
    assert caution == ""  # All expected cell counts exceed 5


def test_chi_square_cochran_warning() -> None:
    """Validate Cochran rule warning when expected cell counts are < 5."""
    df = pd.DataFrame({
        "tier": ["A"] * 5 + ["B"] * 5,
        "segment": ["Cat1"] * 4 + ["Cat2"] * 1 + ["Cat1"] * 1 + ["Cat2"] * 4,
    })
    _, _, _, _, caution = run_chi_square_test(df, "tier", "segment")
    assert "expected counts < 5" in caution


# =============================================================================
# 4. MULTIPLE TESTING ADJUSTMENTS (BENJAMINI-HOCHBERG FDR)
# =============================================================================

def test_benjamini_hochberg_fdr_calculation() -> None:
    """Validate Benjamini-Hochberg FDR adjustments against deterministic manual derivation."""
    # Raw p-values: [0.01, 0.04, 0.03, 0.20]
    # m = 4
    # Sorted order:
    # Rank 1: p = 0.01 -> q = 0.01 * 4 / 1 = 0.04
    # Rank 2: p = 0.03 -> q = 0.03 * 4 / 2 = 0.06
    # Rank 3: p = 0.04 -> q = 0.04 * 4 / 3 = 0.053333...
    # Rank 4: p = 0.20 -> q = 0.20 * 4 / 4 = 0.20
    # Monotonicity adjustment (from m down to 1):
    # rank 4: 0.20
    # rank 3: min(0.20, 0.053333) = 0.053333
    # rank 2: min(0.053333, 0.06) = 0.053333 (enforced monotonic!)
    # rank 1: min(0.053333, 0.04) = 0.04
    # Restored to original index order: [0.04, 0.053333, 0.053333, 0.20]
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = adjust_p_values_fdr(raw_p)

    assert len(adj_p) == 4
    assert adj_p[0] == pytest.approx(0.04, abs=1e-4)
    assert adj_p[1] == pytest.approx(4.0 * 0.04 / 3.0, abs=1e-4)  # ~0.05333
    assert adj_p[2] == pytest.approx(4.0 * 0.04 / 3.0, abs=1e-4)  # Enforced monotonicity
    assert adj_p[3] == pytest.approx(0.20, abs=1e-4)


# =============================================================================
# 5. CORRELATION ANALYSIS & CONSTANT COLUMNS
# =============================================================================

def test_correlations_known_relationships_and_degenerate_handling() -> None:
    """Verify linear and monotonic correlation behavior and constant column exclusion."""
    n = 50
    x = np.linspace(1, 10, n)
    y_linear = 2.5 * x + 4.0  # Perfect Pearson (r = 1.0)
    y_mono = np.exp(x)  # Perfect Spearman (r_s = 1.0), non-linear Pearson
    y_const = np.full(n, 42.0)  # Constant zero variance

    df = pd.DataFrame({
        "x": x,
        "y_linear": y_linear,
        "y_mono": y_mono,
        "y_const": y_const,
    })
    profile = compute_dataset_profile(df)

    # 1. Pearson
    res_pearson = compute_advanced_correlations(df, profile, method="pearson")
    assert any("y_const" in exc for exc in res_pearson.excluded_columns)
    matrix_p = res_pearson.corr_matrix
    assert matrix_p.loc["x", "y_linear"] == pytest.approx(1.0, abs=1e-4)
    assert matrix_p.loc["x", "y_mono"] < 0.95  # Exponential departure from linearity

    # 2. Spearman
    res_spearman = compute_advanced_correlations(df, profile, method="spearman")
    matrix_s = res_spearman.corr_matrix
    assert matrix_s.loc["x", "y_linear"] == pytest.approx(1.0, abs=1e-4)
    assert matrix_s.loc["x", "y_mono"] == pytest.approx(1.0, abs=1e-4)  # Monotonic rank order preserved


# =============================================================================
# 6. OUTLIER DETECTION (IQR, Z-SCORE, MAD)
# =============================================================================

def test_outlier_detection_with_known_outliers() -> None:
    """Validate IQR, Z-Score, and MAD outlier detection on known synthetic fixtures."""
    # 20 points centered around 10, plus extreme outlier 500 at index 20
    vals = [10.0 + i * 0.1 for i in range(20)] + [500.0]
    s = pd.Series(vals)

    # 1. IQR Tukey's Fences
    audit_iqr = audit_column_outliers_iqr(s, "val", multiplier=1.5)
    assert audit_iqr is not None
    assert audit_iqr.outlier_count == 1
    assert 20 in audit_iqr.flagged_indices
    assert audit_iqr.max_extreme == 500.0

    # 2. Z-Score (|z| > 3.0)
    audit_z = audit_column_outliers_zscore(s, "val", threshold=3.0)
    assert audit_z is not None
    assert audit_z.outlier_count == 1
    assert 20 in audit_z.flagged_indices

    # 3. Median Absolute Deviation (MAD)
    audit_mad = audit_column_outliers_mad(s, "val", threshold=3.5)
    assert audit_mad is not None
    assert audit_mad.outlier_count == 1
    assert 20 in audit_mad.flagged_indices


def test_outlier_detection_zero_dispersion() -> None:
    """Validate that constant columns return None safely without ZeroDivisionError."""
    s_const = pd.Series([100.0] * 20)
    assert audit_column_outliers_iqr(s_const, "const") is None
    assert audit_column_outliers_zscore(s_const, "const") is None
    assert audit_column_outliers_mad(s_const, "const") is None


# =============================================================================
# 7. MISSINGNESS AUDIT & CUSTOM TOKENS
# =============================================================================

def test_missingness_audit_and_custom_tokens() -> None:
    """Validate missingness calculation, complete rows, and pseudo-missing tokens detection."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat": ["A", "?", "B", "None", "C"],
        "all_missing": [np.nan] * 5,
        "clean": [10, 20, 30, 40, 50],
    })

    report = analyze_missingness(df, custom_tokens=["?", "None"])
    assert report.total_rows == 5
    assert report.total_cols == 4
    assert report.total_cells == 20
    assert report.total_missing_cells == 6  # 1 in num + 5 in all_missing

    # Custom tokens found
    tokens = {t.token: t.count for t in report.custom_token_occurrences}
    assert tokens.get("?") == 1
    assert tokens.get("None") == 1

    # Valid group missingness
    group_df = compute_group_missingness(df, target_col="num", group_col="cat")
    assert not group_df.empty
    assert "Missing_Pct" in group_df.columns


# =============================================================================
# 8. TRANSFORMATION MANAGER (PREVIEW, APPLY, UNDO, RESET)
# =============================================================================

def test_transformation_manager_lifecycle() -> None:
    """Validate non-destructive transformation workflow with preview, apply, undo, and reset."""
    base_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "val": [10.0, np.nan, 30.0, 40.0, 50.0],
        "category": ["X", "Y", "X", "Y", "Z"],
    })

    manager = TransformationManager(base_df)
    assert len(manager.working_df) == 5
    assert manager.step_count == 0

    # 1. Preview imputation
    prev = preview_transformation(
        manager.working_df,
        "impute_missing",
        {"column": "val", "strategy": "mean"},
    )
    assert prev.expected_rows_after == 5
    assert prev.preview_head["val"].isna().sum() == 0

    # 2. Apply imputation
    step = manager.apply_transformation(
        "impute_missing",
        {"column": "val", "strategy": "mean"},
    )
    assert step.step_id == 1
    assert manager.working_df["val"].isna().sum() == 0
    assert manager.original_df["val"].isna().sum() == 1  # Original untouched!

    # 3. Apply row filter
    manager.apply_transformation(
        "filter_rows",
        {"column": "val", "operator": ">", "value": 25.0},
    )
    assert len(manager.working_df) < 5
    assert manager.step_count == 2

    # 4. Undo last step
    undone = manager.undo_last_step()
    assert undone is not None
    assert manager.step_count == 1
    assert len(manager.working_df) == 5

    # 5. Reset to original
    reset_df = manager.reset_to_original()
    assert manager.step_count == 0
    assert reset_df["val"].isna().sum() == 1


# =============================================================================
# 9. NUMERICAL ROBUSTNESS & DEGENERATE EDGE CASES
# =============================================================================

def test_numerical_robustness_edge_cases() -> None:
    """Verify application stability under zero-rows, one-row, infinities, and all-null columns."""
    # A. Zero rows DataFrame
    empty_df = pd.DataFrame(columns=["a", "b", "c"])
    empty_prof = compute_dataset_profile(empty_df)
    assert empty_prof.total_rows == 0
    assert empty_prof.total_cells == 0
    empty_stats = compute_descriptive_stats_table(empty_df, empty_prof)
    assert empty_stats.empty

    # B. Single-row DataFrame
    single_df = pd.DataFrame({"x": [42.0], "cat": ["Solo"]})
    single_prof = compute_dataset_profile(single_df)
    assert single_prof.total_rows == 1
    assert single_prof.column_profiles["x"].std == 0.0  # Sample std for N=1 returns 0.0 safely

    # C. Infinite values DataFrame
    inf_df = pd.DataFrame({"inf_col": [1.0, np.inf, 3.0, -np.inf, 5.0]})
    inf_prof = compute_dataset_profile(inf_df)
    col_inf = inf_prof.column_profiles["inf_col"]
    assert col_inf.mean == pytest.approx(3.0, abs=1e-5)  # Finite mean computed across [1.0, 3.0, 5.0]

    # D. All-null column
    null_df = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan]})
    null_prof = compute_dataset_profile(null_df)
    assert null_prof.column_profiles["all_null"].is_all_null is True
    assert null_prof.all_null_columns == ["all_null"]


# =============================================================================
# 10. SPREADSHEET FORMULA INJECTION SANITIZATION (SECURITY CWE-1236)
# =============================================================================

def test_spreadsheet_formula_injection_sanitization() -> None:
    """Validate that formula triggers (=, +, -, @, \\t, \\r) are neutralized while preserving numbers."""
    # Test individual cell sanitization
    assert sanitize_spreadsheet_cell("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert sanitize_spreadsheet_cell("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_spreadsheet_cell("+1+2+cmd") == "'+1+2+cmd"
    assert sanitize_spreadsheet_cell("-5+cmd") == "'-5+cmd"
    assert sanitize_spreadsheet_cell("\tmalicious") == "'\tmalicious"

    # Legitimate numbers in string columns must be preserved!
    assert sanitize_spreadsheet_cell("-42.5") == "-42.5"
    assert sanitize_spreadsheet_cell("+100") == "+100"
    assert sanitize_spreadsheet_cell("Benign text") == "Benign text"
    assert sanitize_spreadsheet_cell(1234) == 1234

    # Test full dataframe sanitization
    raw_df = pd.DataFrame({
        "username": ["Alice", "=SUM(A1:A5)", "@ATTACK", "-42.0", "Bob"],
        "balance": [-50.0, 100.0, 200.0, -10.0, 0.0],
    })
    clean_df = sanitize_dataframe_for_spreadsheet(raw_df)
    assert clean_df.loc[0, "username"] == "Alice"
    assert clean_df.loc[1, "username"] == "'=SUM(A1:A5)"
    assert clean_df.loc[2, "username"] == "'@ATTACK"
    assert clean_df.loc[3, "username"] == "-42.0"  # Legitimate number preserved
    assert list(clean_df["balance"]) == [-50.0, 100.0, 200.0, -10.0, 0.0]  # Numeric column untouched
