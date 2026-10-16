"""Advanced statistical analysis engine for EDA Intelligence Workbench.

Provides extended descriptive statistics, distribution normality diagnostics
(Shapiro-Wilk, D'Agostino-Pearson, Q-Q quantiles), parametric and bootstrap confidence intervals,
and inferential group comparisons (Welch's t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis, Chi-square).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from core.profiling import DatasetProfile


@dataclass
class NormalityTestResult:
    """Outcome of a statistical normality test."""

    feature: str
    test_name: str
    statistic: float
    p_value: float
    sample_size: int
    is_normal_at_alpha: bool
    status: str = "success"
    null_hypothesis: str = "The data is drawn from a normally distributed population."
    interpretation: str = ""
    assumptions: str = ""
    limitations: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConfidenceIntervalResult:
    """Confidence interval estimation for mean or median."""

    feature: str
    statistic_name: str  # "Mean (Student's t)" or "Median (Bootstrap)"
    point_estimate: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    margin_of_error: float | None
    sample_size: int
    method_details: str
    status: str = "success"
    warnings: list[str] = field(default_factory=list)

    @property
    def ci_lower(self) -> float:
        """Alias for lower_bound."""
        return self.lower_bound

    @property
    def ci_upper(self) -> float:
        """Alias for upper_bound."""
        return self.upper_bound


@dataclass
class GroupComparisonTestResult:
    """Result of a bivariate or multi-group statistical hypothesis test."""

    test_name: str
    target_metric: str
    grouping_feature: str
    groups_compared: list[str]
    statistic: float
    p_value: float
    degrees_of_freedom: float | None
    is_statistically_significant: bool
    status: str = "success"
    effect_size_name: str | None = None
    effect_size_value: float | None = None
    group_sizes: dict[str, int] = field(default_factory=dict)
    group_means: dict[str, float] = field(default_factory=dict)
    group_medians: dict[str, float] = field(default_factory=dict)
    null_hypothesis: str = ""
    interpretation: str = ""
    assumptions_and_cautions: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        """Alias for is_statistically_significant."""
        return self.is_statistically_significant


def compute_extended_descriptive_stats(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Compute comprehensive descriptive metrics across all valid numeric columns."""
    records = []

    for col in df.select_dtypes(include=[np.number]).columns:
        col_str = str(col)
        clean = df[col].dropna()

        # Filter to finite
        finite_vals = clean.to_numpy(dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        n = len(finite_vals)

        if n == 0:
            continue

        p_col = profile.column_profiles.get(col_str)
        mean_v = float(np.mean(finite_vals))
        median_v = float(np.median(finite_vals))
        std_v = float(np.std(finite_vals, ddof=1)) if n > 1 else 0.0
        var_v = float(np.var(finite_vals, ddof=1)) if n > 1 else 0.0
        min_v = float(np.min(finite_vals))
        max_v = float(np.max(finite_vals))
        range_v = max_v - min_v

        q25 = float(np.percentile(finite_vals, 25))
        q75 = float(np.percentile(finite_vals, 75))
        iqr_v = q75 - q25

        p05 = float(np.percentile(finite_vals, 5))
        p95 = float(np.percentile(finite_vals, 95))

        # Coefficient of variation (CV) = std / mean
        cv = (std_v / abs(mean_v)) if abs(mean_v) > 1e-9 else np.nan

        # Mode calculation
        try:
            mode_res = stats.mode(finite_vals, keepdims=False)
            mode_v = float(mode_res.mode) if hasattr(mode_res, "mode") else float(mode_res[0])
        except (ValueError, IndexError):
            mode_v = np.nan

        skew_v = p_col.skewness if p_col and p_col.skewness is not None else np.nan
        kurt_v = p_col.kurtosis if p_col and p_col.kurtosis is not None else np.nan

        records.append({
            "Feature": col_str,
            "Count (N)": n,
            "Missing": len(df) - n,
            "Missing %": f"{((len(df) - n) / max(1, len(df)) * 100):.1f}%",
            "Mean": round(mean_v, 3),
            "Median": round(median_v, 3),
            "Mode": round(mode_v, 3) if not math.isnan(mode_v) else "—",
            "Std Dev": round(std_v, 3),
            "Variance": round(var_v, 3),
            "CV (Dispersion)": round(cv, 3) if not math.isnan(cv) else "—",
            "Min": round(min_v, 3),
            "P05": round(p05, 3),
            "Q25 (Q1)": round(q25, 3),
            "Q75 (Q3)": round(q75, 3),
            "P95": round(p95, 3),
            "Max": round(max_v, 3),
            "Range": round(range_v, 3),
            "IQR": round(iqr_v, 3),
            "Skewness": round(skew_v, 3) if not math.isnan(skew_v) else "—",
            "Kurtosis": round(kurt_v, 3) if not math.isnan(kurt_v) else "—",
        })

    return pd.DataFrame(records)


def run_normality_test(
    series: pd.Series,
    feature_name: str,
    test_type: str = "shapiro",
    alpha: float = 0.05,
) -> NormalityTestResult:
    """Run Shapiro-Wilk or D'Agostino-Pearson test of normality with sample size checks."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    finite_vals = clean.to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    n = len(finite_vals)

    if n < 3:
        interp = "Insufficient non-missing finite data (minimum N >= 3 required)."
        return NormalityTestResult(
            feature=feature_name,
            test_name=test_type.capitalize(),
            statistic=0.0,
            p_value=1.0,
            sample_size=n,
            is_normal_at_alpha=False,
            status="error",
            interpretation=interp,
            limitations="Sample size is too small to assess normality.",
            warnings=[interp],
        )

    # Check zero variance
    if np.all(finite_vals == finite_vals[0]):
        interp = "Constant feature with zero variance; distribution is degenerate, not normal."
        return NormalityTestResult(
            feature=feature_name,
            test_name=test_type.capitalize(),
            statistic=0.0,
            p_value=0.0,
            sample_size=n,
            is_normal_at_alpha=False,
            status="error",
            interpretation=interp,
            limitations="Statistical tests of variance cannot be evaluated on constant series.",
            warnings=[interp],
        )

    if test_type.lower() == "shapiro":
        # Shapiro-Wilk is bounded to N <= 5000 in scipy
        sample_to_test = finite_vals[:5000] if n > 5000 else finite_vals
        try:
            stat_val, p_val = stats.shapiro(sample_to_test)
        except (ValueError, IndexError, UserWarning):
            stat_val, p_val = 0.0, 1.0

        is_norm = bool(p_val >= alpha)
        interp = (
            f"Fail to reject H0 (p = {p_val:.4f} >= {alpha}). Distribution is consistent with normal."
            if is_norm
            else f"Reject H0 (p = {p_val:.4e} < {alpha}). Distribution departs significantly from normal."
        )

        return NormalityTestResult(
            feature=feature_name,
            test_name="Shapiro-Wilk Test",
            statistic=round(float(stat_val), 4),
            p_value=float(p_val),
            sample_size=len(sample_to_test),
            is_normal_at_alpha=is_norm,
            interpretation=interp,
            assumptions="Observations are assumed to be independent and identically distributed.",
            limitations=(
                "Highly sensitive to minor departures from normality in large sample sizes (N > 500). "
                "Failure to reject H0 does not prove that the underlying population is exactly normal."
            ),
        )

    # D'Agostino-Pearson omnibus test
    if n < 20:
        interp = "Requires at least 20 observations to evaluate skewness and kurtosis moments."
        return NormalityTestResult(
            feature=feature_name,
            test_name="D'Agostino-Pearson Test",
            statistic=0.0,
            p_value=1.0,
            sample_size=n,
            is_normal_at_alpha=False,
            status="error",
            interpretation=interp,
            limitations="Sample size N < 20.",
            warnings=[interp],
        )

    try:
        stat_val, p_val = stats.normaltest(finite_vals)
    except (ValueError, IndexError, UserWarning):
        stat_val, p_val = 0.0, 1.0

    is_norm = bool(p_val >= alpha)
    interp = (
        f"Fail to reject H0 (p = {p_val:.4f} >= {alpha}). Skewness and kurtosis conform to normal expectation."
        if is_norm
        else f"Reject H0 (p = {p_val:.4e} < {alpha}). Skewness or kurtosis departs significantly from normal."
    )

    return NormalityTestResult(
        feature=feature_name,
        test_name="D'Agostino-Pearson Test",
        statistic=round(float(stat_val), 4),
        p_value=float(p_val),
        sample_size=n,
        is_normal_at_alpha=is_norm,
        interpretation=interp,
        assumptions="Combines skewness and kurtosis omnibus test statistics.",
        limitations="Requires N >= 20. Does not prove exact normality, only tests moment conformity.",
    )


def compute_confidence_intervals(
    series: pd.Series,
    feature_name: str,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 500,
    seed: int = 42,
) -> list[ConfidenceIntervalResult]:
    """Compute Student's t confidence interval for mean and Bootstrap interval for median."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    vals = clean.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)

    results: list[ConfidenceIntervalResult] = []
    if n < 2:
        return results

    # 1. Student's t CI for Mean
    mean_v = float(np.mean(vals))
    std_err = float(stats.sem(vals))
    dof = n - 1

    if std_err == 0.0 or math.isnan(std_err):
        lower_t, upper_t = mean_v, mean_v
        moe = 0.0
    else:
        t_crit = float(stats.t.ppf((1 + confidence_level) / 2.0, df=dof))
        moe = t_crit * std_err
        lower_t = mean_v - moe
        upper_t = mean_v + moe

    results.append(
        ConfidenceIntervalResult(
            feature=feature_name,
            statistic_name=f"Mean ({int(confidence_level * 100)}% Student's t)",
            point_estimate=round(mean_v, 3),
            confidence_level=confidence_level,
            lower_bound=round(lower_t, 3),
            upper_bound=round(upper_t, 3),
            margin_of_error=round(moe, 3),
            sample_size=n,
            method_details=f"Parametric Student's t-distribution with {dof} degrees of freedom.",
        )
    )

    # 2. Bootstrap CI for Median (Non-parametric)
    median_v = float(np.median(vals))
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(bootstrap_resamples, n))
    boot_samples = vals[boot_indices]
    boot_medians = np.median(boot_samples, axis=1)

    alpha_half = (1.0 - confidence_level) / 2.0
    lower_boot = float(np.percentile(boot_medians, alpha_half * 100))
    upper_boot = float(np.percentile(boot_medians, (1.0 - alpha_half) * 100))

    results.append(
        ConfidenceIntervalResult(
            feature=feature_name,
            statistic_name=f"Median ({int(confidence_level * 100)}% Bootstrap)",
            point_estimate=round(median_v, 3),
            confidence_level=confidence_level,
            lower_bound=round(lower_boot, 3),
            upper_bound=round(upper_boot, 3),
            margin_of_error=None,
            sample_size=n,
            method_details=f"Empirical percentile bootstrap ({bootstrap_resamples} resamples, seed={seed}).",
        )
    )

    return results


def run_group_comparison_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    test_type: str = "auto",  # "auto", "welch_t", "mann_whitney", "anova", "kruskal"
    alpha: float = 0.05,
) -> GroupComparisonTestResult:
    """Run parametric or non-parametric group difference tests across categorical segments."""
    sub = df[[metric_col, group_col]].dropna().copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.dropna()

    groups = [str(g) for g in sub[group_col].unique()]
    group_data: dict[str, np.ndarray] = {
        g: sub[sub[group_col].astype(str) == g][metric_col].to_numpy(dtype=float)
        for g in groups
    }

    # Filter out groups with fewer than 3 samples
    valid_groups = [g for g in groups if len(group_data[g]) >= 3]

    if len(valid_groups) < 2:
        msg = "At least two groups with >= 3 observations are required."
        return GroupComparisonTestResult(
            test_name="Group Comparison",
            target_metric=metric_col,
            grouping_feature=group_col,
            groups_compared=groups,
            statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=None,
            is_statistically_significant=False,
            status="error",
            interpretation=msg,
            warnings=[msg],
        )

    # Compute group summaries
    g_sizes = {g: len(group_data[g]) for g in valid_groups}
    g_means = {g: round(float(np.mean(group_data[g])), 2) for g in valid_groups}
    g_medians = {g: round(float(np.median(group_data[g])), 2) for g in valid_groups}

    # TWO-GROUP COMPARISONS
    if len(valid_groups) == 2:
        g1, g2 = valid_groups[0], valid_groups[1]
        arr1, arr2 = group_data[g1], group_data[g2]

        # Calculate Cohen's d effect size: (mean1 - mean2) / pooled_std
        n1, n2 = len(arr1), len(arr2)
        s1, s2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
        pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, n1 + n2 - 2))
        cohen_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0.0

        if test_type in ("mann_whitney", "non_parametric"):
            res = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
            stat_val, p_val = float(res.statistic), float(res.pvalue)
            t_name = "Mann-Whitney U Test (Non-parametric)"
            null_h = f"Distributions of '{metric_col}' are identical across '{g1}' and '{g2}'."
            cautions = "Does not assume normal distribution or equal variances. Tests stochastic equality."
        else:
            # Welch's t-test (robust against unequal variances)
            res = stats.ttest_ind(arr1, arr2, equal_var=False)
            stat_val, p_val = float(res.statistic), float(res.pvalue)
            t_name = "Welch's Two-Sample t-test (Unequal Variances)"
            null_h = f"True population mean of '{metric_col}' is equal for '{g1}' and '{g2}'."
            cautions = "Robust against heteroscedasticity. Assumes observations are independent."

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant difference detected (p = {p_val:.4f} < {alpha}, Cohen's d = {cohen_d:.2f})."
            if is_sig
            else f"No statistically significant difference detected (p = {p_val:.4f} >= {alpha})."
        )

        return GroupComparisonTestResult(
            test_name=t_name,
            target_metric=metric_col,
            grouping_feature=group_col,
            groups_compared=[g1, g2],
            statistic=round(stat_val, 3),
            p_value=round(p_val, 5),
            degrees_of_freedom=float(n1 + n2 - 2),
            is_statistically_significant=is_sig,
            effect_size_name="Cohen's d",
            effect_size_value=round(cohen_d, 2),
            group_sizes=g_sizes,
            group_means=g_means,
            group_medians=g_medians,
            null_hypothesis=null_h,
            interpretation=interp,
            assumptions_and_cautions=cautions,
        )

    # MULTI-GROUP COMPARISONS (>= 3 groups)
    group_arrays = [group_data[g] for g in valid_groups]

    if test_type in ("kruskal", "non_parametric"):
        res = stats.kruskal(*group_arrays)
        stat_val, p_val = float(res.statistic), float(res.pvalue)
        t_name = f"Kruskal-Wallis H-test ({len(valid_groups)} groups)"
        null_h = f"Population medians of '{metric_col}' are identical across all {len(valid_groups)} groups."
        cautions = "Non-parametric alternative to one-way ANOVA. Does not assume normality."
    else:
        res = stats.f_oneway(*group_arrays)
        stat_val, p_val = float(res.statistic), float(res.pvalue)
        t_name = f"One-Way ANOVA F-test ({len(valid_groups)} groups)"
        null_h = f"Population means of '{metric_col}' are equal across all {len(valid_groups)} groups."
        cautions = "Assumes approximately normal distributions and homogeneity of variance across groups."

    is_sig = p_val < alpha
    interp = (
        f"Statistically significant variance observed among groups (p = {p_val:.4f} < {alpha})."
        if is_sig
        else f"No significant between-group variance detected (p = {p_val:.4f} >= {alpha})."
    )

    return GroupComparisonTestResult(
        test_name=t_name,
        target_metric=metric_col,
        grouping_feature=group_col,
        groups_compared=valid_groups,
        statistic=round(stat_val, 3),
        p_value=round(p_val, 5),
        degrees_of_freedom=float(len(valid_groups) - 1),
        is_statistically_significant=is_sig,
        group_sizes=g_sizes,
        group_means=g_means,
        group_medians=g_medians,
        null_hypothesis=null_h,
        interpretation=interp,
        assumptions_and_cautions=cautions,
    )


def run_chi_square_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, float, float, int, str]:
    """Run Chi-Square Test of Independence for two categorical columns.

    Returns:
        tuple: (contingency_table, chi2_stat, p_value, degrees_of_freedom, caution_message)
    """
    sub = df[[col_a, col_b]].dropna().copy()
    contingency = pd.crosstab(sub[col_a], sub[col_b])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return contingency, 0.0, 1.0, 0, "Requires at least 2 distinct categories in each column."

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    caution = ""
    # Check Cochran's rule: expected counts < 5
    if (expected < 5).sum() > 0:
        pct_low = ((expected < 5).sum() / expected.size) * 100.0
        caution = (
            f"Warning: {pct_low:.1f}% of contingency cells have expected counts < 5. "
            "Chi-square asymptotic p-value may be unreliable."
        )

    return contingency, round(float(chi2), 3), float(p), int(dof), caution


@dataclass
class ChiSquareTestResult:
    """Outcome of a Chi-square test of independence."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    status: str = "success"
    caution: str = ""
    interpretation: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        """Convenience property for significance test outcome."""
        return "statistically significant association detected" in self.interpretation.lower()


def test_shapiro_wilk(data: np.ndarray | pd.Series, alpha: float = 0.05) -> NormalityTestResult:
    """Convenience function for Shapiro-Wilk test on array or series."""
    s = pd.Series(data) if isinstance(data, np.ndarray) else data
    return run_normality_test(s, "sample", test_type="shapiro", alpha=alpha)


def test_dagostino_pearson(data: np.ndarray | pd.Series, alpha: float = 0.05) -> NormalityTestResult:
    """Convenience function for D'Agostino-Pearson test on array or series."""
    s = pd.Series(data) if isinstance(data, np.ndarray) else data
    return run_normality_test(s, "sample", test_type="dagostino", alpha=alpha)


def compute_mean_confidence_interval(
    data: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Convenience function for parametric Student's t CI for the mean."""
    s = pd.Series(data) if isinstance(data, np.ndarray) else data
    res = compute_confidence_intervals(s, "sample", confidence_level=confidence_level)
    return res[0] if res else ConfidenceIntervalResult(
        feature="sample",
        statistic_name="Mean",
        point_estimate=0.0,
        confidence_level=confidence_level,
        lower_bound=0.0,
        upper_bound=0.0,
        margin_of_error=0.0,
        sample_size=0,
        method_details="Insufficient data",
        status="error",
    )


def compute_bootstrap_mean_ci(
    data: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> ConfidenceIntervalResult:
    """Convenience function for bootstrap confidence interval for the mean."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return ConfidenceIntervalResult(
            feature="sample",
            statistic_name="Bootstrap Mean",
            point_estimate=0.0,
            confidence_level=confidence_level,
            lower_bound=0.0,
            upper_bound=0.0,
            margin_of_error=None,
            sample_size=n,
            method_details="Insufficient data",
            status="error",
        )
    rng = np.random.default_rng(random_seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(arr[boot_indices], axis=1)
    alpha_half = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha_half * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha_half) * 100))
    point_est = float(np.mean(arr))
    return ConfidenceIntervalResult(
        feature="sample",
        statistic_name=f"Mean ({int(confidence_level * 100)}% Bootstrap)",
        point_estimate=point_est,
        confidence_level=confidence_level,
        lower_bound=lower,
        upper_bound=upper,
        margin_of_error=None,
        sample_size=n,
        method_details=f"Empirical percentile bootstrap ({n_bootstrap} resamples, seed={random_seed}).",
        status="success",
    )


def compute_cohens_d(g1: np.ndarray | pd.Series, g2: np.ndarray | pd.Series) -> float:
    """Compute Cohen's d standardized effect size between two groups."""
    a1 = np.asarray(g1, dtype=float)
    a2 = np.asarray(g2, dtype=float)
    a1 = a1[np.isfinite(a1)]
    a2 = a2[np.isfinite(a2)]
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(a1, ddof=1), np.var(a2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, n1 + n2 - 2))
    return float((np.mean(a1) - np.mean(a2)) / pooled_std) if pooled_std > 0 else 0.0


def test_welchs_ttest(
    g1: np.ndarray | pd.Series,
    g2: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> GroupComparisonTestResult:
    """Convenience function for Welch's two-sample t-test."""
    df_temp = pd.DataFrame({
        "val": np.concatenate([np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)]),
        "grp": ["G1"] * len(g1) + ["G2"] * len(g2),
    })
    return run_group_comparison_test(df_temp, "val", "grp", test_type="welch_t", alpha=alpha)


def test_mann_whitney_u(
    g1: np.ndarray | pd.Series,
    g2: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> GroupComparisonTestResult:
    """Convenience function for Mann-Whitney U rank-sum test."""
    df_temp = pd.DataFrame({
        "val": np.concatenate([np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)]),
        "grp": ["G1"] * len(g1) + ["G2"] * len(g2),
    })
    return run_group_comparison_test(df_temp, "val", "grp", test_type="mann_whitney", alpha=alpha)


def test_one_way_anova(
    groups: list[np.ndarray | pd.Series],
    alpha: float = 0.05,
) -> GroupComparisonTestResult:
    """Convenience function for One-Way ANOVA F-test across multiple groups."""
    vals = []
    labels = []
    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        vals.append(arr)
        labels.extend([f"G{idx}"] * len(arr))
    df_temp = pd.DataFrame({"val": np.concatenate(vals), "grp": labels})
    return run_group_comparison_test(df_temp, "val", "grp", test_type="anova", alpha=alpha)


def test_kruskal_wallis(
    groups: list[np.ndarray | pd.Series],
    alpha: float = 0.05,
) -> GroupComparisonTestResult:
    """Convenience function for Kruskal-Wallis H-test across multiple groups."""
    vals = []
    labels = []
    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        vals.append(arr)
        labels.extend([f"G{idx}"] * len(arr))
    df_temp = pd.DataFrame({"val": np.concatenate(vals), "grp": labels})
    return run_group_comparison_test(df_temp, "val", "grp", test_type="kruskal", alpha=alpha)


def test_chi_square_independence(
    table: np.ndarray | pd.DataFrame,
    alpha: float = 0.05,
) -> ChiSquareTestResult:
    """Convenience function for Chi-Square test on contingency matrix."""
    mat = np.asarray(table)
    chi2, p, dof, expected = stats.chi2_contingency(mat)
    caution = ""
    if (expected < 5).sum() > 0:
        pct_low = ((expected < 5).sum() / expected.size) * 100.0
        caution = f"Warning: {pct_low:.1f}% expected counts < 5."

    is_sig = p < alpha
    interp = (
        f"Statistically significant association detected (p = {p:.4e} < {alpha})."
        if is_sig
        else f"No statistically significant association detected (p = {p:.4f} >= {alpha})."
    )

    return ChiSquareTestResult(
        statistic=float(chi2),
        p_value=float(p),
        degrees_of_freedom=int(dof),
        status="success",
        caution=caution,
        interpretation=interp,
        warnings=[caution] if caution else [],
    )


