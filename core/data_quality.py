"""Data Quality Analyzer for EDA Intelligence Workbench.

Audits datasets for completeness, uniqueness, consistency, anomalies, non-finite values,
mixed types, and statistical outliers (IQR and Z-score methods) with documented severity rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.profiling import DatasetProfile


@dataclass
class OutlierSummary:
    """Outlier analysis result for a single numeric column."""

    column_name: str
    method: str  # "IQR (Tukey's Fences)" or "Z-Score (|z| > 3)"
    outlier_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float
    min_outlier: float | None = None
    max_outlier: float | None = None
    explanation: str = ""
    assumptions: str = ""


@dataclass
class QualityIssue:
    """Detailed record of an identified data-quality issue or anomaly."""

    name: str
    category: str  # "Completeness", "Uniqueness", "Validity", "Consistency", "Anomalies"
    severity: str  # "Critical", "Warning", "Informational"
    affected_columns: list[str]
    affected_count: int
    affected_percentage: float
    explanation: str
    suggested_next_step: str


@dataclass
class QualityReport:
    """Comprehensive dataset quality audit report."""

    overall_health_score: int  # 0 to 100
    health_status: str  # "Excellent", "Good", "Fair", "Needs Attention"
    total_issues_count: int
    critical_issues_count: int
    warning_issues_count: int
    info_issues_count: int
    issues: list[QualityIssue]
    iqr_outliers: dict[str, OutlierSummary]
    zscore_outliers: dict[str, OutlierSummary]

    @property
    def overall_score(self) -> int:
        """Alias for overall_health_score."""
        return self.overall_health_score


def detect_iqr_outliers(series: pd.Series, col_name: str) -> OutlierSummary | None:
    """Detect outliers using Tukey's IQR rule: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    clean = series.dropna()
    if len(clean) < 10:
        return None

    try:
        vals = clean.astype(float).values
    except (ValueError, TypeError):
        vals = pd.to_numeric(clean, errors="coerce").dropna().values
        if len(vals) < 10:
            return None

    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr = q75 - q25

    if iqr == 0:
        return None

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outlier_mask = (vals < lower_bound) | (vals > upper_bound)
    outlier_count = int(np.sum(outlier_mask))
    outlier_pct = round((outlier_count / len(vals)) * 100.0, 2)

    if outlier_count == 0:
        return None

    outlier_vals = vals[outlier_mask]
    min_out = float(np.min(outlier_vals))
    max_out = float(np.max(outlier_vals))

    return OutlierSummary(
        column_name=col_name,
        method="IQR (Tukey's Fences)",
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        lower_bound=round(lower_bound, 3),
        upper_bound=round(upper_bound, 3),
        min_outlier=round(min_out, 3),
        max_outlier=round(max_out, 3),
        explanation=(
            f"Found {outlier_count:,} values ({outlier_pct}%) outside the fence "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]."
        ),
        assumptions="Does not assume normal distribution. Robust to extreme skew.",
    )


def detect_zscore_outliers(series: pd.Series, col_name: str, threshold: float = 3.0) -> OutlierSummary | None:
    """Detect outliers using standard Z-score rule: |z| > 3.0."""
    clean = series.dropna()
    if len(clean) < 15:
        return None

    try:
        vals = clean.astype(float).values
    except (ValueError, TypeError):
        vals = pd.to_numeric(clean, errors="coerce").dropna().values
        if len(vals) < 15:
            return None

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1))

    if std_val == 0.0:
        return None

    z_scores = np.abs((vals - mean_val) / std_val)
    outlier_mask = z_scores > threshold
    outlier_count = int(np.sum(outlier_mask))
    outlier_pct = round((outlier_count / len(vals)) * 100.0, 2)

    lower_bound = mean_val - (threshold * std_val)
    upper_bound = mean_val + (threshold * std_val)

    if outlier_count == 0:
        return None

    outlier_vals = vals[outlier_mask]
    min_out = float(np.min(outlier_vals))
    max_out = float(np.max(outlier_vals))

    return OutlierSummary(
        column_name=col_name,
        method=f"Z-Score (|z| > {threshold:.1f})",
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        lower_bound=round(lower_bound, 3),
        upper_bound=round(upper_bound, 3),
        min_outlier=round(min_out, 3),
        max_outlier=round(max_out, 3),
        explanation=(
            f"Found {outlier_count:,} values ({outlier_pct}%) more than {threshold} standard deviations from mean."
        ),
        assumptions="Assumes approximately normal distribution. Can be biased by extreme outliers inflating std.",
    )


def analyze_data_quality(df: pd.DataFrame, profile: DatasetProfile) -> QualityReport:
    """Perform a comprehensive, rules-driven data quality audit."""
    issues: list[QualityIssue] = []
    total_rows = len(df)

    if total_rows == 0:
        return QualityReport(
            overall_health_score=0,
            health_status="Needs Attention",
            total_issues_count=1,
            critical_issues_count=1,
            warning_issues_count=0,
            info_issues_count=0,
            issues=[
                QualityIssue(
                    name="Empty Dataset",
                    category="Completeness",
                    severity="Critical",
                    affected_columns=[],
                    affected_count=0,
                    affected_percentage=100.0,
                    explanation="The loaded dataset contains zero rows of data.",
                    suggested_next_step="Upload a file that contains records below the header row.",
                )
            ],
            iqr_outliers={},
            zscore_outliers={},
        )

    # 1. Missingness checks
    for col_name, col_prof in profile.column_profiles.items():
        if col_prof.null_count > 0:
            null_pct = col_prof.null_percentage
            if null_pct >= 60.0:
                issues.append(
                    QualityIssue(
                        name="Severe Missing Values",
                        category="Completeness",
                        severity="Critical",
                        affected_columns=[col_name],
                        affected_count=col_prof.null_count,
                        affected_percentage=null_pct,
                        explanation=(
                            f"Column '{col_name}' has {col_prof.null_count:,} missing values "
                            f"({null_pct:.1f}% of column). Over half the data is absent."
                        ),
                        suggested_next_step=(
                            "Assess whether this feature can be imputed safely or if it should be excluded "
                            "from downstream modeling."
                        ),
                    )
                )
            elif null_pct >= 15.0:
                issues.append(
                    QualityIssue(
                        name="Moderate Missing Values",
                        category="Completeness",
                        severity="Warning",
                        affected_columns=[col_name],
                        affected_count=col_prof.null_count,
                        affected_percentage=null_pct,
                        explanation=(
                            f"Column '{col_name}' has {col_prof.null_count:,} missing values ({null_pct:.1f}%)."
                        ),
                        suggested_next_step="Evaluate missingness mechanism (MCAR, MAR, MNAR) and apply appropriate imputation.",
                    )
                )
            else:
                issues.append(
                    QualityIssue(
                        name="Minor Missing Values",
                        category="Completeness",
                        severity="Informational",
                        affected_columns=[col_name],
                        affected_count=col_prof.null_count,
                        affected_percentage=null_pct,
                        explanation=f"Column '{col_name}' has {col_prof.null_count:,} missing values ({null_pct:.1f}%).",
                        suggested_next_step="Apply simple imputation (mean/median/mode) or drop missing rows if sample size permits.",
                    )
                )

    # 2. Duplicate rows check
    if profile.duplicate_rows_count > 0:
        dup_pct = profile.duplicate_rows_percentage
        severity = "Critical" if dup_pct > 25.0 else ("Warning" if dup_pct > 2.0 else "Informational")
        issues.append(
            QualityIssue(
                name="Duplicate Rows Detected",
                category="Uniqueness",
                severity=severity,
                affected_columns=list(df.columns),
                affected_count=profile.duplicate_rows_count,
                affected_percentage=dup_pct,
                explanation=(
                    f"Found {profile.duplicate_rows_count:,} completely identical rows ({dup_pct:.1f}% of dataset)."
                ),
                suggested_next_step="Investigate whether duplicates represent repeated events or ingestion duplication, then deduplicate.",
            )
        )

    # 3. Constant and zero-variance columns
    for col_name, col_prof in profile.column_profiles.items():
        if col_prof.is_all_null:
            issues.append(
                QualityIssue(
                    name="All-Null Column",
                    category="Validity",
                    severity="Critical",
                    affected_columns=[col_name],
                    affected_count=total_rows,
                    affected_percentage=100.0,
                    explanation=f"Column '{col_name}' contains no non-null observations (100% missing).",
                    suggested_next_step="Drop this column as it contains no analytical information.",
                )
            )
        elif col_prof.is_constant:
            issues.append(
                QualityIssue(
                    name="Constant Column",
                    category="Consistency",
                    severity="Warning",
                    affected_columns=[col_name],
                    affected_count=total_rows,
                    affected_percentage=100.0,
                    explanation=f"Column '{col_name}' has only 1 distinct value across all rows (zero variance).",
                    suggested_next_step="Exclude from regression or classification models as constant features provide zero predictive power.",
                )
            )

    # 4. Non-finite values in numeric columns (inf, -inf)
    for col in df.select_dtypes(include=[np.number]).columns:
        col_str = str(col)
        series = df[col]
        inf_count = int(np.isinf(series.to_numpy(dtype=float, na_value=0.0)).sum())
        if inf_count > 0:
            inf_pct = round((inf_count / total_rows) * 100.0, 2)
            issues.append(
                QualityIssue(
                    name="Non-Finite Numeric Values",
                    category="Validity",
                    severity="Warning",
                    affected_columns=[col_str],
                    affected_count=inf_count,
                    affected_percentage=inf_pct,
                    explanation=f"Column '{col_str}' contains {inf_count:,} infinite (+Inf / -Inf) values.",
                    suggested_next_step="Inspect division operations or replace infinite values with NaN or bounds.",
                )
            )

    # 5. Empty and whitespace-only strings
    for col_name, col_prof in profile.column_profiles.items():
        if col_prof.empty_string_count > 0 or col_prof.whitespace_only_count > 0:
            blank_count = col_prof.empty_string_count + col_prof.whitespace_only_count
            blank_pct = round((blank_count / max(1, col_prof.non_null_count)) * 100.0, 2)
            issues.append(
                QualityIssue(
                    name="Blank / Whitespace Strings",
                    category="Consistency",
                    severity="Warning",
                    affected_columns=[col_name],
                    affected_count=blank_count,
                    affected_percentage=blank_pct,
                    explanation=(
                        f"Column '{col_name}' contains {blank_count:,} empty or whitespace-only text values "
                        "that may mask missingness."
                    ),
                    suggested_next_step="Strip whitespace and convert empty strings to formal NaN values.",
                )
            )

    # 6. High Cardinality Categoricals
    for col_name in profile.high_cardinality_columns:
        col_prof = profile.column_profiles[col_name]
        issues.append(
            QualityIssue(
                name="High Cardinality Category",
                category="Consistency",
                severity="Informational",
                affected_columns=[col_name],
                affected_count=col_prof.distinct_count,
                affected_percentage=round(col_prof.uniqueness_ratio * 100.0, 1),
                explanation=(
                    f"Categorical column '{col_name}' has {col_prof.distinct_count:,} distinct categories "
                    f"({col_prof.uniqueness_ratio:.1%} uniqueness ratio)."
                ),
                suggested_next_step="Group infrequent levels into an 'Other' category or consider target encoding.",
            )
        )

    # 7. Outlier Detection (IQR & Z-score)
    iqr_outliers: dict[str, OutlierSummary] = {}
    zscore_outliers: dict[str, OutlierSummary] = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        col_str = str(col)
        col_prof = profile.column_profiles.get(col_str)
        if col_prof and (col_prof.is_constant or col_prof.is_all_null):
            continue

        iqr_res = detect_iqr_outliers(df[col], col_str)
        if iqr_res:
            iqr_outliers[col_str] = iqr_res
            sev = "Warning" if iqr_res.outlier_percentage > 5.0 else "Informational"
            issues.append(
                QualityIssue(
                    name="Potential Outliers (IQR Method)",
                    category="Anomalies",
                    severity=sev,
                    affected_columns=[col_str],
                    affected_count=iqr_res.outlier_count,
                    affected_percentage=iqr_res.outlier_percentage,
                    explanation=(
                        f"Column '{col_str}' has {iqr_res.outlier_count:,} points ({iqr_res.outlier_percentage}%) "
                        f"outside 1.5x IQR boundaries [{iqr_res.lower_bound}, {iqr_res.upper_bound}]."
                    ),
                    suggested_next_step=(
                        "Inspect extreme values to verify if they are legitimate natural observations or entry errors; "
                        "do not delete blindly."
                    ),
                )
            )

        z_res = detect_zscore_outliers(df[col], col_str)
        if z_res:
            zscore_outliers[col_str] = z_res

    # Calculate overall health score (100 base, deductions for critical/warning/info)
    crit_count = sum(1 for i in issues if i.severity == "Critical")
    warn_count = sum(1 for i in issues if i.severity == "Warning")
    info_count = sum(1 for i in issues if i.severity == "Informational")

    deduction = (crit_count * 15) + (warn_count * 5) + (info_count * 1)
    health_score = max(5, 100 - deduction)

    if health_score >= 85:
        health_status = "Excellent"
    elif health_score >= 70:
        health_status = "Good"
    elif health_score >= 50:
        health_status = "Fair"
    else:
        health_status = "Needs Attention"

    return QualityReport(
        overall_health_score=health_score,
        health_status=health_status,
        total_issues_count=len(issues),
        critical_issues_count=crit_count,
        warning_issues_count=warn_count,
        info_issues_count=info_count,
        issues=issues,
        iqr_outliers=iqr_outliers,
        zscore_outliers=zscore_outliers,
    )
