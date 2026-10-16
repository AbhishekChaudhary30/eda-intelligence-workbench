"""Advanced outlier investigation engine for EDA Intelligence Workbench.

Provides multi-method outlier detection:
1. IQR Tukey's Fences (configurable multiplier)
2. Standard Z-Score (configurable threshold)
3. Median Absolute Deviation (MAD) / Modified Z-score (Iglewicz & Hoaglin)
along with flagged record extraction and explicit assumption documentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnOutlierAudit:
    """Detailed outlier audit result for a single numeric column."""

    column_name: str
    method: str
    threshold_param: float
    outlier_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float
    min_extreme: float | None = None
    max_extreme: float | None = None
    flagged_indices: list[int] = field(default_factory=list)
    assumptions: str = ""
    interpretation: str = ""

    @property
    def upper_threshold(self) -> float:
        """Alias for upper_bound."""
        return self.upper_bound

    @property
    def lower_threshold(self) -> float:
        """Alias for lower_bound."""
        return self.lower_bound


@dataclass
class OutlierAnalysisReport:
    """Consolidated outlier audit report across all numeric features."""

    method_used: str
    total_numeric_cols_evaluated: int
    cols_with_outliers_count: int
    total_outliers_flagged: int
    column_audits: dict[str, ColumnOutlierAudit]
    summary_table: pd.DataFrame
    flagged_records_df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    guidance: str = (
        "Statistically unusual values (outliers) may represent genuine natural extremes, "
        "rare events, or recording errors. Do NOT automatically delete outliers without independent domain evidence."
    )


def audit_column_outliers_iqr(
    series: pd.Series,
    col_name: str,
    multiplier: float = 1.5,
) -> ColumnOutlierAudit | None:
    """Detect outliers using Tukey's IQR rule: [Q1 - k*IQR, Q3 + k*IQR]."""
    clean = series.dropna()
    finite_mask = np.isfinite(clean.to_numpy(dtype=float))
    clean_finite = clean[finite_mask]
    n = len(clean_finite)

    if n < 5:
        return None

    vals = clean_finite.to_numpy(dtype=float)
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr = q75 - q25

    if iqr == 0:
        return None

    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr

    is_outlier = (vals < lower_bound) | (vals > upper_bound)
    outlier_count = int(np.sum(is_outlier))
    outlier_pct = round((outlier_count / n) * 100.0, 2)

    flagged_idx = list(clean_finite.index[is_outlier])
    min_ext = float(np.min(vals[is_outlier])) if outlier_count > 0 else None
    max_ext = float(np.max(vals[is_outlier])) if outlier_count > 0 else None

    return ColumnOutlierAudit(
        column_name=col_name,
        method=f"IQR (k={multiplier:.1f})",
        threshold_param=multiplier,
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        lower_bound=round(lower_bound, 3),
        upper_bound=round(upper_bound, 3),
        min_extreme=round(min_ext, 3) if min_ext is not None else None,
        max_extreme=round(max_ext, 3) if max_ext is not None else None,
        flagged_indices=flagged_idx,
        assumptions="Distribution-agnostic. Resistant to heavy skewness.",
        interpretation=(
            f"{outlier_count:,} observations ({outlier_pct}%) fall outside "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]."
        ),
    )


def audit_column_outliers_zscore(
    series: pd.Series,
    col_name: str,
    threshold: float = 3.0,
) -> ColumnOutlierAudit | None:
    """Detect outliers using standard Z-score rule: |z| > threshold."""
    clean = series.dropna()
    finite_mask = np.isfinite(clean.to_numpy(dtype=float))
    clean_finite = clean[finite_mask]
    n = len(clean_finite)

    if n < 10:
        return None

    vals = clean_finite.to_numpy(dtype=float)
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals, ddof=1))

    if std_v == 0.0 or math.isnan(std_v):
        return None

    z_scores = np.abs((vals - mean_v) / std_v)
    is_outlier = z_scores > threshold
    outlier_count = int(np.sum(is_outlier))
    outlier_pct = round((outlier_count / n) * 100.0, 2)

    lower_bound = mean_v - threshold * std_v
    upper_bound = mean_v + threshold * std_v

    flagged_idx = list(clean_finite.index[is_outlier])
    min_ext = float(np.min(vals[is_outlier])) if outlier_count > 0 else None
    max_ext = float(np.max(vals[is_outlier])) if outlier_count > 0 else None

    return ColumnOutlierAudit(
        column_name=col_name,
        method=f"Z-Score (|z| > {threshold:.1f})",
        threshold_param=threshold,
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        lower_bound=round(lower_bound, 3),
        upper_bound=round(upper_bound, 3),
        min_extreme=round(min_ext, 3) if min_ext is not None else None,
        max_extreme=round(max_ext, 3) if max_ext is not None else None,
        flagged_indices=flagged_idx,
        assumptions="Assumes approximately normal distribution. Can be distorted by extreme outliers inflating std.",
        interpretation=(
            f"{outlier_count:,} observations ({outlier_pct}%) are beyond {threshold} std deviations from mean."
        ),
    )


def audit_column_outliers_mad(
    series: pd.Series,
    col_name: str,
    threshold: float = 3.5,
) -> ColumnOutlierAudit | None:
    """Detect outliers using Median Absolute Deviation (MAD) / Modified Z-score.

    Modified Z_i = 0.6745 * (x_i - median) / MAD
    Outliers defined as |Modified Z_i| > threshold (typically 3.5).
    """
    clean = series.dropna()
    finite_mask = np.isfinite(clean.to_numpy(dtype=float))
    clean_finite = clean[finite_mask]
    n = len(clean_finite)

    if n < 10:
        return None

    vals = clean_finite.to_numpy(dtype=float)
    median_v = float(np.median(vals))
    deviations = np.abs(vals - median_v)
    mad = float(np.median(deviations))

    if mad == 0.0 or math.isnan(mad):
        # Alternative: use mean absolute deviation or return None
        return None

    mod_z = 0.6745 * deviations / mad
    is_outlier = mod_z > threshold
    outlier_count = int(np.sum(is_outlier))
    outlier_pct = round((outlier_count / n) * 100.0, 2)

    lower_bound = median_v - (threshold * mad / 0.6745)
    upper_bound = median_v + (threshold * mad / 0.6745)

    flagged_idx = list(clean_finite.index[is_outlier])
    min_ext = float(np.min(vals[is_outlier])) if outlier_count > 0 else None
    max_ext = float(np.max(vals[is_outlier])) if outlier_count > 0 else None

    return ColumnOutlierAudit(
        column_name=col_name,
        method=f"MAD (Modified Z > {threshold:.1f})",
        threshold_param=threshold,
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        lower_bound=round(lower_bound, 3),
        upper_bound=round(upper_bound, 3),
        min_extreme=round(min_ext, 3) if min_ext is not None else None,
        max_extreme=round(max_ext, 3) if max_ext is not None else None,
        flagged_indices=flagged_idx,
        assumptions="Robust non-parametric method based on median absolute deviation. Resilient to multiple outliers.",
        interpretation=f"{outlier_count:,} observations ({outlier_pct}%) exceed MAD cutoff ({threshold}).",
    )


def run_comprehensive_outlier_analysis(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
    columns: list[str] | None = None,
) -> OutlierAnalysisReport:
    """Run outlier audit across numeric columns using selected method."""
    method_clean = method.lower()
    audits: dict[str, ColumnOutlierAudit] = {}
    if columns is not None:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    all_flagged_indices: set[int] = set()

    for col in numeric_cols:
        col_str = str(col)
        audit_res: ColumnOutlierAudit | None = None
        if method_clean == "zscore":
            audit_res = audit_column_outliers_zscore(df[col], col_str, threshold=threshold)
        elif method_clean == "mad":
            audit_res = audit_column_outliers_mad(df[col], col_str, threshold=threshold)
        else:  # iqr default
            audit_res = audit_column_outliers_iqr(df[col], col_str, multiplier=threshold)

        if audit_res:
            audits[col_str] = audit_res
            if audit_res.outlier_count > 0:
                all_flagged_indices.update(audit_res.flagged_indices)

    # Build summary table
    summary_records = []
    for col_str, audit in audits.items():
        summary_records.append({
            "Feature": col_str,
            "Column_Name": col_str,
            "Outlier Count": audit.outlier_count,
            "Outlier_Count": audit.outlier_count,
            "Outlier %": f"{audit.outlier_percentage:.2f}%",
            "Outlier_Percentage": audit.outlier_percentage,
            "Lower Bound": audit.lower_bound,
            "Lower_Bound": audit.lower_bound,
            "Upper Bound": audit.upper_bound,
            "Upper_Bound": audit.upper_bound,
            "Min Extreme": audit.min_extreme if audit.min_extreme is not None else np.nan,
            "Min_Extreme": audit.min_extreme if audit.min_extreme is not None else np.nan,
            "Max Extreme": audit.max_extreme if audit.max_extreme is not None else np.nan,
            "Max_Extreme": audit.max_extreme if audit.max_extreme is not None else np.nan,
            "Method": audit.method,
        })

    summary_df = (
        pd.DataFrame(summary_records).sort_values("Outlier Count", ascending=False)
        if summary_records
        else pd.DataFrame()
    )

    # Flagged records dataframe slice
    if all_flagged_indices:
        sorted_indices = sorted(all_flagged_indices)[:200]  # Cap at 200 for preview
        flagged_df = df.loc[sorted_indices].copy()
    else:
        flagged_df = pd.DataFrame()

    total_flagged = sum(a.outlier_count for a in audits.values())
    cols_with_out = sum(1 for a in audits.values() if a.outlier_count > 0)

    return OutlierAnalysisReport(
        method_used=method_clean.upper(),
        total_numeric_cols_evaluated=len(numeric_cols),
        cols_with_outliers_count=cols_with_out,
        total_outliers_flagged=total_flagged,
        column_audits=audits,
        summary_table=summary_df,
        flagged_records_df=flagged_df,
    )


def detect_iqr_outliers(series: pd.Series, k: float = 1.5) -> ColumnOutlierAudit:
    """Helper for single-series IQR outlier detection."""
    col_name = str(series.name or "feature")
    res = audit_column_outliers_iqr(series, col_name, multiplier=k)
    return res or ColumnOutlierAudit(
        column_name=col_name,
        method=f"IQR (k={k})",
        threshold_param=k,
        outlier_count=0,
        outlier_percentage=0.0,
        lower_bound=0.0,
        upper_bound=0.0,
    )


def detect_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> ColumnOutlierAudit:
    """Helper for single-series Z-score outlier detection."""
    col_name = str(series.name or "feature")
    res = audit_column_outliers_zscore(series, col_name, threshold=threshold)
    return res or ColumnOutlierAudit(
        column_name=col_name,
        method=f"Z-Score ({threshold})",
        threshold_param=threshold,
        outlier_count=0,
        outlier_percentage=0.0,
        lower_bound=0.0,
        upper_bound=0.0,
    )


def detect_mad_outliers(series: pd.Series, threshold: float = 3.5) -> ColumnOutlierAudit:
    """Helper for single-series MAD modified Z-score outlier detection."""
    col_name = str(series.name or "feature")
    res = audit_column_outliers_mad(series, col_name, threshold=threshold)
    return res or ColumnOutlierAudit(
        column_name=col_name,
        method=f"MAD ({threshold})",
        threshold_param=threshold,
        outlier_count=0,
        outlier_percentage=0.0,
        lower_bound=0.0,
        upper_bound=0.0,
    )


def analyze_all_numeric_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    param_value: float = 1.5,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Batch outlier audit returning summary DataFrame across numeric features."""
    report = run_comprehensive_outlier_analysis(
        df, method=method, threshold=param_value, columns=columns
    )
    return report.summary_table

