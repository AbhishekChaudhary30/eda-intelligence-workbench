"""Export engine for EDA Intelligence Workbench.

Provides robust CSV and machine-readable JSON serializations of dataset profiles,
data-quality audit findings, descriptive statistics, and correlation results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from core.data_quality import QualityReport
from core.first_look import FirstLookReport
from core.ingestion import DatasetMetadata
from core.profiling import DatasetProfile
from core.statistics import compute_descriptive_stats_table
from utils.security import sanitize_dataframe_for_spreadsheet


def export_dataset_as_csv(df: pd.DataFrame, sanitize_formulas: bool = True) -> bytes:
    """Serialize the working dataset to CSV format encoded in UTF-8.

    Args:
        df: The pandas DataFrame to export.
        sanitize_formulas: If True, neutralizes spreadsheet formula injection risks
            for text cells starting with =, +, -, @, \\t, or \\r.
    """
    out_df = sanitize_dataframe_for_spreadsheet(df) if sanitize_formulas else df
    return out_df.to_csv(index=False).encode("utf-8")


def export_profile_summary_as_csv(profile: DatasetProfile) -> bytes:
    """Generate and serialize a tabular per-column profile summary as CSV."""
    records = []
    for col_name, p in profile.column_profiles.items():
        records.append({
            "Column Name": col_name,
            "Pandas Dtype": p.dtype_str,
            "Semantic Type": p.semantic_type,
            "Total Count": p.total_count,
            "Non-Null Count": p.non_null_count,
            "Null Count": p.null_count,
            "Null Percentage": f"{p.null_percentage:.2f}%",
            "Distinct Values": p.distinct_count,
            "Uniqueness Ratio": f"{p.uniqueness_ratio:.4f}",
            "Memory (Bytes)": p.memory_bytes,
            "Is Constant": "Yes" if p.is_constant else "No",
            "Is All Null": "Yes" if p.is_all_null else "No",
            "Is Potential Identifier": "Yes" if p.is_potential_identifier else "No",
            "Mean": p.mean if p.mean is not None else "",
            "Std Dev": p.std if p.std is not None else "",
            "Min": p.min_val if p.min_val is not None else "",
            "Median": p.median if p.median is not None else "",
            "Max": p.max_val if p.max_val is not None else "",
            "IQR": p.iqr if p.iqr is not None else "",
            "Skewness": p.skewness if p.skewness is not None else "",
            "Kurtosis": p.kurtosis if p.kurtosis is not None else "",
        })
    df_prof = sanitize_dataframe_for_spreadsheet(pd.DataFrame(records))
    return df_prof.to_csv(index=False).encode("utf-8")


def export_quality_issues_as_csv(quality_report: QualityReport) -> bytes:
    """Serialize detected data quality issues as CSV."""
    records = []
    for issue in quality_report.issues:
        records.append({
            "Issue Name": issue.name,
            "Category": issue.category,
            "Severity": issue.severity,
            "Affected Columns": ", ".join(issue.affected_columns),
            "Affected Count": issue.affected_count,
            "Affected Percentage": f"{issue.affected_percentage:.2f}%",
            "Explanation": issue.explanation,
            "Suggested Next Step": issue.suggested_next_step,
        })
    df_issues = sanitize_dataframe_for_spreadsheet(pd.DataFrame(records))
    return df_issues.to_csv(index=False).encode("utf-8")


def export_descriptive_stats_as_csv(df: pd.DataFrame, profile: DatasetProfile) -> bytes:
    """Serialize descriptive statistics table as CSV."""
    stats_df = sanitize_dataframe_for_spreadsheet(compute_descriptive_stats_table(df, profile))
    return stats_df.to_csv(index=False).encode("utf-8")


def export_json_audit_summary(
    metadata: DatasetMetadata,
    profile: DatasetProfile,
    quality_report: QualityReport,
    first_look: FirstLookReport | None = None,
) -> str:
    """Generate a clean, structured JSON audit summary of all computed findings."""
    payload: dict[str, Any] = {
        "workbench_version": "1.0.0",
        "dataset_metadata": {
            "filename": metadata.filename,
            "format": metadata.file_format,
            "original_rows": metadata.original_rows,
            "original_cols": metadata.original_cols,
            "loaded_rows": metadata.loaded_rows,
            "loaded_cols": metadata.loaded_cols,
            "is_sampled": metadata.is_sampled,
            "sampling_explanation": metadata.sampling_explanation,
            "encoding": metadata.encoding_used,
            "delimiter": metadata.delimiter_used,
        },
        "structure_summary": {
            "total_cells": profile.total_cells,
            "total_missing_cells": profile.total_missing_cells,
            "missing_cells_percentage": profile.missing_cells_percentage,
            "complete_rows_count": profile.complete_rows_count,
            "complete_rows_percentage": profile.complete_rows_percentage,
            "duplicate_rows_count": profile.duplicate_rows_count,
            "duplicate_rows_percentage": profile.duplicate_rows_percentage,
            "numeric_columns_count": profile.numeric_columns_count,
            "categorical_columns_count": profile.categorical_columns_count,
            "datetime_columns_count": profile.datetime_columns_count,
            "boolean_columns_count": profile.boolean_columns_count,
            "constant_columns": profile.constant_columns,
            "all_null_columns": profile.all_null_columns,
            "potential_identifiers": profile.potential_identifiers,
        },
        "quality_audit": {
            "health_score": quality_report.overall_health_score,
            "health_status": quality_report.health_status,
            "critical_issues_count": quality_report.critical_issues_count,
            "warning_issues_count": quality_report.warning_issues_count,
            "info_issues_count": quality_report.info_issues_count,
            "issues": [asdict(i) for i in quality_report.issues],
        },
    }

    if first_look:
        payload["first_look_findings"] = {
            "facts": [asdict(f) for f in first_look.facts],
            "quality_highlights": [asdict(q) for q in first_look.quality_highlights],
            "relationships": [asdict(r) for r in first_look.relationships],
            "distribution_insights": [asdict(d) for d in first_look.distribution_insights],
            "next_steps": [asdict(n) for n in first_look.next_steps],
            "limitations": [asdict(l) for l in first_look.limitations],
        }

    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_correlation_matrix_as_csv(corr_matrix: pd.DataFrame) -> bytes:
    """Serialize correlation coefficient matrix as CSV bytes."""
    return corr_matrix.to_csv(index=True).encode("utf-8")


def export_ranked_pairs_as_csv(ranked_pairs: list[dict[str, Any]]) -> bytes:
    """Serialize ranked correlation pairs table as CSV bytes."""
    df_pairs = sanitize_dataframe_for_spreadsheet(pd.DataFrame(ranked_pairs))
    return df_pairs.to_csv(index=False).encode("utf-8")


def export_outlier_summary_as_csv(outlier_df: pd.DataFrame) -> bytes:
    """Serialize outlier analysis summary table as CSV bytes."""
    clean_outliers = sanitize_dataframe_for_spreadsheet(outlier_df)
    return clean_outliers.to_csv(index=False).encode("utf-8")


def export_transformation_history_as_json(steps: list[Any]) -> str:
    """Serialize transformation history steps to JSON."""
    records = []
    for s in steps:
        records.append({
            "step_id": getattr(s, "step_id", None),
            "operation": getattr(s, "operation", ""),
            "description": getattr(s, "description", ""),
            "parameters": getattr(s, "parameters", {}),
            "rows_affected": getattr(s, "rows_affected", 0),
            "cols_affected": getattr(s, "cols_affected", 0),
            "timestamp": s.timestamp.isoformat() if hasattr(s, "timestamp") else "",
        })
    return json.dumps(records, indent=2, ensure_ascii=False)


def export_transformation_history_as_csv(steps: list[Any]) -> bytes:
    """Serialize transformation history steps to CSV bytes."""
    records = []
    for s in steps:
        records.append({
            "Step": getattr(s, "step_id", None),
            "Operation": getattr(s, "operation", ""),
            "Description": getattr(s, "description", ""),
            "Rows Affected": getattr(s, "rows_affected", 0),
            "Cols Affected": getattr(s, "cols_affected", 0),
            "Timestamp": s.timestamp.isoformat() if hasattr(s, "timestamp") else "",
        })
    df_steps = sanitize_dataframe_for_spreadsheet(pd.DataFrame(records))
    return df_steps.to_csv(index=False).encode("utf-8")


def export_insights_as_json(insights: list[Any]) -> str:
    """Serialize analytical insights to JSON."""
    records = []
    for ins in insights:
        records.append({
            "insight_id": getattr(ins, "insight_id", ""),
            "category": getattr(ins, "category", ""),
            "severity": getattr(ins, "severity", ""),
            "title": getattr(ins, "title", ""),
            "description": getattr(ins, "description", ""),
            "evidence": getattr(ins, "evidence", {}),
            "affected_columns": getattr(ins, "affected_columns", []),
            "recommended_action": getattr(ins, "recommended_action", ""),
            "limitations": getattr(ins, "limitations", ""),
            "priority_score": getattr(ins, "priority_score", 0.0),
        })
    return json.dumps(records, indent=2, ensure_ascii=False)

