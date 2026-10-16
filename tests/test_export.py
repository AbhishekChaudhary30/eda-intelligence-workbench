"""Tests for Export Engine (CSV and JSON Serializations)."""

from __future__ import annotations

import json

import pandas as pd

from core.data_quality import analyze_data_quality
from core.export import (
    export_dataset_as_csv,
    export_descriptive_stats_as_csv,
    export_json_audit_summary,
    export_profile_summary_as_csv,
    export_quality_issues_as_csv,
)
from core.first_look import run_first_look_analysis
from core.ingestion import DatasetMetadata
from core.profiling import compute_dataset_profile


def test_csv_exports(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify that dataset, profile, quality, and stats serialize to valid UTF-8 CSV bytes."""
    profile = compute_dataset_profile(mixed_dataset_df)
    quality = analyze_data_quality(mixed_dataset_df, profile)

    # 1. Dataset CSV
    ds_csv = export_dataset_as_csv(mixed_dataset_df)
    assert isinstance(ds_csv, bytes)
    assert b"salary" in ds_csv

    # 2. Profile CSV
    prof_csv = export_profile_summary_as_csv(profile)
    assert isinstance(prof_csv, bytes)
    assert b"Column Name" in prof_csv

    # 3. Quality CSV
    qual_csv = export_quality_issues_as_csv(quality)
    assert isinstance(qual_csv, bytes)

    # 4. Descriptive Stats CSV
    stats_csv = export_descriptive_stats_as_csv(mixed_dataset_df, profile)
    assert isinstance(stats_csv, bytes)
    assert b"Mean" in stats_csv


def test_json_audit_summary_export(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify machine-readable JSON report serializes properly without errors."""
    meta = DatasetMetadata(
        filename="test.csv",
        file_format="CSV",
        file_size_bytes=1000,
        original_rows=20,
        original_cols=6,
        loaded_rows=20,
        loaded_cols=6,
    )
    profile = compute_dataset_profile(mixed_dataset_df)
    quality = analyze_data_quality(mixed_dataset_df, profile)
    first_look = run_first_look_analysis(mixed_dataset_df, profile, quality)

    json_str = export_json_audit_summary(meta, profile, quality, first_look)
    data = json.loads(json_str)

    assert data["workbench_version"] == "1.0.0"
    assert data["dataset_metadata"]["filename"] == "test.csv"
    assert "quality_audit" in data
    assert "structure_summary" in data
    assert "first_look_findings" in data
