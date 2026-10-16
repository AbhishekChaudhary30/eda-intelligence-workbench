"""Tests for Automatic First-Look Analysis Engine."""

from __future__ import annotations

import pandas as pd

from core.data_quality import analyze_data_quality
from core.first_look import run_first_look_analysis
from core.profiling import compute_dataset_profile


def test_first_look_analysis_output(mixed_dataset_df: pd.DataFrame) -> None:
    """Verify first-look analysis generates all expected categories of deterministic findings."""
    profile = compute_dataset_profile(mixed_dataset_df)
    quality = analyze_data_quality(mixed_dataset_df, profile)

    report = run_first_look_analysis(mixed_dataset_df, profile, quality)

    assert len(report.facts) >= 2
    assert len(report.quality_highlights) >= 1
    assert len(report.next_steps) >= 1
    assert len(report.limitations) >= 1

    # Check that candidate visualizations were identified
    cand = report.visual_candidates
    assert cand.primary_numeric_col is not None
    assert cand.primary_categorical_col is not None


def test_first_look_on_small_dataset_warnings() -> None:
    """Verify small sample size triggers statistical caution limitation."""
    tiny_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    profile = compute_dataset_profile(tiny_df)
    quality = analyze_data_quality(tiny_df, profile)

    report = run_first_look_analysis(tiny_df, profile, quality)
    limitation_text = " ".join([f"{lim.headline}: {lim.detail}" for lim in report.limitations])
    assert "Small Sample Size" in limitation_text or "N < 50" in limitation_text
