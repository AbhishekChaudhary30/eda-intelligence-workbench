"""Tests for consolidated HTML audit report generation and XSS escaping."""

from __future__ import annotations

import pandas as pd

from core.advanced_statistics import compute_extended_descriptive_stats
from core.data_quality import analyze_data_quality
from core.ingestion import DatasetMetadata
from core.insights import generate_deterministic_insights
from core.profiling import compute_dataset_profile
from core.report_generation import generate_consolidated_html_report
from core.settings import AnalysisSettings


def test_consolidated_html_report_generation():
    """Verify that consolidated HTML report renders all sections with strict escaping."""
    # Create dataset with potentially malicious column and value to test XSS escaping
    malicious_col = "<script>alert('xss')</script>"
    df = pd.DataFrame({
        malicious_col: [10.0, 20.0, 30.0, 40.0],
        "category": ["A", "B", "A", "<img src=x onerror=alert(1)>"],
    })
    meta = DatasetMetadata(
        filename="test_dataset.csv",
        file_format="CSV",
        file_size_bytes=1024,
        original_rows=4,
        original_cols=2,
        loaded_rows=4,
        loaded_cols=2,
        is_sampled=False,
    )
    profile = compute_dataset_profile(df)
    quality = analyze_data_quality(df, profile)
    settings = AnalysisSettings()
    insights = generate_deterministic_insights(df, profile, quality, settings)
    stats_df = compute_extended_descriptive_stats(df, profile)

    html_content = generate_consolidated_html_report(
        metadata=meta,
        profile=profile,
        quality_report=quality,
        insights=insights,
        settings=settings,
        stats_df=stats_df,
    )

    # Validate HTML structure
    assert html_content.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html_content

    # Strict XSS check: unescaped script tag must NOT exist
    assert "<script>alert('xss')</script>" not in html_content
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_content or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html_content

    # Validate key sections
    assert "Executive Audit Report" in html_content
    assert "Automated Analytical Insights &amp; Findings" in html_content or "Automated Analytical Insights & Findings" in html_content
    assert "Comprehensive Descriptive Statistics" in html_content
    assert "Analytical Governance &amp; Disclaimers" in html_content or "Analytical Governance & Disclaimers" in html_content
