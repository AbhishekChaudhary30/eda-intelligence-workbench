"""Tests for deterministic automated insight generation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from core.insights import (
    generate_automated_insights,
)
from core.profiling import compute_dataset_profile


@pytest.fixture
def insight_test_df() -> pd.DataFrame:
    """Create dataset with known quality issues, correlations, and skew."""
    return pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(50)],  # High cardinality identifier
        "income": [20.0 + i * 2.0 for i in range(20)] + [None] * 30,  # 60% missing (severe)
        "expense": [10.0 + i * 1.8 for i in range(50)],
        "skewed_metric": [1.0] * 45 + [100.0, 200.0, 300.0, 400.0, 500.0],  # Heavy right skew
        "category": ["A", "B", "A", "B", "C"] * 10,
    })


def test_insight_generation_rule_triggers(insight_test_df):
    """Verify that deterministic rules fire accurately based on calculated metrics."""
    profile = compute_dataset_profile(insight_test_df)
    catalog = generate_automated_insights(insight_test_df, profile)

    assert catalog.total_insights >= 2
    assert len(catalog.insights) >= 2

    categories_fired = {ins.category for ins in catalog.insights}
    assert "Missingness" in categories_fired

    # Verify priority score ordering (descending)
    scores = [ins.priority_score for ins in catalog.insights]
    assert scores == sorted(scores, reverse=True)


def test_insight_filtering(insight_test_df):
    """Verify filtering insights by category, severity, and column."""
    profile = compute_dataset_profile(insight_test_df)
    catalog = generate_automated_insights(insight_test_df, profile)

    # Filter by category
    miss_only = catalog.filter(category="Missingness")
    for ins in miss_only:
        assert ins.category == "Missingness"

    # Filter by column
    income_only = catalog.filter(column="income")
    for ins in income_only:
        assert "income" in ins.affected_columns
