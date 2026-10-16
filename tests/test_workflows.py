"""End-to-End User Journey and Workflow Integration Tests using Streamlit AppTest.

Tests complete analytical pipelines across all 15 navigation sections:
1. Overview & Getting Started
2. Upload Dataset
3. Dataset Preview
4. Dataset Profile
5. Data Quality Audit
6. Automated Insights
7. Visualization Studio
8. Statistical Analysis
9. Correlation Explorer
10. Missingness Analysis
11. Outlier Analysis
12. Transformation Workbench
13. Automatic First-Look Analysis
14. Analysis Configuration
15. Export & Download
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sample_data.loader import (
    create_customer_churn_sample,
    create_employee_analytics_sample,
)

ALL_SECTIONS: list[str] = [
    "Overview & Getting Started",
    "Upload Dataset",
    "Dataset Preview",
    "Dataset Profile",
    "Data Quality Audit",
    "Automated Insights",
    "Visualization Studio",
    "Statistical Analysis",
    "Correlation Explorer",
    "Missingness Analysis",
    "Outlier Analysis",
    "Transformation Workbench",
    "Automatic First-Look Analysis",
    "Analysis Configuration",
    "Export & Download",
]

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def base_app_test() -> AppTest:
    """Initialize a fresh headless Streamlit AppTest instance."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    return at


def test_empty_state_navigation_resilience(base_app_test: AppTest) -> None:
    """Verify that every section renders a clean empty-state without crashing when no data is loaded."""
    at = base_app_test
    assert len(at.exception) == 0

    for section in ALL_SECTIONS:
        at.session_state["selected_section"] = section
        at.run()
        assert len(at.exception) == 0, f"Section '{section}' raised an unhandled exception in empty state: {at.exception}"


def test_full_employee_analytical_workflow(base_app_test: AppTest) -> None:
    """Verify complete analytical journey with Employee Analytics dataset loaded."""
    at = base_app_test
    df = create_employee_analytics_sample()

    # Ingest dataset into session state
    at.session_state["dataset_df"] = df.copy()
    at.session_state["original_df"] = df.copy()
    at.session_state["working_df"] = df.copy()

    # Step through every navigation section in analytical sequence
    for section in ALL_SECTIONS:
        at.session_state["selected_section"] = section
        at.run()
        assert len(at.exception) == 0, f"Section '{section}' crashed during active analysis: {at.exception}"

    # Verify session state persistence
    assert at.session_state["working_df"] is not None
    assert len(at.session_state["working_df"]) == len(df)


def test_full_customer_churn_analytical_workflow(base_app_test: AppTest) -> None:
    """Verify complete analytical journey with Customer Churn dataset loaded."""
    at = base_app_test
    df = create_customer_churn_sample()

    at.session_state["dataset_df"] = df.copy()
    at.session_state["original_df"] = df.copy()
    at.session_state["working_df"] = df.copy()

    for section in ALL_SECTIONS:
        at.session_state["selected_section"] = section
        at.run()
        assert len(at.exception) == 0, f"SaaS Churn analysis failed on section '{section}': {at.exception}"
