"""Automated Insights UI component for EDA Intelligence Workbench.

Renders deterministic, rule-based analytical insights derived from actual calculations.
Provides transparent priority scoring, category and severity filtering, and JSON export.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.navigation import render_section_navigation
from core.data_quality import QualityReport
from core.insights import (
    filter_insights,
    generate_deterministic_insights,
)
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings


def render_insights_panel(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality_report: QualityReport,
    settings: AnalysisSettings,
) -> None:
    """Render the Automated Insights panel."""
    st.markdown("### 💡 Automated Analytical Insights & Findings")
    st.caption(
        "A 100% deterministic, rule-based insight engine deriving findings directly from mathematical calculations. "
        "No external generative AI APIs, zero hallucinated numbers."
    )
    render_section_navigation("Automated Insights", position="top")

    # Generate insights
    all_insights = generate_deterministic_insights(
        df=df,
        profile=profile,
        quality_report=quality_report,
        settings=settings,
    )

    if not all_insights:
        st.info("No active analytical insights triggered for the current dataset.")
        return

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)

    all_categories = sorted({ins.category for ins in all_insights})
    all_severities = ["all", "critical", "warning", "info", "opportunity"]
    all_affected_cols = sorted({c for ins in all_insights for c in ins.affected_columns})

    with f_col1:
        selected_category = st.selectbox(
            "Filter by Category",
            options=["all"] + all_categories,
            format_func=lambda c: "All Categories" if c == "all" else c.title(),
        )
    with f_col2:
        selected_severity = st.selectbox(
            "Filter by Severity",
            options=all_severities,
            format_func=lambda s: "All Severities" if s == "all" else s.capitalize(),
        )
    with f_col3:
        selected_column = st.selectbox(
            "Filter by Affected Column",
            options=["all"] + all_affected_cols,
            format_func=lambda c: "All Columns" if c == "all" else c,
        )

    # Filter
    filtered = filter_insights(
        all_insights,
        category=selected_category,
        severity=selected_severity,
        affected_column=selected_column,
    )

    # KPI summary bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Findings", f"{len(filtered)} of {len(all_insights)}")
    crit_count = sum(1 for i in all_insights if i.severity == "critical")
    warn_count = sum(1 for i in all_insights if i.severity == "warning")
    opp_count = sum(1 for i in all_insights if i.severity == "opportunity")
    m2.metric("Critical Alerts", str(crit_count), delta="Action Recommended" if crit_count > 0 else "None", delta_color="inverse")
    m3.metric("Warnings", str(warn_count))
    m4.metric("Opportunities", str(opp_count))

    st.markdown("---")

    # Render cards
    for idx, ins in enumerate(filtered, start=1):
        sev_icon = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️",
            "opportunity": "🚀",
        }.get(ins.severity, "📌")

        sev_color = {
            "critical": "#f43f5e",
            "warning": "#f59e0b",
            "info": "#3b82f6",
            "opportunity": "#10b981",
        }.get(ins.severity, "#94a3b8")

        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 4px solid {sev_color}; background-color: rgba(19, 27, 46, 0.7);
                            padding: 16px 20px; border-radius: 8px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span style="font-weight: 700; font-size: 15px; color: #f1f5f9;">
                                {sev_icon} #{idx}. {ins.title}
                            </span>
                            <span style="margin-left: 10px; background: {sev_color}22; color: {sev_color};
                                         padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">
                                {ins.category.upper()}
                            </span>
                        </div>
                        <div style="font-family: monospace; font-size: 12px; color: #94a3b8;">
                            Priority Score: <strong>{ins.priority_score:.1f}</strong>
                        </div>
                    </div>
                    <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5; margin-bottom: 8px;">
                        {ins.description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"View Evidence & Statistical Boundaries for Finding #{idx}", expanded=False):
                e1, e2 = st.columns([1, 1])
                with e1:
                    st.markdown("**Quantitative Evidence:**")
                    st.json(ins.evidence)
                    if ins.affected_columns:
                        st.markdown(f"**Affected Columns:** `{', '.join(ins.affected_columns)}`")
                    st.markdown(f"**Source Engine:** `{ins.source_analysis}`")

                with e2:
                    st.markdown(f"**Recommended Action:**\n\n{ins.recommended_action}")
                    st.markdown(f"**Statistical Limitation:**\n\n*{ins.limitations}*")

    # Export findings
    export_payload = []
    for ins in all_insights:
        export_payload.append({
            "insight_id": ins.insight_id,
            "category": ins.category,
            "severity": ins.severity,
            "title": ins.title,
            "description": ins.description,
            "priority_score": ins.priority_score,
            "evidence": ins.evidence,
            "affected_columns": ins.affected_columns,
            "recommended_action": ins.recommended_action,
            "limitations": ins.limitations,
            "source_analysis": ins.source_analysis,
        })

    json_str = json.dumps(export_payload, indent=2)
    st.download_button(
        "📥 Export All Insights (JSON)",
        data=json_str,
        file_name="analytical_insights.json",
        mime="application/json",
    )
    render_section_navigation("Automated Insights", position="bottom")
