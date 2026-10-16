"""HTML-safe metric cards, health score banners, and badge components.

Follows the dark analytics design system with zero unescaped user string injection.
"""

from __future__ import annotations

import streamlit as st

from core.data_quality import QualityIssue
from core.first_look import ObservationItem
from utils.security import escape_html


def render_header(
    title: str = "EDA Intelligence Workbench",
    tagline: str = "Understand Your Data. Discover What Matters.",
    status_text: str | None = None,
) -> None:
    """Render the top banner header with product identity and status indicator."""
    status_html = ""
    if status_text:
        status_html = f'<span class="workbench-badge-primary">{escape_html(status_text)}</span>'

    html_content = f"""
    <div class="workbench-header">
        <div class="workbench-title-row">
            <div>
                <h1 class="workbench-title">
                    <span>📊 {escape_html(title)}</span>
                </h1>
                <p class="workbench-tagline">{escape_html(tagline)}</p>
            </div>
            <div>{status_html}</div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    badge_text: str | None = None,
    badge_type: str = "neutral",  # "critical", "warning", "info", "success", "neutral"
) -> str:
    """Generate HTML string for a single KPI metric card."""
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="badge badge-{badge_type}" style="margin-top: 4px;">{escape_html(badge_text)}</span>'

    return f"""
    <div class="metric-card">
        <div class="metric-label">{escape_html(label)}</div>
        <div class="metric-value">{escape_html(value)}</div>
        <div class="metric-subtitle">{escape_html(subtitle)}</div>
        {badge_html}
    </div>
    """


def render_metric_grid(cards_html_list: list[str]) -> None:
    """Render a responsive grid of metric cards."""
    grid_inner = "\n".join(cards_html_list)
    html_content = f'<div class="metric-grid">{grid_inner}</div>'
    st.markdown(html_content, unsafe_allow_html=True)


def render_health_score_banner(
    score: int,
    status: str,
    critical_count: int,
    warning_count: int,
    info_count: int,
) -> None:
    """Render the prominent data health audit banner."""
    color_class = "score-high" if score >= 80 else ("score-medium" if score >= 60 else "score-low")
    html_content = f"""
    <div class="health-score-container">
        <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">
            Overall Data Health Index
        </div>
        <div class="health-score-val {color_class}">
            {score}<span style="font-size: 1.25rem; font-weight: 500; color: #64748b;"> / 100</span>
        </div>
        <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.75rem;">
            Status: {escape_html(status)}
        </div>
        <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;">
            <span class="badge badge-critical">{critical_count} Critical Issues</span>
            <span class="badge badge-warning">{warning_count} Warnings</span>
            <span class="badge badge-info">{info_count} Informational</span>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_issue_alert(issue: QualityIssue) -> None:
    """Render a structured alert box for a data quality issue."""
    badge_class = f"badge-{issue.severity.lower()}"
    cols_str = ", ".join(f"`{c}`" for c in issue.affected_columns[:5])
    if len(issue.affected_columns) > 5:
        cols_str += f" and {len(issue.affected_columns) - 5} more"

    html_content = f"""
    <div style="background-color: #111827; border: 1px solid #1f2937; border-left: 4px solid {'#ef4444' if issue.severity == 'Critical' else ('#f59e0b' if issue.severity == 'Warning' else '#38bdf8')}; border-radius: 8px; padding: 1rem; margin-bottom: 0.85rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9;">{escape_html(issue.name)}</span>
            <span class="badge {badge_class}">{escape_html(issue.severity)}</span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem;">
            <strong>Category:</strong> {escape_html(issue.category)} &nbsp;|&nbsp;
            <strong>Impact:</strong> {issue.affected_count:,} records ({issue.affected_percentage:.1f}%)
        </div>
        <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0.5rem; line-height: 1.4;">
            {escape_html(issue.explanation)}
        </div>
        <div style="font-size: 0.82rem; color: #34d399; background: rgba(16, 185, 129, 0.08); padding: 0.35rem 0.6rem; border-radius: 4px;">
            💡 <strong>Next Step:</strong> {escape_html(issue.suggested_next_step)}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_observation_callout(obs: ObservationItem) -> None:
    """Render an individual observation block with category indicator."""
    cat_class_map = {
        "Fact": "insight-box-fact",
        "Quality": "insight-box-quality",
        "Relationship": "insight-box-relationship",
        "Distribution": "insight-box",
        "NextStep": "insight-box-next-step",
        "Limitation": "insight-box-limitation",
    }
    box_class = cat_class_map.get(obs.category, "insight-box")

    html_content = f"""
    <div class="insight-box {box_class}">
        <div style="font-weight: 600; color: #f8fafc; margin-bottom: 0.2rem;">
            {escape_html(obs.headline)}
        </div>
        <div style="color: #cbd5e1;">
            {escape_html(obs.detail)}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
