"""Consolidated HTML Report Generation Engine for EDA Intelligence Workbench.

Generates self-contained, beautifully styled, secure HTML analytics audit reports.
All user-supplied column names and values are strictly escaped with html.escape
to prevent XSS, and no external telemetry or paid APIs are invoked.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from core.data_quality import QualityReport
from core.ingestion import DatasetMetadata
from core.insights import AnalyticalInsight
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings
from core.transformations import TransformationStep


def generate_consolidated_html_report(
    metadata: DatasetMetadata,
    profile: DatasetProfile,
    quality_report: QualityReport,
    insights: list[AnalyticalInsight],
    settings: AnalysisSettings,
    stats_df: pd.DataFrame | None = None,
    outlier_df: pd.DataFrame | None = None,
    ranked_pairs: list[dict[str, Any]] | None = None,
    transformations: list[TransformationStep] | None = None,
) -> str:
    """Produce a standalone, executive-ready HTML audit report."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Safe escaping helpers
    safe_filename = html.escape(metadata.filename)
    safe_format = html.escape(metadata.file_format.upper())

    # Build KPI metrics
    quality_score = getattr(quality_report, "overall_health_score", getattr(quality_report, "overall_score", 100))
    quality_color = "#10b981" if quality_score >= 80 else ("#f59e0b" if quality_score >= 60 else "#f43f5e")

    # Format insights HTML
    insights_html = ""
    if insights:
        for idx, ins in enumerate(insights[:20], start=1):
            sev_color = {
                "critical": "#f43f5e",
                "warning": "#f59e0b",
                "info": "#3b82f6",
                "opportunity": "#10b981",
            }.get(ins.severity, "#94a3b8")

            cols_str = html.escape(", ".join(ins.affected_columns)) if ins.affected_columns else "Dataset-level"
            desc_str = html.escape(ins.description)
            rec_str = html.escape(ins.recommended_action)
            evidence_str = html.escape(str(ins.evidence))
            limit_str = html.escape(ins.limitations)

            insights_html += f"""
            <div class="insight-card" style="border-left: 4px solid {sev_color};">
                <div class="insight-header">
                    <span class="insight-badge" style="background: {sev_color}22; color: {sev_color};">
                        {html.escape(ins.category.upper())}
                    </span>
                    <span class="insight-title">#{idx}. {html.escape(ins.title)}</span>
                    <span class="priority-score">Priority: {ins.priority_score:.1f}</span>
                </div>
                <div class="insight-body">
                    <p><strong>Description:</strong> {desc_str}</p>
                    <p><strong>Affected Feature(s):</strong> <code>{cols_str}</code></p>
                    <p><strong>Quantitative Evidence:</strong> {evidence_str}</p>
                    <p><strong>Recommended Action:</strong> {rec_str}</p>
                    <p class="limitation-text"><strong>Statistical Boundary:</strong> {limit_str}</p>
                </div>
            </div>
            """
    else:
        insights_html = "<p class='muted'>No active analytical findings generated.</p>"

    # Format stats table HTML
    stats_html = ""
    if stats_df is not None and not stats_df.empty:
        header_cols = "".join(f"<th>{html.escape(str(c))}</th>" for c in stats_df.columns)
        rows_html = ""
        for _, row in stats_df.iterrows():
            row_cells = "".join(
                f"<td>{html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</td>"
                for v in row
            )
            rows_html += f"<tr>{row_cells}</tr>"
        stats_html = f"""
        <div class="table-container">
            <table>
                <thead><tr>{header_cols}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
    else:
        stats_html = "<p class='muted'>Descriptive statistics table not provided.</p>"

    # Format outlier table HTML
    outliers_html = ""
    if outlier_df is not None and not outlier_df.empty:
        header_cols = "".join(f"<th>{html.escape(str(c))}</th>" for c in outlier_df.columns)
        rows_html = ""
        for _, row in outlier_df.iterrows():
            row_cells = "".join(
                f"<td>{html.escape(f'{v:.2f}' if isinstance(v, float) else str(v))}</td>"
                for v in row
            )
            rows_html += f"<tr>{row_cells}</tr>"
        outliers_html = f"""
        <div class="table-container">
            <table>
                <thead><tr>{header_cols}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
    else:
        outliers_html = "<p class='muted'>No outliers flagged across evaluated continuous columns.</p>"

    # Format correlation pairs HTML
    corr_html = ""
    if ranked_pairs:
        corr_rows = ""
        for pair in ranked_pairs[:10]:
            col1 = html.escape(str(pair.get("col1", "")))
            col2 = html.escape(str(pair.get("col2", "")))
            r = pair.get("r", 0.0)
            p = pair.get("p_value", 1.0)
            n = pair.get("sample_size", 0)
            sig_badge = "<span class='badge-sig'>p < 0.05</span>" if p < 0.05 else "<span class='badge-insig'>Not Sig</span>"
            corr_rows += f"""
            <tr>
                <td><strong>{col1}</strong> ↔ <strong>{col2}</strong></td>
                <td><code>{r:+.4f}</code></td>
                <td><code>{p:.4e}</code></td>
                <td>{n}</td>
                <td>{sig_badge}</td>
            </tr>
            """
        corr_html = f"""
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Feature Pair</th>
                        <th>Correlation (r)</th>
                        <th>p-value</th>
                        <th>Sample Size (N)</th>
                        <th>Significance</th>
                    </tr>
                </thead>
                <tbody>{corr_rows}</tbody>
            </table>
        </div>
        """
    else:
        corr_html = "<p class='muted'>No paired correlations evaluated.</p>"

    # Format transformations HTML
    trans_html = ""
    if transformations:
        t_rows = ""
        for step in transformations:
            ts = step.timestamp.strftime("%H:%M:%S")
            t_rows += f"""
            <tr>
                <td>#{step.step_id}</td>
                <td><code>{html.escape(step.operation)}</code></td>
                <td>{html.escape(step.description)}</td>
                <td>{step.rows_affected}</td>
                <td>{ts}</td>
            </tr>
            """
        trans_html = f"""
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Operation</th>
                        <th>Description</th>
                        <th>Rows Affected</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>{t_rows}</tbody>
            </table>
        </div>
        """
    else:
        trans_html = "<p class='muted'>Original dataset unmutated. No transformations applied.</p>"

    # Assemble HTML document
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Intelligence Workbench — Audit Report ({safe_filename})</title>
    <style>
        :root {{
            --bg-canvas: #0b0f19;
            --bg-card: #131b2e;
            --bg-elevated: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --primary: #3b82f6;
            --accent: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #f43f5e;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        .report-wrapper {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #131b2e 0%, #1a243d 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }}
        .header-brand {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        .logo-title {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .logo-title span {{ color: var(--primary); }}
        .header-meta {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        section {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 30px;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
            border-left: 3px solid var(--primary);
            padding-left: 12px;
        }}
        .table-container {{
            overflow-x: auto;
            margin-top: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-subtle);
        }}
        th {{
            background: var(--bg-elevated);
            color: var(--text-secondary);
            font-weight: 600;
        }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        code {{
            background: rgba(255, 255, 255, 0.06);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
        .insight-card {{
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }}
        .insight-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }}
        .insight-badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .insight-title {{
            font-weight: 600;
            font-size: 14px;
            flex-grow: 1;
        }}
        .priority-score {{
            font-size: 12px;
            color: var(--text-secondary);
            font-family: monospace;
        }}
        .insight-body p {{
            font-size: 13px;
            margin-bottom: 6px;
            color: #cbd5e1;
        }}
        .limitation-text {{
            font-size: 12px !important;
            color: #64748b !important;
            font-style: italic;
        }}
        .badge-sig {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-insig {{
            background: rgba(148, 163, 184, 0.15);
            color: #94a3b8;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .muted {{ color: var(--text-secondary); font-size: 13px; }}
        footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-secondary);
            font-size: 12px;
            border-top: 1px solid var(--border-subtle);
        }}
    </style>
</head>
<body>
    <div class="report-wrapper">
        <header>
            <div class="header-brand">
                <div>
                    <div class="logo-title">EDA Intelligence <span>Workbench</span></div>
                    <div class="header-meta">Understand Your Data. Discover What Matters. — Phase 2 Executive Audit Report</div>
                </div>
                <div class="header-meta" style="text-align: right;">
                    <div>Target: <strong>{safe_filename}</strong> ({safe_format})</div>
                    <div>Generated: {now_str}</div>
                </div>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Observations</div>
                    <div class="kpi-value">{metadata.loaded_rows:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Features</div>
                    <div class="kpi-value">{metadata.loaded_cols:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Quality Score</div>
                    <div class="kpi-value" style="color: {quality_color};">{quality_score:.1f}/100</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Duplicate Rows</div>
                    <div class="kpi-value">{profile.duplicate_rows_count:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Quality Issues</div>
                    <div class="kpi-value">{len(quality_report.issues)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Active Insights</div>
                    <div class="kpi-value">{len(insights)}</div>
                </div>
            </div>
        </header>

        <section id="insights">
            <h2>1. Automated Analytical Insights & Findings</h2>
            <p class="muted">Ranked rule-based findings derived deterministically from calculations. No external AI APIs used.</p>
            <div style="margin-top: 16px;">
                {insights_html}
            </div>
        </section>

        <section id="descriptive-stats">
            <h2>2. Comprehensive Descriptive Statistics</h2>
            <p class="muted">Parametric and non-parametric central tendencies, dispersion, and higher distributional moments.</p>
            {stats_html}
        </section>

        <section id="correlations">
            <h2>3. Significant Correlation Highlights</h2>
            <p class="muted">Highest pairwise correlations evaluated under hypothesis testing. Note: Correlation does not establish causation.</p>
            {corr_html}
        </section>

        <section id="outliers">
            <h2>4. Statistical Outlier Audit</h2>
            <p class="muted">Multi-method boundary evaluation (IQR, Z-Score, and MAD). Flags statistically unusual observations without asserting defectiveness.</p>
            {outliers_html}
        </section>

        <section id="transformations">
            <h2>5. Transformation Audit Trail</h2>
            <p class="muted">Non-destructive lineage of all cleaning, imputation, filtering, and scaling operations applied to the working dataset.</p>
            {trans_html}
        </section>

        <section id="governance">
            <h2>6. Analytical Governance & Disclaimers</h2>
            <p class="muted">
                This report represents exploratory data analysis (EDA). Observational associations, normality test failures,
                and outlier markers serve as exploratory diagnostic signals rather than confirmatory or causal proof.
                P-values are susceptible to sample size scaling and multiple-comparison inflation unless specifically adjusted.
            </p>
        </section>

        <footer>
            EDA Intelligence Workbench • Built with Streamlit, Plotly, SciPy, Statsmodels & scikit-learn • 100% Local & Deterministic
        </footer>
    </div>
</body>
</html>
"""
