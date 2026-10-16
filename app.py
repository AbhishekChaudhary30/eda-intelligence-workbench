"""EDA Intelligence Workbench - Main Streamlit Application Entry Point.

Tagline: Understand Your Data. Discover What Matters.
Architecture: Python 3.11+ & Streamlit, Plotly, Pandas, SciPy, PyArrow.
Security: 100% Local processing, Zero external telemetry, Safe parsing.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from components.analysis_settings import render_analysis_settings

# UI Components
from components.chart_components import (
    plot_box_by_category,
    plot_boxplot,
    plot_categorical_bar,
    plot_correlation_heatmap,
    plot_ecdf,
    plot_histogram,
    plot_pairwise_scatter,
    plot_scatter,
    plot_time_series,
    plot_treemap,
    plot_violin,
)
from components.chart_explorer import render_chart_explorer
from components.correlation_explorer import render_correlation_explorer
from components.data_table import render_data_preview
from components.insights_panel import render_insights_panel
from components.metric_cards import (
    render_header,
    render_health_score_banner,
    render_issue_alert,
    render_metric_card,
    render_metric_grid,
    render_observation_callout,
)
from components.missingness_panel import render_missingness_panel
from components.outlier_panel import render_outlier_panel
from components.sidebar import render_sidebar
from components.statistical_analysis import render_statistical_analysis
from components.transformation_workbench import render_transformation_workbench
from components.upload_panel import render_upload_panel

# Core Analytics
from core.advanced_statistics import compute_extended_descriptive_stats
from core.correlations import compute_correlation_matrix, extract_ranked_correlations
from core.data_quality import QualityReport, analyze_data_quality
from core.export import (
    export_correlation_matrix_as_csv,
    export_dataset_as_csv,
    export_descriptive_stats_as_csv,
    export_insights_as_json,
    export_json_audit_summary,
    export_outlier_summary_as_csv,
    export_profile_summary_as_csv,
    export_quality_issues_as_csv,
    export_transformation_history_as_csv,
    export_transformation_history_as_json,
)
from core.first_look import FirstLookReport, run_first_look_analysis
from core.ingestion import DatasetMetadata
from core.insights import generate_deterministic_insights
from core.outlier_analysis import analyze_all_numeric_outliers
from core.profiling import DatasetProfile, compute_dataset_profile
from core.report_generation import generate_consolidated_html_report
from core.settings import AnalysisSettings
from core.statistics import (
    compute_correlations,
    compute_descriptive_stats_table,
    compute_frequency_distribution,
    compute_group_summary,
)
from core.transformations import TransformationManager
from sample_data.loader import SAMPLE_DATASETS
from utils.formatting import format_bytes, format_number

# -----------------------------------------------------------------------------
# APPLICATION CONFIGURATION & CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="EDA Intelligence Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Inject custom dark analytics CSS styling safely."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------

if "dataset_df" not in st.session_state:
    st.session_state["dataset_df"] = None

if "original_df" not in st.session_state:
    st.session_state["original_df"] = None

if "working_df" not in st.session_state:
    st.session_state["working_df"] = None

if "transformation_manager" not in st.session_state:
    st.session_state["transformation_manager"] = None

if "analysis_settings" not in st.session_state:
    st.session_state["analysis_settings"] = AnalysisSettings()

if "dataset_metadata" not in st.session_state:
    st.session_state["dataset_metadata"] = None

if "first_look_report" not in st.session_state:
    st.session_state["first_look_report"] = None

if "selected_section" not in st.session_state:
    st.session_state["selected_section"] = "Overview & Getting Started"


# -----------------------------------------------------------------------------
# CACHED ANALYTICS COMPUTATION
# -----------------------------------------------------------------------------

def get_or_compute_profile(df: pd.DataFrame) -> DatasetProfile:
    """Compute and cache dataset profile in session state."""
    cache_key = f"profile_{len(df)}_{len(df.columns)}"
    if cache_key not in st.session_state or st.session_state.get("profile_df_id") != id(df):
        st.session_state[cache_key] = compute_dataset_profile(df)
        st.session_state["profile_df_id"] = id(df)
    return st.session_state[cache_key]


def get_or_compute_quality(df: pd.DataFrame, profile: DatasetProfile) -> QualityReport:
    """Compute and cache quality audit in session state."""
    cache_key = f"quality_{len(df)}_{len(df.columns)}"
    if cache_key not in st.session_state or st.session_state.get("quality_df_id") != id(df):
        st.session_state[cache_key] = analyze_data_quality(df, profile)
        st.session_state["quality_df_id"] = id(df)
    return st.session_state[cache_key]


# -----------------------------------------------------------------------------
# RENDER SECTIONS
# -----------------------------------------------------------------------------

def render_overview_section(df: pd.DataFrame | None, meta: DatasetMetadata | None) -> None:
    """Render landing state, product explanation, and quick action cards."""
    render_header(
        title="EDA Intelligence Workbench",
        tagline="Understand Your Data. Discover What Matters.",
        status_text="Dataset Loaded" if df is not None else "Ready for Ingestion",
    )

    if df is not None and meta is not None:
        st.success(
            f"Active dataset: **{meta.filename}** ({meta.loaded_rows:,} rows &bull; {meta.loaded_cols} columns &bull; {meta.file_format}). "
            "Use the navigation sidebar to explore data quality, profiling, statistics, and visualizations."
        )

    # Hero Intro Grid
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">🔍 Comprehensive Profiling</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                    Inspect dataset architecture, cell completeness, exact duplicate records,
                    and rigorous per-column distribution statistics across numeric, categorical, boolean, and datetime features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_feat2:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">🛡️ Data Quality Audit</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                    Calculate overall data health scores, detect high missingness, zero-variance columns,
                    whitespace contamination, non-finite values, and statistical outliers (IQR & Z-score) with actionable remediation steps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_feat3:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">⚡ First-Look Intelligence</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                    Generate deterministic, evidence-backed findings and curated starter visualizations
                    distinguishing facts, quality highlights, correlations, and analytical limitations with zero hallucinations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Product Guide & Supported Formats
    g1, g2 = st.columns([1.2, 1.0])
    with g1:
        st.markdown("#### 🎯 Platform Capabilities & Analysis Engine")
        st.markdown(
            """
            1. **Multi-Format Ingestion**: Ingest CSV, TSV, XLSX, and Parquet files with automatic encoding fallbacks and delimiter sniffing.
            2. **Conservative Type Inference**: Transparently distinguishes discrete vs continuous numbers, text vs categories, booleans, and datetimes.
            3. **Outlier Detection**: Tukey's Fences (IQR) and Z-score methods with explicit assumption explanations.
            4. **Correlation Analysis**: Pearson (linear) and Spearman (monotonic) rank matrices with degenerate input safeguards.
            5. **Interactive Dark Visualizations**: Plotly charts with automatic category aggregation and dark styling.
            6. **Audited Exports**: Export profile summaries, quality issues, descriptive stats, and machine-readable JSON metadata.
            """
        )

    with g2:
        st.markdown("#### 📂 Supported File Specifications")
        st.markdown(
            """
            - **CSV / TSV**: UTF-8, UTF-8-BOM, Latin-1, CP1252. Auto-sniffs `,`, `\\t`, `;`, `|`.
            - **Excel (.xlsx)**: Multi-worksheet inspection with interactive sheet selector.
            - **Parquet (.parquet)**: Columnar storage parsed via PyArrow.
            - **Memory Safeguard**: Automatic reproducible random sampling for datasets exceeding 250,000 rows.
            - **Privacy Guarantee**: 100% client-side local execution; zero telemetry or external network calls.
            """
        )

    if df is None:
        st.markdown("---")
        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-title">Get Started by Loading a Dataset</div>
                <div class="empty-state-desc">
                    Upload your tabular file in the Upload panel or select one of the bundled synthetic datasets
                    in the sidebar to immediately inspect distributions and quality metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📂 Go to Upload Panel", type="primary", use_container_width=True):
                st.session_state["selected_section"] = "Upload Dataset"
                st.rerun()
        with col_btn2:
            if st.button("🧪 Load Telecom Churn Sample Dataset", use_container_width=True):
                sample_info = SAMPLE_DATASETS["customer_churn"]
                sample_df = sample_info["loader"]()
                meta = DatasetMetadata(
                    filename=sample_info["filename"],
                    file_format="CSV (Bundled Sample)",
                    file_size_bytes=int(sample_df.memory_usage(deep=True).sum()),
                    original_rows=len(sample_df),
                    original_cols=len(sample_df.columns),
                    loaded_rows=len(sample_df),
                    loaded_cols=len(sample_df.columns),
                    is_sampled=False,
                )
                st.session_state["dataset_df"] = sample_df
                st.session_state["dataset_metadata"] = meta
                st.session_state["first_look_report"] = None
                st.session_state["selected_section"] = "Data Preview"
                st.rerun()


def render_profiling_section(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Render comprehensive dataset-level and column-level structural profiles."""
    st.markdown("### 📋 Dataset Architecture & Structural Profile")

    profile = get_or_compute_profile(df)

    # Top Structural KPI Cards
    c1 = render_metric_card("Total Rows", f"{profile.total_rows:,}", f"Original: {meta.original_rows:,}")
    c2 = render_metric_card("Total Columns", f"{profile.total_cols:,}", f"{profile.total_cells:,} total cells")
    c3 = render_metric_card("Memory Footprint", profile.formatted_memory, "Deep allocation")
    c4 = render_metric_card(
        "Missing Cells",
        f"{profile.missing_cells_percentage:.1f}%",
        f"{profile.total_missing_cells:,} cells missing",
        badge_text="Clean" if profile.missing_cells_percentage < 5 else "Missingness",
        badge_type="success" if profile.missing_cells_percentage < 5 else "warning",
    )
    c5 = render_metric_card(
        "Complete Rows",
        f"{profile.complete_rows_percentage:.1f}%",
        f"{profile.complete_rows_count:,} complete rows",
    )
    c6 = render_metric_card(
        "Duplicate Rows",
        f"{profile.duplicate_rows_percentage:.1f}%",
        f"{profile.duplicate_rows_count:,} duplicate rows",
        badge_text="Zero Dups" if profile.duplicate_rows_count == 0 else f"{profile.duplicate_rows_count} Duplicates",
        badge_type="success" if profile.duplicate_rows_count == 0 else "warning",
    )

    render_metric_grid([c1, c2, c3, c4, c5, c6])

    # Type Breakdown Badges
    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.25rem; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8;">Feature Types:</span>
            <span class="badge badge-info">{profile.numeric_columns_count} Numeric</span>
            <span class="badge badge-neutral">{profile.categorical_columns_count} Categorical/Text</span>
            <span class="badge badge-neutral">{profile.datetime_columns_count} Datetime</span>
            <span class="badge badge-neutral">{profile.boolean_columns_count} Boolean</span>
            {f'<span class="badge badge-warning">{len(profile.constant_columns)} Constant</span>' if profile.constant_columns else ''}
            {f'<span class="badge badge-critical">{len(profile.all_null_columns)} All Null</span>' if profile.all_null_columns else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Column Profiles Table
    st.markdown("#### 📑 Per-Column Structural Summary")

    type_filter = st.selectbox(
        "Filter by Inferred Semantic Type",
        options=["All Types", "Numeric", "Categorical / Text", "Datetime", "Boolean"],
    )

    col_records = []
    for col_name, p in profile.column_profiles.items():
        if type_filter == "Numeric" and not ("Numeric" in p.semantic_type):
            continue
        if type_filter == "Categorical / Text" and not ("Categorical" in p.semantic_type or "Text" in p.semantic_type):
            continue
        if type_filter == "Datetime" and p.semantic_type != "Datetime":
            continue
        if type_filter == "Boolean" and p.semantic_type != "Boolean":
            continue

        stats_summary = "—"
        if p.mean is not None:
            stats_summary = f"Mean: {p.mean:.2f} &bull; Median: {p.median:.2f} &bull; Std: {p.std:.2f}"
        elif p.top_values:
            top_name, top_cnt, top_pct = p.top_values[0]
            stats_summary = f"Top: '{top_name}' ({top_cnt:,}, {top_pct}%)"
        elif p.min_date:
            stats_summary = f"Span: {p.min_date} to {p.max_date}"

        col_records.append({
            "Column Name": col_name,
            "Type": p.semantic_type,
            "Distinct": f"{p.distinct_count:,}",
            "Uniqueness": f"{p.uniqueness_ratio:.1%}",
            "Missing": f"{p.null_count:,} ({p.null_percentage:.1f}%)",
            "Key Summary": stats_summary,
            "Flags": "Potential ID" if p.is_potential_identifier else ("Constant" if p.is_constant else "Normal"),
        })

    if col_records:
        st.dataframe(pd.DataFrame(col_records), use_container_width=True, hide_index=True)
    else:
        st.info("No columns match the selected type filter.")

    # Deep Column Inspector
    st.markdown("#### 🔎 Detailed Column Inspector")
    col_to_inspect = st.selectbox("Select a Column for Granular Inspection", options=list(df.columns))
    if col_to_inspect:
        p_col = profile.column_profiles[col_to_inspect]
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f"**Pandas Dtype:** `{p_col.dtype_str}`")
            st.markdown(f"**Semantic Classification:** `{p_col.semantic_type}`")
            st.markdown(f"**Distinct Values:** {p_col.distinct_count:,} ({p_col.uniqueness_ratio:.2%})")
            if p_col.is_potential_identifier:
                st.warning(f"Identifier Flag: {p_col.identifier_reason}")
        with ic2:
            st.markdown(f"**Non-Null Observations:** {p_col.non_null_count:,}")
            st.markdown(f"**Missing Values:** {p_col.null_count:,} ({p_col.null_percentage:.1f}%)")
            st.markdown(f"**Memory Allocation:** {format_bytes(p_col.memory_bytes)}")
        with ic3:
            if p_col.mean is not None:
                st.markdown(f"**Mean:** {format_number(p_col.mean)} &bull; **Std:** {format_number(p_col.std)}")
                st.markdown(f"**Range:** [{format_number(p_col.min_val)} to {format_number(p_col.max_val)}]")
                st.markdown(f"**IQR:** {format_number(p_col.iqr)} &bull; **Skew:** {format_number(p_col.skewness)}")
            elif p_col.top_values:
                st.markdown("**Top Frequent Values:**")
                for v_name, v_cnt, v_pct in p_col.top_values[:3]:
                    st.markdown(f"- `{v_name}`: {v_cnt:,} ({v_pct}%)")


def render_quality_section(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Render data-quality audit, health score banner, outlier detection, and actionable remediation."""
    st.markdown("### 🛡️ Automated Data Quality Audit")

    profile = get_or_compute_profile(df)
    quality = get_or_compute_quality(df, profile)

    # Health score banner
    render_health_score_banner(
        score=quality.overall_health_score,
        status=quality.health_status,
        critical_count=quality.critical_issues_count,
        warning_count=quality.warning_issues_count,
        info_count=quality.info_issues_count,
    )

    # Outlier Detection Tabs
    st.markdown("#### 🎯 Statistical Outlier Audit")
    tab_iqr, tab_zscore = st.tabs(["IQR Method (Tukey's Fences)", "Z-Score Method (|z| > 3.0)"])

    with tab_iqr:
        st.markdown(
            "Detects values outside `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`. "
            "Does not assume a normal distribution and is resilient to heavy skew."
        )
        if quality.iqr_outliers:
            iqr_records = []
            for col_name, out_info in quality.iqr_outliers.items():
                iqr_records.append({
                    "Feature": col_name,
                    "Outlier Count": f"{out_info.outlier_count:,}",
                    "Outlier %": f"{out_info.outlier_percentage:.2f}%",
                    "Lower Bound": out_info.lower_bound,
                    "Upper Bound": out_info.upper_bound,
                    "Extremes Observed": f"[{out_info.min_outlier}, {out_info.max_outlier}]",
                })
            st.dataframe(pd.DataFrame(iqr_records), use_container_width=True, hide_index=True)
            st.caption("⚠️ Note: Statistical outliers may represent genuine high-value events; do not delete without domain verification.")
        else:
            st.success("No numeric features exceeded 1.5x IQR boundaries.")

    with tab_zscore:
        st.markdown(
            "Detects values with standard score `|z| > 3.0` (more than 3 standard deviations from mean). "
            "**Assumption:** Assumes approximately normal distribution."
        )
        if quality.zscore_outliers:
            z_records = []
            for col_name, out_info in quality.zscore_outliers.items():
                z_records.append({
                    "Feature": col_name,
                    "Outlier Count": f"{out_info.outlier_count:,}",
                    "Outlier %": f"{out_info.outlier_percentage:.2f}%",
                    "Lower Bound": out_info.lower_bound,
                    "Upper Bound": out_info.upper_bound,
                    "Extremes Observed": f"[{out_info.min_outlier}, {out_info.max_outlier}]",
                })
            st.dataframe(pd.DataFrame(z_records), use_container_width=True, hide_index=True)
        else:
            st.success("No numeric features contain observations with |z| > 3.0.")

    st.markdown("---")

    # Issue Log with Filters
    st.markdown("#### 🚨 Detected Quality Conditions & Remediation Guidance")

    sev_filter = st.selectbox(
        "Filter Issues by Severity",
        options=["All Severities", "Critical", "Warning", "Informational"],
    )

    filtered_issues = [
        iss for iss in quality.issues
        if sev_filter == "All Severities" or iss.severity == sev_filter
    ]

    if filtered_issues:
        for issue in filtered_issues:
            render_issue_alert(issue)
    else:
        st.success(f"No issues found matching severity filter '{sev_filter}'.")


def render_statistics_section(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Render descriptive statistics, correlation matrices, frequency analysis, and group comparisons."""
    st.markdown("### 📐 Statistical Summary & Relationships")

    profile = get_or_compute_profile(df)

    stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs([
        "Descriptive Statistics",
        "Correlation Analysis",
        "Categorical Frequencies",
        "Group-wise Comparisons",
    ])

    with stat_tab1:
        st.markdown("#### Numeric Feature Distribution Statistics")
        desc_table = compute_descriptive_stats_table(df, profile)
        if not desc_table.empty:
            st.dataframe(desc_table, use_container_width=True, hide_index=True)
            st.caption(
                "Calculates count, central tendencies (mean, median), dispersions (std, IQR, range), "
                "and higher moments (skewness, kurtosis) across non-null numeric observations."
            )
        else:
            st.info("No numeric columns with non-null values available for descriptive statistics.")

    with stat_tab2:
        st.markdown("#### Numeric Correlation Analysis")
        corr_col1, _ = st.columns([1, 2])
        with corr_col1:
            method_choice = st.radio(
                "Correlation Method",
                options=["Spearman (Monotonic / Non-linear)", "Pearson (Linear)"],
            )
            method_key = "spearman" if "Spearman" in method_choice else "pearson"

        corr_res = compute_correlations(df, profile, method=method_key)

        if corr_res.warnings:
            for w in corr_res.warnings:
                st.caption(f"ℹ️ {w}")

        if not corr_res.matrix.empty:
            st.plotly_chart(
                plot_correlation_heatmap(
                    corr_res.matrix,
                    title=f"{corr_res.method} Correlation Matrix Heatmap",
                ),
                use_container_width=True,
            )

            if corr_res.top_pairs:
                st.markdown("##### 🏆 Top Ranked Correlated Feature Pairs")
                pair_records = []
                for p in corr_res.top_pairs[:8]:
                    pair_records.append({
                        "Feature A": p.column_a,
                        "Feature B": p.column_b,
                        "Correlation (r)": f"{p.correlation:.3f}",
                        "Strength": p.strength,
                        "Direction": p.direction,
                        "Pair Sample Size": f"{p.sample_size:,}",
                    })
                st.dataframe(pd.DataFrame(pair_records), use_container_width=True, hide_index=True)

            st.info(f"⚖️ **Methodology & Limitation:** {corr_res.disclaimer}")
        else:
            st.warning("Insufficient non-constant numeric features to compute a correlation matrix.")

    with stat_tab3:
        st.markdown("#### Categorical Frequency Distribution Explorer")
        cat_cols = [
            col for col, p in profile.column_profiles.items()
            if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 25
        ]

        if cat_cols:
            selected_cat = st.selectbox("Select Categorical Feature", options=cat_cols)
            if selected_cat:
                freq_df, is_trunc, other_cnt = compute_frequency_distribution(df[selected_cat], max_categories=15)
                fcol1, fcol2 = st.columns([1.2, 1.0])
                with fcol1:
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)
                    if is_trunc:
                        st.caption(f"Aggregated infrequent categories into 'Other' ({other_cnt:,} records).")
                with fcol2:
                    fig_bar, _, _ = plot_categorical_bar(df, selected_cat, max_categories=12)
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No categorical features found in dataset.")

    with stat_tab4:
        st.markdown("#### Group-wise Metric Summaries")
        cat_options = [
            col for col, p in profile.column_profiles.items()
            if p.distinct_count <= 20 and p.distinct_count >= 2
        ]
        num_options = [
            col for col, p in profile.column_profiles.items()
            if p.mean is not None and not p.is_constant
        ]

        if cat_options and num_options:
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                grp_col = st.selectbox("Group By (Category)", options=cat_options)
            with g_c2:
                met_col = st.selectbox("Measure (Numeric Feature)", options=num_options)

            if grp_col and met_col:
                grp_summary_df = compute_group_summary(df, grp_col, met_col)
                st.dataframe(grp_summary_df, use_container_width=True, hide_index=True)

                fig_box_grp = plot_box_by_category(df, grp_col, met_col)
                st.plotly_chart(fig_box_grp, use_container_width=True)
        else:
            st.info("Group-wise summaries require at least one categorical feature and one numeric feature.")


def render_visualizations_section(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Render interactive chart builder with guardrails and dynamic parameters."""
    st.markdown("### 📊 Interactive Visualizations Engine")

    profile = get_or_compute_profile(df)

    v_category = st.radio(
        "Chart Family",
        options=[
            "Numeric Distributions",
            "Categorical Breakdowns",
            "Relationships & Bivariate",
            "Time-Series Trends",
            "Group Comparisons",
        ],
        horizontal=True,
    )

    numeric_cols = [col for col, p in profile.column_profiles.items() if p.mean is not None]
    cat_cols = [col for col, p in profile.column_profiles.items() if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 30]
    dt_cols = [col for col, p in profile.column_profiles.items() if p.semantic_type == "Datetime" or p.min_date]

    if v_category == "Numeric Distributions":
        if not numeric_cols:
            st.warning("No numeric columns available for distribution plots.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_num = st.selectbox("Select Numeric Column", options=numeric_cols)
        with c2:
            chart_type = st.selectbox("Chart Type", options=["Histogram with Box", "Box Plot", "Violin Plot", "ECDF Plot"])
        with c3:
            bins = st.slider("Histogram Bins", min_value=10, max_value=80, value=30)

        if chart_type == "Histogram with Box":
            fig = plot_histogram(df, sel_num, nbins=bins, show_kde=True)
        elif chart_type == "Box Plot":
            fig = plot_boxplot(df, sel_num)
        elif chart_type == "Violin Plot":
            fig = plot_violin(df, sel_num)
        else:
            fig = plot_ecdf(df, sel_num)

        st.plotly_chart(fig, use_container_width=True)

    elif v_category == "Categorical Breakdowns":
        if not cat_cols:
            st.warning("No categorical columns available for breakdown plots.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_cat = st.selectbox("Select Categorical Column", options=cat_cols)
        with c2:
            cat_type = st.selectbox("Chart Type", options=["Vertical Bar", "Horizontal Bar", "Treemap"])
        with c3:
            max_c = st.slider("Max Categories to Show", min_value=5, max_value=25, value=12)

        if cat_type == "Vertical Bar":
            fig, is_agg, other_c = plot_categorical_bar(df, sel_cat, max_categories=max_c, horizontal=False)
            st.plotly_chart(fig, use_container_width=True)
            if is_agg:
                st.caption(f"Infrequent categories aggregated into 'Other' ({other_c:,} records).")
        elif cat_type == "Horizontal Bar":
            fig, is_agg, other_c = plot_categorical_bar(df, sel_cat, max_categories=max_c, horizontal=True)
            st.plotly_chart(fig, use_container_width=True)
            if is_agg:
                st.caption(f"Infrequent categories aggregated into 'Other' ({other_c:,} records).")
        else:
            fig = plot_treemap(df, sel_cat, max_categories=max_c)
            st.plotly_chart(fig, use_container_width=True)

    elif v_category == "Relationships & Bivariate":
        if len(numeric_cols) < 2:
            st.warning("Scatter and pairwise relationships require at least two numeric columns.")
            return

        rel_type = st.radio("Relationship Type", options=["Bivariate Scatter Plot", "Pairwise Matrix (Matrix of 3-5 Columns)"], horizontal=True)

        if rel_type == "Bivariate Scatter Plot":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x_c = st.selectbox("X-Axis (Numeric)", options=numeric_cols, index=0)
            with c2:
                y_c = st.selectbox("Y-Axis (Numeric)", options=numeric_cols, index=min(1, len(numeric_cols) - 1))
            with c3:
                color_opt = st.selectbox("Color By (Optional Category)", options=["None"] + cat_cols)
                color_arg = None if color_opt == "None" else color_opt
            with c4:
                trend_opt = st.checkbox("OLS Trendline", value=True)

            fig = plot_scatter(df, x_c, y_c, color_col=color_arg, trendline=trend_opt)
            st.plotly_chart(fig, use_container_width=True)

        else:
            pw_cols = st.multiselect(
                "Select 3 to 5 Numeric Columns for Matrix",
                options=numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
            )
            color_opt = st.selectbox("Matrix Color Group (Optional)", options=["None"] + cat_cols)
            color_arg = None if color_opt == "None" else color_opt

            if len(pw_cols) < 2:
                st.info("Select at least 2 numeric features to display the pairwise scatter matrix.")
            else:
                fig = plot_pairwise_scatter(df, pw_cols, color_col=color_arg)
                st.plotly_chart(fig, use_container_width=True)

    elif v_category == "Time-Series Trends":
        if not dt_cols:
            st.warning("No datetime columns detected in dataset. Select or convert a date column to view time-series trends.")
            return
        if not numeric_cols:
            st.warning("Time series plots require at least one numeric measure to track over time.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_dt = st.selectbox("Datetime Column", options=dt_cols)
        with c2:
            sel_val = st.selectbox("Measure Column (Y-Axis)", options=numeric_cols)
        with c3:
            freq_choice = st.selectbox("Resampling Frequency", options=["Raw", "Daily", "Weekly", "Monthly"])

        fig = plot_time_series(df, sel_dt, sel_val, freq=freq_choice, show_rolling=True)
        st.plotly_chart(fig, use_container_width=True)

    elif v_category == "Group Comparisons":
        if not cat_cols or not numeric_cols:
            st.warning("Group comparisons require at least one categorical feature and one numeric feature.")
            return

        c1, c2 = st.columns(2)
        with c1:
            sel_gcat = st.selectbox("Grouping Feature (Category)", options=cat_cols)
        with c2:
            sel_gnum = st.selectbox("Measured Feature (Numeric)", options=numeric_cols)

        fig = plot_box_by_category(df, sel_gcat, sel_gnum)
        st.plotly_chart(fig, use_container_width=True)


def render_first_look_section(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Render deterministic Automatic First-Look Analysis with starter recommendations."""
    st.markdown("### ⚡ Automatic First-Look Analysis")
    st.markdown(
        "Runs an end-to-end evidence-based evaluation across dataset structure, quality issues, "
        "skewness, and correlations to produce deterministic observations and curated starter visualizations."
    )

    profile = get_or_compute_profile(df)
    quality = get_or_compute_quality(df, profile)

    # Prominent Action Button
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("⚡ Run Automatic First-Look Analysis", type="primary", use_container_width=True):
            with st.spinner("Synthesizing observations and curating visualizations..."):
                report = run_first_look_analysis(df, profile, quality)
                st.session_state["first_look_report"] = report

    report: FirstLookReport | None = st.session_state.get("first_look_report")

    # If not yet generated, auto-generate once
    if report is None:
        report = run_first_look_analysis(df, profile, quality)
        st.session_state["first_look_report"] = report

    st.markdown("---")

    # Display Findings in Columns
    f_col1, f_col2 = st.columns(2, gap="large")

    with f_col1:
        st.markdown("#### 📌 Observed Structural Facts")
        for fact in report.facts:
            render_observation_callout(fact)

        st.markdown("#### 🛡️ Data Quality Highlights")
        for q in report.quality_highlights:
            render_observation_callout(q)

        st.markdown("#### 🔗 Key Relationships & Associations")
        for rel in report.relationships:
            render_observation_callout(rel)

    with f_col2:
        st.markdown("#### 📈 Distribution Insights")
        for dist in report.distribution_insights:
            render_observation_callout(dist)

        st.markdown("#### 💡 Suggested Next Steps")
        for n_step in report.next_steps:
            render_observation_callout(n_step)

        st.markdown("#### ⚖️ Analytical Limitations & Disclaimers")
        for lim in report.limitations:
            render_observation_callout(lim)

    # Curated Starter Visualizations
    st.markdown("---")
    st.markdown("#### 🎨 Curated Starter Visualizations")
    cand = report.visual_candidates

    cv1, cv2 = st.columns(2)
    with cv1:
        if cand.top_pair_x and cand.top_pair_y:
            st.markdown(f"**Primary Correlation:** `{cand.top_pair_x}` vs `{cand.top_pair_y}` (r = {cand.top_pair_r:.2f})")
            fig_top = plot_scatter(df, cand.top_pair_x, cand.top_pair_y, trendline=True)
            st.plotly_chart(fig_top, use_container_width=True)
        elif cand.primary_numeric_col:
            st.markdown(f"**Primary Numeric Distribution:** `{cand.primary_numeric_col}`")
            fig_pnum = plot_histogram(df, cand.primary_numeric_col)
            st.plotly_chart(fig_pnum, use_container_width=True)

    with cv2:
        if cand.primary_categorical_col:
            st.markdown(f"**Primary Category Breakdown:** `{cand.primary_categorical_col}`")
            fig_pcat, _, _ = plot_categorical_bar(df, cand.primary_categorical_col)
            st.plotly_chart(fig_pcat, use_container_width=True)
        elif cand.primary_numeric_col:
            st.markdown(f"**Box Quartile Spread:** `{cand.primary_numeric_col}`")
            fig_pbox = plot_boxplot(df, cand.primary_numeric_col)
            st.plotly_chart(fig_pbox, use_container_width=True)


def render_export_section(
    df: pd.DataFrame,
    meta: DatasetMetadata,
    settings: AnalysisSettings,
    manager: TransformationManager | None = None,
) -> None:
    """Render export and download options with UTF-8 encoding and clean schemas."""
    st.markdown("### 💾 Export Center & Executive Reporting")
    st.caption(
        "Download audit findings, descriptive tables, correlation matrices, outlier summaries, "
        "and the self-contained HTML executive audit report."
    )

    profile = get_or_compute_profile(df)
    quality = get_or_compute_quality(df, profile)
    first_look: FirstLookReport | None = st.session_state.get("first_look_report")

    # Generate insights and supplementary tables for HTML report
    all_insights = generate_deterministic_insights(df, profile, quality, settings)
    stats_df = compute_extended_descriptive_stats(df, profile)
    outlier_df = analyze_all_numeric_outliers(df, method=settings.default_outlier_method)
    corr_res = compute_correlation_matrix(df, method=settings.default_correlation_method)
    ranked_pairs = extract_ranked_correlations(corr_res, top_n=10)

    # 1. Executive HTML Report
    st.markdown("#### 📑 Consolidated Executive Audit Report (HTML)")
    st.caption("A standalone, responsive, dark-themed HTML report containing executive KPIs, insights, stats, and audit trails.")

    html_report = generate_consolidated_html_report(
        metadata=meta,
        profile=profile,
        quality_report=quality,
        insights=all_insights,
        settings=settings,
        stats_df=stats_df,
        outlier_df=outlier_df,
        ranked_pairs=ranked_pairs,
        transformations=manager.history if manager else None,
    )

    st.download_button(
        label="⬇️ Download Consolidated HTML Audit Report",
        data=html_report.encode("utf-8"),
        file_name=f"eda_audit_report_{meta.filename}.html",
        mime="text/html",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### 📄 Tabular CSV Downloads")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">Active Working Dataset (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Download active tabular records with normalized column headers in UTF-8 encoding.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        csv_bytes = export_dataset_as_csv(df)
        st.download_button(
            label="⬇️ Download Dataset CSV",
            data=csv_bytes,
            file_name=f"exported_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="workbench-card" style="margin-top: 1rem;">
                <div class="workbench-card-title">Data Quality Issues (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Tabular export of all identified quality issues, affected columns, counts, and remediation steps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        quality_csv = export_quality_issues_as_csv(quality)
        st.download_button(
            label="⬇️ Download Quality Issues CSV",
            data=quality_csv,
            file_name=f"quality_issues_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="workbench-card" style="margin-top: 1rem;">
                <div class="workbench-card-title">Correlation Matrix (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Full pairwise correlation coefficient matrix computed under hypothesis testing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        corr_csv = export_correlation_matrix_as_csv(corr_res.matrix)
        st.download_button(
            label="⬇️ Download Correlation Matrix CSV",
            data=corr_csv,
            file_name=f"correlation_matrix_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">Dataset Profile Summary (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Per-column structural metrics including missing percentages, distinct counts, and moments.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        profile_csv = export_profile_summary_as_csv(profile)
        st.download_button(
            label="⬇️ Download Profile Summary CSV",
            data=profile_csv,
            file_name=f"profile_summary_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="workbench-card" style="margin-top: 1rem;">
                <div class="workbench-card-title">Extended Descriptive Statistics (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Parametric & non-parametric distribution statistics: moments, CV, percentiles.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        stats_csv = export_descriptive_stats_as_csv(df, profile)
        st.download_button(
            label="⬇️ Download Descriptive Stats CSV",
            data=stats_csv,
            file_name=f"descriptive_stats_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="workbench-card" style="margin-top: 1rem;">
                <div class="workbench-card-title">Outlier Analysis Summary (CSV)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Outlier counts, percentages, and upper/lower fences across all continuous features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        outlier_csv = export_outlier_summary_as_csv(outlier_df)
        st.download_button(
            label="⬇️ Download Outlier Summary CSV",
            data=outlier_csv,
            file_name=f"outlier_summary_{meta.filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 🤖 Machine-Readable JSON Exports")
    j1, j2 = st.columns(2)

    with j1:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">Complete Audit Summary (JSON)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Consolidated machine-readable schema of profile metrics, data health, and first look findings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        json_summary = export_json_audit_summary(meta, profile, quality, first_look)
        st.download_button(
            label="⬇️ Download Audit Summary JSON",
            data=json_summary.encode("utf-8"),
            file_name=f"eda_audit_report_{meta.filename}.json",
            mime="application/json",
            use_container_width=True,
        )

    with j2:
        st.markdown(
            """
            <div class="workbench-card">
                <div class="workbench-card-title">Analytical Findings (JSON)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    Ranked deterministic findings across all 9 analytical categories with evidence metadata.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        insights_json = export_insights_as_json(all_insights)
        st.download_button(
            label="⬇️ Download Analytical Insights JSON",
            data=insights_json.encode("utf-8"),
            file_name=f"insights_{meta.filename}.json",
            mime="application/json",
            use_container_width=True,
        )

    if manager and manager.history:
        st.markdown("---")
        st.markdown("#### 📜 Transformation Audit Trail Downloads")
        th1, th2 = st.columns(2)
        with th1:
            t_csv = export_transformation_history_as_csv(manager.history)
            st.download_button(
                "⬇️ Download Transformation History (CSV)",
                data=t_csv,
                file_name=f"transformations_{meta.filename}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with th2:
            t_json = export_transformation_history_as_json(manager.history)
            st.download_button(
                "⬇️ Download Transformation History (JSON)",
                data=t_json.encode("utf-8"),
                file_name=f"transformations_{meta.filename}.json",
                mime="application/json",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# MAIN APPLICATION ROUTING
# -----------------------------------------------------------------------------

def main() -> None:
    """Main routing dispatcher."""
    if "analysis_settings" not in st.session_state:
        st.session_state["analysis_settings"] = AnalysisSettings()
    settings: AnalysisSettings = st.session_state["analysis_settings"]

    raw_df: pd.DataFrame | None = st.session_state.get("dataset_df")
    meta: DatasetMetadata | None = st.session_state.get("dataset_metadata")

    if raw_df is not None and st.session_state.get("transformation_manager") is None:
        st.session_state["transformation_manager"] = TransformationManager(raw_df)
        st.session_state["original_df"] = raw_df.copy()
        st.session_state["working_df"] = raw_df.copy()

    manager: TransformationManager | None = st.session_state.get("transformation_manager")
    df: pd.DataFrame | None = manager.working_df if manager is not None else raw_df

    # Render sidebar navigation
    selected_section = render_sidebar(current_metadata=meta)

    # Route based on selected section
    if selected_section == "Overview & Getting Started":
        render_overview_section(df, meta)

    elif selected_section == "Upload Dataset":
        render_upload_panel()

    elif selected_section == "Data Preview":
        if df is not None and meta is not None:
            render_data_preview(df, meta)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Dataset Profile":
        if df is not None and meta is not None:
            render_profiling_section(df, meta)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Data Quality Audit":
        if df is not None and meta is not None:
            render_quality_section(df, meta)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Automated Insights":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            quality = get_or_compute_quality(df, profile)
            render_insights_panel(df, profile, quality, settings)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Visualization Studio":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            render_chart_explorer(df, profile)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Statistical Analysis":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            render_statistical_analysis(df, profile, settings)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Correlation Explorer":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            render_correlation_explorer(df, profile, settings)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Missingness Analysis":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            render_missingness_panel(df, profile, settings)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Outlier Analysis":
        if df is not None and meta is not None:
            profile = get_or_compute_profile(df)
            render_outlier_panel(df, profile, settings)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Transformation Workbench":
        if df is not None and manager is not None:
            render_transformation_workbench(manager)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Automatic First-Look":
        if df is not None and meta is not None:
            render_first_look_section(df, meta)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()

    elif selected_section == "Analysis Configuration":
        render_analysis_settings(settings)

    elif selected_section == "Export & Download":
        if df is not None and meta is not None:
            render_export_section(df, meta, settings, manager)
        else:
            st.info("No dataset is loaded. Upload a dataset or select a sample from the sidebar.")
            render_upload_panel()


if __name__ == "__main__":
    main()

