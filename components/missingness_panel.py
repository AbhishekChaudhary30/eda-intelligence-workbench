"""Missingness Investigation UI component for EDA Intelligence Workbench.

Provides comprehensive nullity profiling, missingness correlation matrix,
row-level missing distributions, group-wise missingness, and pseudo-missing string audit.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.chart_components import COLOR_PALETTE, apply_dark_theme
from core.missingness import (
    compute_group_missingness,
    compute_nullity_correlation,
    compute_row_missingness_distribution,
    detect_pseudo_missing_tokens,
)
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings


def render_missingness_panel(
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: AnalysisSettings,
) -> None:
    """Render the Missingness Investigation panel."""
    st.markdown("### 🔍 Dedicated Missing-Data Investigation")
    st.caption(
        "Examine missing-data prevalence, co-occurrence nullity correlations, row-level incompleteness, "
        "and disguised missing tokens (e.g. 'N/A', '?', 'none')."
    )

    # 1. Headline KPIs
    missing_cells = profile.total_missing_cells
    missing_pct = profile.missing_cells_percentage
    complete_rows = profile.complete_rows_count
    complete_pct = profile.complete_rows_percentage

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Missing Cells", f"{missing_cells:,}", delta=f"{missing_pct:.2f}% of dataset", delta_color="inverse")
    kpi2.metric("Complete Rows", f"{complete_rows:,}", delta=f"{complete_pct:.1f}% intact")
    kpi3.metric("Total Columns with Nulls", str(sum(1 for p in profile.column_profiles.values() if p.null_count > 0)))
    kpi4.metric("All-Null Columns", str(len(profile.all_null_columns)))

    tab_summary, tab_nullity_corr, tab_rows, tab_tokens, tab_group = st.tabs([
        "📊 Column Missingness Breakdown",
        "🧩 Nullity Correlation Matrix",
        "📉 Row-Level Incompleteness",
        "🏷️ Pseudo-Missing Token Audit",
        "👥 Group-Wise Missingness",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Column Breakdown
    # -------------------------------------------------------------------------
    with tab_summary:
        st.markdown("#### Per-Column Missing Value Counts & Percentages")

        records = []
        for col_name, p in profile.column_profiles.items():
            records.append({
                "Column Name": col_name,
                "Semantic Type": p.semantic_type,
                "Missing Count": p.null_count,
                "Missing Percentage (%)": p.null_percentage,
                "Valid Observations": p.non_null_count,
                "Status": "All Null" if p.is_all_null else ("Clean" if p.null_count == 0 else "Partially Missing"),
            })

        missing_df = pd.DataFrame(records).sort_values("Missing Count", ascending=False)

        # Plot missingness bar chart
        cols_with_missing = missing_df[missing_df["Missing Count"] > 0]
        if not cols_with_missing.empty:
            fig_bar = px.bar(
                cols_with_missing.head(25),
                x="Column Name",
                y="Missing Percentage (%)",
                color="Missing Percentage (%)",
                color_continuous_scale="Reds",
                text="Missing Percentage (%)",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar = apply_dark_theme(fig_bar, title="Top Features by Missing Value Percentage")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("🎉 Outstanding data completeness! Zero missing values detected across all columns.")

        st.dataframe(
            missing_df.style.format({
                "Missing Count": "{:,}",
                "Missing Percentage (%)": "{:.2f}%",
                "Valid Observations": "{:,}",
            }),
            use_container_width=True,
            height=360,
        )

        csv_miss = missing_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Missingness Breakdown (CSV)",
            data=csv_miss,
            file_name="missingness_breakdown.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # TAB 2: Nullity Correlation Matrix
    # -------------------------------------------------------------------------
    with tab_nullity_corr:
        st.markdown("#### Nullity Correlation Matrix")
        st.caption(
            "Measures whether missingness in one feature predicts missingness in another. "
            "A value near +1 indicates variables always go missing together; -1 indicates mutual exclusivity."
        )

        nullity_matrix = compute_nullity_correlation(df)
        if nullity_matrix.empty or len(nullity_matrix) < 2:
            st.info("Insufficient features with variable missingness to compute nullity correlations.")
        else:
            fig_null = px.imshow(
                nullity_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
            )
            fig_null = apply_dark_theme(fig_null, title="Co-occurring Missingness (Nullity Correlation)")
            st.plotly_chart(fig_null, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Row-Level Incompleteness
    # -------------------------------------------------------------------------
    with tab_rows:
        st.markdown("#### Row-Level Missing Value Distribution")
        st.caption("Distribution of missing field counts across individual records.")

        row_dist = compute_row_missingness_distribution(df)
        fig_rows = px.bar(
            row_dist,
            x="Missing_Fields_Count",
            y="Row_Count",
            text="Percentage",
            color_discrete_sequence=[COLOR_PALETTE[0]],
        )
        fig_rows.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rows.update_xaxes(title_text="Number of Missing Features in a Row")
        fig_rows.update_yaxes(title_text="Row Count")
        fig_rows = apply_dark_theme(fig_rows, title="Row Incompleteness Frequency")
        st.plotly_chart(fig_rows, use_container_width=True)

        # Show worst incomplete rows
        st.markdown("##### Records with Highest Missingness")
        null_counts_per_row = df.isnull().sum(axis=1)
        worst_rows_idx = null_counts_per_row.sort_values(ascending=False).head(10).index
        if null_counts_per_row.max() > 0:
            st.dataframe(df.loc[worst_rows_idx], use_container_width=True)
        else:
            st.info("No incomplete rows.")

    # -------------------------------------------------------------------------
    # TAB 4: Pseudo-Missing Token Audit
    # -------------------------------------------------------------------------
    with tab_tokens:
        st.markdown("#### Pseudo-Missing String Tokens Audit")
        st.caption(
            "Identifies disguised missing indicators stored as string text (e.g. 'N/A', 'null', '?', '-', 'none', 'NaN')."
        )

        tokens_input = st.text_input(
            "Audit Tokens (Comma-separated)",
            value=", ".join(settings.custom_missing_tokens),
        )
        parsed_tokens = [t.strip() for t in tokens_input.split(",") if t.strip()]

        detected_tokens = detect_pseudo_missing_tokens(df, tokens=parsed_tokens)
        if not detected_tokens:
            st.success("No text pseudo-missing tokens detected with the specified patterns.")
        else:
            t_records = []
            for item in detected_tokens:
                t_records.append({
                    "Column Name": item["column"],
                    "Detected Token": item["token"],
                    "Occurrence Count": item["count"],
                    "Percentage of Column (%)": item["percentage"],
                })
            df_tokens = pd.DataFrame(t_records)
            st.dataframe(
                df_tokens.style.format({
                    "Occurrence Count": "{:,}",
                    "Percentage of Column (%)": "{:.2f}%",
                }),
                use_container_width=True,
            )
            st.info("💡 You can convert these text tokens into true NaN nulls in the **Transformation Workbench**.")

    # -------------------------------------------------------------------------
    # TAB 5: Group-Wise Missingness
    # -------------------------------------------------------------------------
    with tab_group:
        st.markdown("#### Segmented Missingness by Categorical Factor")
        st.caption("Investigate whether missingness is concentrated in particular operational segments or cohorts.")

        cat_cols = [
            c for c, p in profile.column_profiles.items()
            if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 25
        ]

        if not cat_cols:
            st.info("Requires at least one categorical feature to segment missingness.")
        else:
            grp_col = st.selectbox("Select Categorical Grouping Variable", options=cat_cols)
            if grp_col:
                grp_missing = compute_group_missingness(df, group_col=grp_col)
                if grp_missing.empty:
                    st.info("No missing values found across groups.")
                else:
                    st.dataframe(
                        grp_missing,
                        use_container_width=True,
                    )
