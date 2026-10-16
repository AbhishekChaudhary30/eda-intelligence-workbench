"""Correlation Explorer UI component for EDA Intelligence Workbench.

Provides interactive correlation analysis with Pearson, Spearman, and Kendall methods,
exact pairwise sample sizes, p-values, Benjamini-Hochberg FDR correction,
and individual pair bivariate deep-dive scatter visualizations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_components import plot_correlation_heatmap, plot_scatter
from components.navigation import render_section_navigation
from core.correlations import (
    compute_correlation_matrix,
    extract_ranked_correlations,
)
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings


def render_correlation_explorer(
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: AnalysisSettings,
) -> None:
    """Render the Advanced Correlation Explorer panel."""
    st.markdown("### 🔗 Advanced Correlation Explorer")
    st.caption(
        "Evaluate pairwise linear (Pearson) or monotonic (Spearman, Kendall) dependencies. "
        "Includes exact paired sample sizes, two-tailed p-values, and Benjamini-Hochberg FDR adjustments."
    )
    render_section_navigation("Correlation Explorer", position="top")

    numeric_cols = [
        c for c, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null and not p.is_constant
    ]

    if len(numeric_cols) < 2:
        st.warning("⚠️ Insufficient numeric features for correlation analysis (requires at least 2 non-constant numeric columns).")
        return

    # Configuration bar
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        default_cols = numeric_cols[:min(15, len(numeric_cols))]
        selected_cols = st.multiselect(
            "Select Features to Correlate",
            options=numeric_cols,
            default=default_cols,
            help="To maintain visual readability and performance, matrix displays are optimal for up to 25 features.",
        )
    with c2:
        method = st.selectbox(
            "Method",
            options=["spearman", "pearson", "kendall"],
            index=["spearman", "pearson", "kendall"].index(settings.default_correlation_method)
            if settings.default_correlation_method in ["spearman", "pearson", "kendall"]
            else 0,
            format_func=lambda m: {
                "spearman": "Spearman (Rank)",
                "pearson": "Pearson (Linear)",
                "kendall": "Kendall Tau (Robust)",
            }[m],
        )
    with c3:
        min_thresh = st.slider(
            "Min |r| Threshold",
            min_value=0.0,
            max_value=0.9,
            value=float(settings.correlation_threshold),
            step=0.05,
        )
    with c4:
        use_fdr = st.checkbox("Apply BH FDR Correction", value=True)

    if len(selected_cols) < 2:
        st.info("Select at least 2 numeric features above to compute correlation.")
        return

    # Compute correlation result
    corr_res = compute_correlation_matrix(
        df,
        columns=selected_cols,
        method=method,
        min_threshold=min_thresh,
        apply_fdr=use_fdr,
    )

    tab_heatmap, tab_ranked, tab_deepdive = st.tabs([
        "🗺️ Correlation Heatmap",
        "🏆 Ranked Association Pairs",
        "🔍 Pair Deep Dive",
    ])

    with tab_heatmap:
        st.markdown(f"#### {method.capitalize()} Correlation Coefficient Matrix")
        fig_heat = plot_correlation_heatmap(corr_res.matrix, title=f"{method.capitalize()} Correlation Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

        show_sizes = st.checkbox("Show Paired Sample Sizes Table", value=False)
        if show_sizes and corr_res.sample_size_matrix is not None:
            st.markdown("##### Pairwise Sample Sizes (N)")
            st.dataframe(corr_res.sample_size_matrix, use_container_width=True)

        csv_corr = corr_res.matrix.to_csv(index=True).encode("utf-8")
        st.download_button(
            "📥 Export Correlation Matrix (CSV)",
            data=csv_corr,
            file_name=f"correlation_matrix_{method}.csv",
            mime="text/csv",
        )

    with tab_ranked:
        st.markdown("#### Ranked Pairwise Correlations")
        st.caption("Sorted by absolute correlation magnitude |r|.")

        ranked_pairs = extract_ranked_correlations(corr_res, top_n=30)
        if not ranked_pairs:
            st.info(f"No variable pairs satisfy the minimum |r| threshold of {min_thresh:.2f}.")
        else:
            records = []
            for p in ranked_pairs:
                p_val = p["p_value"]
                is_sig = p.get("significant", p_val < settings.alpha)
                records.append({
                    "Feature A": p["col1"],
                    "Feature B": p["col2"],
                    "Coefficient (r)": p["r"],
                    "p-value": p_val,
                    "Adj p-value (FDR)": p.get("adj_p_value", p_val),
                    "Sample Size (N)": p["sample_size"],
                    "Significance": "Significant" if is_sig else "Not Sig",
                })

            ranked_df = pd.DataFrame(records)
            st.dataframe(
                ranked_df.style.format({
                    "Coefficient (r)": "{:+.4f}",
                    "p-value": "{:.4e}",
                    "Adj p-value (FDR)": "{:.4e}",
                    "Sample Size (N)": "{:,}",
                }),
                use_container_width=True,
                height=400,
            )

            csv_pairs = ranked_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Ranked Pairs (CSV)",
                data=csv_pairs,
                file_name=f"ranked_correlations_{method}.csv",
                mime="text/csv",
            )

    with tab_deepdive:
        st.markdown("#### Bivariate Variable Inspection")
        st.caption("Inspect scatter distribution and linear trend for any selected variable pair.")

        d1, d2, d3 = st.columns([1, 1, 1])
        with d1:
            pair_x = st.selectbox("Feature X", options=selected_cols, index=0)
        with d2:
            pair_y_opts = [c for c in selected_cols if c != pair_x] or selected_cols
            pair_y = st.selectbox("Feature Y", options=pair_y_opts, index=0)
        with d3:
            show_trend = st.checkbox("Overlay OLS Regression Line", value=True)

        if pair_x and pair_y:
            fig_scatter = plot_scatter(
                df,
                x_col=pair_x,
                y_col=pair_y,
                trendline=show_trend,
                title=f"Bivariate Scatter: {pair_x} vs {pair_y}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Pairwise summary metrics
            pair_r = corr_res.matrix.loc[pair_x, pair_y] if pair_x in corr_res.matrix.index and pair_y in corr_res.matrix.columns else 0.0
            p_val = corr_res.p_values.loc[pair_x, pair_y] if corr_res.p_values is not None else 1.0
            n_val = corr_res.sample_size_matrix.loc[pair_x, pair_y] if corr_res.sample_size_matrix is not None else len(df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Correlation (r)", f"{pair_r:+.4f}")
            m2.metric("Two-Tailed p-value", f"{p_val:.4e}")
            m3.metric("Paired Observations (N)", f"{int(n_val):,}")

            st.caption(
                "⚠️ **Methodological Boundary:** High correlation implies linear or monotonic co-variation, "
                "not causal dependency. Lurking confounders or common response variables may account for the observed association."
            )

    render_section_navigation("Correlation Explorer", position="bottom")
