"""Interactive Chart Explorer component with centralized registry integration.

Offers all 7 chart families, validates dataset compatibility before rendering,
dynamically displays applicable controls, and avoids broken placeholders.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_components import (
    plot_autocorrelation_acf,
    plot_box_by_category,
    plot_boxplot,
    plot_categorical_bar,
    plot_correlation_heatmap,
    plot_density_contour,
    plot_donut_chart,
    plot_ecdf,
    plot_grouped_bar,
    plot_histogram,
    plot_kde_density,
    plot_lag_scatter,
    plot_mean_median_comparison,
    plot_pairwise_scatter,
    plot_parallel_coordinates,
    plot_pareto_chart,
    plot_pca_2d,
    plot_pca_3d,
    plot_qq_plot,
    plot_scatter,
    plot_scatter_3d,
    plot_stacked_bar_100,
    plot_sunburst,
    plot_time_series,
    plot_time_series_area,
    plot_treemap,
    plot_violin,
    plot_violin_by_category,
)
from components.navigation import render_section_navigation
from core.chart_registry import (
    CHART_REGISTRY,
    evaluate_chart_compatibility,
    get_available_chart_families,
    get_charts_by_family,
)
from core.profiling import DatasetProfile


def render_chart_explorer(df: pd.DataFrame, profile: DatasetProfile) -> None:
    """Render the centralized Visualization Studio / Chart Explorer."""
    st.markdown("### 📊 Visualization Studio & Dynamic Chart Explorer")
    st.caption(
        "A centralized visualization engine supporting 25+ chart types across 7 families. "
        "Each chart evaluates data compatibility guards dynamically before rendering."
    )
    render_section_navigation("Visualization Studio", position="top")

    # 1. Family selection
    families = get_available_chart_families()
    selected_family = st.radio(
        "Select Chart Family",
        options=families,
        horizontal=True,
        index=0,
    )

    # 2. Charts within selected family
    family_charts = get_charts_by_family(selected_family)
    chart_options = {c.chart_id: f"{c.display_name}" for c in family_charts}

    selected_chart_id = st.selectbox(
        "Choose Visualization Type",
        options=list(chart_options.keys()),
        format_func=lambda cid: chart_options[cid],
    )

    chart_def = CHART_REGISTRY[selected_chart_id]

    # Display chart purpose and guardrail info
    st.info(f"**Purpose:** {chart_def.purpose}\n\n*Constraints:* {chart_def.resource_constraints}")

    # 3. Check compatibility
    is_compatible, reason = evaluate_chart_compatibility(selected_chart_id, df, profile)
    if not is_compatible:
        st.warning(
            f"⚠️ **Incompatible with Current Dataset:** {reason}\n\n"
            f"Required column types: `{', '.join(chart_def.required_column_types)}`."
        )
        return

    # Extract column subsets
    numeric_cols = [
        c for c, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null and not p.is_constant
    ]
    cat_cols = [
        c for c, p in profile.column_profiles.items()
        if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 25
    ]
    dt_cols = [
        c for c, p in profile.column_profiles.items()
        if p.semantic_type == "Datetime" or p.min_date is not None
    ]

    # 4. Dynamic parameter controls based on chart_id
    st.markdown("#### Configure Chart Parameters")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    fig = None

    # Family: Distributions
    if chart_def.chart_id == "histogram_box":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        with ctrl_col2:
            nbins = st.slider("Bins", min_value=10, max_value=100, value=30, step=5)
        with ctrl_col3:
            show_kde = st.checkbox("Overlay Box Marginal", value=True)
        if col:
            fig = plot_histogram(df, col=col, nbins=nbins, show_kde=show_kde)

    elif chart_def.chart_id == "boxplot":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        with ctrl_col2:
            pts = st.selectbox("Points Display", options=["outliers", "all", "suspectedoutliers", False])
        if col:
            fig = plot_boxplot(df, col=col, points=pts)

    elif chart_def.chart_id == "violin_plot":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        with ctrl_col2:
            box = st.checkbox("Show Embedded Box", value=True)
        if col:
            fig = plot_violin(df, col=col, box=box)

    elif chart_def.chart_id == "ecdf_plot":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        if col:
            fig = plot_ecdf(df, col=col)

    elif chart_def.chart_id == "kde_density":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        if col:
            fig = plot_kde_density(df, col=col)

    elif chart_def.chart_id == "qq_plot":
        with ctrl_col1:
            col = st.selectbox("Numeric Column", options=numeric_cols)
        if col:
            fig = plot_qq_plot(df, col=col)

    elif chart_def.chart_id == "pareto_chart":
        with ctrl_col1:
            col = st.selectbox("Categorical Column", options=cat_cols)
        with ctrl_col2:
            max_c = st.slider("Max Categories", min_value=5, max_value=30, value=15)
        if col:
            fig = plot_pareto_chart(df, col=col, max_categories=max_c)

    # Family: Relationships
    elif chart_def.chart_id == "scatter_regression":
        with ctrl_col1:
            x_col = st.selectbox("X-Axis (Numeric)", options=numeric_cols, index=0)
            y_opts = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = st.selectbox("Y-Axis (Numeric)", options=y_opts, index=0)
        with ctrl_col2:
            color_c = st.selectbox("Color Grouping (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        with ctrl_col3:
            trend = st.checkbox("Add OLS Linear Fit", value=True)
        if x_col and y_col:
            fig = plot_scatter(df, x_col=x_col, y_col=y_col, color_col=color_col, trendline=trend)

    elif chart_def.chart_id == "bubble_chart":
        with ctrl_col1:
            x_col = st.selectbox("X-Axis (Numeric)", options=numeric_cols, index=0)
            y_opts = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = st.selectbox("Y-Axis (Numeric)", options=y_opts, index=0)
        with ctrl_col2:
            sz_opts = [c for c in numeric_cols if c not in (x_col, y_col)] or numeric_cols
            sz_col = st.selectbox("Bubble Size (Numeric)", options=sz_opts, index=0)
        with ctrl_col3:
            color_c = st.selectbox("Color Grouping (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        if x_col and y_col and sz_col:
            fig = plot_scatter(df, x_col=x_col, y_col=y_col, size_col=sz_col, color_col=color_col)

    elif chart_def.chart_id == "density_contour":
        with ctrl_col1:
            x_col = st.selectbox("X-Axis (Numeric)", options=numeric_cols, index=0)
            y_opts = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = st.selectbox("Y-Axis (Numeric)", options=y_opts, index=0)
        if x_col and y_col:
            fig = plot_density_contour(df, x_col=x_col, y_col=y_col)

    elif chart_def.chart_id == "correlation_heatmap":
        with ctrl_col1:
            sub_nums = st.multiselect(
                "Select Features (Default: All Bounded)",
                options=numeric_cols,
                default=numeric_cols[:15],
            )
        with ctrl_col2:
            corr_meth = st.selectbox("Method", options=["spearman", "pearson", "kendall"])
        if len(sub_nums) >= 2:
            from core.correlations import compute_correlation_matrix
            res = compute_correlation_matrix(df, columns=sub_nums, method=corr_meth)
            fig = plot_correlation_heatmap(res.matrix, title=f"{corr_meth.capitalize()} Correlation Heatmap")

    elif chart_def.chart_id == "pairwise_scatter":
        with ctrl_col1:
            sub_nums = st.multiselect(
                "Select 3 to 5 Numeric Features",
                options=numeric_cols,
                default=numeric_cols[:4],
                max_selections=5,
            )
        with ctrl_col2:
            color_c = st.selectbox("Color Grouping (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        if len(sub_nums) >= 3:
            fig = plot_pairwise_scatter(df, columns=sub_nums, color_col=color_col)

    # Family: Categoricals
    elif chart_def.chart_id in ("vertical_bar", "horizontal_bar"):
        is_horiz = chart_def.chart_id == "horizontal_bar"
        with ctrl_col1:
            col = st.selectbox("Categorical Column", options=cat_cols)
        with ctrl_col2:
            max_c = st.slider("Top N Categories", min_value=5, max_value=30, value=15)
        if col:
            fig, is_agg, other_n = plot_categorical_bar(
                df, col=col, max_categories=max_c, horizontal=is_horiz
            )
            if is_agg:
                st.caption(f"ℹ️ Categories beyond top {max_c - 1} aggregated into 'Other' ({other_n:,} rows).")

    elif chart_def.chart_id == "grouped_bar":
        with ctrl_col1:
            c1 = st.selectbox("Primary Category", options=cat_cols, index=0)
            c2_opts = [c for c in cat_cols if c != c1] or cat_cols
            c2 = st.selectbox("Secondary Category", options=c2_opts, index=0)
        with ctrl_col2:
            max_c = st.slider("Max Levels per Dimension", min_value=3, max_value=15, value=8)
        if c1 and c2:
            fig = plot_grouped_bar(df, cat_col1=c1, cat_col2=c2, max_categories=max_c)

    elif chart_def.chart_id == "stacked_bar_100":
        with ctrl_col1:
            c1 = st.selectbox("Base Grouping Category", options=cat_cols, index=0)
            c2_opts = [c for c in cat_cols if c != c1] or cat_cols
            c2 = st.selectbox("Segmenting Category", options=c2_opts, index=0)
        with ctrl_col2:
            max_c = st.slider("Max Levels per Dimension", min_value=3, max_value=15, value=8)
        if c1 and c2:
            fig = plot_stacked_bar_100(df, cat_col1=c1, cat_col2=c2, max_categories=max_c)

    elif chart_def.chart_id == "treemap":
        with ctrl_col1:
            cat_c = st.selectbox("Primary Dimension", options=cat_cols)
        with ctrl_col2:
            val_c = st.selectbox("Optional Value Weight", options=["Record Count"] + numeric_cols)
            val_col = None if val_c == "Record Count" else val_c
        if cat_c:
            fig = plot_treemap(df, cat_col=cat_c, val_col=val_col)

    elif chart_def.chart_id == "sunburst":
        with ctrl_col1:
            paths = st.multiselect(
                "Hierarchy Dimensions (2-3 categories)",
                options=cat_cols,
                default=cat_cols[:2],
                max_selections=3,
            )
        with ctrl_col2:
            val_c = st.selectbox("Optional Numeric Weight", options=["Record Count"] + numeric_cols)
            val_col = None if val_c == "Record Count" else val_c
        if len(paths) >= 2:
            fig = plot_sunburst(df, path_cols=paths, val_col=val_col)

    elif chart_def.chart_id == "donut_chart":
        with ctrl_col1:
            col = st.selectbox("Nominal Category (<= 8 levels recommended)", options=cat_cols)
        with ctrl_col2:
            max_c = st.slider("Max Slices", min_value=4, max_value=12, value=8)
        if col:
            fig = plot_donut_chart(df, col=col, max_categories=max_c)

    # Family: Time-Series
    elif chart_def.chart_id in ("time_series_line", "time_series_area"):
        with ctrl_col1:
            d_col = st.selectbox("Datetime Column", options=dt_cols)
            v_col = st.selectbox("Metric Column", options=numeric_cols)
        with ctrl_col2:
            freq = st.selectbox("Resampling Aggregation", options=["Raw", "Daily", "Weekly", "Monthly"])
        with ctrl_col3:
            show_roll = st.checkbox("Overlay Rolling Window", value=True)
            w_size = st.slider("Window Size", min_value=3, max_value=30, value=7)
        if d_col and v_col:
            if chart_def.chart_id == "time_series_line":
                fig = plot_time_series(
                    df, date_col=d_col, val_col=v_col, freq=freq,
                    show_rolling=show_roll, rolling_window=w_size,
                )
            else:
                fig = plot_time_series_area(df, date_col=d_col, val_col=v_col, freq=freq)

    elif chart_def.chart_id == "autocorrelation_acf":
        with ctrl_col1:
            d_col = st.selectbox("Datetime Column", options=dt_cols)
            v_col = st.selectbox("Metric Column", options=numeric_cols)
        with ctrl_col2:
            lags = st.slider("Max Lag Orders", min_value=5, max_value=60, value=30)
        if d_col and v_col:
            fig = plot_autocorrelation_acf(df, date_col=d_col, val_col=v_col, max_lags=lags)

    elif chart_def.chart_id == "lag_scatter":
        with ctrl_col1:
            d_col = st.selectbox("Datetime Column", options=dt_cols)
            v_col = st.selectbox("Metric Column", options=numeric_cols)
        with ctrl_col2:
            lag_k = st.slider("Lag Step (k)", min_value=1, max_value=14, value=1)
        if d_col and v_col:
            fig = plot_lag_scatter(df, date_col=d_col, val_col=v_col, lag=lag_k)

    # Family: Comparisons
    elif chart_def.chart_id == "category_box":
        with ctrl_col1:
            cat_c = st.selectbox("Categorical Group", options=cat_cols)
            val_c = st.selectbox("Numeric Metric", options=numeric_cols)
        with ctrl_col2:
            max_c = st.slider("Max Groups", min_value=3, max_value=20, value=10)
        if cat_c and val_c:
            fig = plot_box_by_category(df, cat_col=cat_c, val_col=val_c, max_categories=max_c)

    elif chart_def.chart_id == "category_violin":
        with ctrl_col1:
            cat_c = st.selectbox("Categorical Group", options=cat_cols)
            val_c = st.selectbox("Numeric Metric", options=numeric_cols)
        with ctrl_col2:
            max_c = st.slider("Max Groups", min_value=3, max_value=20, value=10)
        if cat_c and val_c:
            fig = plot_violin_by_category(df, cat_col=cat_c, val_col=val_c, max_categories=max_c)

    elif chart_def.chart_id == "mean_median_bars":
        with ctrl_col1:
            cat_c = st.selectbox("Categorical Group", options=cat_cols)
            val_c = st.selectbox("Numeric Metric", options=numeric_cols)
        with ctrl_col2:
            max_c = st.slider("Max Groups", min_value=3, max_value=20, value=10)
        if cat_c and val_c:
            fig = plot_mean_median_comparison(df, cat_col=cat_c, val_col=val_c, max_categories=max_c)

    # Family: Multivariate
    elif chart_def.chart_id == "parallel_coordinates":
        with ctrl_col1:
            cols = st.multiselect(
                "Select 3 to 6 Numeric Dimensions",
                options=numeric_cols,
                default=numeric_cols[:4],
                max_selections=6,
            )
        with ctrl_col2:
            color_c = st.selectbox("Color Mapping Variable (Optional)", options=["None"] + numeric_cols)
            color_col = None if color_c == "None" else color_c
        if len(cols) >= 3:
            fig = plot_parallel_coordinates(df, cols=cols, color_col=color_col)

    elif chart_def.chart_id == "pca_2d":
        with ctrl_col1:
            cols = st.multiselect(
                "Continuous Features to Standardize & Project",
                options=numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))],
            )
        with ctrl_col2:
            color_c = st.selectbox("Color Cluster Legend (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        if len(cols) >= 2:
            fig, evr = plot_pca_2d(df, cols=cols, color_col=color_col)
            if evr:
                st.caption(
                    f"💡 **Variance Explained:** PC1: `{evr[0]:.1f}%`, PC2: `{evr[1]:.1f}%` "
                    f"(Cumulative: `{sum(evr):.1f}%`). Features were centered and scaled (Z-score)."
                )

    # Family: 3D Visualizations
    elif chart_def.chart_id == "scatter_3d":
        with ctrl_col1:
            x_col = st.selectbox("X Dimension", options=numeric_cols, index=0)
            y_opts = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = st.selectbox("Y Dimension", options=y_opts, index=0)
            z_opts = [c for c in numeric_cols if c not in (x_col, y_col)] or numeric_cols
            z_col = st.selectbox("Z Dimension", options=z_opts, index=0)
        with ctrl_col2:
            color_c = st.selectbox("Color Grouping (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        st.warning("⚠️ **3D Plotting Note:** 3D projections can obscure overlapping points. Rotate carefully.")
        if x_col and y_col and z_col:
            fig = plot_scatter_3d(df, x_col=x_col, y_col=y_col, z_col=z_col, color_col=color_col)

    elif chart_def.chart_id == "pca_3d":
        with ctrl_col1:
            cols = st.multiselect(
                "Continuous Features for 3D PCA",
                options=numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))],
            )
        with ctrl_col2:
            color_c = st.selectbox("Color Cluster Legend (Optional)", options=["None"] + cat_cols)
            color_col = None if color_c == "None" else color_c
        if len(cols) >= 3:
            fig, evr = plot_pca_3d(df, cols=cols, color_col=color_col)
            if evr:
                st.caption(
                    f"💡 **3D Variance Explained:** PC1: `{evr[0]:.1f}%`, PC2: `{evr[1]:.1f}%`, "
                    f"PC3: `{evr[2]:.1f}%` (Cumulative: `{sum(evr):.1f}%`)."
                )

    # 5. Render figure
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select parameters above to render the chart.")

    render_section_navigation("Visualization Studio", position="bottom")
