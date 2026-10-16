"""Statistical Outlier Analysis UI component for EDA Intelligence Workbench.

Provides multi-method outlier detection (Tukey IQR, Z-Score, Median Absolute Deviation MAD),
threshold visualizations, flagged record inspection, and methodological explanations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.chart_components import (
    COLOR_PALETTE,
    apply_dark_theme,
    plot_boxplot,
    plot_histogram,
)
from core.outlier_analysis import (
    analyze_all_numeric_outliers,
    detect_iqr_outliers,
    detect_mad_outliers,
    detect_zscore_outliers,
)
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings


def render_outlier_panel(
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: AnalysisSettings,
) -> None:
    """Render the Statistical Outlier Analysis panel."""
    st.markdown("### 🎯 Statistical Outlier Investigation")
    st.caption(
        "Detect and evaluate extreme observations across continuous distributions using "
        "Tukey's Interquartile Range (IQR), Standard Z-Score, and Median Absolute Deviation (MAD)."
    )

    numeric_cols = [
        c for c, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null and not p.is_constant
    ]

    if not numeric_cols:
        st.warning("⚠️ No non-constant numeric features available for outlier detection.")
        return

    # Configuration controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        method = st.selectbox(
            "Detection Method",
            options=["iqr", "zscore", "mad"],
            index=["iqr", "zscore", "mad"].index(settings.default_outlier_method)
            if settings.default_outlier_method in ["iqr", "zscore", "mad"]
            else 0,
            format_func=lambda m: {
                "iqr": "Tukey IQR Fences (Non-Parametric)",
                "zscore": "Z-Score (Standard Normal)",
                "mad": "MAD / Modified Z-Score (Robust)",
            }[m],
        )

    with c2:
        if method == "iqr":
            param_val = st.slider("IQR Multiplier (k)", min_value=1.0, max_value=3.5, value=float(settings.iqr_multiplier), step=0.1)
        elif method == "zscore":
            param_val = st.slider("Z-Score Threshold (σ)", min_value=1.5, max_value=5.0, value=float(settings.zscore_threshold), step=0.25)
        else:
            param_val = st.slider("MAD Modified Z Threshold", min_value=2.0, max_value=6.0, value=float(settings.mad_threshold), step=0.25)

    with c3:
        target_feature = st.selectbox("Feature for Detailed Drilldown", options=numeric_cols)

    # Compute outlier results across all numeric features
    outlier_df = analyze_all_numeric_outliers(
        df,
        columns=numeric_cols,
        method=method,
        param_value=param_val,
    )

    tab_summary, tab_drilldown, tab_flagged, tab_methodology = st.tabs([
        "📋 Dataset Outlier Overview",
        "🔬 Single-Feature Diagnostic",
        "🚩 Flagged Anomalous Records",
        "📖 Methodological Governance",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Dataset Overview
    # -------------------------------------------------------------------------
    with tab_summary:
        st.markdown(f"#### Multi-Feature Outlier Summary ({method.upper()})")

        total_flagged_cells = outlier_df["Outlier_Count"].sum()
        cols_with_outliers = (outlier_df["Outlier_Count"] > 0).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Flagged Values", f"{total_flagged_cells:,}")
        m2.metric("Features with Outliers", f"{cols_with_outliers} of {len(numeric_cols)}")
        max_pct_col = outlier_df.sort_values("Outlier_Percentage", ascending=False).iloc[0] if not outlier_df.empty else None
        if max_pct_col is not None and max_pct_col["Outlier_Count"] > 0:
            m3.metric("Highest Outlier Rate", f"{max_pct_col['Outlier_Percentage']:.1f}% ({max_pct_col['Column_Name']})")
        else:
            m3.metric("Highest Outlier Rate", "0.0%")

        # Bar chart of outlier rates
        if total_flagged_cells > 0:
            active_outliers = outlier_df[outlier_df["Outlier_Count"] > 0].sort_values("Outlier_Count", ascending=False)
            fig_bar = px.bar(
                active_outliers.head(20),
                x="Column_Name",
                y="Outlier_Count",
                text="Outlier_Percentage",
                color="Outlier_Count",
                color_continuous_scale="Purples",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar = apply_dark_theme(fig_bar, title=f"Outlier Counts by Feature ({method.upper()})")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            outlier_df.style.format({
                "Outlier_Count": "{:,}",
                "Outlier_Percentage": "{:.2f}%",
                "Lower_Bound": "{:.3f}",
                "Upper_Bound": "{:.3f}",
                "Valid_Observations": "{:,}",
            }),
            use_container_width=True,
        )

        csv_out = outlier_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Outlier Summary (CSV)",
            data=csv_out,
            file_name=f"outlier_summary_{method}.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # TAB 2: Single-Feature Drilldown
    # -------------------------------------------------------------------------
    with tab_drilldown:
        st.markdown(f"#### Diagnostic Drilldown: `{target_feature}`")

        clean_s = pd.to_numeric(df[target_feature], errors="coerce").dropna()
        if method == "iqr":
            diag_res = detect_iqr_outliers(clean_s, k=param_val)
        elif method == "zscore":
            diag_res = detect_zscore_outliers(clean_s, threshold=param_val)
        else:
            diag_res = detect_mad_outliers(clean_s, threshold=param_val)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Outlier Count", f"{diag_res.outlier_count:,}")
        d2.metric("Outlier Rate", f"{diag_res.outlier_percentage:.2f}%")
        d3.metric("Lower Threshold", f"{diag_res.lower_threshold:.3f}")
        d4.metric("Upper Threshold", f"{diag_res.upper_threshold:.3f}")

        p1, p2 = st.columns(2)
        with p1:
            fig_b = plot_boxplot(df, col=target_feature, points="all")
            st.plotly_chart(fig_b, use_container_width=True)
        with p2:
            fig_h = plot_histogram(df, col=target_feature, nbins=35, show_kde=True)
            # Add threshold reference lines
            fig_h.add_vline(x=diag_res.lower_threshold, line_dash="dash", line_color=COLOR_PALETTE[4], annotation_text="Lower Bound")
            fig_h.add_vline(x=diag_res.upper_threshold, line_dash="dash", line_color=COLOR_PALETTE[5], annotation_text="Upper Bound")
            st.plotly_chart(fig_h, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Flagged Records
    # -------------------------------------------------------------------------
    with tab_flagged:
        st.markdown(f"#### Flagged Records for `{target_feature}`")
        clean_s = pd.to_numeric(df[target_feature], errors="coerce").dropna()
        if method == "iqr":
            diag_res = detect_iqr_outliers(clean_s, k=param_val)
        elif method == "zscore":
            diag_res = detect_zscore_outliers(clean_s, threshold=param_val)
        else:
            diag_res = detect_mad_outliers(clean_s, threshold=param_val)

        if not diag_res.flagged_indices:
            st.success(f"No records flagged as outliers in `{target_feature}` under {method.upper()} criteria.")
        else:
            st.caption(f"Displaying {len(diag_res.flagged_indices):,} flagged rows.")
            flagged_rows = df.loc[diag_res.flagged_indices]
            st.dataframe(flagged_rows, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Methodological Governance
    # -------------------------------------------------------------------------
    with tab_methodology:
        st.markdown("#### Outlier Governance, Assumptions & Boundaries")
        st.markdown("""
        **1. Tukey IQR Fences**
        * *Formula:* $[Q1 - k \\cdot IQR, Q3 + k \\cdot IQR]$, where $IQR = Q3 - Q1$. Standard $k=1.5$, extreme $k=3.0$.
        * *Assumptions:* None. Highly resistant to existing extreme values.
        * *Limitations:* Can flag legitimate heavy-tail distribution values as outliers.

        **2. Standard Z-Score**
        * *Formula:* $z = \\frac{x - \\mu}{\\sigma}$. Flagged if $|z| > \\text{threshold}$.
        * *Assumptions:* Population follows a Gaussian normal distribution.
        * *Limitations:* Both sample mean $\\mu$ and standard deviation $\\sigma$ are heavily distorted by the very outliers being tested (masking effect).

        **3. Median Absolute Deviation (MAD)**
        * *Formula:* $\\text{MAD} = \\text{median}(|x_i - \\text{median}(x)|)$. Modified Z-score: $0.6745 \\cdot \\frac{|x - \\text{median}|}{\\text{MAD}}$.
        * *Assumptions:* Symmetric distribution around the median.
        * *Limitations:* If more than 50% of the sample shares identical values, $\\text{MAD} = 0$, rendering the statistic undefined.

        > ⚠️ **Critical Analytical Principle:** Never delete or alter flagged records automatically.
        > Statistically unusual observations may reflect breakthrough signals, rare real-world phenomena,
        > or heavy-tailed processes rather than measurement error.
        """)
