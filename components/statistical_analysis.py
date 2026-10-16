"""Statistical Analysis UI component for EDA Intelligence Workbench.

Provides interactive tabs for extended descriptive statistics, normality diagnostics,
confidence intervals (parametric & bootstrap), and hypothesis testing / group comparisons.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_components import plot_boxplot, plot_histogram, plot_qq_plot
from components.navigation import render_section_navigation
from core.advanced_statistics import (
    compute_bootstrap_mean_ci,
    compute_cohens_d,
    compute_extended_descriptive_stats,
    compute_mean_confidence_interval,
    test_chi_square_independence,
    test_dagostino_pearson,
    test_kruskal_wallis,
    test_mann_whitney_u,
    test_one_way_anova,
    test_shapiro_wilk,
    test_welchs_ttest,
)
from core.profiling import DatasetProfile
from core.settings import AnalysisSettings


def render_statistical_analysis(
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: AnalysisSettings,
) -> None:
    """Render the Advanced Statistical Analysis panel."""
    st.markdown("### 📐 Advanced Statistical Analysis Engine")
    st.caption(
        "Rigorous statistical diagnostics covering distributional moments, parametric & "
        "non-parametric normality tests, confidence intervals, and hypothesis tests with assumption checks."
    )
    render_section_navigation("Statistical Analysis", position="top")

    numeric_cols = [
        c for c, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null and not p.is_constant
    ]
    cat_cols = [
        c for c, p in profile.column_profiles.items()
        if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 25
    ]

    tab_stats, tab_normality, tab_ci, tab_hypothesis = st.tabs([
        "📊 Extended Descriptive Statistics",
        "🔬 Normality Diagnostics",
        "🎯 Confidence Intervals",
        "⚖️ Group Comparisons & Hypothesis Tests",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXTENDED DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------
    with tab_stats:
        st.markdown("#### Comprehensive Distributional Statistics")
        st.caption(
            "Includes central tendencies, dispersion (std, IQR, variance, CV), percentiles, "
            "and shape parameters (Fisher-Pearson skewness, Pearson kurtosis)."
        )

        stats_df = compute_extended_descriptive_stats(df, profile)
        if stats_df.empty:
            st.info("No non-constant numeric features available for descriptive statistics.")
        else:
            # Filtering options
            filter_col1, filter_col2 = st.columns([2, 1])
            with filter_col1:
                search_term = st.text_input("Filter Columns by Name", placeholder="e.g. price, age...")
            with filter_col2:
                show_high_skew = st.checkbox("Show only high skew (|skew| > 1.0)", value=False)

            display_df = stats_df.copy()
            if search_term:
                display_df = display_df[display_df["Column Name"].str.contains(search_term, case=False, na=False)]
            if show_high_skew and "Skewness" in display_df.columns:
                display_df = display_df[display_df["Skewness"].abs() > 1.0]

            st.dataframe(
                display_df.style.format({
                    "Mean": "{:.3f}",
                    "Std Dev": "{:.3f}",
                    "Variance": "{:.3f}",
                    "Min": "{:.3f}",
                    "Q1 (25%)": "{:.3f}",
                    "Median (50%)": "{:.3f}",
                    "Q3 (75%)": "{:.3f}",
                    "Max": "{:.3f}",
                    "IQR": "{:.3f}",
                    "Coef of Variation": "{:.4f}",
                    "Skewness": "{:.3f}",
                    "Kurtosis": "{:.3f}",
                }, na_rep="—"),
                use_container_width=True,
                height=420,
            )

            # Export button
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Descriptive Statistics (CSV)",
                data=csv_bytes,
                file_name="descriptive_statistics.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------------
    # TAB 2: NORMALITY DIAGNOSTICS
    # -------------------------------------------------------------------------
    with tab_normality:
        st.markdown("#### Distribution Normality Diagnostics")
        st.caption(
            "Evaluates whether a continuous numeric feature follows a Gaussian (normal) distribution. "
            "H0: The sample was drawn from a normally distributed population."
        )

        if not numeric_cols:
            st.info("No numeric columns available for normality testing.")
        else:
            norm_col = st.selectbox("Select Numeric Feature for Normality Analysis", options=numeric_cols)
            series = pd.to_numeric(df[norm_col], errors="coerce").dropna()
            n = len(series)

            st.write(f"Sample size: **{n:,}** valid observations.")

            test_col1, test_col2 = st.columns(2)

            with test_col1:
                st.markdown("##### 1. Shapiro-Wilk Test")
                res_sw = test_shapiro_wilk(series, alpha=settings.alpha)
                if res_sw.status == "success":
                    sig_badge = "🔴 Reject H0 (Non-Normal)" if res_sw.p_value < settings.alpha else "🟢 Fail to Reject H0 (Consistent with Normal)"
                    st.metric("Test Statistic (W)", f"{res_sw.statistic:.4f}")
                    st.metric("p-value", f"{res_sw.p_value:.4e}", delta=sig_badge, delta_color="inverse" if res_sw.p_value < settings.alpha else "normal")
                    st.info(f"**Interpretation:** {res_sw.interpretation}")
                else:
                    st.warning(f"Shapiro-Wilk unavailable: {'; '.join(res_sw.warnings)}")

            with test_col2:
                st.markdown("##### 2. D'Agostino-Pearson Omnibus Test")
                res_dp = test_dagostino_pearson(series, alpha=settings.alpha)
                if res_dp.status == "success":
                    sig_badge = "🔴 Reject H0 (Non-Normal)" if res_dp.p_value < settings.alpha else "🟢 Fail to Reject H0 (Consistent with Normal)"
                    st.metric("Omnibus Statistic (K²)", f"{res_dp.statistic:.4f}")
                    st.metric("p-value", f"{res_dp.p_value:.4e}", delta=sig_badge, delta_color="inverse" if res_dp.p_value < settings.alpha else "normal")
                    st.info(f"**Interpretation:** {res_dp.interpretation}")
                else:
                    st.warning(f"D'Agostino-Pearson unavailable: {'; '.join(res_dp.warnings)}")

            st.markdown("##### Visual Normality Inspection")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                fig_hist = plot_histogram(df, col=norm_col, nbins=35, show_kde=True)
                st.plotly_chart(fig_hist, use_container_width=True)
            with plot_col2:
                fig_qq = plot_qq_plot(df, col=norm_col)
                st.plotly_chart(fig_qq, use_container_width=True)

            st.caption(
                "⚠️ **Methodological Boundary:** In large sample sizes (N > 1,000), even minuscule, practically "
                "inconsequential departures from normality will yield p < 0.001. Always examine visual Q-Q plots alongside p-values."
            )

    # -------------------------------------------------------------------------
    # TAB 3: CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    with tab_ci:
        st.markdown("#### Confidence Intervals for Population Mean")
        st.caption(
            "Computes parametric (Student's t) and non-parametric (Bootstrap Percentile) confidence intervals."
        )

        if not numeric_cols:
            st.info("No numeric columns available.")
        else:
            ci_col1, ci_col2, ci_col3 = st.columns(3)
            with ci_col1:
                target_col = st.selectbox("Numeric Column", options=numeric_cols, key="ci_target_col")
            with ci_col2:
                conf_level = st.selectbox("Confidence Level", options=[0.90, 0.95, 0.99], index=1)
            with ci_col3:
                n_boot = st.slider("Bootstrap Iterations", min_value=500, max_value=3000, value=settings.bootstrap_iterations, step=500)

            series = pd.to_numeric(df[target_col], errors="coerce").dropna()

            t_ci = compute_mean_confidence_interval(series, confidence_level=conf_level)
            b_ci = compute_bootstrap_mean_ci(
                series,
                confidence_level=conf_level,
                n_bootstrap=n_boot,
                random_seed=settings.random_seed,
            )

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("##### Parametric Student's t Interval")
                if t_ci.status == "success":
                    st.metric("Sample Mean", f"{t_ci.point_estimate:.4f}")
                    st.metric(f"{int(conf_level * 100)}% CI Lower Bound", f"{t_ci.ci_lower:.4f}")
                    st.metric(f"{int(conf_level * 100)}% CI Upper Bound", f"{t_ci.ci_upper:.4f}")
                    t_moe = f"±{t_ci.margin_of_error:.4f}" if t_ci.margin_of_error is not None else "N/A"
                    st.metric("Margin of Error", t_moe)
                    st.caption(f"Assumes independent sampling and asymptotic normality (N={t_ci.sample_size}).")
                else:
                    st.warning(f"Could not compute parametric CI: {'; '.join(getattr(t_ci, 'warnings', []))}")

            with res_col2:
                st.markdown("##### Non-Parametric Bootstrap Percentile Interval")
                if b_ci.status == "success":
                    st.metric("Sample Mean", f"{b_ci.point_estimate:.4f}")
                    st.metric(f"{int(conf_level * 100)}% Bootstrap Lower", f"{b_ci.ci_lower:.4f}")
                    st.metric(f"{int(conf_level * 100)}% Bootstrap Upper", f"{b_ci.ci_upper:.4f}")
                    b_moe = f"±{b_ci.margin_of_error:.4f}" if b_ci.margin_of_error is not None else "N/A (Empirical Percentiles)"
                    st.metric("Margin of Error", b_moe)
                    st.caption(f"Free from distributional shape assumptions (B={n_boot:,} iterations, seed={settings.random_seed}).")
                else:
                    st.warning(f"Could not compute bootstrap CI: {'; '.join(getattr(b_ci, 'warnings', []))}")

    # -------------------------------------------------------------------------
    # TAB 4: GROUP COMPARISONS & HYPOTHESIS TESTS
    # -------------------------------------------------------------------------
    with tab_hypothesis:
        st.markdown("#### Hypothesis Testing & Group Comparisons")
        st.caption(
            "Select continuous outcomes and categorical group factors to evaluate differences in location and association."
        )

        test_mode = st.radio(
            "Select Comparison Nature",
            options=[
                "Two-Group Comparison (Welch's t-test / Mann-Whitney U)",
                "Multi-Group Comparison (One-Way ANOVA / Kruskal-Wallis)",
                "Categorical Association (Chi-Square Independence Test)",
            ],
            horizontal=True,
        )

        if test_mode == "Two-Group Comparison (Welch's t-test / Mann-Whitney U)":
            if not numeric_cols or not cat_cols:
                st.info("Requires at least 1 numeric outcome and 1 categorical factor.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    metric_c = st.selectbox("Numeric Outcome Metric", options=numeric_cols, key="two_group_metric")
                    group_c = st.selectbox("Categorical Factor", options=cat_cols, key="two_group_factor")

                sub_clean = df[[metric_c, group_c]].dropna().copy()
                sub_clean[metric_c] = pd.to_numeric(sub_clean[metric_c], errors="coerce")
                sub_clean = sub_clean.dropna()

                unique_groups = list(sub_clean[group_c].astype(str).unique())
                if len(unique_groups) < 2:
                    st.warning("Selected categorical factor has fewer than 2 distinct levels.")
                else:
                    with c2:
                        g1 = st.selectbox("Group 1", options=unique_groups, index=0)
                        g2_opts = [g for g in unique_groups if g != g1]
                        g2 = st.selectbox("Group 2", options=g2_opts, index=0)

                    s1 = sub_clean[sub_clean[group_c].astype(str) == g1][metric_c].values
                    s2 = sub_clean[sub_clean[group_c].astype(str) == g2][metric_c].values

                    st.write(f"Sample Sizes: Group 1 ({g1}) = **{len(s1)}**, Group 2 ({g2}) = **{len(s2)}**.")

                    col_w, col_m = st.columns(2)
                    with col_w:
                        st.markdown("##### Welch's t-Test (Unequal Variances)")
                        w_res = test_welchs_ttest(s1, s2, alpha=settings.alpha)
                        if w_res.status == "success":
                            d_val = compute_cohens_d(s1, s2)
                            st.metric("t-Statistic", f"{w_res.statistic:.4f}")
                            st.metric("p-value", f"{w_res.p_value:.4e}")
                            st.metric("Cohen's d Effect Size", f"{d_val:.3f}")
                            st.info(w_res.interpretation)
                        else:
                            st.warning("; ".join(w_res.warnings))

                    with col_m:
                        st.markdown("##### Mann-Whitney U Test (Rank Sum)")
                        m_res = test_mann_whitney_u(s1, s2, alpha=settings.alpha)
                        if m_res.status == "success":
                            st.metric("U-Statistic", f"{m_res.statistic:.1f}")
                            st.metric("p-value", f"{m_res.p_value:.4e}")
                            st.info(m_res.interpretation)
                        else:
                            st.warning("; ".join(m_res.warnings))

                    # Group box plot
                    fig_comp = plot_boxplot(
                        sub_clean[sub_clean[group_c].astype(str).isin([g1, g2])],
                        col=metric_c,
                        title=f"{metric_c} Distribution: {g1} vs {g2}",
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

        elif test_mode == "Multi-Group Comparison (One-Way ANOVA / Kruskal-Wallis)":
            if not numeric_cols or not cat_cols:
                st.info("Requires at least 1 numeric metric and 1 categorical factor.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    metric_c = st.selectbox("Continuous Outcome Metric", options=numeric_cols, key="multi_metric")
                with c2:
                    group_c = st.selectbox("Categorical Group Factor", options=cat_cols, key="multi_factor")

                sub_clean = df[[metric_c, group_c]].dropna().copy()
                sub_clean[metric_c] = pd.to_numeric(sub_clean[metric_c], errors="coerce")
                sub_clean[group_c] = sub_clean[group_c].astype(str)
                sub_clean = sub_clean.dropna()

                group_series = [group.values for _, group in sub_clean.groupby(group_c)[metric_c] if len(group) >= 2]

                if len(group_series) < 2:
                    st.warning("Insufficient groups with >= 2 observations.")
                else:
                    st.write(f"Evaluating across **{len(group_series)}** group levels.")
                    col_a, col_k = st.columns(2)
                    with col_a:
                        st.markdown("##### One-Way ANOVA (Parametric)")
                        a_res = test_one_way_anova(group_series, alpha=settings.alpha)
                        if a_res.status == "success":
                            st.metric("F-Statistic", f"{a_res.statistic:.4f}")
                            st.metric("p-value", f"{a_res.p_value:.4e}")
                            st.info(a_res.interpretation)
                        else:
                            st.warning("; ".join(a_res.warnings))

                    with col_k:
                        st.markdown("##### Kruskal-Wallis H-Test (Non-Parametric)")
                        k_res = test_kruskal_wallis(group_series, alpha=settings.alpha)
                        if k_res.status == "success":
                            st.metric("H-Statistic", f"{k_res.statistic:.4f}")
                            st.metric("p-value", f"{k_res.p_value:.4e}")
                            st.info(k_res.interpretation)
                        else:
                            st.warning("; ".join(k_res.warnings))

        else:
            # Categorical Association
            if len(cat_cols) < 2:
                st.info("Requires at least 2 categorical features for Chi-Square independence analysis.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    f1 = st.selectbox("Categorical Factor A", options=cat_cols, index=0)
                with c2:
                    f2_opts = [c for c in cat_cols if c != f1] or cat_cols
                    f2 = st.selectbox("Categorical Factor B", options=f2_opts, index=0)

                sub_cat = df[[f1, f2]].dropna().copy()
                sub_cat[f1] = sub_cat[f1].astype(str)
                sub_cat[f2] = sub_cat[f2].astype(str)

                # Limit cardinality to avoid massive contingency tables
                top_f1 = sub_cat[f1].value_counts().head(8).index
                top_f2 = sub_cat[f2].value_counts().head(8).index
                sub_cat = sub_cat[sub_cat[f1].isin(top_f1) & sub_cat[f2].isin(top_f2)]

                contingency = pd.crosstab(sub_cat[f1], sub_cat[f2])

                st.markdown("##### Observed Contingency Table")
                st.dataframe(contingency, use_container_width=True)

                chi_res = test_chi_square_independence(contingency.values, alpha=settings.alpha)
                if chi_res.status == "success":
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chi-Square (χ²)", f"{chi_res.statistic:.4f}")
                    m2.metric("p-value", f"{chi_res.p_value:.4e}")
                    m3.metric("Degrees of Freedom", str(chi_res.degrees_of_freedom))
                    st.info(chi_res.interpretation)
                    if chi_res.warnings:
                        st.warning("⚠️ " + " | ".join(chi_res.warnings))
                else:
                    st.warning("; ".join(chi_res.warnings))

    render_section_navigation("Statistical Analysis", position="bottom")
