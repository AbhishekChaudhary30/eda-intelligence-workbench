"""Reproducible Analysis Settings UI component for EDA Intelligence Workbench.

Provides validated parameter tuning across statistical thresholds, outlier bounds,
visualization defaults, and random seeds with full JSON import and export for scientific reproducibility.
"""

import streamlit as st

from core.settings import AnalysisSettings


def render_analysis_settings(settings: AnalysisSettings) -> AnalysisSettings:
    """Render the Analysis Settings & Reproducibility panel."""
    st.markdown("### ⚙️ Analysis Configuration & Reproducibility")
    st.caption(
        "Fine-tune statistical assumptions, outlier thresholds, random seeds, and missingness tokens. "
        "Settings can be saved to or loaded from a JSON configuration file to guarantee reproducible analytical runs."
    )

    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        # Download current settings as JSON
        config_json = settings.to_json()
        st.download_button(
            "📥 Download Analysis Config (JSON)",
            data=config_json,
            file_name="eda_analysis_config.json",
            mime="application/json",
            help="Export complete parameter configuration without data payload.",
        )

    with col_exp2:
        # Upload settings JSON
        uploaded_cfg = st.file_uploader("📤 Restore Config from JSON", type=["json"], label_visibility="collapsed")
        if uploaded_cfg is not None:
            try:
                content = uploaded_cfg.read().decode("utf-8")
                new_settings = AnalysisSettings.from_json(content)
                st.session_state["analysis_settings"] = new_settings
                st.success("Successfully loaded analysis configuration!")
                st.rerun()
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Failed to load configuration: {e}")

    st.markdown("---")

    # Basic Settings
    st.markdown("#### 1. Core Statistical Thresholds")
    b1, b2, b3 = st.columns(3)
    with b1:
        corr_method = st.selectbox(
            "Default Correlation Metric",
            options=["spearman", "pearson", "kendall"],
            index=["spearman", "pearson", "kendall"].index(settings.default_correlation_method),
            format_func=lambda m: {"spearman": "Spearman (Rank)", "pearson": "Pearson (Linear)", "kendall": "Kendall Tau"}[m],
        )
    with b2:
        corr_thresh = st.slider(
            "Minimum |r| Correlation Filter",
            min_value=0.0,
            max_value=0.8,
            value=float(settings.correlation_threshold),
            step=0.05,
        )
    with b3:
        alpha = st.selectbox(
            "Hypothesis Test Alpha (Significance Level)",
            options=[0.01, 0.05, 0.10],
            index=[0.01, 0.05, 0.10].index(settings.alpha) if settings.alpha in [0.01, 0.05, 0.10] else 1,
            format_func=lambda a: f"α = {a} ({int((1-a)*100)}% Confidence)",
        )

    st.markdown("#### 2. Outlier Boundary Parameters")
    o1, o2, o3 = st.columns(3)
    with o1:
        out_method = st.selectbox(
            "Primary Outlier Engine",
            options=["iqr", "zscore", "mad"],
            index=["iqr", "zscore", "mad"].index(settings.default_outlier_method),
            format_func=lambda m: {"iqr": "Tukey IQR Fences", "zscore": "Standard Z-Score", "mad": "Median Absolute Deviation"}[m],
        )
    with o2:
        iqr_mult = st.slider(
            "Tukey IQR Multiplier (k)",
            min_value=1.0,
            max_value=3.5,
            value=float(settings.iqr_multiplier),
            step=0.1,
            help="1.5 = Standard outlier threshold, 3.0 = Extreme outlier fence",
        )
    with o3:
        z_thresh = st.slider(
            "Z-Score Threshold (σ)",
            min_value=2.0,
            max_value=5.0,
            value=float(settings.zscore_threshold),
            step=0.25,
            help="Number of standard deviations from the sample mean",
        )

    # Advanced Progressive Disclosure
    with st.expander("🛠️ Advanced Computation & Reproducibility Settings", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            mad_thresh = st.slider(
                "MAD Modified Z Threshold",
                min_value=2.0,
                max_value=6.0,
                value=float(settings.mad_threshold),
                step=0.25,
            )
            boot_iters = st.slider(
                "Bootstrap CI Resampling Iterations",
                min_value=500,
                max_value=3000,
                value=int(settings.bootstrap_iterations),
                step=250,
            )
        with a2:
            seed = st.number_input(
                "Deterministic Random Seed",
                min_value=1,
                max_value=999999,
                value=int(settings.random_seed),
                step=1,
                help="Ensures identical bootstrap intervals and subsampling across sessions.",
            )
            max_cats = st.slider(
                "Max Categorical Bar Levels (Top N)",
                min_value=5,
                max_value=30,
                value=int(settings.max_categorical_levels),
                step=5,
            )
        with a3:
            ts_freq = st.selectbox(
                "Default Time-Series Frequency",
                options=["Raw", "Daily", "Weekly", "Monthly"],
                index=["Raw", "Daily", "Weekly", "Monthly"].index(settings.default_time_series_freq)
                if settings.default_time_series_freq in ["Raw", "Daily", "Weekly", "Monthly"]
                else 0,
            )
            ts_window = st.slider(
                "Default Rolling Average Window",
                min_value=3,
                max_value=30,
                value=int(settings.rolling_window_size),
                step=1,
            )

        tokens_str = st.text_input(
            "Disguised Missing-Data Text Tokens (Comma-separated)",
            value=", ".join(settings.custom_missing_tokens),
            help="String values recognized during missing-data audit as disguised nulls.",
        )
        custom_tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]

    # Apply button / Reset Defaults button
    st.markdown("---")
    act1, act2, _ = st.columns([1.2, 1.2, 2.6])
    with act1:
        if st.button("💾 Save Settings", type="primary"):
            settings.default_correlation_method = corr_method
            settings.correlation_threshold = corr_thresh
            settings.alpha = alpha
            settings.default_outlier_method = out_method
            settings.iqr_multiplier = iqr_mult
            settings.zscore_threshold = z_thresh
            settings.mad_threshold = mad_thresh
            settings.bootstrap_iterations = boot_iters
            settings.random_seed = seed
            settings.max_categorical_levels = max_cats
            settings.default_time_series_freq = ts_freq
            settings.rolling_window_size = ts_window
            settings.custom_missing_tokens = custom_tokens

            # Validate boundaries
            is_valid, err_msg = settings.validate()
            if not is_valid:
                st.error(f"Validation Error: {err_msg}")
            else:
                st.session_state["analysis_settings"] = settings
                st.success("Analysis settings saved successfully!")
                st.rerun()

    with act2:
        if st.button("🔄 Restore Defaults"):
            def_settings = AnalysisSettings()
            st.session_state["analysis_settings"] = def_settings
            st.info("Restored default analysis settings.")
            st.rerun()

    return settings
