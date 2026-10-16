"""Centralized chart registry and compatibility validation system.

Defines all supported visualization types across 7 chart families, specifying
mathematical requirements, required column data types, and compatibility guards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.profiling import DatasetProfile


@dataclass
class ChartDefinition:
    """Standardized metadata and compatibility specification for a visualization."""

    chart_id: str
    display_name: str
    family: str  # "Distributions", "Relationships", "Categoricals", "Time-Series", "Comparisons", "Multivariate", "3D Visualizations"
    purpose: str
    required_column_types: list[str]  # e.g. ["numeric"], ["categorical"], ["datetime"]
    min_columns: int = 1
    max_columns: int = 1
    supports_grouping: bool = False
    supports_aggregation: bool = False
    applicable_aggregations: list[str] = field(default_factory=list)
    resource_constraints: str = "Standard browser rendering"
    parameters: dict[str, Any] = field(default_factory=dict)
    compatibility_check: Callable[[pd.DataFrame, DatasetProfile], tuple[bool, str]] = field(
        default=lambda df, p: (True, "")
    )


def check_has_numeric(df: pd.DataFrame, profile: DatasetProfile, min_count: int = 1) -> tuple[bool, str]:
    """Compatibility guard requiring at least N non-constant numeric features."""
    valid_nums = [
        col for col, p in profile.column_profiles.items()
        if p.mean is not None and not p.is_all_null and not p.is_constant
    ]
    if len(valid_nums) < min_count:
        return False, f"Requires at least {min_count} non-constant numeric feature(s) (found {len(valid_nums)})."
    return True, ""


def check_has_categorical(df: pd.DataFrame, profile: DatasetProfile, min_count: int = 1) -> tuple[bool, str]:
    """Compatibility guard requiring at least N categorical or discrete features."""
    valid_cats = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type in ("Categorical", "Text", "Boolean") or p.distinct_count <= 25
    ]
    if len(valid_cats) < min_count:
        return False, f"Requires at least {min_count} categorical/discrete feature(s) (found {len(valid_cats)})."
    return True, ""


def check_has_datetime(df: pd.DataFrame, profile: DatasetProfile) -> tuple[bool, str]:
    """Compatibility guard requiring a valid timestamp column."""
    dt_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "Datetime" or p.min_date is not None
    ]
    if not dt_cols:
        return False, "Requires at least one valid Datetime column."
    return True, ""


def check_num_and_cat(df: pd.DataFrame, profile: DatasetProfile) -> tuple[bool, str]:
    """Compatibility guard requiring at least 1 numeric and 1 categorical column."""
    has_num, num_msg = check_has_numeric(df, profile, min_count=1)
    if not has_num:
        return False, num_msg
    has_cat, cat_msg = check_has_categorical(df, profile, min_count=1)
    if not has_cat:
        return False, cat_msg
    return True, ""


# -----------------------------------------------------------------------------
# CHART REGISTRY DEFINITIONS
# -----------------------------------------------------------------------------

CHART_REGISTRY: dict[str, ChartDefinition] = {
    # 1. DISTRIBUTIONS
    "histogram_box": ChartDefinition(
        chart_id="histogram_box",
        display_name="Histogram with Marginal Box / Rug",
        family="Distributions",
        purpose="Visualizes continuous dispersion, modal peaks, and extreme values simultaneously.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=True,
        parameters={"nbins": 30, "marginal": "box"},
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "boxplot": ChartDefinition(
        chart_id="boxplot",
        display_name="Quartile Box Plot",
        family="Distributions",
        purpose="Displays median, IQR boundaries, and Tukey outlier points.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=True,
        parameters={"points": "outliers"},
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "violin_plot": ChartDefinition(
        chart_id="violin_plot",
        display_name="Violin Density Plot",
        family="Distributions",
        purpose="Combines kernel density estimation with embedded quartile markers.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=True,
        parameters={"box": True},
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "ecdf_plot": ChartDefinition(
        chart_id="ecdf_plot",
        display_name="Empirical Cumulative Distribution (ECDF)",
        family="Distributions",
        purpose="Shows step-wise cumulative probability without binning bias.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=True,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "kde_density": ChartDefinition(
        chart_id="kde_density",
        display_name="KDE Continuous Density Curve",
        family="Distributions",
        purpose="Smooth probability density estimation over continuous values.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=False,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "qq_plot": ChartDefinition(
        chart_id="qq_plot",
        display_name="Q-Q Normal Diagnostic Plot",
        family="Distributions",
        purpose="Compares sample quantiles against theoretical normal distribution quantiles.",
        required_column_types=["numeric"],
        min_columns=1,
        max_columns=1,
        supports_grouping=False,
        resource_constraints="Requires sample size >= 10",
        compatibility_check=lambda df, p: check_has_numeric(df, p, 1),
    ),
    "pareto_chart": ChartDefinition(
        chart_id="pareto_chart",
        display_name="Pareto 80/20 Distribution Chart",
        family="Distributions",
        purpose="Displays frequency bars and cumulative percentage curve to identify vital contributors.",
        required_column_types=["categorical"],
        min_columns=1,
        max_columns=1,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 1),
    ),

    # 2. RELATIONSHIPS
    "scatter_regression": ChartDefinition(
        chart_id="scatter_regression",
        display_name="Bivariate Scatter with OLS Regression",
        family="Relationships",
        purpose="Evaluates pairwise linearity, variance homoscedasticity, and slope.",
        required_column_types=["numeric", "numeric"],
        min_columns=2,
        max_columns=2,
        supports_grouping=True,
        parameters={"trendline": True},
        compatibility_check=lambda df, p: check_has_numeric(df, p, 2),
    ),
    "bubble_chart": ChartDefinition(
        chart_id="bubble_chart",
        display_name="3-Variable Bubble Chart (X, Y, Size)",
        family="Relationships",
        purpose="Examines bivariate relationships with a third numeric magnitude encoded by size.",
        required_column_types=["numeric", "numeric", "numeric"],
        min_columns=3,
        max_columns=3,
        supports_grouping=True,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),
    "density_contour": ChartDefinition(
        chart_id="density_contour",
        display_name="2D Density Contour Heatmap",
        family="Relationships",
        purpose="Bivariate 2D kernel density contour avoiding scatter overplotting on large datasets.",
        required_column_types=["numeric", "numeric"],
        min_columns=2,
        max_columns=2,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 2),
    ),
    "correlation_heatmap": ChartDefinition(
        chart_id="correlation_heatmap",
        display_name="Correlation Matrix Heatmap",
        family="Relationships",
        purpose="Displays annotated pairwise linear or monotonic correlation coefficients.",
        required_column_types=["numeric", "numeric"],
        min_columns=2,
        max_columns=40,
        parameters={"method": "spearman"},
        compatibility_check=lambda df, p: check_has_numeric(df, p, 2),
    ),
    "pairwise_scatter": ChartDefinition(
        chart_id="pairwise_scatter",
        display_name="Pairwise Scatter Matrix (3-5 columns)",
        family="Relationships",
        purpose="Multi-panel bivariate matrix revealing pairwise interactions across numeric features.",
        required_column_types=["numeric"],
        min_columns=3,
        max_columns=5,
        supports_grouping=True,
        resource_constraints="Bounded to max 5 columns and 1,500 sample points for performance",
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),

    # 3. CATEGORICALS
    "vertical_bar": ChartDefinition(
        chart_id="vertical_bar",
        display_name="Vertical Frequency Bar Chart",
        family="Categoricals",
        purpose="Compares discrete category frequencies with automated long-tail aggregation.",
        required_column_types=["categorical"],
        min_columns=1,
        max_columns=1,
        parameters={"max_categories": 15},
        compatibility_check=lambda df, p: check_has_categorical(df, p, 1),
    ),
    "horizontal_bar": ChartDefinition(
        chart_id="horizontal_bar",
        display_name="Horizontal Ranking Bar Chart",
        family="Categoricals",
        purpose="Ranks categories horizontally for readable long labels.",
        required_column_types=["categorical"],
        min_columns=1,
        max_columns=1,
        parameters={"max_categories": 15},
        compatibility_check=lambda df, p: check_has_categorical(df, p, 1),
    ),
    "grouped_bar": ChartDefinition(
        chart_id="grouped_bar",
        display_name="Grouped Categorical Comparison Bar",
        family="Categoricals",
        purpose="Compares distributions across two discrete categorical dimensions.",
        required_column_types=["categorical", "categorical"],
        min_columns=2,
        max_columns=2,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 2),
    ),
    "stacked_bar_100": ChartDefinition(
        chart_id="stacked_bar_100",
        display_name="100% Normalized Stacked Bar",
        family="Categoricals",
        purpose="Visualizes relative composition and conditional proportions across groups.",
        required_column_types=["categorical", "categorical"],
        min_columns=2,
        max_columns=2,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 2),
    ),
    "treemap": ChartDefinition(
        chart_id="treemap",
        display_name="Hierarchical Treemap",
        family="Categoricals",
        purpose="Nested rectangular area partitioning reflecting category volume or sum.",
        required_column_types=["categorical"],
        min_columns=1,
        max_columns=2,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 1),
    ),
    "sunburst": ChartDefinition(
        chart_id="sunburst",
        display_name="Concentric Sunburst Chart",
        family="Categoricals",
        purpose="Concentric ring visualization of hierarchical categorical breakdowns.",
        required_column_types=["categorical", "categorical"],
        min_columns=2,
        max_columns=3,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 2),
    ),
    "donut_chart": ChartDefinition(
        chart_id="donut_chart",
        display_name="Donut Share Chart (<= 8 categories)",
        family="Categoricals",
        purpose="Proportional share visualization for low-cardinality nominal features.",
        required_column_types=["categorical"],
        min_columns=1,
        max_columns=1,
        compatibility_check=lambda df, p: check_has_categorical(df, p, 1),
    ),

    # 4. TIME-SERIES
    "time_series_line": ChartDefinition(
        chart_id="time_series_line",
        display_name="Temporal Trend with Rolling Average",
        family="Time-Series",
        purpose="Tracks chronological changes with configurable resampling and moving averages.",
        required_column_types=["datetime", "numeric"],
        min_columns=2,
        max_columns=2,
        supports_aggregation=True,
        applicable_aggregations=["Raw", "Daily", "Weekly", "Monthly"],
        compatibility_check=lambda df, p: (
            check_has_datetime(df, p)[0] and check_has_numeric(df, p, 1)[0],
            "Requires at least 1 Datetime column and 1 Numeric measure column.",
        ),
    ),
    "time_series_area": ChartDefinition(
        chart_id="time_series_area",
        display_name="Cumulative Temporal Area Chart",
        family="Time-Series",
        purpose="Displays chronological volume and continuous accumulation over time.",
        required_column_types=["datetime", "numeric"],
        min_columns=2,
        max_columns=2,
        compatibility_check=lambda df, p: (
            check_has_datetime(df, p)[0] and check_has_numeric(df, p, 1)[0],
            "Requires at least 1 Datetime column and 1 Numeric column.",
        ),
    ),
    "autocorrelation_acf": ChartDefinition(
        chart_id="autocorrelation_acf",
        display_name="Autocorrelation (ACF) Lag Plot",
        family="Time-Series",
        purpose="Computes autocorrelation coefficients at successive time lags to test periodicity.",
        required_column_types=["datetime", "numeric"],
        min_columns=2,
        max_columns=2,
        parameters={"max_lags": 30},
        compatibility_check=lambda df, p: (
            check_has_datetime(df, p)[0] and check_has_numeric(df, p, 1)[0],
            "Requires at least 1 Datetime column and 1 Numeric column.",
        ),
    ),
    "lag_scatter": ChartDefinition(
        chart_id="lag_scatter",
        display_name="Time Lag Scatter Plot (y_t vs y_{t-k})",
        family="Time-Series",
        purpose="Bivariate scatter comparing observations against their historical lag values.",
        required_column_types=["datetime", "numeric"],
        min_columns=2,
        max_columns=2,
        parameters={"lag": 1},
        compatibility_check=lambda df, p: (
            check_has_datetime(df, p)[0] and check_has_numeric(df, p, 1)[0],
            "Requires at least 1 Datetime column and 1 Numeric column.",
        ),
    ),

    # 5. COMPARISONS
    "category_box": ChartDefinition(
        chart_id="category_box",
        display_name="Group-wise Box Plot Comparison",
        family="Comparisons",
        purpose="Compares continuous metric distributions across discrete categorical groups.",
        required_column_types=["categorical", "numeric"],
        min_columns=2,
        max_columns=2,
        compatibility_check=check_num_and_cat,
    ),
    "category_violin": ChartDefinition(
        chart_id="category_violin",
        display_name="Group-wise Violin Plot Comparison",
        family="Comparisons",
        purpose="Evaluates multi-modal densities and variance differences across segments.",
        required_column_types=["categorical", "numeric"],
        min_columns=2,
        max_columns=2,
        compatibility_check=check_num_and_cat,
    ),
    "mean_median_bars": ChartDefinition(
        chart_id="mean_median_bars",
        display_name="Group Mean vs Median Comparison",
        family="Comparisons",
        purpose="Side-by-side comparison of central tendencies to identify group-level skewness.",
        required_column_types=["categorical", "numeric"],
        min_columns=2,
        max_columns=2,
        compatibility_check=check_num_and_cat,
    ),

    # 6. MULTIVARIATE
    "parallel_coordinates": ChartDefinition(
        chart_id="parallel_coordinates",
        display_name="Parallel Coordinates Plot (3-6 variables)",
        family="Multivariate",
        purpose="Visualizes high-dimensional numeric clusters and profile trajectories.",
        required_column_types=["numeric"],
        min_columns=3,
        max_columns=6,
        supports_grouping=True,
        resource_constraints="Bounded to 6 columns and 2,000 points to ensure responsiveness",
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),
    "pca_2d": ChartDefinition(
        chart_id="pca_2d",
        display_name="PCA 2D Dimensionality Projection",
        family="Multivariate",
        purpose="Projects standardized numeric features onto first two principal components.",
        required_column_types=["numeric"],
        min_columns=3,
        max_columns=30,
        supports_grouping=True,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),

    # 7. 3D VISUALIZATIONS
    "scatter_3d": ChartDefinition(
        chart_id="scatter_3d",
        display_name="3D Interactive Spatial Scatter Plot",
        family="3D Visualizations",
        purpose="Rotatable 3D scatter inspecting 3 numeric features with color grouping.",
        required_column_types=["numeric", "numeric", "numeric"],
        min_columns=3,
        max_columns=3,
        supports_grouping=True,
        resource_constraints="Bounded to 3,000 points to prevent GPU choking. Caution: 3D plots can obscure overlap.",
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),
    "pca_3d": ChartDefinition(
        chart_id="pca_3d",
        display_name="PCA 3D Component Space Projection",
        family="3D Visualizations",
        purpose="Projects standardized numeric space onto first three principal components in 3D.",
        required_column_types=["numeric"],
        min_columns=3,
        max_columns=30,
        supports_grouping=True,
        compatibility_check=lambda df, p: check_has_numeric(df, p, 3),
    ),
}


def get_available_chart_families() -> list[str]:
    """Return all unique chart family category names."""
    return [
        "Distributions",
        "Relationships",
        "Categoricals",
        "Time-Series",
        "Comparisons",
        "Multivariate",
        "3D Visualizations",
    ]


def get_charts_by_family(family: str) -> list[ChartDefinition]:
    """Retrieve all chart definitions belonging to a specific family."""
    return [c for c in CHART_REGISTRY.values() if c.family == family]


def evaluate_chart_compatibility(
    chart_id: str,
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> tuple[bool, str]:
    """Check whether a specific chart is compatible with the current dataset."""
    chart_def = CHART_REGISTRY.get(chart_id)
    if not chart_def:
        return False, f"Unknown chart identifier '{chart_id}'."
    return chart_def.compatibility_check(df, profile)
