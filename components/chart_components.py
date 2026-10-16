"""Interactive Plotly Visualization Engine for EDA Intelligence Workbench.

Provides distribution, relationship, time-series, and comparison charts with
a consistent premium dark analytics theme, category aggregation guards, and graceful fallbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

# Design system color tokens
CHART_BG_COLOR = "#131b2e"
CANVAS_BG_COLOR = "#0b0f19"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT_COLOR = "#94a3b8"

COLOR_PALETTE = [
    "#3b82f6",  # Electric Blue
    "#8b5cf6",  # Electric Violet
    "#06b6d4",  # Cyan
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#f43f5e",  # Rose
    "#6366f1",  # Indigo
    "#14b8a6",  # Teal
]


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Apply consistent premium dark analytics styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": TEXT_COLOR, "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        template="plotly_dark",
        paper_bgcolor=CHART_BG_COLOR,
        plot_bgcolor=CHART_BG_COLOR,
        font={"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 50, "r": 40, "t": 60, "b": 50},
        legend={
            "bgcolor": "rgba(17, 24, 39, 0.6)",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "borderwidth": 1,
            "font": {"color": TEXT_COLOR, "size": 11},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont={"color": MUTED_TEXT_COLOR, "size": 11},
        title_font={"color": TEXT_COLOR, "size": 12},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont={"color": MUTED_TEXT_COLOR, "size": 11},
        title_font={"color": TEXT_COLOR, "size": 12},
    )
    return fig


# Alias for backward compatibility
_apply_workbench_theme = apply_dark_theme


# -----------------------------------------------------------------------------
# NUMERIC DISTRIBUTIONS
# -----------------------------------------------------------------------------

def plot_histogram(
    df: pd.DataFrame,
    col: str,
    nbins: int = 30,
    show_kde: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot distribution histogram with optional KDE density curve."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    if clean_series.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Numeric Observations")

    fig = px.histogram(
        clean_series,
        x=clean_series.values,
        nbins=nbins,
        marginal="box" if show_kde else None,
        color_discrete_sequence=[COLOR_PALETTE[0]],
        opacity=0.85,
    )

    t = title or f"Distribution Histogram: {col}"
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Count")
    return apply_dark_theme(fig, title=t)


def plot_boxplot(
    df: pd.DataFrame,
    col: str,
    points: str = "outliers",
    title: str | None = None,
) -> go.Figure:
    """Plot box plot with quartile markers and outliers."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.box(
        clean_series,
        y=clean_series.values,
        points=points,
        color_discrete_sequence=[COLOR_PALETTE[1]],
    )
    t = title or f"Box Plot & Quartiles: {col}"
    fig.update_yaxes(title_text=col)
    return apply_dark_theme(fig, title=t)


def plot_violin(
    df: pd.DataFrame,
    col: str,
    box: bool = True,
    points: str = "outliers",
    title: str | None = None,
) -> go.Figure:
    """Plot violin distribution with embedded box."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.violin(
        clean_series,
        y=clean_series.values,
        box=box,
        points=points,
        color_discrete_sequence=[COLOR_PALETTE[2]],
    )
    t = title or f"Violin Density: {col}"
    fig.update_yaxes(title_text=col)
    return apply_dark_theme(fig, title=t)


def plot_ecdf(
    df: pd.DataFrame,
    col: str,
    title: str | None = None,
) -> go.Figure:
    """Plot empirical cumulative distribution function (ECDF)."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.ecdf(
        clean_series,
        x=clean_series.values,
        color_discrete_sequence=[COLOR_PALETTE[3]],
    )
    t = title or f"Empirical Cumulative Distribution (ECDF): {col}"
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Cumulative Probability")
    return apply_dark_theme(fig, title=t)


# -----------------------------------------------------------------------------
# CATEGORICAL DISTRIBUTIONS
# -----------------------------------------------------------------------------

def plot_categorical_bar(
    df: pd.DataFrame,
    col: str,
    max_categories: int = 15,
    horizontal: bool = False,
    title: str | None = None,
) -> tuple[go.Figure, bool, int]:
    """Plot vertical or horizontal bar chart for categorical distributions.

    Aggregates excess levels into 'Other' if distinct levels exceed max_categories.
    Returns: (fig, is_aggregated, other_count)
    """
    clean = df[col].dropna().astype(str)
    counts = clean.value_counts()
    distinct_n = len(counts)

    is_agg = False
    other_count = 0
    if distinct_n > max_categories:
        is_agg = True
        top_counts = counts.iloc[: max_categories - 1]
        other_count = int(counts.iloc[max_categories - 1 :].sum())
        display_series = pd.concat([top_counts, pd.Series({"Other (Aggregated)": other_count})])
    else:
        display_series = counts

    plot_df = pd.DataFrame({"Category": display_series.index, "Count": display_series.values})

    if horizontal:
        # Reverse order so highest is on top
        plot_df = plot_df.iloc[::-1]
        fig = px.bar(
            plot_df,
            x="Count",
            y="Category",
            orientation="h",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            text="Count",
        )
    else:
        fig = px.bar(
            plot_df,
            x="Category",
            y="Count",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            text="Count",
        )

    fig.update_traces(textposition="outside")
    t = title or f"Frequency Breakdown: {col}"
    return apply_dark_theme(fig, title=t), is_agg, other_count


def plot_treemap(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str | None = None,
    max_categories: int = 20,
    title: str | None = None,
) -> go.Figure:
    """Plot hierarchical treemap for categorical shares."""
    sub = df[[cat_col] + ([val_col] if val_col else [])].dropna()
    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Observations for Treemap")

    sub[cat_col] = sub[cat_col].astype(str)
    counts = sub[cat_col].value_counts()

    if len(counts) > max_categories:
        top_cats = set(counts.iloc[: max_categories - 1].index)
        sub["Display_Cat"] = sub[cat_col].apply(lambda x: x if x in top_cats else "Other")
    else:
        sub["Display_Cat"] = sub[cat_col]

    if val_col:
        sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce").fillna(0)
        agg_df = sub.groupby("Display_Cat")[val_col].sum().reset_index()
        fig = px.treemap(
            agg_df,
            path=["Display_Cat"],
            values=val_col,
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        agg_df = sub["Display_Cat"].value_counts().reset_index()
        agg_df.columns = ["Display_Cat", "Count"]
        fig = px.treemap(
            agg_df,
            path=["Display_Cat"],
            values="Count",
            color_discrete_sequence=COLOR_PALETTE,
        )

    t = title or f"Treemap Composition: {cat_col}"
    return apply_dark_theme(fig, title=t)


# -----------------------------------------------------------------------------
# RELATIONSHIPS
# -----------------------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    trendline: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Plot bivariate scatter plot with optional color grouping and OLS trendline."""
    cols_needed = [x_col, y_col]
    if color_col:
        cols_needed.append(color_col)
    if size_col:
        cols_needed.append(size_col)

    sub = df[cols_needed].dropna().copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col])

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Insufficient Paired Numeric Data for Scatter")

    # If size_col used, ensure positive
    if size_col:
        sub[size_col] = pd.to_numeric(sub[size_col], errors="coerce").fillna(1)
        sub[size_col] = np.maximum(0.1, sub[size_col])

    # Sample if overly large to prevent webgl choking
    if len(sub) > 5000:
        sub = sub.sample(n=5000, random_state=42)

    trend = "ols" if trendline and len(sub) >= 5 else None

    fig = px.scatter(
        sub,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        trendline=trend,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.75,
    )
    t = title or f"Scatter Analysis: {x_col} vs {y_col}"
    return apply_dark_theme(fig, title=t)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
) -> go.Figure:
    """Plot annotated correlation matrix heatmap with diverging colorscale."""
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Insufficient Features for Correlation Heatmap")

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
    )
    fig.update_layout(
        coloraxis_colorbar={
            "title": "Correlation (r)",
            "tickvals": [-1, -0.5, 0, 0.5, 1],
            "ticktext": ["-1.0 (Inverse)", "-0.5", "0.0 (None)", "0.5", "+1.0 (Direct)"],
        }
    )
    return apply_dark_theme(fig, title=title, height=max(450, len(corr_matrix) * 35))


def plot_pairwise_scatter(
    df: pd.DataFrame,
    columns: list[str],
    color_col: str | None = None,
    max_rows: int = 1500,
) -> go.Figure:
    """Plot pairwise scatter matrix for a bounded set of numeric columns (up to 5)."""
    cols_to_use = columns[:5]  # Strict guardrail
    cols_needed = list(cols_to_use)
    if color_col and color_col not in cols_needed:
        cols_needed.append(color_col)

    sub = df[cols_needed].dropna().copy()
    for c in cols_to_use:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols_to_use)

    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42)

    fig = px.scatter_matrix(
        sub,
        dimensions=cols_to_use,
        color=color_col,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.7,
    )
    return apply_dark_theme(fig, title="Pairwise Relationship Matrix", height=600)


# -----------------------------------------------------------------------------
# TIME-SERIES & COMPARISONS
# -----------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    freq: str = "Raw",  # "Raw", "Daily", "Weekly", "Monthly"
    show_rolling: bool = True,
    rolling_window: int = 7,
    title: str | None = None,
) -> go.Figure:
    """Plot temporal trend with optional resampling aggregation and rolling average."""
    sub = df[[date_col, val_col]].dropna().copy()
    sub["dt"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub["val"] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=["dt", "val"]).sort_values("dt")

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Datetime & Value Pairs")

    fig = go.Figure()

    if freq == "Daily":
        agg = sub.set_index("dt")["val"].resample("D").mean().dropna().reset_index()
    elif freq == "Weekly":
        agg = sub.set_index("dt")["val"].resample("W").mean().dropna().reset_index()
    elif freq == "Monthly":
        agg = sub.set_index("dt")["val"].resample("M").mean().dropna().reset_index()
    else:
        agg = sub.rename(columns={"val": "val"})

    # Primary line
    fig.add_trace(
        go.Scatter(
            x=agg["dt"],
            y=agg["val"],
            mode="lines",
            name=f"{val_col} ({freq})",
            line={"color": COLOR_PALETTE[0], "width": 2},
        )
    )

    # Rolling average overlay
    if show_rolling and len(agg) >= rolling_window:
        rolling_series = agg["val"].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=agg["dt"],
                y=rolling_series,
                mode="lines",
                name=f"{rolling_window}-Period Rolling Avg",
                line={"color": COLOR_PALETTE[4], "width": 2.5, "dash": "dash"},
            )
        )

    t = title or f"Temporal Trend: {val_col} over {date_col}"
    fig.update_xaxes(title_text=date_col)
    fig.update_yaxes(title_text=val_col)
    return apply_dark_theme(fig, title=t)


def plot_box_by_category(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str,
    max_categories: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot numeric distribution segmented across categorical groups."""
    sub = df[[cat_col, val_col]].dropna().copy()
    sub[cat_col] = sub[cat_col].astype(str)
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[val_col])

    top_cats = sub[cat_col].value_counts().head(max_categories).index
    sub_filtered = sub[sub[cat_col].isin(top_cats)]

    fig = px.box(
        sub_filtered,
        x=cat_col,
        y=val_col,
        color=cat_col,
        color_discrete_sequence=COLOR_PALETTE,
    )
    t = title or f"{val_col} Distribution Across {cat_col}"
    return apply_dark_theme(fig, title=t)


# -----------------------------------------------------------------------------
# ADVANCED PHASE 2 VISUALIZATIONS
# -----------------------------------------------------------------------------

def plot_kde_density(
    df: pd.DataFrame,
    col: str,
    title: str | None = None,
) -> go.Figure:
    """Plot Gaussian Kernel Density Estimation (KDE) continuous probability curve."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(clean_series.unique()) < 3:
        fig = go.Figure()
        return apply_dark_theme(
            fig,
            title=f"KDE Density requires at least 3 distinct values (found {len(clean_series.unique())})",
        )

    try:
        kde = stats.gaussian_kde(clean_series.values)
        x_min, x_max = clean_series.min(), clean_series.max()
        padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
        x_grid = np.linspace(x_min - padding, x_max + padding, 300)
        density = kde(x_grid)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name="KDE Density",
                line={"color": COLOR_PALETTE[0], "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.25)",
            )
        )
        fig.update_xaxes(title_text=col)
        fig.update_yaxes(title_text="Probability Density")
        t = title or f"Continuous KDE Probability Density: {col}"
        return apply_dark_theme(fig, title=t)
    except (ValueError, np.linalg.LinAlgError) as exc:
        fig = go.Figure()
        return apply_dark_theme(fig, title=f"KDE Calculation Error: {exc}")


def plot_qq_plot(
    df: pd.DataFrame,
    col: str,
    title: str | None = None,
) -> go.Figure:
    """Plot Quantile-Quantile (Q-Q) diagnostic plot against standard normal distribution."""
    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(clean_series) < 10:
        fig = go.Figure()
        return apply_dark_theme(
            fig,
            title="Q-Q Plot requires at least 10 valid observations for diagnostic utility",
        )

    (osm, osr), (slope, intercept, r) = stats.probplot(clean_series, dist="norm")

    fig = go.Figure()
    # Sample quantiles scatter
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Sample Quantiles",
            marker={"color": COLOR_PALETTE[2], "size": 6, "opacity": 0.8},
        )
    )
    # Theoretical normal fit line
    line_x = np.array([osm[0], osm[-1]])
    line_y = slope * line_x + intercept
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name=f"Normal Fit (R² = {r**2:.3f})",
            line={"color": COLOR_PALETTE[5], "width": 2, "dash": "dash"},
        )
    )

    fig.update_xaxes(title_text="Theoretical Normal Quantiles (Z-Score)")
    fig.update_yaxes(title_text=f"Sample Quantiles ({col})")
    t = title or f"Normal Q-Q Diagnostic Plot: {col} (R² = {r**2:.3f})"
    return apply_dark_theme(fig, title=t)


def plot_pareto_chart(
    df: pd.DataFrame,
    col: str,
    max_categories: int = 15,
    title: str | None = None,
) -> go.Figure:
    """Plot Pareto 80/20 frequency distribution with cumulative contribution percentage curve."""
    clean = df[col].dropna().astype(str)
    counts = clean.value_counts()
    if counts.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Data for Pareto Analysis")

    total_count = counts.sum()
    if len(counts) > max_categories:
        top_counts = counts.iloc[: max_categories - 1]
        other_sum = counts.iloc[max_categories - 1 :].sum()
        display_series = pd.concat([top_counts, pd.Series({"Other (Aggregated)": other_sum})])
    else:
        display_series = counts

    categories = list(display_series.index)
    freqs = list(display_series.values)
    cum_pct = (np.cumsum(freqs) / total_count) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Bar trace on primary y
    fig.add_trace(
        go.Bar(
            x=categories,
            y=freqs,
            name="Frequency",
            marker_color=COLOR_PALETTE[0],
            opacity=0.85,
        ),
        secondary_y=False,
    )
    # Cumulative curve on secondary y
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=cum_pct,
            name="Cumulative %",
            mode="lines+markers",
            line={"color": COLOR_PALETTE[4], "width": 2.5},
            marker={"size": 6, "color": COLOR_PALETTE[4]},
        ),
        secondary_y=True,
    )
    # 80% Pareto threshold guideline
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color=COLOR_PALETTE[5],
        annotation_text="80% Cutoff",
        annotation_position="bottom right",
        secondary_y=True,
    )

    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage (%)", range=[0, 105], secondary_y=True)
    t = title or f"Pareto 80/20 Distribution: {col}"
    return apply_dark_theme(fig, title=t)


def plot_density_contour(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
) -> go.Figure:
    """Plot 2D density contour map avoiding overplotting on dense numeric data."""
    sub = df[[x_col, y_col]].dropna().copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 10:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Insufficient paired observations for 2D density contour")

    fig = px.density_contour(
        sub,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[COLOR_PALETTE[2]],
    )
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    t = title or f"2D Density Contours: {x_col} vs {y_col}"
    return apply_dark_theme(fig, title=t)


def plot_grouped_bar(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    max_categories: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot grouped side-by-side bar chart for two categorical dimensions."""
    sub = df[[cat_col1, cat_col2]].dropna().copy()
    sub[cat_col1] = sub[cat_col1].astype(str)
    sub[cat_col2] = sub[cat_col2].astype(str)

    top1 = sub[cat_col1].value_counts().head(max_categories).index
    top2 = sub[cat_col2].value_counts().head(max_categories).index
    sub = sub[sub[cat_col1].isin(top1) & sub[cat_col2].isin(top2)]

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Overlapping Observations")

    agg = sub.groupby([cat_col1, cat_col2]).size().reset_index(name="Count")
    fig = px.bar(
        agg,
        x=cat_col1,
        y="Count",
        color=cat_col2,
        barmode="group",
        color_discrete_sequence=COLOR_PALETTE,
    )
    t = title or f"Grouped Comparison: {cat_col1} by {cat_col2}"
    return apply_dark_theme(fig, title=t)


def plot_stacked_bar_100(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    max_categories: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot 100% normalized stacked bar chart displaying conditional proportions."""
    sub = df[[cat_col1, cat_col2]].dropna().copy()
    sub[cat_col1] = sub[cat_col1].astype(str)
    sub[cat_col2] = sub[cat_col2].astype(str)

    top1 = sub[cat_col1].value_counts().head(max_categories).index
    top2 = sub[cat_col2].value_counts().head(max_categories).index
    sub = sub[sub[cat_col1].isin(top1) & sub[cat_col2].isin(top2)]

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Observations for Stacked Proportions")

    crosstab = pd.crosstab(sub[cat_col1], sub[cat_col2], normalize="index") * 100
    crosstab_melt = crosstab.reset_index().melt(
        id_vars=cat_col1,
        value_name="Percentage",
    )

    fig = px.bar(
        crosstab_melt,
        x=cat_col1,
        y="Percentage",
        color=cat_col2,
        color_discrete_sequence=COLOR_PALETTE,
        text="Percentage",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig.update_yaxes(title_text="Relative Proportion (%)", range=[0, 100])
    t = title or f"100% Composition: {cat_col2} Distribution within {cat_col1}"
    return apply_dark_theme(fig, title=t)


def plot_sunburst(
    df: pd.DataFrame,
    path_cols: list[str],
    val_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot concentric multi-level Sunburst chart for hierarchical categorical exploration."""
    cols = list(path_cols) + ([val_col] if val_col else [])
    sub = df[cols].dropna().copy()
    for c in path_cols:
        sub[c] = sub[c].astype(str)

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Data for Sunburst")

    if val_col:
        sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce").fillna(0)
        fig = px.sunburst(
            sub,
            path=path_cols,
            values=val_col,
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        sub["Count"] = 1
        agg = sub.groupby(path_cols)["Count"].sum().reset_index()
        fig = px.sunburst(
            agg,
            path=path_cols,
            values="Count",
            color_discrete_sequence=COLOR_PALETTE,
        )

    t = title or f"Sunburst Hierarchy: {' → '.join(path_cols)}"
    return apply_dark_theme(fig, title=t, height=520)


def plot_donut_chart(
    df: pd.DataFrame,
    col: str,
    max_categories: int = 8,
    title: str | None = None,
) -> go.Figure:
    """Plot donut share chart for low-cardinality nominal variables."""
    clean = df[col].dropna().astype(str)
    counts = clean.value_counts()
    if counts.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Data for Donut Chart")

    if len(counts) > max_categories:
        top = counts.iloc[: max_categories - 1]
        other = counts.iloc[max_categories - 1 :].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.55,
        color_discrete_sequence=COLOR_PALETTE,
    )
    t = title or f"Proportional Share: {col}"
    return apply_dark_theme(fig, title=t)


def plot_time_series_area(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    freq: str = "Raw",
    title: str | None = None,
) -> go.Figure:
    """Plot cumulative or chronological area chart."""
    sub = df[[date_col, val_col]].dropna().copy()
    sub["dt"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub["val"] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=["dt", "val"]).sort_values("dt")

    if sub.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, title="No Valid Datetime & Value Pairs")

    if freq == "Daily":
        agg = sub.set_index("dt")["val"].resample("D").mean().dropna().reset_index()
    elif freq == "Weekly":
        agg = sub.set_index("dt")["val"].resample("W").mean().dropna().reset_index()
    elif freq == "Monthly":
        agg = sub.set_index("dt")["val"].resample("M").mean().dropna().reset_index()
    else:
        agg = sub

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["dt"],
            y=agg["val"],
            mode="lines",
            fill="tozeroy",
            name=val_col,
            line={"color": COLOR_PALETTE[2], "width": 2},
            fillcolor="rgba(6, 182, 212, 0.25)",
        )
    )
    fig.update_xaxes(title_text=date_col)
    fig.update_yaxes(title_text=val_col)
    t = title or f"Temporal Area Profile: {val_col} ({freq})"
    return apply_dark_theme(fig, title=t)


def plot_autocorrelation_acf(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    max_lags: int = 30,
    title: str | None = None,
) -> go.Figure:
    """Plot sample Autocorrelation Function (ACF) with 95% Bartlett confidence bands."""
    sub = df[[date_col, val_col]].dropna().copy()
    sub["dt"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub["val"] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=["dt", "val"]).sort_values("dt")

    series = sub["val"].values
    n = len(series)
    if n < 15:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Requires at least 15 time-ordered observations for ACF")

    nlags = min(max_lags, n // 2)
    acf_vals = acf(series, nlags=nlags, fft=True)
    lags = list(range(len(acf_vals)))

    conf_bound = 1.96 / np.sqrt(n)

    fig = go.Figure()
    # Confidence envelope
    fig.add_hline(
        y=conf_bound,
        line_dash="dash",
        line_color=COLOR_PALETTE[4],
        annotation_text="+95% CI",
    )
    fig.add_hline(
        y=-conf_bound,
        line_dash="dash",
        line_color=COLOR_PALETTE[4],
        annotation_text="-95% CI",
    )
    fig.add_hline(y=0, line_color=GRID_COLOR)

    # ACF Bars
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_vals,
            marker_color=COLOR_PALETTE[0],
            name="ACF Coefficient",
            width=0.3,
        )
    )

    fig.update_xaxes(title_text="Lag Order (k)")
    fig.update_yaxes(title_text="Autocorrelation r_k", range=[-1.05, 1.05])
    t = title or f"Autocorrelation Diagnostic (ACF): {val_col} (N={n})"
    return apply_dark_theme(fig, title=t)


def plot_lag_scatter(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    lag: int = 1,
    title: str | None = None,
) -> go.Figure:
    """Plot bivariate lag scatter (y_t vs y_{t-k}) to diagnose serial correlation."""
    sub = df[[date_col, val_col]].dropna().copy()
    sub["dt"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub["val"] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=["dt", "val"]).sort_values("dt")

    sub["lag_val"] = sub["val"].shift(lag)
    lag_df = sub.dropna(subset=["val", "lag_val"])

    if len(lag_df) < 5:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Insufficient observations after lag shift")

    fig = px.scatter(
        lag_df,
        x="lag_val",
        y="val",
        color_discrete_sequence=[COLOR_PALETTE[3]],
        opacity=0.75,
    )
    # Add 45-degree diagonal reference line
    min_val = min(lag_df["lag_val"].min(), lag_df["val"].min())
    max_val = max(lag_df["lag_val"].max(), lag_df["val"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Identity Line (y=x)",
            line={"color": COLOR_PALETTE[5], "dash": "dot"},
        )
    )

    fig.update_xaxes(title_text=f"{val_col} at time t-{lag}")
    fig.update_yaxes(title_text=f"{val_col} at time t")
    t = title or f"Lag-{lag} Scatter Plot: {val_col}"
    return apply_dark_theme(fig, title=t)


def plot_violin_by_category(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str,
    max_categories: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot group-wise violin distributions segmented across categorical groups."""
    sub = df[[cat_col, val_col]].dropna().copy()
    sub[cat_col] = sub[cat_col].astype(str)
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[val_col])

    top_cats = sub[cat_col].value_counts().head(max_categories).index
    sub_filtered = sub[sub[cat_col].isin(top_cats)]

    fig = px.violin(
        sub_filtered,
        x=cat_col,
        y=val_col,
        color=cat_col,
        box=True,
        points="outliers",
        color_discrete_sequence=COLOR_PALETTE,
    )
    t = title or f"Violin Density Across {cat_col}: {val_col}"
    return apply_dark_theme(fig, title=t)


def plot_mean_median_comparison(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str,
    max_categories: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot side-by-side grouped bars comparing Mean and Median across categories."""
    sub = df[[cat_col, val_col]].dropna().copy()
    sub[cat_col] = sub[cat_col].astype(str)
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[val_col])

    top_cats = sub[cat_col].value_counts().head(max_categories).index
    sub_filtered = sub[sub[cat_col].isin(top_cats)]

    agg = sub_filtered.groupby(cat_col)[val_col].agg(["mean", "median"]).reset_index()
    agg_melt = agg.melt(id_vars=cat_col, value_vars=["mean", "median"], var_name="Statistic", value_name="Value")

    fig = px.bar(
        agg_melt,
        x=cat_col,
        y="Value",
        color="Statistic",
        barmode="group",
        color_discrete_sequence=[COLOR_PALETTE[0], COLOR_PALETTE[4]],
    )
    t = title or f"Central Tendency Comparison (Mean vs Median): {val_col} by {cat_col}"
    return apply_dark_theme(fig, title=t)


def plot_parallel_coordinates(
    df: pd.DataFrame,
    cols: list[str],
    color_col: str | None = None,
    max_rows: int = 2000,
    title: str | None = None,
) -> go.Figure:
    """Plot parallel coordinates across 3-6 continuous numeric features."""
    cols_to_use = cols[:6]
    needed = list(cols_to_use)
    if color_col and color_col not in needed:
        needed.append(color_col)

    sub = df[needed].dropna().copy()
    for c in cols_to_use:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols_to_use)

    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42)

    fig = px.parallel_coordinates(
        sub,
        dimensions=cols_to_use,
        color=color_col if color_col in sub.columns and pd.api.types.is_numeric_dtype(sub[color_col]) else None,
        color_continuous_scale="Viridis",
    )
    t = title or f"Parallel Coordinates: {', '.join(cols_to_use)}"
    return apply_dark_theme(fig, title=t, height=520)


def plot_pca_2d(
    df: pd.DataFrame,
    cols: list[str],
    color_col: str | None = None,
    max_rows: int = 3000,
    title: str | None = None,
) -> tuple[go.Figure, list[float]]:
    """Standardize numeric columns and project onto first 2 Principal Components (PCA)."""
    needed = list(cols)
    if color_col and color_col not in needed:
        needed.append(color_col)

    sub = df[needed].dropna().copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols)

    if len(sub) < 5 or len(cols) < 2:
        fig = go.Figure()
        return apply_dark_theme(fig, title="PCA requires at least 2 numeric features and 5 observations"), []

    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42)

    features = sub[cols].values
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_feats)
    evr = [float(v) * 100 for v in pca.explained_variance_ratio_]

    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]

    fig = px.scatter(
        sub,
        x="PC1",
        y="PC2",
        color=color_col if color_col in sub.columns else None,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.8,
    )
    fig.update_xaxes(title_text=f"Principal Component 1 ({evr[0]:.1f}% Variance)")
    fig.update_yaxes(title_text=f"Principal Component 2 ({evr[1]:.1f}% Variance)")
    t = title or f"PCA 2D Projection (Total Explained: {sum(evr):.1f}%)"
    return apply_dark_theme(fig, title=t), evr


def plot_scatter_3d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str | None = None,
    max_rows: int = 3000,
    title: str | None = None,
) -> go.Figure:
    """Plot interactive 3D spatial scatter plot with dark scene styling."""
    cols = [x_col, y_col, z_col]
    if color_col:
        cols.append(color_col)

    sub = df[cols].dropna().copy()
    for c in [x_col, y_col, z_col]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col, z_col])

    if len(sub) < 5:
        fig = go.Figure()
        return apply_dark_theme(fig, title="Insufficient paired observations for 3D Scatter")

    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42)

    fig = px.scatter_3d(
        sub,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col if color_col in sub.columns else None,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.8,
    )
    fig.update_layout(
        scene={
            "xaxis": {"backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
            "yaxis": {"backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
            "zaxis": {"backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
        },
        height=550,
    )
    t = title or f"3D Scatter: {x_col} × {y_col} × {z_col}"
    return apply_dark_theme(fig, title=t, height=550)


def plot_pca_3d(
    df: pd.DataFrame,
    cols: list[str],
    color_col: str | None = None,
    max_rows: int = 3000,
    title: str | None = None,
) -> tuple[go.Figure, list[float]]:
    """Project standardized continuous features onto 3 Principal Components in 3D."""
    needed = list(cols)
    if color_col and color_col not in needed:
        needed.append(color_col)

    sub = df[needed].dropna().copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols)

    if len(sub) < 5 or len(cols) < 3:
        fig = go.Figure()
        return apply_dark_theme(fig, title="3D PCA requires at least 3 numeric features and 5 observations"), []

    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42)

    features = sub[cols].values
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)

    pca = PCA(n_components=3)
    coords = pca.fit_transform(scaled_feats)
    evr = [float(v) * 100 for v in pca.explained_variance_ratio_]

    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    sub["PC3"] = coords[:, 2]

    fig = px.scatter_3d(
        sub,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col if color_col in sub.columns else None,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.8,
    )
    fig.update_layout(
        scene={
            "xaxis": {"title": f"PC1 ({evr[0]:.1f}%)", "backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
            "yaxis": {"title": f"PC2 ({evr[1]:.1f}%)", "backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
            "zaxis": {"title": f"PC3 ({evr[2]:.1f}%)", "backgroundcolor": CHART_BG_COLOR, "gridcolor": GRID_COLOR, "color": TEXT_COLOR},
        },
        height=580,
    )
    t = title or f"PCA 3D Space (Total Explained: {sum(evr):.1f}%)"
    return apply_dark_theme(fig, title=t, height=580), evr

