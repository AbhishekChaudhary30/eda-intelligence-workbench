"""Automatic First-Look Analysis Engine for EDA Intelligence Workbench.

Synthesizes data validation, profiling, quality findings, and correlation statistics
into deterministic, mathematically grounded analytical observations and starter recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.data_quality import QualityReport
from core.profiling import DatasetProfile
from core.statistics import CorrelationAnalysisResult, compute_correlations


@dataclass
class ObservationItem:
    """A distinct analytical finding or recommendation."""

    category: str  # "Fact", "Quality", "Relationship", "Distribution", "NextStep", "Limitation"
    headline: str
    detail: str
    metric_value: str | None = None


@dataclass
class CuratedVisualCandidates:
    """Candidate columns chosen deterministically for automatic starter visualization."""

    top_pair_x: str | None = None
    top_pair_y: str | None = None
    top_pair_r: float | None = None
    primary_numeric_col: str | None = None
    primary_categorical_col: str | None = None
    primary_datetime_col: str | None = None


@dataclass
class FirstLookReport:
    """Full synthesis output for the Automatic First-Look Analysis."""

    facts: list[ObservationItem] = field(default_factory=list)
    quality_highlights: list[ObservationItem] = field(default_factory=list)
    relationships: list[ObservationItem] = field(default_factory=list)
    distribution_insights: list[ObservationItem] = field(default_factory=list)
    next_steps: list[ObservationItem] = field(default_factory=list)
    limitations: list[ObservationItem] = field(default_factory=list)
    visual_candidates: CuratedVisualCandidates = field(default_factory=CuratedVisualCandidates)


def run_first_look_analysis(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality_report: QualityReport,
) -> FirstLookReport:
    """Run a deterministic, evidence-based automatic analysis across the dataset."""
    facts: list[ObservationItem] = []
    quality_highlights: list[ObservationItem] = []
    relationships: list[ObservationItem] = []
    distribution_insights: list[ObservationItem] = []
    next_steps: list[ObservationItem] = []
    limitations: list[ObservationItem] = []

    total_rows = profile.total_rows
    total_cols = profile.total_cols

    if total_rows == 0 or total_cols == 0:
        facts.append(
            ObservationItem(
                category="Fact",
                headline="Empty Dataset",
                detail="The dataset contains zero records or zero columns.",
            )
        )
        return FirstLookReport(facts=facts)

    # 1. Observed Facts
    facts.append(
        ObservationItem(
            category="Fact",
            headline="Dataset Dimensions",
            detail=(
                f"Dataset comprises {total_rows:,} rows across {total_cols:,} columns, "
                f"consuming approximately {profile.formatted_memory} in memory."
            ),
        )
    )

    type_summary = (
        f"{profile.numeric_columns_count} numeric, {profile.categorical_columns_count} categorical/text, "
        f"{profile.datetime_columns_count} datetime, and {profile.boolean_columns_count} boolean columns."
    )
    facts.append(
        ObservationItem(
            category="Fact",
            headline="Feature Composition",
            detail=f"Detected semantic types: {type_summary}",
        )
    )

    completeness_detail = (
        f"{profile.complete_rows_count:,} out of {total_rows:,} rows ({profile.complete_rows_percentage}%) "
        f"are completely intact without any missing values."
    )
    facts.append(
        ObservationItem(
            category="Fact",
            headline="Row Completeness",
            detail=completeness_detail,
        )
    )

    # 2. Quality Highlights
    quality_highlights.append(
        ObservationItem(
            category="Quality",
            headline=f"Health Audit Score: {quality_report.overall_health_score}/100 ({quality_report.health_status})",
            detail=(
                f"Audited {quality_report.total_issues_count} total data-quality conditions: "
                f"{quality_report.critical_issues_count} Critical, {quality_report.warning_issues_count} Warning, "
                f"and {quality_report.info_issues_count} Informational."
            ),
        )
    )

    if profile.duplicate_rows_count > 0:
        quality_highlights.append(
            ObservationItem(
                category="Quality",
                headline=f"{profile.duplicate_rows_count:,} Exact Duplicate Rows",
                detail=(
                    f"Duplicate records account for {profile.duplicate_rows_percentage}% of the dataset. "
                    "Review if duplicates are legitimate recurring occurrences or ETL re-ingestion artifacts."
                ),
            )
        )

    # 3. Relationships & Correlations
    candidates = CuratedVisualCandidates()

    if profile.numeric_columns_count >= 2:
        corr_res: CorrelationAnalysisResult = compute_correlations(
            df, profile, method="spearman", max_columns=25
        )
        if corr_res.top_pairs:
            top_p = corr_res.top_pairs[0]
            candidates.top_pair_x = top_p.column_a
            candidates.top_pair_y = top_p.column_b
            candidates.top_pair_r = top_p.correlation

            relationships.append(
                ObservationItem(
                    category="Relationship",
                    headline=f"{top_p.strength} {top_p.direction} Correlation (r = {top_p.correlation:.2f})",
                    detail=(
                        f"Features '{top_p.column_a}' and '{top_p.column_b}' exhibit a {top_p.strength.lower()} "
                        f"monotonic association based on {top_p.sample_size:,} paired observations."
                    ),
                    metric_value=f"r={top_p.correlation:.2f}",
                )
            )

            # Check if there is another notable correlation
            if len(corr_res.top_pairs) > 1 and corr_res.top_pairs[1].abs_correlation >= 0.4:
                second_p = corr_res.top_pairs[1]
                relationships.append(
                    ObservationItem(
                        category="Relationship",
                        headline=f"{second_p.strength} Correlation: '{second_p.column_a}' vs '{second_p.column_b}'",
                        detail=f"Spearman r = {second_p.correlation:.2f} (n = {second_p.sample_size:,}).",
                        metric_value=f"r={second_p.correlation:.2f}",
                    )
                )
        else:
            relationships.append(
                ObservationItem(
                    category="Relationship",
                    headline="No Significant Linear/Monotonic Numeric Correlations",
                    detail="Numeric features showed insufficient variation or low cross-correlation.",
                )
            )

    # 4. Distribution Insights
    # Find most skewed column
    skewed_cols = [
        (col, p.skewness)
        for col, p in profile.column_profiles.items()
        if p.skewness is not None and not p.is_zero_variance
    ]
    if skewed_cols:
        skewed_cols.sort(key=lambda x: abs(x[1]), reverse=True)
        top_skew_col, top_skew_val = skewed_cols[0]
        direction = "right-skewed (positive skew)" if top_skew_val > 0 else "left-skewed (negative skew)"
        distribution_insights.append(
            ObservationItem(
                category="Distribution",
                headline=f"Substantial Asymmetry in '{top_skew_col}'",
                detail=(
                    f"Column '{top_skew_col}' is heavily {direction} with skewness = {top_skew_val:.2f}. "
                    "Consider log or Box-Cox transformations if linear models will be trained on this feature."
                ),
            )
        )
        candidates.primary_numeric_col = top_skew_col
    elif profile.numeric_columns_count > 0:
        # Default to first non-constant numeric column
        for col, p in profile.column_profiles.items():
            if p.mean is not None and not p.is_constant:
                candidates.primary_numeric_col = col
                break

    # Find candidate categorical column (2 to 15 distinct values)
    best_cat: str | None = None
    for col, p in profile.column_profiles.items():
        if (
            (p.semantic_type in ("Categorical", "Boolean") or p.distinct_count <= 15)
            and p.distinct_count >= 2
            and not p.is_potential_identifier
        ):
            best_cat = col
            break
    candidates.primary_categorical_col = best_cat

    # Find candidate datetime column
    for col, p in profile.column_profiles.items():
        if p.semantic_type == "Datetime" and p.min_date:
            candidates.primary_datetime_col = col
            break

    # 5. Suggested Next Steps
    if quality_report.critical_issues_count > 0:
        next_steps.append(
            ObservationItem(
                category="NextStep",
                headline="Address Critical Data Quality Warnings",
                detail="Inspect features with >60% missingness or all-null values before statistical inference.",
            )
        )

    if profile.duplicate_rows_count > 0:
        next_steps.append(
            ObservationItem(
                category="NextStep",
                headline="Resolve Exact Duplicate Rows",
                detail="Verify if identical records represent repeated interactions or duplicate pipeline ingestion.",
            )
        )

    if candidates.top_pair_x and candidates.top_pair_y:
        next_steps.append(
            ObservationItem(
                category="NextStep",
                headline=f"Examine Scatter Relationship: '{candidates.top_pair_x}' vs '{candidates.top_pair_y}'",
                detail="Generate an interactive scatter plot with trendline to evaluate homoscedasticity and non-linear patterns.",
            )
        )

    next_steps.append(
        ObservationItem(
            category="NextStep",
            headline="Formulate Targeted Hypotheses",
            detail="Transition to Manual Exploration to test group-wise variances and distribution splits.",
        )
    )

    # 6. Analytical Limitations & Warnings
    limitations.append(
        ObservationItem(
            category="Limitation",
            headline="Correlation vs Causation",
            detail=(
                "All reported correlations reflect mathematical co-variation in the uploaded sample. "
                "Confounding variables, reverse causality, or selection bias cannot be ruled out."
            ),
        )
    )

    if total_rows < 50:
        limitations.append(
            ObservationItem(
                category="Limitation",
                headline="Small Sample Size Warning (N < 50)",
                detail=(
                    f"The dataset contains only {total_rows} observations. Statistical estimates "
                    "(skewness, kurtosis, correlation coefficients) carry wider confidence intervals."
                ),
            )
        )

    return FirstLookReport(
        facts=facts,
        quality_highlights=quality_highlights,
        relationships=relationships,
        distribution_insights=distribution_insights,
        next_steps=next_steps,
        limitations=limitations,
        visual_candidates=candidates,
    )
