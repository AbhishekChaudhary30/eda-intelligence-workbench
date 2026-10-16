"""Deterministic rule-based automated insight engine for EDA Intelligence Workbench.

Extracts evidence-based findings across 9 analytical categories, assigns transparent
priority scores, provides actionable remediation steps, and avoids hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.correlations import compute_advanced_correlations
from core.missingness import analyze_missingness
from core.outlier_analysis import run_comprehensive_outlier_analysis
from core.profiling import DatasetProfile


@dataclass
class StructuredInsight:
    """A single deterministic, evidence-backed finding."""

    id: str
    category: str  # "Data Quality", "Missingness", "Distribution", "Outliers", "Relationships", "Categorical Patterns", "Time-Series Patterns", "Potential Data Problems", "Analytical Opportunities"
    title: str
    description: str
    evidence: str
    severity: str  # "Critical", "Warning", "Informational", "Opportunity"
    priority_score: float  # 0.0 to 100.0
    affected_columns: list[str]
    recommended_next_step: str
    confidence: str = "High"  # "High" or "Medium"
    limitations: str = ""
    source_analysis: str = ""

    @property
    def recommended_action(self) -> str:
        """Alias for recommended_next_step."""
        return self.recommended_next_step

    @property
    def insight_id(self) -> str:
        """Alias for id."""
        return self.id


AnalyticalInsight = StructuredInsight


@dataclass
class InsightCatalog:
    """Collection of ranked and filterable analytical insights."""

    total_insights: int
    insights: list[StructuredInsight]
    category_counts: dict[str, int]
    severity_counts: dict[str, int]

    def filter(
        self,
        category: str | None = None,
        severity: str | None = None,
        column: str | None = None,
    ) -> list[StructuredInsight]:
        """Filter insights by category, severity, or affected column."""
        res = self.insights
        if category and category != "All Categories":
            res = [i for i in res if i.category == category]
        if severity and severity != "All Severities":
            res = [i for i in res if i.severity == severity]
        if column and column != "All Columns":
            res = [i for i in res if column in i.affected_columns]
        return res


def generate_automated_insights(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> InsightCatalog:
    """Generate deterministic, ranked analytical insights across the dataset."""
    insights: list[StructuredInsight] = []
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0 or total_cols == 0:
        return InsightCatalog(
            total_insights=0,
            insights=[],
            category_counts={},
            severity_counts={},
        )

    # 1. MISSINGNESS INSIGHTS
    miss_rep = analyze_missingness(df)
    if miss_rep.total_missing_cells > 0:
        for _, row in miss_rep.column_missing_summary.iterrows():
            col_name = str(row["Column Name"])
            n_miss = int(row["Missing Count"])
            pct_miss = float(row["Missing %"])

            if pct_miss >= 50.0:
                insights.append(
                    StructuredInsight(
                        id=f"miss_critical_{col_name}",
                        category="Missingness",
                        title=f"Severe Missingness in '{col_name}' ({pct_miss:.1f}%)",
                        description=f"Over half of all values in column '{col_name}' are absent.",
                        evidence=f"{n_miss:,} missing records out of {total_rows:,} total rows ({pct_miss:.1f}%).",
                        severity="Critical",
                        priority_score=85.0 + min(15.0, pct_miss / 10.0),
                        affected_columns=[col_name],
                        recommended_next_step=(
                            "Assess whether this column can be imputed or should be excluded from modeling "
                            "to prevent significant bias."
                        ),
                        confidence="High",
                        limitations="Missingness mechanism cannot be deduced solely from frequency.",
                        source_analysis="Missingness Audit",
                    )
                )
            elif pct_miss >= 10.0:
                insights.append(
                    StructuredInsight(
                        id=f"miss_warning_{col_name}",
                        category="Missingness",
                        title=f"Moderate Missingness in '{col_name}' ({pct_miss:.1f}%)",
                        description=f"Feature '{col_name}' has a substantial portion of missing records.",
                        evidence=f"{n_miss:,} missing records ({pct_miss:.1f}%).",
                        severity="Warning",
                        priority_score=50.0 + (pct_miss / 2.0),
                        affected_columns=[col_name],
                        recommended_next_step="Apply median/mode imputation or explore group-wise missingness patterns.",
                        confidence="High",
                        limitations="Assumes data is missing at random or completely at random.",
                        source_analysis="Missingness Audit",
                    )
                )

    # Check for custom tokens
    if miss_rep.custom_token_occurrences:
        tok = miss_rep.custom_token_occurrences[0]
        insights.append(
            StructuredInsight(
                id=f"token_problem_{tok.column_name}_{tok.token}",
                category="Potential Data Problems",
                title=f"Pseudo-Missing Text Token '{tok.token}' in '{tok.column_name}'",
                description=f"Found text token '{tok.token}' masquerading as valid data.",
                evidence=f"{tok.count:,} occurrences ({tok.percentage:.1f}% of column).",
                severity="Warning",
                priority_score=55.0,
                affected_columns=[tok.column_name],
                recommended_next_step="Replace pseudo-missing string representations with standard NaN values.",
                confidence="High",
                source_analysis="Token Pattern Scanner",
            )
        )

    # 2. DATA QUALITY & DEGENERATE COLUMNS
    if profile.duplicate_rows_count > 0:
        dup_pct = profile.duplicate_rows_percentage
        insights.append(
            StructuredInsight(
                id="dup_rows_audit",
                category="Data Quality",
                title=f"{profile.duplicate_rows_count:,} Exact Duplicate Rows Detected",
                description="The dataset contains identical repeated rows across all columns.",
                evidence=f"{profile.duplicate_rows_count:,} duplicate rows ({dup_pct:.1f}% of dataset).",
                severity="Warning" if dup_pct < 20 else "Critical",
                priority_score=60.0 + min(30.0, dup_pct),
                affected_columns=list(df.columns),
                recommended_next_step="Inspect whether duplicates represent legitimate repeated transactions or pipeline duplication.",
                confidence="High",
                source_analysis="Duplicate Row Audit",
            )
        )

    for col_str in profile.constant_columns:
        insights.append(
            StructuredInsight(
                id=f"const_col_{col_str}",
                category="Potential Data Problems",
                title=f"Zero Variance / Constant Feature: '{col_str}'",
                description=f"Feature '{col_str}' contains only one single unique value across all records.",
                evidence="Variance = 0.0. 100% of observations contain the identical value.",
                severity="Warning",
                priority_score=65.0,
                affected_columns=[col_str],
                recommended_next_step="Exclude from predictive modeling as zero-variance features contribute no information.",
                confidence="High",
                source_analysis="Variance Audit",
            )
        )

    # 3. OUTLIER INSIGHTS
    outlier_rep = run_comprehensive_outlier_analysis(df, method="iqr", threshold=1.5)
    for col_str, audit in outlier_rep.column_audits.items():
        if audit.outlier_count > 0 and audit.outlier_percentage >= 3.0:
            sev = "Warning" if audit.outlier_percentage >= 5.0 else "Informational"
            insights.append(
                StructuredInsight(
                    id=f"outlier_iqr_{col_str}",
                    category="Outliers",
                    title=f"Statistical Outliers in '{col_str}' ({audit.outlier_percentage:.1f}%)",
                    description=f"{audit.outlier_count:,} values fall beyond Tukey's fences [{audit.lower_bound}, {audit.upper_bound}].",
                    evidence=f"Range: [{audit.min_extreme}, {audit.max_extreme}] vs bounds [{audit.lower_bound}, {audit.upper_bound}].",
                    severity=sev,
                    priority_score=40.0 + min(30.0, audit.outlier_percentage * 2.0),
                    affected_columns=[col_str],
                    recommended_next_step="Inspect flagged extreme records in the Outlier Investigation panel; avoid blind deletion.",
                    confidence="High",
                    limitations="IQR method identifies statistical dispersion, not confirmed domain errors.",
                    source_analysis="IQR Outlier Audit",
                )
            )

    # 4. RELATIONSHIP & CORRELATION INSIGHTS
    if profile.numeric_columns_count >= 2:
        corr_res = compute_advanced_correlations(df, profile, method="spearman", min_threshold=0.35)
        for pair in corr_res.ranked_pairs[:4]:
            if pair.abs_correlation >= 0.5:
                direction_desc = "positive (co-varying)" if pair.correlation > 0 else "inverse (opposing)"
                sig_desc = f"Statistically significant (adjusted p = {pair.p_value_adjusted:.4f})" if pair.is_statistically_significant else "Exploratory sample"
                insights.append(
                    StructuredInsight(
                        id=f"corr_{pair.column_a}_{pair.column_b}",
                        category="Relationships",
                        title=f"{pair.strength} {pair.direction} Association: '{pair.column_a}' & '{pair.column_b}'",
                        description=f"Observed {direction_desc} monotonic rank correlation (r = {pair.correlation:.2f}).",
                        evidence=f"Spearman r = {pair.correlation:.3f}, N = {pair.sample_size:,}. {sig_desc}.",
                        severity="Opportunity" if pair.is_statistically_significant else "Informational",
                        priority_score=45.0 + (pair.abs_correlation * 40.0),
                        affected_columns=[pair.column_a, pair.column_b],
                        recommended_next_step="Inspect pairwise scatter plot with trendline to evaluate linearity and check for confounding variables.",
                        confidence="High" if pair.sample_size >= 30 else "Medium",
                        limitations="Correlation establishes mathematical association, NEVER causality.",
                        source_analysis="Bivariate Correlation Analysis",
                    )
                )

    # 5. DISTRIBUTION & NORMALITY INSIGHTS
    for col_str, col_p in profile.column_profiles.items():
        if col_p.skewness is not None and abs(col_p.skewness) >= 1.5:
            skew_dir = "right (positive)" if col_p.skewness > 0 else "left (negative)"
            insights.append(
                StructuredInsight(
                    id=f"skew_{col_str}",
                    category="Distribution",
                    title=f"Substantial Asymmetry in '{col_str}' (Skewness = {col_p.skewness:.2f})",
                    description=f"Feature '{col_str}' exhibits significant {skew_dir} skewness.",
                    evidence=f"Skewness = {col_p.skewness:.2f}, Mean = {col_p.mean:.2f}, Median = {col_p.median:.2f}.",
                    severity="Informational",
                    priority_score=40.0 + min(20.0, abs(col_p.skewness) * 5.0),
                    affected_columns=[col_str],
                    recommended_next_step="Consider logarithmic (log1p) or Box-Cox transformations if fitting linear regression models.",
                    confidence="High",
                    limitations="Sample moments can be sensitive to isolated extreme observations.",
                    source_analysis="Distribution Moment Analysis",
                )
            )

    # 6. CATEGORICAL PATTERNS & IDENTIFIERS
    for col_str, col_p in profile.column_profiles.items():
        if col_p.is_potential_identifier:
            insights.append(
                StructuredInsight(
                    id=f"id_candidate_{col_str}",
                    category="Categorical Patterns",
                    title=f"Potential Key / Identifier Detected: '{col_str}'",
                    description=f"Column '{col_str}' exhibits identifier-like characteristics.",
                    evidence=f"Uniqueness ratio = {col_p.uniqueness_ratio:.1%} ({col_p.distinct_count:,} unique values). Reason: {col_p.identifier_reason}",
                    severity="Informational",
                    priority_score=35.0,
                    affected_columns=[col_str],
                    recommended_next_step="Exclude from predictive features to prevent data leakage and excessive cardinality.",
                    confidence="Medium",
                    limitations="Identifier detection is heuristic based on uniqueness and nomenclature.",
                    source_analysis="Type Inference & Key Audit",
                )
            )
        elif col_p.semantic_type == "Categorical" and col_p.distinct_count > 50 and col_p.uniqueness_ratio > 0.4:
            insights.append(
                StructuredInsight(
                    id=f"high_card_{col_str}",
                    category="Categorical Patterns",
                    title=f"High-Cardinality Category: '{col_str}' ({col_p.distinct_count:,} levels)",
                    description=f"Feature '{col_str}' contains many distinct levels relative to dataset size.",
                    evidence=f"{col_p.distinct_count:,} distinct categories in {col_p.non_null_count:,} observations.",
                    severity="Informational",
                    priority_score=30.0,
                    affected_columns=[col_str],
                    recommended_next_step="Group long-tail infrequent categories into an 'Other' bucket using the Transformation Workbench.",
                    confidence="High",
                    source_analysis="Categorical Cardinality Analysis",
                )
            )

    # 7. TIME-SERIES PATTERNS (if datetime exists)
    for col_str, col_p in profile.column_profiles.items():
        if col_p.semantic_type == "Datetime" and col_p.min_date:
            insights.append(
                StructuredInsight(
                    id=f"ts_temporal_{col_str}",
                    category="Time-Series Patterns",
                    title=f"Temporal Coverage in '{col_str}'",
                    description="Detected valid timestamp sequence suitable for trend and lag exploration.",
                    evidence=f"Span: {col_p.min_date} to {col_p.max_date} ({col_p.temporal_span_days:.0f} days).",
                    severity="Opportunity",
                    priority_score=50.0,
                    affected_columns=[col_str],
                    recommended_next_step="Explore seasonality, rolling averages, and autocorrelation in the Visualization Studio.",
                    confidence="High",
                    source_analysis="Datetime Profiling",
                )
            )
            break

    # Sort descending by priority score
    insights.sort(key=lambda x: x.priority_score, reverse=True)

    # Deduplicate by title
    seen_titles: set[str] = set()
    deduped: list[StructuredInsight] = []
    for ins in insights:
        if ins.title not in seen_titles:
            seen_titles.add(ins.title)
            deduped.append(ins)

    cat_counts: dict[str, int] = {}
    sev_counts: dict[str, int] = {}
    for ins in deduped:
        cat_counts[ins.category] = cat_counts.get(ins.category, 0) + 1
        sev_counts[ins.severity] = sev_counts.get(ins.severity, 0) + 1

    return InsightCatalog(
        total_insights=len(deduped),
        insights=deduped,
        category_counts=cat_counts,
        severity_counts=sev_counts,
    )


def generate_deterministic_insights(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality_report: Any = None,
    settings: Any = None,
) -> list[StructuredInsight]:
    """Compatibility function returning list of deterministic StructuredInsight objects."""
    catalog = generate_automated_insights(df, profile)
    return catalog.insights


def filter_insights(
    insights: list[StructuredInsight],
    category: str | None = None,
    severity: str | None = None,
    affected_column: str | None = None,
) -> list[StructuredInsight]:
    """Filter insight list by category, severity, or affected column."""
    res = list(insights)
    if category:
        res = [i for i in res if i.category.lower() == category.lower()]
    if severity:
        res = [i for i in res if i.severity.lower() == severity.lower()]
    if affected_column:
        res = [i for i in res if affected_column in i.affected_columns]
    return res
