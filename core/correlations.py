"""Advanced correlation exploration engine for EDA Intelligence Workbench.

Computes Pearson, Spearman, and Kendall correlation matrices along with exact
pairwise sample sizes, p-values, multiple-testing adjustments (FDR / Bonferroni),
and ranked relationship tables with degenerate input safeguards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.profiling import DatasetProfile, compute_dataset_profile


@dataclass
class CorrelatedPairExtended:
    """Detailed record of a bivariate correlation pair with sample size and inference."""

    column_a: str
    column_b: str
    correlation: float
    abs_correlation: float
    p_value: float | None
    p_value_adjusted: float | None
    is_statistically_significant: bool
    sample_size: int
    method: str
    strength: str  # "Very Strong", "Strong", "Moderate", "Weak"
    direction: str  # "Positive", "Negative", "None"


@dataclass
class CorrelationExplorerResult:
    """Structured output from the advanced correlation analysis."""

    method: str
    corr_matrix: pd.DataFrame
    p_value_matrix: pd.DataFrame
    sample_size_matrix: pd.DataFrame
    ranked_pairs: list[CorrelatedPairExtended]
    excluded_columns: list[str]
    warnings: list[str]
    disclaimer: str = (
        "Correlation measures mathematical association, NOT causation. A high correlation "
        "does not establish that changes in one variable cause changes in the other."
    )

    @property
    def matrix(self) -> pd.DataFrame:
        """Alias for corr_matrix."""
        return self.corr_matrix

    @property
    def p_values(self) -> pd.DataFrame:
        """Alias for p_value_matrix."""
        return self.p_value_matrix


def adjust_p_values_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg False Discovery Rate (FDR) adjustment for multiple comparisons."""
    m = len(p_values)
    if m <= 1:
        return p_values

    # Pair with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m

    cum_min = 1.0
    for rank, (orig_idx, p_val) in reversed(list(enumerate(indexed, start=1))):
        adj_p = min(1.0, (p_val * m) / rank)
        cum_min = min(cum_min, adj_p)
        adjusted[orig_idx] = cum_min

    return adjusted


# Alias for backwards compatibility
adjust_pvalues_benjamini_hochberg = adjust_p_values_fdr


def compute_advanced_correlations(
    df: pd.DataFrame,
    profile: DatasetProfile,
    method: str = "spearman",
    min_threshold: float = 0.0,
    max_columns: int = 40,
    alpha: float = 0.05,
) -> CorrelationExplorerResult:
    """Compute pairwise correlation coefficients, p-values, and sample sizes across valid numeric columns."""
    warnings: list[str] = []
    excluded: list[str] = []
    method_clean = method.lower()

    if method_clean not in ("pearson", "spearman", "kendall"):
        method_clean = "spearman"

    # Screen numeric columns
    valid_cols: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_str = str(col)
        col_prof = profile.column_profiles.get(col_str)
        if col_prof:
            if col_prof.is_all_null:
                excluded.append(f"{col_str} (100% missing)")
                continue
            if col_prof.is_constant or col_prof.is_zero_variance:
                excluded.append(f"{col_str} (Zero variance / constant)")
                continue
            if col_prof.non_null_count < 3:
                excluded.append(f"{col_str} (< 3 observations)")
                continue
        valid_cols.append(col_str)

    if len(valid_cols) > max_columns:
        warnings.append(
            f"Dataset has {len(valid_cols)} numeric columns. Analysis was bounded to the first "
            f"{max_columns} columns to ensure system responsiveness."
        )
        valid_cols = valid_cols[:max_columns]

    n_cols = len(valid_cols)
    if n_cols < 2:
        return CorrelationExplorerResult(
            method=method_clean.capitalize(),
            corr_matrix=pd.DataFrame(),
            p_value_matrix=pd.DataFrame(),
            sample_size_matrix=pd.DataFrame(),
            ranked_pairs=[],
            excluded_columns=excluded,
            warnings=["At least two non-constant numeric columns with >= 3 observations are required."],
        )

    # Initialize matrices
    corr_mat = pd.DataFrame(np.eye(n_cols), index=valid_cols, columns=valid_cols)
    pval_mat = pd.DataFrame(np.zeros((n_cols, n_cols)), index=valid_cols, columns=valid_cols)
    n_mat = pd.DataFrame(np.zeros((n_cols, n_cols), dtype=int), index=valid_cols, columns=valid_cols)

    pairs_data: list[tuple[str, str, float, float, int]] = []

    for i in range(n_cols):
        col_a = valid_cols[i]
        for j in range(i, n_cols):
            col_b = valid_cols[j]
            if i == j:
                corr_mat.loc[col_a, col_b] = 1.0
                pval_mat.loc[col_a, col_b] = 0.0
                n_mat.loc[col_a, col_b] = int(df[col_a].notna().sum())
                continue

            sub = df[[col_a, col_b]].dropna()
            pair_n = len(sub)
            n_mat.loc[col_a, col_b] = pair_n
            n_mat.loc[col_b, col_a] = pair_n

            if pair_n < 3:
                corr_mat.loc[col_a, col_b] = np.nan
                corr_mat.loc[col_b, col_a] = np.nan
                pval_mat.loc[col_a, col_b] = np.nan
                pval_mat.loc[col_b, col_a] = np.nan
                continue

            arr_a = sub[col_a].to_numpy(dtype=float)
            arr_b = sub[col_b].to_numpy(dtype=float)

            if np.all(arr_a == arr_a[0]) or np.all(arr_b == arr_b[0]):
                r, p = 0.0, 1.0
            else:
                try:
                    if method_clean == "pearson":
                        res = stats.pearsonr(arr_a, arr_b)
                        r, p = float(res.statistic), float(res.pvalue)
                    elif method_clean == "kendall":
                        res = stats.kendalltau(arr_a, arr_b)
                        r, p = float(res.statistic), float(res.pvalue)
                    else:  # spearman
                        res = stats.spearmanr(arr_a, arr_b)
                        r, p = float(res.statistic), float(res.pvalue)

                    if math.isnan(r):
                        r, p = 0.0, 1.0
                except (ValueError, ZeroDivisionError, RuntimeWarning):
                    r, p = 0.0, 1.0

            corr_mat.loc[col_a, col_b] = round(r, 4)
            corr_mat.loc[col_b, col_a] = round(r, 4)
            pval_mat.loc[col_a, col_b] = round(p, 5)
            pval_mat.loc[col_b, col_a] = round(p, 5)

            if abs(r) >= min_threshold:
                pairs_data.append((col_a, col_b, r, p, pair_n))

    # Apply FDR correction on collected pairs
    ranked_pairs: list[CorrelatedPairExtended] = []
    if pairs_data:
        raw_p_vals = [p[3] for p in pairs_data]
        adj_p_vals = adjust_p_values_fdr(raw_p_vals)

        for (col_a, col_b, r, p, pair_n), adj_p in zip(pairs_data, adj_p_vals):
            abs_r = abs(r)
            if abs_r >= 0.8:
                strength = "Very Strong"
            elif abs_r >= 0.6:
                strength = "Strong"
            elif abs_r >= 0.35:
                strength = "Moderate"
            else:
                strength = "Weak"

            direction = "Positive" if r > 0.05 else ("Negative" if r < -0.05 else "None")
            is_sig = adj_p < alpha

            ranked_pairs.append(
                CorrelatedPairExtended(
                    column_a=col_a,
                    column_b=col_b,
                    correlation=round(r, 4),
                    abs_correlation=round(abs_r, 4),
                    p_value=round(p, 5),
                    p_value_adjusted=round(adj_p, 5),
                    is_statistically_significant=is_sig,
                    sample_size=pair_n,
                    method=method_clean.capitalize(),
                    strength=strength,
                    direction=direction,
                )
            )

        ranked_pairs.sort(key=lambda x: x.abs_correlation, reverse=True)

    return CorrelationExplorerResult(
        method=method_clean.capitalize(),
        corr_matrix=corr_mat,
        p_value_matrix=pval_mat,
        sample_size_matrix=n_mat,
        ranked_pairs=ranked_pairs,
        excluded_columns=excluded,
        warnings=warnings,
    )


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "spearman",
    min_threshold: float = 0.0,
    max_columns: int = 40,
    alpha: float = 0.05,
    apply_fdr: bool = True,
) -> CorrelationExplorerResult:
    """Compute correlation matrix with optional column subsetting and auto-profile generation."""
    sub_df = df[columns].copy() if columns else df
    profile = compute_dataset_profile(sub_df)
    return compute_advanced_correlations(
        sub_df,
        profile,
        method=method,
        min_threshold=min_threshold,
        max_columns=max_columns,
        alpha=alpha,
    )


def extract_ranked_correlations(
    corr_result: CorrelationExplorerResult,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Extract top N correlated pairs as dictionaries for tables, reports, and legacy APIs."""
    output: list[dict[str, Any]] = []
    for pair in corr_result.ranked_pairs[:top_n]:
        adj_p = pair.p_value_adjusted if pair.p_value_adjusted is not None else 1.0
        output.append({
            "col1": pair.column_a,
            "col2": pair.column_b,
            "feature_a": pair.column_a,
            "feature_b": pair.column_b,
            "r": pair.correlation,
            "correlation": pair.correlation,
            "abs_r": pair.abs_correlation,
            "p_value": pair.p_value if pair.p_value is not None else 1.0,
            "p_value_adjusted": adj_p,
            "adj_p_value": adj_p,
            "sample_size": pair.sample_size,
            "is_significant": pair.is_statistically_significant,
            "significant": pair.is_statistically_significant,
            "strength": pair.strength,
            "direction": pair.direction,
        })
    return output

