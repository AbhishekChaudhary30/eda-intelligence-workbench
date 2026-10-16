"""Dedicated missingness analysis engine for EDA Intelligence Workbench.

Audits null counts, row-level missingness distributions, nullity correlation matrices,
group-wise missingness breakdowns, and custom missing-data tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CustomTokenOccurrence:
    """Detection of pseudo-missing string tokens (e.g. '?', 'NA', '-999')."""

    column_name: str
    token: str
    count: int
    percentage: float


@dataclass
class MissingnessReport:
    """Comprehensive structured output for the missingness investigation panel."""

    total_rows: int
    total_cols: int
    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    complete_rows_count: int
    complete_rows_pct: float
    columns_with_missing_count: int
    column_missing_summary: pd.DataFrame
    row_missing_distribution: pd.DataFrame
    nullity_correlation_matrix: pd.DataFrame
    custom_token_occurrences: list[CustomTokenOccurrence]
    warnings: list[str] = field(default_factory=list)


def analyze_missingness(
    df: pd.DataFrame,
    custom_tokens: list[str] | None = None,
) -> MissingnessReport:
    """Compute detailed missingness metrics, pattern correlation, and custom tokens."""
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    warnings: list[str] = []

    if total_rows == 0 or total_cols == 0:
        return MissingnessReport(
            total_rows=total_rows,
            total_cols=total_cols,
            total_cells=0,
            total_missing_cells=0,
            overall_missing_pct=0.0,
            complete_rows_count=0,
            complete_rows_pct=0.0,
            columns_with_missing_count=0,
            column_missing_summary=pd.DataFrame(),
            row_missing_distribution=pd.DataFrame(),
            nullity_correlation_matrix=pd.DataFrame(),
            custom_token_occurrences=[],
            warnings=["Dataset has zero records or zero columns."],
        )

    # 1. Column-level missingness table
    col_records = []
    null_bool_df = df.isna()
    total_missing_cells = int(null_bool_df.to_numpy().sum())
    overall_pct = (total_missing_cells / max(1, total_cells)) * 100.0

    for col in df.columns:
        col_str = str(col)
        n_missing = int(null_bool_df[col].sum())
        pct_missing = (n_missing / total_rows) * 100.0

        status = "Complete (0%)"
        if pct_missing >= 60.0:
            status = "Critical Missing (>60%)"
        elif pct_missing >= 15.0:
            status = "Moderate Missing (15-60%)"
        elif pct_missing > 0.0:
            status = "Low Missing (<15%)"

        col_records.append({
            "Column Name": col_str,
            "Missing Count": n_missing,
            "Missing %": round(pct_missing, 2),
            "Present Count": total_rows - n_missing,
            "Status": status,
        })

    col_missing_df = pd.DataFrame(col_records).sort_values("Missing %", ascending=False)
    cols_with_missing = int((col_missing_df["Missing Count"] > 0).sum())

    # 2. Row-level missingness distribution
    missing_per_row = null_bool_df.sum(axis=1)
    complete_rows = int((missing_per_row == 0).sum())
    complete_pct = (complete_rows / total_rows) * 100.0

    row_dist_counts = missing_per_row.value_counts().sort_index()
    row_dist_records = []
    for num_missing, count in row_dist_counts.items():
        pct = round((count / total_rows) * 100.0, 2)
        row_dist_records.append({
            "Missing Columns Per Row": int(num_missing),
            "Number of Rows": int(count),
            "Percentage of Dataset": pct,
            "Percentage": pct,
        })
    row_dist_df = pd.DataFrame(row_dist_records)

    # 3. Nullity correlation matrix (correlation of missingness flags between columns)
    cols_with_some_missing = [c for c in df.columns if 0 < null_bool_df[c].sum() < total_rows]
    if len(cols_with_some_missing) >= 2:
        # Compute correlation between boolean missingness indicators
        nullity_corr = null_bool_df[cols_with_some_missing].astype(int).corr().round(3)
    else:
        nullity_corr = pd.DataFrame()

    # 4. Custom missing tokens scan
    tokens_to_search = custom_tokens or ["NA", "N/A", "null", "None", "none", "?", "-999", "missing", "nan"]
    token_findings: list[CustomTokenOccurrence] = []

    for col in df.columns:
        col_str = str(col)
        s = df[col].dropna()
        if s.empty:
            continue

        if pd.api.types.is_numeric_dtype(s):
            # For numeric features, only search tokens that represent valid numeric sentinels (e.g. -999)
            for tok in tokens_to_search:
                try:
                    num_val = float(tok)
                    cnt = int((s == num_val).sum())
                    if cnt > 0:
                        token_findings.append(
                            CustomTokenOccurrence(
                                column_name=col_str,
                                token=tok,
                                count=cnt,
                                percentage=round((cnt / total_rows) * 100.0, 2),
                            )
                        )
                except ValueError:
                    continue
        else:
            # For text/object columns, use fast value_counts() hash table lookup
            val_counts = s.astype(str).str.strip().value_counts()
            for tok in tokens_to_search:
                cnt = int(val_counts.get(tok, 0))
                if cnt > 0:
                    token_findings.append(
                        CustomTokenOccurrence(
                            column_name=col_str,
                            token=tok,
                            count=cnt,
                            percentage=round((cnt / total_rows) * 100.0, 2),
                        )
                    )

    if token_findings:
        warnings.append(
            f"Detected {len(token_findings)} instances of potential pseudo-missing text tokens "
            f"(e.g. {', '.join({t.token for t in token_findings[:3]})}) that may mask true missingness."
        )

    return MissingnessReport(
        total_rows=total_rows,
        total_cols=total_cols,
        total_cells=total_cells,
        total_missing_cells=total_missing_cells,
        overall_missing_pct=round(overall_pct, 2),
        complete_rows_count=complete_rows,
        complete_rows_pct=round(complete_pct, 1),
        columns_with_missing_count=cols_with_missing,
        column_missing_summary=col_missing_df,
        row_missing_distribution=row_dist_df,
        nullity_correlation_matrix=nullity_corr,
        custom_token_occurrences=token_findings,
        warnings=warnings,
    )


def compute_group_missingness(
    df: pd.DataFrame,
    target_col: str | None = None,
    group_col: str = "",
) -> pd.DataFrame:
    """Compute missingness rates of target_col (or overall dataset) across categorical segments of group_col."""
    if group_col not in df.columns:
        return pd.DataFrame()

    if target_col is not None and target_col in df.columns:
        sub = df[[target_col, group_col]].copy()
        sub["is_missing"] = sub[target_col].isna().astype(int)

        agg = sub.groupby(group_col).agg(
            Total_Count=("is_missing", "count"),
            Missing_Count=("is_missing", "sum"),
        ).reset_index()

        agg["Missing_Pct"] = (agg["Missing_Count"] / agg["Total_Count"] * 100.0).round(2)
        agg["Present_Count"] = agg["Total_Count"] - agg["Missing_Count"]
        return agg.sort_values("Missing_Pct", ascending=False)

    # Overall dataset row missingness segmented by group_col
    sub = pd.DataFrame({
        group_col: df[group_col],
        "is_missing": df.isna().any(axis=1).astype(int),
    })
    agg = sub.groupby(group_col).agg(
        Total_Count=("is_missing", "count"),
        Rows_With_Missing=("is_missing", "sum"),
    ).reset_index()
    agg["Missing_Pct"] = (agg["Rows_With_Missing"] / agg["Total_Count"] * 100.0).round(2)
    agg["Complete_Rows"] = agg["Total_Count"] - agg["Rows_With_Missing"]
    return agg.sort_values("Missing_Pct", ascending=False)


def compute_nullity_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute nullity correlation matrix between features."""
    report = analyze_missingness(df)
    return report.nullity_correlation_matrix


def compute_row_missingness_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of missing features per row."""
    report = analyze_missingness(df)
    df_dist = report.row_missing_distribution.copy()
    if not df_dist.empty and "Missing Columns Per Row" in df_dist.columns:
        df_dist["Missing_Fields_Count"] = df_dist["Missing Columns Per Row"]
        df_dist["Row_Count"] = df_dist["Number of Rows"]
    return df_dist


def detect_pseudo_missing_tokens(
    df: pd.DataFrame,
    tokens: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Scan dataset for pseudo-missing tokens and return dictionary representations."""
    report = analyze_missingness(df, custom_tokens=tokens)
    return [
        {
            "column": occ.column_name,
            "token": occ.token,
            "count": occ.count,
            "percentage": occ.percentage,
        }
        for occ in report.custom_token_occurrences
    ]

