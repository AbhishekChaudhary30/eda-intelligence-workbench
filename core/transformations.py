"""Interactive data transformation engine for EDA Intelligence Workbench.

Provides validated, traceable, non-destructive transformations across columns, rows,
missing values, numerical scaling, and categorical encodings with full undo and replay support.
Zero use of eval or exec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


class TransformationError(Exception):
    """User-facing actionable error during dataset transformation."""


@dataclass
class TransformationStep:
    """Audit record of an applied dataset transformation."""

    step_id: int
    operation_type: str  # "drop_column", "rename_column", "cast_type", "filter_rows", "drop_duplicates", "impute_missing", "scale_numeric", "log_transform", "group_categories", "one_hot_encode"
    description: str
    parameters: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0


@dataclass
class TransformationPreview:
    """Pre-execution preview showing expected impact and potential data loss."""

    operation_name: str
    expected_rows_after: int
    expected_cols_after: int
    row_delta: int
    col_delta: int
    preview_head: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def preview_transformation(
    df: pd.DataFrame,
    operation_type: str,
    params: dict[str, Any],
) -> TransformationPreview:
    """Generate a lightweight preview of an operation on a defensive copy."""
    try:
        temp_df = df.head(100).copy() if len(df) > 100 else df.copy()
        transformed_df, _ = apply_single_transformation(temp_df, operation_type, params)

        # Estimate full row count if filtering
        if operation_type in ("filter_rows", "drop_duplicates", "drop_na_rows"):
            full_sim, _ = apply_single_transformation(df, operation_type, params)
            est_rows = len(full_sim)
        else:
            est_rows = len(df)

        est_cols = len(transformed_df.columns)
        warnings: list[str] = []

        if est_rows == 0:
            warnings.append("Caution: This operation will filter out 100% of rows (resulting dataset is empty).")

        return TransformationPreview(
            operation_name=operation_type,
            expected_rows_after=est_rows,
            expected_cols_after=est_cols,
            row_delta=est_rows - len(df),
            col_delta=est_cols - len(df.columns),
            preview_head=transformed_df.head(10),
            warnings=warnings,
        )
    except Exception as exc:
        raise TransformationError(f"Preview failed: {exc}") from exc


def apply_single_transformation(
    df: pd.DataFrame,
    operation_type: str,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Execute a single validated transformation on a DataFrame copy.

    Returns:
        tuple: (new_transformed_df, description_text)
    """
    res_df = df.copy()
    op = operation_type.lower()

    # 1. DROP COLUMNS
    if op == "drop_columns":
        cols_to_drop = [c for c in params.get("columns", []) if c in res_df.columns]
        if not cols_to_drop:
            raise TransformationError("No matching columns specified to drop.")
        res_df = res_df.drop(columns=cols_to_drop)
        desc = f"Dropped column(s): {', '.join(cols_to_drop)}"

    # 2. RENAME COLUMN
    elif op == "rename_column":
        old_name = params.get("old_name", "")
        new_name = str(params.get("new_name", "")).strip()
        if old_name not in res_df.columns:
            raise TransformationError(f"Column '{old_name}' not found.")
        if not new_name:
            raise TransformationError("New column name cannot be empty.")
        if new_name in res_df.columns and new_name != old_name:
            raise TransformationError(f"Column '{new_name}' already exists in dataset.")
        res_df = res_df.rename(columns={old_name: new_name})
        desc = f"Renamed '{old_name}' to '{new_name}'"

    # 3. CAST DATA TYPE
    elif op == "cast_type":
        col = params.get("column", "")
        target_type = params.get("target_type", "").lower()
        if col not in res_df.columns:
            raise TransformationError(f"Column '{col}' not found.")

        try:
            if target_type == "float":
                res_df[col] = pd.to_numeric(res_df[col], errors="coerce").astype(float)
            elif target_type == "int":
                res_df[col] = pd.to_numeric(res_df[col], errors="coerce").round().astype("Int64")
            elif target_type == "string":
                res_df[col] = res_df[col].astype(str)
            elif target_type == "datetime":
                res_df[col] = pd.to_datetime(res_df[col], errors="coerce")
            elif target_type == "category":
                res_df[col] = res_df[col].astype("category")
            elif target_type == "boolean":
                res_df[col] = res_df[col].astype(bool)
            else:
                raise TransformationError(f"Unsupported target type '{target_type}'.")
        except Exception as e:
            raise TransformationError(f"Type casting failed: {e}") from e
        desc = f"Cast column '{col}' to '{target_type}'"

    # 4. FILTER ROWS (Safe condition without eval/exec)
    elif op == "filter_rows":
        col = params.get("column", "")
        operator = params.get("operator", "==")
        val = params.get("value")

        if col not in res_df.columns:
            raise TransformationError(f"Filter column '{col}' not found.")

        s = res_df[col]
        if operator == "is_null":
            mask = s.isna()
        elif operator == "not_null":
            mask = s.notna()
        elif operator == "contains":
            mask = s.astype(str).str.contains(str(val), case=False, na=False)
        else:
            # Numeric or equality comparisons
            try:
                if pd.api.types.is_numeric_dtype(s):
                    cmp_val = float(val)
                    col_vals = pd.to_numeric(s, errors="coerce")
                    if operator == "==":
                        mask = col_vals == cmp_val
                    elif operator == "!=":
                        mask = col_vals != cmp_val
                    elif operator == ">":
                        mask = col_vals > cmp_val
                    elif operator == ">=":
                        mask = col_vals >= cmp_val
                    elif operator == "<":
                        mask = col_vals < cmp_val
                    elif operator == "<=":
                        mask = col_vals <= cmp_val
                    else:
                        raise TransformationError(f"Unsupported operator '{operator}' for numeric comparison.")
                else:
                    # String comparison
                    str_val = str(val)
                    if operator == "==":
                        mask = s.astype(str) == str_val
                    elif operator == "!=":
                        mask = s.astype(str) != str_val
                    else:
                        raise TransformationError(f"Operator '{operator}' not supported for text column.")
            except Exception as e:
                raise TransformationError(f"Invalid filter evaluation: {e}") from e

        res_df = res_df[mask].reset_index(drop=True)
        desc = f"Filtered rows where `{col}` {operator} `{val if val is not None else ''}`"

    # 5. DROP EXACT DUPLICATES
    elif op == "drop_duplicates":
        before_cnt = len(res_df)
        res_df = res_df.drop_duplicates().reset_index(drop=True)
        removed = before_cnt - len(res_df)
        desc = f"Removed {removed:,} exact duplicate rows"

    # 6. IMPUTE MISSING VALUES
    elif op == "impute_missing":
        col = params.get("column", "")
        strategy = params.get("strategy", "mean").lower()  # "mean", "median", "mode", "constant"
        constant_val = params.get("constant_value", "")

        if col not in res_df.columns:
            raise TransformationError(f"Column '{col}' not found.")

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(res_df[col]):
                raise TransformationError("Mean imputation requires a numeric column.")
            mean_val = res_df[col].mean()
            res_df[col] = res_df[col].fillna(mean_val)
            desc = f"Imputed missing values in '{col}' using mean ({mean_val:.2f})"
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(res_df[col]):
                raise TransformationError("Median imputation requires a numeric column.")
            med_val = res_df[col].median()
            res_df[col] = res_df[col].fillna(med_val)
            desc = f"Imputed missing values in '{col}' using median ({med_val:.2f})"
        elif strategy == "mode":
            mode_series = res_df[col].mode(dropna=True)
            if mode_series.empty:
                raise TransformationError(f"Cannot compute mode for column '{col}'.")
            mode_val = mode_series.iloc[0]
            res_df[col] = res_df[col].fillna(mode_val)
            desc = f"Imputed missing values in '{col}' using mode ('{mode_val}')"
        elif strategy == "constant":
            res_df[col] = res_df[col].fillna(constant_val)
            desc = f"Imputed missing values in '{col}' with constant '{constant_val}'"
        else:
            raise TransformationError(f"Unsupported imputation strategy '{strategy}'.")

    # 7. SCALE NUMERIC FEATURE
    elif op == "scale_numeric":
        col = params.get("column", "")
        method = params.get("method", "standard").lower()  # "standard", "minmax", "robust"

        if col not in res_df.columns or not pd.api.types.is_numeric_dtype(res_df[col]):
            raise TransformationError(f"Column '{col}' is not a valid numeric column.")

        s = pd.to_numeric(res_df[col], errors="coerce")
        if method == "standard":
            std_v = s.std()
            if std_v == 0 or np.isnan(std_v):
                raise TransformationError("Cannot standardize column with zero variance.")
            res_df[col] = (s - s.mean()) / std_v
            desc = f"Standard scaled (Z-score) column '{col}'"
        elif method == "minmax":
            range_v = s.max() - s.min()
            if range_v == 0 or np.isnan(range_v):
                raise TransformationError("Cannot Min-Max scale column with zero range.")
            res_df[col] = (s - s.min()) / range_v
            desc = f"Min-Max scaled [0, 1] column '{col}'"
        elif method == "robust":
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr_v = q75 - q25
            if iqr_v == 0 or np.isnan(iqr_v):
                raise TransformationError("Cannot Robust-scale column with zero IQR.")
            res_df[col] = (s - s.median()) / iqr_v
            desc = f"Robust scaled (IQR-based) column '{col}'"
        else:
            raise TransformationError(f"Unsupported scaling method '{method}'.")

    # 8. LOG TRANSFORMATION (log1p)
    elif op == "log_transform":
        col = params.get("column", "")
        if col not in res_df.columns or not pd.api.types.is_numeric_dtype(res_df[col]):
            raise TransformationError(f"Column '{col}' must be numeric.")

        s = pd.to_numeric(res_df[col], errors="coerce")
        if (s < 0).any():
            raise TransformationError(
                f"Column '{col}' contains negative numbers. Logarithm is undefined for negative values."
            )
        res_df[col] = np.log1p(s)
        desc = f"Applied log(1 + x) transformation to '{col}'"

    # 9. GROUP INFREQUENT CATEGORIES
    elif op == "group_categories":
        col = params.get("column", "")
        top_n = int(params.get("top_n", 10))
        other_label = str(params.get("other_label", "Other"))

        if col not in res_df.columns:
            raise TransformationError(f"Column '{col}' not found.")

        top_cats = set(res_df[col].astype(str).value_counts().head(top_n).index)
        res_df[col] = res_df[col].astype(str).apply(lambda x: x if x in top_cats else other_label)
        desc = f"Grouped long-tail categories of '{col}' into '{other_label}' (preserved top {top_n})"

    # 10. ONE-HOT ENCODING
    elif op == "one_hot_encode":
        col = params.get("column", "")
        if col not in res_df.columns:
            raise TransformationError(f"Column '{col}' not found.")

        distinct_n = res_df[col].nunique(dropna=True)
        if distinct_n > 12:
            raise TransformationError(
                f"Column '{col}' has {distinct_n} categories. One-hot encoding is restricted to <= 12 levels."
            )

        dummies = pd.get_dummies(res_df[col], prefix=col, drop_first=params.get("drop_first", False), dtype=int)
        res_df = pd.concat([res_df.drop(columns=[col]), dummies], axis=1)
        desc = f"One-hot encoded '{col}' into {len(dummies.columns)} binary features"

    else:
        raise TransformationError(f"Unknown transformation operation '{operation_type}'.")

    return res_df, desc


class TransformationManager:
    """Manages working dataset mutations, undo history, and original preservation."""

    def __init__(self, original_df: pd.DataFrame):
        self.original_df = original_df.copy()
        self.working_df = original_df.copy()
        self.history: list[TransformationStep] = []

    def apply(self, operation_type: str, params: dict[str, Any]) -> TransformationStep:
        """Apply an operation to the working dataset and record step in history."""
        r_before, c_before = len(self.working_df), len(self.working_df.columns)
        new_df, desc = apply_single_transformation(self.working_df, operation_type, params)
        r_after, c_after = len(new_df), len(new_df.columns)

        step = TransformationStep(
            step_id=len(self.history) + 1,
            operation_type=operation_type,
            description=desc,
            parameters=params,
            rows_before=r_before,
            rows_after=r_after,
            cols_before=c_before,
            cols_after=c_after,
        )

        self.working_df = new_df
        self.history.append(step)
        return step

    def undo(self) -> bool:
        """Replay history up to step N - 1 from original dataset."""
        if not self.history:
            return False

        self.history.pop()
        current_df = self.original_df.copy()

        # Replay remaining steps
        for step in self.history:
            current_df, _ = apply_single_transformation(current_df, step.operation_type, step.parameters)

        self.working_df = current_df
        return True

    def reset(self) -> None:
        """Reset working dataset back to pristine original state."""
        self.working_df = self.original_df.copy()
        self.history.clear()

    @property
    def step_count(self) -> int:
        """Total number of applied transformation steps."""
        return len(self.history)

    def apply_transformation(self, operation_type: str, params: dict[str, Any]) -> TransformationStep:
        """Convenience alias for apply()."""
        return self.apply(operation_type, params)

    def undo_last_step(self) -> bool:
        """Convenience alias for undo()."""
        return self.undo()

    def reset_to_original(self) -> pd.DataFrame:
        """Convenience alias for reset() returning the pristine working DataFrame."""
        self.reset()
        return self.working_df
