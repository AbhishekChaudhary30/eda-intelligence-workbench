"""Interactive Data Transformation Workbench UI component.

Provides non-destructive column/row operations, type casting, imputation,
numeric scaling, category grouping, and one-hot encoding with strict previewing,
undo history, and original dataset preservation. Zero eval or exec.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.navigation import render_section_navigation
from core.transformations import (
    TransformationError,
    TransformationManager,
    preview_transformation,
)


def render_transformation_workbench(
    manager: TransformationManager,
) -> None:
    """Render the controlled Data Transformation Workbench."""
    st.markdown("### 🛠️ Interactive Data Transformation Workbench")
    st.caption(
        "Apply validated, traceable transformations to prepare and clean the active working dataset. "
        "The original uploaded dataset is immutably preserved in memory. Every operation can be previewed, undone, or reset."
    )
    render_section_navigation("Transformation Workbench", position="top")

    working_df = manager.working_df
    orig_df = manager.original_df

    # 1. State and Lineage Header
    h1, h2, h3, h4 = st.columns([1.5, 1.5, 1.2, 1.2])
    h1.metric("Original Dataset", f"{len(orig_df):,} rows × {len(orig_df.columns)} cols")
    h2.metric(
        "Working Dataset",
        f"{len(working_df):,} rows × {len(working_df.columns)} cols",
        delta=f"{len(working_df) - len(orig_df):+d} rows",
    )
    h3.metric("Applied Transformations", f"{len(manager.history)} step(s)")

    with h4:
        st.write("")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("↩️ Undo", help="Revert the most recent transformation step", disabled=len(manager.history) == 0) and manager.undo():
                st.success("Reverted last transformation step.")
                st.rerun()
        with col_btn2:
            if st.button("🔄 Reset", help="Restore pristine original dataset", disabled=len(manager.history) == 0):
                manager.reset()
                st.warning("Reset working dataset back to original uploaded state.")
                st.rerun()

    st.markdown("---")

    tab_cols, tab_rows, tab_missing, tab_numeric, tab_cat, tab_history = st.tabs([
        "📑 Column Operations",
        "🔍 Row Filtering & Duplicates",
        "🩹 Missing Value Imputation",
        "📐 Scaling & Math Transforms",
        "🏷️ Categorical Encodings",
        "📜 Audit History",
    ])

    current_cols = list(working_df.columns)
    numeric_cols = [c for c in current_cols if pd.api.types.is_numeric_dtype(working_df[c])]
    cat_cols = [c for c in current_cols if not pd.api.types.is_numeric_dtype(working_df[c])]

    # -------------------------------------------------------------------------
    # TAB 1: COLUMN OPERATIONS
    # -------------------------------------------------------------------------
    with tab_cols:
        st.markdown("#### Column Modifications")
        col_action = st.radio(
            "Select Column Operation",
            options=["Drop Columns", "Rename Column", "Cast Data Type"],
            horizontal=True,
        )

        if col_action == "Drop Columns":
            cols_to_drop = st.multiselect("Select Columns to Remove", options=current_cols)
            if cols_to_drop:
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    if st.button("👁️ Preview Drop"):
                        try:
                            prev = preview_transformation(working_df, "drop_columns", {"columns": cols_to_drop})
                            st.info(f"Will remove **{len(cols_to_drop)}** column(s). New shape: {prev.expected_rows_after} rows × {prev.expected_cols_after} cols.")
                            st.dataframe(prev.preview_head, use_container_width=True)
                        except TransformationError as e:
                            st.error(str(e))
                with p_col2:
                    if st.button("⚡ Apply Column Drop", type="primary"):
                        try:
                            step = manager.apply("drop_columns", {"columns": cols_to_drop})
                            st.success(f"Success: {step.description}")
                            st.rerun()
                        except TransformationError as e:
                            st.error(str(e))

        elif col_action == "Rename Column":
            r1, r2 = st.columns(2)
            with r1:
                target_col = st.selectbox("Column to Rename", options=current_cols)
            with r2:
                new_name = st.text_input("New Column Name", value=target_col if target_col else "")

            if target_col and new_name and new_name != target_col and st.button("⚡ Apply Rename", type="primary"):
                try:
                    step = manager.apply("rename_column", {"old_name": target_col, "new_name": new_name})
                    st.success(f"Success: {step.description}")
                    st.rerun()
                except TransformationError as e:
                    st.error(str(e))

        else:
            # Cast Data Type
            c1, c2 = st.columns(2)
            with c1:
                target_col = st.selectbox("Column to Cast", options=current_cols)
            with c2:
                target_type = st.selectbox("Target Data Type", options=["float", "int", "string", "category", "datetime", "boolean"])

            if target_col and target_type and st.button("⚡ Apply Type Cast", type="primary"):
                try:
                    step = manager.apply("cast_type", {"column": target_col, "target_type": target_type})
                    st.success(f"Success: {step.description}")
                    st.rerun()
                except TransformationError as e:
                    st.error(str(e))

    # -------------------------------------------------------------------------
    # TAB 2: ROW FILTERING & DUPLICATES
    # -------------------------------------------------------------------------
    with tab_rows:
        st.markdown("#### Row Filtering & Deduplication")
        row_action = st.radio("Select Row Action", options=["Filter Rows by Safe Condition", "Remove Exact Duplicate Rows"], horizontal=True)

        if row_action == "Filter Rows by Safe Condition":
            st.caption("Apply deterministic row filtering without eval() or exec().")
            f1, f2, f3 = st.columns(3)
            with f1:
                f_col = st.selectbox("Filter Target Feature", options=current_cols)
            with f2:
                is_num = pd.api.types.is_numeric_dtype(working_df[f_col])
                op_options = ["==", "!=", ">", ">=", "<", "<=", "is_null", "not_null"] if is_num else ["==", "!=", "contains", "is_null", "not_null"]
                f_op = st.selectbox("Comparison Operator", options=op_options)
            with f3:
                f_val = None
                if f_op not in ("is_null", "not_null"):
                    f_val = st.text_input("Comparison Value", placeholder="e.g. 100 or 'Pending'")

            p1, p2 = st.columns(2)
            with p1:
                if st.button("👁️ Preview Row Filter"):
                    try:
                        prev = preview_transformation(working_df, "filter_rows", {"column": f_col, "operator": f_op, "value": f_val})
                        st.info(f"Filtered Rows Remaining: **{prev.expected_rows_after:,}** (Rows removed: {abs(prev.row_delta):,}).")
                        if prev.warnings:
                            st.warning("; ".join(prev.warnings))
                        st.dataframe(prev.preview_head, use_container_width=True)
                    except TransformationError as e:
                        st.error(str(e))
            with p2:
                if st.button("⚡ Apply Filter", type="primary"):
                    try:
                        step = manager.apply("filter_rows", {"column": f_col, "operator": f_op, "value": f_val})
                        st.success(f"Success: {step.description}")
                        st.rerun()
                    except TransformationError as e:
                        st.error(str(e))

        else:
            # Remove exact duplicate rows
            dup_count = working_df.duplicated().sum()
            st.write(f"Detected **{dup_count:,}** exact duplicate rows in active working dataset.")
            if dup_count > 0:
                if st.button("⚡ Remove All Exact Duplicates", type="primary"):
                    try:
                        step = manager.apply("drop_duplicates", {})
                        st.success(f"Success: {step.description}")
                        st.rerun()
                    except TransformationError as e:
                        st.error(str(e))
            else:
                st.info("No duplicate rows present.")

    # -------------------------------------------------------------------------
    # TAB 3: MISSING VALUE IMPUTATION
    # -------------------------------------------------------------------------
    with tab_missing:
        st.markdown("#### Imputation of Missing Values")
        st.caption("Fill null values using robust statistical aggregates or custom constants.")

        cols_with_nulls = [c for c in current_cols if working_df[c].isnull().sum() > 0]
        if not cols_with_nulls:
            st.success("🎉 Working dataset contains zero missing values in any column.")
        else:
            i1, i2, i3 = st.columns(3)
            with i1:
                imp_col = st.selectbox("Feature with Missing Data", options=cols_with_nulls)
            with i2:
                is_num = pd.api.types.is_numeric_dtype(working_df[imp_col])
                strat_opts = ["mean", "median", "mode", "constant"] if is_num else ["mode", "constant"]
                strategy = st.selectbox("Imputation Strategy", options=strat_opts)
            with i3:
                const_val = ""
                if strategy == "constant":
                    const_val = st.text_input("Constant Value to Fill", value="Missing")

            missing_cnt = working_df[imp_col].isnull().sum()
            st.write(f"Target column `{imp_col}` currently contains **{missing_cnt:,}** null values.")

            if st.button("⚡ Apply Imputation", type="primary"):
                try:
                    step = manager.apply("impute_missing", {
                        "column": imp_col,
                        "strategy": strategy,
                        "constant_value": const_val,
                    })
                    st.success(f"Success: {step.description}")
                    st.rerun()
                except TransformationError as e:
                    st.error(str(e))

    # -------------------------------------------------------------------------
    # TAB 4: SCALING & NUMERIC TRANSFORMS
    # -------------------------------------------------------------------------
    with tab_numeric:
        st.markdown("#### Mathematical Scaling & Transformations")
        if not numeric_cols:
            st.info("No numeric columns available for scaling.")
        else:
            scale_action = st.radio("Transformation Type", options=["Numeric Feature Scaling", "Logarithmic Transformation (log1p)"], horizontal=True)

            if scale_action == "Numeric Feature Scaling":
                s1, s2 = st.columns(2)
                with s1:
                    num_col = st.selectbox("Numeric Column to Scale", options=numeric_cols, key="scale_col")
                with s2:
                    scale_meth = st.selectbox("Scaling Method", options=["standard", "minmax", "robust"], format_func=lambda m: {
                        "standard": "Standardization (Z-Score: mean=0, std=1)",
                        "minmax": "Min-Max Scaling (Bounded [0, 1])",
                        "robust": "Robust Scaling (Median & IQR)",
                    }[m])

                if st.button("⚡ Apply Scaling", type="primary"):
                    try:
                        step = manager.apply("scale_numeric", {"column": num_col, "method": scale_meth})
                        st.success(f"Success: {step.description}")
                        st.rerun()
                    except TransformationError as e:
                        st.error(str(e))

            else:
                l1, _ = st.columns(2)
                with l1:
                    log_col = st.selectbox("Numeric Column for Log(1 + x)", options=numeric_cols, key="log_col")

                clean_s = pd.to_numeric(working_df[log_col], errors="coerce").dropna()
                if (clean_s < 0).any():
                    st.warning("⚠️ Column contains negative values. Logarithm is mathematically undefined for negatives.")
                else:
                    if st.button("⚡ Apply Log(1 + x)", type="primary"):
                        try:
                            step = manager.apply("log_transform", {"column": log_col})
                            st.success(f"Success: {step.description}")
                            st.rerun()
                        except TransformationError as e:
                            st.error(str(e))

    # -------------------------------------------------------------------------
    # TAB 5: CATEGORICAL ENCODINGS
    # -------------------------------------------------------------------------
    with tab_cat:
        st.markdown("#### Categorical Transformations")
        cat_action = st.radio("Encoding Type", options=["Group Infrequent Categories ('Other')", "One-Hot Encode (Dummies)"], horizontal=True)

        if cat_action == "Group Infrequent Categories ('Other')":
            g1, g2, g3 = st.columns(3)
            with g1:
                grp_col = st.selectbox("Categorical Column", options=cat_cols if cat_cols else current_cols, key="grp_cat_col")
            with g2:
                top_n = st.slider("Keep Top N Levels", min_value=3, max_value=25, value=10)
            with g3:
                other_lbl = st.text_input("Label for Remaining Levels", value="Other")

            if st.button("⚡ Apply Category Grouping", type="primary"):
                try:
                    step = manager.apply("group_categories", {"column": grp_col, "top_n": top_n, "other_label": other_lbl})
                    st.success(f"Success: {step.description}")
                    st.rerun()
                except TransformationError as e:
                    st.error(str(e))

        else:
            # One-Hot Encode
            o1, o2 = st.columns(2)
            with o1:
                ohe_col = st.selectbox("Categorical Column (<= 12 levels)", options=cat_cols if cat_cols else current_cols, key="ohe_col")
            with o2:
                drop_first = st.checkbox("Drop First Level (Avoid multicollinearity)", value=False)

            distinct_k = working_df[ohe_col].nunique()
            st.caption(f"Column `{ohe_col}` currently contains **{distinct_k}** distinct levels.")
            if distinct_k > 12:
                st.warning("One-hot encoding is guarded to max 12 levels to avoid dimensionality explosion. Please group infrequent levels first.")
            else:
                if st.button("⚡ Apply One-Hot Encoding", type="primary"):
                    try:
                        step = manager.apply("one_hot_encode", {"column": ohe_col, "drop_first": drop_first})
                        st.success(f"Success: {step.description}")
                        st.rerun()
                    except TransformationError as e:
                        st.error(str(e))

    # -------------------------------------------------------------------------
    # TAB 6: AUDIT HISTORY
    # -------------------------------------------------------------------------
    with tab_history:
        st.markdown("#### Transformation Audit Trail & Lineage")
        if not manager.history:
            st.info("No transformations applied yet. Working dataset is identical to original uploaded data.")
        else:
            records = []
            for s in manager.history:
                records.append({
                    "Step": f"#{s.step_id}",
                    "Operation": s.operation_type,
                    "Description": s.description,
                    "Rows Before": s.rows_before,
                    "Rows After": s.rows_after,
                    "Cols Before": s.cols_before,
                    "Cols After": s.cols_after,
                    "Timestamp": s.timestamp,
                })
            hist_df = pd.DataFrame(records)
            st.dataframe(hist_df, use_container_width=True)

            csv_hist = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Transformation History (CSV)",
                data=csv_hist,
                file_name="transformation_audit_history.csv",
                mime="text/csv",
            )

    render_section_navigation("Transformation Workbench", position="bottom")
