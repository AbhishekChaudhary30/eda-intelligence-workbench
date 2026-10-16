"""Interactive data preview component with bounded rendering and type inspection.

Allows users to inspect head, tail, or seeded random samples, select specific columns,
and review inferred data types without modifying the underlying analytical dataset.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.navigation import render_section_navigation
from core.ingestion import DatasetMetadata
from core.validation import ValidationReport, validate_dataset
from utils.formatting import format_bytes


def render_data_preview(df: pd.DataFrame, metadata: DatasetMetadata) -> None:
    """Render interactive tabular preview with safe row bounding and column filters."""
    st.markdown("### 🔍 Dataset Preview & Inspection")
    render_section_navigation("Data Preview", position="top")

    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0 or total_cols == 0:
        st.warning("The dataset is empty (0 rows or 0 columns).")
        return

    # Ingest warnings callout if any
    if metadata.warnings:
        with st.expander(f"⚠️ Ingestion & Normalization Notices ({len(metadata.warnings)})", expanded=False):
            for w in metadata.warnings:
                st.markdown(f"- {w}")

    # Top summary metrics
    val_report: ValidationReport = validate_dataset(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows", f"{metadata.original_rows:,}")
    with c2:
        st.metric("Active Loaded Rows", f"{metadata.loaded_rows:,}")
    with c3:
        st.metric("Total Columns", f"{total_cols:,}")
    with c4:
        st.metric("File Size", format_bytes(metadata.file_size_bytes))

    st.markdown("---")

    # Preview Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 2.0])

    with ctrl_col1:
        view_mode = st.radio(
            "Preview Mode",
            options=["First Rows (Head)", "Last Rows (Tail)", "Random Sample"],
            horizontal=True,
        )

    with ctrl_col2:
        row_limit = st.select_slider(
            "Rows to Display",
            options=[5, 10, 25, 50, 100],
            value=10,
            help="Limits rendered rows in the browser to ensure fast page response.",
        )

    with ctrl_col3:
        selected_cols = st.multiselect(
            "Filter Columns to Display",
            options=list(df.columns),
            default=list(df.columns)[:15] if len(df.columns) > 15 else list(df.columns),
            help="Select specific columns to view in the table below.",
        )

    # Sample slice calculation
    seed_used: int = 42
    if view_mode == "Random Sample":
        seed_col, _ = st.columns([1, 3])
        with seed_col:
            seed_used = st.number_input("Random Seed", value=42, step=1)
        preview_slice = df.sample(n=min(row_limit, total_rows), random_state=int(seed_used))
        st.caption(f"Showing {len(preview_slice)} randomly sampled rows (seed={seed_used}).")
    elif view_mode == "Last Rows (Tail)":
        preview_slice = df.tail(row_limit)
        st.caption(f"Showing last {len(preview_slice)} rows of dataset.")
    else:
        preview_slice = df.head(row_limit)
        st.caption(f"Showing first {len(preview_slice)} rows of dataset.")

    if not selected_cols:
        st.info("Select at least one column to display the table preview.")
        return

    # Render interactive DataFrame
    st.dataframe(
        preview_slice[selected_cols],
        use_container_width=True,
        hide_index=False,
    )

    # Inferred Schema Inspection Table
    with st.expander("📋 Inspect Column Names & Inferred Types", expanded=False):
        schema_records = []
        for col_name in selected_cols:
            t_info = val_report.inferred_types.get(col_name)
            if t_info:
                schema_records.append({
                    "Column Name": col_name,
                    "Pandas Dtype": t_info.pandas_dtype,
                    "Inferred Semantic Type": t_info.semantic_type,
                    "Potential Identifier?": "Yes (Heuristic)" if t_info.is_potential_identifier else "No",
                    "Mixed Python Types?": f"Yes: {t_info.detected_python_types}" if t_info.is_mixed_type else "No",
                })
        schema_df = pd.DataFrame(schema_records)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    render_section_navigation("Data Preview", position="bottom")
