"""Upload panel with multi-format parsing options and bundled sample datasets.

Handles file validation, encoding and delimiter overrides, multi-sheet selection,
and actionable error reporting.
"""

from __future__ import annotations

import streamlit as st

from core.ingestion import (
    IngestionError,
    IngestionResult,
    inspect_excel_sheets,
    load_dataset,
)
from sample_data.loader import SAMPLE_DATASETS


def render_upload_panel() -> IngestionResult | None:
    """Render file upload interface and return IngestionResult upon successful load."""
    st.markdown("### 📂 Ingest Tabular Dataset")
    st.markdown(
        "Upload a supported tabular dataset to begin automated exploratory data analysis. "
        "Files are processed locally and securely."
    )

    col_up, col_samples = st.columns([1.6, 1.0], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            label="Upload CSV, TSV, Excel, or Parquet file",
            type=["csv", "tsv", "xlsx", "parquet"],
            help="Maximum permitted upload size is 100 MB.",
        )

        # Advanced configuration options
        with st.expander("⚙️ Advanced Ingestion Settings", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                delim_choice = st.selectbox(
                    "Delimiter Override (CSV/TSV)",
                    options=["Auto-detect", "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Pipe (|)"],
                )
                delim_map = {
                    "Auto-detect": None,
                    "Comma (,)": ",",
                    "Tab (\\t)": "\t",
                    "Semicolon (;)": ";",
                    "Pipe (|)": "|",
                }
                user_delim = delim_map[delim_choice]

            with col_adv2:
                enc_choice = st.selectbox(
                    "Encoding Override",
                    options=["Auto-detect", "UTF-8", "UTF-8-SIG (BOM)", "Latin-1 / CP1252", "ISO-8859-1"],
                )
                enc_map = {
                    "Auto-detect": None,
                    "UTF-8": "utf-8",
                    "UTF-8-SIG (BOM)": "utf-8-sig",
                    "Latin-1 / CP1252": "cp1252",
                    "ISO-8859-1": "iso-8859-1",
                }
                user_enc = enc_map[enc_choice]

            allow_sampling = st.checkbox(
                "Enable automatic random sampling for massive datasets (>250,000 rows)",
                value=True,
                help="Samples 50,000 rows with seed=42 to prevent browser freezing.",
            )

        # Multi-sheet Excel detection
        selected_sheet: str | None = None
        if uploaded_file is not None and uploaded_file.name.lower().endswith(".xlsx"):
            try:
                raw_bytes = uploaded_file.getvalue()
                sheet_names = inspect_excel_sheets(raw_bytes)
                if len(sheet_names) > 1:
                    st.info(f"The Excel workbook contains {len(sheet_names)} worksheets.")
                    selected_sheet = st.selectbox("Select Worksheet to Analyze", options=sheet_names)
            except (ValueError, OSError, IngestionError) as e:
                st.warning(f"Note: Could not inspect worksheets: {e}")

        if uploaded_file is not None and st.button(
            "🚀 Ingest & Process Dataset", type="primary", use_container_width=True
        ):
            with st.spinner("Validating, parsing, and normalizing dataset..."):
                try:
                    raw_bytes = uploaded_file.getvalue()
                    result = load_dataset(
                        file_source=raw_bytes,
                        filename=uploaded_file.name,
                        user_delimiter=user_delim,
                        user_encoding=user_enc,
                        sheet_name=selected_sheet,
                        allow_sampling=allow_sampling,
                    )

                    from core.transformations import TransformationManager

                    st.session_state["original_df"] = result.df.copy()
                    st.session_state["working_df"] = result.df.copy()
                    st.session_state["dataset_df"] = result.df.copy()
                    st.session_state["transformation_manager"] = TransformationManager(result.df)
                    st.session_state["dataset_metadata"] = result.metadata
                    st.session_state["first_look_report"] = None
                    st.session_state["selected_section"] = "Data Preview"

                    st.success(
                        f"Successfully ingested **{result.metadata.filename}** "
                        f"({result.metadata.loaded_rows:,} rows, {result.metadata.loaded_cols} columns)!"
                    )
                    st.rerun()

                except IngestionError as ing_err:
                    st.error(f"⚠️ **Ingestion Failed:** {ing_err}")
                except (ValueError, OSError, TypeError) as exc:
                    st.error(
                        f"⚠️ **Unexpected Error:** The file could not be parsed. "
                        f"Please verify the format or delimiter. Details: {exc}"
                    )

    with col_samples:
        st.markdown("#### 🧪 Bundled Sample Datasets")
        st.markdown(
            "Test all platform capabilities instantly with pre-configured synthetic datasets "
            "containing realistic types, missing values, and anomalies."
        )

        for key, s_info in SAMPLE_DATASETS.items():
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.75rem;">
                    <div style="font-weight: 600; font-size: 0.9rem; color: #f8fafc;">{s_info['name']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin: 0.25rem 0 0.5rem 0;">{s_info['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Load {s_info['name'].split('(')[0].strip()}", key=f"btn_sample_{key}", use_container_width=True):
                sample_df = s_info["loader"]()
                from core.ingestion import DatasetMetadata

                meta = DatasetMetadata(
                    filename=s_info["filename"],
                    file_format="CSV (Bundled Sample)",
                    file_size_bytes=int(sample_df.memory_usage(deep=True).sum()),
                    original_rows=len(sample_df),
                    original_cols=len(sample_df.columns),
                    loaded_rows=len(sample_df),
                    loaded_cols=len(sample_df.columns),
                    is_sampled=False,
                )
                from core.transformations import TransformationManager

                st.session_state["original_df"] = sample_df.copy()
                st.session_state["working_df"] = sample_df.copy()
                st.session_state["dataset_df"] = sample_df.copy()
                st.session_state["transformation_manager"] = TransformationManager(sample_df)
                st.session_state["dataset_metadata"] = meta
                st.session_state["first_look_report"] = None
                st.session_state["selected_section"] = "Data Preview"
                st.rerun()

    return None
