"""Sidebar navigation and active dataset state manager.

Provides seamless section switching, dataset status monitoring,
and quick access to bundled sample datasets and session clearing.
"""

from __future__ import annotations

import streamlit as st

from core.ingestion import DatasetMetadata
from sample_data.loader import SAMPLE_DATASETS

SECTIONS = [
    "Overview & Getting Started",
    "Upload Dataset",
    "Data Preview",
    "Dataset Profile",
    "Data Quality Audit",
    "Automated Insights",
    "Visualization Studio",
    "Statistical Analysis",
    "Correlation Explorer",
    "Missingness Analysis",
    "Outlier Analysis",
    "Transformation Workbench",
    "Automatic First-Look",
    "Analysis Configuration",
    "Export & Download",
]


def render_sidebar(
    current_metadata: DatasetMetadata | None = None,
) -> str:
    """Render the sidebar controls and return the selected section name."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.25rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">⚡</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">
                        EDA Workbench
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                    v1.0.0 &bull; Local &bull; Air-gapped
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active Dataset Status Card
        if current_metadata:
            sampled_badge = ""
            if current_metadata.is_sampled:
                sampled_badge = '<div style="margin-top:4px;"><span class="badge badge-warning">Sampled Dataset</span></div>'

            status_card = f"""
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 0.85rem; margin-bottom: 1.25rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #3b82f6; letter-spacing: 0.05em; margin-bottom: 0.25rem;">
                    Active Dataset
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #f1f5f9; word-break: break-all;">
                    {current_metadata.filename}
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.25rem;">
                    {current_metadata.loaded_rows:,} rows &times; {current_metadata.loaded_cols} cols
                </div>
                <div style="font-size: 0.72rem; color: #64748b;">
                    Format: {current_metadata.file_format}
                </div>
                {sampled_badge}
            </div>
            """
            st.markdown(status_card, unsafe_allow_html=True)

            if st.button("✕ Clear Dataset", use_container_width=True, help="Unload active dataset and reset workspace"):
                st.session_state["dataset_df"] = None
                st.session_state["original_df"] = None
                st.session_state["working_df"] = None
                st.session_state["transformation_manager"] = None
                st.session_state["dataset_metadata"] = None
                st.session_state["first_look_report"] = None
                st.session_state["selected_section"] = "Overview & Getting Started"
                st.rerun()
        else:
            empty_status = """
            <div style="background: #0f172a; border: 1px dashed #334155; border-radius: 8px; padding: 0.75rem; margin-bottom: 1.25rem; text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8;">No Dataset Loaded</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Upload a file or choose a sample</div>
            </div>
            """
            st.markdown(empty_status, unsafe_allow_html=True)

        st.markdown('<div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Navigation</div>', unsafe_allow_html=True)

        # Track default selection
        default_index = 0
        if "selected_section" in st.session_state:
            try:
                default_index = SECTIONS.index(st.session_state["selected_section"])
            except ValueError:
                default_index = 0

        selected = st.radio(
            label="Workspace Sections",
            options=SECTIONS,
            index=default_index,
            label_visibility="collapsed",
        )
        st.session_state["selected_section"] = selected

        st.markdown("---")

        # Quick Sample Loader in Sidebar
        st.markdown('<div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Load Sample Dataset</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Churn (502)", use_container_width=True, help="Load Telecom Customer Churn dataset"):
                sample_info = SAMPLE_DATASETS["customer_churn"]
                sample_df = sample_info["loader"]()
                from core.ingestion import DatasetMetadata
                from core.transformations import TransformationManager

                meta = DatasetMetadata(
                    filename=sample_info["filename"],
                    file_format="CSV (Bundled Sample)",
                    file_size_bytes=int(sample_df.memory_usage(deep=True).sum()),
                    original_rows=len(sample_df),
                    original_cols=len(sample_df.columns),
                    loaded_rows=len(sample_df),
                    loaded_cols=len(sample_df.columns),
                    is_sampled=False,
                )
                st.session_state["original_df"] = sample_df.copy()
                st.session_state["working_df"] = sample_df.copy()
                st.session_state["dataset_df"] = sample_df.copy()
                st.session_state["transformation_manager"] = TransformationManager(sample_df)
                st.session_state["dataset_metadata"] = meta
                st.session_state["first_look_report"] = None
                st.session_state["selected_section"] = "Data Preview"
                st.rerun()

        with col_s2:
            if st.button("HR Data (400)", use_container_width=True, help="Load Employee HR & Compensation dataset"):
                sample_info = SAMPLE_DATASETS["employee_analytics"]
                sample_df = sample_info["loader"]()
                from core.ingestion import DatasetMetadata
                from core.transformations import TransformationManager

                meta = DatasetMetadata(
                    filename=sample_info["filename"],
                    file_format="CSV (Bundled Sample)",
                    file_size_bytes=int(sample_df.memory_usage(deep=True).sum()),
                    original_rows=len(sample_df),
                    original_cols=len(sample_df.columns),
                    loaded_rows=len(sample_df),
                    loaded_cols=len(sample_df.columns),
                    is_sampled=False,
                )
                st.session_state["original_df"] = sample_df.copy()
                st.session_state["working_df"] = sample_df.copy()
                st.session_state["dataset_df"] = sample_df.copy()
                st.session_state["transformation_manager"] = TransformationManager(sample_df)
                st.session_state["dataset_metadata"] = meta
                st.session_state["first_look_report"] = None
                st.session_state["selected_section"] = "Data Preview"
                st.rerun()

        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #475569; margin-top: 2rem; line-height: 1.4;">
                <strong>Privacy & Architecture:</strong><br>
                Runs entirely in your local Python environment. Zero external API calls, tracking, or cloud storage.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected
