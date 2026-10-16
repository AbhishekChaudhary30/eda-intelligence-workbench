"""Sequential workflow navigation component for EDA Intelligence Workbench.

Provides synchronized Top and Bottom 'Next Step' navigation buttons across
all 15 registered workflow sections in strict sidebar sequence.
"""

from __future__ import annotations

import streamlit as st

from components.sidebar import SECTIONS


def render_section_navigation(
    current_section: str,
    position: str = "top",
) -> None:
    """Render sequential workflow navigation controls.

    Parameters
    ----------
    current_section : str
        The exact name of the current active workflow section (must match SECTIONS).
    position : str
        Position on the page: "top" or "bottom".
    """
    if current_section not in SECTIONS:
        return

    curr_idx = SECTIONS.index(current_section)
    is_last = curr_idx >= len(SECTIONS) - 1

    if is_last:
        if position == "bottom":
            st.markdown("---")
            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 1.25rem; margin-top: 1.5rem; text-align: center;">
                    <div style="font-size: 1.25rem; margin-bottom: 0.25rem;">🎉</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #10b981; letter-spacing: -0.01em;">
                        Workflow Complete!
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.35rem; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                        You have reached the final step of the EDA Intelligence Workbench. All executive reports, audit logs, and sanitized datasets can be downloaded above.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif position == "top":
            _col_space, col_badge = st.columns([2.5, 1.5])
            with col_badge:
                st.markdown(
                    """
                    <div style="text-align: right; padding-top: 0.25rem;">
                        <span class="badge badge-success">✓ Final Step: Export & Download</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    next_section = SECTIONS[curr_idx + 1]
    btn_label = f"Next: {next_section} →"
    btn_key = f"btn_next_{position}_{curr_idx}"

    if position == "top":
        _col_space, col_btn = st.columns([2.5, 1.5])
        with col_btn:
            if st.button(
                btn_label,
                key=btn_key,
                use_container_width=True,
                type="secondary",
                help=f"Proceed sequentially to {next_section}",
            ):
                st.session_state["selected_section"] = next_section
                st.rerun()
    else:  # bottom
        st.markdown("---")
        col_info, col_btn = st.columns([2.5, 1.5])
        with col_info:
            st.markdown(
                f"""
                <div style="padding-top: 0.5rem; color: #94a3b8; font-size: 0.85rem;">
                    Completed step <strong>{curr_idx + 1} of {len(SECTIONS)}</strong>: <em>{current_section}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(
                btn_label,
                key=btn_key,
                use_container_width=True,
                type="primary",
                help=f"Proceed sequentially to {next_section}",
            ):
                st.session_state["selected_section"] = next_section
                st.rerun()
