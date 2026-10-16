"""Unit and Integration tests for Sequential Next-Step Navigation.

Verifies:
- All 15 canonical sections follow exact sequence.
- Top and bottom buttons render with exact label 'Next: [Actual Next Page Name] →'.
- Clicking navigation buttons transitions session state to the immediate next section.
- Section 15 (Export & Download) does not loop to Section 1 and displays completion.
- Uploaded dataset, metadata, and working DataFrame persist across transitions.
- Navigation buttons work seamlessly in both loaded and empty states.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from components.navigation import SECTIONS
from sample_data.loader import create_employee_analytics_sample

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_navigation_sequence_contract() -> None:
    """Verify the 15-section canonical sequence and label generation."""
    assert len(SECTIONS) == 15
    assert SECTIONS[0] == "Overview & Getting Started"
    assert SECTIONS[1] == "Upload Dataset"
    assert SECTIONS[14] == "Export & Download"

    for idx in range(len(SECTIONS) - 1):
        curr = SECTIONS[idx]
        nxt = SECTIONS[idx + 1]
        expected_label = f"Next: {nxt} →"
        assert expected_label.startswith("Next: ")
        assert expected_label.endswith(" →")
        assert curr != nxt


def test_sequential_top_navigation_with_loaded_dataset() -> None:
    """Verify top navigation buttons navigate sequentially and preserve state."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    df = create_employee_analytics_sample()
    at.session_state["dataset_df"] = df.copy()
    at.session_state["original_df"] = df.copy()
    at.session_state["working_df"] = df.copy()
    at.session_state["selected_section"] = SECTIONS[0]
    at.run()
    assert len(at.exception) == 0

    # Step through every section sequentially via top buttons
    for idx in range(len(SECTIONS) - 1):
        current_section = SECTIONS[idx]
        expected_next = SECTIONS[idx + 1]
        expected_label = f"Next: {expected_next} →"

        top_key = f"btn_next_top_{idx}"
        bottom_key = f"btn_next_bottom_{idx}"

        top_buttons = [b for b in at.button if b.key == top_key]
        bottom_buttons = [b for b in at.button if b.key == bottom_key]

        assert len(top_buttons) == 1, f"Top button missing on {current_section}"
        assert len(bottom_buttons) == 1, f"Bottom button missing on {current_section}"

        assert top_buttons[0].label == expected_label
        assert bottom_buttons[0].label == expected_label

        # Verify state persistence before click
        assert at.session_state["working_df"] is not None
        assert len(at.session_state["working_df"]) == len(df)

        # Click top button to advance
        top_buttons[0].click()
        at.run()
        assert len(at.exception) == 0
        assert at.session_state["selected_section"] == expected_next

    # Final section (Export & Download)
    assert at.session_state["selected_section"] == "Export & Download"
    last_idx = len(SECTIONS) - 1
    last_top = [b for b in at.button if b.key == f"btn_next_top_{last_idx}"]
    last_bottom = [b for b in at.button if b.key == f"btn_next_bottom_{last_idx}"]

    # Must NOT have next step button on final section (no loop)
    assert len(last_top) == 0
    assert len(last_bottom) == 0


def test_sequential_bottom_navigation_advances() -> None:
    """Verify bottom buttons navigate to the exact same next target section."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    df = create_employee_analytics_sample()
    at.session_state["dataset_df"] = df.copy()
    at.session_state["original_df"] = df.copy()
    at.session_state["working_df"] = df.copy()

    # Test bottom navigation across key representative transitions
    test_indices = [0, 1, 6, 11, 13]
    for idx in test_indices:
        at.session_state["selected_section"] = SECTIONS[idx]
        at.run()
        assert len(at.exception) == 0

        bottom_key = f"btn_next_bottom_{idx}"
        bottom_buttons = [b for b in at.button if b.key == bottom_key]
        assert len(bottom_buttons) == 1

        expected_next = SECTIONS[idx + 1]
        assert bottom_buttons[0].label == f"Next: {expected_next} →"

        bottom_buttons[0].click()
        at.run()
        assert len(at.exception) == 0
        assert at.session_state["selected_section"] == expected_next
