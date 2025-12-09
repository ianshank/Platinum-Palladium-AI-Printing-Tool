"""
Unit tests for dashboard tab functionality.

Tests dashboard metrics calculation, stat card generation, and UI interactions.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from ptpd_calibration.session import PrintRecord, PrintResult, PrintSession
from ptpd_calibration.ui.tabs.dashboard import build_dashboard_tab


@pytest.fixture
def mock_session_logger():
    """Create a mock session logger with test data."""
    logger = MagicMock()
    logger.list_sessions.return_value = [
        {"filepath": "session1.json", "started_at": "2023-01-01"},
        {"filepath": "session2.json", "started_at": "2023-01-02"},
    ]

    session1 = MagicMock(spec=PrintSession)
    session1.duration_hours = 2.0
    session1.records = [
        MagicMock(
            spec=PrintRecord,
            timestamp=datetime.now(),
            paper_type="Arches",
            result=PrintResult.EXCELLENT,
            curve_name="Curve A",
        )
    ]

    session2 = MagicMock(spec=PrintSession)
    session2.duration_hours = 1.5
    session2.records = []  # Empty session

    def load_side_effect(path):
        if "session1" in str(path):
            return session1
        if "session2" in str(path):
            return session2
        raise FileNotFoundError

    logger.load_session.side_effect = load_side_effect
    return logger


def test_dashboard_metrics_calculation(mock_session_logger):
    """Test dashboard metrics calculation and display."""
    with (
        patch("gradio.Markdown"),
        patch("gradio.Row"),
        patch("gradio.Column"),
        patch("gradio.Button"),
        patch("gradio.Dataframe"),
        patch("gradio.Number"),
        patch("gradio.Textbox"),
        patch("gradio.HTML"),
        patch("gradio.Group"),
        patch("gradio.TabItem"),
        patch("gradio.Timer") as MockTimer,
    ):
        state = MagicMock()
        mock_timer_instance = MockTimer.return_value

        build_dashboard_tab(state, mock_session_logger)

        # Extract the callback function passed to timer.tick
        callback = mock_timer_instance.tick.call_args[0][0]

        # Execute the callback to test logic
        results = callback()

        # Results are now HTML stat cards, not raw values
        # results[0] = Print Volume stat card HTML
        # results[1] = Success Rate stat card HTML
        # results[2] = Active Curve stat card HTML
        # results[3] = Total Lab Time stat card HTML
        # results[4] = summary_rows list

        # Verify stat cards contain expected values
        assert "Print Volume (7d)" in results[0]
        assert "1" in results[0]  # 1 recent record

        assert "Success Rate" in results[1]
        assert "100.0%" in results[1]  # 1 success / 1 total

        assert "Active Curve" in results[2]
        assert "Curve A" in results[2]

        assert "Total Lab Time" in results[3]
        assert "3.5h" in results[3]  # 2.0 + 1.5

        # Verify summary rows
        assert isinstance(results[4], list)
        assert len(results[4]) == 2  # 2 sessions processed


def test_dashboard_jump_js():
    """Test that dashboard buttons have JavaScript for tab navigation."""
    with (
        patch("gradio.Markdown"),
        patch("gradio.Row"),
        patch("gradio.Column"),
        patch("gradio.Dataframe"),
        patch("gradio.Number"),
        patch("gradio.Textbox"),
        patch("gradio.HTML"),
        patch("gradio.Group"),
        patch("gradio.TabItem"),
        patch("gradio.Timer"),
        patch("gradio.Button") as MockButton,
    ):
        state = MagicMock()
        build_dashboard_tab(state, MagicMock())

        # Verify button clicks are registered with JS
        # We expect 5 quick action buttons
        assert MockButton.call_count >= 5

        # Check that click was called with js argument for some buttons
        js_calls = 0
        for instance in MockButton.return_value.click.call_args_list:
            if instance.kwargs and "js" in instance.kwargs:
                if "document.querySelectorAll" in instance.kwargs["js"]:
                    js_calls += 1

        # At least some buttons should have JS navigation
        assert js_calls >= 4


def test_stat_card_html_format():
    """Test that stat cards generate valid HTML."""
    with (
        patch("gradio.Markdown"),
        patch("gradio.Row"),
        patch("gradio.Column"),
        patch("gradio.Button"),
        patch("gradio.Dataframe"),
        patch("gradio.Number"),
        patch("gradio.Textbox"),
        patch("gradio.HTML"),
        patch("gradio.Group"),
        patch("gradio.TabItem"),
        patch("gradio.Timer") as MockTimer,
    ):
        state = MagicMock()
        logger = MagicMock()
        logger.list_sessions.return_value = []

        mock_timer_instance = MockTimer.return_value
        build_dashboard_tab(state, logger)

        callback = mock_timer_instance.tick.call_args[0][0]
        results = callback()

        # Verify HTML structure
        for html_card in results[:4]:
            assert '<div class="stat-card">' in html_card
            assert '<div class="stat-value">' in html_card
            assert '<div class="stat-label">' in html_card
