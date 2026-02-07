from unittest.mock import MagicMock, patch

from ptpd_calibration.ui.tabs.session_log import build_session_log_tab


def test_session_log_structure():
    with patch('gradio.TabItem'), \
         patch('gradio.Row'), \
         patch('gradio.Column'), \
         patch('gradio.Markdown'), \
         patch('gradio.Button'), \
         patch('gradio.Dropdown'), \
         patch('gradio.HTML'), \
         patch('gradio.State'), \
         patch('gradio.Plot'):

         mock_logger = MagicMock()
         build_session_log_tab(mock_logger)
         # Pass if no error

