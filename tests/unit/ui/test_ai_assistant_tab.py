from unittest.mock import patch

from ptpd_calibration.ui.tabs.ai_assistant import build_ai_assistant_tab


def test_ai_assistant_structure():
    with patch('gradio.TabItem'), \
         patch('gradio.Markdown'), \
         patch('gradio.Row'), \
         patch('gradio.Column'), \
         patch('gradio.Chatbot'), \
         patch('gradio.Button'), \
         patch('gradio.Textbox'), \
         patch('gradio.Dropdown'), \
         patch('gradio.Image'):

         build_ai_assistant_tab()
         # Pass if no error

