import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from ptpd_calibration.ui.tabs.dashboard import build_dashboard_tab
from ptpd_calibration.session import PrintRecord, PrintSession, PrintResult

@pytest.fixture
def mock_session_logger():
    logger = MagicMock()
    logger.list_sessions.return_value = [
        {"filepath": "session1.json", "started_at": "2023-01-01"},
        {"filepath": "session2.json", "started_at": "2023-01-02"}
    ]
    
    session1 = MagicMock(spec=PrintSession)
    session1.duration_hours = 2.0
    session1.records = [
        MagicMock(spec=PrintRecord, timestamp=datetime.now(), paper_type="Arches", result=PrintResult.EXCELLENT, curve_name="Curve A")
    ]
    
    session2 = MagicMock(spec=PrintSession)
    session2.duration_hours = 1.5
    session2.records = [] # Empty session

    def load_side_effect(path):
        if "session1" in str(path): return session1
        if "session2" in str(path): return session2
        raise FileNotFoundError

    logger.load_session.side_effect = load_side_effect
    return logger

def test_dashboard_metrics_calculation(mock_session_logger):
    # We need to extract the inner compute_dashboard_metrics function or test the side effects
    # Since build_dashboard_tab is a builder, we can test it by mocking gradio components and inspecting calls
    
    with patch('gradio.Markdown'), \
         patch('gradio.Row'), \
         patch('gradio.Column'), \
         patch('gradio.Button'), \
         patch('gradio.Dataframe'), \
         patch('gradio.Number'), \
         patch('gradio.Textbox'), \
         patch('gradio.Timer') as MockTimer:
        
        state = MagicMock()
        # Mocking the timer tick to capture the callback
        mock_timer_instance = MockTimer.return_value
        
        build_dashboard_tab(state, mock_session_logger)
        
        # Extract the callback function passed to timer.tick
        # args[0] is usually the function
        callback = mock_timer_instance.tick.call_args[0][0]
        
        # Execute the callback to test logic
        results = callback()
        
        # Unpack results: recent_records, success_rate, active_curve, total_hours, summary_rows
        assert results[0] == 1 # 1 recent record in session1 (mocked as now)
        assert results[1] == "100.0%" # 1 success / 1 total
        assert results[2] == "Curve A"
        assert results[3] == "3.5h" # 2.0 + 1.5
        assert len(results[4]) == 2 # 2 sessions processed

def test_dashboard_jump_js():
    # The file has a local helper jump_js, we can't import it directly easily unless we expose it
    # But we can verify the Button clicks have js arguments
    with patch('gradio.Markdown'), \
         patch('gradio.Row'), \
         patch('gradio.Column'), \
         patch('gradio.Dataframe'), \
         patch('gradio.Number'), \
         patch('gradio.Textbox'), \
         patch('gradio.Timer'):
             
        with patch('gradio.Button') as MockButton:
            state = MagicMock()
            build_dashboard_tab(state, MagicMock())
            
            # Verify button clicks are registered with JS
            # We expect 5 quick action buttons
            assert MockButton.call_count >= 5
            
            # Check that click was called with js argument for some buttons
            # We can iterate through all instances created
            js_calls = 0
            for instance in MockButton.return_value.click.call_args_list:
                if 'js' in instance.kwargs and "document.querySelectorAll" in instance.kwargs['js']:
                    js_calls += 1
            
            assert js_calls >= 4 # Scan, Chem, Expo, Neg, AI (some might share logic or be separate)

