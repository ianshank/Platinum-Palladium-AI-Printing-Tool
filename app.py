"""
Huggingface Spaces entry point for Pt/Pd Calibration Studio.

This file launches the Gradio web UI for platinum/palladium print calibration.
"""

import os

# Set environment variables for Huggingface Spaces
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")

# Import and launch the Gradio app
from ptpd_calibration.ui.gradio_app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
