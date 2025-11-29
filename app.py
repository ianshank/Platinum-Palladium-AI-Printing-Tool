"""
Huggingface Spaces entry point for Pt/Pd Calibration Studio.

This file launches the Gradio web UI for platinum/palladium print calibration.
"""

import os
import sys

# Add src directory to Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set environment variables for Huggingface Spaces
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")

# Work around a Gradio 4.44 JSON schema regression where boolean
# additionalProperties values surface as bare booleans that crash the
# API metadata generation step. Treat bool schemas as permissive/empty.
try:
    from gradio_client import utils as gradio_client_utils

    _orig_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

    def _bool_safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_json_schema_to_python_type(schema, defs)

    gradio_client_utils._json_schema_to_python_type = _bool_safe_json_schema_to_python_type
except Exception:  # pragma: no cover - best effort patching
    pass

# Import and launch the Gradio app
from ptpd_calibration.ui.gradio_app import create_gradio_app

# Create the app
demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
