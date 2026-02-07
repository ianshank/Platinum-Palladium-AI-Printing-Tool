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

# --- PATCH START ---
# Work around a Gradio 4.44 JSON schema regression where boolean
# additionalProperties values surface as bare booleans that crash clients.
print("Applying Gradio schema patch...")
try:
    from gradio.blocks import Blocks

    _orig_get_config_file = Blocks.get_config_file

    def _fixed_get_config_file(self):
        config = _orig_get_config_file(self)

        def fix_schema(schema):
            if isinstance(schema, dict):
                if schema.get("additionalProperties") is True:
                    schema["additionalProperties"] = {}
                for v in schema.values():
                    fix_schema(v)
            elif isinstance(schema, list):
                for item in schema:
                    fix_schema(item)

        if "components" in config:
            for comp in config["components"]:
                if "api_info" in comp:
                    fix_schema(comp["api_info"])

        return config

    Blocks.get_config_file = _fixed_get_config_file
    print("Gradio schema patch applied successfully.")
except Exception as e:
    print(f"Failed to apply Gradio schema patch: {e}")
# --- PATCH END ---

# Import and launch the Gradio app
from ptpd_calibration.ui.gradio_app import create_gradio_app

# Create the app
demo = create_gradio_app()

if __name__ == "__main__":
    # Allow port override via environment variable
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    print(f"Launching on {server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
    )
