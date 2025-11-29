"""
FastAPI web API for PTPD Calibration System.
"""

from ptpd_calibration.api.server import create_app, main

__all__ = ["create_app", "main"]
