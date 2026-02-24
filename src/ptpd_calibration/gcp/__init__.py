"""
Google Cloud Platform Integration Module.

This package provides interfaces for interacting with Google Cloud services
including Cloud Storage (GCS) and Vertex AI. It uses a protocol-based
architecture to allow for easy mocking and local testing.
"""

from ptpd_calibration.gcp.config import GCPConfig, get_gcp_config

__all__ = ["GCPConfig", "get_gcp_config"]
