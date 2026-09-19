"""Hypothesis configuration for the property-based suites.

Two settings profiles are registered:

* ``ci`` (default): deterministic and database-free so CI runs are
  reproducible; 200 examples per property, no deadline.
* ``nightly``: 2000 randomised examples per property for deeper exploration.

The active profile is selected with the ``PTPD_HYPOTHESIS_PROFILE``
environment variable (default ``ci``).
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

PROFILE_ENV_VAR = "PTPD_HYPOTHESIS_PROFILE"
CI_PROFILE = "ci"
NIGHTLY_PROFILE = "nightly"
CI_MAX_EXAMPLES = 200
NIGHTLY_MAX_EXAMPLES = 2000

settings.register_profile(
    CI_PROFILE,
    max_examples=CI_MAX_EXAMPLES,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    NIGHTLY_PROFILE,
    max_examples=NIGHTLY_MAX_EXAMPLES,
    deadline=None,
    derandomize=False,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get(PROFILE_ENV_VAR, CI_PROFILE))


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``property`` marker so the suite runs under ``--strict-markers``
    even when the marker is not declared in ``pyproject.toml``."""
    config.addinivalue_line("markers", "property: property-based (Hypothesis) test")
