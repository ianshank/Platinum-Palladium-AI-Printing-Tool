"""
Tests for the API-facing settings added for SEC-03 / SEC-07 / SCI-06.

Every limit used by the API layer must be a settings field with an
environment override and a sensible default; these tests pin those contracts.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ptpd_calibration.config import APISettings, Settings
from ptpd_calibration.mcts.config import MCTSSettings

# =============================================================================
# APISettings
# =============================================================================


class TestAPISettingsDefaults:
    def test_cors_credentials_default_off(self) -> None:
        settings = APISettings()
        assert settings.cors_origins == ["*"]
        assert settings.cors_allow_credentials is False

    def test_upload_limits(self) -> None:
        settings = APISettings()
        assert settings.max_upload_size_mb == 50
        assert settings.upload_chunk_size_kb == 64
        assert ".png" in settings.allowed_scan_extensions
        assert ".tif" in settings.allowed_scan_extensions
        assert settings.allowed_quad_extensions == [".quad", ".txt"]
        assert settings.max_export_name_length == 64

    def test_request_bounds(self) -> None:
        settings = APISettings()
        assert settings.max_request_body_mb == 50
        assert settings.max_list_length == 4096
        assert settings.max_string_length == 4096
        assert settings.max_synthetic_samples == 5000
        assert settings.max_hidden_layers == 16
        assert settings.max_hidden_dim == 4096

    def test_aggregate_settings_carry_api_defaults(self) -> None:
        settings = Settings()
        assert settings.api.cors_allow_credentials is False
        assert settings.api.max_request_body_mb == 50


class TestAPISettingsEnvOverrides:
    @pytest.mark.parametrize(
        ("env_name", "env_value", "attribute", "expected"),
        [
            ("PTPD_API_MAX_REQUEST_BODY_MB", "7", "max_request_body_mb", 7),
            ("PTPD_API_MAX_LIST_LENGTH", "12", "max_list_length", 12),
            ("PTPD_API_MAX_STRING_LENGTH", "99", "max_string_length", 99),
            ("PTPD_API_MAX_UPLOAD_SIZE_MB", "3", "max_upload_size_mb", 3),
            ("PTPD_API_UPLOAD_CHUNK_SIZE_KB", "16", "upload_chunk_size_kb", 16),
            ("PTPD_API_MAX_EXPORT_NAME_LENGTH", "32", "max_export_name_length", 32),
            ("PTPD_API_MAX_SYNTHETIC_SAMPLES", "10", "max_synthetic_samples", 10),
            ("PTPD_API_MAX_HIDDEN_LAYERS", "2", "max_hidden_layers", 2),
            ("PTPD_API_MAX_HIDDEN_DIM", "8", "max_hidden_dim", 8),
            ("PTPD_API_ALLOWED_QUAD_EXTENSIONS", '[".quad"]', "allowed_quad_extensions", [".quad"]),
            ("PTPD_API_ALLOWED_SCAN_EXTENSIONS", '[".png"]', "allowed_scan_extensions", [".png"]),
        ],
    )
    def test_env_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        env_value: str,
        attribute: str,
        expected: object,
    ) -> None:
        monkeypatch.setenv(env_name, env_value)
        assert getattr(APISettings(), attribute) == expected

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_request_body_mb": 0},
            {"max_list_length": 0},
            {"max_string_length": 0},
            {"upload_chunk_size_kb": 1},
            {"max_export_name_length": 1},
        ],
    )
    def test_out_of_range_limits_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            APISettings(**kwargs)


class TestCORSValidator:
    def test_wildcard_with_credentials_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="wildcard"):
            APISettings(cors_origins=["*"], cors_allow_credentials=True)

    def test_wildcard_with_credentials_via_env_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PTPD_API_CORS_ALLOW_CREDENTIALS", "true")
        with pytest.raises(ValueError, match="PTPD_API_CORS_ORIGINS"):
            APISettings()

    def test_explicit_origins_with_credentials_is_allowed(self) -> None:
        settings = APISettings(
            cors_origins=["https://app.example"],
            cors_allow_credentials=True,
        )
        assert settings.cors_allow_credentials is True

    def test_wildcard_among_explicit_origins_with_credentials_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="wildcard"):
            APISettings(cors_origins=["https://app.example", "*"], cors_allow_credentials=True)

    def test_wildcard_without_credentials_is_allowed(self) -> None:
        settings = APISettings(cors_origins=["*"], cors_allow_credentials=False)
        assert settings.cors_allow_credentials is False

    def test_aggregate_settings_reject_unsafe_combination(self) -> None:
        with pytest.raises(ValueError, match="wildcard"):
            Settings(api=APISettings(cors_origins=["*"], cors_allow_credentials=True))


# =============================================================================
# MCTSSettings fields consumed by the API
# =============================================================================


class TestMCTSSettingsAPIFields:
    def test_defaults(self) -> None:
        settings = MCTSSettings()
        assert settings.feedback_path is None
        assert settings.max_simulations_per_request == 2000

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_MCTS_FEEDBACK_PATH", "/var/lib/ptpd/feedback.jsonl")
        monkeypatch.setenv("PTPD_MCTS_MAX_SIMULATIONS_PER_REQUEST", "100")
        settings = MCTSSettings()
        assert settings.feedback_path == "/var/lib/ptpd/feedback.jsonl"
        assert settings.max_simulations_per_request == 100

    @pytest.mark.parametrize("value", [10, 20000])
    def test_max_simulations_bounds(self, value: int) -> None:
        with pytest.raises(ValidationError):
            MCTSSettings(max_simulations_per_request=value)
