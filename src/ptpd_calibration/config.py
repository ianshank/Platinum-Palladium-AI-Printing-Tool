"""
Configuration management for PTPD Calibration System.

Uses pydantic-settings for environment-based configuration with validation.
All settings can be overridden via environment variables with PTPD_ prefix.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ExportFormat(str, Enum):
    """Supported curve export formats."""

    QTR = "qtr"
    PIEZOGRAPHY = "piezography"
    CSV = "csv"
    JSON = "json"


class InterpolationMethod(str, Enum):
    """Curve interpolation methods."""

    LINEAR = "linear"
    CUBIC = "cubic"
    MONOTONIC = "monotonic"
    PCHIP = "pchip"


class TabletType(str, Enum):
    """Supported step tablet types."""

    STOUFFER_21 = "stouffer_21"
    STOUFFER_31 = "stouffer_31"
    STOUFFER_41 = "stouffer_41"
    CUSTOM = "custom"


class DetectionSettings(BaseSettings):
    """Settings for step tablet detection."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DETECTION_")

    # Edge detection parameters
    canny_low_threshold: int = Field(default=50, ge=0, le=255)
    canny_high_threshold: int = Field(default=150, ge=0, le=255)

    # Morphological operations
    morph_kernel_size: int = Field(default=5, ge=1, le=21)
    morph_iterations: int = Field(default=2, ge=1, le=10)

    # Contour detection
    min_contour_area_ratio: float = Field(default=0.01, ge=0.001, le=0.5)
    max_contour_area_ratio: float = Field(default=0.95, ge=0.5, le=1.0)

    # Rotation correction
    max_rotation_angle: float = Field(default=15.0, ge=0.0, le=45.0)
    rotation_threshold: float = Field(default=0.5, ge=0.1, le=5.0)

    # Patch segmentation
    gradient_threshold: float = Field(default=0.1, ge=0.01, le=0.5)
    min_patch_width_ratio: float = Field(default=0.02, ge=0.01, le=0.1)


class ExtractionSettings(BaseSettings):
    """Settings for density/color extraction."""

    model_config = SettingsConfigDict(env_prefix="PTPD_EXTRACTION_")

    # Sampling parameters
    sample_margin_ratio: float = Field(default=0.15, ge=0.05, le=0.4)
    min_sample_pixels: int = Field(default=100, ge=10, le=10000)

    # Outlier rejection
    outlier_rejection_method: str = Field(default="mad")
    mad_threshold: float = Field(default=3.0, ge=1.0, le=10.0)

    # Density calculation
    reference_white_reflectance: float = Field(default=0.9, ge=0.5, le=1.0)
    status_a_weights: tuple[float, float, float] = Field(default=(0.299, 0.587, 0.114))

    # Paper base detection
    paper_margin_ratio: float = Field(default=0.05, ge=0.01, le=0.2)
    paper_sample_size: int = Field(default=50, ge=10, le=500)


class CurveSettings(BaseSettings):
    """Settings for curve generation and export."""

    model_config = SettingsConfigDict(env_prefix="PTPD_CURVE_")

    # Interpolation
    default_interpolation: InterpolationMethod = Field(default=InterpolationMethod.MONOTONIC)
    num_output_points: int = Field(default=256, ge=16, le=4096)

    # Smoothing
    smoothing_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    monotonicity_enforcement: bool = Field(default=True)

    # Highlight preservation
    highlight_hold_point: float = Field(default=0.05, ge=0.0, le=0.2)
    shadow_hold_point: float = Field(default=0.95, ge=0.8, le=1.0)

    # Export defaults
    default_export_format: ExportFormat = Field(default=ExportFormat.QTR)
    qtr_ink_limit: float = Field(default=100.0, ge=0.0, le=100.0)
    qtr_resolution: int = Field(default=2880, ge=360, le=5760)


class MLSettings(BaseSettings):
    """Settings for ML prediction and refinement."""

    model_config = SettingsConfigDict(env_prefix="PTPD_ML_")

    # Model selection
    default_model_type: str = Field(default="gradient_boosting")
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int = Field(default=6, ge=2, le=20)

    # Training
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)
    min_training_samples: int = Field(default=5, ge=3, le=50)
    cross_validation_folds: int = Field(default=5, ge=2, le=10)

    # Active learning
    uncertainty_threshold: float = Field(default=0.1, ge=0.01, le=0.5)
    exploration_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Persistence
    auto_save_model: bool = Field(default=True)
    model_cache_dir: Optional[Path] = Field(default=None)


class LLMSettings(BaseSettings):
    """Settings for LLM integration."""

    model_config = SettingsConfigDict(env_prefix="PTPD_LLM_")

    # Provider configuration
    provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC)
    api_key: Optional[str] = Field(default=None)

    # Model selection
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    openai_model: str = Field(default="gpt-4o")

    # Request parameters
    max_tokens: int = Field(default=4096, ge=100, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=10, le=300)

    # Rate limiting
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)


class AgentSettings(BaseSettings):
    """Settings for agentic system."""

    model_config = SettingsConfigDict(env_prefix="PTPD_AGENT_")

    # Planning
    max_plan_steps: int = Field(default=10, ge=3, le=50)
    max_iterations: int = Field(default=20, ge=5, le=100)
    planning_timeout_seconds: int = Field(default=300, ge=30, le=1800)

    # Memory
    enable_memory: bool = Field(default=True)
    memory_file: Optional[Path] = Field(default=None)
    max_memory_items: int = Field(default=1000, ge=100, le=10000)
    working_memory_size: int = Field(default=10, ge=3, le=50)

    # Tool execution
    tool_timeout_seconds: int = Field(default=30, ge=5, le=300)
    parallel_tool_calls: bool = Field(default=True)

    # Reflection
    enable_reflection: bool = Field(default=True)
    reflection_frequency: int = Field(default=3, ge=1, le=10)


class APISettings(BaseSettings):
    """Settings for API server."""

    model_config = SettingsConfigDict(env_prefix="PTPD_API_")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    reload: bool = Field(default=False)

    # CORS
    cors_origins: list[str] = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)

    # File uploads
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    upload_dir: Optional[Path] = Field(default=None)

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)


class Settings(BaseSettings):
    """Main application settings aggregating all subsettings."""

    model_config = SettingsConfigDict(
        env_prefix="PTPD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application info
    app_name: str = Field(default="PTPD Calibration Studio")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Data directories
    data_dir: Path = Field(default=Path.home() / ".ptpd")
    calibrations_dir: Optional[Path] = Field(default=None)
    exports_dir: Optional[Path] = Field(default=None)

    # Subsettings
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    curves: CurveSettings = Field(default_factory=CurveSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("calibrations_dir", "exports_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v: Optional[Path], info) -> Optional[Path]:
        """Resolve paths relative to data_dir if not absolute."""
        if v is None:
            return None
        path = Path(v)
        if not path.is_absolute():
            # Get data_dir from values if available
            data = info.data if hasattr(info, "data") else {}
            data_dir = data.get("data_dir", Path.home() / ".ptpd")
            return data_dir / path
        return path

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.calibrations_dir:
            self.calibrations_dir.mkdir(parents=True, exist_ok=True)
        if self.exports_dir:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
        if self.ml.model_cache_dir:
            self.ml.model_cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance - lazy loaded
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, creating it if necessary."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(settings: Optional[Settings] = None, **kwargs) -> Settings:
    """
    Configure the global settings.

    Args:
        settings: Optional Settings instance to use directly
        **kwargs: Settings overrides

    Returns:
        The configured Settings instance
    """
    global _settings
    if settings is not None:
        _settings = settings
    elif kwargs:
        _settings = Settings(**kwargs)
    else:
        _settings = Settings()
    return _settings
