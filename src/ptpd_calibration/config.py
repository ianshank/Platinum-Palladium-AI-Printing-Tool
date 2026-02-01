"""
Configuration management for PTPD Calibration System.

Uses pydantic-settings for environment-based configuration with validation.
All settings can be overridden via environment variables with PTPD_ prefix.
"""

from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


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
    status_a_weights: tuple[float, float, float] = Field(default=(0.2126, 0.7152, 0.0722))

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
    model_cache_dir: Path | None = Field(default=None)


class DeepLearningSettings(BaseSettings):
    """Settings for deep learning-based curve prediction.

    These settings control PyTorch-based neural network models for
    learning tone curves from calibration data.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_")

    # Model architecture
    model_type: str = Field(
        default="curve_mlp",
        description="Model type: curve_mlp, curve_cnn, content_aware, uniformity",
    )
    num_control_points: int = Field(
        default=16, ge=4, le=64, description="Number of control points for curve representation"
    )
    lut_size: int = Field(
        default=256, ge=16, le=4096, description="Size of output LUT (lookup table)"
    )

    # MLP architecture
    hidden_dims: list[int] = Field(
        default_factory=lambda: [128, 256, 128], description="Hidden layer dimensions for MLP"
    )
    dropout_rate: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Dropout rate for regularization"
    )
    use_batch_norm: bool = Field(default=True, description="Use batch normalization in MLP layers")

    # Process simulation
    process_gamma_init: float = Field(
        default=1.8, ge=0.5, le=4.0, description="Initial gamma value for process simulator"
    )
    process_dmin_init: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Initial Dmin for process simulator"
    )
    process_dmax_init: float = Field(
        default=2.0, ge=1.0, le=4.0, description="Initial Dmax for process simulator"
    )

    # Uniformity correction
    uniformity_kernel_size: int = Field(
        default=31, ge=3, le=101, description="Kernel size for uniformity smoothing (must be odd)"
    )
    uniformity_sigma: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Sigma for Gaussian smoothing in uniformity correction",
    )
    uniformity_range: tuple[float, float] = Field(
        default=(0.8, 1.2), description="Range for multiplicative correction factor"
    )

    # Training parameters
    learning_rate: float = Field(
        default=1e-3, ge=1e-6, le=1.0, description="Learning rate for optimizer"
    )
    weight_decay: float = Field(
        default=1e-4, ge=0.0, le=1e-1, description="L2 regularization weight decay"
    )
    batch_size: int = Field(default=32, ge=1, le=512, description="Training batch size")
    num_epochs: int = Field(
        default=100, ge=1, le=10000, description="Maximum number of training epochs"
    )
    early_stopping_patience: int = Field(
        default=10, ge=1, le=100, description="Epochs to wait before early stopping"
    )
    min_delta: float = Field(
        default=1e-4, ge=0.0, le=1e-1, description="Minimum improvement for early stopping"
    )

    # Loss function weights
    mse_weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Weight for MSE loss component"
    )
    monotonicity_weight: float = Field(
        default=0.1, ge=0.0, le=10.0, description="Weight for monotonicity regularization"
    )
    smoothness_weight: float = Field(
        default=0.05, ge=0.0, le=10.0, description="Weight for smoothness regularization"
    )
    perceptual_weight: float = Field(
        default=0.1, ge=0.0, le=10.0, description="Weight for perceptual loss (Lab deltaE)"
    )

    # Data augmentation
    augmentation_enabled: bool = Field(
        default=True, description="Enable data augmentation during training"
    )
    noise_std: float = Field(
        default=0.02, ge=0.0, le=0.1, description="Standard deviation of noise augmentation"
    )
    exposure_jitter: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Exposure time jitter factor for augmentation"
    )

    # Inference
    use_ensemble: bool = Field(default=True, description="Use ensemble of models for prediction")
    num_ensemble_models: int = Field(
        default=5, ge=1, le=20, description="Number of models in ensemble"
    )
    mc_dropout_samples: int = Field(
        default=10, ge=1, le=100, description="Number of MC dropout samples for uncertainty"
    )

    # Device settings
    device: str = Field(default="auto", description="Device for training: auto, cpu, cuda, mps")
    mixed_precision: bool = Field(default=False, description="Use mixed precision (FP16) training")

    # Persistence
    checkpoint_dir: Path | None = Field(default=None, description="Directory for model checkpoints")
    save_best_only: bool = Field(default=True, description="Only save best model checkpoint")


class LLMSettings(BaseSettings):
    """Settings for LLM integration.

    Users can provide their API key via:
    1. Environment variable: PTPD_LLM_API_KEY or PTPD_LLM_ANTHROPIC_API_KEY
    2. Runtime: Through the Settings tab in the UI (for HuggingFace deployment)
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_LLM_")

    # Provider configuration
    provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC)
    api_key: str | None = Field(
        default=None, description="Primary API key (used if provider-specific key not set)"
    )
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key for Claude models"
    )
    openai_api_key: str | None = Field(default=None, description="OpenAI API key for GPT models")

    # Runtime API key (can be set via UI, takes precedence)
    runtime_api_key: str | None = Field(
        default=None, description="Runtime API key set via UI (takes precedence over env vars)"
    )

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

    def get_active_api_key(self) -> str | None:
        """Get the active API key with runtime key taking precedence."""
        if self.runtime_api_key:
            return self.runtime_api_key
        if self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key or self.api_key
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_api_key or self.api_key
        return self.api_key


class AgentSettings(BaseSettings):
    """Settings for agentic system."""

    model_config = SettingsConfigDict(env_prefix="PTPD_AGENT_")

    # Planning
    max_plan_steps: int = Field(default=10, ge=3, le=50)
    max_iterations: int = Field(default=20, ge=5, le=100)
    planning_timeout_seconds: int = Field(default=300, ge=30, le=1800)

    # Memory
    enable_memory: bool = Field(default=True)
    memory_file: Path | None = Field(default=None)
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
    upload_dir: Path | None = Field(default=None)

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)


class VisualizationSettings(BaseSettings):
    """Settings for curve visualization."""

    model_config = SettingsConfigDict(env_prefix="PTPD_VIS_")

    # Figure dimensions
    figure_width: float = Field(default=10.0, ge=4.0, le=20.0)
    figure_height: float = Field(default=6.0, ge=3.0, le=15.0)
    dpi: int = Field(default=100, ge=50, le=300)

    # Colors and styling
    background_color: str = Field(default="#FAF8F5")
    grid_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    line_width: float = Field(default=2.0, ge=0.5, le=5.0)
    marker_size: float = Field(default=6.0, ge=2.0, le=15.0)

    # Color scheme (platinum, monochrome, vibrant, pastel, accessible)
    color_scheme: str = Field(default="platinum")

    # Font sizes
    title_fontsize: int = Field(default=14, ge=8, le=24)
    label_fontsize: int = Field(default=12, ge=6, le=20)
    tick_fontsize: int = Field(default=10, ge=6, le=16)
    legend_fontsize: int = Field(default=10, ge=6, le=16)

    # Display options
    show_grid: bool = Field(default=True)
    show_legend: bool = Field(default=True)
    show_reference_line: bool = Field(default=True)


class ChemistrySettings(BaseSettings):
    """Settings for platinum/palladium chemistry calculations.

    Based on Bostick-Sullivan formulas and industry standards.
    Reference: https://www.bostick-sullivan.com/wp-content/uploads/2022/03/platinum-and-palladium-kit-instructions.pdf

    Standard formula: A (FO #1) + B (FO #2 contrast) + C (metals) = total coating solution
    Rule: Drops of metals (C) should equal drops of ferric oxalate (A + B)
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_CHEM_")

    # Standard drops per square inch (based on 46 drops for 8x10" = 99 sq in)
    drops_per_square_inch: float = Field(
        default=0.465,
        ge=0.2,
        le=1.0,
        description="Base drops per square inch for average absorbency paper",
    )

    # Drops per ml (standard plastic dropper)
    drops_per_ml: float = Field(default=20.0, ge=15.0, le=25.0)

    # Default metal ratio (platinum percentage, rest is palladium)
    default_platinum_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Default platinum ratio (0.0 = all palladium, 1.0 = all platinum)",
    )

    # Contrast agent (Na2) default drops per 24 drops of metal
    default_na2_drops_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="Na2 drops as ratio of metal drops (e.g., 6 drops Na2 per 24 drops metal = 0.25)",
    )

    # Paper absorbency multipliers
    low_absorbency_multiplier: float = Field(
        default=0.80, ge=0.5, le=1.0, description="Multiplier for hot press/low absorbency papers"
    )
    medium_absorbency_multiplier: float = Field(default=1.0, ge=0.8, le=1.2)
    high_absorbency_multiplier: float = Field(
        default=1.20, ge=1.0, le=1.5, description="Multiplier for cold press/high absorbency papers"
    )

    # Coating method adjustments
    brush_coating_multiplier: float = Field(default=1.0, ge=0.8, le=1.2)
    rod_coating_multiplier: float = Field(
        default=0.75, ge=0.5, le=1.0, description="Glass rod coating uses less solution"
    )

    # Default margin (inches to subtract from each side for coating area)
    default_margin_inches: float = Field(default=0.5, ge=0.0, le=2.0)

    # Solution costs (USD per ml) for cost estimation
    ferric_oxalate_cost_per_ml: float = Field(default=0.50, ge=0.0, le=10.0)
    palladium_cost_per_ml: float = Field(default=2.00, ge=0.0, le=50.0)
    platinum_cost_per_ml: float = Field(default=8.00, ge=0.0, le=100.0)
    na2_cost_per_ml: float = Field(default=4.00, ge=0.0, le=50.0)


class WedgeAnalysisSettings(BaseSettings):
    """Settings for step wedge analysis."""

    model_config = SettingsConfigDict(env_prefix="PTPD_WEDGE_")

    # Default tablet type (stouffer_21, stouffer_31, stouffer_41, custom)
    default_tablet_type: str = Field(default="stouffer_21")

    # Quality thresholds
    min_density_range: float = Field(default=1.5, ge=0.5, le=3.0)
    max_dmin: float = Field(default=0.15, ge=0.0, le=0.5)
    min_dmax: float = Field(default=1.8, ge=1.0, le=3.5)
    uniformity_threshold: float = Field(default=0.7, ge=0.3, le=1.0)

    # Monotonicity settings
    max_reversal_tolerance: float = Field(default=0.02, ge=0.0, le=0.1)
    smoothing_window: int = Field(default=3, ge=1, le=7)

    # Curve generation defaults
    default_curve_type: str = Field(default="linear")
    num_output_points: int = Field(default=256, ge=16, le=4096)
    apply_smoothing: bool = Field(default=True)
    smoothing_factor: float = Field(default=0.05, ge=0.0, le=0.5)
    enforce_monotonicity: bool = Field(default=True)

    # Advanced settings
    auto_fix_reversals: bool = Field(default=True)
    outlier_rejection: bool = Field(default=True)
    outlier_threshold: float = Field(default=2.5, ge=1.0, le=5.0)


class WorkflowSettings(BaseSettings):
    """Settings for workflow automation and recipe management."""

    model_config = SettingsConfigDict(env_prefix="PTPD_WORKFLOW_")

    # Database settings
    recipe_db_path: Path | None = Field(
        default=None, description="Path to recipe database (defaults to data_dir/recipes.db)"
    )
    auto_backup: bool = Field(default=True, description="Automatically backup recipe database")
    backup_interval_hours: int = Field(
        default=24, ge=1, le=168, description="Hours between automatic backups"
    )

    # Batch processing
    default_max_workers: int = Field(
        default=4, ge=1, le=32, description="Default number of parallel workers"
    )
    batch_timeout_minutes: int = Field(
        default=60, ge=5, le=480, description="Timeout for batch jobs (minutes)"
    )

    # Workflow execution
    enable_scheduling: bool = Field(default=True, description="Enable workflow scheduling")
    max_concurrent_workflows: int = Field(
        default=3, ge=1, le=10, description="Maximum concurrent workflows"
    )


class QASettings(BaseSettings):
    """Quality assurance settings."""

    model_config = SettingsConfigDict(env_prefix="PTPD_QA_")

    # Density validation
    min_density: float = Field(default=0.0, ge=0.0, le=1.0)
    max_density: float = Field(default=3.5, ge=1.0, le=5.0)
    highlight_warning_threshold: float = Field(default=0.10, ge=0.0, le=0.5)
    shadow_warning_threshold: float = Field(default=2.0, ge=1.0, le=4.0)
    density_step_warning: float = Field(default=0.15, ge=0.05, le=0.5)

    # Density warning thresholds (additional)
    density_min_warning: float = Field(default=0.05, ge=0.0, le=0.2)
    density_max_warning: float = Field(default=3.0, ge=2.0, le=4.0)
    density_range_warning: float = Field(default=1.5, ge=0.5, le=3.0)
    density_uniformity_warning: float = Field(default=0.05, ge=0.01, le=0.2)

    # Chemistry freshness (in days)
    ferric_oxalate_shelf_life: int = Field(default=180, ge=1, le=730)
    palladium_shelf_life: int = Field(default=365, ge=1, le=1095)
    platinum_shelf_life: int = Field(default=365, ge=1, le=1095)
    na2_shelf_life: int = Field(default=365, ge=1, le=1095)
    developer_shelf_life: int = Field(default=90, ge=1, le=365)
    clearing_bath_shelf_life: int = Field(default=90, ge=1, le=365)
    edta_shelf_life: int = Field(default=90, ge=1, le=365)

    # Chemistry usage alerts (percentage remaining)
    low_volume_warning_percent: float = Field(default=20.0, ge=0.0, le=50.0)
    critical_volume_percent: float = Field(default=10.0, ge=0.0, le=25.0)

    # Expiration warnings (days before expiration)
    expiration_warning_days: int = Field(default=30, ge=1, le=90)
    expiration_critical_days: int = Field(default=7, ge=1, le=30)

    # Paper humidity
    ideal_humidity_min: float = Field(default=40.0, ge=20.0, le=60.0)
    ideal_humidity_max: float = Field(default=60.0, ge=40.0, le=80.0)
    humidity_tolerance: float = Field(default=5.0, ge=2.0, le=15.0)
    humidity_optimal_range: tuple[float, float] = Field(
        default=(45.0, 55.0), description="Optimal humidity range (min, max) in percent"
    )

    # Drying time estimation (hours per % humidity difference)
    drying_time_factor: float = Field(default=0.5, ge=0.1, le=2.0)

    # UV light
    uv_intensity_target: float = Field(default=100.0, ge=10.0, le=1000.0)
    uv_intensity_tolerance: float = Field(default=10.0, ge=5.0, le=30.0)
    uv_intensity_reference: float = Field(default=100.0, ge=10.0, le=1000.0)
    uv_wavelength_target: float = Field(default=365.0, ge=300.0, le=420.0)
    bulb_degradation_threshold: float = Field(default=15.0, ge=5.0, le=30.0)
    bulb_replacement_hours: int = Field(default=1000, ge=100, le=5000)

    # Alert retention
    alert_history_days: int = Field(default=90, ge=7, le=365)


class PlatinumPalladiumAISettings(BaseSettings):
    """Settings for AI-powered platinum/palladium printing features."""

    model_config = SettingsConfigDict(env_prefix="PTPD_AI_")

    # Tonality analysis
    tonality_analysis_enabled: bool = Field(default=True)
    tonality_histogram_bins: int = Field(default=256, ge=64, le=1024)
    tonality_percentile_threshold: float = Field(default=0.02, ge=0.0, le=0.1)

    # Exposure prediction
    exposure_prediction_enabled: bool = Field(default=True)
    exposure_model_type: str = Field(default="gradient_boosting")
    exposure_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Digital negative generation
    digital_negative_enabled: bool = Field(default=True)
    digital_negative_bit_depth: int = Field(default=16, ge=8, le=16)
    digital_negative_dpi: int = Field(default=1440, ge=300, le=5760)

    # Confidence thresholds
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    uncertainty_warning_threshold: float = Field(default=0.15, ge=0.0, le=0.5)


class SplitGradeSettings(BaseSettings):
    """Settings for split-grade printing technique."""

    model_config = SettingsConfigDict(env_prefix="PTPD_SPLIT_GRADE_")

    # Default grade values
    default_shadow_grade: float = Field(default=5.0, ge=0.0, le=5.0)
    default_highlight_grade: float = Field(default=0.0, ge=0.0, le=5.0)

    # Thresholds
    shadow_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    highlight_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Blending
    blend_gamma: float = Field(default=2.2, ge=1.0, le=3.0)
    blend_smoothness: float = Field(default=0.5, ge=0.0, le=1.0)

    # Advanced
    auto_split_enabled: bool = Field(default=True)
    split_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class RecipeSettings(BaseSettings):
    """Settings for recipe management and versioning."""

    model_config = SettingsConfigDict(env_prefix="PTPD_RECIPE_")

    # Database settings
    database_path: Path | None = Field(
        default=None, description="Path to recipe database (defaults to data_dir/recipes.db)"
    )

    # Versioning
    max_recipe_versions: int = Field(default=50, ge=5, le=500)
    auto_version_on_save: bool = Field(default=True)

    # Auto-save
    auto_save_enabled: bool = Field(default=True)
    auto_save_interval_seconds: int = Field(default=300, ge=30, le=3600)

    # Search and filtering
    enable_full_text_search: bool = Field(default=True)
    search_result_limit: int = Field(default=50, ge=10, le=500)

    # Tags and categorization
    default_tags: list[str] = Field(default_factory=list)
    max_tags_per_recipe: int = Field(default=20, ge=1, le=100)


class CyanotypeSettings(BaseSettings):
    """Settings for cyanotype chemistry and exposure calculations.

    Based on traditional cyanotype formulas:
    - Classic (Sir John Herschel): FAC + Potassium Ferricyanide
    - New Cyanotype (Mike Ware): Modified iron salts for better results
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_CYANOTYPE_")

    # Base coating amounts
    ml_per_square_inch: float = Field(
        default=0.015, ge=0.005, le=0.05, description="Milliliters of sensitizer per square inch"
    )
    drops_per_ml: float = Field(default=20.0, ge=15.0, le=25.0)

    # Default margin
    default_margin_inches: float = Field(default=0.5, ge=0.0, le=2.0)

    # Solution costs (USD per ml of stock solution)
    solution_a_cost_per_ml: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Cost per ml of ferric ammonium citrate solution"
    )
    solution_b_cost_per_ml: float = Field(
        default=0.08, ge=0.0, le=1.0, description="Cost per ml of potassium ferricyanide solution"
    )

    # Exposure defaults
    base_sunlight_exposure_minutes: float = Field(
        default=15.0, ge=5.0, le=60.0, description="Base exposure time in direct sunlight"
    )
    base_bl_tube_exposure_minutes: float = Field(
        default=15.0, ge=5.0, le=60.0, description="Base exposure time with BL fluorescent tubes"
    )

    # Process characteristics
    typical_dmax: float = Field(default=1.9, ge=1.0, le=3.0)
    typical_dmin: float = Field(default=0.12, ge=0.0, le=0.5)

    # Paper recommendations
    recommended_paper_types: list[str] = Field(
        default_factory=lambda: ["Arches Platine", "Stonehenge", "Fabriano Artistico"]
    )


class SilverGelatinSettings(BaseSettings):
    """Settings for silver gelatin darkroom processing.

    Based on standard darkroom chemistry and procedures.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_SILVER_GELATIN_")

    # Developer defaults
    default_developer: str = Field(
        default="dektol", description="Default paper developer (dektol, selectol, etc.)"
    )
    default_dilution: str = Field(default="1:2", description="Default developer dilution ratio")
    default_temperature_c: float = Field(
        default=20.0, ge=18.0, le=24.0, description="Default developer temperature in Celsius"
    )
    default_temperature_f: float = Field(
        default=68.0, ge=65.0, le=75.0, description="Default developer temperature in Fahrenheit"
    )

    # Development times (seconds)
    default_development_time_seconds: int = Field(
        default=90, ge=30, le=300, description="Default development time in seconds"
    )
    stop_bath_time_seconds: int = Field(
        default=30, ge=15, le=60, description="Stop bath time in seconds"
    )

    # Fixer settings
    fixer_time_fb_seconds: int = Field(
        default=300, ge=120, le=600, description="Fixer time for fiber-based paper (seconds)"
    )
    fixer_time_rc_seconds: int = Field(
        default=120, ge=60, le=300, description="Fixer time for RC paper (seconds)"
    )

    # Wash settings
    wash_time_fb_minutes: int = Field(
        default=60, ge=30, le=120, description="Wash time for fiber-based paper (minutes)"
    )
    wash_time_rc_minutes: int = Field(
        default=4, ge=2, le=10, description="Wash time for RC paper (minutes)"
    )

    # Chemistry costs (USD per liter of working solution)
    developer_cost_per_liter: float = Field(default=0.50, ge=0.0, le=5.0)
    stop_bath_cost_per_liter: float = Field(default=0.10, ge=0.0, le=2.0)
    fixer_cost_per_liter: float = Field(default=0.30, ge=0.0, le=3.0)
    hypo_clear_cost_per_liter: float = Field(default=0.20, ge=0.0, le=2.0)

    # Paper recommendations
    recommended_papers: list[str] = Field(
        default_factory=lambda: [
            "Ilford MGIV FB",
            "Ilford MGIV RC",
            "Ilford MGFB Warmtone",
            "Foma Fomabrom",
            "Bergger Prestige CB",
        ]
    )

    # Safelight settings
    safelight_filter: str = Field(
        default="OC", description="Recommended safelight filter (OC, OA, etc.)"
    )


class AdvancedFeaturesSettings(BaseSettings):
    """Settings for advanced features like QR codes, style transfer, and process simulation."""

    model_config = SettingsConfigDict(env_prefix="PTPD_ADVANCED_")

    # QR code settings
    qr_code_enabled: bool = Field(default=True)
    qr_code_size: int = Field(default=100, ge=50, le=500)
    qr_error_correction: str = Field(
        default="M", description="QR error correction level (L, M, Q, H)"
    )
    qr_border_size: int = Field(default=4, ge=1, le=10)

    # Style transfer
    style_transfer_enabled: bool = Field(default=True)
    style_transfer_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    style_transfer_model: str = Field(default="neural_style")

    # Process simulation gamma values
    process_simulation_enabled: bool = Field(default=True)
    simulation_gamma_cyanotype: float = Field(default=1.8, ge=1.0, le=3.0)
    simulation_gamma_vandyke: float = Field(default=2.0, ge=1.0, le=3.0)
    simulation_gamma_kallittype: float = Field(default=1.9, ge=1.0, le=3.0)
    simulation_gamma_argyrotype: float = Field(default=2.1, ge=1.0, le=3.0)

    # Additional process parameters
    simulation_contrast_boost: float = Field(default=1.1, ge=0.5, le=2.0)
    simulation_color_tint_enabled: bool = Field(default=True)


class IntegrationSettings(BaseSettings):
    """Settings for hardware and API integrations."""

    model_config = SettingsConfigDict(env_prefix="PTPD_INTEGRATION_")

    # Weather API settings
    weather_api_key: str | None = Field(
        default=None, description="OpenWeatherMap API key for environmental data"
    )
    weather_api_url: str = Field(
        default="https://api.openweathermap.org/data/2.5/weather",
        description="Weather API endpoint URL",
    )
    weather_location: str = Field(
        default="Portland, OR", description="Default location for weather data"
    )
    weather_cache_minutes: int = Field(
        default=10, ge=1, le=60, description="Weather data cache duration (minutes)"
    )

    # Spectrophotometer settings
    spectrophotometer_port: str | None = Field(
        default=None, description="Serial port for spectrophotometer (e.g., /dev/ttyUSB0 or COM3)"
    )
    spectro_device_id: str | None = Field(
        default=None, description="Spectrophotometer device identifier"
    )
    spectro_measurement_mode: str = Field(
        default="reflection",
        description="Default measurement mode (reflection, transmission, density)",
    )
    spectro_aperture_size: str = Field(
        default="medium", description="Aperture size (small, medium, large)"
    )
    spectro_baud_rate: int = Field(default=9600, ge=1200, le=115200)
    spectro_timeout_seconds: int = Field(default=5, ge=1, le=30)
    spectro_simulate: bool = Field(
        default=True, description="Use simulated spectrophotometer for testing"
    )

    # Printer settings
    default_printer_name: str | None = Field(
        default=None, description="Default printer for digital negatives"
    )
    default_printer_brand: str = Field(
        default="epson", description="Default printer brand (epson, canon, hp)"
    )
    default_printer_model: str = Field(default="R2400", description="Default printer model")
    printer_driver: str = Field(
        default="gutenprint", description="Printer driver (gutenprint, cups, native)"
    )
    printer_resolution: int = Field(default=2880, ge=360, le=5760)
    printer_paper_feed: str = Field(default="sheet", description="Paper feed type")
    printer_simulate: bool = Field(default=True, description="Use simulated printer for testing")

    # ICC Profile settings
    icc_profile_paths: list[str] = Field(
        default_factory=lambda: [
            "/usr/share/color/icc",
            "~/.local/share/color/icc",
        ],
        description="System ICC profile search paths",
    )
    custom_profile_dir: Path | None = Field(
        default=None, description="Custom directory for ICC profiles"
    )
    default_rendering_intent: str = Field(
        default="perceptual",
        description="Default ICC rendering intent (perceptual, relative, saturation, absolute)",
    )
    default_icc_profile: str | None = Field(default=None, description="Default ICC profile name")

    # Paper drying time defaults
    default_paper_type: str = Field(
        default="cold_press", description="Default paper type for drying time calculations"
    )


class EducationSettings(BaseSettings):
    """Settings for educational features and tutorials."""

    model_config = SettingsConfigDict(env_prefix="PTPD_EDUCATION_")

    # Tutorial paths
    tutorials_path: Path | None = Field(
        default=None, description="Path to tutorials directory (defaults to data_dir/tutorials)"
    )
    glossary_path: Path | None = Field(
        default=None, description="Path to glossary file (defaults to data_dir/glossary.json)"
    )

    # Display settings
    show_tips: bool = Field(default=True, description="Show tips and hints in UI")
    show_tooltips: bool = Field(default=True, description="Show detailed tooltips")
    tutorial_auto_advance: bool = Field(default=False)

    # Tutorial difficulty
    default_difficulty_level: str = Field(
        default="beginner",
        description="Default tutorial difficulty (beginner, intermediate, advanced)",
    )

    # Progress tracking
    track_progress: bool = Field(default=True, description="Track tutorial progress")
    progress_file: Path | None = Field(default=None, description="Path to progress tracking file")


class PerformanceSettings(BaseSettings):
    """Settings for performance monitoring and optimization."""

    model_config = SettingsConfigDict(env_prefix="PTPD_PERFORMANCE_")

    # Profiling
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    profiling_output_dir: Path | None = Field(
        default=None, description="Directory for profiling output"
    )
    profile_memory: bool = Field(default=False)
    profile_cpu: bool = Field(default=True)

    # Caching
    enable_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Cache time-to-live in seconds"
    )
    cache_max_size: int = Field(
        default=1000, ge=10, le=100000, description="Maximum number of cached items"
    )
    cache_backend: str = Field(default="memory", description="Cache backend (memory, redis, disk)")

    # Metrics
    enable_metrics: bool = Field(default=True)
    metrics_retention_days: int = Field(
        default=30, ge=1, le=365, description="Days to retain performance metrics"
    )
    metrics_export_path: Path | None = Field(default=None, description="Path to export metrics")

    # Optimization
    lazy_loading: bool = Field(default=True)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=32)


class DataManagementSettings(BaseSettings):
    """Settings for data management, backup, and sync."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DATA_")

    # Database settings
    database_path: Path | None = Field(
        default=None, description="Path to main database (defaults to data_dir/ptpd.db)"
    )
    database_backup_path: Path | None = Field(
        default=None, description="Path to database backups directory"
    )

    # Backup settings
    auto_backup: bool = Field(default=True)
    backup_interval_hours: int = Field(
        default=24, ge=1, le=168, description="Hours between automatic backups"
    )
    max_backup_count: int = Field(
        default=30, ge=1, le=365, description="Maximum number of backups to keep"
    )
    backup_compression: bool = Field(default=True)

    # Cloud sync
    cloud_sync_enabled: bool = Field(default=False)
    cloud_provider: str = Field(
        default="s3", description="Cloud provider (s3, gcs, azure, dropbox)"
    )
    cloud_bucket_name: str | None = Field(default=None)
    cloud_api_key: str | None = Field(default=None)
    cloud_sync_interval_minutes: int = Field(default=60, ge=5, le=1440)

    # Data retention
    data_retention_days: int = Field(
        default=365, ge=7, le=3650, description="Days to retain historical data"
    )
    archive_old_data: bool = Field(default=True)

    # Export settings
    default_export_format: str = Field(default="json")
    export_path: Path | None = Field(default=None)


class NeuroSymbolicSettings(BaseSettings):
    """Settings for neuro-symbolic AI curve generation and reasoning.

    This module combines neural networks with symbolic reasoning for
    physically-constrained curve generation and interpretable predictions.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_NEUROSYM_")

    # Constraint Settings
    monotonicity_weight: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Weight for monotonicity constraint in loss function",
    )
    density_bounds_weight: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Weight for density bounds constraint"
    )
    physics_constraint_weight: float = Field(
        default=8.0,
        ge=0.0,
        le=100.0,
        description="Weight for physics-based constraints (H&D curve laws)",
    )
    smoothness_weight: float = Field(
        default=2.0, ge=0.0, le=100.0, description="Weight for smoothness regularization"
    )

    # Density Bounds (configurable physical limits)
    min_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum valid density (paper base)"
    )
    max_density: float = Field(
        default=3.5, ge=1.0, le=5.0, description="Maximum valid density (Dmax)"
    )
    expected_dmin_range: tuple[float, float] = Field(
        default=(0.02, 0.15), description="Expected Dmin range for valid calibrations"
    )
    expected_dmax_range: tuple[float, float] = Field(
        default=(1.8, 3.2), description="Expected Dmax range for valid calibrations"
    )

    # Optimization Settings
    optimizer_learning_rate: float = Field(
        default=0.01, ge=0.0001, le=1.0, description="Learning rate for constrained optimization"
    )
    optimizer_max_iterations: int = Field(
        default=1000, ge=10, le=10000, description="Maximum iterations for curve optimization"
    )
    optimizer_tolerance: float = Field(
        default=1e-6, ge=1e-10, le=1e-2, description="Convergence tolerance for optimization"
    )
    optimizer_method: str = Field(
        default="L-BFGS-B", description="Optimization method (L-BFGS-B, Adam, SGD)"
    )

    # Knowledge Graph Settings
    kg_embedding_dim: int = Field(
        default=64, ge=16, le=512, description="Embedding dimension for knowledge graph entities"
    )
    kg_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for entity similarity in analogical reasoning",
    )
    kg_max_inference_depth: int = Field(
        default=3, ge=1, le=10, description="Maximum depth for graph traversal inference"
    )
    kg_enable_learning: bool = Field(
        default=True, description="Enable learning new relationships from calibration data"
    )

    # Symbolic Regression Settings
    sr_max_expression_depth: int = Field(
        default=5, ge=2, le=10, description="Maximum depth of symbolic expressions"
    )
    sr_population_size: int = Field(
        default=100, ge=20, le=1000, description="Population size for genetic programming"
    )
    sr_generations: int = Field(
        default=50, ge=10, le=500, description="Number of generations for symbolic regression"
    )
    sr_mutation_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Mutation rate for genetic operators"
    )
    sr_crossover_rate: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Crossover rate for genetic operators"
    )
    sr_parsimony_coefficient: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Parsimony pressure for simpler expressions"
    )
    sr_allowed_operators: list[str] = Field(
        default_factory=lambda: ["add", "sub", "mul", "div", "pow", "log", "exp", "sqrt"],
        description="Allowed operators in symbolic expressions",
    )

    # Physics Model Settings
    physics_toe_exponent_range: tuple[float, float] = Field(
        default=(0.3, 0.8), description="Expected toe region exponent range (sqrt-like behavior)"
    )
    physics_shoulder_saturation_rate: tuple[float, float] = Field(
        default=(0.5, 2.0), description="Expected shoulder saturation rate range"
    )
    physics_reciprocity_failure_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for reciprocity failure detection"
    )

    # Uncertainty Quantification
    uncertainty_num_samples: int = Field(
        default=100, ge=10, le=1000, description="Number of samples for uncertainty estimation"
    )
    uncertainty_confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for uncertainty intervals"
    )

    # Explanation Settings
    enable_explanations: bool = Field(
        default=True, description="Generate explanations for predictions"
    )
    explanation_verbosity: str = Field(
        default="medium", description="Explanation verbosity (brief, medium, detailed)"
    )


class CalculationsSettings(BaseSettings):
    """Settings for environmental calculations and cost estimation."""

    model_config = SettingsConfigDict(env_prefix="PTPD_CALC_")

    # Environmental factor coefficients
    temperature_coefficient: float = Field(
        default=0.02, ge=0.0, le=0.1, description="Exposure adjustment per degree C from optimal"
    )
    humidity_coefficient: float = Field(
        default=0.015, ge=0.0, le=0.1, description="Exposure adjustment per % humidity from optimal"
    )
    altitude_coefficient: float = Field(
        default=0.001, ge=0.0, le=0.01, description="Exposure adjustment per 100m elevation"
    )

    # Optimal conditions
    optimal_temperature_c: float = Field(
        default=20.0, ge=15.0, le=25.0, description="Optimal working temperature in Celsius"
    )
    optimal_humidity_percent: float = Field(
        default=50.0, ge=30.0, le=70.0, description="Optimal relative humidity percentage"
    )
    optimal_uv_intensity: float = Field(
        default=100.0, ge=10.0, le=1000.0, description="Optimal UV intensity (arbitrary units)"
    )

    # Cost per ml values (USD)
    ferric_oxalate_cost_per_ml: float = Field(default=0.50, ge=0.0, le=10.0)
    palladium_cost_per_ml: float = Field(default=2.00, ge=0.0, le=50.0)
    platinum_cost_per_ml: float = Field(default=8.00, ge=0.0, le=100.0)
    na2_cost_per_ml: float = Field(default=4.00, ge=0.0, le=50.0)
    developer_cost_per_ml: float = Field(default=0.10, ge=0.0, le=5.0)
    clearing_bath_cost_per_ml: float = Field(default=0.15, ge=0.0, le=5.0)

    # Paper costs (per sheet)
    default_paper_cost_per_sheet: float = Field(default=5.0, ge=0.0, le=100.0)

    # Labor and overhead
    hourly_labor_rate: float = Field(default=0.0, ge=0.0, le=500.0)
    overhead_multiplier: float = Field(default=1.2, ge=1.0, le=3.0)

    # Exposure time calculations
    exposure_base_time_seconds: float = Field(default=180.0, ge=1.0, le=3600.0)
    uv_intensity_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)


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
    calibrations_dir: Path | None = Field(default=None)
    exports_dir: Path | None = Field(default=None)

    # Subsettings
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    curves: CurveSettings = Field(default_factory=CurveSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    deep_learning: DeepLearningSettings = Field(default_factory=DeepLearningSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    wedge_analysis: WedgeAnalysisSettings = Field(default_factory=WedgeAnalysisSettings)
    chemistry: ChemistrySettings = Field(default_factory=ChemistrySettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    qa: QASettings = Field(default_factory=QASettings)
    integrations: IntegrationSettings = Field(default_factory=IntegrationSettings)

    # New subsettings for expanded features
    ai: PlatinumPalladiumAISettings = Field(default_factory=PlatinumPalladiumAISettings)
    split_grade: SplitGradeSettings = Field(default_factory=SplitGradeSettings)
    recipe: RecipeSettings = Field(default_factory=RecipeSettings)
    advanced: AdvancedFeaturesSettings = Field(default_factory=AdvancedFeaturesSettings)
    education: EducationSettings = Field(default_factory=EducationSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    data_management: DataManagementSettings = Field(default_factory=DataManagementSettings)
    calculations: CalculationsSettings = Field(default_factory=CalculationsSettings)
    neuro_symbolic: NeuroSymbolicSettings = Field(default_factory=NeuroSymbolicSettings)

    # Alternative process settings
    cyanotype: CyanotypeSettings = Field(default_factory=CyanotypeSettings)
    silver_gelatin: SilverGelatinSettings = Field(default_factory=SilverGelatinSettings)

    @field_validator("calibrations_dir", "exports_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v: Path | None, info) -> Path | None:
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
        if self.deep_learning.checkpoint_dir:
            self.deep_learning.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance - lazy loaded
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, creating it if necessary."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(settings: Settings | None = None, **kwargs) -> Settings:
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
