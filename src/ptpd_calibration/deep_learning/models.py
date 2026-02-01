"""
Pydantic models for deep learning components.

Provides result models, input/output schemas, and data structures
for all AI/ML features with full validation.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptpd_calibration.deep_learning.types import (
    ComparisonResult,
    DefectSeverity,
    DefectType,
    DetectionConfidence,
    EnhancementMode,
    QualityLevel,
    RecipeCategory,
)

# =============================================================================
# Base Models
# =============================================================================


class BaseAIResult(BaseModel):
    """Base class for AI result models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    inference_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Inference time in milliseconds",
    )
    device_used: str = Field(
        default="cpu",
        description="Device used for inference",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version used",
    )


# =============================================================================
# Detection Models (YOLOv8 + SAM)
# =============================================================================


class DetectedPatch(BaseModel):
    """A detected step tablet patch with segmentation mask."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(..., ge=0, description="Patch index (0 = lightest)")
    bbox: tuple[int, int, int, int] = Field(..., description="Bounding box (x, y, width, height)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    mask: np.ndarray | None = Field(default=None, description="Segmentation mask")
    mask_area: int = Field(default=0, ge=0, description="Mask area in pixels")
    centroid: tuple[float, float] = Field(default=(0.0, 0.0), description="Centroid (x, y)")

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: Any) -> tuple[int, int, int, int]:
        """Validate and convert bounding box."""
        if isinstance(v, (list, np.ndarray)):
            return tuple(int(x) for x in v[:4])
        return v


class DeepDetectionResult(BaseAIResult):
    """Result from deep learning step tablet detection."""

    # Detection info
    tablet_bbox: tuple[int, int, int, int] = Field(..., description="Tablet bounding box")
    tablet_confidence: float = Field(..., ge=0.0, le=1.0, description="Tablet detection confidence")
    rotation_angle: float = Field(default=0.0, description="Detected rotation angle in degrees")
    orientation: str = Field(default="horizontal", description="Tablet orientation")

    # Patches
    patches: list[DetectedPatch] = Field(default_factory=list, description="Detected patches")
    num_patches: int = Field(default=0, ge=0, description="Number of detected patches")

    # Quality metrics
    detection_quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall detection quality"
    )
    confidence_level: DetectionConfidence = Field(
        default=DetectionConfidence.HIGH, description="Confidence level"
    )

    # Fallback info
    used_fallback: bool = Field(default=False, description="Whether classical fallback was used")
    fallback_reason: str | None = Field(default=None, description="Reason for fallback")

    # Warnings
    warnings: list[str] = Field(default_factory=list, description="Detection warnings")

    def get_patch_bounds(self) -> list[tuple[int, int, int, int]]:
        """Get list of patch bounding boxes."""
        return [p.bbox for p in self.patches]


# =============================================================================
# Image Quality Assessment Models
# =============================================================================


class ZoneQualityScore(BaseModel):
    """Quality score for a specific tonal zone."""

    zone: int = Field(..., ge=0, le=10, description="Zone number (0-10)")
    zone_name: str = Field(..., description="Zone name")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    pixel_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of pixels in zone"
    )
    detail_preservation: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detail preservation score"
    )
    issues: list[str] = Field(default_factory=list, description="Zone-specific issues")


class ImageQualityResult(BaseAIResult):
    """Result from Vision Transformer image quality assessment."""

    # Overall quality
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    quality_level: QualityLevel = Field(..., description="Quality classification")

    # Individual metrics
    metric_scores: dict[str, float] = Field(
        default_factory=dict, description="Individual metric scores"
    )
    primary_metric_name: str = Field(default="maniqa", description="Primary metric used")
    primary_metric_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Primary metric score"
    )

    # Zone analysis
    zone_scores: list[ZoneQualityScore] = Field(
        default_factory=list, description="Per-zone quality scores"
    )
    highlight_quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Highlight zone quality"
    )
    midtone_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Midtone zone quality")
    shadow_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Shadow zone quality")

    # Technical metrics
    sharpness: float = Field(default=1.0, ge=0.0, le=1.0, description="Sharpness score")
    noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Noise level")
    dynamic_range: float = Field(default=0.0, ge=0.0, description="Dynamic range")
    contrast: float = Field(default=1.0, ge=0.0, le=2.0, description="Contrast level")

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )
    issues: list[str] = Field(default_factory=list, description="Detected issues")

    # Embeddings (for comparison)
    embedding: list[float] | None = Field(
        default=None, description="Image embedding for comparison"
    )


# =============================================================================
# Diffusion Enhancement Models
# =============================================================================


class EnhancementRegion(BaseModel):
    """A region targeted for enhancement."""

    bbox: tuple[int, int, int, int] = Field(..., description="Region bounding box")
    mask: np.ndarray | None = Field(default=None, description="Region mask")
    enhancement_type: EnhancementMode = Field(..., description="Type of enhancement applied")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Enhancement strength")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DiffusionEnhancementResult(BaseAIResult):
    """Result from diffusion model enhancement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Enhanced image
    enhanced_image: np.ndarray | None = Field(default=None, description="Enhanced image array")
    original_size: tuple[int, int] = Field(..., description="Original image size (width, height)")
    output_size: tuple[int, int] = Field(..., description="Output image size (width, height)")

    # Enhancement info
    enhancement_mode: EnhancementMode = Field(..., description="Enhancement mode used")
    regions_enhanced: list[EnhancementRegion] = Field(
        default_factory=list, description="Enhanced regions"
    )
    num_inference_steps: int = Field(default=30, ge=1, description="Inference steps used")

    # Quality metrics
    quality_improvement: float = Field(default=0.0, description="Quality improvement percentage")
    structure_preservation: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Structure preservation score"
    )
    tone_fidelity: float = Field(default=1.0, ge=0.0, le=1.0, description="Tonal fidelity score")

    # Prompts used
    prompt_used: str | None = Field(default=None, description="Generation prompt used")
    negative_prompt: str | None = Field(default=None, description="Negative prompt used")

    # Warnings
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# Neural Curve Prediction Models
# =============================================================================


class CurvePredictionResult(BaseAIResult):
    """Result from neural curve prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Predicted curve
    input_values: list[float] = Field(..., min_length=2, description="Input values (0-1)")
    output_values: list[float] = Field(..., min_length=2, description="Output values (0-1)")
    num_points: int = Field(..., ge=2, description="Number of curve points")

    # Uncertainty
    uncertainty: list[float] | None = Field(default=None, description="Per-point uncertainty")
    mean_uncertainty: float = Field(default=0.0, ge=0.0, description="Mean uncertainty")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence")

    # Curve properties
    is_monotonic: bool = Field(default=True, description="Curve is monotonic")
    max_slope: float = Field(default=1.0, ge=0.0, description="Maximum slope")
    min_slope: float = Field(default=0.0, description="Minimum slope")

    # Conditioning used
    conditioning_factors: dict[str, Any] = Field(
        default_factory=dict, description="Conditioning factors used"
    )

    # Comparison with baseline
    baseline_mae: float | None = Field(default=None, description="MAE vs baseline/target")

    @field_validator("input_values", "output_values", mode="before")
    @classmethod
    def convert_to_list(cls, v: Any) -> list[float]:
        """Convert numpy arrays to lists."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return np.array(self.input_values), np.array(self.output_values)


# =============================================================================
# UV Exposure Prediction Models
# =============================================================================


class UVExposurePrediction(BaseAIResult):
    """Result from neural UV exposure prediction."""

    # Prediction
    predicted_seconds: float = Field(..., ge=0.0, description="Predicted exposure time in seconds")
    predicted_minutes: float = Field(..., ge=0.0, description="Predicted exposure time in minutes")

    # Confidence interval
    lower_bound_seconds: float = Field(..., ge=0.0, description="Lower confidence bound")
    upper_bound_seconds: float = Field(..., ge=0.0, description="Upper confidence bound")
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence level")

    # Input factors
    input_factors: dict[str, Any] = Field(default_factory=dict, description="Input factors used")

    # Factor contributions
    factor_contributions: dict[str, float] = Field(
        default_factory=dict, description="Contribution of each factor"
    )
    base_exposure: float = Field(..., ge=0.0, description="Base exposure before adjustments")
    adjustment_factor: float = Field(default=1.0, ge=0.0, description="Total adjustment factor")

    # Recommendations
    recommendations: list[str] = Field(default_factory=list, description="Exposure recommendations")
    warnings: list[str] = Field(default_factory=list, description="Exposure warnings")

    def format_time(self) -> str:
        """Format predicted time as human-readable string."""
        if self.predicted_minutes < 1:
            return f"{self.predicted_seconds:.0f} seconds"
        mins = int(self.predicted_minutes)
        secs = int((self.predicted_minutes - mins) * 60)
        if secs == 0:
            return f"{mins} minutes"
        return f"{mins} min {secs} sec"


# =============================================================================
# Defect Detection Models
# =============================================================================


class DetectedDefect(BaseModel):
    """A detected print defect."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defect_type: DefectType = Field(..., description="Type of defect")
    severity: DefectSeverity = Field(..., description="Defect severity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: tuple[int, int, int, int] = Field(..., description="Bounding box")
    mask: np.ndarray | None = Field(default=None, description="Defect mask")
    area_pixels: int = Field(default=0, ge=0, description="Defect area in pixels")
    area_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Defect area percentage"
    )
    remediation: str | None = Field(default=None, description="Suggested remediation")


class DefectDetectionResult(BaseAIResult):
    """Result from defect detection."""

    # Defects
    defects: list[DetectedDefect] = Field(default_factory=list, description="Detected defects")
    num_defects: int = Field(default=0, ge=0, description="Number of defects")

    # Summary by type
    defects_by_type: dict[str, int] = Field(
        default_factory=dict, description="Defect counts by type"
    )
    defects_by_severity: dict[str, int] = Field(
        default_factory=dict, description="Defect counts by severity"
    )

    # Overall assessment
    overall_severity: DefectSeverity = Field(
        default=DefectSeverity.NEGLIGIBLE, description="Overall severity"
    )
    print_acceptable: bool = Field(default=True, description="Whether print is acceptable")

    # Segmentation mask
    full_mask: np.ndarray | None = Field(default=None, description="Full segmentation mask")
    defect_coverage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Total defect coverage percentage"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Remediation recommendations"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# Recipe Recommendation Models
# =============================================================================


class RecipeRecommendation(BaseModel):
    """A recommended recipe."""

    recipe_id: UUID = Field(..., description="Recipe ID")
    recipe_name: str = Field(..., description="Recipe name")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Recommendation rank")

    # Recipe details
    paper_type: str = Field(..., description="Paper type")
    chemistry_type: str = Field(..., description="Chemistry type")
    metal_ratio: float = Field(..., ge=0.0, le=1.0, description="Metal ratio")
    exposure_time: float = Field(..., ge=0.0, description="Exposure time")

    # Categories
    categories: list[RecipeCategory] = Field(default_factory=list, description="Recipe categories")

    # Explanation
    explanation: str | None = Field(default=None, description="Why this recipe was recommended")
    matching_factors: list[str] = Field(default_factory=list, description="Matching factors")


class RecipeRecommendationResult(BaseAIResult):
    """Result from recipe recommendation engine."""

    # Query info
    query_image_used: bool = Field(
        default=False, description="Whether image was used for recommendation"
    )
    query_parameters: dict[str, Any] = Field(default_factory=dict, description="Query parameters")

    # Recommendations
    recommendations: list[RecipeRecommendation] = Field(
        default_factory=list, description="Recommended recipes"
    )
    num_recommendations: int = Field(default=0, ge=0, description="Number of recommendations")

    # User preferences considered
    preferences_used: dict[str, Any] = Field(
        default_factory=dict, description="User preferences considered"
    )

    # Diversity metrics
    recommendation_diversity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Diversity of recommendations"
    )


# =============================================================================
# Print Comparison Models
# =============================================================================


class ZoneComparison(BaseModel):
    """Comparison for a specific tonal zone."""

    zone: str = Field(..., description="Zone name")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Zone similarity score")
    lpips_score: float = Field(default=0.0, ge=0.0, description="Zone LPIPS score")
    ssim_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Zone SSIM score")
    differences: list[str] = Field(default_factory=list, description="Notable differences")


class PrintComparisonResult(BaseAIResult):
    """Result from LPIPS print comparison."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Overall comparison
    overall_similarity: float = Field(..., ge=0.0, le=1.0, description="Overall similarity score")
    comparison_result: ComparisonResult = Field(..., description="Comparison classification")

    # Metric scores
    lpips_score: float = Field(default=0.0, ge=0.0, description="LPIPS distance")
    ssim_score: float = Field(default=1.0, ge=0.0, le=1.0, description="SSIM score")
    psnr_score: float = Field(default=0.0, ge=0.0, description="PSNR in dB")
    additional_metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional metric scores"
    )

    # Zone comparison
    zone_comparisons: list[ZoneComparison] = Field(
        default_factory=list, description="Per-zone comparisons"
    )

    # Difference visualization
    difference_map: np.ndarray | None = Field(default=None, description="Difference heatmap")
    attention_map: np.ndarray | None = Field(
        default=None, description="Attention-based difference map"
    )

    # Detailed differences
    major_differences: list[str] = Field(
        default_factory=list, description="Major differences found"
    )
    minor_differences: list[str] = Field(
        default_factory=list, description="Minor differences found"
    )

    # Recommendations
    adjustment_suggestions: list[dict[str, Any]] = Field(
        default_factory=list, description="Parameter adjustment suggestions"
    )


# =============================================================================
# Multi-Modal Assistant Models
# =============================================================================


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    tool_name: str = Field(..., description="Tool name")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="Tool input")
    tool_output: Any = Field(default=None, description="Tool output")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Execution time")
    success: bool = Field(default=True, description="Whether tool call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")


class ImageAnalysis(BaseModel):
    """Analysis of an image by the assistant."""

    image_index: int = Field(..., ge=0, description="Image index")
    description: str = Field(..., description="Image description")
    detected_issues: list[str] = Field(default_factory=list, description="Detected issues")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    extracted_data: dict[str, Any] = Field(
        default_factory=dict, description="Extracted data from image"
    )


class MultiModalResponse(BaseAIResult):
    """Response from multi-modal AI assistant."""

    # Response
    response_text: str = Field(..., description="Main response text")
    response_type: str = Field(
        default="chat", description="Response type: chat, analysis, workflow"
    )

    # Tool usage
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls made")
    num_tool_calls: int = Field(default=0, ge=0, description="Number of tool calls")

    # Image analysis
    image_analyses: list[ImageAnalysis] = Field(
        default_factory=list, description="Image analyses performed"
    )
    images_analyzed: int = Field(default=0, ge=0, description="Number of images analyzed")

    # RAG context
    rag_sources_used: list[str] = Field(default_factory=list, description="RAG sources retrieved")

    # Conversation
    conversation_id: UUID | None = Field(default=None, description="Conversation ID")
    turn_number: int = Field(default=1, ge=1, description="Conversation turn number")

    # Token usage
    input_tokens: int = Field(default=0, ge=0, description="Input tokens used")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")


# =============================================================================
# Federated Learning Models
# =============================================================================


class FederatedUpdate(BaseModel):
    """A federated learning update from a client."""

    client_id: str = Field(..., description="Client identifier")
    round_number: int = Field(..., ge=0, description="Training round number")
    num_samples: int = Field(..., ge=1, description="Number of local samples")
    local_loss: float = Field(..., ge=0.0, description="Local training loss")
    local_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, description="Local accuracy")
    training_time_seconds: float = Field(default=0.0, ge=0.0, description="Local training time")


class FederatedRoundResult(BaseAIResult):
    """Result from a federated learning round."""

    # Round info
    round_number: int = Field(..., ge=0, description="Round number")
    num_participants: int = Field(..., ge=1, description="Number of participants")

    # Aggregation
    aggregation_strategy: str = Field(..., description="Aggregation strategy used")
    global_loss: float = Field(..., ge=0.0, description="Global aggregated loss")
    global_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Global accuracy"
    )

    # Participant updates
    participant_updates: list[FederatedUpdate] = Field(
        default_factory=list, description="Individual participant updates"
    )

    # Privacy
    privacy_level: str = Field(..., description="Privacy level used")
    noise_added: bool = Field(default=False, description="Whether DP noise was added")

    # Model improvement
    loss_improvement: float = Field(default=0.0, description="Loss improvement from previous round")

    # Communication stats
    bytes_communicated: int = Field(default=0, ge=0, description="Total bytes communicated")
    compression_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Gradient compression ratio"
    )
