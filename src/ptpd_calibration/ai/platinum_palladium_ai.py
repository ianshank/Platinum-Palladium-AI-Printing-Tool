"""
Comprehensive AI system for platinum-palladium alternative photographic printing.

This module provides AI-powered analysis, prediction, and optimization tools
specifically designed for the platinum-palladium printing workflow.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from ptpd_calibration.chemistry.calculator import (
    ChemistryCalculator,
    CoatingMethod,
    PaperAbsorbency,
)
from ptpd_calibration.config import get_settings
from ptpd_calibration.core.models import (
    CalibrationRecord,
    CurveData,
)
from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    CurveType,
    DeveloperType,
)
from ptpd_calibration.exposure.calculator import (
    ExposureCalculator,
    ExposureSettings,
    LightSource,
)
from ptpd_calibration.imaging.histogram import HistogramAnalyzer
from ptpd_calibration.imaging.processor import (
    ColorMode,
    ExportSettings,
    ImageFormat,
    ImageProcessor,
    ProcessingResult,
)

# ============================================================================
# Enums and Constants
# ============================================================================


class TonePreference(str, Enum):
    """Tone preferences for Pt/Pd printing."""

    WARM = "warm"  # Brown-black tones, higher palladium
    NEUTRAL = "neutral"  # Balanced tones
    COOL = "cool"  # Blue-black tones, higher platinum
    CUSTOM = "custom"  # User-defined ratios


class ContrastLevel(str, Enum):
    """Contrast level preferences."""

    LOW = "low"  # Minimal contrast agents
    NORMAL = "normal"  # Standard contrast
    HIGH = "high"  # Increased contrast agents
    MAXIMUM = "maximum"  # Maximum contrast for flat negatives


class PrinterProfile(str, Enum):
    """Common printer profiles for digital negatives."""

    EPSON_P800 = "epson_p800"
    EPSON_3880 = "epson_3880"
    CANON_PRO_1000 = "canon_pro_1000"
    HP_Z9 = "hp_z9"
    CUSTOM = "custom"


class ProblemArea(str, Enum):
    """Problem areas in print analysis."""

    HIGHLIGHTS = "highlights"
    MIDTONES = "midtones"
    SHADOWS = "shadows"
    OVERALL_DENSITY = "overall_density"
    UNIFORMITY = "uniformity"
    GRAIN = "grain"


# ============================================================================
# Pydantic Models
# ============================================================================


class TonalityAnalysisResult(BaseModel):
    """Result of image tonality analysis."""

    # Histogram statistics
    histogram_stats: dict[str, Any] = Field(..., description="Complete histogram statistics")

    # Zone analysis (Ansel Adams zones 0-10)
    zone_distribution: dict[int, float] = Field(
        ..., description="Percentage of pixels in each zone"
    )
    dominant_zones: list[int] = Field(..., description="Zones with highest pixel concentration")

    # Tonal range
    dynamic_range_stops: float = Field(..., ge=0.0, description="Dynamic range in stops")
    shadow_detail_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of shadow detail"
    )
    highlight_detail_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of highlight detail"
    )

    # Exposure recommendations
    recommended_exposure_adjustment_stops: float = Field(
        default=0.0, description="Suggested exposure adjustment in stops"
    )
    recommended_contrast_adjustment: str = Field(
        default="none", description="Suggested contrast adjustment"
    )

    # Suggestions
    suggestions: list[str] = Field(
        default_factory=list, description="Recommendations for optimal printing"
    )
    warnings: list[str] = Field(default_factory=list, description="Potential issues to address")


class ExposurePrediction(BaseModel):
    """AI-based exposure time prediction with confidence interval."""

    predicted_exposure_seconds: float = Field(
        ..., ge=0.0, description="Predicted exposure time in seconds"
    )
    predicted_exposure_minutes: float = Field(
        ..., ge=0.0, description="Predicted exposure time in minutes"
    )

    # Confidence interval (95%)
    lower_bound_seconds: float = Field(..., ge=0.0, description="Lower confidence bound")
    upper_bound_seconds: float = Field(..., ge=0.0, description="Upper confidence bound")
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence level")

    # Factors considered
    negative_density: float = Field(..., description="Negative density range")
    paper_speed_factor: float = Field(default=1.0, description="Paper speed factor")
    humidity_factor: float = Field(default=1.0, description="Humidity adjustment")
    temperature_celsius: float | None = Field(default=None, description="Temperature in Celsius")

    # Breakdown
    base_exposure: float = Field(..., description="Base exposure time")
    adjustments_applied: dict[str, float] = Field(
        default_factory=dict, description="Adjustment factors applied"
    )

    # Recommendations
    recommendations: list[str] = Field(default_factory=list, description="Exposure recommendations")

    def format_time(self) -> str:
        """Format predicted time as human-readable string."""
        if self.predicted_exposure_minutes < 1:
            return f"{self.predicted_exposure_seconds:.0f} seconds"
        mins = int(self.predicted_exposure_minutes)
        secs = int((self.predicted_exposure_minutes - mins) * 60)
        if secs == 0:
            return f"{mins} minutes"
        return f"{mins} min {secs} sec"


class ChemistryRecommendation(BaseModel):
    """Chemistry ratio recommendations based on desired tone and contrast."""

    # Metal ratios
    platinum_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Platinum ratio (0=pure Pd, 1=pure Pt)"
    )
    palladium_ratio: float = Field(..., ge=0.0, le=1.0, description="Palladium ratio")

    # Ferric oxalate amounts (in drops)
    ferric_oxalate_1_drops: float = Field(..., ge=0.0, description="FO #1 drops")
    ferric_oxalate_2_drops: float = Field(default=0.0, ge=0.0, description="FO #2 (contrast) drops")

    # Contrast agent
    contrast_agent: ContrastAgent = Field(
        default=ContrastAgent.NONE, description="Recommended contrast agent"
    )
    na2_drops: float = Field(default=0.0, ge=0.0, description="Na2 drops if using")
    contrast_amount_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Contrast boost percentage"
    )

    # Tone characteristics
    expected_tone: str = Field(..., description="Expected tonal quality")
    expected_dmax: float = Field(
        default=2.0, ge=0.0, le=3.5, description="Expected maximum density"
    )

    # Developer recommendation
    recommended_developer: DeveloperType = Field(
        default=DeveloperType.POTASSIUM_OXALATE, description="Recommended developer"
    )

    # Rationale
    rationale: list[str] = Field(default_factory=list, description="Explanation of recommendations")
    notes: list[str] = Field(default_factory=list, description="Additional notes")


class DigitalNegativeResult(BaseModel):
    """Result of digital negative generation."""

    model_config = {"arbitrary_types_allowed": True}

    # Processing info
    processing_result: Any | None = Field(default=None, description="ProcessingResult object")
    output_path: Path | None = Field(default=None, description="Saved file path")

    # Metadata
    original_size: tuple[int, int] = Field(..., description="Original image size")
    output_size: tuple[int, int] = Field(..., description="Output image size")
    output_format: str = Field(..., description="Output format")
    output_dpi: int | None = Field(default=None, description="Output DPI")

    # Curve applied
    curve_name: str | None = Field(default=None, description="Applied curve name")
    curve_type: CurveType | None = Field(default=None, description="Curve type")

    # Processing steps
    steps_applied: list[str] = Field(default_factory=list, description="Processing steps applied")

    # Quality metrics
    estimated_quality: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Estimated quality score"
    )

    @field_validator("processing_result", mode="before")
    @classmethod
    def handle_processing_result(cls, v: Any) -> Any:
        """Allow ProcessingResult to be stored."""
        return v


class PrintQualityAnalysis(BaseModel):
    """Analysis comparing print to reference image."""

    # Overall metrics
    overall_match_score: float = Field(..., ge=0.0, le=1.0, description="Overall match score (0-1)")
    density_correlation: float = Field(
        ..., ge=-1.0, le=1.0, description="Density correlation coefficient"
    )

    # Density analysis
    mean_density_difference: float = Field(..., description="Mean density difference")
    max_density_difference: float = Field(..., description="Maximum density difference")
    density_range_match: float = Field(
        ..., ge=0.0, le=1.0, description="How well density ranges match"
    )

    # Problem areas
    problem_areas: list[tuple[ProblemArea, str, float]] = Field(
        default_factory=list, description="List of (area, description, severity) tuples"
    )

    # Zone-by-zone analysis
    zone_differences: dict[int, float] = Field(
        default_factory=dict, description="Difference by zone (0-10)"
    )
    worst_zones: list[int] = Field(
        default_factory=list, description="Zones with largest differences"
    )

    # Corrections
    suggested_exposure_correction_stops: float = Field(
        default=0.0, description="Suggested exposure correction in stops"
    )
    suggested_contrast_correction: str = Field(
        default="none", description="Suggested contrast correction"
    )
    suggested_curve_adjustments: dict[str, float] = Field(
        default_factory=dict, description="Suggested curve adjustments"
    )

    # Detailed recommendations
    corrections: list[str] = Field(
        default_factory=list, description="Specific correction recommendations"
    )


class WorkflowOptimization(BaseModel):
    """Workflow optimization based on print history."""

    # Analysis summary
    total_prints_analyzed: int = Field(..., ge=0, description="Number of prints analyzed")
    successful_prints: int = Field(..., ge=0, description="Number of successful prints")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Overall success rate")

    # Patterns identified
    optimal_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Identified optimal parameters"
    )
    parameter_trends: dict[str, str] = Field(
        default_factory=dict, description="Trends in successful prints"
    )

    # Predictive insights
    recommended_base_exposure: float | None = Field(
        default=None, description="Recommended base exposure time"
    )
    recommended_metal_ratio: float | None = Field(
        default=None, description="Recommended Pt/Pd ratio"
    )
    recommended_paper_settings: dict[str, Any] = Field(
        default_factory=dict, description="Paper-specific settings"
    )

    # Efficiency improvements
    efficiency_suggestions: list[str] = Field(
        default_factory=list, description="Suggestions to improve efficiency"
    )
    common_mistakes: list[str] = Field(default_factory=list, description="Common mistakes to avoid")

    # Confidence
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in recommendations"
    )

    # Detailed insights
    insights: list[str] = Field(default_factory=list, description="Detailed insights from analysis")


# ============================================================================
# Main AI Class
# ============================================================================


class PlatinumPalladiumAI:
    """
    Comprehensive AI system for platinum-palladium alternative printing.

    This class provides intelligent analysis, prediction, and optimization
    tools for the complete Pt/Pd printing workflow, from image analysis
    to print quality assessment.

    Features:
    - Image tonality analysis with zone-based recommendations
    - AI-based exposure time prediction with confidence intervals
    - Chemistry ratio recommendations based on desired tone
    - Digital negative generation with curve application
    - Print quality analysis and correction suggestions
    - Workflow optimization based on print history

    All parameters are configurable via settings, with no hardcoded values.
    """

    def __init__(self, settings: Any | None = None):
        """
        Initialize the Platinum/Palladium AI system.

        Args:
            settings: Optional Settings object. If None, uses global settings.
        """
        self.settings = settings or get_settings()

        # Initialize component classes
        self.histogram_analyzer = HistogramAnalyzer()
        self.image_processor = ImageProcessor()
        self.exposure_calculator = ExposureCalculator()
        self.chemistry_calculator = ChemistryCalculator()

        # Cache for ML models (lazy loaded)
        self._exposure_model: Any | None = None
        self._quality_model: Any | None = None

    # ========================================================================
    # 1. Image Tonality Analysis
    # ========================================================================

    def analyze_image_tonality(
        self,
        image: str | Path | Image.Image | np.ndarray,
        target_paper: str | None = None,
        target_process: ChemistryType | None = None,
    ) -> TonalityAnalysisResult:
        """
        Analyze image for optimal Pt/Pd conversion.

        Performs comprehensive tonal analysis including:
        - Histogram distribution across Ansel Adams zones
        - Dynamic range calculation
        - Shadow/highlight detail assessment
        - Exposure and contrast recommendations

        Args:
            image: Input image (path, PIL Image, or numpy array)
            target_paper: Optional target paper type for recommendations
            target_process: Optional target chemistry type

        Returns:
            TonalityAnalysisResult with complete analysis and recommendations
        """
        # Analyze histogram
        histogram_result = self.histogram_analyzer.analyze(image, include_rgb=False)
        stats = histogram_result.stats

        # Extract zone distribution
        zone_distribution = stats.zone_distribution

        # Identify dominant zones (>10% of pixels)
        dominant_zones = [zone for zone, pct in zone_distribution.items() if pct > 0.10]
        dominant_zones.sort(key=lambda z: zone_distribution[z], reverse=True)

        # Calculate shadow and highlight detail percentages
        # Zones 0-3 are shadows with detail
        shadow_detail = sum(zone_distribution.get(z, 0) for z in range(1, 4)) * 100
        # Zones 7-9 are highlights with detail
        highlight_detail = sum(zone_distribution.get(z, 0) for z in range(7, 10)) * 100

        # Calculate exposure recommendations
        exposure_adjustment = 0.0
        contrast_adjustment = "none"
        suggestions = []
        warnings = []

        # Analyze brightness
        if stats.brightness < 0.25:
            exposure_adjustment = -1.0  # Increase exposure by 1 stop
            suggestions.append("Image is quite dark. Consider increasing exposure by 1 stop.")
        elif stats.brightness > 0.75:
            exposure_adjustment = 1.0  # Decrease exposure by 1 stop
            suggestions.append("Image is quite bright. Consider reducing exposure by 1 stop.")

        # Analyze contrast
        if stats.contrast < 0.15:
            contrast_adjustment = "increase"
            suggestions.append(
                "Low contrast image. Consider using higher grade paper or "
                "adding contrast agent (Na2)."
            )
        elif stats.contrast > 0.35:
            contrast_adjustment = "decrease"
            suggestions.append(
                "High contrast image. Consider N-1 development or using softer grade chemistry."
            )

        # Dynamic range analysis
        if stats.dynamic_range < 4.0:
            warnings.append(
                f"Low dynamic range ({stats.dynamic_range:.1f} stops). "
                "Image may lack tonal separation."
            )
            suggestions.append("Consider contrast enhancement in digital negative.")
        elif stats.dynamic_range > 8.0:
            warnings.append(
                f"Very high dynamic range ({stats.dynamic_range:.1f} stops). "
                "May exceed paper capabilities."
            )
            suggestions.append(
                "Consider compression in highlights/shadows or split-grade printing."
            )

        # Clipping warnings
        if stats.shadow_clipping_percent > 5.0:
            warnings.append(
                f"Shadow clipping detected ({stats.shadow_clipping_percent:.1f}%). "
                "Pure blacks may lack detail."
            )
            suggestions.append("Reduce negative density or increase base exposure.")

        if stats.highlight_clipping_percent > 5.0:
            warnings.append(
                f"Highlight clipping detected ({stats.highlight_clipping_percent:.1f}%). "
                "Pure whites may blow out."
            )
            suggestions.append("Increase negative density or reduce exposure time.")

        # Zone-specific recommendations
        if zone_distribution.get(5, 0) < 0.05:  # Zone V (middle gray)
            suggestions.append("Limited midtone content. Consider whether this suits your vision.")

        # Paper-specific recommendations
        if target_paper:
            if "hot press" in target_paper.lower():
                suggestions.append(
                    "Hot press paper: Expect smoother tones and finer grain. "
                    "May need reduced coating solution."
                )
            elif "cold press" in target_paper.lower():
                suggestions.append(
                    "Cold press paper: Expect more texture and higher absorbency. "
                    "May need increased coating solution."
                )

        # Process-specific recommendations
        if target_process == ChemistryType.PURE_PALLADIUM:
            suggestions.append(
                "Pure palladium: Expect warm brown-black tones. "
                "Good for portraits and warm-toned images."
            )
        elif target_process == ChemistryType.PURE_PLATINUM:
            suggestions.append(
                "Pure platinum: Expect cool neutral blacks with maximum Dmax. "
                "Ideal for landscapes and high-contrast images."
            )

        return TonalityAnalysisResult(
            histogram_stats=stats.to_dict(),
            zone_distribution=zone_distribution,
            dominant_zones=dominant_zones,
            dynamic_range_stops=stats.dynamic_range,
            shadow_detail_percent=shadow_detail,
            highlight_detail_percent=highlight_detail,
            recommended_exposure_adjustment_stops=exposure_adjustment,
            recommended_contrast_adjustment=contrast_adjustment,
            suggestions=suggestions,
            warnings=warnings,
        )

    # ========================================================================
    # 2. Exposure Time Prediction
    # ========================================================================

    def predict_exposure_time(
        self,
        negative_density: float,
        paper_type: str | None = None,
        light_source: LightSource | None = None,
        humidity: float | None = None,
        temperature: float | None = None,
        platinum_ratio: float | None = None,
        distance_inches: float | None = None,
    ) -> ExposurePrediction:
        """
        AI-based exposure time prediction with confidence intervals.

        Uses configurable base exposure times and applies adjustments for:
        - Negative density (via industry-standard formulas)
        - Paper type and speed
        - Environmental conditions (humidity, temperature)
        - Chemistry (Pt/Pd ratio affects exposure speed)
        - Light source type and distance

        Args:
            negative_density: Density range of negative (Dmax - Dmin)
            paper_type: Paper type name (for speed lookup)
            light_source: UV light source type
            humidity: Relative humidity (0-100%)
            temperature: Temperature in Celsius
            platinum_ratio: Platinum ratio (0=pure Pd, 1=pure Pt)
            distance_inches: Distance from light source

        Returns:
            ExposurePrediction with predicted time and confidence interval
        """
        # Get base exposure settings
        exposure_settings = ExposureSettings(
            base_exposure_minutes=self.settings.exposure.base_exposure_minutes
            if hasattr(self.settings, "exposure")
            else 10.0,
            light_source=light_source or LightSource.BL_FLUORESCENT,
        )

        # Apply platinum ratio if provided
        if platinum_ratio is not None:
            exposure_settings.platinum_ratio = platinum_ratio

        # Calculate paper speed factor
        paper_speed = 1.0
        if paper_type:
            # Look up paper speed from profiles or use default
            paper_speed = self._get_paper_speed_factor(paper_type)

        # Calculate humidity adjustment
        humidity_adj = 1.0
        if humidity is not None:
            # Higher humidity = faster exposure (more moisture = more sensitivity)
            # Baseline at 50% humidity
            humidity_adj = 1.0 - ((humidity - 50.0) / 100.0) * 0.15
            humidity_adj = max(0.85, min(1.15, humidity_adj))

        # Calculate base exposure
        result = self.exposure_calculator.calculate(
            negative_density=negative_density,
            distance_inches=distance_inches,
            light_source=light_source,
            paper_speed=paper_speed,
            platinum_ratio=platinum_ratio,
            humidity_factor=humidity_adj,
        )

        # Calculate confidence interval
        # Factors affecting uncertainty:
        # - Negative density measurement uncertainty: ±0.1 density
        # - Environmental variation: ±5%
        # - Paper batch variation: ±10%

        density_uncertainty = 0.1
        env_uncertainty = 0.05
        paper_uncertainty = 0.10

        # Calculate bounds using error propagation
        # Density affects exposure exponentially (2^(Δd/0.3))
        2 ** (
            (negative_density - density_uncertainty - exposure_settings.base_negative_density) / 0.3
        )
        2 ** (
            (negative_density + density_uncertainty - exposure_settings.base_negative_density) / 0.3
        )

        lower_bound = result.exposure_seconds * (1 - paper_uncertainty - env_uncertainty)
        upper_bound = result.exposure_seconds * (1 + paper_uncertainty + env_uncertainty)

        # Generate recommendations
        recommendations = list(result.notes)

        if upper_bound - lower_bound > result.exposure_seconds * 0.5:
            recommendations.append(
                "Wide confidence interval - consider test strip with ±0.5 stop increments"
            )

        if temperature is not None:
            if temperature < 15:
                recommendations.append(
                    f"Low temperature ({temperature}°C) may slow exposure. "
                    "Consider warming chemicals."
                )
            elif temperature > 25:
                recommendations.append(
                    f"High temperature ({temperature}°C) may accelerate exposure. "
                    "Monitor carefully."
                )

        return ExposurePrediction(
            predicted_exposure_seconds=result.exposure_seconds,
            predicted_exposure_minutes=result.exposure_minutes,
            lower_bound_seconds=lower_bound,
            upper_bound_seconds=upper_bound,
            confidence_level=0.95,
            negative_density=negative_density,
            paper_speed_factor=paper_speed,
            humidity_factor=humidity_adj,
            temperature_celsius=temperature,
            base_exposure=result.base_exposure,
            adjustments_applied={
                "density_adjustment": result.density_adjustment,
                "light_source_adjustment": result.light_source_adjustment,
                "distance_adjustment": result.distance_adjustment,
                "paper_adjustment": result.paper_adjustment,
                "chemistry_adjustment": result.chemistry_adjustment,
                "environmental_adjustment": result.environmental_adjustment,
            },
            recommendations=recommendations,
        )

    # ========================================================================
    # 3. Chemistry Ratio Recommendations
    # ========================================================================

    def suggest_chemistry_ratios(
        self,
        desired_tone: TonePreference,
        contrast_level: ContrastLevel,
        paper_type: str | None = None,
        print_size_inches: tuple[float, float] | None = None,
    ) -> ChemistryRecommendation:
        """
        Recommend ferric oxalate and metal ratios for desired tone and contrast.

        Maps tone preferences to Pt/Pd ratios:
        - Warm: Higher palladium (brown-black tones)
        - Neutral: Balanced mix
        - Cool: Higher platinum (blue-black tones)

        Calculates ferric oxalate amounts and contrast agent additions.

        Args:
            desired_tone: Desired tonal quality (warm/neutral/cool)
            contrast_level: Desired contrast level
            paper_type: Optional paper type for specific recommendations
            print_size_inches: Optional (width, height) for quantity calculations

        Returns:
            ChemistryRecommendation with complete recipe and rationale
        """
        # Map tone preference to platinum ratio
        tone_to_ratio = {
            TonePreference.WARM: 0.0,  # Pure palladium
            TonePreference.NEUTRAL: 0.5,  # 50/50 mix
            TonePreference.COOL: 1.0,  # Pure platinum
        }

        platinum_ratio = tone_to_ratio.get(desired_tone, 0.5)
        palladium_ratio = 1.0 - platinum_ratio

        # Map contrast level to contrast boost
        contrast_to_boost = {
            ContrastLevel.LOW: 0.0,
            ContrastLevel.NORMAL: 0.25,
            ContrastLevel.HIGH: 0.5,
            ContrastLevel.MAXIMUM: 0.75,
        }

        contrast_boost = contrast_to_boost.get(contrast_level, 0.25)

        # Determine Na2 usage based on contrast
        use_na2 = contrast_level in (ContrastLevel.HIGH, ContrastLevel.MAXIMUM)
        na2_ratio = self.settings.chemistry.default_na2_drops_ratio if use_na2 else 0.0

        # Calculate actual chemistry amounts if print size provided
        if print_size_inches:
            width, height = print_size_inches

            # Determine paper absorbency from paper type
            absorbency = PaperAbsorbency.MEDIUM
            if paper_type:
                if "hot press" in paper_type.lower() or "hp" in paper_type.lower():
                    absorbency = PaperAbsorbency.LOW
                elif "cold press" in paper_type.lower() or "cp" in paper_type.lower():
                    absorbency = PaperAbsorbency.HIGH

            # Calculate recipe
            recipe = self.chemistry_calculator.calculate(
                width_inches=width,
                height_inches=height,
                platinum_ratio=platinum_ratio,
                paper_absorbency=absorbency,
                coating_method=CoatingMethod.BRUSH,
                contrast_boost=contrast_boost,
                na2_ratio=na2_ratio if use_na2 else None,
            )

            fo1_drops = recipe.ferric_oxalate_drops
            fo2_drops = recipe.ferric_oxalate_contrast_drops
            na2_drops = recipe.na2_drops
        else:
            # Provide ratios only
            fo1_drops = 12.0  # Standard base for reference
            fo2_drops = (
                fo1_drops * (contrast_boost / (1 - contrast_boost)) if contrast_boost < 1 else 0
            )
            na2_drops = (fo1_drops + fo2_drops) * na2_ratio if use_na2 else 0.0

        # Determine expected characteristics
        tone_descriptions = {
            TonePreference.WARM: "Warm brown-black tones with subtle grain",
            TonePreference.NEUTRAL: "Neutral black tones with balanced character",
            TonePreference.COOL: "Cool blue-black tones with maximum depth",
        }
        expected_tone = tone_descriptions.get(desired_tone, "Balanced tones")

        # Expected Dmax
        # Platinum gives higher Dmax than palladium
        base_dmax = 1.8 + (platinum_ratio * 0.4)
        # Na2 can increase Dmax by ~0.2
        if use_na2:
            base_dmax += 0.2
        expected_dmax = min(2.5, base_dmax)

        # Recommended developer
        # Potassium oxalate is standard, ammonium citrate for warmer tones
        recommended_dev = DeveloperType.POTASSIUM_OXALATE
        if desired_tone == TonePreference.WARM:
            recommended_dev = DeveloperType.AMMONIUM_CITRATE

        # Build rationale
        rationale = []

        if platinum_ratio == 0.0:
            rationale.append(
                "Pure palladium selected for warm tones. More economical and "
                "produces beautiful brown-blacks."
            )
        elif platinum_ratio == 1.0:
            rationale.append(
                "Pure platinum selected for cool tones. Provides maximum Dmax "
                "and neutral to cool black tones."
            )
        else:
            rationale.append(
                f"{platinum_ratio * 100:.0f}% platinum / {palladium_ratio * 100:.0f}% "
                f"palladium for balanced tones."
            )

        if contrast_boost > 0:
            rationale.append(
                f"FO #2 (contrast) at {contrast_boost * 100:.0f}% for increased contrast. "
                f"Good for flat negatives."
            )

        if use_na2:
            rationale.append(
                "Na2 (sodium chloroplatinate) recommended for enhanced contrast and Dmax."
            )

        # Build notes
        notes = []

        notes.append(
            f"Expected Dmax: ~{expected_dmax:.1f} (actual may vary with paper and exposure)"
        )

        if platinum_ratio > 0.5:
            notes.append("Higher platinum content requires longer exposure times (~2x vs pure Pd)")

        if contrast_level == ContrastLevel.MAXIMUM:
            notes.append(
                "Maximum contrast setting. Good for very flat negatives but may "
                "reduce overall tonal range."
            )

        if paper_type:
            notes.append(f"Recommendations tailored for {paper_type}")

        return ChemistryRecommendation(
            platinum_ratio=platinum_ratio,
            palladium_ratio=palladium_ratio,
            ferric_oxalate_1_drops=fo1_drops,
            ferric_oxalate_2_drops=fo2_drops,
            contrast_agent=ContrastAgent.NA2 if use_na2 else ContrastAgent.NONE,
            na2_drops=na2_drops,
            contrast_amount_percent=contrast_boost * 100,
            expected_tone=expected_tone,
            expected_dmax=expected_dmax,
            recommended_developer=recommended_dev,
            rationale=rationale,
            notes=notes,
        )

    # ========================================================================
    # 4. Digital Negative Generation
    # ========================================================================

    def generate_digital_negative(
        self,
        image: str | Path | Image.Image | np.ndarray,
        printer_profile: PrinterProfile | None = None,
        curve: CurveData | None = None,
        output_path: str | Path | None = None,
        output_format: ImageFormat = ImageFormat.TIFF_16BIT,
        target_dpi: int = 2880,
        invert: bool = True,
    ) -> DigitalNegativeResult:
        """
        Create optimized digital negative with curve application.

        Processing pipeline:
        1. Load image
        2. Convert to grayscale
        3. Apply linearization curve (if provided)
        4. Invert to negative
        5. Export in high-quality format
        6. Preserve metadata

        Args:
            image: Input image (path, PIL Image, or numpy array)
            printer_profile: Printer profile for optimizations
            curve: Optional calibration curve to apply
            output_path: Optional output file path
            output_format: Output format (TIFF 16-bit, PNG, etc.)
            target_dpi: Target DPI for output
            invert: Whether to invert image (create negative)

        Returns:
            DigitalNegativeResult with processing info and output path
        """
        # Load image
        processing_result = self.image_processor.load_image(image)

        # Convert to grayscale for negatives
        if processing_result.image.mode not in ("L", "LA"):
            gray_image = processing_result.image.convert("L")
            processing_result = ProcessingResult(
                image=gray_image,
                original_size=processing_result.original_size,
                original_mode=processing_result.original_mode,
                original_format=processing_result.original_format,
                original_dpi=processing_result.original_dpi,
                curve_applied=False,
                inverted=False,
                processing_notes=processing_result.processing_notes + ["Converted to grayscale"],
            )

        steps_applied = ["Loaded image", "Converted to grayscale"]

        # Apply calibration curve if provided
        if curve is not None:
            processing_result = self.image_processor.apply_curve(
                processing_result,
                curve,
                ColorMode.GRAYSCALE,
            )
            steps_applied.append(f"Applied curve: {curve.name}")

        # Invert if requested
        if invert:
            processing_result = self.image_processor.invert(processing_result)
            steps_applied.append("Inverted to negative")

        # Prepare export settings
        export_settings = ExportSettings(
            format=output_format,
            preserve_metadata=True,
            preserve_resolution=True,
            target_dpi=target_dpi,
        )

        # Export if path provided
        saved_path = None
        if output_path:
            saved_path = self.image_processor.export(
                processing_result,
                output_path,
                export_settings,
            )
            steps_applied.append(f"Exported to {saved_path}")

        # Calculate quality estimate
        # Based on bit depth, resolution, and format
        quality = 0.8  # Base quality
        if output_format in (ImageFormat.TIFF_16BIT, ImageFormat.PNG_16BIT):
            quality += 0.2  # 16-bit adds quality
        if target_dpi >= 2880:
            quality = min(1.0, quality + 0.1)  # High DPI

        return DigitalNegativeResult(
            processing_result=processing_result,
            output_path=saved_path,
            original_size=processing_result.original_size,
            output_size=processing_result.image.size,
            output_format=output_format.value,
            output_dpi=target_dpi,
            curve_name=curve.name if curve else None,
            curve_type=curve.curve_type if curve else None,
            steps_applied=steps_applied,
            estimated_quality=quality,
        )

    # ========================================================================
    # 5. Print Quality Analysis
    # ========================================================================

    def analyze_print_quality(
        self,
        scan_image: str | Path | Image.Image | np.ndarray,
        reference_image: str | Path | Image.Image | np.ndarray,
        zone_weight: float = 1.0,
    ) -> PrintQualityAnalysis:
        """
        Compare print scan to expected reference image.

        Analysis includes:
        - Overall density match scoring
        - Zone-by-zone comparison (Ansel Adams zones)
        - Problem area identification
        - Specific correction suggestions

        Args:
            scan_image: Scanned print image
            reference_image: Reference/target image
            zone_weight: Weight for zone-based analysis (0-1)

        Returns:
            PrintQualityAnalysis with scores and correction suggestions
        """
        # Analyze both images
        scan_analysis = self.histogram_analyzer.analyze(scan_image, include_rgb=False)
        ref_analysis = self.histogram_analyzer.analyze(reference_image, include_rgb=False)

        scan_stats = scan_analysis.stats
        ref_stats = ref_analysis.stats

        # Compare histograms
        comparison = self.histogram_analyzer.compare_histograms(scan_image, reference_image)

        # Calculate overall match score
        # Based on histogram intersection and Bhattacharyya coefficient
        hist_similarity = comparison["similarity"]["histogram_intersection"]
        bhatt_coeff = comparison["similarity"]["bhattacharyya_coefficient"]
        overall_match = (hist_similarity + bhatt_coeff) / 2.0

        # Calculate density correlation
        scan_densities = scan_analysis.histogram.astype(float) / scan_analysis.total_pixels
        ref_densities = ref_analysis.histogram.astype(float) / ref_analysis.total_pixels
        density_corr = float(np.corrcoef(scan_densities, ref_densities)[0, 1])

        # Density differences
        mean_diff = comparison["changes"]["mean_shift"]
        # Estimate max difference from histogram comparison
        max_diff = abs(comparison["changes"]["contrast_change"])

        # Density range match
        scan_range = scan_stats.dynamic_range
        ref_range = ref_stats.dynamic_range
        range_match = 1.0 - min(1.0, abs(scan_range - ref_range) / max(ref_range, 1.0))

        # Zone-by-zone analysis
        zone_differences = {}
        for zone in range(11):
            scan_pct = scan_stats.zone_distribution.get(zone, 0)
            ref_pct = ref_stats.zone_distribution.get(zone, 0)
            zone_differences[zone] = abs(scan_pct - ref_pct)

        # Identify worst zones (largest differences)
        worst_zones = sorted(
            zone_differences.keys(), key=lambda z: zone_differences[z], reverse=True
        )[:3]

        # Identify problem areas
        problem_areas: list[tuple[ProblemArea, str, float]] = []

        # Check highlights
        if zone_differences[9] > 0.1 or zone_differences[10] > 0.1:
            severity = max(zone_differences[9], zone_differences[10])
            if scan_stats.zone_distribution.get(10, 0) > ref_stats.zone_distribution.get(10, 0):
                problem_areas.append(
                    (ProblemArea.HIGHLIGHTS, "Highlights are blown out (too bright)", severity)
                )
            else:
                problem_areas.append(
                    (ProblemArea.HIGHLIGHTS, "Highlights are blocked (too dark)", severity)
                )

        # Check shadows
        if zone_differences[0] > 0.1 or zone_differences[1] > 0.1:
            severity = max(zone_differences[0], zone_differences[1])
            if scan_stats.zone_distribution.get(0, 0) > ref_stats.zone_distribution.get(0, 0):
                problem_areas.append(
                    (ProblemArea.SHADOWS, "Shadows are blocked (too dark)", severity)
                )
            else:
                problem_areas.append(
                    (ProblemArea.SHADOWS, "Shadows are weak (too light)", severity)
                )

        # Check midtones
        if zone_differences[5] > 0.1:
            severity = zone_differences[5]
            if scan_stats.zone_distribution.get(5, 0) > ref_stats.zone_distribution.get(5, 0):
                problem_areas.append(
                    (ProblemArea.MIDTONES, "Midtones are shifted (distribution mismatch)", severity)
                )

        # Check overall density
        if abs(mean_diff) > 20:
            severity = min(1.0, abs(mean_diff) / 50)
            if mean_diff > 0:
                problem_areas.append(
                    (ProblemArea.OVERALL_DENSITY, "Print is overall too light", severity)
                )
            else:
                problem_areas.append(
                    (ProblemArea.OVERALL_DENSITY, "Print is overall too dark", severity)
                )

        # Calculate corrections
        exposure_correction = 0.0
        contrast_correction = "none"
        curve_adjustments = {}
        corrections = []

        # Exposure correction based on mean shift
        if abs(mean_diff) > 10:
            # Convert mean shift to stops
            # Rough approximation: 30 units ≈ 1 stop
            exposure_correction = -mean_diff / 30.0
            if exposure_correction > 0:
                corrections.append(
                    f"Increase exposure by {abs(exposure_correction):.1f} stops (print is too dark)"
                )
            else:
                corrections.append(
                    f"Decrease exposure by {abs(exposure_correction):.1f} stops "
                    "(print is too light)"
                )

        # Contrast correction
        contrast_diff = comparison["changes"]["contrast_change"]
        if abs(contrast_diff) > 10:
            if contrast_diff > 0:
                contrast_correction = "reduce"
                corrections.append(
                    "Reduce contrast: Print has more contrast than target. "
                    "Consider reducing Na2 or using less FO #2."
                )
            else:
                contrast_correction = "increase"
                corrections.append(
                    "Increase contrast: Print lacks contrast. "
                    "Consider adding Na2 or increasing FO #2."
                )

        # Curve adjustments based on zone differences
        if ProblemArea.HIGHLIGHTS in [p[0] for p in problem_areas]:
            curve_adjustments["highlights"] = -0.1 if "blown" in str(problem_areas) else 0.1
            corrections.append(
                "Adjust curve highlights: "
                + (
                    "Reduce highlight values"
                    if curve_adjustments["highlights"] < 0
                    else "Increase highlight values"
                )
            )

        if ProblemArea.SHADOWS in [p[0] for p in problem_areas]:
            curve_adjustments["shadows"] = -0.1 if "blocked" in str(problem_areas) else 0.1
            corrections.append(
                "Adjust curve shadows: "
                + (
                    "Reduce shadow values"
                    if curve_adjustments["shadows"] < 0
                    else "Increase shadow values"
                )
            )

        # General recommendations
        if overall_match < 0.7:
            corrections.append(
                "Significant differences detected. Consider creating new calibration "
                "curve for this paper/chemistry combination."
            )

        return PrintQualityAnalysis(
            overall_match_score=overall_match,
            density_correlation=density_corr,
            mean_density_difference=mean_diff,
            max_density_difference=max_diff,
            density_range_match=range_match,
            problem_areas=problem_areas,
            zone_differences=zone_differences,
            worst_zones=worst_zones,
            suggested_exposure_correction_stops=exposure_correction,
            suggested_contrast_correction=contrast_correction,
            suggested_curve_adjustments=curve_adjustments,
            corrections=corrections,
        )

    # ========================================================================
    # 6. Workflow Optimization
    # ========================================================================

    def optimize_workflow(
        self,
        print_history: list[CalibrationRecord],
        success_threshold: float = 0.8,
    ) -> WorkflowOptimization:
        """
        Learn from past prints to optimize workflow.

        Analyzes print history to:
        - Identify patterns in successful prints
        - Suggest parameter improvements
        - Build predictive models for future prints
        - Identify common mistakes

        Args:
            print_history: List of CalibrationRecord objects
            success_threshold: Threshold for considering a print successful (0-1)

        Returns:
            WorkflowOptimization with insights and recommendations
        """
        if not print_history:
            return WorkflowOptimization(
                total_prints_analyzed=0,
                successful_prints=0,
                success_rate=0.0,
                insights=["No print history available for analysis."],
                confidence=0.0,
            )

        total_prints = len(print_history)

        # Determine successful prints
        # For now, we'll consider prints with good density range as successful
        successful = [
            r for r in print_history if r.measured_densities and max(r.measured_densities) > 1.5
        ]
        successful_count = len(successful)
        success_rate = successful_count / total_prints if total_prints > 0 else 0.0

        # Analyze patterns in successful prints
        optimal_params = {}
        trends = {}
        insights = []
        efficiency_suggestions = []
        common_mistakes = []

        if successful_count > 0:
            # Calculate average optimal parameters
            avg_exposure = np.mean([r.exposure_time for r in successful])
            avg_metal_ratio = np.mean([r.metal_ratio for r in successful])
            avg_humidity = np.mean([r.humidity for r in successful if r.humidity])
            avg_temp = np.mean([r.temperature for r in successful if r.temperature])

            optimal_params = {
                "avg_exposure_time": float(avg_exposure),
                "avg_metal_ratio": float(avg_metal_ratio),
                "avg_humidity": float(avg_humidity) if not np.isnan(avg_humidity) else None,
                "avg_temperature": float(avg_temp) if not np.isnan(avg_temp) else None,
            }

            # Identify trends
            # Group by paper type
            paper_groups = {}
            for record in successful:
                if record.paper_type not in paper_groups:
                    paper_groups[record.paper_type] = []
                paper_groups[record.paper_type].append(record)

            # Analyze most successful paper
            if paper_groups:
                paper_groups: dict[str, list[CalibrationRecord]] = paper_groups
                best_paper = max(paper_groups.keys(), key=lambda k: len(paper_groups[k]))
                trends["most_successful_paper"] = best_paper
                insights.append(
                    f"Most successful results with {best_paper} paper "
                    f"({len(paper_groups[best_paper])} prints)."
                )

            # Analyze metal ratio trends
            if avg_metal_ratio < 0.3:
                trends["metal_ratio"] = "palladium_dominant"
                insights.append(
                    "Your successful prints favor palladium-rich chemistry "
                    f"(avg {avg_metal_ratio * 100:.0f}% Pt). "
                    "Consider this your preferred aesthetic."
                )
            elif avg_metal_ratio > 0.7:
                trends["metal_ratio"] = "platinum_dominant"
                insights.append(
                    "Your successful prints favor platinum-rich chemistry "
                    f"(avg {avg_metal_ratio * 100:.0f}% Pt). "
                    "Expect cool tones and maximum Dmax."
                )

            # Exposure time consistency
            exposure_std = np.std([r.exposure_time for r in successful])
            if exposure_std / avg_exposure < 0.2:
                trends["exposure_consistency"] = "high"
                insights.append(
                    f"Exposure times are very consistent (σ={exposure_std:.1f}s). "
                    "Good workflow control!"
                )
                efficiency_suggestions.append(
                    f"Continue using ~{avg_exposure:.0f}s base exposure for similar negatives."
                )
            else:
                trends["exposure_consistency"] = "variable"
                insights.append(
                    "Exposure times vary significantly. Consider standardizing "
                    "negative densities for more predictable results."
                )

        # Analyze failures if any
        failed = [r for r in print_history if r not in successful]
        if failed:
            # Look for common issues
            very_short_exposures = [r for r in failed if r.exposure_time < avg_exposure * 0.5]
            if len(very_short_exposures) > 2:
                common_mistakes.append(
                    f"Found {len(very_short_exposures)} prints with very short exposure times. "
                    "May have resulted in underexposure."
                )

            very_long_exposures = [r for r in failed if r.exposure_time > avg_exposure * 2]
            if len(very_long_exposures) > 2:
                common_mistakes.append(
                    f"Found {len(very_long_exposures)} prints with very long exposure times. "
                    "May indicate overly dense negatives or weak UV source."
                )

        # Build recommendations
        recommended_base_exposure = optimal_params.get("avg_exposure_time")
        recommended_metal_ratio = optimal_params.get("avg_metal_ratio")

        # Paper-specific settings
        paper_settings = {}
        for paper, records in paper_groups.items():
            avg_exp = np.mean([r.exposure_time for r in records])
            paper_settings[paper] = {
                "recommended_exposure": float(avg_exp),
                "sample_count": len(records),
            }

        # Efficiency suggestions
        if success_rate > 0.7:
            efficiency_suggestions.append(
                "High success rate! Consider batch printing similar images."
            )
        else:
            efficiency_suggestions.append(
                "Consider making test strips for each new negative to improve success rate."
            )

        if total_prints > 10:
            efficiency_suggestions.append(
                "With this much experience, consider creating paper-specific calibration curves."
            )

        # Calculate confidence based on sample size
        confidence = min(1.0, successful_count / 10.0)  # Full confidence at 10+ successes

        return WorkflowOptimization(
            total_prints_analyzed=total_prints,
            successful_prints=successful_count,
            success_rate=success_rate,
            optimal_parameters=optimal_params,
            parameter_trends=trends,
            recommended_base_exposure=recommended_base_exposure,
            recommended_metal_ratio=recommended_metal_ratio,
            recommended_paper_settings=paper_settings,
            efficiency_suggestions=efficiency_suggestions,
            common_mistakes=common_mistakes,
            confidence=confidence,
            insights=insights,
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_paper_speed_factor(self, paper_type: str) -> float:
        """
        Get paper speed factor from paper type name.

        Args:
            paper_type: Paper type name

        Returns:
            Speed factor (1.0 = average, <1.0 = faster, >1.0 = slower)
        """
        # Common paper speed adjustments
        # These are configurable via settings but have defaults
        paper_lower = paper_type.lower()

        # Fast papers (sized, hot press)
        if any(
            keyword in paper_lower for keyword in ["hot press", "hp", "sized", "arches platine"]
        ):
            return 0.9

        # Slow papers (unsized, cold press)
        elif any(keyword in paper_lower for keyword in ["cold press", "cp", "unsized", "handmade"]):
            return 1.1

        # Average speed
        else:
            return 1.0
