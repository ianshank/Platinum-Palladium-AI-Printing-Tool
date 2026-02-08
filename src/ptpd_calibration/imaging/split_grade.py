"""
Split-grade printing simulation for platinum/palladium printing.

Split-grade printing is a technique where shadows and highlights are printed
separately with different contrast levels, then combined to achieve optimal
tonal separation. This module provides simulation and preview capabilities
for split-grade Pt/Pd printing workflows.

Key Features:
- Automatic analysis of optimal split-grade parameters
- Shadow and highlight mask generation
- Separate contrast curve application
- Pt/Pd metal characteristic simulation
- Exposure time calculation for dual printing
- Preview and comparison tools

References:
- Split-Grade Printing: A Comprehensive Guide to B&W Printing
- Platinum/Palladium Printing: A Contemporary Guide
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.ndimage import gaussian_filter


class BlendMode(str, Enum):
    """Methods for blending shadow and highlight exposures."""

    LINEAR = "linear"  # Simple linear blend
    GAMMA = "gamma"  # Gamma-corrected blend
    CUSTOM = "custom"  # Custom blend curve
    SOFT_LIGHT = "soft_light"  # Soft light compositing
    OVERLAY = "overlay"  # Overlay compositing


class MetalType(str, Enum):
    """Platinum/Palladium metal types."""

    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    MIXED = "mixed"


class SplitGradeSettings(BaseSettings):
    """Configuration for split-grade printing simulation.

    All settings can be overridden via environment variables with
    PTPD_SPLIT_GRADE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTPD_SPLIT_GRADE_",
        validate_assignment=True,
    )

    # Contrast grade settings (0-5 scale, similar to traditional VC papers)
    shadow_grade: float = Field(
        default=2.5,
        ge=0.0,
        le=5.0,
        description="Contrast grade for shadow regions (0=softest, 5=hardest)",
    )
    highlight_grade: float = Field(
        default=1.5,
        ge=0.0,
        le=5.0,
        description="Contrast grade for highlight regions (0=softest, 5=hardest)",
    )

    # Exposure ratio (percentage of total exposure time)
    shadow_exposure_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Ratio of total exposure time for shadows (0.0-1.0)",
    )

    # Blend settings
    blend_mode: BlendMode = Field(
        default=BlendMode.GAMMA, description="Method for blending shadow and highlight exposures"
    )
    blend_gamma: float = Field(
        default=2.2, ge=0.5, le=4.0, description="Gamma value for gamma blend mode"
    )
    blend_softness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Softness of blend transition (0=hard, 1=soft)"
    )

    # Tonal thresholds (0-1 range)
    shadow_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Threshold separating shadows from midtones (0-1)"
    )
    highlight_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold separating highlights from midtones (0-1)",
    )

    # Mask generation
    mask_blur_radius: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Gaussian blur radius for mask smoothing (pixels)",
    )
    mask_feather_amount: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Amount of feathering at mask edges"
    )

    # Metal characteristics
    platinum_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Platinum to palladium ratio (0=pure Pd, 1=pure Pt)",
    )

    # Advanced settings
    preserve_highlights: bool = Field(
        default=True, description="Prevent highlight blocking in bright areas"
    )
    preserve_shadows: bool = Field(
        default=True, description="Prevent shadow crushing in dark areas"
    )
    highlight_hold_point: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Point above which highlights are held to paper white",
    )
    shadow_hold_point: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Point below which shadows are held to maximum density",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "SplitGradeSettings":
        """Ensure thresholds are in correct order."""
        if self.shadow_threshold >= self.highlight_threshold:
            raise ValueError(
                f"shadow_threshold ({self.shadow_threshold}) must be less than "
                f"highlight_threshold ({self.highlight_threshold})"
            )
        return self


@dataclass
class TonalAnalysis:
    """Results of image tonal analysis for split-grade determination."""

    # Histogram statistics
    mean_luminance: float
    median_luminance: float
    std_luminance: float

    # Distribution percentiles
    p05: float  # 5th percentile
    p25: float  # 25th percentile
    p50: float  # 50th percentile (median)
    p75: float  # 75th percentile
    p95: float  # 95th percentile

    # Tonal regions (percentage of pixels)
    shadow_percentage: float
    midtone_percentage: float
    highlight_percentage: float

    # Recommended split-grade parameters
    recommended_shadow_grade: float
    recommended_highlight_grade: float
    recommended_shadow_threshold: float
    recommended_highlight_threshold: float
    recommended_exposure_ratio: float

    # Image characteristics
    is_low_key: bool  # Predominantly dark
    is_high_key: bool  # Predominantly light
    needs_split_grade: bool  # Would benefit from split grading

    # Quality metrics
    tonal_range: float  # 0-1, how much of the tonal range is used
    contrast_score: float  # Estimated contrast level

    notes: list[str] = field(default_factory=list)


@dataclass
class ExposureCalculation:
    """Calculated exposure times for split-grade printing."""

    # Total exposure
    total_exposure_seconds: float

    # Individual exposures
    shadow_exposure_seconds: float
    highlight_exposure_seconds: float

    # Exposure ratios
    shadow_ratio: float
    highlight_ratio: float

    # Grades used
    shadow_grade: float
    highlight_grade: float

    # Recommendations
    notes: list[str] = field(default_factory=list)

    def format_exposure_info(self) -> str:
        """Format exposure information as readable text."""
        lines = [
            "SPLIT-GRADE EXPOSURE CALCULATION",
            "=" * 50,
            f"Total Exposure Time: {self.total_exposure_seconds:.1f} seconds",
            "",
            "SHADOW EXPOSURE:",
            f"  Time: {self.shadow_exposure_seconds:.1f} seconds ({self.shadow_ratio * 100:.0f}%)",
            f"  Grade: {self.shadow_grade:.1f}",
            "",
            "HIGHLIGHT EXPOSURE:",
            f"  Time: {self.highlight_exposure_seconds:.1f} seconds ({self.highlight_ratio * 100:.0f}%)",
            f"  Grade: {self.highlight_grade:.1f}",
        ]

        if self.notes:
            lines.extend(["", "NOTES:"])
            for note in self.notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)


class TonalCurveAdjuster:
    """Generate and apply tonal curves for Pt/Pd printing with metal characteristics.

    Simulates the characteristic response curves of platinum and palladium
    printing processes, including their different contrast and tonal behaviors.
    """

    def __init__(self, settings: SplitGradeSettings | None = None):
        """Initialize the tonal curve adjuster.

        Args:
            settings: Optional split-grade settings. If None, uses defaults.
        """
        self.settings = settings or SplitGradeSettings()
        self._curve_cache: dict[str, np.ndarray] = {}

    def create_contrast_curve(
        self,
        grade: float,
        num_points: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate contrast curve for a given grade level.

        Grade scale:
        - 0: Very soft, low contrast (long toe, gentle shoulder)
        - 1-2: Soft to normal contrast
        - 3: Normal contrast (linear in midtones)
        - 4-5: Hard to very hard contrast (short toe, steep midtones)

        Args:
            grade: Contrast grade (0-5 scale)
            num_points: Number of points in the curve

        Returns:
            Tuple of (input_values, output_values) arrays
        """
        # Input values from 0 to 1
        x = np.linspace(0, 1, num_points)

        # Grade affects the gamma and shoulder/toe compression
        # Grade 0: gamma ~0.6, soft shoulder/toe
        # Grade 3: gamma ~1.0, moderate shoulder/toe
        # Grade 5: gamma ~1.8, hard shoulder/toe

        # Calculate gamma from grade
        gamma = 0.55 + (grade * 0.25)  # Range: 0.55 to 1.8

        # Calculate toe and shoulder compression
        toe_compress = max(0.0, 0.3 - (grade * 0.05))  # Decreases with grade
        shoulder_compress = max(0.0, 0.3 - (grade * 0.05))

        # Apply power curve (gamma)
        y = np.power(x, gamma)

        # Apply toe compression (lift shadows)
        if toe_compress > 0:
            toe_mask = x < 0.3
            toe_lift = toe_compress * (1 - x / 0.3)
            y[toe_mask] += toe_lift[toe_mask] * y[toe_mask]

        # Apply shoulder compression (compress highlights)
        if shoulder_compress > 0:
            shoulder_mask = x > 0.7
            shoulder_factor = shoulder_compress * ((x - 0.7) / 0.3)
            y[shoulder_mask] = y[shoulder_mask] * (1 - shoulder_factor[shoulder_mask] * 0.3)

        # Ensure endpoints are preserved
        y[0] = 0.0
        y[-1] = 1.0

        # Ensure monotonicity
        y = self._ensure_monotonic(y)

        return x, y

    def apply_platinum_characteristic(
        self,
        image: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Apply platinum characteristic response curve.

        Platinum produces:
        - Higher maximum density (deeper blacks)
        - Cooler, more neutral tones
        - Slightly more contrast in highlights
        - Longer tonal scale

        Args:
            image: Input image array (0-1 normalized)
            strength: Strength of effect (0-1)

        Returns:
            Processed image array
        """
        if strength <= 0:
            return image.copy()

        # Platinum characteristic: slightly higher contrast, extended blacks
        # S-curve with emphasis on shadow separation
        x = np.linspace(0, 1, 256)

        # Platinum curve: gamma ~1.1 with extended toe
        gamma = 1.1
        y = np.power(x, gamma)

        # Extend toe for better shadow detail
        toe_mask = x < 0.3
        toe_extension = 0.15 * (1 - x / 0.3)
        y[toe_mask] -= toe_extension[toe_mask] * y[toe_mask]

        # Slight highlight compression for smoother highlights
        shoulder_mask = x > 0.8
        shoulder_factor = (x - 0.8) / 0.2
        y[shoulder_mask] = y[shoulder_mask] * (1 - shoulder_factor[shoulder_mask] * 0.1)

        # Ensure valid range
        y = np.clip(y, 0, 1)
        y = self._ensure_monotonic(y)

        # Apply curve
        img_8bit = (image * 255).astype(np.uint8)
        lut = (y * 255).astype(np.uint8)
        processed = lut[img_8bit] / 255.0

        # Blend with original based on strength
        return image * (1 - strength) + processed * strength

    def apply_palladium_characteristic(
        self,
        image: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Apply palladium characteristic response curve.

        Palladium produces:
        - Warmer, brown-black tones
        - Slightly lower maximum density
        - Softer highlight rolloff
        - Smoother midtone transitions

        Args:
            image: Input image array (0-1 normalized)
            strength: Strength of effect (0-1)

        Returns:
            Processed image array
        """
        if strength <= 0:
            return image.copy()

        # Palladium characteristic: softer, warmer response
        x = np.linspace(0, 1, 256)

        # Palladium curve: gamma ~0.9 with gentler toe and shoulder
        gamma = 0.9
        y = np.power(x, gamma)

        # Gentler toe for smoother shadow transitions
        toe_mask = x < 0.25
        toe_lift = 0.1 * (1 - x / 0.25)
        y[toe_mask] += toe_lift[toe_mask] * y[toe_mask]

        # Softer shoulder for highlight preservation
        shoulder_mask = x > 0.75
        shoulder_factor = (x - 0.75) / 0.25
        y[shoulder_mask] = y[shoulder_mask] * (1 - shoulder_factor[shoulder_mask] * 0.15)

        # Slightly compressed Dmax (palladium doesn't go as dark as platinum)
        y = y * 0.95 + 0.05

        # Ensure valid range
        y = np.clip(y, 0, 1)
        y = self._ensure_monotonic(y)

        # Apply curve
        img_8bit = (image * 255).astype(np.uint8)
        lut = (y * 255).astype(np.uint8)
        processed = lut[img_8bit] / 255.0

        # Blend with original based on strength
        return image * (1 - strength) + processed * strength

    def blend_metal_characteristics(
        self,
        image: np.ndarray,
        pt_ratio: float,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Blend platinum and palladium characteristics based on chemistry ratio.

        Args:
            image: Input image array (0-1 normalized)
            pt_ratio: Platinum ratio (0=pure Pd, 1=pure Pt)
            strength: Overall strength of metal characteristic effect (0-1)

        Returns:
            Processed image array with blended metal characteristics
        """
        if strength <= 0:
            return image.copy()

        pt_ratio = np.clip(pt_ratio, 0, 1)
        pd_ratio = 1 - pt_ratio

        # Apply individual characteristics
        pt_processed = self.apply_platinum_characteristic(image, strength=1.0)
        pd_processed = self.apply_palladium_characteristic(image, strength=1.0)

        # Blend based on ratio
        blended = pt_processed * pt_ratio + pd_processed * pd_ratio

        # Apply overall strength
        return image * (1 - strength) + blended * strength

    def apply_curve_to_image(
        self,
        image: np.ndarray,
        grade: float,
        apply_metal_characteristic: bool = True,
    ) -> np.ndarray:
        """Apply contrast curve and optional metal characteristics to image.

        Args:
            image: Input image array (0-1 normalized)
            grade: Contrast grade (0-5)
            apply_metal_characteristic: Whether to apply Pt/Pd characteristics

        Returns:
            Processed image array
        """
        # Generate contrast curve
        x_curve, y_curve = self.create_contrast_curve(grade)

        # Apply curve via lookup table
        img_8bit = (image * 255).astype(np.uint8)
        lut = (y_curve * 255).astype(np.uint8)
        processed = lut[img_8bit] / 255.0

        # Apply metal characteristics if requested
        if apply_metal_characteristic:
            processed = self.blend_metal_characteristics(
                processed,
                self.settings.platinum_ratio,
                strength=1.0,
            )

        return processed

    @staticmethod
    def _ensure_monotonic(y: np.ndarray) -> np.ndarray:
        """Ensure array is monotonically increasing.

        Args:
            y: Input array

        Returns:
            Monotonic array
        """
        result = y.copy()
        for i in range(1, len(result)):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
        return result


class SplitGradeSimulator:
    """Simulate split-grade printing for platinum/palladium processes.

    Provides comprehensive tools for:
    - Analyzing images for optimal split-grade parameters
    - Creating shadow and highlight masks
    - Applying separate contrast grades to different tonal regions
    - Simulating dual exposures
    - Calculating exposure times
    """

    def __init__(self, settings: SplitGradeSettings | None = None):
        """Initialize the split-grade simulator.

        Args:
            settings: Optional split-grade settings. If None, uses defaults.
        """
        self.settings = settings or SplitGradeSettings()
        self.curve_adjuster = TonalCurveAdjuster(self.settings)

    def analyze_image(
        self,
        image: np.ndarray | Image.Image,
    ) -> TonalAnalysis:
        """Analyze image to determine optimal split-grade parameters.

        Examines the tonal distribution and recommends:
        - Shadow and highlight grades
        - Tonal thresholds
        - Exposure ratios

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            TonalAnalysis with recommendations
        """
        # Convert to normalized array
        img_array = self._prepare_image(image)

        # Convert to luminance if color
        if img_array.ndim == 3:
            # Convert RGB to luminance
            luminance = (
                0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            )
        else:
            luminance = img_array

        # Calculate histogram statistics
        flat_lum = luminance.flatten()
        mean_lum = float(np.mean(flat_lum))
        median_lum = float(np.median(flat_lum))
        std_lum = float(np.std(flat_lum))

        # Calculate percentiles
        percentiles = np.percentile(flat_lum, [5, 25, 50, 75, 95])
        p05, p25, p50, p75, p95 = percentiles

        # Calculate tonal region percentages
        shadow_pct = float(np.sum(flat_lum < 0.3) / len(flat_lum))
        highlight_pct = float(np.sum(flat_lum > 0.7) / len(flat_lum))
        midtone_pct = 1.0 - shadow_pct - highlight_pct

        # Determine key
        is_low_key = mean_lum < 0.35
        is_high_key = mean_lum > 0.65

        # Calculate tonal range and contrast
        tonal_range = float(p95 - p05)
        contrast_score = float(std_lum * 2)  # Normalized to ~0-1 range

        # Determine if split-grade would be beneficial
        # Benefits: wide tonal range, significant shadows AND highlights
        needs_split_grade = tonal_range > 0.5 and shadow_pct > 0.15 and highlight_pct > 0.15

        # Recommend parameters based on analysis
        rec_shadow_grade, rec_highlight_grade = self._recommend_grades(
            mean_lum, contrast_score, is_low_key, is_high_key
        )

        rec_shadow_thresh, rec_highlight_thresh = self._recommend_thresholds(
            p25, p75, shadow_pct, highlight_pct
        )

        rec_exposure_ratio = self._recommend_exposure_ratio(
            shadow_pct, highlight_pct, is_low_key, is_high_key
        )

        # Generate notes
        notes = []
        if is_low_key:
            notes.append("Low-key image: Emphasize shadow detail with softer shadow grade")
        elif is_high_key:
            notes.append("High-key image: Protect highlights with softer highlight grade")

        if tonal_range < 0.4:
            notes.append("Limited tonal range: Consider single-grade printing")
        elif tonal_range > 0.7:
            notes.append("Wide tonal range: Split-grade printing highly recommended")

        if contrast_score < 0.3:
            notes.append("Low contrast: Use harder grades to increase separation")
        elif contrast_score > 0.7:
            notes.append("High contrast: Use softer grades to preserve detail")

        return TonalAnalysis(
            mean_luminance=mean_lum,
            median_luminance=median_lum,
            std_luminance=std_lum,
            p05=float(p05),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p95=float(p95),
            shadow_percentage=shadow_pct,
            midtone_percentage=midtone_pct,
            highlight_percentage=highlight_pct,
            recommended_shadow_grade=rec_shadow_grade,
            recommended_highlight_grade=rec_highlight_grade,
            recommended_shadow_threshold=rec_shadow_thresh,
            recommended_highlight_threshold=rec_highlight_thresh,
            recommended_exposure_ratio=rec_exposure_ratio,
            is_low_key=is_low_key,
            is_high_key=is_high_key,
            needs_split_grade=needs_split_grade,
            tonal_range=tonal_range,
            contrast_score=contrast_score,
            notes=notes,
        )

    def create_shadow_mask(
        self,
        image: np.ndarray | Image.Image,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Create mask selecting shadow regions of the image.

        Args:
            image: Input image
            threshold: Shadow threshold (0-1). If None, uses settings.

        Returns:
            Shadow mask array (0-1, where 1=shadow, 0=not shadow)
        """
        # Convert to normalized array
        img_array = self._prepare_image(image)
        luminance = self._get_luminance(img_array)

        # Use threshold from settings if not provided
        thresh = threshold if threshold is not None else self.settings.shadow_threshold

        # Create base mask (inverse: dark areas get high values)
        mask = np.clip((thresh - luminance) / thresh, 0, 1)

        # Apply feathering and blur
        mask = self._apply_mask_processing(mask)

        return mask

    def create_highlight_mask(
        self,
        image: np.ndarray | Image.Image,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Create mask selecting highlight regions of the image.

        Args:
            image: Input image
            threshold: Highlight threshold (0-1). If None, uses settings.

        Returns:
            Highlight mask array (0-1, where 1=highlight, 0=not highlight)
        """
        # Convert to normalized array
        img_array = self._prepare_image(image)
        luminance = self._get_luminance(img_array)

        # Use threshold from settings if not provided
        thresh = threshold if threshold is not None else self.settings.highlight_threshold

        # Create base mask (bright areas get high values)
        mask = np.clip((luminance - thresh) / (1 - thresh), 0, 1)

        # Apply feathering and blur
        mask = self._apply_mask_processing(mask)

        return mask

    def simulate_split_grade(
        self,
        image: np.ndarray | Image.Image,
        settings: SplitGradeSettings | None = None,
    ) -> np.ndarray:
        """Apply split-grade simulation to an image.

        Simulates the effect of printing with separate shadow and highlight
        grades, including Pt/Pd metal characteristics.

        Args:
            image: Input image
            settings: Optional custom settings. If None, uses instance settings.

        Returns:
            Processed image array (0-1 normalized)
        """
        settings = settings or self.settings

        # Prepare image
        img_array = self._prepare_image(image)
        luminance = self._get_luminance(img_array)

        # Create masks
        shadow_mask = self.create_shadow_mask(img_array, settings.shadow_threshold)
        highlight_mask = self.create_highlight_mask(img_array, settings.highlight_threshold)

        # Apply shadow grade
        shadow_processed = self.curve_adjuster.apply_curve_to_image(
            luminance,
            grade=settings.shadow_grade,
            apply_metal_characteristic=True,
        )

        # Apply highlight grade
        highlight_processed = self.curve_adjuster.apply_curve_to_image(
            luminance,
            grade=settings.highlight_grade,
            apply_metal_characteristic=True,
        )

        # Blend exposures
        result = self.blend_exposures(
            shadow_processed,
            highlight_processed,
            settings,
            shadow_mask,
            highlight_mask,
        )

        # Apply hold points if requested
        if settings.preserve_highlights:
            highlight_hold_mask = luminance > settings.highlight_hold_point
            result[highlight_hold_mask] = luminance[highlight_hold_mask]

        if settings.preserve_shadows:
            shadow_hold_mask = luminance < settings.shadow_hold_point
            result[shadow_hold_mask] = luminance[shadow_hold_mask]

        return result

    def blend_exposures(
        self,
        shadow_image: np.ndarray,
        highlight_image: np.ndarray,
        settings: SplitGradeSettings | None = None,
        shadow_mask: np.ndarray | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend shadow and highlight exposures using specified blend mode.

        Args:
            shadow_image: Shadow-graded image array
            highlight_image: Highlight-graded image array
            settings: Optional custom settings
            shadow_mask: Optional pre-computed shadow mask
            highlight_mask: Optional pre-computed highlight mask

        Returns:
            Blended image array
        """
        settings = settings or self.settings

        # Create masks if not provided
        if shadow_mask is None:
            shadow_mask = np.ones_like(shadow_image) * settings.shadow_exposure_ratio
        else:
            shadow_mask = shadow_mask * settings.shadow_exposure_ratio

        if highlight_mask is None:
            highlight_mask = np.ones_like(highlight_image) * (1 - settings.shadow_exposure_ratio)
        else:
            highlight_mask = highlight_mask * (1 - settings.shadow_exposure_ratio)

        # Normalize masks so they sum to 1
        total_mask = shadow_mask + highlight_mask
        total_mask = np.maximum(total_mask, 1e-6)  # Avoid division by zero
        shadow_mask = shadow_mask / total_mask
        highlight_mask = highlight_mask / total_mask

        # Apply blend mode
        if settings.blend_mode == BlendMode.LINEAR:
            result = shadow_image * shadow_mask + highlight_image * highlight_mask

        elif settings.blend_mode == BlendMode.GAMMA:
            # Gamma-corrected blend (more perceptually uniform)
            gamma = settings.blend_gamma
            shadow_gamma = np.power(shadow_image, gamma)
            highlight_gamma = np.power(highlight_image, gamma)
            blended_gamma = shadow_gamma * shadow_mask + highlight_gamma * highlight_mask
            result = np.power(blended_gamma, 1.0 / gamma)

        elif settings.blend_mode == BlendMode.SOFT_LIGHT:
            # Soft light compositing
            result = self._soft_light_blend(shadow_image, highlight_image, shadow_mask)

        elif settings.blend_mode == BlendMode.OVERLAY:
            # Overlay compositing
            result = self._overlay_blend(shadow_image, highlight_image, shadow_mask)

        else:  # CUSTOM or fallback
            result = shadow_image * shadow_mask + highlight_image * highlight_mask

        return np.clip(result, 0, 1)

    def preview_result(
        self,
        image: np.ndarray | Image.Image,
        settings: SplitGradeSettings | None = None,
        include_masks: bool = False,
    ) -> dict[str, np.ndarray | Image.Image]:
        """Generate preview comparison showing original and processed results.

        Args:
            image: Input image
            settings: Optional custom settings
            include_masks: Whether to include shadow/highlight masks in output

        Returns:
            Dictionary with 'original', 'processed', and optionally 'shadow_mask',
            'highlight_mask' keys
        """
        settings = settings or self.settings

        # Prepare image
        img_array = self._prepare_image(image)
        luminance = self._get_luminance(img_array)

        # Process image
        processed = self.simulate_split_grade(img_array, settings)

        # Build result dictionary
        result = {
            "original": luminance,
            "processed": processed,
        }

        if include_masks:
            result["shadow_mask"] = self.create_shadow_mask(img_array, settings.shadow_threshold)
            result["highlight_mask"] = self.create_highlight_mask(
                img_array, settings.highlight_threshold
            )

        return result

    def calculate_exposure_times(
        self,
        base_time: float,
        settings: SplitGradeSettings | None = None,
    ) -> ExposureCalculation:
        """Calculate separate exposure times for shadow and highlight grades.

        Args:
            base_time: Base exposure time in seconds
            settings: Optional custom settings

        Returns:
            ExposureCalculation with timing details
        """
        settings = settings or self.settings

        # Calculate individual exposures with ratios clamped to 0-1
        shadow_ratio = float(np.clip(settings.shadow_exposure_ratio, 0.0, 1.0))
        shadow_ratio = float(round(shadow_ratio, 6))
        highlight_ratio = float(round(np.clip(1.0 - shadow_ratio, 0.0, 1.0), 6))
        shadow_time = base_time * shadow_ratio
        highlight_time = base_time * highlight_ratio

        # Generate notes
        notes = []

        if settings.shadow_grade > 3.5:
            notes.append("Hard shadow grade: Watch for blocked shadows")
        elif settings.shadow_grade < 1.5:
            notes.append("Soft shadow grade: May lose shadow contrast")

        if settings.highlight_grade > 3.5:
            notes.append("Hard highlight grade: May block highlights")
        elif settings.highlight_grade < 1.5:
            notes.append("Soft highlight grade: Good for preserving highlight detail")

        if abs(settings.shadow_grade - settings.highlight_grade) < 0.5:
            notes.append("Similar grades: Consider single-grade printing instead")

        if shadow_time < 5:
            notes.append(
                f"Short shadow exposure ({shadow_time:.1f}s): Consider increasing base time"
            )
        if highlight_time < 5:
            notes.append(
                f"Short highlight exposure ({highlight_time:.1f}s): Consider increasing base time"
            )

        return ExposureCalculation(
            total_exposure_seconds=base_time,
            shadow_exposure_seconds=shadow_time,
            highlight_exposure_seconds=highlight_time,
            shadow_ratio=shadow_ratio,
            highlight_ratio=highlight_ratio,
            shadow_grade=settings.shadow_grade,
            highlight_grade=settings.highlight_grade,
            notes=notes,
        )

    # Private helper methods

    def _prepare_image(
        self,
        image: np.ndarray | Image.Image,
    ) -> np.ndarray:
        """Convert image to normalized numpy array.

        Args:
            image: Input image (PIL or numpy)

        Returns:
            Normalized array (0-1 range)
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                img_array = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                img_array = image.astype(np.float32) / 65535.0
            else:
                img_array = image.astype(np.float32)
                # Normalize if outside 0-1 range
                if img_array.max() > 1.0:
                    img_array = img_array / img_array.max()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return np.clip(img_array, 0, 1)

    def _get_luminance(self, img_array: np.ndarray) -> np.ndarray:
        """Extract luminance from image array.

        Args:
            img_array: Image array (may be grayscale or color)

        Returns:
            Luminance array (2D)
        """
        if img_array.ndim == 2:
            return img_array
        elif img_array.ndim == 3:
            # Convert RGB to luminance using standard weights
            return (
                0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            )
        else:
            raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")

    def _apply_mask_processing(self, mask: np.ndarray) -> np.ndarray:
        """Apply blur and feathering to mask.

        Args:
            mask: Raw mask array

        Returns:
            Processed mask array
        """
        # Apply Gaussian blur for smooth transitions
        if self.settings.mask_blur_radius > 0:
            mask = gaussian_filter(mask, sigma=self.settings.mask_blur_radius)

        # Apply feathering (soften the mask values)
        if self.settings.mask_feather_amount > 0:
            feather = self.settings.mask_feather_amount
            mask = np.power(mask, 1.0 / (1.0 + feather))

        return np.clip(mask, 0, 1)

    def _recommend_grades(
        self,
        mean_lum: float,
        contrast: float,
        is_low_key: bool,
        is_high_key: bool,
    ) -> tuple[float, float]:
        """Recommend shadow and highlight grades based on image analysis.

        Args:
            mean_lum: Mean luminance
            contrast: Contrast score
            is_low_key: Whether image is low-key
            is_high_key: Whether image is high-key

        Returns:
            Tuple of (shadow_grade, highlight_grade)
        """
        # Default grades
        shadow_grade = 2.5
        highlight_grade = 1.5

        # Adjust based on contrast
        if contrast < 0.3:
            # Low contrast: use harder grades
            shadow_grade += 1.0
            highlight_grade += 0.5
        elif contrast > 0.7:
            # High contrast: use softer grades
            shadow_grade -= 1.0
            highlight_grade -= 0.5

        # Adjust based on key
        if is_low_key:
            # Low key: softer shadow grade to preserve detail
            shadow_grade -= 0.5
        elif is_high_key:
            # High key: softer highlight grade to preserve detail
            highlight_grade -= 0.5

        # Clamp to valid range
        shadow_grade = np.clip(shadow_grade, 0, 5)
        highlight_grade = np.clip(highlight_grade, 0, 5)

        return shadow_grade, highlight_grade

    def _recommend_thresholds(
        self,
        p25: float,
        p75: float,
        shadow_pct: float,
        highlight_pct: float,
    ) -> tuple[float, float]:
        """Recommend shadow and highlight thresholds.

        Args:
            p25: 25th percentile
            p75: 75th percentile
            shadow_pct: Percentage of shadow pixels
            highlight_pct: Percentage of highlight pixels

        Returns:
            Tuple of (shadow_threshold, highlight_threshold)
        """
        # Start with percentile-based thresholds
        shadow_thresh = float(np.clip(p25 + 0.1, 0.2, 0.5))
        highlight_thresh = float(np.clip(p75 - 0.1, 0.6, 0.85))

        # Ensure minimum separation
        if highlight_thresh - shadow_thresh < 0.2:
            mid = (shadow_thresh + highlight_thresh) / 2
            shadow_thresh = mid - 0.1
            highlight_thresh = mid + 0.1

        return shadow_thresh, highlight_thresh

    def _recommend_exposure_ratio(
        self,
        shadow_pct: float,
        highlight_pct: float,
        is_low_key: bool,
        is_high_key: bool,
    ) -> float:
        """Recommend shadow exposure ratio.

        Args:
            shadow_pct: Percentage of shadow pixels
            highlight_pct: Percentage of highlight pixels
            is_low_key: Whether image is low-key
            is_high_key: Whether image is high-key

        Returns:
            Recommended shadow exposure ratio (0-1)
        """
        # Start with default
        ratio = 0.6

        # Adjust based on tonal distribution
        if shadow_pct > highlight_pct * 1.5:
            # More shadows: increase shadow exposure
            ratio = 0.65
        elif highlight_pct > shadow_pct * 1.5:
            # More highlights: decrease shadow exposure
            ratio = 0.55

        # Adjust based on key
        if is_low_key:
            ratio += 0.05
        elif is_high_key:
            ratio -= 0.05

        return np.clip(ratio, 0.4, 0.75)

    @staticmethod
    def _soft_light_blend(
        shadow: np.ndarray,
        highlight: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply soft light blending mode.

        Args:
            shadow: Shadow image
            highlight: Highlight image
            mask: Blend mask (0-1)

        Returns:
            Blended image
        """
        # Soft light formula
        result = np.where(
            shadow < 0.5,
            highlight - (1 - 2 * shadow) * highlight * (1 - highlight),
            highlight + (2 * shadow - 1) * (np.sqrt(highlight) - highlight),
        )
        # Blend with mask
        return shadow * (1 - mask) + result * mask

    @staticmethod
    def _overlay_blend(
        shadow: np.ndarray,
        highlight: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply overlay blending mode.

        Args:
            shadow: Shadow image
            highlight: Highlight image
            mask: Blend mask (0-1)

        Returns:
            Blended image
        """
        # Overlay formula
        result = np.where(
            highlight < 0.5, 2 * shadow * highlight, 1 - 2 * (1 - shadow) * (1 - highlight)
        )
        # Blend with mask
        return shadow * (1 - mask) + result * mask
