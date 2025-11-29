"""
Step wedge analysis module for PTPD Calibration System.

Provides comprehensive step wedge scan analysis with automatic curve generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from ptpd_calibration.config import TabletType, get_settings
from ptpd_calibration.core.models import CurveData, ExtractionResult
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.generator import CurveGenerator, TargetCurve
from ptpd_calibration.detection.reader import StepTabletReader


class AnalysisWarningLevel(str, Enum):
    """Warning severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityGrade(str, Enum):
    """Overall quality grade for analysis."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class AnalysisWarning:
    """Warning or issue found during analysis."""

    level: AnalysisWarningLevel
    code: str
    message: str
    suggestion: Optional[str] = None
    affected_patches: Optional[list[int]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "level": self.level.value,
            "code": self.code,
            "message": self.message,
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.affected_patches:
            result["affected_patches"] = self.affected_patches
        return result


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment of step wedge analysis."""

    # Overall grade
    grade: QualityGrade
    score: float  # 0-100

    # Individual metrics
    density_range_score: float  # Based on Dmax - Dmin
    uniformity_score: float  # Patch uniformity
    monotonicity_score: float  # Smooth progression
    detection_confidence: float  # Detection reliability
    signal_to_noise: float  # S/N ratio estimate

    # Issues found
    warnings: list[AnalysisWarning] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "grade": self.grade.value,
            "score": round(self.score, 1),
            "metrics": {
                "density_range": round(self.density_range_score, 2),
                "uniformity": round(self.uniformity_score, 2),
                "monotonicity": round(self.monotonicity_score, 2),
                "detection_confidence": round(self.detection_confidence, 2),
                "signal_to_noise": round(self.signal_to_noise, 2),
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendations": self.recommendations,
        }


@dataclass
class WedgeAnalysisConfig:
    """Configuration for step wedge analysis."""

    # Tablet settings
    tablet_type: TabletType = TabletType.STOUFFER_21

    # Quality thresholds
    min_density_range: float = 1.5  # Minimum acceptable density range
    max_dmin: float = 0.15  # Maximum acceptable Dmin
    min_dmax: float = 1.8  # Minimum expected Dmax
    uniformity_threshold: float = 0.7  # Minimum patch uniformity

    # Monotonicity settings
    max_reversal_tolerance: float = 0.02  # Maximum allowed reversal in density
    smoothing_window: int = 3  # Window size for smoothing analysis

    # Curve generation settings
    default_curve_type: CurveType = CurveType.LINEAR
    num_output_points: int = 256
    apply_smoothing: bool = True
    smoothing_factor: float = 0.05
    enforce_monotonicity: bool = True

    # Advanced settings
    auto_fix_reversals: bool = True
    outlier_rejection: bool = True
    outlier_threshold: float = 2.5  # MAD multiplier for outliers

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tablet_type": self.tablet_type.value,
            "min_density_range": self.min_density_range,
            "max_dmin": self.max_dmin,
            "min_dmax": self.min_dmax,
            "uniformity_threshold": self.uniformity_threshold,
            "default_curve_type": self.default_curve_type.value,
            "num_output_points": self.num_output_points,
        }


@dataclass
class WedgeAnalysisResult:
    """Complete result from step wedge analysis."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Source information
    source_path: Optional[Path] = None
    tablet_type: TabletType = TabletType.STOUFFER_21

    # Detection results
    extraction: Optional[ExtractionResult] = None
    detection_success: bool = False

    # Density measurements
    densities: list[float] = field(default_factory=list)
    input_values: list[float] = field(default_factory=list)  # Normalized 0-1
    dmin: Optional[float] = None
    dmax: Optional[float] = None
    density_range: Optional[float] = None

    # Original and processed densities
    raw_densities: list[float] = field(default_factory=list)
    corrected_densities: list[float] = field(default_factory=list)

    # Quality assessment
    quality: Optional[QualityAssessment] = None

    # Generated curve
    curve: Optional[CurveData] = None
    curve_generated: bool = False

    # Configuration used
    config: Optional[WedgeAnalysisConfig] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "source_path": str(self.source_path) if self.source_path else None,
            "tablet_type": self.tablet_type.value,
            "detection_success": self.detection_success,
            "num_patches": len(self.densities),
            "densities": [round(d, 4) for d in self.densities],
            "dmin": round(self.dmin, 4) if self.dmin is not None else None,
            "dmax": round(self.dmax, 4) if self.dmax is not None else None,
            "density_range": round(self.density_range, 4) if self.density_range is not None else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "curve_generated": self.curve_generated,
            "curve_name": self.curve.name if self.curve else None,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Step Wedge Analysis Results",
            f"=" * 40,
            f"Detection: {'Success' if self.detection_success else 'Failed'}",
            f"Patches: {len(self.densities)}",
        ]

        if self.dmin is not None:
            lines.append(f"Dmin: {self.dmin:.3f}")
        if self.dmax is not None:
            lines.append(f"Dmax: {self.dmax:.3f}")
        if self.density_range is not None:
            lines.append(f"Density Range: {self.density_range:.3f}")

        if self.quality:
            lines.append(f"Quality Grade: {self.quality.grade.value.upper()}")
            lines.append(f"Quality Score: {self.quality.score:.1f}/100")

        if self.curve_generated:
            lines.append(f"Curve Generated: {self.curve.name if self.curve else 'Yes'}")

        return "\n".join(lines)


class StepWedgeAnalyzer:
    """
    Comprehensive step wedge analyzer for Pt/Pd calibration.

    Performs automatic detection, extraction, quality assessment,
    and curve generation from step wedge scans.
    """

    def __init__(self, config: Optional[WedgeAnalysisConfig] = None):
        """
        Initialize the analyzer.

        Args:
            config: Analysis configuration. Uses defaults if not provided.
        """
        self.config = config or WedgeAnalysisConfig()
        self._reader: Optional[StepTabletReader] = None

    def _get_reader(self) -> StepTabletReader:
        """Get or create the step tablet reader."""
        if self._reader is None or self._reader.tablet_type != self.config.tablet_type:
            self._reader = StepTabletReader(tablet_type=self.config.tablet_type)
        return self._reader

    def analyze(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        curve_name: Optional[str] = None,
        paper_type: Optional[str] = None,
        chemistry: Optional[str] = None,
        generate_curve: bool = True,
        curve_type: Optional[CurveType] = None,
    ) -> WedgeAnalysisResult:
        """
        Perform complete step wedge analysis.

        Args:
            image: Input image (array, PIL Image, or path).
            curve_name: Name for generated curve.
            paper_type: Paper type for metadata.
            chemistry: Chemistry description for metadata.
            generate_curve: Whether to generate calibration curve.
            curve_type: Type of curve to generate.

        Returns:
            WedgeAnalysisResult with all analysis data.
        """
        result = WedgeAnalysisResult(
            tablet_type=self.config.tablet_type,
            config=self.config,
        )

        # Handle path input
        if isinstance(image, (str, Path)):
            result.source_path = Path(image)

        # Step 1: Detection and extraction
        try:
            reader = self._get_reader()
            tablet_result = reader.read(image)
            result.extraction = tablet_result.extraction
            result.detection_success = True
        except Exception as e:
            result.detection_success = False
            result.quality = QualityAssessment(
                grade=QualityGrade.FAILED,
                score=0.0,
                density_range_score=0.0,
                uniformity_score=0.0,
                monotonicity_score=0.0,
                detection_confidence=0.0,
                signal_to_noise=0.0,
                warnings=[
                    AnalysisWarning(
                        level=AnalysisWarningLevel.ERROR,
                        code="DETECTION_FAILED",
                        message=f"Step wedge detection failed: {str(e)}",
                        suggestion="Ensure the step wedge is clearly visible and properly aligned.",
                    )
                ],
            )
            return result

        # Step 2: Extract densities
        raw_densities = result.extraction.get_densities()
        if not raw_densities:
            result.quality = QualityAssessment(
                grade=QualityGrade.FAILED,
                score=0.0,
                density_range_score=0.0,
                uniformity_score=0.0,
                monotonicity_score=0.0,
                detection_confidence=result.extraction.overall_quality,
                signal_to_noise=0.0,
                warnings=[
                    AnalysisWarning(
                        level=AnalysisWarningLevel.ERROR,
                        code="NO_DENSITIES",
                        message="No density values could be extracted.",
                        suggestion="Check image quality and step wedge visibility.",
                    )
                ],
            )
            return result

        result.raw_densities = raw_densities.copy()
        num_patches = len(raw_densities)
        result.input_values = [i / (num_patches - 1) for i in range(num_patches)]

        # Step 3: Process and correct densities
        processed_densities = self._process_densities(raw_densities, result)
        result.densities = processed_densities
        result.corrected_densities = processed_densities.copy()

        # Step 4: Calculate statistics
        result.dmin = float(np.min(processed_densities))
        result.dmax = float(np.max(processed_densities))
        result.density_range = result.dmax - result.dmin

        # Step 5: Quality assessment
        result.quality = self._assess_quality(result)

        # Step 6: Generate curve if requested
        if generate_curve and result.quality.grade != QualityGrade.FAILED:
            curve_type = curve_type or self.config.default_curve_type
            result.curve = self._generate_curve(
                result,
                curve_name=curve_name,
                paper_type=paper_type,
                chemistry=chemistry,
                curve_type=curve_type,
            )
            result.curve_generated = result.curve is not None

        return result

    def _process_densities(
        self, densities: list[float], result: WedgeAnalysisResult
    ) -> list[float]:
        """Process and correct density values."""
        if len(densities) < 2:
            return densities.copy() if densities else []

        processed = np.array(densities, dtype=float)

        # Outlier rejection
        if self.config.outlier_rejection:
            processed = self._reject_outliers(processed, result)

        # Fix reversals if enabled
        if self.config.auto_fix_reversals:
            processed = self._fix_reversals(processed, result)

        return processed.tolist()

    def _reject_outliers(
        self, densities: np.ndarray, result: WedgeAnalysisResult
    ) -> np.ndarray:
        """Reject outlier density values using MAD."""
        if len(densities) < 5:
            return densities

        # Calculate expected trend
        expected = np.linspace(densities[0], densities[-1], len(densities))
        deviations = densities - expected

        # MAD (Median Absolute Deviation)
        median_dev = np.median(np.abs(deviations - np.median(deviations)))
        if median_dev > 0:
            mad_score = np.abs(deviations - np.median(deviations)) / median_dev
            outlier_mask = mad_score > self.config.outlier_threshold

            if np.any(outlier_mask):
                outlier_indices = np.where(outlier_mask)[0].tolist()
                # Interpolate outliers
                densities = densities.copy()
                for idx in outlier_indices:
                    if idx > 0 and idx < len(densities) - 1:
                        densities[idx] = (densities[idx - 1] + densities[idx + 1]) / 2

        return densities

    def _fix_reversals(
        self, densities: np.ndarray, result: WedgeAnalysisResult
    ) -> np.ndarray:
        """Fix any reversals in density progression."""
        densities = densities.copy()
        reversals_fixed = []

        # Determine expected direction (should be increasing)
        if densities[-1] < densities[0]:
            # Reverse the array temporarily
            densities = densities[::-1]
            was_reversed = True
        else:
            was_reversed = False

        # Fix reversals
        for i in range(1, len(densities)):
            if densities[i] < densities[i - 1] - self.config.max_reversal_tolerance:
                reversals_fixed.append(i if not was_reversed else len(densities) - 1 - i)
                densities[i] = densities[i - 1]

        if was_reversed:
            densities = densities[::-1]

        return densities

    def _assess_quality(self, result: WedgeAnalysisResult) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        warnings: list[AnalysisWarning] = []
        recommendations: list[str] = []

        densities = np.array(result.densities)
        num_patches = len(densities)

        # 1. Density range score
        density_range = result.density_range or 0
        if density_range >= 2.0:
            density_range_score = 100.0
        elif density_range >= self.config.min_density_range:
            density_range_score = 60 + 40 * (density_range - self.config.min_density_range) / (2.0 - self.config.min_density_range)
        else:
            density_range_score = max(0, 60 * density_range / self.config.min_density_range)
            warnings.append(
                AnalysisWarning(
                    level=AnalysisWarningLevel.WARNING,
                    code="LOW_DENSITY_RANGE",
                    message=f"Density range ({density_range:.2f}) is below recommended minimum ({self.config.min_density_range:.2f}).",
                    suggestion="Consider increasing exposure time or using a more concentrated sensitizer.",
                )
            )
            recommendations.append("Increase exposure time to extend density range")

        # 2. Dmin check
        dmin = result.dmin or 0
        if dmin <= self.config.max_dmin:
            dmin_penalty = 0
        else:
            dmin_penalty = min(20, (dmin - self.config.max_dmin) * 100)
            warnings.append(
                AnalysisWarning(
                    level=AnalysisWarningLevel.WARNING,
                    code="HIGH_DMIN",
                    message=f"Dmin ({dmin:.3f}) is above expected ({self.config.max_dmin:.3f}).",
                    suggestion="Check for chemical fog, light leaks, or clearing issues.",
                )
            )
            recommendations.append("Review clearing process to reduce paper staining")

        # 3. Dmax check
        dmax = result.dmax or 0
        if dmax >= self.config.min_dmax:
            dmax_penalty = 0
        else:
            dmax_penalty = min(15, (self.config.min_dmax - dmax) * 30)
            warnings.append(
                AnalysisWarning(
                    level=AnalysisWarningLevel.INFO,
                    code="LOW_DMAX",
                    message=f"Dmax ({dmax:.3f}) is below typical ({self.config.min_dmax:.3f}).",
                    suggestion="May be normal for pure platinum or low contrast papers.",
                )
            )

        # 4. Uniformity score (from extraction)
        if result.extraction and result.extraction.patches:
            uniformities = [p.uniformity for p in result.extraction.patches]
            uniformity_score = float(np.mean(uniformities)) * 100
            if uniformity_score < self.config.uniformity_threshold * 100:
                warnings.append(
                    AnalysisWarning(
                        level=AnalysisWarningLevel.WARNING,
                        code="LOW_UNIFORMITY",
                        message=f"Patch uniformity ({uniformity_score:.1f}%) is below threshold.",
                        suggestion="Check for uneven coating, brush strokes, or scanning artifacts.",
                    )
                )
        else:
            uniformity_score = 80.0  # Default

        # 5. Monotonicity score
        diffs = np.diff(densities)
        reversals = np.sum(diffs < -self.config.max_reversal_tolerance)
        if reversals == 0:
            monotonicity_score = 100.0
        else:
            monotonicity_score = max(0, 100 - reversals * 20)
            if reversals > 0:
                reversal_indices = np.where(diffs < -self.config.max_reversal_tolerance)[0] + 1
                warnings.append(
                    AnalysisWarning(
                        level=AnalysisWarningLevel.WARNING,
                        code="DENSITY_REVERSAL",
                        message=f"Found {reversals} density reversal(s) in the step wedge.",
                        suggestion="This may indicate solarization or measurement error.",
                        affected_patches=reversal_indices.tolist(),
                    )
                )

        # 6. Detection confidence
        if result.extraction:
            detection_confidence = result.extraction.overall_quality * 100
        else:
            detection_confidence = 50.0

        # 7. Signal-to-noise estimate
        if num_patches > 3:
            # Estimate from local variance
            expected_progression = np.linspace(densities[0], densities[-1], num_patches)
            noise = densities - expected_progression
            signal = result.density_range or 1.0
            noise_level = float(np.std(noise))
            snr = signal / max(noise_level, 0.001)
            signal_to_noise = min(100, snr * 10)
        else:
            signal_to_noise = 50.0

        # Calculate overall score
        weights = {
            "density_range": 0.30,
            "uniformity": 0.20,
            "monotonicity": 0.25,
            "detection": 0.15,
            "snr": 0.10,
        }

        overall_score = (
            weights["density_range"] * density_range_score
            + weights["uniformity"] * uniformity_score
            + weights["monotonicity"] * monotonicity_score
            + weights["detection"] * detection_confidence
            + weights["snr"] * signal_to_noise
            - dmin_penalty
            - dmax_penalty
        )
        overall_score = max(0, min(100, overall_score))

        # Determine grade
        if overall_score >= 85:
            grade = QualityGrade.EXCELLENT
        elif overall_score >= 70:
            grade = QualityGrade.GOOD
        elif overall_score >= 50:
            grade = QualityGrade.ACCEPTABLE
        elif overall_score >= 30:
            grade = QualityGrade.POOR
        else:
            grade = QualityGrade.FAILED

        # Add grade-specific recommendations
        if grade == QualityGrade.EXCELLENT:
            recommendations.append("Results are excellent - proceed with curve generation")
        elif grade == QualityGrade.GOOD:
            recommendations.append("Results are good - generated curve should work well")
        elif grade == QualityGrade.ACCEPTABLE:
            recommendations.append("Consider re-running analysis with improved conditions")
        elif grade in (QualityGrade.POOR, QualityGrade.FAILED):
            recommendations.append("Strongly recommend re-exposing and re-scanning the step wedge")

        return QualityAssessment(
            grade=grade,
            score=overall_score,
            density_range_score=density_range_score,
            uniformity_score=uniformity_score,
            monotonicity_score=monotonicity_score,
            detection_confidence=detection_confidence,
            signal_to_noise=signal_to_noise,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _generate_curve(
        self,
        result: WedgeAnalysisResult,
        curve_name: Optional[str] = None,
        paper_type: Optional[str] = None,
        chemistry: Optional[str] = None,
        curve_type: CurveType = CurveType.LINEAR,
    ) -> Optional[CurveData]:
        """Generate calibration curve from analysis result."""
        if not result.densities:
            return None

        settings = get_settings()

        # Create curve generator with custom settings
        generator = CurveGenerator(settings.curves)

        # Generate curve name
        if not curve_name:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            curve_name = f"Calibration_{timestamp}"

        try:
            curve = generator.generate(
                result.densities,
                curve_type=curve_type,
                name=curve_name,
                paper_type=paper_type,
                chemistry=chemistry,
            )

            # Link to source extraction
            if result.extraction:
                curve.source_extraction_id = result.extraction.id

            return curve

        except Exception:
            return None

    def analyze_from_densities(
        self,
        densities: list[float],
        curve_name: Optional[str] = None,
        paper_type: Optional[str] = None,
        chemistry: Optional[str] = None,
        generate_curve: bool = True,
        curve_type: Optional[CurveType] = None,
    ) -> WedgeAnalysisResult:
        """
        Analyze from pre-extracted density values.

        Args:
            densities: List of density values (Dmin to Dmax).
            curve_name: Name for generated curve.
            paper_type: Paper type for metadata.
            chemistry: Chemistry description.
            generate_curve: Whether to generate calibration curve.
            curve_type: Type of curve to generate.

        Returns:
            WedgeAnalysisResult with analysis data.
        """
        result = WedgeAnalysisResult(
            tablet_type=self.config.tablet_type,
            config=self.config,
            detection_success=True,
        )

        # Handle edge cases for empty or single-element lists
        if not densities:
            result.detection_success = False
            result.densities = []
            result.quality = QualityAssessment(
                grade=QualityGrade.FAILED,
                score=0.0,
                density_range_score=0.0,
                uniformity_score=0.0,
                monotonicity_score=0.0,
                detection_confidence=0.0,
                signal_to_noise=0.0,
                warnings=[
                    AnalysisWarning(
                        level=AnalysisWarningLevel.ERROR,
                        code="NO_DENSITIES",
                        message="No density values provided.",
                    )
                ],
            )
            return result

        # Store raw densities
        result.raw_densities = densities.copy()
        num_patches = len(densities)
        if num_patches > 1:
            result.input_values = [i / (num_patches - 1) for i in range(num_patches)]
        else:
            result.input_values = [0.0]

        # Process densities
        processed_densities = self._process_densities(densities, result)
        result.densities = processed_densities
        result.corrected_densities = processed_densities.copy()

        # Calculate statistics
        result.dmin = float(np.min(processed_densities))
        result.dmax = float(np.max(processed_densities))
        result.density_range = result.dmax - result.dmin

        # Quality assessment
        result.quality = self._assess_quality(result)

        # Generate curve if requested
        if generate_curve and result.quality.grade != QualityGrade.FAILED:
            curve_type = curve_type or self.config.default_curve_type
            result.curve = self._generate_curve(
                result,
                curve_name=curve_name,
                paper_type=paper_type,
                chemistry=chemistry,
                curve_type=curve_type,
            )
            result.curve_generated = result.curve is not None

        return result

    def generate_target_curve(
        self,
        num_points: int,
        curve_type: CurveType = CurveType.LINEAR,
    ) -> TargetCurve:
        """
        Generate a target curve for visualization.

        Args:
            num_points: Number of points in the curve.
            curve_type: Type of target curve.

        Returns:
            TargetCurve for comparison.
        """
        if curve_type == CurveType.LINEAR:
            return TargetCurve.linear(num_points)
        elif curve_type == CurveType.PAPER_WHITE:
            return TargetCurve.paper_white_preserve(num_points)
        elif curve_type == CurveType.AESTHETIC:
            return TargetCurve.aesthetic(num_points)
        else:
            return TargetCurve.linear(num_points)

    @staticmethod
    def get_supported_tablet_types() -> list[TabletType]:
        """Get list of supported tablet types."""
        return list(TabletType)

    @staticmethod
    def get_default_config() -> WedgeAnalysisConfig:
        """Get default configuration."""
        return WedgeAnalysisConfig()
