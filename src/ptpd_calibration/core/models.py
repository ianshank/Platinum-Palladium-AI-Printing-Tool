"""
Core data models for PTPD Calibration System.

All models use Pydantic for validation and serialization.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    CurveType,
    DeveloperType,
    MeasurementUnit,
    PaperSizing,
)


class PatchData(BaseModel):
    """Data for a single patch in a step tablet."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(..., ge=0, description="Patch index (0 = lightest)")
    position: tuple[int, int, int, int] = Field(
        ..., description="Bounding box (x, y, width, height)"
    )
    rgb_mean: tuple[float, float, float] = Field(..., description="Mean RGB values (0-255)")
    rgb_std: tuple[float, float, float] = Field(..., description="RGB standard deviations")
    lab_mean: Optional[tuple[float, float, float]] = Field(
        default=None, description="Mean L*a*b* values"
    )
    density: Optional[float] = Field(default=None, ge=0.0, description="Calculated density")
    uniformity: float = Field(default=1.0, ge=0.0, le=1.0, description="Patch uniformity score")

    @field_validator("rgb_mean", "rgb_std", mode="before")
    @classmethod
    def convert_to_tuple(cls, v: Any) -> tuple[float, float, float]:
        """Convert various inputs to tuple of floats."""
        if isinstance(v, (list, np.ndarray)):
            return tuple(float(x) for x in v[:3])
        return v


class DensityMeasurement(BaseModel):
    """A single density measurement with metadata."""

    step: int = Field(..., ge=0, description="Step number")
    input_value: float = Field(..., ge=0.0, le=1.0, description="Input value (0-1)")
    density: float = Field(..., ge=0.0, description="Measured density")
    lab: Optional[tuple[float, float, float]] = Field(default=None, description="L*a*b* values")
    unit: MeasurementUnit = Field(default=MeasurementUnit.VISUAL_DENSITY)


class ExtractionResult(BaseModel):
    """Complete extraction result from a step tablet scan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Image info
    source_path: Optional[Path] = Field(default=None)
    image_size: tuple[int, int] = Field(..., description="Image dimensions (width, height)")
    image_dpi: Optional[int] = Field(default=None)

    # Detection info
    tablet_bounds: tuple[int, int, int, int] = Field(
        ..., description="Tablet bounding box (x, y, w, h)"
    )
    rotation_angle: float = Field(default=0.0, description="Applied rotation in degrees")
    orientation: str = Field(default="horizontal", description="Tablet orientation")

    # Patch data
    patches: list[PatchData] = Field(default_factory=list)
    num_patches: int = Field(default=0, ge=0)

    # Paper base reference
    paper_base_rgb: Optional[tuple[float, float, float]] = Field(default=None)
    paper_base_density: Optional[float] = Field(default=None)

    # Quality metrics
    overall_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patches(self) -> "ExtractionResult":
        """Ensure num_patches matches patches list."""
        if self.patches:
            self.num_patches = len(self.patches)
        return self

    def get_densities(self) -> list[float]:
        """Get list of density values from patches."""
        return [p.density for p in self.patches if p.density is not None]

    def get_rgb_values(self) -> list[tuple[float, float, float]]:
        """Get list of RGB values from patches."""
        return [p.rgb_mean for p in self.patches]

    @property
    def dmin(self) -> Optional[float]:
        """Minimum density (paper base)."""
        densities = self.get_densities()
        return min(densities) if densities else None

    @property
    def dmax(self) -> Optional[float]:
        """Maximum density."""
        densities = self.get_densities()
        return max(densities) if densities else None

    @property
    def density_range(self) -> Optional[float]:
        """Total density range."""
        if self.dmin is not None and self.dmax is not None:
            return self.dmax - self.dmin
        return None


class StepTabletResult(BaseModel):
    """High-level result from step tablet reading."""

    extraction: ExtractionResult
    measurements: list[DensityMeasurement] = Field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary string."""
        ext = self.extraction
        dmin = ext.dmin or 0.0
        dmax = ext.dmax or 0.0
        dr = ext.density_range or 0.0
        return (
            f"Steps: {ext.num_patches}, "
            f"Dmin: {dmin:.2f}, Dmax: {dmax:.2f}, "
            f"Range: {dr:.2f}, Quality: {ext.overall_quality:.0%}"
        )

    def to_csv(self) -> str:
        """Export measurements to CSV format."""
        lines = ["step,input,density"]
        for m in self.measurements:
            lines.append(f"{m.step},{m.input_value:.4f},{m.density:.4f}")
        return "\n".join(lines)


class CurveData(BaseModel):
    """Calibration curve data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=256)
    created_at: datetime = Field(default_factory=datetime.now)

    # Curve type and metadata
    curve_type: CurveType = Field(default=CurveType.LINEAR)
    paper_type: Optional[str] = Field(default=None)
    chemistry: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)

    # Input/output points
    input_values: list[float] = Field(..., min_length=2)
    output_values: list[float] = Field(..., min_length=2)

    # Source calibration
    source_extraction_id: Optional[UUID] = Field(default=None)
    target_curve_type: Optional[str] = Field(default=None)

    @field_validator("input_values", "output_values", mode="before")
    @classmethod
    def convert_to_list(cls, v: Any) -> list[float]:
        """Convert arrays to lists."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v)

    @model_validator(mode="after")
    def validate_curve_lengths(self) -> "CurveData":
        """Ensure input and output have same length."""
        if len(self.input_values) != len(self.output_values):
            raise ValueError("Input and output values must have the same length")
        return self

    def interpolate(self, x: float) -> float:
        """Interpolate a single value."""
        from scipy.interpolate import PchipInterpolator

        interp = PchipInterpolator(self.input_values, self.output_values)
        return float(interp(x))

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return np.array(self.input_values), np.array(self.output_values)

    def save(self, path: Path, format: str = "json") -> None:
        """Save curve to file."""
        from ptpd_calibration.curves import save_curve

        save_curve(self, path, format)


class PaperProfile(BaseModel):
    """Profile for a specific paper type."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    manufacturer: Optional[str] = Field(default=None)
    weight_gsm: Optional[int] = Field(default=None, ge=50, le=1000)
    sizing: PaperSizing = Field(default=PaperSizing.INTERNAL)

    # Characteristics
    base_density: Optional[float] = Field(default=None, ge=0.0)
    max_density: Optional[float] = Field(default=None, ge=0.0)
    recommended_exposure_factor: float = Field(default=1.0, ge=0.1, le=10.0)

    # Notes and metadata
    notes: Optional[str] = Field(default=None)
    calibration_ids: list[UUID] = Field(default_factory=list)


class CalibrationRecord(BaseModel):
    """Complete calibration session record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    name: Optional[str] = Field(default=None)

    # Paper info
    paper_type: str = Field(..., min_length=1)
    paper_weight: Optional[int] = Field(default=None, ge=50, le=1000)
    paper_sizing: PaperSizing = Field(default=PaperSizing.INTERNAL)

    # Chemistry
    chemistry_type: ChemistryType = Field(default=ChemistryType.PLATINUM_PALLADIUM)
    metal_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Pt ratio (0=Pd, 1=Pt)")
    contrast_agent: ContrastAgent = Field(default=ContrastAgent.NONE)
    contrast_amount: float = Field(default=0.0, ge=0.0)
    developer: DeveloperType = Field(default=DeveloperType.POTASSIUM_OXALATE)

    # Process parameters
    exposure_time: float = Field(..., ge=0.0, description="Exposure time in seconds")
    uv_source: Optional[str] = Field(default=None)
    humidity: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    temperature: Optional[float] = Field(default=None, ge=-20.0, le=50.0)

    # Results
    measured_densities: list[float] = Field(default_factory=list)
    extraction_id: Optional[UUID] = Field(default=None)
    curve_id: Optional[UUID] = Field(default=None)

    # Metadata
    notes: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)

    def get_feature_vector(self) -> list[float]:
        """Convert to feature vector for ML."""
        return [
            self.metal_ratio,
            float(self.contrast_agent != ContrastAgent.NONE),
            self.contrast_amount,
            self.exposure_time,
            self.humidity or 50.0,
            self.temperature or 21.0,
        ]
