"""
Pydantic response models for FastAPI endpoints.

These models ensure type safety and proper OpenAPI schema generation.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Root & Health Endpoints
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")


class RootResponse(BaseModel):
    """Root endpoint response."""

    message: str = Field(..., description="Service description")
    version: str = Field(..., description="API version")


# ============================================================================
# Analysis Endpoints
# ============================================================================


class DensityAnalysis(BaseModel):
    """Analysis of density measurements."""

    is_monotonic: bool = Field(..., description="Whether density values are monotonically increasing")
    max_error: float = Field(..., description="Maximum deviation from ideal curve")
    rms_error: float = Field(..., description="Root mean square error")


class AnalyzeResponse(BaseModel):
    """Response from density analysis."""

    dmin: float = Field(..., ge=0.0, description="Minimum density")
    dmax: float = Field(..., ge=0.0, description="Maximum density")
    range: float = Field(..., ge=0.0, description="Density range")
    is_monotonic: bool = Field(..., description="Whether values are monotonic")
    max_error: float = Field(..., description="Maximum error")
    rms_error: float = Field(..., description="RMS error")
    suggestions: list[str] = Field(default_factory=list, description="Adjustment suggestions")


# ============================================================================
# Scan Upload Endpoints
# ============================================================================


class ScanUploadResponse(BaseModel):
    """Response from step tablet scan upload."""

    success: bool = Field(..., description="Whether upload was successful")
    extraction_id: str = Field(..., description="Unique extraction ID")
    original_filename: str = Field(..., description="Original filename")
    num_patches: int = Field(..., ge=0, description="Number of patches detected")
    densities: list[float] = Field(..., description="Extracted density values")
    dmin: float | None = Field(default=None, description="Minimum density")
    dmax: float | None = Field(default=None, description="Maximum density")
    range: float | None = Field(default=None, description="Density range")
    quality: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")


# ============================================================================
# Curve Endpoints
# ============================================================================


class CurveGenerateResponse(BaseModel):
    """Response from curve generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    num_points: int = Field(..., ge=2, description="Number of points in curve")
    input_values: list[float] = Field(..., description="Sample input values")
    output_values: list[float] = Field(..., description="Sample output values")


class CurveModifyResponse(BaseModel):
    """Response from curve modification."""

    success: bool = Field(..., description="Whether modification succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    adjustment_applied: str = Field(..., description="Type of adjustment applied")
    input_values: list[float] = Field(..., description="Modified input values")
    output_values: list[float] = Field(..., description="Modified output values")


class CurveSmoothResponse(BaseModel):
    """Response from curve smoothing."""

    success: bool = Field(..., description="Whether smoothing succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    method_applied: str = Field(..., description="Smoothing method used")
    input_values: list[float] = Field(..., description="Smoothed input values")
    output_values: list[float] = Field(..., description="Smoothed output values")


class CurveBlendResponse(BaseModel):
    """Response from curve blending."""

    success: bool = Field(..., description="Whether blending succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    mode_applied: str = Field(..., description="Blending mode used")
    input_values: list[float] = Field(..., description="Blended input values")
    output_values: list[float] = Field(..., description="Blended output values")


class CurveEnhanceResponse(BaseModel):
    """Response from AI curve enhancement."""

    success: bool = Field(..., description="Whether enhancement succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    goal: str = Field(..., description="Enhancement goal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in enhancement")
    analysis: str = Field(..., description="Analysis of the curve")
    changes_made: list[str] = Field(default_factory=list, description="List of changes made")
    input_values: list[float] = Field(..., description="Enhanced input values")
    output_values: list[float] = Field(..., description="Enhanced output values")


class CurveRetrieveResponse(BaseModel):
    """Response from curve retrieval."""

    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    curve_type: str | None = Field(default=None, description="Type of curve")
    paper_type: str | None = Field(default=None, description="Paper type")
    input_values: list[float] = Field(..., description="Input values")
    output_values: list[float] = Field(..., description="Output values")
    notes: str | None = Field(default=None, description="Notes about the curve")


class CurveMonotonicityResponse(BaseModel):
    """Response from monotonicity enforcement."""

    success: bool = Field(..., description="Whether enforcement succeeded")
    curve_id: str = Field(..., description="Unique curve ID")
    name: str = Field(..., description="Curve name")
    input_values: list[float] = Field(..., description="Input values")
    output_values: list[float] = Field(..., description="Output values")


# ============================================================================
# Quad File Endpoints
# ============================================================================


class CurveDataSample(BaseModel):
    """Sample of curve data."""

    input_values: list[float] = Field(..., description="Input values")
    output_values: list[float] = Field(..., description="Output values")


class QuadUploadResponse(BaseModel):
    """Response from quad file upload."""

    success: bool = Field(..., description="Whether upload succeeded")
    profile_name: str = Field(..., description="Profile name")
    resolution: int = Field(..., ge=0, description="Resolution")
    ink_limit: float | None = Field(default=None, description="Ink limit")
    media_type: str | None = Field(default=None, description="Media type")
    all_channels: list[str] = Field(default_factory=list, description="All available channels")
    active_channels: list[str] = Field(default_factory=list, description="Active channels")
    curve_id: str | None = Field(default=None, description="ID of extracted curve")
    curve_data: CurveDataSample | None = Field(default=None, description="Sample curve data")
    summary: str | None = Field(default=None, description="Profile summary")


class QuadParseResponse(BaseModel):
    """Response from quad content parsing."""

    success: bool = Field(..., description="Whether parsing succeeded")
    profile_name: str = Field(..., description="Profile name")
    active_channels: list[str] = Field(default_factory=list, description="Active channels")
    curve_id: str | None = Field(default=None, description="ID of extracted curve")
    curve_data: CurveDataSample | None = Field(default=None, description="Sample curve data")


# ============================================================================
# Calibration Endpoints
# ============================================================================


class CalibrationRecord(BaseModel):
    """Summary of a calibration record."""

    id: str = Field(..., description="Unique calibration ID")
    paper_type: str = Field(..., description="Paper type")
    exposure_time: float = Field(..., ge=0.0, description="Exposure time in seconds")
    metal_ratio: float = Field(..., ge=0.0, le=1.0, description="Platinum to palladium ratio")
    timestamp: str = Field(..., description="ISO-format timestamp")
    dmax: float = Field(..., ge=0.0, description="Maximum density achieved")


class ListCalibrationsResponse(BaseModel):
    """Response from calibration listing."""

    count: int = Field(..., ge=0, description="Number of records")
    records: list[CalibrationRecord] = Field(..., description="Calibration records")


class CreateCalibrationResponse(BaseModel):
    """Response from calibration creation."""

    success: bool = Field(..., description="Whether creation succeeded")
    id: str = Field(..., description="Unique calibration ID")
    message: str = Field(..., description="Confirmation message")


# ============================================================================
# Chat Endpoints
# ============================================================================


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    response: str = Field(..., description="AI assistant response")


class RecipeResponse(BaseModel):
    """Response from recipe suggestion."""

    response: str = Field(..., description="Recipe recommendation")


class TroubleshootResponse(BaseModel):
    """Response from troubleshooting endpoint."""

    response: str = Field(..., description="Troubleshooting advice")


# ============================================================================
# Statistics Endpoints
# ============================================================================


class StatisticsResponse(BaseModel):
    """Response from statistics endpoint."""

    total_calibrations: int = Field(..., ge=0, description="Total calibration records")
    unique_papers: int = Field(..., ge=0, description="Number of unique paper types")
    avg_exposure_time: float = Field(..., ge=0.0, description="Average exposure time")
    data: dict[str, Any] = Field(default_factory=dict, description="Additional statistics")


# ============================================================================
# Error Responses
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error_code: str = Field(..., description="Machine-readable error code")
    detail: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., ge=400, le=599, description="HTTP status code")
