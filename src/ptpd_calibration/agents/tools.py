"""
Tool definitions for the calibration agent.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ptpd_calibration.core.models import CalibrationRecord, CurveData
from ptpd_calibration.ml.database import CalibrationDatabase


class ToolCategory(str, Enum):
    """Categories of tools."""

    ANALYSIS = "analysis"
    DATABASE = "database"
    PREDICTION = "prediction"
    CURVES = "curves"
    PLANNING = "planning"
    MEMORY = "memory"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: Any = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if not self.success:
            return f"Error: {self.error}"

        if isinstance(self.data, (dict, list)):
            return json.dumps(self.data, indent=2, default=str)
        return str(self.data)


@dataclass
class Tool:
    """Definition of a tool for the agent."""

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., ToolResult]
    category: ToolCategory = ToolCategory.ANALYSIS

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            result = self.handler(**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> list[Tool]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def to_anthropic_format(self) -> list[dict]:
        """Convert all tools to Anthropic format."""
        return [t.to_anthropic_format() for t in self._tools.values()]


def create_calibration_tools(
    database: Optional[CalibrationDatabase] = None,
    predictor: Optional[Any] = None,
) -> ToolRegistry:
    """
    Create the standard set of calibration tools.

    Args:
        database: CalibrationDatabase for data access.
        predictor: CurvePredictor for ML predictions.

    Returns:
        ToolRegistry with all calibration tools.
    """
    registry = ToolRegistry()

    # Analysis tools
    registry.register(
        Tool(
            name="analyze_densities",
            description="Analyze a list of density measurements and provide quality metrics",
            parameters=[
                ToolParameter(
                    name="densities",
                    type="array",
                    description="List of density values from step tablet",
                ),
            ],
            handler=_analyze_densities,
            category=ToolCategory.ANALYSIS,
        )
    )

    registry.register(
        Tool(
            name="compare_calibrations",
            description="Compare two calibration records",
            parameters=[
                ToolParameter(name="id1", type="string", description="First calibration ID"),
                ToolParameter(name="id2", type="string", description="Second calibration ID"),
            ],
            handler=lambda id1, id2: _compare_calibrations(database, id1, id2),
            category=ToolCategory.ANALYSIS,
        )
    )

    # Database tools
    if database:
        registry.register(
            Tool(
                name="search_calibrations",
                description="Search calibration database",
                parameters=[
                    ToolParameter(
                        name="paper_type",
                        type="string",
                        description="Paper type to search for",
                        required=False,
                    ),
                    ToolParameter(
                        name="chemistry_type",
                        type="string",
                        description="Chemistry type",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results",
                        required=False,
                        default=10,
                    ),
                ],
                handler=lambda **kwargs: _search_calibrations(database, **kwargs),
                category=ToolCategory.DATABASE,
            )
        )

        registry.register(
            Tool(
                name="get_calibration",
                description="Get a specific calibration record by ID",
                parameters=[
                    ToolParameter(name="id", type="string", description="Calibration ID"),
                ],
                handler=lambda id: _get_calibration(database, id),
                category=ToolCategory.DATABASE,
            )
        )

        registry.register(
            Tool(
                name="save_calibration",
                description="Save a new calibration record",
                parameters=[
                    ToolParameter(
                        name="paper_type", type="string", description="Paper type"
                    ),
                    ToolParameter(
                        name="exposure_time",
                        type="number",
                        description="Exposure time in seconds",
                    ),
                    ToolParameter(
                        name="metal_ratio",
                        type="number",
                        description="Platinum ratio (0-1)",
                    ),
                    ToolParameter(
                        name="densities",
                        type="array",
                        description="Measured densities",
                    ),
                    ToolParameter(
                        name="notes",
                        type="string",
                        description="Notes",
                        required=False,
                    ),
                ],
                handler=lambda **kwargs: _save_calibration(database, **kwargs),
                category=ToolCategory.DATABASE,
            )
        )

    # Prediction tools
    if predictor:
        registry.register(
            Tool(
                name="predict_response",
                description="Predict density response for given parameters",
                parameters=[
                    ToolParameter(
                        name="paper_type", type="string", description="Paper type"
                    ),
                    ToolParameter(
                        name="metal_ratio",
                        type="number",
                        description="Platinum ratio (0-1)",
                    ),
                    ToolParameter(
                        name="exposure_time",
                        type="number",
                        description="Exposure time in seconds",
                    ),
                    ToolParameter(
                        name="contrast_amount",
                        type="number",
                        description="Contrast agent drops",
                        required=False,
                        default=0,
                    ),
                ],
                handler=lambda **kwargs: _predict_response(predictor, **kwargs),
                category=ToolCategory.PREDICTION,
            )
        )

    # Curve tools
    registry.register(
        Tool(
            name="generate_curve",
            description="Generate a linearization curve from densities",
            parameters=[
                ToolParameter(
                    name="densities",
                    type="array",
                    description="Measured densities",
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Curve name",
                ),
                ToolParameter(
                    name="curve_type",
                    type="string",
                    description="Curve type",
                    enum=["linear", "paper_white", "aesthetic"],
                    default="linear",
                    required=False,
                ),
            ],
            handler=_generate_curve,
            category=ToolCategory.CURVES,
        )
    )

    registry.register(
        Tool(
            name="suggest_parameters",
            description="Suggest printing parameters for desired characteristics",
            parameters=[
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Target paper",
                ),
                ToolParameter(
                    name="target_dmax",
                    type="number",
                    description="Desired maximum density",
                    required=False,
                    default=2.0,
                ),
                ToolParameter(
                    name="tone",
                    type="string",
                    description="Desired tone (warm, neutral, cool)",
                    enum=["warm", "neutral", "cool"],
                    required=False,
                    default="neutral",
                ),
            ],
            handler=_suggest_parameters,
            category=ToolCategory.PLANNING,
        )
    )

    registry.register(
        Tool(
            name="create_test_plan",
            description="Create a test plan for calibrating a new paper",
            parameters=[
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Paper to calibrate",
                ),
                ToolParameter(
                    name="starting_exposure",
                    type="number",
                    description="Starting exposure time estimate",
                    required=False,
                    default=180,
                ),
            ],
            handler=_create_test_plan,
            category=ToolCategory.PLANNING,
        )
    )

    return registry


# Tool handlers


def _analyze_densities(densities: list[float]) -> ToolResult:
    """Analyze density measurements."""
    from ptpd_calibration.curves.analysis import CurveAnalyzer

    analysis = CurveAnalyzer.analyze_linearity(densities)
    suggestions = CurveAnalyzer.suggest_adjustments(densities)

    return ToolResult(
        success=True,
        data={
            "dmin": min(densities),
            "dmax": max(densities),
            "range": max(densities) - min(densities),
            "num_steps": len(densities),
            "is_monotonic": analysis.is_monotonic,
            "max_error": analysis.max_error,
            "rms_error": analysis.rms_error,
            "suggestions": suggestions,
        },
    )


def _compare_calibrations(
    database: Optional[CalibrationDatabase], id1: str, id2: str
) -> ToolResult:
    """Compare two calibrations."""
    if not database:
        return ToolResult(success=False, error="Database not available")

    from uuid import UUID

    rec1 = database.get_record(UUID(id1))
    rec2 = database.get_record(UUID(id2))

    if not rec1 or not rec2:
        return ToolResult(success=False, error="One or both records not found")

    comparison = {
        "record1": {
            "paper": rec1.paper_type,
            "dmax": max(rec1.measured_densities) if rec1.measured_densities else 0,
        },
        "record2": {
            "paper": rec2.paper_type,
            "dmax": max(rec2.measured_densities) if rec2.measured_densities else 0,
        },
        "differences": {
            "metal_ratio": rec1.metal_ratio - rec2.metal_ratio,
            "exposure": rec1.exposure_time - rec2.exposure_time,
        },
    }

    return ToolResult(success=True, data=comparison)


def _search_calibrations(
    database: CalibrationDatabase,
    paper_type: Optional[str] = None,
    chemistry_type: Optional[str] = None,
    limit: int = 10,
) -> ToolResult:
    """Search calibrations."""
    results = database.query(paper_type=paper_type, chemistry_type=chemistry_type)

    records = []
    for rec in results[:limit]:
        records.append(
            {
                "id": str(rec.id),
                "paper": rec.paper_type,
                "metal_ratio": rec.metal_ratio,
                "exposure": rec.exposure_time,
                "dmax": max(rec.measured_densities) if rec.measured_densities else 0,
            }
        )

    return ToolResult(success=True, data={"count": len(records), "records": records})


def _get_calibration(database: CalibrationDatabase, id: str) -> ToolResult:
    """Get a specific calibration."""
    from uuid import UUID

    record = database.get_record(UUID(id))
    if not record:
        return ToolResult(success=False, error=f"Record {id} not found")

    return ToolResult(success=True, data=record.model_dump(mode="json"))


def _save_calibration(
    database: CalibrationDatabase,
    paper_type: str,
    exposure_time: float,
    metal_ratio: float,
    densities: list[float],
    notes: Optional[str] = None,
) -> ToolResult:
    """Save a new calibration."""
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType

    record = CalibrationRecord(
        paper_type=paper_type,
        exposure_time=exposure_time,
        metal_ratio=metal_ratio,
        measured_densities=densities,
        notes=notes,
        chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
        contrast_agent=ContrastAgent.NONE,
        developer=DeveloperType.POTASSIUM_OXALATE,
    )

    database.add_record(record)

    return ToolResult(
        success=True, data={"id": str(record.id), "message": "Calibration saved"}
    )


def _predict_response(
    predictor: Any,
    paper_type: str,
    metal_ratio: float,
    exposure_time: float,
    contrast_amount: float = 0,
) -> ToolResult:
    """Predict density response."""
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType

    record = CalibrationRecord(
        paper_type=paper_type,
        exposure_time=exposure_time,
        metal_ratio=metal_ratio,
        contrast_amount=contrast_amount,
        chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
        contrast_agent=ContrastAgent.NA2 if contrast_amount > 0 else ContrastAgent.NONE,
        developer=DeveloperType.POTASSIUM_OXALATE,
    )

    try:
        import asyncio

        prediction, uncertainty = asyncio.get_event_loop().run_until_complete(
            predictor.predict(record, return_uncertainty=True)
        )
        return ToolResult(
            success=True,
            data={
                "predicted_densities": prediction,
                "uncertainty": uncertainty,
                "dmax": max(prediction) if prediction else 0,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _generate_curve(
    densities: list[float],
    name: str,
    curve_type: str = "linear",
) -> ToolResult:
    """Generate a linearization curve."""
    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.curves.generator import CurveGenerator

    generator = CurveGenerator()
    ct = CurveType(curve_type)

    curve = generator.generate(densities, curve_type=ct, name=name)

    return ToolResult(
        success=True,
        data={
            "id": str(curve.id),
            "name": curve.name,
            "num_points": len(curve.input_values),
            "curve_type": curve.curve_type.value,
        },
    )


def _suggest_parameters(
    paper_type: str,
    target_dmax: float = 2.0,
    tone: str = "neutral",
) -> ToolResult:
    """Suggest printing parameters."""
    # Base recommendations
    suggestions = {
        "paper_type": paper_type,
        "target_dmax": target_dmax,
    }

    # Adjust metal ratio for tone
    if tone == "warm":
        suggestions["metal_ratio"] = 0.2  # More Pd
        suggestions["metal_description"] = "20% Pt / 80% Pd for warm tones"
    elif tone == "cool":
        suggestions["metal_ratio"] = 0.8  # More Pt
        suggestions["metal_description"] = "80% Pt / 20% Pd for cool tones"
    else:
        suggestions["metal_ratio"] = 0.5
        suggestions["metal_description"] = "50% Pt / 50% Pd for neutral tones"

    # Estimate exposure
    suggestions["exposure_estimate"] = 180.0  # 3 minutes baseline
    if target_dmax > 2.2:
        suggestions["exposure_estimate"] *= 1.3
    elif target_dmax < 1.8:
        suggestions["exposure_estimate"] *= 0.8

    suggestions["contrast_agent"] = "na2"
    suggestions["contrast_drops"] = 5

    return ToolResult(success=True, data=suggestions)


def _create_test_plan(
    paper_type: str,
    starting_exposure: float = 180,
) -> ToolResult:
    """Create a test plan for new paper."""
    plan = {
        "paper_type": paper_type,
        "steps": [
            {
                "step": 1,
                "name": "Exposure bracket",
                "description": "Test 5 exposures around starting point",
                "exposures": [
                    starting_exposure * 0.5,
                    starting_exposure * 0.71,
                    starting_exposure,
                    starting_exposure * 1.41,
                    starting_exposure * 2.0,
                ],
            },
            {
                "step": 2,
                "name": "Metal ratio test",
                "description": "Test 3 metal ratios at optimal exposure",
                "ratios": [0.0, 0.5, 1.0],
            },
            {
                "step": 3,
                "name": "Contrast test",
                "description": "Test contrast agent amounts",
                "amounts": [0, 5, 10],
            },
            {
                "step": 4,
                "name": "Final calibration",
                "description": "Full step tablet with optimal parameters",
            },
        ],
    }

    return ToolResult(success=True, data=plan)


# =============================================================================
# Composite Tools - High-value operations combining multiple steps
# =============================================================================


def _full_calibration(
    densities: list[float],
    paper_type: str,
    name: str,
    curve_type: str = "linear",
    export_format: str = "qtr",
    target_dmax: float = 2.0,
) -> ToolResult:
    """
    Perform full calibration workflow: analyze, generate curve, and prepare export.

    This composite tool combines density analysis, curve generation, and
    quality assessment into a single operation.
    """
    from ptpd_calibration.curves.analysis import CurveAnalyzer
    from ptpd_calibration.core.types import CurveType
    from ptpd_calibration.curves.generator import CurveGenerator

    # Step 1: Analyze densities
    analysis = CurveAnalyzer.analyze_linearity(densities)
    suggestions = CurveAnalyzer.suggest_adjustments(densities)

    # Step 2: Generate curve
    generator = CurveGenerator()
    ct = CurveType(curve_type)
    curve = generator.generate(densities, curve_type=ct, name=name)

    # Step 3: Quality assessment
    dmin = min(densities)
    dmax = max(densities)
    density_range = dmax - dmin

    quality_grade = "excellent"
    if density_range < 1.5:
        quality_grade = "poor"
    elif density_range < 1.8:
        quality_grade = "fair"
    elif density_range < 2.0:
        quality_grade = "good"

    if not analysis.is_monotonic:
        quality_grade = "needs_adjustment"

    return ToolResult(
        success=True,
        data={
            "analysis": {
                "dmin": dmin,
                "dmax": dmax,
                "range": density_range,
                "is_monotonic": analysis.is_monotonic,
                "max_error": analysis.max_error,
                "rms_error": analysis.rms_error,
            },
            "curve": {
                "id": str(curve.id),
                "name": curve.name,
                "num_points": len(curve.input_values),
                "type": curve.curve_type.value,
            },
            "quality": {
                "grade": quality_grade,
                "suggestions": suggestions,
            },
            "export_ready": {
                "format": export_format,
                "paper_type": paper_type,
            },
        },
        metadata={"composite_tool": "full_calibration"},
    )


def _quality_report(
    densities: list[float],
    paper_type: Optional[str] = None,
    expected_dmin: float = 0.1,
    expected_dmax: float = 2.0,
) -> ToolResult:
    """
    Generate comprehensive quality report for calibration data.

    Combines multiple quality checks into a single assessment.
    """
    from ptpd_calibration.curves.analysis import CurveAnalyzer

    # Analyze linearity
    analysis = CurveAnalyzer.analyze_linearity(densities)

    # Calculate metrics
    dmin = min(densities)
    dmax = max(densities)
    density_range = dmax - dmin
    num_steps = len(densities)

    # Calculate step uniformity
    step_sizes = [densities[i + 1] - densities[i] for i in range(len(densities) - 1)]
    avg_step = sum(step_sizes) / len(step_sizes) if step_sizes else 0
    step_uniformity = 1.0 - (max(step_sizes) - min(step_sizes)) / avg_step if avg_step > 0 else 0

    # Quality checks
    checks = {
        "dmin_acceptable": dmin <= expected_dmin * 1.5,
        "dmax_acceptable": dmax >= expected_dmax * 0.9,
        "range_acceptable": density_range >= 1.5,
        "monotonicity": analysis.is_monotonic,
        "step_uniformity": step_uniformity >= 0.7,
        "sufficient_steps": num_steps >= 11,
    }

    # Overall grade
    passed_checks = sum(1 for v in checks.values() if v)
    total_checks = len(checks)
    grade_score = passed_checks / total_checks

    if grade_score >= 0.9:
        overall_grade = "A"
    elif grade_score >= 0.8:
        overall_grade = "B"
    elif grade_score >= 0.7:
        overall_grade = "C"
    elif grade_score >= 0.6:
        overall_grade = "D"
    else:
        overall_grade = "F"

    # Generate recommendations
    recommendations = []
    if not checks["dmin_acceptable"]:
        recommendations.append("Consider reducing exposure time to lower Dmin")
    if not checks["dmax_acceptable"]:
        recommendations.append("Consider increasing exposure time or metal concentration for higher Dmax")
    if not checks["range_acceptable"]:
        recommendations.append("Density range is narrow - review chemistry and exposure")
    if not checks["monotonicity"]:
        recommendations.append("Non-monotonic response detected - check for measurement errors")
    if not checks["step_uniformity"]:
        recommendations.append("Uneven step response - consider curve smoothing")

    return ToolResult(
        success=True,
        data={
            "metrics": {
                "dmin": dmin,
                "dmax": dmax,
                "range": density_range,
                "num_steps": num_steps,
                "step_uniformity": round(step_uniformity, 3),
                "linearity_error": round(analysis.rms_error, 4),
            },
            "checks": checks,
            "grade": {
                "overall": overall_grade,
                "score": round(grade_score, 2),
                "passed": passed_checks,
                "total": total_checks,
            },
            "recommendations": recommendations,
            "paper_type": paper_type,
        },
        metadata={"composite_tool": "quality_report"},
    )


def _troubleshoot_print(
    symptoms: list[str],
    densities: Optional[list[float]] = None,
    paper_type: Optional[str] = None,
    metal_ratio: Optional[float] = None,
    exposure_time: Optional[float] = None,
) -> ToolResult:
    """
    Diagnose print problems based on symptoms and data.

    Analyzes common issues and provides targeted solutions.
    """
    # Symptom-to-cause mapping (configurable knowledge base)
    symptom_causes = {
        "muddy": [
            ("Overexposure", "Reduce exposure time by 20-30%"),
            ("Paper humidity too high", "Condition paper in lower humidity"),
            ("Developer too active", "Dilute developer or reduce development time"),
        ],
        "faded": [
            ("Underexposure", "Increase exposure time by 30-50%"),
            ("Chemistry exhausted", "Prepare fresh coating solution"),
            ("Paper humidity too low", "Humidify paper before coating"),
        ],
        "blocked": [
            ("Shadow blocking", "Reduce exposure or increase negative density range"),
            ("Paper too absorbent", "Consider sizing or different paper"),
            ("Too much contrast agent", "Reduce Na2 by 2-3 drops"),
        ],
        "flat": [
            ("Insufficient exposure", "Increase exposure time"),
            ("Low negative density range", "Adjust digital negative curve"),
            ("Developer exhausted", "Use fresh developer"),
        ],
        "uneven": [
            ("Uneven coating", "Use glass rod or improve brush technique"),
            ("Paper not flat", "Use vacuum frame or glass pressure"),
            ("UV light uneven", "Check light source uniformity"),
        ],
        "staining": [
            ("Incomplete clearing", "Extend clearing bath time"),
            ("Paper residue", "Use distilled water for final rinse"),
            ("Old chemistry", "Replace clearing agents"),
        ],
        "bronzing": [
            ("Over-coating", "Reduce sensitizer volume"),
            ("Too much platinum", "Increase palladium ratio"),
            ("Incomplete wash", "Extend final wash time"),
        ],
    }

    # Analyze symptoms
    diagnoses = []
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for key, causes in symptom_causes.items():
            if key in symptom_lower:
                for cause, solution in causes:
                    diagnoses.append({
                        "symptom": symptom,
                        "likely_cause": cause,
                        "suggested_fix": solution,
                    })

    # Add density-based analysis if available
    density_analysis = None
    if densities:
        dmin = min(densities)
        dmax = max(densities)

        if dmin > 0.15:
            diagnoses.append({
                "symptom": "High Dmin detected",
                "likely_cause": "Fog or staining in highlights",
                "suggested_fix": "Check paper humidity, exposure, and clearing process",
            })

        if dmax < 1.6:
            diagnoses.append({
                "symptom": "Low Dmax detected",
                "likely_cause": "Insufficient density in shadows",
                "suggested_fix": "Increase exposure time or check chemistry freshness",
            })

        density_analysis = {
            "dmin": dmin,
            "dmax": dmax,
            "range": dmax - dmin,
        }

    # Generate priority recommendations
    priority_actions = []
    seen_causes = set()
    for diag in diagnoses[:5]:  # Top 5 priorities
        if diag["likely_cause"] not in seen_causes:
            priority_actions.append(diag["suggested_fix"])
            seen_causes.add(diag["likely_cause"])

    return ToolResult(
        success=True,
        data={
            "diagnoses": diagnoses,
            "priority_actions": priority_actions,
            "density_analysis": density_analysis,
            "context": {
                "paper_type": paper_type,
                "metal_ratio": metal_ratio,
                "exposure_time": exposure_time,
            },
            "confidence": len(diagnoses) / max(len(symptoms), 1),
        },
        metadata={"composite_tool": "troubleshoot_print"},
    )


def _optimize_recipe(
    paper_type: str,
    target_characteristics: Optional[dict] = None,
    database: Optional[CalibrationDatabase] = None,
    current_params: Optional[dict] = None,
) -> ToolResult:
    """
    Optimize printing recipe for desired characteristics.

    Analyzes similar recipes and suggests optimal parameters.
    """
    target = target_characteristics or {}
    target_dmax = target.get("dmax", 2.0)
    target_tone = target.get("tone", "neutral")
    target_contrast = target.get("contrast", "normal")

    # Base recommendations by tone
    tone_params = {
        "warm": {"metal_ratio": 0.2, "description": "20% Pt / 80% Pd"},
        "neutral": {"metal_ratio": 0.5, "description": "50% Pt / 50% Pd"},
        "cool": {"metal_ratio": 0.8, "description": "80% Pt / 20% Pd"},
    }

    # Base recommendations by contrast
    contrast_params = {
        "low": {"na2_drops": 0, "description": "No contrast agent"},
        "normal": {"na2_drops": 5, "description": "Standard contrast"},
        "high": {"na2_drops": 10, "description": "High contrast"},
    }

    tone_rec = tone_params.get(target_tone, tone_params["neutral"])
    contrast_rec = contrast_params.get(target_contrast, contrast_params["normal"])

    # Calculate exposure estimate
    base_exposure = 180.0  # seconds
    if target_dmax > 2.2:
        base_exposure *= 1.3
    elif target_dmax < 1.8:
        base_exposure *= 0.8

    # Adjust for metal ratio
    metal_ratio = tone_rec["metal_ratio"]
    if metal_ratio > 0.6:
        base_exposure *= 1.1  # Platinum needs slightly more exposure

    # Search for similar recipes if database available
    similar_recipes = []
    if database:
        try:
            records = database.query(paper_type=paper_type)
            for rec in records[:3]:
                similar_recipes.append({
                    "id": str(rec.id),
                    "metal_ratio": rec.metal_ratio,
                    "exposure": rec.exposure_time,
                    "dmax": max(rec.measured_densities) if rec.measured_densities else 0,
                })
        except Exception:
            pass

    optimized_recipe = {
        "paper_type": paper_type,
        "metal_ratio": metal_ratio,
        "metal_description": tone_rec["description"],
        "na2_drops": contrast_rec["na2_drops"],
        "contrast_description": contrast_rec["description"],
        "exposure_time": round(base_exposure, 1),
        "target_dmax": target_dmax,
    }

    # Add adjustments if current params provided
    adjustments = []
    if current_params:
        if current_params.get("metal_ratio") != metal_ratio:
            adjustments.append(
                f"Change metal ratio from {current_params.get('metal_ratio')} to {metal_ratio}"
            )
        if current_params.get("exposure_time") != base_exposure:
            adjustments.append(
                f"Adjust exposure from {current_params.get('exposure_time')}s to {base_exposure}s"
            )

    return ToolResult(
        success=True,
        data={
            "optimized_recipe": optimized_recipe,
            "similar_recipes": similar_recipes,
            "adjustments": adjustments,
            "targets": {
                "dmax": target_dmax,
                "tone": target_tone,
                "contrast": target_contrast,
            },
        },
        metadata={"composite_tool": "optimize_recipe"},
    )


def register_composite_tools(registry: ToolRegistry, database: Optional[CalibrationDatabase] = None) -> None:
    """
    Register composite tools with the registry.

    Args:
        registry: The tool registry to add tools to.
        database: Optional calibration database for data access.
    """
    # Full calibration workflow
    registry.register(
        Tool(
            name="full_calibration",
            description="Perform complete calibration: analyze densities, generate curve, assess quality, and prepare export",
            parameters=[
                ToolParameter(
                    name="densities",
                    type="array",
                    description="List of density measurements from step tablet",
                ),
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Type of paper being calibrated",
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Name for the generated curve",
                ),
                ToolParameter(
                    name="curve_type",
                    type="string",
                    description="Type of curve (linear, paper_white, aesthetic)",
                    required=False,
                    default="linear",
                ),
                ToolParameter(
                    name="export_format",
                    type="string",
                    description="Export format (qtr, piezography, csv, json)",
                    required=False,
                    default="qtr",
                ),
                ToolParameter(
                    name="target_dmax",
                    type="number",
                    description="Target maximum density",
                    required=False,
                    default=2.0,
                ),
            ],
            handler=_full_calibration,
            category=ToolCategory.CURVES,
        )
    )

    # Quality report
    registry.register(
        Tool(
            name="quality_report",
            description="Generate comprehensive quality assessment report for calibration data",
            parameters=[
                ToolParameter(
                    name="densities",
                    type="array",
                    description="List of density measurements",
                ),
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Paper type (for context)",
                    required=False,
                ),
                ToolParameter(
                    name="expected_dmin",
                    type="number",
                    description="Expected minimum density",
                    required=False,
                    default=0.1,
                ),
                ToolParameter(
                    name="expected_dmax",
                    type="number",
                    description="Expected maximum density",
                    required=False,
                    default=2.0,
                ),
            ],
            handler=_quality_report,
            category=ToolCategory.ANALYSIS,
        )
    )

    # Troubleshooting
    registry.register(
        Tool(
            name="troubleshoot_print",
            description="Diagnose print problems and suggest solutions based on symptoms",
            parameters=[
                ToolParameter(
                    name="symptoms",
                    type="array",
                    description="List of observed symptoms (e.g., 'muddy shadows', 'faded highlights')",
                ),
                ToolParameter(
                    name="densities",
                    type="array",
                    description="Measured densities if available",
                    required=False,
                ),
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Paper type used",
                    required=False,
                ),
                ToolParameter(
                    name="metal_ratio",
                    type="number",
                    description="Platinum ratio (0-1)",
                    required=False,
                ),
                ToolParameter(
                    name="exposure_time",
                    type="number",
                    description="Exposure time in seconds",
                    required=False,
                ),
            ],
            handler=_troubleshoot_print,
            category=ToolCategory.ANALYSIS,
        )
    )

    # Recipe optimization
    registry.register(
        Tool(
            name="optimize_recipe",
            description="Optimize printing parameters for desired characteristics and paper",
            parameters=[
                ToolParameter(
                    name="paper_type",
                    type="string",
                    description="Target paper type",
                ),
                ToolParameter(
                    name="target_characteristics",
                    type="object",
                    description="Desired characteristics: dmax, tone (warm/neutral/cool), contrast (low/normal/high)",
                    required=False,
                ),
                ToolParameter(
                    name="current_params",
                    type="object",
                    description="Current parameters to compare against",
                    required=False,
                ),
            ],
            handler=lambda **kwargs: _optimize_recipe(database=database, **kwargs),
            category=ToolCategory.PLANNING,
        )
    )
