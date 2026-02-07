"""
Tool definitions for the calibration agent.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ptpd_calibration.core.models import CalibrationRecord
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
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
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

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: ToolCategory | None = None) -> list[Tool]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def to_anthropic_format(self) -> list[dict]:
        """Convert all tools to Anthropic format."""
        return [t.to_anthropic_format() for t in self._tools.values()]


def create_calibration_tools(
    database: CalibrationDatabase | None = None,
    predictor: Any | None = None,
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
    database: CalibrationDatabase | None, id1: str, id2: str
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
    paper_type: str | None = None,
    chemistry_type: str | None = None,
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
    notes: str | None = None,
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
