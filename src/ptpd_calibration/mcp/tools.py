"""
MCP tool definitions for calibration operations.

Tools in MCP allow LLMs to perform actions and call functions,
such as analyzing densities, generating curves, and more.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class MCPToolCategory(str, Enum):
    """Categories of MCP tools."""

    ANALYSIS = "analysis"
    CURVES = "curves"
    DATABASE = "database"
    PREDICTION = "prediction"
    CHEMISTRY = "chemistry"
    UTILITY = "utility"


@dataclass
class MCPToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items_type: Optional[str] = None  # For array types

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.items_type and self.type == "array":
            schema["items"] = {"type": self.items_type}

        return schema


@dataclass
class MCPToolResult:
    """Result from tool execution."""

    success: bool
    content: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    is_error: bool = False

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP protocol format."""
        if self.is_error or not self.success:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": self.error or "Unknown error occurred",
                    }
                ],
            }

        return {
            "content": self.content,
        }

    @classmethod
    def text(cls, text: str) -> "MCPToolResult":
        """Create a text result."""
        return cls(
            success=True,
            content=[{"type": "text", "text": text}],
        )

    @classmethod
    def json_result(cls, data: Any) -> "MCPToolResult":
        """Create a JSON result."""
        return cls(
            success=True,
            content=[
                {
                    "type": "text",
                    "text": json.dumps(data, indent=2, default=str),
                }
            ],
        )

    @classmethod
    def error_result(cls, error: str) -> "MCPToolResult":
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            is_error=True,
        )


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    parameters: list[MCPToolParameter]
    handler: Callable[..., MCPToolResult]
    category: MCPToolCategory = MCPToolCategory.UTILITY

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP protocol format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Dictionary of argument names to values.

        Returns:
            MCPToolResult with the execution result.
        """
        try:
            # Apply defaults for missing optional parameters
            kwargs = {}
            for param in self.parameters:
                if param.name in arguments:
                    kwargs[param.name] = arguments[param.name]
                elif not param.required and param.default is not None:
                    kwargs[param.name] = param.default
                elif param.required and param.name not in arguments:
                    return MCPToolResult.error_result(
                        f"Missing required parameter: {param.name}"
                    )

            result = self.handler(**kwargs)

            if isinstance(result, MCPToolResult):
                return result

            # Wrap non-MCPToolResult returns
            if isinstance(result, (dict, list)):
                return MCPToolResult.json_result(result)

            return MCPToolResult.text(str(result))

        except Exception as e:
            return MCPToolResult.error_result(f"Tool execution failed: {str(e)}")


class MCPToolRegistry:
    """Registry of available MCP tools."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """
        Register a tool.

        Args:
            tool: MCPTool to register.
        """
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name to unregister.

        Returns:
            True if tool was found and removed, False otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            MCPTool if found, None otherwise.
        """
        return self._tools.get(name)

    def list_tools(
        self, category: Optional[MCPToolCategory] = None
    ) -> list[MCPTool]:
        """
        List all tools, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of MCPTool objects.
        """
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def to_mcp_format(self) -> list[dict[str, Any]]:
        """
        Convert all tools to MCP protocol format.

        Returns:
            List of tool definitions in MCP format.
        """
        return [t.to_mcp_format() for t in self._tools.values()]


def create_mcp_tools(
    database: Optional[Any] = None,
    predictor: Optional[Any] = None,
) -> MCPToolRegistry:
    """
    Create the standard set of MCP tools for calibration.

    Args:
        database: Optional CalibrationDatabase for data access.
        predictor: Optional CurvePredictor for ML predictions.

    Returns:
        MCPToolRegistry with all calibration tools.
    """
    registry = MCPToolRegistry()

    # Analysis tools
    registry.register(
        MCPTool(
            name="analyze_densities",
            description="Analyze a list of density measurements and provide quality metrics including linearity, Dmax, and suggestions for improvement",
            parameters=[
                MCPToolParameter(
                    name="densities",
                    type="array",
                    description="List of density values from step tablet (typically 21 values)",
                    items_type="number",
                ),
            ],
            handler=_analyze_densities,
            category=MCPToolCategory.ANALYSIS,
        )
    )

    registry.register(
        MCPTool(
            name="calculate_density_range",
            description="Calculate the usable density range from measurements",
            parameters=[
                MCPToolParameter(
                    name="densities",
                    type="array",
                    description="List of density values",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="threshold",
                    type="number",
                    description="Minimum density change to consider significant",
                    required=False,
                    default=0.02,
                    minimum=0.001,
                    maximum=0.1,
                ),
            ],
            handler=_calculate_density_range,
            category=MCPToolCategory.ANALYSIS,
        )
    )

    # Curve tools
    registry.register(
        MCPTool(
            name="generate_linearization_curve",
            description="Generate a linearization curve from density measurements",
            parameters=[
                MCPToolParameter(
                    name="densities",
                    type="array",
                    description="Measured densities from step tablet",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="curve_name",
                    type="string",
                    description="Name for the generated curve",
                ),
                MCPToolParameter(
                    name="curve_type",
                    type="string",
                    description="Type of curve to generate",
                    required=False,
                    default="linear",
                    enum=["linear", "paper_white", "aesthetic"],
                ),
                MCPToolParameter(
                    name="interpolation",
                    type="string",
                    description="Interpolation method",
                    required=False,
                    default="monotonic",
                    enum=["linear", "cubic", "monotonic", "pchip"],
                ),
            ],
            handler=_generate_linearization_curve,
            category=MCPToolCategory.CURVES,
        )
    )

    registry.register(
        MCPTool(
            name="smooth_curve",
            description="Apply smoothing to curve data",
            parameters=[
                MCPToolParameter(
                    name="input_values",
                    type="array",
                    description="Input values (0-255)",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="output_values",
                    type="array",
                    description="Output values (0-255)",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="smoothing_factor",
                    type="number",
                    description="Smoothing factor (0-1)",
                    required=False,
                    default=0.1,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            handler=_smooth_curve,
            category=MCPToolCategory.CURVES,
        )
    )

    # Chemistry tools
    registry.register(
        MCPTool(
            name="calculate_chemistry",
            description="Calculate Pt/Pd chemistry volumes for a given print size",
            parameters=[
                MCPToolParameter(
                    name="width_inches",
                    type="number",
                    description="Print width in inches",
                    minimum=1.0,
                    maximum=40.0,
                ),
                MCPToolParameter(
                    name="height_inches",
                    type="number",
                    description="Print height in inches",
                    minimum=1.0,
                    maximum=40.0,
                ),
                MCPToolParameter(
                    name="metal_ratio",
                    type="number",
                    description="Platinum ratio (0=all Pd, 1=all Pt)",
                    required=False,
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
                MCPToolParameter(
                    name="contrast_drops",
                    type="integer",
                    description="Number of contrast agent drops",
                    required=False,
                    default=0,
                    minimum=0,
                    maximum=20,
                ),
            ],
            handler=_calculate_chemistry,
            category=MCPToolCategory.CHEMISTRY,
        )
    )

    registry.register(
        MCPTool(
            name="suggest_exposure",
            description="Suggest exposure time based on parameters",
            parameters=[
                MCPToolParameter(
                    name="paper_type",
                    type="string",
                    description="Paper type being used",
                ),
                MCPToolParameter(
                    name="uv_source",
                    type="string",
                    description="UV light source type",
                    required=False,
                    default="led",
                    enum=["led", "fluorescent", "sunlight", "mercury_vapor"],
                ),
                MCPToolParameter(
                    name="metal_ratio",
                    type="number",
                    description="Platinum ratio (0-1)",
                    required=False,
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            handler=_suggest_exposure,
            category=MCPToolCategory.CHEMISTRY,
        )
    )

    # Database tools (if available)
    if database:
        registry.register(
            MCPTool(
                name="search_calibrations",
                description="Search the calibration database",
                parameters=[
                    MCPToolParameter(
                        name="paper_type",
                        type="string",
                        description="Filter by paper type",
                        required=False,
                    ),
                    MCPToolParameter(
                        name="chemistry_type",
                        type="string",
                        description="Filter by chemistry type",
                        required=False,
                    ),
                    MCPToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results to return",
                        required=False,
                        default=10,
                        minimum=1,
                        maximum=100,
                    ),
                ],
                handler=lambda **kwargs: _search_calibrations(database, **kwargs),
                category=MCPToolCategory.DATABASE,
            )
        )

        registry.register(
            MCPTool(
                name="save_calibration",
                description="Save a new calibration record",
                parameters=[
                    MCPToolParameter(
                        name="paper_type",
                        type="string",
                        description="Paper type used",
                    ),
                    MCPToolParameter(
                        name="exposure_time",
                        type="number",
                        description="Exposure time in seconds",
                        minimum=1.0,
                    ),
                    MCPToolParameter(
                        name="metal_ratio",
                        type="number",
                        description="Platinum ratio (0-1)",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    MCPToolParameter(
                        name="densities",
                        type="array",
                        description="Measured density values",
                        items_type="number",
                    ),
                    MCPToolParameter(
                        name="notes",
                        type="string",
                        description="Additional notes",
                        required=False,
                    ),
                ],
                handler=lambda **kwargs: _save_calibration(database, **kwargs),
                category=MCPToolCategory.DATABASE,
            )
        )

    # Prediction tools (if available)
    if predictor:
        registry.register(
            MCPTool(
                name="predict_response",
                description="Predict density response using ML model",
                parameters=[
                    MCPToolParameter(
                        name="paper_type",
                        type="string",
                        description="Paper type",
                    ),
                    MCPToolParameter(
                        name="metal_ratio",
                        type="number",
                        description="Platinum ratio (0-1)",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    MCPToolParameter(
                        name="exposure_time",
                        type="number",
                        description="Exposure time in seconds",
                        minimum=1.0,
                    ),
                ],
                handler=lambda **kwargs: _predict_response(predictor, **kwargs),
                category=MCPToolCategory.PREDICTION,
            )
        )

    # Utility tools
    registry.register(
        MCPTool(
            name="format_qtr_curve",
            description="Format curve data as QTR (QuadToneRIP) output",
            parameters=[
                MCPToolParameter(
                    name="input_values",
                    type="array",
                    description="Input values (0-255)",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="output_values",
                    type="array",
                    description="Output values (0-255)",
                    items_type="number",
                ),
                MCPToolParameter(
                    name="curve_name",
                    type="string",
                    description="Name of the curve",
                    required=False,
                    default="Linearization",
                ),
            ],
            handler=_format_qtr_curve,
            category=MCPToolCategory.UTILITY,
        )
    )

    return registry


# Tool handlers


def _analyze_densities(densities: list[float]) -> MCPToolResult:
    """Analyze density measurements."""
    if not densities:
        return MCPToolResult.error_result("No density values provided")

    if len(densities) < 3:
        return MCPToolResult.error_result("At least 3 density values required")

    try:
        import numpy as np

        densities_arr = np.array(densities)

        # Basic statistics
        dmin = float(np.min(densities_arr))
        dmax = float(np.max(densities_arr))
        density_range = dmax - dmin

        # Check monotonicity
        diffs = np.diff(densities_arr)
        is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
        non_monotonic_count = int(np.sum(diffs < 0)) if np.mean(diffs) > 0 else int(np.sum(diffs > 0))

        # Linearity analysis
        ideal_linear = np.linspace(dmin, dmax, len(densities))
        max_deviation = float(np.max(np.abs(densities_arr - ideal_linear)))
        rms_error = float(np.sqrt(np.mean((densities_arr - ideal_linear) ** 2)))

        # Gradient analysis
        gradients = np.abs(diffs)
        avg_gradient = float(np.mean(gradients))
        gradient_uniformity = 1.0 - float(np.std(gradients) / avg_gradient) if avg_gradient > 0 else 0

        # Quality assessment
        quality = "good"
        suggestions = []

        if not is_monotonic:
            quality = "poor"
            suggestions.append(f"Found {non_monotonic_count} non-monotonic transitions - check for measurement errors")

        if density_range < 1.5:
            if quality != "poor":
                quality = "fair"
            suggestions.append(f"Low density range ({density_range:.2f}) - consider increasing exposure or contrast")

        if max_deviation > 0.15:
            if quality != "poor":
                quality = "fair"
            suggestions.append("High deviation from linear - curve may need significant correction")

        if gradient_uniformity < 0.7:
            suggestions.append("Uneven gradient distribution - check for blocked shadows or highlights")

        if not suggestions:
            suggestions.append("Measurements look good for curve generation")

        analysis = {
            "dmin": round(dmin, 4),
            "dmax": round(dmax, 4),
            "density_range": round(density_range, 4),
            "num_steps": len(densities),
            "is_monotonic": is_monotonic,
            "non_monotonic_transitions": non_monotonic_count,
            "max_deviation_from_linear": round(max_deviation, 4),
            "rms_error": round(rms_error, 4),
            "gradient_uniformity": round(gradient_uniformity, 4),
            "quality": quality,
            "suggestions": suggestions,
        }

        return MCPToolResult.json_result(analysis)

    except ImportError:
        return MCPToolResult.error_result(
            "numpy is required for density analysis. Install with: pip install numpy"
        )
    except Exception as e:
        return MCPToolResult.error_result(f"Analysis failed: {str(e)}")


def _calculate_density_range(
    densities: list[float],
    threshold: float = 0.02,
) -> MCPToolResult:
    """Calculate usable density range."""
    if not densities:
        return MCPToolResult.error_result("No density values provided")

    try:
        import numpy as np

        densities_arr = np.array(densities)
        diffs = np.abs(np.diff(densities_arr))

        # Find usable range (where changes exceed threshold)
        usable_mask = diffs >= threshold
        usable_indices = np.where(usable_mask)[0]

        if len(usable_indices) == 0:
            return MCPToolResult.json_result({
                "usable_start": 0,
                "usable_end": len(densities) - 1,
                "usable_range": float(densities[-1] - densities[0]),
                "warning": "No significant density changes found",
            })

        start_idx = int(usable_indices[0])
        end_idx = int(usable_indices[-1] + 1)

        result = {
            "usable_start_step": start_idx,
            "usable_end_step": end_idx,
            "usable_steps": end_idx - start_idx + 1,
            "total_steps": len(densities),
            "usable_dmin": round(float(densities_arr[start_idx]), 4),
            "usable_dmax": round(float(densities_arr[end_idx]), 4),
            "usable_range": round(float(densities_arr[end_idx] - densities_arr[start_idx]), 4),
        }

        return MCPToolResult.json_result(result)

    except Exception as e:
        return MCPToolResult.error_result(f"Calculation failed: {str(e)}")


def _generate_linearization_curve(
    densities: list[float],
    curve_name: str,
    curve_type: str = "linear",
    interpolation: str = "monotonic",
) -> MCPToolResult:
    """Generate a linearization curve."""
    if not densities:
        return MCPToolResult.error_result("No density values provided")

    if len(densities) < 3:
        return MCPToolResult.error_result("At least 3 density values required")

    try:
        import numpy as np

        n_points = len(densities)
        densities_arr = np.array(densities)

        # Normalize densities to 0-1 range
        dmin = np.min(densities_arr)
        dmax = np.max(densities_arr)

        if dmax - dmin < 0.01:
            return MCPToolResult.error_result("Density range too small for linearization")

        normalized = (densities_arr - dmin) / (dmax - dmin)

        # Input values (0-255 in n_points steps)
        input_values = np.linspace(0, 255, n_points)

        # For linearization, we need to invert the response
        # Output = correction to achieve linear response
        target_linear = np.linspace(0, 1, n_points)

        # Create correction curve
        try:
            from scipy import interpolate

            if interpolation == "pchip":
                interp = interpolate.PchipInterpolator(normalized, target_linear)
            elif interpolation == "cubic":
                interp = interpolate.interp1d(
                    normalized, target_linear, kind="cubic", fill_value="extrapolate"
                )
            elif interpolation == "monotonic":
                interp = interpolate.PchipInterpolator(normalized, target_linear)
            else:
                interp = interpolate.interp1d(
                    normalized, target_linear, kind="linear", fill_value="extrapolate"
                )

            # Generate output curve at 256 points
            output_normalized = np.linspace(0, 1, 256)
            corrected = interp(output_normalized)
            corrected = np.clip(corrected, 0, 1)
            output_values = (corrected * 255).astype(int)

        except ImportError:
            # Fallback to simple linear interpolation
            output_values = np.interp(
                np.linspace(0, 1, 256),
                normalized,
                target_linear * 255,
            ).astype(int)

        # Apply curve type adjustments
        if curve_type == "paper_white":
            # Preserve paper white in highlights
            output_values[:20] = np.linspace(0, output_values[20], 20).astype(int)

        elif curve_type == "aesthetic":
            # S-curve for more pleasing tonality
            x = np.linspace(0, 1, 256)
            s_curve = 0.5 * (np.tanh(3 * (x - 0.5)) / np.tanh(1.5) + 1)
            output_values = (output_values * (0.7 + 0.3 * s_curve)).astype(int)
            output_values = np.clip(output_values, 0, 255)

        result = {
            "name": curve_name,
            "curve_type": curve_type,
            "interpolation": interpolation,
            "input_points": n_points,
            "output_points": 256,
            "input_values": [int(v) for v in np.linspace(0, 255, 256)],
            "output_values": [int(v) for v in output_values],
            "summary": {
                "min_output": int(np.min(output_values)),
                "max_output": int(np.max(output_values)),
                "midpoint_output": int(output_values[128]),
            },
        }

        return MCPToolResult.json_result(result)

    except Exception as e:
        return MCPToolResult.error_result(f"Curve generation failed: {str(e)}")


def _smooth_curve(
    input_values: list[float],
    output_values: list[float],
    smoothing_factor: float = 0.1,
) -> MCPToolResult:
    """Apply smoothing to curve data."""
    if len(input_values) != len(output_values):
        return MCPToolResult.error_result("Input and output arrays must have same length")

    try:
        import numpy as np

        output_arr = np.array(output_values)

        # Apply Gaussian smoothing
        if smoothing_factor > 0:
            try:
                from scipy.ndimage import gaussian_filter1d

                sigma = smoothing_factor * len(output_arr) / 10
                smoothed = gaussian_filter1d(output_arr, sigma=sigma)
            except ImportError:
                # Simple moving average fallback
                window = max(3, int(smoothing_factor * len(output_arr) / 5))
                if window % 2 == 0:
                    window += 1
                smoothed = np.convolve(
                    output_arr, np.ones(window) / window, mode="same"
                )
        else:
            smoothed = output_arr

        # Ensure monotonicity
        for i in range(1, len(smoothed)):
            if smoothed[i] < smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]

        smoothed = np.clip(smoothed, 0, 255)

        result = {
            "input_values": [int(v) for v in input_values],
            "output_values": [int(v) for v in smoothed],
            "smoothing_applied": smoothing_factor,
        }

        return MCPToolResult.json_result(result)

    except Exception as e:
        return MCPToolResult.error_result(f"Smoothing failed: {str(e)}")


def _calculate_chemistry(
    width_inches: float,
    height_inches: float,
    metal_ratio: float = 0.5,
    contrast_drops: int = 0,
) -> MCPToolResult:
    """Calculate chemistry volumes."""
    try:
        # Standard coating rate: ~2ml per 100 sq inches
        area_sq_inches = width_inches * height_inches
        coating_rate_ml_per_sq_inch = 0.02

        total_solution_ml = area_sq_inches * coating_rate_ml_per_sq_inch
        total_solution_ml = max(1.0, total_solution_ml)  # Minimum 1ml

        # Calculate individual components (typical 1:1:1 ratio of sensitizer:metal:FO)
        sensitizer_ml = total_solution_ml / 3
        ferric_oxalate_ml = total_solution_ml / 3
        metal_solution_ml = total_solution_ml / 3

        # Split metal solution by ratio
        platinum_ml = metal_solution_ml * metal_ratio
        palladium_ml = metal_solution_ml * (1 - metal_ratio)

        # Convert to drops (approximately 20 drops per ml)
        drops_per_ml = 20

        result = {
            "print_size": {
                "width_inches": width_inches,
                "height_inches": height_inches,
                "area_sq_inches": round(area_sq_inches, 2),
            },
            "volumes_ml": {
                "total_solution": round(total_solution_ml, 2),
                "ferric_oxalate": round(ferric_oxalate_ml, 2),
                "platinum_solution": round(platinum_ml, 2),
                "palladium_solution": round(palladium_ml, 2),
            },
            "drops": {
                "ferric_oxalate": int(ferric_oxalate_ml * drops_per_ml),
                "platinum": int(platinum_ml * drops_per_ml),
                "palladium": int(palladium_ml * drops_per_ml),
                "contrast_agent": contrast_drops,
            },
            "metal_ratio": {
                "platinum_percent": round(metal_ratio * 100, 1),
                "palladium_percent": round((1 - metal_ratio) * 100, 1),
            },
            "notes": [
                "Add ferric oxalate first, then metal solutions",
                "Mix thoroughly before coating",
                "Apply under safe light conditions",
            ],
        }

        return MCPToolResult.json_result(result)

    except Exception as e:
        return MCPToolResult.error_result(f"Calculation failed: {str(e)}")


def _suggest_exposure(
    paper_type: str,
    uv_source: str = "led",
    metal_ratio: float = 0.5,
) -> MCPToolResult:
    """Suggest exposure time."""
    # Base exposure times by UV source (in seconds)
    base_exposures = {
        "led": 180,
        "fluorescent": 300,
        "sunlight": 120,
        "mercury_vapor": 90,
    }

    base_time = base_exposures.get(uv_source, 180)

    # Adjust for metal ratio (more Pt = longer exposure typically)
    metal_adjustment = 1.0 + (metal_ratio - 0.5) * 0.2

    # Paper adjustments (rough estimates)
    paper_adjustments = {
        "arches_platine": 1.0,
        "bergger_cot": 0.95,
        "hahnemuhle": 1.05,
        "stonehenge": 0.9,
        "washi": 0.85,
    }

    paper_key = paper_type.lower().replace(" ", "_")
    paper_adjustment = paper_adjustments.get(paper_key, 1.0)

    suggested_time = base_time * metal_adjustment * paper_adjustment

    # Create bracket
    bracket = [
        round(suggested_time * 0.5),
        round(suggested_time * 0.71),
        round(suggested_time),
        round(suggested_time * 1.41),
        round(suggested_time * 2.0),
    ]

    result = {
        "suggested_exposure_seconds": round(suggested_time),
        "suggested_exposure_formatted": f"{int(suggested_time // 60)}m {int(suggested_time % 60)}s",
        "exposure_bracket": bracket,
        "bracket_formatted": [f"{int(t // 60)}m {int(t % 60)}s" for t in bracket],
        "factors": {
            "uv_source": uv_source,
            "base_time": base_time,
            "metal_ratio_adjustment": round(metal_adjustment, 2),
            "paper_adjustment": round(paper_adjustment, 2),
        },
        "notes": [
            "These are starting estimates - actual times vary significantly",
            "Run a step wedge test to determine optimal exposure",
            "Exposure depends on negative density and humidity",
        ],
    }

    return MCPToolResult.json_result(result)


def _search_calibrations(
    database: Any,
    paper_type: Optional[str] = None,
    chemistry_type: Optional[str] = None,
    limit: int = 10,
) -> MCPToolResult:
    """Search calibrations in database."""
    try:
        results = database.query(
            paper_type=paper_type,
            chemistry_type=chemistry_type,
        )

        records = []
        for rec in results[:limit]:
            records.append({
                "id": str(rec.id),
                "paper_type": rec.paper_type,
                "metal_ratio": rec.metal_ratio,
                "exposure_time": rec.exposure_time,
                "dmax": max(rec.measured_densities) if rec.measured_densities else None,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
            })

        return MCPToolResult.json_result({
            "count": len(records),
            "records": records,
        })

    except Exception as e:
        return MCPToolResult.error_result(f"Search failed: {str(e)}")


def _save_calibration(
    database: Any,
    paper_type: str,
    exposure_time: float,
    metal_ratio: float,
    densities: list[float],
    notes: Optional[str] = None,
) -> MCPToolResult:
    """Save a calibration record."""
    try:
        from ptpd_calibration.core.models import CalibrationRecord
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

        return MCPToolResult.json_result({
            "success": True,
            "id": str(record.id),
            "message": "Calibration saved successfully",
        })

    except Exception as e:
        return MCPToolResult.error_result(f"Save failed: {str(e)}")


def _predict_response(
    predictor: Any,
    paper_type: str,
    metal_ratio: float,
    exposure_time: float,
) -> MCPToolResult:
    """Predict density response using ML model."""
    try:
        import asyncio
        from ptpd_calibration.core.models import CalibrationRecord
        from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType

        record = CalibrationRecord(
            paper_type=paper_type,
            exposure_time=exposure_time,
            metal_ratio=metal_ratio,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NONE,
            developer=DeveloperType.POTASSIUM_OXALATE,
        )

        # Run async prediction
        loop = asyncio.get_event_loop()
        prediction, uncertainty = loop.run_until_complete(
            predictor.predict(record, return_uncertainty=True)
        )

        return MCPToolResult.json_result({
            "predicted_densities": prediction,
            "uncertainty": uncertainty,
            "dmax": max(prediction) if prediction else None,
            "dmin": min(prediction) if prediction else None,
        })

    except Exception as e:
        return MCPToolResult.error_result(f"Prediction failed: {str(e)}")


def _format_qtr_curve(
    input_values: list[float],
    output_values: list[float],
    curve_name: str = "Linearization",
) -> MCPToolResult:
    """Format curve as QTR output."""
    if len(input_values) != len(output_values):
        return MCPToolResult.error_result("Input and output arrays must have same length")

    try:
        lines = [
            f"# QTR Curve: {curve_name}",
            f"# Generated by PTPD Calibration MCP",
            "",
            "[Gray]",
        ]

        # QTR expects 256 entries
        if len(output_values) != 256:
            import numpy as np

            output_values = np.interp(
                np.arange(256),
                np.linspace(0, 255, len(output_values)),
                output_values,
            ).astype(int).tolist()

        for i, val in enumerate(output_values):
            lines.append(f"{i}={int(val)}")

        qtr_content = "\n".join(lines)

        return MCPToolResult.text(qtr_content)

    except Exception as e:
        return MCPToolResult.error_result(f"Formatting failed: {str(e)}")
