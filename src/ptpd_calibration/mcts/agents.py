"""
MCTS subagents for calibration parameter optimization.

Provides specialized subagents for chemistry, exposure, and coordination
that integrate with the agentic infrastructure and MCTS search engine.
"""

from __future__ import annotations

import logging
from typing import Any

from ptpd_calibration.agents.logging import get_agent_logger
from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentResult,
    register_subagent,
)
from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings, PhysicsConstants
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator

logger = logging.getLogger(__name__)


@register_subagent
class ChemistrySubagent(BaseSubagent):
    """Subagent for chemistry parameter optimization in MCTS.

    Manages metal_ratio, ferric_oxalate_pct, and coating_weight decisions.
    Maps aesthetic preferences to chemistry parameters.
    """

    AGENT_TYPE = "mcts_chemistry"
    CAPABILITIES = [SubagentCapability.ANALYSIS]
    DESCRIPTION = "Optimizes chemistry parameters for Pt/Pd calibration"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize ChemistrySubagent.

        Args:
            config: Subagent configuration.
        """
        super().__init__(config)
        self.settings = MCTSSettings()
        self.physics = PhysicsConstants()
        self._logger = get_agent_logger()

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """Execute chemistry analysis task.

        Args:
            task: Task description (e.g., "suggest_chemistry", "analyze_parameters").
            context: Optional context data with target_aesthetics, fixed_parameters, etc.

        Returns:
            SubagentResult with chemistry parameter suggestions.
        """
        self._start_execution(task)

        try:
            context = context or {}
            target_aesthetics = context.get("target_aesthetics", {})

            if task == "suggest_chemistry":
                result_data = self.suggest_chemistry(target_aesthetics)
            elif task == "analyze_parameters":
                params = context.get("parameters", {})
                result_data = self._analyze_chemistry_params(params)
            else:
                raise ValueError(f"Unknown chemistry task: {task}")

            result = SubagentResult(
                success=True,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                result=result_data,
            )

        except Exception as e:
            logger.exception("ChemistrySubagent failed: %s", e)
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                error=str(e),
            )

        return self._complete_execution(result)

    def capabilities(self) -> list[SubagentCapability]:
        """Return the capabilities this subagent provides."""
        return self.CAPABILITIES

    def suggest_chemistry(
        self,
        target_aesthetics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Suggest chemistry parameters based on target aesthetics.

        Maps aesthetic preferences (contrast, warmth, tonal range) to
        chemistry parameters (metal_ratio, FO%, coating_weight).

        Args:
            target_aesthetics: Optional dict with keys:
                - "contrast": 0.0-1.0 (higher = more contrast)
                - "warmth": 0.0-1.0 (higher = warmer tones, more Pd)
                - "tonal_range": 0.0-1.0 (higher = wider range, more coating)

        Returns:
            Dictionary with suggested chemistry parameters.
        """
        aesthetics = target_aesthetics or {}

        # Extract aesthetic preferences with defaults
        contrast_pref = aesthetics.get("contrast", 0.5)
        warmth_pref = aesthetics.get("warmth", 0.5)
        tonal_range_pref = aesthetics.get("tonal_range", 0.5)

        # Map aesthetics to chemistry parameters
        # Warmth: Higher warmth -> more Pd (lower metal_ratio)
        metal_ratio_range = DEFAULT_PARAMETER_RANGES["metal_ratio"]
        metal_ratio = metal_ratio_range.min_value + (1.0 - warmth_pref) * (
            metal_ratio_range.max_value - metal_ratio_range.min_value
        )

        # Contrast: Higher contrast -> higher FO%
        fo_range = DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"]
        fo_center = self.physics.fo_contrast_center
        # Map 0.5 contrast to center, 0.0 to min, 1.0 to max
        if contrast_pref < 0.5:
            fo_pct = fo_range.min_value + (contrast_pref / 0.5) * (fo_center - fo_range.min_value)
        else:
            fo_pct = fo_center + ((contrast_pref - 0.5) / 0.5) * (fo_range.max_value - fo_center)

        # Tonal range: Higher range -> more coating weight
        coating_range = DEFAULT_PARAMETER_RANGES["coating_weight"]
        coating_weight = coating_range.min_value + tonal_range_pref * (
            coating_range.max_value - coating_range.min_value
        )

        suggested = {
            "metal_ratio": float(metal_ratio),
            "ferric_oxalate_pct": float(fo_pct),
            "coating_weight": float(coating_weight),
        }

        logger.debug(
            "Chemistry suggestion from aesthetics: contrast=%.2f, warmth=%.2f, range=%.2f -> %s",
            contrast_pref,
            warmth_pref,
            tonal_range_pref,
            suggested,
        )

        return suggested

    def _analyze_chemistry_params(self, params: dict[str, float]) -> dict[str, Any]:
        """Analyze chemistry parameters for validity and expected characteristics.

        Args:
            params: Chemistry parameters to analyze.

        Returns:
            Analysis results with validity, warnings, and expected characteristics.
        """
        analysis = {
            "valid": True,
            "warnings": [],
            "expected_characteristics": {},
        }

        # Check ranges
        for param_name in ["metal_ratio", "ferric_oxalate_pct", "coating_weight"]:
            if param_name in params:
                value = params[param_name]
                param_range = DEFAULT_PARAMETER_RANGES[param_name]
                if value < param_range.min_value or value > param_range.max_value:
                    analysis["valid"] = False
                    analysis["warnings"].append(
                        f"{param_name}={value:.2f} is out of valid range "
                        f"[{param_range.min_value}, {param_range.max_value}]"
                    )

        # Infer expected characteristics
        metal_ratio = params.get("metal_ratio", 0.5)
        fo_pct = params.get("ferric_oxalate_pct", self.physics.fo_contrast_center)

        # Warmth from metal ratio (lower Pt = warmer)
        warmth = 1.0 - metal_ratio

        # Contrast from FO%
        fo_deviation = abs(fo_pct - self.physics.fo_contrast_center)
        contrast = 0.5 + fo_deviation / (
            DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"].max_value
            - self.physics.fo_contrast_center
        )

        analysis["expected_characteristics"] = {
            "warmth": float(warmth),
            "contrast": float(min(1.0, contrast)),
        }

        return analysis


@register_subagent
class ExposureSubagent(BaseSubagent):
    """Subagent for exposure and development parameter optimization."""

    AGENT_TYPE = "mcts_exposure"
    CAPABILITIES = [SubagentCapability.ANALYSIS]
    DESCRIPTION = "Optimizes exposure and development parameters"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize ExposureSubagent.

        Args:
            config: Subagent configuration.
        """
        super().__init__(config)
        self.settings = MCTSSettings()
        self.physics = PhysicsConstants()
        self._logger = get_agent_logger()

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """Execute exposure analysis task.

        Args:
            task: Task description (e.g., "suggest_exposure").
            context: Optional context data with chemistry_params, uv_source, etc.

        Returns:
            SubagentResult with exposure parameter suggestions.
        """
        self._start_execution(task)

        try:
            context = context or {}
            chemistry_params = context.get("chemistry_params", {})
            uv_source = context.get("uv_source")

            if task == "suggest_exposure":
                result_data = self.suggest_exposure(chemistry_params, uv_source)
            else:
                raise ValueError(f"Unknown exposure task: {task}")

            result = SubagentResult(
                success=True,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                result=result_data,
            )

        except Exception as e:
            logger.exception("ExposureSubagent failed: %s", e)
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                error=str(e),
            )

        return self._complete_execution(result)

    def capabilities(self) -> list[SubagentCapability]:
        """Return the capabilities this subagent provides."""
        return self.CAPABILITIES

    def suggest_exposure(
        self,
        chemistry_params: dict[str, float],
        uv_source: str | None = None,
    ) -> dict[str, float]:
        """Suggest exposure parameters given chemistry.

        Args:
            chemistry_params: Chemistry parameters (metal_ratio, coating_weight, etc.).
            uv_source: Optional UV source type (e.g., "sun", "uv_led", "metal_halide").

        Returns:
            Dictionary with suggested exposure parameters.
        """
        # Extract chemistry parameters
        coating_weight = chemistry_params.get("coating_weight", 1.5)

        # Base exposure time from coating weight (more coating = more time)
        exposure_range = DEFAULT_PARAMETER_RANGES["exposure_time"]
        # Heavier coating needs longer exposure
        coating_factor = (coating_weight - DEFAULT_PARAMETER_RANGES["coating_weight"].min_value) / (
            DEFAULT_PARAMETER_RANGES["coating_weight"].max_value
            - DEFAULT_PARAMETER_RANGES["coating_weight"].min_value
        )
        base_exposure = (
            exposure_range.default_value
            + coating_factor * (exposure_range.max_value - exposure_range.default_value) * 0.5
        )

        # Adjust for UV source intensity
        uv_multipliers = {
            "sun": 0.7,  # Faster
            "uv_led": 1.2,  # Slower
            "metal_halide": 1.0,  # Standard
        }
        uv_multiplier = uv_multipliers.get(uv_source or "metal_halide", 1.0)
        exposure_time = base_exposure * uv_multiplier

        # Ensure within bounds
        exposure_time = max(exposure_range.min_value, min(exposure_range.max_value, exposure_time))

        # Developer temp: standard default
        dev_temp_range = DEFAULT_PARAMETER_RANGES["developer_temp"]
        developer_temp = dev_temp_range.default_value

        # Humidity: optimal default
        humidity = self.physics.humidity_optimal

        suggested = {
            "exposure_time": float(exposure_time),
            "developer_temp": float(developer_temp),
            "humidity": float(humidity),
        }

        logger.debug(
            "Exposure suggestion for coating_weight=%.2f, uv_source=%s -> %s",
            coating_weight,
            uv_source,
            suggested,
        )

        return suggested


@register_subagent
class CalibrationCoordinatorSubagent(BaseSubagent):
    """Coordinates chemistry and exposure subagents for full calibration.

    Orchestrates the MCTS search by delegating to specialized subagents
    and resolving multi-objective trade-offs.
    """

    AGENT_TYPE = "mcts_coordinator"
    CAPABILITIES = [SubagentCapability.ORCHESTRATION, SubagentCapability.ANALYSIS]
    DESCRIPTION = "Coordinates MCTS calibration search across subagents"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize CalibrationCoordinatorSubagent.

        Args:
            config: Subagent configuration.
        """
        super().__init__(config)
        self.settings = MCTSSettings()
        self.simulator = ExtendedProcessSimulator()
        self.scorer = QualityScorer(self.settings)
        self._logger = get_agent_logger()

        # Create specialized subagents
        self.chemistry_agent = ChemistrySubagent(config)
        self.exposure_agent = ExposureSubagent(config)

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """Execute coordination task.

        Args:
            task: Task description (e.g., "coordinate_search", "evaluate_parameters").
            context: Optional context data.

        Returns:
            SubagentResult with coordination results.
        """
        self._start_execution(task)

        try:
            context = context or {}

            if task == "coordinate_search":
                result_data = await self._coordinate_search(context)
            elif task == "evaluate_parameters":
                params = context.get("parameters", {})
                result_data = self._evaluate_parameters(params)
            else:
                raise ValueError(f"Unknown coordination task: {task}")

            result = SubagentResult(
                success=True,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                result=result_data,
            )

        except Exception as e:
            logger.exception("CalibrationCoordinatorSubagent failed: %s", e)
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                error=str(e),
            )

        return self._complete_execution(result)

    def capabilities(self) -> list[SubagentCapability]:
        """Return the capabilities this subagent provides."""
        return self.CAPABILITIES

    async def _coordinate_search(self, context: dict) -> dict[str, Any]:
        """Coordinate a full calibration search.

        Args:
            context: Search context with target_aesthetics, fixed_parameters, etc.

        Returns:
            Search results with best parameters and alternatives.
        """
        target_aesthetics = context.get("target_aesthetics", {})
        fixed_params = context.get("fixed_parameters", {})

        # Step 1: Get chemistry suggestions from chemistry agent
        chem_result = await self.chemistry_agent.run(
            "suggest_chemistry",
            context={"target_aesthetics": target_aesthetics},
        )
        chemistry_params = chem_result.result if chem_result.success else {}

        # Apply fixed parameters
        chemistry_params.update(fixed_params)

        # Step 2: Get exposure suggestions from exposure agent
        exp_result = await self.exposure_agent.run(
            "suggest_exposure",
            context={
                "chemistry_params": chemistry_params,
                "uv_source": context.get("uv_source"),
            },
        )
        exposure_params = exp_result.result if exp_result.success else {}

        # Step 3: Combine parameters
        full_params = {**chemistry_params, **exposure_params}

        # Step 4: Evaluate combined parameters
        evaluation = self._evaluate_parameters(full_params)

        return {
            "chemistry_suggestion": chemistry_params,
            "exposure_suggestion": exposure_params,
            "full_parameters": full_params,
            "evaluation": evaluation,
        }

    def _evaluate_parameters(self, params: dict[str, float]) -> dict[str, Any]:
        """Evaluate a parameter set using the simulator and scorer.

        Args:
            params: Full parameter set to evaluate.

        Returns:
            Evaluation results with quality score and predicted characteristics.
        """
        # Run simulation
        sim_result = self.simulator.simulate(params)

        # Compute quality score
        quality_score = self.scorer.score(sim_result)

        return {
            "quality_score": quality_score,
            "predicted_curve": sim_result.density_curve,
            "dmin": sim_result.dmin,
            "dmax": sim_result.dmax,
            "density_range": sim_result.density_range,
            "gamma": sim_result.gamma,
        }
