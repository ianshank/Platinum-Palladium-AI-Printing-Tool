"""
Tests for MCTS agents and API router.

Tests ChemistrySubagent, ExposureSubagent, CalibrationCoordinatorSubagent,
and MCTS API router endpoints.
"""

from __future__ import annotations

import pytest

from ptpd_calibration.agents.subagents.base import SubagentCapability, SubagentResult
from ptpd_calibration.mcts.agents import (
    CalibrationCoordinatorSubagent,
    ChemistrySubagent,
    ExposureSubagent,
)
from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, PhysicsConstants

# =============================================================================
# ChemistrySubagent Tests
# =============================================================================


class TestChemistrySubagent:
    """Test ChemistrySubagent functionality."""

    @pytest.fixture
    def chemistry_agent(self):
        """Create ChemistrySubagent instance."""
        return ChemistrySubagent()

    def test_initialization(self, chemistry_agent):
        """Test ChemistrySubagent initializes correctly."""
        assert chemistry_agent.AGENT_TYPE == "mcts_chemistry"
        assert SubagentCapability.ANALYSIS in chemistry_agent.CAPABILITIES
        assert chemistry_agent.settings is not None
        assert chemistry_agent.physics is not None

    def test_capabilities(self, chemistry_agent):
        """Test capabilities method returns correct list."""
        capabilities = chemistry_agent.capabilities()
        assert SubagentCapability.ANALYSIS in capabilities
        assert len(capabilities) == 1

    def test_suggest_chemistry_returns_dict(self, chemistry_agent):
        """Test suggest_chemistry returns dict with required keys."""
        result = chemistry_agent.suggest_chemistry()

        assert isinstance(result, dict)
        assert "metal_ratio" in result
        assert "ferric_oxalate_pct" in result
        assert "coating_weight" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_suggest_chemistry_default_aesthetics(self, chemistry_agent):
        """Test suggest_chemistry with default aesthetics (0.5, 0.5, 0.5)."""
        result = chemistry_agent.suggest_chemistry()

        # Default (0.5) should give mid-range values
        assert DEFAULT_PARAMETER_RANGES["metal_ratio"].min_value <= result["metal_ratio"]
        assert result["metal_ratio"] <= DEFAULT_PARAMETER_RANGES["metal_ratio"].max_value

        assert DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"].min_value <= result[
            "ferric_oxalate_pct"
        ]
        assert (
            result["ferric_oxalate_pct"]
            <= DEFAULT_PARAMETER_RANGES["ferric_oxalate_pct"].max_value
        )

        assert DEFAULT_PARAMETER_RANGES["coating_weight"].min_value <= result["coating_weight"]
        assert result["coating_weight"] <= DEFAULT_PARAMETER_RANGES["coating_weight"].max_value

    def test_suggest_chemistry_warmth_preference_affects_metal_ratio(self, chemistry_agent):
        """Test warmth preference affects metal_ratio (higher warmth = lower ratio, more Pd)."""
        # High warmth -> more Pd -> lower metal_ratio
        warm_result = chemistry_agent.suggest_chemistry({"warmth": 1.0})

        # Low warmth -> more Pt -> higher metal_ratio
        cool_result = chemistry_agent.suggest_chemistry({"warmth": 0.0})

        assert warm_result["metal_ratio"] < cool_result["metal_ratio"]

    def test_suggest_chemistry_contrast_preference_affects_ferric_oxalate(self, chemistry_agent):
        """Test contrast preference affects ferric_oxalate_pct."""
        physics = PhysicsConstants()

        # Default contrast (0.5) should be near center
        default_result = chemistry_agent.suggest_chemistry({"contrast": 0.5})
        assert abs(default_result["ferric_oxalate_pct"] - physics.fo_contrast_center) < 5.0

        # High contrast -> higher FO%
        high_contrast = chemistry_agent.suggest_chemistry({"contrast": 1.0})

        # Low contrast -> lower FO%
        low_contrast = chemistry_agent.suggest_chemistry({"contrast": 0.0})

        assert low_contrast["ferric_oxalate_pct"] < high_contrast["ferric_oxalate_pct"]

    def test_suggest_chemistry_tonal_range_affects_coating_weight(self, chemistry_agent):
        """Test tonal_range preference affects coating_weight."""
        # High tonal range -> more coating
        high_range = chemistry_agent.suggest_chemistry({"tonal_range": 1.0})

        # Low tonal range -> less coating
        low_range = chemistry_agent.suggest_chemistry({"tonal_range": 0.0})

        assert low_range["coating_weight"] < high_range["coating_weight"]

    def test_analyze_chemistry_params_returns_valid_analysis(self, chemistry_agent):
        """Test _analyze_chemistry_params returns valid analysis structure."""
        params = {
            "metal_ratio": 0.5,
            "ferric_oxalate_pct": 20.0,
            "coating_weight": 1.5,
        }

        analysis = chemistry_agent._analyze_chemistry_params(params)

        assert "valid" in analysis
        assert "warnings" in analysis
        assert "expected_characteristics" in analysis
        assert isinstance(analysis["valid"], bool)
        assert isinstance(analysis["warnings"], list)
        assert isinstance(analysis["expected_characteristics"], dict)

    def test_analyze_chemistry_params_valid_parameters(self, chemistry_agent):
        """Test analysis marks valid parameters as valid."""
        params = {
            "metal_ratio": 0.5,
            "ferric_oxalate_pct": 20.0,
            "coating_weight": 1.5,
        }

        analysis = chemistry_agent._analyze_chemistry_params(params)

        assert analysis["valid"] is True
        assert len(analysis["warnings"]) == 0

    def test_analyze_chemistry_params_out_of_range(self, chemistry_agent):
        """Test analysis detects out-of-range parameters."""
        params = {
            "metal_ratio": 10.0,  # Way out of range
            "ferric_oxalate_pct": 100.0,  # Way out of range
            "coating_weight": 0.01,  # Too low
        }

        analysis = chemistry_agent._analyze_chemistry_params(params)

        assert analysis["valid"] is False
        assert len(analysis["warnings"]) > 0

    def test_analyze_chemistry_params_expected_characteristics(self, chemistry_agent):
        """Test analysis computes expected characteristics correctly."""
        params = {
            "metal_ratio": 0.3,  # Lower ratio = warmer
            "ferric_oxalate_pct": 25.0,
        }

        analysis = chemistry_agent._analyze_chemistry_params(params)
        chars = analysis["expected_characteristics"]

        assert "warmth" in chars
        assert "contrast" in chars
        # Lower metal_ratio = higher warmth
        assert chars["warmth"] > 0.5
        assert 0.0 <= chars["warmth"] <= 1.0
        assert 0.0 <= chars["contrast"] <= 1.0

    @pytest.mark.asyncio
    async def test_run_suggest_chemistry_task(self, chemistry_agent):
        """Test run method with 'suggest_chemistry' task."""
        context = {
            "target_aesthetics": {
                "contrast": 0.7,
                "warmth": 0.6,
                "tonal_range": 0.5,
            }
        }

        result = await chemistry_agent.run("suggest_chemistry", context=context)

        assert isinstance(result, SubagentResult)
        assert result.success is True
        assert result.agent_type == "mcts_chemistry"
        assert result.task == "suggest_chemistry"
        assert "metal_ratio" in result.result
        assert "ferric_oxalate_pct" in result.result
        assert "coating_weight" in result.result

    @pytest.mark.asyncio
    async def test_run_analyze_parameters_task(self, chemistry_agent):
        """Test run method with 'analyze_parameters' task."""
        context = {
            "parameters": {
                "metal_ratio": 0.5,
                "ferric_oxalate_pct": 20.0,
                "coating_weight": 1.5,
            }
        }

        result = await chemistry_agent.run("analyze_parameters", context=context)

        assert isinstance(result, SubagentResult)
        assert result.success is True
        assert result.task == "analyze_parameters"
        assert "valid" in result.result
        assert "expected_characteristics" in result.result

    @pytest.mark.asyncio
    async def test_run_unknown_task_fails(self, chemistry_agent):
        """Test run method with unknown task returns error."""
        result = await chemistry_agent.run("unknown_task")

        assert isinstance(result, SubagentResult)
        assert result.success is False
        assert "unknown" in result.error.lower()


# =============================================================================
# ExposureSubagent Tests
# =============================================================================


class TestExposureSubagent:
    """Test ExposureSubagent functionality."""

    @pytest.fixture
    def exposure_agent(self):
        """Create ExposureSubagent instance."""
        return ExposureSubagent()

    def test_initialization(self, exposure_agent):
        """Test ExposureSubagent initializes correctly."""
        assert exposure_agent.AGENT_TYPE == "mcts_exposure"
        assert SubagentCapability.ANALYSIS in exposure_agent.CAPABILITIES

    def test_capabilities(self, exposure_agent):
        """Test capabilities method returns correct list."""
        capabilities = exposure_agent.capabilities()
        assert SubagentCapability.ANALYSIS in capabilities

    def test_suggest_exposure_returns_dict(self, exposure_agent):
        """Test suggest_exposure returns dict with required keys."""
        chemistry_params = {"coating_weight": 1.5}
        result = exposure_agent.suggest_exposure(chemistry_params)

        assert isinstance(result, dict)
        assert "exposure_time" in result
        assert "developer_temp" in result
        assert "humidity" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_suggest_exposure_adjusts_for_coating_weight(self, exposure_agent):
        """Test exposure time adjusts based on coating weight."""
        # Heavier coating needs more time
        heavy_result = exposure_agent.suggest_exposure({"coating_weight": 3.0})
        light_result = exposure_agent.suggest_exposure({"coating_weight": 1.0})

        assert heavy_result["exposure_time"] > light_result["exposure_time"]

    def test_suggest_exposure_adjusts_for_uv_source(self, exposure_agent):
        """Test exposure time adjusts for UV source type."""
        chemistry_params = {"coating_weight": 1.5}

        # Sun is faster (0.7x multiplier)
        sun_result = exposure_agent.suggest_exposure(chemistry_params, uv_source="sun")

        # UV LED is slower (1.2x multiplier)
        led_result = exposure_agent.suggest_exposure(chemistry_params, uv_source="uv_led")

        # Metal halide is standard (1.0x multiplier)
        halide_result = exposure_agent.suggest_exposure(
            chemistry_params, uv_source="metal_halide"
        )

        assert sun_result["exposure_time"] < halide_result["exposure_time"]
        assert led_result["exposure_time"] > halide_result["exposure_time"]

    def test_suggest_exposure_respects_bounds(self, exposure_agent):
        """Test exposure time stays within parameter ranges."""
        # Extreme values should still be bounded
        extreme_result = exposure_agent.suggest_exposure(
            {"coating_weight": 10.0}, uv_source="uv_led"
        )

        exposure_range = DEFAULT_PARAMETER_RANGES["exposure_time"]
        assert exposure_range.min_value <= extreme_result["exposure_time"]
        assert extreme_result["exposure_time"] <= exposure_range.max_value

    def test_suggest_exposure_default_developer_temp(self, exposure_agent):
        """Test developer temp uses default value."""
        result = exposure_agent.suggest_exposure({"coating_weight": 1.5})

        dev_temp_range = DEFAULT_PARAMETER_RANGES["developer_temp"]
        assert result["developer_temp"] == dev_temp_range.default_value

    def test_suggest_exposure_optimal_humidity(self, exposure_agent):
        """Test humidity uses optimal default."""
        result = exposure_agent.suggest_exposure({"coating_weight": 1.5})

        physics = PhysicsConstants()
        assert result["humidity"] == physics.humidity_optimal

    @pytest.mark.asyncio
    async def test_run_suggest_exposure_task(self, exposure_agent):
        """Test run method with 'suggest_exposure' task."""
        context = {
            "chemistry_params": {"coating_weight": 2.0},
            "uv_source": "sun",
        }

        result = await exposure_agent.run("suggest_exposure", context=context)

        assert isinstance(result, SubagentResult)
        assert result.success is True
        assert result.task == "suggest_exposure"
        assert "exposure_time" in result.result
        assert "developer_temp" in result.result
        assert "humidity" in result.result

    @pytest.mark.asyncio
    async def test_run_unknown_task_fails(self, exposure_agent):
        """Test run method with unknown task returns error."""
        result = await exposure_agent.run("unknown_task")

        assert isinstance(result, SubagentResult)
        assert result.success is False
        assert "unknown" in result.error.lower()


# =============================================================================
# CalibrationCoordinatorSubagent Tests
# =============================================================================


class TestCalibrationCoordinatorSubagent:
    """Test CalibrationCoordinatorSubagent functionality."""

    @pytest.fixture
    def coordinator(self):
        """Create CalibrationCoordinatorSubagent instance."""
        return CalibrationCoordinatorSubagent()

    def test_initialization(self, coordinator):
        """Test CalibrationCoordinatorSubagent initializes correctly."""
        assert coordinator.AGENT_TYPE == "mcts_coordinator"
        assert SubagentCapability.ORCHESTRATION in coordinator.CAPABILITIES
        assert SubagentCapability.ANALYSIS in coordinator.CAPABILITIES
        assert coordinator.chemistry_agent is not None
        assert coordinator.exposure_agent is not None
        assert coordinator.simulator is not None
        assert coordinator.scorer is not None

    def test_capabilities(self, coordinator):
        """Test capabilities method returns correct list."""
        capabilities = coordinator.capabilities()
        assert SubagentCapability.ORCHESTRATION in capabilities
        assert SubagentCapability.ANALYSIS in capabilities

    @pytest.mark.asyncio
    async def test_coordinate_search_returns_complete_result(self, coordinator):
        """Test _coordinate_search returns all required fields."""
        context = {
            "target_aesthetics": {"contrast": 0.7, "warmth": 0.6},
            "fixed_parameters": {},
            "uv_source": "metal_halide",
        }

        result = await coordinator._coordinate_search(context)

        assert "chemistry_suggestion" in result
        assert "exposure_suggestion" in result
        assert "full_parameters" in result
        assert "evaluation" in result

        # Check chemistry suggestion has all keys
        chem = result["chemistry_suggestion"]
        assert "metal_ratio" in chem
        assert "ferric_oxalate_pct" in chem
        assert "coating_weight" in chem

        # Check exposure suggestion has all keys
        exp = result["exposure_suggestion"]
        assert "exposure_time" in exp
        assert "developer_temp" in exp
        assert "humidity" in exp

        # Check full parameters combines both
        full = result["full_parameters"]
        assert len(full) >= 6  # At least 6 parameters total

        # Check evaluation has metrics
        evaluation = result["evaluation"]
        assert "quality_score" in evaluation
        assert "predicted_curve" in evaluation
        assert "dmin" in evaluation
        assert "dmax" in evaluation
        assert "gamma" in evaluation

    @pytest.mark.asyncio
    async def test_coordinate_search_respects_fixed_parameters(self, coordinator):
        """Test fixed parameters override suggested values."""
        fixed_metal_ratio = 0.42
        context = {
            "target_aesthetics": {"warmth": 0.8},  # Would normally suggest different ratio
            "fixed_parameters": {"metal_ratio": fixed_metal_ratio},
        }

        result = await coordinator._coordinate_search(context)

        # Fixed parameter should be in final result
        assert result["full_parameters"]["metal_ratio"] == fixed_metal_ratio

    def test_evaluate_parameters_returns_quality_score(self, coordinator):
        """Test _evaluate_parameters returns quality score."""
        params = {
            "metal_ratio": 0.5,
            "ferric_oxalate_pct": 20.0,
            "coating_weight": 1.5,
            "exposure_time": 10.0,
            "developer_temp": 20.0,
            "humidity": 50.0,
        }

        evaluation = coordinator._evaluate_parameters(params)

        assert "quality_score" in evaluation
        assert 0.0 <= evaluation["quality_score"] <= 1.0
        assert "predicted_curve" in evaluation
        assert isinstance(evaluation["predicted_curve"], list)
        assert len(evaluation["predicted_curve"]) > 0

    def test_evaluate_parameters_includes_simulation_metrics(self, coordinator):
        """Test _evaluate_parameters includes all simulation metrics."""
        params = {
            "metal_ratio": 0.5,
            "ferric_oxalate_pct": 20.0,
            "coating_weight": 1.5,
            "exposure_time": 10.0,
            "developer_temp": 20.0,
            "humidity": 50.0,
        }

        evaluation = coordinator._evaluate_parameters(params)

        assert "dmin" in evaluation
        assert "dmax" in evaluation
        assert "density_range" in evaluation
        assert "gamma" in evaluation
        assert evaluation["dmin"] >= 0.0
        assert evaluation["dmax"] >= evaluation["dmin"]
        assert evaluation["gamma"] > 0.0

    @pytest.mark.asyncio
    async def test_run_coordinate_search_task(self, coordinator):
        """Test run method with 'coordinate_search' task."""
        context = {
            "target_aesthetics": {"contrast": 0.6, "warmth": 0.5},
            "fixed_parameters": {},
        }

        result = await coordinator.run("coordinate_search", context=context)

        assert isinstance(result, SubagentResult)
        assert result.success is True
        assert result.task == "coordinate_search"
        assert "chemistry_suggestion" in result.result
        assert "evaluation" in result.result

    @pytest.mark.asyncio
    async def test_run_evaluate_parameters_task(self, coordinator):
        """Test run method with 'evaluate_parameters' task."""
        context = {
            "parameters": {
                "metal_ratio": 0.5,
                "ferric_oxalate_pct": 20.0,
                "coating_weight": 1.5,
                "exposure_time": 10.0,
                "developer_temp": 20.0,
                "humidity": 50.0,
            }
        }

        result = await coordinator.run("evaluate_parameters", context=context)

        assert isinstance(result, SubagentResult)
        assert result.success is True
        assert result.task == "evaluate_parameters"
        assert "quality_score" in result.result

    @pytest.mark.asyncio
    async def test_run_unknown_task_fails(self, coordinator):
        """Test run method with unknown task returns error."""
        result = await coordinator.run("unknown_task")

        assert isinstance(result, SubagentResult)
        assert result.success is False
        assert "unknown" in result.error.lower()


# =============================================================================
# MCTS Router Tests (require FastAPI)
# =============================================================================

# Check if FastAPI is available
try:
    import fastapi  # noqa: F401

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestMCTSRouter:
    """Test MCTS API router functionality."""

    def test_create_mcts_router_returns_router(self):
        """Test create_mcts_router returns APIRouter."""
        from ptpd_calibration.api.mcts_router import create_mcts_router

        router = create_mcts_router()

        assert router is not None
        assert hasattr(router, "routes")
        assert router.prefix == "/api/mcts"
        assert "mcts" in router.tags


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestMCTSModels:
    """Test MCTS Pydantic model validation."""

    def test_mcts_search_request_validation(self):
        """Test MCTSSearchRequest model validation."""
        from ptpd_calibration.api.mcts_router import MCTSSearchRequest

        # Valid request with all fields
        request = MCTSSearchRequest(
            paper_type="Arches Platine",
            uv_source="metal_halide",
            target_curve=[0.0, 0.5, 1.0, 1.5, 2.0],
            fixed_parameters={"metal_ratio": 0.5},
            target_aesthetics={"contrast": 0.7},
            num_simulations=500,
        )

        assert request.paper_type == "Arches Platine"
        assert request.uv_source == "metal_halide"
        assert len(request.target_curve) == 5
        assert request.fixed_parameters["metal_ratio"] == 0.5
        assert request.target_aesthetics["contrast"] == 0.7
        assert request.num_simulations == 500

    def test_mcts_search_request_defaults(self):
        """Test MCTSSearchRequest uses defaults for optional fields."""
        from ptpd_calibration.api.mcts_router import MCTSSearchRequest

        request = MCTSSearchRequest()

        assert request.paper_type is None
        assert request.uv_source is None
        assert request.target_curve is None
        assert request.fixed_parameters == {}
        assert request.target_aesthetics == {}
        assert request.num_simulations is None

    def test_mcts_search_request_num_simulations_bounds(self):
        """Test MCTSSearchRequest validates num_simulations bounds."""
        from pydantic import ValidationError

        from ptpd_calibration.api.mcts_router import MCTSSearchRequest

        # Too low
        with pytest.raises(ValidationError):
            MCTSSearchRequest(num_simulations=10)

        # Too high
        with pytest.raises(ValidationError):
            MCTSSearchRequest(num_simulations=20000)

        # Valid
        request = MCTSSearchRequest(num_simulations=100)
        assert request.num_simulations == 100

    def test_mcts_evaluate_request_validation(self):
        """Test MCTSEvaluateRequest model validation."""
        from ptpd_calibration.api.mcts_router import MCTSEvaluateRequest

        params = {
            "metal_ratio": 0.5,
            "ferric_oxalate_pct": 20.0,
            "coating_weight": 1.5,
        }

        request = MCTSEvaluateRequest(parameters=params)

        assert request.parameters == params
        assert request.parameters["metal_ratio"] == 0.5

    def test_mcts_status_response_model(self):
        """Test MCTSStatusResponse model."""
        from ptpd_calibration.api.mcts_router import MCTSStatusResponse

        response = MCTSStatusResponse(
            engine_ready=True,
            networks_loaded=False,
            torch_available=True,
            parameter_ranges={
                "metal_ratio": {
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "unit": "ratio",
                }
            },
        )

        assert response.engine_ready is True
        assert response.networks_loaded is False
        assert response.torch_available is True
        assert "metal_ratio" in response.parameter_ranges

    def test_mcts_recommendation_model(self):
        """Test MCTSRecommendation model."""
        from ptpd_calibration.api.mcts_router import MCTSRecommendation

        rec = MCTSRecommendation(
            parameters={"metal_ratio": 0.5},
            predicted_quality=0.85,
            rationale="Optimized for high contrast",
        )

        assert rec.parameters["metal_ratio"] == 0.5
        assert rec.predicted_quality == 0.85
        assert "contrast" in rec.rationale

    def test_mcts_train_request_validation(self):
        """Test MCTSTrainRequest validates episode bounds."""
        from pydantic import ValidationError

        from ptpd_calibration.api.mcts_router import MCTSTrainRequest

        # Too low
        with pytest.raises(ValidationError):
            MCTSTrainRequest(num_episodes=5)

        # Too high
        with pytest.raises(ValidationError):
            MCTSTrainRequest(num_episodes=15000)

        # Valid
        request = MCTSTrainRequest(num_episodes=100)
        assert request.num_episodes == 100

    def test_mcts_feedback_request_quality_bounds(self):
        """Test MCTSFeedbackRequest validates quality rating bounds."""
        from pydantic import ValidationError

        from ptpd_calibration.api.mcts_router import MCTSFeedbackRequest

        params = {"metal_ratio": 0.5}
        curve = [0.0, 1.0, 2.0]

        # Too low
        with pytest.raises(ValidationError):
            MCTSFeedbackRequest(
                parameters=params,
                measured_curve=curve,
                quality_rating=-0.1,
            )

        # Too high
        with pytest.raises(ValidationError):
            MCTSFeedbackRequest(
                parameters=params,
                measured_curve=curve,
                quality_rating=1.5,
            )

        # Valid
        request = MCTSFeedbackRequest(
            parameters=params,
            measured_curve=curve,
            quality_rating=0.75,
        )
        assert request.quality_rating == 0.75
