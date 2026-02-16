"""
Unit tests for MCTS subagents.
"""

from __future__ import annotations

import pytest

from ptpd_calibration.agents.subagents.base import SubagentCapability, get_subagent_registry
from ptpd_calibration.mcts.agents import (
    CalibrationCoordinatorSubagent,
    ChemistrySubagent,
    ExposureSubagent,
)


class TestChemistrySubagent:
    """Tests for ChemistrySubagent."""

    def test_agent_registration(self):
        """Test that ChemistrySubagent is registered in the registry."""
        registry = get_subagent_registry()
        agent_class = registry.get_agent_class("mcts_chemistry")
        assert agent_class is ChemistrySubagent

    def test_capabilities(self):
        """Test that ChemistrySubagent has correct capabilities."""
        agent = ChemistrySubagent()
        capabilities = agent.capabilities()
        assert SubagentCapability.ANALYSIS in capabilities

    def test_agent_type(self):
        """Test that ChemistrySubagent has correct type."""
        assert ChemistrySubagent.AGENT_TYPE == "mcts_chemistry"

    def test_suggest_chemistry_default(self):
        """Test chemistry suggestion with default aesthetics."""
        agent = ChemistrySubagent()
        result = agent.suggest_chemistry()

        assert "metal_ratio" in result
        assert "ferric_oxalate_pct" in result
        assert "coating_weight" in result

        # Check values are in valid ranges
        assert 0.0 <= result["metal_ratio"] <= 1.0
        assert 15.0 <= result["ferric_oxalate_pct"] <= 27.0
        assert 0.5 <= result["coating_weight"] <= 3.0

    def test_suggest_chemistry_high_warmth(self):
        """Test chemistry suggestion with high warmth preference."""
        agent = ChemistrySubagent()
        result = agent.suggest_chemistry(target_aesthetics={"warmth": 1.0})

        # High warmth should give low metal_ratio (more palladium)
        assert result["metal_ratio"] < 0.3

    def test_suggest_chemistry_high_contrast(self):
        """Test chemistry suggestion with high contrast preference."""
        agent = ChemistrySubagent()
        result = agent.suggest_chemistry(target_aesthetics={"contrast": 1.0})

        # High contrast should give high FO%
        assert result["ferric_oxalate_pct"] > 22.0

    def test_suggest_chemistry_high_tonal_range(self):
        """Test chemistry suggestion with high tonal range preference."""
        agent = ChemistrySubagent()
        result = agent.suggest_chemistry(target_aesthetics={"tonal_range": 1.0})

        # High tonal range should give high coating weight
        assert result["coating_weight"] > 2.0

    @pytest.mark.asyncio
    async def test_run_suggest_chemistry(self):
        """Test run method with suggest_chemistry task."""
        agent = ChemistrySubagent()
        result = await agent.run(
            "suggest_chemistry",
            context={"target_aesthetics": {"warmth": 0.7}},
        )

        assert result.success
        assert result.agent_type == "mcts_chemistry"
        assert "metal_ratio" in result.result

    @pytest.mark.asyncio
    async def test_run_invalid_task(self):
        """Test run method with invalid task."""
        agent = ChemistrySubagent()
        result = await agent.run("invalid_task")

        assert not result.success
        assert result.error is not None


class TestExposureSubagent:
    """Tests for ExposureSubagent."""

    def test_agent_registration(self):
        """Test that ExposureSubagent is registered in the registry."""
        registry = get_subagent_registry()
        agent_class = registry.get_agent_class("mcts_exposure")
        assert agent_class is ExposureSubagent

    def test_capabilities(self):
        """Test that ExposureSubagent has correct capabilities."""
        agent = ExposureSubagent()
        capabilities = agent.capabilities()
        assert SubagentCapability.ANALYSIS in capabilities

    def test_agent_type(self):
        """Test that ExposureSubagent has correct type."""
        assert ExposureSubagent.AGENT_TYPE == "mcts_exposure"

    def test_suggest_exposure_default(self):
        """Test exposure suggestion with default chemistry."""
        agent = ExposureSubagent()
        result = agent.suggest_exposure({})

        assert "exposure_time" in result
        assert "developer_temp" in result
        assert "humidity" in result

        # Check values are in valid ranges
        assert 30.0 <= result["exposure_time"] <= 600.0
        assert 20.0 <= result["developer_temp"] <= 50.0
        assert 30.0 <= result["humidity"] <= 80.0

    def test_suggest_exposure_heavy_coating(self):
        """Test exposure suggestion with heavy coating."""
        agent = ExposureSubagent()
        result = agent.suggest_exposure({"coating_weight": 2.5})

        # Heavy coating should require longer exposure
        assert result["exposure_time"] > 200.0

    def test_suggest_exposure_sun_uv(self):
        """Test exposure suggestion with sun UV source."""
        agent = ExposureSubagent()
        result_sun = agent.suggest_exposure({}, uv_source="sun")
        result_led = agent.suggest_exposure({}, uv_source="uv_led")

        # Sun should be faster than UV LED
        assert result_sun["exposure_time"] < result_led["exposure_time"]

    @pytest.mark.asyncio
    async def test_run_suggest_exposure(self):
        """Test run method with suggest_exposure task."""
        agent = ExposureSubagent()
        result = await agent.run(
            "suggest_exposure",
            context={"chemistry_params": {"coating_weight": 2.0}},
        )

        assert result.success
        assert result.agent_type == "mcts_exposure"
        assert "exposure_time" in result.result


class TestCalibrationCoordinatorSubagent:
    """Tests for CalibrationCoordinatorSubagent."""

    def test_agent_registration(self):
        """Test that CalibrationCoordinatorSubagent is registered in the registry."""
        registry = get_subagent_registry()
        agent_class = registry.get_agent_class("mcts_coordinator")
        assert agent_class is CalibrationCoordinatorSubagent

    def test_capabilities(self):
        """Test that CalibrationCoordinatorSubagent has correct capabilities."""
        agent = CalibrationCoordinatorSubagent()
        capabilities = agent.capabilities()
        assert SubagentCapability.ORCHESTRATION in capabilities
        assert SubagentCapability.ANALYSIS in capabilities

    def test_agent_type(self):
        """Test that CalibrationCoordinatorSubagent has correct type."""
        assert CalibrationCoordinatorSubagent.AGENT_TYPE == "mcts_coordinator"

    def test_has_specialized_agents(self):
        """Test that coordinator has chemistry and exposure agents."""
        agent = CalibrationCoordinatorSubagent()
        assert isinstance(agent.chemistry_agent, ChemistrySubagent)
        assert isinstance(agent.exposure_agent, ExposureSubagent)

    @pytest.mark.asyncio
    async def test_run_coordinate_search(self):
        """Test run method with coordinate_search task."""
        agent = CalibrationCoordinatorSubagent()
        result = await agent.run(
            "coordinate_search",
            context={
                "target_aesthetics": {"warmth": 0.6, "contrast": 0.7},
                "uv_source": "sun",
            },
        )

        assert result.success
        assert result.agent_type == "mcts_coordinator"
        assert "chemistry_suggestion" in result.result
        assert "exposure_suggestion" in result.result
        assert "full_parameters" in result.result
        assert "evaluation" in result.result

    @pytest.mark.asyncio
    async def test_run_evaluate_parameters(self):
        """Test run method with evaluate_parameters task."""
        agent = CalibrationCoordinatorSubagent()
        params = {
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        }
        result = await agent.run(
            "evaluate_parameters",
            context={"parameters": params},
        )

        assert result.success
        assert "quality_score" in result.result
        assert "predicted_curve" in result.result
        assert "dmin" in result.result
        assert "dmax" in result.result

    @pytest.mark.asyncio
    async def test_coordination_produces_valid_parameters(self):
        """Test that coordination produces valid parameter ranges."""
        agent = CalibrationCoordinatorSubagent()
        result = await agent.run(
            "coordinate_search",
            context={"target_aesthetics": {"warmth": 0.5}},
        )

        assert result.success
        full_params = result.result["full_parameters"]

        # Check all parameters are present
        expected_params = [
            "metal_ratio",
            "coating_weight",
            "ferric_oxalate_pct",
            "exposure_time",
            "developer_temp",
            "humidity",
        ]
        for param in expected_params:
            assert param in full_params
