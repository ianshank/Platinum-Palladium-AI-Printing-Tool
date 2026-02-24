"""
Tests for ExtendedProcessSimulator.

Validates physics model mapping from calibration parameters to process characteristics
and density curve generation.
"""

import logging

import pytest

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, PhysicsConstants
from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator

logger = logging.getLogger(__name__)


@pytest.fixture
def simulator() -> ExtendedProcessSimulator:
    """Create a simulator with default settings."""
    return ExtendedProcessSimulator()


@pytest.fixture
def pure_platinum_params() -> dict[str, float]:
    """Pure platinum calibration parameters."""
    return {
        "metal_ratio": 1.0,  # 100% Pt
        "coating_weight": 1.5,
        "ferric_oxalate_pct": 20.0,
        "exposure_time": 180.0,
        "developer_temp": 25.0,
        "humidity": 50.0,
    }


@pytest.fixture
def pure_palladium_params() -> dict[str, float]:
    """Pure palladium calibration parameters."""
    return {
        "metal_ratio": 0.0,  # 100% Pd
        "coating_weight": 1.5,
        "ferric_oxalate_pct": 20.0,
        "exposure_time": 180.0,
        "developer_temp": 25.0,
        "humidity": 50.0,
    }


@pytest.fixture
def default_params() -> dict[str, float]:
    """Default calibration parameters."""
    return {
        param: param_range.default_value
        for param, param_range in DEFAULT_PARAMETER_RANGES.items()
    }


class TestComputeProcessParameters:
    """Tests for compute_process_parameters method."""

    def test_pure_platinum_gamma(
        self,
        simulator: ExtendedProcessSimulator,
        pure_platinum_params: dict[str, float],
    ) -> None:
        """Pure platinum should produce gamma near pt_gamma_base."""
        process_params = simulator.compute_process_parameters(pure_platinum_params)
        expected_gamma = simulator.physics.pt_gamma_base
        assert abs(process_params.gamma - expected_gamma) < 0.01

    def test_pure_palladium_gamma(
        self,
        simulator: ExtendedProcessSimulator,
        pure_palladium_params: dict[str, float],
    ) -> None:
        """Pure palladium should produce gamma near pd_gamma_base."""
        process_params = simulator.compute_process_parameters(pure_palladium_params)
        expected_gamma = simulator.physics.pd_gamma_base
        assert abs(process_params.gamma - expected_gamma) < 0.01

    def test_mixed_metal_gamma(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """50/50 mix should produce gamma between pt and pd."""
        process_params = simulator.compute_process_parameters(default_params)
        pt_gamma = simulator.physics.pt_gamma_base
        pd_gamma = simulator.physics.pd_gamma_base
        assert min(pt_gamma, pd_gamma) <= process_params.gamma <= max(pt_gamma, pd_gamma)

    def test_exposure_increases_dmax(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Higher exposure time should produce higher dmax."""
        params_short = {**default_params, "exposure_time": 60.0}
        params_long = {**default_params, "exposure_time": 300.0}

        process_short = simulator.compute_process_parameters(params_short)
        process_long = simulator.compute_process_parameters(params_long)

        assert process_long.dmax > process_short.dmax

    def test_coating_weight_increases_dmax(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Higher coating weight should produce higher dmax."""
        params_light = {**default_params, "coating_weight": 0.8}
        params_heavy = {**default_params, "coating_weight": 2.5}

        process_light = simulator.compute_process_parameters(params_light)
        process_heavy = simulator.compute_process_parameters(params_heavy)

        # Note: dmax is also limited by exposure, so this might not always hold
        # if exposure is very short. We use default exposure (180s).
        assert process_heavy.dmax >= process_light.dmax

    def test_ferric_oxalate_affects_contrast(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """FO% deviation from center should affect contrast."""
        params_low = {**default_params, "ferric_oxalate_pct": 16.0}
        params_center = {**default_params, "ferric_oxalate_pct": 20.0}
        params_high = {**default_params, "ferric_oxalate_pct": 25.0}

        process_low = simulator.compute_process_parameters(params_low)
        process_center = simulator.compute_process_parameters(params_center)
        process_high = simulator.compute_process_parameters(params_high)

        # Low FO should have lower contrast, high FO higher contrast
        assert process_low.contrast < process_center.contrast
        assert process_high.contrast > process_center.contrast

    def test_humidity_affects_toe(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Humidity away from optimal should affect toe position."""
        params_optimal = {**default_params, "humidity": 50.0}
        params_dry = {**default_params, "humidity": 30.0}
        params_humid = {**default_params, "humidity": 70.0}

        process_optimal = simulator.compute_process_parameters(params_optimal)
        process_dry = simulator.compute_process_parameters(params_dry)
        process_humid = simulator.compute_process_parameters(params_humid)

        # Deviation from optimal should change toe (exact direction depends on physics model)
        # At least verify it has some effect
        assert (
            process_dry.toe_position != process_optimal.toe_position
            or process_humid.toe_position != process_optimal.toe_position
        )

    def test_edge_ferric_oxalate_values(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """FO% outside normal range should still produce valid parameters."""
        params_min = {**default_params, "ferric_oxalate_pct": 15.0}
        params_max = {**default_params, "ferric_oxalate_pct": 27.0}

        process_min = simulator.compute_process_parameters(params_min)
        process_max = simulator.compute_process_parameters(params_max)

        # Should produce valid results
        assert 0.0 < process_min.gamma < 5.0
        assert 0.0 < process_max.gamma < 5.0
        assert 0.0 <= process_min.dmin < process_min.dmax
        assert 0.0 <= process_max.dmin < process_max.dmax

    def test_developer_temp_affects_shoulder(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Developer temperature should affect shoulder position."""
        params_cold = {**default_params, "developer_temp": 20.0}
        params_hot = {**default_params, "developer_temp": 35.0}

        process_cold = simulator.compute_process_parameters(params_cold)
        process_hot = simulator.compute_process_parameters(params_hot)

        # Temperature should affect shoulder
        assert process_cold.shoulder_position != process_hot.shoulder_position


class TestSimulate:
    """Tests for simulate method."""

    def test_returns_correct_num_steps(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Simulation should return requested number of density points."""
        for num_steps in [11, 21, 31, 41]:
            result = simulator.simulate(default_params, num_steps=num_steps)
            assert len(result.density_curve) == num_steps

    def test_density_monotonic_increasing(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Density curve should be monotonically increasing (or at least non-decreasing)."""
        result = simulator.simulate(default_params, num_steps=21)
        curve = result.density_curve

        # Check monotonicity
        for i in range(len(curve) - 1):
            assert curve[i + 1] >= curve[i] - 1e-6  # Allow tiny numerical errors

    def test_dmin_dmax_match_curve_extremes(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Reported dmin and dmax should match curve extremes."""
        result = simulator.simulate(default_params, num_steps=21)

        curve_min = min(result.density_curve)
        curve_max = max(result.density_curve)

        assert abs(result.dmin - curve_min) < 1e-6
        assert abs(result.dmax - curve_max) < 1e-6

    def test_density_range_computed_correctly(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Density range should be dmax - dmin."""
        result = simulator.simulate(default_params, num_steps=21)
        expected_range = result.dmax - result.dmin
        assert abs(result.density_range - expected_range) < 1e-6

    def test_parameters_stored(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Simulation result should store input parameters."""
        result = simulator.simulate(default_params, num_steps=21)
        assert result.parameters == default_params

    def test_gamma_stored(
        self,
        simulator: ExtendedProcessSimulator,
        pure_platinum_params: dict[str, float],
    ) -> None:
        """Simulation result should store computed gamma."""
        result = simulator.simulate(pure_platinum_params, num_steps=21)
        assert result.gamma > 0.0
        # Should be close to pt_gamma_base
        assert abs(result.gamma - simulator.physics.pt_gamma_base) < 0.1


class TestNumpyTorchEquivalence:
    """Tests comparing NumPy and PyTorch simulation paths."""

    def test_numpy_torch_similar_results(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """NumPy and PyTorch paths should produce similar density curves."""
        if not simulator.use_torch:
            pytest.skip("PyTorch not available")

        num_steps = 21

        # Force NumPy path
        result_numpy = simulator.simulate_with_numpy(default_params, num_steps)

        # Use PyTorch path
        result_torch = simulator.simulate_with_torch(default_params, num_steps)

        # Compare density curves (allow small numerical differences)
        assert len(result_numpy.density_curve) == len(result_torch.density_curve)
        for i in range(len(result_numpy.density_curve)):
            np_val = result_numpy.density_curve[i]
            torch_val = result_torch.density_curve[i]
            assert abs(np_val - torch_val) < 0.01  # 1% tolerance

        # Compare key metrics
        assert abs(result_numpy.dmin - result_torch.dmin) < 0.01
        assert abs(result_numpy.dmax - result_torch.dmax) < 0.01
        assert abs(result_numpy.gamma - result_torch.gamma) < 0.01

    def test_numpy_fallback_when_torch_unavailable(
        self,
        default_params: dict[str, float],
    ) -> None:
        """Simulator should work with NumPy even if PyTorch is unavailable."""
        # Create simulator and force NumPy mode
        sim = ExtendedProcessSimulator()
        original_use_torch = sim.use_torch
        sim.use_torch = False

        try:
            result = sim.simulate(default_params, num_steps=21)
            assert len(result.density_curve) == 21
            assert result.dmin >= 0.0
            assert result.dmax > result.dmin
        finally:
            # Restore original state
            sim.use_torch = original_use_torch


class TestPhysicsConstants:
    """Tests for custom physics constants."""

    def test_custom_physics_constants(
        self,
        default_params: dict[str, float],
    ) -> None:
        """Simulator should use custom physics constants."""
        # Create custom physics with different pt_gamma_base
        custom_physics = PhysicsConstants(pt_gamma_base=2.5, pd_gamma_base=3.0)
        sim = ExtendedProcessSimulator(physics=custom_physics)

        # Pure platinum should now have gamma ~2.5
        params_pt = {**default_params, "metal_ratio": 1.0}
        process_params = sim.compute_process_parameters(params_pt)

        assert abs(process_params.gamma - 2.5) < 0.01

    def test_physics_constants_validation(self) -> None:
        """Physics constants should validate field constraints."""
        # This should raise ValidationError due to pt_gamma_base out of range
        with pytest.raises(Exception):  # Pydantic ValidationError
            PhysicsConstants(pt_gamma_base=5.0)  # Max is 3.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_parameters(
        self,
        simulator: ExtendedProcessSimulator,
    ) -> None:
        """Empty parameters should use defaults from method."""
        result = simulator.simulate({}, num_steps=21)
        # Should still produce valid result
        assert len(result.density_curve) == 21
        assert result.dmin >= 0.0
        assert result.dmax > result.dmin

    def test_partial_parameters(
        self,
        simulator: ExtendedProcessSimulator,
    ) -> None:
        """Partial parameters should use defaults for missing values."""
        params = {"metal_ratio": 0.8, "exposure_time": 240.0}
        result = simulator.simulate(params, num_steps=21)

        # Should still produce valid result
        assert len(result.density_curve) == 21
        assert result.dmin >= 0.0

    def test_extreme_exposure_time(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Very long exposure should asymptotically approach dmax ceiling."""
        params_extreme = {**default_params, "exposure_time": 1200.0}  # 20 minutes
        result = simulator.simulate(params_extreme, num_steps=21)

        # Dmax should be near ceiling (within reasonable range)
        # But not exceed it
        assert result.dmax <= simulator.physics.exposure_dmax_ceiling + 0.1

    def test_zero_exposure_time(
        self,
        simulator: ExtendedProcessSimulator,
        default_params: dict[str, float],
    ) -> None:
        """Zero exposure should produce very low dmax."""
        params_zero = {**default_params, "exposure_time": 0.0}
        result = simulator.simulate(params_zero, num_steps=21)

        # Dmax should be close to dmin (very low)
        assert result.dmax < 1.0  # Should be significantly below normal
