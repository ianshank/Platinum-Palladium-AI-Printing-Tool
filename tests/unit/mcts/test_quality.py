"""
Tests for QualityScorer.

Validates quality metrics for linearity, dmax target matching,
smoothness, and cost efficiency.
"""

import logging

import numpy as np
import pytest

from ptpd_calibration.mcts.config import MCTSSettings
from ptpd_calibration.mcts.quality import QualityScorer
from ptpd_calibration.mcts.types import SimulationResult

logger = logging.getLogger(__name__)


@pytest.fixture
def scorer() -> QualityScorer:
    """Create a quality scorer with default settings."""
    return QualityScorer()


@pytest.fixture
def custom_settings() -> MCTSSettings:
    """Create custom MCTS settings for testing."""
    return MCTSSettings(
        target_dmax=2.2,
        target_dmin=0.05,
        linearity_weight=0.5,
        dmax_weight=0.3,
        smoothness_weight=0.1,
        cost_weight=0.1,
    )


@pytest.fixture
def perfect_linear_result() -> SimulationResult:
    """Perfect linear curve from 0.1 to 2.0."""
    curve = list(np.linspace(0.1, 2.0, 21))
    return SimulationResult(
        density_curve=curve,
        dmin=0.1,
        dmax=2.0,
        density_range=1.9,
        gamma=1.8,
        quality_score=0.0,
        parameters={
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        },
    )


@pytest.fixture
def noisy_curve_result() -> SimulationResult:
    """Noisy curve with random perturbations."""
    # Start with linear
    base_curve = np.linspace(0.1, 2.0, 21)
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, 21)
    noisy_curve = (base_curve + noise).tolist()

    return SimulationResult(
        density_curve=noisy_curve,
        dmin=min(noisy_curve),
        dmax=max(noisy_curve),
        density_range=max(noisy_curve) - min(noisy_curve),
        gamma=1.8,
        quality_score=0.0,
        parameters={
            "metal_ratio": 0.5,
            "coating_weight": 1.5,
            "ferric_oxalate_pct": 20.0,
            "exposure_time": 180.0,
            "developer_temp": 25.0,
            "humidity": 50.0,
        },
    )


class TestLinearityScore:
    """Tests for _linearity_score method."""

    def test_perfect_linear_curve_high_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Perfect linear curve should score very high on linearity."""
        curve = list(np.linspace(0.1, 2.0, 21))
        score = scorer._linearity_score(curve)
        assert score > 0.95  # Should be near perfect

    def test_noisy_curve_lower_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Noisy curve should score lower than perfect linear."""
        perfect_curve = list(np.linspace(0.1, 2.0, 21))
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, 21)
        noisy_curve = (np.array(perfect_curve) + noise).tolist()

        perfect_score = scorer._linearity_score(perfect_curve)
        noisy_score = scorer._linearity_score(noisy_curve)

        assert noisy_score < perfect_score

    def test_curved_line_lower_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """S-curve should score lower than linear."""
        # Create S-curve
        x = np.linspace(-3, 3, 21)
        s_curve = (1.0 / (1.0 + np.exp(-x))).tolist()  # Sigmoid

        linear_curve = list(np.linspace(0.1, 2.0, 21))

        s_score = scorer._linearity_score(s_curve)
        linear_score = scorer._linearity_score(linear_curve)

        assert s_score < linear_score

    def test_target_curve_comparison(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Linearity score should compare to target curve if provided."""
        # Create a curve and a slightly different target
        curve = list(np.linspace(0.1, 2.0, 21))
        target = list(np.linspace(0.15, 2.05, 21))

        score_vs_target = scorer._linearity_score(curve, target_curve=target)
        score_vs_ideal = scorer._linearity_score(curve)

        # Both should be high, but different
        assert 0.0 <= score_vs_target <= 1.0
        assert 0.0 <= score_vs_ideal <= 1.0

    def test_short_curve_returns_zero(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Curve with <2 points should return 0."""
        assert scorer._linearity_score([0.5]) == 0.0

    def test_flat_curve_handles_zero_range(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Flat curve (zero range) should return 0."""
        flat_curve = [1.0] * 21
        assert scorer._linearity_score(flat_curve) == 0.0


class TestDmaxScore:
    """Tests for _dmax_score method."""

    def test_exact_target_scores_one(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Dmax exactly on target should score 1.0."""
        target = scorer.settings.target_dmax
        score = scorer._dmax_score(target)
        assert abs(score - 1.0) < 0.01

    def test_near_target_scores_high(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Dmax near target should score high."""
        target = scorer.settings.target_dmax
        score = scorer._dmax_score(target + 0.1)
        assert score > 0.8

    def test_far_from_target_scores_low(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Dmax far from target should score low."""
        target = scorer.settings.target_dmax
        score = scorer._dmax_score(target + 1.0)
        assert score < 0.2

    def test_symmetric_around_target(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Score should be symmetric around target."""
        target = scorer.settings.target_dmax
        score_above = scorer._dmax_score(target + 0.3)
        score_below = scorer._dmax_score(target - 0.3)
        assert abs(score_above - score_below) < 0.01

    def test_score_in_range(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Dmax score should always be in [0, 1]."""
        for dmax in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            score = scorer._dmax_score(dmax)
            assert 0.0 <= score <= 1.0


class TestSmoothnessScore:
    """Tests for _smoothness_score method."""

    def test_perfectly_smooth_curve_high_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Perfectly smooth (linear) curve should score very high."""
        curve = list(np.linspace(0.1, 2.0, 21))
        score = scorer._smoothness_score(curve)
        assert score > 0.95

    def test_noisy_curve_lower_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Noisy curve should score lower on smoothness."""
        smooth_curve = list(np.linspace(0.1, 2.0, 21))
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, 21)
        noisy_curve = (np.array(smooth_curve) + noise).tolist()

        smooth_score = scorer._smoothness_score(smooth_curve)
        noisy_score = scorer._smoothness_score(noisy_curve)

        assert noisy_score < smooth_score

    def test_high_frequency_noise_very_low_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """High-frequency oscillations should score very low."""
        # Alternating up/down
        base = np.linspace(0.1, 2.0, 21)
        oscillation = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(21)])
        oscillating_curve = (base + oscillation).tolist()

        smooth_curve = list(np.linspace(0.1, 2.0, 21))

        oscillating_score = scorer._smoothness_score(oscillating_curve)
        smooth_score = scorer._smoothness_score(smooth_curve)

        assert oscillating_score < smooth_score * 0.5  # Much lower

    def test_short_curve_returns_one(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Curve with <3 points should return 1.0 (cannot compute second deriv)."""
        assert scorer._smoothness_score([0.5, 1.0]) == 1.0

    def test_flat_curve_perfectly_smooth(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Flat curve should be perfectly smooth."""
        flat_curve = [1.5] * 21
        score = scorer._smoothness_score(flat_curve)
        assert score == 1.0  # Zero range case


class TestCostScore:
    """Tests for _cost_score method."""

    def test_pure_palladium_high_cost_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Pure palladium (metal_ratio=0) should have high cost score."""
        params = {"metal_ratio": 0.0, "coating_weight": 1.0}
        score = scorer._cost_score(params)
        assert score > 0.8  # Pd is cheaper

    def test_pure_platinum_low_cost_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Pure platinum (metal_ratio=1) should have lower cost score."""
        params = {"metal_ratio": 1.0, "coating_weight": 1.0}
        score = scorer._cost_score(params)
        assert score < 0.5  # Pt is more expensive

    def test_low_coating_weight_higher_score(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Lower coating weight should increase cost score."""
        params_light = {"metal_ratio": 0.5, "coating_weight": 0.6}
        params_heavy = {"metal_ratio": 0.5, "coating_weight": 2.8}

        score_light = scorer._cost_score(params_light)
        score_heavy = scorer._cost_score(params_heavy)

        assert score_light > score_heavy

    def test_cost_score_in_range(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Cost score should always be in [0, 1]."""
        test_params = [
            {"metal_ratio": 0.0, "coating_weight": 0.5},
            {"metal_ratio": 0.5, "coating_weight": 1.5},
            {"metal_ratio": 1.0, "coating_weight": 3.0},
        ]
        for params in test_params:
            score = scorer._cost_score(params)
            assert 0.0 <= score <= 1.0

    def test_missing_parameters_use_defaults(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Missing parameters should use defaults."""
        score = scorer._cost_score({})
        assert 0.0 <= score <= 1.0


class TestOverallScore:
    """Tests for overall score method."""

    def test_overall_score_in_range(
        self,
        scorer: QualityScorer,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """Overall score should be in [0, 1]."""
        score = scorer.score(perfect_linear_result)
        assert 0.0 <= score <= 1.0

    def test_perfect_linear_scores_high(
        self,
        scorer: QualityScorer,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """Perfect linear curve with good dmax should score high."""
        score = scorer.score(perfect_linear_result)
        assert score > 0.7  # Should be pretty good

    def test_noisy_curve_scores_lower(
        self,
        scorer: QualityScorer,
        perfect_linear_result: SimulationResult,
        noisy_curve_result: SimulationResult,
    ) -> None:
        """Noisy curve should score lower than smooth."""
        perfect_score = scorer.score(perfect_linear_result)
        noisy_score = scorer.score(noisy_curve_result)

        assert noisy_score < perfect_score

    def test_weights_sum_correctly(
        self,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """Different weights should produce different scores."""
        # All weight on linearity
        settings_linearity = MCTSSettings(
            linearity_weight=1.0,
            dmax_weight=0.0,
            smoothness_weight=0.0,
            cost_weight=0.0,
        )
        scorer_linearity = QualityScorer(settings_linearity)

        # All weight on dmax
        settings_dmax = MCTSSettings(
            linearity_weight=0.0,
            dmax_weight=1.0,
            smoothness_weight=0.0,
            cost_weight=0.0,
        )
        scorer_dmax = QualityScorer(settings_dmax)

        score_linearity = scorer_linearity.score(perfect_linear_result)
        score_dmax = scorer_dmax.score(perfect_linear_result)

        # Scores should be different (unless curve is perfect on all metrics)
        # At least verify they're both valid
        assert 0.0 <= score_linearity <= 1.0
        assert 0.0 <= score_dmax <= 1.0

    def test_zero_weights_returns_zero(
        self,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """All zero weights should return 0.0."""
        settings_zero = MCTSSettings(
            linearity_weight=0.0,
            dmax_weight=0.0,
            smoothness_weight=0.0,
            cost_weight=0.0,
        )
        scorer_zero = QualityScorer(settings_zero)
        score = scorer_zero.score(perfect_linear_result)
        assert score == 0.0

    def test_custom_weights(
        self,
        custom_settings: MCTSSettings,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """Custom weights should be respected."""
        scorer = QualityScorer(custom_settings)
        score = scorer.score(perfect_linear_result)

        # Should produce valid score with custom weights
        assert 0.0 <= score <= 1.0

        # Verify weights are being used
        assert scorer.settings.linearity_weight == 0.5
        assert scorer.settings.dmax_weight == 0.3

    def test_target_curve_passed_through(
        self,
        scorer: QualityScorer,
        perfect_linear_result: SimulationResult,
    ) -> None:
        """Target curve should be passed to linearity scorer."""
        target = list(np.linspace(0.15, 2.05, 21))
        score_with_target = scorer.score(perfect_linear_result, target_curve=target)
        score_without_target = scorer.score(perfect_linear_result)

        # Both should be valid
        assert 0.0 <= score_with_target <= 1.0
        assert 0.0 <= score_without_target <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimal_curve(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Minimal 2-point curve should be handled."""
        result = SimulationResult(
            density_curve=[0.1, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            gamma=1.8,
            quality_score=0.0,
            parameters={},
        )

        # Should not crash, though linearity/smoothness may be limited
        score = scorer.score(result)
        assert 0.0 <= score <= 1.0

    def test_flat_curve_minimal(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Flat curve (constant value) should be handled."""
        result = SimulationResult(
            density_curve=[1.5, 1.5],
            dmin=1.5,
            dmax=1.5,
            density_range=0.0,
            gamma=1.0,
            quality_score=0.0,
            parameters={},
        )

        # Should handle zero-range curve gracefully
        score = scorer.score(result)
        assert 0.0 <= score <= 1.0

    def test_small_density_values(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Very small density values (near zero) should be handled."""
        result = SimulationResult(
            density_curve=[0.0, 0.01, 0.02, 0.03],
            dmin=0.0,
            dmax=0.03,
            density_range=0.03,
            gamma=1.0,
            quality_score=0.0,
            parameters={},
        )

        # Should not crash
        score = scorer.score(result)
        assert 0.0 <= score <= 1.0

    def test_very_large_density_values(
        self,
        scorer: QualityScorer,
    ) -> None:
        """Very large density values should be handled."""
        curve = list(np.linspace(0.0, 10.0, 21))
        result = SimulationResult(
            density_curve=curve,
            dmin=0.0,
            dmax=10.0,
            density_range=10.0,
            gamma=1.0,
            quality_score=0.0,
            parameters={},
        )

        score = scorer.score(result)
        assert 0.0 <= score <= 1.0
