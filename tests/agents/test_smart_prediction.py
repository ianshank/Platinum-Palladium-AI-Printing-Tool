"""Tests for the predictive quality assessment system.

Tests cover:
- Quality feature handling
- Quality prediction
- Historical data management
- Risk identification
- Recommendation generation
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from ptpd_calibration.agents.smart.prediction import (
    HistoricalData,
    PredictionResult,
    PredictionSettings,
    QualityFeatures,
    QualityGrade,
    QualityPredictor,
    RiskLevel,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def prediction_settings() -> PredictionSettings:
    """Create prediction settings for tests."""
    return PredictionSettings(
        min_data_points=5,
        target_dmax=2.0,
        target_dmin=0.1,
        target_contrast=1.8,
    )


@pytest.fixture
def quality_predictor(prediction_settings: PredictionSettings) -> QualityPredictor:
    """Create quality predictor for tests."""
    return QualityPredictor(settings=prediction_settings)


@pytest.fixture
def optimal_features() -> QualityFeatures:
    """Create optimal quality features."""
    return QualityFeatures(
        dmin=0.1,
        dmax=2.0,
        contrast=1.8,
        curve_smoothness=0.9,
        metal_ratio=1.2,
        coating_amount=1.75,
        sensitizer_ratio=0.2,
        temperature=21.0,
        humidity=50.0,
        exposure_time=180.0,
        paper_type="platine",
        coating_age_hours=1.0,
    )


@pytest.fixture
def suboptimal_features() -> QualityFeatures:
    """Create suboptimal quality features."""
    return QualityFeatures(
        dmin=0.2,
        dmax=1.5,
        contrast=1.2,
        curve_smoothness=0.5,
        metal_ratio=0.8,
        coating_amount=1.0,
        sensitizer_ratio=0.1,
        temperature=28.0,
        humidity=70.0,
        exposure_time=100.0,
        paper_type="unknown",
        coating_age_hours=20.0,
    )


# ============================================================================
# QualityGrade Tests
# ============================================================================


class TestQualityGrade:
    """Tests for QualityGrade enum."""

    def test_grades_exist(self) -> None:
        """Test all quality grades are defined."""
        assert QualityGrade.EXCELLENT == "excellent"
        assert QualityGrade.GOOD == "good"
        assert QualityGrade.ACCEPTABLE == "acceptable"
        assert QualityGrade.POOR == "poor"
        assert QualityGrade.UNACCEPTABLE == "unacceptable"

    def test_grade_count(self) -> None:
        """Test correct number of grades."""
        assert len(QualityGrade) == 5


# ============================================================================
# RiskLevel Tests
# ============================================================================


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels_exist(self) -> None:
        """Test all risk levels are defined."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"


# ============================================================================
# QualityFeatures Tests
# ============================================================================


class TestQualityFeatures:
    """Tests for QualityFeatures dataclass."""

    def test_default_creation(self) -> None:
        """Test creation with defaults."""
        features = QualityFeatures()

        assert features.dmin == 0.0
        assert features.dmax == 0.0
        assert features.temperature == 21.0
        assert features.humidity == 50.0

    def test_full_creation(self, optimal_features: QualityFeatures) -> None:
        """Test creation with all fields."""
        assert optimal_features.dmax == 2.0
        assert optimal_features.metal_ratio == 1.2
        assert optimal_features.paper_type == "platine"

    def test_to_vector(self, optimal_features: QualityFeatures) -> None:
        """Test conversion to feature vector."""
        vector = optimal_features.to_vector()

        assert isinstance(vector, list)
        assert len(vector) == 11  # All numerical features
        assert all(isinstance(v, (int, float)) for v in vector)

    def test_feature_names(self) -> None:
        """Test feature names."""
        names = QualityFeatures.feature_names()

        assert len(names) == 11
        assert "dmax" in names
        assert "temperature" in names
        assert "paper_type" not in names  # Non-numerical


# ============================================================================
# HistoricalData Tests
# ============================================================================


class TestHistoricalData:
    """Tests for HistoricalData dataclass."""

    def test_creation(self, optimal_features: QualityFeatures) -> None:
        """Test historical data creation."""
        data = HistoricalData(
            features=optimal_features,
            actual_quality=85.0,
            quality_grade=QualityGrade.GOOD,
            timestamp=datetime.now(timezone.utc),
            issues=[],
            notes="Test print",
        )

        assert data.actual_quality == 85.0
        assert data.quality_grade == QualityGrade.GOOD

    def test_to_dict(self, optimal_features: QualityFeatures) -> None:
        """Test conversion to dictionary."""
        data = HistoricalData(
            features=optimal_features,
            actual_quality=85.0,
            quality_grade=QualityGrade.GOOD,
            timestamp=datetime.now(timezone.utc),
        )

        result = data.to_dict()

        assert "features" in result
        assert result["actual_quality"] == 85.0
        assert result["quality_grade"] == "good"
        assert "timestamp" in result

    def test_with_issues(self, optimal_features: QualityFeatures) -> None:
        """Test historical data with issues."""
        data = HistoricalData(
            features=optimal_features,
            actual_quality=60.0,
            quality_grade=QualityGrade.ACCEPTABLE,
            timestamp=datetime.now(timezone.utc),
            issues=["uneven coating", "dust particles"],
        )

        assert len(data.issues) == 2


# ============================================================================
# PredictionSettings Tests
# ============================================================================


class TestPredictionSettings:
    """Tests for PredictionSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = PredictionSettings()

        assert settings.min_data_points == 10
        assert settings.target_dmax == 2.0
        assert settings.target_dmin == 0.1
        assert settings.high_risk_threshold == 0.7

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = PredictionSettings(
            min_data_points=20,
            target_dmax=2.2,
        )

        assert settings.min_data_points == 20
        assert settings.target_dmax == 2.2

    def test_weight_sum(self) -> None:
        """Test that weights sum appropriately."""
        settings = PredictionSettings()

        total = (
            settings.dmax_weight +
            settings.contrast_weight +
            settings.smoothness_weight +
            settings.environment_weight +
            settings.chemistry_weight +
            settings.process_weight
        )

        assert abs(total - 1.0) < 0.01


# ============================================================================
# PredictionResult Tests
# ============================================================================


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_get_highest_risk_empty(self) -> None:
        """Test getting highest risk with no risks."""
        result = PredictionResult(
            predicted_quality=85.0,
            quality_grade=QualityGrade.GOOD,
            confidence=0.8,
            confidence_interval=(80.0, 90.0),
            risk_factors=[],
            recommendations=[],
            feature_contributions={},
            data_points_used=10,
        )

        assert result.get_highest_risk() is None

    def test_get_highest_risk(self) -> None:
        """Test getting highest risk."""
        risks = [
            {"factor": "low", "severity": 0.3},
            {"factor": "high", "severity": 0.8},
            {"factor": "medium", "severity": 0.5},
        ]

        result = PredictionResult(
            predicted_quality=70.0,
            quality_grade=QualityGrade.ACCEPTABLE,
            confidence=0.7,
            confidence_interval=(60.0, 80.0),
            risk_factors=risks,
            recommendations=[],
            feature_contributions={},
            data_points_used=10,
        )

        highest = result.get_highest_risk()
        assert highest["factor"] == "high"

    def test_is_acceptable_true(self) -> None:
        """Test acceptable quality."""
        for grade in [QualityGrade.EXCELLENT, QualityGrade.GOOD, QualityGrade.ACCEPTABLE]:
            result = PredictionResult(
                predicted_quality=75.0,
                quality_grade=grade,
                confidence=0.8,
                confidence_interval=(70.0, 80.0),
                risk_factors=[],
                recommendations=[],
                feature_contributions={},
                data_points_used=10,
            )
            assert result.is_acceptable() is True

    def test_is_acceptable_false(self) -> None:
        """Test unacceptable quality."""
        for grade in [QualityGrade.POOR, QualityGrade.UNACCEPTABLE]:
            result = PredictionResult(
                predicted_quality=40.0,
                quality_grade=grade,
                confidence=0.8,
                confidence_interval=(30.0, 50.0),
                risk_factors=[],
                recommendations=[],
                feature_contributions={},
                data_points_used=10,
            )
            assert result.is_acceptable() is False


# ============================================================================
# QualityPredictor Tests
# ============================================================================


class TestQualityPredictor:
    """Tests for QualityPredictor."""

    def test_creation(self, quality_predictor: QualityPredictor) -> None:
        """Test predictor creation."""
        assert quality_predictor.settings is not None
        assert len(quality_predictor._history) == 0

    def test_predict_optimal_features(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test prediction with optimal features."""
        result = quality_predictor.predict(optimal_features)

        assert result.predicted_quality > 70  # Should be good or better
        assert result.quality_grade in [
            QualityGrade.EXCELLENT,
            QualityGrade.GOOD,
            QualityGrade.ACCEPTABLE
        ]
        assert 0 <= result.confidence <= 1

    def test_predict_suboptimal_features(
        self,
        quality_predictor: QualityPredictor,
        suboptimal_features: QualityFeatures
    ) -> None:
        """Test prediction with suboptimal features."""
        result = quality_predictor.predict(suboptimal_features)

        assert len(result.risk_factors) > 0
        assert len(result.recommendations) > 0

    def test_predict_confidence_interval(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test confidence interval calculation."""
        result = quality_predictor.predict(optimal_features)

        low, high = result.confidence_interval
        assert low <= result.predicted_quality <= high
        assert low >= 0
        assert high <= 100

    def test_predict_feature_contributions(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test feature contributions in prediction."""
        result = quality_predictor.predict(optimal_features)

        assert "calibration" in result.feature_contributions
        assert "chemistry" in result.feature_contributions
        assert "environment" in result.feature_contributions
        assert "process" in result.feature_contributions

    def test_add_historical_data(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test adding historical data."""
        record = quality_predictor.add_historical_data(
            features=optimal_features,
            actual_quality=85.0,
            issues=["minor dust"],
            notes="Test print",
        )

        assert record.actual_quality == 85.0
        assert len(quality_predictor._history) == 1

    def test_get_similar_prints_empty(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test getting similar prints with no history."""
        similar = quality_predictor.get_similar_prints(optimal_features)

        assert len(similar) == 0

    def test_get_similar_prints_with_history(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test getting similar prints with history."""
        # Add some historical data
        quality_predictor.add_historical_data(
            features=optimal_features,
            actual_quality=85.0,
        )

        # Query with same features
        similar = quality_predictor.get_similar_prints(optimal_features)

        assert len(similar) == 1
        assert similar[0][1] == 1.0  # Perfect match

    def test_get_statistics_empty(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test statistics with no history."""
        stats = quality_predictor.get_statistics()

        assert stats["total_prints"] == 0
        assert stats["has_sufficient_data"] is False

    def test_get_statistics_with_data(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test statistics with history."""
        # Add multiple records
        for quality in [90, 85, 80, 75, 70]:
            quality_predictor.add_historical_data(
                features=optimal_features,
                actual_quality=float(quality),
            )

        stats = quality_predictor.get_statistics()

        assert stats["total_prints"] == 5
        assert stats["has_sufficient_data"] is True
        assert stats["quality"]["mean"] == 80.0
        assert stats["quality"]["min"] == 70.0
        assert stats["quality"]["max"] == 90.0


# ============================================================================
# Risk Identification Tests
# ============================================================================


class TestRiskIdentification:
    """Tests for risk identification."""

    def test_identify_temperature_risk_low(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test low temperature risk identification."""
        features = QualityFeatures(temperature=15.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "low_temperature" in risk_factors

    def test_identify_temperature_risk_high(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test high temperature risk identification."""
        features = QualityFeatures(temperature=30.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "high_temperature" in risk_factors

    def test_identify_humidity_risk_low(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test low humidity risk identification."""
        features = QualityFeatures(humidity=30.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "low_humidity" in risk_factors

    def test_identify_humidity_risk_high(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test high humidity risk identification."""
        features = QualityFeatures(humidity=70.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "high_humidity" in risk_factors

    def test_identify_old_coating_risk(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test old coating risk identification."""
        features = QualityFeatures(coating_age_hours=20.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "old_coating" in risk_factors

    def test_identify_low_dmax_risk(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test low Dmax risk identification."""
        features = QualityFeatures(dmax=1.2)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "low_dmax" in risk_factors

    def test_identify_exposure_risk_short(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test short exposure risk identification."""
        features = QualityFeatures(exposure_time=60.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "short_exposure" in risk_factors

    def test_identify_exposure_risk_long(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test long exposure risk identification."""
        features = QualityFeatures(exposure_time=500.0)
        result = quality_predictor.predict(features)

        risk_factors = [r["factor"] for r in result.risk_factors]
        assert "long_exposure" in risk_factors

    def test_risk_severity_ordering(
        self,
        quality_predictor: QualityPredictor,
        suboptimal_features: QualityFeatures
    ) -> None:
        """Test risks are ordered by severity."""
        result = quality_predictor.predict(suboptimal_features)

        if len(result.risk_factors) >= 2:
            severities = [r["severity"] for r in result.risk_factors]
            assert severities == sorted(severities, reverse=True)


# ============================================================================
# Scoring Tests
# ============================================================================


class TestScoring:
    """Tests for quality scoring components."""

    def test_score_calibration_optimal(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test calibration scoring with optimal values."""
        features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            curve_smoothness=0.9,
        )

        score = quality_predictor._score_calibration(features)
        # Optimal values should score well (above 0.7)
        assert score > 0.7

    def test_score_calibration_suboptimal(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test calibration scoring with suboptimal values."""
        features = QualityFeatures(
            dmin=0.3,
            dmax=1.2,
            contrast=1.0,
            curve_smoothness=0.4,
        )

        score = quality_predictor._score_calibration(features)
        assert score < 0.7

    def test_score_chemistry_optimal(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test chemistry scoring with optimal values."""
        features = QualityFeatures(
            metal_ratio=1.2,
            coating_amount=1.75,
            sensitizer_ratio=0.2,
        )

        score = quality_predictor._score_chemistry(features)
        assert score > 0.8

    def test_score_environment_optimal(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test environment scoring with optimal values."""
        features = QualityFeatures(
            temperature=21.0,
            humidity=50.0,
        )

        score = quality_predictor._score_environment(features)
        assert score > 0.9

    def test_score_process_optimal(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test process scoring with optimal values."""
        features = QualityFeatures(
            exposure_time=180.0,
            coating_age_hours=1.0,
        )

        score = quality_predictor._score_process(features)
        assert score > 0.9


# ============================================================================
# Quality Grade Tests
# ============================================================================


class TestQualityGrading:
    """Tests for quality grading."""

    @pytest.mark.parametrize("quality,expected_grade", [
        (95, QualityGrade.EXCELLENT),
        (90, QualityGrade.EXCELLENT),
        (85, QualityGrade.GOOD),
        (75, QualityGrade.GOOD),
        (70, QualityGrade.ACCEPTABLE),
        (60, QualityGrade.ACCEPTABLE),
        (50, QualityGrade.POOR),
        (40, QualityGrade.POOR),
        (30, QualityGrade.UNACCEPTABLE),
        (0, QualityGrade.UNACCEPTABLE),
    ])
    def test_quality_to_grade(
        self,
        quality_predictor: QualityPredictor,
        quality: int,
        expected_grade: QualityGrade
    ) -> None:
        """Test quality to grade conversion."""
        grade = quality_predictor._quality_to_grade(float(quality))
        assert grade == expected_grade


# ============================================================================
# Integration Tests
# ============================================================================


class TestQualityPredictorIntegration:
    """Integration tests for QualityPredictor."""

    def test_full_prediction_workflow(
        self,
        quality_predictor: QualityPredictor,
        optimal_features: QualityFeatures
    ) -> None:
        """Test complete prediction workflow."""
        # Make initial prediction
        result1 = quality_predictor.predict(optimal_features)

        # Record actual result
        quality_predictor.add_historical_data(
            features=optimal_features,
            actual_quality=85.0,
            notes="Test print 1",
        )

        # Make another prediction (should have higher confidence)
        result2 = quality_predictor.predict(optimal_features)

        # Find similar prints
        similar = quality_predictor.get_similar_prints(optimal_features)

        # Check statistics
        stats = quality_predictor.get_statistics()

        assert result1.predicted_quality > 0
        assert len(similar) >= 1
        assert stats["total_prints"] == 1

    def test_learning_from_history(
        self,
        quality_predictor: QualityPredictor
    ) -> None:
        """Test predictor learns from history."""
        base_features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            temperature=21.0,
            humidity=50.0,
            exposure_time=180.0,
        )

        # Add historical prints with known outcomes
        for i in range(10):
            features = QualityFeatures(
                dmin=0.1 + i * 0.01,
                dmax=2.0 - i * 0.05,
                contrast=1.8,
                temperature=21.0,
                humidity=50.0,
                exposure_time=180.0,
            )
            quality = 90 - i * 3  # Quality decreases as features worsen
            quality_predictor.add_historical_data(
                features=features,
                actual_quality=float(quality),
            )

        # Get statistics
        stats = quality_predictor.get_statistics()

        assert stats["total_prints"] == 10
        assert stats["has_sufficient_data"] is True

        # Make prediction
        result = quality_predictor.predict(base_features)

        # Confidence should be higher with more data
        assert result.confidence > 0.5

    def test_recommendations_actionable(
        self,
        quality_predictor: QualityPredictor,
        suboptimal_features: QualityFeatures
    ) -> None:
        """Test recommendations are actionable."""
        result = quality_predictor.predict(suboptimal_features)

        # Should have recommendations
        assert len(result.recommendations) > 0

        # Recommendations should contain action words
        action_words = ["warm", "cool", "increase", "reduce", "allow", "consider", "check", "mix"]
        has_actionable = any(
            any(word in rec.lower() for word in action_words)
            for rec in result.recommendations
        )
        assert has_actionable
