"""
Tests for step wedge analyzer module.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.analysis.wedge_analyzer import (
    AnalysisWarning,
    AnalysisWarningLevel,
    QualityAssessment,
    QualityGrade,
    StepWedgeAnalyzer,
    WedgeAnalysisConfig,
    WedgeAnalysisResult,
)
from ptpd_calibration.config import TabletType
from ptpd_calibration.core.types import CurveType


class TestAnalysisWarning:
    """Tests for AnalysisWarning."""

    def test_warning_creation(self):
        """Test warning object creation."""
        warning = AnalysisWarning(
            level=AnalysisWarningLevel.WARNING,
            code="TEST_WARNING",
            message="This is a test warning",
            suggestion="Fix the issue",
        )

        assert warning.level == AnalysisWarningLevel.WARNING
        assert warning.code == "TEST_WARNING"
        assert warning.message == "This is a test warning"
        assert warning.suggestion == "Fix the issue"

    def test_warning_to_dict(self):
        """Test warning dictionary conversion."""
        warning = AnalysisWarning(
            level=AnalysisWarningLevel.ERROR,
            code="ERROR_CODE",
            message="Error message",
            affected_patches=[1, 2, 3],
        )

        d = warning.to_dict()

        assert d["level"] == "error"
        assert d["code"] == "ERROR_CODE"
        assert d["message"] == "Error message"
        assert d["affected_patches"] == [1, 2, 3]

    def test_warning_without_optional_fields(self):
        """Test warning without optional fields."""
        warning = AnalysisWarning(
            level=AnalysisWarningLevel.INFO,
            code="INFO_CODE",
            message="Info message",
        )

        d = warning.to_dict()

        assert "suggestion" not in d
        assert "affected_patches" not in d


class TestQualityAssessment:
    """Tests for QualityAssessment."""

    def test_quality_assessment_creation(self):
        """Test quality assessment creation."""
        assessment = QualityAssessment(
            grade=QualityGrade.GOOD,
            score=75.0,
            density_range_score=80.0,
            uniformity_score=70.0,
            monotonicity_score=90.0,
            detection_confidence=85.0,
            signal_to_noise=65.0,
        )

        assert assessment.grade == QualityGrade.GOOD
        assert assessment.score == 75.0
        assert assessment.density_range_score == 80.0

    def test_quality_assessment_to_dict(self):
        """Test quality assessment dictionary conversion."""
        assessment = QualityAssessment(
            grade=QualityGrade.EXCELLENT,
            score=92.5,
            density_range_score=95.0,
            uniformity_score=90.0,
            monotonicity_score=100.0,
            detection_confidence=88.0,
            signal_to_noise=85.0,
            warnings=[
                AnalysisWarning(
                    level=AnalysisWarningLevel.INFO,
                    code="TEST",
                    message="Test",
                )
            ],
            recommendations=["Good results!"],
        )

        d = assessment.to_dict()

        assert d["grade"] == "excellent"
        assert d["score"] == 92.5
        assert "metrics" in d
        assert d["metrics"]["density_range"] == 95.0
        assert len(d["warnings"]) == 1
        assert d["recommendations"] == ["Good results!"]


class TestWedgeAnalysisConfig:
    """Tests for WedgeAnalysisConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WedgeAnalysisConfig()

        assert config.tablet_type == TabletType.STOUFFER_21
        assert config.min_density_range == 1.5
        assert config.max_dmin == 0.15
        assert config.min_dmax == 1.8
        assert config.default_curve_type == CurveType.LINEAR
        assert config.auto_fix_reversals is True
        assert config.outlier_rejection is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = WedgeAnalysisConfig(
            tablet_type=TabletType.STOUFFER_31,
            min_density_range=1.2,
            default_curve_type=CurveType.AESTHETIC,
            auto_fix_reversals=False,
        )

        assert config.tablet_type == TabletType.STOUFFER_31
        assert config.min_density_range == 1.2
        assert config.default_curve_type == CurveType.AESTHETIC
        assert config.auto_fix_reversals is False

    def test_config_to_dict(self):
        """Test config dictionary conversion."""
        config = WedgeAnalysisConfig()
        d = config.to_dict()

        assert d["tablet_type"] == "stouffer_21"
        assert "min_density_range" in d
        assert "default_curve_type" in d


class TestWedgeAnalysisResult:
    """Tests for WedgeAnalysisResult."""

    def test_result_creation(self):
        """Test result object creation."""
        result = WedgeAnalysisResult(
            tablet_type=TabletType.STOUFFER_21,
            detection_success=True,
            densities=[0.1, 0.5, 1.0, 1.5, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
        )

        assert result.tablet_type == TabletType.STOUFFER_21
        assert result.detection_success is True
        assert len(result.densities) == 5
        assert result.dmin == 0.1
        assert result.dmax == 2.0

    def test_result_to_dict(self):
        """Test result dictionary conversion."""
        result = WedgeAnalysisResult(
            tablet_type=TabletType.STOUFFER_21,
            detection_success=True,
            densities=[0.1, 0.5, 1.0, 1.5, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
        )

        d = result.to_dict()

        assert "id" in d
        assert "timestamp" in d
        assert d["tablet_type"] == "stouffer_21"
        assert d["detection_success"] is True
        assert d["num_patches"] == 5
        assert d["dmin"] == 0.1

    def test_result_summary(self):
        """Test result summary generation."""
        quality = QualityAssessment(
            grade=QualityGrade.GOOD,
            score=75.0,
            density_range_score=80.0,
            uniformity_score=70.0,
            monotonicity_score=90.0,
            detection_confidence=85.0,
            signal_to_noise=65.0,
        )

        result = WedgeAnalysisResult(
            tablet_type=TabletType.STOUFFER_21,
            detection_success=True,
            densities=[0.1, 0.5, 1.0, 1.5, 2.0],
            dmin=0.1,
            dmax=2.0,
            density_range=1.9,
            quality=quality,
        )

        summary = result.summary()

        assert "Success" in summary
        assert "Dmin: 0.100" in summary
        assert "Dmax: 2.000" in summary
        assert "GOOD" in summary


class TestStepWedgeAnalyzer:
    """Tests for StepWedgeAnalyzer."""

    @pytest.fixture
    def sample_densities(self):
        """Generate sample density measurements."""
        # Typical Pt/Pd response curve
        steps = np.linspace(0, 1, 21)
        densities = 0.1 + 2.0 * (steps**0.85)
        return list(densities)

    @pytest.fixture
    def low_range_densities(self):
        """Generate low density range measurements."""
        steps = np.linspace(0, 1, 21)
        densities = 0.1 + 0.8 * steps  # Only 0.8 density range
        return list(densities)

    @pytest.fixture
    def high_dmin_densities(self):
        """Generate high Dmin measurements."""
        steps = np.linspace(0, 1, 21)
        densities = 0.3 + 1.7 * steps  # Dmin of 0.3
        return list(densities)

    @pytest.fixture
    def reversal_densities(self):
        """Generate measurements with reversals."""
        densities = [0.1, 0.3, 0.5, 0.7, 0.65, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        return densities

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StepWedgeAnalyzer()

    def test_init_default_config(self):
        """Test analyzer initialization with default config."""
        analyzer = StepWedgeAnalyzer()

        assert analyzer.config is not None
        assert analyzer.config.tablet_type == TabletType.STOUFFER_21

    def test_init_custom_config(self):
        """Test analyzer initialization with custom config."""
        config = WedgeAnalysisConfig(
            tablet_type=TabletType.STOUFFER_31,
            min_density_range=1.2,
        )
        analyzer = StepWedgeAnalyzer(config)

        assert analyzer.config.tablet_type == TabletType.STOUFFER_31
        assert analyzer.config.min_density_range == 1.2

    def test_analyze_from_densities_basic(self, analyzer, sample_densities):
        """Test basic density analysis."""
        result = analyzer.analyze_from_densities(
            sample_densities,
            curve_name="Test Curve",
            generate_curve=True,
        )

        assert result.detection_success is True
        assert len(result.densities) == 21
        assert result.dmin is not None
        assert result.dmax is not None
        assert result.density_range is not None
        assert result.quality is not None

    def test_analyze_from_densities_generates_curve(self, analyzer, sample_densities):
        """Test that curve is generated."""
        result = analyzer.analyze_from_densities(
            sample_densities,
            curve_name="Generated Curve",
            paper_type="Arches Platine",
            chemistry="50% Pt",
            generate_curve=True,
        )

        assert result.curve_generated is True
        assert result.curve is not None
        assert result.curve.name == "Generated Curve"
        assert result.curve.paper_type == "Arches Platine"

    def test_analyze_from_densities_no_curve(self, analyzer, sample_densities):
        """Test analysis without curve generation."""
        result = analyzer.analyze_from_densities(
            sample_densities,
            generate_curve=False,
        )

        assert result.curve_generated is False
        assert result.curve is None

    def test_analyze_from_densities_quality_grades(self, analyzer, sample_densities):
        """Test quality grading."""
        result = analyzer.analyze_from_densities(sample_densities)

        # Good density range should give good grade
        assert result.quality is not None
        assert result.quality.grade in [
            QualityGrade.EXCELLENT,
            QualityGrade.GOOD,
            QualityGrade.ACCEPTABLE,
        ]

    def test_analyze_low_density_range(self, analyzer, low_range_densities):
        """Test analysis with low density range."""
        result = analyzer.analyze_from_densities(low_range_densities)

        assert result.quality is not None
        # Should have warning about low density range
        warning_codes = [w.code for w in result.quality.warnings]
        assert "LOW_DENSITY_RANGE" in warning_codes

    def test_analyze_high_dmin(self, analyzer, high_dmin_densities):
        """Test analysis with high Dmin."""
        result = analyzer.analyze_from_densities(high_dmin_densities)

        assert result.quality is not None
        # Should have warning about high Dmin
        warning_codes = [w.code for w in result.quality.warnings]
        assert "HIGH_DMIN" in warning_codes

    def test_analyze_with_reversals(self, reversal_densities):
        """Test analysis with density reversals."""
        config = WedgeAnalysisConfig(auto_fix_reversals=True)
        analyzer = StepWedgeAnalyzer(config)

        result = analyzer.analyze_from_densities(reversal_densities)

        # Should fix the reversal
        assert result.densities[4] >= result.densities[3]  # No reversal

    def test_analyze_without_reversal_fix(self, reversal_densities):
        """Test analysis without reversal fixing."""
        # Disable both auto_fix_reversals AND outlier_rejection to preserve reversals
        config = WedgeAnalysisConfig(auto_fix_reversals=False, outlier_rejection=False)
        analyzer = StepWedgeAnalyzer(config)

        result = analyzer.analyze_from_densities(reversal_densities)

        # Reversal warning should be present
        warning_codes = [w.code for w in result.quality.warnings]
        assert "DENSITY_REVERSAL" in warning_codes

    def test_analyze_different_curve_types(self, analyzer, sample_densities):
        """Test analysis with different curve types."""
        for curve_type in [CurveType.LINEAR, CurveType.PAPER_WHITE, CurveType.AESTHETIC]:
            result = analyzer.analyze_from_densities(
                sample_densities,
                curve_type=curve_type,
                generate_curve=True,
            )

            assert result.curve is not None
            assert result.curve.curve_type == curve_type

    def test_quality_score_calculation(self, analyzer, sample_densities):
        """Test quality score is within valid range."""
        result = analyzer.analyze_from_densities(sample_densities)

        assert result.quality is not None
        assert 0 <= result.quality.score <= 100
        assert 0 <= result.quality.density_range_score <= 100
        assert 0 <= result.quality.uniformity_score <= 100
        assert 0 <= result.quality.monotonicity_score <= 100

    def test_input_values_normalized(self, analyzer, sample_densities):
        """Test that input values are properly normalized."""
        result = analyzer.analyze_from_densities(sample_densities)

        assert len(result.input_values) == len(sample_densities)
        assert result.input_values[0] == pytest.approx(0.0)
        assert result.input_values[-1] == pytest.approx(1.0)

    def test_get_supported_tablet_types(self):
        """Test getting supported tablet types."""
        types = StepWedgeAnalyzer.get_supported_tablet_types()

        assert TabletType.STOUFFER_21 in types
        assert TabletType.STOUFFER_31 in types
        assert TabletType.STOUFFER_41 in types
        assert TabletType.CUSTOM in types

    def test_get_default_config(self):
        """Test getting default config."""
        config = StepWedgeAnalyzer.get_default_config()

        assert isinstance(config, WedgeAnalysisConfig)
        assert config.tablet_type == TabletType.STOUFFER_21

    def test_generate_target_curve(self, analyzer):
        """Test target curve generation."""
        for curve_type in [CurveType.LINEAR, CurveType.PAPER_WHITE, CurveType.AESTHETIC]:
            target = analyzer.generate_target_curve(21, curve_type)

            assert len(target.input_values) == 21
            assert len(target.output_values) == 21
            assert target.input_values[0] == pytest.approx(0.0)
            assert target.input_values[-1] == pytest.approx(1.0)


class TestAnalysisWithImage:
    """Tests for analysis with actual images."""

    @pytest.fixture
    def sample_step_tablet_image(self, tmp_path):
        """Create a synthetic step tablet image for testing."""
        width, height = 420, 100
        num_patches = 21
        patch_width = width // num_patches

        img = np.zeros((height, width), dtype=np.uint8)

        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img[:, x_start:x_end] = value

        # Add margin
        full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
        full_img[20 : height + 20, 20 : width + 20, 0] = img
        full_img[20 : height + 20, 20 : width + 20, 1] = img
        full_img[20 : height + 20, 20 : width + 20, 2] = img

        image_path = tmp_path / "step_tablet.png"
        Image.fromarray(full_img).save(image_path)

        return image_path

    def test_analyze_image(self, sample_step_tablet_image):
        """Test analysis from image file."""
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze(
            sample_step_tablet_image,
            curve_name="Test Image Curve",
            generate_curve=True,
        )

        assert result.detection_success is True
        assert len(result.densities) > 0
        assert result.quality is not None

    def test_analyze_pil_image(self, sample_step_tablet_image):
        """Test analysis from PIL Image."""
        analyzer = StepWedgeAnalyzer()
        pil_image = Image.open(sample_step_tablet_image)

        result = analyzer.analyze(
            pil_image,
            generate_curve=False,
        )

        assert result.detection_success is True

    def test_analyze_numpy_array(self, sample_step_tablet_image):
        """Test analysis from numpy array."""
        analyzer = StepWedgeAnalyzer()
        img_array = np.array(Image.open(sample_step_tablet_image))

        result = analyzer.analyze(
            img_array,
            generate_curve=False,
        )

        assert result.detection_success is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_densities(self):
        """Test with empty densities."""
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze_from_densities([])

        assert result.densities == []
        assert result.quality.grade == QualityGrade.FAILED

    def test_single_density(self):
        """Test with single density value."""
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze_from_densities([1.0])

        assert len(result.densities) == 1

    def test_two_densities(self):
        """Test with two density values."""
        analyzer = StepWedgeAnalyzer()
        result = analyzer.analyze_from_densities([0.1, 2.0], generate_curve=True)

        assert len(result.densities) == 2

    def test_extreme_density_values(self):
        """Test with extreme density values."""
        analyzer = StepWedgeAnalyzer()
        densities = list(np.linspace(0.0, 5.0, 21))

        result = analyzer.analyze_from_densities(densities, generate_curve=True)

        assert result.dmin == pytest.approx(0.0)
        assert result.dmax == pytest.approx(5.0)
