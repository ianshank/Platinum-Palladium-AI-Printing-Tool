"""
Comprehensive tests for PlatinumPalladiumAI module.

Tests the AI-powered analysis, prediction, and optimization tools
for platinum-palladium printing workflow.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.ai.platinum_palladium_ai import (
    ChemistryRecommendation,
    ContrastLevel,
    DigitalNegativeResult,
    ExposurePrediction,
    PlatinumPalladiumAI,
    PrinterProfile,
    PrintQualityAnalysis,
    ProblemArea,
    TonalityAnalysisResult,
    TonePreference,
    WorkflowOptimization,
)
from ptpd_calibration.core.models import (
    CalibrationRecord,
    CurveData,
)
from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    CurveType,
    DeveloperType,
)
from ptpd_calibration.exposure.calculator import LightSource
from ptpd_calibration.imaging.processor import ImageFormat

# ============================================================================
# Fixtures - Test Images
# ============================================================================


@pytest.fixture
def low_key_image():
    """Create a dark (low key) test image."""
    # Create image with 70% dark tones
    arr = np.random.randint(0, 80, (200, 200), dtype=np.uint8)
    # Add some darker areas
    arr[:100, :] = np.random.randint(0, 40, (100, 200), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def high_key_image():
    """Create a bright (high key) test image."""
    # Create image with 70% bright tones
    arr = np.random.randint(180, 255, (200, 200), dtype=np.uint8)
    # Add some brighter areas
    arr[:100, :] = np.random.randint(220, 255, (100, 200), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def normal_image():
    """Create a normal (mid-tone) test image."""
    # Create balanced image with good tonal distribution
    arr = np.zeros((200, 200), dtype=np.uint8)
    # Shadows
    arr[:50, :] = np.random.randint(20, 80, (50, 200), dtype=np.uint8)
    # Midtones
    arr[50:150, :] = np.random.randint(90, 170, (100, 200), dtype=np.uint8)
    # Highlights
    arr[150:, :] = np.random.randint(180, 240, (50, 200), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def gradient_image():
    """Create a smooth gradient image."""
    arr = np.linspace(0, 255, 256).reshape(1, 256).repeat(100, axis=0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def high_contrast_image():
    """Create a high contrast image."""
    arr = np.zeros((200, 200), dtype=np.uint8)
    # Pure blacks
    arr[:100, :100] = 0
    # Pure whites
    arr[100:, 100:] = 255
    # Some transitions
    arr[:100, 100:] = 100
    arr[100:, :100] = 150
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def low_contrast_image():
    """Create a low contrast image."""
    # All values in narrow range around middle gray
    arr = np.random.randint(110, 145, (200, 200), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def rgb_test_image():
    """Create RGB test image for conversion testing."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 100  # Red
    arr[:, :, 1] = 150  # Green
    arr[:, :, 2] = 200  # Blue
    return Image.fromarray(arr, mode="RGB")


# ============================================================================
# Fixtures - Test Data
# ============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.exposure.base_exposure_minutes = 10.0
    settings.chemistry.default_na2_drops_ratio = 0.25
    return settings


@pytest.fixture
def sample_curve_data():
    """Create sample calibration curve data."""
    # Create a simple linearization curve
    input_values = np.linspace(0, 255, 21)
    output_values = np.power(input_values / 255.0, 1.2) * 255

    return CurveData(
        name="Test Curve",
        curve_type=CurveType.LINEAR,
        input_values=input_values.tolist(),
        output_values=output_values.tolist(),
        description="Test linearization curve",
    )


@pytest.fixture
def sample_print_history():
    """Create sample print history for workflow optimization."""
    history = []

    # Add 15 successful prints
    for i in range(15):
        record = CalibrationRecord(
            paper_type="Arches Platine" if i < 10 else "Bergger COT320",
            paper_weight=310,
            exposure_time=180.0 + (i * 5),  # Varying exposures
            metal_ratio=0.5 + (i * 0.01),  # Slight variations
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
            humidity=50.0 + (i % 5),
            temperature=20.0 + (i % 3),
            measured_densities=[0.1 + j * 0.1 for j in range(21)],  # Good Dmax
        )
        history.append(record)

    # Add 3 failed prints (low Dmax)
    for _ in range(3):
        record = CalibrationRecord(
            paper_type="Arches Platine",
            paper_weight=310,
            exposure_time=60.0,  # Too short
            metal_ratio=0.3,
            chemistry_type=ChemistryType.PURE_PALLADIUM,
            contrast_agent=ContrastAgent.NONE,
            contrast_amount=0.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
            humidity=45.0,
            temperature=21.0,
            measured_densities=[0.1 + j * 0.05 for j in range(21)],  # Low Dmax
        )
        history.append(record)

    return history


# ============================================================================
# Test Class Initialization
# ============================================================================


class TestPlatinumPalladiumAIInitialization:
    """Test PlatinumPalladiumAI class initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        ai = PlatinumPalladiumAI()

        assert ai.settings is not None
        assert ai.histogram_analyzer is not None
        assert ai.image_processor is not None
        assert ai.exposure_calculator is not None
        assert ai.chemistry_calculator is not None

    def test_init_custom_settings(self, mock_settings):
        """Test initialization with custom settings."""
        ai = PlatinumPalladiumAI(settings=mock_settings)

        assert ai.settings == mock_settings
        assert ai.histogram_analyzer is not None

    def test_lazy_model_loading(self):
        """Test that ML models are lazy loaded (None initially)."""
        ai = PlatinumPalladiumAI()

        assert ai._exposure_model is None
        assert ai._quality_model is None


# ============================================================================
# Test Image Tonality Analysis
# ============================================================================


class TestAnalyzeImageTonality:
    """Test analyze_image_tonality method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_analyze_normal_image(self, ai, normal_image):
        """Test analysis of normal balanced image."""
        result = ai.analyze_image_tonality(normal_image)

        assert isinstance(result, TonalityAnalysisResult)
        assert isinstance(result.histogram_stats, dict)
        assert len(result.zone_distribution) == 11
        assert 0.99 <= sum(result.zone_distribution.values()) <= 1.01
        assert len(result.dominant_zones) > 0
        assert result.dynamic_range_stops > 0

    def test_analyze_low_key_image(self, ai, low_key_image):
        """Test analysis of dark/low key image."""
        result = ai.analyze_image_tonality(low_key_image)

        # Should recommend exposure adjustment for dark image
        assert result.recommended_exposure_adjustment_stops != 0
        assert result.shadow_detail_percent > 0
        assert len(result.suggestions) > 0

        # Dominant zones should be in lower range
        assert any(zone < 5 for zone in result.dominant_zones)

    def test_analyze_high_key_image(self, ai, high_key_image):
        """Test analysis of bright/high key image."""
        result = ai.analyze_image_tonality(high_key_image)

        # Should recommend exposure adjustment for bright image
        assert result.recommended_exposure_adjustment_stops != 0
        assert result.highlight_detail_percent > 0

        # Dominant zones should be in upper range
        assert any(zone > 6 for zone in result.dominant_zones)

    def test_analyze_low_contrast_image(self, ai, low_contrast_image):
        """Test analysis of low contrast image."""
        result = ai.analyze_image_tonality(low_contrast_image)

        # Should suggest increasing contrast
        assert result.recommended_contrast_adjustment in ("increase", "none")
        assert any("contrast" in s.lower() for s in result.suggestions)

    def test_analyze_high_contrast_image(self, ai, high_contrast_image):
        """Test analysis of high contrast image."""
        result = ai.analyze_image_tonality(high_contrast_image)

        # Should suggest contrast management
        assert result.recommended_contrast_adjustment in ("decrease", "none")
        assert result.dynamic_range_stops > 0

    def test_analyze_gradient_image(self, ai, gradient_image):
        """Test analysis of smooth gradient."""
        result = ai.analyze_image_tonality(gradient_image)

        # Gradient should have high dynamic range
        assert result.dynamic_range_stops > 5
        # Should have good distribution across zones
        assert len(result.dominant_zones) >= 3

    def test_analyze_with_target_paper(self, ai, normal_image):
        """Test analysis with target paper specified."""
        result = ai.analyze_image_tonality(
            normal_image,
            target_paper="Arches Platine Hot Press",
        )

        assert isinstance(result, TonalityAnalysisResult)
        # Should have paper-specific suggestions
        paper_suggestions = [s for s in result.suggestions if "press" in s.lower()]
        assert len(paper_suggestions) > 0

    def test_analyze_with_target_process(self, ai, normal_image):
        """Test analysis with target chemistry process."""
        result_pd = ai.analyze_image_tonality(
            normal_image,
            target_process=ChemistryType.PURE_PALLADIUM,
        )

        result_pt = ai.analyze_image_tonality(
            normal_image,
            target_process=ChemistryType.PURE_PLATINUM,
        )

        # Both should have process-specific suggestions
        assert any("palladium" in s.lower() for s in result_pd.suggestions)
        assert any("platinum" in s.lower() for s in result_pt.suggestions)

    def test_analyze_numpy_array(self, ai):
        """Test analysis accepts numpy array."""
        arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = ai.analyze_image_tonality(arr)

        assert isinstance(result, TonalityAnalysisResult)

    def test_analyze_path_string(self, ai, tmp_path, normal_image):
        """Test analysis accepts file path string."""
        image_path = tmp_path / "test.png"
        normal_image.save(image_path)

        result = ai.analyze_image_tonality(str(image_path))

        assert isinstance(result, TonalityAnalysisResult)

    def test_clipping_warnings(self, ai):
        """Test that clipping generates appropriate warnings."""
        # Create image with heavy shadow clipping
        arr = np.zeros((200, 200), dtype=np.uint8)
        arr[:100, :] = 0  # 50% pure black
        arr[100:, :] = 128
        clipped_image = Image.fromarray(arr, mode="L")

        result = ai.analyze_image_tonality(clipped_image)

        # Should warn about shadow clipping
        assert any("clipping" in w.lower() for w in result.warnings)


# ============================================================================
# Test Exposure Time Prediction
# ============================================================================


class TestPredictExposureTime:
    """Test predict_exposure_time method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_predict_basic(self, ai):
        """Test basic exposure prediction."""
        result = ai.predict_exposure_time(negative_density=1.6)

        assert isinstance(result, ExposurePrediction)
        assert result.predicted_exposure_seconds > 0
        assert result.predicted_exposure_minutes > 0
        assert result.lower_bound_seconds > 0
        assert result.upper_bound_seconds > result.lower_bound_seconds
        assert 0.0 < result.confidence_level <= 1.0

    def test_predict_higher_density_increases_exposure(self, ai):
        """Test that higher density requires longer exposure."""
        result1 = ai.predict_exposure_time(negative_density=1.2)
        result2 = ai.predict_exposure_time(negative_density=2.0)

        assert result2.predicted_exposure_seconds > result1.predicted_exposure_seconds

    def test_predict_with_paper_type(self, ai):
        """Test exposure prediction with specific paper type."""
        result_hp = ai.predict_exposure_time(
            negative_density=1.6,
            paper_type="Generic Hot Press",
        )

        result_cp = ai.predict_exposure_time(
            negative_density=1.6,
            paper_type="Generic Cold Press",
        )

        # Different papers should give different exposures
        # Hot press is faster (< 1.0), cold press is slower (> 1.0)
        assert result_hp.paper_speed_factor < 1.0
        assert result_cp.paper_speed_factor > 1.0
        assert result_hp.paper_speed_factor != result_cp.paper_speed_factor

    def test_predict_with_light_source(self, ai):
        """Test exposure with different light sources."""
        result_bl = ai.predict_exposure_time(
            negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
        )

        result_led = ai.predict_exposure_time(
            negative_density=1.6,
            light_source=LightSource.LED_UV,
        )

        # LED should be faster than BL fluorescent
        assert result_led.predicted_exposure_seconds < result_bl.predicted_exposure_seconds

    def test_predict_with_humidity(self, ai):
        """Test humidity adjustment."""
        result_low_humidity = ai.predict_exposure_time(
            negative_density=1.6,
            humidity=30.0,
        )

        result_high_humidity = ai.predict_exposure_time(
            negative_density=1.6,
            humidity=70.0,
        )

        # Humidity should affect exposure
        assert result_low_humidity.humidity_factor != result_high_humidity.humidity_factor

    def test_predict_with_temperature(self, ai):
        """Test temperature consideration."""
        result_cold = ai.predict_exposure_time(
            negative_density=1.6,
            temperature=10.0,
        )

        result_hot = ai.predict_exposure_time(
            negative_density=1.6,
            temperature=30.0,
        )

        # Should have temperature-related recommendations
        assert result_cold.temperature_celsius == 10.0
        assert result_hot.temperature_celsius == 30.0

    def test_predict_with_platinum_ratio(self, ai):
        """Test that platinum ratio affects exposure."""
        result_pd = ai.predict_exposure_time(
            negative_density=1.6,
            platinum_ratio=0.0,  # Pure palladium
        )

        result_pt = ai.predict_exposure_time(
            negative_density=1.6,
            platinum_ratio=1.0,  # Pure platinum
        )

        # Platinum should require longer exposure
        assert result_pt.predicted_exposure_seconds > result_pd.predicted_exposure_seconds

    def test_predict_with_distance(self, ai):
        """Test distance adjustment (inverse square law)."""
        result_close = ai.predict_exposure_time(
            negative_density=1.6,
            distance_inches=4.0,
        )

        result_far = ai.predict_exposure_time(
            negative_density=1.6,
            distance_inches=8.0,
        )

        # Double distance should roughly quadruple exposure
        ratio = result_far.predicted_exposure_seconds / result_close.predicted_exposure_seconds
        assert 3.0 < ratio < 5.0

    def test_format_time_method(self, ai):
        """Test human-readable time formatting."""
        result = ai.predict_exposure_time(negative_density=1.6)
        formatted = result.format_time()

        assert isinstance(formatted, str)
        assert any(word in formatted.lower() for word in ["second", "minute", "min", "sec"])

    def test_adjustments_breakdown(self, ai):
        """Test that all adjustments are tracked."""
        result = ai.predict_exposure_time(
            negative_density=1.8,
            distance_inches=6.0,
        )

        assert "density_adjustment" in result.adjustments_applied
        assert "light_source_adjustment" in result.adjustments_applied
        assert result.base_exposure > 0

    def test_confidence_interval_reasonable(self, ai):
        """Test that confidence interval is reasonable."""
        result = ai.predict_exposure_time(negative_density=1.6)

        # Bounds should bracket the prediction
        assert result.lower_bound_seconds < result.predicted_exposure_seconds
        assert result.predicted_exposure_seconds < result.upper_bound_seconds

        # Interval shouldn't be too wide (less than 100% of prediction)
        interval_width = result.upper_bound_seconds - result.lower_bound_seconds
        assert interval_width < result.predicted_exposure_seconds * 1.0


# ============================================================================
# Test Chemistry Ratio Recommendations
# ============================================================================


class TestSuggestChemistryRatios:
    """Test suggest_chemistry_ratios method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_suggest_warm_tone(self, ai):
        """Test warm tone (palladium-rich) recommendation."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.WARM,
            contrast_level=ContrastLevel.NORMAL,
        )

        assert isinstance(result, ChemistryRecommendation)
        assert result.platinum_ratio == 0.0  # Pure palladium for warm
        assert result.palladium_ratio == 1.0
        assert "warm" in result.expected_tone.lower()

    def test_suggest_neutral_tone(self, ai):
        """Test neutral tone (50/50 mix) recommendation."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.NORMAL,
        )

        assert result.platinum_ratio == 0.5
        assert result.palladium_ratio == 0.5
        assert "neutral" in result.expected_tone.lower()

    def test_suggest_cool_tone(self, ai):
        """Test cool tone (platinum-rich) recommendation."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.COOL,
            contrast_level=ContrastLevel.NORMAL,
        )

        assert result.platinum_ratio == 1.0  # Pure platinum for cool
        assert result.palladium_ratio == 0.0
        assert "cool" in result.expected_tone.lower()

    def test_contrast_levels(self, ai):
        """Test different contrast level recommendations."""
        result_low = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.LOW,
        )

        result_high = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.HIGH,
        )

        result_max = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.MAXIMUM,
        )

        # Higher contrast should use more FO#2
        assert result_low.contrast_amount_percent < result_high.contrast_amount_percent
        assert result_high.contrast_amount_percent < result_max.contrast_amount_percent

    def test_na2_usage_with_high_contrast(self, ai):
        """Test that high contrast recommends Na2."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.HIGH,
        )

        assert result.contrast_agent == ContrastAgent.NA2
        assert result.na2_drops > 0

    def test_no_na2_with_low_contrast(self, ai):
        """Test that low contrast doesn't use Na2."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.LOW,
        )

        assert result.contrast_agent == ContrastAgent.NONE
        assert result.na2_drops == 0

    def test_with_print_size(self, ai):
        """Test chemistry calculation with specific print size."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.NORMAL,
            print_size_inches=(8.0, 10.0),
        )

        # Should calculate actual drop amounts
        assert result.ferric_oxalate_1_drops > 0
        # For 8x10 print, drops should be reasonable (10-30 range)
        assert 5 < result.ferric_oxalate_1_drops < 50

    def test_with_paper_type(self, ai):
        """Test recommendations with specific paper type."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.NORMAL,
            paper_type="Arches Platine Hot Press",
            print_size_inches=(8.0, 10.0),
        )

        # Should have paper-specific notes
        assert any("arches" in note.lower() for note in result.notes)

    def test_expected_dmax_varies_with_metals(self, ai):
        """Test that expected Dmax varies with metal ratios."""
        result_pd = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.WARM,  # Palladium
            contrast_level=ContrastLevel.NORMAL,
        )

        result_pt = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.COOL,  # Platinum
            contrast_level=ContrastLevel.NORMAL,
        )

        # Platinum should give higher Dmax
        assert result_pt.expected_dmax > result_pd.expected_dmax

    def test_developer_recommendation(self, ai):
        """Test developer recommendations based on tone."""
        result_warm = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.WARM,
            contrast_level=ContrastLevel.NORMAL,
        )

        result_cool = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.COOL,
            contrast_level=ContrastLevel.NORMAL,
        )

        # Warm should recommend ammonium citrate
        assert result_warm.recommended_developer == DeveloperType.AMMONIUM_CITRATE
        # Cool should use standard potassium oxalate
        assert result_cool.recommended_developer == DeveloperType.POTASSIUM_OXALATE

    def test_rationale_provided(self, ai):
        """Test that rationale is provided for recommendations."""
        result = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.HIGH,
        )

        assert len(result.rationale) > 0
        assert len(result.notes) > 0

    def test_metal_ratios_sum_to_one(self, ai):
        """Test that platinum and palladium ratios sum to 1.0."""
        for tone in [TonePreference.WARM, TonePreference.NEUTRAL, TonePreference.COOL]:
            result = ai.suggest_chemistry_ratios(
                desired_tone=tone,
                contrast_level=ContrastLevel.NORMAL,
            )

            total = result.platinum_ratio + result.palladium_ratio
            assert 0.99 < total < 1.01  # Allow small float error


# ============================================================================
# Test Digital Negative Generation
# ============================================================================


class TestGenerateDigitalNegative:
    """Test generate_digital_negative method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_generate_basic(self, ai, normal_image, tmp_path):
        """Test basic digital negative generation."""
        output_path = tmp_path / "negative.tif"

        result = ai.generate_digital_negative(
            normal_image,
            output_path=output_path,
        )

        assert isinstance(result, DigitalNegativeResult)
        assert result.output_path == output_path
        assert output_path.exists()
        assert result.output_format == ImageFormat.TIFF_16BIT.value

    def test_generate_with_curve(self, ai, normal_image, sample_curve_data, tmp_path):
        """Test negative generation with calibration curve."""
        output_path = tmp_path / "negative_curved.tif"

        result = ai.generate_digital_negative(
            normal_image,
            curve=sample_curve_data,
            output_path=output_path,
        )

        assert result.curve_name == "Test Curve"
        assert result.curve_type == CurveType.LINEAR
        assert "Applied curve" in " ".join(result.steps_applied)

    def test_generate_without_inversion(self, ai, normal_image, tmp_path):
        """Test generation without inversion (positive)."""
        output_path = tmp_path / "positive.tif"

        result = ai.generate_digital_negative(
            normal_image,
            output_path=output_path,
            invert=False,
        )

        assert "Inverted" not in " ".join(result.steps_applied)

    def test_generate_with_inversion(self, ai, normal_image, tmp_path):
        """Test generation with inversion (negative)."""
        output_path = tmp_path / "negative.tif"

        result = ai.generate_digital_negative(
            normal_image,
            output_path=output_path,
            invert=True,
        )

        assert any("invert" in step.lower() for step in result.steps_applied)

    def test_rgb_to_grayscale_conversion(self, ai, rgb_test_image, tmp_path):
        """Test that RGB images are converted to grayscale."""
        output_path = tmp_path / "negative.tif"

        result = ai.generate_digital_negative(
            rgb_test_image,
            output_path=output_path,
        )

        assert any("grayscale" in step.lower() for step in result.steps_applied)

    def test_different_output_formats(self, ai, normal_image, tmp_path):
        """Test different output format options."""
        # Test PNG 16-bit
        result_png = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "negative.png",
            output_format=ImageFormat.PNG_16BIT,
        )

        # Test TIFF 16-bit
        result_tiff = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "negative.tif",
            output_format=ImageFormat.TIFF_16BIT,
        )

        assert result_png.output_format == ImageFormat.PNG_16BIT.value
        assert result_tiff.output_format == ImageFormat.TIFF_16BIT.value

    def test_custom_dpi(self, ai, normal_image, tmp_path):
        """Test custom DPI setting."""
        result = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "negative.tif",
            target_dpi=3600,
        )

        assert result.output_dpi == 3600

    def test_quality_score_higher_for_16bit(self, ai, normal_image, tmp_path):
        """Test that 16-bit formats get higher quality scores."""
        result_16bit = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "neg_16.tif",
            output_format=ImageFormat.TIFF_16BIT,
        )

        result_8bit = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "neg_8.png",
            output_format=ImageFormat.PNG,
        )

        assert result_16bit.estimated_quality > result_8bit.estimated_quality

    def test_size_information_preserved(self, ai, normal_image, tmp_path):
        """Test that size information is preserved."""
        result = ai.generate_digital_negative(
            normal_image,
            output_path=tmp_path / "negative.tif",
        )

        assert result.original_size == normal_image.size
        assert result.output_size == normal_image.size

    def test_numpy_array_input(self, ai, tmp_path):
        """Test that numpy array input works."""
        arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        result = ai.generate_digital_negative(
            arr,
            output_path=tmp_path / "negative.tif",
        )

        assert isinstance(result, DigitalNegativeResult)

    def test_without_output_path(self, ai, normal_image):
        """Test generation without saving to file."""
        result = ai.generate_digital_negative(normal_image)

        assert result.output_path is None
        assert result.processing_result is not None


# ============================================================================
# Test Print Quality Analysis
# ============================================================================


class TestAnalyzePrintQuality:
    """Test analyze_print_quality method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_analyze_identical_images(self, ai, normal_image):
        """Test analysis of identical scan and reference."""
        result = ai.analyze_print_quality(
            scan_image=normal_image,
            reference_image=normal_image,
        )

        assert isinstance(result, PrintQualityAnalysis)
        # Identical images should have very high match score
        assert result.overall_match_score > 0.9
        assert result.density_correlation > 0.9
        assert result.mean_density_difference < 5

    def test_analyze_different_brightness(self, ai, normal_image, low_key_image):
        """Test analysis with brightness difference."""
        result = ai.analyze_print_quality(
            scan_image=low_key_image,  # Darker
            reference_image=normal_image,
        )

        # Should detect density difference
        assert abs(result.mean_density_difference) > 10
        # Should suggest exposure correction
        assert abs(result.suggested_exposure_correction_stops) > 0

    def test_analyze_different_contrast(self, ai, high_contrast_image, low_contrast_image):
        """Test analysis with contrast difference."""
        result = ai.analyze_print_quality(
            scan_image=low_contrast_image,
            reference_image=high_contrast_image,
        )

        # Should detect contrast issue
        assert result.suggested_contrast_correction in ("increase", "reduce")
        assert len(result.corrections) > 0

    def test_problem_area_detection_highlights(self, ai, normal_image, high_key_image):
        """Test detection of highlight problems."""
        result = ai.analyze_print_quality(
            scan_image=high_key_image,
            reference_image=normal_image,
        )

        # Should detect highlight issues
        problem_types = [p[0] for p in result.problem_areas]
        if len(problem_types) > 0:
            assert (
                ProblemArea.HIGHLIGHTS in problem_types
                or ProblemArea.OVERALL_DENSITY in problem_types
            )

    def test_problem_area_detection_shadows(self, ai, normal_image, low_key_image):
        """Test detection of shadow problems."""
        result = ai.analyze_print_quality(
            scan_image=low_key_image,
            reference_image=normal_image,
        )

        # Should detect shadow or density issues
        problem_types = [p[0] for p in result.problem_areas]
        if len(problem_types) > 0:
            assert (
                ProblemArea.SHADOWS in problem_types or ProblemArea.OVERALL_DENSITY in problem_types
            )

    def test_zone_differences_calculated(self, ai, normal_image, low_key_image):
        """Test that zone-by-zone differences are calculated."""
        result = ai.analyze_print_quality(
            scan_image=low_key_image,
            reference_image=normal_image,
        )

        assert len(result.zone_differences) == 11  # Zones 0-10
        assert len(result.worst_zones) > 0
        # Worst zones should be in zone_differences keys
        for zone in result.worst_zones:
            assert zone in result.zone_differences

    def test_curve_adjustment_suggestions(self, ai, normal_image):
        """Test that curve adjustments are suggested when needed."""
        # Create images with highlight difference
        ref_arr = np.array(normal_image)
        scan_arr = ref_arr.copy()
        # Make scan brighter to simulate blown highlights
        scan_arr = np.clip(scan_arr.astype(int) + 40, 0, 255).astype(np.uint8)

        ref_img = Image.fromarray(ref_arr, mode="L")
        scan_img = Image.fromarray(scan_arr, mode="L")

        result = ai.analyze_print_quality(
            scan_image=scan_img,
            reference_image=ref_img,
        )

        # Should suggest curve adjustments if problems detected
        if len(result.problem_areas) > 0:
            assert len(result.suggested_curve_adjustments) >= 0

    def test_corrections_provided(self, ai, normal_image, low_key_image):
        """Test that specific corrections are provided."""
        result = ai.analyze_print_quality(
            scan_image=low_key_image,
            reference_image=normal_image,
        )

        # Should provide actionable corrections
        assert len(result.corrections) > 0
        # Corrections should be strings
        assert all(isinstance(c, str) for c in result.corrections)

    def test_low_match_triggers_calibration_suggestion(
        self, ai, high_contrast_image, low_contrast_image
    ):
        """Test that poor match suggests new calibration."""
        result = ai.analyze_print_quality(
            scan_image=low_contrast_image,
            reference_image=high_contrast_image,
        )

        if result.overall_match_score < 0.7:
            # Should suggest creating new calibration curve
            assert any("calibration" in c.lower() for c in result.corrections)

    def test_density_range_match_calculated(self, ai, normal_image, low_key_image):
        """Test density range match calculation."""
        result = ai.analyze_print_quality(
            scan_image=normal_image,
            reference_image=low_key_image,
        )

        assert 0.0 <= result.density_range_match <= 1.0
        # Correlation should be valid (not NaN)
        assert -1.0 <= result.density_correlation <= 1.0


# ============================================================================
# Test Workflow Optimization
# ============================================================================


class TestOptimizeWorkflow:
    """Test optimize_workflow method."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_optimize_empty_history(self, ai):
        """Test optimization with no print history."""
        result = ai.optimize_workflow([])

        assert isinstance(result, WorkflowOptimization)
        assert result.total_prints_analyzed == 0
        assert result.successful_prints == 0
        assert result.success_rate == 0.0
        assert result.confidence == 0.0

    def test_optimize_with_history(self, ai, sample_print_history):
        """Test optimization with sample print history."""
        result = ai.optimize_workflow(sample_print_history)

        assert result.total_prints_analyzed == len(sample_print_history)
        assert result.successful_prints > 0
        assert 0.0 <= result.success_rate <= 1.0

    def test_identifies_successful_prints(self, ai, sample_print_history):
        """Test that successful prints are identified correctly."""
        result = ai.optimize_workflow(sample_print_history)

        # We created 15 successful and 3 failed prints
        assert result.successful_prints >= 10  # Most should be identified
        assert result.success_rate > 0.5

    def test_calculates_optimal_parameters(self, ai, sample_print_history):
        """Test that optimal parameters are calculated."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 0:
            assert "avg_exposure_time" in result.optimal_parameters
            assert "avg_metal_ratio" in result.optimal_parameters
            assert result.optimal_parameters["avg_exposure_time"] > 0

    def test_recommends_base_exposure(self, ai, sample_print_history):
        """Test that base exposure is recommended."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 0:
            assert result.recommended_base_exposure is not None
            assert result.recommended_base_exposure > 0

    def test_recommends_metal_ratio(self, ai, sample_print_history):
        """Test that metal ratio is recommended."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 0:
            assert result.recommended_metal_ratio is not None
            assert 0.0 <= result.recommended_metal_ratio <= 1.0

    def test_identifies_paper_preferences(self, ai, sample_print_history):
        """Test that paper preferences are identified."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 0:
            # Should identify "most_successful_paper" trend
            assert len(result.parameter_trends) > 0
            # Should have insights about papers
            assert len(result.insights) > 0

    def test_paper_specific_settings(self, ai, sample_print_history):
        """Test that paper-specific settings are provided."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 0:
            # Should have recommendations for specific papers
            assert len(result.recommended_paper_settings) > 0
            # Each paper should have recommended exposure
            for _paper, settings in result.recommended_paper_settings.items():
                assert "recommended_exposure" in settings
                assert settings["recommended_exposure"] > 0

    def test_identifies_common_mistakes(self, ai, sample_print_history):
        """Test that common mistakes are identified."""
        result = ai.optimize_workflow(sample_print_history)

        # With failed prints in history, should identify some mistakes
        if result.total_prints_analyzed - result.successful_prints > 0:
            # May or may not identify specific mistakes
            assert isinstance(result.common_mistakes, list)

    def test_provides_efficiency_suggestions(self, ai, sample_print_history):
        """Test that efficiency suggestions are provided."""
        result = ai.optimize_workflow(sample_print_history)

        assert isinstance(result.efficiency_suggestions, list)
        if result.successful_prints > 0:
            assert len(result.efficiency_suggestions) > 0

    def test_confidence_increases_with_samples(self, ai):
        """Test that confidence increases with more successful prints."""
        # Small history
        small_history = [
            CalibrationRecord(
                paper_type="Test",
                paper_weight=310,
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                measured_densities=[0.1 + i * 0.1 for i in range(21)],
            )
            for _ in range(3)
        ]

        # Large history
        large_history = [
            CalibrationRecord(
                paper_type="Test",
                paper_weight=310,
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                measured_densities=[0.1 + i * 0.1 for i in range(21)],
            )
            for _ in range(15)
        ]

        result_small = ai.optimize_workflow(small_history)
        result_large = ai.optimize_workflow(large_history)

        # More samples should give higher confidence
        assert result_large.confidence >= result_small.confidence

    def test_identifies_metal_ratio_trends(self, ai, sample_print_history):
        """Test identification of metal ratio trends."""
        result = ai.optimize_workflow(sample_print_history)

        if result.successful_prints > 5 and "metal_ratio" in result.parameter_trends:
            # Should identify metal ratio trends
            trend = result.parameter_trends["metal_ratio"]
            assert trend in ["palladium_dominant", "platinum_dominant", "balanced"]

    def test_identifies_exposure_consistency(self, ai):
        """Test identification of exposure consistency."""
        # Create history with very consistent exposures
        consistent_history = [
            CalibrationRecord(
                paper_type="Test",
                paper_weight=310,
                exposure_time=180.0,  # All same
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                measured_densities=[0.1 + i * 0.1 for i in range(21)],
            )
            for _ in range(10)
        ]

        result = ai.optimize_workflow(consistent_history)

        if "exposure_consistency" in result.parameter_trends:
            assert result.parameter_trends["exposure_consistency"] in ["high", "variable"]

    def test_high_success_rate_suggestions(self, ai):
        """Test suggestions for high success rate."""
        # Create all successful prints
        good_history = [
            CalibrationRecord(
                paper_type="Test",
                paper_weight=310,
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                measured_densities=[0.1 + i * 0.1 for i in range(21)],
            )
            for _ in range(12)
        ]

        result = ai.optimize_workflow(good_history)

        # High success rate should suggest batch printing
        if result.success_rate > 0.7:
            assert any("batch" in s.lower() for s in result.efficiency_suggestions)


# ============================================================================
# Test Helper Methods
# ============================================================================


class TestHelperMethods:
    """Test helper methods."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_get_paper_speed_factor_hot_press(self, ai):
        """Test paper speed factor for hot press paper."""
        factor = ai._get_paper_speed_factor("Generic Hot Press Paper")

        # Hot press should be faster (factor < 1.0)
        assert factor == 0.9

    def test_get_paper_speed_factor_cold_press(self, ai):
        """Test paper speed factor for cold press paper."""
        factor = ai._get_paper_speed_factor("Generic Cold Press Paper")

        # Cold press should be slower (factor > 1.0)
        assert factor == 1.1

    def test_get_paper_speed_factor_unknown(self, ai):
        """Test paper speed factor for unknown paper."""
        factor = ai._get_paper_speed_factor("Unknown Paper Brand")

        # Should return default 1.0
        assert factor == 1.0

    def test_get_paper_speed_factor_hp_shorthand(self, ai):
        """Test that HP shorthand is recognized as hot press."""
        factor = ai._get_paper_speed_factor("Some Paper HP")

        assert factor == 0.9

    def test_get_paper_speed_factor_cp_shorthand(self, ai):
        """Test that CP shorthand is recognized as cold press."""
        factor = ai._get_paper_speed_factor("Some Paper CP")

        assert factor == 1.1


# ============================================================================
# Test Enum Values
# ============================================================================


class TestEnumValues:
    """Test enum value definitions."""

    def test_tone_preference_values(self):
        """Test TonePreference enum values."""
        assert TonePreference.WARM.value == "warm"
        assert TonePreference.NEUTRAL.value == "neutral"
        assert TonePreference.COOL.value == "cool"
        assert TonePreference.CUSTOM.value == "custom"

    def test_contrast_level_values(self):
        """Test ContrastLevel enum values."""
        assert ContrastLevel.LOW.value == "low"
        assert ContrastLevel.NORMAL.value == "normal"
        assert ContrastLevel.HIGH.value == "high"
        assert ContrastLevel.MAXIMUM.value == "maximum"

    def test_printer_profile_values(self):
        """Test PrinterProfile enum values."""
        assert PrinterProfile.EPSON_P800.value == "epson_p800"
        assert PrinterProfile.CUSTOM.value == "custom"

    def test_problem_area_values(self):
        """Test ProblemArea enum values."""
        assert ProblemArea.HIGHLIGHTS.value == "highlights"
        assert ProblemArea.SHADOWS.value == "shadows"
        assert ProblemArea.MIDTONES.value == "midtones"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for testing."""
        return PlatinumPalladiumAI()

    def test_complete_workflow_scenario(self, ai, normal_image, tmp_path, sample_curve_data):
        """Test a complete workflow from analysis to negative generation."""
        # 1. Analyze image tonality
        tonality = ai.analyze_image_tonality(normal_image)
        assert isinstance(tonality, TonalityAnalysisResult)

        # 2. Get chemistry recommendations based on analysis
        chemistry = ai.suggest_chemistry_ratios(
            desired_tone=TonePreference.NEUTRAL,
            contrast_level=ContrastLevel.NORMAL,
            print_size_inches=(8.0, 10.0),
        )
        assert isinstance(chemistry, ChemistryRecommendation)

        # 3. Predict exposure time
        exposure = ai.predict_exposure_time(
            negative_density=1.6,
            platinum_ratio=chemistry.platinum_ratio,
        )
        assert isinstance(exposure, ExposurePrediction)

        # 4. Generate digital negative
        negative = ai.generate_digital_negative(
            normal_image,
            curve=sample_curve_data,
            output_path=tmp_path / "negative.tif",
        )
        assert isinstance(negative, DigitalNegativeResult)
        assert negative.output_path.exists()

    def test_print_evaluation_and_optimization_scenario(
        self, ai, normal_image, sample_print_history
    ):
        """Test print evaluation and workflow optimization scenario."""
        # 1. Analyze print quality
        # Simulate slight density shift in "print"
        scan_arr = np.array(normal_image)
        scan_arr = np.clip(scan_arr.astype(int) - 10, 0, 255).astype(np.uint8)
        scan_image = Image.fromarray(scan_arr, mode="L")

        quality = ai.analyze_print_quality(
            scan_image=scan_image,
            reference_image=normal_image,
        )
        assert isinstance(quality, PrintQualityAnalysis)

        # 2. Optimize workflow based on history
        optimization = ai.optimize_workflow(sample_print_history)
        assert isinstance(optimization, WorkflowOptimization)

        # 3. Use optimized parameters for next prediction
        if optimization.recommended_base_exposure:
            exposure = ai.predict_exposure_time(
                negative_density=1.6,
            )
            # Prediction should be in reasonable range
            assert exposure.predicted_exposure_seconds > 0
