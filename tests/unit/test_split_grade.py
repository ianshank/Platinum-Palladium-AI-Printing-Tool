"""
Comprehensive tests for split-grade printing module.

Tests cover:
- SplitGradeSettings validation and defaults
- TonalCurveAdjuster curve generation and metal characteristics
- SplitGradeSimulator image analysis, mask generation, and simulation
- All blend modes and exposure calculations
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.imaging.split_grade import (
    BlendMode,
    MetalType,
    SplitGradeSettings,
    TonalCurveAdjuster,
    SplitGradeSimulator,
    TonalAnalysis,
    ExposureCalculation,
)


# ==============================================================================
# Test Fixtures - Synthetic Test Images
# ==============================================================================


@pytest.fixture
def linear_gradient():
    """Create a simple linear gradient from black to white."""
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    gradient = np.repeat(gradient, 100, axis=0)  # Make it 100 pixels tall
    return gradient


@pytest.fixture
def low_key_image():
    """Create a low-key (predominantly dark) test image."""
    # Create image with most values in shadow range
    img = np.random.gamma(0.5, 0.3, (100, 100))
    return np.clip(img, 0, 1)


@pytest.fixture
def high_key_image():
    """Create a high-key (predominantly light) test image."""
    # Create image with most values in highlight range
    img = 0.7 + np.random.gamma(2.0, 0.15, (100, 100))
    return np.clip(img, 0, 1)


@pytest.fixture
def wide_range_image():
    """Create image with wide tonal range (shadows and highlights)."""
    img = np.zeros((100, 100))
    # Left half: shadows
    img[:, :50] = np.random.uniform(0, 0.3, (100, 50))
    # Right half: highlights
    img[:, 50:] = np.random.uniform(0.7, 1.0, (100, 50))
    return img


@pytest.fixture
def normal_contrast_image():
    """Create a normal contrast image with even distribution."""
    # Normal distribution centered at 0.5
    img = np.random.normal(0.5, 0.2, (100, 100))
    return np.clip(img, 0, 1)


@pytest.fixture
def step_wedge_image():
    """Create a step wedge test image with discrete tonal steps."""
    img = np.zeros((100, 210))
    step_width = 10
    num_steps = 21

    for i in range(num_steps):
        value = i / (num_steps - 1)
        x_start = i * step_width
        x_end = (i + 1) * step_width
        img[:, x_start:x_end] = value

    return img


@pytest.fixture
def color_test_image():
    """Create a color test image (RGB)."""
    img = np.zeros((100, 100, 3))
    # Red channel: gradient
    img[:, :, 0] = np.linspace(0, 1, 100).reshape(1, 100)
    # Green channel: inverse gradient
    img[:, :, 1] = np.linspace(1, 0, 100).reshape(1, 100)
    # Blue channel: uniform
    img[:, :, 2] = 0.5
    return img


# ==============================================================================
# Test SplitGradeSettings
# ==============================================================================


class TestSplitGradeSettings:
    """Tests for SplitGradeSettings validation and defaults."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = SplitGradeSettings()

        assert settings.shadow_grade == 2.5
        assert settings.highlight_grade == 1.5
        assert settings.shadow_exposure_ratio == 0.6
        assert settings.blend_mode == BlendMode.GAMMA
        assert settings.blend_gamma == 2.2
        assert settings.blend_softness == 0.5
        assert settings.shadow_threshold == 0.4
        assert settings.highlight_threshold == 0.7
        assert settings.mask_blur_radius == 10.0
        assert settings.mask_feather_amount == 0.2
        assert settings.platinum_ratio == 0.0
        assert settings.preserve_highlights is True
        assert settings.preserve_shadows is True
        assert settings.highlight_hold_point == 0.95
        assert settings.shadow_hold_point == 0.05

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = SplitGradeSettings(
            shadow_grade=3.0,
            highlight_grade=2.0,
            shadow_exposure_ratio=0.55,
            blend_mode=BlendMode.LINEAR,
        )

        assert settings.shadow_grade == 3.0
        assert settings.highlight_grade == 2.0
        assert settings.shadow_exposure_ratio == 0.55
        assert settings.blend_mode == BlendMode.LINEAR

    def test_grade_validation_min(self):
        """Test grade validation at minimum boundary."""
        settings = SplitGradeSettings(shadow_grade=0.0, highlight_grade=0.0)
        assert settings.shadow_grade == 0.0
        assert settings.highlight_grade == 0.0

    def test_grade_validation_max(self):
        """Test grade validation at maximum boundary."""
        settings = SplitGradeSettings(shadow_grade=5.0, highlight_grade=5.0)
        assert settings.shadow_grade == 5.0
        assert settings.highlight_grade == 5.0

    def test_grade_validation_out_of_range(self):
        """Test that grades outside valid range raise errors."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            SplitGradeSettings(shadow_grade=-0.1)

        with pytest.raises(Exception):
            SplitGradeSettings(shadow_grade=5.1)

    def test_exposure_ratio_validation(self):
        """Test exposure ratio validation."""
        settings = SplitGradeSettings(shadow_exposure_ratio=0.0)
        assert settings.shadow_exposure_ratio == 0.0

        settings = SplitGradeSettings(shadow_exposure_ratio=1.0)
        assert settings.shadow_exposure_ratio == 1.0

        with pytest.raises(Exception):
            SplitGradeSettings(shadow_exposure_ratio=-0.1)

        with pytest.raises(Exception):
            SplitGradeSettings(shadow_exposure_ratio=1.1)

    def test_threshold_order_validation(self):
        """Test that shadow_threshold must be less than highlight_threshold."""
        # Valid thresholds
        settings = SplitGradeSettings(
            shadow_threshold=0.3,
            highlight_threshold=0.7
        )
        assert settings.shadow_threshold == 0.3
        assert settings.highlight_threshold == 0.7

        # Invalid: shadow >= highlight
        with pytest.raises(ValueError, match="must be less than"):
            SplitGradeSettings(
                shadow_threshold=0.7,
                highlight_threshold=0.3
            )

        with pytest.raises(ValueError, match="must be less than"):
            SplitGradeSettings(
                shadow_threshold=0.5,
                highlight_threshold=0.5
            )

    def test_blend_gamma_validation(self):
        """Test blend gamma validation."""
        settings = SplitGradeSettings(blend_gamma=0.5)
        assert settings.blend_gamma == 0.5

        settings = SplitGradeSettings(blend_gamma=4.0)
        assert settings.blend_gamma == 4.0

        with pytest.raises(Exception):
            SplitGradeSettings(blend_gamma=0.4)

        with pytest.raises(Exception):
            SplitGradeSettings(blend_gamma=4.1)

    def test_platinum_ratio_validation(self):
        """Test platinum ratio validation."""
        settings = SplitGradeSettings(platinum_ratio=0.0)
        assert settings.platinum_ratio == 0.0

        settings = SplitGradeSettings(platinum_ratio=1.0)
        assert settings.platinum_ratio == 1.0

        with pytest.raises(Exception):
            SplitGradeSettings(platinum_ratio=-0.1)

        with pytest.raises(Exception):
            SplitGradeSettings(platinum_ratio=1.1)

    def test_all_blend_modes(self):
        """Test all blend mode options."""
        for mode in list(BlendMode):
            settings = SplitGradeSettings(blend_mode=mode)
            assert settings.blend_mode == mode


# ==============================================================================
# Test TonalCurveAdjuster
# ==============================================================================


class TestTonalCurveAdjuster:
    """Tests for TonalCurveAdjuster curve generation and metal characteristics."""

    @pytest.fixture
    def adjuster(self):
        """Create a TonalCurveAdjuster instance."""
        return TonalCurveAdjuster()

    def test_initialization(self):
        """Test TonalCurveAdjuster initialization."""
        adjuster = TonalCurveAdjuster()
        assert adjuster.settings is not None
        assert isinstance(adjuster.settings, SplitGradeSettings)

    def test_initialization_with_custom_settings(self):
        """Test initialization with custom settings."""
        settings = SplitGradeSettings(shadow_grade=3.0)
        adjuster = TonalCurveAdjuster(settings)
        assert adjuster.settings.shadow_grade == 3.0

    @pytest.mark.parametrize("grade", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    def test_create_contrast_curve_all_grades(self, adjuster, grade):
        """Test curve generation for all grade levels."""
        x, y = adjuster.create_contrast_curve(grade)

        assert len(x) == 256
        assert len(y) == 256
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0)
        assert y[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(1.0)

    def test_create_contrast_curve_monotonicity(self, adjuster):
        """Test that curves are monotonically increasing."""
        for grade in [0.0, 1.5, 3.0, 5.0]:
            x, y = adjuster.create_contrast_curve(grade)
            diffs = np.diff(y)
            assert np.all(diffs >= -1e-6), f"Curve not monotonic for grade {grade}"

    def test_create_contrast_curve_grade_zero_soft(self, adjuster):
        """Test that grade 0 produces softer (lower) contrast."""
        x_soft, y_soft = adjuster.create_contrast_curve(0.0)
        x_normal, y_normal = adjuster.create_contrast_curve(3.0)

        # At midpoint, soft curve should be higher (lifted)
        mid_idx = 128
        assert y_soft[mid_idx] > y_normal[mid_idx]

    def test_create_contrast_curve_grade_five_hard(self, adjuster):
        """Test that grade 5 produces harder (higher) contrast."""
        x_hard, y_hard = adjuster.create_contrast_curve(5.0)
        x_normal, y_normal = adjuster.create_contrast_curve(3.0)

        # At midpoint, hard curve should be lower (darkened)
        mid_idx = 128
        assert y_hard[mid_idx] < y_normal[mid_idx]

    def test_create_contrast_curve_custom_num_points(self, adjuster):
        """Test curve generation with custom number of points."""
        x, y = adjuster.create_contrast_curve(2.5, num_points=128)
        assert len(x) == 128
        assert len(y) == 128

    def test_apply_platinum_characteristic(self, adjuster, linear_gradient):
        """Test platinum characteristic application."""
        processed = adjuster.apply_platinum_characteristic(linear_gradient, strength=1.0)

        assert processed.shape == linear_gradient.shape
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)
        # Platinum should extend shadows (darker blacks)
        assert processed[0, 50] < linear_gradient[0, 50]

    def test_apply_platinum_characteristic_zero_strength(self, adjuster, linear_gradient):
        """Test platinum characteristic with zero strength returns unchanged."""
        processed = adjuster.apply_platinum_characteristic(linear_gradient, strength=0.0)
        np.testing.assert_array_almost_equal(processed, linear_gradient)

    def test_apply_platinum_characteristic_partial_strength(self, adjuster, linear_gradient):
        """Test platinum characteristic with partial strength."""
        full = adjuster.apply_platinum_characteristic(linear_gradient, strength=1.0)
        half = adjuster.apply_platinum_characteristic(linear_gradient, strength=0.5)

        # Half strength should be between original and full
        mid_val = linear_gradient[0, 128]
        full_val = full[0, 128]
        half_val = half[0, 128]

        # Check that half is between original and full
        if full_val < mid_val:
            assert half_val < mid_val
            assert half_val > full_val
        else:
            assert half_val > mid_val
            assert half_val < full_val

    def test_apply_palladium_characteristic(self, adjuster, linear_gradient):
        """Test palladium characteristic application."""
        processed = adjuster.apply_palladium_characteristic(linear_gradient, strength=1.0)

        assert processed.shape == linear_gradient.shape
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)
        # Palladium should have softer shadows (not as dark)
        assert processed[0, 10] > linear_gradient[0, 10]

    def test_apply_palladium_characteristic_zero_strength(self, adjuster, linear_gradient):
        """Test palladium characteristic with zero strength."""
        processed = adjuster.apply_palladium_characteristic(linear_gradient, strength=0.0)
        np.testing.assert_array_almost_equal(processed, linear_gradient)

    def test_palladium_vs_platinum_characteristics(self, adjuster, linear_gradient):
        """Test that palladium and platinum produce different results."""
        pt = adjuster.apply_platinum_characteristic(linear_gradient, strength=1.0)
        pd = adjuster.apply_palladium_characteristic(linear_gradient, strength=1.0)

        # Should be different
        assert not np.allclose(pt, pd)

        # Palladium should be lighter in shadows
        assert np.mean(pd[:, :50]) > np.mean(pt[:, :50])

    def test_blend_metal_characteristics_pure_platinum(self, adjuster, linear_gradient):
        """Test blending with 100% platinum."""
        blended = adjuster.blend_metal_characteristics(linear_gradient, pt_ratio=1.0, strength=1.0)
        pt = adjuster.apply_platinum_characteristic(linear_gradient, strength=1.0)

        np.testing.assert_array_almost_equal(blended, pt, decimal=5)

    def test_blend_metal_characteristics_pure_palladium(self, adjuster, linear_gradient):
        """Test blending with 100% palladium."""
        blended = adjuster.blend_metal_characteristics(linear_gradient, pt_ratio=0.0, strength=1.0)
        pd = adjuster.apply_palladium_characteristic(linear_gradient, strength=1.0)

        np.testing.assert_array_almost_equal(blended, pd, decimal=5)

    def test_blend_metal_characteristics_fifty_fifty(self, adjuster, linear_gradient):
        """Test blending with 50/50 platinum/palladium mix."""
        blended = adjuster.blend_metal_characteristics(linear_gradient, pt_ratio=0.5, strength=1.0)
        pt = adjuster.apply_platinum_characteristic(linear_gradient, strength=1.0)
        pd = adjuster.apply_palladium_characteristic(linear_gradient, strength=1.0)

        # Should be between pure platinum and pure palladium
        assert np.all(blended >= np.minimum(pt, pd) - 0.01)
        assert np.all(blended <= np.maximum(pt, pd) + 0.01)

    def test_blend_metal_characteristics_zero_strength(self, adjuster, linear_gradient):
        """Test metal blending with zero strength."""
        blended = adjuster.blend_metal_characteristics(linear_gradient, pt_ratio=0.5, strength=0.0)
        np.testing.assert_array_almost_equal(blended, linear_gradient)

    def test_apply_curve_to_image(self, adjuster, linear_gradient):
        """Test applying curve to image."""
        processed = adjuster.apply_curve_to_image(
            linear_gradient,
            grade=2.5,
            apply_metal_characteristic=False
        )

        assert processed.shape == linear_gradient.shape
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_apply_curve_to_image_with_metal(self, adjuster, linear_gradient):
        """Test applying curve with metal characteristics."""
        settings = SplitGradeSettings(platinum_ratio=0.5)
        adjuster = TonalCurveAdjuster(settings)

        processed = adjuster.apply_curve_to_image(
            linear_gradient,
            grade=2.5,
            apply_metal_characteristic=True
        )

        assert processed.shape == linear_gradient.shape
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_ensure_monotonic(self, adjuster):
        """Test monotonic array enforcement."""
        # Non-monotonic array
        arr = np.array([0.0, 0.2, 0.5, 0.4, 0.6, 0.8, 1.0])
        result = adjuster._ensure_monotonic(arr)

        # Should be monotonic now
        diffs = np.diff(result)
        assert np.all(diffs >= 0.0)

        # First and last values should be preserved
        assert result[0] == arr[0]
        assert result[-1] == arr[-1]


# ==============================================================================
# Test SplitGradeSimulator - Image Analysis
# ==============================================================================


class TestSplitGradeSimulatorAnalysis:
    """Tests for SplitGradeSimulator image analysis."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        return SplitGradeSimulator()

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = SplitGradeSimulator()
        assert simulator.settings is not None
        assert simulator.curve_adjuster is not None

    def test_initialization_with_settings(self):
        """Test initialization with custom settings."""
        settings = SplitGradeSettings(shadow_grade=3.0)
        simulator = SplitGradeSimulator(settings)
        assert simulator.settings.shadow_grade == 3.0

    def test_analyze_image_normal_contrast(self, simulator, normal_contrast_image):
        """Test image analysis on normal contrast image."""
        analysis = simulator.analyze_image(normal_contrast_image)

        assert isinstance(analysis, TonalAnalysis)
        assert 0.0 <= analysis.mean_luminance <= 1.0
        assert 0.0 <= analysis.median_luminance <= 1.0
        assert analysis.std_luminance >= 0.0
        assert 0.0 <= analysis.p05 <= 1.0
        assert 0.0 <= analysis.p95 <= 1.0
        assert analysis.p05 <= analysis.p95
        assert 0.0 <= analysis.shadow_percentage <= 1.0
        assert 0.0 <= analysis.midtone_percentage <= 1.0
        assert 0.0 <= analysis.highlight_percentage <= 1.0
        # Percentages should sum to approximately 1.0
        assert pytest.approx(
            analysis.shadow_percentage +
            analysis.midtone_percentage +
            analysis.highlight_percentage,
            abs=0.01
        ) == 1.0

    def test_analyze_image_low_key(self, simulator, low_key_image):
        """Test analysis identifies low-key images."""
        analysis = simulator.analyze_image(low_key_image)

        assert analysis.is_low_key
        assert not analysis.is_high_key
        assert analysis.mean_luminance < 0.35

    def test_analyze_image_high_key(self, simulator, high_key_image):
        """Test analysis identifies high-key images."""
        analysis = simulator.analyze_image(high_key_image)

        assert analysis.is_high_key
        assert not analysis.is_low_key
        assert analysis.mean_luminance > 0.65

    def test_analyze_image_wide_range(self, simulator, wide_range_image):
        """Test analysis on wide tonal range image."""
        analysis = simulator.analyze_image(wide_range_image)

        assert analysis.tonal_range > 0.5
        assert analysis.shadow_percentage > 0.15
        assert analysis.highlight_percentage > 0.15
        # Should recommend split-grade
        assert analysis.needs_split_grade

    def test_analyze_image_recommendations(self, simulator, normal_contrast_image):
        """Test that analysis provides recommendations."""
        analysis = simulator.analyze_image(normal_contrast_image)

        assert 0.0 <= analysis.recommended_shadow_grade <= 5.0
        assert 0.0 <= analysis.recommended_highlight_grade <= 5.0
        assert 0.0 <= analysis.recommended_shadow_threshold <= 1.0
        assert 0.0 <= analysis.recommended_highlight_threshold <= 1.0
        assert analysis.recommended_shadow_threshold < analysis.recommended_highlight_threshold
        assert 0.0 <= analysis.recommended_exposure_ratio <= 1.0

    def test_analyze_color_image(self, simulator, color_test_image):
        """Test analysis on color image (should convert to luminance)."""
        analysis = simulator.analyze_image(color_test_image)

        assert isinstance(analysis, TonalAnalysis)
        assert 0.0 <= analysis.mean_luminance <= 1.0

    def test_analyze_pil_image(self, simulator, normal_contrast_image):
        """Test analysis accepts PIL Image."""
        pil_image = Image.fromarray((normal_contrast_image * 255).astype(np.uint8))
        analysis = simulator.analyze_image(pil_image)

        assert isinstance(analysis, TonalAnalysis)

    def test_analyze_image_notes(self, simulator, low_key_image):
        """Test that analysis includes helpful notes."""
        analysis = simulator.analyze_image(low_key_image)

        assert isinstance(analysis.notes, list)
        if analysis.is_low_key:
            assert any("low-key" in note.lower() for note in analysis.notes)


# ==============================================================================
# Test SplitGradeSimulator - Mask Generation
# ==============================================================================


class TestSplitGradeSimulatorMasks:
    """Tests for shadow and highlight mask generation."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        settings = SplitGradeSettings(
            mask_blur_radius=5.0,
            mask_feather_amount=0.1
        )
        return SplitGradeSimulator(settings)

    def test_create_shadow_mask_shape(self, simulator, normal_contrast_image):
        """Test shadow mask has correct shape."""
        mask = simulator.create_shadow_mask(normal_contrast_image)

        assert mask.shape == normal_contrast_image.shape

    def test_create_shadow_mask_range(self, simulator, normal_contrast_image):
        """Test shadow mask values are in valid range."""
        mask = simulator.create_shadow_mask(normal_contrast_image)

        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_create_shadow_mask_dark_pixels(self, simulator, linear_gradient):
        """Test that shadow mask selects dark pixels."""
        mask = simulator.create_shadow_mask(linear_gradient, threshold=0.5)

        # Dark areas (left side) should have high mask values
        assert np.mean(mask[:, :50]) > np.mean(mask[:, -50:])

    def test_create_highlight_mask_shape(self, simulator, normal_contrast_image):
        """Test highlight mask has correct shape."""
        mask = simulator.create_highlight_mask(normal_contrast_image)

        assert mask.shape == normal_contrast_image.shape

    def test_create_highlight_mask_range(self, simulator, normal_contrast_image):
        """Test highlight mask values are in valid range."""
        mask = simulator.create_highlight_mask(normal_contrast_image)

        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_create_highlight_mask_bright_pixels(self, simulator, linear_gradient):
        """Test that highlight mask selects bright pixels."""
        mask = simulator.create_highlight_mask(linear_gradient, threshold=0.5)

        # Bright areas (right side) should have high mask values
        assert np.mean(mask[:, -50:]) > np.mean(mask[:, :50])

    def test_create_shadow_mask_custom_threshold(self, simulator, linear_gradient):
        """Test shadow mask with custom threshold."""
        mask_low = simulator.create_shadow_mask(linear_gradient, threshold=0.3)
        mask_high = simulator.create_shadow_mask(linear_gradient, threshold=0.7)

        # Higher threshold should select more pixels
        assert np.sum(mask_high > 0.5) > np.sum(mask_low > 0.5)

    def test_create_highlight_mask_custom_threshold(self, simulator, linear_gradient):
        """Test highlight mask with custom threshold."""
        mask_low = simulator.create_highlight_mask(linear_gradient, threshold=0.3)
        mask_high = simulator.create_highlight_mask(linear_gradient, threshold=0.7)

        # Lower threshold should select more pixels
        assert np.sum(mask_low > 0.5) > np.sum(mask_high > 0.5)

    def test_mask_blur_effect(self, simulator, step_wedge_image):
        """Test that blur smooths mask transitions."""
        # Create simulator with no blur
        no_blur_sim = SplitGradeSimulator(SplitGradeSettings(mask_blur_radius=0.0))
        mask_no_blur = no_blur_sim.create_shadow_mask(step_wedge_image)

        # Create simulator with blur
        blur_sim = SplitGradeSimulator(SplitGradeSettings(mask_blur_radius=10.0))
        mask_blur = blur_sim.create_shadow_mask(step_wedge_image)

        # Blurred mask should be smoother (less variation)
        assert np.std(mask_blur) < np.std(mask_no_blur)

    def test_mask_feather_effect(self, simulator, linear_gradient):
        """Test that feathering affects mask values."""
        no_feather_sim = SplitGradeSimulator(
            SplitGradeSettings(mask_feather_amount=0.0, mask_blur_radius=0.0)
        )
        mask_no_feather = no_feather_sim.create_shadow_mask(linear_gradient)

        feather_sim = SplitGradeSimulator(
            SplitGradeSettings(mask_feather_amount=0.5, mask_blur_radius=0.0)
        )
        mask_feather = feather_sim.create_shadow_mask(linear_gradient)

        # Feathered mask should be different
        assert not np.allclose(mask_no_feather, mask_feather)


# ==============================================================================
# Test SplitGradeSimulator - Blend Modes
# ==============================================================================


class TestSplitGradeSimulatorBlendModes:
    """Tests for all blend modes in exposure blending."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        return SplitGradeSimulator()

    @pytest.fixture
    def shadow_image(self, linear_gradient):
        """Create shadow-processed image."""
        return linear_gradient * 0.8  # Darker

    @pytest.fixture
    def highlight_image(self, linear_gradient):
        """Create highlight-processed image."""
        return linear_gradient * 1.2  # Lighter (will be clipped)

    @pytest.mark.parametrize("blend_mode", [
        BlendMode.LINEAR,
        BlendMode.GAMMA,
        BlendMode.SOFT_LIGHT,
        BlendMode.OVERLAY,
        BlendMode.CUSTOM,
    ])
    def test_blend_exposures_all_modes(self, simulator, shadow_image, highlight_image, blend_mode):
        """Test all blend modes produce valid output."""
        settings = SplitGradeSettings(blend_mode=blend_mode)

        result = simulator.blend_exposures(
            shadow_image,
            highlight_image,
            settings,
        )

        assert result.shape == shadow_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_blend_exposures_linear(self, simulator, shadow_image, highlight_image):
        """Test linear blend mode."""
        settings = SplitGradeSettings(
            blend_mode=BlendMode.LINEAR,
            shadow_exposure_ratio=0.5
        )

        result = simulator.blend_exposures(
            shadow_image,
            highlight_image,
            settings,
        )

        # With 50/50 ratio and no masks, should be average
        expected = np.clip((shadow_image + highlight_image) / 2, 0, 1)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_blend_exposures_gamma(self, simulator, shadow_image, highlight_image):
        """Test gamma blend mode."""
        settings = SplitGradeSettings(
            blend_mode=BlendMode.GAMMA,
            blend_gamma=2.2,
            shadow_exposure_ratio=0.5
        )

        result = simulator.blend_exposures(
            shadow_image,
            highlight_image,
            settings,
        )

        assert result.shape == shadow_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_blend_exposures_with_masks(self, simulator, shadow_image, highlight_image):
        """Test blending with provided masks."""
        shadow_mask = np.ones_like(shadow_image) * 0.7
        highlight_mask = np.ones_like(highlight_image) * 0.3

        settings = SplitGradeSettings(blend_mode=BlendMode.LINEAR)

        result = simulator.blend_exposures(
            shadow_image,
            highlight_image,
            settings,
            shadow_mask=shadow_mask,
            highlight_mask=highlight_mask,
        )

        assert result.shape == shadow_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_blend_exposures_exposure_ratio_effect(self, simulator, shadow_image, highlight_image):
        """Test that exposure ratio affects the blend."""
        settings_shadow = SplitGradeSettings(
            blend_mode=BlendMode.LINEAR,
            shadow_exposure_ratio=0.8
        )
        result_shadow = simulator.blend_exposures(
            shadow_image, highlight_image, settings_shadow
        )

        settings_highlight = SplitGradeSettings(
            blend_mode=BlendMode.LINEAR,
            shadow_exposure_ratio=0.2
        )
        result_highlight = simulator.blend_exposures(
            shadow_image, highlight_image, settings_highlight
        )

        # Different ratios should produce different results
        assert not np.allclose(result_shadow, result_highlight)


# ==============================================================================
# Test SplitGradeSimulator - Full Simulation
# ==============================================================================


class TestSplitGradeSimulatorSimulation:
    """Tests for complete split-grade simulation."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        return SplitGradeSimulator()

    def test_simulate_split_grade_basic(self, simulator, normal_contrast_image):
        """Test basic split-grade simulation."""
        result = simulator.simulate_split_grade(normal_contrast_image)

        assert result.shape == normal_contrast_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_simulate_split_grade_with_settings(self, simulator, normal_contrast_image):
        """Test simulation with custom settings."""
        settings = SplitGradeSettings(
            shadow_grade=3.0,
            highlight_grade=1.5,
            shadow_threshold=0.35,
            highlight_threshold=0.75,
        )

        result = simulator.simulate_split_grade(normal_contrast_image, settings)

        assert result.shape == normal_contrast_image.shape

    def test_simulate_split_grade_preserve_highlights(self, simulator, linear_gradient):
        """Test highlight preservation."""
        settings = SplitGradeSettings(
            preserve_highlights=True,
            highlight_hold_point=0.95
        )

        result = simulator.simulate_split_grade(linear_gradient, settings)

        # Very bright pixels should be approximately preserved
        bright_pixels = linear_gradient > 0.95
        if np.any(bright_pixels):
            original_bright = linear_gradient[bright_pixels]
            result_bright = result[bright_pixels]
            np.testing.assert_array_almost_equal(result_bright, original_bright, decimal=2)

    def test_simulate_split_grade_preserve_shadows(self, simulator, linear_gradient):
        """Test shadow preservation."""
        settings = SplitGradeSettings(
            preserve_shadows=True,
            shadow_hold_point=0.05
        )

        result = simulator.simulate_split_grade(linear_gradient, settings)

        # Very dark pixels should be approximately preserved
        dark_pixels = linear_gradient < 0.05
        if np.any(dark_pixels):
            original_dark = linear_gradient[dark_pixels]
            result_dark = result[dark_pixels]
            np.testing.assert_array_almost_equal(result_dark, original_dark, decimal=2)

    def test_simulate_split_grade_different_grades(self, simulator, normal_contrast_image):
        """Test that different grades produce different results."""
        settings_soft = SplitGradeSettings(shadow_grade=1.0, highlight_grade=0.5)
        result_soft = simulator.simulate_split_grade(normal_contrast_image, settings_soft)

        settings_hard = SplitGradeSettings(shadow_grade=4.0, highlight_grade=3.5)
        result_hard = simulator.simulate_split_grade(normal_contrast_image, settings_hard)

        # Different grades should produce different results
        assert not np.allclose(result_soft, result_hard)

    def test_simulate_split_grade_pil_image(self, simulator, normal_contrast_image):
        """Test simulation accepts PIL Image."""
        pil_image = Image.fromarray((normal_contrast_image * 255).astype(np.uint8))
        result = simulator.simulate_split_grade(pil_image)

        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_preview_result(self, simulator, normal_contrast_image):
        """Test preview result generation."""
        result_dict = simulator.preview_result(normal_contrast_image)

        assert 'original' in result_dict
        assert 'processed' in result_dict
        assert result_dict['original'].shape == normal_contrast_image.shape
        assert result_dict['processed'].shape == normal_contrast_image.shape

    def test_preview_result_with_masks(self, simulator, normal_contrast_image):
        """Test preview with masks included."""
        result_dict = simulator.preview_result(
            normal_contrast_image,
            include_masks=True
        )

        assert 'original' in result_dict
        assert 'processed' in result_dict
        assert 'shadow_mask' in result_dict
        assert 'highlight_mask' in result_dict
        assert result_dict['shadow_mask'].shape == normal_contrast_image.shape
        assert result_dict['highlight_mask'].shape == normal_contrast_image.shape


# ==============================================================================
# Test SplitGradeSimulator - Exposure Calculations
# ==============================================================================


class TestSplitGradeSimulatorExposureCalculations:
    """Tests for exposure time calculations."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        return SplitGradeSimulator()

    def test_calculate_exposure_times_basic(self, simulator):
        """Test basic exposure time calculation."""
        calc = simulator.calculate_exposure_times(base_time=120.0)

        assert isinstance(calc, ExposureCalculation)
        assert calc.total_exposure_seconds == 120.0
        assert calc.shadow_exposure_seconds > 0
        assert calc.highlight_exposure_seconds > 0
        assert pytest.approx(
            calc.shadow_exposure_seconds + calc.highlight_exposure_seconds
        ) == 120.0

    def test_calculate_exposure_times_custom_ratio(self, simulator):
        """Test exposure calculation with custom ratio."""
        settings = SplitGradeSettings(shadow_exposure_ratio=0.7)
        calc = simulator.calculate_exposure_times(base_time=100.0, settings=settings)

        assert calc.shadow_exposure_seconds == pytest.approx(70.0)
        assert calc.highlight_exposure_seconds == pytest.approx(30.0)
        assert calc.shadow_ratio == 0.7
        assert calc.highlight_ratio == 0.3

    def test_calculate_exposure_times_fifty_fifty(self, simulator):
        """Test 50/50 exposure split."""
        settings = SplitGradeSettings(shadow_exposure_ratio=0.5)
        calc = simulator.calculate_exposure_times(base_time=100.0, settings=settings)

        assert calc.shadow_exposure_seconds == pytest.approx(50.0)
        assert calc.highlight_exposure_seconds == pytest.approx(50.0)

    def test_calculate_exposure_times_grades_included(self, simulator):
        """Test that grades are included in calculation."""
        settings = SplitGradeSettings(shadow_grade=3.5, highlight_grade=1.5)
        calc = simulator.calculate_exposure_times(base_time=120.0, settings=settings)

        assert calc.shadow_grade == 3.5
        assert calc.highlight_grade == 1.5

    def test_calculate_exposure_times_notes_generated(self, simulator):
        """Test that helpful notes are generated."""
        settings = SplitGradeSettings(shadow_grade=4.5, highlight_grade=0.5)
        calc = simulator.calculate_exposure_times(base_time=120.0, settings=settings)

        assert isinstance(calc.notes, list)

    def test_calculate_exposure_times_short_exposure_warning(self, simulator):
        """Test warning for very short exposures."""
        calc = simulator.calculate_exposure_times(base_time=8.0)

        # Should warn about short exposures
        assert any("short" in note.lower() for note in calc.notes)

    def test_calculate_exposure_times_similar_grades_note(self, simulator):
        """Test note when grades are similar."""
        settings = SplitGradeSettings(shadow_grade=2.5, highlight_grade=2.3)
        calc = simulator.calculate_exposure_times(base_time=120.0, settings=settings)

        # Should suggest single-grade printing
        assert any("similar" in note.lower() for note in calc.notes)

    def test_exposure_calculation_format_info(self, simulator):
        """Test exposure calculation formatting."""
        calc = simulator.calculate_exposure_times(base_time=120.0)

        info = calc.format_exposure_info()

        assert isinstance(info, str)
        assert "SPLIT-GRADE EXPOSURE CALCULATION" in info
        assert "Total Exposure Time" in info
        assert "SHADOW EXPOSURE" in info
        assert "HIGHLIGHT EXPOSURE" in info

    def test_calculate_exposure_times_various_base_times(self, simulator):
        """Test exposure calculations with various base times."""
        for base_time in [60.0, 120.0, 180.0, 240.0]:
            calc = simulator.calculate_exposure_times(base_time=base_time)

            assert calc.total_exposure_seconds == base_time
            assert pytest.approx(
                calc.shadow_exposure_seconds + calc.highlight_exposure_seconds
            ) == base_time


# ==============================================================================
# Test SplitGradeSimulator - Helper Methods
# ==============================================================================


class TestSplitGradeSimulatorHelpers:
    """Tests for private helper methods."""

    @pytest.fixture
    def simulator(self):
        """Create a SplitGradeSimulator instance."""
        return SplitGradeSimulator()

    def test_prepare_image_numpy_uint8(self, simulator):
        """Test image preparation from uint8 numpy array."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        prepared = simulator._prepare_image(img)

        assert prepared.dtype == np.float32
        assert np.all(prepared >= 0.0)
        assert np.all(prepared <= 1.0)

    def test_prepare_image_numpy_uint16(self, simulator):
        """Test image preparation from uint16 numpy array."""
        img = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        prepared = simulator._prepare_image(img)

        assert prepared.dtype == np.float32
        assert np.all(prepared >= 0.0)
        assert np.all(prepared <= 1.0)

    def test_prepare_image_numpy_float(self, simulator):
        """Test image preparation from float numpy array."""
        img = np.random.rand(100, 100).astype(np.float32)
        prepared = simulator._prepare_image(img)

        assert prepared.dtype == np.float32
        np.testing.assert_array_almost_equal(prepared, img)

    def test_prepare_image_pil(self, simulator):
        """Test image preparation from PIL Image."""
        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        prepared = simulator._prepare_image(pil_img)

        assert prepared.dtype == np.float32
        assert np.all(prepared >= 0.0)
        assert np.all(prepared <= 1.0)

    def test_prepare_image_out_of_range_normalization(self, simulator):
        """Test that out-of-range values are normalized."""
        img = np.array([[0, 127, 255], [500, 1000, 2000]], dtype=np.float32)
        prepared = simulator._prepare_image(img)

        assert np.all(prepared >= 0.0)
        assert np.all(prepared <= 1.0)
        assert prepared[-1, -1] == 1.0  # Max value should be 1.0

    def test_get_luminance_grayscale(self, simulator):
        """Test luminance extraction from grayscale image."""
        img = np.random.rand(100, 100)
        luminance = simulator._get_luminance(img)

        np.testing.assert_array_equal(luminance, img)

    def test_get_luminance_rgb(self, simulator, color_test_image):
        """Test luminance extraction from RGB image."""
        luminance = simulator._get_luminance(color_test_image)

        assert luminance.ndim == 2
        assert luminance.shape == color_test_image.shape[:2]
        assert np.all(luminance >= 0.0)
        assert np.all(luminance <= 1.0)

    def test_get_luminance_invalid_dimensions(self, simulator):
        """Test error on invalid image dimensions."""
        img = np.random.rand(100, 100, 100, 100)  # 4D

        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            simulator._get_luminance(img)

    def test_soft_light_blend(self, simulator):
        """Test soft light blending formula."""
        shadow = np.array([[0.2, 0.5, 0.8]])
        highlight = np.array([[0.3, 0.6, 0.9]])
        mask = np.array([[0.5, 0.5, 0.5]])

        result = simulator._soft_light_blend(shadow, highlight, mask)

        assert result.shape == shadow.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_overlay_blend(self, simulator):
        """Test overlay blending formula."""
        shadow = np.array([[0.2, 0.5, 0.8]])
        highlight = np.array([[0.3, 0.6, 0.9]])
        mask = np.array([[0.5, 0.5, 0.5]])

        result = simulator._overlay_blend(shadow, highlight, mask)

        assert result.shape == shadow.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_recommend_grades_low_contrast(self, simulator):
        """Test grade recommendations for low contrast."""
        shadow_grade, highlight_grade = simulator._recommend_grades(
            mean_lum=0.5,
            contrast=0.2,  # Low contrast
            is_low_key=False,
            is_high_key=False
        )

        # Low contrast should get harder grades
        assert shadow_grade > 2.5
        assert highlight_grade > 1.5

    def test_recommend_grades_high_contrast(self, simulator):
        """Test grade recommendations for high contrast."""
        shadow_grade, highlight_grade = simulator._recommend_grades(
            mean_lum=0.5,
            contrast=0.8,  # High contrast
            is_low_key=False,
            is_high_key=False
        )

        # High contrast should get softer grades
        assert shadow_grade < 2.5
        assert highlight_grade < 1.5

    def test_recommend_thresholds(self, simulator):
        """Test threshold recommendations."""
        shadow_thresh, highlight_thresh = simulator._recommend_thresholds(
            p25=0.25,
            p75=0.75,
            shadow_pct=0.3,
            highlight_pct=0.3
        )

        assert 0.0 <= shadow_thresh <= 1.0
        assert 0.0 <= highlight_thresh <= 1.0
        assert shadow_thresh < highlight_thresh
        # Minimum separation
        assert (highlight_thresh - shadow_thresh) >= 0.2

    def test_recommend_exposure_ratio(self, simulator):
        """Test exposure ratio recommendations."""
        ratio = simulator._recommend_exposure_ratio(
            shadow_pct=0.4,
            highlight_pct=0.2,
            is_low_key=False,
            is_high_key=False
        )

        assert 0.4 <= ratio <= 0.75


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestSplitGradeIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_analysis_to_simulation(self, normal_contrast_image):
        """Test complete workflow from analysis to simulation."""
        simulator = SplitGradeSimulator()

        # Step 1: Analyze image
        analysis = simulator.analyze_image(normal_contrast_image)

        # Step 2: Use recommendations to create settings
        settings = SplitGradeSettings(
            shadow_grade=analysis.recommended_shadow_grade,
            highlight_grade=analysis.recommended_highlight_grade,
            shadow_threshold=analysis.recommended_shadow_threshold,
            highlight_threshold=analysis.recommended_highlight_threshold,
            shadow_exposure_ratio=analysis.recommended_exposure_ratio,
        )

        # Step 3: Simulate with recommended settings
        result = simulator.simulate_split_grade(normal_contrast_image, settings)

        assert result.shape == normal_contrast_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_full_workflow_with_exposure_calculation(self, wide_range_image):
        """Test workflow including exposure calculation."""
        simulator = SplitGradeSimulator()

        # Analyze
        analysis = simulator.analyze_image(wide_range_image)

        # Create settings
        settings = SplitGradeSettings(
            shadow_grade=analysis.recommended_shadow_grade,
            highlight_grade=analysis.recommended_highlight_grade,
        )

        # Simulate
        result = simulator.simulate_split_grade(wide_range_image, settings)

        # Calculate exposures
        exposure_calc = simulator.calculate_exposure_times(120.0, settings)

        assert result is not None
        assert exposure_calc.total_exposure_seconds == 120.0

        # Get formatted info
        info = exposure_calc.format_exposure_info()
        assert len(info) > 0

    def test_comparison_all_blend_modes(self, normal_contrast_image):
        """Compare results from all blend modes."""
        simulator = SplitGradeSimulator()
        results = {}

        for mode in list(BlendMode):
            settings = SplitGradeSettings(blend_mode=mode)
            result = simulator.simulate_split_grade(normal_contrast_image, settings)
            results[mode] = result

        # All should produce valid results
        for mode, result in results.items():
            assert result.shape == normal_contrast_image.shape
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_platinum_vs_palladium_workflow(self, normal_contrast_image):
        """Compare pure platinum vs pure palladium workflows."""
        # Platinum workflow
        pt_settings = SplitGradeSettings(platinum_ratio=1.0)
        pt_simulator = SplitGradeSimulator(pt_settings)
        pt_result = pt_simulator.simulate_split_grade(normal_contrast_image)

        # Palladium workflow
        pd_settings = SplitGradeSettings(platinum_ratio=0.0)
        pd_simulator = SplitGradeSimulator(pd_settings)
        pd_result = pd_simulator.simulate_split_grade(normal_contrast_image)

        # Should produce different results
        assert not np.allclose(pt_result, pd_result)

        # Palladium should generally be lighter in shadows
        shadow_region = normal_contrast_image < 0.3
        if np.any(shadow_region):
            assert np.mean(pd_result[shadow_region]) > np.mean(pt_result[shadow_region])
