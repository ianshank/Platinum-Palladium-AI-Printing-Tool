"""
Tests for scanner calibration module.

Tests scanner profiling, per-channel corrections, and uniformity mapping.
"""

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.detection.scanner import (
    ScannerCalibration,
    ScannerProfile,
    ChannelCurve,
)


class TestChannelCurve:
    """Tests for ChannelCurve dataclass."""

    def test_default_curve(self):
        """Default curve should be identity."""
        curve = ChannelCurve()
        assert len(curve.input_values) == 256
        assert len(curve.output_values) == 256
        # Identity: input == output
        assert curve.input_values[0] == curve.output_values[0]
        assert curve.input_values[255] == curve.output_values[255]

    def test_apply_identity(self):
        """Identity curve should not change value."""
        curve = ChannelCurve()
        assert curve.apply(0) == pytest.approx(0, abs=1)
        assert curve.apply(128) == pytest.approx(128, abs=1)
        assert curve.apply(255) == pytest.approx(255, abs=1)

    def test_apply_interpolation(self):
        """Curve should interpolate between points."""
        curve = ChannelCurve(
            input_values=list(range(256)),
            output_values=[v * 0.5 for v in range(256)],  # Half values
        )
        assert curve.apply(128) == pytest.approx(64, abs=2)

    def test_apply_clamps_input(self):
        """Apply should handle out-of-range values."""
        curve = ChannelCurve()
        # Should not crash with edge values
        result = curve.apply(0)
        assert result >= 0
        result = curve.apply(255)
        assert result <= 256


class TestScannerProfile:
    """Tests for ScannerProfile dataclass."""

    def test_default_profile(self):
        """Profile should have identity curves."""
        profile = ScannerProfile(name="Test Profile")
        assert profile.red_curve is not None
        assert profile.green_curve is not None
        assert profile.blue_curve is not None
        assert profile.uniformity_map is None

    def test_profile_with_custom_curves(self):
        """Profile should accept custom curves."""
        red = ChannelCurve(
            input_values=list(range(256)),
            output_values=[v * 1.1 for v in range(256)],
        )
        profile = ScannerProfile(name="Custom", red_curve=red)
        assert profile.red_curve == red

    def test_profile_resolution(self):
        """Profile should store resolution."""
        profile = ScannerProfile(name="Test", resolution_dpi=600)
        assert profile.resolution_dpi == 600


class TestScannerCalibration:
    """Tests for ScannerCalibration class."""

    @pytest.fixture
    def calibration(self):
        """Create scanner calibration instance with default profile."""
        profile = ScannerProfile(name="Test")
        return ScannerCalibration(profile=profile)

    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image."""
        arr = np.arange(256).reshape(16, 16).astype(np.uint8)
        return arr

    @pytest.fixture
    def rgb_image(self):
        """Create RGB test image with distinct channels."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = 100  # Red
        arr[:, :, 1] = 150  # Green
        arr[:, :, 2] = 200  # Blue
        return arr

    def test_calibration_creation(self):
        """Calibration should be created without profile."""
        calibration = ScannerCalibration()
        assert calibration.profile is None

    def test_calibration_with_profile(self):
        """Calibration should accept custom profile."""
        profile = ScannerProfile(name="Test", resolution_dpi=1200)
        calibration = ScannerCalibration(profile=profile)
        assert calibration.profile.resolution_dpi == 1200

    def test_apply_correction_identity(self, calibration, rgb_image):
        """Identity profile should not change image significantly."""
        corrected = calibration.apply_correction(rgb_image)

        assert corrected.shape == rgb_image.shape
        assert np.allclose(corrected, rgb_image, atol=2)

    def test_apply_correction_per_channel(self, rgb_image):
        """Per-channel curves should modify each channel independently."""
        # Create profile with different curves per channel
        red_curve = ChannelCurve(
            input_values=list(range(256)),
            output_values=[min(255, int(v * 1.2)) for v in range(256)],  # Boost red
        )
        green_curve = ChannelCurve(
            input_values=list(range(256)),
            output_values=[int(v * 0.8) for v in range(256)],  # Reduce green
        )
        profile = ScannerProfile(
            name="Per-channel test",
            red_curve=red_curve,
            green_curve=green_curve,
        )
        calibration = ScannerCalibration(profile=profile)

        corrected = calibration.apply_correction(rgb_image)

        # Red should be boosted
        assert corrected[:, :, 0].mean() > rgb_image[:, :, 0].mean()
        # Green should be reduced
        assert corrected[:, :, 1].mean() < rgb_image[:, :, 1].mean()
        # Blue unchanged (identity)
        assert np.allclose(corrected[:, :, 2], rgb_image[:, :, 2], atol=2)

    def test_apply_correction_grayscale(self, calibration, grayscale_image):
        """Correction should work on grayscale images."""
        corrected = calibration.apply_correction(grayscale_image)

        assert corrected.shape == grayscale_image.shape
        assert corrected.dtype == np.uint8

    def test_apply_correction_clips_values(self, rgb_image):
        """Corrected values should be clipped to valid range."""
        # Create extreme boost curve
        boost_curve = ChannelCurve(
            input_values=list(range(256)),
            output_values=[min(255, v * 2) for v in range(256)],
        )
        profile = ScannerProfile(name="Boost", red_curve=boost_curve)
        calibration = ScannerCalibration(profile=profile)

        corrected = calibration.apply_correction(rgb_image)

        assert corrected.min() >= 0
        assert corrected.max() <= 255


class TestUniformityCorrection:
    """Tests for uniformity map handling."""

    def test_uniformity_map_applied(self):
        """Uniformity map should modify image."""
        # Create image with uniform values
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Create uniformity map with variation
        uniformity = np.ones((100, 100), dtype=np.float64)
        uniformity[50:, :] = 1.1  # Bottom half is 10% brighter

        profile = ScannerProfile(name="Uniformity", uniformity_map=uniformity)
        calibration = ScannerCalibration(profile=profile)

        corrected = calibration.apply_correction(arr)

        # Bottom half should be different from top half
        top_mean = corrected[:50, :, :].mean()
        bottom_mean = corrected[50:, :, :].mean()
        assert abs(top_mean - bottom_mean) > 1

    def test_uniformity_map_resizing(self):
        """Uniformity map should resize to match image."""
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 128

        # Create smaller uniformity map
        uniformity = np.ones((50, 50), dtype=np.float64) * 1.05

        profile = ScannerProfile(name="Resize", uniformity_map=uniformity)
        calibration = ScannerCalibration(profile=profile)

        # Should not crash, map will be resized
        corrected = calibration.apply_correction(arr)
        assert corrected.shape == arr.shape


class TestApplyCorrectionNoProfile:
    """Tests for apply_correction without profile."""

    def test_no_profile_returns_unchanged(self):
        """Without profile, image should be returned unchanged."""
        calibration = ScannerCalibration()  # No profile
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = calibration.apply_correction(arr)
        assert np.array_equal(result, arr)

    def test_grayscale_with_uniformity(self):
        """Grayscale image with uniformity correction."""
        uniformity = np.ones((50, 50), dtype=np.float64) * 1.1
        profile = ScannerProfile(name="Gray Uniformity", uniformity_map=uniformity)
        calibration = ScannerCalibration(profile=profile)

        arr = np.ones((50, 50), dtype=np.uint8) * 100
        corrected = calibration.apply_correction(arr)

        # Should be scaled by uniformity factor
        assert corrected.mean() > arr.mean()


class TestCalibrateSimple:
    """Tests for simple two-point calibration."""

    def test_calibrate_simple_linear(self):
        """Simple calibration should create linear correction curves."""
        calibration = ScannerCalibration()

        # Scanner reads 10-240 for actual 0-255 range
        profile = calibration.calibrate_simple(
            white_sample=(240.0, 240.0, 240.0),
            black_sample=(10.0, 10.0, 10.0),
            target_white=(255.0, 255.0, 255.0),
            target_black=(0.0, 0.0, 0.0),
            name="Simple Cal",
        )

        assert profile.name == "Simple Cal"
        assert calibration.profile == profile

    def test_calibrate_simple_per_channel(self):
        """Simple calibration with different per-channel values."""
        calibration = ScannerCalibration()

        profile = calibration.calibrate_simple(
            white_sample=(230.0, 240.0, 250.0),
            black_sample=(5.0, 10.0, 15.0),
            name="Per-channel Cal",
        )

        # Each channel should have different correction
        assert profile.red_curve.output_values != profile.green_curve.output_values

    def test_calibrate_simple_handles_same_black_white(self):
        """Should handle case where black equals white."""
        calibration = ScannerCalibration()

        # Edge case: same value for black and white
        profile = calibration.calibrate_simple(
            white_sample=(128.0, 128.0, 128.0),
            black_sample=(128.0, 128.0, 128.0),
            name="Edge Case",
        )
        # Should not crash
        assert profile is not None


class TestCalibrateFromTarget:
    """Tests for target-based calibration."""

    @pytest.fixture
    def target_image(self):
        """Create calibration target image."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(10):
            val = i * 25
            arr[i * 10:(i + 1) * 10, :, :] = val
        return arr

    def test_calibrate_from_target_creates_profile(self, target_image):
        """Calibration from target should create profile with uniformity map."""
        calibration = ScannerCalibration()

        profile = calibration.calibrate_from_target(
            target_scan=target_image,
            reference_values={"patch_1": (50, 50, 50)},
            name="Target Cal",
        )

        assert profile.name == "Target Cal"
        assert profile.uniformity_map is not None
        assert profile.uniformity_map_size == (100, 100)
        assert calibration.profile == profile


class TestSaveLoad:
    """Tests for profile save/load."""

    def test_save_profile(self, tmp_path):
        """Should save profile to JSON."""
        profile = ScannerProfile(
            name="Save Test",
            scanner_model="Test Scanner",
            resolution_dpi=600,
        )
        calibration = ScannerCalibration(profile=profile)

        path = tmp_path / "scanner_profile.json"
        calibration.save(path)

        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["name"] == "Save Test"
        assert data["scanner_model"] == "Test Scanner"
        assert data["resolution_dpi"] == 600

    def test_save_profile_with_uniformity(self, tmp_path):
        """Should save profile with uniformity map."""
        uniformity = np.ones((10, 10), dtype=np.float64) * 1.05
        profile = ScannerProfile(
            name="Uniformity Save",
            uniformity_map=uniformity,
            uniformity_map_size=(10, 10),
        )
        calibration = ScannerCalibration(profile=profile)

        path = tmp_path / "scanner_uniformity.json"
        calibration.save(path)

        import json
        data = json.loads(path.read_text())
        assert "uniformity_map" in data
        assert data["uniformity_map_size"] == [10, 10]

    def test_save_no_profile_raises_error(self, tmp_path):
        """Should raise error when saving without profile."""
        calibration = ScannerCalibration()
        path = tmp_path / "no_profile.json"

        with pytest.raises(ValueError, match="No profile"):
            calibration.save(path)

    def test_load_profile(self, tmp_path):
        """Should load profile from JSON."""
        # First save a profile
        profile = ScannerProfile(
            name="Load Test",
            resolution_dpi=1200,
        )
        calibration = ScannerCalibration(profile=profile)
        path = tmp_path / "load_test.json"
        calibration.save(path)

        # Then load it
        loaded = ScannerCalibration.load(path)
        assert loaded.profile.name == "Load Test"
        assert loaded.profile.resolution_dpi == 1200

    def test_load_profile_with_uniformity(self, tmp_path):
        """Should load profile with uniformity map."""
        uniformity = np.ones((20, 20), dtype=np.float64) * 1.1
        profile = ScannerProfile(
            name="Load Uniformity",
            uniformity_map=uniformity,
            uniformity_map_size=(20, 20),
        )
        calibration = ScannerCalibration(profile=profile)
        path = tmp_path / "load_uniformity.json"
        calibration.save(path)

        loaded = ScannerCalibration.load(path)
        assert loaded.profile.uniformity_map is not None
        assert loaded.profile.uniformity_map.shape == (20, 20)


class TestLoadImage:
    """Tests for _load_image helper."""

    def test_load_from_numpy(self):
        """Should pass through numpy array."""
        calibration = ScannerCalibration()
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        result = calibration._load_image(arr)
        assert np.array_equal(result, arr)

    def test_load_from_pil(self):
        """Should convert PIL Image to array."""
        calibration = ScannerCalibration()
        img = Image.new("RGB", (50, 50), color=(100, 150, 200))
        result = calibration._load_image(img)
        assert result.shape == (50, 50, 3)
        assert result[0, 0, 0] == 100
        assert result[0, 0, 1] == 150

    def test_load_from_path(self, tmp_path):
        """Should load from file path."""
        calibration = ScannerCalibration()

        # Create test image
        img = Image.new("RGB", (50, 50), color=(50, 100, 150))
        path = tmp_path / "test_image.png"
        img.save(path)

        result = calibration._load_image(path)
        assert result.shape == (50, 50, 3)

    def test_load_from_string_path(self, tmp_path):
        """Should load from string path."""
        calibration = ScannerCalibration()

        img = Image.new("RGB", (30, 30), color=(200, 100, 50))
        path = tmp_path / "string_path.png"
        img.save(path)

        result = calibration._load_image(str(path))
        assert result.shape == (30, 30, 3)

    def test_load_invalid_type_raises(self):
        """Should raise error for invalid type."""
        calibration = ScannerCalibration()

        with pytest.raises(TypeError, match="Unsupported image type"):
            calibration._load_image(12345)


class TestAnalyzeUniformity:
    """Tests for uniformity analysis."""

    def test_analyze_uniform_image(self):
        """Uniform image should have near-identity uniformity map."""
        calibration = ScannerCalibration()
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128

        uniformity = calibration._analyze_uniformity(arr)

        # Should be close to 1.0 everywhere
        assert uniformity.shape == (100, 100)
        assert np.allclose(uniformity, 1.0, atol=0.01)

    def test_analyze_grayscale_image(self):
        """Should handle grayscale image."""
        calibration = ScannerCalibration()
        arr = np.ones((100, 100), dtype=np.uint8) * 128

        uniformity = calibration._analyze_uniformity(arr)

        assert uniformity.shape == (100, 100)

    def test_analyze_vignetting(self):
        """Should detect vignetting pattern."""
        calibration = ScannerCalibration()

        # Create image with vignetting (darker at edges)
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 200
        # Make center brighter
        for i in range(100):
            for j in range(100):
                dist = np.sqrt((i - 50) ** 2 + (j - 50) ** 2)
                factor = max(0.5, 1 - dist / 100)
                arr[i, j] = int(200 * factor)

        uniformity = calibration._analyze_uniformity(arr)

        # Center should have lower correction factor (it's brighter)
        center_val = uniformity[50, 50]
        edge_val = uniformity[0, 0]
        # Edge needs more correction to match center
        assert edge_val > center_val


class TestScannerCalibrationWorkflow:
    """Tests for complete calibration workflow."""

    @pytest.fixture
    def target_image(self):
        """Create calibration target image."""
        # Simulate step wedge with known values
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        for i in range(10):
            val = i * 25
            arr[:, i * 20:(i + 1) * 20, :] = val
        return arr

    def test_create_profile_from_target(self, target_image):
        """Create profile from calibration target scan."""
        # This tests that the workflow can complete
        calibration = ScannerCalibration()

        # Simulate creating profile (basic test)
        profile = ScannerProfile(
            name="Target Profile",
            resolution_dpi=300,
        )
        calibration = ScannerCalibration(profile=profile)

        # Apply correction (should work with default identity curves)
        corrected = calibration.apply_correction(target_image)
        assert corrected is not None

    def test_round_trip_correction(self):
        """Apply and then inverse correction."""
        arr = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)

        # With identity curves, round trip should preserve values
        profile = ScannerProfile(name="Identity")
        calibration = ScannerCalibration(profile=profile)
        corrected = calibration.apply_correction(arr)

        # Should be very close to original
        assert np.allclose(corrected, arr, atol=2)

    def test_full_workflow(self, tmp_path, target_image):
        """Test complete calibration workflow: calibrate, save, load, apply."""
        # 1. Calibrate from target
        cal1 = ScannerCalibration()
        profile = cal1.calibrate_from_target(
            target_scan=target_image,
            reference_values={},
            name="Workflow Test",
        )

        # 2. Save profile
        path = tmp_path / "workflow_profile.json"
        cal1.save(path)

        # 3. Load profile
        cal2 = ScannerCalibration.load(path)
        assert cal2.profile.name == "Workflow Test"

        # 4. Apply correction
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        corrected = cal2.apply_correction(test_image)
        assert corrected.shape == test_image.shape
