"""
Tests for the ACV export functionality.

Verifies curve export to Photoshop format works correctly.
"""

import struct

import numpy as np
import pytest

from ptpd_calibration.alphazero.export.acv import (
    ACVExporter,
    check_acv_export,
    export_to_acv,
)


class TestACVExporter:
    """Tests for the ACVExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create an exporter instance."""
        return ACVExporter()

    @pytest.fixture
    def sample_densities(self):
        """Create sample density values."""
        steps = np.linspace(0, 1, 21)
        return 0.1 + 1.9 * np.power(steps, 1.8)

    def test_create_exporter(self, exporter):
        """Test creating an exporter."""
        assert exporter is not None

    def test_export_creates_file(self, exporter, sample_densities, tmp_path):
        """Test that export creates a file."""
        output_path = tmp_path / "test_curve.acv"
        result_path = exporter.export(sample_densities, output_path)

        assert result_path.exists()
        assert result_path.suffix == ".acv"

    def test_export_adds_suffix(self, exporter, sample_densities, tmp_path):
        """Test that .acv suffix is added if missing."""
        output_path = tmp_path / "test_curve"  # No suffix
        result_path = exporter.export(sample_densities, output_path)

        assert result_path.suffix == ".acv"

    def test_export_file_format(self, exporter, sample_densities, tmp_path):
        """Test the exported file format."""
        output_path = tmp_path / "test_curve.acv"
        exporter.export(sample_densities, output_path)

        with open(output_path, "rb") as f:
            # Read header
            version = struct.unpack(">H", f.read(2))[0]
            num_curves = struct.unpack(">H", f.read(2))[0]

            assert version == 4  # ACV version 4
            assert num_curves == 5  # Master + RGBK

            # Read first curve (master)
            num_points = struct.unpack(">H", f.read(2))[0]
            assert 2 <= num_points <= 16

    def test_export_curve_points(self, exporter, sample_densities, tmp_path):
        """Test that curve points are valid."""
        output_path = tmp_path / "test_curve.acv"
        exporter.export(sample_densities, output_path)

        with open(output_path, "rb") as f:
            # Skip header
            f.read(4)

            # Read master curve
            num_points = struct.unpack(">H", f.read(2))[0]

            points = []
            for _ in range(num_points):
                output = struct.unpack(">H", f.read(2))[0]
                input = struct.unpack(">H", f.read(2))[0]
                points.append((input, output))

            # Check all points are in valid range
            for input_val, output_val in points:
                assert 0 <= input_val <= 255
                assert 0 <= output_val <= 255

            # Check endpoints
            assert points[0][0] == 0  # First input is 0
            assert points[-1][0] == 255  # Last input is 255

    def test_linear_curve(self, exporter):
        """Test creating a linear curve."""
        curve = exporter.create_linear_curve()

        assert len(curve) == 2
        assert curve[0] == (0, 0)
        assert curve[1] == (255, 255)

    def test_contrast_curve(self, exporter):
        """Test creating a contrast curve."""
        curve = exporter.create_contrast_curve(amount=0.3)

        assert len(curve) > 2
        assert curve[0][0] == 0
        assert curve[-1][0] == 255

    def test_densities_to_curve(self, exporter, sample_densities):
        """Test converting densities to curve points."""
        curve = exporter._densities_to_curve(sample_densities)

        assert 2 <= len(curve) <= 16
        assert curve[0] == (0, 0)
        assert curve[-1] == (255, 255)

        # Check monotonicity of input values
        input_values = [p[0] for p in curve]
        assert input_values == sorted(input_values)


class TestExportConvenience:
    """Tests for the convenience export function."""

    @pytest.fixture
    def sample_densities(self):
        """Create sample density values."""
        return np.linspace(0.1, 2.0, 21)

    def test_export_to_acv(self, sample_densities, tmp_path):
        """Test the convenience function."""
        output_path = tmp_path / "test.acv"
        result = export_to_acv(sample_densities, output_path)

        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_with_string_path(self, sample_densities, tmp_path):
        """Test export with string path."""
        output_path = str(tmp_path / "test.acv")
        result = export_to_acv(sample_densities, output_path)

        assert result.exists()

    def test_check_acv_export(self):
        """Run the built-in export check."""
        assert check_acv_export()


class TestCurveCorrection:
    """Tests for curve correction logic."""

    @pytest.fixture
    def exporter(self):
        """Create an exporter instance."""
        return ACVExporter()

    def test_linear_densities_give_linear_curve(self, exporter):
        """Test that linear densities produce nearly linear curve."""
        linear_densities = np.linspace(0.1, 2.0, 21)
        curve = exporter._densities_to_curve(linear_densities)

        # For perfectly linear response, correction should be minimal
        for input_val, output_val in curve:
            # Should be close to identity
            assert abs(input_val - output_val) < 50  # Allow some tolerance

    def test_nonlinear_densities_create_correction(self, exporter):
        """Test that non-linear densities create correction curve."""
        # High gamma (non-linear) densities
        steps = np.linspace(0, 1, 21)
        nonlinear_densities = 0.1 + 1.9 * np.power(steps, 2.5)

        curve = exporter._densities_to_curve(nonlinear_densities)

        # Correction should differ from identity
        # Check that curve has more than just endpoints
        assert len(curve) > 2, "Non-linear response should create multi-point curve"
