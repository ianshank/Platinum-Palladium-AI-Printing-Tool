"""
Tests for core data models.
"""

import pytest
from datetime import datetime
from uuid import UUID

import numpy as np

from ptpd_calibration.core.models import (
    PatchData,
    DensityMeasurement,
    ExtractionResult,
    StepTabletResult,
    CurveData,
    PaperProfile,
    CalibrationRecord,
)
from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    CurveType,
    DeveloperType,
    MeasurementUnit,
    PaperSizing,
)


class TestPatchData:
    """Tests for PatchData model."""

    def test_create_patch_data(self):
        """Test basic patch data creation."""
        patch = PatchData(
            index=0,
            position=(10, 20, 100, 50),
            rgb_mean=(128.0, 128.0, 128.0),
            rgb_std=(5.0, 5.0, 5.0),
        )

        assert patch.index == 0
        assert patch.position == (10, 20, 100, 50)
        assert patch.rgb_mean == (128.0, 128.0, 128.0)
        assert patch.density is None
        assert patch.uniformity == 1.0

    def test_patch_with_density(self):
        """Test patch data with density value."""
        patch = PatchData(
            index=5,
            position=(0, 0, 50, 50),
            rgb_mean=(64.0, 64.0, 64.0),
            rgb_std=(2.0, 2.0, 2.0),
            density=1.5,
            uniformity=0.95,
        )

        assert patch.density == 1.5
        assert patch.uniformity == 0.95

    def test_patch_with_numpy_arrays(self):
        """Test that numpy arrays are converted to tuples."""
        rgb = np.array([100.0, 150.0, 200.0])
        patch = PatchData(
            index=0,
            position=(0, 0, 10, 10),
            rgb_mean=rgb,
            rgb_std=(1.0, 1.0, 1.0),
        )

        assert isinstance(patch.rgb_mean, tuple)
        assert patch.rgb_mean == (100.0, 150.0, 200.0)


class TestDensityMeasurement:
    """Tests for DensityMeasurement model."""

    def test_create_measurement(self):
        """Test density measurement creation."""
        measurement = DensityMeasurement(
            step=10,
            input_value=0.5,
            density=1.2,
        )

        assert measurement.step == 10
        assert measurement.input_value == 0.5
        assert measurement.density == 1.2
        assert measurement.unit == MeasurementUnit.VISUAL_DENSITY

    def test_measurement_with_lab(self):
        """Test measurement with Lab values."""
        measurement = DensityMeasurement(
            step=5,
            input_value=0.25,
            density=0.8,
            lab=(50.0, 2.0, 5.0),
        )

        assert measurement.lab == (50.0, 2.0, 5.0)


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_extraction_result(self):
        """Test extraction result creation."""
        patches = [
            PatchData(
                index=i,
                position=(i * 50, 0, 50, 100),
                rgb_mean=(200.0 - i * 20, 200.0 - i * 20, 200.0 - i * 20),
                rgb_std=(3.0, 3.0, 3.0),
                density=0.1 + i * 0.2,
            )
            for i in range(5)
        ]

        result = ExtractionResult(
            image_size=(500, 100),
            tablet_bounds=(0, 0, 500, 100),
            patches=patches,
        )

        assert result.num_patches == 5
        assert len(result.patches) == 5
        assert result.image_size == (500, 100)

    def test_get_densities(self):
        """Test density extraction from patches."""
        patches = [
            PatchData(
                index=i,
                position=(0, 0, 10, 10),
                rgb_mean=(128.0, 128.0, 128.0),
                rgb_std=(1.0, 1.0, 1.0),
                density=0.1 + i * 0.3,
            )
            for i in range(4)
        ]

        result = ExtractionResult(
            image_size=(100, 100),
            tablet_bounds=(0, 0, 100, 100),
            patches=patches,
        )

        densities = result.get_densities()
        assert len(densities) == 4
        assert densities[0] == pytest.approx(0.1, rel=0.01)
        assert densities[3] == pytest.approx(1.0, rel=0.01)

    def test_dmin_dmax_properties(self):
        """Test Dmin and Dmax property calculations."""
        patches = [
            PatchData(
                index=i,
                position=(0, 0, 10, 10),
                rgb_mean=(128.0, 128.0, 128.0),
                rgb_std=(1.0, 1.0, 1.0),
                density=0.1 + i * 0.5,
            )
            for i in range(5)
        ]

        result = ExtractionResult(
            image_size=(100, 100),
            tablet_bounds=(0, 0, 100, 100),
            patches=patches,
        )

        assert result.dmin == pytest.approx(0.1, rel=0.01)
        assert result.dmax == pytest.approx(2.1, rel=0.01)
        assert result.density_range == pytest.approx(2.0, rel=0.01)


class TestCurveData:
    """Tests for CurveData model."""

    def test_create_curve_data(self):
        """Test curve data creation."""
        curve = CurveData(
            name="Test Curve",
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.6, 1.0],
        )

        assert curve.name == "Test Curve"
        assert len(curve.input_values) == 3
        assert len(curve.output_values) == 3
        assert isinstance(curve.id, UUID)

    def test_curve_interpolation(self):
        """Test curve interpolation."""
        curve = CurveData(
            name="Test",
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.4, 1.0],
        )

        # Test interpolation at known point
        assert curve.interpolate(0.0) == pytest.approx(0.0, abs=0.01)
        assert curve.interpolate(1.0) == pytest.approx(1.0, abs=0.01)

        # Test interpolation at intermediate point
        mid = curve.interpolate(0.25)
        assert 0.1 < mid < 0.3  # Should be between endpoints

    def test_curve_to_numpy(self):
        """Test conversion to numpy arrays."""
        curve = CurveData(
            name="Test",
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.6, 1.0],
        )

        inp, out = curve.to_numpy()
        assert isinstance(inp, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert len(inp) == 3

    def test_curve_validation(self):
        """Test that curve requires matching lengths."""
        with pytest.raises(ValueError):
            CurveData(
                name="Bad Curve",
                input_values=[0.0, 0.5, 1.0],
                output_values=[0.0, 1.0],  # Wrong length
            )


class TestCalibrationRecord:
    """Tests for CalibrationRecord model."""

    def test_create_calibration_record(self):
        """Test calibration record creation."""
        record = CalibrationRecord(
            paper_type="Arches Platine",
            exposure_time=180.0,
            metal_ratio=0.5,
        )

        assert record.paper_type == "Arches Platine"
        assert record.exposure_time == 180.0
        assert record.metal_ratio == 0.5
        assert isinstance(record.id, UUID)
        assert isinstance(record.timestamp, datetime)

    def test_full_calibration_record(self):
        """Test record with all fields."""
        record = CalibrationRecord(
            paper_type="Bergger COT320",
            paper_weight=320,
            paper_sizing=PaperSizing.INTERNAL,
            chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            metal_ratio=0.6,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
            exposure_time=200.0,
            humidity=50.0,
            temperature=21.0,
            measured_densities=[0.1, 0.3, 0.6, 1.0, 1.5, 2.0],
            notes="Test calibration",
        )

        assert record.chemistry_type == ChemistryType.PLATINUM_PALLADIUM
        assert record.contrast_agent == ContrastAgent.NA2
        assert len(record.measured_densities) == 6

    def test_feature_vector_generation(self):
        """Test feature vector for ML."""
        record = CalibrationRecord(
            paper_type="Test Paper",
            exposure_time=180.0,
            metal_ratio=0.5,
            contrast_agent=ContrastAgent.NA2,
            contrast_amount=5.0,
            humidity=50.0,
            temperature=21.0,
        )

        features = record.get_feature_vector()
        assert len(features) == 6
        assert features[0] == 0.5  # metal_ratio
        assert features[1] == 1.0  # has_contrast_agent


class TestPaperProfile:
    """Tests for PaperProfile model."""

    def test_create_paper_profile(self):
        """Test paper profile creation."""
        profile = PaperProfile(
            name="Arches Platine",
            manufacturer="Arches",
            weight_gsm=310,
            sizing=PaperSizing.INTERNAL,
        )

        assert profile.name == "Arches Platine"
        assert profile.weight_gsm == 310
        assert profile.sizing == PaperSizing.INTERNAL
