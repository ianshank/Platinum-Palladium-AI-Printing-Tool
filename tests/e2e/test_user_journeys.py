"""
End-to-end user journey tests for the Pt/Pd Calibration Studio.

These tests simulate complete user workflows without requiring browser automation.
They test the full integration of modules as a user would experience them.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.core.types import CurveType


class TestCalibrationJourney:
    """Test the complete calibration workflow."""

    @pytest.fixture
    def sample_step_tablet(self):
        """Create a sample step tablet image."""
        width, height = 420, 100
        num_patches = 21
        patch_width = width // num_patches

        img = np.zeros((height, width), dtype=np.uint8)

        for i in range(num_patches):
            value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            img[:, x_start:x_end] = value

        full_img = np.full((height + 40, width + 40, 3), 250, dtype=np.uint8)
        full_img[20 : height + 20, 20 : width + 20, 0] = img
        full_img[20 : height + 20, 20 : width + 20, 1] = img
        full_img[20 : height + 20, 20 : width + 20, 2] = img

        return Image.fromarray(full_img)

    def test_full_calibration_workflow(self, sample_step_tablet, tmp_path):
        """
        Complete workflow: Scan → Analyze → Generate Curve → Export
        """
        from ptpd_calibration.detection import StepTabletReader
        from ptpd_calibration.curves import CurveGenerator, save_curve
        from ptpd_calibration.analysis import StepWedgeAnalyzer

        # Step 1: Save the sample image
        image_path = tmp_path / "step_tablet.png"
        sample_step_tablet.save(image_path)

        # Step 2: Read step tablet
        reader = StepTabletReader()
        result = reader.read(str(image_path))

        assert result is not None
        assert result.extraction is not None
        densities = result.extraction.get_densities()
        assert len(densities) > 0
        dmax = max(densities)
        dmin = min(densities)
        assert dmax > dmin

        # Step 3: Analyze with step wedge analyzer (skip if not enough points)
        if len(densities) >= 5:
            analyzer = StepWedgeAnalyzer()
            analysis = analyzer.analyze_from_densities(densities)
            assert analysis is not None
        # With synthetic images, detection may not extract enough patches

        # Step 4: Generate linearization curve
        generator = CurveGenerator()
        curve = generator.generate_from_extraction(
            result.extraction,
            curve_type=CurveType.LINEAR,
            name="Test Calibration",
            paper_type="Test Paper",
        )

        assert curve is not None
        assert len(curve.input_values) > 0
        assert len(curve.output_values) > 0

        # Step 5: Export curve
        export_path = tmp_path / "test_curve.txt"
        save_curve(curve, export_path, format="qtr")

        assert export_path.exists()
        content = export_path.read_text()
        assert len(content) > 0


class TestDigitalNegativeJourney:
    """Test digital negative creation workflow."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample grayscale image."""
        arr = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
        arr = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)
        return Image.fromarray(arr, mode="L")

    @pytest.fixture
    def sample_curve(self):
        """Create a sample correction curve."""
        return CurveData(
            name="Test Curve",
            input_values=[i / 10.0 for i in range(11)],
            output_values=[i / 10.0 for i in range(11)],
        )

    def test_digital_negative_workflow(self, sample_image, sample_curve, tmp_path):
        """
        Workflow: Load Image → Preview → Apply Curve → Export Negative
        """
        from ptpd_calibration.imaging import (
            ImageProcessor,
            ImageFormat,
            ExportSettings,
        )

        # Step 1: Save sample image
        image_path = tmp_path / "sample.png"
        sample_image.save(image_path)

        # Step 2: Initialize processor
        processor = ImageProcessor()

        # Step 3: Preview curve effect
        original, processed = processor.preview_curve_effect(
            str(image_path),
            sample_curve,
        )

        assert original is not None
        assert processed is not None

        # Step 4: Create digital negative
        result = processor.create_digital_negative(
            str(image_path),
            curve=sample_curve,
            invert=True,
        )

        assert result is not None
        assert result.image is not None

        # Step 5: Export
        export_path = tmp_path / "negative.tiff"
        settings = ExportSettings(format=ImageFormat.TIFF)
        processor.export(result, str(export_path), settings)

        assert export_path.exists()


class TestChemistryJourney:
    """Test chemistry calculation workflow."""

    def test_chemistry_calculation_workflow(self):
        """
        Workflow: Select Size → Choose Metal Mix → Get Recipe
        """
        from ptpd_calibration.chemistry import ChemistryCalculator

        # Step 1: Create calculator
        calculator = ChemistryCalculator()

        # Step 2: Calculate for 8x10 print
        recipe = calculator.calculate(
            width_inches=8.0,
            height_inches=10.0,
            platinum_ratio=0.5,
        )

        assert recipe is not None
        assert recipe.platinum_drops + recipe.palladium_drops > 0
        assert recipe.ferric_oxalate_drops > 0

        # Step 3: Verify recipe formatting
        formatted = recipe.format_recipe()
        assert "drops" in formatted.lower()

        # Step 4: Try different presets
        for size_name, (w, h) in ChemistryCalculator.get_standard_sizes().items():
            result = calculator.calculate(width_inches=w, height_inches=h)
            assert result.platinum_drops + result.palladium_drops > 0


class TestExposureJourney:
    """Test exposure calculation workflow."""

    def test_exposure_calculation_workflow(self):
        """
        Workflow: Set Base → Adjust Parameters → Calculate → Generate Test Strip
        """
        from ptpd_calibration.exposure import (
            ExposureCalculator,
            ExposureSettings,
            LightSource,
        )

        # Step 1: Create calculator with base settings
        settings = ExposureSettings(
            base_exposure_minutes=10.0,
            base_negative_density=1.6,
            light_source=LightSource.BL_FLUORESCENT,
        )
        calculator = ExposureCalculator(settings)

        # Step 2: Calculate for current negative
        result = calculator.calculate(
            negative_density=1.8,
            distance_inches=6.0,
        )

        assert result.exposure_minutes > 0
        formatted = result.format_time()
        assert len(formatted) > 0

        # Step 3: Generate test strip
        test_times = calculator.calculate_test_strip(
            center_exposure=result.exposure_minutes,
            steps=5,
            increment_stops=0.5,
        )

        assert len(test_times) == 5
        assert min(test_times) < result.exposure_minutes < max(test_times)


class TestZoneSystemJourney:
    """Test zone system analysis workflow."""

    @pytest.fixture
    def sample_gradient(self):
        """Create a gradient image for zone analysis."""
        arr = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
        arr = np.repeat(np.repeat(arr, 10, axis=0), 10, axis=1)
        return Image.fromarray(arr, mode="L")

    def test_zone_analysis_workflow(self, sample_gradient, tmp_path):
        """
        Workflow: Upload Image → Analyze Zones → Get Recommendations
        """
        from ptpd_calibration.zones import (
            ZoneMapper,
            ZoneMapping,
            Zone,
        )

        # Step 1: Set up paper characteristics
        mapping = ZoneMapping(paper_dmax=1.6, paper_dmin=0.08)
        mapper = ZoneMapper(mapping)

        # Step 2: Analyze image
        analysis = mapper.analyze_image(sample_gradient)

        assert analysis is not None
        assert len(analysis.zone_histogram) == 11

        # Step 3: Check zone distribution
        for zone in Zone:
            assert zone in analysis.zone_histogram

        # Step 4: Get development recommendation
        assert analysis.development_adjustment in ["N-2", "N-1", "N", "N+1", "N+2"]


class TestSoftProofingJourney:
    """Test soft proofing workflow."""

    @pytest.fixture
    def sample_image(self):
        """Create a test image for proofing."""
        arr = np.linspace(50, 200, 100).reshape(10, 10).astype(np.uint8)
        arr = np.repeat(np.repeat(arr, 10, axis=0), 10, axis=1)
        return Image.fromarray(arr, mode="L")

    def test_soft_proofing_workflow(self, sample_image):
        """
        Workflow: Select Paper → Adjust Settings → Generate Proof
        """
        from ptpd_calibration.proofing import (
            SoftProofer,
            ProofSettings,
            PaperSimulation,
        )

        # Step 1: Try different paper presets
        for paper in [
            PaperSimulation.ARCHES_PLATINE,
            PaperSimulation.BERGGER_COT320,
            PaperSimulation.STONEHENGE,
        ]:
            settings = ProofSettings.from_paper_preset(paper)
            proofer = SoftProofer(settings)
            result = proofer.proof(sample_image)

            assert result.image is not None
            assert result.image.mode == "RGB"

        # Step 2: Custom settings
        custom_settings = ProofSettings(
            paper_dmax=1.7,
            paper_dmin=0.06,
            platinum_ratio=0.5,
        )
        proofer = SoftProofer(custom_settings)
        result = proofer.proof(sample_image)

        assert result.image is not None


class TestHistogramJourney:
    """Test histogram analysis workflow."""

    @pytest.fixture
    def sample_image(self):
        """Create a test image."""
        arr = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    def test_histogram_analysis_workflow(self, sample_image):
        """
        Workflow: Upload Image → Analyze → View Statistics → Get Recommendations
        """
        from ptpd_calibration.imaging import (
            HistogramAnalyzer,
            HistogramScale,
        )

        # Step 1: Create analyzer
        analyzer = HistogramAnalyzer()

        # Step 2: Analyze image
        result = analyzer.analyze(sample_image)

        assert result is not None
        assert len(result.histogram) == 256

        # Step 3: Check statistics
        stats = result.stats
        assert 0 <= stats.mean <= 255
        assert 0 <= stats.median <= 255
        assert stats.brightness >= 0

        # Step 4: Check zone distribution
        assert len(stats.zone_distribution) == 11

        # Step 5: Generate plot
        fig = analyzer.create_histogram_plot(
            result,
            scale=HistogramScale.LINEAR,
            show_zones=True,
        )
        assert fig is not None


class TestAutoLinearizationJourney:
    """Test auto-linearization workflow."""

    def test_auto_linearization_workflow(self):
        """
        Workflow: Enter Densities → Select Method → Generate Curve → Export
        """
        from ptpd_calibration.curves import (
            AutoLinearizer,
            LinearizationMethod,
            TargetResponse,
        )

        # Step 1: Prepare density measurements
        densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

        # Step 2: Create linearizer
        linearizer = AutoLinearizer()

        # Step 3: Test different methods
        for method in LinearizationMethod:
            result = linearizer.linearize(
                densities,
                curve_name=f"Test {method.value}",
                method=method,
            )

            assert result.curve is not None
            assert result.residual_error >= 0

        # Step 4: Test different targets
        for target in [TargetResponse.LINEAR, TargetResponse.GAMMA_22]:
            result = linearizer.linearize(
                densities,
                target=target,
            )
            assert result.target_response == target


class TestPaperProfilesJourney:
    """Test paper profiles workflow."""

    def test_paper_profiles_workflow(self):
        """
        Workflow: Browse Papers → View Details → Add Custom
        """
        from ptpd_calibration.papers import (
            PaperDatabase,
            PaperProfile,
        )

        # Step 1: Create database
        db = PaperDatabase()

        # Step 2: Get all papers
        papers = db.list_papers()
        assert len(papers) >= 4  # Should have built-in papers

        # Step 3: View specific paper
        arches = db.get_paper("arches_platine")
        assert arches is not None
        assert arches.characteristics is not None

        # Step 4: Convert to dict
        profile_dict = arches.to_dict()
        assert "name" in profile_dict
        assert "characteristics" in profile_dict


class TestPrintSessionJourney:
    """Test print session logging workflow."""

    def test_print_session_workflow(self, tmp_path):
        """
        Workflow: Start Session → Log Prints → Review History → Get Stats
        """
        from ptpd_calibration.session import (
            SessionLogger,
            PrintRecord,
        )
        from ptpd_calibration.session.logger import ChemistryUsed, PrintResult

        # Step 1: Create logger
        logger = SessionLogger(storage_dir=tmp_path / "sessions")

        # Step 2: Start a session
        session = logger.start_session("Test Session")

        # Step 3: Log some prints
        for i in range(3):
            record = PrintRecord(
                image_name=f"Test Image {i+1}",
                paper_type="Arches Platine",
                exposure_time_minutes=10.0 + i,
                chemistry=ChemistryUsed(
                    ferric_oxalate_drops=10.0,
                    palladium_drops=5.0,
                    platinum_drops=5.0,
                ),
                notes=f"Test notes {i}",
                result=PrintResult.GOOD,
            )
            logger.log_print(record)

        # Step 4: Check session records
        current_session = logger.get_current_session()
        assert current_session is not None
        assert len(current_session.records) == 3

        # Step 5: Get statistics
        stats = current_session.get_statistics()
        assert stats["total_prints"] == 3


class TestIntegratedWorkflow:
    """Test complex integrated workflows."""

    @pytest.fixture
    def sample_image(self):
        """Create a test image."""
        arr = np.linspace(30, 220, 256).reshape(16, 16).astype(np.uint8)
        arr = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)
        return Image.fromarray(arr, mode="L")

    def test_complete_print_preparation(self, sample_image, tmp_path):
        """
        Complete workflow: Analyze → Calculate Chemistry → Calculate Exposure →
        Generate Negative → Preview Proof
        """
        from ptpd_calibration.imaging import ImageProcessor, HistogramAnalyzer
        from ptpd_calibration.chemistry import ChemistryCalculator
        from ptpd_calibration.exposure import ExposureCalculator, ExposureSettings
        from ptpd_calibration.zones import ZoneMapper
        from ptpd_calibration.proofing import SoftProofer, ProofSettings

        # Step 1: Analyze image histogram
        hist_analyzer = HistogramAnalyzer()
        hist_result = hist_analyzer.analyze(sample_image)
        assert hist_result.stats.dynamic_range > 0

        # Step 2: Zone analysis
        zone_mapper = ZoneMapper()
        zone_analysis = zone_mapper.analyze_image(sample_image)
        dev_adjustment = zone_analysis.development_adjustment

        # Step 3: Calculate chemistry for 8x10
        chem_calc = ChemistryCalculator()
        recipe = chem_calc.calculate(
            width_inches=8.0,
            height_inches=10.0,
            platinum_ratio=0.5,
        )
        assert recipe.platinum_drops + recipe.palladium_drops > 0

        # Step 4: Calculate exposure
        exp_settings = ExposureSettings(
            base_exposure_minutes=10.0,
            platinum_ratio=0.5,
        )
        exp_calc = ExposureCalculator(exp_settings)
        exposure = exp_calc.calculate(negative_density=1.6)
        assert exposure.exposure_minutes > 0

        # Step 5: Create digital negative
        processor = ImageProcessor()
        image_path = tmp_path / "sample.png"
        sample_image.save(image_path)

        negative_result = processor.create_digital_negative(
            str(image_path),
            invert=True,
        )
        assert negative_result.image is not None

        # Step 6: Generate soft proof
        proofer = SoftProofer(ProofSettings())
        proof = proofer.proof(sample_image)
        assert proof.image is not None

        # Verify complete workflow succeeded
        total_metal = recipe.platinum_drops + recipe.palladium_drops
        print(f"Image analysis: {hist_result.stats.brightness:.2f} brightness")
        print(f"Zone recommendation: {dev_adjustment} development")
        print(f"Chemistry: {total_metal:.0f} drops metal")
        print(f"Exposure: {exposure.format_time()}")
