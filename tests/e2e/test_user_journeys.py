"""
End-to-end user journey tests for the Pt/Pd Calibration Studio.

These tests simulate complete user workflows without requiring browser automation.
They test the full integration of modules as a user would experience them.
"""

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
        from ptpd_calibration.analysis import StepWedgeAnalyzer
        from ptpd_calibration.curves import CurveGenerator, save_curve
        from ptpd_calibration.detection import StepTabletReader

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

    def test_import_quad_journey(self, real_quad_path):
        """
        Journey: Import existing .quad file → Validate → Check Channels
        """
        from ptpd_calibration.curves import load_quad_file

        # Step 1: Load the real-world quad file
        profile = load_quad_file(real_quad_path)

        # Step 2: Validate basic metadata
        assert profile is not None
        # The file has "Platinum-Palladium" in comments
        assert any("Platinum-Palladium" in c for c in profile.comments)

        # Step 3: Check Channels
        # The file has K, C, M, Y, LC, LM, LK, LLK, V, MK headers
        # Check for active channels. Based on file content, K seems active.
        assert "K" in profile.channels
        k_curve = profile.channels["K"]
        assert len(k_curve.values) == 256
        assert k_curve.values[-1] > 0  # Should have some density

        # Step 4: Check specific metadata if parsed (e.g. ink load)
        # The parser might extract these if implemented, otherwise check comments
        assert len(profile.comments) > 0


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

    def test_digital_negative_workflow(self, sample_image, sample_curve, real_quad_path, tmp_path):
        """
        Workflow: Load Image → Preview → Apply Curve → Export Negative
        """
        from ptpd_calibration.curves import load_quad_file
        from ptpd_calibration.imaging import (
            ExportSettings,
            ImageFormat,
            ImageProcessor,
        )

        # Step 1: Save sample image
        image_path = tmp_path / "sample.png"
        sample_image.save(image_path)

        # Step 2: Initialize processor
        processor = ImageProcessor()

        # Step 3: Load real curve if available, otherwise use sample
        try:
            profile = load_quad_file(real_quad_path)
            # Use K channel curve
            curve = profile.to_curve_data("K")
        except Exception:
            curve = sample_curve

        # Step 4: Preview curve effect
        original, processed = processor.preview_curve_effect(
            str(image_path),
            curve,
        )

        assert original is not None
        assert processed is not None

        # Step 5: Create digital negative
        result = processor.create_digital_negative(
            str(image_path),
            curve=curve,
            invert=True,
        )

        assert result is not None
        assert result.image is not None

        # Step 6: Export
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
        for _size_name, (w, h) in ChemistryCalculator.get_standard_sizes().items():
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

    def test_zone_analysis_workflow(self, sample_gradient, tmp_path):  # noqa: ARG002
        """
        Workflow: Upload Image → Analyze Zones → Get Recommendations
        """
        from ptpd_calibration.zones import (
            Zone,
            ZoneMapper,
            ZoneMapping,
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

    def test_soft_proofing_workflow(self, sample_image, real_quad_path):
        """
        Workflow: Select Paper → Adjust Settings → Generate Proof
        """
        from ptpd_calibration.curves import load_quad_file
        from ptpd_calibration.proofing import (
            PaperSimulation,
            ProofSettings,
            SoftProofer,
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

        # Step 2: Custom settings with Real Curve
        # Load real curve to get some characteristics if possible,
        # but SoftProofer mainly uses density/color.
        # We'll just ensure we can read it and potentially use it in a more advanced test.
        profile = load_quad_file(real_quad_path)
        assert profile is not None

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
        for method in list(LinearizationMethod):
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
            PrintRecord,
            SessionLogger,
        )
        from ptpd_calibration.session.logger import ChemistryUsed, PrintResult

        # Step 1: Create logger
        logger = SessionLogger(storage_dir=tmp_path / "sessions")

        # Step 2: Start a session
        logger.start_session("Test Session")

        # Step 3: Log some prints
        for i in range(3):
            record = PrintRecord(
                image_name=f"Test Image {i + 1}",
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
        from ptpd_calibration.chemistry import ChemistryCalculator
        from ptpd_calibration.exposure import ExposureCalculator, ExposureSettings
        from ptpd_calibration.imaging import HistogramAnalyzer, ImageProcessor
        from ptpd_calibration.proofing import ProofSettings, SoftProofer
        from ptpd_calibration.zones import ZoneMapper

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


class TestWizardStep3Linearization:
    """Test Calibration Wizard Step 3 linearization configuration journey."""

    def test_wizard_step3_mode_configuration(self):
        """
        Journey: Select Mode → Configure Options → Validate Configuration
        """
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            get_linearization_mode_choices,
            get_mode_by_label,
            get_paper_preset_choices,
            get_strategy_labels,
            get_target_labels,
            wizard_is_valid_config,
            wizard_on_config_change,
            wizard_on_mode_change,
        )

        # Step 1: Get available modes
        modes = get_linearization_mode_choices()
        assert len(modes) == 4
        assert "Single-curve linearization (recommended)" in modes

        # Step 2: Select single-curve mode
        mode_label = "Single-curve linearization (recommended)"
        mode_config = get_mode_by_label(mode_label)
        assert mode_config is not None
        assert mode_config.requires_target is True
        assert mode_config.requires_strategy is True

        # Step 3: Mode change updates visibility
        visibility_updates = wizard_on_mode_change(mode_label)
        assert len(visibility_updates) == 7
        assert visibility_updates[0]["visible"] is True  # Target visible
        assert visibility_updates[1]["visible"] is True  # Strategy visible

        # Step 4: Configure valid options
        target_label = get_target_labels()[0]  # "Even tonal steps (linear)"
        strategy_label = get_strategy_labels()[0]  # "Smooth spline (recommended)"
        paper_preset = get_paper_preset_choices()[0]  # First paper

        # Step 5: Validate configuration
        is_valid, msg = wizard_is_valid_config(
            mode_label=mode_label,
            target_label=target_label,
            strategy_label=strategy_label,
            paper_preset=paper_preset,
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Wizard Curve",
        )
        assert is_valid is True
        assert msg == ""

        # Step 6: Config change enables button
        button_update, validation_msg = wizard_on_config_change(
            mode_label=mode_label,
            target_label=target_label,
            strategy_label=strategy_label,
            paper_preset=paper_preset,
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Wizard Curve",
        )
        assert button_update["interactive"] is True

    def test_wizard_step3_no_linearization_mode(self):
        """
        Journey: Select No Linearization → Generate Identity Curve
        """
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            WizardLinearizationMode,
            get_mode_by_label,
            get_mode_value_by_label,
            wizard_is_valid_config,
            wizard_on_mode_change,
        )

        # Step 1: Select no linearization mode
        mode_label = "No linearization (straight curve)"
        mode_config = get_mode_by_label(mode_label)

        assert mode_config is not None
        assert mode_config.requires_target is False
        assert mode_config.requires_strategy is False
        assert mode_config.requires_paper_preset is True

        # Step 2: Mode change hides target/strategy
        visibility_updates = wizard_on_mode_change(mode_label)
        assert visibility_updates[0]["visible"] is False  # Target hidden
        assert visibility_updates[1]["visible"] is False  # Strategy hidden
        assert visibility_updates[2]["visible"] is True  # Paper visible

        # Step 3: Validate configuration
        is_valid, msg = wizard_is_valid_config(
            mode_label=mode_label,
            target_label="",  # Not required
            strategy_label="",  # Not required
            paper_preset="Arches Platine",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Identity Curve",
        )
        assert is_valid is True

        # Step 4: Verify mode value
        mode_value = get_mode_value_by_label(mode_label)
        assert mode_value == WizardLinearizationMode.NO_LINEARIZATION.value

    def test_wizard_step3_use_existing_mode(self):
        """
        Journey: Select Use Existing → Select Profile → Skip Generation
        """
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            get_mode_by_label,
            wizard_is_valid_config,
            wizard_on_mode_change,
        )

        # Step 1: Select use existing mode
        mode_label = "Use existing profile"
        mode_config = get_mode_by_label(mode_label)

        assert mode_config is not None
        assert mode_config.requires_existing_profile is True
        assert mode_config.requires_target is False
        assert mode_config.requires_strategy is False

        # Step 2: Mode change shows existing profile dropdown
        visibility_updates = wizard_on_mode_change(mode_label)
        assert visibility_updates[0]["visible"] is False  # Target hidden
        assert visibility_updates[1]["visible"] is False  # Strategy hidden
        assert visibility_updates[2]["visible"] is False  # Paper hidden
        assert visibility_updates[3]["visible"] is True  # Existing profile visible

        # Step 3: Invalid without profile
        is_valid, msg = wizard_is_valid_config(
            mode_label=mode_label,
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test",
        )
        assert is_valid is False
        assert "profile" in msg.lower()

        # Step 4: Valid with profile selected
        is_valid, msg = wizard_is_valid_config(
            mode_label=mode_label,
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile="My Existing Curve",
            custom_chemistry="",
            curve_name="Test",
        )
        assert is_valid is True

    def test_wizard_step3_custom_paper_workflow(self):
        """
        Journey: Select Custom Paper → Enter Chemistry Notes → Validate
        """
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            get_paper_preset_choices,
            wizard_is_valid_config,
            wizard_on_paper_change,
        )

        # Step 1: Select custom paper
        paper_preset = "Other / custom"
        assert paper_preset in get_paper_preset_choices()

        # Step 2: Paper change shows chemistry input
        custom_visible, notes_update = wizard_on_paper_change(paper_preset)
        assert custom_visible["visible"] is True
        assert custom_visible["interactive"] is True
        assert notes_update["value"] == ""

        # Step 3: Invalid without chemistry notes
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset=paper_preset,
            existing_profile=None,
            custom_chemistry="",
            curve_name="Test Curve",
        )
        assert is_valid is False
        assert "chemistry" in msg.lower()

        # Step 4: Valid with chemistry notes
        is_valid, msg = wizard_is_valid_config(
            mode_label="Single-curve linearization (recommended)",
            target_label="Even tonal steps (linear)",
            strategy_label="Smooth spline (recommended)",
            paper_preset=paper_preset,
            existing_profile=None,
            custom_chemistry="50/50 Pt/Pd, ammonium citrate, 5 drops Na2",
            curve_name="Test Curve",
        )
        assert is_valid is True

    def test_wizard_step3_linearization_integration(self):
        """
        Journey: Configure Wizard → Generate Curve with AutoLinearizer
        """
        from ptpd_calibration.curves.linearization import (
            AutoLinearizer,
            LinearizationConfig,
            LinearizationMethod,
            TargetResponse,
        )
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            get_strategy_value_by_label,
            get_target_value_by_label,
        )

        # Step 1: Get configuration values from UI labels
        strategy_label = "Smooth spline (recommended)"
        target_label = "Even tonal steps (linear)"

        strategy_value = get_strategy_value_by_label(strategy_label)
        target_value = get_target_value_by_label(target_label)

        assert strategy_value == LinearizationMethod.SPLINE_FIT.value
        assert target_value == TargetResponse.LINEAR.value

        # Step 2: Create linearizer config using UI-derived values
        method_enum = LinearizationMethod(strategy_value)
        target_enum = TargetResponse(target_value)

        config = LinearizationConfig(
            method=method_enum,
            target=target_enum,
            output_points=256,
            smoothing=0.1,
        )

        # Step 3: Generate curve with sample densities
        densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]
        linearizer = AutoLinearizer(config)

        result = linearizer.linearize(
            measured_densities=densities,
            curve_name="Wizard Generated Curve",
            target=target_enum,
            method=method_enum,
        )

        # Step 4: Verify curve generation
        assert result.curve is not None
        assert result.curve.name == "Wizard Generated Curve"
        assert result.method_used == method_enum
        assert result.target_response == target_enum
        assert len(result.curve.output_values) > 0
        assert result.residual_error >= 0

    def test_wizard_step3_all_strategies(self):
        """
        Journey: Test all strategy options generate valid curves
        """
        from ptpd_calibration.curves.linearization import (
            AutoLinearizer,
            LinearizationMethod,
        )
        from ptpd_calibration.ui.tabs.calibration_wizard import get_strategy_choices

        densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

        # Test each strategy from UI choices
        for label, value in get_strategy_choices():
            method = LinearizationMethod(value)
            linearizer = AutoLinearizer()

            result = linearizer.linearize(
                measured_densities=densities,
                curve_name=f"Test {label}",
                method=method,
            )

            assert result.curve is not None, f"Strategy '{label}' failed to generate curve"
            assert len(result.curve.output_values) > 0, f"Strategy '{label}' generated empty curve"

    def test_wizard_step3_all_targets(self):
        """
        Journey: Test all target options generate valid curves
        """
        from ptpd_calibration.curves.linearization import (
            AutoLinearizer,
            TargetResponse,
        )
        from ptpd_calibration.ui.tabs.calibration_wizard import get_target_choices

        densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

        # Test each target from UI choices
        for label, value in get_target_choices():
            target = TargetResponse(value)
            linearizer = AutoLinearizer()

            result = linearizer.linearize(
                measured_densities=densities,
                curve_name=f"Test {label}",
                target=target,
            )

            assert result.curve is not None, f"Target '{label}' failed to generate curve"
            assert result.target_response == target, f"Target '{label}' mismatch"
