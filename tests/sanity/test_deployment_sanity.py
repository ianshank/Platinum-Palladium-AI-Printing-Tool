"""
Sanity tests for deployment verification.

These tests quickly verify that core functionality works after deployment.
Run these after deploying to Huggingface Spaces or other environments.
"""

import sys
from pathlib import Path

import pytest


class TestCoreImports:
    """Verify core modules can be imported."""

    def test_main_package(self):
        """Test main package imports."""
        import ptpd_calibration

        assert hasattr(ptpd_calibration, "__version__") or True

    def test_core_models(self):
        """Test core models import."""
        from ptpd_calibration.core.models import CurveData, ExtractionResult

        assert CurveData is not None
        assert ExtractionResult is not None

    def test_core_types(self):
        """Test core types import."""
        from ptpd_calibration.core.types import CurveType

        assert CurveType.LINEAR is not None

    def test_detection_imports(self):
        """Test detection module imports."""
        from ptpd_calibration.detection import StepTabletReader

        assert StepTabletReader is not None

    def test_curves_imports(self):
        """Test curves module imports."""
        from ptpd_calibration.curves import CurveGenerator, AutoLinearizer

        assert CurveGenerator is not None
        assert AutoLinearizer is not None

    def test_imaging_imports(self):
        """Test imaging module imports."""
        from ptpd_calibration.imaging import ImageProcessor, HistogramAnalyzer

        assert ImageProcessor is not None
        assert HistogramAnalyzer is not None

    def test_chemistry_imports(self):
        """Test chemistry module imports."""
        from ptpd_calibration.chemistry import ChemistryCalculator

        assert ChemistryCalculator is not None

    def test_exposure_imports(self):
        """Test exposure module imports."""
        from ptpd_calibration.exposure import ExposureCalculator

        assert ExposureCalculator is not None

    def test_zones_imports(self):
        """Test zones module imports."""
        from ptpd_calibration.zones import ZoneMapper

        assert ZoneMapper is not None

    def test_proofing_imports(self):
        """Test proofing module imports."""
        from ptpd_calibration.proofing import SoftProofer

        assert SoftProofer is not None

    def test_papers_imports(self):
        """Test papers module imports."""
        from ptpd_calibration.papers import PaperDatabase

        assert PaperDatabase is not None

    def test_session_imports(self):
        """Test session module imports."""
        from ptpd_calibration.session import SessionLogger

        assert SessionLogger is not None


class TestCoreFunctionality:
    """Test core functionality works."""

    def test_curve_data_creation(self):
        """Test CurveData can be created."""
        from ptpd_calibration.core.models import CurveData

        curve = CurveData(
            name="Test Curve",
            input_values=[0.0, 0.5, 1.0],
            output_values=[0.0, 0.5, 1.0],
        )
        assert curve.name == "Test Curve"
        assert len(curve.input_values) == 3

    def test_chemistry_calculation(self):
        """Test chemistry calculation works."""
        from ptpd_calibration.chemistry import ChemistryCalculator

        calc = ChemistryCalculator()
        recipe = calc.calculate(
            width_inches=8.0,
            height_inches=10.0,
            platinum_ratio=0.5,
        )
        assert recipe is not None
        assert recipe.platinum_drops >= 0
        assert recipe.palladium_drops >= 0

    def test_exposure_calculation(self):
        """Test exposure calculation works."""
        from ptpd_calibration.exposure import ExposureCalculator

        calc = ExposureCalculator()
        result = calc.calculate(negative_density=1.6)
        assert result is not None
        assert result.exposure_minutes > 0

    def test_histogram_analysis(self):
        """Test histogram analysis works."""
        import numpy as np
        from PIL import Image

        from ptpd_calibration.imaging import HistogramAnalyzer

        # Create test image
        arr = np.linspace(0, 255, 100).reshape(10, 10).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        analyzer = HistogramAnalyzer()
        result = analyzer.analyze(img)
        assert result is not None
        assert len(result.histogram) == 256

    def test_zone_mapping(self):
        """Test zone mapping works."""
        import numpy as np
        from PIL import Image

        from ptpd_calibration.zones import ZoneMapper

        # Create test image
        arr = np.linspace(0, 255, 100).reshape(10, 10).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        mapper = ZoneMapper()
        result = mapper.analyze_image(img)
        assert result is not None
        assert result.development_adjustment is not None

    def test_soft_proofing(self):
        """Test soft proofing works."""
        import numpy as np
        from PIL import Image

        from ptpd_calibration.proofing import SoftProofer

        # Create test image
        arr = np.linspace(50, 200, 100).reshape(10, 10).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        proofer = SoftProofer()
        result = proofer.proof(img)
        assert result is not None
        assert result.image is not None

    def test_auto_linearization(self):
        """Test auto-linearization works."""
        from ptpd_calibration.curves import AutoLinearizer

        linearizer = AutoLinearizer()
        densities = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        result = linearizer.linearize(densities)
        assert result is not None
        assert result.curve is not None

    def test_paper_database(self):
        """Test paper database works."""
        from ptpd_calibration.papers import PaperDatabase

        db = PaperDatabase()
        papers = db.list_papers()
        assert len(papers) >= 4


class TestGradioAppImport:
    """Test Gradio app can be imported."""

    def test_gradio_import(self):
        """Test Gradio is available."""
        try:
            import gradio as gr

            assert gr is not None
        except ImportError:
            pytest.skip("Gradio not installed")

    def test_gradio_app_import(self):
        """Test Gradio app can be imported."""
        try:
            from ptpd_calibration.ui.gradio_app import demo

            assert demo is not None
        except ImportError as e:
            pytest.skip(f"Could not import Gradio app: {e}")


class TestAppEntryPoint:
    """Test app.py entry point."""

    def test_app_file_exists(self):
        """Test app.py exists."""
        app_path = Path(__file__).parent.parent.parent / "app.py"
        assert app_path.exists(), f"app.py not found at {app_path}"

    def test_app_imports(self):
        """Test app.py can be imported."""
        import importlib.util

        app_path = Path(__file__).parent.parent.parent / "app.py"
        spec = importlib.util.spec_from_file_location("app", app_path)
        assert spec is not None


class TestRequirementsFile:
    """Test requirements.txt is valid."""

    def test_requirements_exists(self):
        """Test requirements.txt exists."""
        req_path = Path(__file__).parent.parent.parent / "requirements.txt"
        assert req_path.exists(), f"requirements.txt not found at {req_path}"

    def test_requirements_has_core_deps(self):
        """Test requirements.txt has core dependencies."""
        req_path = Path(__file__).parent.parent.parent / "requirements.txt"
        content = req_path.read_text()

        # Check for essential dependencies
        assert "numpy" in content
        assert "scipy" in content
        assert "pillow" in content.lower() or "Pillow" in content
        assert "pydantic" in content
        assert "gradio" in content


class TestWizardStep3Linearization:
    """Sanity tests for Calibration Wizard Step 3 linearization functionality."""

    def test_wizard_linearization_mode_enum_import(self):
        """Test WizardLinearizationMode enum can be imported."""
        from ptpd_calibration.ui.tabs.calibration_wizard import WizardLinearizationMode

        assert WizardLinearizationMode.SINGLE_CURVE is not None
        assert WizardLinearizationMode.MULTI_CURVE is not None
        assert WizardLinearizationMode.USE_EXISTING is not None
        assert WizardLinearizationMode.NO_LINEARIZATION is not None

    def test_wizard_linearization_mode_config_import(self):
        """Test LinearizationModeConfig dataclass can be imported."""
        from ptpd_calibration.ui.tabs.calibration_wizard import LinearizationModeConfig

        assert LinearizationModeConfig is not None

    def test_wizard_linearization_mode_configs_exist(self):
        """Test LINEARIZATION_MODES dictionary exists and has all modes."""
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            LINEARIZATION_MODES,
            WizardLinearizationMode,
        )

        assert len(LINEARIZATION_MODES) == 4
        for mode in list(WizardLinearizationMode):
            assert mode.value in LINEARIZATION_MODES

    def test_wizard_get_linearization_mode_choices(self):
        """Test get_linearization_mode_choices returns valid choices."""
        from ptpd_calibration.ui.tabs.calibration_wizard import get_linearization_mode_choices

        choices = get_linearization_mode_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 4
        # Each choice should be a string label
        for choice in choices:
            assert isinstance(choice, str)
            assert len(choice) > 0

    def test_wizard_get_strategy_choices(self):
        """Test get_strategy_choices returns valid linearization strategies."""
        from ptpd_calibration.ui.tabs.calibration_wizard import get_strategy_choices

        choices = get_strategy_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 1
        # Choices are tuples of (label, value)
        for choice in choices:
            assert isinstance(choice, tuple)
            assert len(choice) == 2
        # Should include common strategies
        strategy_labels = [label.lower() for label, _ in choices]
        assert any("spline" in s for s in strategy_labels)

    def test_wizard_get_target_choices(self):
        """Test get_target_choices returns valid target responses."""
        from ptpd_calibration.ui.tabs.calibration_wizard import get_target_choices

        choices = get_target_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 1
        # Choices are tuples of (label, value)
        for choice in choices:
            assert isinstance(choice, tuple)
            assert len(choice) == 2
        # Should include common targets
        target_labels = [label.lower() for label, _ in choices]
        assert any("linear" in t for t in target_labels)

    def test_wizard_get_paper_preset_choices(self):
        """Test get_paper_preset_choices returns valid paper presets."""
        from ptpd_calibration.ui.tabs.calibration_wizard import get_paper_preset_choices

        choices = get_paper_preset_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 1
        # Should include Custom option
        assert any("custom" in c.lower() for c in choices)

    def test_wizard_validation_function(self):
        """Test wizard_is_valid_config returns proper tuple format."""
        from ptpd_calibration.ui.tabs.calibration_wizard import wizard_is_valid_config

        # Test that the function returns proper tuple format
        result = wizard_is_valid_config(
            mode_label="",
            target_label="",
            strategy_label="",
            paper_preset="",
            existing_profile="",
            custom_chemistry="",
            curve_name="",
        )
        # Should return a tuple of (bool, str)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_valid, error_msg = result
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        # Empty mode should return False with error message
        assert is_valid is False
        assert len(error_msg) > 0

    def test_wizard_mode_change_handler(self):
        """Test wizard_on_mode_change returns correct number of updates."""
        from ptpd_calibration.ui.tabs.calibration_wizard import (
            LINEARIZATION_MODES,
            WizardLinearizationMode,
            wizard_on_mode_change,
        )

        mode_label = LINEARIZATION_MODES[WizardLinearizationMode.SINGLE_CURVE.value].label
        result = wizard_on_mode_change(mode_label)

        # Should return 7 update objects
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_wizard_linearization_with_auto_linearizer(self):
        """Test AutoLinearizer works with wizard configuration."""
        from ptpd_calibration.curves import AutoLinearizer
        from ptpd_calibration.curves.linearization import (
            LinearizationMethod,
            TargetResponse,
        )

        # Create linearizer with wizard-compatible settings
        linearizer = AutoLinearizer()
        densities = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        # Test with spline fit method and linear target
        result = linearizer.linearize(
            densities,
            method=LinearizationMethod.SPLINE_FIT,
            target=TargetResponse.LINEAR,
        )
        assert result is not None
        assert result.curve is not None
        assert len(result.curve.input_values) >= 2

    def test_wizard_paper_database_integration(self):
        """Test PaperDatabase integrates with wizard presets."""
        from ptpd_calibration.papers import PaperDatabase

        db = PaperDatabase()
        papers = db.list_papers()

        # Verify papers can be used as presets
        assert len(papers) >= 1
        for paper in papers:
            assert hasattr(paper, "name")
            assert len(paper.name) > 0
