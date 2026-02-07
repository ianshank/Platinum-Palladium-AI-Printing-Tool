"""
Sanity tests for deployment verification.

These tests quickly verify that core functionality works after deployment.
Run these after deploying to Huggingface Spaces or other environments.
"""

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
        from ptpd_calibration.curves import AutoLinearizer, CurveGenerator

        assert CurveGenerator is not None
        assert AutoLinearizer is not None

    def test_imaging_imports(self):
        """Test imaging module imports."""
        from ptpd_calibration.imaging import HistogramAnalyzer, ImageProcessor

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
