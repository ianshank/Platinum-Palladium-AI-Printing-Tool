"""
Tests for module imports and package structure.

Ensures all public modules can be imported correctly.
"""

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_import_core_models(self):
        """Import core models."""
        from ptpd_calibration.core.models import (
            CurveData,
            PatchData,
            DensityMeasurement,
            ExtractionResult,
            StepTabletResult,
            PaperProfile,
            CalibrationRecord,
        )
        assert CurveData is not None
        assert PatchData is not None
        assert DensityMeasurement is not None
        assert CalibrationRecord is not None

    def test_import_core_types(self):
        """Import core types."""
        from ptpd_calibration.core.types import (
            CurveType,
            ChemistryType,
            PaperSizing,
            ContrastAgent,
            DeveloperType,
            MeasurementUnit,
        )
        assert CurveType is not None
        assert ChemistryType is not None

    def test_import_core_package(self):
        """Import core package."""
        from ptpd_calibration import core
        assert hasattr(core, "CurveData")


class TestImagingImports:
    """Test imaging module imports."""

    def test_import_processor(self):
        """Import image processor."""
        from ptpd_calibration.imaging.processor import (
            ImageProcessor,
            ImageFormat,
            ColorMode,
            ExportSettings,
            ProcessingResult,
        )
        assert ImageProcessor is not None
        assert ImageFormat is not None
        assert ColorMode is not None

    def test_import_histogram(self):
        """Import histogram analyzer."""
        from ptpd_calibration.imaging.histogram import (
            HistogramAnalyzer,
            HistogramResult,
            HistogramStats,
        )
        assert HistogramAnalyzer is not None
        assert HistogramResult is not None

    def test_import_imaging_package(self):
        """Import imaging package."""
        from ptpd_calibration import imaging
        # Check exported names
        assert hasattr(imaging, "ImageProcessor")


class TestCurvesImports:
    """Test curves module imports."""

    def test_import_parser(self):
        """Import curve parser."""
        from ptpd_calibration.curves.parser import (
            QuadFileParser,
            ChannelCurve,
            QuadProfile,
        )
        assert QuadFileParser is not None
        assert ChannelCurve is not None

    def test_import_generator(self):
        """Import curve generator."""
        from ptpd_calibration.curves.generator import (
            CurveGenerator,
            TargetCurve,
            generate_linearization_curve,
        )
        assert CurveGenerator is not None
        assert TargetCurve is not None

    def test_import_export(self):
        """Import curve exporters."""
        from ptpd_calibration.curves.export import (
            QTRExporter,
            PiezographyExporter,
            CSVExporter,
            JSONExporter,
            save_curve,
            load_curve,
        )
        assert QTRExporter is not None
        assert save_curve is not None

    def test_import_modifier(self):
        """Import curve modifier."""
        from ptpd_calibration.curves.modifier import (
            CurveModifier,
            SmoothingMethod,
        )
        assert CurveModifier is not None
        assert SmoothingMethod is not None

    def test_import_linearization(self):
        """Import linearization."""
        from ptpd_calibration.curves.linearization import (
            AutoLinearizer,
            LinearizationResult,
            LinearizationConfig,
        )
        assert AutoLinearizer is not None
        assert LinearizationResult is not None

    def test_import_visualization(self):
        """Import visualization."""
        from ptpd_calibration.curves.visualization import (
            CurveVisualizer,
            VisualizationConfig,
        )
        assert CurveVisualizer is not None

    def test_import_curves_package(self):
        """Import curves package."""
        from ptpd_calibration import curves
        assert hasattr(curves, "CurveGenerator")


class TestDetectionImports:
    """Test detection module imports."""

    def test_import_detector(self):
        """Import step tablet detector."""
        from ptpd_calibration.detection.detector import (
            StepTabletDetector,
            DetectionResult,
        )
        assert StepTabletDetector is not None

    def test_import_extractor(self):
        """Import density extractor."""
        from ptpd_calibration.detection.extractor import (
            DensityExtractor,
            ExtractionSettings,
        )
        assert DensityExtractor is not None

    def test_import_scanner(self):
        """Import scanner calibration."""
        from ptpd_calibration.detection.scanner import (
            ScannerCalibration,
            ScannerProfile,
            ChannelCurve,
        )
        assert ScannerCalibration is not None
        assert ScannerProfile is not None

    def test_import_reader(self):
        """Import step tablet reader."""
        from ptpd_calibration.detection.reader import StepTabletReader
        assert StepTabletReader is not None


class TestProofingImports:
    """Test proofing module imports."""

    def test_import_simulation(self):
        """Import soft proofer."""
        from ptpd_calibration.proofing.simulation import (
            SoftProofer,
            ProofSettings,
            ProofResult,
        )
        assert SoftProofer is not None
        assert ProofSettings is not None


class TestChemistryImports:
    """Test chemistry module imports."""

    def test_import_calculator(self):
        """Import chemistry calculator."""
        from ptpd_calibration.chemistry.calculator import (
            ChemistryCalculator,
            MetalMix,
            ChemistryRecipe,
            PaperAbsorbency,
        )
        assert ChemistryCalculator is not None
        assert MetalMix is not None


class TestExposureImports:
    """Test exposure module imports."""

    def test_import_calculator(self):
        """Import exposure calculator."""
        from ptpd_calibration.exposure.calculator import (
            ExposureCalculator,
            LightSource,
            ExposureResult,
        )
        assert ExposureCalculator is not None
        assert LightSource is not None


class TestMLImports:
    """Test ML module imports."""

    def test_import_database(self):
        """Import calibration database."""
        from ptpd_calibration.ml.database import CalibrationDatabase
        assert CalibrationDatabase is not None

    def test_import_active_learning(self):
        """Import active learner."""
        from ptpd_calibration.ml.active_learning import ActiveLearner
        assert ActiveLearner is not None

    def test_import_transfer(self):
        """Import transfer learning."""
        from ptpd_calibration.ml.transfer import TransferLearner
        assert TransferLearner is not None


class TestZonesImports:
    """Test zones module imports."""

    def test_import_mapping(self):
        """Import zone mapper."""
        from ptpd_calibration.zones.mapping import (
            ZoneMapper,
            ZoneAnalysis,
        )
        assert ZoneMapper is not None


class TestAnalysisImports:
    """Test analysis module imports."""

    def test_import_wedge_analyzer(self):
        """Import wedge analyzer."""
        from ptpd_calibration.analysis.wedge_analyzer import (
            StepWedgeAnalyzer,
            WedgeAnalysisResult,
        )
        assert StepWedgeAnalyzer is not None


class TestLLMImports:
    """Test LLM module imports."""

    def test_import_client(self):
        """Import LLM client."""
        from ptpd_calibration.llm.client import LLMClient
        assert LLMClient is not None

    def test_import_assistant(self):
        """Import calibration assistant."""
        from ptpd_calibration.llm.assistant import CalibrationAssistant
        assert CalibrationAssistant is not None

    def test_import_prompts(self):
        """Import prompts."""
        from ptpd_calibration.llm.prompts import SYSTEM_PROMPT
        assert SYSTEM_PROMPT is not None


class TestAgentsImports:
    """Test agents module imports."""

    def test_import_agent(self):
        """Import calibration agent."""
        from ptpd_calibration.agents.agent import CalibrationAgent
        assert CalibrationAgent is not None

    def test_import_memory(self):
        """Import agent memory."""
        from ptpd_calibration.agents.memory import AgentMemory
        assert AgentMemory is not None

    def test_import_planning(self):
        """Import agent planning."""
        from ptpd_calibration.agents.planning import (
            Planner,
            Plan,
            PlanStep,
            PlanStatus,
        )
        assert Planner is not None
        assert Plan is not None


class TestConfigImports:
    """Test config imports."""

    def test_import_config(self):
        """Import configuration."""
        from ptpd_calibration.config import (
            Settings,
            CurveSettings,
            DetectionSettings,
            get_settings,
        )
        assert Settings is not None
        assert get_settings is not None


class TestTopLevelImports:
    """Test top-level package imports."""

    def test_import_main_package(self):
        """Import main package."""
        import ptpd_calibration
        assert ptpd_calibration is not None

    def test_package_has_version(self):
        """Package should have version."""
        import ptpd_calibration
        assert hasattr(ptpd_calibration, "__version__")

    def test_package_exports(self):
        """Package should export key classes."""
        from ptpd_calibration import (
            CurveData,
            CurveGenerator,
            Settings,
            get_settings,
        )
        assert CurveData is not None
        assert CurveGenerator is not None
        assert Settings is not None


class TestOptionalImports:
    """Test optional dependencies."""

    def test_tifffile_available(self):
        """Check if tifffile is available for 16-bit TIFF."""
        try:
            import tifffile
            assert tifffile is not None
        except ImportError:
            pytest.skip("tifffile not installed")

    def test_processor_tifffile_flag(self):
        """Check processor has tifffile flag."""
        from ptpd_calibration.imaging.processor import HAS_TIFFFILE
        # Just verify the flag exists (True or False)
        assert isinstance(HAS_TIFFFILE, bool)
