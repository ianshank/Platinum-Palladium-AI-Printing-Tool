"""
PTPD Calibration - AI-powered calibration system for platinum/palladium printing.

This package provides comprehensive tools for calibrating digital negatives
for Pt/Pd printing, including:

- Step tablet detection and density extraction
- Curve generation for QTR and Piezography
- ML-based prediction and active learning
- AI assistant for calibration guidance
- Agentic system for autonomous calibration tasks
- Advanced features: alternative process simulation, negative blending,
  QR metadata generation, historic style transfer, and print comparison
- Platinum/Palladium AI for intelligent printing analysis
- Split-grade printing simulation
- Recipe management and workflow automation
- Enhanced technical calculations
- Data management with cloud sync and version control
- Hardware integrations (spectrophotometer, weather, printers, ICC)
- Quality assurance and validation
- Educational tutorials and glossary
- Performance monitoring and profiling
"""

from contextlib import suppress

__version__ = "1.1.0"

# Core models
# Configuration
from ptpd_calibration.config import (
    ExportFormat,
    InterpolationMethod,
    LLMProvider,
    Settings,
    TabletType,
    configure,
    get_settings,
)
from ptpd_calibration.core.models import (
    CalibrationRecord,
    CurveData,
    DensityMeasurement,
    ExtractionResult,
    PaperProfile,
    PatchData,
    StepTabletResult,
)
from ptpd_calibration.core.types import (
    ChemistryType,
    ContrastAgent,
    CurveType,
    DeveloperType,
    MeasurementUnit,
    PaperSizing,
)

# Curves
from ptpd_calibration.curves import (
    CurveAnalyzer,
    CurveExporter,
    CurveGenerator,
    PiezographyExporter,
    QTRExporter,
    TargetCurve,
    generate_linearization_curve,
    load_curve,
    save_curve,
)

# Detection
from ptpd_calibration.detection import (
    DensityExtractor,
    ScannerCalibration,
    StepTabletDetector,
    StepTabletReader,
)

# ML (optional - requires scikit-learn)
with suppress(ImportError):
    from ptpd_calibration.ml import (
        ActiveLearner,
        CalibrationDatabase,
        CurvePredictor,
        TransferLearner,
    )

# LLM (optional - requires anthropic/openai)
with suppress(ImportError):
    from ptpd_calibration.llm import (
        CalibrationAssistant,
        create_assistant,
    )

# Agents (optional - requires LLM)
with suppress(ImportError):
    from ptpd_calibration.agents import (
        AgentMemory,
        CalibrationAgent,
        Plan,
        Planner,
        Tool,
        ToolRegistry,
        create_agent,
    )

# Advanced features
with suppress(ImportError):
    from ptpd_calibration.advanced import (
        AlternativeProcessParams,
        AlternativeProcessSimulator,
        BlendMode,
        HistoricStyle,
        NegativeBlender,
        PrintComparison,
        PrintMetadata,
        QRMetadataGenerator,
        StyleParameters,
        StyleTransfer,
    )

# Platinum/Palladium AI
with suppress(ImportError):
    from ptpd_calibration.ai import (
        ChemistryRecommendation,
        DigitalNegativeResult,
        ExposurePrediction,
        PlatinumPalladiumAI,
        PrintQualityAnalysis,
        TonalityAnalysisResult,
        WorkflowOptimization,
    )

# Split-grade printing
with suppress(ImportError):
    from ptpd_calibration.imaging.split_grade import (
        SplitGradeSettings,
        SplitGradeSimulator,
        TonalCurveAdjuster,
    )

# Recipe management and workflow
with suppress(ImportError):
    from ptpd_calibration.workflow import (
        PrintRecipe,
        RecipeDatabase,
        RecipeManager,
        WorkflowAutomation,
    )

# Enhanced calculations
with suppress(ImportError):
    from ptpd_calibration.calculations import (
        CoatingVolumeCalculator,
        CostCalculator,
        DilutionCalculator,
        EnvironmentalCompensation,
        UVExposureCalculator,
    )

# Data management
with suppress(ImportError):
    from ptpd_calibration.data import (
        DataExporter,
        DataImporter,
        LocalStorageProvider,
        PrintDatabase,
        PrintRecord,
        SyncManager,
        VersionController,
    )

# Hardware integrations
with suppress(ImportError):
    from ptpd_calibration.integrations import (
        CanonDriver,
        EpsonDriver,
        ICCProfileManager,
        OpenWeatherMapProvider,
        PrinterInterface,
        SpectrophotometerInterface,
        WeatherProvider,
        XRiteIntegration,
    )

# Quality assurance
with suppress(ImportError):
    from ptpd_calibration.qa import (
        AlertSystem,
        ChemistryFreshnessTracker,
        NegativeDensityValidator,
        PaperHumidityChecker,
        QualityReport,
        UVLightMeterIntegration,
    )

# Education
with suppress(ImportError):
    from ptpd_calibration.education import (
        Glossary,
        TipsManager,
        TutorialManager,
    )

# Performance monitoring
with suppress(ImportError):
    from ptpd_calibration.monitoring import (
        APIPerformanceTracker,
        CacheManager,
        ImageProcessingProfiler,
        PerformanceMonitor,
        PerformanceReport,
        ResourceMonitor,
    )

__all__ = [
    # Version
    "__version__",
    # Core models
    "CalibrationRecord",
    "CurveData",
    "DensityMeasurement",
    "ExtractionResult",
    "PaperProfile",
    "PatchData",
    "StepTabletResult",
    # Types
    "ChemistryType",
    "ContrastAgent",
    "CurveType",
    "DeveloperType",
    "MeasurementUnit",
    "PaperSizing",
    # Config
    "Settings",
    "get_settings",
    "configure",
    "TabletType",
    "ExportFormat",
    "InterpolationMethod",
    "LLMProvider",
    # Detection
    "StepTabletDetector",
    "DensityExtractor",
    "StepTabletReader",
    "ScannerCalibration",
    # Curves
    "CurveGenerator",
    "TargetCurve",
    "generate_linearization_curve",
    "CurveExporter",
    "QTRExporter",
    "PiezographyExporter",
    "save_curve",
    "load_curve",
    "CurveAnalyzer",
    # ML
    "CalibrationDatabase",
    "CurvePredictor",
    "ActiveLearner",
    "TransferLearner",
    # LLM
    "CalibrationAssistant",
    "create_assistant",
    # Agents
    "CalibrationAgent",
    "create_agent",
    "Tool",
    "ToolRegistry",
    "AgentMemory",
    "Plan",
    "Planner",
    # Advanced features
    "AlternativeProcessSimulator",
    "NegativeBlender",
    "QRMetadataGenerator",
    "StyleTransfer",
    "PrintComparison",
    "AlternativeProcessParams",
    "StyleParameters",
    "PrintMetadata",
    "BlendMode",
    "HistoricStyle",
    # Platinum/Palladium AI
    "PlatinumPalladiumAI",
    "TonalityAnalysisResult",
    "ExposurePrediction",
    "ChemistryRecommendation",
    "DigitalNegativeResult",
    "PrintQualityAnalysis",
    "WorkflowOptimization",
    # Split-grade printing
    "SplitGradeSimulator",
    "SplitGradeSettings",
    "TonalCurveAdjuster",
    # Recipe management and workflow
    "PrintRecipe",
    "RecipeManager",
    "RecipeDatabase",
    "WorkflowAutomation",
    # Enhanced calculations
    "UVExposureCalculator",
    "CoatingVolumeCalculator",
    "CostCalculator",
    "DilutionCalculator",
    "EnvironmentalCompensation",
    # Data management
    "PrintDatabase",
    "PrintRecord",
    "DataExporter",
    "DataImporter",
    "SyncManager",
    "LocalStorageProvider",
    "VersionController",
    # Hardware integrations
    "SpectrophotometerInterface",
    "XRiteIntegration",
    "WeatherProvider",
    "OpenWeatherMapProvider",
    "PrinterInterface",
    "EpsonDriver",
    "CanonDriver",
    "ICCProfileManager",
    # Quality assurance
    "NegativeDensityValidator",
    "ChemistryFreshnessTracker",
    "PaperHumidityChecker",
    "UVLightMeterIntegration",
    "QualityReport",
    "AlertSystem",
    # Education
    "TutorialManager",
    "Glossary",
    "TipsManager",
    # Performance monitoring
    "PerformanceMonitor",
    "ImageProcessingProfiler",
    "APIPerformanceTracker",
    "CacheManager",
    "ResourceMonitor",
    "PerformanceReport",
]
