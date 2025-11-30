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

__version__ = "1.1.0"

# Core models
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

# Configuration
from ptpd_calibration.config import (
    Settings,
    get_settings,
    configure,
    TabletType,
    ExportFormat,
    InterpolationMethod,
    LLMProvider,
)

# Detection
from ptpd_calibration.detection import (
    StepTabletDetector,
    DensityExtractor,
    StepTabletReader,
    ScannerCalibration,
)

# Curves
from ptpd_calibration.curves import (
    CurveGenerator,
    TargetCurve,
    generate_linearization_curve,
    CurveExporter,
    QTRExporter,
    PiezographyExporter,
    save_curve,
    load_curve,
    CurveAnalyzer,
)

# ML (optional - requires scikit-learn)
try:
    from ptpd_calibration.ml import (
        CalibrationDatabase,
        CurvePredictor,
        ActiveLearner,
        TransferLearner,
    )
except ImportError:
    pass

# LLM (optional - requires anthropic/openai)
try:
    from ptpd_calibration.llm import (
        CalibrationAssistant,
        create_assistant,
    )
except ImportError:
    pass

# Agents (optional - requires LLM)
try:
    from ptpd_calibration.agents import (
        CalibrationAgent,
        create_agent,
        Tool,
        ToolRegistry,
        AgentMemory,
        Plan,
        Planner,
    )
except ImportError:
    pass

# Advanced features
try:
    from ptpd_calibration.advanced import (
        AlternativeProcessSimulator,
        NegativeBlender,
        QRMetadataGenerator,
        StyleTransfer,
        PrintComparison,
        AlternativeProcessParams,
        StyleParameters,
        PrintMetadata,
        BlendMode,
        HistoricStyle,
    )
except ImportError:
    pass

# Platinum/Palladium AI
try:
    from ptpd_calibration.ai import (
        PlatinumPalladiumAI,
        TonalityAnalysisResult,
        ExposurePrediction,
        ChemistrySuggestion,
        DigitalNegativeResult,
        PrintQualityAnalysis,
        WorkflowOptimization,
    )
except ImportError:
    pass

# Split-grade printing
try:
    from ptpd_calibration.imaging.split_grade import (
        SplitGradeSimulator,
        SplitGradeSettings,
        TonalCurveAdjuster,
    )
except ImportError:
    pass

# Recipe management and workflow
try:
    from ptpd_calibration.workflow import (
        PrintRecipe,
        RecipeManager,
        RecipeDatabase,
        WorkflowAutomation,
    )
except ImportError:
    pass

# Enhanced calculations
try:
    from ptpd_calibration.calculations import (
        UVExposureCalculator,
        CoatingVolumeCalculator,
        CostCalculator,
        DilutionCalculator,
        EnvironmentalCompensation,
    )
except ImportError:
    pass

# Data management
try:
    from ptpd_calibration.data import (
        PrintDatabase,
        PrintRecord,
        DataExporter,
        DataImporter,
        SyncManager,
        LocalStorageProvider,
        VersionController,
    )
except ImportError:
    pass

# Hardware integrations
try:
    from ptpd_calibration.integrations import (
        SpectrophotometerInterface,
        XRiteIntegration,
        WeatherProvider,
        OpenWeatherMapProvider,
        PrinterInterface,
        EpsonDriver,
        CanonDriver,
        ICCProfileManager,
    )
except ImportError:
    pass

# Quality assurance
try:
    from ptpd_calibration.qa import (
        NegativeDensityValidator,
        ChemistryFreshnessTracker,
        PaperHumidityChecker,
        UVLightMeterIntegration,
        QualityReport,
        AlertSystem,
    )
except ImportError:
    pass

# Education
try:
    from ptpd_calibration.education import (
        TutorialManager,
        Glossary,
        TipsManager,
    )
except ImportError:
    pass

# Performance monitoring
try:
    from ptpd_calibration.monitoring import (
        PerformanceMonitor,
        ImageProcessingProfiler,
        APIPerformanceTracker,
        CacheManager,
        ResourceMonitor,
        PerformanceReport,
    )
except ImportError:
    pass

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
    "ChemistrySuggestion",
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
