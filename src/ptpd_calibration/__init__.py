"""
PTPD Calibration - AI-powered calibration system for platinum/palladium printing.

This package provides comprehensive tools for calibrating digital negatives
for Pt/Pd printing, including:

- Step tablet detection and density extraction
- Curve generation for QTR and Piezography
- ML-based prediction and active learning
- AI assistant for calibration guidance
- Agentic system for autonomous calibration tasks
"""

__version__ = "1.0.0"

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
]
