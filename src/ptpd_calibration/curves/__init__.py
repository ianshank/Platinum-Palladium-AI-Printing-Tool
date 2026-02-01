"""
Curve generation, export, and modification module.
"""

from ptpd_calibration.curves.ai_enhance import (
    CurveAIEnhancer,
    EnhancementGoal,
    EnhancementResult,
    enhance_curve,
)
from ptpd_calibration.curves.analysis import CurveAnalyzer
from ptpd_calibration.curves.export import (
    CurveExporter,
    PiezographyExporter,
    QTRExporter,
    load_curve,
    save_curve,
)
from ptpd_calibration.curves.generator import (
    CurveGenerator,
    TargetCurve,
    generate_linearization_curve,
)
from ptpd_calibration.curves.linearization import (
    AutoLinearizer,
    LinearizationConfig,
    LinearizationMethod,
    LinearizationResult,
    TargetResponse,
)
from ptpd_calibration.curves.modifier import (
    AdjustmentType,
    BlendMode,
    CurveAdjustment,
    CurveModifier,
    SmoothingMethod,
    adjust_curve,
    blend_curves,
    smooth_curve,
)
from ptpd_calibration.curves.parser import (
    ChannelCurve,
    QuadFileParser,
    QuadProfile,
    load_quad_file,
    load_quad_string,
)
from ptpd_calibration.curves.visualization import (
    ColorScheme,
    CurveComparisonResult,
    CurveStatistics,
    CurveVisualizer,
    PlotStyle,
    VisualizationConfig,
)

__all__ = [
    # Generator
    "CurveGenerator",
    "TargetCurve",
    "generate_linearization_curve",
    # Export
    "CurveExporter",
    "QTRExporter",
    "PiezographyExporter",
    "save_curve",
    "load_curve",
    # Analysis
    "CurveAnalyzer",
    # Parser
    "QuadProfile",
    "QuadFileParser",
    "ChannelCurve",
    "load_quad_file",
    "load_quad_string",
    # Modifier
    "CurveModifier",
    "CurveAdjustment",
    "AdjustmentType",
    "SmoothingMethod",
    "BlendMode",
    "adjust_curve",
    "smooth_curve",
    "blend_curves",
    # AI Enhancement
    "CurveAIEnhancer",
    "EnhancementGoal",
    "EnhancementResult",
    "enhance_curve",
    # Visualization
    "CurveVisualizer",
    "CurveStatistics",
    "CurveComparisonResult",
    "VisualizationConfig",
    "PlotStyle",
    "ColorScheme",
    # Auto-linearization
    "AutoLinearizer",
    "LinearizationMethod",
    "TargetResponse",
    "LinearizationConfig",
    "LinearizationResult",
]
