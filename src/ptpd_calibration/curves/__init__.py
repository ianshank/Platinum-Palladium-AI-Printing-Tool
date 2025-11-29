"""
Curve generation, export, and modification module.
"""

from ptpd_calibration.curves.generator import (
    CurveGenerator,
    TargetCurve,
    generate_linearization_curve,
)
from ptpd_calibration.curves.export import (
    CurveExporter,
    QTRExporter,
    PiezographyExporter,
    save_curve,
    load_curve,
)
from ptpd_calibration.curves.analysis import CurveAnalyzer
from ptpd_calibration.curves.parser import (
    QuadProfile,
    QuadFileParser,
    ChannelCurve,
    load_quad_file,
    load_quad_string,
)
from ptpd_calibration.curves.modifier import (
    CurveModifier,
    CurveAdjustment,
    AdjustmentType,
    SmoothingMethod,
    BlendMode,
    adjust_curve,
    smooth_curve,
    blend_curves,
)
from ptpd_calibration.curves.ai_enhance import (
    CurveAIEnhancer,
    EnhancementGoal,
    EnhancementResult,
    enhance_curve,
)
from ptpd_calibration.curves.visualization import (
    CurveVisualizer,
    CurveStatistics,
    CurveComparisonResult,
    VisualizationConfig,
    PlotStyle,
    ColorScheme,
)
from ptpd_calibration.curves.linearization import (
    AutoLinearizer,
    LinearizationMethod,
    TargetResponse,
    LinearizationConfig,
    LinearizationResult,
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
