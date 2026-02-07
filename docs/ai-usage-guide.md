# PlatinumPalladiumAI - Usage Guide

Comprehensive AI tool for platinum-palladium alternative photographic printing.

## Installation

The class is located at `/home/user/Platinum-Palladium-AI-Printing-Tool/src/ptpd_calibration/ai/platinum_palladium_ai.py`

## Quick Start

```python
from ptpd_calibration.ai import PlatinumPalladiumAI, TonePreference, ContrastLevel
from ptpd_calibration.core.types import ChemistryType
from pathlib import Path

# Initialize the AI system
ai = PlatinumPalladiumAI()
```

## Method 1: Image Tonality Analysis

Analyze an image for optimal Pt/Pd conversion with zone-based recommendations.

```python
# Analyze image tonality
result = ai.analyze_image_tonality(
    image="path/to/image.jpg",
    target_paper="Arches Platine",
    target_process=ChemistryType.PURE_PALLADIUM
)

# Access results
print(f"Dynamic Range: {result.dynamic_range_stops:.1f} stops")
print(f"Dominant Zones: {result.dominant_zones}")
print(f"Shadow Detail: {result.shadow_detail_percent:.1f}%")
print(f"Highlight Detail: {result.highlight_detail_percent:.1f}%")

# View suggestions
for suggestion in result.suggestions:
    print(f"• {suggestion}")

# Check warnings
for warning in result.warnings:
    print(f"⚠ {warning}")
```

**Returns**: `TonalityAnalysisResult` with:
- Histogram statistics
- Zone distribution (Ansel Adams zones 0-10)
- Dynamic range analysis
- Exposure and contrast recommendations

## Method 2: Exposure Time Prediction

AI-based exposure prediction with confidence intervals.

```python
from ptpd_calibration.exposure.calculator import LightSource

# Predict exposure time
prediction = ai.predict_exposure_time(
    negative_density=1.8,
    paper_type="Arches Platine",
    light_source=LightSource.BL_FLUORESCENT,
    humidity=50.0,
    temperature=21.0,
    platinum_ratio=0.0,  # Pure palladium
    distance_inches=4.0
)

# View prediction
print(f"Predicted Exposure: {prediction.format_time()}")
print(f"95% Confidence Interval: {prediction.lower_bound_seconds:.0f}s - {prediction.upper_bound_seconds:.0f}s")

# View adjustments
for factor, value in prediction.adjustments_applied.items():
    print(f"{factor}: {value:.2f}x")

# View recommendations
for rec in prediction.recommendations:
    print(f"• {rec}")
```

**Returns**: `ExposurePrediction` with:
- Predicted exposure time (seconds and minutes)
- 95% confidence interval bounds
- Breakdown of adjustment factors
- Environmental considerations

## Method 3: Chemistry Ratio Recommendations

Get chemistry recommendations based on desired tone and contrast.

```python
# Get chemistry recommendations
chem = ai.suggest_chemistry_ratios(
    desired_tone=TonePreference.WARM,
    contrast_level=ContrastLevel.NORMAL,
    paper_type="Arches Platine",
    print_size_inches=(8, 10)
)

# View metal ratios
print(f"Platinum: {chem.platinum_ratio*100:.0f}%")
print(f"Palladium: {chem.palladium_ratio*100:.0f}%")

# View amounts
print(f"\nFerric Oxalate #1: {chem.ferric_oxalate_1_drops:.1f} drops")
print(f"Ferric Oxalate #2: {chem.ferric_oxalate_2_drops:.1f} drops")
print(f"Na2: {chem.na2_drops:.1f} drops")

# View expected results
print(f"\nExpected Tone: {chem.expected_tone}")
print(f"Expected Dmax: {chem.expected_dmax:.1f}")

# View rationale
for reason in chem.rationale:
    print(f"• {reason}")
```

**Returns**: `ChemistryRecommendation` with:
- Platinum/Palladium ratios
- Ferric oxalate amounts (drops)
- Contrast agent recommendations
- Expected tonal characteristics

## Method 4: Digital Negative Generation

Create optimized digital negatives with curve application.

```python
from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging.processor import ImageFormat

# Load or create a calibration curve
curve = CurveData(
    name="Arches Platine Linear",
    input_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    output_values=[0.0, 0.22, 0.48, 0.73, 1.0]
)

# Generate digital negative
negative = ai.generate_digital_negative(
    image="path/to/image.jpg",
    curve=curve,
    output_path="output/negative.tif",
    output_format=ImageFormat.TIFF_16BIT,
    target_dpi=2880,
    invert=True
)

# View results
print(f"Created: {negative.output_path}")
print(f"Size: {negative.output_size}")
print(f"Format: {negative.output_format}")
print(f"Quality Score: {negative.estimated_quality:.1%}")

# View processing steps
for step in negative.steps_applied:
    print(f"✓ {step}")
```

**Returns**: `DigitalNegativeResult` with:
- Processing result object
- Output file path
- Size and format information
- Quality metrics

## Method 5: Print Quality Analysis

Compare print scan to reference image and get correction suggestions.

```python
# Analyze print quality
analysis = ai.analyze_print_quality(
    scan_image="scans/print_001.jpg",
    reference_image="reference/original.jpg",
    zone_weight=1.0
)

# View overall scores
print(f"Overall Match Score: {analysis.overall_match_score:.1%}")
print(f"Density Correlation: {analysis.density_correlation:.3f}")
print(f"Mean Density Difference: {analysis.mean_density_difference:.1f}")

# View problem areas
print("\nProblem Areas:")
for area, description, severity in analysis.problem_areas:
    print(f"  {area.value}: {description} (severity: {severity:.2f})")

# View worst zones
print(f"\nWorst Zones: {analysis.worst_zones}")

# View corrections
print("\nSuggested Corrections:")
for correction in analysis.corrections:
    print(f"• {correction}")

# Exposure correction
if analysis.suggested_exposure_correction_stops != 0:
    print(f"\nExposure Correction: {analysis.suggested_exposure_correction_stops:+.1f} stops")

# Curve adjustments
if analysis.suggested_curve_adjustments:
    print("\nCurve Adjustments:")
    for param, value in analysis.suggested_curve_adjustments.items():
        print(f"  {param}: {value:+.2f}")
```

**Returns**: `PrintQualityAnalysis` with:
- Overall match score (0-1)
- Density correlation
- Problem area identification
- Zone-by-zone differences
- Specific correction suggestions

## Method 6: Workflow Optimization

Learn from print history and optimize workflow.

```python
from ptpd_calibration.core.models import CalibrationRecord

# Load print history (example)
print_history = [
    CalibrationRecord(
        paper_type="Arches Platine",
        chemistry_type=ChemistryType.PURE_PALLADIUM,
        metal_ratio=0.0,
        exposure_time=600,
        measured_densities=[0.1, 0.5, 1.0, 1.5, 2.0],
        # ... other parameters
    ),
    # ... more records
]

# Optimize workflow
optimization = ai.optimize_workflow(
    print_history=print_history,
    success_threshold=0.8
)

# View analysis
print(f"Total Prints: {optimization.total_prints_analyzed}")
print(f"Successful: {optimization.successful_prints}")
print(f"Success Rate: {optimization.success_rate:.1%}")
print(f"Confidence: {optimization.confidence:.1%}")

# View optimal parameters
print("\nOptimal Parameters:")
for param, value in optimization.optimal_parameters.items():
    print(f"  {param}: {value}")

# View trends
print("\nParameter Trends:")
for param, trend in optimization.parameter_trends.items():
    print(f"  {param}: {trend}")

# View recommendations
print("\nRecommendations:")
if optimization.recommended_base_exposure:
    print(f"  Base Exposure: {optimization.recommended_base_exposure:.0f}s")
if optimization.recommended_metal_ratio:
    print(f"  Metal Ratio: {optimization.recommended_metal_ratio:.1%} Pt")

# View efficiency suggestions
print("\nEfficiency Suggestions:")
for suggestion in optimization.efficiency_suggestions:
    print(f"• {suggestion}")

# View common mistakes
if optimization.common_mistakes:
    print("\nCommon Mistakes to Avoid:")
    for mistake in optimization.common_mistakes:
        print(f"⚠ {mistake}")

# View insights
print("\nInsights:")
for insight in optimization.insights:
    print(f"💡 {insight}")
```

**Returns**: `WorkflowOptimization` with:
- Success rate analysis
- Optimal parameter identification
- Parameter trends
- Efficiency suggestions
- Common mistakes
- Detailed insights

## Configuration

All parameters are configurable via the settings system:

```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Access chemistry settings
print(f"Base drops per sq in: {settings.chemistry.drops_per_square_inch}")
print(f"Default Pt ratio: {settings.chemistry.default_platinum_ratio}")

# Access exposure settings (if available)
# print(f"Base exposure: {settings.exposure.base_exposure_minutes}")

# You can also pass custom settings to the AI
from ptpd_calibration.config import Settings

custom_settings = Settings()
ai = PlatinumPalladiumAI(settings=custom_settings)
```

## Complete Workflow Example

```python
from ptpd_calibration.ai import (
    PlatinumPalladiumAI,
    TonePreference,
    ContrastLevel
)
from ptpd_calibration.exposure.calculator import LightSource
from ptpd_calibration.imaging.processor import ImageFormat
from ptpd_calibration.core.types import ChemistryType

# Initialize AI
ai = PlatinumPalladiumAI()

# 1. Analyze source image
print("=== ANALYZING IMAGE ===")
tonality = ai.analyze_image_tonality(
    image="source.jpg",
    target_paper="Arches Platine",
    target_process=ChemistryType.PURE_PALLADIUM
)
print(f"Dynamic Range: {tonality.dynamic_range_stops:.1f} stops")
for suggestion in tonality.suggestions[:3]:
    print(f"• {suggestion}")

# 2. Get chemistry recommendations
print("\n=== CHEMISTRY RECOMMENDATIONS ===")
chemistry = ai.suggest_chemistry_ratios(
    desired_tone=TonePreference.WARM,
    contrast_level=ContrastLevel.NORMAL,
    print_size_inches=(8, 10)
)
print(f"Metal Ratio: {chemistry.platinum_ratio*100:.0f}% Pt / {chemistry.palladium_ratio*100:.0f}% Pd")
print(f"Expected Tone: {chemistry.expected_tone}")

# 3. Predict exposure
print("\n=== EXPOSURE PREDICTION ===")
exposure = ai.predict_exposure_time(
    negative_density=1.8,
    light_source=LightSource.BL_FLUORESCENT,
    platinum_ratio=chemistry.platinum_ratio,
    humidity=50.0,
    temperature=21.0
)
print(f"Predicted: {exposure.format_time()}")
print(f"Range: {exposure.lower_bound_seconds:.0f}s - {exposure.upper_bound_seconds:.0f}s")

# 4. Generate digital negative
print("\n=== GENERATING NEGATIVE ===")
# (Assuming you have a curve)
# negative = ai.generate_digital_negative(
#     image="source.jpg",
#     curve=my_curve,
#     output_path="negative.tif",
#     output_format=ImageFormat.TIFF_16BIT
# )
# print(f"Created: {negative.output_path}")

print("\n=== WORKFLOW COMPLETE ===")
```

## Notes

- All methods use Pydantic models for type safety and validation
- Settings are configurable via the global settings system
- No hardcoded values - all parameters can be adjusted
- Comprehensive type hints throughout
- Detailed docstrings for all methods
- Follows existing codebase patterns

## File Locations

- **Main Class**: `/home/user/Platinum-Palladium-AI-Printing-Tool/src/ptpd_calibration/ai/platinum_palladium_ai.py`
- **Module Init**: `/home/user/Platinum-Palladium-AI-Printing-Tool/src/ptpd_calibration/ai/__init__.py`
- **Total Lines**: 1,414 lines of comprehensive implementation
