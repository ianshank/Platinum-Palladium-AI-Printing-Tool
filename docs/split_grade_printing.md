# Split-Grade Printing Simulation for Pt/Pd

## Overview

The split-grade printing module provides comprehensive simulation and analysis tools for split-grade platinum/palladium printing. Split-grade printing is an advanced technique where shadows and highlights are printed separately with different contrast levels, then combined to achieve optimal tonal separation and detail preservation.

## Key Concepts

### What is Split-Grade Printing?

Split-grade printing involves:
1. **Shadow Exposure**: Printing with a harder contrast grade to increase shadow detail and separation
2. **Highlight Exposure**: Printing with a softer contrast grade to preserve highlight detail and smooth tonal transitions
3. **Blending**: Combining the two exposures using masks or sequential printing

### Benefits for Pt/Pd Printing

- **Enhanced Tonal Range**: Achieve both deep shadow detail and smooth highlight transitions
- **Better Control**: Independently control shadow and highlight contrast
- **Improved Detail**: Preserve detail in both shadows and highlights
- **Flexible Workflow**: Adjust individual exposures without reprinting
- **Pt/Pd Optimization**: Account for the unique tonal characteristics of platinum and palladium

## Module Components

### 1. SplitGradeSettings

Pydantic settings model for configuring split-grade simulation.

**Key Parameters:**
- `shadow_grade` (0-5): Contrast grade for shadow regions (0=softest, 5=hardest)
- `highlight_grade` (0-5): Contrast grade for highlight regions
- `shadow_exposure_ratio` (0.0-1.0): Percentage of total exposure for shadows
- `blend_mode`: Method for blending exposures (linear, gamma, soft_light, overlay, custom)
- `shadow_threshold` (0-1): Luminance threshold separating shadows from midtones
- `highlight_threshold` (0-1): Luminance threshold separating highlights from midtones
- `platinum_ratio` (0-1): Pt to Pd ratio (0=pure Pd, 1=pure Pt)

**Environment Variables:**
All settings can be configured via environment variables with `PTPD_SPLIT_GRADE_` prefix:
```bash
export PTPD_SPLIT_GRADE_SHADOW_GRADE=3.0
export PTPD_SPLIT_GRADE_HIGHLIGHT_GRADE=1.5
export PTPD_SPLIT_GRADE_PLATINUM_RATIO=0.5
```

### 2. SplitGradeSimulator

Main class for split-grade simulation and analysis.

**Key Methods:**

#### `analyze_image(image)`
Analyzes an image to determine optimal split-grade parameters.

**Returns:** `TonalAnalysis` with:
- Histogram statistics (mean, median, std, percentiles)
- Tonal distribution (shadow, midtone, highlight percentages)
- Recommended split-grade parameters
- Image characteristics (low-key, high-key, contrast score)
- Quality metrics and notes

**Example:**
```python
from ptpd_calibration.imaging import SplitGradeSimulator
from PIL import Image

simulator = SplitGradeSimulator()
image = Image.open("my_image.jpg")

analysis = simulator.analyze_image(image)
print(f"Recommended shadow grade: {analysis.recommended_shadow_grade}")
print(f"Recommended highlight grade: {analysis.recommended_highlight_grade}")
print(f"Needs split-grade: {analysis.needs_split_grade}")
```

#### `create_shadow_mask(image, threshold)`
Creates a mask selecting shadow regions.

**Returns:** NumPy array (0-1) where 1=shadow, 0=not shadow

**Example:**
```python
shadow_mask = simulator.create_shadow_mask(image, threshold=0.4)
```

#### `create_highlight_mask(image, threshold)`
Creates a mask selecting highlight regions.

**Returns:** NumPy array (0-1) where 1=highlight, 0=not highlight

#### `simulate_split_grade(image, settings)`
Applies complete split-grade simulation to an image.

**Returns:** Processed image array (0-1 normalized)

**Example:**
```python
from ptpd_calibration.imaging import SplitGradeSettings, BlendMode

# Custom settings
settings = SplitGradeSettings(
    shadow_grade=3.5,
    highlight_grade=1.0,
    shadow_exposure_ratio=0.65,
    blend_mode=BlendMode.GAMMA,
    platinum_ratio=0.5,  # 50/50 Pt/Pd mix
)

processed = simulator.simulate_split_grade(image, settings)
```

#### `preview_result(image, settings, include_masks)`
Generates preview comparison showing original and processed results.

**Returns:** Dictionary with 'original', 'processed', and optionally 'shadow_mask', 'highlight_mask'

**Example:**
```python
preview = simulator.preview_result(image, include_masks=True)

# Access results
original = preview['original']
processed = preview['processed']
shadow_mask = preview['shadow_mask']
highlight_mask = preview['highlight_mask']
```

#### `calculate_exposure_times(base_time, settings)`
Calculates separate exposure times for shadow and highlight grades.

**Returns:** `ExposureCalculation` with timing details and recommendations

**Example:**
```python
exposure = simulator.calculate_exposure_times(base_time=60.0)
print(exposure.format_exposure_info())

# Output:
# SPLIT-GRADE EXPOSURE CALCULATION
# ==================================================
# Total Exposure Time: 60.0 seconds
#
# SHADOW EXPOSURE:
#   Time: 36.0 seconds (60%)
#   Grade: 2.5
#
# HIGHLIGHT EXPOSURE:
#   Time: 24.0 seconds (40%)
#   Grade: 1.5
```

#### `blend_exposures(shadow_image, highlight_image, settings, shadow_mask, highlight_mask)`
Blends shadow and highlight exposures using specified blend mode.

### 3. TonalCurveAdjuster

Generates and applies tonal curves with Pt/Pd metal characteristics.

**Key Methods:**

#### `create_contrast_curve(grade, num_points)`
Generates contrast curve for a given grade level (0-5 scale).

**Returns:** Tuple of (input_values, output_values) arrays

**Grade Scale:**
- 0: Very soft, low contrast (long toe, gentle shoulder)
- 1-2: Soft to normal contrast
- 3: Normal contrast (linear in midtones)
- 4-5: Hard to very hard contrast (short toe, steep midtones)

**Example:**
```python
from ptpd_calibration.imaging import TonalCurveAdjuster

adjuster = TonalCurveAdjuster()

# Generate curves for different grades
for grade in [0, 1, 2, 3, 4, 5]:
    x, y = adjuster.create_contrast_curve(grade)
    # Use x, y arrays to plot or apply curve
```

#### `apply_platinum_characteristic(image, strength)`
Applies platinum characteristic response curve.

**Platinum characteristics:**
- Higher maximum density (deeper blacks)
- Cooler, more neutral tones
- Slightly more contrast in highlights
- Longer tonal scale

**Example:**
```python
pt_processed = adjuster.apply_platinum_characteristic(image, strength=1.0)
```

#### `apply_palladium_characteristic(image, strength)`
Applies palladium characteristic response curve.

**Palladium characteristics:**
- Warmer, brown-black tones
- Slightly lower maximum density
- Softer highlight rolloff
- Smoother midtone transitions

**Example:**
```python
pd_processed = adjuster.apply_palladium_characteristic(image, strength=1.0)
```

#### `blend_metal_characteristics(image, pt_ratio, strength)`
Blends platinum and palladium characteristics based on chemistry ratio.

**Example:**
```python
# 75% Platinum, 25% Palladium
blended = adjuster.blend_metal_characteristics(
    image,
    pt_ratio=0.75,
    strength=1.0
)
```

#### `apply_curve_to_image(image, grade, apply_metal_characteristic)`
Applies contrast curve and optional metal characteristics to image.

## Complete Workflow Examples

### Example 1: Basic Split-Grade Workflow

```python
from ptpd_calibration.imaging import SplitGradeSimulator
from PIL import Image
import matplotlib.pyplot as plt

# Load image
image = Image.open("negative.jpg")

# Create simulator with defaults
simulator = SplitGradeSimulator()

# Step 1: Analyze the image
analysis = simulator.analyze_image(image)

print("Analysis Results:")
print(f"  Shadow Grade: {analysis.recommended_shadow_grade:.1f}")
print(f"  Highlight Grade: {analysis.recommended_highlight_grade:.1f}")
print(f"  Needs Split-Grade: {analysis.needs_split_grade}")

# Step 2: Apply split-grade simulation
processed = simulator.simulate_split_grade(image)

# Step 3: Calculate exposure times
base_exposure = 60.0  # seconds
exposure = simulator.calculate_exposure_times(base_exposure)
print(f"\n{exposure.format_exposure_info()}")

# Step 4: Generate preview
preview = simulator.preview_result(image, include_masks=True)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(preview['original'], cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(preview['processed'], cmap='gray')
axes[0, 1].set_title('Split-Grade Processed')
axes[1, 0].imshow(preview['shadow_mask'], cmap='gray')
axes[1, 0].set_title('Shadow Mask')
axes[1, 1].imshow(preview['highlight_mask'], cmap='gray')
axes[1, 1].set_title('Highlight Mask')
plt.tight_layout()
plt.savefig('split_grade_preview.png')
```

### Example 2: Custom Settings for High-Contrast Pt/Pd

```python
from ptpd_calibration.imaging import (
    SplitGradeSimulator,
    SplitGradeSettings,
    BlendMode,
)

# Configure for high-contrast platinum printing
settings = SplitGradeSettings(
    shadow_grade=4.0,          # Hard shadow grade
    highlight_grade=1.0,       # Soft highlight grade
    shadow_exposure_ratio=0.7, # More shadow exposure
    blend_mode=BlendMode.GAMMA,
    blend_gamma=2.4,
    shadow_threshold=0.35,
    highlight_threshold=0.75,
    platinum_ratio=1.0,        # Pure platinum
    mask_blur_radius=20.0,     # Smooth mask transitions
    preserve_highlights=True,
    preserve_shadows=True,
)

simulator = SplitGradeSimulator(settings=settings)

# Process image
processed = simulator.simulate_split_grade(image)

# Calculate exposures
exposure = simulator.calculate_exposure_times(base_time=90.0)
print(exposure.format_exposure_info())
```

### Example 3: Comparing Different Metal Ratios

```python
from ptpd_calibration.imaging import TonalCurveAdjuster
import numpy as np
import matplotlib.pyplot as plt

adjuster = TonalCurveAdjuster()

# Test image
test_image = np.linspace(0, 1, 256).reshape(1, -1)
test_image = np.tile(test_image, (100, 1))

# Compare different Pt/Pd ratios
ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

for pt_ratio in ratios:
    processed = adjuster.blend_metal_characteristics(
        test_image,
        pt_ratio=pt_ratio,
        strength=1.0
    )
    results.append(processed)

# Plot comparison
fig, axes = plt.subplots(len(ratios), 1, figsize=(12, 8))
for i, (ratio, result) in enumerate(zip(ratios, results)):
    axes[i].imshow(result, cmap='gray', aspect='auto')
    axes[i].set_title(f'{ratio*100:.0f}% Pt / {(1-ratio)*100:.0f}% Pd')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('metal_ratio_comparison.png')
```

### Example 4: Interactive Parameter Exploration

```python
from ptpd_calibration.imaging import SplitGradeSimulator, SplitGradeSettings
import numpy as np

def explore_grades(image, shadow_grades, highlight_grades):
    """Explore different grade combinations."""

    results = {}

    for sg in shadow_grades:
        for hg in highlight_grades:
            settings = SplitGradeSettings(
                shadow_grade=sg,
                highlight_grade=hg,
            )

            simulator = SplitGradeSimulator(settings=settings)
            processed = simulator.simulate_split_grade(image)

            # Calculate quality metrics
            contrast = processed.std()
            tonal_range = processed.max() - processed.min()

            results[(sg, hg)] = {
                'image': processed,
                'contrast': contrast,
                'tonal_range': tonal_range,
            }

    return results

# Explore grade combinations
shadow_grades = [2.0, 2.5, 3.0, 3.5]
highlight_grades = [0.5, 1.0, 1.5, 2.0]

results = explore_grades(image, shadow_grades, highlight_grades)

# Find optimal combination
best = max(results.items(), key=lambda x: x[1]['contrast'])
print(f"Optimal grades: Shadow={best[0][0]}, Highlight={best[0][1]}")
```

## Blend Modes

### LINEAR
Simple weighted average of shadow and highlight images.
- Best for: Quick previews, subtle effects
- Characteristics: Most straightforward, may lack tonal depth

### GAMMA (Recommended)
Gamma-corrected blend for perceptually uniform transitions.
- Best for: Most Pt/Pd printing scenarios
- Characteristics: Natural-looking transitions, preserves midtone detail
- Parameter: `blend_gamma` (default 2.2)

### SOFT_LIGHT
Soft light compositing mode.
- Best for: Subtle contrast enhancement
- Characteristics: Gentle, preserves original tones

### OVERLAY
Overlay compositing mode.
- Best for: Increased contrast
- Characteristics: More aggressive blending, higher contrast

### CUSTOM
Reserved for custom blend curves (advanced use).

## Technical Notes

### Grade Scale Reference

The 0-5 grade scale corresponds to traditional variable contrast paper grades:

| Grade | Contrast | Use Case |
|-------|----------|----------|
| 0 | Very soft | High-contrast negatives, highlight preservation |
| 1 | Soft | Gentle contrast increase |
| 2 | Normal-soft | Standard printing |
| 3 | Normal | Linear midtone response |
| 4 | Hard | Low-contrast negatives, shadow separation |
| 5 | Very hard | Maximum contrast, split-grade shadows |

### Pt/Pd Metal Characteristics

**Platinum (Pt):**
- Cooler, more neutral tone
- Higher Dmax (deeper blacks)
- Slightly increased highlight contrast
- Extended shadow detail

**Palladium (Pd):**
- Warmer, brown-black tone
- Slightly lower Dmax
- Softer highlight rolloff
- Smoother midtone transitions

**Mixed Ratios:**
- 100% Pd (0.0): Warmest, most economical
- 75% Pd / 25% Pt (0.25): Warm with good blacks
- 50% Pd / 50% Pt (0.5): Balanced, classic look
- 25% Pd / 75% Pt (0.75): Cooler, deeper blacks
- 100% Pt (1.0): Coolest, maximum Dmax

### Mask Processing

Masks are processed with:
1. **Gaussian blur**: Smooths transitions (configurable radius)
2. **Feathering**: Softens mask edges (0-1 parameter)
3. **Normalization**: Ensures masks sum to 1.0 for proper blending

### Performance Tips

1. **Use thumbnails for previews**: Resize large images before analysis
2. **Cache results**: Analysis results can be reused for multiple processing attempts
3. **Batch processing**: Process multiple images with same settings for consistency
4. **Incremental refinement**: Start with analysis recommendations, then fine-tune

## Integration with Pt/Pd Workflow

### Digital Negative Creation

```python
from ptpd_calibration.imaging import ImageProcessor, SplitGradeSimulator

# Load and process image
processor = ImageProcessor()
result = processor.load_image("original.jpg")

# Apply split-grade simulation
simulator = SplitGradeSimulator()
processed = simulator.simulate_split_grade(result.image)

# Create digital negative
from PIL import Image
negative_image = Image.fromarray((processed * 255).astype('uint8'), mode='L')
inverted = processor.invert(
    processor.load_image(negative_image)
)

# Export for printing
processor.export(
    inverted,
    "split_grade_negative.tif",
    settings=ExportSettings(
        format=ImageFormat.TIFF_16BIT,
        preserve_resolution=True,
    )
)
```

### Chemistry Calculation Integration

```python
from ptpd_calibration.chemistry import ChemistryCalculator, MetalMix
from ptpd_calibration.imaging import SplitGradeSimulator, SplitGradeSettings

# Configure split-grade with chemistry
settings = SplitGradeSettings(
    platinum_ratio=0.5,  # 50/50 mix
)

# Calculate chemistry
calc = ChemistryCalculator()
recipe = calc.calculate_from_preset(
    width_inches=8,
    height_inches=10,
    metal_mix=MetalMix.CLASSIC_MIX,  # Matches 50/50 ratio
)

print(recipe.format_recipe())

# Process negative with matching Pt/Pd ratio
simulator = SplitGradeSimulator(settings=settings)
processed = simulator.simulate_split_grade(image)
```

## Troubleshooting

### Issue: Split-grade not improving results
**Solution:** Image may not benefit from split-grade printing. Check `analysis.needs_split_grade`. Try single-grade printing if tonal range is limited.

### Issue: Harsh transitions between tonal regions
**Solution:** Increase `mask_blur_radius` and `mask_feather_amount` for smoother blending.

### Issue: Lost shadow detail
**Solution:** Decrease `shadow_grade` or increase `shadow_exposure_ratio`. Enable `preserve_shadows`.

### Issue: Blocked highlights
**Solution:** Decrease `highlight_grade` or enable `preserve_highlights`. Adjust `highlight_hold_point`.

### Issue: Unnatural-looking results
**Solution:** Try different `blend_mode` (gamma recommended). Adjust `blend_gamma` for gamma mode.

## References

- Bostick-Sullivan: Platinum and Palladium Printing Instructions
- Dick Arentz: Platinum and Palladium Printing, 2nd Edition
- Irving Poor: The Print (Split-Grade Printing Techniques)
- Ctein: Post Exposure (Advanced Printing Control)

## See Also

- `ImageProcessor`: For general image processing and digital negative creation
- `ChemistryCalculator`: For Pt/Pd coating solution calculations
- `CurveGenerator`: For calibration curve creation
- `HistogramAnalyzer`: For detailed tonal analysis
