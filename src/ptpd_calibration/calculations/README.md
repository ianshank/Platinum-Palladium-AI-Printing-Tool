# Enhanced Calculations Module

Advanced technical calculations for platinum/palladium printing with comprehensive environmental compensation and cost tracking.

## Overview

The enhanced calculations module provides five specialized calculators:

1. **UVExposureCalculator** - Multi-factor UV exposure calculations
2. **CoatingVolumeCalculator** - Optimal coating volume determination
3. **CostCalculator** - Print and session cost tracking
4. **DilutionCalculator** - Solution dilution and replenishment
5. **EnvironmentalCompensation** - Environmental and seasonal adjustments

All calculations are **configuration-driven** with **no hardcoded values**.

## Installation

The module is part of the `ptpd_calibration` package:

```python
from ptpd_calibration.calculations import (
    UVExposureCalculator,
    CoatingVolumeCalculator,
    CostCalculator,
    DilutionCalculator,
    EnvironmentalCompensation,
)
```

## Calculators

### 1. UVExposureCalculator

Calculate adjusted UV exposure times accounting for all environmental factors.

**Features:**
- Negative density compensation (industry-standard 0.3D = 1 stop)
- Humidity effects on paper sensitivity
- Temperature effects on chemical reactions
- UV intensity variations
- Paper speed factors
- Chemistry-based exposure adjustments
- Confidence intervals

**Example:**

```python
calc = UVExposureCalculator()

result = calc.calculate_uv_exposure(
    base_time=10.0,              # Base exposure in minutes
    negative_density=1.8,         # Negative Dmax - Dmin
    humidity=55.0,                # Relative humidity %
    temperature=70.0,             # Temperature in °F
    uv_intensity=95.0,           # UV intensity % of reference
    paper_factor=1.0,            # Paper speed multiplier
    chemistry_factor=1.2,        # Chemistry multiplier (e.g., high Pt)
    base_density=1.6,            # Reference density
    optimal_humidity=50.0,
    optimal_temperature=68.0,
)

print(f"Adjusted exposure: {result.adjusted_exposure_minutes:.2f} minutes")
print(f"Confidence interval: {result.confidence_lower_minutes:.2f} - {result.confidence_upper_minutes:.2f} min")
```

**Adjustment Factors:**
- **Humidity Factor**: ±15% per 100% humidity delta
- **Temperature Factor**: ±5% per 10°F delta
- **Density Factor**: 2^(density_delta / 0.3)
- **Intensity Factor**: Inverse of intensity %

### 2. CoatingVolumeCalculator

Determine optimal sensitizer volume based on paper type, coating method, and conditions.

**Features:**
- Paper absorbency profiles (configurable)
- Coating method efficiency factors
- Humidity absorption adjustments
- Waste factor calculations
- Practical recommendations (ml and drops)

**Paper Profiles (default):**
- `arches_platine`: 0.0465 ml/sq in (hot press, low absorbency)
- `arches_88`: 0.0510 ml/sq in (medium)
- `hahnemuhle_platinum`: 0.0445 ml/sq in (hot press, sized)
- `bergger_cot320`: 0.0520 ml/sq in (medium-high)
- `fabriano_artistico_hp`: 0.0450 ml/sq in (hot press)
- `fabriano_artistico_cp`: 0.0530 ml/sq in (cold press)
- `custom_hot_press`: 0.0450 ml/sq in
- `custom_cold_press`: 0.0540 ml/sq in
- `custom_rough`: 0.0580 ml/sq in

**Coating Method Efficiency:**
- `brush` / `hake_brush`: 1.0x (baseline)
- `glass_rod` / `puddle_pusher`: 0.75x (25% savings)
- `coating_rod`: 0.70x (30% savings)

**Example:**

```python
calc = CoatingVolumeCalculator()

result = calc.determine_coating_volume(
    paper_area=80.0,              # 8x10 = 80 sq in
    paper_type="arches_platine",
    coating_method="glass_rod",
    humidity=50.0,
    waste_factor=1.15,           # 15% extra
)

print(f"Recommended volume: {result.recommended_ml:.1f} ml")
print(f"                    ({result.recommended_drops:.0f} drops)")
```

### 3. CostCalculator

Track printing costs including chemistry, paper, and processing supplies.

**Features:**
- Per-print cost breakdown
- Session cost aggregation
- Solution usage estimation
- Stock level recommendations
- Cost reports

**Methods:**
- `calculate_print_cost()` - Total cost per print
- `calculate_session_cost()` - Total session cost
- `estimate_solution_usage()` - Estimate chemistry usage
- `generate_cost_report()` - Formatted cost analysis

**Example:**

```python
calc = CostCalculator()

# Single print cost
chemistry = {
    'ferric_oxalate_ml': 1.8,
    'platinum_ml': 0.9,
    'palladium_ml': 0.9,
    'na2_ml': 0.45,
}

result = calc.calculate_print_cost(
    paper_size="8x10",
    chemistry=chemistry,
    paper_type="arches_platine",
    include_processing=True,
)

print(f"Total cost: ${result.total_cost_usd:.2f}")
print(f"Chemistry: ${result.total_chemistry_ml:.2f} for {result.total_chemistry_ml:.1f} ml")

# Session cost
prints = [result1, result2, result3]  # List of PrintCostResult
session = calc.calculate_session_cost(prints)

print(f"Session total: ${session.total_session_cost_usd:.2f}")
print(f"Average per print: ${session.average_cost_per_print:.2f}")

# Usage estimation
estimate = calc.estimate_solution_usage(
    num_prints=10,
    avg_size="8x10",
    avg_platinum_ratio=0.5,
    coating_method="glass_rod",
)

print(f"Recommended stock - Platinum: {estimate.recommended_stock_platinum_ml:.0f} ml")
```

**Default Costs (configurable):**
- Ferric Oxalate: $0.50/ml
- Palladium: $2.00/ml
- Platinum: $8.00/ml
- Na2: $4.00/ml
- Developer: $5.00/L
- Clearing Bath: $3.00/L

### 4. DilutionCalculator

Calculate dilutions for developers, clearing baths, and replenishment.

**Features:**
- Developer dilution calculations
- Clearing bath preparation (3-bath sequence)
- Solution replenishment recommendations
- Exhaustion tracking

**Methods:**
- `calculate_developer_dilution()` - Developer mixing
- `calculate_clearing_bath()` - Clearing solution prep
- `suggest_replenishment()` - Replenishment amounts

**Example:**

```python
calc = DilutionCalculator()

# Developer dilution
result = calc.calculate_developer_dilution(
    concentrate_strength=20.0,  # 20% EDTA stock
    target_strength=3.0,         # 3% working solution
    volume=500.0,                # 500 ml needed
)

print(f"Mix {result.concentrate_ml:.1f} ml concentrate")
print(f"with {result.water_ml:.1f} ml water")
print(f"Ratio: {result.dilution_ratio}")

# Clearing bath
for bath_num in [1, 2, 3]:
    result = calc.calculate_clearing_bath(
        volume=1000.0,
        bath_number=bath_num,
    )
    # Bath 1: 1% citric acid
    # Bath 2: 0.5% citric acid
    # Bath 3: water rinse

# Replenishment
result = calc.suggest_replenishment(
    solution="developer",
    usage=250.0,              # Used 250 ml
    current_volume=1000.0,    # 1L working solution
    exhaustion_threshold=0.30,  # Replace at 30%
)

if result.should_replace:
    print(f"Replace solution: {result.replenish_ml:.0f} ml")
else:
    print(f"Top up: {result.replenish_ml:.0f} ml")
```

### 5. EnvironmentalCompensation

Environmental and seasonal adjustments for drying, exposure, and timing.

**Features:**
- Altitude adjustments (drying, UV intensity)
- Seasonal compensation
- Optimal conditions reference
- Drying time estimation

**Methods:**
- `adjust_for_altitude()` - Altitude adjustment
- `adjust_for_season()` - Seasonal adjustment
- `get_optimal_conditions()` - Optimal working conditions
- `calculate_drying_time()` - Drying time estimate

**Example:**

```python
calc = EnvironmentalCompensation()

# Altitude adjustment
result = calc.adjust_for_altitude(
    base_value=15.0,        # 15 min base drying
    altitude=5000.0,        # 5000 ft
    value_type="drying_time",
)

print(f"Adjusted drying: {result.adjusted_value:.1f} minutes")

# Seasonal adjustment
result = calc.adjust_for_season(
    base_value=10.0,        # 10 min base exposure
    month=7,                # July (summer)
    value_type="exposure_time",
    latitude=40.0,
)

print(f"Seasonal adjustment: {result.adjusted_value:.1f} minutes")

# Optimal conditions
conditions = calc.get_optimal_conditions()

print(f"Temp: {conditions.temperature_f_ideal:.0f}°F")
print(f"Range: {conditions.temperature_f_min:.0f}-{conditions.temperature_f_max:.0f}°F")
print(f"Humidity: {conditions.humidity_percent_ideal:.0f}% RH")

# Drying time
result = calc.calculate_drying_time(
    humidity=60.0,
    temperature=72.0,
    paper="arches_platine",
    forced_air=False,
)

print(f"Drying time: {result.drying_minutes:.1f} minutes")
print(f"Range: {result.estimated_range_minutes[0]:.1f} - {result.estimated_range_minutes[1]:.1f} min")
if result.forced_air_recommended:
    print("Forced air recommended")
```

**Altitude Effects:**
- Drying: -5% per 1000 ft (faster evaporation)
- UV Exposure: -4% per 1000 ft (higher UV intensity)

**Seasonal Variation:**
- Drying: ±15% summer to winter
- Exposure: ±10% summer to winter
- Varies with latitude (higher = more variation)

## Result Models

All calculators return **Pydantic models** with validation:

- `ExposureResult` - UV exposure calculation result
- `CoatingResult` - Coating volume calculation result
- `PrintCostResult` - Print cost breakdown
- `SessionCostResult` - Session cost summary
- `SolutionUsageEstimate` - Usage estimation
- `DilutionResult` - Dilution instructions
- `ReplenishmentResult` - Replenishment recommendation
- `EnvironmentalAdjustment` - Environmental adjustment
- `OptimalConditions` - Optimal working conditions
- `DryingTimeEstimate` - Drying time estimate

All models are **immutable** (frozen) and include:
- Type validation
- Range validation (where applicable)
- Helpful notes and warnings
- JSON serialization support

## Configuration

All calculations use values from `ChemistrySettings` (configurable via environment variables with `PTPD_CHEM_` prefix):

```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Access chemistry settings
print(settings.chemistry.drops_per_ml)                    # 20.0
print(settings.chemistry.platinum_cost_per_ml)             # 8.00
print(settings.chemistry.ferric_oxalate_cost_per_ml)       # 0.50
print(settings.chemistry.drops_per_square_inch)            # 0.465
```

**Key Settings:**
- `drops_per_square_inch`: Base coverage (default: 0.465)
- `drops_per_ml`: Dropper calibration (default: 20.0)
- `*_cost_per_ml`: Solution costs for cost calculations
- `*_absorbency_multiplier`: Paper absorbency factors
- `*_coating_multiplier`: Coating method efficiency

## Examples

See `/examples/enhanced_calculations_demo.py` for comprehensive demonstrations of all calculators.

Run the demo:

```bash
cd /home/user/Platinum-Palladium-AI-Printing-Tool
python examples/enhanced_calculations_demo.py
```

## Best Practices

1. **Always use configuration** - Don't hardcode values
2. **Check warnings** - Results include warnings for extreme conditions
3. **Use confidence intervals** - Account for uncertainty in exposure
4. **Round up volumes** - Better to have excess than shortage
5. **Track costs** - Use session cost tracking for budgeting
6. **Monitor exhaustion** - Replace solutions before quality degrades
7. **Consider environment** - Use environmental compensation in variable conditions

## Technical Notes

### UV Exposure Formula

```
adjusted_exposure = base_time ×
                   humidity_factor ×
                   temperature_factor ×
                   density_factor ×
                   intensity_factor ×
                   paper_factor ×
                   chemistry_factor
```

Where:
- `humidity_factor = 1.0 - (humidity_delta × 0.15)`
- `temperature_factor = 1.0 - (temp_delta/10 × 0.05)`
- `density_factor = 2^(density_delta / 0.3)`
- `intensity_factor = 100 / uv_intensity`

### Coating Volume Formula

```
total_volume = (paper_area × absorption_rate ×
                method_efficiency ×
                humidity_adjustment) ×
                waste_factor
```

### Drying Time Formula

```
drying_time = base_time ×
              humidity_factor ×
              temperature_factor ×
              paper_absorbency_factor
```

With forced air: `drying_time × 0.5`

## Integration

The enhanced calculations integrate seamlessly with existing calculators:

```python
from ptpd_calibration.exposure.calculator import ExposureCalculator
from ptpd_calibration.chemistry.calculator import ChemistryCalculator
from ptpd_calibration.calculations import UVExposureCalculator, CostCalculator

# Use existing calculator for base calculation
exposure_calc = ExposureCalculator()
base_result = exposure_calc.calculate(negative_density=1.8)

# Enhance with environmental compensation
uv_calc = UVExposureCalculator()
enhanced_result = uv_calc.calculate_uv_exposure(
    base_time=base_result.exposure_minutes,
    negative_density=1.8,
    humidity=current_humidity,
    temperature=current_temp,
    # ... other factors
)

# Calculate cost
chem_calc = ChemistryCalculator()
recipe = chem_calc.calculate(width_inches=8, height_inches=10)

cost_calc = CostCalculator()
cost = cost_calc.calculate_print_cost(
    paper_size="8x10",
    chemistry={
        'ferric_oxalate_ml': recipe.ferric_oxalate_ml,
        'platinum_ml': recipe.platinum_ml,
        'palladium_ml': recipe.palladium_ml,
        'na2_ml': recipe.na2_ml,
    },
)
```

## License

Part of the Platinum-Palladium AI Printing Tool.
