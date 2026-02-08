# Quick Start Guide - Quality Assurance Module

## Installation

```python
from ptpd_calibration.qa import (
    NegativeDensityValidator,
    ChemistryFreshnessTracker,
    PaperHumidityChecker,
    UVLightMeterIntegration,
    QualityReport,
    AlertSystem,
)
```

## Quick Examples

### 1. Validate Negative Density (30 seconds)

```python
from ptpd_calibration.qa import NegativeDensityValidator
from PIL import Image

validator = NegativeDensityValidator()
negative = Image.open("your_negative.tif")
analysis = validator.validate_density_range(negative)

print(f"Density Range: {analysis.density_range:.2f}")
print(f"Ready to print: {not (analysis.highlight_blocked or analysis.shadow_blocked)}")
```

### 2. Track Chemistry (1 minute)

```python
from ptpd_calibration.qa import ChemistryFreshnessTracker, SolutionType
from datetime import datetime

tracker = ChemistryFreshnessTracker()

# Register your solutions
pd_id = tracker.register_solution(
    SolutionType.PALLADIUM,
    datetime.now(),
    volume_ml=100.0
)

# Check before printing
is_fresh, msg = tracker.check_freshness(pd_id)
print(msg)

# Log after printing
tracker.log_usage(pd_id, amount_ml=8.5)
```

### 3. Monitor Paper Humidity (30 seconds)

```python
from ptpd_calibration.qa import PaperHumidityChecker

checker = PaperHumidityChecker()

# Take a reading
checker.measure_paper_humidity(52.0, temperature_celsius=21.0)

# Check if ready
is_ready, msg = checker.is_paper_ready()
print(msg)
```

### 4. Complete Pre-Print Checklist (2 minutes)

```python
from ptpd_calibration.qa import QualityReport
from PIL import Image

# Set up your tracking (from above examples)
# ... validator, tracker, checker, uv_meter ...

report = QualityReport()
checklist = report.generate_pre_print_checklist(
    image=Image.open("negative.tif"),
    chemistry_tracker=tracker,
    humidity_checker=checker,
    uv_meter=uv_meter,
)

if checklist['ready_to_print']:
    print("✓ All systems go!")
else:
    print("⚠ Issues detected:")
    for error in checklist['errors']:
        print(f"  - {error}")
```

## Common Workflows

### Before Each Print Session

```python
# 1. Check chemistry
alerts = chemistry_tracker.get_alerts()
if alerts:
    for alert in alerts:
        print(f"[{alert['severity']}] {alert['message']}")

# 2. Check paper
is_ready, msg = humidity_checker.is_paper_ready()
print(f"Paper: {msg}")

# 3. Check UV
uv_meter.read_intensity(current_reading)
adjustment, msg = uv_meter.calculate_exposure_adjustment()
print(f"UV: {msg}")

# 4. Validate negative
analysis = validator.validate_density_range(negative)
print(f"Density: {analysis.density_range:.2f}")
```

### After Printing

```python
# Log chemistry usage
chemistry_tracker.log_usage(solution_id, amount_used)

# Analyze print quality
scan = Image.open("print_scan.tif")
post_analysis = report.generate_post_print_analysis(scan)
print(f"Quality Score: {post_analysis['quality_score']}/100")
```

## Configuration

### Environment Variables

```bash
# Set in your shell or .env file
export PTPD_QA_MAX_DENSITY=3.5
export PTPD_QA_IDEAL_HUMIDITY_MIN=40.0
export PTPD_QA_IDEAL_HUMIDITY_MAX=60.0
export PTPD_QA_PALLADIUM_SHELF_LIFE=365
```

### Programmatic

```python
from ptpd_calibration.qa import QASettings

settings = QASettings(
    max_density=3.5,
    ideal_humidity_min=40.0,
    palladium_shelf_life=365,
)

validator = NegativeDensityValidator(settings)
```

## Key Thresholds (Defaults)

| Parameter | Default | Unit |
|-----------|---------|------|
| Max Density | 3.5 | D |
| Highlight Warning | 0.10 | D |
| Shadow Warning | 2.0 | D |
| Ideal Humidity (min) | 40% | RH |
| Ideal Humidity (max) | 60% | RH |
| Palladium Shelf Life | 365 | days |
| FO Shelf Life | 180 | days |
| UV Bulb Life | 1000 | hours |

## Files

- **Module**: `/src/ptpd_calibration/qa/`
- **Examples**: `/examples/qa_example.py`
- **Docs**: `/src/ptpd_calibration/qa/README.md`

## Getting Help

```python
from ptpd_calibration.qa import NegativeDensityValidator
help(NegativeDensityValidator)
```

## Pro Tips

1. **Always validate negatives before critical prints**
2. **Register chemistry immediately after mixing**
3. **Monitor humidity 24hrs before coating**
4. **Calibrate UV meter at session start**
5. **Generate pre-print checklists for archival work**
6. **Review alerts before each session**

## Next Steps

- Read the full documentation: `/src/ptpd_calibration/qa/README.md`
- Run examples: `python examples/qa_example.py`
- Integrate with your workflow
- Customize settings for your process
