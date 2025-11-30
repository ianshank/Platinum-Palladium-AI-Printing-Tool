# Quality Assurance Module

Comprehensive quality assurance tools for platinum/palladium printing workflows.

## Overview

The QA module provides six main components for ensuring print quality:

1. **NegativeDensityValidator** - Validates negative density for optimal printing
2. **ChemistryFreshnessTracker** - Tracks chemistry solution freshness and usage
3. **PaperHumidityChecker** - Monitors paper humidity for coating readiness
4. **UVLightMeterIntegration** - Integrates UV light measurements and calibration
5. **QualityReport** - Generates comprehensive pre/post-print reports
6. **AlertSystem** - Manages quality alerts and notifications

## Installation

The QA module is part of the main ptpd_calibration package:

```python
from ptpd_calibration.qa import (
    NegativeDensityValidator,
    ChemistryFreshnessTracker,
    PaperHumidityChecker,
    UVLightMeterIntegration,
    QualityReport,
    AlertSystem,
    QASettings,
)
```

## Components

### 1. NegativeDensityValidator

Validates negative density values to ensure they're in the printable range for platinum/palladium processes.

**Features:**
- Density range validation (min/max)
- Highlight detail checking
- Shadow detail checking
- Zone-based histogram analysis (Ansel Adams zones)
- Automatic correction suggestions

**Example:**

```python
from ptpd_calibration.qa import NegativeDensityValidator
from PIL import Image

# Initialize validator
validator = NegativeDensityValidator()

# Load and validate negative
negative = Image.open("negative.tif")
analysis = validator.validate_density_range(negative)

print(f"Density Range: {analysis.density_range:.2f}")
print(f"Warnings: {analysis.warnings}")
print(f"Suggestions: {analysis.suggestions}")

# Check specific areas
has_detail, msg = validator.check_highlight_detail(negative)
print(f"Highlight Detail: {msg}")
```

**Methods:**
- `validate_density_range(image)` - Complete density analysis
- `check_highlight_detail(image)` - Check if highlights have detail
- `check_shadow_detail(image)` - Check if shadows have detail
- `get_density_histogram(image, bins)` - Get density histogram
- `suggest_corrections(...)` - Get correction suggestions

**Configuration:**
```python
from ptpd_calibration.qa import QASettings

settings = QASettings(
    min_density=0.0,
    max_density=3.5,
    highlight_warning_threshold=0.10,
    shadow_warning_threshold=2.0,
)
validator = NegativeDensityValidator(settings)
```

### 2. ChemistryFreshnessTracker

Tracks chemistry solutions, monitoring expiration dates and volume usage.

**Features:**
- Solution registration and tracking
- Expiration date calculation
- Volume usage logging
- Automatic alerts for expiring/low solutions
- Replenishment recommendations

**Example:**

```python
from ptpd_calibration.qa import ChemistryFreshnessTracker, SolutionType
from datetime import datetime

# Initialize tracker
tracker = ChemistryFreshnessTracker()

# Register a new solution
solution_id = tracker.register_solution(
    solution_type=SolutionType.PALLADIUM,
    date_mixed=datetime.now(),
    volume_ml=100.0,
    notes="Fresh batch for portrait series"
)

# Check freshness
is_fresh, msg = tracker.check_freshness(solution_id)
print(msg)  # "Fresh (365 days remaining)"

# Log usage
tracker.log_usage(solution_id, amount_ml=10.0)

# Get remaining volume
remaining = tracker.get_remaining_volume(solution_id)
print(f"Remaining: {remaining:.1f} ml")

# Check for alerts
alerts = tracker.get_alerts()
for alert in alerts:
    print(f"[{alert['severity']}] {alert['message']}")

# Get replenishment recommendation
rec = tracker.recommend_replenishment(solution_id)
print(rec)
```

**Solution Types:**
- `FERRIC_OXALATE_1` - Ferric oxalate (standard)
- `FERRIC_OXALATE_2` - Ferric oxalate with contrast agent
- `PALLADIUM` - Palladium solution
- `PLATINUM` - Platinum solution
- `NA2` - Sodium dichromate (contrast agent)
- `DEVELOPER` - Ammonium citrate developer
- `CLEARING_BATH` - EDTA clearing bath
- `EDTA` - EDTA solution

**Shelf Life Defaults:**
- Ferric Oxalate: 180 days
- Palladium/Platinum: 365 days
- Na2: 365 days
- Developer: 90 days
- Clearing Bath/EDTA: 90 days

### 3. PaperHumidityChecker

Monitors paper humidity to ensure optimal coating conditions.

**Features:**
- Humidity reading logging
- Paper readiness checking
- Drying time estimation
- Ambient condition tracking
- Humidity adjustment recommendations

**Example:**

```python
from ptpd_calibration.qa import PaperHumidityChecker

# Initialize checker
checker = PaperHumidityChecker()

# Log ambient conditions
checker.log_ambient_conditions(
    humidity_percent=45.0,
    temperature_celsius=22.0
)

# Measure paper humidity
reading = checker.measure_paper_humidity(
    humidity_percent=52.0,
    temperature_celsius=21.5,
    paper_type="Hahnemuhle Platinum Rag",
    notes="Stored in dry cabinet 24hrs"
)

# Check if ready to coat
is_ready, msg = checker.is_paper_ready()
print(msg)  # "Paper ready (52.0% RH in ideal range)"

# Estimate drying time if needed
hours, msg = checker.estimate_drying_time(current_humidity=65.0)
print(msg)  # "Approximately 7.5 hours of drying needed"

# Get recommendation
rec = checker.recommend_humidity_adjustment()
print(rec)
```

**Ideal Humidity Range:**
- Minimum: 40% RH (configurable)
- Maximum: 60% RH (configurable)
- Tolerance: ±5% (configurable)

### 4. UVLightMeterIntegration

Integrates UV light meter readings for exposure control.

**Features:**
- Meter calibration
- Intensity reading logging
- Exposure time adjustment calculation
- Bulb degradation detection
- Replacement timing recommendations

**Example:**

```python
from ptpd_calibration.qa import UVLightMeterIntegration

# Initialize UV meter
uv_meter = UVLightMeterIntegration()

# Calibrate meter
status = uv_meter.calibrate_meter(reference_intensity=100.0)
print(status)

# Take readings
reading = uv_meter.read_intensity(
    intensity=95.0,
    wavelength=365.0,
    bulb_hours=150.0,
    notes="Morning reading"
)

# Calculate exposure adjustment
adjustment, msg = uv_meter.calculate_exposure_adjustment()
print(f"Adjust exposure by {adjustment:.2f}x")
print(msg)

# Check bulb degradation
degraded, msg = uv_meter.check_bulb_degradation()
print(msg)

# Get replacement recommendation
rec = uv_meter.recommend_bulb_replacement()
print(rec)
```

**Default Settings:**
- Target Intensity: 100.0 (arbitrary units)
- Target Wavelength: 365nm
- Bulb Degradation Threshold: 15%
- Replacement Hours: 1000 hours

### 5. QualityReport

Generates comprehensive quality reports for pre-print and post-print analysis.

**Features:**
- Pre-print checklist generation
- Post-print analysis
- Multi-format export (JSON, Markdown, HTML, PDF)
- Quality scoring and grading
- Actionable recommendations

**Example:**

```python
from ptpd_calibration.qa import QualityReport, ReportFormat
from PIL import Image

# Initialize report generator
report = QualityReport()

# Generate pre-print checklist
checklist = report.generate_pre_print_checklist(
    image=negative_image,
    chemistry_tracker=tracker,
    humidity_checker=humidity_checker,
    uv_meter=uv_meter,
)

print(f"Ready to Print: {checklist['ready_to_print']}")

# Check individual components
for check_name, check_data in checklist['checks'].items():
    print(f"{check_name}: {check_data['status']}")

# Generate post-print analysis
scan = Image.open("finished_print_scan.tif")
analysis = report.generate_post_print_analysis(
    scan=scan,
    expected_density_range=(0.10, 2.0)
)

print(f"Quality Score: {analysis['quality_score']}/100")
print(f"Grade: {analysis['grade']}")

# Export report
markdown = report.export_report(checklist, ReportFormat.MARKDOWN)
print(markdown)

# Save to file
report.export_report(
    checklist,
    ReportFormat.HTML,
    output_path="/path/to/report.html"
)
```

**Quality Grading:**
- A: 90-100 points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: < 60 points

### 6. AlertSystem

Manages quality assurance alerts and notifications.

**Features:**
- Alert creation and tracking
- Severity-based filtering
- Alert dismissal
- Alert history
- Automatic cleanup of old alerts

**Example:**

```python
from ptpd_calibration.qa import AlertSystem, AlertType, AlertSeverity

# Initialize alert system
alerts = AlertSystem()

# Add alerts
alert_id = alerts.add_alert(
    alert_type=AlertType.CHEMISTRY,
    message="Palladium solution expires in 3 days",
    severity=AlertSeverity.ERROR
)

# Get active alerts
active = alerts.get_active_alerts()
for alert in active:
    print(f"[{alert.severity.value}] {alert.message}")

# Filter by severity
critical = alerts.get_active_alerts(severity=AlertSeverity.CRITICAL)

# Get alert summary
summary = alerts.get_alert_summary()
print(f"Total: {summary['total']}, Critical: {summary['critical']}")

# Dismiss alert
alerts.dismiss_alert(alert_id)

# Get history
history = alerts.get_alert_history(hours=24)

# Clean up old alerts
removed = alerts.clear_old_alerts()
print(f"Removed {removed} old alerts")
```

**Alert Types:**
- `DENSITY` - Negative density issues
- `CHEMISTRY` - Chemistry freshness/volume
- `HUMIDITY` - Paper humidity conditions
- `UV_LIGHT` - UV light intensity/bulb issues
- `GENERAL` - General QA alerts

**Severity Levels:**
- `INFO` - Informational
- `WARNING` - Warning, attention recommended
- `ERROR` - Error, action needed
- `CRITICAL` - Critical, immediate action required

## Configuration

All QA settings are configurable via environment variables or programmatically:

### Environment Variables

```bash
# Density validation
export PTPD_QA_MIN_DENSITY=0.0
export PTPD_QA_MAX_DENSITY=3.5
export PTPD_QA_HIGHLIGHT_WARNING_THRESHOLD=0.10
export PTPD_QA_SHADOW_WARNING_THRESHOLD=2.0

# Chemistry freshness (days)
export PTPD_QA_FERRIC_OXALATE_SHELF_LIFE=180
export PTPD_QA_PALLADIUM_SHELF_LIFE=365
export PTPD_QA_PLATINUM_SHELF_LIFE=365
export PTPD_QA_DEVELOPER_SHELF_LIFE=90

# Paper humidity
export PTPD_QA_IDEAL_HUMIDITY_MIN=40.0
export PTPD_QA_IDEAL_HUMIDITY_MAX=60.0
export PTPD_QA_HUMIDITY_TOLERANCE=5.0

# UV light
export PTPD_QA_UV_INTENSITY_TARGET=100.0
export PTPD_QA_UV_WAVELENGTH_TARGET=365.0
export PTPD_QA_BULB_REPLACEMENT_HOURS=1000
```

### Programmatic Configuration

```python
from ptpd_calibration.qa import QASettings
from ptpd_calibration.config import configure

# Create custom settings
qa_settings = QASettings(
    min_density=0.0,
    max_density=3.5,
    highlight_warning_threshold=0.10,
    shadow_warning_threshold=2.0,
    ferric_oxalate_shelf_life=180,
    palladium_shelf_life=365,
    ideal_humidity_min=40.0,
    ideal_humidity_max=60.0,
)

# Configure global settings
settings = configure()
settings.qa = qa_settings

# Or pass to individual components
from ptpd_calibration.qa import NegativeDensityValidator

validator = NegativeDensityValidator(settings=qa_settings)
```

## Complete Workflow Example

```python
from datetime import datetime
from PIL import Image
from ptpd_calibration.qa import (
    NegativeDensityValidator,
    ChemistryFreshnessTracker,
    PaperHumidityChecker,
    UVLightMeterIntegration,
    QualityReport,
    SolutionType,
    ReportFormat,
)

# 1. Set up tracking systems
chemistry = ChemistryFreshnessTracker()
pd_id = chemistry.register_solution(SolutionType.PALLADIUM, datetime.now(), 100.0)

humidity = PaperHumidityChecker()
humidity.measure_paper_humidity(52.0, 21.0, "Arches Platine")

uv_meter = UVLightMeterIntegration()
uv_meter.calibrate_meter()
uv_meter.read_intensity(98.0, 365.0, 200.0)

# 2. Validate negative
negative = Image.open("portrait_negative.tif")
validator = NegativeDensityValidator()
density_analysis = validator.validate_density_range(negative)

# 3. Generate pre-print checklist
report = QualityReport()
checklist = report.generate_pre_print_checklist(
    image=negative,
    chemistry_tracker=chemistry,
    humidity_checker=humidity,
    uv_meter=uv_meter,
)

# 4. Review checklist
if checklist['ready_to_print']:
    print("✓ Ready to print!")

    # Export checklist
    report.export_report(
        checklist,
        ReportFormat.HTML,
        output_path="pre_print_checklist.html"
    )
else:
    print("✗ Not ready to print")
    print("Errors:", checklist['errors'])
    print("Warnings:", checklist['warnings'])

# 5. After printing, analyze result
print_scan = Image.open("finished_print_scan.tif")
post_analysis = report.generate_post_print_analysis(
    scan=print_scan,
    expected_density_range=(0.10, 2.0)
)

print(f"Quality Score: {post_analysis['quality_score']}/100")
print(f"Grade: {post_analysis['grade']}")

# Export post-print report
report.export_report(
    post_analysis,
    ReportFormat.MARKDOWN,
    output_path="post_print_analysis.md"
)
```

## Data Models

### DensityAnalysis

```python
@dataclass
class DensityAnalysis:
    min_density: float
    max_density: float
    mean_density: float
    density_range: float
    highlight_blocked: bool
    shadow_blocked: bool
    histogram: np.ndarray
    zone_distribution: Dict[int, float]
    warnings: List[str]
    suggestions: List[str]
```

### ChemistrySolution

```python
@dataclass
class ChemistrySolution:
    solution_id: str
    solution_type: SolutionType
    date_mixed: datetime
    initial_volume_ml: float
    current_volume_ml: float
    shelf_life_days: int
    usage_log: List[Tuple[datetime, float]]
    notes: str
```

### HumidityReading

```python
@dataclass
class HumidityReading:
    timestamp: datetime
    humidity_percent: float
    temperature_celsius: Optional[float]
    paper_type: Optional[str]
    notes: str
```

### UVReading

```python
@dataclass
class UVReading:
    timestamp: datetime
    intensity: float
    wavelength: Optional[float]
    bulb_hours: Optional[float]
    notes: str
```

### Alert

```python
@dataclass
class Alert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    dismissed: bool
    dismissed_at: Optional[datetime]
```

## Best Practices

### 1. Negative Validation
- Always validate negatives before printing
- Pay attention to density range warnings
- Aim for 1.5-2.5 density range for optimal results
- Check both highlight and shadow detail

### 2. Chemistry Tracking
- Register solutions immediately after mixing
- Log usage after each print session
- Review alerts before starting a session
- Replace solutions before expiration

### 3. Paper Conditioning
- Monitor humidity for at least 24 hours before coating
- Maintain consistent ambient conditions
- Allow adequate drying time
- Keep detailed notes on paper behavior

### 4. UV Exposure
- Calibrate meter regularly
- Log readings at session start and end
- Monitor bulb degradation trends
- Plan bulb replacement before critical degradation

### 5. Quality Reports
- Generate pre-print checklists for all critical work
- Archive reports for process documentation
- Review post-print analysis for continuous improvement
- Use recommendations to refine techniques

## Troubleshooting

### Common Issues

**"Highlights blocked"**
- Reduce overall negative density
- Check exposure settings
- Verify development time

**"Shadows blocked"**
- Increase negative density
- Consider higher platinum ratio
- Check negative contrast

**"Chemistry expired"**
- Mix fresh solutions
- Review storage conditions
- Update shelf life settings if using alternative formulas

**"Paper not ready to coat"**
- Allow more drying/humidifying time
- Check ambient conditions
- Verify target humidity range is appropriate

**"UV intensity unstable"**
- Allow bulb warm-up time (10-15 minutes)
- Check bulb age and degradation
- Verify meter calibration

## Integration with Other Modules

The QA module integrates seamlessly with other ptpd_calibration modules:

```python
# With curves module
from ptpd_calibration.curves import CurveGenerator
from ptpd_calibration.qa import NegativeDensityValidator

generator = CurveGenerator()
curve = generator.generate_from_wedge(scan)

# Validate curve application
validator = NegativeDensityValidator()
analysis = validator.validate_density_range(negative_with_curve)

# With chemistry calculator
from ptpd_calibration.chemistry import ChemistryCalculator
from ptpd_calibration.qa import ChemistryFreshnessTracker

calc = ChemistryCalculator()
recipe = calc.calculate(width_inches=8, height_inches=10)

# Track the solutions
tracker = ChemistryFreshnessTracker()
# Register and track each solution component
```

## API Reference

See the main module docstrings for complete API documentation:

```python
from ptpd_calibration.qa import NegativeDensityValidator
help(NegativeDensityValidator)
```

## Examples

Complete working examples are available in:
- `/examples/qa_example.py` - Comprehensive examples for all components

## License

Part of the Platinum-Palladium AI Printing Tool. See main project LICENSE.
