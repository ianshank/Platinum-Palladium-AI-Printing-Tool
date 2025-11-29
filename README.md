# Platinum-Palladium AI Printing Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-460%20passed-brightgreen.svg)](tests/)

An AI-powered calibration system for platinum/palladium alternative photographic printing. This comprehensive toolkit helps photographers achieve consistent, high-quality Pt/Pd prints through automated calibration, intelligent curve generation, and AI-assisted guidance.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Feature Guide](#complete-feature-guide)
- [Web UI](#web-ui)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Features

### Core Calibration
- **Step Tablet Reading**: Automated detection and density extraction from scanned step tablets
- **Step Wedge Analysis**: Comprehensive step wedge analysis with quality assessment
- **Curve Generation**: Create linearization curves for digital negatives with multiple target options
- **Auto-Linearization**: Multiple algorithms (spline, polynomial, iterative) for automatic curve generation
- **Curve Editor**: Interactive curve editing with smoothing and AI enhancement
- **Multi-Format Export**: Export to QuadTone RIP, Piezography, CSV, and JSON formats

### Image Processing
- **Image Preview**: Preview curve effects on images before processing
- **Digital Negative Creation**: Create inverted negatives with curves applied
- **Batch Processing**: Process multiple images with the same curve efficiently
- **Histogram Analysis**: Zone-based tonal distribution analysis with clipping detection

### Printing Tools
- **Chemistry Calculator**: Calculate coating solutions based on Bostick-Sullivan formulas
- **Exposure Calculator**: Industry-standard UV exposure calculations with test strip generator
- **Zone System**: Ansel Adams zone analysis with development recommendations
- **Soft Proofing**: Preview prints on different paper types with Dmax/Dmin simulation
- **Paper Profiles Database**: Browse and manage paper profiles with recommended settings
- **Print Session Log**: Track prints and build process knowledge over time

### Machine Learning
- **Density Prediction**: ML models learn from your calibration history to predict results
- **Active Learning**: Intelligent suggestions for the most informative test prints
- **Transfer Learning**: Bootstrap new paper calibrations from similar known papers

### AI Assistance
- **Natural Language Chat**: Ask questions about Pt/Pd printing in plain language
- **Recipe Suggestions**: Get customized coating recipes for your paper and goals
- **Troubleshooting**: Diagnose common problems with AI-powered guidance

## Installation

### Basic Installation

```bash
pip install ptpd-calibration
```

### With Optional Dependencies

```bash
# Machine Learning features
pip install ptpd-calibration[ml]

# LLM integration (Claude/GPT)
pip install ptpd-calibration[llm]

# Gradio UI
pip install ptpd-calibration[ui]

# Everything
pip install ptpd-calibration[all]
```

### From Source

```bash
git clone https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool.git
cd Platinum-Palladium-AI-Printing-Tool
pip install -e ".[all]"
```

## Quick Start

### 1. Basic Calibration Workflow

```python
from ptpd_calibration import StepTabletReader, CurveGenerator, CurveType, save_curve

# Read your scanned step tablet
reader = StepTabletReader()
result = reader.read("step_tablet_scan.tiff")

# Check the results
densities = result.extraction.get_densities()
print(f"Dmax: {max(densities):.2f}")
print(f"Dmin: {min(densities):.2f}")

# Generate linearization curve
generator = CurveGenerator()
curve = generator.generate_from_extraction(
    result.extraction,
    curve_type=CurveType.LINEAR,
    name="My Paper - Standard",
    paper_type="Arches Platine",
)

# Export for your RIP
save_curve(curve, "my_curve.txt", format="qtr")
```

### 2. Chemistry Calculation

```python
from ptpd_calibration.chemistry import ChemistryCalculator

calculator = ChemistryCalculator()

# Calculate for an 8x10 print
recipe = calculator.calculate(
    width_inches=8.0,
    height_inches=10.0,
    platinum_ratio=0.5,  # 50/50 Pt/Pd
)

print(f"Ferric Oxalate: {recipe.ferric_oxalate_drops:.0f} drops")
print(f"Platinum:       {recipe.platinum_drops:.0f} drops")
print(f"Palladium:      {recipe.palladium_drops:.0f} drops")
print(f"Total:          {recipe.total_drops:.0f} drops ({recipe.total_ml:.2f} ml)")
```

### 3. Exposure Calculation

```python
from ptpd_calibration.exposure import ExposureCalculator, ExposureSettings

settings = ExposureSettings(
    base_exposure_minutes=10.0,
    base_negative_density=1.6,
)
calculator = ExposureCalculator(settings)

# Calculate for your negative
result = calculator.calculate(negative_density=1.8)
print(f"Exposure time: {result.format_time()}")

# Generate test strip
times = calculator.calculate_test_strip(10.0, steps=5, increment_stops=0.5)
for i, t in enumerate(times, 1):
    print(f"Strip {i}: {t:.1f} min")
```

### 4. Run the Web UI

```bash
python -m ptpd_calibration.ui.gradio_app
```

Then open http://localhost:7860

## Complete Feature Guide

### Auto-Linearization

Generate curves automatically from density measurements:

```python
from ptpd_calibration.curves import AutoLinearizer, LinearizationMethod

# Your measured densities from a step wedge
densities = [0.08, 0.15, 0.28, 0.45, 0.68, 0.95, 1.25, 1.48, 1.60]

linearizer = AutoLinearizer()
result = linearizer.linearize(
    densities,
    method=LinearizationMethod.SPLINE_FIT,
    curve_name="Auto-Linearized Curve",
)

print(f"Method: {result.method_used.value}")
print(f"Residual error: {result.residual_error:.4f}")
```

Available methods:
- `DIRECT_INVERSION`: Simple inversion of measured response
- `SPLINE_FIT`: Smooth cubic spline interpolation
- `POLYNOMIAL_FIT`: Polynomial curve fitting
- `ITERATIVE`: Iterative refinement for best fit
- `HYBRID`: Combines multiple methods

### Zone System Analysis

Analyze images using Ansel Adams' zone system:

```python
from ptpd_calibration.zones import ZoneMapper
from PIL import Image

image = Image.open("your_image.tiff")
mapper = ZoneMapper()
analysis = mapper.analyze_image(image)

print(f"Development: {analysis.development_adjustment}")
# Outputs: N-2, N-1, N, N+1, or N+2
```

### Histogram Analysis

Get detailed tonal analysis with zone distribution:

```python
from ptpd_calibration.imaging import HistogramAnalyzer
from PIL import Image

image = Image.open("your_image.tiff")
analyzer = HistogramAnalyzer()
result = analyzer.analyze(image)

print(f"Mean: {result.stats.mean:.1f}")
print(f"Dynamic range: {result.stats.dynamic_range:.0f} levels")
print(f"Brightness: {result.stats.brightness:.2f}")
print(f"Contrast: {result.stats.contrast:.2f}")
```

### Soft Proofing

Preview how your image will look on different papers:

```python
from ptpd_calibration.proofing import SoftProofer, ProofSettings, PaperSimulation
from PIL import Image

image = Image.open("your_image.tiff")

# Use a paper preset
settings = ProofSettings.from_paper_preset(PaperSimulation.ARCHES_PLATINE)
proofer = SoftProofer(settings)
result = proofer.proof(image)

# Save the proof
result.image.save("proof_arches.jpg")
```

Available paper presets:
- `ARCHES_PLATINE`: Dmax 1.6, smooth surface
- `BERGGER_COT320`: Dmax 1.55, cotton rag
- `HAHNEMUHLE_PLATINUM`: Dmax 1.65, fine art paper
- `STONEHENGE`: Dmax 1.4, warm tone
- `REVERE_PLATINUM`: Dmax 1.5, bright white

### Paper Profiles

Browse and use paper characteristics:

```python
from ptpd_calibration.papers import PaperDatabase

db = PaperDatabase()

# List all papers
for paper in db.list_papers():
    print(f"{paper.name}: Dmax {paper.characteristics.typical_dmax}")

# Get specific paper
arches = db.get_paper("arches_platine")
print(f"Sizing: {arches.characteristics.sizing}")
print(f"Surface: {arches.characteristics.surface}")
```

### Digital Negative Creation

Create print-ready negatives with curve correction:

```python
from ptpd_calibration.imaging import ImageProcessor
from ptpd_calibration.core.models import CurveData

processor = ImageProcessor()

# Create negative with curve
result = processor.create_digital_negative(
    "your_image.tiff",
    curve=your_curve,
    invert=True,
)

result.image.save("negative.tiff")
```

### Print Session Logging

Track your prints over time:

```python
from ptpd_calibration.session import SessionLogger, PrintRecord
from ptpd_calibration.session.logger import ChemistryUsed, PrintResult

logger = SessionLogger()
session = logger.start_session("Evening Print Session")

# Log a print
record = PrintRecord(
    image_name="Portrait Study",
    paper_type="Arches Platine",
    exposure_time_minutes=12.0,
    chemistry=ChemistryUsed(
        ferric_oxalate_drops=15.0,
        platinum_drops=8.0,
        palladium_drops=7.0,
    ),
    result=PrintResult.EXCELLENT,
)
logger.log_print(record)

# Get statistics
stats = logger.get_current_session().get_statistics()
print(f"Total prints: {stats['total_prints']}")
print(f"Success rate: {stats['success_rate']}")
```

### Curve Export Formats

Export curves in multiple formats:

```python
from ptpd_calibration.curves import save_curve

# QuadTone RIP format
save_curve(curve, "curve.txt", format="qtr")

# CSV for spreadsheets
save_curve(curve, "curve.csv", format="csv")

# JSON with full metadata
save_curve(curve, "curve.json", format="json")
```

## Web UI

The Gradio web interface provides 21 tabs for all functionality:

| Tab | Description |
|-----|-------------|
| Step Tablet | Read and analyze step tablet scans |
| Curve Editor | Create and edit linearization curves |
| AI Enhance | Improve curves with AI suggestions |
| Step Wedge Analysis | Comprehensive quality analysis |
| Curve Visualizer | View curve statistics |
| Curve Manager | Save, load, and export curves |
| Scanner Calibration | Calibrate your scanner |
| Chemistry Calculator | Calculate coating recipes |
| Settings | Configure API keys and preferences |
| Image Preview | Preview curve effects |
| Digital Negative | Create print-ready negatives |
| Batch Processing | Process multiple images |
| Histogram Analysis | Zone-based image analysis |
| Exposure Calculator | Calculate UV exposure times |
| Zone System | Ansel Adams zone mapping |
| Soft Proofing | Preview on different papers |
| Paper Profiles | Browse paper characteristics |
| Auto-Linearization | Automatic curve generation |
| Print Session Log | Track prints over time |
| AI Assistant | Chat about Pt/Pd printing |
| About | Help and information |

Launch the UI:

```bash
python -m ptpd_calibration.ui.gradio_app
```

## Configuration

Configure via environment variables with the `PTPD_` prefix:

```bash
# LLM Configuration
export PTPD_LLM_API_KEY="sk-ant-..."
export PTPD_LLM_PROVIDER="anthropic"

# Detection Settings
export PTPD_DETECTION_CANNY_LOW_THRESHOLD=50
export PTPD_DETECTION_CANNY_HIGH_THRESHOLD=150

# Curve Settings
export PTPD_CURVE_NUM_OUTPUT_POINTS=256
export PTPD_CURVE_DEFAULT_EXPORT_FORMAT="qtr"
```

Or use a `.env` file in your project directory.

## Project Structure

```
src/ptpd_calibration/
├── __init__.py              # Main API exports
├── config.py                # Configuration management
├── core/                    # Core data models
│   ├── models.py           # Pydantic models
│   └── types.py            # Domain types
├── detection/              # Step tablet detection
│   ├── detector.py         # Patch detection
│   ├── extractor.py        # Density extraction
│   ├── reader.py           # High-level reader
│   └── scanner.py          # Scanner calibration
├── curves/                 # Curve generation
│   ├── generator.py        # Curve calculation
│   ├── export.py           # Format export
│   ├── analysis.py         # Curve analysis
│   ├── visualization.py    # Curve display & statistics
│   ├── linearization.py    # Auto-linearization algorithms
│   └── modifier.py         # Curve editing & smoothing
├── imaging/                # Image processing
│   ├── processor.py        # Digital negative creation
│   └── histogram.py        # Histogram & zone analysis
├── chemistry/              # Chemistry calculations
│   └── calculator.py       # Coating recipe calculator
├── exposure/               # Exposure tools
│   └── calculator.py       # UV exposure calculator
├── zones/                  # Zone System
│   └── mapping.py          # Ansel Adams zone mapping
├── proofing/               # Soft proofing
│   └── simulation.py       # Print preview simulation
├── papers/                 # Paper database
│   └── profiles.py         # Paper profiles & settings
├── session/                # Session management
│   └── logger.py           # Print session logging
├── analysis/               # Analysis tools
│   └── wedge_analyzer.py   # Step wedge analysis
├── ml/                     # Machine learning
│   ├── database.py         # Calibration storage
│   ├── predictor.py        # ML prediction
│   ├── active_learning.py  # Experiment suggestion
│   └── transfer.py         # Transfer learning
├── llm/                    # LLM integration
│   ├── client.py           # API clients
│   ├── assistant.py        # Chat assistant
│   └── prompts.py          # Domain prompts
├── agents/                 # Agentic system
│   ├── agent.py            # Main agent
│   ├── tools.py            # Tool definitions
│   └── memory.py           # Persistent memory
└── ui/                     # User interfaces
    └── gradio_app.py       # Gradio UI (21 tabs)
```

## Examples

Run the comprehensive demo to see all features:

```bash
python examples/comprehensive_demo.py
```

Run the quick start guide:

```bash
python examples/quick_start.py
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool.git
cd Platinum-Palladium-AI-Printing-Tool
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,all]"
```

### Running Tests

```bash
# All tests (460+ tests)
pytest

# With coverage
pytest --cov=src/ptpd_calibration --cov-report=html

# E2E user journey tests
pytest tests/e2e/test_user_journeys.py -v

# Sanity tests for deployment
pytest tests/sanity/ -v
```

### Test Coverage

- Unit tests: 427 tests covering all modules
- E2E tests: 11 complete user journey tests
- Sanity tests: 25 deployment verification tests

## Supported Formats

### Step Tablets
- Stouffer 21/31/41-step transmission wedge
- Custom tablets (auto-detection)

### Image Formats
- TIFF (recommended)
- PNG, JPEG
- Most formats supported by Pillow

### Curve Export
- **QuadTone RIP**: `.txt` and `.quad` profiles
- **Piezography**: `.ppt` format
- **CSV**: Standard comma-separated values
- **JSON**: Full metadata preservation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- The alternative photography community for process knowledge
- Anthropic and OpenAI for LLM APIs
- The developers of QuadTone RIP and Piezography

## Support

- **Issues**: [GitHub Issues](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/discussions)
