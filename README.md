# Platinum-Palladium AI Printing Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered calibration system for platinum/palladium alternative photographic printing. This comprehensive toolkit helps photographers achieve consistent, high-quality Pt/Pd prints through automated calibration, intelligent curve generation, and AI-assisted guidance.

## Features

### Core Calibration
- **Step Tablet Reading**: Automated detection and density extraction from scanned step tablets
- **Curve Generation**: Create linearization curves for digital negatives with multiple target options
- **Multi-Format Export**: Export to QuadTone RIP, Piezography, CSV, and JSON formats
- **Scanner Calibration**: Profile your scanner for accurate measurements

### Machine Learning
- **Density Prediction**: ML models learn from your calibration history to predict results
- **Active Learning**: Intelligent suggestions for the most informative test prints
- **Transfer Learning**: Bootstrap new paper calibrations from similar known papers
- **Pattern Recognition**: Identify trends and optimize your workflow

### AI Assistance
- **Natural Language Chat**: Ask questions about Pt/Pd printing in plain language
- **Recipe Suggestions**: Get customized coating recipes for your paper and goals
- **Troubleshooting**: Diagnose common problems with AI-powered guidance
- **Calibration Analysis**: Automatic analysis of your results with recommendations

### Agentic System
- **Autonomous Agents**: Let AI handle complex calibration workflows
- **ReAct Reasoning**: Transparent thought → action → observation loops
- **Tool Integration**: 15+ specialized tools for calibration tasks
- **Persistent Memory**: Agents learn and remember across sessions

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

# Web API server
pip install ptpd-calibration[api]

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

### Basic Calibration Workflow

```python
from ptpd_calibration import (
    StepTabletReader,
    CurveGenerator,
    CurveType,
    QTRExporter,
)
from pathlib import Path

# 1. Read step tablet scan
reader = StepTabletReader()
result = reader.read("step_tablet_scan.tiff")

print(f"Dmin: {result.extraction.dmin:.2f}")
print(f"Dmax: {result.extraction.dmax:.2f}")
print(f"Range: {result.extraction.density_range:.2f}")

# 2. Generate linearization curve
generator = CurveGenerator()
curve = generator.generate_from_extraction(
    result.extraction,
    curve_type=CurveType.LINEAR,
    name="Arches Platine - Standard",
    paper_type="Arches Platine",
    chemistry="50% Pt / 50% Pd, 5 drops Na2",
)

# 3. Export for your RIP
exporter = QTRExporter()
exporter.export(curve, Path("arches_linear.txt"))

print("Curve exported successfully!")
```

### Using the AI Assistant

```python
import asyncio
from ptpd_calibration import create_assistant

async def main():
    # Create assistant (requires PTPD_LLM_API_KEY env var)
    assistant = create_assistant()

    # Ask questions
    response = await assistant.chat(
        "What metal ratio should I use for warm-toned portraits?"
    )
    print(response)

    # Get recipe suggestions
    recipe = await assistant.suggest_recipe(
        "Bergger COT320",
        "Maximum density range with good shadow detail"
    )
    print(recipe)

    # Troubleshoot problems
    help_text = await assistant.troubleshoot(
        "My highlights are blocking up and I'm seeing yellow staining"
    )
    print(help_text)

asyncio.run(main())
```

### ML-Powered Predictions

```python
from ptpd_calibration import (
    CalibrationDatabase,
    CalibrationRecord,
    CurvePredictor,
    ActiveLearner,
)
from pathlib import Path

# Load your calibration history
db = CalibrationDatabase.load(Path("my_calibrations.json"))

# Train prediction model
predictor = CurvePredictor()
stats = predictor.train(db)
print(f"Validation MAE: {stats['validation_mae']:.4f}")

# Predict for new setup
new_setup = CalibrationRecord(
    paper_type="New Japanese Paper",
    exposure_time=200.0,
    metal_ratio=0.4,
)

predicted_densities, uncertainty = predictor.predict(
    new_setup,
    return_uncertainty=True
)
print(f"Predicted Dmax: {max(predicted_densities):.2f} ± {uncertainty:.2f}")

# Get suggestions for next experiment
learner = ActiveLearner(predictor)
suggestion = learner.suggest_next_experiment(
    new_setup,
    variations=[
        {"metal_ratio": 0.3},
        {"metal_ratio": 0.5},
        {"exposure_time": 180},
        {"exposure_time": 220},
    ]
)
print(f"Suggestion: {suggestion['rationale']}")
```

### Running the Web UI

```bash
# Start the Gradio interface
python -m ptpd_calibration.ui.gradio_app

# Or start the API server
python -m ptpd_calibration.api.server
```

Then open http://localhost:7860 (Gradio) or http://localhost:8000 (API).

## Configuration

All settings can be configured via environment variables with the `PTPD_` prefix:

```bash
# LLM Configuration
export PTPD_LLM_API_KEY="sk-ant-..."
export PTPD_LLM_PROVIDER="anthropic"  # or "openai"
export PTPD_LLM_ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# Detection Settings
export PTPD_DETECTION_CANNY_LOW_THRESHOLD=50
export PTPD_DETECTION_CANNY_HIGH_THRESHOLD=150

# Curve Settings
export PTPD_CURVE_NUM_OUTPUT_POINTS=256
export PTPD_CURVE_DEFAULT_EXPORT_FORMAT="qtr"

# API Settings
export PTPD_API_HOST="0.0.0.0"
export PTPD_API_PORT=8000
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
│   └── analysis.py         # Curve analysis
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
│   ├── memory.py           # Persistent memory
│   └── planning.py         # Task planning
├── api/                    # REST API
│   └── server.py           # FastAPI server
└── ui/                     # User interfaces
    └── gradio_app.py       # Gradio UI
```

## Supported Formats

### Step Tablets
- Stouffer 21-step transmission wedge
- Stouffer 31-step transmission wedge
- Stouffer 41-step transmission wedge
- Custom tablets (auto-detection)

### Image Formats
- TIFF (recommended for accuracy)
- PNG
- JPEG
- Most formats supported by Pillow

### Curve Export
- **QuadTone RIP**: `.txt` and `.quad` profiles
- **Piezography**: `.ppt` format
- **CSV**: Standard comma-separated values
- **JSON**: Full metadata preservation

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool.git
cd Platinum-Palladium-AI-Printing-Tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/ptpd_calibration --cov-report=html

# Specific test file
pytest tests/unit/test_curves.py

# Verbose output
pytest -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The alternative photography community for process knowledge
- Anthropic and OpenAI for LLM APIs
- The developers of QuadTone RIP and Piezography

## Support

- **Issues**: [GitHub Issues](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/discussions)
