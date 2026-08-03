# Configuration System Guide

## Overview

The PTPD Calibration system uses **pydantic-settings** for robust, validated configuration management. All settings are environment-based with sensible defaults, enabling deployment flexibility without code changes.

## Quick Start

### Basic Usage

```python
from ptpd_calibration.config import get_settings

# Get configuration (singleton)
settings = get_settings()

# Access nested settings
print(settings.api.host)  # "0.0.0.0"
print(settings.detection.canny_low_threshold)  # 50
print(settings.llm.provider)  # "anthropic"
```

### Environment Variables

Override any setting with `PTPD_` prefixed environment variables:

```bash
# Set nested settings with double underscore
export PTPD_API__HOST=127.0.0.1
export PTPD_API__PORT=8001
export PTPD_DETECTION__CANNY_LOW_THRESHOLD=75
export PTPD_LLM__PROVIDER=openai

# Run application
uvicorn src.ptpd_calibration.api.server:app --reload
```

### Configuration Files

For complex deployments, load from dotenv files:

```bash
# .env in project root
PTPD_API__RELOAD=true
PTPD_DETECTION__MORPH_KERNEL_SIZE=7
PTPD_LLM__ANTHROPIC_API_KEY=sk-ant-...
```

The `load_dotenv()` call in `config.py` automatically loads these on import.

## Settings Structure

### Root Settings Class

```
Settings (root)
├── api              # API server configuration
├── detection        # Step tablet detection
├── extraction       # Density extraction
├── curves           # Curve generation
├── llm              # Language model settings
├── ml               # Machine learning
├── monitoring       # Performance monitoring
└── data             # Data storage and paths
```

### Settings Classes Reference

#### `APISettings`

Server configuration and CORS settings:

```python
from ptpd_calibration.config import get_settings
settings = get_settings()

# Access API settings
print(settings.api.host)              # "0.0.0.0"
print(settings.api.port)              # 8000
print(settings.api.reload)            # False (True in dev)
print(settings.api.cors_origins)      # ["http://localhost:3000"]
print(settings.api.cors_allow_credentials)  # True
print(settings.api.upload_dir)        # Path object or None
print(settings.api.max_upload_size)   # 104857600 (100MB)
```

Environment variables:
```bash
PTPD_API__HOST=0.0.0.0
PTPD_API__PORT=8000
PTPD_API__RELOAD=false
PTPD_API__CORS_ORIGINS=http://localhost:3000,https://example.com
PTPD_API__CORS_ALLOW_CREDENTIALS=true
PTPD_API__UPLOAD_DIR=/tmp/uploads
PTPD_API__MAX_UPLOAD_SIZE=104857600
```

#### `DetectionSettings`

Step tablet detection parameters:

```python
settings = get_settings()

# Edge detection
print(settings.detection.canny_low_threshold)    # 50 (0-255)
print(settings.detection.canny_high_threshold)   # 150 (0-255)

# Morphological operations
print(settings.detection.morph_kernel_size)      # 5 (1-21, odd)
print(settings.detection.morph_iterations)       # 2 (1-10)

# Contour detection
print(settings.detection.min_contour_area_ratio)  # 0.01 (0.1%-50%)
print(settings.detection.max_contour_area_ratio)  # 0.95 (50%-100%)

# Rotation correction
print(settings.detection.max_rotation_angle)      # 15.0 degrees
print(settings.detection.rotation_threshold)      # 0.5
```

Environment variables:
```bash
PTPD_DETECTION__CANNY_LOW_THRESHOLD=50
PTPD_DETECTION__CANNY_HIGH_THRESHOLD=150
PTPD_DETECTION__MORPH_KERNEL_SIZE=5
PTPD_DETECTION__MORPH_ITERATIONS=2
PTPD_DETECTION__MIN_CONTOUR_AREA_RATIO=0.01
PTPD_DETECTION__MAX_CONTOUR_AREA_RATIO=0.95
PTPD_DETECTION__MAX_ROTATION_ANGLE=15.0
PTPD_DETECTION__ROTATION_THRESHOLD=0.5
```

#### `ExtractionSettings`

Density and color extraction parameters:

```python
settings = get_settings()

# Sampling parameters
print(settings.extraction.sample_margin_ratio)         # 0.15 (margin %)
print(settings.extraction.min_sample_pixels)           # 100 pixels

# Outlier rejection
print(settings.extraction.outlier_rejection_method)    # "mad"
print(settings.extraction.mad_threshold)               # 3.0 standard deviations

# Density calculation
print(settings.extraction.reference_white_reflectance) # 0.9
print(settings.extraction.status_a_weights)            # (0.2126, 0.7152, 0.0722)

# Paper base detection
print(settings.extraction.paper_margin_ratio)          # 0.05
```

Environment variables:
```bash
PTPD_EXTRACTION__SAMPLE_MARGIN_RATIO=0.15
PTPD_EXTRACTION__MIN_SAMPLE_PIXELS=100
PTPD_EXTRACTION__OUTLIER_REJECTION_METHOD=mad
PTPD_EXTRACTION__MAD_THRESHOLD=3.0
PTPD_EXTRACTION__REFERENCE_WHITE_REFLECTANCE=0.9
PTPD_EXTRACTION__PAPER_MARGIN_RATIO=0.05
```

#### `LLMSettings`

Language model provider and credentials:

```python
settings = get_settings()

# Provider and models
print(settings.llm.provider)              # "anthropic" | "openai" | "vertex_ai"
print(settings.llm.model)                 # "claude-3-sonnet" | "gpt-4" etc.
print(settings.llm.temperature)           # 0.7 (0.0-1.0)
print(settings.llm.max_tokens)            # 2000

# API Keys (loaded from environment or .env)
print(settings.llm.anthropic_api_key)     # Hidden in logs
print(settings.llm.openai_api_key)        # Hidden in logs
```

Environment variables:
```bash
PTPD_LLM__PROVIDER=anthropic
PTPD_LLM__MODEL=claude-3-sonnet-20240229
PTPD_LLM__TEMPERATURE=0.7
PTPD_LLM__MAX_TOKENS=2000
PTPD_LLM__ANTHROPIC_API_KEY=sk-ant-...
PTPD_LLM__OPENAI_API_KEY=sk-...
```

#### `CurveSettings`

Curve generation and export parameters:

```python
settings = get_settings()

# Interpolation
print(settings.curves.interpolation_method)  # "cubic"
print(settings.curves.smoothing_method)      # "gaussian"

# Bounds
print(settings.curves.min_curve_points)      # 5
print(settings.curves.max_curve_points)      # 256

# Export formats
print(settings.curves.supported_formats)     # ["qtr", "piezography", "csv", "json"]
```

#### `MLSettings`

Machine learning model parameters:

```python
settings = get_settings()

# Model loading
print(settings.ml.model_cache_dir)      # Path to cached models
print(settings.ml.use_gpu)              # True if CUDA available

# Training
print(settings.ml.batch_size)           # 32
print(settings.ml.learning_rate)        # 0.001
print(settings.ml.epochs)               # 100
```

#### `MonitoringSettings`

Performance monitoring configuration:

```python
settings = get_settings()

# Metrics
print(settings.monitoring.enable_metrics)    # True
print(settings.monitoring.metrics_interval)  # 60 seconds
print(settings.monitoring.slow_request_threshold_ms)  # 1000

# Profiling
print(settings.monitoring.enable_profiling)  # False in production
print(settings.monitoring.profile_samples)   # 1000
```

#### `DataSettings`

Data storage and paths:

```python
settings = get_settings()

# Directories
print(settings.data.data_dir)           # Path("./data")
print(settings.data.cache_dir)          # Path("./data/cache")
print(settings.data.models_dir)         # Path("./data/models")

# Retention
print(settings.data.retention_days)     # 90
print(settings.data.backup_enabled)     # True
```

## Validation and Constraints

All settings are validated with type checking and constraints:

```python
from pydantic import Field, field_validator, ValidationInfo

class DetectionSettings(BaseSettings):
    # Range constraints
    canny_low_threshold: int = Field(default=50, ge=0, le=255)
    
    # Custom validation
    @field_validator('morph_kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        if v % 2 == 0:
            raise ValueError('Kernel size must be odd')
        return v
```

This means invalid environment variables raise errors on startup:

```bash
# Invalid: threshold out of range
export PTPD_DETECTION__CANNY_LOW_THRESHOLD=300  # Error: <= 255

# Invalid: kernel size even
export PTPD_DETECTION__MORPH_KERNEL_SIZE=6  # Error: must be odd
```

## Adding New Settings

### Step 1: Define Settings Class

```python
# In config.py

from pydantic import BaseSettings, Field, SettingsConfigDict

class NewFeatureSettings(BaseSettings):
    """Settings for new feature."""
    
    model_config = SettingsConfigDict(env_prefix="PTPD_NEWFEATURE_")
    
    enabled: bool = Field(default=True)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_items: int = Field(default=100, ge=1)
```

### Step 2: Add to Root Settings

```python
class Settings(BaseSettings):
    """Root configuration."""
    
    # ... existing settings ...
    
    # NEW
    newfeature: NewFeatureSettings = Field(default_factory=NewFeatureSettings)
```

### Step 3: Use in Code

```python
from ptpd_calibration.config import get_settings

settings = get_settings()
if settings.newfeature.enabled:
    process_with_threshold(settings.newfeature.threshold)
```

### Step 4: Document Environment Variables

In your .env or deployment docs:

```bash
# New Feature Settings
PTPD_NEWFEATURE__ENABLED=true
PTPD_NEWFEATURE__THRESHOLD=0.5
PTPD_NEWFEATURE__MAX_ITEMS=100
```

## Best Practices

### 1. Use Type Hints

```python
# Good: Clear types with validation
threshold: float = Field(default=0.5, ge=0.0, le=1.0)

# Avoid: Ambiguous types
threshold = 0.5
```

### 2. Provide Sensible Defaults

```python
# Good: Reasonable defaults
model: str = Field(default="claude-3-sonnet-20240229")

# Avoid: Requiring environment variables
model: str  # Raises error if not set
```

### 3. Group Related Settings

```python
# Good: Nested settings class
class LLMSettings(BaseSettings):
    provider: str
    model: str
    temperature: float
    api_key: str

class Settings(BaseSettings):
    llm: LLMSettings

# Usage: settings.llm.provider

# Avoid: Flat structure
class Settings(BaseSettings):
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_api_key: str
```

### 4. Use Path for File Paths

```python
# Good: Type safety
data_dir: Path = Field(default=Path("./data"))
model_path = settings.data_dir / "model.pt"

# Avoid: String paths
data_dir: str = "./data"
model_path = f"{settings.data_dir}/model.pt"
```

### 5. Hide Sensitive Data

```python
# Good: Pydantic hides in logs
class LLMSettings(BaseSettings):
    api_key: str = Field(default="", exclude=True)  # Won't show in repr

# Avoid: Exposing secrets
api_key = os.environ.get("PTPD_API_KEY")
print(settings)  # Would leak the key
```

## Testing with Configuration

### Override Settings in Tests

```python
import pytest
from ptpd_calibration.config import get_settings, Settings

@pytest.fixture
def test_settings():
    """Provide test configuration."""
    return Settings(
        detection=DetectionSettings(canny_low_threshold=100),
        api=APISettings(reload=False)
    )

def test_detection_with_custom_threshold(test_settings):
    detector = StepTabletDetector()
    # Use test_settings for the test
```

### Mock Configuration

```python
from unittest.mock import patch

def test_with_mocked_config():
    with patch('ptpd_calibration.config.get_settings') as mock:
        mock.return_value.llm.provider = "openai"
        # Test with mocked configuration
```

## Deployment Scenarios

### Local Development

```bash
# .env
PTPD_API__RELOAD=true
PTPD_API__HOST=127.0.0.1
PTPD_LLM__PROVIDER=anthropic
PTPD_LLM__ANTHROPIC_API_KEY=sk-ant-...
PTPD_DATA__DATA_DIR=./data-dev
```

### Docker Container

```dockerfile
ENV PTPD_API__HOST=0.0.0.0
ENV PTPD_API__RELOAD=false
ENV PTPD_API__CORS_ORIGINS=https://app.example.com
ENV PTPD_DATA__DATA_DIR=/app/data
```

### Kubernetes with ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ptpd-config
data:
  PTPD_API__HOST: "0.0.0.0"
  PTPD_API__PORT: "8000"
  PTPD_DETECTION__CANNY_LOW_THRESHOLD: "50"
```

### Cloud Deployment (GCP)

```bash
# Using Cloud Run environment variables
gcloud run deploy ptpd-api \
  --set-env-vars="PTPD_API__RELOAD=false" \
  --set-env-vars="PTPD_LLM__PROVIDER=vertex_ai" \
  --set-env-vars="PTPD_DATA__DATA_DIR=/workspace/data"
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Configuration as part of system design
- [LOGGING.md](LOGGING.md) - Logging configuration integration
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Error configuration
- [src/ptpd_calibration/config.py](../src/ptpd_calibration/config.py) - Full implementation
