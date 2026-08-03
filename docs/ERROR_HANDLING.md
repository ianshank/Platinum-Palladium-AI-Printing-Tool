# Error Handling Guide

## Overview

The PTPD Calibration system uses a unified exception hierarchy that enables predictable error handling, consistent logging, and proper HTTP status code mapping.

## Exception Hierarchy

```
Exception (Python built-in)
├── PTCDError (base for all PTPD exceptions)
│   ├── ConfigurationError
│   │   └── InvalidSettingsError
│   │
│   ├── DetectionError (detection/)
│   │   ├── TabletDetectionError
│   │   ├── PatchExtractionError
│   │   └── ScannerCalibrationError
│   │
│   ├── CurveError (curves/)
│   │   ├── CurveGenerationError
│   │   ├── CurveExportError
│   │   ├── CurveModificationError
│   │   └── CurveLoadError
│   │
│   ├── LLMError (llm/)
│   │   ├── ProviderError
│   │   ├── APIError
│   │   └── ModelNotFoundError
│   │
│   ├── AgentError (agents/)
│   │   ├── ToolExecutionError
│   │   └── PlanningError
│   │
│   └── ... (domain-specific errors)
```

## Using the Exception Hierarchy

### Define Custom Exceptions

Each domain module should define its own exception base and specific exceptions:

```python
# src/ptpd_calibration/detection/exceptions.py

from ptpd_calibration.core.exceptions import PTCDError

class DetectionError(PTCDError):
    """Base exception for detection module."""
    
    error_code: str = "DETECTION_ERROR"
    http_status_code: int = 400

class TabletDetectionError(DetectionError):
    """Raised when step tablet cannot be detected."""
    
    error_code: str = "TABLET_NOT_FOUND"
    http_status_code: int = 422  # Unprocessable entity

class PatchExtractionError(DetectionError):
    """Raised when patch extraction fails."""
    
    error_code: str = "PATCH_EXTRACTION_FAILED"
    http_status_code: int = 422
```

### Raise Domain-Specific Exceptions

```python
from ptpd_calibration.detection.exceptions import TabletDetectionError
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

class StepTabletDetector:
    def detect(self, image):
        """Detect step tablet in image."""
        try:
            # Detection logic
            if not self._find_tablet_contours(image):
                raise TabletDetectionError(
                    "No tablet contours found in image. "
                    "Ensure image is well-lit and tablet is fully visible."
                )
        except Exception as e:
            logger.error(
                "Tablet detection failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "image_shape": image.shape,
                }
            )
            raise
```

### Catch and Handle Exceptions

```python
# In FastAPI route

from fastapi import HTTPException
from ptpd_calibration.detection.exceptions import TabletDetectionError

@app.post("/scan/upload")
async def upload_scan(file: UploadFile):
    """Upload and analyze step tablet scan."""
    try:
        image = await load_image(file)
        result = detector.detect(image)
        return {"status": "success", "patches": result.patches}
    
    except TabletDetectionError as e:
        logger.warning("Tablet detection returned user-friendly error")
        raise HTTPException(
            status_code=e.http_status_code,
            detail={
                "error": e.error_code,
                "message": str(e),
            }
        )
    
    except Exception as e:
        logger.error("Unexpected error during scan upload", extra={"error": str(e)})
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again.",
            }
        )
```

## HTTP Status Code Mapping

### Standard Status Codes

| HTTP Status | Use Case | Exception Example |
|-------------|----------|-------------------|
| 400 (Bad Request) | Invalid input format | `ValueError` |
| 401 (Unauthorized) | Missing/invalid credentials | `AuthenticationError` |
| 403 (Forbidden) | User lacks permission | `PermissionError` |
| 404 (Not Found) | Resource doesn't exist | `CurveNotFoundError` |
| 409 (Conflict) | Resource already exists | `DuplicateError` |
| 422 (Unprocessable Entity) | Input valid but unsupported | `TabletDetectionError` |
| 429 (Too Many Requests) | Rate limited | `RateLimitError` |
| 500 (Internal Server Error) | Unexpected server error | `Exception` (unhandled) |
| 503 (Service Unavailable) | External service down | `LLMProviderError` |

### Exception to Status Code Mapping

```python
# In core/exceptions.py

HTTP_STATUS_CODE_MAP = {
    "VALIDATION_ERROR": 400,
    "MALFORMED_REQUEST": 400,
    "INVALID_SETTINGS": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "DUPLICATE_ERROR": 409,
    "TABLET_NOT_FOUND": 422,
    "PATCH_EXTRACTION_FAILED": 422,
    "CURVE_NOT_FOUND": 404,
    "CURVE_EXPORT_ERROR": 422,
    "LLM_PROVIDER_ERROR": 503,
    "LLM_API_ERROR": 503,
    "RATE_LIMITED": 429,
    "INTERNAL_SERVER_ERROR": 500,
}
```

## Best Practices

### 1. Be Specific in Error Messages

```python
# Good: Clear, actionable error messages
if density < 0:
    raise ValueError(
        f"Density must be non-negative, got {density}. "
        "Density ranges from 0 (white) to 4.0 (black)."
    )

if not image.data:
    raise TabletDetectionError(
        "Image data is empty. "
        "Make sure the uploaded image file is not corrupted."
    )

# Avoid: Vague errors
if not image.data:
    raise ValueError("Invalid image")
```

### 2. Add Context to Errors

```python
# Good: Include relevant context
try:
    curve = load_curve(path)
except FileNotFoundError as e:
    raise CurveLoadError(
        f"Curve file not found at {path}. "
        f"Expected path format: {EXPECTED_FORMAT}"
    ) from e

# Avoid: Swallowing original error
except FileNotFoundError:
    raise CurveLoadError("File not found")
```

### 3. Use Exception Chaining

```python
# Good: Preserve original traceback
try:
    result = llm_client.generate_text(prompt)
except OpenAIError as e:
    raise LLMError(f"LLM generation failed: {e}") from e

# Avoid: Losing original error
except OpenAIError:
    raise LLMError("LLM generation failed")
```

### 4. Avoid Catching Too Broad

```python
# Good: Catch specific exceptions
try:
    value = float(user_input)
except ValueError:
    raise ValidationError(f"Expected number, got '{user_input}'")

# Avoid: Catching all exceptions
try:
    value = float(user_input)
except Exception:  # Too broad!
    raise ValidationError("Invalid value")
```

### 5. Provide Recovery Suggestions

```python
# Good: Help users recover
def validate_tablet_image(image):
    if image.width < MIN_WIDTH:
        raise TabletDetectionError(
            f"Image width {image.width}px is below minimum {MIN_WIDTH}px. "
            "Try: (1) Using a higher resolution scan, "
            "(2) Scanning at 600+ DPI, or "
            "(3) Using a larger step tablet."
        )

# Avoid: Just stating the problem
raise TabletDetectionError("Image too small")
```

## Error Logging Strategy

### Structured Error Logging

```python
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

try:
    result = process_densities(densities)
except DensityValidationError as e:
    logger.warning(
        "Density validation failed",
        extra={
            "error_code": e.error_code,
            "error_message": str(e),
            "densities": densities,
            "user_id": user_id,
            "request_id": request_id,
        }
    )
    raise

except Exception as e:
    logger.error(
        "Unexpected error during processing",
        extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "densities": densities,
            "request_id": request_id,
        }
    )
    raise
```

### Sentry Integration (Optional)

```python
import sentry_sdk

# Initialize Sentry
sentry_sdk.init(
    dsn="https://examplePublicKey@sentry.example.com/123456",
    environment="production",
    traces_sample_rate=0.1,
)

# Capture exceptions automatically
try:
    result = risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

## FastAPI Error Response Format

### Standard Error Response

```python
# Error response from API

{
    "error": "TABLET_NOT_FOUND",
    "message": "No tablet contours found in image. Ensure image is well-lit...",
    "status_code": 422,
    "timestamp": "2026-08-03T12:34:56Z",
    "request_id": "req_1234567890",
    "details": {
        "image_width": 1920,
        "image_height": 1440,
        "min_contours_found": 0
    }
}
```

### Create Custom Exception Handler

```python
# In api/server.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ptpd_calibration.core.exceptions import PTCDError

app = FastAPI()

@app.exception_handler(PTCDError)
async def ptpd_error_handler(request: Request, exc: PTCDError):
    """Handle PTPD-specific errors with proper HTTP status codes."""
    return JSONResponse(
        status_code=getattr(exc, 'http_status_code', 500),
        content={
            "error": getattr(exc, 'error_code', 'INTERNAL_ERROR'),
            "message": str(exc),
            "status_code": getattr(exc, 'http_status_code', 500),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("x-request-id", "unknown"),
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(
        "Unhandled exception",
        extra={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "path": request.url.path,
            "request_id": request.headers.get("x-request-id"),
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("x-request-id", "unknown"),
        }
    )
```

## Testing Error Handling

### Unit Tests

```python
import pytest
from ptpd_calibration.detection.exceptions import TabletDetectionError

def test_detection_raises_on_empty_image():
    """Test that detection raises appropriate error."""
    detector = StepTabletDetector()
    empty_image = Image.new('RGB', (100, 100), color='white')
    
    with pytest.raises(TabletDetectionError) as exc_info:
        detector.detect(empty_image)
    
    assert "No tablet contours found" in str(exc_info.value)

def test_detection_error_has_http_status_code():
    """Test error contains correct HTTP status code."""
    error = TabletDetectionError("Test error")
    assert error.http_status_code == 422
    assert error.error_code == "TABLET_NOT_FOUND"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_upload_scan_returns_422_on_detection_error(client):
    """Test API returns 422 when tablet detection fails."""
    # Create invalid image
    invalid_image = Image.new('RGB', (100, 100), color='white')
    
    response = await client.post(
        "/scan/upload",
        files={"file": ("blank.png", invalid_image.tobytes())}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "TABLET_NOT_FOUND"
    assert "contours" in data["message"].lower()
```

## Common Error Patterns

### Pattern 1: Validation Error with Recovery Suggestions

```python
class DensityValidationError(PTCDError):
    """Density value is outside expected range."""
    
    error_code = "INVALID_DENSITY"
    http_status_code = 422
    
    def __init__(self, value: float, valid_range: tuple[float, float]):
        min_val, max_val = valid_range
        super().__init__(
            f"Density {value} is outside valid range [{min_val}, {max_val}]. "
            "Typical densities range from 0 (white) to 4.0 (black). "
            "Check your scanning equipment calibration."
        )
        self.value = value
        self.valid_range = valid_range
```

### Pattern 2: Optional Dependency Error

```python
class MissingOptionalDependencyError(PTCDError):
    """Required optional dependency is not installed."""
    
    error_code = "MISSING_DEPENDENCY"
    http_status_code = 503  # Service unavailable
    
    def __init__(self, feature: str, package: str):
        super().__init__(
            f"{feature} requires {package}. "
            f"Install with: pip install ptpd-calibration[{feature.lower()}]"
        )
        self.feature = feature
        self.package = package
```

### Pattern 3: Timeout Error

```python
class OperationTimeoutError(PTCDError):
    """Operation exceeded time limit."""
    
    error_code = "TIMEOUT"
    http_status_code = 504  # Gateway timeout
    
    def __init__(self, operation: str, timeout_seconds: int):
        super().__init__(
            f"{operation} exceeded {timeout_seconds}s timeout. "
            "Try: (1) Reducing input size, (2) Increasing timeout, "
            "(3) Running on a faster machine."
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Error handling as part of system design
- [LOGGING.md](LOGGING.md) - How errors are logged
- [src/ptpd_calibration/core/exceptions.py](../src/ptpd_calibration/core/exceptions.py) - Base exception classes
- [FASTAPI Documentation](https://fastapi.tiangolo.com/tutorial/handling-errors/) - FastAPI error handling
