# Logging Architecture Guide

## Overview

The PTPD Calibration system uses centralized, structured logging with support for JSON formatting, context tracking, and development-friendly console output.

## Quick Start

### Basic Logging

```python
from ptpd_calibration.core.logging import get_logger

# Get logger for your module
logger = get_logger(__name__)

# Log at different levels
logger.debug("Detailed information for diagnosing problems")
logger.info("Confirmation that things are working as expected")
logger.warning("An indication that something unexpected happened")
logger.error("A serious problem, something has failed")
logger.critical("A very serious error, system is at risk")
```

### Structured Logging with Context

```python
logger.info(
    "Curve generated successfully",
    extra={
        "curve_id": curve.id,
        "num_points": len(curve.points),
        "duration_ms": elapsed_time,
        "user_id": user.id,
    }
)
```

### Initialize Logging

```python
# At application startup (typically in api/server.py or main)
from ptpd_calibration.core.logging import setup_logging

setup_logging(
    level="INFO",
    json_format=True,  # Use JSON formatting
    log_file="app.log",  # Optional file logging
)
```

## Architecture

### Log Formatting

#### JSON Format (Production)

```python
setup_logging(json_format=True)

# Output:
{
  "timestamp": "2026-08-03T12:34:56.789Z",
  "level": "INFO",
  "logger": "ptpd_calibration.curves.generator",
  "message": "Curve generated successfully",
  "module": "generator",
  "function": "generate",
  "line": 42,
  "context": {
    "request_id": "req_abc123",
    "user_id": "user_123"
  },
  "curve_id": "curve_456",
  "num_points": 256,
  "duration_ms": 1234
}
```

#### Colored Console (Development)

```python
setup_logging(json_format=False)  # Default

# Output (with colors):
[INFO] 2026-08-03 12:34:56 - ptpd_calibration.curves.generator - Curve generated successfully
```

### Context Management

Track request/operation context across function calls:

```python
from ptpd_calibration.core.logging import log_context
from contextvars import ContextVar

# Set context for request
async def handle_request(request_id: str, user_id: str):
    with log_context(request_id=request_id, user_id=user_id):
        logger.info("Request started")
        result = await process()
        logger.info("Request completed")
        return result

# All logs within the context include the context variables
# Output: {..., "context": {"request_id": "...", "user_id": "..."}}
```

## Usage Patterns

### 1. Operation Logging

```python
from time import time
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

def detect_tablet(image):
    """Detect step tablet in image."""
    start = time()
    
    logger.info("Tablet detection started", extra={"image_shape": image.shape})
    
    try:
        contours = _find_contours(image)
        
        if not contours:
            logger.warning(
                "No contours found",
                extra={"min_contours_required": MIN_CONTOURS}
            )
            return None
        
        logger.info(
            "Tablet detection completed",
            extra={
                "num_contours": len(contours),
                "duration_ms": (time() - start) * 1000,
            }
        )
        return contours
    
    except Exception as e:
        logger.error(
            "Tablet detection failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": (time() - start) * 1000,
            }
        )
        raise
```

### 2. Performance Logging

```python
import logging
from functools import wraps
from time import time

def log_performance(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            try:
                result = func(*args, **kwargs)
                duration = (time() - start) * 1000
                logger.info(
                    f"{func.__name__} completed",
                    extra={"duration_ms": duration}
                )
                return result
            except Exception as e:
                duration = (time() - start) * 1000
                logger.error(
                    f"{func.__name__} failed",
                    extra={
                        "error": str(e),
                        "duration_ms": duration,
                    }
                )
                raise
        return wrapper
    return decorator

# Usage
logger = get_logger(__name__)

@log_performance(logger)
def generate_curve(densities):
    """Curve generation is automatically timed."""
    return CurveGenerator().generate(densities)
```

### 3. Request/Response Logging (FastAPI Middleware)

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from ptpd_calibration.core.logging import get_logger, log_context
import time
import uuid

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Log HTTP requests and responses."""
        request_id = str(uuid.uuid4())
        start = time.time()
        
        with log_context(request_id=request_id):
            logger.info(
                "Request started",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                }
            )
            
            try:
                response = await call_next(request)
                duration = time.time() - start
                
                logger.info(
                    "Request completed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": duration * 1000,
                    }
                )
                
                # Add request ID to response headers for tracing
                response.headers["X-Request-ID"] = request_id
                return response
            
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    "Request failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e),
                        "duration_ms": duration * 1000,
                    }
                )
                raise

# Register in FastAPI
from ptpd_calibration.api.server import app
app.add_middleware(RequestLoggingMiddleware)
```

### 4. Data Processing Logging

```python
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

def extract_density_data(image, patches):
    """Extract density values from image patches."""
    densities = []
    errors = []
    
    logger.info(
        "Density extraction started",
        extra={"num_patches": len(patches)}
    )
    
    for i, patch in enumerate(patches):
        try:
            density = _calculate_density(image, patch)
            densities.append(density)
            
            if i % 10 == 0:  # Log progress
                logger.debug(
                    "Progress update",
                    extra={
                        "processed_patches": i,
                        "total_patches": len(patches),
                        "progress_percent": (i / len(patches)) * 100,
                    }
                )
        
        except Exception as e:
            logger.warning(
                f"Failed to extract density from patch {i}",
                extra={
                    "patch_index": i,
                    "error": str(e),
                }
            )
            errors.append((i, e))
    
    logger.info(
        "Density extraction completed",
        extra={
            "successful": len(densities),
            "failed": len(errors),
            "total": len(patches),
        }
    )
    
    return densities, errors
```

### 5. Debug Logging with Conditionals

```python
from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

def process_curve(curve_data, debug=False):
    """Process curve with optional debug output."""
    logger.debug(
        "Starting curve processing",
        extra={"curve_points": len(curve_data.points)}
    )
    
    # Only log detailed internals if debug is enabled
    if debug:
        logger.debug(
            "Curve details",
            extra={
                "input_values": curve_data.input_values[:5],  # First 5
                "output_values": curve_data.output_values[:5],
                "min_input": min(curve_data.input_values),
                "max_input": max(curve_data.input_values),
            }
        )
    
    result = _process(curve_data)
    
    logger.info("Curve processing completed")
    return result
```

## Configuration

### Setup Logging at Application Startup

```python
# In main entry point (uvicorn, CLI, etc.)
from ptpd_calibration.core.logging import setup_logging
from ptpd_calibration.config import get_settings

def main():
    """Initialize application."""
    settings = get_settings()
    
    # Configure logging based on environment
    setup_logging(
        level=settings.log_level,
        json_format=not settings.development_mode,
        log_file=settings.log_file if settings.log_file else None,
    )
    
    # Now all loggers will use this configuration
    from ptpd_calibration.api.server import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

### Environment Variables

```bash
# Set log level
export PTPD_LOG_LEVEL=DEBUG

# Use JSON formatting (production)
export PTPD_JSON_LOGS=true

# Log to file
export PTPD_LOG_FILE=/var/log/ptpd.log

# Development mode (colored console output)
export PTPD_DEVELOPMENT_MODE=true
```

## Log Levels

### DEBUG
Use for detailed information useful only when diagnosing problems:

```python
logger.debug("Cache hit for curve", extra={"cache_key": key})
logger.debug("Interpolation method: cubic")
logger.debug("Configuration loaded", extra={"settings": settings.dict()})
```

### INFO
Use for confirmation that things are working as expected:

```python
logger.info("Curve generated successfully")
logger.info("File uploaded", extra={"filename": "scan.png", "size_bytes": 1024})
logger.info("User authenticated", extra={"user_id": user.id})
```

### WARNING
Use when something unexpected happened but functionality continues:

```python
logger.warning("Low density values detected", extra={"min_density": 0.01})
logger.warning("Degraded image quality", extra={"image_quality_score": 0.45})
logger.warning("Fallback to default parameters")
```

### ERROR
Use for serious problems where functionality failed:

```python
logger.error("Tablet detection failed", extra={"error": str(e)})
logger.error("LLM request timeout after 30s")
logger.error("Database connection lost")
```

### CRITICAL
Use for very serious errors where system integrity is at risk:

```python
logger.critical("Memory allocation failed - system unstable")
logger.critical("Configuration validation failed - cannot start")
logger.critical("Unrecoverable disk error")
```

## Best Practices

### 1. Use Consistent Field Names

```python
# Good: Standardized extra fields
logger.info("Operation completed", extra={
    "operation": "detect_tablet",
    "duration_ms": 1234,
    "status": "success",
})

# Avoid: Inconsistent field names
logger.info("Op done", extra={
    "op": "detect_tablet",
    "time": 1234,
    "result": "success",
})
```

### 2. Include Identifiers for Tracing

```python
# Good: Include IDs for correlation
logger.info("Processing started", extra={
    "request_id": request_id,
    "user_id": user_id,
    "operation_id": operation_id,
})

# Avoid: Missing context
logger.info("Processing started")
```

### 3. Don't Log Sensitive Data

```python
# Good: Mask or omit sensitive information
logger.info("Authentication attempt", extra={
    "username": username,
    "api_key_prefix": api_key[:4] + "***",
})

# Avoid: Logging secrets
logger.info("Auth attempt", extra={
    "api_key": api_key,  # DON'T DO THIS
    "password": password,  # DON'T DO THIS
})
```

### 4. Use Appropriate Log Levels

```python
# Good: Useful in production
logger.info("Important milestone reached")
logger.warning("Recoverable issue detected")

# Avoid: Too verbose for production
logger.debug("Inside loop iteration 42")
logger.debug("Variable x = 123")
```

### 5. Structure Logs for Parsing

```python
# Good: Machine-readable
logger.info("Request completed", extra={
    "status_code": 200,
    "duration_ms": 1234,
})

# Avoid: Unstructured text
logger.info(f"Request completed with code {code} in {time}ms")
```

## Monitoring and Analysis

### Log Aggregation Tools

With JSON logging, logs can be easily aggregated and analyzed:

```bash
# Using jq to filter and analyze logs
cat app.log | jq 'select(.level == "ERROR")'

# Count errors by type
cat app.log | jq 'select(.level == "ERROR") | .error_type' | sort | uniq -c

# Timeline of specific operation
cat app.log | jq 'select(.operation == "detect_tablet")'

# Performance analysis
cat app.log | jq 'select(.duration_ms) | .duration_ms' | jq -s 'add/length'
```

### Sentry Integration

```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://examplePublicKey@sentry.example.com/123456",
    environment="production",
)

# Sentry captures logged errors and critical messages
logger.error("This will be sent to Sentry")
logger.critical("This will also be sent")
```

### CloudLogging (GCP)

```python
from google.cloud import logging as cloud_logging

# Integrate with Google Cloud Logging
client = cloud_logging.Client()
client.setup_logging()

# Logs will be sent to Cloud Logging
logger.info("This goes to Google Cloud")
```

## Troubleshooting

### Issue: Logs not showing

```python
# Ensure setup_logging was called
from ptpd_calibration.core.logging import setup_logging
setup_logging()  # Must be called before logging

# Check log level
setup_logging(level="DEBUG")  # Set lower level
```

### Issue: Missing context in logs

```python
# Use log_context correctly
from ptpd_calibration.core.logging import log_context

# Good
with log_context(request_id=req_id):
    logger.info("Message")  # Will include context

# Avoid
log_context(request_id=req_id)  # Not a context manager!
logger.info("Message")  # Context NOT included
```

### Issue: Performance impact

```python
# Use lazy evaluation for expensive operations
# Avoid
logger.debug(f"Large object: {expensive_object.to_dict()}")

# Better: Only evaluate if needed
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Large object: {expensive_object.to_dict()}")

# Best: Use extra dict
logger.debug("Operation", extra={"key": expensive_object.key})
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Logging as part of system design
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Error logging patterns
- [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) - Logging configuration
- [src/ptpd_calibration/core/logging.py](../src/ptpd_calibration/core/logging.py) - Implementation
- [Python logging docs](https://docs.python.org/3/library/logging.html)
