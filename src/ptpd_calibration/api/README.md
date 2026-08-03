# PTPD Calibration API

Fast, type-safe API for the platinum/palladium printing calibration system built with FastAPI.

## Quick Start

### Starting the Server

```bash
# Development (with auto-reload)
uvicorn src.ptpd_calibration.api.server:app --reload

# Production
uvicorn src.ptpd_calibration.api.server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Architecture

### Request/Response Models

All API endpoints use Pydantic models for strict type validation:

```python
# Request model - defines what the client sends
class AnalyzeRequest(BaseModel):
    densities: list[float]  # List of density values to analyze

# Response model - defines what the server returns
class AnalyzeResponse(BaseModel):
    dmin: float             # Minimum density
    dmax: float             # Maximum density
    range: float            # Total range
    is_monotonic: bool      # Whether values increase monotonically
    max_error: float        # Maximum deviation
    rms_error: float        # Root mean square error
    suggestions: list[str]  # Adjustment suggestions
```

### Endpoint Pattern

All endpoints follow this pattern:

```python
@app.post("/api/resource", response_model=ResourceResponse)
async def create_resource(request: ResourceRequest) -> ResourceResponse:
    """
    Create a new resource.
    
    Detailed description of what this endpoint does.
    """
    # Process request
    result = process(request.field)
    
    # Return typed response
    return ResourceResponse(field=result)
```

## Models

### Response Models

Response models define the structure of all API responses. They are located in `models.py`:

#### Health & Root Endpoints
- `HealthResponse` - Health check status
- `RootResponse` - API information

#### Analysis Endpoints
- `AnalyzeResponse` - Density analysis results

#### Scan Processing
- `ScanUploadResponse` - Step tablet scan processing results

#### Curve Operations
- `CurveGenerateResponse` - Curve generation results
- `CurveModifyResponse` - Curve modification results
- `CurveSmoothResponse` - Curve smoothing results
- `CurveBlendResponse` - Curve blending results
- `CurveEnhanceResponse` - AI curve enhancement results
- `CurveRetrieveResponse` - Stored curve retrieval
- `CurveMonotonicityResponse` - Monotonicity enforcement

#### Quad Files
- `QuadUploadResponse` - QTR file upload results
- `QuadParseResponse` - QTR content parsing results

#### Calibrations
- `ListCalibrationsResponse` - List of calibration records
- `CreateCalibrationResponse` - Calibration creation confirmation

#### Chat & Assistance
- `ChatResponse` - AI assistant response
- `RecipeResponse` - Recipe suggestions
- `TroubleshootResponse` - Troubleshooting advice

#### Statistics
- `StatisticsResponse` - Database statistics

### Request Models

Request models are defined inline in `server.py`:

- `AnalyzeRequest` - Density values to analyze
- `CurveRequest` - Parameters for curve generation
- `CurveModifyRequest` - Curve modification parameters
- `CurveSmoothRequest` - Smoothing parameters
- `CurveBlendRequest` - Blending parameters
- `CurveEnhanceRequest` - Enhancement parameters
- `CalibrationRequest` - Calibration creation parameters
- `ChatRequest` - Chat message and history flag
- `RecipeRequest` - Recipe request parameters
- `TroubleshootRequest` - Problem description

## Endpoints

### Health & Information

```http
GET /
GET /api/health
```

### Analysis

```http
POST /api/analyze
Content-Type: application/json
{
  "densities": [0.1, 0.5, 1.0, 1.5, 2.0]
}
```

### Scan Upload

```http
POST /api/scan/upload
Content-Type: multipart/form-data
file: <scan-image.tif>
tablet_type: stouffer_21
```

### Curves

```http
POST /api/curves/generate
POST /api/curves/modify
POST /api/curves/smooth
POST /api/curves/blend
POST /api/curves/enhance
POST /api/curves/upload-quad
POST /api/curves/parse-quad
GET /api/curves/{curve_id}
POST /api/curves/{curve_id}/enforce-monotonicity
POST /api/curves/{curve_id}/export
```

### Calibrations

```http
GET /api/calibrations?paper_type=<type>&limit=50
POST /api/calibrations
GET /api/calibrations/{calibration_id}
```

### Chat

```http
POST /api/chat
POST /api/chat/recipe
POST /api/chat/troubleshoot
```

### Statistics

```http
GET /api/statistics
```

## Error Handling

All errors follow a consistent format:

```python
class ErrorResponse(BaseModel):
    error_code: str          # Machine-readable error code
    detail: str              # Human-readable error message
    status_code: int         # HTTP status code (400-599)
```

### Example Error Response

```json
{
  "error_code": "FILE_TOO_LARGE",
  "detail": "Upload exceeds maximum size of 50 MB",
  "status_code": 413
}
```

### Common Error Codes

- `INVALID_FILE_TYPE` - Unsupported file format
- `FILE_TOO_LARGE` - Upload exceeds size limit
- `INVALID_PARAMETERS` - Request validation failed
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `PROCESSING_ERROR` - Error during processing
- `EXTERNAL_SERVICE_ERROR` - LLM or external service failed

## Type Safety

### Backend Type Hints

All endpoints have full type hints:

```python
# Request type is validated
# Response type ensures consistent structure
# Return type annotation enables IDE support
@app.post("/api/resource", response_model=ResourceResponse)
async def create_resource(request: ResourceRequest) -> ResourceResponse:
    pass
```

### Frontend TypeScript Types

TypeScript types are automatically generated from the OpenAPI schema:

```typescript
import type { paths, components } from "./generated/schema";

type MyRequest = components["schemas"]["MyRequest"];
type MyResponse = components["schemas"]["MyResponse"];

const response: MyResponse = await api.post("/api/resource", {
  field: "value"
});
```

## Validation

### Request Validation

Pydantic automatically validates all requests:

```python
# Type validation
class CurveRequest(BaseModel):
    densities: list[float]           # Must be list of numbers
    name: str = "Curve"              # String with default
    curve_type: str = "linear"       # Constrained values
    paper_type: str | None = None    # Optional string

# Field constraints
class CalibrationRequest(BaseModel):
    exposure_time: float = Field(..., ge=0.0, description="Must be >= 0")
    metal_ratio: float = Field(..., ge=0.0, le=1.0, description="0 to 1")
```

### Response Validation

All responses are validated before sending:

```python
return AnalyzeResponse(
    dmin=0.1,           # Validated as float
    dmax=2.5,           # Validated as float
    range=2.4,          # Validated as float
    is_monotonic=True,  # Validated as bool
    max_error=0.05,     # Validated as float
    rms_error=0.02,     # Validated as float
    suggestions=[],     # Validated as list[str]
)  # Type checking ensures all required fields present
```

## Development

### Adding a New Endpoint

1. **Define request model** in `server.py` or `models.py`:
   ```python
   class NewRequest(BaseModel):
       field: str = Field(..., description="Field description")
   ```

2. **Define response model** in `models.py`:
   ```python
   class NewResponse(BaseModel):
       result: str = Field(..., description="Result description")
   ```

3. **Implement endpoint** in `server.py`:
   ```python
   @app.post("/api/new", response_model=NewResponse)
   async def create_new(request: NewRequest) -> NewResponse:
       """Do something new."""
       return NewResponse(result="value")
   ```

4. **Test endpoint**:
   ```bash
   curl -X POST http://localhost:8000/api/new \
     -H "Content-Type: application/json" \
     -d '{"field": "value"}'
   ```

5. **Regenerate OpenAPI schema**:
   ```bash
   python scripts/generate_openapi_schema.py
   ```

6. **Regenerate TypeScript types**:
   ```bash
   cd frontend && pnpm run generate:types
   ```

### Testing

Run API tests:

```bash
# All API tests
pytest tests/api/ -v

# Specific endpoint tests
pytest tests/api/test_curve_endpoints.py -v

# Schema validation tests
pytest tests/api/test_openapi_schema.py -v
```

## CORS Configuration

CORS is configured in `create_app()`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

In development, `http://localhost:3000` is automatically allowed.

## Performance

### Response Sizes

All responses are optimized for size:
- Large arrays are sampled (e.g., curve points)
- Unnecessary fields are omitted
- Responses are gzip compressed

### Request Limits

- Maximum upload size: 50 MB (configurable)
- Maximum request body: 10 MB
- Connection timeout: 30 seconds

## Security

### Input Validation

- All file uploads are scanned and sanitized
- File extensions are checked against allowlist
- File names are never trusted from client

### Rate Limiting

Not currently implemented. Consider adding:
- Per-IP rate limiting
- Per-endpoint rate limiting
- Authentication/API key support

### Error Messages

- Detailed error messages in development
- Generic error messages in production
- No sensitive information leaked in errors

## Troubleshooting

### "Module not found" Error

```bash
pip install -e .
```

### API won't start

```bash
# Check dependencies
pip install fastapi uvicorn python-multipart

# Check port isn't in use
lsof -i :8000
```

### Type validation fails

Check that all Pydantic models inherit from `BaseModel`:

```python
from pydantic import BaseModel

class MyModel(BaseModel):  # ✓ Correct
    field: str
```

## References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [OpenAPI Spec](https://spec.openapis.org/oas/v3.0.3)
