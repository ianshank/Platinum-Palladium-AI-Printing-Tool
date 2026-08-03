# API Types and Schema Guide

## Overview

The PTPD Calibration API uses OpenAPI 3.0 schema auto-generated from Pydantic models, ensuring type safety and enabling code generation for frontend clients.

## Quick Start

### Access API Documentation

Once the FastAPI server is running:

- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Code Generation

Generate TypeScript types from OpenAPI schema:

```bash
# Install OpenAPI generator
npm install -g @openapitools/openapi-generator-cli

# Generate TypeScript client
openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g typescript-fetch \
  -o frontend/src/api/generated

# Or use openapi-typescript
npm install -g openapi-typescript
openapi-typescript http://localhost:8000/openapi.json > frontend/src/api/types.ts
```

## Pydantic Models

### Request Models

Request models define the shape of data sent to the API:

```python
# In api/models.py

from pydantic import BaseModel, Field
from typing import Optional, List

class CurveGenerateRequest(BaseModel):
    """Request to generate a calibration curve."""
    
    densities: List[float] = Field(
        ...,
        description="Density measurements (0.0 to 4.0)",
        min_items=2,
        max_items=256,
        example=[0.1, 0.5, 1.0, 1.5, 2.0],
    )
    
    name: str = Field(
        default="Calibration Curve",
        description="Curve name for identification",
        max_length=100,
        example="My Calibration",
    )
    
    curve_type: str = Field(
        default="linear",
        description="Curve interpolation type",
        regex="^(linear|cubic|monotonic)$",
        example="cubic",
    )
    
    paper_type: Optional[str] = Field(
        default=None,
        description="Optional paper type for context",
        example="platinum_palladium",
    )

class CurveGenerateResponse(BaseModel):
    """Response with generated curve."""
    
    status: str = Field(example="success")
    curve: 'CurveData'  # Reference to another model
    duration_ms: float = Field(description="Processing time in milliseconds")
```

### Response Models

Response models define the shape of data returned by the API:

```python
class CurveData(BaseModel):
    """Curve data structure."""
    
    id: str = Field(description="Unique curve identifier")
    name: str
    input_values: List[float] = Field(description="Input density values")
    output_values: List[float] = Field(description="Output linear values")
    curve_type: str
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "curve_abc123",
                "name": "My Calibration",
                "input_values": [0.0, 0.5, 1.0, 1.5, 2.0],
                "output_values": [0.0, 0.3, 0.6, 0.8, 1.0],
                "curve_type": "cubic",
                "created_at": "2026-08-03T12:34:56Z",
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    status_code: int = Field(description="HTTP status code")
    timestamp: datetime
    request_id: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "TABLET_NOT_FOUND",
                "message": "No tablet contours found in image.",
                "status_code": 422,
                "timestamp": "2026-08-03T12:34:56Z",
                "request_id": "req_abc123",
            }
        }
```

## Type Validation

### Field Constraints

Pydantic automatically validates fields based on constraints:

```python
from pydantic import BaseModel, Field, constr, conint, confloat

class DetectionSettings(BaseModel):
    # String constraints
    method: constr(regex="^(edge|contour|hough)$")  # Pattern match
    
    # Integer constraints
    threshold: conint(ge=0, le=255)  # Range 0-255
    kernel_size: conint(ge=1, le=21, multiple_of=2)  # Even numbers only
    
    # Float constraints
    confidence: confloat(ge=0.0, le=1.0)  # Probability
    
    # List constraints
    channels: list[str] = Field(min_items=1, max_items=3)
    
    # Length constraints
    description: str = Field(min_length=1, max_length=500)
```

### Custom Validation

Define custom validators for complex logic:

```python
from pydantic import field_validator, model_validator

class CurveGenerateRequest(BaseModel):
    densities: List[float]
    
    @field_validator('densities')
    @classmethod
    def validate_densities(cls, v):
        """Validate density values are in expected range."""
        if not v:
            raise ValueError("Densities list cannot be empty")
        if any(d < 0 or d > 4.0 for d in v):
            raise ValueError("All densities must be between 0 and 4.0")
        if len(set(v)) < len(v):
            raise ValueError("Duplicate density values not allowed")
        return sorted(v)

class CurveData(BaseModel):
    input_values: List[float]
    output_values: List[float]
    
    @model_validator(mode='after')
    def validate_matching_lengths(self):
        """Validate input and output have same length."""
        if len(self.input_values) != len(self.output_values):
            raise ValueError(
                f"Length mismatch: {len(self.input_values)} inputs "
                f"but {len(self.output_values)} outputs"
            )
        return self
```

## OpenAPI Schema

### Viewing Generated Schema

The schema is auto-generated and available at `/openapi.json`:

```bash
# Download schema
curl http://localhost:8000/openapi.json > openapi.json

# Pretty print
curl http://localhost:8000/openapi.json | jq . | less
```

### Key Schema Components

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "PTPD Calibration API",
    "version": "1.0.0",
    "description": "AI-powered calibration system for platinum/palladium printing"
  },
  "paths": {
    "/curves/generate": {
      "post": {
        "summary": "Generate calibration curve",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/CurveGenerateRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Curve generated successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CurveGenerateResponse"
                }
              }
            }
          },
          "422": {
            "description": "Validation error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "CurveGenerateRequest": { ... },
      "CurveGenerateResponse": { ... },
      "CurveData": { ... },
      "ErrorResponse": { ... }
    }
  }
}
```

## Frontend Type Generation

### TypeScript Client Code Generation

```bash
# Install openapi-typescript
npm install -D openapi-typescript

# Generate types from running server
npx openapi-typescript http://localhost:8000/openapi.json \
  --output frontend/src/api/types.ts

# Generate from saved schema
npx openapi-typescript openapi.json --output frontend/src/api/types.ts
```

### Using Generated Types

```typescript
// frontend/src/api/types.ts (auto-generated)

export interface CurveGenerateRequest {
  densities: number[];
  name?: string;
  curve_type?: "linear" | "cubic" | "monotonic";
  paper_type?: string;
}

export interface CurveData {
  id: string;
  name: string;
  input_values: number[];
  output_values: number[];
  curve_type: string;
  created_at: string;
}

// frontend/src/api/client.ts (manual)

import type { CurveGenerateRequest, CurveData } from './types';

export async function generateCurve(request: CurveGenerateRequest): Promise<CurveData> {
  const response = await fetch('http://localhost:8000/curves/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}
```

## Common Response Patterns

### Success Response

```python
class SuccessResponse(BaseModel):
    """Standard success response."""
    status: str = "success"
    data: dict  # Actual data
    timestamp: datetime

# Usage
return SuccessResponse(
    data={"curve": curve.dict()},
)
```

### Paginated Response

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated list response."""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    
    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size

# Usage
return PaginatedResponse[CurveData](
    items=curves,
    total=100,
    page=1,
    page_size=20,
    has_next=True,
)
```

### Streaming Response

```python
from fastapi.responses import StreamingResponse

@app.post("/curves/generate/stream")
async def generate_curve_stream(request: CurveGenerateRequest):
    """Stream curve generation updates."""
    async def generate():
        generator = CurveGenerator()
        for progress in generator.generate_with_progress(request.densities):
            yield f"data: {json.dumps(progress)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
```

## API Documentation Examples

### Parameter Documentation

```python
from fastapi import Query, Body, Path

@app.get("/curves/{curve_id}")
async def get_curve(
    curve_id: str = Path(
        ...,
        description="Unique curve identifier",
        example="curve_abc123",
    ),
    include_history: bool = Query(
        False,
        description="Include edit history in response",
    ),
):
    """Get curve by ID with optional history."""
    pass
```

### Response Documentation

```python
@app.post("/curves/generate")
async def generate_curve(request: CurveGenerateRequest) -> CurveGenerateResponse:
    """Generate calibration curve from density measurements.
    
    ## Details
    
    This endpoint uses a cubic spline interpolation algorithm to create
    a smooth calibration curve from discrete density measurements.
    
    ## Examples
    
    ### Request
    ```json
    {
        "densities": [0.1, 0.5, 1.0, 1.5, 2.0],
        "name": "My Calibration",
        "curve_type": "cubic"
    }
    ```
    
    ### Response
    ```json
    {
        "status": "success",
        "curve": {
            "id": "curve_abc123",
            "name": "My Calibration",
            "input_values": [...],
            "output_values": [...],
            "created_at": "2026-08-03T12:34:56Z"
        },
        "duration_ms": 234
    }
    ```
    
    ## Errors
    
    - 422: Invalid densities (outside 0-4.0 range or too few points)
    - 503: LLM-based enhancement requested but service unavailable
    """
    pass
```

## Testing API Types

### Unit Test Types

```python
def test_curve_generate_request_validation():
    """Test request model validation."""
    # Valid request
    valid = CurveGenerateRequest(
        densities=[0.1, 0.5, 1.0, 1.5, 2.0],
        name="Test",
    )
    assert valid.curve_type == "linear"  # Default
    
    # Invalid: density out of range
    with pytest.raises(ValueError):
        CurveGenerateRequest(
            densities=[5.0, 6.0],  # Too high
        )
    
    # Invalid: too few densities
    with pytest.raises(ValueError):
        CurveGenerateRequest(densities=[1.0])

def test_curve_data_model():
    """Test response model."""
    curve = CurveData(
        id="test",
        name="Test Curve",
        input_values=[0.0, 0.5, 1.0],
        output_values=[0.0, 0.6, 1.0],
        curve_type="cubic",
        created_at=datetime.now(),
    )
    
    # Can serialize to JSON
    json_str = curve.model_dump_json()
    assert "test" in json_str
    
    # Can deserialize
    loaded = CurveData.model_validate_json(json_str)
    assert loaded.id == curve.id
```

### Integration Test Types

```python
@pytest.mark.asyncio
async def test_api_returns_valid_schema(client):
    """Test that API responses match schema."""
    response = client.post(
        "/curves/generate",
        json={"densities": [0.1, 0.5, 1.0, 1.5, 2.0]}
    )
    
    # Parse response with model
    data = CurveGenerateResponse(**response.json())
    
    # Validates against schema automatically
    assert data.status == "success"
    assert isinstance(data.curve, CurveData)
```

## Version Management

### API Versioning

```python
# api/v1/server.py
from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")

@v1_router.post("/curves/generate")
async def generate_curve_v1(request: CurveGenerateRequest):
    pass

# api/v2/server.py - If major changes needed
v2_router = APIRouter(prefix="/api/v2")

@v2_router.post("/curves/generate")
async def generate_curve_v2(request: CurveGenerateRequestV2):
    pass

# main server.py
app.include_router(v1_router)
app.include_router(v2_router)
```

### Deprecation

```python
from fastapi import Header, deprecated

@app.post("/old-endpoint", deprecated=True)
async def old_endpoint():
    """This endpoint is deprecated. Use /new-endpoint instead."""
    pass

@app.post("/new-endpoint")
async def new_endpoint():
    """Replacement for /old-endpoint."""
    pass
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - API as part of system architecture
- [src/ptpd_calibration/api/models.py](../src/ptpd_calibration/api/models.py) - Pydantic models
- [src/ptpd_calibration/api/server.py](../src/ptpd_calibration/api/server.py) - API endpoints
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.0)
- [OpenAPI Tools](https://openapi.tools/)
