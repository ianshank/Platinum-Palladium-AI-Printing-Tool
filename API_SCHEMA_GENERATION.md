# OpenAPI Schema Generation & TypeScript Type Safety

This document describes the OpenAPI schema generation pipeline and how TypeScript types are automatically generated from the FastAPI backend.

## Overview

This project uses a **schema-first** approach to API development:

1. **Backend** (FastAPI) → Defines Pydantic request/response models with full type hints
2. **OpenAPI Schema** → Automatically generated from FastAPI app
3. **TypeScript Types** → Generated from OpenAPI schema for frontend type safety
4. **API Hooks** → Consume generated types for compile-time safety

This ensures backend and frontend types are always in sync, preventing runtime errors from type mismatches.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + Pydantic)             │
│                                                              │
│  ├─ Request Models (AnalyzeRequest, CurveRequest, etc.)     │
│  ├─ Response Models (AnalyzeResponse, ChatResponse, etc.)   │
│  └─ Endpoint Type Annotations (→ ResponseModel)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Generate OpenAPI Schema        │
        │  (3.0+, JSON format)             │
        │  • All endpoints documented      │
        │  • All models defined            │
        │  • Error responses typed         │
        └──────────────────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌──────────────────────┐     ┌──────────────────────┐
    │   openapi.json       │     │   API Documentation  │
    │   (Repository Root)  │     │   (Swagger UI)       │
    └──────────────────────┘     └──────────────────────┘
                │
                │ openapi-typescript
                │
                ▼
    ┌──────────────────────────────────────┐
    │  src/api/generated/schema.ts         │
    │  • paths: All endpoints typed         │
    │  • components.schemas: All models    │
    │  • Full TypeScript intellisense      │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │   API Hooks (hooks.ts)               │
    │   • useAnalyze                       │
    │   • useUploadScan                    │
    │   • useCurves                        │
    │   • useCalibrations                  │
    │   • useChat                          │
    └──────────────────────────────────────┘
```

---

## Workflow

### 1. Update Backend Types

When adding or modifying an endpoint:

```python
# src/ptpd_calibration/api/server.py

# Define request model
class MyRequest(BaseModel):
    """Clear description of what this does."""
    field1: str = Field(..., description="What field1 does")
    field2: int = Field(default=0, ge=0, description="Non-negative number")

# Define response model
class MyResponse(BaseModel):
    """Clear description of the response."""
    result: str = Field(..., description="The result")
    status: int = Field(..., ge=200, le=599, description="HTTP status code")

# Add endpoint with proper types
@app.post("/api/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest) -> MyResponse:
    """Endpoint documentation."""
    return MyResponse(result="success", status=200)
```

**Requirements:**
- All request/response classes inherit from Pydantic `BaseModel`
- All fields have type hints
- All fields have `Field()` with description
- All endpoints have `response_model=` parameter
- All endpoints have return type annotation (`→ ResponseModel`)

### 2. Generate OpenAPI Schema

Run the schema generation script:

```bash
# From repository root
python scripts/generate_openapi_schema.py
```

This will:
- Create FastAPI app
- Extract OpenAPI schema (3.1.0)
- Validate schema structure
- Write to `openapi.json`
- Report summary statistics

**Output:**
```
OpenAPI Schema Generation Summary
============================================================
API Title:     PTPD Calibration API
API Version:   1.0.0
OpenAPI Ver:   3.1.0
Endpoints:     38
Schemas:       49
Output File:   openapi.json

✓ Schema validation passed!
```

### 3. Generate TypeScript Types

Run the type generation script:

```bash
# From frontend directory
cd frontend
pnpm run generate:types
```

Or run the shell script:

```bash
bash scripts/generate-api-types.sh
```

This will:
- Copy `openapi.json` to expected location
- Run `openapi-typescript` tool
- Generate `frontend/src/api/generated/schema.ts`
- Output TypeScript interface types for all API endpoints/models

**Generated file:** `frontend/src/api/generated/schema.ts` (~3200 lines)

### 4. Use Generated Types in API Hooks

The generated types are consumed by API hooks:

```typescript
// frontend/src/api/hooks.ts
import type { paths, components } from "./generated/schema";

type AnalyzeRequest = components["schemas"]["AnalyzeRequest"];
type AnalyzeResponse = components["schemas"]["AnalyzeResponse"];

export const useAnalyze = () => {
  return useMutation({
    mutationFn: async (request: AnalyzeRequest): Promise<AnalyzeResponse> => {
      const response = await apiClient.post("/api/analyze", request);
      return response.data;
    },
  });
};
```

---

## File Structure

```
project-root/
├── openapi.json                          # Generated OpenAPI schema
├── src/ptpd_calibration/
│   └── api/
│       ├── server.py                     # FastAPI app with endpoints
│       ├── models.py                     # Pydantic response models (NEW)
│       └── openapi.json                  # Copy for frontend (from generation)
├── frontend/
│   ├── src/api/
│   │   ├── generated/
│   │   │   ├── schema.ts                 # AUTO-GENERATED TypeScript types
│   │   │   └── .gitignore                # Keep generated file out of git
│   │   ├── hooks.ts                      # API hooks using generated types
│   │   └── client.ts                     # API client
│   └── package.json                      # Has generate:types script
└── scripts/
    ├── generate_openapi_schema.py         # Python script for schema generation
    └── generate-api-types.sh              # Shell script for TypeScript generation
```

---

## Key Features

### Type Safety

All API calls are now fully typed:

```typescript
// ✓ Good: Types from generated schema
const { mutate } = useAnalyze();
mutate({ densities: [0.1, 0.5, 1.0] });  // Type-checked!

// ✗ Bad: Trying to pass wrong type would error
// mutate({ wrong_field: 123 });  // ❌ Compile error!
```

### No Circular Dependencies

The generated schema is validated to ensure no circular references that could cause issues in TypeScript.

### Error Response Types

All error responses include `error_code` field for machine-readable error handling:

```typescript
type ErrorResponse = {
  error_code: string;      // e.g., "FILE_TOO_LARGE"
  detail: string;          // e.g., "Upload exceeds maximum size..."
  status_code: number;     // e.g., 413
};
```

### Request/Response Models

All endpoints have dedicated request and response models:

```typescript
// Request models for validation
type CurveRequest = {
  densities: number[];
  name?: string;
  curve_type?: string;
  paper_type?: string | null;
  chemistry?: string | null;
};

// Response models ensure type safety
type CurveGenerateResponse = {
  success: boolean;
  curve_id: string;
  name: string;
  num_points: number;
  input_values: number[];
  output_values: number[];
};
```

---

## Workflow Checklist

When adding a new endpoint:

- [ ] **Backend**: Define `Request` model in `server.py` or `models.py`
- [ ] **Backend**: Define `Response` model in `models.py`
- [ ] **Backend**: Add `response_model=` to `@app.method()` decorator
- [ ] **Backend**: Add return type annotation to function (`→ ResponseModel`)
- [ ] **Backend**: Run tests to ensure no breaking changes
- [ ] **Backend**: Verify endpoint works with `curl` or Postman
- [ ] **Schema**: Run `python scripts/generate_openapi_schema.py`
- [ ] **Schema**: Commit `openapi.json` to repo
- [ ] **Frontend**: Run `pnpm run generate:types` (or `bash scripts/generate-api-types.sh`)
- [ ] **Frontend**: Update API hooks to use new types
- [ ] **Frontend**: Add tests for hook using new types
- [ ] **Frontend**: Verify TypeScript compilation passes

---

## Testing

### Python Backend Tests

```bash
# Test schema generation
pytest tests/api/test_openapi_schema.py -v

# Test specific schema aspects
pytest tests/api/test_openapi_schema.py::TestOpenAPISchemaGeneration::test_response_models_defined
```

**What's tested:**
- Schema has required OpenAPI fields
- All endpoints define responses
- Response models are defined
- Error response includes `error_code`
- No breaking changes from previous version
- Minimum endpoint count is met

### TypeScript Frontend Tests

```bash
# Test generated types
cd frontend
pnpm run test -- src/api/__tests__/schema.test.ts

# Run all tests
pnpm run test
```

**What's tested:**
- Generated types have correct structure
- Response models are complete
- Request models work correctly
- API endpoint paths exist
- Type safety enforcement
- Required/optional fields
- Array and enum types

---

## Troubleshooting

### Schema Generation Fails

**Problem:** `ModuleNotFoundError` when running schema generation

**Solution:** Install package in development mode:
```bash
pip install -e .
pip install fastapi uvicorn python-multipart
```

### TypeScript Generation Fails

**Problem:** `openapi-typescript: not found`

**Solution:** Install frontend dependencies:
```bash
cd frontend
pnpm install
```

### Missing openapi.json

**Problem:** `Cannot find module '../src/ptpd_calibration/api/openapi.json'`

**Solution:** Regenerate schema:
```bash
python scripts/generate_openapi_schema.py
```

### Schema Validation Fails

**Problem:** Schema validation reports errors

**Solution:**
1. Check that all endpoints have response models
2. Verify all response models are Pydantic BaseModel
3. Run `pnpm run typecheck` in frontend to catch type errors
4. Check `openapi.json` is valid JSON: `jq . openapi.json`

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/api-types.yml
name: Generate API Types

on:
  push:
    paths:
      - 'src/ptpd_calibration/api/**'
      - 'openapi.json'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e . && pip install openapi-typescript
      - run: python scripts/generate_openapi_schema.py
      - run: npm run generate:types
      - uses: actions/upload-artifact@v3
        with:
          name: schema-files
          path: |
            openapi.json
            frontend/src/api/generated/schema.ts
```

---

## Best Practices

1. **Always use Pydantic models** for requests/responses - no inline dicts
2. **Add descriptions to all fields** using `Field(..., description="...")`
3. **Keep models focused** - one model per request/response type
4. **Document required fields** - use `...` for required, `None` for optional
5. **Validate types** - use constraints like `ge=0`, `le=1`, `min_length=1`
6. **Export models properly** - import in server.py and use as response_model
7. **Test after changes** - run schema generation and TypeScript tests
8. **Commit both files** - `openapi.json` and `frontend/src/api/generated/schema.ts`

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI 3.0 Specification](https://spec.openapis.org/oas/v3.0.3)
- [openapi-typescript](https://github.com/drwpow/openapi-typescript)
