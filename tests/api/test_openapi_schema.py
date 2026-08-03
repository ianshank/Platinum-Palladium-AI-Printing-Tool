"""
Tests for OpenAPI schema generation and validation.

These tests ensure:
1. OpenAPI schema is valid OpenAPI 3.0+ specification
2. All endpoints have proper response models
3. Error responses include error_code field
4. No breaking changes from previous API versions
"""

import json
from pathlib import Path

import pytest
from ptpd_calibration.api.server import create_app


class TestOpenAPISchemaGeneration:
    """Tests for OpenAPI schema generation."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        return create_app()

    @pytest.fixture
    def openapi_schema(self, app):
        """Extract OpenAPI schema from app."""
        return app.openapi()

    def test_schema_has_required_fields(self, openapi_schema):
        """Test that schema has all required OpenAPI fields."""
        assert "openapi" in openapi_schema
        assert "info" in openapi_schema
        assert "paths" in openapi_schema
        assert "components" in openapi_schema

    def test_schema_version(self, openapi_schema):
        """Test that schema is OpenAPI 3.0+."""
        version = openapi_schema.get("openapi", "")
        assert version.startswith("3."), f"Expected OpenAPI 3.x, got {version}"

    def test_api_info_present(self, openapi_schema):
        """Test that API info is complete."""
        info = openapi_schema.get("info", {})
        assert info.get("title") == "PTPD Calibration API"
        assert "version" in info
        assert "description" in info

    def test_all_endpoints_have_responses(self, openapi_schema):
        """Test that all endpoints define responses."""
        paths = openapi_schema.get("paths", {})
        endpoints_without_responses = []

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.startswith("x-"):
                    continue  # Skip vendor extensions
                if isinstance(operation, dict) and "responses" not in operation:
                    endpoints_without_responses.append(f"{method.upper()} {path}")

        assert not endpoints_without_responses, (
            f"Found {len(endpoints_without_responses)} endpoints without responses: "
            f"{endpoints_without_responses[:5]}"
        )

    def test_endpoints_have_200_response(self, openapi_schema):
        """Test that endpoints define 200 response."""
        paths = openapi_schema.get("paths", {})
        endpoints_missing_200 = []

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.startswith("x-"):
                    continue
                if isinstance(operation, dict):
                    responses = operation.get("responses", {})
                    # POST/GET/PUT endpoints should have 200 response
                    if method in ["get", "post", "put", "delete"] and "200" not in responses:
                        endpoints_missing_200.append(f"{method.upper()} {path}")

        assert not endpoints_missing_200, (
            f"Found {len(endpoints_missing_200)} endpoints without 200 response: "
            f"{endpoints_missing_200[:5]}"
        )

    def test_response_models_defined(self, openapi_schema):
        """Test that response models are defined in components."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        required_schemas = [
            "HealthResponse",
            "RootResponse",
            "AnalyzeResponse",
            "ScanUploadResponse",
            "CurveGenerateResponse",
            "CurveModifyResponse",
            "CurveSmoothResponse",
            "CurveBlendResponse",
            "CurveEnhanceResponse",
            "CurveRetrieveResponse",
            "QuadUploadResponse",
            "QuadParseResponse",
            "ListCalibrationsResponse",
            "CreateCalibrationResponse",
            "ChatResponse",
            "RecipeResponse",
            "TroubleshootResponse",
            "StatisticsResponse",
        ]

        missing_schemas = [s for s in required_schemas if s not in schemas]
        assert not missing_schemas, f"Missing schemas: {missing_schemas}"

    def test_error_response_schema(self, openapi_schema):
        """Test that error response includes error_code field."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})
        error_schema = schemas.get("ErrorResponse", {})

        properties = error_schema.get("properties", {})
        required = error_schema.get("required", [])

        assert "error_code" in properties, "ErrorResponse missing error_code property"
        assert "detail" in properties, "ErrorResponse missing detail property"
        assert "error_code" in required, "error_code not in required fields"

    def test_request_models_defined(self, openapi_schema):
        """Test that request models are properly defined."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        required_request_models = [
            "AnalyzeRequest",
            "CurveRequest",
            "CurveModifyRequest",
            "CurveSmoothRequest",
            "CurveBlendRequest",
            "CurveEnhanceRequest",
            "CalibrationRequest",
            "ChatRequest",
            "RecipeRequest",
            "TroubleshootRequest",
        ]

        missing_models = [m for m in required_request_models if m not in schemas]
        assert not missing_models, f"Missing request models: {missing_models}"

    def test_endpoint_count(self, openapi_schema):
        """Test minimum number of endpoints."""
        paths = openapi_schema.get("paths", {})
        endpoint_count = sum(
            len([m for m in methods if not m.startswith("x-")])
            for methods in paths.values()
        )

        # Should have at least the core endpoints
        assert endpoint_count >= 15, f"Expected at least 15 endpoints, got {endpoint_count}"

    def test_schema_models_count(self, openapi_schema):
        """Test minimum number of defined models."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        # Should have response, request, and core models
        assert len(schemas) >= 30, f"Expected at least 30 schemas, got {len(schemas)}"

    def test_endpoint_paths_are_documented(self, openapi_schema):
        """Test that all endpoint paths are properly documented."""
        paths = openapi_schema.get("paths", {})

        # Check for api endpoints
        api_paths = [p for p in paths if p.startswith("/api/")]
        assert len(api_paths) > 15, f"Expected many /api/ paths, got {len(api_paths)}"

        # Check key endpoints exist
        key_endpoints = [
            "/api/health",
            "/api/analyze",
            "/api/scan/upload",
            "/api/curves/generate",
            "/api/calibrations",
            "/api/chat",
        ]

        missing_endpoints = [
            ep for ep in key_endpoints if not any(p == ep for p in paths)
        ]
        assert not missing_endpoints, f"Missing key endpoints: {missing_endpoints}"

    def test_schema_json_validity(self, openapi_schema):
        """Test that schema can be serialized to JSON."""
        try:
            schema_json = json.dumps(openapi_schema)
            assert len(schema_json) > 1000  # Should be a substantial schema
        except TypeError as e:
            pytest.fail(f"Schema not JSON serializable: {e}")

    def test_type_validation_in_schemas(self, openapi_schema):
        """Test that schemas have proper type definitions."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        # Check a few key schemas have proper types
        test_schemas = ["HealthResponse", "AnalyzeResponse", "ChatResponse"]

        for schema_name in test_schemas:
            schema = schemas.get(schema_name, {})
            assert "properties" in schema, f"{schema_name} missing properties"
            assert "type" in schema or "properties" in schema, (
                f"{schema_name} missing type definition"
            )

    def test_schema_properties_have_descriptions(self, openapi_schema):
        """Test that schema properties have descriptions."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        test_schema = schemas.get("CurveGenerateResponse", {})
        properties = test_schema.get("properties", {})

        # At least some properties should have descriptions
        described_properties = [
            p for p in properties.values() if "description" in p
        ]
        assert len(described_properties) > 0, "No properties with descriptions"


class TestSchemaBackwardsCompatibility:
    """Tests for API backwards compatibility."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        return create_app()

    @pytest.fixture
    def openapi_schema(self, app):
        """Extract OpenAPI schema from app."""
        return app.openapi()

    def test_core_endpoints_exist(self, openapi_schema):
        """Test that core endpoints still exist (backwards compatibility)."""
        paths = openapi_schema.get("paths", {})

        core_endpoints = {
            "/": "get",
            "/api/health": "get",
            "/api/analyze": "post",
            "/api/scan/upload": "post",
            "/api/curves/generate": "post",
            "/api/curves/modify": "post",
            "/api/curves/smooth": "post",
            "/api/curves/blend": "post",
            "/api/curves/enhance": "post",
            "/api/calibrations": ["get", "post"],
            "/api/chat": "post",
            "/api/statistics": "get",
        }

        for path, methods in core_endpoints.items():
            assert path in paths, f"Endpoint {path} missing"
            path_item = paths[path]

            if isinstance(methods, str):
                assert methods in path_item, f"Method {methods} missing for {path}"
            else:
                for method in methods:
                    assert method in path_item, f"Method {method} missing for {path}"

    def test_response_field_presence(self, openapi_schema):
        """Test that all response models have required fields."""
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        test_cases = {
            "AnalyzeResponse": ["dmin", "dmax", "range", "is_monotonic"],
            "ScanUploadResponse": ["success", "extraction_id", "original_filename"],
            "CurveGenerateResponse": ["success", "curve_id", "name"],
        }

        for schema_name, required_fields in test_cases.items():
            schema = schemas.get(schema_name, {})
            properties = schema.get("properties", {})
            missing = [f for f in required_fields if f not in properties]
            assert not missing, f"{schema_name} missing fields: {missing}"


class TestSchemaExport:
    """Tests for schema export functionality."""

    def test_schema_file_exists(self):
        """Test that openapi.json file exists after generation."""
        schema_file = Path(__file__).parent.parent.parent / "openapi.json"
        assert schema_file.exists(), "openapi.json file not found"

    def test_schema_file_is_valid_json(self):
        """Test that openapi.json is valid JSON."""
        schema_file = Path(__file__).parent.parent.parent / "openapi.json"
        if schema_file.exists():
            with open(schema_file) as f:
                try:
                    schema = json.load(f)
                    assert isinstance(schema, dict)
                    assert "openapi" in schema
                except json.JSONDecodeError as e:
                    pytest.fail(f"openapi.json is not valid JSON: {e}")
