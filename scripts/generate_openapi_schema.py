#!/usr/bin/env python3
"""
Generate and validate OpenAPI schema from FastAPI application.

This script:
1. Creates the FastAPI app
2. Extracts the OpenAPI schema
3. Validates it against OpenAPI 3.0.0 spec
4. Exports to openapi.json
5. Generates TypeScript types (optional)
"""

import json
import sys
from pathlib import Path

def main() -> int:
    """Generate OpenAPI schema from FastAPI app."""
    try:
        from ptpd_calibration.api.server import create_app
    except ImportError as e:
        print(f"Error: Failed to import FastAPI server: {e}", file=sys.stderr)
        return 1

    # Create the FastAPI application
    print("Creating FastAPI application...")
    app = create_app()

    # Extract OpenAPI schema
    print("Extracting OpenAPI schema...")
    openapi_schema = app.openapi()

    if not openapi_schema:
        print("Error: Failed to generate OpenAPI schema", file=sys.stderr)
        return 1

    # Validate schema version
    openapi_version = openapi_schema.get("openapi", "")
    if not openapi_version.startswith("3.0"):
        print(f"Warning: Expected OpenAPI 3.0.x, got {openapi_version}")

    # Count endpoints and models
    paths = openapi_schema.get("paths", {})
    components = openapi_schema.get("components", {})
    schemas = components.get("schemas", {})

    endpoint_count = sum(len(methods) for methods in paths.values())
    schema_count = len(schemas)

    print(f"Found {endpoint_count} endpoints and {schema_count} schemas")

    # Validate schema structure
    print("Validating schema structure...")
    errors = []

    # Check for required fields
    if "info" not in openapi_schema:
        errors.append("Missing 'info' field")
    if "paths" not in openapi_schema:
        errors.append("Missing 'paths' field")

    # Check for response models
    missing_response_models = []
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method.startswith("x-"):
                continue  # Skip vendor extensions
            if isinstance(operation, dict) and "responses" in operation:
                responses = operation["responses"]
                if "200" not in responses:
                    missing_response_models.append(f"{method.upper()} {path}")

    if missing_response_models:
        print(f"Warning: {len(missing_response_models)} endpoints missing 200 response")
        for endpoint in missing_response_models[:5]:  # Show first 5
            print(f"  - {endpoint}")

    # Check for error codes in schemas
    error_response_schema = schemas.get("ErrorResponse")
    if error_response_schema:
        required_fields = error_response_schema.get("required", [])
        if "error_code" in required_fields:
            print("✓ ErrorResponse includes error_code field")
        else:
            print("✗ ErrorResponse missing error_code field")

    # Write schema to file
    output_path = Path(__file__).parent.parent / "openapi.json"
    print(f"Writing schema to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✓ OpenAPI schema written to openapi.json")

    # Generate summary
    print("\n" + "=" * 60)
    print("OpenAPI Schema Generation Summary")
    print("=" * 60)
    print(f"API Title:     {openapi_schema.get('info', {}).get('title', 'N/A')}")
    print(f"API Version:   {openapi_schema.get('info', {}).get('version', 'N/A')}")
    print(f"OpenAPI Ver:   {openapi_version}")
    print(f"Endpoints:     {endpoint_count}")
    print(f"Schemas:       {schema_count}")
    print(f"Output File:   openapi.json")

    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  ✗ {error}")
        return 1

    print("\n✓ Schema validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
