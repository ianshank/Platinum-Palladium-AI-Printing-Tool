#!/bin/bash
# Generate TypeScript types from OpenAPI schema

set -e

echo "Generating TypeScript types from OpenAPI schema..."

# Ensure openapi.json exists
if [ ! -f openapi.json ]; then
    echo "Error: openapi.json not found. Run 'python scripts/generate_openapi_schema.py' first."
    exit 1
fi

# Create generated types directory
TYPES_DIR="frontend/src/api/generated"
mkdir -p "$TYPES_DIR"

# Copy openapi.json to a location accessible by frontend
cp openapi.json "$TYPES_DIR/../openapi.json"

# Run openapi-typescript to generate types
echo "Running openapi-typescript..."
cd frontend
pnpm run generate:api

echo "✓ TypeScript types generated successfully"
echo "Generated types are in: $TYPES_DIR/schema.ts"

# Print summary
echo ""
echo "Files generated:"
ls -lh src/api/generated/schema.ts 2>/dev/null || echo "  (check if file exists)"
