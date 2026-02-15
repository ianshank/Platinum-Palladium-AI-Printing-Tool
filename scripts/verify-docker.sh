#!/bin/bash
# Verify Docker setup for Platinum-Palladium AI Printing Tool

set -e

echo "=========================================="
echo "Docker Setup Verification"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is missing"
        return 1
    fi
}

check_version() {
    local cmd=$1
    local min_version=$2
    local current_version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [ -z "$current_version" ]; then
        echo -e "${YELLOW}⚠${NC} Could not determine $cmd version"
        return 1
    fi

    echo -e "${GREEN}✓${NC} $cmd version: $current_version"
    return 0
}

# 1. Check prerequisites
echo "1. Checking prerequisites..."
check_command docker
check_command docker-compose || check_command "docker compose"
echo ""

# 2. Check Docker files
echo "2. Checking Docker configuration files..."
check_file "Dockerfile"
check_file "docker-compose.yml"
check_file ".dockerignore"
check_file ".env.example"
echo ""

# 3. Check project structure
echo "3. Checking project structure..."
check_file "pyproject.toml"
check_file "frontend/package.json"
check_file "src/ptpd_calibration/api/server.py"
echo ""

# 4. Verify .env file
echo "4. Checking environment configuration..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"

    # Check for required variables
    required_vars=("PTPD_LLM_PROVIDER")
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" .env; then
            echo -e "  ${GREEN}✓${NC} $var is set"
        else
            echo -e "  ${YELLOW}⚠${NC} $var is not set in .env"
        fi
    done
else
    echo -e "${YELLOW}⚠${NC} .env file does not exist"
    echo "  Copy .env.example to .env and configure:"
    echo "  cp .env.example .env"
fi
echo ""

# 5. Test Docker build (dry run)
echo "5. Testing Docker build configuration..."
if docker compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose.yml is valid"
else
    echo -e "${RED}✗${NC} docker-compose.yml has errors"
    docker compose config
fi
echo ""

# 6. Check Docker daemon
echo "6. Checking Docker daemon..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"

    # Check Docker resources
    echo "  Docker info:"
    docker info --format '  - CPUs: {{.NCPU}}'
    docker info --format '  - Memory: {{.MemTotal}}'
else
    echo -e "${RED}✗${NC} Docker daemon is not running"
    echo "  Start Docker Desktop or run: sudo systemctl start docker"
fi
echo ""

# 7. Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Copy and configure .env:"
echo "     cp .env.example .env"
echo "     # Edit .env with your API keys"
echo ""
echo "  2. Build and start services:"
echo "     make build && make up"
echo "     # or: docker compose build && docker compose up -d api"
echo ""
echo "  3. For development with hot reload:"
echo "     make dev"
echo "     # or: docker compose --profile dev up"
echo ""
echo "  4. Check health:"
echo "     curl http://localhost:8000/api/health"
echo ""
echo "See DOCKER.md for detailed documentation."
echo ""
