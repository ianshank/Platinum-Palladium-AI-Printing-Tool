# Multi-stage build for PTPD Calibration API

# ---- Stage 1: Build frontend ----
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Enable pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Install dependencies
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Build frontend
COPY frontend/ ./
RUN pnpm build

# ---- Stage 2: Python base with system deps ----
FROM python:3.11-slim AS base

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage 3: Build Python dependencies ----
FROM base AS python-builder

WORKDIR /app

# Copy package configuration
COPY pyproject.toml ./

# Copy source code (required by hatchling build)
COPY src/ ./src/

# Install the package with api and llm extras
RUN pip install --no-cache-dir --prefix=/install ".[api,llm,ml]"

# ---- Stage 4: Production image ----
FROM base AS production

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=python-builder /install /usr/local

# Copy application source code
COPY src/ ./src/

# Copy built frontend static files
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create non-root user
RUN useradd -m -r ptpd && \
    mkdir -p /app/data && \
    chown -R ptpd:ptpd /app

USER ptpd

# Environment variables
ENV PYTHONPATH=/app \
    PTPD_DATA_DIR=/app/data \
    PTPD_API_HOST=0.0.0.0 \
    PTPD_API_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["ptpd-server"]
