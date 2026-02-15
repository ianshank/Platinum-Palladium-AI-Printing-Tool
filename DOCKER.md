# Docker Setup Guide

This guide explains how to run the Platinum-Palladium AI Printing Tool using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- (Optional) API keys for Anthropic Claude and/or OpenAI

## Quick Start

### Production Build

Build and run the production container (API + Frontend static files):

```bash
# Build the image
docker compose build

# Start the API service
docker compose up api

# Or run in detached mode
docker compose up -d api
```

The API will be available at `http://localhost:8000`

### Development Mode

Run with live frontend development server (hot reload):

```bash
# Start API + frontend dev server
docker compose --profile dev up

# Or in detached mode
docker compose --profile dev up -d
```

- API: `http://localhost:8000`
- Frontend Dev Server: `http://localhost:5173`

## Environment Configuration

Create a `.env` file in the project root:

```bash
# API Configuration
PTPD_API_PORT=8000
PTPD_LOG_LEVEL=INFO

# LLM Provider (anthropic or openai)
PTPD_LLM_PROVIDER=anthropic
PTPD_LLM_ANTHROPIC_API_KEY=your-anthropic-api-key-here
PTPD_LLM_OPENAI_API_KEY=your-openai-api-key-here

# Frontend Dev Server
VITE_PORT=5173
```

See `.env.example` for a complete template.

## Docker Architecture

### Multi-Stage Build

The Dockerfile uses a multi-stage build process:

1. **frontend-builder**: Builds the React frontend with pnpm
2. **base**: Python base image with system dependencies
3. **python-builder**: Installs Python packages
4. **production**: Final production image with frontend static files

### Services

#### api (Production)
- FastAPI backend with static frontend files
- Non-root user (`ptpd`)
- Health check endpoint: `/api/health`
- Persistent volume: `ptpd-data` mounted at `/app/data`

#### frontend-dev (Development)
- Vite dev server with hot reload
- Enabled only with `--profile dev`
- Volume-mounted source code for live updates

## Common Commands

### Build and Run

```bash
# Build only
docker compose build

# Start services
docker compose up

# Start in background
docker compose up -d

# Start with frontend dev server
docker compose --profile dev up

# Rebuild and start
docker compose up --build
```

### Management

```bash
# View logs
docker compose logs -f api

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# Restart service
docker compose restart api

# Execute command in running container
docker compose exec api python -m ptpd_calibration.cli --help
```

### Debugging

```bash
# View health check status
docker compose ps

# Inspect container logs
docker compose logs api --tail=100

# Open shell in running container
docker compose exec api /bin/bash

# Run tests inside container
docker compose exec api pytest tests/
```

## Volume Management

### Data Persistence

The `ptpd-data` volume stores:
- Calibration profiles
- User settings
- Session logs
- Generated curves

```bash
# Backup data volume
docker compose exec api tar czf - /app/data > backup.tar.gz

# Restore data volume
docker compose exec -T api tar xzf - -C /app < backup.tar.gz

# Inspect volume
docker volume inspect ptpd-ai-printing-tool_ptpd-data

# Remove volume (WARNING: deletes all data)
docker volume rm ptpd-ai-printing-tool_ptpd-data
```

## Production Deployment

### Build for Production

```bash
# Build production image
docker compose build api

# Tag for registry
docker tag ptpd-ai-printing-tool_api:latest your-registry/ptpd-api:1.0.0

# Push to registry
docker push your-registry/ptpd-api:1.0.0
```

### Environment-Specific Configs

Use separate compose files for different environments:

```bash
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Security Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use secrets management** for API keys in production
3. **Enable HTTPS** with a reverse proxy (nginx, traefik)
4. **Scan images** for vulnerabilities: `docker scan ptpd-ai-printing-tool_api`
5. **Update base images** regularly

## Networking

### Container-to-Container

Services communicate via Docker's internal network:

```yaml
# Frontend calls API via internal hostname
VITE_API_URL=http://api:8000
```

### External Access

Map ports to host:

```yaml
ports:
  - "8000:8000"  # API
  - "5173:5173"  # Vite dev server
```

### Custom Network

Create custom network for isolation:

```bash
docker network create ptpd-network

# Update docker-compose.yml to use it
networks:
  default:
    external: true
    name: ptpd-network
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs api

# Verify health check
docker compose ps

# Inspect container
docker inspect ptpd-ai-printing-tool_api
```

### Permission Errors

The container runs as non-root user `ptpd`. If you encounter permission errors:

```bash
# Fix volume permissions
docker compose exec -u root api chown -R ptpd:ptpd /app/data
```

### Frontend Not Loading

1. Check API health: `curl http://localhost:8000/api/health`
2. Verify static files: `docker compose exec api ls -la /app/frontend/dist`
3. Check CORS settings in API

### Build Failures

```bash
# Clean build cache
docker compose build --no-cache

# Verify system resources
docker system df

# Prune unused resources
docker system prune -a
```

## Performance Optimization

### Image Size

Current image size: ~1.2GB (optimized with multi-stage build)

```bash
# View image size
docker images ptpd-ai-printing-tool_api

# Analyze layers
docker history ptpd-ai-printing-tool_api
```

### Resource Limits

Set resource constraints in `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Build Cache

Speed up builds with BuildKit:

```bash
export DOCKER_BUILDKIT=1
docker compose build
```

## Monitoring

### Health Checks

Built-in health check monitors `/api/health` endpoint:

```bash
# Manual health check
curl http://localhost:8000/api/health

# View health status
docker inspect --format='{{.State.Health.Status}}' ptpd-ai-printing-tool_api
```

### Metrics

Export metrics with Prometheus:

```bash
# Add metrics endpoint to API
docker compose exec api curl http://localhost:8000/metrics
```

## Advanced Usage

### Multi-Container Scaling

Run multiple API replicas:

```bash
docker compose up --scale api=3
```

### Custom Entrypoint

Override command for debugging:

```bash
docker compose run --rm api /bin/bash
```

### Development with Live Code

Mount source code for development:

```yaml
services:
  api:
    volumes:
      - ./src:/app/src:ro
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [React Docker Deployment](https://create-react-app.dev/docs/deployment/#docker)
