# Docker Quick Start

Get the Platinum-Palladium AI Printing Tool running in 5 minutes.

## Prerequisites

- Docker Engine 20.10+ ([Install](https://docs.docker.com/engine/install/))
- Docker Compose 2.0+ (included with Docker Desktop)

## Quick Start

### 1. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings (optional - will work without API keys for basic features)
nano .env
```

**Minimum configuration:**
```bash
PTPD_LLM_PROVIDER=anthropic
# Add your API key if you want AI features
# PTPD_LLM_ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Start the Application

**Production Mode** (recommended for regular use):
```bash
make build    # Build the Docker image
make up       # Start the API server
```

Or without Make:
```bash
docker compose build
docker compose up -d api
```

**Development Mode** (with hot-reload frontend):
```bash
make dev
```

Or without Make:
```bash
docker compose --profile dev up
```

### 3. Access the Application

- **Production**: http://localhost:8000
- **Development**:
  - Frontend: http://localhost:5173
  - API: http://localhost:8000

### 4. Verify Health

```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{"status": "healthy"}
```

## Common Tasks

### View Logs
```bash
make logs
# or
docker compose logs -f api
```

### Stop Services
```bash
make down
# or
docker compose down
```

### Restart Services
```bash
docker compose restart api
```

### Open Shell in Container
```bash
make shell
# or
docker compose exec api /bin/bash
```

### Backup Data
```bash
make backup
# Creates timestamped backup in backups/ folder
```

### Restore Data
```bash
make restore FILE=backups/ptpd-data-20260215-120000.tar.gz
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs api

# Verify configuration
docker compose config

# Rebuild from scratch
make rebuild
```

### Permission errors
```bash
# Fix volume permissions
docker compose exec -u root api chown -R ptpd:ptpd /app/data
```

### Port already in use
```bash
# Change port in .env
echo "PTPD_API_PORT=8080" >> .env

# Restart
docker compose down && docker compose up -d api
```

### Out of disk space
```bash
# Clean unused Docker resources
docker system prune -a --volumes
```

## Environment Variables Reference

### Required
- `PTPD_LLM_PROVIDER` - LLM provider (anthropic or openai)

### Optional
- `PTPD_API_PORT` - API port (default: 8000)
- `PTPD_LLM_ANTHROPIC_API_KEY` - Anthropic API key
- `PTPD_LLM_OPENAI_API_KEY` - OpenAI API key
- `PTPD_LOG_LEVEL` - Log level (DEBUG, INFO, WARNING, ERROR)
- `VITE_PORT` - Frontend dev server port (default: 5173)

## Directory Structure

```
/app/
├── src/                    # Python backend code
├── frontend/dist/          # Built frontend (production)
└── data/                   # Persistent data volume
    ├── calibrations/       # Saved calibration profiles
    ├── curves/             # Generated curves
    └── logs/               # Application logs
```

## Next Steps

- Read full documentation: [DOCKER.md](DOCKER.md)
- Configure nginx reverse proxy: [nginx.conf.example](nginx.conf.example)
- Deploy to production: [docker-compose.prod.yml](docker-compose.prod.yml)
- Contribute: See [CLAUDE.md](CLAUDE.md) for development setup

## Support

- GitHub Issues: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/issues
- Documentation: See README.md and DOCKER.md
