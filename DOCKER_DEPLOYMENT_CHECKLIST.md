# Docker Deployment Checklist

Use this checklist to validate your Docker deployment before going to production.

## Pre-Deployment

### Environment Setup
- [ ] Copy `.env.example` to `.env`
- [ ] Configure `PTPD_LLM_PROVIDER` (anthropic or openai)
- [ ] Set `PTPD_LLM_ANTHROPIC_API_KEY` or `PTPD_LLM_OPENAI_API_KEY`
- [ ] Set `PTPD_LOG_LEVEL` to `WARNING` or `ERROR` for production
- [ ] Set `PTPD_API_PORT` (default: 8000)
- [ ] Verify `.env` is in `.gitignore`

### Docker Setup
- [ ] Docker Engine 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] Docker daemon is running
- [ ] Sufficient disk space (minimum 5 GB)
- [ ] Network connectivity for image pulls

### Code Validation
- [ ] Run verification script: `bash scripts/verify-docker.sh`
- [ ] Validate docker-compose: `docker compose config`
- [ ] Review Dockerfile for customizations
- [ ] Check .dockerignore excludes sensitive files

## Build Phase

### Image Build
- [ ] Build image: `make build` or `docker compose build`
- [ ] Build completes without errors
- [ ] Check image size: `docker images | grep ptpd`
- [ ] Image size is reasonable (~1-2 GB)
- [ ] No sensitive data in image layers

### Build Verification
- [ ] Frontend build artifacts exist in image: `docker run --rm IMAGE ls /app/frontend/dist`
- [ ] Python packages installed: `docker run --rm IMAGE pip list`
- [ ] Source code copied: `docker run --rm IMAGE ls /app/src`
- [ ] Non-root user created: `docker run --rm IMAGE whoami` (should be `ptpd`)

## Testing Phase

### Container Startup
- [ ] Start container: `make up` or `docker compose up -d api`
- [ ] Container starts successfully
- [ ] Container remains running (not restarting)
- [ ] Health check passes: `docker compose ps` shows "healthy"

### API Validation
- [ ] Health endpoint responds: `curl http://localhost:8000/api/health`
- [ ] Response is `{"status": "healthy"}` or similar
- [ ] API docs available: `curl http://localhost:8000/docs`
- [ ] OpenAPI spec available: `curl http://localhost:8000/openapi.json`

### Frontend Validation (Production)
- [ ] Frontend loads: `curl http://localhost:8000/`
- [ ] Static assets load: `curl http://localhost:8000/assets/`
- [ ] No 404 errors in browser console

### Development Mode (Optional)
- [ ] Start dev mode: `make dev`
- [ ] Frontend dev server starts on port 5173
- [ ] Hot reload works (modify a file and check browser)
- [ ] API proxy works (frontend calls to /api/)

### Data Persistence
- [ ] Create test calibration via API
- [ ] Stop container: `docker compose down`
- [ ] Start container: `docker compose up -d api`
- [ ] Test calibration still exists
- [ ] Data volume persists: `docker volume ls | grep ptpd-data`

### Logs
- [ ] View logs: `make logs` or `docker compose logs api`
- [ ] No critical errors in logs
- [ ] Log level appropriate for environment
- [ ] Log format is structured and parseable

## Security Review

### Container Security
- [ ] Container runs as non-root user (ptpd)
- [ ] No privileged containers
- [ ] Security options configured (production):
  - [ ] `no-new-privileges:true`
  - [ ] `read-only` filesystem
- [ ] Health check configured

### Network Security
- [ ] Only necessary ports exposed
- [ ] HTTPS configured (if using nginx)
- [ ] SSL certificates valid and not expired
- [ ] Security headers configured (nginx)

### Secrets Management
- [ ] No API keys in Dockerfile
- [ ] No secrets in docker-compose.yml
- [ ] All secrets in .env or external secret manager
- [ ] .env not committed to git

### Image Security
- [ ] Run security scan: `docker scan IMAGE` or Trivy
- [ ] Address critical vulnerabilities
- [ ] Base image is recent and supported
- [ ] Minimal attack surface (slim base image)

## Performance Testing

### Resource Usage
- [ ] Monitor CPU usage: `docker stats`
- [ ] Monitor memory usage: `docker stats`
- [ ] Resource limits configured (production)
- [ ] No memory leaks over time

### Load Testing
- [ ] API handles expected request volume
- [ ] Response times acceptable under load
- [ ] Graceful degradation under heavy load
- [ ] Container doesn't crash under stress

### Startup Performance
- [ ] Container starts within 30 seconds
- [ ] Health check passes within 60 seconds
- [ ] API responds within 5 seconds of health check

## Production Configuration

### docker-compose.prod.yml
- [ ] Resource limits configured
- [ ] Logging driver configured
- [ ] Restart policy set to `always`
- [ ] Security options enabled
- [ ] Production environment variables set

### Nginx (if applicable)
- [ ] nginx.conf customized for your domain
- [ ] SSL certificates in place
- [ ] HTTPS redirect configured
- [ ] Rate limiting configured
- [ ] Client max body size appropriate for uploads

### Backups
- [ ] Backup strategy defined
- [ ] Test backup: `make backup`
- [ ] Test restore: `make restore FILE=backup.tar.gz`
- [ ] Automated backup scheduled (cron/systemd)

## Deployment

### Initial Deployment
- [ ] Stop any running instances
- [ ] Deploy: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
- [ ] Verify health: `make health`
- [ ] Test all critical endpoints
- [ ] Monitor logs for errors

### Smoke Testing
- [ ] Load homepage: works
- [ ] Upload image: works
- [ ] Generate calibration: works
- [ ] Export curve: works
- [ ] AI features: work (if configured)

### Monitoring Setup
- [ ] Health check endpoint monitored
- [ ] Logs aggregated (if using external logging)
- [ ] Alerts configured for downtime
- [ ] Metrics collection (if using Prometheus)

## Post-Deployment

### Verification
- [ ] Application accessible from expected URLs
- [ ] All features working as expected
- [ ] Performance within acceptable range
- [ ] No errors in logs

### Documentation
- [ ] Update deployment documentation
- [ ] Document any customizations
- [ ] Record deployment date and version
- [ ] Share access credentials with team

### Maintenance Plan
- [ ] Schedule regular backups
- [ ] Plan for image updates
- [ ] Define rollback procedure
- [ ] Document troubleshooting steps

## Rollback Plan

If deployment fails:

- [ ] Stop new deployment: `docker compose down`
- [ ] Restore previous version: `docker compose pull OLD_IMAGE && docker compose up -d`
- [ ] Restore data backup if needed
- [ ] Verify rollback successful
- [ ] Document issues for post-mortem

## Continuous Monitoring

### Daily
- [ ] Check container status: `docker compose ps`
- [ ] Review logs for errors: `make logs`
- [ ] Verify health endpoint: `curl http://localhost:8000/api/health`

### Weekly
- [ ] Backup data: `make backup`
- [ ] Review resource usage: `docker stats`
- [ ] Check disk space: `docker system df`

### Monthly
- [ ] Update base images: `docker compose pull && docker compose up -d`
- [ ] Review security advisories
- [ ] Prune unused resources: `docker system prune -a`
- [ ] Test restore from backup

## Troubleshooting Quick Reference

| Issue | Command |
|-------|---------|
| Container won't start | `docker compose logs api` |
| Health check failing | `docker compose exec api curl http://localhost:8000/api/health` |
| Permission errors | `docker compose exec -u root api chown -R ptpd:ptpd /app/data` |
| Port conflicts | Edit `PTPD_API_PORT` in .env |
| Out of memory | Reduce resource limits or increase host resources |
| Slow performance | Check `docker stats` and adjust limits |

## Support Resources

- **Documentation**: DOCKER.md, DOCKER_QUICKSTART.md
- **GitHub Issues**: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/issues
- **Logs**: `make logs`
- **Health**: `make health`
- **Shell Access**: `make shell`

---

## Deployment Sign-off

Date: ________________

Deployed by: ________________

Environment: ☐ Development  ☐ Staging  ☐ Production

All checks completed: ☐ Yes  ☐ No (see notes below)

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

Approved by: ________________  Date: ________________
