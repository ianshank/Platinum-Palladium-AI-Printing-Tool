"""
Health Check System

Provides health check endpoints and monitoring for:
- Application startup verification
- Dependency availability
- Resource monitoring
- Graceful degradation detection
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System resource metrics."""

    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    cpu_percent: float | None = None
    disk_used_gb: float | None = None
    disk_available_gb: float | None = None
    disk_percent: float | None = None
    open_file_descriptors: int | None = None
    thread_count: int | None = None


class HealthReport(BaseModel):
    """Complete health report."""

    status: HealthStatus
    version: str
    environment: str
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: list[ComponentHealth] = Field(default_factory=list)
    metrics: SystemMetrics | None = None
    features: dict[str, bool] = Field(default_factory=dict)


@dataclass
class HealthChecker:
    """
    Health check function registration.

    Attributes:
        name: Component name
        check_fn: Async function that returns (status, message)
        critical: If True, failure makes overall status UNHEALTHY
        timeout_seconds: Check timeout
        cache_seconds: How long to cache results
    """

    name: str
    check_fn: Callable[[], Any]
    critical: bool = False
    timeout_seconds: float = 5.0
    cache_seconds: float = 10.0
    _last_result: ComponentHealth | None = field(default=None, repr=False)
    _last_check_time: float = field(default=0.0, repr=False)


class HealthCheck:
    """
    Centralized health check manager.

    Usage:
        health = HealthCheck(
            app_name="my_app",
            version="1.0.0",
            environment="production"
        )

        # Register component checks
        health.register("database", check_database, critical=True)
        health.register("cache", check_cache, critical=False)

        # Get health report
        report = await health.check_all()
    """

    _instance: HealthCheck | None = None
    _start_time: float = time.time()

    def __init__(
        self,
        app_name: str = "Application",
        version: str = "0.1.0",
        environment: str = "development",
    ):
        """Initialize health check manager."""
        self.app_name = app_name
        self.version = version
        self.environment = environment
        self._checkers: dict[str, HealthChecker] = {}
        self._start_time = time.time()

        # Register built-in checks
        self._register_builtin_checks()

    @classmethod
    def get_instance(cls) -> HealthCheck:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = HealthCheck()
        return cls._instance

    @classmethod
    def configure(
        cls,
        app_name: str,
        version: str,
        environment: str,
    ) -> HealthCheck:
        """Configure and return singleton instance."""
        cls._instance = HealthCheck(
            app_name=app_name,
            version=version,
            environment=environment,
        )
        return cls._instance

    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""
        self.register("system", self._check_system, critical=True)
        self.register("python", self._check_python, critical=True)
        self.register("filesystem", self._check_filesystem, critical=False)

    def register(
        self,
        name: str,
        check_fn: Callable[[], Any],
        critical: bool = False,
        timeout_seconds: float = 5.0,
        cache_seconds: float = 10.0,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Component name
            check_fn: Function returning (status, message) or just status
            critical: If True, failure causes overall UNHEALTHY status
            timeout_seconds: Check timeout
            cache_seconds: Result cache duration
        """
        self._checkers[name] = HealthChecker(
            name=name,
            check_fn=check_fn,
            critical=critical,
            timeout_seconds=timeout_seconds,
            cache_seconds=cache_seconds,
        )
        logger.debug(f"Registered health check: {name}", critical=critical)

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        if name in self._checkers:
            del self._checkers[name]
            logger.debug(f"Unregistered health check: {name}")

    async def _run_check(
        self,
        checker: HealthChecker,
    ) -> ComponentHealth:
        """Run a single health check with timeout and caching."""
        # Check cache
        now = time.time()
        if (
            checker._last_result
            and (now - checker._last_check_time) < checker.cache_seconds
        ):
            return checker._last_result

        start_time = time.perf_counter()

        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(checker.check_fn):
                result = await asyncio.wait_for(
                    checker.check_fn(),
                    timeout=checker.timeout_seconds,
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, checker.check_fn),
                    timeout=checker.timeout_seconds,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse result
            if isinstance(result, tuple):
                status, message = result
            elif isinstance(result, HealthStatus):
                status, message = result, None
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = None
            else:
                status = HealthStatus.HEALTHY
                message = str(result) if result else None

            health = ComponentHealth(
                name=checker.name,
                status=status,
                message=message,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            health = ComponentHealth(
                name=checker.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {checker.timeout_seconds}s",
                latency_ms=checker.timeout_seconds * 1000,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            health = ComponentHealth(
                name=checker.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency_ms,
            )

        # Update cache
        checker._last_result = health
        checker._last_check_time = now

        return health

    async def check_component(self, name: str) -> ComponentHealth | None:
        """Check a specific component."""
        if name not in self._checkers:
            return None
        return await self._run_check(self._checkers[name])

    async def check_all(
        self,
        include_metrics: bool = True,
        include_features: bool = True,
    ) -> HealthReport:
        """
        Run all health checks and return report.

        Args:
            include_metrics: Include system metrics
            include_features: Include feature flags status

        Returns:
            Complete health report
        """
        # Run all checks concurrently
        results = await asyncio.gather(
            *[self._run_check(checker) for checker in self._checkers.values()],
            return_exceptions=True,
        )

        # Process results
        components: list[ComponentHealth] = []
        has_critical_failure = False
        has_any_failure = False

        for i, result in enumerate(results):
            checker = list(self._checkers.values())[i]

            if isinstance(result, Exception):
                health = ComponentHealth(
                    name=checker.name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                )
            else:
                health = result

            components.append(health)

            if health.status == HealthStatus.UNHEALTHY:
                has_any_failure = True
                if checker.critical:
                    has_critical_failure = True

        # Determine overall status
        if has_critical_failure:
            overall_status = HealthStatus.UNHEALTHY
        elif has_any_failure:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Build report
        report = HealthReport(
            status=overall_status,
            version=self.version,
            environment=self.environment,
            uptime_seconds=time.time() - self._start_time,
            components=components,
        )

        if include_metrics:
            report.metrics = self._get_system_metrics()

        if include_features:
            report.features = self._get_feature_status()

        return report

    def check_all_sync(
        self,
        include_metrics: bool = True,
        include_features: bool = True,
    ) -> HealthReport:
        """Synchronous version of check_all."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.check_all(include_metrics, include_features)
        )

    # Built-in checks

    def _check_system(self) -> tuple[HealthStatus, str]:
        """Check system resources."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            if memory.percent > 95:
                return HealthStatus.UNHEALTHY, f"Memory critical: {memory.percent}%"
            elif memory.percent > 85:
                return HealthStatus.DEGRADED, f"Memory high: {memory.percent}%"

            return HealthStatus.HEALTHY, f"Memory: {memory.percent}%"

        except ImportError:
            # psutil not available, do basic check
            return HealthStatus.HEALTHY, "Basic check (psutil not available)"

    def _check_python(self) -> tuple[HealthStatus, str]:
        """Check Python environment."""
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return HealthStatus.HEALTHY, f"Python {version}"

    def _check_filesystem(self) -> tuple[HealthStatus, str]:
        """Check filesystem access."""
        try:
            from ptpd_calibration.template.config import get_template_config

            config = get_template_config()
            data_dir = config.data_dir

            # Check if directory exists or can be created
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)

            # Check write access
            test_file = data_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            return HealthStatus.HEALTHY, f"Data dir accessible: {data_dir}"

        except Exception as e:
            return HealthStatus.DEGRADED, f"Filesystem issue: {e}"

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu = psutil.cpu_percent(interval=0.1)

            return SystemMetrics(
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent,
                cpu_percent=cpu,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_available_gb=disk.free / (1024 * 1024 * 1024),
                disk_percent=disk.percent,
                thread_count=len(psutil.Process().threads()),
            )

        except ImportError:
            # Fallback without psutil
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)

            return SystemMetrics(
                memory_used_mb=rusage.ru_maxrss / 1024,  # KB to MB on Linux
                memory_available_mb=0,  # Unknown
                memory_percent=0,  # Unknown
            )

    def _get_feature_status(self) -> dict[str, bool]:
        """Get feature availability status."""
        features: dict[str, bool] = {}

        # Check deep learning
        try:
            import torch
            features["deep_learning"] = True
            features["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            features["deep_learning"] = False
            features["cuda_available"] = False

        # Check LLM
        try:
            import anthropic
            features["llm_anthropic"] = True
        except ImportError:
            features["llm_anthropic"] = False

        try:
            import openai
            features["llm_openai"] = True
        except ImportError:
            features["llm_openai"] = False

        # Check UI
        try:
            import gradio
            features["gradio_ui"] = True
        except ImportError:
            features["gradio_ui"] = False

        # Check API
        try:
            import fastapi
            features["fastapi_api"] = True
        except ImportError:
            features["fastapi_api"] = False

        return features


# FastAPI health endpoint
def health_check_endpoint():
    """
    Create FastAPI health check router.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(health_check_endpoint())
    """
    try:
        from fastapi import APIRouter, Response
        from fastapi.responses import JSONResponse
    except ImportError:
        return None

    router = APIRouter(tags=["health"])
    health = HealthCheck.get_instance()

    @router.get("/health")
    async def get_health() -> JSONResponse:
        """Get health status."""
        report = await health.check_all()

        status_code = {
            HealthStatus.HEALTHY: 200,
            HealthStatus.DEGRADED: 200,
            HealthStatus.UNHEALTHY: 503,
            HealthStatus.UNKNOWN: 503,
        }.get(report.status, 503)

        return JSONResponse(
            status_code=status_code,
            content=report.model_dump(mode="json"),
        )

    @router.get("/health/live")
    async def get_liveness() -> Response:
        """Kubernetes liveness probe."""
        return Response(status_code=200, content="OK")

    @router.get("/health/ready")
    async def get_readiness() -> JSONResponse:
        """Kubernetes readiness probe."""
        report = await health.check_all(include_metrics=False, include_features=False)

        if report.status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "status": report.status.value},
            )

        return JSONResponse(
            status_code=200,
            content={"ready": True, "status": report.status.value},
        )

    @router.get("/health/{component}")
    async def get_component_health(component: str) -> JSONResponse:
        """Get health of specific component."""
        result = await health.check_component(component)

        if result is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Component '{component}' not found"},
            )

        return JSONResponse(
            status_code=200 if result.status != HealthStatus.UNHEALTHY else 503,
            content=result.model_dump(mode="json"),
        )

    return router


# Gradio health component
def create_gradio_health_display():
    """
    Create Gradio component for health status display.

    Usage:
        with gr.Blocks() as demo:
            health_display = create_gradio_health_display()
    """
    try:
        import gradio as gr
    except ImportError:
        return None

    health = HealthCheck.get_instance()

    def get_health_html() -> str:
        """Generate health status HTML."""
        report = health.check_all_sync()

        status_colors = {
            HealthStatus.HEALTHY: "#22c55e",
            HealthStatus.DEGRADED: "#f59e0b",
            HealthStatus.UNHEALTHY: "#ef4444",
            HealthStatus.UNKNOWN: "#6b7280",
        }

        html = f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background: #1f2937;">
            <h3 style="margin: 0 0 1rem 0; color: white;">
                System Status:
                <span style="color: {status_colors[report.status]};">
                    {report.status.value.upper()}
                </span>
            </h3>
            <div style="display: grid; gap: 0.5rem;">
        """

        for component in report.components:
            html += f"""
                <div style="display: flex; justify-content: space-between;
                            padding: 0.5rem; background: #374151; border-radius: 0.25rem;">
                    <span style="color: white;">{component.name}</span>
                    <span style="color: {status_colors[component.status]};">
                        {component.status.value}
                    </span>
                </div>
            """

        html += "</div></div>"
        return html

    with gr.Blocks() as health_block:
        health_html = gr.HTML(value=get_health_html)
        refresh_btn = gr.Button("Refresh Status", size="sm")
        refresh_btn.click(fn=get_health_html, outputs=health_html)

    return health_block
