"""
Health check system for agentic operations.

Provides health monitoring, status reporting, and dependency checking
for the agent system with configurable thresholds.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.agents.logging import EventType, get_agent_logger


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DependencyType(str, Enum):
    """Types of dependencies that can be checked."""

    LLM_SERVICE = "llm_service"
    MESSAGE_BUS = "message_bus"
    MEMORY_SYSTEM = "memory_system"
    TOOL_REGISTRY = "tool_registry"
    SUBAGENT_REGISTRY = "subagent_registry"
    DATABASE = "database"
    HARDWARE = "hardware"


class HealthCheckSettings(BaseSettings):
    """Settings for health checks."""

    model_config = SettingsConfigDict(env_prefix="PTPD_HEALTH_")

    # Check intervals
    check_interval_seconds: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Interval between automatic health checks"
    )
    check_timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Timeout for individual health checks"
    )

    # Memory thresholds
    memory_warning_mb: float = Field(
        default=500.0, ge=100.0, le=10000.0, description="Memory usage warning threshold (MB)"
    )
    memory_critical_mb: float = Field(
        default=1000.0, ge=200.0, le=20000.0, description="Memory usage critical threshold (MB)"
    )

    # Queue thresholds
    queue_warning_depth: int = Field(
        default=100, ge=10, le=10000, description="Message queue warning depth"
    )
    queue_critical_depth: int = Field(
        default=500, ge=50, le=50000, description="Message queue critical depth"
    )

    # Workflow thresholds
    max_active_workflows: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent workflows before warning"
    )

    # Response time thresholds
    response_time_warning_ms: float = Field(
        default=1000.0, ge=100.0, le=30000.0, description="Response time warning threshold (ms)"
    )
    response_time_critical_ms: float = Field(
        default=5000.0, ge=500.0, le=60000.0, description="Response time critical threshold (ms)"
    )

    # Failure thresholds
    failure_rate_warning: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Failure rate warning threshold (0-1)"
    )
    failure_rate_critical: float = Field(
        default=0.25, ge=0.05, le=1.0, description="Failure rate critical threshold (0-1)"
    )


class DependencyHealth(BaseModel):
    """Health status of a single dependency."""

    name: str = Field(..., description="Dependency name")
    dependency_type: DependencyType = Field(..., description="Type of dependency")
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Current status")
    latency_ms: float | None = Field(default=None, description="Check latency in milliseconds")
    message: str | None = Field(default=None, description="Status message or error")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last check time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentHealthReport(BaseModel):
    """Comprehensive health report for the agent system."""

    status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Overall status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Report timestamp")

    # Component health
    llm_connected: bool = Field(default=False, description="LLM service connectivity")
    message_bus_active: bool = Field(default=False, description="Message bus status")
    memory_system_active: bool = Field(default=False, description="Memory system status")

    # Resource metrics
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage (MB)")
    active_workflows: int = Field(default=0, description="Number of active workflows")
    message_queue_depth: int = Field(default=0, description="Messages in queue")
    registered_tools: int = Field(default=0, description="Number of registered tools")
    registered_subagents: int = Field(default=0, description="Number of registered subagents")

    # Performance metrics
    avg_response_time_ms: float | None = Field(default=None, description="Average response time")
    success_rate: float | None = Field(default=None, description="Recent success rate (0-1)")
    total_requests: int = Field(default=0, description="Total requests processed")

    # Dependency details
    dependencies: list[DependencyHealth] = Field(
        default_factory=list, description="Individual dependency health"
    )

    # Issues
    issues: list[str] = Field(default_factory=list, description="Current health issues")
    warnings: list[str] = Field(default_factory=list, description="Health warnings")


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    healthy: bool
    latency_ms: float
    message: str | None = None
    metadata: dict = field(default_factory=dict)


class HealthChecker:
    """
    Health checker for the agent system.

    Performs health checks on all agent components and dependencies,
    aggregating results into a comprehensive health report.
    """

    def __init__(self, settings: HealthCheckSettings | None = None):
        """
        Initialize the health checker.

        Args:
            settings: Health check settings. Uses defaults if not provided.
        """
        self.settings = settings or HealthCheckSettings()
        self._logger = get_agent_logger()
        self._last_report: AgentHealthReport | None = None
        self._check_history: list[AgentHealthReport] = []
        self._max_history = 100

        # Performance tracking
        self._request_times: list[float] = []
        self._request_successes: list[bool] = []
        self._max_samples = 1000

    async def check_llm_connectivity(self) -> HealthCheckResult:
        """
        Check LLM service connectivity.

        Returns:
            HealthCheckResult with connectivity status.
        """
        start = time.time()
        try:
            # Lazy import to avoid circular dependency
            from ptpd_calibration.config import get_settings

            settings = get_settings()
            api_key = settings.llm.get_active_api_key()

            if not api_key:
                return HealthCheckResult(
                    name="llm_service",
                    healthy=False,
                    latency_ms=(time.time() - start) * 1000,
                    message="No API key configured",
                )

            # Check if we can create a client (without making actual API call)
            from ptpd_calibration.llm.client import LLMClient

            client = LLMClient()
            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name="llm_service",
                healthy=True,
                latency_ms=latency_ms,
                message="LLM client initialized",
                metadata={"provider": settings.llm.provider.value},
            )

        except ImportError as e:
            return HealthCheckResult(
                name="llm_service",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"LLM module not available: {e}",
            )
        except Exception as e:
            return HealthCheckResult(
                name="llm_service",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"LLM check failed: {e}",
            )

    async def check_message_bus(self) -> HealthCheckResult:
        """
        Check message bus health.

        Returns:
            HealthCheckResult with message bus status.
        """
        start = time.time()
        try:
            from ptpd_calibration.agents.communication import get_message_bus

            bus = get_message_bus()
            queue_depth = bus.queue_size() if bus else 0
            latency_ms = (time.time() - start) * 1000

            healthy = bus is not None
            status = "active" if healthy else "not initialized"

            # Check queue depth thresholds
            if queue_depth > self.settings.queue_critical_depth:
                healthy = False
                status = f"queue overloaded ({queue_depth} messages)"
            elif queue_depth > self.settings.queue_warning_depth:
                status = f"queue high ({queue_depth} messages)"

            return HealthCheckResult(
                name="message_bus",
                healthy=healthy,
                latency_ms=latency_ms,
                message=status,
                metadata={"queue_depth": queue_depth},
            )

        except Exception as e:
            return HealthCheckResult(
                name="message_bus",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Message bus check failed: {e}",
            )

    async def check_memory_system(self) -> HealthCheckResult:
        """
        Check memory system health and resource usage.

        Returns:
            HealthCheckResult with memory status.
        """
        start = time.time()
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            latency_ms = (time.time() - start) * 1000

            healthy = True
            status = f"Memory usage: {memory_mb:.1f} MB"

            if memory_mb > self.settings.memory_critical_mb:
                healthy = False
                status = f"Critical memory usage: {memory_mb:.1f} MB"
            elif memory_mb > self.settings.memory_warning_mb:
                status = f"High memory usage: {memory_mb:.1f} MB"

            return HealthCheckResult(
                name="memory_system",
                healthy=healthy,
                latency_ms=latency_ms,
                message=status,
                metadata={"memory_mb": memory_mb},
            )

        except ImportError:
            # psutil not installed
            return HealthCheckResult(
                name="memory_system",
                healthy=True,
                latency_ms=(time.time() - start) * 1000,
                message="Memory check unavailable (psutil not installed)",
                metadata={"memory_mb": 0},
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory_system",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Memory check failed: {e}",
            )

    async def check_tool_registry(self) -> HealthCheckResult:
        """
        Check tool registry health.

        Returns:
            HealthCheckResult with tool registry status.
        """
        start = time.time()
        try:
            from ptpd_calibration.agents.tools import ToolRegistry

            registry = ToolRegistry()
            tool_count = len(registry.list_tools())
            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name="tool_registry",
                healthy=True,
                latency_ms=latency_ms,
                message=f"{tool_count} tools registered",
                metadata={"tool_count": tool_count},
            )

        except Exception as e:
            return HealthCheckResult(
                name="tool_registry",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Tool registry check failed: {e}",
            )

    async def check_subagent_registry(self) -> HealthCheckResult:
        """
        Check subagent registry health.

        Returns:
            HealthCheckResult with subagent registry status.
        """
        start = time.time()
        try:
            from ptpd_calibration.agents.subagents.base import get_subagent_registry

            registry = get_subagent_registry()
            subagent_count = len(registry.list_subagents())
            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name="subagent_registry",
                healthy=True,
                latency_ms=latency_ms,
                message=f"{subagent_count} subagents registered",
                metadata={"subagent_count": subagent_count},
            )

        except Exception as e:
            return HealthCheckResult(
                name="subagent_registry",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Subagent registry check failed: {e}",
            )

    def record_request(self, duration_ms: float, success: bool) -> None:
        """
        Record a request for performance tracking.

        Args:
            duration_ms: Request duration in milliseconds.
            success: Whether the request succeeded.
        """
        self._request_times.append(duration_ms)
        self._request_successes.append(success)

        # Limit sample size
        if len(self._request_times) > self._max_samples:
            self._request_times = self._request_times[-self._max_samples :]
            self._request_successes = self._request_successes[-self._max_samples :]

    def _calculate_performance_metrics(self) -> tuple[float | None, float | None]:
        """
        Calculate performance metrics from recorded requests.

        Returns:
            Tuple of (average response time ms, success rate).
        """
        if not self._request_times:
            return None, None

        avg_time = sum(self._request_times) / len(self._request_times)
        success_rate = sum(self._request_successes) / len(self._request_successes)

        return avg_time, success_rate

    def _determine_overall_status(
        self,
        results: list[HealthCheckResult],
        memory_mb: float,
        avg_response_time: float | None,
        success_rate: float | None,
    ) -> tuple[HealthStatus, list[str], list[str]]:
        """
        Determine overall health status from individual check results.

        Args:
            results: List of health check results.
            memory_mb: Current memory usage.
            avg_response_time: Average response time.
            success_rate: Current success rate.

        Returns:
            Tuple of (status, issues list, warnings list).
        """
        issues: list[str] = []
        warnings: list[str] = []

        # Check individual results
        for result in results:
            if not result.healthy:
                issues.append(f"{result.name}: {result.message}")

        # Check memory thresholds
        if memory_mb > self.settings.memory_critical_mb:
            issues.append(f"Critical memory usage: {memory_mb:.1f} MB")
        elif memory_mb > self.settings.memory_warning_mb:
            warnings.append(f"High memory usage: {memory_mb:.1f} MB")

        # Check response time
        if avg_response_time is not None:
            if avg_response_time > self.settings.response_time_critical_ms:
                issues.append(f"Critical response time: {avg_response_time:.0f} ms")
            elif avg_response_time > self.settings.response_time_warning_ms:
                warnings.append(f"High response time: {avg_response_time:.0f} ms")

        # Check success rate
        if success_rate is not None:
            if success_rate < (1 - self.settings.failure_rate_critical):
                issues.append(f"High failure rate: {(1 - success_rate) * 100:.1f}%")
            elif success_rate < (1 - self.settings.failure_rate_warning):
                warnings.append(f"Elevated failure rate: {(1 - success_rate) * 100:.1f}%")

        # Determine overall status
        if issues:
            status = HealthStatus.UNHEALTHY
        elif warnings:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return status, issues, warnings

    async def check_health(self) -> AgentHealthReport:
        """
        Perform comprehensive health check.

        Returns:
            AgentHealthReport with full health status.
        """
        start = time.time()

        # Run all checks concurrently
        results = await asyncio.gather(
            self.check_llm_connectivity(),
            self.check_message_bus(),
            self.check_memory_system(),
            self.check_tool_registry(),
            self.check_subagent_registry(),
            return_exceptions=True,
        )

        # Process results
        check_results: list[HealthCheckResult] = []
        dependencies: list[DependencyHealth] = []

        type_mapping = {
            "llm_service": DependencyType.LLM_SERVICE,
            "message_bus": DependencyType.MESSAGE_BUS,
            "memory_system": DependencyType.MEMORY_SYSTEM,
            "tool_registry": DependencyType.TOOL_REGISTRY,
            "subagent_registry": DependencyType.SUBAGENT_REGISTRY,
        }

        for result in results:
            if isinstance(result, Exception):
                # Handle exception as unhealthy result
                check_results.append(
                    HealthCheckResult(
                        name="unknown",
                        healthy=False,
                        latency_ms=0,
                        message=str(result),
                    )
                )
            else:
                check_results.append(result)
                dependencies.append(
                    DependencyHealth(
                        name=result.name,
                        dependency_type=type_mapping.get(
                            result.name, DependencyType.DATABASE
                        ),
                        status=HealthStatus.HEALTHY if result.healthy else HealthStatus.UNHEALTHY,
                        latency_ms=result.latency_ms,
                        message=result.message,
                        metadata=result.metadata,
                    )
                )

        # Extract metrics from results
        memory_mb = 0.0
        queue_depth = 0
        tool_count = 0
        subagent_count = 0

        for result in check_results:
            if result.name == "memory_system":
                memory_mb = result.metadata.get("memory_mb", 0.0)
            elif result.name == "message_bus":
                queue_depth = result.metadata.get("queue_depth", 0)
            elif result.name == "tool_registry":
                tool_count = result.metadata.get("tool_count", 0)
            elif result.name == "subagent_registry":
                subagent_count = result.metadata.get("subagent_count", 0)

        # Calculate performance metrics
        avg_response_time, success_rate = self._calculate_performance_metrics()

        # Determine overall status
        status, issues, warnings = self._determine_overall_status(
            check_results, memory_mb, avg_response_time, success_rate
        )

        # Build report
        report = AgentHealthReport(
            status=status,
            timestamp=datetime.utcnow(),
            llm_connected=any(
                r.healthy for r in check_results if r.name == "llm_service"
            ),
            message_bus_active=any(
                r.healthy for r in check_results if r.name == "message_bus"
            ),
            memory_system_active=any(
                r.healthy for r in check_results if r.name == "memory_system"
            ),
            memory_usage_mb=memory_mb,
            active_workflows=0,  # Would need orchestrator integration
            message_queue_depth=queue_depth,
            registered_tools=tool_count,
            registered_subagents=subagent_count,
            avg_response_time_ms=avg_response_time,
            success_rate=success_rate,
            total_requests=len(self._request_times),
            dependencies=dependencies,
            issues=issues,
            warnings=warnings,
        )

        # Log health check
        self._logger.info(
            f"Health check completed: {status.value}",
            event_type=EventType.HEALTH_CHECK,
            duration_ms=(time.time() - start) * 1000,
            data={
                "status": status.value,
                "issues_count": len(issues),
                "warnings_count": len(warnings),
            },
        )

        # Store in history
        self._last_report = report
        self._check_history.append(report)
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history :]

        return report

    def get_last_report(self) -> AgentHealthReport | None:
        """Get the most recent health report."""
        return self._last_report

    def get_history(self, limit: int = 10) -> list[AgentHealthReport]:
        """
        Get recent health check history.

        Args:
            limit: Maximum number of reports to return.

        Returns:
            List of recent health reports.
        """
        return self._check_history[-limit:]

    def is_healthy(self) -> bool:
        """
        Quick check if system is healthy based on last report.

        Returns:
            True if last health status was HEALTHY.
        """
        if self._last_report is None:
            return True  # Assume healthy if not checked
        return self._last_report.status == HealthStatus.HEALTHY


# Global health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def check_agent_health() -> AgentHealthReport:
    """
    Convenience function to check agent health.

    Returns:
        AgentHealthReport with current health status.
    """
    checker = get_health_checker()
    return await checker.check_health()
