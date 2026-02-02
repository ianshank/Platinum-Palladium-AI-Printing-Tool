"""
Comprehensive tests for observability modules.

Tests health checks, circuit breakers, metrics, and workflow persistence.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Health check tests
from ptpd_calibration.agents.health import (
    AgentHealthReport,
    DependencyHealth,
    DependencyType,
    HealthChecker,
    HealthCheckResult,
    HealthCheckSettings,
    HealthStatus,
    check_agent_health,
    get_health_checker,
)

# Circuit breaker tests
from ptpd_calibration.agents.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerSettings,
    CircuitBreakerState,
    CircuitOpenError,
    CircuitState,
    get_all_circuit_breaker_stats,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

# Metrics tests
from ptpd_calibration.agents.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricSample,
    MetricsRegistry,
    MetricsSettings,
    MetricType,
    get_agent_metrics,
    get_metrics_registry,
    record_request,
    record_token_usage,
    record_tool_call,
)

# Persistence tests
from ptpd_calibration.agents.persistence import (
    PersistenceSettings,
    TaskCheckpoint,
    WorkflowCheckpoint,
    WorkflowPersistence,
    WorkflowState,
    create_workflow_checkpoint,
    get_persistence,
)


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Verify all status values exist."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"

    def test_health_status_is_string_enum(self):
        """Verify HealthStatus can be used as string."""
        assert str(HealthStatus.HEALTHY) == "HealthStatus.HEALTHY"
        assert HealthStatus.HEALTHY.value == "healthy"


class TestDependencyType:
    """Tests for DependencyType enum."""

    def test_dependency_types_exist(self):
        """Verify all dependency types."""
        assert DependencyType.LLM_SERVICE == "llm_service"
        assert DependencyType.MESSAGE_BUS == "message_bus"
        assert DependencyType.MEMORY_SYSTEM == "memory_system"
        assert DependencyType.TOOL_REGISTRY == "tool_registry"
        assert DependencyType.SUBAGENT_REGISTRY == "subagent_registry"


class TestHealthCheckSettings:
    """Tests for HealthCheckSettings."""

    def test_default_settings(self):
        """Verify default settings are sensible."""
        settings = HealthCheckSettings()
        assert settings.check_interval_seconds == 30.0
        assert settings.check_timeout_seconds == 10.0
        assert settings.memory_warning_mb == 500.0
        assert settings.memory_critical_mb == 1000.0
        assert settings.queue_warning_depth == 100
        assert settings.queue_critical_depth == 500

    def test_custom_settings(self):
        """Verify custom settings work."""
        settings = HealthCheckSettings(
            check_interval_seconds=60.0,
            memory_warning_mb=1000.0,
        )
        assert settings.check_interval_seconds == 60.0
        assert settings.memory_warning_mb == 1000.0


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_create_healthy_result(self):
        """Create a healthy result."""
        result = HealthCheckResult(
            name="test_check",
            healthy=True,
            latency_ms=10.5,
            message="All good",
        )
        assert result.name == "test_check"
        assert result.healthy is True
        assert result.latency_ms == 10.5
        assert result.message == "All good"

    def test_create_unhealthy_result(self):
        """Create an unhealthy result."""
        result = HealthCheckResult(
            name="test_check",
            healthy=False,
            latency_ms=100.0,
            message="Service unavailable",
        )
        assert result.healthy is False

    def test_result_with_metadata(self):
        """Result can include metadata."""
        result = HealthCheckResult(
            name="test_check",
            healthy=True,
            latency_ms=5.0,
            metadata={"key": "value"},
        )
        assert result.metadata["key"] == "value"


class TestDependencyHealth:
    """Tests for DependencyHealth model."""

    def test_create_dependency_health(self):
        """Create a dependency health record."""
        health = DependencyHealth(
            name="llm_service",
            dependency_type=DependencyType.LLM_SERVICE,
            status=HealthStatus.HEALTHY,
            latency_ms=50.0,
            message="Connected",
        )
        assert health.name == "llm_service"
        assert health.dependency_type == DependencyType.LLM_SERVICE
        assert health.status == HealthStatus.HEALTHY

    def test_dependency_health_defaults(self):
        """Verify default values."""
        health = DependencyHealth(
            name="test",
            dependency_type=DependencyType.DATABASE,
        )
        assert health.status == HealthStatus.UNKNOWN
        assert health.latency_ms is None
        assert health.metadata == {}


class TestAgentHealthReport:
    """Tests for AgentHealthReport model."""

    def test_create_report(self):
        """Create a health report."""
        report = AgentHealthReport(
            status=HealthStatus.HEALTHY,
            llm_connected=True,
            message_bus_active=True,
            memory_usage_mb=256.0,
            active_workflows=2,
        )
        assert report.status == HealthStatus.HEALTHY
        assert report.llm_connected is True
        assert report.memory_usage_mb == 256.0

    def test_report_with_issues(self):
        """Report can track issues and warnings."""
        report = AgentHealthReport(
            status=HealthStatus.DEGRADED,
            issues=["Service slow"],
            warnings=["High memory"],
        )
        assert len(report.issues) == 1
        assert len(report.warnings) == 1

    def test_report_defaults(self):
        """Verify default values."""
        report = AgentHealthReport()
        assert report.status == HealthStatus.UNKNOWN
        assert report.llm_connected is False
        assert report.memory_usage_mb == 0.0
        assert report.dependencies == []


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a health checker instance."""
        return HealthChecker(HealthCheckSettings())

    def test_create_checker(self, checker):
        """Verify checker creation."""
        assert checker is not None
        assert checker.settings is not None

    def test_record_request(self, checker):
        """Record request performance data."""
        checker.record_request(100.0, True)
        checker.record_request(200.0, False)

        assert len(checker._request_times) == 2
        assert len(checker._request_successes) == 2

    def test_calculate_performance_metrics_empty(self, checker):
        """Calculate metrics with no data."""
        avg_time, success_rate = checker._calculate_performance_metrics()
        assert avg_time is None
        assert success_rate is None

    def test_calculate_performance_metrics(self, checker):
        """Calculate metrics with data."""
        checker.record_request(100.0, True)
        checker.record_request(200.0, True)
        checker.record_request(300.0, False)

        avg_time, success_rate = checker._calculate_performance_metrics()
        assert avg_time == 200.0
        assert success_rate == pytest.approx(2 / 3)

    def test_determine_status_healthy(self, checker):
        """Determine healthy status."""
        results = [
            HealthCheckResult("test1", True, 10.0),
            HealthCheckResult("test2", True, 20.0),
        ]
        status, issues, warnings = checker._determine_overall_status(
            results, 100.0, None, None
        )
        assert status == HealthStatus.HEALTHY
        assert len(issues) == 0
        assert len(warnings) == 0

    def test_determine_status_unhealthy_failed_check(self, checker):
        """Determine unhealthy status from failed check."""
        results = [
            HealthCheckResult("test1", False, 10.0, "Error"),
        ]
        status, issues, warnings = checker._determine_overall_status(
            results, 100.0, None, None
        )
        assert status == HealthStatus.UNHEALTHY
        assert len(issues) > 0

    def test_determine_status_degraded_high_memory(self, checker):
        """Determine degraded status from high memory."""
        results = [
            HealthCheckResult("test1", True, 10.0),
        ]
        status, issues, warnings = checker._determine_overall_status(
            results, 600.0, None, None  # Above warning threshold
        )
        assert status == HealthStatus.DEGRADED
        assert len(warnings) > 0

    def test_is_healthy_no_report(self, checker):
        """Check healthy status with no report."""
        assert checker.is_healthy() is True  # Assume healthy if not checked

    def test_get_last_report_none(self, checker):
        """Get last report when none exists."""
        assert checker.get_last_report() is None

    @pytest.mark.asyncio
    async def test_check_health_runs(self, checker):
        """Verify health check runs without errors."""
        with patch.object(checker, "check_llm_connectivity") as mock_llm, \
             patch.object(checker, "check_message_bus") as mock_bus, \
             patch.object(checker, "check_memory_system") as mock_mem, \
             patch.object(checker, "check_tool_registry") as mock_tools, \
             patch.object(checker, "check_subagent_registry") as mock_subagents:

            mock_llm.return_value = HealthCheckResult("llm_service", True, 10.0)
            mock_bus.return_value = HealthCheckResult("message_bus", True, 5.0, metadata={"queue_depth": 0})
            mock_mem.return_value = HealthCheckResult("memory_system", True, 2.0, metadata={"memory_mb": 100.0})
            mock_tools.return_value = HealthCheckResult("tool_registry", True, 1.0, metadata={"tool_count": 5})
            mock_subagents.return_value = HealthCheckResult("subagent_registry", True, 1.0, metadata={"subagent_count": 4})

            report = await checker.check_health()

            assert report is not None
            assert report.status == HealthStatus.HEALTHY
            assert checker.get_last_report() == report


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_states_exist(self):
        """Verify all circuit states."""
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"


class TestCircuitBreakerSettings:
    """Tests for CircuitBreakerSettings."""

    def test_default_settings(self):
        """Verify default settings."""
        settings = CircuitBreakerSettings()
        assert settings.failure_threshold == 3
        assert settings.success_threshold == 2
        assert settings.cooldown_seconds == 30.0
        assert settings.enable_fallback is True

    def test_custom_settings(self):
        """Verify custom settings work."""
        settings = CircuitBreakerSettings(
            failure_threshold=5,
            cooldown_seconds=60.0,
        )
        assert settings.failure_threshold == 5
        assert settings.cooldown_seconds == 60.0


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_create_error(self):
        """Create circuit open error."""
        error = CircuitOpenError(
            message="Circuit open",
            service_name="llm_service",
            retry_after=30.0,
        )
        assert str(error) == "Circuit open"
        assert error.service_name == "llm_service"
        assert error.retry_after == 30.0

    def test_error_defaults(self):
        """Error has sensible defaults."""
        error = CircuitOpenError()
        assert "open" in str(error).lower()
        assert error.service_name is None
        assert error.retry_after is None


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker instance."""
        # Create breaker with default settings, then modify state directly for tests
        breaker = CircuitBreaker("test_service")
        # Override cooldown for testing
        breaker.settings.cooldown_seconds = 5.0  # Use minimum valid value
        return breaker

    def test_create_breaker(self, breaker):
        """Verify breaker creation."""
        assert breaker.name == "test_service"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    def test_initial_state_closed(self, breaker):
        """Breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        """Successful calls pass through."""
        async def success():
            return "result"

        result = await breaker.call(success)
        assert result == "result"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_increments_count(self, breaker):
        """Failures increment failure count."""
        async def failure():
            raise ValueError("test error")

        # Set threshold high so we don't open on first failure
        breaker.settings.failure_threshold = 5

        with pytest.raises(ValueError):
            await breaker.call(failure)

        assert breaker._state.failure_count == 1
        assert breaker.state == CircuitState.CLOSED  # Not open yet

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        """Circuit opens after failure threshold."""
        async def failure():
            raise ValueError("test error")

        # Override failure threshold for this test
        breaker.settings.failure_threshold = 2

        # First failure
        with pytest.raises(ValueError):
            await breaker.call(failure)
        assert breaker.state == CircuitState.CLOSED

        # Second failure (threshold=2)
        with pytest.raises(ValueError):
            await breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, breaker):
        """Open circuit blocks calls."""
        async def failure():
            raise ValueError("test error")

        # Override failure threshold for this test
        breaker.settings.failure_threshold = 2

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

        # Next call should be blocked
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(failure)

        assert exc_info.value.service_name == "test_service"

    @pytest.mark.asyncio
    async def test_fallback_when_open(self, breaker):
        """Fallback value returned when circuit is open."""
        async def failure():
            raise ValueError("test error")

        # Override failure threshold for this test
        breaker.settings.failure_threshold = 2

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Call with fallback
        result = await breaker.call(failure, fallback="fallback_value")
        assert result == "fallback_value"

    @pytest.mark.asyncio
    async def test_half_open_after_cooldown(self, breaker):
        """Circuit transitions to half-open after cooldown."""
        async def failure():
            raise ValueError("test error")

        # Override failure threshold for this test
        breaker.settings.failure_threshold = 2

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

        # Manually set last_failure_time to simulate cooldown passing
        breaker._state.last_failure_time = time.time() - 10.0

        # Next check should transition to half-open
        assert breaker._should_attempt_reset() is True

    @pytest.mark.asyncio
    async def test_closes_from_half_open_on_success(self, breaker):
        """Circuit closes from half-open on success."""
        async def failure():
            raise ValueError("test error")

        async def success():
            return "ok"

        # Override settings for test
        breaker.settings.failure_threshold = 2
        breaker.settings.success_threshold = 1

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Transition to half-open directly
        breaker._transition_to_half_open()
        assert breaker.state == CircuitState.HALF_OPEN

        # Success should close
        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_reset(self, breaker):
        """Manual reset works."""
        breaker._state.failure_count = 10
        breaker._state.state = CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._state.failure_count == 0

    def test_get_stats(self, breaker):
        """Get statistics."""
        stats = breaker.get_stats()

        assert stats["name"] == "test_service"
        assert stats["state"] == "closed"
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_failures" in stats


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry functions."""

    def test_get_circuit_breaker(self):
        """Get or create circuit breaker."""
        breaker = get_circuit_breaker("test_registry_breaker")
        assert breaker is not None
        assert breaker.name == "test_registry_breaker"

        # Getting again should return same instance
        breaker2 = get_circuit_breaker("test_registry_breaker")
        assert breaker is breaker2

    def test_reset_all(self):
        """Reset all circuit breakers."""
        breaker = get_circuit_breaker("test_reset_all")
        breaker._state.state = CircuitState.OPEN

        reset_all_circuit_breakers()

        assert breaker.state == CircuitState.CLOSED

    def test_get_all_stats(self):
        """Get all circuit breaker stats."""
        get_circuit_breaker("test_stats_1")
        get_circuit_breaker("test_stats_2")

        stats = get_all_circuit_breaker_stats()
        assert "test_stats_1" in stats
        assert "test_stats_2" in stats


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types_exist(self):
        """Verify all metric types."""
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.HISTOGRAM == "histogram"
        assert MetricType.SUMMARY == "summary"


class TestMetricsSettings:
    """Tests for MetricsSettings."""

    def test_default_settings(self):
        """Verify default settings."""
        settings = MetricsSettings()
        assert settings.enabled is True
        assert settings.retention_seconds == 3600.0
        assert len(settings.histogram_buckets) > 0

    def test_custom_settings(self):
        """Verify custom settings work."""
        settings = MetricsSettings(
            enabled=False,
            retention_seconds=7200.0,
        )
        assert settings.enabled is False
        assert settings.retention_seconds == 7200.0


class TestCounter:
    """Tests for Counter metric."""

    def test_create_counter(self):
        """Create a counter."""
        counter = Counter(
            "test_counter",
            "Test counter description",
            labels=["label1"],
        )
        assert counter.name == "test_counter"
        assert counter.description == "Test counter description"

    def test_increment(self):
        """Increment counter."""
        counter = Counter("test_inc", "Test")
        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5.0)
        assert counter.get() == 6.0

    def test_increment_with_labels(self):
        """Increment counter with labels."""
        counter = Counter("test_labels", "Test", labels=["status"])
        counter.inc(labels={"status": "success"})
        counter.inc(labels={"status": "failure"})

        assert counter.get(labels={"status": "success"}) == 1.0
        assert counter.get(labels={"status": "failure"}) == 1.0

    def test_increment_negative_raises(self):
        """Incrementing by negative raises."""
        counter = Counter("test_neg", "Test")
        with pytest.raises(ValueError):
            counter.inc(-1.0)

    def test_collect(self):
        """Collect counter samples."""
        counter = Counter("test_collect", "Test")
        counter.inc(10.0)

        samples = counter.collect()
        assert len(samples) == 1
        assert samples[0].value == 10.0


class TestGauge:
    """Tests for Gauge metric."""

    def test_create_gauge(self):
        """Create a gauge."""
        gauge = Gauge(
            "test_gauge",
            "Test gauge description",
        )
        assert gauge.name == "test_gauge"

    def test_set(self):
        """Set gauge value."""
        gauge = Gauge("test_set", "Test")
        gauge.set(42.0)
        assert gauge.get() == 42.0

    def test_inc_dec(self):
        """Increment and decrement gauge."""
        gauge = Gauge("test_inc_dec", "Test")
        gauge.set(10.0)
        gauge.inc(5.0)
        assert gauge.get() == 15.0

        gauge.dec(3.0)
        assert gauge.get() == 12.0

    def test_gauge_with_labels(self):
        """Gauge with labels."""
        gauge = Gauge("test_labels", "Test", labels=["type"])
        gauge.set(10.0, labels={"type": "a"})
        gauge.set(20.0, labels={"type": "b"})

        assert gauge.get(labels={"type": "a"}) == 10.0
        assert gauge.get(labels={"type": "b"}) == 20.0

    def test_collect(self):
        """Collect gauge samples."""
        gauge = Gauge("test_collect", "Test")
        gauge.set(100.0)

        samples = gauge.collect()
        assert len(samples) == 1
        assert samples[0].value == 100.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_create_histogram(self):
        """Create a histogram."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=[0.1, 0.5, 1.0],
        )
        assert histogram.name == "test_histogram"
        assert histogram.buckets == [0.1, 0.5, 1.0]

    def test_default_buckets(self):
        """Histogram has default buckets."""
        histogram = Histogram("test_default", "Test")
        assert len(histogram.buckets) > 0

    def test_observe(self):
        """Observe values."""
        histogram = Histogram(
            "test_observe",
            "Test",
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)

        assert histogram.get_count() == 3
        assert histogram.get_sum() == pytest.approx(1.15)

    def test_observe_with_labels(self):
        """Observe with labels."""
        histogram = Histogram(
            "test_labels",
            "Test",
            labels=["method"],
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05, labels={"method": "GET"})
        histogram.observe(0.3, labels={"method": "POST"})

        assert histogram.get_count(labels={"method": "GET"}) == 1
        assert histogram.get_count(labels={"method": "POST"}) == 1

    def test_time_context_manager(self):
        """Time context manager."""
        histogram = Histogram("test_time", "Test")

        with histogram.time():
            time.sleep(0.01)

        assert histogram.get_count() == 1
        assert histogram.get_sum() > 0.01

    def test_collect_includes_buckets(self):
        """Collect includes bucket samples."""
        histogram = Histogram(
            "test_collect",
            "Test",
            buckets=[0.1, 0.5],
        )
        histogram.observe(0.3)

        samples = histogram.collect()
        # Should include buckets, sum, and count
        assert len(samples) > 2


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return MetricsRegistry(MetricsSettings())

    def test_create_registry(self, registry):
        """Create a registry."""
        assert registry is not None

    def test_counter(self, registry):
        """Create counter through registry."""
        counter = registry.counter("test_counter", "Test counter")
        assert counter is not None

        # Getting again returns same instance
        counter2 = registry.counter("test_counter", "Test counter")
        assert counter is counter2

    def test_gauge(self, registry):
        """Create gauge through registry."""
        gauge = registry.gauge("test_gauge", "Test gauge")
        assert gauge is not None

    def test_histogram(self, registry):
        """Create histogram through registry."""
        histogram = registry.histogram("test_histogram", "Test histogram")
        assert histogram is not None

    def test_collect_all(self, registry):
        """Collect all metrics."""
        registry.counter("test_c", "Test").inc(10)
        registry.gauge("test_g", "Test").set(20)
        registry.histogram("test_h", "Test").observe(0.5)

        all_metrics = registry.collect_all()
        assert "test_c" in all_metrics
        assert "test_g" in all_metrics
        assert "test_h" in all_metrics

    def test_export_prometheus(self, registry):
        """Export in Prometheus format."""
        registry.counter("test_export", "Test").inc(100)

        output = registry.export_prometheus()
        assert "# HELP test_export" in output
        assert "# TYPE test_export counter" in output
        assert "test_export" in output

    def test_export_json(self, registry):
        """Export as JSON."""
        registry.counter("test_json", "Test").inc(50)

        output = registry.export_json()
        assert "timestamp" in output
        assert "counters" in output
        assert "test_json" in output["counters"]


class TestMetricsConvenienceFunctions:
    """Tests for metrics convenience functions."""

    def test_record_request(self):
        """Record request convenience function."""
        record_request("calibration", "success", 0.5, "generate")
        # Should not raise

    def test_record_tool_call(self):
        """Record tool call convenience function."""
        record_tool_call("analyze_densities", True)
        record_tool_call("predict_response", False)
        # Should not raise

    def test_record_token_usage(self):
        """Record token usage convenience function."""
        record_token_usage("calibration", "claude-3", 100, 50)
        # Should not raise

    def test_get_agent_metrics(self):
        """Get pre-defined agent metrics."""
        metrics = get_agent_metrics()
        assert "requests_total" in metrics
        assert "request_latency" in metrics
        assert "active_agents" in metrics
        assert "tool_calls_total" in metrics


# ============================================================================
# Persistence Tests
# ============================================================================


class TestWorkflowState:
    """Tests for WorkflowState enum."""

    def test_workflow_states_exist(self):
        """Verify all workflow states."""
        assert WorkflowState.PENDING == "pending"
        assert WorkflowState.RUNNING == "running"
        assert WorkflowState.PAUSED == "paused"
        assert WorkflowState.COMPLETED == "completed"
        assert WorkflowState.FAILED == "failed"
        assert WorkflowState.CANCELLED == "cancelled"


class TestPersistenceSettings:
    """Tests for PersistenceSettings."""

    def test_default_settings(self):
        """Verify default settings."""
        settings = PersistenceSettings()
        assert settings.max_checkpoints == 100
        assert settings.auto_checkpoint is True
        assert settings.cleanup_completed is True

    def test_custom_settings(self):
        """Verify custom settings work."""
        settings = PersistenceSettings(
            max_checkpoints=50,
            auto_checkpoint=False,
        )
        assert settings.max_checkpoints == 50
        assert settings.auto_checkpoint is False


class TestTaskCheckpoint:
    """Tests for TaskCheckpoint dataclass."""

    def test_create_task_checkpoint(self):
        """Create a task checkpoint."""
        task = TaskCheckpoint(
            task_id="task-1",
            status="pending",
        )
        assert task.task_id == "task-1"
        assert task.status == "pending"
        assert task.result is None
        assert task.error is None

    def test_task_checkpoint_with_result(self):
        """Task checkpoint with result."""
        task = TaskCheckpoint(
            task_id="task-1",
            status="completed",
            started_at=1000.0,
            completed_at=1010.0,
            result={"output": "value"},
        )
        assert task.result == {"output": "value"}
        assert task.completed_at == 1010.0


class TestWorkflowCheckpoint:
    """Tests for WorkflowCheckpoint dataclass."""

    def test_create_workflow_checkpoint(self):
        """Create a workflow checkpoint."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-123",
            name="test_workflow",
            state=WorkflowState.PENDING,
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert checkpoint.workflow_id == "wf-123"
        assert checkpoint.name == "test_workflow"
        assert checkpoint.state == WorkflowState.PENDING

    def test_checkpoint_with_tasks(self):
        """Checkpoint with tasks."""
        tasks = [
            TaskCheckpoint(task_id="t1", status="completed"),
            TaskCheckpoint(task_id="t2", status="pending"),
        ]
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-123",
            name="test",
            state=WorkflowState.RUNNING,
            created_at=1000.0,
            updated_at=1000.0,
            tasks=tasks,
            current_task_index=1,
        )
        assert len(checkpoint.tasks) == 2
        assert checkpoint.current_task_index == 1

    def test_to_dict(self):
        """Convert to dictionary."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-123",
            name="test",
            state=WorkflowState.RUNNING,
            created_at=1000.0,
            updated_at=1000.0,
        )
        data = checkpoint.to_dict()
        assert data["workflow_id"] == "wf-123"
        assert data["state"] == "running"  # String value

    def test_from_dict(self):
        """Create from dictionary."""
        data = {
            "workflow_id": "wf-456",
            "name": "from_dict_test",
            "state": "completed",
            "created_at": 1000.0,
            "updated_at": 2000.0,
            "current_task_index": 0,
            "tasks": [],
            "context": {},
            "metadata": {},
        }
        checkpoint = WorkflowCheckpoint.from_dict(data)
        assert checkpoint.workflow_id == "wf-456"
        assert checkpoint.state == WorkflowState.COMPLETED


class TestWorkflowPersistence:
    """Tests for WorkflowPersistence class."""

    @pytest.fixture
    def persistence(self):
        """Create persistence with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = PersistenceSettings(
                checkpoint_dir=Path(tmpdir),
                checkpoint_interval_seconds=10.0,  # Use minimum valid value
            )
            persist = WorkflowPersistence(settings)
            # For testing, bypass interval check by clearing timestamps
            persist._last_checkpoint_time = {}
            yield persist

    def test_create_persistence(self, persistence):
        """Create persistence instance."""
        assert persistence is not None
        assert persistence.settings.checkpoint_dir.exists()

    def test_save_and_load_checkpoint(self, persistence):
        """Save and load a checkpoint."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-test-save",
            name="save_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Save
        saved = persistence.save_checkpoint(checkpoint, force=True)
        assert saved is True

        # Load
        loaded = persistence.load_checkpoint("wf-test-save")
        assert loaded is not None
        assert loaded.workflow_id == "wf-test-save"
        assert loaded.name == "save_test"
        assert loaded.state == WorkflowState.RUNNING

    def test_delete_checkpoint(self, persistence):
        """Delete a checkpoint."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-test-delete",
            name="delete_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(checkpoint, force=True)

        # Delete
        deleted = persistence.delete_checkpoint("wf-test-delete")
        assert deleted is True

        # Verify deleted
        loaded = persistence.load_checkpoint("wf-test-delete")
        assert loaded is None

    def test_delete_nonexistent(self, persistence):
        """Delete nonexistent checkpoint."""
        deleted = persistence.delete_checkpoint("nonexistent")
        assert deleted is False

    def test_list_checkpoints(self, persistence):
        """List all checkpoints."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint = WorkflowCheckpoint(
                workflow_id=f"wf-list-{i}",
                name=f"list_test_{i}",
                state=WorkflowState.RUNNING,
                created_at=time.time(),
                updated_at=time.time() + i,
            )
            persistence.save_checkpoint(checkpoint, force=True)

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 3

    def test_list_checkpoints_by_state(self, persistence):
        """List checkpoints filtered by state."""
        # Create checkpoints with different states
        cp1 = WorkflowCheckpoint(
            workflow_id="wf-state-1",
            name="state_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        cp2 = WorkflowCheckpoint(
            workflow_id="wf-state-2",
            name="state_test",
            state=WorkflowState.COMPLETED,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(cp1, force=True)
        persistence.save_checkpoint(cp2, force=True)

        running = persistence.list_checkpoints(state=WorkflowState.RUNNING)
        completed = persistence.list_checkpoints(state=WorkflowState.COMPLETED)

        assert any(cp.workflow_id == "wf-state-1" for cp in running)
        assert any(cp.workflow_id == "wf-state-2" for cp in completed)

    def test_list_incomplete_workflows(self, persistence):
        """List incomplete workflows."""
        # Create one complete and one running
        running = WorkflowCheckpoint(
            workflow_id="wf-incomplete-1",
            name="incomplete",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        completed = WorkflowCheckpoint(
            workflow_id="wf-incomplete-2",
            name="complete",
            state=WorkflowState.COMPLETED,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(running, force=True)
        persistence.save_checkpoint(completed, force=True)

        incomplete = persistence.list_incomplete_workflows()
        ids = [cp.workflow_id for cp in incomplete]

        assert "wf-incomplete-1" in ids
        assert "wf-incomplete-2" not in ids

    def test_mark_completed(self, persistence):
        """Mark workflow as completed."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-complete",
            name="complete_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(checkpoint, force=True)

        # Mark completed
        result = persistence.mark_completed("wf-complete")
        assert result is True

        # Verify
        loaded = persistence.load_checkpoint("wf-complete")
        assert loaded.state == WorkflowState.COMPLETED

    def test_mark_failed(self, persistence):
        """Mark workflow as failed."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-fail",
            name="fail_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
        )
        persistence.save_checkpoint(checkpoint, force=True)

        # Mark failed
        result = persistence.mark_failed("wf-fail", "Test error")
        assert result is True

        # Verify
        loaded = persistence.load_checkpoint("wf-fail")
        assert loaded.state == WorkflowState.FAILED
        assert loaded.error == "Test error"

    def test_advance_workflow(self, persistence):
        """Advance workflow to next task."""
        checkpoint = WorkflowCheckpoint(
            workflow_id="wf-advance",
            name="advance_test",
            state=WorkflowState.RUNNING,
            created_at=time.time(),
            updated_at=time.time(),
            current_task_index=0,
        )
        persistence.save_checkpoint(checkpoint, force=True)

        # Advance
        result = persistence.advance_workflow("wf-advance")
        assert result is True

        # Verify
        loaded = persistence.load_checkpoint("wf-advance")
        assert loaded.current_task_index == 1


class TestCreateWorkflowCheckpoint:
    """Tests for create_workflow_checkpoint function."""

    def test_create_basic(self):
        """Create basic checkpoint."""
        checkpoint = create_workflow_checkpoint(
            workflow_id="wf-create-1",
            name="create_test",
        )
        assert checkpoint.workflow_id == "wf-create-1"
        assert checkpoint.name == "create_test"
        assert checkpoint.state == WorkflowState.PENDING

    def test_create_with_tasks(self):
        """Create checkpoint with tasks."""
        tasks = [
            {"id": "task-1", "metadata": {"type": "analysis"}},
            {"id": "task-2"},
        ]
        checkpoint = create_workflow_checkpoint(
            workflow_id="wf-create-2",
            name="with_tasks",
            tasks=tasks,
        )
        assert len(checkpoint.tasks) == 2
        assert checkpoint.tasks[0].task_id == "task-1"
        assert checkpoint.tasks[0].status == "pending"

    def test_create_with_context(self):
        """Create checkpoint with context."""
        checkpoint = create_workflow_checkpoint(
            workflow_id="wf-create-3",
            name="with_context",
            context={"key": "value"},
            metadata={"created_by": "test"},
        )
        assert checkpoint.context["key"] == "value"
        assert checkpoint.metadata["created_by"] == "test"


# ============================================================================
# Integration Tests
# ============================================================================


class TestObservabilityIntegration:
    """Integration tests for observability components."""

    @pytest.mark.asyncio
    async def test_health_metrics_circuit_breaker_integration(self):
        """Test health checker with metrics and circuit breaker."""
        # Create components
        checker = HealthChecker()
        breaker = CircuitBreaker("integration_test")
        registry = MetricsRegistry()

        # Record some metrics
        counter = registry.counter("integration_test_counter", "Test")
        counter.inc(10)

        # Record a request
        checker.record_request(100.0, True)

        # Get circuit breaker stats
        stats = breaker.get_stats()
        assert stats["state"] == "closed"

        # Health check should work
        with patch.object(checker, "check_llm_connectivity") as mock_llm, \
             patch.object(checker, "check_message_bus") as mock_bus, \
             patch.object(checker, "check_memory_system") as mock_mem, \
             patch.object(checker, "check_tool_registry") as mock_tools, \
             patch.object(checker, "check_subagent_registry") as mock_subagents:

            mock_llm.return_value = HealthCheckResult("llm", True, 10.0)
            mock_bus.return_value = HealthCheckResult("bus", True, 5.0, metadata={"queue_depth": 0})
            mock_mem.return_value = HealthCheckResult("mem", True, 2.0, metadata={"memory_mb": 100.0})
            mock_tools.return_value = HealthCheckResult("tools", True, 1.0, metadata={"tool_count": 5})
            mock_subagents.return_value = HealthCheckResult("subagents", True, 1.0, metadata={"subagent_count": 4})

            report = await checker.check_health()

            assert report.status == HealthStatus.HEALTHY

    def test_persistence_with_task_progress(self):
        """Test persistence tracking task progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = PersistenceSettings(
                checkpoint_dir=Path(tmpdir),
                checkpoint_interval_seconds=10.0,  # Use minimum valid value
            )
            persistence = WorkflowPersistence(settings)
            persistence._last_checkpoint_time = {}  # Clear for testing

            # Create workflow with tasks
            tasks = [
                {"id": "analyze"},
                {"id": "generate"},
                {"id": "export"},
            ]
            checkpoint = create_workflow_checkpoint(
                workflow_id="wf-progress",
                name="progress_test",
                tasks=tasks,
            )

            # Save initial
            persistence.save_checkpoint(checkpoint, force=True)

            # Update task status
            persistence.update_task_status(
                "wf-progress", "analyze", "completed", result={"density": 1.5}
            )

            # Advance
            persistence.advance_workflow("wf-progress")

            # Verify
            loaded = persistence.load_checkpoint("wf-progress")
            assert loaded.current_task_index == 1
            assert loaded.tasks[0].status == "completed"
            assert loaded.tasks[0].result == {"density": 1.5}
