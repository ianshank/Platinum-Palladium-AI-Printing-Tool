"""
Performance metrics collection for agentic operations.

Provides Prometheus-style metrics for monitoring agent performance,
resource usage, and operational health.
"""

import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.agents.logging import get_agent_logger


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


class MetricsSettings(BaseSettings):
    """Settings for metrics collection."""

    model_config = SettingsConfigDict(env_prefix="PTPD_METRICS_")

    # Collection settings
    enabled: bool = Field(default=True, description="Enable metrics collection")
    retention_seconds: float = Field(
        default=3600.0, ge=60.0, le=86400.0, description="Retention period for metric data"
    )

    # Histogram settings
    histogram_buckets: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Default histogram bucket boundaries (seconds)",
    )

    # Export settings
    export_format: str = Field(
        default="prometheus", description="Export format (prometheus, json)"
    )
    include_timestamps: bool = Field(default=True, description="Include timestamps in export")


@dataclass
class MetricSample:
    """A single metric sample."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    description: str
    labels: list[str] = field(default_factory=list)
    buckets: list[float] | None = None  # For histograms


class Counter:
    """
    Monotonically increasing counter metric.

    Example:
        ```python
        requests_total = Counter(
            "agent_requests_total",
            "Total agent requests",
            labels=["agent_type", "status"]
        )
        requests_total.inc(labels={"agent_type": "calibration", "status": "success"})
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ):
        """
        Initialize the counter.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names for this metric.
        """
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._created: dict[tuple, float] = {}

    def _label_key(self, labels: dict[str, str] | None) -> tuple:
        """Convert labels dict to hashable tuple key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """
        Increment the counter.

        Args:
            value: Amount to increment (must be >= 0).
            labels: Label values.
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")

        key = self._label_key(labels)
        if key not in self._created:
            self._created[key] = time.time()
        self._values[key] += value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        return self._values[self._label_key(labels)]

    def collect(self) -> list[MetricSample]:
        """Collect all counter samples."""
        samples = []
        for key, value in self._values.items():
            samples.append(
                MetricSample(
                    value=value,
                    labels=dict(key) if key else {},
                )
            )
        return samples


class Gauge:
    """
    Gauge metric that can increase or decrease.

    Example:
        ```python
        active_workflows = Gauge(
            "agent_active_workflows",
            "Number of active workflows"
        )
        active_workflows.set(5)
        active_workflows.inc()
        active_workflows.dec()
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ):
        """
        Initialize the gauge.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names for this metric.
        """
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)

    def _label_key(self, labels: dict[str, str] | None) -> tuple:
        """Convert labels dict to hashable tuple key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set the gauge value."""
        self._values[self._label_key(labels)] = value

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment the gauge."""
        self._values[self._label_key(labels)] += value

    def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Decrement the gauge."""
        self._values[self._label_key(labels)] -= value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        return self._values[self._label_key(labels)]

    def collect(self) -> list[MetricSample]:
        """Collect all gauge samples."""
        samples = []
        for key, value in self._values.items():
            samples.append(
                MetricSample(
                    value=value,
                    labels=dict(key) if key else {},
                )
            )
        return samples


class Histogram:
    """
    Histogram metric for measuring distributions.

    Example:
        ```python
        request_latency = Histogram(
            "agent_request_latency_seconds",
            "Request latency in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        request_latency.observe(0.35)
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        """
        Initialize the histogram.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names for this metric.
            buckets: Bucket boundaries (defaults to standard values).
        """
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])

        self._sum: dict[tuple, float] = defaultdict(float)
        self._count: dict[tuple, int] = defaultdict(int)
        self._buckets: dict[tuple, dict[float, int]] = {}

    def _label_key(self, labels: dict[str, str] | None) -> tuple:
        """Convert labels dict to hashable tuple key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def _init_buckets(self, key: tuple) -> None:
        """Initialize bucket counters for a label combination."""
        if key not in self._buckets:
            self._buckets[key] = dict.fromkeys(self.buckets, 0)
            self._buckets[key][float("inf")] = 0

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Record an observation.

        Args:
            value: Observed value.
            labels: Label values.
        """
        key = self._label_key(labels)
        self._init_buckets(key)

        self._sum[key] += value
        self._count[key] += 1

        for bucket in self._buckets[key]:
            if value <= bucket:
                self._buckets[key][bucket] += 1

    @contextmanager
    def time(self, labels: dict[str, str] | None = None) -> Generator[None, None, None]:
        """
        Context manager for timing operations.

        Example:
            ```python
            with histogram.time():
                do_something()
            ```
        """
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start, labels)

    def get_sum(self, labels: dict[str, str] | None = None) -> float:
        """Get sum of all observations."""
        return self._sum[self._label_key(labels)]

    def get_count(self, labels: dict[str, str] | None = None) -> int:
        """Get count of observations."""
        return self._count[self._label_key(labels)]

    def collect(self) -> list[MetricSample]:
        """Collect all histogram samples."""
        samples = []
        for key in self._count:
            base_labels = dict(key) if key else {}

            # Add bucket samples
            for bucket, count in self._buckets.get(key, {}).items():
                bucket_labels = {**base_labels, "le": str(bucket)}
                samples.append(
                    MetricSample(
                        value=count,
                        labels=bucket_labels,
                    )
                )

            # Add sum and count
            samples.append(
                MetricSample(
                    value=self._sum[key],
                    labels={**base_labels, "_type": "sum"},
                )
            )
            samples.append(
                MetricSample(
                    value=self._count[key],
                    labels={**base_labels, "_type": "count"},
                )
            )

        return samples


class MetricsRegistry:
    """
    Registry for all metrics.

    Provides centralized management and export of metrics.
    """

    def __init__(self, settings: MetricsSettings | None = None):
        """
        Initialize the registry.

        Args:
            settings: Metrics settings.
        """
        self.settings = settings or MetricsSettings()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._logger = get_agent_logger()

    def counter(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ) -> Counter:
        """
        Get or create a counter metric.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names.

        Returns:
            Counter instance.
        """
        if name not in self._counters:
            self._counters[name] = Counter(name, description, labels)
        return self._counters[name]

    def gauge(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ) -> Gauge:
        """
        Get or create a gauge metric.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names.

        Returns:
            Gauge instance.
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description, labels)
        return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """
        Get or create a histogram metric.

        Args:
            name: Metric name.
            description: Human-readable description.
            labels: Label names.
            buckets: Histogram bucket boundaries.

        Returns:
            Histogram instance.
        """
        if name not in self._histograms:
            self._histograms[name] = Histogram(
                name, description, labels, buckets or self.settings.histogram_buckets
            )
        return self._histograms[name]

    def collect_all(self) -> dict[str, list[MetricSample]]:
        """
        Collect all metrics.

        Returns:
            Dictionary mapping metric names to their samples.
        """
        result = {}

        for name, counter in self._counters.items():
            result[name] = counter.collect()

        for name, gauge in self._gauges.items():
            result[name] = gauge.collect()

        for name, histogram in self._histograms.items():
            result[name] = histogram.collect()

        return result

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []

        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            for sample in counter.collect():
                labels_str = self._format_labels(sample.labels)
                lines.append(f"{name}{labels_str} {sample.value}")

        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            for sample in gauge.collect():
                labels_str = self._format_labels(sample.labels)
                lines.append(f"{name}{labels_str} {sample.value}")

        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            for sample in histogram.collect():
                labels_str = self._format_labels(sample.labels)
                metric_type = sample.labels.pop("_type", None)
                if metric_type == "sum":
                    lines.append(f"{name}_sum{labels_str} {sample.value}")
                elif metric_type == "count":
                    lines.append(f"{name}_count{labels_str} {sample.value}")
                elif "le" in sample.labels:
                    lines.append(f"{name}_bucket{labels_str} {sample.value}")

        return "\n".join(lines)

    def export_json(self) -> dict[str, Any]:
        """
        Export metrics as JSON.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "counters": {
                name: [
                    {"value": s.value, "labels": s.labels}
                    for s in counter.collect()
                ]
                for name, counter in self._counters.items()
            },
            "gauges": {
                name: [
                    {"value": s.value, "labels": s.labels}
                    for s in gauge.collect()
                ]
                for name, gauge in self._gauges.items()
            },
            "histograms": {
                name: {
                    "buckets": histogram.buckets,
                    "samples": [
                        {"value": s.value, "labels": s.labels}
                        for s in histogram.collect()
                    ],
                }
                for name, histogram in self._histograms.items()
            },
        }

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items()) if not k.startswith("_")]
        return "{" + ",".join(pairs) + "}" if pairs else ""


# Global metrics registry
_metrics_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Pre-defined agent metrics
def get_agent_metrics() -> dict[str, Counter | Gauge | Histogram]:
    """
    Get pre-defined agent metrics.

    Returns:
        Dictionary of standard agent metrics.
    """
    registry = get_metrics_registry()

    return {
        "requests_total": registry.counter(
            "agent_requests_total",
            "Total number of agent requests",
            labels=["agent_type", "status"],
        ),
        "request_latency": registry.histogram(
            "agent_request_latency_seconds",
            "Agent request latency in seconds",
            labels=["agent_type", "operation"],
        ),
        "active_agents": registry.gauge(
            "agent_active_count",
            "Number of currently active agents",
            labels=["agent_type"],
        ),
        "tool_calls_total": registry.counter(
            "agent_tool_calls_total",
            "Total number of tool invocations",
            labels=["tool_name", "status"],
        ),
        "token_usage_total": registry.counter(
            "agent_token_usage_total",
            "Total LLM tokens consumed",
            labels=["agent_type", "model", "token_type"],
        ),
        "memory_items": registry.gauge(
            "agent_memory_items",
            "Number of items in agent memory",
            labels=["memory_type"],
        ),
        "message_queue_depth": registry.gauge(
            "agent_message_queue_depth",
            "Current message queue depth",
        ),
        "circuit_breaker_state": registry.gauge(
            "agent_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            labels=["circuit_name"],
        ),
    }


# Convenience functions
def record_request(
    agent_type: str,
    status: str,
    duration_seconds: float,
    operation: str = "default",
) -> None:
    """
    Record an agent request.

    Args:
        agent_type: Type of agent.
        status: Request status (success, failure, etc.).
        duration_seconds: Request duration.
        operation: Operation type.
    """
    metrics = get_agent_metrics()
    requests_counter = metrics["requests_total"]
    latency_histogram = metrics["request_latency"]
    if isinstance(requests_counter, Counter):
        requests_counter.inc(labels={"agent_type": agent_type, "status": status})
    if isinstance(latency_histogram, Histogram):
        latency_histogram.observe(
            duration_seconds, labels={"agent_type": agent_type, "operation": operation}
        )


def record_tool_call(tool_name: str, success: bool) -> None:
    """
    Record a tool invocation.

    Args:
        tool_name: Name of the tool.
        success: Whether the call succeeded.
    """
    metrics = get_agent_metrics()
    status = "success" if success else "failure"
    tool_counter = metrics["tool_calls_total"]
    if isinstance(tool_counter, Counter):
        tool_counter.inc(labels={"tool_name": tool_name, "status": status})


def record_token_usage(
    agent_type: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """
    Record LLM token usage.

    Args:
        agent_type: Type of agent.
        model: LLM model used.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
    """
    metrics = get_agent_metrics()
    token_counter = metrics["token_usage_total"]
    if isinstance(token_counter, Counter):
        token_counter.inc(
            prompt_tokens,
            labels={"agent_type": agent_type, "model": model, "token_type": "prompt"},
        )
        token_counter.inc(
            completion_tokens,
            labels={"agent_type": agent_type, "model": model, "token_type": "completion"},
        )
