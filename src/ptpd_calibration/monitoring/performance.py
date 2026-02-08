"""
Performance monitoring module for PTPD Calibration System.

Provides comprehensive performance tracking, profiling, caching, and resource monitoring.
All thresholds are configurable via settings.
"""

import csv
import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================


class PerformanceMetric(BaseModel):
    """Performance metric data model."""

    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ResourceUsage(BaseModel):
    """System resource usage snapshot."""

    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    memory_used_mb: float = Field(..., description="Memory used in MB")
    memory_available_mb: float = Field(..., description="Memory available in MB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    disk_used_gb: float = Field(..., description="Disk used in GB")
    disk_free_gb: float = Field(..., description="Disk free in GB")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    gpu_percent: float | None = Field(None, description="GPU usage percentage if available")
    gpu_memory_mb: float | None = Field(None, description="GPU memory used in MB")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class APIMetric(BaseModel):
    """API request metric."""

    endpoint: str = Field(..., description="API endpoint")
    duration_ms: float = Field(..., description="Request duration in milliseconds")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    method: str = Field(default="GET", description="HTTP method")
    error: str | None = Field(None, description="Error message if any")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CacheStats(BaseModel):
    """Cache statistics."""

    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    evictions: int = Field(default=0, description="Number of evictions")
    expirations: int = Field(default=0, description="Number of expirations")
    size: int = Field(default=0, description="Current cache size")
    max_size: int = Field(..., description="Maximum cache size")

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


# ============================================================================
# Performance Monitor
# ============================================================================


class PerformanceMonitor:
    """
    Core performance monitoring class.
    Tracks operation timings, records arbitrary metrics, and provides statistics.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self._timers: dict[str, float] = {}
        self._metrics: dict[str, list[PerformanceMetric]] = defaultdict(list)
        self._max_history = max_history
        self._lock = threading.RLock()
        logger.info(f"PerformanceMonitor initialized with max_history={max_history}")

    def start_timer(self, operation_name: str) -> None:
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation to time
        """
        with self._lock:
            self._timers[operation_name] = time.perf_counter()
            logger.debug(f"Started timer for: {operation_name}")

    def stop_timer(self, operation_name: str) -> float | None:
        """
        Stop timing an operation and record the duration.

        Args:
            operation_name: Name of the operation

        Returns:
            Duration in seconds, or None if timer wasn't started
        """
        with self._lock:
            if operation_name not in self._timers:
                logger.warning(f"Timer not started for: {operation_name}")
                return None

            start_time = self._timers.pop(operation_name)
            duration = time.perf_counter() - start_time

            # Record as metric
            self.record_metric(
                name=operation_name,
                value=duration,
                unit="seconds",
                metadata={"type": "timer"},
            )

            logger.debug(f"Stopped timer for {operation_name}: {duration:.4f}s")
            return duration

    @contextmanager
    def timer(self, operation_name: str):
        """
        Context manager for timing operations.

        Args:
            operation_name: Name of the operation

        Example:
            with monitor.timer("image_processing"):
                process_image(img)
        """
        self.start_timer(operation_name)
        try:
            yield
        finally:
            self.stop_timer(operation_name)

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an arbitrary performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metadata: Optional metadata
        """
        with self._lock:
            metric = PerformanceMetric(
                metric_name=name,
                value=value,
                unit=unit,
                metadata=metadata or {},
            )

            self._metrics[name].append(metric)

            # Trim history if needed
            if len(self._metrics[name]) > self._max_history:
                self._metrics[name] = self._metrics[name][-self._max_history :]

            logger.debug(f"Recorded metric: {name}={value} {unit}")

    def get_metrics(
        self,
        operation_name: str,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[PerformanceMetric]:
        """
        Get metrics for an operation, optionally filtered by time range.

        Args:
            operation_name: Name of the operation
            time_range: Optional (start, end) datetime tuple

        Returns:
            List of metrics
        """
        with self._lock:
            metrics = self._metrics.get(operation_name, [])

            if time_range:
                start, end = time_range
                metrics = [m for m in metrics if start <= m.timestamp <= end]

            return metrics

    def get_average(self, operation_name: str) -> float | None:
        """
        Get average value for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Average value, or None if no metrics exist
        """
        metrics = self.get_metrics(operation_name)
        if not metrics:
            return None

        return np.mean([m.value for m in metrics])

    def get_percentiles(
        self, operation_name: str, percentiles: list[int] = None
    ) -> dict[str, float]:
        """
        Get percentile values for an operation.

        Args:
            operation_name: Name of the operation
            percentiles: List of percentiles to calculate

        Returns:
            Dict mapping percentile to value
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        metrics = self.get_metrics(operation_name)
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        result = {}

        for p in percentiles:
            result[f"p{p}"] = np.percentile(values, p)

        return result

    def get_statistics(self, operation_name: str) -> dict[str, Any]:
        """
        Get comprehensive statistics for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dict with min, max, mean, median, std, percentiles, count
        """
        metrics = self.get_metrics(operation_name)
        if not metrics:
            return {"count": 0}

        values = [m.value for m in metrics]
        unit = metrics[0].unit if metrics else "unknown"

        return {
            "count": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "unit": unit,
            **self.get_percentiles(operation_name),
        }

    def export_metrics(
        self,
        format: str = "json",
        output_path: Path | None = None,
        operation_name: str | None = None,
    ) -> str | None:
        """
        Export metrics to JSON or CSV format.

        Args:
            format: Export format ("json" or "csv")
            output_path: Optional path to write file
            operation_name: Optional operation name to filter

        Returns:
            JSON string if no output_path, else None
        """
        with self._lock:
            if operation_name:
                metrics = self.get_metrics(operation_name)
            else:
                # Export all metrics
                metrics = []
                for metric_list in self._metrics.values():
                    metrics.extend(metric_list)

            if format.lower() == "json":
                data = [m.model_dump() for m in metrics]
                json_str = json.dumps(data, indent=2, default=str)

                if output_path:
                    output_path.write_text(json_str)
                    logger.info(f"Exported {len(metrics)} metrics to {output_path}")
                    return None
                return json_str

            elif format.lower() == "csv":
                if not metrics:
                    return "" if not output_path else None

                if output_path:
                    with open(output_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "metric_name",
                                "value",
                                "unit",
                                "timestamp",
                                "metadata",
                            ],
                        )
                        writer.writeheader()
                        for m in metrics:
                            row = m.model_dump()
                            row["metadata"] = json.dumps(row["metadata"])
                            writer.writerow(row)
                    logger.info(f"Exported {len(metrics)} metrics to {output_path}")
                    return None
                else:
                    # Return CSV string
                    import io

                    output = io.StringIO()
                    writer = csv.DictWriter(
                        output,
                        fieldnames=[
                            "metric_name",
                            "value",
                            "unit",
                            "timestamp",
                            "metadata",
                        ],
                    )
                    writer.writeheader()
                    for m in metrics:
                        row = m.model_dump()
                        row["metadata"] = json.dumps(row["metadata"])
                        writer.writerow(row)
                    return output.getvalue()

            else:
                raise ValueError(f"Unsupported format: {format}")

    def clear_metrics(self, operation_name: str | None = None) -> None:
        """
        Clear metrics for an operation or all metrics.

        Args:
            operation_name: Optional operation name to clear
        """
        with self._lock:
            if operation_name:
                self._metrics.pop(operation_name, None)
                logger.info(f"Cleared metrics for: {operation_name}")
            else:
                self._metrics.clear()
                logger.info("Cleared all metrics")


# ============================================================================
# Image Processing Profiler
# ============================================================================


class ImageProcessingProfiler:
    """
    Profiler specifically for image processing operations.
    Tracks processing speed, memory usage, and identifies bottlenecks.
    """

    def __init__(self, monitor: PerformanceMonitor | None = None):
        """
        Initialize image processing profiler.

        Args:
            monitor: Optional PerformanceMonitor instance
        """
        self.monitor = monitor or PerformanceMonitor()
        self._memory_tracking: dict[str, float] = {}
        self._lock = threading.RLock()
        logger.info("ImageProcessingProfiler initialized")

    def profile_operation(self, func: Callable, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
        """
        Profile any operation and return result with profiling data.

        Args:
            func: Function to profile
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, profile_data)
        """
        operation_name = func.__name__
        profile_data = {"operation": operation_name}

        # Track memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time the operation
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            result = func(*args, **kwargs)
            profile_data["success"] = True
        except Exception as e:
            logger.error(f"Error profiling {operation_name}: {e}")
            profile_data["success"] = False
            profile_data["error"] = str(e)
            raise
        finally:
            # Calculate metrics
            wall_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_cpu
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before

            profile_data.update(
                {
                    "wall_time_s": wall_time,
                    "cpu_time_s": cpu_time,
                    "memory_delta_mb": mem_delta,
                    "memory_after_mb": mem_after,
                }
            )

            # Record metrics
            self.monitor.record_metric(
                f"{operation_name}_wall_time",
                wall_time,
                "seconds",
                metadata=profile_data,
            )
            self.monitor.record_metric(
                f"{operation_name}_memory",
                mem_delta,
                "MB",
                metadata=profile_data,
            )

            logger.info(f"Profiled {operation_name}: {wall_time:.3f}s, {mem_delta:.1f}MB")

        return result, profile_data

    def get_processing_speed(
        self, image_size: tuple[int, int], operation_name: str
    ) -> float | None:
        """
        Get processing speed in pixels per second.

        Args:
            image_size: (width, height) tuple
            operation_name: Name of the operation

        Returns:
            Pixels per second, or None if no data
        """
        metrics = self.monitor.get_metrics(f"{operation_name}_wall_time")
        if not metrics:
            return None

        total_pixels = image_size[0] * image_size[1]
        avg_time = np.mean([m.value for m in metrics])

        if avg_time == 0:
            return None

        return total_pixels / avg_time

    def estimate_batch_time(
        self, images: list[tuple[int, int]], operation_name: str
    ) -> float | None:
        """
        Estimate total time to process a batch of images.

        Args:
            images: List of (width, height) tuples
            operation_name: Name of the operation

        Returns:
            Estimated time in seconds, or None if no data
        """
        metrics = self.monitor.get_metrics(f"{operation_name}_wall_time")
        if not metrics:
            return None

        # Get average processing speed
        avg_time = np.mean([m.value for m in metrics])

        # Estimate based on number of images
        estimated_time = avg_time * len(images)

        # Add 10% overhead for batch operations
        return estimated_time * 1.1

    def track_memory_usage(self, operation: str) -> None:
        """
        Track current memory usage for an operation.

        Args:
            operation: Operation name
        """
        with self._lock:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            self._memory_tracking[operation] = mem_mb
            self.monitor.record_metric(f"{operation}_memory_snapshot", mem_mb, "MB")

    def get_memory_stats(self) -> dict[str, float]:
        """
        Get current memory statistics.

        Returns:
            Dict with memory info
        """
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def identify_bottlenecks(
        self, operations: list[str], threshold_percentile: float = 90
    ) -> list[dict[str, Any]]:
        """
        Identify slow operations that may be bottlenecks.

        Args:
            operations: List of operation names to check
            threshold_percentile: Percentile threshold for slow operations

        Returns:
            List of bottleneck information dicts
        """
        bottlenecks = []

        for op in operations:
            stats = self.monitor.get_statistics(f"{op}_wall_time")
            if not stats or stats["count"] == 0:
                continue

            # Check if p90 is significantly higher than median
            p90 = stats.get("p90", 0)
            median = stats.get("median", 0)

            if median > 0 and p90 / median > 2.0:
                bottlenecks.append(
                    {
                        "operation": op,
                        "median_time": median,
                        "p90_time": p90,
                        "max_time": stats["max"],
                        "slowdown_factor": p90 / median,
                        "count": stats["count"],
                    }
                )

        # Sort by slowdown factor
        bottlenecks.sort(key=lambda x: x["slowdown_factor"], reverse=True)

        return bottlenecks


# ============================================================================
# API Performance Tracker
# ============================================================================


class APIPerformanceTracker:
    """
    Track API endpoint performance, response times, and error rates.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize API performance tracker.

        Args:
            max_history: Maximum number of requests to keep
        """
        self._requests: list[APIMetric] = deque(maxlen=max_history)
        self._lock = threading.RLock()
        logger.info(f"APIPerformanceTracker initialized with max_history={max_history}")

    def track_request(
        self,
        endpoint: str,
        duration: float,
        status: int,
        method: str = "GET",
        error: str | None = None,
    ) -> None:
        """
        Track an API request.

        Args:
            endpoint: API endpoint path
            duration: Request duration in seconds
            status: HTTP status code
            method: HTTP method
            error: Optional error message
        """
        with self._lock:
            metric = APIMetric(
                endpoint=endpoint,
                duration_ms=duration * 1000,  # Convert to ms
                status_code=status,
                method=method,
                error=error,
            )
            self._requests.append(metric)
            logger.debug(f"Tracked {method} {endpoint}: {status} in {duration * 1000:.1f}ms")

    @contextmanager
    def track(self, endpoint: str, method: str = "GET"):
        """
        Context manager for tracking requests.

        Args:
            endpoint: API endpoint
            method: HTTP method

        Example:
            with tracker.track("/api/process", "POST"):
                response = process_request()
        """
        start_time = time.perf_counter()
        status = 200
        error = None

        try:
            yield
        except Exception as e:
            status = 500
            error = str(e)
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.track_request(endpoint, duration, status, method, error)

    def get_endpoint_stats(
        self, endpoint: str, time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """
        Get statistics for a specific endpoint.

        Args:
            endpoint: API endpoint path
            time_range: Optional (start, end) datetime tuple

        Returns:
            Statistics dict
        """
        with self._lock:
            requests = [r for r in self._requests if r.endpoint == endpoint]

            if time_range:
                start, end = time_range
                requests = [r for r in requests if start <= r.timestamp <= end]

            if not requests:
                return {"count": 0}

            durations = [r.duration_ms for r in requests]
            errors = sum(1 for r in requests if r.status_code >= 400)

            return {
                "count": len(requests),
                "error_count": errors,
                "error_rate": errors / len(requests),
                "min_ms": float(np.min(durations)),
                "max_ms": float(np.max(durations)),
                "mean_ms": float(np.mean(durations)),
                "median_ms": float(np.median(durations)),
                "p90_ms": float(np.percentile(durations, 90)),
                "p95_ms": float(np.percentile(durations, 95)),
                "p99_ms": float(np.percentile(durations, 99)),
            }

    def get_error_rate(
        self, endpoint: str, time_range: tuple[datetime, datetime] | None = None
    ) -> float:
        """
        Calculate error rate for an endpoint.

        Args:
            endpoint: API endpoint path
            time_range: Optional (start, end) datetime tuple

        Returns:
            Error rate (0.0 to 1.0)
        """
        stats = self.get_endpoint_stats(endpoint, time_range)
        return stats.get("error_rate", 0.0)

    def get_response_times(self) -> dict[str, list[float]]:
        """
        Get response time distribution by endpoint.

        Returns:
            Dict mapping endpoint to list of response times
        """
        with self._lock:
            response_times = defaultdict(list)
            for req in self._requests:
                response_times[req.endpoint].append(req.duration_ms)
            return dict(response_times)

    def generate_api_report(self) -> dict[str, Any]:
        """
        Generate comprehensive API performance report.

        Returns:
            Report dict
        """
        with self._lock:
            if not self._requests:
                return {"total_requests": 0}

            endpoints = {r.endpoint for r in self._requests}
            endpoint_stats = {ep: self.get_endpoint_stats(ep) for ep in endpoints}

            # Overall stats
            all_durations = [r.duration_ms for r in self._requests]
            all_errors = sum(1 for r in self._requests if r.status_code >= 400)

            return {
                "total_requests": len(self._requests),
                "total_errors": all_errors,
                "overall_error_rate": all_errors / len(self._requests),
                "overall_mean_ms": float(np.mean(all_durations)),
                "overall_median_ms": float(np.median(all_durations)),
                "overall_p95_ms": float(np.percentile(all_durations, 95)),
                "endpoints": endpoint_stats,
                "slowest_endpoints": self._get_slowest_endpoints(endpoint_stats),
                "highest_error_endpoints": self._get_highest_error_endpoints(endpoint_stats),
            }

    def _get_slowest_endpoints(
        self, endpoint_stats: dict[str, dict[str, Any]], top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Get slowest endpoints by p95 response time."""
        endpoints = [
            {"endpoint": ep, "p95_ms": stats.get("p95_ms", 0)}
            for ep, stats in endpoint_stats.items()
            if stats.get("count", 0) > 0
        ]
        endpoints.sort(key=lambda x: x["p95_ms"], reverse=True)
        return endpoints[:top_n]

    def _get_highest_error_endpoints(
        self, endpoint_stats: dict[str, dict[str, Any]], top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Get endpoints with highest error rates."""
        endpoints = [
            {"endpoint": ep, "error_rate": stats.get("error_rate", 0)}
            for ep, stats in endpoint_stats.items()
            if stats.get("count", 0) > 0
        ]
        endpoints.sort(key=lambda x: x["error_rate"], reverse=True)
        return endpoints[:top_n]


# ============================================================================
# Cache Manager
# ============================================================================


class CacheManager:
    """
    Simple dict-based LRU cache with TTL support.
    Thread-safe with configurable size and expiration.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._expiry: dict[str, datetime] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }
        self._lock = threading.RLock()
        logger.info(f"CacheManager initialized: max_size={max_size}, default_ttl={default_ttl}s")

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            # Check expiry
            if key in self._expiry and datetime.now() > self._expiry[key]:
                self._cache.pop(key)
                self._expiry.pop(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
        """
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._cache.pop(key)
                self._expiry.pop(key, None)

            # Add new entry
            self._cache[key] = value

            # Set expiry
            ttl_seconds = ttl if ttl is not None else self._default_ttl
            if ttl_seconds > 0:
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl_seconds)

            # Evict oldest if over limit
            if len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
                self._expiry.pop(oldest_key, None)
                self._stats["evictions"] += 1

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self._expiry.pop(key, None)
                return True
            return False

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats model
        """
        with self._lock:
            return CacheStats(
                hits=self._stats["hits"],
                misses=self._stats["misses"],
                evictions=self._stats["evictions"],
                expirations=self._stats["expirations"],
                size=len(self._cache),
                max_size=self._max_size,
            )

    def get_size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of items in cache
        """
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.now()
            expired_keys = [key for key, expiry in self._expiry.items() if now > expiry]

            for key in expired_keys:
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
                self._stats["expirations"] += 1

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)


# ============================================================================
# Resource Monitor
# ============================================================================


class ResourceMonitor:
    """
    Monitor system resources (CPU, memory, disk, GPU).
    Provides alerts for high resource usage.
    """

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
    ):
        """
        Initialize resource monitor.

        Args:
            cpu_threshold: CPU usage alert threshold (%)
            memory_threshold: Memory usage alert threshold (%)
            disk_threshold: Disk usage alert threshold (%)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self._lock = threading.RLock()
        logger.info("ResourceMonitor initialized")

    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.

        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=0.1)

    def get_memory_usage(self) -> dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dict with memory info
        """
        mem = psutil.virtual_memory()
        return {
            "percent": mem.percent,
            "used_mb": mem.used / 1024 / 1024,
            "available_mb": mem.available / 1024 / 1024,
            "total_mb": mem.total / 1024 / 1024,
        }

    def get_disk_usage(self, path: str = "/") -> dict[str, float]:
        """
        Get disk usage for a path.

        Args:
            path: Path to check

        Returns:
            Dict with disk info
        """
        disk = psutil.disk_usage(path)
        return {
            "percent": disk.percent,
            "used_gb": disk.used / 1024 / 1024 / 1024,
            "free_gb": disk.free / 1024 / 1024 / 1024,
            "total_gb": disk.total / 1024 / 1024 / 1024,
        }

    def get_gpu_usage(self) -> dict[str, Any] | None:
        """
        Get GPU usage if available.

        Returns:
            Dict with GPU info or None if not available
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {
                "gpu_percent": utilization.gpu,
                "memory_percent": (memory_info.used / memory_info.total) * 100,
                "memory_used_mb": memory_info.used / 1024 / 1024,
                "memory_total_mb": memory_info.total / 1024 / 1024,
            }
        except (ImportError, Exception) as e:
            logger.debug(f"GPU monitoring not available: {e}")
            return None

    def check_resources(self) -> ResourceUsage:
        """
        Perform comprehensive resource check.

        Returns:
            ResourceUsage model
        """
        cpu = self.get_cpu_usage()
        mem = self.get_memory_usage()
        disk = self.get_disk_usage()
        gpu = self.get_gpu_usage()

        return ResourceUsage(
            cpu_percent=cpu,
            memory_percent=mem["percent"],
            memory_used_mb=mem["used_mb"],
            memory_available_mb=mem["available_mb"],
            disk_percent=disk["percent"],
            disk_used_gb=disk["used_gb"],
            disk_free_gb=disk["free_gb"],
            gpu_percent=gpu["gpu_percent"] if gpu else None,
            gpu_memory_mb=gpu["memory_used_mb"] if gpu else None,
        )

    def get_alerts_for_high_usage(self) -> list[dict[str, Any]]:
        """
        Get alerts for resource usage exceeding thresholds.

        Returns:
            List of alert dicts
        """
        alerts = []
        resources = self.check_resources()

        if resources.cpu_percent > self.cpu_threshold:
            alerts.append(
                {
                    "resource": "cpu",
                    "usage": resources.cpu_percent,
                    "threshold": self.cpu_threshold,
                    "severity": "warning" if resources.cpu_percent < 95 else "critical",
                    "message": f"CPU usage at {resources.cpu_percent:.1f}%",
                }
            )

        if resources.memory_percent > self.memory_threshold:
            alerts.append(
                {
                    "resource": "memory",
                    "usage": resources.memory_percent,
                    "threshold": self.memory_threshold,
                    "severity": "warning" if resources.memory_percent < 95 else "critical",
                    "message": f"Memory usage at {resources.memory_percent:.1f}%",
                }
            )

        if resources.disk_percent > self.disk_threshold:
            alerts.append(
                {
                    "resource": "disk",
                    "usage": resources.disk_percent,
                    "threshold": self.disk_threshold,
                    "severity": "warning" if resources.disk_percent < 95 else "critical",
                    "message": f"Disk usage at {resources.disk_percent:.1f}%",
                }
            )

        if alerts:
            logger.warning(f"Resource alerts: {len(alerts)} threshold(s) exceeded")

        return alerts


# ============================================================================
# Performance Report
# ============================================================================


class PerformanceReport:
    """
    Generate performance reports and summaries.
    """

    def __init__(
        self,
        monitor: PerformanceMonitor,
        api_tracker: APIPerformanceTracker | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ):
        """
        Initialize performance report generator.

        Args:
            monitor: PerformanceMonitor instance
            api_tracker: Optional APIPerformanceTracker instance
            resource_monitor: Optional ResourceMonitor instance
        """
        self.monitor = monitor
        self.api_tracker = api_tracker
        self.resource_monitor = resource_monitor
        logger.info("PerformanceReport initialized")

    def generate_daily_report(self, date: datetime | None = None) -> dict[str, Any]:
        """
        Generate daily performance summary.

        Args:
            date: Date to generate report for (default: today)

        Returns:
            Report dict
        """
        if date is None:
            date = datetime.now()

        start = datetime.combine(date.date(), datetime.min.time())
        end = datetime.combine(date.date(), datetime.max.time())

        report = {
            "date": date.date().isoformat(),
            "generated_at": datetime.now().isoformat(),
            "operations": {},
        }

        # Get all operation names
        with self.monitor._lock:
            operation_names = list(self.monitor._metrics.keys())

        # Generate stats for each operation
        for op_name in operation_names:
            metrics = self.monitor.get_metrics(op_name, (start, end))
            if metrics:
                stats = self.monitor.get_statistics(op_name)
                report["operations"][op_name] = stats

        # Add API stats if available
        if self.api_tracker:
            report["api"] = self.api_tracker.generate_api_report()

        # Add resource info if available
        if self.resource_monitor:
            report["resources"] = self.resource_monitor.check_resources().model_dump()
            report["resource_alerts"] = self.resource_monitor.get_alerts_for_high_usage()

        return report

    def generate_session_report(self, session_id: str) -> dict[str, Any]:
        """
        Generate report for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Report dict
        """
        report = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "operations": {},
        }

        # Filter metrics by session_id in metadata
        with self.monitor._lock:
            for op_name, metrics in self.monitor._metrics.items():
                session_metrics = [m for m in metrics if m.metadata.get("session_id") == session_id]
                if session_metrics:
                    values = [m.value for m in session_metrics]
                    report["operations"][op_name] = {
                        "count": len(values),
                        "total": sum(values),
                        "mean": np.mean(values),
                        "unit": session_metrics[0].unit,
                    }

        return report

    def compare_performance(
        self, period1: tuple[datetime, datetime], period2: tuple[datetime, datetime]
    ) -> dict[str, Any]:
        """
        Compare performance between two time periods.

        Args:
            period1: First (start, end) datetime tuple
            period2: Second (start, end) datetime tuple

        Returns:
            Comparison dict
        """
        comparison = {"period1": {}, "period2": {}, "changes": {}}

        with self.monitor._lock:
            operation_names = list(self.monitor._metrics.keys())

        for op_name in operation_names:
            metrics1 = self.monitor.get_metrics(op_name, period1)
            metrics2 = self.monitor.get_metrics(op_name, period2)

            if not metrics1 or not metrics2:
                continue

            values1 = [m.value for m in metrics1]
            values2 = [m.value for m in metrics2]

            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            change_pct = ((mean2 - mean1) / mean1 * 100) if mean1 > 0 else 0

            comparison["period1"][op_name] = {"mean": mean1, "count": len(values1)}
            comparison["period2"][op_name] = {"mean": mean2, "count": len(values2)}
            comparison["changes"][op_name] = {
                "absolute": mean2 - mean1,
                "percentage": change_pct,
                "direction": "improved" if change_pct < 0 else "degraded",
            }

        return comparison

    def export_report(
        self, report: dict[str, Any], format: str = "json", path: Path | None = None
    ) -> str | None:
        """
        Export report to file.

        Args:
            report: Report dict
            format: Export format ("json" or "csv")
            path: Optional path to write file

        Returns:
            Report string if no path, else None
        """
        if format.lower() == "json":
            json_str = json.dumps(report, indent=2, default=str)
            if path:
                path.write_text(json_str)
                logger.info(f"Exported report to {path}")
                return None
            return json_str

        elif format.lower() == "csv":
            # Flatten report for CSV
            rows = []
            for op_name, stats in report.get("operations", {}).items():
                row = {"operation": op_name}
                row.update(stats)
                rows.append(row)

            if path:
                if rows:
                    with open(path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                logger.info(f"Exported report to {path}")
                return None
            else:
                import io

                output = io.StringIO()
                if rows:
                    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                return output.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# Module-level instances
# ============================================================================

# Global instances for easy access
_global_monitor: PerformanceMonitor | None = None
_global_profiler: ImageProcessingProfiler | None = None
_global_api_tracker: APIPerformanceTracker | None = None
_global_cache: CacheManager | None = None
_global_resource_monitor: ResourceMonitor | None = None


def get_monitor() -> PerformanceMonitor:
    """Get or create global PerformanceMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def get_profiler() -> ImageProcessingProfiler:
    """Get or create global ImageProcessingProfiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ImageProcessingProfiler(monitor=get_monitor())
    return _global_profiler


def get_api_tracker() -> APIPerformanceTracker:
    """Get or create global APIPerformanceTracker instance."""
    global _global_api_tracker
    if _global_api_tracker is None:
        _global_api_tracker = APIPerformanceTracker()
    return _global_api_tracker


def get_cache() -> CacheManager:
    """Get or create global CacheManager instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def get_resource_monitor() -> ResourceMonitor:
    """Get or create global ResourceMonitor instance."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
    return _global_resource_monitor
