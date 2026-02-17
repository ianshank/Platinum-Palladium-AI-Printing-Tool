"""
Comprehensive unit tests for performance monitoring and data repository modules.

Tests cover:
Performance Module (src/ptpd_calibration/monitoring/performance.py):
- PerformanceMonitor: timing, metrics, statistics, export
- ImageProcessingProfiler: profiling, speed calculation, bottlenecks
- APIPerformanceTracker: request tracking, stats, reports
- CacheManager: get/set, TTL, LRU eviction, stats
- ResourceMonitor: CPU/memory/disk usage, alerts
- PerformanceReport: daily/session reports, comparisons
- Global instance getters
- Thread safety and concurrent operations

Repository Module (src/ptpd_calibration/data/repository.py):
- Repository ABC interface
- SQLiteRepository: CRUD operations, search, indexed fields
- InMemoryRepository: CRUD operations, testing
- Edge cases and error handling
"""

import concurrent.futures
import json
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import BaseModel, Field

from ptpd_calibration.data.repository import (
    InMemoryRepository,
    Repository,
    SQLiteRepository,
)
from ptpd_calibration.monitoring.performance import (
    APIMetric,
    APIPerformanceTracker,
    CacheManager,
    CacheStats,
    ImageProcessingProfiler,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
    ResourceMonitor,
    ResourceUsage,
    get_api_tracker,
    get_cache,
    get_monitor,
    get_profiler,
    get_resource_monitor,
)


# ============================================================================
# Test Models for Repository Tests
# ============================================================================


class TestModel(BaseModel):
    """Test model for repository tests."""

    id: str | None = None
    name: str
    value: int
    category: str = "default"


class PersonModel(BaseModel):
    """Another test model for repository tests."""

    id: str | None = None
    first_name: str
    last_name: str
    age: int
    email: str


# ============================================================================
# PerformanceMonitor Tests
# ============================================================================


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_init(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(max_history=5000)
        assert monitor._max_history == 5000
        assert len(monitor._timers) == 0
        assert len(monitor._metrics) == 0

    def test_init_default_history(self):
        """Test PerformanceMonitor with default max_history."""
        monitor = PerformanceMonitor()
        assert monitor._max_history == 10000

    def test_start_stop_timer(self):
        """Test basic timer operations."""
        monitor = PerformanceMonitor()

        # Start timer
        monitor.start_timer("test_operation")
        assert "test_operation" in monitor._timers

        # Sleep briefly to get measurable time
        time.sleep(0.01)

        # Stop timer
        duration = monitor.stop_timer("test_operation")
        assert duration is not None
        assert duration >= 0.01
        assert "test_operation" not in monitor._timers

        # Metric should be recorded
        metrics = monitor.get_metrics("test_operation")
        assert len(metrics) == 1
        assert metrics[0].value >= 0.01
        assert metrics[0].unit == "seconds"

    def test_stop_timer_not_started(self):
        """Test stopping a timer that wasn't started."""
        monitor = PerformanceMonitor()
        duration = monitor.stop_timer("nonexistent")
        assert duration is None

    def test_timer_context_manager(self):
        """Test timer context manager."""
        monitor = PerformanceMonitor()

        with monitor.timer("context_test"):
            time.sleep(0.01)

        metrics = monitor.get_metrics("context_test")
        assert len(metrics) == 1
        assert metrics[0].value >= 0.01

    def test_timer_context_manager_with_exception(self):
        """Test timer context manager handles exceptions."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            with monitor.timer("error_test"):
                raise ValueError("Test error")

        # Metric should still be recorded
        metrics = monitor.get_metrics("error_test")
        assert len(metrics) == 1

    def test_record_metric(self):
        """Test recording custom metrics."""
        monitor = PerformanceMonitor()

        monitor.record_metric("cpu_usage", 75.5, "percent", {"host": "test"})

        metrics = monitor.get_metrics("cpu_usage")
        assert len(metrics) == 1
        assert metrics[0].value == 75.5
        assert metrics[0].unit == "percent"
        assert metrics[0].metadata["host"] == "test"

    def test_record_metric_without_metadata(self):
        """Test recording metric without metadata."""
        monitor = PerformanceMonitor()
        monitor.record_metric("memory", 512, "MB")

        metrics = monitor.get_metrics("memory")
        assert len(metrics) == 1
        assert metrics[0].metadata == {}

    def test_max_history_trimming(self):
        """Test that history is trimmed when max_history is exceeded."""
        monitor = PerformanceMonitor(max_history=10)

        # Record 20 metrics
        for i in range(20):
            monitor.record_metric("test", i, "count")

        metrics = monitor.get_metrics("test")
        assert len(metrics) == 10
        # Should keep the last 10
        assert metrics[0].value == 10
        assert metrics[-1].value == 19

    def test_get_metrics_with_time_range(self):
        """Test filtering metrics by time range."""
        monitor = PerformanceMonitor()

        # Record metrics at different times
        start_time = datetime.now()
        monitor.record_metric("test", 1, "count")
        time.sleep(0.01)
        middle_time = datetime.now()
        time.sleep(0.01)
        monitor.record_metric("test", 2, "count")
        time.sleep(0.01)
        end_time = datetime.now()

        # Get metrics in middle range
        metrics = monitor.get_metrics("test", (start_time, middle_time))
        assert len(metrics) == 1
        assert metrics[0].value == 1

        # Get all metrics
        metrics = monitor.get_metrics("test", (start_time, end_time))
        assert len(metrics) == 2

    def test_get_average(self):
        """Test calculating average value."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 10, "ms")
        monitor.record_metric("test", 20, "ms")
        monitor.record_metric("test", 30, "ms")

        avg = monitor.get_average("test")
        assert avg == 20.0

    def test_get_average_no_metrics(self):
        """Test get_average with no metrics."""
        monitor = PerformanceMonitor()
        avg = monitor.get_average("nonexistent")
        assert avg is None

    def test_get_percentiles(self):
        """Test calculating percentiles."""
        monitor = PerformanceMonitor()

        # Record 100 metrics with values 0-99
        for i in range(100):
            monitor.record_metric("test", i, "count")

        percentiles = monitor.get_percentiles("test")
        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] == pytest.approx(49.5, rel=0.1)
        assert percentiles["p90"] == pytest.approx(89.1, rel=0.1)

    def test_get_percentiles_custom(self):
        """Test calculating custom percentiles."""
        monitor = PerformanceMonitor()

        for i in range(100):
            monitor.record_metric("test", i, "count")

        percentiles = monitor.get_percentiles("test", [25, 75])
        assert "p25" in percentiles
        assert "p75" in percentiles
        assert percentiles["p25"] == pytest.approx(24.75, rel=0.1)
        assert percentiles["p75"] == pytest.approx(74.25, rel=0.1)

    def test_get_percentiles_no_metrics(self):
        """Test get_percentiles with no metrics."""
        monitor = PerformanceMonitor()
        percentiles = monitor.get_percentiles("nonexistent")
        assert percentiles == {}

    def test_get_statistics(self):
        """Test comprehensive statistics."""
        monitor = PerformanceMonitor()

        values = [10, 20, 30, 40, 50]
        for v in values:
            monitor.record_metric("test", v, "ms")

        stats = monitor.get_statistics("test")
        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["unit"] == "ms"
        assert "p50" in stats
        assert "p90" in stats

    def test_get_statistics_no_metrics(self):
        """Test get_statistics with no metrics."""
        monitor = PerformanceMonitor()
        stats = monitor.get_statistics("nonexistent")
        assert stats == {"count": 0}

    def test_export_metrics_json(self):
        """Test exporting metrics to JSON."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 123, "ms")
        monitor.record_metric("test", 456, "ms")

        json_str = monitor.export_metrics(format="json")
        assert json_str is not None

        data = json.loads(json_str)
        assert len(data) == 2
        assert data[0]["metric_name"] == "test"
        assert data[0]["value"] == 123

    def test_export_metrics_json_to_file(self, tmp_path):
        """Test exporting metrics to JSON file."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 123, "ms")

        output_path = tmp_path / "metrics.json"
        result = monitor.export_metrics(format="json", output_path=output_path)
        assert result is None
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert len(data) == 1
        assert data[0]["value"] == 123

    def test_export_metrics_csv(self):
        """Test exporting metrics to CSV."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 123, "ms", {"host": "server1"})

        csv_str = monitor.export_metrics(format="csv")
        assert csv_str is not None
        assert "metric_name" in csv_str
        assert "test" in csv_str
        assert "123" in csv_str

    def test_export_metrics_csv_to_file(self, tmp_path):
        """Test exporting metrics to CSV file."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 123, "ms")

        output_path = tmp_path / "metrics.csv"
        result = monitor.export_metrics(format="csv", output_path=output_path)
        assert result is None
        assert output_path.exists()

        content = output_path.read_text()
        assert "metric_name" in content
        assert "123" in content

    def test_export_metrics_csv_empty(self):
        """Test exporting empty metrics to CSV."""
        monitor = PerformanceMonitor()
        csv_str = monitor.export_metrics(format="csv")
        assert csv_str == ""

    def test_export_metrics_filter_operation(self):
        """Test exporting metrics for specific operation."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 100, "ms")
        monitor.record_metric("op2", 200, "ms")

        json_str = monitor.export_metrics(format="json", operation_name="op1")
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["metric_name"] == "op1"

    def test_export_metrics_invalid_format(self):
        """Test exporting with invalid format."""
        monitor = PerformanceMonitor()
        monitor.record_metric("test", 123, "ms")

        with pytest.raises(ValueError, match="Unsupported format"):
            monitor.export_metrics(format="xml")

    def test_clear_metrics_single(self):
        """Test clearing metrics for single operation."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 100, "ms")
        monitor.record_metric("op2", 200, "ms")

        monitor.clear_metrics("op1")

        assert len(monitor.get_metrics("op1")) == 0
        assert len(monitor.get_metrics("op2")) == 1

    def test_clear_metrics_all(self):
        """Test clearing all metrics."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 100, "ms")
        monitor.record_metric("op2", 200, "ms")

        monitor.clear_metrics()

        assert len(monitor.get_metrics("op1")) == 0
        assert len(monitor.get_metrics("op2")) == 0

    def test_thread_safety(self):
        """Test thread safety of PerformanceMonitor."""
        monitor = PerformanceMonitor()

        def record_metrics(thread_id):
            for i in range(100):
                monitor.record_metric(f"thread_{thread_id}", i, "count")

        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have 100 metrics
        for i in range(5):
            metrics = monitor.get_metrics(f"thread_{i}")
            assert len(metrics) == 100


# ============================================================================
# ImageProcessingProfiler Tests
# ============================================================================


class TestImageProcessingProfiler:
    """Tests for ImageProcessingProfiler class."""

    def test_init(self):
        """Test ImageProcessingProfiler initialization."""
        profiler = ImageProcessingProfiler()
        assert profiler.monitor is not None
        assert isinstance(profiler.monitor, PerformanceMonitor)

    def test_init_with_monitor(self):
        """Test ImageProcessingProfiler with custom monitor."""
        monitor = PerformanceMonitor()
        profiler = ImageProcessingProfiler(monitor=monitor)
        assert profiler.monitor is monitor

    def test_profile_operation(self):
        """Test profiling an operation."""
        profiler = ImageProcessingProfiler()

        def test_func(x, y):
            time.sleep(0.01)
            return x + y

        result, profile_data = profiler.profile_operation(test_func, 5, 10)

        assert result == 15
        assert profile_data["operation"] == "test_func"
        assert profile_data["success"] is True
        assert profile_data["wall_time_s"] >= 0.01
        assert "cpu_time_s" in profile_data
        assert "memory_delta_mb" in profile_data

    def test_profile_operation_with_exception(self):
        """Test profiling operation that raises exception."""
        profiler = ImageProcessingProfiler()

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            profiler.profile_operation(failing_func)

    def test_get_processing_speed(self):
        """Test calculating processing speed."""
        profiler = ImageProcessingProfiler()

        # Simulate processing a 100x100 image
        def process_image():
            time.sleep(0.01)

        profiler.profile_operation(process_image)

        speed = profiler.get_processing_speed((100, 100), "process_image")
        assert speed is not None
        assert speed > 0  # pixels per second

    def test_get_processing_speed_no_data(self):
        """Test get_processing_speed with no data."""
        profiler = ImageProcessingProfiler()
        speed = profiler.get_processing_speed((100, 100), "nonexistent")
        assert speed is None

    def test_get_processing_speed_zero_time(self):
        """Test get_processing_speed with zero average time."""
        profiler = ImageProcessingProfiler()
        profiler.monitor.record_metric("test_wall_time", 0, "seconds")
        speed = profiler.get_processing_speed((100, 100), "test")
        assert speed is None

    def test_estimate_batch_time(self):
        """Test estimating batch processing time."""
        profiler = ImageProcessingProfiler()

        # Profile a single operation
        def process_image():
            time.sleep(0.01)

        profiler.profile_operation(process_image)

        # Estimate batch of 5 images
        images = [(100, 100)] * 5
        estimated = profiler.estimate_batch_time(images, "process_image")

        assert estimated is not None
        assert estimated > 0.05  # At least 5 * 0.01 * 1.1 (with overhead)

    def test_estimate_batch_time_no_data(self):
        """Test estimate_batch_time with no data."""
        profiler = ImageProcessingProfiler()
        estimated = profiler.estimate_batch_time([(100, 100)], "nonexistent")
        assert estimated is None

    def test_track_memory_usage(self):
        """Test tracking memory usage."""
        profiler = ImageProcessingProfiler()
        profiler.track_memory_usage("test_operation")

        metrics = profiler.monitor.get_metrics("test_operation_memory_snapshot")
        assert len(metrics) == 1
        assert metrics[0].value > 0

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        profiler = ImageProcessingProfiler()
        stats = profiler.get_memory_stats()

        assert "rss_mb" in stats
        assert "vms_mb" in stats
        assert "percent" in stats
        assert "available_mb" in stats
        assert stats["rss_mb"] > 0

    def test_identify_bottlenecks(self):
        """Test identifying bottlenecks."""
        profiler = ImageProcessingProfiler()

        # Create operations with varying performance
        for i in range(10):
            profiler.monitor.record_metric("fast_op_wall_time", 0.01, "seconds")

        for i in range(10):
            profiler.monitor.record_metric("slow_op_wall_time", 0.05, "seconds")
            if i % 3 == 0:
                # Add some outliers
                profiler.monitor.record_metric("slow_op_wall_time", 0.5, "seconds")

        bottlenecks = profiler.identify_bottlenecks(["fast_op", "slow_op"])

        # slow_op should be identified as a bottleneck due to outliers
        assert len(bottlenecks) > 0
        if bottlenecks:
            assert bottlenecks[0]["operation"] == "slow_op"
            assert bottlenecks[0]["slowdown_factor"] > 2.0

    def test_identify_bottlenecks_no_operations(self):
        """Test identifying bottlenecks with no operations."""
        profiler = ImageProcessingProfiler()
        bottlenecks = profiler.identify_bottlenecks(["nonexistent"])
        assert bottlenecks == []


# ============================================================================
# APIPerformanceTracker Tests
# ============================================================================


class TestAPIPerformanceTracker:
    """Tests for APIPerformanceTracker class."""

    def test_init(self):
        """Test APIPerformanceTracker initialization."""
        tracker = APIPerformanceTracker(max_history=5000)
        assert len(tracker._requests) == 0

    def test_track_request(self):
        """Test tracking an API request."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.123, 200, method="GET")

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.endpoint == "/api/test"
        assert req.duration_ms == 123.0  # Converted to ms
        assert req.status_code == 200
        assert req.method == "GET"
        assert req.error is None

    def test_track_request_with_error(self):
        """Test tracking request with error."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.5, 500, error="Internal error")

        req = tracker._requests[0]
        assert req.status_code == 500
        assert req.error == "Internal error"

    def test_track_context_manager(self):
        """Test tracking with context manager."""
        tracker = APIPerformanceTracker()

        with tracker.track("/api/process", "POST"):
            time.sleep(0.01)

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.endpoint == "/api/process"
        assert req.method == "POST"
        assert req.duration_ms >= 10

    def test_track_context_manager_with_exception(self):
        """Test tracking context manager with exception."""
        tracker = APIPerformanceTracker()

        with pytest.raises(ValueError):
            with tracker.track("/api/fail", "POST"):
                raise ValueError("Test error")

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.status_code == 500
        assert req.error == "Test error"

    def test_get_endpoint_stats(self):
        """Test getting endpoint statistics."""
        tracker = APIPerformanceTracker()

        # Track multiple requests
        for i in range(10):
            tracker.track_request("/api/test", 0.01 * (i + 1), 200)

        stats = tracker.get_endpoint_stats("/api/test")

        assert stats["count"] == 10
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["min_ms"] > 0
        assert stats["max_ms"] > stats["min_ms"]
        assert "mean_ms" in stats
        assert "p90_ms" in stats

    def test_get_endpoint_stats_with_errors(self):
        """Test endpoint stats with error requests."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.01, 200)
        tracker.track_request("/api/test", 0.02, 500)
        tracker.track_request("/api/test", 0.03, 404)

        stats = tracker.get_endpoint_stats("/api/test")

        assert stats["count"] == 3
        assert stats["error_count"] == 2
        assert stats["error_rate"] == pytest.approx(2 / 3)

    def test_get_endpoint_stats_no_data(self):
        """Test endpoint stats with no data."""
        tracker = APIPerformanceTracker()
        stats = tracker.get_endpoint_stats("/api/nonexistent")
        assert stats == {"count": 0}

    def test_get_endpoint_stats_with_time_range(self):
        """Test endpoint stats with time range filter."""
        tracker = APIPerformanceTracker()

        start = datetime.now()
        tracker.track_request("/api/test", 0.01, 200)
        time.sleep(0.01)
        middle = datetime.now()
        time.sleep(0.01)
        tracker.track_request("/api/test", 0.02, 200)
        end = datetime.now()

        stats = tracker.get_endpoint_stats("/api/test", (start, middle))
        assert stats["count"] == 1

        stats = tracker.get_endpoint_stats("/api/test", (start, end))
        assert stats["count"] == 2

    def test_get_error_rate(self):
        """Test calculating error rate."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.01, 200)
        tracker.track_request("/api/test", 0.02, 500)

        error_rate = tracker.get_error_rate("/api/test")
        assert error_rate == 0.5

    def test_get_error_rate_no_data(self):
        """Test error rate with no data."""
        tracker = APIPerformanceTracker()
        error_rate = tracker.get_error_rate("/api/nonexistent")
        assert error_rate == 0.0

    def test_get_response_times(self):
        """Test getting response time distribution."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test1", 0.01, 200)
        tracker.track_request("/api/test1", 0.02, 200)
        tracker.track_request("/api/test2", 0.03, 200)

        response_times = tracker.get_response_times()

        assert "/api/test1" in response_times
        assert "/api/test2" in response_times
        assert len(response_times["/api/test1"]) == 2
        assert len(response_times["/api/test2"]) == 1

    def test_generate_api_report(self):
        """Test generating comprehensive API report."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/fast", 0.01, 200)
        tracker.track_request("/api/slow", 0.5, 200)
        tracker.track_request("/api/error", 0.1, 500)

        report = tracker.generate_api_report()

        assert report["total_requests"] == 3
        assert report["total_errors"] == 1
        assert report["overall_error_rate"] == pytest.approx(1 / 3)
        assert "overall_mean_ms" in report
        assert "endpoints" in report
        assert len(report["endpoints"]) == 3
        assert "slowest_endpoints" in report
        assert "highest_error_endpoints" in report

    def test_generate_api_report_empty(self):
        """Test generating report with no data."""
        tracker = APIPerformanceTracker()
        report = tracker.generate_api_report()
        assert report == {"total_requests": 0}

    def test_max_history_deque(self):
        """Test that deque respects maxlen."""
        tracker = APIPerformanceTracker(max_history=10)

        for i in range(20):
            tracker.track_request("/api/test", 0.01, 200)

        assert len(tracker._requests) == 10


# ============================================================================
# CacheManager Tests
# ============================================================================


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init(self):
        """Test CacheManager initialization."""
        cache = CacheManager(max_size=100, default_ttl=3600)
        assert cache._max_size == 100
        assert cache._default_ttl == 3600
        assert cache.get_size() == 0

    def test_get_set(self):
        """Test basic get/set operations."""
        cache = CacheManager()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = CacheManager()
        assert cache.get("nonexistent") is None

    def test_get_updates_stats(self):
        """Test that get updates hit/miss stats."""
        cache = CacheManager()

        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats.misses == 1

        cache.set("key1", "value1")
        cache.get("key1")
        stats = cache.get_stats()
        assert stats.hits == 1

    def test_set_with_ttl(self):
        """Test setting value with TTL."""
        cache = CacheManager()

        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

        stats = cache.get_stats()
        assert stats.expirations == 1

    def test_set_replaces_existing(self):
        """Test that set replaces existing value."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"
        assert cache.get_size() == 1

    def test_lru_eviction(self):
        """Test LRU eviction when max_size is exceeded."""
        cache = CacheManager(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

        stats = cache.get_stats()
        assert stats.evictions == 1

    def test_lru_ordering(self):
        """Test that accessing keys updates LRU order."""
        cache = CacheManager(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_delete(self):
        """Test deleting key from cache."""
        cache = CacheManager()

        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        cache = CacheManager()
        assert cache.delete("nonexistent") is False

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = CacheManager(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1
        assert stats.max_size == 10
        assert stats.hit_rate == 0.5
        assert stats.miss_rate == 0.5

    def test_get_stats_no_requests(self):
        """Test stats with no requests."""
        cache = CacheManager()
        stats = cache.get_stats()
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0

    def test_clear(self):
        """Test clearing cache."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get_size() == 0
        assert cache.get("key1") is None

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        cache = CacheManager()

        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=10)

        time.sleep(1.1)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_thread_safety(self):
        """Test thread safety of CacheManager."""
        cache = CacheManager()

        def worker(thread_id):
            for i in range(100):
                cache.set(f"key_{thread_id}_{i}", f"value_{i}")
                cache.get(f"key_{thread_id}_{i}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = cache.get_stats()
        assert stats.hits > 0


# ============================================================================
# ResourceMonitor Tests
# ============================================================================


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(
            cpu_threshold=80.0, memory_threshold=80.0, disk_threshold=90.0
        )
        assert monitor.cpu_threshold == 80.0
        assert monitor.memory_threshold == 80.0
        assert monitor.disk_threshold == 90.0

    def test_get_cpu_usage(self):
        """Test getting CPU usage."""
        monitor = ResourceMonitor()
        cpu = monitor.get_cpu_usage()
        assert 0 <= cpu <= 100

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        monitor = ResourceMonitor()
        mem = monitor.get_memory_usage()

        assert "percent" in mem
        assert "used_mb" in mem
        assert "available_mb" in mem
        assert "total_mb" in mem
        assert 0 <= mem["percent"] <= 100
        assert mem["used_mb"] > 0

    def test_get_disk_usage(self):
        """Test getting disk usage."""
        monitor = ResourceMonitor()
        disk = monitor.get_disk_usage("/")

        assert "percent" in disk
        assert "used_gb" in disk
        assert "free_gb" in disk
        assert "total_gb" in disk
        assert 0 <= disk["percent"] <= 100

    def test_get_gpu_usage_unavailable(self):
        """Test GPU usage when unavailable."""
        monitor = ResourceMonitor()
        gpu = monitor.get_gpu_usage()
        # GPU likely not available in test environment
        assert gpu is None or isinstance(gpu, dict)

    def test_check_resources(self):
        """Test comprehensive resource check."""
        monitor = ResourceMonitor()
        resources = monitor.check_resources()

        assert isinstance(resources, ResourceUsage)
        assert 0 <= resources.cpu_percent <= 100
        assert 0 <= resources.memory_percent <= 100
        assert 0 <= resources.disk_percent <= 100
        assert resources.memory_used_mb > 0
        assert resources.disk_used_gb >= 0

    def test_get_alerts_for_high_usage(self):
        """Test getting alerts for high usage."""
        monitor = ResourceMonitor(cpu_threshold=0.0, memory_threshold=0.0, disk_threshold=0.0)

        alerts = monitor.get_alerts_for_high_usage()

        # With 0% thresholds, we should get alerts for memory and disk at minimum
        assert len(alerts) >= 1
        # Memory and disk should always have some usage
        assert any(a["resource"] in ["memory", "disk"] for a in alerts)

    def test_get_alerts_severity(self):
        """Test alert severity levels."""
        monitor = ResourceMonitor(cpu_threshold=0.0)
        alerts = monitor.get_alerts_for_high_usage()

        if alerts:
            alert = alerts[0]
            assert alert["severity"] in ["warning", "critical"]
            assert "usage" in alert
            assert "threshold" in alert
            assert "message" in alert


# ============================================================================
# PerformanceReport Tests
# ============================================================================


class TestPerformanceReport:
    """Tests for PerformanceReport class."""

    def test_init(self):
        """Test PerformanceReport initialization."""
        monitor = PerformanceMonitor()
        report = PerformanceReport(monitor)
        assert report.monitor is monitor

    def test_init_with_trackers(self):
        """Test PerformanceReport with all components."""
        monitor = PerformanceMonitor()
        api_tracker = APIPerformanceTracker()
        resource_monitor = ResourceMonitor()

        report = PerformanceReport(monitor, api_tracker, resource_monitor)
        assert report.api_tracker is api_tracker
        assert report.resource_monitor is resource_monitor

    def test_generate_daily_report(self):
        """Test generating daily report."""
        monitor = PerformanceMonitor()
        monitor.record_metric("test_op", 123, "ms")

        report_gen = PerformanceReport(monitor)
        report = report_gen.generate_daily_report()

        assert "date" in report
        assert "generated_at" in report
        assert "operations" in report

    def test_generate_daily_report_specific_date(self):
        """Test generating daily report for specific date."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        specific_date = datetime(2024, 1, 15)
        report = report_gen.generate_daily_report(specific_date)

        assert report["date"] == "2024-01-15"

    def test_generate_daily_report_with_api_tracker(self):
        """Test daily report includes API stats."""
        monitor = PerformanceMonitor()
        api_tracker = APIPerformanceTracker()
        api_tracker.track_request("/api/test", 0.01, 200)

        report_gen = PerformanceReport(monitor, api_tracker)
        report = report_gen.generate_daily_report()

        assert "api" in report
        assert report["api"]["total_requests"] == 1

    def test_generate_daily_report_with_resource_monitor(self):
        """Test daily report includes resource stats."""
        monitor = PerformanceMonitor()
        resource_monitor = ResourceMonitor()

        report_gen = PerformanceReport(monitor, resource_monitor=resource_monitor)
        report = report_gen.generate_daily_report()

        assert "resources" in report
        assert "resource_alerts" in report

    def test_generate_session_report(self):
        """Test generating session report."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 10, "ms", {"session_id": "session123"})
        monitor.record_metric("op1", 20, "ms", {"session_id": "session123"})
        monitor.record_metric("op2", 30, "ms", {"session_id": "other"})

        report_gen = PerformanceReport(monitor)
        report = report_gen.generate_session_report("session123")

        assert report["session_id"] == "session123"
        assert "op1" in report["operations"]
        assert "op2" not in report["operations"]
        assert report["operations"]["op1"]["count"] == 2

    def test_compare_performance(self):
        """Test comparing performance between periods."""
        monitor = PerformanceMonitor()

        # Period 1
        period1_start = datetime.now()
        monitor.record_metric("test_op", 10, "ms")
        monitor.record_metric("test_op", 12, "ms")
        period1_end = datetime.now()

        time.sleep(0.01)

        # Period 2
        period2_start = datetime.now()
        monitor.record_metric("test_op", 20, "ms")
        monitor.record_metric("test_op", 22, "ms")
        period2_end = datetime.now()

        report_gen = PerformanceReport(monitor)
        comparison = report_gen.compare_performance(
            (period1_start, period1_end), (period2_start, period2_end)
        )

        assert "period1" in comparison
        assert "period2" in comparison
        assert "changes" in comparison
        assert "test_op" in comparison["changes"]
        assert comparison["changes"]["test_op"]["direction"] == "degraded"

    def test_export_report_json(self, tmp_path):
        """Test exporting report to JSON."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        report = {"test": "data"}
        output_path = tmp_path / "report.json"

        result = report_gen.export_report(report, format="json", path=output_path)
        assert result is None
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert data["test"] == "data"

    def test_export_report_json_string(self):
        """Test exporting report to JSON string."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        report = {"test": "data"}
        json_str = report_gen.export_report(report, format="json")

        assert json_str is not None
        data = json.loads(json_str)
        assert data["test"] == "data"

    def test_export_report_csv(self, tmp_path):
        """Test exporting report to CSV."""
        monitor = PerformanceMonitor()
        monitor.record_metric("op1", 10, "ms")

        report_gen = PerformanceReport(monitor)
        report = report_gen.generate_daily_report()

        output_path = tmp_path / "report.csv"
        result = report_gen.export_report(report, format="csv", path=output_path)

        assert result is None
        assert output_path.exists()

    def test_export_report_invalid_format(self):
        """Test exporting with invalid format."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        with pytest.raises(ValueError, match="Unsupported format"):
            report_gen.export_report({}, format="xml")


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstances:
    """Tests for global instance getters."""

    def test_get_monitor(self):
        """Test getting global monitor instance."""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        assert monitor1 is monitor2
        assert isinstance(monitor1, PerformanceMonitor)

    def test_get_profiler(self):
        """Test getting global profiler instance."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        assert profiler1 is profiler2
        assert isinstance(profiler1, ImageProcessingProfiler)

    def test_get_api_tracker(self):
        """Test getting global API tracker instance."""
        tracker1 = get_api_tracker()
        tracker2 = get_api_tracker()
        assert tracker1 is tracker2
        assert isinstance(tracker1, APIPerformanceTracker)

    def test_get_cache(self):
        """Test getting global cache instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2
        assert isinstance(cache1, CacheManager)

    def test_get_resource_monitor(self):
        """Test getting global resource monitor instance."""
        monitor1 = get_resource_monitor()
        monitor2 = get_resource_monitor()
        assert monitor1 is monitor2
        assert isinstance(monitor1, ResourceMonitor)


# ============================================================================
# SQLiteRepository Tests
# ============================================================================


class TestSQLiteRepository:
    """Tests for SQLiteRepository class."""

    def test_init(self, tmp_path):
        """Test SQLiteRepository initialization."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        assert repo.model_class == TestModel
        assert repo.table_name == "test_table"
        assert repo.db_path == db_path
        assert db_path.exists()

    def test_init_creates_schema(self, tmp_path):
        """Test that init creates database schema."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["category"]
        )

        # Check table exists
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_add(self, tmp_path):
        """Test adding entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        assert saved.id is not None
        assert saved.name == "Test"
        assert saved.value == 123

    def test_add_with_existing_id(self, tmp_path):
        """Test adding entity with existing ID."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(id="custom_id", name="Test", value=123)
        saved = repo.add(entity)

        assert saved.id == "custom_id"

    def test_get(self, tmp_path):
        """Test getting entity by ID."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        retrieved = repo.get(saved.id)
        assert retrieved is not None
        assert retrieved.name == "Test"
        assert retrieved.value == 123

    def test_get_nonexistent(self, tmp_path):
        """Test getting nonexistent entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        result = repo.get("nonexistent")
        assert result is None

    def test_get_all(self, tmp_path):
        """Test getting all entities."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        for i in range(5):
            repo.add(TestModel(name=f"Test{i}", value=i))

        all_entities = repo.get_all()
        assert len(all_entities) == 5

    def test_get_all_with_pagination(self, tmp_path):
        """Test getting entities with pagination."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        for i in range(10):
            repo.add(TestModel(name=f"Test{i}", value=i))

        page1 = repo.get_all(limit=5, offset=0)
        page2 = repo.get_all(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].name != page2[0].name

    def test_update(self, tmp_path):
        """Test updating entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(name="Original", value=100)
        saved = repo.add(entity)

        updated_entity = TestModel(name="Updated", value=200)
        result = repo.update(saved.id, updated_entity)

        assert result is not None
        assert result.name == "Updated"

        # Verify in database
        retrieved = repo.get(saved.id)
        assert retrieved.name == "Updated"
        assert retrieved.value == 200

    def test_update_nonexistent(self, tmp_path):
        """Test updating nonexistent entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(name="Test", value=123)
        result = repo.update("nonexistent", entity)
        assert result is None

    def test_delete(self, tmp_path):
        """Test deleting entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        assert repo.delete(saved.id) is True
        assert repo.get(saved.id) is None

    def test_delete_nonexistent(self, tmp_path):
        """Test deleting nonexistent entity."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        assert repo.delete("nonexistent") is False

    def test_find_with_indexed_field(self, tmp_path):
        """Test finding entities with indexed field."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["category"]
        )

        repo.add(TestModel(name="Test1", value=1, category="A"))
        repo.add(TestModel(name="Test2", value=2, category="B"))
        repo.add(TestModel(name="Test3", value=3, category="A"))

        results = repo.find(category="A")
        assert len(results) == 2
        assert all(r.category == "A" for r in results)

    def test_find_with_non_indexed_field(self, tmp_path):
        """Test finding entities with non-indexed field (string field)."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        repo.add(TestModel(name="Test1", value=100))
        repo.add(TestModel(name="Test2", value=200))

        # Use string field (name) which is not indexed
        results = repo.find(name="Test1")
        assert len(results) == 1
        assert results[0].name == "Test1"

    def test_find_no_results(self, tmp_path):
        """Test find with no matching results."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        results = repo.find(category="nonexistent")
        assert len(results) == 0

    def test_count(self, tmp_path):
        """Test counting entities."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["category"]
        )

        repo.add(TestModel(name="Test1", value=1, category="A"))
        repo.add(TestModel(name="Test2", value=2, category="B"))
        repo.add(TestModel(name="Test3", value=3, category="A"))

        count = repo.count(category="A")
        assert count == 2

    def test_count_all(self, tmp_path):
        """Test counting all entities."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        for i in range(5):
            repo.add(TestModel(name=f"Test{i}", value=i))

        count = repo.count()
        assert count == 5

    def test_search(self, tmp_path):
        """Test full-text search."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["name", "category"]
        )

        repo.add(TestModel(name="Apple", value=1, category="Fruit"))
        repo.add(TestModel(name="Banana", value=2, category="Fruit"))
        repo.add(TestModel(name="Carrot", value=3, category="Vegetable"))

        results = repo.search("Fruit")
        assert len(results) == 2

        results = repo.search("Ban")
        assert len(results) == 1
        assert results[0].name == "Banana"

    def test_search_with_limit(self, tmp_path):
        """Test search with limit."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["category"]
        )

        for i in range(10):
            repo.add(TestModel(name=f"Test{i}", value=i, category="common"))

        results = repo.search("common", limit=5)
        assert len(results) == 5

    def test_indexed_fields_extraction(self, tmp_path):
        """Test that indexed fields are properly extracted and stored."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            TestModel, "test_table", db_path=db_path, indexed_fields=["category", "name"]
        )

        entity = TestModel(name="Test", value=123, category="TestCat")
        saved = repo.add(entity)

        # Verify indexed fields in database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT category, name FROM {repo.table_name} WHERE id = ?", (saved.id,)
        )
        row = cursor.fetchone()
        assert row[0] == "TestCat"
        assert row[1] == "Test"
        conn.close()

    def test_concurrent_access(self, tmp_path):
        """Test concurrent access to SQLite repository."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        def worker(thread_id):
            for i in range(10):
                repo.add(TestModel(name=f"Thread{thread_id}_Item{i}", value=i))

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            concurrent.futures.wait(futures)

        count = repo.count()
        assert count == 30


# ============================================================================
# InMemoryRepository Tests
# ============================================================================


class TestInMemoryRepository:
    """Tests for InMemoryRepository class."""

    def test_init(self):
        """Test InMemoryRepository initialization."""
        repo = InMemoryRepository(TestModel)
        assert repo.model_class == TestModel
        assert len(repo._store) == 0
        assert repo._counter == 0

    def test_add(self):
        """Test adding entity."""
        repo = InMemoryRepository(TestModel)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        assert saved.id is not None
        assert saved.name == "Test"
        assert saved.value == 123

    def test_add_multiple(self):
        """Test adding multiple entities."""
        repo = InMemoryRepository(TestModel)

        entity1 = repo.add(TestModel(name="Test1", value=1))
        entity2 = repo.add(TestModel(name="Test2", value=2))

        assert entity1.id != entity2.id

    def test_get(self):
        """Test getting entity by ID."""
        repo = InMemoryRepository(TestModel)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        retrieved = repo.get(saved.id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_nonexistent(self):
        """Test getting nonexistent entity."""
        repo = InMemoryRepository(TestModel)
        result = repo.get("999")
        assert result is None

    def test_get_all(self):
        """Test getting all entities."""
        repo = InMemoryRepository(TestModel)

        for i in range(5):
            repo.add(TestModel(name=f"Test{i}", value=i))

        all_entities = repo.get_all()
        assert len(all_entities) == 5

    def test_get_all_with_pagination(self):
        """Test getting entities with pagination."""
        repo = InMemoryRepository(TestModel)

        for i in range(10):
            repo.add(TestModel(name=f"Test{i}", value=i))

        page1 = repo.get_all(limit=5, offset=0)
        page2 = repo.get_all(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5

    def test_update(self):
        """Test updating entity."""
        repo = InMemoryRepository(TestModel)

        entity = TestModel(name="Original", value=100)
        saved = repo.add(entity)

        updated_entity = TestModel(name="Updated", value=200)
        result = repo.update(saved.id, updated_entity)

        assert result is not None
        assert result.name == "Updated"
        assert result.value == 200

    def test_update_nonexistent(self):
        """Test updating nonexistent entity."""
        repo = InMemoryRepository(TestModel)

        entity = TestModel(name="Test", value=123)
        result = repo.update("999", entity)
        assert result is None

    def test_delete(self):
        """Test deleting entity."""
        repo = InMemoryRepository(TestModel)

        entity = TestModel(name="Test", value=123)
        saved = repo.add(entity)

        assert repo.delete(saved.id) is True
        assert repo.get(saved.id) is None

    def test_delete_nonexistent(self):
        """Test deleting nonexistent entity."""
        repo = InMemoryRepository(TestModel)
        assert repo.delete("999") is False

    def test_find(self):
        """Test finding entities."""
        repo = InMemoryRepository(TestModel)

        repo.add(TestModel(name="Test1", value=100, category="A"))
        repo.add(TestModel(name="Test2", value=200, category="B"))
        repo.add(TestModel(name="Test3", value=300, category="A"))

        results = repo.find(category="A")
        assert len(results) == 2
        assert all(r.category == "A" for r in results)

    def test_find_multiple_criteria(self):
        """Test finding with multiple criteria."""
        repo = InMemoryRepository(TestModel)

        repo.add(TestModel(name="Test1", value=100, category="A"))
        repo.add(TestModel(name="Test2", value=100, category="B"))
        repo.add(TestModel(name="Test3", value=200, category="A"))

        results = repo.find(value=100, category="A")
        assert len(results) == 1
        assert results[0].name == "Test1"

    def test_find_no_results(self):
        """Test find with no matching results."""
        repo = InMemoryRepository(TestModel)
        results = repo.find(category="nonexistent")
        assert len(results) == 0

    def test_count(self):
        """Test counting entities."""
        repo = InMemoryRepository(TestModel)

        repo.add(TestModel(name="Test1", value=1, category="A"))
        repo.add(TestModel(name="Test2", value=2, category="B"))
        repo.add(TestModel(name="Test3", value=3, category="A"))

        count = repo.count(category="A")
        assert count == 2

    def test_count_all(self):
        """Test counting all entities."""
        repo = InMemoryRepository(TestModel)

        for i in range(5):
            repo.add(TestModel(name=f"Test{i}", value=i))

        count = repo.count()
        assert count == 5

    def test_clear(self):
        """Test clearing repository."""
        repo = InMemoryRepository(TestModel)

        for i in range(5):
            repo.add(TestModel(name=f"Test{i}", value=i))

        repo.clear()

        assert len(repo._store) == 0
        assert repo._counter == 0
        assert len(repo.get_all()) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestRepositoryIntegration:
    """Integration tests for repository implementations."""

    def test_sqlite_and_inmemory_equivalence(self, tmp_path):
        """Test that SQLite and InMemory repos behave the same."""
        db_path = tmp_path / "test.db"
        sqlite_repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)
        inmem_repo = InMemoryRepository(TestModel)

        # Add same data to both
        for i in range(5):
            entity = TestModel(name=f"Test{i}", value=i, category="A")
            sqlite_repo.add(entity)
            inmem_repo.add(entity)

        # Both should have 5 entities
        assert len(sqlite_repo.get_all()) == 5
        assert len(inmem_repo.get_all()) == 5

        # Both should find same results
        sqlite_results = sqlite_repo.find(category="A")
        inmem_results = inmem_repo.find(category="A")
        assert len(sqlite_results) == len(inmem_results)

    def test_complex_workflow(self, tmp_path):
        """Test complex CRUD workflow."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(
            PersonModel, "persons", db_path=db_path, indexed_fields=["last_name", "email"]
        )

        # Create
        person1 = repo.add(
            PersonModel(first_name="John", last_name="Doe", age=30, email="john@example.com")
        )
        person2 = repo.add(
            PersonModel(first_name="Jane", last_name="Smith", age=25, email="jane@example.com")
        )

        # Read
        assert repo.count() == 2
        retrieved = repo.get(person1.id)
        assert retrieved.first_name == "John"

        # Update
        updated = PersonModel(
            first_name="John", last_name="Doe", age=31, email="john.doe@example.com"
        )
        repo.update(person1.id, updated)

        # Verify update
        retrieved = repo.get(person1.id)
        assert retrieved.age == 31

        # Search
        results = repo.search("Doe")
        assert len(results) == 1

        # Delete
        repo.delete(person2.id)
        assert repo.count() == 1

    def test_performance_monitoring_with_repository(self, tmp_path):
        """Test using performance monitoring with repository operations."""
        monitor = PerformanceMonitor()
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(TestModel, "test_table", db_path=db_path)

        # Time repository operations
        with monitor.timer("repo_add"):
            for i in range(10):
                repo.add(TestModel(name=f"Test{i}", value=i))

        with monitor.timer("repo_get_all"):
            all_entities = repo.get_all()

        with monitor.timer("repo_find"):
            results = repo.find(value=5)

        # Verify metrics were recorded
        assert len(monitor.get_metrics("repo_add")) == 1
        assert len(monitor.get_metrics("repo_get_all")) == 1
        assert len(monitor.get_metrics("repo_find")) == 1

        # Get statistics
        stats = monitor.get_statistics("repo_add")
        assert stats["count"] == 1
        assert stats["mean"] > 0
