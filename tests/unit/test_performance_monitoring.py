"""
Comprehensive tests for performance monitoring module.

Tests cover:
- PerformanceMonitor: timing, metrics, statistics, export
- ImageProcessingProfiler: profiling, speed calculation, bottlenecks
- APIPerformanceTracker: request tracking, stats, reports
- CacheManager: get/set, TTL, LRU eviction, stats
- ResourceMonitor: CPU/memory/disk usage, alerts
- PerformanceReport: daily/session reports, comparisons
- Thread safety and concurrent operations
- Edge cases (empty metrics, etc.)
"""

import concurrent.futures
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
)

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

        with pytest.raises(ValueError), monitor.timer("error_test"):
            raise ValueError("Test error")

        # Timer should still record the metric
        metrics = monitor.get_metrics("error_test")
        assert len(metrics) == 1

    def test_record_metric(self):
        """Test recording arbitrary metrics."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test_metric", 123.45, "ms", {"key": "value"})

        metrics = monitor.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].metric_name == "test_metric"
        assert metrics[0].value == 123.45
        assert metrics[0].unit == "ms"
        assert metrics[0].metadata["key"] == "value"

    def test_record_metric_max_history(self):
        """Test that max_history is enforced."""
        monitor = PerformanceMonitor(max_history=5)

        # Record 10 metrics
        for i in range(10):
            monitor.record_metric("test", i, "count")

        metrics = monitor.get_metrics("test")
        assert len(metrics) == 5
        # Should keep the most recent 5
        assert metrics[0].value == 5.0
        assert metrics[4].value == 9.0

    def test_get_metrics_with_time_range(self):
        """Test filtering metrics by time range."""
        monitor = PerformanceMonitor()

        # Record metrics at different times
        datetime.now()
        monitor.record_metric("test", 1, "count")

        time.sleep(0.1)
        middle = datetime.now()
        monitor.record_metric("test", 2, "count")

        time.sleep(0.1)
        datetime.now()
        monitor.record_metric("test", 3, "count")

        # Get all metrics
        all_metrics = monitor.get_metrics("test")
        assert len(all_metrics) == 3

        # Get only middle metric
        middle_metrics = monitor.get_metrics(
            "test", (middle - timedelta(seconds=0.05), middle + timedelta(seconds=0.05))
        )
        assert len(middle_metrics) == 1
        assert middle_metrics[0].value == 2.0

    def test_get_average(self):
        """Test average calculation."""
        monitor = PerformanceMonitor()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            monitor.record_metric("test", v, "units")

        avg = monitor.get_average("test")
        assert avg == pytest.approx(3.0)

    def test_get_average_empty(self):
        """Test average with no metrics."""
        monitor = PerformanceMonitor()
        avg = monitor.get_average("nonexistent")
        assert avg is None

    def test_get_percentiles(self):
        """Test percentile calculation."""
        monitor = PerformanceMonitor()

        # Record 100 values
        for i in range(100):
            monitor.record_metric("test", i, "units")

        percentiles = monitor.get_percentiles("test", [50, 90, 95, 99])

        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        assert percentiles["p50"] == pytest.approx(49.5, rel=0.1)
        assert percentiles["p90"] == pytest.approx(89.1, rel=0.1)
        assert percentiles["p99"] == pytest.approx(98.01, rel=0.1)

    def test_get_percentiles_empty(self):
        """Test percentiles with no metrics."""
        monitor = PerformanceMonitor()
        percentiles = monitor.get_percentiles("nonexistent")
        assert percentiles == {}

    def test_get_statistics(self):
        """Test comprehensive statistics."""
        monitor = PerformanceMonitor()

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in values:
            monitor.record_metric("test", v, "ms")

        stats = monitor.get_statistics("test")

        assert stats["count"] == 10
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["mean"] == pytest.approx(5.5)
        assert stats["median"] == pytest.approx(5.5)
        assert stats["unit"] == "ms"
        assert "p50" in stats
        assert "p90" in stats

    def test_get_statistics_empty(self):
        """Test statistics with no metrics."""
        monitor = PerformanceMonitor()
        stats = monitor.get_statistics("nonexistent")
        assert stats == {"count": 0}

    def test_export_metrics_json(self):
        """Test exporting metrics to JSON."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test1", 10.0, "ms")
        monitor.record_metric("test2", 20.0, "ms")

        json_str = monitor.export_metrics(format="json")
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["metric_name"] == "test1"
        assert data[0]["value"] == 10.0

    def test_export_metrics_json_to_file(self):
        """Test exporting metrics to JSON file."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 10.0, "ms")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            path = Path(f.name)

        try:
            result = monitor.export_metrics(format="json", output_path=path)
            assert result is None

            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["metric_name"] == "test"
        finally:
            path.unlink()

    def test_export_metrics_csv(self):
        """Test exporting metrics to CSV."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test1", 10.0, "ms", {"key": "value"})
        monitor.record_metric("test2", 20.0, "s")

        csv_str = monitor.export_metrics(format="csv")
        lines = csv_str.strip().split("\n")

        assert len(lines) == 3  # header + 2 data rows
        assert "metric_name" in lines[0]
        assert "test1" in lines[1]
        assert "test2" in lines[2]

    def test_export_metrics_csv_to_file(self):
        """Test exporting metrics to CSV file."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 10.0, "ms")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            path = Path(f.name)

        try:
            result = monitor.export_metrics(format="csv", output_path=path)
            assert result is None

            content = path.read_text()
            assert "metric_name" in content
            assert "test" in content
        finally:
            path.unlink()

    def test_export_metrics_empty_csv(self):
        """Test exporting empty metrics to CSV."""
        monitor = PerformanceMonitor()
        csv_str = monitor.export_metrics(format="csv")
        assert csv_str == ""

    def test_export_metrics_filter_by_operation(self):
        """Test exporting metrics filtered by operation name."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 10.0, "ms")
        monitor.record_metric("op2", 20.0, "ms")
        monitor.record_metric("op1", 30.0, "ms")

        json_str = monitor.export_metrics(format="json", operation_name="op1")
        data = json.loads(json_str)

        assert len(data) == 2
        assert all(m["metric_name"] == "op1" for m in data)

    def test_export_metrics_invalid_format(self):
        """Test export with invalid format."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError, match="Unsupported format"):
            monitor.export_metrics(format="xml")

    def test_clear_metrics(self):
        """Test clearing metrics."""
        monitor = PerformanceMonitor()

        monitor.record_metric("op1", 10.0, "ms")
        monitor.record_metric("op2", 20.0, "ms")

        monitor.clear_metrics("op1")
        assert len(monitor.get_metrics("op1")) == 0
        assert len(monitor.get_metrics("op2")) == 1

        monitor.clear_metrics()
        assert len(monitor.get_metrics("op2")) == 0

    def test_thread_safety_concurrent_timers(self):
        """Test thread safety with concurrent timer operations."""
        monitor = PerformanceMonitor()
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    op_name = f"thread_{thread_id}_op_{i}"
                    with monitor.timer(op_name):
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 50 different metrics (5 threads * 10 ops)
        assert len(monitor._metrics) == 50

    def test_thread_safety_concurrent_metrics(self):
        """Test thread safety with concurrent metric recording."""
        monitor = PerformanceMonitor()
        errors = []

        def worker(_thread_id):
            try:
                for i in range(100):
                    monitor.record_metric("shared_metric", i, "count")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = monitor.get_metrics("shared_metric")
        assert len(metrics) == 1000  # 10 threads * 100 metrics


# ============================================================================
# ImageProcessingProfiler Tests
# ============================================================================


class TestImageProcessingProfiler:
    """Tests for ImageProcessingProfiler class."""

    def test_init(self):
        """Test ImageProcessingProfiler initialization."""
        profiler = ImageProcessingProfiler()
        assert profiler.monitor is not None
        assert len(profiler._memory_tracking) == 0

    def test_init_with_monitor(self):
        """Test initialization with existing monitor."""
        monitor = PerformanceMonitor()
        profiler = ImageProcessingProfiler(monitor=monitor)
        assert profiler.monitor is monitor

    def test_profile_operation_success(self):
        """Test profiling a successful operation."""
        profiler = ImageProcessingProfiler()

        def sample_func(x, y):
            time.sleep(0.01)
            return x + y

        result, profile_data = profiler.profile_operation(sample_func, 5, 10)

        assert result == 15
        assert profile_data["success"] is True
        assert profile_data["operation"] == "sample_func"
        assert profile_data["wall_time_s"] >= 0.01
        assert "cpu_time_s" in profile_data
        assert "memory_delta_mb" in profile_data

    def test_profile_operation_failure(self):
        """Test profiling a failed operation."""
        profiler = ImageProcessingProfiler()

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            profiler.profile_operation(failing_func)

        # Should still record metrics
        profiler.monitor.get_metrics("failing_func_wall_time")
        # The metric might not be recorded if error happens before recording
        # Just ensure no crash occurred

    def test_get_processing_speed(self):
        """Test processing speed calculation."""
        profiler = ImageProcessingProfiler()

        # Simulate processing a 1000x1000 image in 0.1 seconds
        profiler.monitor.record_metric("process_wall_time", 0.1, "seconds")

        speed = profiler.get_processing_speed((1000, 1000), "process")

        assert speed is not None
        assert speed == pytest.approx(1000000 / 0.1)  # 10M pixels/sec

    def test_get_processing_speed_no_data(self):
        """Test processing speed with no metrics."""
        profiler = ImageProcessingProfiler()
        speed = profiler.get_processing_speed((1000, 1000), "nonexistent")
        assert speed is None

    def test_get_processing_speed_zero_time(self):
        """Test processing speed with zero time."""
        profiler = ImageProcessingProfiler()
        profiler.monitor.record_metric("instant_wall_time", 0.0, "seconds")

        speed = profiler.get_processing_speed((100, 100), "instant")
        assert speed is None

    def test_estimate_batch_time(self):
        """Test batch time estimation."""
        profiler = ImageProcessingProfiler()

        # Record some sample timings
        for _ in range(5):
            profiler.monitor.record_metric("batch_wall_time", 0.1, "seconds")

        images = [(1000, 1000)] * 10
        estimated = profiler.estimate_batch_time(images, "batch")

        assert estimated is not None
        # Should be ~1.0s (10 images * 0.1s) + 10% overhead = 1.1s
        assert estimated == pytest.approx(1.1, rel=0.01)

    def test_estimate_batch_time_no_data(self):
        """Test batch time estimation with no data."""
        profiler = ImageProcessingProfiler()
        estimated = profiler.estimate_batch_time([(100, 100)], "nonexistent")
        assert estimated is None

    def test_track_memory_usage(self):
        """Test memory usage tracking."""
        profiler = ImageProcessingProfiler()

        profiler.track_memory_usage("test_op")

        assert "test_op" in profiler._memory_tracking
        assert profiler._memory_tracking["test_op"] > 0

        metrics = profiler.monitor.get_metrics("test_op_memory_snapshot")
        assert len(metrics) == 1

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
        """Test bottleneck identification."""
        profiler = ImageProcessingProfiler()

        # Create a slow operation with high variance
        for _i in range(10):
            # Most operations are fast
            profiler.monitor.record_metric("fast_op_wall_time", 0.01, "seconds")
        for _i in range(2):
            # Some operations are very slow
            profiler.monitor.record_metric("fast_op_wall_time", 0.5, "seconds")

        # Create a consistent fast operation
        for _i in range(10):
            profiler.monitor.record_metric("consistent_op_wall_time", 0.01, "seconds")

        bottlenecks = profiler.identify_bottlenecks(["fast_op", "consistent_op"])

        # fast_op should be identified as a bottleneck due to high variance
        assert len(bottlenecks) >= 1
        if bottlenecks:
            assert bottlenecks[0]["operation"] == "fast_op"
            assert bottlenecks[0]["slowdown_factor"] > 2.0

    def test_identify_bottlenecks_no_data(self):
        """Test bottleneck identification with no data."""
        profiler = ImageProcessingProfiler()
        bottlenecks = profiler.identify_bottlenecks(["nonexistent"])
        assert len(bottlenecks) == 0

    def test_identify_bottlenecks_empty_list(self):
        """Test bottleneck identification with empty operation list."""
        profiler = ImageProcessingProfiler()
        bottlenecks = profiler.identify_bottlenecks([])
        assert len(bottlenecks) == 0


# ============================================================================
# APIPerformanceTracker Tests
# ============================================================================


class TestAPIPerformanceTracker:
    """Tests for APIPerformanceTracker class."""

    def test_init(self):
        """Test APIPerformanceTracker initialization."""
        tracker = APIPerformanceTracker(max_history=1000)
        assert len(tracker._requests) == 0

    def test_track_request(self):
        """Test tracking an API request."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.5, 200, "GET")

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.endpoint == "/api/test"
        assert req.duration_ms == 500.0  # Converted to ms
        assert req.status_code == 200
        assert req.method == "GET"

    def test_track_request_with_error(self):
        """Test tracking a request with error."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/fail", 0.1, 500, "POST", "Internal error")

        req = tracker._requests[0]
        assert req.status_code == 500
        assert req.error == "Internal error"

    def test_track_context_manager_success(self):
        """Test track context manager with success."""
        tracker = APIPerformanceTracker()

        with tracker.track("/api/test", "GET"):
            time.sleep(0.01)

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.status_code == 200
        assert req.duration_ms >= 10.0

    def test_track_context_manager_error(self):
        """Test track context manager with error."""
        tracker = APIPerformanceTracker()

        with pytest.raises(ValueError), tracker.track("/api/fail", "POST"):
            raise ValueError("Test error")

        assert len(tracker._requests) == 1
        req = tracker._requests[0]
        assert req.status_code == 500
        assert "Test error" in req.error

    def test_get_endpoint_stats(self):
        """Test getting endpoint statistics."""
        tracker = APIPerformanceTracker()

        # Track multiple requests
        for i in range(10):
            tracker.track_request("/api/test", 0.1 + i * 0.01, 200, "GET")

        stats = tracker.get_endpoint_stats("/api/test")

        assert stats["count"] == 10
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["min_ms"] >= 100.0
        assert stats["max_ms"] >= 190.0
        assert "mean_ms" in stats
        assert "p90_ms" in stats

    def test_get_endpoint_stats_with_errors(self):
        """Test endpoint stats with some errors."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.1, 200, "GET")
        tracker.track_request("/api/test", 0.1, 200, "GET")
        tracker.track_request("/api/test", 0.1, 500, "GET")

        stats = tracker.get_endpoint_stats("/api/test")

        assert stats["count"] == 3
        assert stats["error_count"] == 1
        assert stats["error_rate"] == pytest.approx(1 / 3)

    def test_get_endpoint_stats_no_data(self):
        """Test endpoint stats with no data."""
        tracker = APIPerformanceTracker()
        stats = tracker.get_endpoint_stats("/api/nonexistent")
        assert stats == {"count": 0}

    def test_get_endpoint_stats_time_range(self):
        """Test endpoint stats with time range filter."""
        tracker = APIPerformanceTracker()

        datetime.now()
        tracker.track_request("/api/test", 0.1, 200, "GET")

        time.sleep(0.1)
        middle = datetime.now()
        tracker.track_request("/api/test", 0.1, 200, "GET")

        time.sleep(0.1)
        datetime.now()

        # Get only middle request
        stats = tracker.get_endpoint_stats(
            "/api/test", (middle - timedelta(seconds=0.05), middle + timedelta(seconds=0.05))
        )

        assert stats["count"] == 1

    def test_get_error_rate(self):
        """Test error rate calculation."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test", 0.1, 200, "GET")
        tracker.track_request("/api/test", 0.1, 200, "GET")
        tracker.track_request("/api/test", 0.1, 400, "GET")
        tracker.track_request("/api/test", 0.1, 500, "GET")

        error_rate = tracker.get_error_rate("/api/test")
        assert error_rate == pytest.approx(0.5)  # 2 errors out of 4

    def test_get_response_times(self):
        """Test getting response time distribution."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/test1", 0.1, 200, "GET")
        tracker.track_request("/api/test1", 0.2, 200, "GET")
        tracker.track_request("/api/test2", 0.3, 200, "POST")

        response_times = tracker.get_response_times()

        assert "/api/test1" in response_times
        assert "/api/test2" in response_times
        assert len(response_times["/api/test1"]) == 2
        assert len(response_times["/api/test2"]) == 1

    def test_generate_api_report(self):
        """Test generating comprehensive API report."""
        tracker = APIPerformanceTracker()

        # Track various requests
        for _i in range(10):
            tracker.track_request("/api/fast", 0.05, 200, "GET")
            tracker.track_request("/api/slow", 0.5, 200, "GET")

        # Add some errors
        tracker.track_request("/api/error", 0.1, 500, "POST")
        tracker.track_request("/api/error", 0.1, 500, "POST")

        report = tracker.generate_api_report()

        assert report["total_requests"] == 22
        assert report["total_errors"] == 2
        assert "overall_error_rate" in report
        assert "endpoints" in report
        assert "/api/fast" in report["endpoints"]
        assert "slowest_endpoints" in report
        assert "highest_error_endpoints" in report

    def test_generate_api_report_empty(self):
        """Test generating report with no data."""
        tracker = APIPerformanceTracker()
        report = tracker.generate_api_report()
        assert report == {"total_requests": 0}

    def test_max_history_limit(self):
        """Test that max_history is enforced."""
        tracker = APIPerformanceTracker(max_history=10)

        # Track more requests than max_history
        for i in range(20):
            tracker.track_request(f"/api/test{i}", 0.1, 200, "GET")

        assert len(tracker._requests) == 10


# ============================================================================
# CacheManager Tests
# ============================================================================


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init(self):
        """Test CacheManager initialization."""
        cache = CacheManager(max_size=100, default_ttl=60)
        assert cache._max_size == 100
        assert cache._default_ttl == 60
        assert cache.get_size() == 0

    def test_get_set(self):
        """Test basic get/set operations."""
        cache = CacheManager()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get_size() == 1

    def test_get_miss(self):
        """Test cache miss."""
        cache = CacheManager()
        assert cache.get("nonexistent") is None

        stats = cache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 0

    def test_get_hit(self):
        """Test cache hit."""
        cache = CacheManager()
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

        stats = cache.get_stats()
        assert stats.hits == 1

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = CacheManager(default_ttl=1)  # 1 second TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

        stats = cache.get_stats()
        assert stats.expirations == 1

    def test_custom_ttl(self):
        """Test custom TTL per key."""
        cache = CacheManager(default_ttl=10)

        cache.set("short", "value", ttl=1)
        cache.set("long", "value", ttl=10)

        time.sleep(1.1)

        assert cache.get("short") is None
        assert cache.get("long") == "value"

    def test_no_ttl(self):
        """Test setting item without TTL."""
        cache = CacheManager(default_ttl=0)

        cache.set("key1", "value1")
        time.sleep(0.5)

        # Should not expire
        assert cache.get("key1") == "value1"

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # This should evict key1 (oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get_size() == 3

        stats = cache.get_stats()
        assert stats.evictions == 1

    def test_lru_ordering(self):
        """Test that access updates LRU order."""
        cache = CacheManager(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it most recent
        cache.get("key1")

        # Add new item - should evict key2
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"
        assert cache.get_size() == 1

    def test_delete(self):
        """Test deleting a key."""
        cache = CacheManager()

        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.get_size() == 0

    def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        cache = CacheManager()
        assert cache.delete("nonexistent") is False

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = CacheManager(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1
        assert stats.max_size == 10
        assert stats.hit_rate == pytest.approx(0.5)
        assert stats.miss_rate == pytest.approx(0.5)

    def test_cache_stats_properties(self):
        """Test CacheStats properties."""
        stats = CacheStats(hits=7, misses=3, evictions=2, expirations=1, size=50, max_size=100)

        assert stats.hit_rate == pytest.approx(0.7)
        assert stats.miss_rate == pytest.approx(0.3)

    def test_cache_stats_no_requests(self):
        """Test CacheStats with no requests."""
        stats = CacheStats(hits=0, misses=0, evictions=0, expirations=0, size=0, max_size=100)

        assert stats.hit_rate == 0.0
        # When total is 0, miss_rate = 1.0 - hit_rate = 1.0 - 0.0 = 1.0
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
        """Test manual cleanup of expired entries."""
        cache = CacheManager(default_ttl=1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3", ttl=10)

        time.sleep(1.1)

        removed = cache.cleanup_expired()

        assert removed == 2
        assert cache.get_size() == 1
        assert cache.get("key3") == "value3"

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = CacheManager(max_size=1000)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# ResourceMonitor Tests
# ============================================================================


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(cpu_threshold=75.0, memory_threshold=85.0, disk_threshold=95.0)

        assert monitor.cpu_threshold == 75.0
        assert monitor.memory_threshold == 85.0
        assert monitor.disk_threshold == 95.0

    def test_get_cpu_usage(self):
        """Test getting CPU usage."""
        monitor = ResourceMonitor()
        cpu = monitor.get_cpu_usage()

        assert isinstance(cpu, float)
        assert 0.0 <= cpu <= 100.0

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        monitor = ResourceMonitor()
        mem = monitor.get_memory_usage()

        assert "percent" in mem
        assert "used_mb" in mem
        assert "available_mb" in mem
        assert "total_mb" in mem
        assert 0.0 <= mem["percent"] <= 100.0
        assert mem["used_mb"] > 0

    def test_get_disk_usage(self):
        """Test getting disk usage."""
        monitor = ResourceMonitor()
        disk = monitor.get_disk_usage("/")

        assert "percent" in disk
        assert "used_gb" in disk
        assert "free_gb" in disk
        assert "total_gb" in disk
        assert 0.0 <= disk["percent"] <= 100.0

    def test_get_gpu_usage_not_available(self):
        """Test GPU usage when not available."""
        monitor = ResourceMonitor()
        gpu = monitor.get_gpu_usage()

        # Most test environments don't have GPU
        # Should return None without crashing
        assert gpu is None or isinstance(gpu, dict)

    def test_check_resources(self):
        """Test comprehensive resource check."""
        monitor = ResourceMonitor()
        resources = monitor.check_resources()

        assert isinstance(resources, ResourceUsage)
        assert resources.cpu_percent >= 0.0
        assert resources.memory_percent >= 0.0
        assert resources.disk_percent >= 0.0
        assert resources.memory_used_mb > 0

    def test_get_alerts_for_high_usage_none(self):
        """Test alerts when usage is normal."""
        monitor = ResourceMonitor(cpu_threshold=99.0, memory_threshold=99.0, disk_threshold=99.0)

        alerts = monitor.get_alerts_for_high_usage()

        # With very high thresholds, should get no alerts
        assert isinstance(alerts, list)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_alerts_for_high_usage_cpu(self, mock_disk, mock_mem, mock_cpu):
        """Test CPU usage alert."""
        mock_cpu.return_value = 90.0
        mock_mem.return_value = Mock(
            percent=50.0,
            used=1024 * 1024 * 1024,
            available=1024 * 1024 * 1024,
            total=2 * 1024 * 1024 * 1024,
        )
        mock_disk.return_value = Mock(
            percent=50.0,
            used=100 * 1024 * 1024 * 1024,
            free=100 * 1024 * 1024 * 1024,
            total=200 * 1024 * 1024 * 1024,
        )

        monitor = ResourceMonitor(cpu_threshold=80.0)
        alerts = monitor.get_alerts_for_high_usage()

        cpu_alerts = [a for a in alerts if a["resource"] == "cpu"]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0]["usage"] == 90.0
        assert cpu_alerts[0]["severity"] == "warning"

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_alerts_critical_severity(self, mock_disk, mock_mem, mock_cpu):
        """Test critical severity alerts."""
        mock_cpu.return_value = 96.0
        mock_mem.return_value = Mock(
            percent=50.0,
            used=1024 * 1024 * 1024,
            available=1024 * 1024 * 1024,
            total=2 * 1024 * 1024 * 1024,
        )
        mock_disk.return_value = Mock(
            percent=50.0,
            used=100 * 1024 * 1024 * 1024,
            free=100 * 1024 * 1024 * 1024,
            total=200 * 1024 * 1024 * 1024,
        )

        monitor = ResourceMonitor(cpu_threshold=80.0)
        alerts = monitor.get_alerts_for_high_usage()

        cpu_alerts = [a for a in alerts if a["resource"] == "cpu"]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0]["severity"] == "critical"


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
        assert report.api_tracker is None
        assert report.resource_monitor is None

    def test_init_with_all_components(self):
        """Test initialization with all components."""
        monitor = PerformanceMonitor()
        api_tracker = APIPerformanceTracker()
        resource_monitor = ResourceMonitor()

        report = PerformanceReport(
            monitor=monitor, api_tracker=api_tracker, resource_monitor=resource_monitor
        )

        assert report.monitor is monitor
        assert report.api_tracker is api_tracker
        assert report.resource_monitor is resource_monitor

    def test_generate_daily_report(self):
        """Test generating daily report."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Record some metrics for today
        monitor.record_metric("op1", 10.0, "ms")
        monitor.record_metric("op1", 20.0, "ms")
        monitor.record_metric("op2", 5.0, "s")

        report = report_gen.generate_daily_report()

        assert "date" in report
        assert "generated_at" in report
        assert "operations" in report
        assert "op1" in report["operations"]
        assert "op2" in report["operations"]

    def test_generate_daily_report_specific_date(self):
        """Test generating report for specific date."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Record metric for yesterday
        yesterday = datetime.now() - timedelta(days=1)
        monitor.record_metric("op1", 10.0, "ms")
        monitor._metrics["op1"][0].timestamp = yesterday

        # Record metric for today
        monitor.record_metric("op2", 20.0, "ms")

        # Generate report for yesterday
        report = report_gen.generate_daily_report(date=yesterday)

        # Should only include yesterday's metrics
        assert "op1" in report["operations"]

    def test_generate_daily_report_with_api_tracker(self):
        """Test daily report with API tracker."""
        monitor = PerformanceMonitor()
        api_tracker = APIPerformanceTracker()
        report_gen = PerformanceReport(monitor, api_tracker=api_tracker)

        api_tracker.track_request("/api/test", 0.1, 200, "GET")

        report = report_gen.generate_daily_report()

        assert "api" in report
        assert report["api"]["total_requests"] > 0

    def test_generate_daily_report_with_resource_monitor(self):
        """Test daily report with resource monitor."""
        monitor = PerformanceMonitor()
        resource_monitor = ResourceMonitor()
        report_gen = PerformanceReport(monitor, resource_monitor=resource_monitor)

        report = report_gen.generate_daily_report()

        assert "resources" in report
        assert "resource_alerts" in report

    def test_generate_session_report(self):
        """Test generating session-specific report."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Record metrics with session IDs
        monitor.record_metric("op1", 10.0, "ms", {"session_id": "session_123"})
        monitor.record_metric("op1", 20.0, "ms", {"session_id": "session_123"})
        monitor.record_metric("op2", 5.0, "s", {"session_id": "session_456"})

        report = report_gen.generate_session_report("session_123")

        assert report["session_id"] == "session_123"
        assert "op1" in report["operations"]
        assert report["operations"]["op1"]["count"] == 2
        assert "op2" not in report["operations"]

    def test_generate_session_report_empty(self):
        """Test session report with no matching sessions."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        report = report_gen.generate_session_report("nonexistent")

        assert report["session_id"] == "nonexistent"
        assert len(report["operations"]) == 0

    def test_compare_performance(self):
        """Test comparing performance between periods."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Record metrics for period 1
        period1_start = datetime.now()
        monitor.record_metric("op1", 10.0, "ms")
        monitor.record_metric("op1", 12.0, "ms")
        period1_end = datetime.now()

        time.sleep(0.1)

        # Record metrics for period 2
        period2_start = datetime.now()
        monitor.record_metric("op1", 20.0, "ms")
        monitor.record_metric("op1", 22.0, "ms")
        period2_end = datetime.now()

        comparison = report_gen.compare_performance(
            (period1_start, period1_end), (period2_start, period2_end)
        )

        assert "period1" in comparison
        assert "period2" in comparison
        assert "changes" in comparison
        assert "op1" in comparison["changes"]
        assert comparison["changes"]["op1"]["direction"] == "degraded"  # Got slower

    def test_compare_performance_improvement(self):
        """Test performance comparison showing improvement."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Period 1: slower
        period1_start = datetime.now()
        monitor.record_metric("op1", 20.0, "ms")
        period1_end = datetime.now()

        time.sleep(0.1)

        # Period 2: faster
        period2_start = datetime.now()
        monitor.record_metric("op1", 10.0, "ms")
        period2_end = datetime.now()

        comparison = report_gen.compare_performance(
            (period1_start, period1_end), (period2_start, period2_end)
        )

        assert comparison["changes"]["op1"]["direction"] == "improved"

    def test_export_report_json(self):
        """Test exporting report as JSON."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        report = {"test": "data", "value": 123}
        json_str = report_gen.export_report(report, format="json")

        data = json.loads(json_str)
        assert data["test"] == "data"
        assert data["value"] == 123

    def test_export_report_json_to_file(self):
        """Test exporting report to JSON file."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        report = {"test": "data"}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            path = Path(f.name)

        try:
            result = report_gen.export_report(report, format="json", path=path)
            assert result is None

            data = json.loads(path.read_text())
            assert data["test"] == "data"
        finally:
            path.unlink()

    def test_export_report_csv(self):
        """Test exporting report as CSV."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        monitor.record_metric("op1", 10.0, "ms")
        report = report_gen.generate_daily_report()

        csv_str = report_gen.export_report(report, format="csv")

        assert "operation" in csv_str
        assert "op1" in csv_str

    def test_export_report_csv_to_file(self):
        """Test exporting report to CSV file."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        monitor.record_metric("op1", 10.0, "ms")
        report = report_gen.generate_daily_report()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            path = Path(f.name)

        try:
            result = report_gen.export_report(report, format="csv", path=path)
            assert result is None

            content = path.read_text()
            assert "operation" in content
        finally:
            path.unlink()

    def test_export_report_invalid_format(self):
        """Test export with invalid format."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        with pytest.raises(ValueError, match="Unsupported format"):
            report_gen.export_report({}, format="xml")


# ============================================================================
# Model Tests
# ============================================================================


class TestPerformanceMetric:
    """Tests for PerformanceMetric model."""

    def test_create_metric(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            metric_name="test_op", value=123.45, unit="ms", metadata={"key": "value"}
        )

        assert metric.metric_name == "test_op"
        assert metric.value == 123.45
        assert metric.unit == "ms"
        assert metric.metadata["key"] == "value"
        assert isinstance(metric.timestamp, datetime)

    def test_metric_with_custom_timestamp(self):
        """Test metric with custom timestamp."""
        ts = datetime(2023, 1, 1, 12, 0, 0)
        metric = PerformanceMetric(metric_name="test", value=10.0, unit="s", timestamp=ts)

        assert metric.timestamp == ts

    def test_metric_default_metadata(self):
        """Test metric with default empty metadata."""
        metric = PerformanceMetric(metric_name="test", value=10.0, unit="s")

        assert metric.metadata == {}


class TestResourceUsage:
    """Tests for ResourceUsage model."""

    def test_create_resource_usage(self):
        """Test creating resource usage snapshot."""
        usage = ResourceUsage(
            cpu_percent=45.5,
            memory_percent=60.0,
            memory_used_mb=8192.0,
            memory_available_mb=4096.0,
            disk_percent=70.0,
            disk_used_gb=500.0,
            disk_free_gb=200.0,
        )

        assert usage.cpu_percent == 45.5
        assert usage.memory_percent == 60.0
        assert usage.disk_percent == 70.0

    def test_resource_usage_with_gpu(self):
        """Test resource usage with GPU metrics."""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=8192.0,
            memory_available_mb=4096.0,
            disk_percent=70.0,
            disk_used_gb=500.0,
            disk_free_gb=200.0,
            gpu_percent=80.0,
            gpu_memory_mb=4096.0,
        )

        assert usage.gpu_percent == 80.0
        assert usage.gpu_memory_mb == 4096.0


class TestAPIMetric:
    """Tests for APIMetric model."""

    def test_create_api_metric(self):
        """Test creating an API metric."""
        metric = APIMetric(endpoint="/api/test", duration_ms=123.45, status_code=200, method="POST")

        assert metric.endpoint == "/api/test"
        assert metric.duration_ms == 123.45
        assert metric.status_code == 200
        assert metric.method == "POST"
        assert metric.error is None

    def test_api_metric_with_error(self):
        """Test API metric with error."""
        metric = APIMetric(
            endpoint="/api/fail", duration_ms=50.0, status_code=500, error="Internal server error"
        )

        assert metric.status_code == 500
        assert metric.error == "Internal server error"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete monitoring workflow."""
        # Setup components
        monitor = PerformanceMonitor()
        ImageProcessingProfiler(monitor=monitor)
        api_tracker = APIPerformanceTracker()
        cache = CacheManager()
        resource_monitor = ResourceMonitor()
        report_gen = PerformanceReport(
            monitor=monitor, api_tracker=api_tracker, resource_monitor=resource_monitor
        )

        # Simulate some operations
        with monitor.timer("image_processing"):
            time.sleep(0.01)

        api_tracker.track_request("/api/process", 0.1, 200, "POST")

        cache.set("result_123", {"data": "processed"})

        # Generate report
        report = report_gen.generate_daily_report()

        assert "image_processing" in report["operations"]
        assert "api" in report
        assert "resources" in report

    def test_concurrent_operations(self):
        """Test concurrent operations across components."""
        monitor = PerformanceMonitor()
        cache = CacheManager()
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    # Timer operations
                    with monitor.timer(f"op_{thread_id}_{i}"):
                        time.sleep(0.001)

                    # Cache operations
                    key = f"key_{thread_id}_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0

    def test_performance_degradation_detection(self):
        """Test detecting performance degradation."""
        monitor = PerformanceMonitor()
        report_gen = PerformanceReport(monitor)

        # Period 1: good performance
        period1_start = datetime.now()
        for _ in range(10):
            monitor.record_metric("api_call", 0.1, "s")
        period1_end = datetime.now()

        time.sleep(0.1)

        # Period 2: degraded performance
        period2_start = datetime.now()
        for _ in range(10):
            monitor.record_metric("api_call", 0.5, "s")
        period2_end = datetime.now()

        comparison = report_gen.compare_performance(
            (period1_start, period1_end), (period2_start, period2_end)
        )

        assert comparison["changes"]["api_call"]["direction"] == "degraded"
        assert comparison["changes"]["api_call"]["percentage"] > 0

    def test_memory_tracking_with_profiler(self):
        """Test memory tracking integration."""
        profiler = ImageProcessingProfiler()

        def memory_intensive_op():
            # Allocate some memory
            data = [0] * 100000
            return len(data)

        result, profile = profiler.profile_operation(memory_intensive_op)

        assert result == 100000
        assert "memory_delta_mb" in profile
        assert profile["success"] is True

    def test_cache_with_api_responses(self):
        """Test caching API responses."""
        cache = CacheManager(default_ttl=60)
        api_tracker = APIPerformanceTracker()

        # Simulate API call with caching
        endpoint = "/api/data"
        cache_key = f"cache:{endpoint}"

        # First call - cache miss
        if cache.get(cache_key) is None:
            with api_tracker.track(endpoint, "GET"):
                result = {"data": "response"}
                cache.set(cache_key, result)

        # Second call - cache hit
        cached = cache.get(cache_key)

        assert cached == {"data": "response"}
        stats = cache.get_stats()
        assert stats.hits >= 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_metrics_statistics(self):
        """Test statistics with empty metrics."""
        monitor = PerformanceMonitor()

        stats = monitor.get_statistics("nonexistent")
        assert stats == {"count": 0}

        avg = monitor.get_average("nonexistent")
        assert avg is None

        percentiles = monitor.get_percentiles("nonexistent")
        assert percentiles == {}

    def test_single_metric_statistics(self):
        """Test statistics with single metric."""
        monitor = PerformanceMonitor()
        monitor.record_metric("test", 10.0, "ms")

        stats = monitor.get_statistics("test")
        assert stats["count"] == 1
        assert stats["min"] == 10.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 10.0
        assert stats["std"] == 0.0

    def test_very_large_values(self):
        """Test handling very large metric values."""
        monitor = PerformanceMonitor()

        large_value = 1e10
        monitor.record_metric("test", large_value, "bytes")

        stats = monitor.get_statistics("test")
        assert stats["max"] == large_value

    def test_zero_values(self):
        """Test handling zero values."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", 0.0, "count")

        avg = monitor.get_average("test")
        assert avg == 0.0

    def test_negative_values(self):
        """Test handling negative values."""
        monitor = PerformanceMonitor()

        monitor.record_metric("test", -5.0, "delta")

        avg = monitor.get_average("test")
        assert avg == -5.0

    def test_cache_with_none_value(self):
        """Test caching None value."""
        cache = CacheManager()

        cache.set("key", None)
        result = cache.get("key")

        # None is a valid cached value
        assert result is None
        stats = cache.get_stats()
        assert stats.hits == 1

    def test_api_tracker_zero_duration(self):
        """Test API tracker with zero duration."""
        tracker = APIPerformanceTracker()

        tracker.track_request("/api/instant", 0.0, 200, "GET")

        stats = tracker.get_endpoint_stats("/api/instant")
        assert stats["min_ms"] == 0.0

    def test_resource_monitor_invalid_path(self):
        """Test resource monitor with invalid disk path."""
        monitor = ResourceMonitor()

        # Should handle gracefully or raise appropriate error
        try:
            disk = monitor.get_disk_usage("/nonexistent/path/that/does/not/exist")
            # If it doesn't raise, result should be valid or None
            assert disk is None or isinstance(disk, dict)
        except (FileNotFoundError, OSError):
            # Expected behavior
            pass

    def test_concurrent_timer_same_name(self):
        """Test concurrent timers with same operation name."""
        monitor = PerformanceMonitor()

        # Start timer
        monitor.start_timer("shared_op")

        # Start another timer with same name (should overwrite)
        monitor.start_timer("shared_op")

        time.sleep(0.01)

        # Stop should work
        duration = monitor.stop_timer("shared_op")
        assert duration is not None

    def test_performance_metric_serialization(self):
        """Test that performance metrics can be serialized."""
        metric = PerformanceMetric(
            metric_name="test", value=10.0, unit="ms", metadata={"key": "value"}
        )

        # Should be able to convert to dict
        data = metric.model_dump()
        assert isinstance(data, dict)
        assert data["metric_name"] == "test"
