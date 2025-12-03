#!/usr/bin/env python3
"""
Example usage of the performance monitoring module.

This script demonstrates all features of the monitoring system:
- Performance monitoring and timing
- Image processing profiling
- API performance tracking
- Caching
- Resource monitoring
- Report generation
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptpd_calibration.monitoring import (
    PerformanceMonitor,
    ImageProcessingProfiler,
    APIPerformanceTracker,
    CacheManager,
    ResourceMonitor,
    PerformanceReport,
    get_monitor,
    get_profiler,
    get_api_tracker,
    get_cache,
    get_resource_monitor,
)


def demo_performance_monitor():
    """Demonstrate PerformanceMonitor functionality."""
    print("\n" + "=" * 70)
    print("PERFORMANCE MONITOR DEMO")
    print("=" * 70)

    monitor = PerformanceMonitor()

    # Example 1: Manual timer
    print("\n1. Manual timer example:")
    monitor.start_timer("database_query")
    time.sleep(0.1)  # Simulate work
    duration = monitor.stop_timer("database_query")
    print(f"   Database query took: {duration:.4f}s")

    # Example 2: Context manager
    print("\n2. Context manager example:")
    for i in range(3):
        with monitor.timer("api_call"):
            time.sleep(0.05)  # Simulate API call

    # Example 3: Get statistics
    print("\n3. Statistics:")
    stats = monitor.get_statistics("api_call")
    print(f"   API calls: {stats['count']}")
    print(f"   Mean: {stats['mean']:.4f}s")
    print(f"   Min: {stats['min']:.4f}s")
    print(f"   Max: {stats['max']:.4f}s")
    print(f"   P50: {stats.get('p50', 0):.4f}s")
    print(f"   P90: {stats.get('p90', 0):.4f}s")
    print(f"   P99: {stats.get('p99', 0):.4f}s")

    # Example 4: Record custom metrics
    print("\n4. Custom metrics:")
    monitor.record_metric("user_count", 1250, "users")
    monitor.record_metric("image_size", 2048 * 1536, "pixels")
    monitor.record_metric("conversion_rate", 0.042, "percentage")

    # Example 5: Export metrics
    print("\n5. Export metrics:")
    json_export = monitor.export_metrics(format="json", operation_name="api_call")
    print(f"   Exported {len(json_export)} bytes of JSON data")


def demo_image_processing_profiler():
    """Demonstrate ImageProcessingProfiler functionality."""
    print("\n" + "=" * 70)
    print("IMAGE PROCESSING PROFILER DEMO")
    print("=" * 70)

    profiler = ImageProcessingProfiler()

    # Example function to profile
    def process_image(width, height, complexity=1.0):
        """Simulate image processing."""
        iterations = int(width * height * complexity / 1000)
        total = 0
        for i in range(iterations):
            total += i ** 0.5
        return total

    # Example 1: Profile operation
    print("\n1. Profile single operation:")
    result, profile = profiler.profile_operation(process_image, 1024, 768, complexity=1.0)
    print(f"   Operation: {profile['operation']}")
    print(f"   Wall time: {profile['wall_time_s']:.4f}s")
    print(f"   CPU time: {profile['cpu_time_s']:.4f}s")
    print(f"   Memory delta: {profile['memory_delta_mb']:.2f}MB")

    # Example 2: Profile multiple operations
    print("\n2. Profile batch operations:")
    for i in range(3):
        profiler.profile_operation(process_image, 800, 600, complexity=0.8)

    # Example 3: Get processing speed
    print("\n3. Processing speed:")
    speed = profiler.get_processing_speed((800, 600), "process_image")
    if speed:
        print(f"   Processing speed: {speed:,.0f} pixels/second")
        print(f"   That's {speed / 1_000_000:.2f} megapixels/second")

    # Example 4: Estimate batch time
    print("\n4. Estimate batch processing time:")
    batch = [(1024, 768), (800, 600), (1920, 1080), (640, 480)]
    estimated_time = profiler.estimate_batch_time(batch, "process_image")
    if estimated_time:
        print(f"   Estimated time for {len(batch)} images: {estimated_time:.2f}s")

    # Example 5: Memory stats
    print("\n5. Current memory statistics:")
    mem_stats = profiler.get_memory_stats()
    print(f"   RSS: {mem_stats['rss_mb']:.1f}MB")
    print(f"   VMS: {mem_stats['vms_mb']:.1f}MB")
    print(f"   Process memory: {mem_stats['percent']:.1f}%")
    print(f"   Available: {mem_stats['available_mb']:.1f}MB")

    # Example 6: Identify bottlenecks
    print("\n6. Identify bottlenecks:")
    # Create some varied performance data
    for i in range(5):
        profiler.profile_operation(process_image, 800, 600, complexity=1.0)
    for i in range(2):
        profiler.profile_operation(process_image, 800, 600, complexity=5.0)  # Slower

    bottlenecks = profiler.identify_bottlenecks(["process_image"])
    if bottlenecks:
        for b in bottlenecks:
            print(f"   Operation: {b['operation']}")
            print(f"   Slowdown factor: {b['slowdown_factor']:.2f}x")
            print(f"   Median: {b['median_time']:.4f}s")
            print(f"   P90: {b['p90_time']:.4f}s")


def demo_api_performance_tracker():
    """Demonstrate APIPerformanceTracker functionality."""
    print("\n" + "=" * 70)
    print("API PERFORMANCE TRACKER DEMO")
    print("=" * 70)

    tracker = APIPerformanceTracker()

    # Example 1: Track requests manually
    print("\n1. Track API requests:")
    for i in range(5):
        tracker.track_request("/api/process", 0.05 + i * 0.01, 200, "POST")
    tracker.track_request("/api/process", 0.15, 500, "POST", error="Timeout")
    tracker.track_request("/api/status", 0.002, 200, "GET")

    # Example 2: Context manager
    print("\n2. Track with context manager:")
    with tracker.track("/api/upload", "POST"):
        time.sleep(0.03)

    # Example 3: Get endpoint stats
    print("\n3. Endpoint statistics:")
    stats = tracker.get_endpoint_stats("/api/process")
    print(f"   Endpoint: /api/process")
    print(f"   Total requests: {stats['count']}")
    print(f"   Error rate: {stats['error_rate']:.1%}")
    print(f"   Mean response: {stats['mean_ms']:.1f}ms")
    print(f"   Median response: {stats['median_ms']:.1f}ms")
    print(f"   P90: {stats['p90_ms']:.1f}ms")
    print(f"   P95: {stats['p95_ms']:.1f}ms")
    print(f"   P99: {stats['p99_ms']:.1f}ms")

    # Example 4: Generate API report
    print("\n4. API Performance Report:")
    report = tracker.generate_api_report()
    print(f"   Total requests: {report['total_requests']}")
    print(f"   Total errors: {report['total_errors']}")
    print(f"   Overall error rate: {report['overall_error_rate']:.1%}")
    print(f"   Overall mean: {report['overall_mean_ms']:.1f}ms")
    print(f"   Overall P95: {report['overall_p95_ms']:.1f}ms")

    if report['slowest_endpoints']:
        print("\n   Slowest endpoints:")
        for ep in report['slowest_endpoints'][:3]:
            print(f"      {ep['endpoint']}: {ep['p95_ms']:.1f}ms (P95)")


def demo_cache_manager():
    """Demonstrate CacheManager functionality."""
    print("\n" + "=" * 70)
    print("CACHE MANAGER DEMO")
    print("=" * 70)

    cache = CacheManager(max_size=100, default_ttl=60)

    # Example 1: Basic get/set
    print("\n1. Basic cache operations:")
    cache.set("user:123", {"name": "John", "email": "john@example.com"})
    user = cache.get("user:123")
    print(f"   Cached user: {user}")

    # Example 2: TTL expiration
    print("\n2. TTL expiration:")
    cache.set("temp_key", "temp_value", ttl=1)
    print(f"   Immediately: {cache.get('temp_key')}")
    time.sleep(1.1)
    print(f"   After 1s: {cache.get('temp_key')}")

    # Example 3: Cache stats
    print("\n3. Cache statistics:")
    for i in range(10):
        cache.set(f"key{i}", f"value{i}")
    for i in range(5):
        cache.get(f"key{i}")  # Hits
    cache.get("nonexistent")  # Miss

    stats = cache.get_stats()
    print(f"   Hits: {stats.hits}")
    print(f"   Misses: {stats.misses}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    print(f"   Cache size: {stats.size}/{stats.max_size}")

    # Example 4: Cleanup expired
    print("\n4. Cleanup expired entries:")
    cache.set("expire1", "val1", ttl=1)
    cache.set("expire2", "val2", ttl=1)
    time.sleep(1.1)
    removed = cache.cleanup_expired()
    print(f"   Removed {removed} expired entries")


def demo_resource_monitor():
    """Demonstrate ResourceMonitor functionality."""
    print("\n" + "=" * 70)
    print("RESOURCE MONITOR DEMO")
    print("=" * 70)

    monitor = ResourceMonitor(cpu_threshold=80, memory_threshold=80, disk_threshold=90)

    # Example 1: CPU usage
    print("\n1. CPU usage:")
    cpu = monitor.get_cpu_usage()
    print(f"   CPU: {cpu:.1f}%")

    # Example 2: Memory usage
    print("\n2. Memory usage:")
    mem = monitor.get_memory_usage()
    print(f"   Memory: {mem['percent']:.1f}%")
    print(f"   Used: {mem['used_mb']:.1f}MB")
    print(f"   Available: {mem['available_mb']:.1f}MB")
    print(f"   Total: {mem['total_mb']:.1f}MB")

    # Example 3: Disk usage
    print("\n3. Disk usage:")
    disk = monitor.get_disk_usage("/")
    print(f"   Disk: {disk['percent']:.1f}%")
    print(f"   Used: {disk['used_gb']:.1f}GB")
    print(f"   Free: {disk['free_gb']:.1f}GB")
    print(f"   Total: {disk['total_gb']:.1f}GB")

    # Example 4: GPU usage (if available)
    print("\n4. GPU usage:")
    gpu = monitor.get_gpu_usage()
    if gpu:
        print(f"   GPU: {gpu['gpu_percent']:.1f}%")
        print(f"   GPU memory: {gpu['memory_percent']:.1f}%")
        print(f"   GPU memory used: {gpu['memory_used_mb']:.1f}MB")
    else:
        print("   GPU monitoring not available")

    # Example 5: Comprehensive check
    print("\n5. Comprehensive resource check:")
    resources = monitor.check_resources()
    print(f"   CPU: {resources.cpu_percent:.1f}%")
    print(f"   Memory: {resources.memory_percent:.1f}%")
    print(f"   Disk: {resources.disk_percent:.1f}%")

    # Example 6: Alerts
    print("\n6. Resource alerts:")
    alerts = monitor.get_alerts_for_high_usage()
    if alerts:
        for alert in alerts:
            print(f"   [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("   No resource alerts")


def demo_performance_report():
    """Demonstrate PerformanceReport functionality."""
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT DEMO")
    print("=" * 70)

    # Create some test data
    monitor = PerformanceMonitor()
    api_tracker = APIPerformanceTracker()

    # Generate some metrics
    for i in range(10):
        with monitor.timer("operation_a"):
            time.sleep(0.01)
        with monitor.timer("operation_b"):
            time.sleep(0.02)
        api_tracker.track_request("/api/test", 0.05, 200)

    # Create report generator
    report_gen = PerformanceReport(
        monitor=monitor, api_tracker=api_tracker, resource_monitor=ResourceMonitor()
    )

    # Example 1: Daily report
    print("\n1. Daily report:")
    daily_report = report_gen.generate_daily_report()
    print(f"   Date: {daily_report['date']}")
    print(f"   Operations tracked: {len(daily_report['operations'])}")
    for op_name, stats in daily_report['operations'].items():
        if stats['count'] > 0:
            print(f"   - {op_name}: {stats['count']} calls, avg {stats['mean']:.4f}s")

    # Example 2: Session report
    print("\n2. Session report:")
    monitor.record_metric(
        "session_metric", 1.5, "seconds", metadata={"session_id": "sess123"}
    )
    session_report = report_gen.generate_session_report("sess123")
    print(f"   Session: {session_report['session_id']}")
    print(f"   Operations: {len(session_report['operations'])}")

    # Example 3: Compare periods
    print("\n3. Compare performance periods:")
    now = datetime.now()
    period1 = (now - timedelta(hours=2), now - timedelta(hours=1))
    period2 = (now - timedelta(hours=1), now)

    # Add more metrics for comparison
    for i in range(5):
        monitor.record_metric("test_op", 0.5, "seconds")
    time.sleep(0.1)
    for i in range(5):
        monitor.record_metric("test_op", 0.6, "seconds")

    comparison = report_gen.compare_performance(period1, period2)
    print(f"   Periods compared: 2")
    print(f"   Operations analyzed: {len(comparison['changes'])}")

    # Example 4: Export report
    print("\n4. Export report:")
    json_report = report_gen.export_report(daily_report, format="json")
    print(f"   JSON export size: {len(json_report)} bytes")


def demo_global_instances():
    """Demonstrate using global singleton instances."""
    print("\n" + "=" * 70)
    print("GLOBAL INSTANCES DEMO")
    print("=" * 70)

    # Get global instances
    monitor = get_monitor()
    profiler = get_profiler()
    api_tracker = get_api_tracker()
    cache = get_cache()
    resource_monitor = get_resource_monitor()

    print("\n✓ All global instances created successfully")
    print(f"  - PerformanceMonitor: {type(monitor).__name__}")
    print(f"  - ImageProcessingProfiler: {type(profiler).__name__}")
    print(f"  - APIPerformanceTracker: {type(api_tracker).__name__}")
    print(f"  - CacheManager: {type(cache).__name__}")
    print(f"  - ResourceMonitor: {type(resource_monitor).__name__}")

    # Verify singleton behavior
    monitor2 = get_monitor()
    assert monitor is monitor2, "Global instances should be singletons"
    print("\n✓ Singleton behavior verified")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PTPD CALIBRATION - PERFORMANCE MONITORING MODULE")
    print("Comprehensive Feature Demonstration")
    print("=" * 70)

    try:
        demo_performance_monitor()
        demo_image_processing_profiler()
        demo_api_performance_tracker()
        demo_cache_manager()
        demo_resource_monitor()
        demo_performance_report()
        demo_global_instances()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
