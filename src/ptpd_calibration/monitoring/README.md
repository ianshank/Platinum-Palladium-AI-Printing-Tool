# Performance Monitoring Module

Comprehensive performance tracking, profiling, and resource monitoring for the PTPD Calibration System.

## Overview

This module provides a complete suite of performance monitoring tools:

- **PerformanceMonitor**: Core timing and metrics tracking
- **ImageProcessingProfiler**: Image processing-specific profiling
- **APIPerformanceTracker**: API endpoint performance tracking
- **CacheManager**: LRU cache with TTL support
- **ResourceMonitor**: System resource monitoring (CPU, memory, disk, GPU)
- **PerformanceReport**: Report generation and analysis

## Quick Start

```python
from ptpd_calibration.monitoring import (
    get_monitor,
    get_profiler,
    get_api_tracker,
    get_cache,
    get_resource_monitor,
)

# Time an operation
monitor = get_monitor()
with monitor.timer("my_operation"):
    do_something()

# Get statistics
stats = monitor.get_statistics("my_operation")
print(f"Average: {stats['mean']:.3f}s")
print(f"P95: {stats['p95']:.3f}s")

# Profile image processing
profiler = get_profiler()
result, profile = profiler.profile_operation(process_image, image)
print(f"Processing time: {profile['wall_time_s']:.3f}s")
print(f"Memory used: {profile['memory_delta_mb']:.1f}MB")

# Track API requests
api_tracker = get_api_tracker()
with api_tracker.track("/api/process", "POST"):
    handle_request()

# Use cache
cache = get_cache()
value = cache.get("expensive_result")
if value is None:
    value = expensive_computation()
    cache.set("expensive_result", value, ttl=300)

# Monitor resources
resource_monitor = get_resource_monitor()
resources = resource_monitor.check_resources()
alerts = resource_monitor.get_alerts_for_high_usage()
```

## Features

### 1. PerformanceMonitor

Track operation timings and custom metrics.

#### Methods

- `start_timer(operation_name)` - Start timing an operation
- `stop_timer(operation_name)` - Stop timer and record duration
- `timer(operation_name)` - Context manager for timing
- `record_metric(name, value, unit, metadata)` - Record arbitrary metric
- `get_metrics(operation_name, time_range)` - Get metrics for operation
- `get_average(operation_name)` - Get average time
- `get_percentiles(operation_name, percentiles)` - Get p50, p90, p95, p99
- `get_statistics(operation_name)` - Get comprehensive statistics
- `export_metrics(format, output_path, operation_name)` - Export to JSON/CSV
- `clear_metrics(operation_name)` - Clear metrics

#### Examples

```python
from ptpd_calibration.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(max_history=10000)

# Manual timing
monitor.start_timer("database_query")
result = db.query("SELECT * FROM users")
duration = monitor.stop_timer("database_query")

# Context manager (recommended)
with monitor.timer("api_call"):
    response = requests.get("https://api.example.com")

# Record custom metrics
monitor.record_metric("user_count", 1250, "users")
monitor.record_metric("cache_hit_rate", 0.85, "percentage")

# Get statistics
stats = monitor.get_statistics("api_call")
print(f"Min: {stats['min']:.3f}s")
print(f"Max: {stats['max']:.3f}s")
print(f"Mean: {stats['mean']:.3f}s")
print(f"Median: {stats['median']:.3f}s")
print(f"P90: {stats['p90']:.3f}s")
print(f"P95: {stats['p95']:.3f}s")
print(f"P99: {stats['p99']:.3f}s")

# Export to JSON
monitor.export_metrics(format="json", output_path=Path("metrics.json"))

# Export to CSV
monitor.export_metrics(format="csv", output_path=Path("metrics.csv"))
```

### 2. ImageProcessingProfiler

Specialized profiling for image processing operations.

#### Methods

- `profile_operation(func, *args, **kwargs)` - Profile any operation
- `get_processing_speed(image_size, operation_name)` - Get pixels/second
- `estimate_batch_time(images, operation_name)` - Estimate batch time
- `track_memory_usage(operation)` - Track memory consumption
- `get_memory_stats()` - Current memory statistics
- `identify_bottlenecks(operations, threshold_percentile)` - Find slow operations

#### Examples

```python
from ptpd_calibration.monitoring import ImageProcessingProfiler

profiler = ImageProcessingProfiler()

# Profile an operation
def process_image(image):
    # Your image processing code
    return processed_image

result, profile = profiler.profile_operation(process_image, my_image)
print(f"Wall time: {profile['wall_time_s']:.3f}s")
print(f"CPU time: {profile['cpu_time_s']:.3f}s")
print(f"Memory delta: {profile['memory_delta_mb']:.1f}MB")

# Get processing speed
speed = profiler.get_processing_speed((1920, 1080), "process_image")
print(f"Processing speed: {speed:,.0f} pixels/second")

# Estimate batch processing time
batch = [(1920, 1080), (1024, 768), (800, 600)]
estimated_time = profiler.estimate_batch_time(batch, "process_image")
print(f"Estimated time for {len(batch)} images: {estimated_time:.2f}s")

# Get memory stats
mem_stats = profiler.get_memory_stats()
print(f"RSS: {mem_stats['rss_mb']:.1f}MB")
print(f"Available: {mem_stats['available_mb']:.1f}MB")

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(
    ["detect_tablet", "extract_densities", "generate_curve"]
)
for b in bottlenecks:
    print(f"{b['operation']}: {b['slowdown_factor']:.2f}x slower")
```

### 3. APIPerformanceTracker

Track API endpoint performance and error rates.

#### Methods

- `track_request(endpoint, duration, status, method, error)` - Track request
- `track(endpoint, method)` - Context manager for tracking
- `get_endpoint_stats(endpoint, time_range)` - Get stats for endpoint
- `get_error_rate(endpoint, time_range)` - Calculate error rate
- `get_response_times()` - Get response time distribution
- `generate_api_report()` - Generate performance report

#### Examples

```python
from ptpd_calibration.monitoring import APIPerformanceTracker

tracker = APIPerformanceTracker(max_history=10000)

# Manual tracking
tracker.track_request("/api/process", duration=0.125, status=200, method="POST")
tracker.track_request("/api/status", duration=0.005, status=200, method="GET")

# Context manager (recommended)
with tracker.track("/api/upload", "POST"):
    handle_upload()

# Get endpoint statistics
stats = tracker.get_endpoint_stats("/api/process")
print(f"Requests: {stats['count']}")
print(f"Error rate: {stats['error_rate']:.2%}")
print(f"Mean response: {stats['mean_ms']:.1f}ms")
print(f"P95: {stats['p95_ms']:.1f}ms")

# Generate API report
report = tracker.generate_api_report()
print(f"Total requests: {report['total_requests']}")
print(f"Overall error rate: {report['overall_error_rate']:.2%}")
print(f"Overall P95: {report['overall_p95_ms']:.1f}ms")

# Slowest endpoints
for ep in report['slowest_endpoints']:
    print(f"{ep['endpoint']}: {ep['p95_ms']:.1f}ms")
```

### 4. CacheManager

Simple dict-based LRU cache with TTL support.

#### Methods

- `get(key)` - Get cached value
- `set(key, value, ttl)` - Set value with TTL
- `delete(key)` - Delete key
- `get_stats()` - Get cache statistics
- `get_size()` - Current cache size
- `clear()` - Clear all entries
- `cleanup_expired()` - Remove expired entries

#### Examples

```python
from ptpd_calibration.monitoring import CacheManager

cache = CacheManager(max_size=1000, default_ttl=3600)

# Basic usage
cache.set("user:123", {"name": "John", "email": "john@example.com"})
user = cache.get("user:123")

# Custom TTL (5 minutes)
cache.set("session:abc", session_data, ttl=300)

# Cache pattern for expensive operations
def get_user(user_id):
    cache_key = f"user:{user_id}"
    user = cache.get(cache_key)
    if user is None:
        user = database.fetch_user(user_id)
        cache.set(cache_key, user, ttl=600)
    return user

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Miss rate: {stats.miss_rate:.2%}")
print(f"Size: {stats.size}/{stats.max_size}")
print(f"Evictions: {stats.evictions}")

# Cleanup expired entries
removed = cache.cleanup_expired()
print(f"Removed {removed} expired entries")
```

### 5. ResourceMonitor

Monitor system resources and alert on high usage.

#### Methods

- `get_cpu_usage()` - Get CPU usage percentage
- `get_memory_usage()` - Get memory usage
- `get_disk_usage(path)` - Get disk usage
- `get_gpu_usage()` - Get GPU usage (if available)
- `check_resources()` - Comprehensive resource check
- `get_alerts_for_high_usage()` - Alert if resources exceed thresholds

#### Examples

```python
from ptpd_calibration.monitoring import ResourceMonitor

monitor = ResourceMonitor(
    cpu_threshold=80.0,
    memory_threshold=80.0,
    disk_threshold=90.0
)

# Get CPU usage
cpu = monitor.get_cpu_usage()
print(f"CPU: {cpu:.1f}%")

# Get memory usage
mem = monitor.get_memory_usage()
print(f"Memory: {mem['percent']:.1f}%")
print(f"Available: {mem['available_mb']:.1f}MB")

# Get disk usage
disk = monitor.get_disk_usage("/")
print(f"Disk: {disk['percent']:.1f}%")
print(f"Free: {disk['free_gb']:.1f}GB")

# Check GPU (if available)
gpu = monitor.get_gpu_usage()
if gpu:
    print(f"GPU: {gpu['gpu_percent']:.1f}%")
    print(f"GPU memory: {gpu['memory_percent']:.1f}%")

# Comprehensive check
resources = monitor.check_resources()
print(f"CPU: {resources.cpu_percent:.1f}%")
print(f"Memory: {resources.memory_percent:.1f}%")
print(f"Disk: {resources.disk_percent:.1f}%")

# Get alerts
alerts = monitor.get_alerts_for_high_usage()
for alert in alerts:
    print(f"[{alert['severity'].upper()}] {alert['message']}")
```

### 6. PerformanceReport

Generate performance reports and summaries.

#### Methods

- `generate_daily_report(date)` - Daily summary
- `generate_session_report(session_id)` - Session summary
- `compare_performance(period1, period2)` - Compare time periods
- `export_report(report, format, path)` - Export report

#### Examples

```python
from ptpd_calibration.monitoring import (
    PerformanceMonitor,
    APIPerformanceTracker,
    ResourceMonitor,
    PerformanceReport,
)
from datetime import datetime, timedelta

monitor = PerformanceMonitor()
api_tracker = APIPerformanceTracker()
resource_monitor = ResourceMonitor()

report_gen = PerformanceReport(
    monitor=monitor,
    api_tracker=api_tracker,
    resource_monitor=resource_monitor
)

# Generate daily report
daily_report = report_gen.generate_daily_report()
print(f"Date: {daily_report['date']}")
print(f"Operations: {len(daily_report['operations'])}")

# Generate session report
session_report = report_gen.generate_session_report("session_123")
print(f"Session: {session_report['session_id']}")

# Compare performance periods
now = datetime.now()
period1 = (now - timedelta(days=7), now - timedelta(days=6))
period2 = (now - timedelta(days=1), now)
comparison = report_gen.compare_performance(period1, period2)

for op_name, change in comparison['changes'].items():
    print(f"{op_name}: {change['percentage']:.1f}% {change['direction']}")

# Export report
report_gen.export_report(daily_report, format="json", path=Path("report.json"))
```

## Configuration

All thresholds and limits can be configured:

```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Configure via environment variables
# PTPD_MONITORING_MAX_HISTORY=5000
# PTPD_MONITORING_CPU_THRESHOLD=75.0
# PTPD_MONITORING_MEMORY_THRESHOLD=85.0
# PTPD_MONITORING_CACHE_SIZE=2000

# Or programmatically
monitor = PerformanceMonitor(max_history=5000)
cache = CacheManager(max_size=2000, default_ttl=1800)
resource_monitor = ResourceMonitor(
    cpu_threshold=75.0,
    memory_threshold=85.0,
    disk_threshold=95.0
)
```

## Global Instances

For convenience, the module provides global singleton instances:

```python
from ptpd_calibration.monitoring import (
    get_monitor,
    get_profiler,
    get_api_tracker,
    get_cache,
    get_resource_monitor,
)

# All calls return the same instance
monitor1 = get_monitor()
monitor2 = get_monitor()
assert monitor1 is monitor2  # True
```

## Thread Safety

All components are thread-safe and use appropriate locking mechanisms:

- `PerformanceMonitor`: Uses `threading.RLock`
- `ImageProcessingProfiler`: Uses `threading.RLock`
- `APIPerformanceTracker`: Uses `threading.RLock`
- `CacheManager`: Uses `threading.RLock`
- `ResourceMonitor`: Uses `threading.RLock`

## Best Practices

1. **Use context managers**: Prefer `with monitor.timer()` over manual start/stop
2. **Set appropriate TTL**: Cache expensive operations with reasonable TTL values
3. **Monitor regularly**: Check resource alerts periodically
4. **Export metrics**: Regularly export metrics for long-term analysis
5. **Clean up cache**: Run `cleanup_expired()` periodically in production
6. **Use global instances**: Use `get_*()` functions for singleton behavior
7. **Profile selectively**: Don't profile every operation in production

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from ptpd_calibration.monitoring import get_api_tracker
import time

app = FastAPI()
tracker = get_api_tracker()

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    tracker.track_request(
        endpoint=request.url.path,
        duration=duration,
        status=response.status_code,
        method=request.method
    )

    return response
```

### Background Monitoring

```python
import threading
from ptpd_calibration.monitoring import get_resource_monitor
import time

def background_resource_monitor():
    monitor = get_resource_monitor()
    while True:
        alerts = monitor.get_alerts_for_high_usage()
        for alert in alerts:
            logger.warning(f"Resource alert: {alert['message']}")
        time.sleep(60)  # Check every minute

# Start background thread
thread = threading.Thread(target=background_resource_monitor, daemon=True)
thread.start()
```

### Cache Cleanup Task

```python
import threading
from ptpd_calibration.monitoring import get_cache
import time

def cleanup_cache_periodically():
    cache = get_cache()
    while True:
        time.sleep(300)  # Every 5 minutes
        removed = cache.cleanup_expired()
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache entries")

# Start background thread
thread = threading.Thread(target=cleanup_cache_periodically, daemon=True)
thread.start()
```

## See Also

- [Performance Monitoring Example](../../../examples/performance_monitoring_example.py)
- [PTPD Configuration](../config.py)
- [API Documentation](../api/)
