# Performance Monitoring - Quick Reference

## Import

```python
from ptpd_calibration.monitoring import (
    get_monitor,
    get_profiler,
    get_api_tracker,
    get_cache,
    get_resource_monitor,
    PerformanceReport,
)
```

## Core Classes

### PerformanceMonitor

```python
monitor = get_monitor()

# Time operations
with monitor.timer("operation"):
    do_work()

# Get stats
stats = monitor.get_statistics("operation")
# → {count, min, max, mean, median, std, p50, p90, p95, p99}

# Export
monitor.export_metrics(format="json", output_path=Path("metrics.json"))
```

### ImageProcessingProfiler

```python
profiler = get_profiler()

# Profile operation
result, profile = profiler.profile_operation(func, *args)
# → profile: {wall_time_s, cpu_time_s, memory_delta_mb}

# Processing speed
speed = profiler.get_processing_speed((width, height), "operation")
# → pixels/second

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(["op1", "op2"])
# → [{operation, slowdown_factor, median_time, p90_time}]
```

### APIPerformanceTracker

```python
tracker = get_api_tracker()

# Track requests
with tracker.track("/api/endpoint", "POST"):
    handle_request()

# Get stats
stats = tracker.get_endpoint_stats("/api/endpoint")
# → {count, error_rate, mean_ms, p90_ms, p95_ms, p99_ms}

# Generate report
report = tracker.generate_api_report()
# → {total_requests, overall_error_rate, endpoints, slowest_endpoints}
```

### CacheManager

```python
cache = get_cache()

# Get/Set
value = cache.get("key")
cache.set("key", value, ttl=300)  # TTL in seconds

# Stats
stats = cache.get_stats()
# → {hits, misses, hit_rate, miss_rate, size, evictions}

# Cleanup
removed = cache.cleanup_expired()
```

### ResourceMonitor

```python
monitor = get_resource_monitor()

# Check resources
resources = monitor.check_resources()
# → ResourceUsage(cpu_percent, memory_percent, disk_percent, ...)

# Get alerts
alerts = monitor.get_alerts_for_high_usage()
# → [{resource, usage, threshold, severity, message}]

# Individual checks
cpu = monitor.get_cpu_usage()
mem = monitor.get_memory_usage()
disk = monitor.get_disk_usage("/")
gpu = monitor.get_gpu_usage()  # If available
```

### PerformanceReport

```python
report_gen = PerformanceReport(
    monitor=get_monitor(),
    api_tracker=get_api_tracker(),
    resource_monitor=get_resource_monitor()
)

# Daily report
report = report_gen.generate_daily_report()

# Session report
report = report_gen.generate_session_report("session_id")

# Compare periods
comparison = report_gen.compare_performance(period1, period2)

# Export
report_gen.export_report(report, format="json", path=Path("report.json"))
```

## Common Patterns

### Time an operation

```python
from ptpd_calibration.monitoring import get_monitor

monitor = get_monitor()
with monitor.timer("my_operation"):
    result = expensive_operation()
```

### Cache expensive results

```python
from ptpd_calibration.monitoring import get_cache

cache = get_cache()
cache_key = f"result:{param}"
result = cache.get(cache_key)
if result is None:
    result = expensive_computation(param)
    cache.set(cache_key, result, ttl=600)
```

### Track API performance

```python
from ptpd_calibration.monitoring import get_api_tracker

tracker = get_api_tracker()
with tracker.track(request.url.path, request.method):
    response = await process_request(request)
```

### Monitor resources

```python
from ptpd_calibration.monitoring import get_resource_monitor

monitor = get_resource_monitor()
alerts = monitor.get_alerts_for_high_usage()
for alert in alerts:
    logger.warning(f"Resource alert: {alert['message']}")
```

### Profile image processing

```python
from ptpd_calibration.monitoring import get_profiler

profiler = get_profiler()
result, profile = profiler.profile_operation(process_image, image)
logger.info(f"Processed in {profile['wall_time_s']:.2f}s, "
           f"used {profile['memory_delta_mb']:.1f}MB")
```

## Data Models

### PerformanceMetric
- `metric_name: str`
- `value: float`
- `unit: str`
- `timestamp: datetime`
- `metadata: dict`

### ResourceUsage
- `cpu_percent: float`
- `memory_percent: float`
- `memory_used_mb: float`
- `disk_percent: float`
- `gpu_percent: Optional[float]`

### APIMetric
- `endpoint: str`
- `duration_ms: float`
- `status_code: int`
- `method: str`
- `error: Optional[str]`

### CacheStats
- `hits: int`
- `misses: int`
- `hit_rate: float` (property)
- `miss_rate: float` (property)
- `size: int`
- `evictions: int`

## Configuration

Configure via environment variables or settings:

```bash
# Environment variables (with PTPD_ prefix)
export PTPD_MONITORING_MAX_HISTORY=5000
export PTPD_MONITORING_CPU_THRESHOLD=75.0
export PTPD_MONITORING_MEMORY_THRESHOLD=85.0
export PTPD_MONITORING_CACHE_SIZE=2000
```

Or programmatically:

```python
monitor = PerformanceMonitor(max_history=5000)
cache = CacheManager(max_size=2000, default_ttl=1800)
resource_monitor = ResourceMonitor(
    cpu_threshold=75.0,
    memory_threshold=85.0,
    disk_threshold=95.0
)
```
