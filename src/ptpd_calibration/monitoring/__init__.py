"""
Performance monitoring and profiling for PTPD Calibration System.

This module provides comprehensive performance tracking capabilities including:
- Operation timing and metrics collection
- Image processing profiling
- API performance tracking
- LRU caching with TTL
- System resource monitoring
- Performance reporting

Example usage:
    from ptpd_calibration.monitoring import get_monitor, get_profiler

    # Time an operation
    monitor = get_monitor()
    with monitor.timer("my_operation"):
        do_something()

    # Profile image processing
    profiler = get_profiler()
    result, profile = profiler.profile_operation(process_image, image)

    # Track API request
    from ptpd_calibration.monitoring import get_api_tracker
    tracker = get_api_tracker()
    with tracker.track("/api/process", "POST"):
        handle_request()

    # Use cache
    from ptpd_calibration.monitoring import get_cache
    cache = get_cache()
    value = cache.get("key")
    if value is None:
        value = expensive_computation()
        cache.set("key", value, ttl=300)
"""

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

__all__ = [
    # Models
    "PerformanceMetric",
    "ResourceUsage",
    "APIMetric",
    "CacheStats",
    # Core classes
    "PerformanceMonitor",
    "ImageProcessingProfiler",
    "APIPerformanceTracker",
    "CacheManager",
    "ResourceMonitor",
    "PerformanceReport",
    # Global instance getters
    "get_monitor",
    "get_profiler",
    "get_api_tracker",
    "get_cache",
    "get_resource_monitor",
]
