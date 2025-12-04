"""
Database Performance Tests.

Benchmark tests for database operations.
"""

import pytest

try:
    import pytest_benchmark

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.benchmark
class TestDatabasePerformance:
    """Benchmark tests for database operations."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_add_record(self, benchmark):
        """Benchmark adding records to database."""
        from ptpd_calibration.ml.database import CalibrationDatabase
        from ptpd_calibration.core.models import CalibrationRecord
        from ptpd_calibration.core.types import (
            ChemistryType,
            ContrastAgent,
            DeveloperType,
        )

        db = CalibrationDatabase()

        def add_record():
            record = CalibrationRecord(
                paper_type="Benchmark Paper",
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
                measured_densities=[0.1 + i * 0.1 for i in range(21)],
            )
            db.add_record(record)
            return record

        result = benchmark(add_record)
        assert result is not None

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_query_by_paper(self, benchmark, populated_database):
        """Benchmark querying by paper type."""
        result = benchmark(
            populated_database.query, paper_type="Arches Platine"
        )
        assert len(result) > 0

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_query_all(self, benchmark, populated_database):
        """Benchmark querying all records."""
        result = benchmark(populated_database.query)
        assert len(result) > 0

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_get_statistics(self, benchmark, populated_database):
        """Benchmark getting database statistics."""
        result = benchmark(populated_database.get_statistics)
        assert result is not None


@pytest.mark.performance
@pytest.mark.benchmark
class TestLargeDatabasePerformance:
    """Benchmark tests for large database operations."""

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_query_large_database(self, benchmark, large_database):
        """Benchmark querying a large database."""
        result = benchmark(large_database.query)
        assert len(result) >= 1000

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    def test_query_with_filter_large(self, benchmark, large_database):
        """Benchmark filtered query on large database."""
        result = benchmark(large_database.query, paper_type="Paper_5")
        assert len(result) > 0


@pytest.mark.performance
class TestDatabasePerformanceThresholds:
    """Test that database operations meet performance thresholds."""

    def test_add_record_under_10ms(self):
        """Adding a record should complete in under 10ms."""
        import time

        from ptpd_calibration.ml.database import CalibrationDatabase
        from ptpd_calibration.core.models import CalibrationRecord
        from ptpd_calibration.core.types import (
            ChemistryType,
            ContrastAgent,
            DeveloperType,
        )

        db = CalibrationDatabase()

        times = []
        for i in range(100):
            record = CalibrationRecord(
                paper_type=f"Paper_{i}",
                exposure_time=180.0,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
                measured_densities=[0.1 + j * 0.1 for j in range(21)],
            )

            start = time.perf_counter()
            db.add_record(record)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.01, f"Average add time {avg_time:.3f}s (>10ms)"

    def test_query_under_100ms(self, populated_database):
        """Querying should complete in under 100ms."""
        import time

        start = time.perf_counter()
        for _ in range(10):
            populated_database.query(paper_type="Arches Platine")
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.1, f"Query took {elapsed:.3f}s (>100ms)"
