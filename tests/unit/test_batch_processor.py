"""
Tests for batch processor module.

Tests batch processing of multiple images with shared settings.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.batch.processor import (
    BatchProcessor,
    BatchSettings,
    BatchJob,
    BatchResult,
    JobStatus,
)
from ptpd_calibration.imaging import ImageFormat
from ptpd_calibration.imaging.processor import ColorMode


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """All status values should exist."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestBatchSettings:
    """Tests for BatchSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = BatchSettings()
        assert settings.curve is None
        assert settings.invert is True
        assert settings.color_mode == ColorMode.GRAYSCALE
        assert settings.export_format == ImageFormat.TIFF
        assert settings.jpeg_quality == 95
        assert settings.max_workers == 4
        assert settings.continue_on_error is True

    def test_custom_settings(self):
        """Custom settings should be applied."""
        curve = CurveData(
            name="Test",
            input_values=[0, 1],
            output_values=[0, 1],
        )
        settings = BatchSettings(
            curve=curve,
            invert=False,
            color_mode=ColorMode.RGB,
            export_format=ImageFormat.PNG,
            max_workers=2,
        )
        assert settings.curve == curve
        assert settings.invert is False
        assert settings.color_mode == ColorMode.RGB
        assert settings.export_format == ImageFormat.PNG
        assert settings.max_workers == 2


class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_default_job(self):
        """Default job should have pending status."""
        job = BatchJob()
        assert job.status == JobStatus.PENDING
        assert job.error_message is None
        assert job.started_at is None
        assert job.completed_at is None

    def test_job_with_path(self):
        """Job should store input path."""
        job = BatchJob(input_path=Path("/test/image.png"))
        assert job.input_path == Path("/test/image.png")

    def test_duration_none_if_not_complete(self):
        """Duration should be None if not completed."""
        job = BatchJob()
        assert job.duration_seconds is None


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_result(self):
        """Empty result should have zero counts."""
        result = BatchResult()
        assert result.total_jobs == 0
        assert result.completed == 0
        assert result.failed == 0
        assert result.cancelled == 0
        assert result.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Success rate should be calculated correctly."""
        result = BatchResult(total_jobs=10, completed=7, failed=3)
        assert result.success_rate == 70.0

    def test_get_failed_jobs(self):
        """Should return only failed jobs."""
        jobs = [
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.FAILED),
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.FAILED),
        ]
        result = BatchResult(jobs=jobs)
        failed = result.get_failed_jobs()
        assert len(failed) == 2
        assert all(j.status == JobStatus.FAILED for j in failed)

    def test_get_completed_jobs(self):
        """Should return only completed jobs."""
        jobs = [
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.FAILED),
            BatchJob(status=JobStatus.COMPLETED),
        ]
        result = BatchResult(jobs=jobs)
        completed = result.get_completed_jobs()
        assert len(completed) == 2
        assert all(j.status == JobStatus.COMPLETED for j in completed)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = BatchResult(total_jobs=5, completed=4, failed=1)
        d = result.to_dict()
        assert d["total_jobs"] == 5
        assert d["completed"] == 4
        assert d["failed"] == 1
        assert "success_rate" in d


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create batch processor."""
        return BatchProcessor()

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample test images."""
        images = []
        for i in range(3):
            arr = np.ones((50, 50), dtype=np.uint8) * (50 + i * 50)
            img = Image.fromarray(arr, mode="L")
            path = tmp_path / f"input_{i}.png"
            img.save(path)
            images.append(path)
        return images

    @pytest.fixture
    def linear_curve(self):
        """Create linear curve."""
        return CurveData(
            name="Linear",
            input_values=[0, 0.5, 1],
            output_values=[0, 0.5, 1],
        )

    def test_processor_creation(self):
        """Processor should be created with default settings."""
        processor = BatchProcessor()
        assert processor.settings is not None
        assert processor.settings.max_workers == 4

    def test_processor_with_custom_settings(self, linear_curve):
        """Processor should accept custom settings."""
        settings = BatchSettings(
            curve=linear_curve,
            max_workers=2,
        )
        processor = BatchProcessor(settings=settings)
        assert processor.settings.curve == linear_curve
        assert processor.settings.max_workers == 2

    def test_process_batch_sequential(self, sample_images, tmp_path):
        """Process batch with single worker."""
        settings = BatchSettings(
            max_workers=1,
            invert=True,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        result = processor.process_batch(sample_images, output_dir)

        assert result.total_jobs == 3
        assert result.completed == 3
        assert result.failed == 0
        assert output_dir.exists()

    def test_process_batch_parallel(self, sample_images, tmp_path):
        """Process batch with multiple workers."""
        settings = BatchSettings(
            max_workers=2,
            invert=True,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        result = processor.process_batch(sample_images, output_dir)

        assert result.total_jobs == 3
        assert result.completed == 3

    def test_process_batch_with_curve(self, sample_images, tmp_path, linear_curve):
        """Process batch with curve applied."""
        settings = BatchSettings(
            curve=linear_curve,
            invert=True,
            max_workers=1,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        result = processor.process_batch(sample_images, output_dir)

        assert result.completed == 3
        # Check output files exist
        assert len(list(output_dir.glob("*.tiff"))) == 3

    def test_process_batch_creates_output_dir(self, sample_images, tmp_path):
        """Batch processing should create output directory."""
        settings = BatchSettings(max_workers=1)
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "new_output_dir"

        assert not output_dir.exists()
        processor.process_batch(sample_images, output_dir)
        assert output_dir.exists()

    def test_process_batch_progress_callback(self, sample_images, tmp_path):
        """Progress callback should be called."""
        settings = BatchSettings(max_workers=1)
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        progress_calls = []

        def callback(current, total, message):
            progress_calls.append((current, total, message))

        processor.process_batch(sample_images, output_dir, progress_callback=callback)

        assert len(progress_calls) > 0
        # First call should be (0, 3, "Starting...")
        assert progress_calls[0][1] == 3

    def test_process_batch_handles_missing_file(self, sample_images, tmp_path):
        """Should handle missing input files gracefully."""
        settings = BatchSettings(
            max_workers=1,
            continue_on_error=True,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        # Add a non-existent file
        paths = sample_images + [tmp_path / "nonexistent.png"]

        result = processor.process_batch(paths, output_dir)

        assert result.total_jobs == 4
        assert result.completed == 3
        assert result.failed == 1

    def test_process_batch_different_formats(self, sample_images, tmp_path):
        """Test different output formats."""
        for format_type in [ImageFormat.PNG, ImageFormat.TIFF, ImageFormat.JPEG]:
            settings = BatchSettings(
                max_workers=1,
                export_format=format_type,
            )
            processor = BatchProcessor(settings=settings)
            output_dir = tmp_path / f"output_{format_type.value}"

            result = processor.process_batch(sample_images, output_dir)
            assert result.completed == 3


class TestBatchOutputNaming:
    """Tests for output file naming."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create single sample image."""
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        img = Image.fromarray(arr, mode="L")
        path = tmp_path / "my_image.png"
        img.save(path)
        return [path]

    def test_default_output_naming(self, sample_image, tmp_path):
        """Default naming should add suffix."""
        settings = BatchSettings(
            max_workers=1,
            output_suffix="_negative",
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        processor.process_batch(sample_image, output_dir)

        output_files = list(output_dir.glob("*"))
        assert len(output_files) == 1
        assert "_negative" in output_files[0].stem

    def test_custom_output_suffix(self, sample_image, tmp_path):
        """Custom suffix should be used."""
        settings = BatchSettings(
            max_workers=1,
            output_suffix="_processed",
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        processor.process_batch(sample_image, output_dir)

        output_files = list(output_dir.glob("*"))
        assert "_processed" in output_files[0].stem


class TestBatchColorModes:
    """Tests for color mode handling in batch processing."""

    @pytest.fixture
    def rgb_images(self, tmp_path):
        """Create RGB test images."""
        images = []
        for i in range(2):
            arr = np.ones((50, 50, 3), dtype=np.uint8) * 128
            arr[:, :, 0] = 100 + i * 50  # Vary red
            img = Image.fromarray(arr, mode="RGB")
            path = tmp_path / f"rgb_{i}.png"
            img.save(path)
            images.append(path)
        return images

    def test_batch_grayscale_mode(self, rgb_images, tmp_path):
        """RGB images should convert to grayscale."""
        settings = BatchSettings(
            max_workers=1,
            color_mode=ColorMode.GRAYSCALE,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        result = processor.process_batch(rgb_images, output_dir)

        assert result.completed == 2

    def test_batch_preserve_mode(self, rgb_images, tmp_path):
        """RGB images should preserve mode when requested."""
        settings = BatchSettings(
            max_workers=1,
            color_mode=ColorMode.PRESERVE,
        )
        processor = BatchProcessor(settings=settings)
        output_dir = tmp_path / "output"

        result = processor.process_batch(rgb_images, output_dir)

        assert result.completed == 2
