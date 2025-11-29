"""
Batch processor for processing multiple images with the same curve and settings.

Provides efficient batch workflows for digital negative creation,
enabling processing of entire editions or test strips in one operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union
from uuid import UUID, uuid4
import concurrent.futures
import threading

from PIL import Image

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.imaging import ImageProcessor, ImageFormat, ExportSettings
from ptpd_calibration.imaging.processor import ColorMode


class JobStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchSettings:
    """Settings for batch processing."""

    # Curve application
    curve: Optional[CurveData] = None
    invert: bool = True
    color_mode: ColorMode = ColorMode.GRAYSCALE

    # Export settings
    export_format: ImageFormat = ImageFormat.TIFF
    jpeg_quality: int = 95
    preserve_resolution: bool = True

    # Processing options
    max_workers: int = 4
    continue_on_error: bool = True

    # Output naming
    output_suffix: str = "_negative"
    preserve_original_name: bool = True


@dataclass
class BatchJob:
    """A single job in the batch queue."""

    id: UUID = field(default_factory=uuid4)
    input_path: Path = field(default_factory=Path)
    output_path: Optional[Path] = None
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class BatchResult:
    """Result of batch processing operation."""

    total_jobs: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    jobs: list[BatchJob] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed / self.total_jobs) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get total processing duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_failed_jobs(self) -> list[BatchJob]:
        """Get list of failed jobs."""
        return [j for j in self.jobs if j.status == JobStatus.FAILED]

    def get_completed_jobs(self) -> list[BatchJob]:
        """Get list of completed jobs."""
        return [j for j in self.jobs if j.status == JobStatus.COMPLETED]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_jobs": self.total_jobs,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration_seconds": self.duration_seconds,
            "jobs": [
                {
                    "input": str(j.input_path),
                    "output": str(j.output_path) if j.output_path else None,
                    "status": j.status.value,
                    "error": j.error_message,
                    "duration": j.duration_seconds,
                }
                for j in self.jobs
            ],
        }


class BatchProcessor:
    """Process multiple images with the same curve and settings.

    Supports:
    - Parallel processing with configurable workers
    - Progress callbacks for UI integration
    - Error handling with continue-on-error option
    - Cancellation support
    """

    def __init__(self, settings: Optional[BatchSettings] = None):
        """Initialize batch processor.

        Args:
            settings: Batch processing settings. If None, uses defaults.
        """
        self.settings = settings or BatchSettings()
        self._processor = ImageProcessor()
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    def process_batch(
        self,
        input_paths: list[Union[str, Path]],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """Process a batch of images.

        Args:
            input_paths: List of input image paths
            output_dir: Directory for output files
            progress_callback: Optional callback(current, total, message)

        Returns:
            BatchResult with processing summary
        """
        self._cancelled.clear()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create jobs
        jobs = []
        for path in input_paths:
            path = Path(path)
            output_name = self._generate_output_name(path)
            output_path = output_dir / output_name

            jobs.append(BatchJob(
                input_path=path,
                output_path=output_path,
            ))

        result = BatchResult(
            total_jobs=len(jobs),
            jobs=jobs,
            started_at=datetime.now(),
        )

        if progress_callback:
            progress_callback(0, len(jobs), "Starting batch processing...")

        # Process jobs
        if self.settings.max_workers > 1:
            self._process_parallel(jobs, progress_callback)
        else:
            self._process_sequential(jobs, progress_callback)

        # Compile results
        result.completed_at = datetime.now()
        result.completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        result.failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
        result.cancelled = sum(1 for j in jobs if j.status == JobStatus.CANCELLED)

        if progress_callback:
            progress_callback(
                len(jobs), len(jobs),
                f"Completed: {result.completed} successful, {result.failed} failed"
            )

        return result

    def cancel(self) -> None:
        """Cancel ongoing batch processing."""
        self._cancelled.set()

    def _process_sequential(
        self,
        jobs: list[BatchJob],
        progress_callback: Optional[Callable],
    ) -> None:
        """Process jobs sequentially."""
        for i, job in enumerate(jobs):
            if self._cancelled.is_set():
                job.status = JobStatus.CANCELLED
                continue

            self._process_single_job(job)

            if progress_callback:
                progress_callback(i + 1, len(jobs), f"Processed: {job.input_path.name}")

    def _process_parallel(
        self,
        jobs: list[BatchJob],
        progress_callback: Optional[Callable],
    ) -> None:
        """Process jobs in parallel."""
        completed_count = 0

        def process_and_update(job: BatchJob) -> BatchJob:
            nonlocal completed_count

            if self._cancelled.is_set():
                job.status = JobStatus.CANCELLED
                return job

            self._process_single_job(job)

            with self._lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(
                        completed_count, len(jobs),
                        f"Processed: {job.input_path.name}"
                    )

            return job

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.settings.max_workers
        ) as executor:
            futures = [executor.submit(process_and_update, job) for job in jobs]

            for future in concurrent.futures.as_completed(futures):
                if self._cancelled.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    future.result()
                except Exception:
                    pass  # Errors handled in process_and_update

    def _process_single_job(self, job: BatchJob) -> None:
        """Process a single job."""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()

        try:
            # Create digital negative
            result = self._processor.create_digital_negative(
                job.input_path,
                curve=self.settings.curve,
                invert=self.settings.invert,
                color_mode=self.settings.color_mode,
            )

            # Export
            export_settings = ExportSettings(
                format=self.settings.export_format,
                jpeg_quality=self.settings.jpeg_quality,
                preserve_resolution=self.settings.preserve_resolution,
            )

            self._processor.export(result, job.output_path, export_settings)

            job.status = JobStatus.COMPLETED

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)

            if not self.settings.continue_on_error:
                raise

        finally:
            job.completed_at = datetime.now()

    def _generate_output_name(self, input_path: Path) -> str:
        """Generate output filename."""
        ext_map = {
            ImageFormat.TIFF: ".tiff",
            ImageFormat.TIFF_16BIT: ".tiff",
            ImageFormat.PNG: ".png",
            ImageFormat.PNG_16BIT: ".png",
            ImageFormat.JPEG: ".jpg",
            ImageFormat.JPEG_HIGH: ".jpg",
            ImageFormat.ORIGINAL: input_path.suffix,
        }

        ext = ext_map.get(self.settings.export_format, ".tiff")

        if self.settings.preserve_original_name:
            return f"{input_path.stem}{self.settings.output_suffix}{ext}"
        else:
            return f"{uuid4().hex[:8]}{ext}"

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported input formats."""
        return ImageProcessor.get_supported_formats()
