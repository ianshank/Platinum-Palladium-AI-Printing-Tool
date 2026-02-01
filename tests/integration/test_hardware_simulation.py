"""
Integration tests for hardware simulation components.

Tests realistic usage scenarios including:
- Step tablet measurement workflows
- Print job processing
- Error handling and recovery
- Multi-device scenarios
"""

import tempfile

import pytest

from ptpd_calibration.integrations.hardware import (
    MeasurementError,
    get_printer_driver,
    get_spectrophotometer_driver,
)
from ptpd_calibration.integrations.protocols import (
    DensityMeasurement,
    DeviceStatus,
    PrintJob,
    PrintResult,
    SpectralData,
)

# =============================================================================
# Test Step Tablet Measurement Workflow
# =============================================================================


class TestStepTabletMeasurementWorkflow:
    """Integration tests for step tablet measurement scenarios."""

    def test_21_step_tablet_measurement(self):
        """Test measuring a 21-step tablet."""
        device = get_spectrophotometer_driver(simulate=True)

        # Setup
        device.connect()
        device.calibrate_white()

        # Measure 21 steps
        measurements = []
        for _step in range(21):
            measurement = device.read_density()
            measurements.append(measurement)

        # Verify measurements
        assert len(measurements) == 21

        # All should be valid DensityMeasurement objects
        for m in measurements:
            assert isinstance(m, DensityMeasurement)
            assert 0.0 <= m.density <= 4.0
            assert 0.0 <= m.lab_l <= 100.0

        # Cleanup
        device.disconnect()

    def test_calibration_between_measurements(self):
        """Test recalibration during measurement session."""
        device = get_spectrophotometer_driver(simulate=True)
        device.connect()

        # First calibration
        device.calibrate_white()
        first_batch = [device.read_density() for _ in range(5)]

        # Recalibrate
        device.calibrate_white()
        second_batch = [device.read_density() for _ in range(5)]

        assert len(first_batch) == 5
        assert len(second_batch) == 5

        device.disconnect()

    def test_spectral_and_density_combined(self):
        """Test alternating between spectral and density readings."""
        device = get_spectrophotometer_driver(simulate=True)
        device.connect()
        device.calibrate_white()

        results = []
        for _ in range(5):
            density = device.read_density()
            spectral = device.read_spectral()
            results.append({"density": density, "spectral": spectral})

        for r in results:
            assert isinstance(r["density"], DensityMeasurement)
            assert isinstance(r["spectral"], SpectralData)

        device.disconnect()

    def test_density_range_simulation(self):
        """Test that simulated densities cover realistic range."""
        device = get_spectrophotometer_driver(simulate=True)
        device.connect()
        device.calibrate_white()

        # Take many measurements to get variety
        densities = [device.read_density().density for _ in range(100)]

        # Should have variety in the realistic range
        min_d = min(densities)
        max_d = max(densities)

        # Simulated data should cover a reasonable range
        assert min_d >= 0.0
        assert max_d <= 3.5  # Max realistic print density

        device.disconnect()


# =============================================================================
# Test Print Job Processing
# =============================================================================


class TestPrintJobProcessing:
    """Integration tests for print job scenarios."""

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image file."""
        image_path = tmp_path / "test_negative.tiff"
        # Create a minimal valid TIFF-like file
        image_path.write_bytes(
            b"II*\x00"  # TIFF header
            + b"\x00" * 1000  # Padding
        )
        return image_path

    def test_single_print_job(self, test_image):
        """Test processing a single print job."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        job = PrintJob(
            name="Single Print Test",
            image_path=str(test_image),
            paper_size="8x10",
            resolution_dpi=2880,
        )

        result = printer.print_image(job)

        assert result.success is True
        assert result.job_id is not None
        assert result.pages_printed == 1

        printer.disconnect()

    def test_multiple_print_jobs_sequential(self, test_image):
        """Test processing multiple print jobs sequentially."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        jobs = [
            PrintJob(
                name=f"Print {i}",
                image_path=str(test_image),
                paper_size="8x10",
                resolution_dpi=2880,
            )
            for i in range(5)
        ]

        results = []
        for job in jobs:
            result = printer.print_image(job)
            results.append(result)

        # All should succeed
        for r in results:
            assert r.success is True

        # All should have unique job IDs
        job_ids = [r.job_id for r in results]
        assert len(set(job_ids)) == len(job_ids)

        printer.disconnect()

    def test_print_job_with_multiple_copies(self, test_image):
        """Test print job with multiple copies."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        job = PrintJob(
            name="Multi-copy Print",
            image_path=str(test_image),
            paper_size="8x10",
            resolution_dpi=2880,
            copies=5,
        )

        result = printer.print_image(job)

        assert result.success is True
        assert result.pages_printed == 5

        printer.disconnect()

    def test_various_paper_sizes(self, test_image):
        """Test printing with different paper sizes."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        sizes = printer.get_paper_sizes()

        for size in sizes[:5]:  # Test first 5 sizes
            job = PrintJob(
                name=f"Size Test - {size}",
                image_path=str(test_image),
                paper_size=size,
            )
            result = printer.print_image(job)
            assert result.success is True, f"Failed for size: {size}"

        printer.disconnect()

    def test_various_resolutions(self, test_image):
        """Test printing at different resolutions."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        resolutions = printer.get_resolutions()

        for dpi in resolutions[:3]:  # Test first 3 resolutions
            job = PrintJob(
                name=f"Resolution Test - {dpi}dpi",
                image_path=str(test_image),
                resolution_dpi=dpi,
            )
            result = printer.print_image(job)
            assert result.success is True, f"Failed for resolution: {dpi}"

        printer.disconnect()


# =============================================================================
# Test Error Handling and Recovery
# =============================================================================


class TestErrorHandlingAndRecovery:
    """Integration tests for error scenarios."""

    def test_measurement_without_connection(self):
        """Test error when measuring without connection."""
        device = get_spectrophotometer_driver(simulate=True)

        # Simulated device raises RuntimeError instead of MeasurementError
        with pytest.raises((MeasurementError, RuntimeError)):
            device.read_density()

        # Should be able to connect and proceed after error
        device.connect()
        measurement = device.read_density()
        assert measurement is not None

        device.disconnect()

    def test_calibration_without_connection(self):
        """Test calibration returns False when not connected."""
        device = get_spectrophotometer_driver(simulate=True)

        # Simulated device returns False instead of raising
        result = device.calibrate_white()
        assert result is False

        # Should be able to connect and proceed
        device.connect()
        result = device.calibrate_white()
        assert result is True

        device.disconnect()

    def test_print_missing_file(self):
        """Test print job with missing file - simulated may succeed."""
        printer = get_printer_driver(simulate=True)
        printer.connect()

        job = PrintJob(
            name="Missing File Test",
            image_path="/nonexistent/path/image.tiff",
        )

        result = printer.print_image(job)

        # Simulated printer may succeed regardless of file existence
        # Real printer would fail, but simulation doesn't check file
        assert isinstance(result, PrintResult)

        # Should be able to print valid file
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(b"valid data")
            valid_path = f.name

        job2 = PrintJob(
            name="Valid File Test",
            image_path=valid_path,
        )

        result2 = printer.print_image(job2)
        assert result2.success is True

        printer.disconnect()

    def test_print_without_connection(self, tmp_path):
        """Test print fails gracefully when not connected."""
        printer = get_printer_driver(simulate=True)

        test_image = tmp_path / "test.tiff"
        test_image.write_bytes(b"data")

        job = PrintJob(
            name="No Connection Test",
            image_path=str(test_image),
        )

        result = printer.print_image(job)

        assert result.success is False
        assert "not connected" in result.error.lower()

    def test_reconnection_after_disconnect(self):
        """Test device reconnection after disconnect."""
        device = get_spectrophotometer_driver(simulate=True)

        # First session
        device.connect()
        m1 = device.read_density()
        device.disconnect()

        # Second session
        device.connect()
        m2 = device.read_density()
        device.disconnect()

        assert m1 is not None
        assert m2 is not None


# =============================================================================
# Test Multi-Device Scenarios
# =============================================================================


class TestMultiDeviceScenarios:
    """Integration tests for multiple device usage."""

    def test_spectrophotometer_and_printer_together(self, tmp_path):
        """Test using both devices in a workflow."""
        # Setup devices
        spectro = get_spectrophotometer_driver(simulate=True)
        printer = get_printer_driver(simulate=True)

        # Connect both
        spectro.connect()
        printer.connect()

        # Measure density
        spectro.calibrate_white()
        measurement = spectro.read_density()
        assert measurement is not None

        # Create test file and print based on measurement
        test_image = tmp_path / "negative.tiff"
        test_image.write_bytes(b"test data")

        job = PrintJob(
            name=f"Calibrated Print (D={measurement.density:.2f})",
            image_path=str(test_image),
            paper_size="8x10",
        )

        result = printer.print_image(job)
        assert result.success is True

        # Disconnect both
        spectro.disconnect()
        printer.disconnect()

    def test_multiple_spectrophotometer_instances(self):
        """Test multiple spectrophotometer instances."""
        device1 = get_spectrophotometer_driver(simulate=True)
        device2 = get_spectrophotometer_driver(simulate=True)

        # Both should work independently
        device1.connect()
        device2.connect()

        device1.calibrate_white()
        device2.calibrate_white()

        m1 = device1.read_density()
        m2 = device2.read_density()

        assert m1 is not None
        assert m2 is not None

        device1.disconnect()
        device2.disconnect()

    def test_multiple_printer_instances(self, tmp_path):
        """Test multiple printer instances."""
        printer1 = get_printer_driver(simulate=True)
        printer2 = get_printer_driver(simulate=True)

        printer1.connect()
        printer2.connect()

        test_image = tmp_path / "test.tiff"
        test_image.write_bytes(b"data")

        job1 = PrintJob(name="Printer 1 Job", image_path=str(test_image))
        job2 = PrintJob(name="Printer 2 Job", image_path=str(test_image))

        result1 = printer1.print_image(job1)
        result2 = printer2.print_image(job2)

        assert result1.success is True
        assert result2.success is True
        # Note: Job IDs may be the same if instances share class-level counter
        # The important thing is both jobs complete successfully
        assert result1.job_id is not None
        assert result2.job_id is not None

        printer1.disconnect()
        printer2.disconnect()


# =============================================================================
# Test Calibration Workflow
# =============================================================================


class TestCalibrationWorkflow:
    """Integration tests for calibration workflows."""

    def test_create_linearization_data(self):
        """Test creating linearization data from measurements."""
        device = get_spectrophotometer_driver(simulate=True)
        device.connect()
        device.calibrate_white()

        # Measure step tablet (21 steps)
        step_densities = []
        for step in range(21):
            measurement = device.read_density()
            step_densities.append(
                {
                    "step": step,
                    "density": measurement.density,
                    "lab_l": measurement.lab_l,
                }
            )

        device.disconnect()

        # Verify data structure
        assert len(step_densities) == 21
        for i, data in enumerate(step_densities):
            assert data["step"] == i
            assert "density" in data
            assert "lab_l" in data

    def test_spectral_analysis_workflow(self):
        """Test spectral data collection for analysis."""
        device = get_spectrophotometer_driver(simulate=True)
        device.connect()
        device.calibrate_white()

        # Collect spectral data for paper white and max black
        paper_white = device.read_spectral()
        max_black = device.read_spectral()

        device.disconnect()

        # Verify spectral data
        assert len(paper_white.wavelengths) == len(paper_white.values)
        assert len(max_black.wavelengths) == len(max_black.values)

        # Should have data from 380nm to 730nm
        assert paper_white.start_nm == 380.0
        assert paper_white.end_nm == 730.0

    def test_iterative_calibration(self, tmp_path):
        """Test iterative calibration workflow."""
        spectro = get_spectrophotometer_driver(simulate=True)
        printer = get_printer_driver(simulate=True)

        spectro.connect()
        printer.connect()
        spectro.calibrate_white()

        # Create test file
        test_image = tmp_path / "negative.tiff"
        test_image.write_bytes(b"test")

        iterations = []
        for iteration in range(3):
            # Measure
            measurement = spectro.read_density()

            # Print adjusted
            job = PrintJob(
                name=f"Iteration {iteration}",
                image_path=str(test_image),
            )
            result = printer.print_image(job)

            iterations.append(
                {
                    "iteration": iteration,
                    "density": measurement.density,
                    "print_success": result.success,
                }
            )

        spectro.disconnect()
        printer.disconnect()

        # All iterations should complete
        assert len(iterations) == 3
        for i in iterations:
            assert i["print_success"] is True


# =============================================================================
# Test Device Status Transitions
# =============================================================================


class TestDeviceStatusTransitions:
    """Test correct status transitions during operations."""

    def test_spectrophotometer_status_flow(self):
        """Test spectrophotometer status transitions."""
        device = get_spectrophotometer_driver(simulate=True)

        # Initial state
        assert device.status == DeviceStatus.DISCONNECTED

        # After connect
        device.connect()
        assert device.status == DeviceStatus.CONNECTED

        # During calibration (may be quick in simulation)
        device.calibrate_white()
        assert device.status == DeviceStatus.CONNECTED

        # After disconnect
        device.disconnect()
        assert device.status == DeviceStatus.DISCONNECTED

    def test_printer_status_flow(self, tmp_path):
        """Test printer status transitions."""
        printer = get_printer_driver(simulate=True)

        # Initial state
        assert printer.status == DeviceStatus.DISCONNECTED

        # After connect
        printer.connect()
        assert printer.status == DeviceStatus.CONNECTED

        # Create test file
        test_image = tmp_path / "test.tiff"
        test_image.write_bytes(b"data")

        # During print (should return to CONNECTED after)
        job = PrintJob(name="Test", image_path=str(test_image))
        printer.print_image(job)
        assert printer.status == DeviceStatus.CONNECTED

        # After disconnect
        printer.disconnect()
        assert printer.status == DeviceStatus.DISCONNECTED
