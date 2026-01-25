"""Unit tests for simulated hardware drivers.

Tests verify the SimulatedSpectrophotometer and SimulatedPrinter functionality.
"""

from pathlib import Path

import pytest

from ptpd_calibration.integrations.hardware.simulated import (
    SimulatedPrinter,
    SimulatedSpectrophotometer,
)
from ptpd_calibration.integrations.protocols import DeviceStatus, PrintJob


class TestSimulatedSpectrophotometerInit:
    """Test SimulatedSpectrophotometer initialization."""

    def test_initial_status_is_disconnected(self) -> None:
        """Test that initial status is DISCONNECTED."""
        device = SimulatedSpectrophotometer()
        assert device.status == DeviceStatus.DISCONNECTED

    def test_initial_device_info_is_none(self) -> None:
        """Test that initial device info is None."""
        device = SimulatedSpectrophotometer()
        assert device.device_info is None

    def test_custom_noise_level(self) -> None:
        """Test that custom noise level is accepted."""
        device = SimulatedSpectrophotometer(noise_level=0.05)
        assert device._noise_level == 0.05

    def test_custom_density_range(self) -> None:
        """Test that custom density range is accepted."""
        device = SimulatedSpectrophotometer(base_density=0.1, max_density=2.5)
        assert device._base_density == 0.1
        assert device._max_density == 2.5


class TestSimulatedSpectrophotometerConnect:
    """Test SimulatedSpectrophotometer connect method."""

    def test_connect_success(self) -> None:
        """Test successful connection."""
        device = SimulatedSpectrophotometer(simulate_delay=False)

        result = device.connect()

        assert result is True
        assert device.status == DeviceStatus.CONNECTED
        assert device.device_info is not None

    def test_connect_sets_device_info(self) -> None:
        """Test that connect sets device info."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()

        info = device.device_info
        assert info.vendor == "Simulated"
        assert "Virtual Spectro" in info.model


class TestSimulatedSpectrophotometerCalibration:
    """Test SimulatedSpectrophotometer calibration methods."""

    def test_calibrate_white_success(self) -> None:
        """Test successful white calibration."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()

        result = device.calibrate_white()

        assert result is True
        assert device._calibrated is True

    def test_calibrate_white_not_connected(self) -> None:
        """Test white calibration raises error when not connected."""
        device = SimulatedSpectrophotometer()

        with pytest.raises(RuntimeError, match="not connected"):
            device.calibrate_white()

    def test_calibrate_black_success(self) -> None:
        """Test successful black calibration."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        result = device.calibrate_black()

        assert result is True


class TestSimulatedSpectrophotometerReadDensity:
    """Test SimulatedSpectrophotometer read_density method."""

    def test_read_density_returns_measurement(self) -> None:
        """Test that read_density returns a DensityMeasurement."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        measurement = device.read_density()

        assert measurement is not None
        assert hasattr(measurement, "density")
        assert hasattr(measurement, "lab_l")
        assert hasattr(measurement, "lab_a")
        assert hasattr(measurement, "lab_b")

    def test_read_density_not_connected(self) -> None:
        """Test read_density fails when not connected."""
        device = SimulatedSpectrophotometer()

        with pytest.raises(RuntimeError):
            device.read_density()

    def test_read_density_values_in_range(self) -> None:
        """Test that density values are in valid range."""
        device = SimulatedSpectrophotometer(simulate_delay=False, noise_level=0.0)
        device.connect()
        device.calibrate_white()

        measurement = device.read_density()

        # Density should be positive
        assert measurement.density >= 0
        # Lab L should be 0-100
        assert 0 <= measurement.lab_l <= 100

    def test_read_density_increments_count(self) -> None:
        """Test that measurement count increments."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        device.read_density()
        count1 = device._measurement_count
        device.read_density()
        count2 = device._measurement_count

        assert count2 == count1 + 1

    def test_read_density_21_steps(self) -> None:
        """Test reading 21-step tablet produces valid curve."""
        device = SimulatedSpectrophotometer(simulate_delay=False, noise_level=0.0)
        device.connect()
        device.calibrate_white()

        densities = []
        for _ in range(21):
            measurement = device.read_density()
            densities.append(measurement.density)

        # Densities should generally increase
        # (allowing for some variation)
        assert densities[-1] > densities[0]


class TestSimulatedSpectrophotometerReadSpectral:
    """Test SimulatedSpectrophotometer read_spectral method."""

    def test_read_spectral_returns_data(self) -> None:
        """Test that read_spectral returns SpectralData."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        spectral = device.read_spectral()

        assert spectral is not None
        assert hasattr(spectral, "wavelengths")
        assert hasattr(spectral, "values")

    def test_read_spectral_wavelength_range(self) -> None:
        """Test that spectral covers visible range."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        spectral = device.read_spectral()

        # Should cover visible range
        assert spectral.wavelengths[0] >= 380
        assert spectral.wavelengths[-1] <= 780

    def test_read_spectral_not_connected(self) -> None:
        """Test read_spectral fails when not connected."""
        device = SimulatedSpectrophotometer()

        with pytest.raises(RuntimeError):
            device.read_spectral()


class TestSimulatedSpectrophotometerDisconnect:
    """Test SimulatedSpectrophotometer disconnect method."""

    def test_disconnect_clears_state(self) -> None:
        """Test that disconnect clears all state."""
        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        device.disconnect()

        assert device.status == DeviceStatus.DISCONNECTED
        assert device.device_info is None
        assert device._calibrated is False


class TestSimulatedPrinterInit:
    """Test SimulatedPrinter initialization."""

    def test_initial_status_is_disconnected(self) -> None:
        """Test that initial status is DISCONNECTED."""
        printer = SimulatedPrinter()
        assert printer.status == DeviceStatus.DISCONNECTED

    def test_custom_failure_rate(self) -> None:
        """Test that custom failure rate is accepted."""
        printer = SimulatedPrinter(failure_rate=0.5)
        assert printer._failure_rate == 0.5


class TestSimulatedPrinterConnect:
    """Test SimulatedPrinter connect method."""

    def test_connect_success(self) -> None:
        """Test successful connection."""
        printer = SimulatedPrinter(simulate_delay=False)

        result = printer.connect()

        assert result is True
        assert printer.status == DeviceStatus.CONNECTED
        assert printer.device_info is not None


class TestSimulatedPrinterPrintImage:
    """Test SimulatedPrinter print_image method."""

    def test_print_image_success(self, tmp_path: Path) -> None:
        """Test successful print job."""
        printer = SimulatedPrinter(simulate_delay=False, failure_rate=0.0)
        printer.connect()

        image_file = tmp_path / "test.tiff"
        image_file.write_bytes(b"test")

        job = PrintJob(
            name="Test Print",
            image_path=str(image_file),
            paper_size="8x10",
            resolution_dpi=1440,
        )

        result = printer.print_image(job)

        assert result.success is True
        assert result.job_id is not None

    def test_print_image_not_connected(self, tmp_path: Path) -> None:
        """Test print fails when not connected."""
        printer = SimulatedPrinter()

        image_file = tmp_path / "test.tiff"
        image_file.write_bytes(b"test")

        job = PrintJob(
            name="Test Print",
            image_path=str(image_file),
            paper_size="8x10",
            resolution_dpi=1440,
        )

        result = printer.print_image(job)

        assert result.success is False

    def test_print_image_with_failure_rate(self, tmp_path: Path) -> None:
        """Test print with 100% failure rate always fails."""
        printer = SimulatedPrinter(simulate_delay=False, failure_rate=1.0)
        printer.connect()

        image_file = tmp_path / "test.tiff"
        image_file.write_bytes(b"test")

        job = PrintJob(
            name="Test Print",
            image_path=str(image_file),
            paper_size="8x10",
            resolution_dpi=1440,
        )

        result = printer.print_image(job)

        assert result.success is False


class TestSimulatedPrinterGetPaperSizes:
    """Test SimulatedPrinter get_paper_sizes method."""

    def test_get_paper_sizes_returns_list(self) -> None:
        """Test that paper sizes returns a list."""
        printer = SimulatedPrinter()

        sizes = printer.get_paper_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) > 0
        assert "8x10" in sizes


class TestSimulatedPrinterGetResolutions:
    """Test SimulatedPrinter get_resolutions method."""

    def test_get_resolutions_returns_standard_values(self) -> None:
        """Test that standard resolutions are returned."""
        printer = SimulatedPrinter()

        resolutions = printer.get_resolutions()

        assert 360 in resolutions
        assert 1440 in resolutions
        assert 2880 in resolutions


class TestSimulatedPrinterGetInkLevels:
    """Test SimulatedPrinter get_ink_levels method."""

    def test_get_ink_levels_not_connected(self) -> None:
        """Test ink levels when not connected."""
        printer = SimulatedPrinter()

        levels = printer.get_ink_levels()

        assert levels == {}

    def test_get_ink_levels_connected(self) -> None:
        """Test ink levels when connected."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()

        levels = printer.get_ink_levels()

        assert len(levels) > 0
        for _color, data in levels.items():
            assert "level" in data
            assert 0 <= data["level"] <= 100
            assert "status" in data


class TestSimulatedSpectrophotometerConstants:
    """Test that simulated spectrophotometer uses constants module."""

    def test_uses_standard_spectral_range(self) -> None:
        """Test that spectral range matches constants."""
        from ptpd_calibration.integrations.hardware.constants import (
            DEFAULT_SPECTRAL_END_NM,
            DEFAULT_SPECTRAL_START_NM,
        )

        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()
        device.calibrate_white()

        spectral = device.read_spectral()

        assert spectral.start_nm == DEFAULT_SPECTRAL_START_NM
        assert spectral.end_nm == DEFAULT_SPECTRAL_END_NM

    def test_uses_device_info_from_constants(self) -> None:
        """Test that device info comes from constants."""
        from ptpd_calibration.integrations.hardware.constants import (
            SIMULATED_SPECTRO_MODEL,
            SIMULATED_SPECTRO_VENDOR,
        )

        device = SimulatedSpectrophotometer(simulate_delay=False)
        device.connect()

        assert device.device_info.vendor == SIMULATED_SPECTRO_VENDOR
        assert device.device_info.model == SIMULATED_SPECTRO_MODEL


class TestSimulatedPrinterConstants:
    """Test that simulated printer uses constants module."""

    def test_paper_sizes_match_constants(self) -> None:
        """Test that paper sizes match constants."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_PAPER_SIZES,
        )

        printer = SimulatedPrinter(simulate_delay=False)

        sizes = printer.get_paper_sizes()

        assert sizes == list(STANDARD_PAPER_SIZES)

    def test_resolutions_match_constants(self) -> None:
        """Test that resolutions match constants."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_RESOLUTIONS,
        )

        printer = SimulatedPrinter(simulate_delay=False)

        resolutions = printer.get_resolutions()

        assert resolutions == list(STANDARD_RESOLUTIONS)


class TestSimulatedPrinterDisconnect:
    """Test SimulatedPrinter disconnect method."""

    def test_disconnect_clears_state(self) -> None:
        """Test that disconnect clears all state."""
        printer = SimulatedPrinter(simulate_delay=False)
        printer.connect()

        printer.disconnect()

        assert printer.status == DeviceStatus.DISCONNECTED
        assert printer.device_info is None
