"""Unit tests for hardware constants module.

Tests verify that constants are properly defined and have valid values.
"""

import re


class TestPrinterConstants:
    """Test printer-related constants."""

    def test_standard_paper_sizes_not_empty(self) -> None:
        """Test that paper sizes list is not empty."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_PAPER_SIZES,
        )

        assert len(STANDARD_PAPER_SIZES) > 0

    def test_standard_paper_sizes_contains_common_sizes(self) -> None:
        """Test that common paper sizes are included."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_PAPER_SIZES,
        )

        common_sizes = ["8x10", "letter", "a4"]
        for size in common_sizes:
            assert size in STANDARD_PAPER_SIZES

    def test_standard_resolutions_are_valid_dpi(self) -> None:
        """Test that resolutions are positive integers."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_RESOLUTIONS,
        )

        for res in STANDARD_RESOLUTIONS:
            assert isinstance(res, int)
            assert res > 0

    def test_standard_resolutions_sorted(self) -> None:
        """Test that resolutions are in ascending order."""
        from ptpd_calibration.integrations.hardware.constants import (
            STANDARD_RESOLUTIONS,
        )

        assert sorted(STANDARD_RESOLUTIONS) == STANDARD_RESOLUTIONS

    def test_ink_level_threshold_in_valid_range(self) -> None:
        """Test that ink level threshold is between 0 and 100."""
        from ptpd_calibration.integrations.hardware.constants import (
            INK_LEVEL_LOW_THRESHOLD,
        )

        assert 0 < INK_LEVEL_LOW_THRESHOLD <= 100

    def test_paper_size_validation_pattern_is_valid_regex(self) -> None:
        """Test that paper size pattern is valid regex."""
        from ptpd_calibration.integrations.hardware.constants import (
            PAPER_SIZE_VALIDATION_PATTERN,
        )

        # Should compile without error
        pattern = re.compile(PAPER_SIZE_VALIDATION_PATTERN)

        # Should match valid sizes
        assert pattern.match("8x10")
        assert pattern.match("letter")
        assert pattern.match("A4")

        # Should not match invalid sizes
        assert not pattern.match("8x10; DROP TABLE")
        assert not pattern.match("../../../etc/passwd")


class TestSpectrophotometerConstants:
    """Test spectrophotometer-related constants."""

    def test_spectral_range_is_valid(self) -> None:
        """Test that spectral range is valid (visible light)."""
        from ptpd_calibration.integrations.hardware.constants import (
            DEFAULT_SPECTRAL_END_NM,
            DEFAULT_SPECTRAL_INTERVAL_NM,
            DEFAULT_SPECTRAL_START_NM,
        )

        # Visible light range is roughly 380-750nm
        assert 350 <= DEFAULT_SPECTRAL_START_NM <= 400
        assert 700 <= DEFAULT_SPECTRAL_END_NM <= 800
        assert DEFAULT_SPECTRAL_START_NM < DEFAULT_SPECTRAL_END_NM
        assert DEFAULT_SPECTRAL_INTERVAL_NM > 0

    def test_xrite_vendor_id_is_hex(self) -> None:
        """Test that X-Rite vendor ID is valid hex."""
        from ptpd_calibration.integrations.hardware.constants import (
            XRITE_VENDOR_ID,
        )

        assert isinstance(XRITE_VENDOR_ID, int)
        assert XRITE_VENDOR_ID > 0

    def test_serial_defaults_are_valid(self) -> None:
        """Test that serial communication defaults are valid."""
        from ptpd_calibration.integrations.hardware.constants import (
            DEFAULT_BAUD_RATE,
            DEFAULT_TIMEOUT_SECONDS,
        )

        # Common baud rates
        valid_baud_rates = [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
        assert DEFAULT_BAUD_RATE in valid_baud_rates
        assert DEFAULT_TIMEOUT_SECONDS > 0

    def test_protocol_patterns_are_valid_regex(self) -> None:
        """Test that protocol patterns compile as valid regex."""
        from ptpd_calibration.integrations.hardware.constants import (
            FIRMWARE_VERSION_PATTERN,
            SERIAL_NUMBER_PATTERN,
        )

        # Should compile without error
        fw_pattern = re.compile(FIRMWARE_VERSION_PATTERN)
        sn_pattern = re.compile(SERIAL_NUMBER_PATTERN)

        # Test firmware version pattern
        match = fw_pattern.search("V1.23.456")
        assert match is not None

        # Test serial number pattern
        match = sn_pattern.search("SN:ABC123")
        assert match is not None


class TestSimulatedConstants:
    """Test simulated device constants."""

    def test_simulation_delays_are_positive(self) -> None:
        """Test that simulation delays are positive."""
        from ptpd_calibration.integrations.hardware.constants import (
            SIMULATED_CALIBRATE_BLACK_DELAY_SEC,
            SIMULATED_CALIBRATE_WHITE_DELAY_SEC,
            SIMULATED_CONNECT_DELAY_SEC,
            SIMULATED_MEASURE_DELAY_SEC,
        )

        assert SIMULATED_CONNECT_DELAY_SEC > 0
        assert SIMULATED_CALIBRATE_WHITE_DELAY_SEC > 0
        assert SIMULATED_CALIBRATE_BLACK_DELAY_SEC > 0
        assert SIMULATED_MEASURE_DELAY_SEC > 0

    def test_step_tablet_steps_is_standard(self) -> None:
        """Test that step tablet uses standard 21 steps."""
        from ptpd_calibration.integrations.hardware.constants import (
            SIMULATED_STEP_TABLET_STEPS,
        )

        # Standard step tablets have 21 steps (0-20)
        assert SIMULATED_STEP_TABLET_STEPS == 21

    def test_density_gamma_in_valid_range(self) -> None:
        """Test that density gamma is in photographic range."""
        from ptpd_calibration.integrations.hardware.constants import (
            SIMULATED_DENSITY_GAMMA,
        )

        # Typical photographic gamma is 0.5-2.0
        assert 0.5 <= SIMULATED_DENSITY_GAMMA <= 2.0

    def test_ink_level_ranges_are_valid(self) -> None:
        """Test that ink level ranges are valid percentages."""
        from ptpd_calibration.integrations.hardware.constants import (
            SIMULATED_INK_LEVEL_RANGES,
        )

        for color, (min_level, max_level) in SIMULATED_INK_LEVEL_RANGES.items():
            assert 0 <= min_level <= 100, f"Invalid min level for {color}"
            assert 0 <= max_level <= 100, f"Invalid max level for {color}"
            assert min_level <= max_level, f"Min > max for {color}"


class TestCapabilitiesConstants:
    """Test device capabilities constants."""

    def test_printer_capabilities_not_empty(self) -> None:
        """Test that printer capabilities are defined."""
        from ptpd_calibration.integrations.hardware.constants import (
            PRINTER_CAPABILITIES,
        )

        assert len(PRINTER_CAPABILITIES) > 0
        assert "grayscale" in PRINTER_CAPABILITIES

    def test_spectro_capabilities_not_empty(self) -> None:
        """Test that spectro capabilities are defined."""
        from ptpd_calibration.integrations.hardware.constants import (
            SPECTRO_CAPABILITIES,
        )

        assert len(SPECTRO_CAPABILITIES) > 0
        assert "density" in SPECTRO_CAPABILITIES
