"""
Unit tests for DeviceRegistry.

Tests the singleton device registry pattern, device registration,
discovery handler management, and device lifecycle.
"""

import pytest

from ptpd_calibration.integrations.hardware.registry import (
    DeviceRegistry,
    DeviceType,
    DiscoveredDevice,
)
from ptpd_calibration.integrations.hardware.simulated import (
    SimulatedPrinter,
    SimulatedSpectrophotometer,
)
from ptpd_calibration.integrations.protocols import DeviceInfo


@pytest.fixture
def clean_registry():
    """Provide a clean registry instance for each test."""
    # Reset singleton before test
    DeviceRegistry.reset_singleton()
    registry = DeviceRegistry()
    yield registry
    # Clean up after test
    DeviceRegistry.reset_singleton()


@pytest.fixture
def sample_device_info():
    """Sample device info for testing."""
    return DeviceInfo(
        vendor="Test",
        model="TestDevice",
        serial_number="TEST-001",
        firmware_version="1.0.0",
        capabilities=["density", "lab"],
    )


@pytest.fixture
def sample_discovered_device(sample_device_info):
    """Sample discovered device for testing."""
    return DiscoveredDevice(
        device_type=DeviceType.SPECTROPHOTOMETER,
        device_id="test-spectro-001",
        device_info=sample_device_info,
        connection_params={"port": "/dev/ttyUSB0"},
        driver_hint="XRiteI1ProDriver",
        is_simulated=False,
    )


class TestDeviceRegistrySingleton:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self, clean_registry):  # noqa: ARG002
        """Verify singleton returns same instance."""
        registry1 = DeviceRegistry()
        registry2 = DeviceRegistry()

        assert registry1 is registry2

    def test_reset_singleton_creates_new_instance(self, clean_registry):  # noqa: ARG002
        """Verify reset_singleton creates new instance."""
        registry1 = DeviceRegistry()
        DeviceRegistry.reset_singleton()
        registry2 = DeviceRegistry()

        assert registry1 is not registry2


class TestDeviceRegistration:
    """Tests for device registration."""

    def test_register_device_manually(self, clean_registry, sample_device_info):
        """Test manual device registration."""
        clean_registry.register_device(
            device_id="manual-device-001",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
            device_info=sample_device_info,
            is_simulated=True,
        )

        devices = clean_registry.get_all_devices()
        assert "manual-device-001" in devices
        assert devices["manual-device-001"].device_type == DeviceType.SPECTROPHOTOMETER

    def test_register_device_without_info(self, clean_registry):
        """Test registration without device info (uses defaults)."""
        clean_registry.register_device(
            device_id="minimal-device",
            device_type=DeviceType.PRINTER,
            driver_class=SimulatedPrinter,
        )

        device = clean_registry.get_all_devices().get("minimal-device")
        assert device is not None
        assert device.discovered_info.device_info.vendor == "Manual"

    def test_unregister_device(self, clean_registry, sample_device_info):
        """Test device unregistration."""
        clean_registry.register_device(
            device_id="to-remove",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
            device_info=sample_device_info,
        )

        assert "to-remove" in clean_registry.get_all_devices()

        result = clean_registry.unregister_device("to-remove")
        assert result is True
        assert "to-remove" not in clean_registry.get_all_devices()

    def test_unregister_nonexistent_device(self, clean_registry):
        """Test unregistering nonexistent device returns False."""
        result = clean_registry.unregister_device("nonexistent")
        assert result is False


class TestDeviceRetrieval:
    """Tests for device retrieval."""

    def test_get_device_returns_instance(self, clean_registry):
        """Test get_device creates and returns device instance."""
        clean_registry.register_device(
            device_id="get-test",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
            is_simulated=True,
        )

        device = clean_registry.get_device("get-test")
        assert device is not None
        assert isinstance(device, SimulatedSpectrophotometer)

    def test_get_device_caches_instance(self, clean_registry):
        """Test get_device returns same instance on subsequent calls."""
        clean_registry.register_device(
            device_id="cache-test",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
            is_simulated=True,
        )

        device1 = clean_registry.get_device("cache-test")
        device2 = clean_registry.get_device("cache-test")
        assert device1 is device2

    def test_get_device_nonexistent_returns_none(self, clean_registry):
        """Test get_device returns None for nonexistent device."""
        device = clean_registry.get_device("nonexistent")
        assert device is None

    def test_get_devices_by_type(self, clean_registry):
        """Test filtering devices by type."""
        clean_registry.register_device(
            device_id="spectro-1",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
        )
        clean_registry.register_device(
            device_id="printer-1",
            device_type=DeviceType.PRINTER,
            driver_class=SimulatedPrinter,
        )
        clean_registry.register_device(
            device_id="spectro-2",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
        )

        spectros = clean_registry.get_devices_by_type(DeviceType.SPECTROPHOTOMETER)
        printers = clean_registry.get_devices_by_type(DeviceType.PRINTER)

        assert len(spectros) == 2
        assert len(printers) == 1


class TestDiscoveryHandlers:
    """Tests for discovery handler management."""

    def test_register_discovery_handler(self, clean_registry):
        """Test registering a discovery handler."""
        handler_called = [False]

        def mock_handler():
            handler_called[0] = True
            return []

        clean_registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            mock_handler,
        )

        clean_registry.discover_all()
        assert handler_called[0] is True

    def test_unregister_discovery_handler(self, clean_registry):
        """Test unregistering a discovery handler."""
        call_count = [0]

        def mock_handler():
            call_count[0] += 1
            return []

        clean_registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            mock_handler,
        )
        clean_registry.discover_all()
        assert call_count[0] == 1

        clean_registry.unregister_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            mock_handler,
        )
        clean_registry.discover_all()
        assert call_count[0] == 1  # Not incremented

    def test_discovery_handler_returns_devices(self, clean_registry, sample_device_info):
        """Test discovery handler that returns devices."""

        def mock_handler():
            return [
                DiscoveredDevice(
                    device_type=DeviceType.SPECTROPHOTOMETER,
                    device_id="discovered-001",
                    device_info=sample_device_info,
                    is_simulated=True,
                )
            ]

        clean_registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            mock_handler,
        )

        devices = clean_registry.discover_all()
        assert "discovered-001" in devices

    def test_discovery_handler_error_handling(self, clean_registry):
        """Test that failing handlers don't crash discovery."""

        def failing_handler():
            raise RuntimeError("Discovery failed")

        def working_handler():
            return []

        clean_registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            failing_handler,
        )
        clean_registry.register_discovery_handler(
            DeviceType.SPECTROPHOTOMETER,
            working_handler,
        )

        # Should not raise
        devices = clean_registry.discover_all()
        assert isinstance(devices, dict)


class TestChangeCallbacks:
    """Tests for change notification callbacks."""

    def test_on_change_callback_registered(self, clean_registry):
        """Test registering change callback."""
        events = []

        def callback(device_id, event_type):
            events.append((device_id, event_type))

        clean_registry.on_change(callback)
        clean_registry.register_device(
            device_id="callback-test",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
        )

        assert len(events) == 1
        assert events[0] == ("callback-test", "registered")

    def test_on_change_callback_for_unregister(self, clean_registry):
        """Test change callback on unregister."""
        events = []

        def callback(device_id, event_type):
            events.append((device_id, event_type))

        clean_registry.register_device(
            device_id="unregister-test",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
        )

        clean_registry.on_change(callback)
        clean_registry.unregister_device("unregister-test")

        assert ("unregister-test", "unregistered") in events


class TestDiscoveredDevice:
    """Tests for DiscoveredDevice dataclass."""

    def test_discovered_device_hash(self, sample_discovered_device):
        """Test DiscoveredDevice is hashable."""
        device_set = {sample_discovered_device}
        assert sample_discovered_device in device_set

    def test_discovered_device_equality(self, sample_device_info):
        """Test DiscoveredDevice equality based on device_id."""
        device1 = DiscoveredDevice(
            device_type=DeviceType.SPECTROPHOTOMETER,
            device_id="same-id",
            device_info=sample_device_info,
        )
        device2 = DiscoveredDevice(
            device_type=DeviceType.PRINTER,  # Different type
            device_id="same-id",  # Same ID
            device_info=sample_device_info,
        )

        assert device1 == device2  # Equal by ID only


class TestRegistryClear:
    """Tests for registry cleanup."""

    def test_clear_removes_all_devices(self, clean_registry):
        """Test clear removes all registered devices."""
        clean_registry.register_device(
            device_id="device-1",
            device_type=DeviceType.SPECTROPHOTOMETER,
            driver_class=SimulatedSpectrophotometer,
        )
        clean_registry.register_device(
            device_id="device-2",
            device_type=DeviceType.PRINTER,
            driver_class=SimulatedPrinter,
        )

        assert len(clean_registry.get_all_devices()) == 2

        clean_registry.clear()
        assert len(clean_registry.get_all_devices()) == 0
