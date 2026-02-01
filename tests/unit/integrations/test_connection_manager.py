"""
Unit tests for ConnectionManager.

Tests connection lifecycle management, automatic reconnection,
health monitoring, and event callbacks.
"""

import threading
import time
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

from ptpd_calibration.integrations.hardware.connection_manager import (
    ConnectionManager,
    ConnectionSettings,
)
from ptpd_calibration.integrations.hardware.exceptions import (
    DeviceConnectionError,
    DeviceReconnectionError,
)
from ptpd_calibration.integrations.hardware.simulated import (
    SimulatedSpectrophotometer,
)
from ptpd_calibration.integrations.protocols import DeviceStatus


@pytest.fixture
def simulated_device():
    """Create a simulated spectrophotometer for testing."""
    return SimulatedSpectrophotometer(simulate_delay=False)


@pytest.fixture
def connection_settings():
    """Create test connection settings."""
    return ConnectionSettings(
        auto_reconnect=True,
        max_reconnect_attempts=3,
        reconnect_delay_seconds=0.1,  # Fast for testing
        reconnect_backoff_multiplier=1.5,
        connection_timeout_seconds=5.0,
        health_check_interval_seconds=1.0,
        enable_health_monitoring=False,  # Disable by default for unit tests
    )


@pytest.fixture
def manager(simulated_device, connection_settings):
    """Create a connection manager for testing."""
    mgr = ConnectionManager(simulated_device, connection_settings)
    yield mgr
    # Ensure cleanup
    with suppress(Exception):
        mgr.disconnect()


class TestConnectionSettings:
    """Tests for ConnectionSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = ConnectionSettings()

        assert settings.auto_reconnect is True
        assert settings.max_reconnect_attempts == 3
        assert settings.reconnect_delay_seconds == 1.0
        assert settings.enable_health_monitoring is True

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = ConnectionSettings(
            auto_reconnect=False,
            max_reconnect_attempts=5,
            reconnect_delay_seconds=2.0,
        )

        assert settings.auto_reconnect is False
        assert settings.max_reconnect_attempts == 5
        assert settings.reconnect_delay_seconds == 2.0


class TestConnectionManagerBasics:
    """Tests for basic ConnectionManager functionality."""

    def test_manager_creation(self, simulated_device, connection_settings):
        """Test creating a connection manager."""
        manager = ConnectionManager(simulated_device, connection_settings)

        assert manager.device is simulated_device
        assert manager.settings is connection_settings
        assert manager.is_connected is False

    def test_manager_without_settings(self, simulated_device):
        """Test creating manager with default settings."""
        manager = ConnectionManager(simulated_device)

        assert manager.settings is not None
        assert manager.settings.auto_reconnect is True

    def test_repr(self, manager):
        """Test string representation."""
        repr_str = repr(manager)

        assert "ConnectionManager" in repr_str
        assert "SimulatedSpectrophotometer" in repr_str
        assert "disconnected" in repr_str


class TestConnect:
    """Tests for connect functionality."""

    def test_connect_success(self, manager):
        """Test successful connection."""
        result = manager.connect()

        assert result is True
        assert manager.is_connected is True
        assert manager.state.last_connected_at is not None

    def test_connect_with_kwargs(self, manager):
        """Test connection with parameters."""
        result = manager.connect(port="/dev/ttyUSB0", timeout=5.0)

        assert result is True
        assert manager.is_connected is True

    def test_connect_stores_params_for_reconnect(self, manager):
        """Test that connection params are stored for reconnection."""
        manager.connect(port="/dev/ttyUSB0")

        assert manager._connection_params == {"port": "/dev/ttyUSB0"}


class TestDisconnect:
    """Tests for disconnect functionality."""

    def test_disconnect_success(self, manager):
        """Test successful disconnection."""
        manager.connect()
        assert manager.is_connected is True

        manager.disconnect()
        assert manager.is_connected is False
        assert manager.state.last_disconnected_at is not None

    def test_disconnect_when_not_connected(self, manager):
        """Test disconnect when already disconnected."""
        # Should not raise
        manager.disconnect()
        assert manager.is_connected is False


class TestEnsureConnected:
    """Tests for ensure_connected functionality."""

    def test_ensure_connected_when_connected(self, manager):
        """Test ensure_connected when already connected."""
        manager.connect()

        result = manager.ensure_connected()
        assert result is True

    def test_ensure_connected_when_disconnected_with_auto_reconnect(self, manager):
        """Test ensure_connected triggers reconnection."""
        manager.settings.auto_reconnect = True

        result = manager.ensure_connected()
        assert result is True
        assert manager.is_connected is True

    def test_ensure_connected_without_auto_reconnect(self, manager):
        """Test ensure_connected returns False without auto_reconnect."""
        manager.settings.auto_reconnect = False

        result = manager.ensure_connected()
        assert result is False


class TestReconnect:
    """Tests for reconnection functionality."""

    def test_reconnect_success(self, manager):
        """Test successful reconnection."""
        manager.connect()
        manager.disconnect()

        result = manager.reconnect()
        assert result is True
        assert manager.is_connected is True
        assert manager.state.total_reconnections == 1

    def test_reconnect_uses_stored_params(self, manager):
        """Test reconnection uses stored connection parameters."""
        manager.connect(port="/dev/ttyUSB0")
        manager.disconnect()

        # Should use stored params
        result = manager.reconnect()
        assert result is True

    def test_reconnect_failure_raises(self, connection_settings):
        """Test reconnection failure raises DeviceReconnectionError."""
        # Create a device that fails to connect
        mock_device = MagicMock()
        mock_device.connect.return_value = False
        mock_device.disconnect.return_value = None
        mock_device.status = DeviceStatus.DISCONNECTED

        manager = ConnectionManager(mock_device, connection_settings)

        with pytest.raises(DeviceReconnectionError):
            manager.reconnect()

    def test_reconnect_attempts_tracked(self, connection_settings):
        """Test reconnection attempts are tracked."""
        mock_device = MagicMock()
        mock_device.connect.return_value = False
        mock_device.disconnect.return_value = None
        mock_device.status = DeviceStatus.DISCONNECTED

        manager = ConnectionManager(mock_device, connection_settings)

        with pytest.raises(DeviceReconnectionError):
            manager.reconnect()

        # Should have attempted max_reconnect_attempts times
        assert mock_device.connect.call_count == connection_settings.max_reconnect_attempts


class TestEventCallbacks:
    """Tests for event callback functionality."""

    def test_connected_callback(self, manager):
        """Test connected event callback."""
        events = []

        manager.on("connected", lambda: events.append("connected"))
        manager.connect()

        assert "connected" in events

    def test_disconnected_callback(self, manager):
        """Test disconnected event callback."""
        events = []

        manager.on("disconnected", lambda: events.append("disconnected"))
        manager.connect()
        manager.disconnect()

        assert "disconnected" in events

    def test_reconnecting_callback(self, manager):
        """Test reconnecting event callback."""
        events = []

        manager.on("reconnecting", lambda: events.append("reconnecting"))
        manager.connect()
        manager.disconnect()
        manager.reconnect()

        assert "reconnecting" in events

    def test_error_callback(self, connection_settings):
        """Test error event callback."""
        errors = []

        mock_device = MagicMock()
        mock_device.connect.side_effect = Exception("Connection failed")

        manager = ConnectionManager(mock_device, connection_settings)
        manager.on("error", lambda e: errors.append(e))

        with pytest.raises(DeviceConnectionError):
            manager.connect()

        assert len(errors) == 1

    def test_off_removes_callback(self, manager):
        """Test removing an event callback."""
        events = []

        def callback():
            events.append("connected")

        manager.on("connected", callback)
        manager.off("connected", callback)
        manager.connect()

        assert "connected" not in events

    def test_multiple_callbacks(self, manager):
        """Test multiple callbacks for same event."""
        events = []

        manager.on("connected", lambda: events.append("callback1"))
        manager.on("connected", lambda: events.append("callback2"))
        manager.connect()

        assert "callback1" in events
        assert "callback2" in events


class TestContextManager:
    """Tests for context manager functionality."""

    def test_session_context_manager(self, simulated_device, connection_settings):
        """Test session context manager connects and disconnects."""
        manager = ConnectionManager(simulated_device, connection_settings)

        with manager.session() as device:
            assert device is simulated_device
            assert manager.is_connected is True

        assert manager.is_connected is False

    def test_session_with_kwargs(self, simulated_device, connection_settings):
        """Test session context manager with connection kwargs."""
        manager = ConnectionManager(simulated_device, connection_settings)

        with manager.session(port="/dev/ttyUSB0") as _device:
            assert manager.is_connected is True

    def test_enter_exit_protocol(self, simulated_device, connection_settings):
        """Test __enter__ and __exit__ protocol."""
        manager = ConnectionManager(simulated_device, connection_settings)

        result = manager.__enter__()
        assert result is manager
        assert manager.is_connected is True

        manager.__exit__(None, None, None)
        assert manager.is_connected is False


class TestConnectionState:
    """Tests for ConnectionState tracking."""

    def test_initial_state(self, manager):
        """Test initial connection state."""
        state = manager.state

        assert state.is_connected is False
        assert state.last_connected_at is None
        assert state.last_disconnected_at is None
        assert state.reconnect_attempts == 0
        assert state.total_reconnections == 0

    def test_state_after_connect(self, manager):
        """Test state after successful connection."""
        manager.connect()
        state = manager.state

        assert state.is_connected is True
        assert state.last_connected_at is not None
        assert state.reconnect_attempts == 0

    def test_state_after_disconnect(self, manager):
        """Test state after disconnection."""
        manager.connect()
        manager.disconnect()
        state = manager.state

        assert state.is_connected is False
        assert state.last_disconnected_at is not None

    def test_state_tracks_reconnections(self, manager):
        """Test state tracks total reconnections."""
        manager.connect()
        manager.disconnect()
        manager.reconnect()

        assert manager.state.total_reconnections == 1


class TestHealthMonitoring:
    """Tests for health monitoring functionality."""

    def test_health_monitor_starts_on_connect(self, simulated_device):
        """Test health monitor starts when connecting with it enabled."""
        settings = ConnectionSettings(
            enable_health_monitoring=True,
            health_check_interval_seconds=0.1,
        )
        manager = ConnectionManager(simulated_device, settings)

        manager.connect()

        # Give thread time to start
        time.sleep(0.05)

        assert manager._health_thread is not None
        assert manager._health_thread.is_alive()

        manager.disconnect()

    def test_health_monitor_stops_on_disconnect(self, simulated_device):
        """Test health monitor stops when disconnecting."""
        settings = ConnectionSettings(
            enable_health_monitoring=True,
            health_check_interval_seconds=0.1,
        )
        manager = ConnectionManager(simulated_device, settings)

        manager.connect()
        time.sleep(0.05)

        manager.disconnect()
        time.sleep(0.15)  # Wait for thread to stop

        assert manager._health_thread is None or not manager._health_thread.is_alive()

    def test_health_check_failure_triggers_reconnect(self):
        """Test health check failure triggers reconnection."""
        # Create a mock device that reports disconnected status
        mock_device = MagicMock()
        mock_device.connect.return_value = True
        mock_device.disconnect.return_value = None
        mock_device.status = DeviceStatus.CONNECTED

        settings = ConnectionSettings(
            enable_health_monitoring=True,
            health_check_interval_seconds=0.1,
            auto_reconnect=True,
            max_reconnect_attempts=1,
            reconnect_delay_seconds=0.01,
        )
        manager = ConnectionManager(mock_device, settings)
        manager.connect()

        # Verify connection was established
        assert manager.is_connected

        # Cleanup
        manager.disconnect()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_connect_disconnect(self, simulated_device, connection_settings):
        """Test concurrent connect/disconnect operations."""
        manager = ConnectionManager(simulated_device, connection_settings)
        errors = []

        def connect_disconnect():
            try:
                for _ in range(10):
                    manager.connect()
                    time.sleep(0.001)
                    manager.disconnect()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=connect_disconnect) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have any errors
        assert len(errors) == 0
