"""
Event-driven communication system for PTPD Calibration.

Provides a publish-subscribe event bus for decoupled communication
between components. Supports both sync and async handlers.

Usage:
    from ptpd_calibration.core.events import EventBus, Event

    # Define event types
    class CalibrationCompleted(Event):
        event_type: str = "calibration.completed"
        record_id: str
        curve_quality: float

    # Subscribe to events
    def on_calibration_completed(event: CalibrationCompleted):
        print(f"Calibration {event.record_id} completed!")

    bus = EventBus()
    bus.subscribe("calibration.completed", on_calibration_completed)

    # Publish events
    bus.publish(CalibrationCompleted(record_id="abc", curve_quality=0.95))
"""

import asyncio
import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from ptpd_calibration.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound="Event")


class Event(BaseModel):
    """Base event class for all system events.

    All events should inherit from this class and specify their
    event_type as a class attribute or field.

    Attributes:
        event_type: Unique identifier for the event type.
        timestamp: When the event was created.
        metadata: Additional event metadata.
    """

    event_type: str = Field(description="Event type identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Event creation timestamp"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

    model_config = {"extra": "allow"}


# Type aliases for handlers
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
Handler = SyncHandler | AsyncHandler


class EventBus:
    """Publish-subscribe event bus for component communication.

    Singleton pattern - use EventBus() to get the shared instance.
    Thread-safe for publishing, but handlers should manage their
    own thread safety.

    Features:
    - Sync and async handler support
    - Weak references to prevent memory leaks
    - Event filtering by type
    - Error isolation (one handler failure doesn't affect others)

    Example:
        bus = EventBus()

        # Subscribe with decorator
        @bus.on("calibration.completed")
        def handle_completion(event):
            print(f"Completed: {event}")

        # Subscribe to multiple events
        @bus.on("calibration.*")
        def handle_all_calibration(event):
            print(f"Calibration event: {event.event_type}")
    """

    _instance: "EventBus | None" = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "EventBus":
        """Singleton pattern implementation."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._subscribers: dict[str, list[Handler]] = defaultdict(list)
            instance._weak_subscribers: dict[str, list[weakref.ref]] = defaultdict(list)
            cls._instance = instance
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing).

        Warning: This will clear all subscriptions.
        """
        cls._instance = None

    def subscribe(
        self,
        event_type: str,
        handler: Handler,
        *,
        weak: bool = False,
    ) -> Callable[[], None]:
        """Subscribe a handler to an event type.

        Args:
            event_type: Event type to subscribe to. Supports wildcards:
                - "calibration.completed" - exact match
                - "calibration.*" - matches all calibration events
                - "*" - matches all events
            handler: Function to call when event is published.
            weak: Use weak reference (handler removed when object is collected).

        Returns:
            Unsubscribe function - call to remove the subscription.

        Example:
            unsubscribe = bus.subscribe("calibration.completed", my_handler)
            # Later...
            unsubscribe()  # Remove subscription
        """
        if weak:
            ref = weakref.ref(handler)
            self._weak_subscribers[event_type].append(ref)
            logger.debug(f"Subscribed (weak) to {event_type}: {handler.__name__}")

            def unsubscribe():
                if ref in self._weak_subscribers[event_type]:
                    self._weak_subscribers[event_type].remove(ref)

            return unsubscribe
        else:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to {event_type}: {handler.__name__}")

            def unsubscribe():
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)

            return unsubscribe

    def unsubscribe(self, event_type: str, handler: Handler) -> bool:
        """Remove a handler subscription.

        Args:
            event_type: Event type the handler was subscribed to.
            handler: The handler function to remove.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from {event_type}: {handler.__name__}")
            return True
        return False

    def on(self, event_type: str) -> Callable[[Handler], Handler]:
        """Decorator for subscribing handlers.

        Args:
            event_type: Event type to subscribe to.

        Returns:
            Decorator function.

        Example:
            @bus.on("calibration.completed")
            def handle_completion(event):
                print(f"Completed: {event}")
        """

        def decorator(handler: Handler) -> Handler:
            self.subscribe(event_type, handler)
            return handler

        return decorator

    def _get_handlers(self, event_type: str) -> list[Handler]:
        """Get all handlers matching an event type.

        Handles wildcard matching and cleans up dead weak references.
        """
        handlers: list[Handler] = []

        # Check exact match
        handlers.extend(self._subscribers[event_type])

        # Check wildcard subscribers
        for pattern, pattern_handlers in self._subscribers.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event_type.startswith(prefix):
                    handlers.extend(pattern_handlers)
            elif pattern == "*":
                handlers.extend(pattern_handlers)

        # Check weak references
        for pattern, weak_refs in self._weak_subscribers.items():
            if (
                pattern == event_type
                or pattern == "*"
                or (pattern.endswith(".*") and event_type.startswith(pattern[:-2]))
            ):
                # Clean up dead references
                alive_refs = []
                for ref in weak_refs:
                    handler = ref()
                    if handler is not None:
                        handlers.append(handler)
                        alive_refs.append(ref)
                self._weak_subscribers[pattern] = alive_refs

        return handlers

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Handlers are called synchronously in order of subscription.
        Errors in handlers are logged but don't prevent other handlers
        from being called.

        Args:
            event: Event to publish.
        """
        handlers = self._get_handlers(event.event_type)
        logger.debug(f"Publishing {event.event_type} to {len(handlers)} handlers")

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # For async handlers, schedule in event loop if available
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        # No running loop, run sync
                        asyncio.run(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.exception(f"Handler {handler.__name__} failed for {event.event_type}: {e}")

    async def publish_async(self, event: Event) -> None:
        """Publish an event asynchronously.

        All handlers are called concurrently and awaited.

        Args:
            event: Event to publish.
        """
        handlers = self._get_handlers(event.event_type)
        logger.debug(f"Publishing async {event.event_type} to {len(handlers)} handlers")

        async def call_handler(handler: Handler) -> None:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.exception(f"Handler {handler.__name__} failed for {event.event_type}: {e}")

        tasks = [call_handler(h) for h in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self, event_type: str | None = None) -> None:
        """Clear all subscribers for an event type or all events.

        Args:
            event_type: Specific event type to clear, or None for all.
        """
        if event_type:
            self._subscribers[event_type].clear()
            self._weak_subscribers[event_type].clear()
            logger.debug(f"Cleared subscribers for {event_type}")
        else:
            self._subscribers.clear()
            self._weak_subscribers.clear()
            logger.debug("Cleared all subscribers")


# Pre-defined event types for the calibration system
class CalibrationStarted(Event):
    """Event emitted when a calibration session starts."""

    event_type: str = "calibration.started"
    session_id: str
    paper_type: str | None = None


class CalibrationCompleted(Event):
    """Event emitted when a calibration session completes."""

    event_type: str = "calibration.completed"
    session_id: str
    record_id: str
    quality_score: float = 0.0


class CalibrationFailed(Event):
    """Event emitted when a calibration session fails."""

    event_type: str = "calibration.failed"
    session_id: str
    error: str
    error_type: str | None = None


class CurveGenerated(Event):
    """Event emitted when a curve is generated."""

    event_type: str = "curve.generated"
    curve_name: str
    num_points: int
    dmax: float = 0.0
    dmin: float = 0.0


class CurveExported(Event):
    """Event emitted when a curve is exported."""

    event_type: str = "curve.exported"
    curve_name: str
    format: str
    file_path: str


class HardwareConnected(Event):
    """Event emitted when hardware device connects."""

    event_type: str = "hardware.connected"
    device_type: str
    device_id: str
    vendor: str | None = None
    model: str | None = None


class HardwareDisconnected(Event):
    """Event emitted when hardware device disconnects."""

    event_type: str = "hardware.disconnected"
    device_type: str
    device_id: str
    reason: str | None = None


class MeasurementTaken(Event):
    """Event emitted when a measurement is taken."""

    event_type: str = "measurement.taken"
    device_id: str
    measurement_type: str
    value: float
    unit: str | None = None


# Convenience function for getting the global event bus
def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Returns:
        The singleton EventBus instance.
    """
    return EventBus()
