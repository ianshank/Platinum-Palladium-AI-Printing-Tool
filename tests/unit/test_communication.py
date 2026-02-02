"""
Comprehensive unit tests for agents/communication.py module.

Tests MessageBus, MessageHandler, AgentMessage, and conversation management.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ptpd_calibration.agents.communication import (
    AgentMessage,
    ConversationContext,
    ConversationManager,
    MessageBus,
    MessageHandler,
    MessagePriority,
    MessageType,
    get_message_bus,
)


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert MessagePriority.LOW.value == 0
        assert MessagePriority.NORMAL.value == 5
        assert MessagePriority.HIGH.value == 8
        assert MessagePriority.URGENT.value == 10

    def test_priority_comparison(self):
        """Test priority comparison."""
        assert MessagePriority.URGENT.value > MessagePriority.HIGH.value
        assert MessagePriority.HIGH.value > MessagePriority.NORMAL.value
        assert MessagePriority.NORMAL.value > MessagePriority.LOW.value


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Test all message types exist."""
        assert MessageType.REQUEST
        assert MessageType.RESPONSE
        assert MessageType.NOTIFICATION
        assert MessageType.ERROR
        assert MessageType.HEARTBEAT


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="planner",
            action="plan",
            payload={"task": "test"},
        )
        assert msg.sender_id == "agent-1"
        assert msg.sender_type == "planner"
        assert msg.action == "plan"
        assert msg.payload == {"task": "test"}

    def test_message_defaults(self):
        """Test message default values."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
        )
        assert msg.payload == {}
        assert msg.recipient_id is None
        assert msg.recipient_type is None
        assert msg.correlation_id is None
        assert msg.ttl_seconds == 300
        assert msg.id is not None
        assert msg.timestamp is not None

    def test_message_with_priority(self):
        """Test creating message with priority."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
            action="urgent",
            priority=MessagePriority.URGENT,
        )
        assert msg.priority == MessagePriority.URGENT

    def test_message_with_recipient(self):
        """Test creating message with recipient."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="planner",
            action="delegate",
            recipient_id="agent-2",
            recipient_type="coder",
        )
        assert msg.recipient_id == "agent-2"
        assert msg.recipient_type == "coder"


class TestMessageHandler:
    """Tests for MessageHandler class."""

    def test_handler_creation(self):
        """Test creating a message handler."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="planner",
            handler_func=callback,
            actions=["plan", "analyze"],
        )
        assert handler.agent_id == "agent-1"
        assert handler.agent_type == "planner"
        assert handler.actions == {"plan", "analyze"}

    def test_handler_creation_no_actions(self):
        """Test creating handler without action filter."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=callback,
        )
        assert handler.actions is None

    def test_can_handle_matching_recipient_id(self):
        """Test handler matching by recipient ID."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="planner",
            handler_func=callback,
        )

        msg_match = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_id="agent-1",
        )
        msg_no_match = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_id="agent-2",
        )

        assert handler.can_handle(msg_match)
        assert not handler.can_handle(msg_no_match)

    def test_can_handle_matching_recipient_type(self):
        """Test handler matching by recipient type."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="planner",
            handler_func=callback,
        )

        msg_match = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_type="planner",
        )
        msg_no_match = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_type="coder",
        )

        assert handler.can_handle(msg_match)
        assert not handler.can_handle(msg_no_match)

    def test_can_handle_matching_action(self):
        """Test handler matching by action."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=callback,
            actions=["plan", "execute"],
        )

        msg_plan = AgentMessage(
            sender_id="sender",
            sender_type="test",
            action="plan",
        )
        msg_other = AgentMessage(
            sender_id="sender",
            sender_type="test",
            action="review",
        )

        assert handler.can_handle(msg_plan)
        assert not handler.can_handle(msg_other)

    def test_can_handle_broadcast(self):
        """Test handler accepts broadcast messages."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="any",
            handler_func=callback,
        )

        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
        )
        assert handler.can_handle(msg)

    @pytest.mark.asyncio
    async def test_handle_async_function(self):
        """Test handler with async function."""

        async def async_callback(msg):
            return {"status": "handled", "action": msg.action}

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=async_callback,
        )

        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
            action="test_action",
        )
        result = await handler.handle(msg)
        assert result == {"status": "handled", "action": "test_action"}

    @pytest.mark.asyncio
    async def test_handle_sync_function(self):
        """Test handler with sync function."""

        def sync_callback(msg):
            return {"status": "sync_handled"}

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=sync_callback,
        )

        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
        )
        result = await handler.handle(msg)
        assert result == {"status": "sync_handled"}


class TestMessageBus:
    """Tests for MessageBus class."""

    @pytest.fixture
    def message_bus(self):
        """Create a fresh message bus for each test."""
        return MessageBus()

    def test_message_bus_creation(self, message_bus):
        """Test creating message bus."""
        assert message_bus._handlers == []
        assert message_bus._running is False

    def test_register_handler(self, message_bus):
        """Test registering a handler."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=callback,
        )
        message_bus.register_handler(handler)
        assert len(message_bus._handlers) == 1

    def test_unregister_handler(self, message_bus):
        """Test unregistering a handler."""

        def callback(msg):
            return msg

        handler = MessageHandler(
            agent_id="agent-1",
            agent_type="test",
            handler_func=callback,
        )
        message_bus.register_handler(handler)
        assert len(message_bus._handlers) == 1

        message_bus.unregister_handler("agent-1")
        assert len(message_bus._handlers) == 0

    def test_unregister_nonexistent_handler(self, message_bus):
        """Test unregistering nonexistent handler."""
        message_bus.unregister_handler("nonexistent")

    @pytest.mark.asyncio
    async def test_send_message(self, message_bus):
        """Test sending a message."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
            action="test",
        )
        await message_bus.send(msg)
        assert not message_bus._queue.empty()

    @pytest.mark.asyncio
    async def test_send_stores_history(self, message_bus):
        """Test that send stores message in history."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
        )
        await message_bus.send(msg)
        assert len(message_bus._message_history) == 1

    @pytest.mark.asyncio
    async def test_broadcast(self, message_bus):
        """Test broadcasting to all handlers."""
        await message_bus.broadcast(
            sender_id="sender-1",
            sender_type="orchestrator",
            action="notify",
            payload={"message": "hello"},
        )
        assert not message_bus._queue.empty()


class TestConversationContext:
    """Tests for ConversationContext model."""

    def test_create_context(self):
        """Test creating conversation context."""
        context = ConversationContext(
            participants=["agent-1", "agent-2"],
        )
        assert "agent-1" in context.participants
        assert "agent-2" in context.participants
        assert context.messages == []
        assert context.id is not None

    def test_context_defaults(self):
        """Test context default values."""
        context = ConversationContext(
            participants=["agent-1"],
        )
        assert context.messages == []
        assert context.metadata == {}


class TestConversationManager:
    """Tests for ConversationManager class."""

    @pytest.fixture
    def manager(self):
        """Create conversation manager."""
        return ConversationManager()

    def test_manager_creation(self, manager):
        """Test creating conversation manager."""
        assert manager._conversations == {}

    def test_create_conversation(self, manager):
        """Test creating a new conversation."""
        context = manager.create_conversation(initiator_id="agent-1")
        assert context is not None
        assert context.id in manager._conversations

    def test_get_conversation(self, manager):
        """Test getting a conversation."""
        context = manager.create_conversation(initiator_id="agent-1")
        found = manager.get_conversation(context.id)
        assert found is not None
        assert found.id == context.id

    def test_get_nonexistent_conversation(self, manager):
        """Test getting nonexistent conversation."""
        found = manager.get_conversation("nonexistent")
        assert found is None

    def test_list_agent_conversations(self, manager):
        """Test listing agent's conversations."""
        manager.create_conversation(initiator_id="agent-1")
        manager.create_conversation(initiator_id="agent-1")
        convs = manager.get_agent_conversations("agent-1")
        assert len(convs) == 2


class TestConversationContextMethods:
    """Additional tests for ConversationContext methods."""

    def test_add_message(self):
        """Test adding message to conversation."""
        context = ConversationContext(
            participants=["agent-1"],
        )
        msg = AgentMessage(
            sender_id="agent-2",
            sender_type="coder",
            action="test",
        )
        context.add_message(msg)
        assert len(context.messages) == 1
        assert "agent-2" in context.participants

    def test_add_message_existing_participant(self):
        """Test adding message from existing participant."""
        context = ConversationContext(
            participants=["agent-1"],
        )
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="planner",
        )
        context.add_message(msg)
        assert len(context.participants) == 1  # No duplicate

    def test_get_summary(self):
        """Test getting conversation summary."""
        context = ConversationContext(
            participants=["agent-1", "agent-2"],
        )
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
        )
        context.add_message(msg)
        summary = context.get_summary()
        assert "2 participants" in summary
        assert "1 messages" in summary


class TestConversationManagerAdvanced:
    """Additional tests for ConversationManager."""

    @pytest.fixture
    def manager(self):
        """Create conversation manager."""
        return ConversationManager()

    def test_add_to_conversation(self, manager):
        """Test adding message to conversation."""
        context = manager.create_conversation(initiator_id="agent-1")
        msg = AgentMessage(
            sender_id="agent-2",
            sender_type="coder",
        )
        result = manager.add_to_conversation(context.id, msg)
        assert result is True
        assert len(context.messages) == 1

    def test_add_to_nonexistent_conversation(self, manager):
        """Test adding to nonexistent conversation."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
        )
        result = manager.add_to_conversation("fake-id", msg)
        assert result is False

    def test_close_conversation(self, manager):
        """Test closing a conversation."""
        context = manager.create_conversation(initiator_id="agent-1")
        conv_id = context.id
        manager.close_conversation(conv_id)
        assert manager.get_conversation(conv_id) is None

    def test_close_nonexistent_conversation(self, manager):
        """Test closing nonexistent conversation (no error)."""
        manager.close_conversation("fake-id")  # Should not raise


class TestMessageBusAdvanced:
    """Advanced tests for MessageBus."""

    @pytest.fixture
    def message_bus(self):
        """Create a fresh message bus."""
        return MessageBus()

    def test_get_queue_size(self, message_bus):
        """Test getting queue size."""
        assert message_bus.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_get_queue_size_after_send(self, message_bus):
        """Test queue size after sending messages."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
        )
        await message_bus.send(msg)
        assert message_bus.get_queue_size() == 1

    def test_stop(self, message_bus):
        """Test stopping the message bus."""
        message_bus._running = True
        message_bus.stop()
        assert message_bus._running is False

    @pytest.mark.asyncio
    async def test_get_message_history(self, message_bus):
        """Test getting message history."""
        msg1 = AgentMessage(
            sender_id="agent-1",
            sender_type="planner",
            action="plan",
        )
        msg2 = AgentMessage(
            sender_id="agent-2",
            sender_type="coder",
            action="code",
        )
        await message_bus.send(msg1)
        await message_bus.send(msg2)

        history = message_bus.get_message_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_message_history_by_sender(self, message_bus):
        """Test filtering history by sender type."""
        msg1 = AgentMessage(
            sender_id="agent-1",
            sender_type="planner",
            action="plan",
        )
        msg2 = AgentMessage(
            sender_id="agent-2",
            sender_type="coder",
            action="code",
        )
        await message_bus.send(msg1)
        await message_bus.send(msg2)

        history = message_bus.get_message_history(sender_type="planner")
        assert len(history) == 1
        assert history[0].sender_type == "planner"

    @pytest.mark.asyncio
    async def test_get_message_history_by_action(self, message_bus):
        """Test filtering history by action."""
        msg1 = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
            action="plan",
        )
        msg2 = AgentMessage(
            sender_id="agent-2",
            sender_type="test",
            action="code",
        )
        await message_bus.send(msg1)
        await message_bus.send(msg2)

        history = message_bus.get_message_history(action="code")
        assert len(history) == 1
        assert history[0].action == "code"

    @pytest.mark.asyncio
    async def test_get_message_history_limit(self, message_bus):
        """Test history limit."""
        for i in range(5):
            msg = AgentMessage(
                sender_id=f"agent-{i}",
                sender_type="test",
            )
            await message_bus.send(msg)

        history = message_bus.get_message_history(limit=3)
        assert len(history) == 3


class TestGlobalMessageBus:
    """Tests for global message bus functions."""

    def test_get_message_bus(self):
        """Test getting global message bus."""
        import ptpd_calibration.agents.communication as comm

        comm._message_bus = None
        bus = get_message_bus()
        assert bus is not None
        bus2 = get_message_bus()
        assert bus is bus2


class TestMessageBusSendAndWait:
    """Tests for MessageBus.send_and_wait functionality."""

    @pytest.fixture
    def message_bus(self):
        """Create a fresh message bus."""
        return MessageBus()

    @pytest.mark.asyncio
    async def test_send_and_wait_timeout(self, message_bus):
        """Test send_and_wait times out when no response."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="test",
            action="request",
        )
        result = await message_bus.send_and_wait(msg, timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_and_wait_with_response(self, message_bus):
        """Test send_and_wait receives response."""
        msg = AgentMessage(
            sender_id="agent-1",
            sender_type="requester",
            action="request",
        )

        async def respond_after_delay():
            await asyncio.sleep(0.05)
            await message_bus.respond(
                original=msg,
                response_payload={"result": "success"},
                sender_id="agent-2",
                sender_type="responder",
            )

        # Start responding in background
        await message_bus.send(msg)
        message_bus._pending_responses[msg.id] = asyncio.Future()

        # Simulate setting the result
        response_msg = AgentMessage(
            sender_id="agent-2",
            sender_type="responder",
            recipient_id="agent-1",
            recipient_type="requester",
            action="request",
            payload={"result": "success"},
            correlation_id=msg.id,
        )
        message_bus._pending_responses[msg.id].set_result(response_msg)

        result = await message_bus._pending_responses[msg.id]
        assert result is not None
        assert result.payload == {"result": "success"}


class TestMessageBusRespond:
    """Tests for MessageBus.respond functionality."""

    @pytest.fixture
    def message_bus(self):
        """Create a fresh message bus."""
        return MessageBus()

    @pytest.mark.asyncio
    async def test_respond_to_message(self, message_bus):
        """Test responding to a message."""
        original = AgentMessage(
            sender_id="agent-1",
            sender_type="requester",
            action="get_data",
        )

        await message_bus.respond(
            original=original,
            response_payload={"data": "test"},
            sender_id="agent-2",
            sender_type="responder",
        )

        assert message_bus.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_respond_with_pending_future(self, message_bus):
        """Test responding when there's a pending future."""
        original = AgentMessage(
            sender_id="agent-1",
            sender_type="requester",
            action="get_data",
        )

        # Set up pending response
        future: asyncio.Future = asyncio.Future()
        message_bus._pending_responses[original.id] = future

        await message_bus.respond(
            original=original,
            response_payload={"data": "result"},
            sender_id="agent-2",
            sender_type="responder",
        )

        # Future should be resolved
        assert future.done()
        result = await future
        assert result.payload == {"data": "result"}


class TestMessageBusProcessMessages:
    """Tests for MessageBus.process_messages functionality."""

    @pytest.fixture
    def message_bus(self):
        """Create a fresh message bus."""
        return MessageBus()

    @pytest.mark.asyncio
    async def test_process_messages_starts_and_stops(self, message_bus):
        """Test process_messages can be started and stopped."""
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            message_bus.stop()

        # Start stopper task
        stopper = asyncio.create_task(stop_after_delay())

        # Run processor briefly
        processor = asyncio.create_task(message_bus.process_messages())

        await stopper
        await asyncio.sleep(0.05)  # Let processor notice stop
        processor.cancel()

        assert message_bus._running is False

    @pytest.mark.asyncio
    async def test_process_messages_handles_message(self, message_bus):
        """Test process_messages processes queued messages."""
        handled = []

        async def handler_callback(msg):
            handled.append(msg)
            return {"handled": True}

        handler = MessageHandler(
            agent_id="receiver",
            agent_type="test",
            handler_func=handler_callback,
        )
        message_bus.register_handler(handler)

        # Queue a message
        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_id="receiver",
            action="process",
        )
        await message_bus.send(msg)

        # Start processor
        async def stop_after_process():
            await asyncio.sleep(0.2)
            message_bus.stop()

        stopper = asyncio.create_task(stop_after_process())
        processor = asyncio.create_task(message_bus.process_messages())

        await stopper
        await asyncio.sleep(0.05)
        processor.cancel()

        assert len(handled) == 1
        assert handled[0].action == "process"

    @pytest.mark.asyncio
    async def test_process_messages_skips_expired(self, message_bus):
        """Test process_messages skips expired messages."""
        from datetime import timedelta

        handled = []

        async def handler_callback(msg):
            handled.append(msg)

        handler = MessageHandler(
            agent_id="receiver",
            agent_type="test",
            handler_func=handler_callback,
        )
        message_bus.register_handler(handler)

        # Create expired message
        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_id="receiver",
            ttl_seconds=0,  # Already expired
        )
        # Manually set old timestamp
        msg.timestamp = datetime.now() - timedelta(seconds=10)
        await message_bus.send(msg)

        async def stop_after_delay():
            await asyncio.sleep(0.15)
            message_bus.stop()

        stopper = asyncio.create_task(stop_after_delay())
        processor = asyncio.create_task(message_bus.process_messages())

        await stopper
        await asyncio.sleep(0.05)
        processor.cancel()

        # Message was expired, should not be handled
        assert len(handled) == 0

    @pytest.mark.asyncio
    async def test_process_messages_handler_error(self, message_bus):
        """Test process_messages handles handler errors gracefully."""

        async def failing_handler(msg):
            raise ValueError("Handler error")

        handler = MessageHandler(
            agent_id="receiver",
            agent_type="test",
            handler_func=failing_handler,
        )
        message_bus.register_handler(handler)

        msg = AgentMessage(
            sender_id="sender",
            sender_type="test",
            recipient_id="receiver",
        )
        await message_bus.send(msg)

        async def stop_after_delay():
            await asyncio.sleep(0.15)
            message_bus.stop()

        stopper = asyncio.create_task(stop_after_delay())
        processor = asyncio.create_task(message_bus.process_messages())

        await stopper
        await asyncio.sleep(0.05)
        processor.cancel()
        # Should not raise - error is logged


class TestStartStopMessageBus:
    """Tests for start_message_bus and stop_message_bus functions."""

    def test_stop_message_bus(self):
        """Test stopping global message bus."""
        import ptpd_calibration.agents.communication as comm

        from ptpd_calibration.agents.communication import stop_message_bus

        # Get or create the bus
        bus = get_message_bus()
        bus._running = True

        stop_message_bus()
        assert bus._running is False

    def test_stop_message_bus_when_none(self):
        """Test stopping when no bus exists."""
        import ptpd_calibration.agents.communication as comm

        from ptpd_calibration.agents.communication import stop_message_bus

        comm._message_bus = None
        stop_message_bus()  # Should not raise


class TestMessageTypeCancel:
    """Tests for CANCEL message type."""

    def test_cancel_message_type_exists(self):
        """Test CANCEL message type exists."""
        assert MessageType.CANCEL == "cancel"

    def test_create_cancel_message(self):
        """Test creating a cancel message."""
        msg = AgentMessage(
            sender_id="orchestrator",
            sender_type="orchestrator",
            message_type=MessageType.CANCEL,
            action="cancel_task",
            payload={"task_id": "task-123"},
        )
        assert msg.message_type == "cancel"
