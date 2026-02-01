"""
Inter-agent communication system.

Provides message bus, protocols, and coordination for multi-agent workflows.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ptpd_calibration.agents.logging import AgentLogger, EventType, get_agent_logger


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 5
    HIGH = 8
    URGENT = 10


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CANCEL = "cancel"


class AgentMessage(BaseModel):
    """Message for agent-to-agent communication."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender_id: str
    sender_type: str
    recipient_id: str | None = None  # None = broadcast
    recipient_type: str | None = None  # Can target type instead of ID
    message_type: MessageType = MessageType.NOTIFICATION
    action: str = ""  # What action is requested
    payload: dict = Field(default_factory=dict)
    correlation_id: str | None = None  # Links responses to requests
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: int = 300  # Time to live
    requires_response: bool = False

    class Config:
        """Pydantic config."""

        use_enum_values = True


class MessageHandler:
    """Handler for processing messages."""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        handler_func: Callable[[AgentMessage], Any],
        actions: list[str] | None = None,
    ):
        """
        Initialize message handler.

        Args:
            agent_id: ID of the owning agent.
            agent_type: Type of the owning agent.
            handler_func: Function to handle messages.
            actions: List of actions this handler can process (None = all).
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.handler_func = handler_func
        self.actions = set(actions) if actions else None

    def can_handle(self, message: AgentMessage) -> bool:
        """Check if this handler can process the message."""
        # Check recipient
        if message.recipient_id and message.recipient_id != self.agent_id:
            return False
        if message.recipient_type and message.recipient_type != self.agent_type:
            return False

        # Check action
        if self.actions and message.action not in self.actions:
            return False

        return True

    async def handle(self, message: AgentMessage) -> Any:
        """Handle the message."""
        if asyncio.iscoroutinefunction(self.handler_func):
            return await self.handler_func(message)
        return self.handler_func(message)


class MessageBus:
    """
    Central message bus for inter-agent communication.

    Supports:
    - Point-to-point messaging
    - Broadcast messaging
    - Request-response patterns
    - Priority-based delivery
    - Message persistence
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the message bus.

        Args:
            max_queue_size: Maximum messages in queue.
        """
        self._handlers: list[MessageHandler] = []
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._message_history: list[AgentMessage] = []
        self._max_history = 1000
        self._running = False
        self._logger = get_agent_logger()

    def register_handler(self, handler: MessageHandler) -> None:
        """
        Register a message handler.

        Args:
            handler: Handler to register.
        """
        self._handlers.append(handler)
        self._logger.info(
            f"Registered handler for {handler.agent_type}",
            data={"agent_id": handler.agent_id, "actions": list(handler.actions) if handler.actions else "all"},
        )

    def unregister_handler(self, agent_id: str) -> None:
        """
        Unregister all handlers for an agent.

        Args:
            agent_id: ID of agent to unregister.
        """
        self._handlers = [h for h in self._handlers if h.agent_id != agent_id]

    async def send(self, message: AgentMessage) -> None:
        """
        Send a message to the bus.

        Args:
            message: Message to send.
        """
        # Priority queue uses (priority, counter, item) for ordering
        # Negate priority so higher priority comes first
        priority = -message.priority
        await self._queue.put((priority, message.timestamp.timestamp(), message))

        self._logger.log_message_sent(
            from_agent=message.sender_type,
            to_agent=message.recipient_type or "broadcast",
            message_type=message.message_type,
        )

        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)

    async def send_and_wait(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
    ) -> AgentMessage | None:
        """
        Send a message and wait for response.

        Args:
            message: Message to send.
            timeout: Timeout in seconds.

        Returns:
            Response message or None if timeout.
        """
        message.requires_response = True
        message.message_type = MessageType.REQUEST

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_responses[message.id] = future

        try:
            await self.send(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._logger.warning(
                f"Message timeout: {message.action}",
                data={"message_id": message.id},
            )
            return None
        finally:
            self._pending_responses.pop(message.id, None)

    async def respond(
        self,
        original: AgentMessage,
        response_payload: dict,
        sender_id: str,
        sender_type: str,
    ) -> None:
        """
        Send a response to a request message.

        Args:
            original: Original request message.
            response_payload: Response data.
            sender_id: ID of responding agent.
            sender_type: Type of responding agent.
        """
        response = AgentMessage(
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=original.sender_id,
            recipient_type=original.sender_type,
            message_type=MessageType.RESPONSE,
            action=original.action,
            payload=response_payload,
            correlation_id=original.id,
            priority=original.priority,
        )

        # Check for pending future
        if original.id in self._pending_responses:
            self._pending_responses[original.id].set_result(response)
        else:
            await self.send(response)

    async def broadcast(
        self,
        sender_id: str,
        sender_type: str,
        action: str,
        payload: dict,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """
        Broadcast a message to all agents.

        Args:
            sender_id: Sender agent ID.
            sender_type: Sender agent type.
            action: Action name.
            payload: Message payload.
            priority: Message priority.
        """
        message = AgentMessage(
            sender_id=sender_id,
            sender_type=sender_type,
            message_type=MessageType.NOTIFICATION,
            action=action,
            payload=payload,
            priority=priority,
        )
        await self.send(message)

    async def process_messages(self) -> None:
        """Process messages from the queue."""
        self._running = True

        while self._running:
            try:
                # Get message with timeout to allow stopping
                try:
                    _, _, message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Check TTL
                age = (datetime.now() - message.timestamp).total_seconds()
                if age > message.ttl_seconds:
                    self._logger.debug(
                        f"Message expired: {message.action}",
                        data={"message_id": message.id, "age": age},
                    )
                    continue

                # Find handlers
                handlers = [h for h in self._handlers if h.can_handle(message)]

                if not handlers:
                    self._logger.debug(
                        f"No handlers for message: {message.action}",
                        data={"message_id": message.id},
                    )
                    continue

                # Process with all matching handlers
                for handler in handlers:
                    try:
                        await handler.handle(message)
                    except Exception as e:
                        self._logger.error(
                            f"Handler error: {e}",
                            event_type=EventType.ERROR,
                            data={
                                "message_id": message.id,
                                "handler_agent": handler.agent_id,
                            },
                        )

            except Exception as e:
                self._logger.error(f"Message processing error: {e}")

    def stop(self) -> None:
        """Stop message processing."""
        self._running = False

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def get_message_history(
        self,
        sender_type: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AgentMessage]:
        """
        Get message history with optional filters.

        Args:
            sender_type: Filter by sender type.
            action: Filter by action.
            limit: Maximum messages to return.

        Returns:
            List of messages.
        """
        result = self._message_history

        if sender_type:
            result = [m for m in result if m.sender_type == sender_type]
        if action:
            result = [m for m in result if m.action == action]

        return result[-limit:]


@dataclass
class ConversationContext:
    """Context for a multi-turn conversation between agents."""

    id: str = field(default_factory=lambda: str(uuid4()))
    participants: list[str] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        if message.sender_id not in self.participants:
            self.participants.append(message.sender_id)

    def get_summary(self) -> str:
        """Get conversation summary."""
        return (
            f"Conversation {self.id[:8]} with {len(self.participants)} participants, "
            f"{len(self.messages)} messages"
        )


class ConversationManager:
    """Manages conversations between agents."""

    def __init__(self):
        """Initialize the conversation manager."""
        self._conversations: dict[str, ConversationContext] = {}
        self._agent_conversations: dict[str, list[str]] = defaultdict(list)

    def create_conversation(
        self,
        initiator_id: str,
        metadata: dict | None = None,
    ) -> ConversationContext:
        """
        Create a new conversation.

        Args:
            initiator_id: ID of the initiating agent.
            metadata: Optional conversation metadata.

        Returns:
            New ConversationContext.
        """
        context = ConversationContext(
            participants=[initiator_id],
            metadata=metadata or {},
        )
        self._conversations[context.id] = context
        self._agent_conversations[initiator_id].append(context.id)
        return context

    def get_conversation(self, conversation_id: str) -> ConversationContext | None:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def add_to_conversation(
        self,
        conversation_id: str,
        message: AgentMessage,
    ) -> bool:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID.
            message: Message to add.

        Returns:
            True if added, False if conversation not found.
        """
        context = self._conversations.get(conversation_id)
        if context:
            context.add_message(message)
            return True
        return False

    def get_agent_conversations(self, agent_id: str) -> list[ConversationContext]:
        """Get all conversations for an agent."""
        conv_ids = self._agent_conversations.get(agent_id, [])
        return [self._conversations[cid] for cid in conv_ids if cid in self._conversations]

    def close_conversation(self, conversation_id: str) -> None:
        """Close and archive a conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]


# Global message bus instance
_message_bus: MessageBus | None = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


async def start_message_bus() -> None:
    """Start the global message bus."""
    bus = get_message_bus()
    await bus.process_messages()


def stop_message_bus() -> None:
    """Stop the global message bus."""
    global _message_bus
    if _message_bus:
        _message_bus.stop()
