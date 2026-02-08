"""
Base subagent infrastructure with registry and communication protocols.

Provides abstract base class for all subagents and centralized registry.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ptpd_calibration.agents.logging import EventType, LogContext, get_agent_logger
from ptpd_calibration.agents.tools import ToolRegistry
from ptpd_calibration.config import AgentSettings, LLMSettings, get_settings
from ptpd_calibration.llm.client import LLMClient, create_client


class SubagentStatus(str, Enum):
    """Status of a subagent."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class SubagentCapability(str, Enum):
    """Capabilities that subagents can provide."""

    PLANNING = "planning"
    TESTING = "testing"
    CODING = "coding"
    REVIEWING = "reviewing"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    ORCHESTRATION = "orchestration"


class SubagentMessage(BaseModel):
    """Message for inter-agent communication."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender_id: str
    sender_type: str
    recipient_id: str | None = None
    recipient_type: str | None = None
    message_type: str  # request, response, notification, error
    content: dict = Field(default_factory=dict)
    correlation_id: str | None = None  # Links response to request
    priority: int = Field(default=0, ge=0, le=10)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class SubagentResult(BaseModel):
    """Result from subagent execution."""

    success: bool
    agent_id: str
    agent_type: str
    task: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0
    metadata: dict = Field(default_factory=dict)
    artifacts: list[dict] = Field(default_factory=list)  # Generated files, tests, etc.

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    llm_settings: LLMSettings | None = None
    agent_settings: AgentSettings | None = None
    tools: ToolRegistry | None = None
    parent_context: LogContext | None = None
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_logging: bool = True


class BaseSubagent(ABC):
    """
    Abstract base class for all subagents.

    Provides common functionality for:
    - LLM interaction
    - Logging and observability
    - Message passing
    - Tool execution
    """

    # Class-level attributes to be overridden
    AGENT_TYPE: str = "base"
    CAPABILITIES: list[SubagentCapability] = []
    DESCRIPTION: str = "Base subagent"

    def __init__(self, config: SubagentConfig | None = None):
        """
        Initialize the subagent.

        Args:
            config: Subagent configuration.
        """
        self.id = str(uuid4())
        self.config = config or SubagentConfig()
        self.settings = config.agent_settings if config else get_settings().agent
        self.llm_settings = config.llm_settings if config else get_settings().llm
        self.status = SubagentStatus.IDLE

        # Lazy-initialized components
        self._client: LLMClient | None = None
        self._tools: ToolRegistry | None = config.tools if config else None

        # Logging
        self._logger = get_agent_logger()
        if config and config.parent_context:
            self._context = config.parent_context.child_span()
        else:
            self._context = LogContext(
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
            )
        self._logger.set_context(self._context)

        # Execution tracking
        self._start_time: float | None = None
        self._messages: list[SubagentMessage] = []

    @property
    def client(self) -> LLMClient:
        """Get or create the LLM client (lazy initialization)."""
        if self._client is None:
            self._client = create_client(self.llm_settings)
        return self._client

    @property
    def tools(self) -> ToolRegistry | None:
        """Get the tool registry."""
        return self._tools

    @abstractmethod
    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """
        Execute the subagent's primary task.

        Args:
            task: Task description.
            context: Optional context data.

        Returns:
            SubagentResult with execution outcome.
        """
        pass

    @abstractmethod
    def capabilities(self) -> list[SubagentCapability]:
        """Return the capabilities this subagent provides."""
        pass

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: Operation name for logging.
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: If all retries fail.
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self._logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} for {operation}",
                    error=str(e),
                )
                if attempt < self.config.max_retries - 1:
                    await self._backoff(attempt)

        raise last_error or Exception(f"All retries failed for {operation}")

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retries."""
        import asyncio

        delay = min(2**attempt, 30)  # Max 30 seconds
        await asyncio.sleep(delay)

    def _start_execution(self, task: str) -> None:
        """Mark start of execution."""
        self._start_time = time.time()
        self.status = SubagentStatus.RUNNING
        self._logger.log_agent_started(
            agent_id=self.id,
            agent_type=self.AGENT_TYPE,
            task=task,
        )

    def _complete_execution(self, result: SubagentResult) -> SubagentResult:
        """Mark completion of execution."""
        if self._start_time:
            result.duration_ms = (time.time() - self._start_time) * 1000

        if result.success:
            self.status = SubagentStatus.COMPLETED
            self._logger.log_agent_completed(
                agent_id=self.id,
                duration_ms=result.duration_ms,
                result_summary=str(result.result)[:200] if result.result else "No result",
            )
        else:
            self.status = SubagentStatus.FAILED
            self._logger.log_agent_failed(
                agent_id=self.id,
                error=result.error or "Unknown error",
                duration_ms=result.duration_ms,
            )

        return result

    async def _llm_complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Complete a prompt using the LLM.

        Args:
            prompt: User prompt.
            system: System prompt override.
            max_tokens: Max tokens override.

        Returns:
            LLM response text.
        """
        start = time.time()
        self._logger.log_llm_request(
            provider=self.llm_settings.provider.value,
            model=self.llm_settings.anthropic_model,
        )

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens or self.llm_settings.max_tokens,
        )

        duration_ms = (time.time() - start) * 1000
        self._logger.log_llm_response(
            provider=self.llm_settings.provider.value,
            model=self.llm_settings.anthropic_model,
            duration_ms=duration_ms,
        )

        return response

    def send_message(
        self,
        recipient_id: str | None,
        recipient_type: str | None,
        message_type: str,
        content: dict,
        correlation_id: str | None = None,
    ) -> SubagentMessage:
        """
        Send a message to another agent.

        Args:
            recipient_id: Target agent ID.
            recipient_type: Target agent type.
            message_type: Type of message.
            content: Message content.
            correlation_id: ID of related request.

        Returns:
            Created SubagentMessage.
        """
        message = SubagentMessage(
            sender_id=self.id,
            sender_type=self.AGENT_TYPE,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
        )
        self._messages.append(message)
        self._logger.log_message_sent(
            from_agent=self.AGENT_TYPE,
            to_agent=recipient_type or "broadcast",
            message_type=message_type,
        )
        return message


class SubagentRegistry:
    """
    Registry for discovering and managing subagents.

    Provides dynamic subagent discovery, instantiation, and health tracking.
    """

    def __init__(self):
        """Initialize the registry."""
        self._agents: dict[str, type[BaseSubagent]] = {}
        self._instances: dict[str, BaseSubagent] = {}
        self._health: dict[str, dict] = {}
        self._logger = get_agent_logger()

    def register(self, agent_class: type[BaseSubagent]) -> None:
        """
        Register a subagent class.

        Args:
            agent_class: Subagent class to register.
        """
        agent_type = agent_class.AGENT_TYPE
        self._agents[agent_type] = agent_class
        self._logger.info(
            f"Registered subagent: {agent_type}",
            data={"capabilities": [c.value for c in agent_class.CAPABILITIES]},
        )

    def unregister(self, agent_type: str) -> bool:
        """
        Unregister a subagent class.

        Args:
            agent_type: Type of agent to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if agent_type in self._agents:
            del self._agents[agent_type]
            return True
        return False

    def get_agent_class(self, agent_type: str) -> type[BaseSubagent] | None:
        """
        Get a registered agent class.

        Args:
            agent_type: Type of agent.

        Returns:
            Agent class or None.
        """
        return self._agents.get(agent_type)

    def create_agent(
        self,
        agent_type: str,
        config: SubagentConfig | None = None,
    ) -> BaseSubagent | None:
        """
        Create a new agent instance.

        Args:
            agent_type: Type of agent to create.
            config: Configuration for the agent.

        Returns:
            New agent instance or None if type not found.
        """
        agent_class = self.get_agent_class(agent_type)
        if agent_class is None:
            self._logger.warning(f"Unknown agent type: {agent_type}")
            return None

        agent = agent_class(config)
        self._instances[agent.id] = agent
        self._logger.info(
            "Created agent instance",
            event_type=EventType.SUBAGENT_SPAWNED,
            data={"agent_id": agent.id, "agent_type": agent_type},
        )
        return agent

    def get_instance(self, agent_id: str) -> BaseSubagent | None:
        """
        Get an existing agent instance.

        Args:
            agent_id: ID of the agent.

        Returns:
            Agent instance or None.
        """
        return self._instances.get(agent_id)

    def list_agent_types(self) -> list[str]:
        """List all registered agent types."""
        return list(self._agents.keys())

    def find_by_capability(
        self,
        capability: SubagentCapability,
    ) -> list[type[BaseSubagent]]:
        """
        Find agents with a specific capability.

        Args:
            capability: Required capability.

        Returns:
            List of agent classes with the capability.
        """
        return [
            agent_class
            for agent_class in self._agents.values()
            if capability in agent_class.CAPABILITIES
        ]

    def get_agent_info(self) -> list[dict]:
        """Get information about all registered agents."""
        return [
            {
                "type": agent_class.AGENT_TYPE,
                "description": agent_class.DESCRIPTION,
                "capabilities": [c.value for c in agent_class.CAPABILITIES],
            }
            for agent_class in self._agents.values()
        ]

    def update_health(self, agent_id: str, status: dict) -> None:
        """
        Update health status for an agent.

        Args:
            agent_id: Agent ID.
            status: Health status data.
        """
        self._health[agent_id] = {
            "timestamp": datetime.now().isoformat(),
            **status,
        }

    def get_health(self, agent_id: str) -> dict | None:
        """Get health status for an agent."""
        return self._health.get(agent_id)

    def cleanup_idle(self, max_idle_seconds: int = 3600) -> int:
        """
        Remove idle agent instances.

        Args:
            max_idle_seconds: Maximum idle time before cleanup.

        Returns:
            Number of instances removed.
        """
        to_remove = []
        for agent_id, agent in self._instances.items():
            if agent.status == SubagentStatus.IDLE:
                # Check if idle for too long
                health = self._health.get(agent_id, {})
                if "timestamp" in health:
                    last_active = datetime.fromisoformat(health["timestamp"])
                    idle_seconds = (datetime.now() - last_active).total_seconds()
                    if idle_seconds > max_idle_seconds:
                        to_remove.append(agent_id)

        for agent_id in to_remove:
            del self._instances[agent_id]
            if agent_id in self._health:
                del self._health[agent_id]

        return len(to_remove)


# Global registry instance
_subagent_registry: SubagentRegistry | None = None


def get_subagent_registry() -> SubagentRegistry:
    """Get the global subagent registry."""
    global _subagent_registry
    if _subagent_registry is None:
        _subagent_registry = SubagentRegistry()
    return _subagent_registry


def register_subagent(agent_class: type[BaseSubagent]) -> type[BaseSubagent]:
    """
    Decorator to register a subagent class.

    Usage:
        @register_subagent
        class MyAgent(BaseSubagent):
            ...
    """
    get_subagent_registry().register(agent_class)
    return agent_class
