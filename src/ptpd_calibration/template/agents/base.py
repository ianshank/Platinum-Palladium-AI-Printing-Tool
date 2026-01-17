"""
Base Agent Classes

Provides foundation for building specialized agents with:
- Lifecycle management
- Context handling
- Result standardization
- State tracking
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from ptpd_calibration.template.errors import TemplateError, TimeoutError
from ptpd_calibration.template.logging_config import LogContext, get_logger

logger = get_logger(__name__)

# Type variables
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentState(str, Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentContext(BaseModel):
    """
    Context passed to agents during execution.

    Contains all information needed for the agent to perform its task.
    """

    # Identification
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    parent_task_id: Optional[str] = None

    # Task information
    task_type: str = "default"
    priority: int = Field(default=5, ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Execution parameters
    timeout_seconds: float = 60.0
    max_iterations: int = 10
    allow_parallel: bool = True

    # State
    iteration: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # History
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)

    def add_thought(self, thought: str, **metadata: Any) -> None:
        """Add a thought to the reasoning trace."""
        self.reasoning_trace.append({
            "type": "thought",
            "content": thought,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata,
        })

    def add_action(self, action: str, tool: str, **params: Any) -> None:
        """Add an action to the reasoning trace."""
        self.reasoning_trace.append({
            "type": "action",
            "action": action,
            "tool": tool,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def add_observation(self, observation: str, **metadata: Any) -> None:
        """Add an observation to the reasoning trace."""
        self.reasoning_trace.append({
            "type": "observation",
            "content": observation,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata,
        })


class AgentResult(BaseModel, Generic[OutputT]):
    """
    Standardized result from agent execution.

    Generic over the output type for type safety.
    """

    # Status
    success: bool
    state: AgentState

    # Output
    output: Optional[OutputT] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Metrics
    iterations_used: int = 0
    execution_time_ms: float = 0.0
    tokens_used: int = 0

    # Trace
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        output: OutputT,
        **kwargs: Any,
    ) -> "AgentResult[OutputT]":
        """Create a successful result."""
        return cls(
            success=True,
            state=AgentState.COMPLETED,
            output=output,
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        error_code: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentResult[OutputT]":
        """Create a failure result."""
        return cls(
            success=False,
            state=AgentState.FAILED,
            error=error,
            error_code=error_code,
            **kwargs,
        )


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True
    timeout_seconds: float = 60.0
    max_iterations: int = 10
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    tools: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentBase(ABC, Generic[InputT, OutputT]):
    """
    Base class for all agents.

    Provides:
    - Lifecycle management (initialize, execute, cleanup)
    - State tracking
    - Error handling
    - Logging integration
    - Timeout enforcement

    Usage:
        class MyAgent(AgentBase[MyInput, MyOutput]):
            async def _execute(
                self,
                input_data: MyInput,
                context: AgentContext,
            ) -> AgentResult[MyOutput]:
                # Agent logic here
                return AgentResult.success_result(output)

        agent = MyAgent(config)
        result = await agent.run(input_data)
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration."""
        self.config = config
        self.name = config.name
        self._state = AgentState.IDLE
        self._logger = get_logger(f"agent.{config.name}")
        self._tools: dict[str, Any] = {}
        self._initialized = False

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        """Set agent state with logging."""
        old_state = self._state
        self._state = value
        self._logger.debug(
            f"State transition: {old_state.value} -> {value.value}",
            agent=self.name,
        )

    async def initialize(self) -> None:
        """
        Initialize agent resources.

        Override to set up connections, load models, etc.
        """
        self.state = AgentState.INITIALIZING
        self._initialized = True
        self.state = AgentState.IDLE
        self._logger.info(f"Agent initialized: {self.name}")

    async def cleanup(self) -> None:
        """
        Clean up agent resources.

        Override to close connections, release resources, etc.
        """
        self._initialized = False
        self.state = AgentState.IDLE
        self._logger.info(f"Agent cleaned up: {self.name}")

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool for this agent."""
        self._tools[name] = tool
        self._logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a registered tool."""
        return self._tools.get(name)

    @abstractmethod
    async def _execute(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        """
        Execute the agent's main logic.

        Must be implemented by subclasses.

        Args:
            input_data: Input to process
            context: Execution context

        Returns:
            Result of execution
        """
        pass

    async def run(
        self,
        input_data: InputT,
        context: Optional[AgentContext] = None,
        timeout: Optional[float] = None,
    ) -> AgentResult[OutputT]:
        """
        Run the agent with full lifecycle management.

        Args:
            input_data: Input to process
            context: Optional execution context
            timeout: Optional timeout override

        Returns:
            Result of execution
        """
        # Create context if not provided
        if context is None:
            context = AgentContext(
                task_type=self.name,
                timeout_seconds=timeout or self.config.timeout_seconds,
                max_iterations=self.config.max_iterations,
            )

        # Use log context for tracing
        with LogContext(
            request_id=context.task_id,
            session_id=context.session_id,
            operation=f"agent:{self.name}",
        ):
            return await self._run_with_timeout(input_data, context)

    async def _run_with_timeout(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        """Run agent with timeout enforcement."""
        import time

        start_time = time.perf_counter()

        try:
            # Initialize if needed
            if not self._initialized:
                await self.initialize()

            # Execute with timeout
            self.state = AgentState.EXECUTING

            result = await asyncio.wait_for(
                self._execute(input_data, context),
                timeout=context.timeout_seconds,
            )

            # Add execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time_ms
            result.reasoning_trace = context.reasoning_trace

            self._logger.info(
                f"Agent completed: {self.name}",
                success=result.success,
                execution_time_ms=execution_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            self.state = AgentState.FAILED
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.error(
                f"Agent timeout: {self.name}",
                timeout_seconds=context.timeout_seconds,
            )

            return AgentResult.failure_result(
                error=f"Agent timed out after {context.timeout_seconds}s",
                error_code="AGENT_TIMEOUT",
                execution_time_ms=execution_time_ms,
                reasoning_trace=context.reasoning_trace,
            )

        except TemplateError as e:
            self.state = AgentState.FAILED
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.error(
                f"Agent error: {self.name}",
                error=str(e),
                error_code=e.error_code,
            )

            return AgentResult.failure_result(
                error=str(e),
                error_code=e.error_code,
                execution_time_ms=execution_time_ms,
                reasoning_trace=context.reasoning_trace,
            )

        except Exception as e:
            self.state = AgentState.FAILED
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.exception(f"Agent unexpected error: {self.name}")

            return AgentResult.failure_result(
                error=str(e),
                error_code="AGENT_ERROR",
                execution_time_ms=execution_time_ms,
                reasoning_trace=context.reasoning_trace,
            )

        finally:
            if self.state == AgentState.EXECUTING:
                self.state = AgentState.COMPLETED

    async def __aenter__(self) -> "AgentBase[InputT, OutputT]":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()


class ReActAgent(AgentBase[InputT, OutputT]):
    """
    Agent using ReAct (Reasoning + Acting) pattern.

    Implements the think-act-observe loop for complex reasoning tasks.
    """

    async def _execute(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        """Execute using ReAct pattern."""
        self.state = AgentState.PLANNING

        while context.iteration < context.max_iterations:
            context.iteration += 1

            # Think
            thought = await self._think(input_data, context)
            context.add_thought(thought)

            if self._is_final_answer(thought):
                # Extract and return final answer
                output = await self._extract_answer(thought, context)
                return AgentResult.success_result(
                    output=output,
                    iterations_used=context.iteration,
                )

            # Act
            action, tool_name, params = await self._decide_action(thought, context)
            context.add_action(action, tool_name, **params)

            # Observe
            self.state = AgentState.EXECUTING
            observation = await self._execute_action(tool_name, params, context)
            context.add_observation(observation)

            self.state = AgentState.PLANNING

        # Max iterations reached
        return AgentResult.failure_result(
            error=f"Max iterations ({context.max_iterations}) reached",
            error_code="MAX_ITERATIONS",
            iterations_used=context.iteration,
        )

    async def _think(self, input_data: InputT, context: AgentContext) -> str:
        """Generate a thought about the current state."""
        # Override in subclass with LLM call
        raise NotImplementedError

    async def _decide_action(
        self,
        thought: str,
        context: AgentContext,
    ) -> tuple[str, str, dict[str, Any]]:
        """Decide on an action based on the thought."""
        # Override in subclass
        raise NotImplementedError

    async def _execute_action(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: AgentContext,
    ) -> str:
        """Execute the decided action."""
        tool = self.get_tool(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found"

        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**params)
            else:
                result = tool(**params)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _is_final_answer(self, thought: str) -> bool:
        """Check if the thought contains a final answer."""
        # Override in subclass for custom logic
        return "FINAL ANSWER:" in thought.upper()

    async def _extract_answer(
        self,
        thought: str,
        context: AgentContext,
    ) -> OutputT:
        """Extract the final answer from the thought."""
        # Override in subclass
        raise NotImplementedError
