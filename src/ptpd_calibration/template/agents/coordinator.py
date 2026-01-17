"""
Agent Coordinator System

Provides coordination for multiple agents with:
- Dynamic task routing
- Parallel execution
- Planning and orchestration
- Load balancing
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel, Field

from ptpd_calibration.template.agents.base import (
    AgentBase,
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
)
from ptpd_calibration.template.agents.memory import AgentMemory, MemoryType
from ptpd_calibration.template.errors import TemplateError
from ptpd_calibration.template.logging_config import LogContext, get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class TaskPriority(int, Enum):
    """Task priority levels."""

    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 9


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PlanStepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A task to be executed by an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: str = "default"
    input_data: Any = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # Execution
    assigned_agent: Optional[str] = None
    result: Optional[AgentResult] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_count: int = 0

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PlanStep(BaseModel):
    """A step in an execution plan."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    task_type: str
    input_data: Any = None
    status: PlanStepStatus = PlanStepStatus.PENDING

    # Dependencies
    depends_on: list[str] = Field(default_factory=list)
    parallel_with: list[str] = Field(default_factory=list)

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None


class ExecutionPlan(BaseModel):
    """An execution plan with multiple steps."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_step(
        self,
        name: str,
        task_type: str,
        input_data: Any = None,
        depends_on: Optional[list[str]] = None,
        parallel_with: Optional[list[str]] = None,
    ) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(
            name=name,
            task_type=task_type,
            input_data=input_data,
            depends_on=depends_on or [],
            parallel_with=parallel_with or [],
        )
        self.steps.append(step)
        return step

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute."""
        completed_ids = {
            s.id for s in self.steps
            if s.status == PlanStepStatus.COMPLETED
        }

        ready = []
        for step in self.steps:
            if step.status != PlanStepStatus.PENDING:
                continue

            # Check dependencies
            deps_met = all(
                dep_id in completed_ids
                for dep_id in step.depends_on
            )

            if deps_met:
                ready.append(step)

        return ready

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(
            s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failed(self) -> bool:
        """Check if plan has failed."""
        return any(s.status == PlanStepStatus.FAILED for s in self.steps)


class TaskRouter:
    """
    Routes tasks to appropriate agents.

    Provides:
    - Rule-based routing
    - Load balancing
    - Capability matching
    """

    def __init__(self):
        """Initialize router."""
        self._routes: dict[str, str] = {}  # task_type -> agent_name
        self._rules: list[tuple[Callable[[Task], bool], str]] = []
        self._load: dict[str, int] = {}  # agent_name -> current tasks

    def add_route(self, task_type: str, agent_name: str) -> None:
        """Add a direct route for a task type."""
        self._routes[task_type] = agent_name

    def add_rule(
        self,
        condition: Callable[[Task], bool],
        agent_name: str,
    ) -> None:
        """Add a conditional routing rule."""
        self._rules.append((condition, agent_name))

    def route(self, task: Task, available_agents: list[str]) -> Optional[str]:
        """
        Route a task to an agent.

        Args:
            task: Task to route
            available_agents: List of available agent names

        Returns:
            Agent name or None if no suitable agent
        """
        # Check direct routes
        if task.task_type in self._routes:
            agent = self._routes[task.task_type]
            if agent in available_agents:
                return agent

        # Check rules
        for condition, agent in self._rules:
            if condition(task) and agent in available_agents:
                return agent

        # Load balancing among available agents
        if available_agents:
            # Choose agent with lowest load
            min_load = float("inf")
            best_agent = available_agents[0]

            for agent in available_agents:
                load = self._load.get(agent, 0)
                if load < min_load:
                    min_load = load
                    best_agent = agent

            return best_agent

        return None

    def record_assignment(self, agent_name: str) -> None:
        """Record a task assignment."""
        self._load[agent_name] = self._load.get(agent_name, 0) + 1

    def record_completion(self, agent_name: str) -> None:
        """Record a task completion."""
        if agent_name in self._load:
            self._load[agent_name] = max(0, self._load[agent_name] - 1)


class AgentCoordinator:
    """
    Coordinates multiple agents for complex tasks.

    Provides:
    - Agent lifecycle management
    - Task queue and scheduling
    - Parallel execution
    - Plan orchestration
    - Shared memory

    Usage:
        coordinator = AgentCoordinator()

        # Register agents
        coordinator.register_agent("analyzer", AnalyzerAgent(config))
        coordinator.register_agent("processor", ProcessorAgent(config))

        # Run a single task
        result = await coordinator.run_task(task)

        # Run a plan
        results = await coordinator.execute_plan(plan)
    """

    def __init__(
        self,
        max_parallel_agents: int = 3,
        shared_memory: Optional[AgentMemory] = None,
        router: Optional[TaskRouter] = None,
    ):
        """
        Initialize coordinator.

        Args:
            max_parallel_agents: Maximum agents running in parallel
            shared_memory: Shared memory for all agents
            router: Task router (creates default if None)
        """
        self.max_parallel_agents = max_parallel_agents
        self.memory = shared_memory or AgentMemory()
        self.router = router or TaskRouter()

        # Agent registry
        self._agents: dict[str, AgentBase] = {}
        self._agent_configs: dict[str, AgentConfig] = {}

        # Task management
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_tasks: dict[str, Task] = {}
        self._completed_tasks: dict[str, Task] = {}

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_parallel_agents)
        self._shutdown = False

        self._logger = get_logger("agent.coordinator")

    def register_agent(
        self,
        name: str,
        agent: AgentBase,
        config: Optional[AgentConfig] = None,
    ) -> None:
        """
        Register an agent.

        Args:
            name: Unique agent name
            agent: Agent instance
            config: Optional configuration override
        """
        self._agents[name] = agent
        self._agent_configs[name] = config or agent.config

        # Register route if task_type matches agent name
        self.router.add_route(name, name)

        self._logger.info(f"Registered agent: {name}")

    def unregister_agent(self, name: str) -> Optional[AgentBase]:
        """Unregister an agent."""
        agent = self._agents.pop(name, None)
        self._agent_configs.pop(name, None)
        return agent

    def get_agent(self, name: str) -> Optional[AgentBase]:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List registered agent names."""
        return list(self._agents.keys())

    async def initialize_all(self) -> None:
        """Initialize all registered agents."""
        for name, agent in self._agents.items():
            try:
                await agent.initialize()
                self._logger.debug(f"Initialized agent: {name}")
            except Exception as e:
                self._logger.error(f"Failed to initialize agent {name}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all registered agents."""
        for name, agent in self._agents.items():
            try:
                await agent.cleanup()
            except Exception as e:
                self._logger.error(f"Failed to cleanup agent {name}: {e}")

    async def run_task(
        self,
        task: Task,
        agent_name: Optional[str] = None,
    ) -> AgentResult:
        """
        Run a single task.

        Args:
            task: Task to run
            agent_name: Optional specific agent to use

        Returns:
            Result from agent execution
        """
        # Route task if no agent specified
        if agent_name is None:
            agent_name = self.router.route(task, self.list_agents())

        if agent_name is None:
            return AgentResult.failure_result(
                error="No suitable agent found for task",
                error_code="NO_AGENT",
            )

        agent = self._agents.get(agent_name)
        if agent is None:
            return AgentResult.failure_result(
                error=f"Agent not found: {agent_name}",
                error_code="AGENT_NOT_FOUND",
            )

        # Update task status
        task.assigned_agent = agent_name
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        # Record assignment
        self.router.record_assignment(agent_name)
        self._running_tasks[task.id] = task

        try:
            # Create context
            context = AgentContext(
                task_id=task.id,
                task_type=task.task_type,
                priority=task.priority.value,
                timeout_seconds=task.timeout_seconds,
            )

            # Add to memory
            self.memory.add(
                f"Starting task: {task.name or task.task_type}",
                MemoryType.ACTION,
                task_id=task.id,
            )

            # Run agent
            async with self._semaphore:
                result = await agent.run(task.input_data, context)

            # Update task
            task.result = result
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()

            # Record completion
            self.memory.add(
                f"Completed task: {task.name} - {'success' if result.success else 'failed'}",
                MemoryType.RESULT,
                task_id=task.id,
                importance=0.7,
            )

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()

            self._logger.exception(f"Task execution error: {task.id}")

            return AgentResult.failure_result(
                error=str(e),
                error_code="EXECUTION_ERROR",
            )

        finally:
            self.router.record_completion(agent_name)
            self._running_tasks.pop(task.id, None)
            self._completed_tasks[task.id] = task

    async def run_parallel(
        self,
        tasks: list[Task],
    ) -> list[AgentResult]:
        """
        Run multiple tasks in parallel.

        Args:
            tasks: Tasks to run

        Returns:
            List of results in same order as tasks
        """
        # Sort by priority
        tasks_sorted = sorted(tasks, key=lambda t: t.priority.value)

        # Run in parallel
        coroutines = [self.run_task(task) for task in tasks_sorted]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to failure results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    AgentResult.failure_result(
                        error=str(result),
                        error_code="PARALLEL_EXECUTION_ERROR",
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def execute_plan(
        self,
        plan: ExecutionPlan,
    ) -> dict[str, AgentResult]:
        """
        Execute an execution plan.

        Handles dependencies and parallel execution.

        Args:
            plan: Plan to execute

        Returns:
            Dictionary of step_id -> result
        """
        plan.status = TaskStatus.RUNNING
        results: dict[str, AgentResult] = {}

        self._logger.info(f"Executing plan: {plan.name} ({len(plan.steps)} steps)")

        self.memory.add(
            f"Starting plan: {plan.name}",
            MemoryType.ACTION,
            importance=0.8,
        )

        while not plan.is_complete() and not plan.has_failed():
            # Get ready steps
            ready = plan.get_ready_steps()

            if not ready:
                # No ready steps but plan not complete - deadlock
                self._logger.error("Plan deadlock detected")
                break

            # Group parallel steps
            parallel_groups: list[list[PlanStep]] = []
            current_group: list[PlanStep] = []

            for step in ready:
                if not current_group:
                    current_group.append(step)
                elif any(s.id in step.parallel_with for s in current_group):
                    current_group.append(step)
                else:
                    parallel_groups.append(current_group)
                    current_group = [step]

            if current_group:
                parallel_groups.append(current_group)

            # Execute each group
            for group in parallel_groups:
                if len(group) == 1:
                    # Single step
                    step = group[0]
                    step.status = PlanStepStatus.RUNNING

                    task = Task(
                        name=step.name,
                        task_type=step.task_type,
                        input_data=step.input_data,
                    )

                    result = await self.run_task(task)
                    results[step.id] = result

                    if result.success:
                        step.status = PlanStepStatus.COMPLETED
                        step.result = result.output
                    else:
                        step.status = PlanStepStatus.FAILED
                        step.error = result.error

                else:
                    # Parallel steps
                    tasks = []
                    for step in group:
                        step.status = PlanStepStatus.RUNNING
                        tasks.append(
                            Task(
                                name=step.name,
                                task_type=step.task_type,
                                input_data=step.input_data,
                            )
                        )

                    group_results = await self.run_parallel(tasks)

                    for step, result in zip(group, group_results):
                        results[step.id] = result

                        if result.success:
                            step.status = PlanStepStatus.COMPLETED
                            step.result = result.output
                        else:
                            step.status = PlanStepStatus.FAILED
                            step.error = result.error

        # Update plan status
        if plan.has_failed():
            plan.status = TaskStatus.FAILED
        elif plan.is_complete():
            plan.status = TaskStatus.COMPLETED
        else:
            plan.status = TaskStatus.FAILED  # Deadlock

        self.memory.add(
            f"Completed plan: {plan.name} - {plan.status.value}",
            MemoryType.RESULT,
            importance=0.8,
        )

        self._logger.info(f"Plan completed: {plan.name} - {plan.status.value}")

        return results

    def create_plan(
        self,
        name: str,
        description: str = "",
    ) -> ExecutionPlan:
        """Create a new execution plan."""
        return ExecutionPlan(name=name, description=description)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        if task_id in self._running_tasks:
            return self._running_tasks[task_id].status
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "agents": {
                name: {
                    "state": agent.state.value,
                    "config": {
                        "timeout": config.timeout_seconds,
                        "max_iterations": config.max_iterations,
                    },
                }
                for name, (agent, config) in zip(
                    self._agents.keys(),
                    zip(self._agents.values(), self._agent_configs.values()),
                )
            },
            "tasks": {
                "running": len(self._running_tasks),
                "completed": len(self._completed_tasks),
            },
            "memory": self.memory.get_stats(),
        }

    async def __aenter__(self) -> "AgentCoordinator":
        """Async context manager entry."""
        await self.initialize_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.cleanup_all()
