"""
Orchestrator agent for coordinating multi-agent workflows.

Coordinates Planner, SQE, Coder, and Reviewer subagents.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ptpd_calibration.agents.communication import (
    AgentMessage,
    ConversationManager,
    MessageBus,
    MessageHandler,
    MessagePriority,
    MessageType,
    get_message_bus,
)
from ptpd_calibration.agents.logging import AgentLogger, EventType, LogContext, get_agent_logger
from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
    get_subagent_registry,
)
from ptpd_calibration.config import AgentSettings, LLMSettings, get_settings


class WorkflowStatus(str, Enum):
    """Status of a workflow."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of a task within a workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowTask(BaseModel):
    """A task within a workflow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    agent_type: str  # Which subagent to use
    input_data: dict = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)  # Task IDs
    status: TaskStatus = TaskStatus.PENDING
    result: SubagentResult | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    max_retries: int = 3

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class Workflow(BaseModel):
    """A multi-agent workflow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    tasks: list[WorkflowTask] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    context: dict = Field(default_factory=dict)  # Shared context for tasks
    results: dict = Field(default_factory=dict)  # Task ID -> result

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_ready_tasks(self) -> list[WorkflowTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        completed_ids = {
            t.id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and all(d in completed_ids for d in t.dependencies)
        ]

    @property
    def progress(self) -> float:
        """Get workflow progress (0-1)."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    @property
    def has_failed(self) -> bool:
        """Check if workflow has failed."""
        return any(
            t.status == TaskStatus.FAILED and t.retries >= t.max_retries
            for t in self.tasks
        )


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_concurrent_tasks: int = Field(default=3, ge=1, le=10)
    task_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    enable_parallel_execution: bool = True
    retry_failed_tasks: bool = True
    llm_settings: LLMSettings | None = None
    agent_settings: AgentSettings | None = None


class OrchestratorAgent:
    """
    Orchestrator for coordinating multi-agent workflows.

    Manages the execution of complex tasks by delegating to
    specialized subagents (Planner, SQE, Coder, Reviewer).
    """

    AGENT_TYPE = "orchestrator"

    def __init__(self, config: OrchestratorConfig | None = None):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self.id = str(uuid4())
        self.config = config or OrchestratorConfig()
        self.status = SubagentStatus.IDLE

        # Components
        self._registry = get_subagent_registry()
        self._message_bus = get_message_bus()
        self._conversation_manager = ConversationManager()

        # State
        self._workflows: dict[str, Workflow] = {}
        self._active_agents: dict[str, BaseSubagent] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

        # Logging
        self._logger = get_agent_logger()
        self._context = LogContext(
            agent_id=self.id,
            agent_type=self.AGENT_TYPE,
        )
        self._logger.set_context(self._context)

        # Register message handler
        self._register_message_handler()

    def _register_message_handler(self) -> None:
        """Register with message bus."""
        handler = MessageHandler(
            agent_id=self.id,
            agent_type=self.AGENT_TYPE,
            handler_func=self._handle_message,
        )
        self._message_bus.register_handler(handler)

    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming messages."""
        self._logger.debug(
            f"Received message: {message.action}",
            data={"from": message.sender_type},
        )

        if message.message_type == MessageType.RESPONSE:
            # Handle subagent response
            self._logger.debug(
                f"Subagent response received",
                data={"correlation_id": message.correlation_id},
            )

    async def run_workflow(self, workflow: Workflow) -> Workflow:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute.

        Returns:
            Completed workflow with results.
        """
        self._logger.info(
            f"Starting workflow: {workflow.name}",
            event_type=EventType.AGENT_STARTED,
            data={
                "workflow_id": workflow.id,
                "num_tasks": len(workflow.tasks),
            },
        )

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        self._workflows[workflow.id] = workflow
        self.status = SubagentStatus.RUNNING

        try:
            while not workflow.is_complete and not workflow.has_failed:
                ready_tasks = workflow.get_ready_tasks()

                if not ready_tasks:
                    # No tasks ready, check if blocked
                    running = [t for t in workflow.tasks if t.status == TaskStatus.RUNNING]
                    if not running:
                        # Deadlock or all done
                        break
                    # Wait for running tasks
                    await asyncio.sleep(0.1)
                    continue

                # Execute ready tasks
                if self.config.enable_parallel_execution:
                    # Run up to max_concurrent_tasks in parallel
                    batch = ready_tasks[: self.config.max_concurrent_tasks]
                    await self._execute_tasks_parallel(workflow, batch)
                else:
                    # Run sequentially
                    for task in ready_tasks:
                        await self._execute_task(workflow, task)

            # Determine final status
            if workflow.has_failed:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED

        except Exception as e:
            self._logger.error(
                f"Workflow failed: {e}",
                event_type=EventType.AGENT_FAILED,
                error=str(e),
            )
            workflow.status = WorkflowStatus.FAILED

        finally:
            workflow.completed_at = datetime.now()
            self.status = SubagentStatus.COMPLETED if workflow.status == WorkflowStatus.COMPLETED else SubagentStatus.FAILED

        duration_ms = (workflow.completed_at - workflow.started_at).total_seconds() * 1000

        self._logger.info(
            f"Workflow completed: {workflow.status.value}",
            event_type=EventType.AGENT_COMPLETED,
            duration_ms=duration_ms,
            data={
                "workflow_id": workflow.id,
                "progress": workflow.progress,
                "failed_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.FAILED]),
            },
        )

        return workflow

    async def _execute_tasks_parallel(
        self,
        workflow: Workflow,
        tasks: list[WorkflowTask],
    ) -> None:
        """Execute multiple tasks in parallel."""
        async_tasks = []
        for task in tasks:
            async_task = asyncio.create_task(self._execute_task(workflow, task))
            self._running_tasks[task.id] = async_task
            async_tasks.append(async_task)

        # Wait for all tasks
        await asyncio.gather(*async_tasks, return_exceptions=True)

        # Cleanup
        for task in tasks:
            self._running_tasks.pop(task.id, None)

    async def _execute_task(
        self,
        workflow: Workflow,
        task: WorkflowTask,
    ) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        self._logger.info(
            f"Starting task: {task.name}",
            event_type=EventType.PLAN_STEP_STARTED,
            data={
                "task_id": task.id,
                "agent_type": task.agent_type,
            },
        )

        try:
            # Get or create subagent
            agent = await self._get_agent(task.agent_type)
            if agent is None:
                raise ValueError(f"Unknown agent type: {task.agent_type}")

            # Build task context with workflow context and dependency results
            task_context = {**workflow.context}
            for dep_id in task.dependencies:
                if dep_id in workflow.results:
                    task_context[f"dep_{dep_id}"] = workflow.results[dep_id]

            # Execute with timeout
            result = await asyncio.wait_for(
                agent.run(task.description, task_context),
                timeout=self.config.task_timeout_seconds,
            )

            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result
                workflow.results[task.id] = result.result
            else:
                raise Exception(result.error or "Task failed")

        except asyncio.TimeoutError:
            self._logger.warning(
                f"Task timeout: {task.name}",
                data={"task_id": task.id},
            )
            task.status = TaskStatus.FAILED
            task.result = SubagentResult(
                success=False,
                agent_id="",
                agent_type=task.agent_type,
                task=task.description,
                error="Task timed out",
            )

        except Exception as e:
            self._logger.error(
                f"Task failed: {task.name}",
                error=str(e),
                data={"task_id": task.id},
            )

            task.retries += 1
            if self.config.retry_failed_tasks and task.retries < task.max_retries:
                task.status = TaskStatus.PENDING
                self._logger.info(f"Retrying task: {task.name} (attempt {task.retries + 1})")
            else:
                task.status = TaskStatus.FAILED
                task.result = SubagentResult(
                    success=False,
                    agent_id="",
                    agent_type=task.agent_type,
                    task=task.description,
                    error=str(e),
                )

        finally:
            task.completed_at = datetime.now()

    async def _get_agent(self, agent_type: str) -> BaseSubagent | None:
        """Get or create a subagent."""
        if agent_type in self._active_agents:
            return self._active_agents[agent_type]

        # Create subagent config
        subagent_config = SubagentConfig(
            llm_settings=self.config.llm_settings,
            agent_settings=self.config.agent_settings,
            parent_context=self._context,
        )

        # Create agent
        agent = self._registry.create_agent(agent_type, subagent_config)
        if agent:
            self._active_agents[agent_type] = agent

        return agent

    def create_development_workflow(
        self,
        feature_description: str,
        include_tests: bool = True,
        include_review: bool = True,
    ) -> Workflow:
        """
        Create a standard development workflow.

        Args:
            feature_description: Description of feature to implement.
            include_tests: Include test generation.
            include_review: Include code review.

        Returns:
            Configured Workflow.
        """
        tasks = []

        # 1. Planning task
        plan_task = WorkflowTask(
            name="Create Implementation Plan",
            description=f"Create a detailed implementation plan for: {feature_description}",
            agent_type="planner",
        )
        tasks.append(plan_task)

        # 2. Coding task (depends on planning)
        code_task = WorkflowTask(
            name="Implement Feature",
            description=f"Implement the feature according to the plan: {feature_description}",
            agent_type="coder",
            dependencies=[plan_task.id],
        )
        tasks.append(code_task)

        # 3. Testing task (depends on coding)
        if include_tests:
            test_task = WorkflowTask(
                name="Generate Tests",
                description="Generate comprehensive tests for the implementation",
                agent_type="sqa",
                dependencies=[code_task.id],
            )
            tasks.append(test_task)

        # 4. Review task (depends on coding and optionally testing)
        if include_review:
            review_deps = [code_task.id]
            if include_tests:
                review_deps.append(test_task.id)

            review_task = WorkflowTask(
                name="Code Review",
                description="Review the implementation for quality, security, and best practices",
                agent_type="reviewer",
                dependencies=review_deps,
            )
            tasks.append(review_task)

        return Workflow(
            name=f"Development: {feature_description[:50]}",
            description=feature_description,
            tasks=tasks,
            context={"feature_description": feature_description},
        )

    def create_review_workflow(self, code: str) -> Workflow:
        """
        Create a code review workflow.

        Args:
            code: Code to review.

        Returns:
            Configured Workflow.
        """
        return Workflow(
            name="Code Review Workflow",
            description="Review code for quality and security",
            tasks=[
                WorkflowTask(
                    name="Security Scan",
                    description="Scan code for security vulnerabilities",
                    agent_type="reviewer",
                    input_data={"code": code, "focus_areas": ["security"]},
                ),
                WorkflowTask(
                    name="Quality Review",
                    description="Review code quality and maintainability",
                    agent_type="reviewer",
                    input_data={"code": code, "focus_areas": ["quality", "maintainability"]},
                ),
            ],
            context={"code": code},
        )

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self, status: WorkflowStatus | None = None) -> list[Workflow]:
        """List workflows, optionally filtered by status."""
        workflows = list(self._workflows.values())
        if status:
            workflows = [w for w in workflows if w.status == status]
        return workflows

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of workflow to cancel.

        Returns:
            True if cancelled, False if not found/not running.
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow or workflow.status != WorkflowStatus.RUNNING:
            return False

        # Cancel running tasks
        for task in workflow.tasks:
            if task.id in self._running_tasks:
                self._running_tasks[task.id].cancel()

        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()

        self._logger.info(
            f"Workflow cancelled: {workflow_id}",
            data={"workflow_name": workflow.name},
        )

        return True

    def get_active_agents(self) -> dict[str, dict]:
        """Get information about active agents."""
        return {
            agent_type: {
                "id": agent.id,
                "type": agent.AGENT_TYPE,
                "status": agent.status.value,
            }
            for agent_type, agent in self._active_agents.items()
        }


# Convenience function for quick orchestration
async def orchestrate_development(
    feature_description: str,
    config: OrchestratorConfig | None = None,
) -> Workflow:
    """
    Quick function to orchestrate feature development.

    Args:
        feature_description: What to implement.
        config: Optional orchestrator config.

    Returns:
        Completed workflow with results.
    """
    orchestrator = OrchestratorAgent(config)
    workflow = orchestrator.create_development_workflow(feature_description)
    return await orchestrator.run_workflow(workflow)
