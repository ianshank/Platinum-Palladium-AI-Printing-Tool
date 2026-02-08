"""
Unit tests for agents/orchestrator.py module.

Tests workflow management, task execution, and orchestrator functionality.
"""


import pytest

from ptpd_calibration.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorConfig,
    TaskStatus,
    Workflow,
    WorkflowStatus,
    WorkflowTask,
)


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test all task status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestWorkflowTask:
    """Tests for WorkflowTask model."""

    def test_create_task(self):
        """Test creating a workflow task."""
        task = WorkflowTask(
            name="Test Task",
            description="A test task",
            agent_type="planner",
        )
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.agent_type == "planner"
        assert task.status == TaskStatus.PENDING
        assert task.id is not None

    def test_task_defaults(self):
        """Test task default values."""
        task = WorkflowTask(
            name="Test",
            description="Test",
            agent_type="coder",
        )
        assert task.input_data == {}
        assert task.dependencies == []
        assert task.result is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.retries == 0
        assert task.max_retries == 3

    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task = WorkflowTask(
            name="Dependent Task",
            description="Depends on other tasks",
            agent_type="reviewer",
            dependencies=["task-1", "task-2"],
        )
        assert "task-1" in task.dependencies
        assert "task-2" in task.dependencies


class TestWorkflow:
    """Tests for Workflow model."""

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing."""
        task1 = WorkflowTask(
            name="Task 1",
            description="First task",
            agent_type="planner",
        )
        task2 = WorkflowTask(
            name="Task 2",
            description="Second task",
            agent_type="coder",
            dependencies=[task1.id],
        )
        return Workflow(
            name="Test Workflow",
            description="A test workflow",
            tasks=[task1, task2],
        )

    def test_create_workflow(self, simple_workflow):
        """Test creating a workflow."""
        assert simple_workflow.name == "Test Workflow"
        assert simple_workflow.status == WorkflowStatus.PENDING
        assert len(simple_workflow.tasks) == 2
        assert simple_workflow.id is not None

    def test_workflow_defaults(self):
        """Test workflow default values."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
        )
        assert workflow.tasks == []
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.context == {}
        assert workflow.results == {}
        assert workflow.started_at is None
        assert workflow.completed_at is None

    def test_get_ready_tasks_initial(self, simple_workflow):
        """Test getting ready tasks at start."""
        ready = simple_workflow.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].name == "Task 1"

    def test_get_ready_tasks_after_completion(self, simple_workflow):
        """Test ready tasks after first task completes."""
        simple_workflow.tasks[0].status = TaskStatus.COMPLETED
        ready = simple_workflow.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].name == "Task 2"

    def test_progress_initial(self, simple_workflow):
        """Test progress at start."""
        assert simple_workflow.progress == 0.0

    def test_progress_partial(self, simple_workflow):
        """Test progress with one task completed."""
        simple_workflow.tasks[0].status = TaskStatus.COMPLETED
        assert simple_workflow.progress == 0.5

    def test_progress_complete(self, simple_workflow):
        """Test progress when all done."""
        for task in simple_workflow.tasks:
            task.status = TaskStatus.COMPLETED
        assert simple_workflow.progress == 1.0

    def test_progress_empty_workflow(self):
        """Test progress with no tasks."""
        workflow = Workflow(name="Empty", description="Empty workflow")
        assert workflow.progress == 0.0

    def test_is_complete_false(self, simple_workflow):
        """Test is_complete when not done."""
        assert not simple_workflow.is_complete

    def test_is_complete_true(self, simple_workflow):
        """Test is_complete when all done."""
        for task in simple_workflow.tasks:
            task.status = TaskStatus.COMPLETED
        assert simple_workflow.is_complete

    def test_is_complete_with_skipped(self, simple_workflow):
        """Test is_complete counts skipped as done."""
        simple_workflow.tasks[0].status = TaskStatus.COMPLETED
        simple_workflow.tasks[1].status = TaskStatus.SKIPPED
        assert simple_workflow.is_complete

    def test_has_failed_false(self, simple_workflow):
        """Test has_failed when no failures."""
        assert not simple_workflow.has_failed

    def test_has_failed_with_retries_remaining(self, simple_workflow):
        """Test has_failed with retries remaining."""
        simple_workflow.tasks[0].status = TaskStatus.FAILED
        simple_workflow.tasks[0].retries = 1
        simple_workflow.tasks[0].max_retries = 3
        assert not simple_workflow.has_failed

    def test_has_failed_after_max_retries(self, simple_workflow):
        """Test has_failed after max retries."""
        simple_workflow.tasks[0].status = TaskStatus.FAILED
        simple_workflow.tasks[0].retries = 3
        simple_workflow.tasks[0].max_retries = 3
        assert simple_workflow.has_failed


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.max_concurrent_tasks == 3
        assert config.task_timeout_seconds == 300
        assert config.enable_parallel_execution is True
        assert config.retry_failed_tasks is True
        assert config.llm_settings is None
        assert config.agent_settings is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            max_concurrent_tasks=5,
            task_timeout_seconds=600,
            enable_parallel_execution=False,
            retry_failed_tasks=False,
        )
        assert config.max_concurrent_tasks == 5
        assert config.task_timeout_seconds == 600
        assert config.enable_parallel_execution is False
        assert config.retry_failed_tasks is False


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent class."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return OrchestratorAgent()

    def test_create_orchestrator(self, orchestrator):
        """Test creating an orchestrator."""
        assert orchestrator.id is not None
        assert orchestrator.AGENT_TYPE == "orchestrator"
        assert orchestrator.config is not None

    def test_create_with_config(self):
        """Test creating orchestrator with custom config."""
        config = OrchestratorConfig(max_concurrent_tasks=5)
        orchestrator = OrchestratorAgent(config=config)
        assert orchestrator.config.max_concurrent_tasks == 5

    def test_create_development_workflow(self, orchestrator):
        """Test creating development workflow."""
        workflow = orchestrator.create_development_workflow(
            feature_description="Add user authentication",
            include_tests=True,
            include_review=True,
        )
        assert "Development:" in workflow.name
        assert len(workflow.tasks) == 4
        task_types = [t.agent_type for t in workflow.tasks]
        assert "planner" in task_types
        assert "coder" in task_types
        assert "sqa" in task_types
        assert "reviewer" in task_types

    def test_create_development_workflow_no_tests(self, orchestrator):
        """Test development workflow without tests."""
        workflow = orchestrator.create_development_workflow(
            feature_description="Simple feature",
            include_tests=False,
            include_review=True,
        )
        task_types = [t.agent_type for t in workflow.tasks]
        assert "sqa" not in task_types

    def test_create_development_workflow_no_review(self, orchestrator):
        """Test development workflow without review."""
        workflow = orchestrator.create_development_workflow(
            feature_description="Quick fix",
            include_tests=True,
            include_review=False,
        )
        task_types = [t.agent_type for t in workflow.tasks]
        assert "reviewer" not in task_types

    def test_create_review_workflow(self, orchestrator):
        """Test creating review workflow."""
        workflow = orchestrator.create_review_workflow(
            code="def foo(): pass"
        )
        assert workflow.name == "Code Review Workflow"
        assert len(workflow.tasks) == 2
        assert all(t.agent_type == "reviewer" for t in workflow.tasks)

    def test_list_workflows_empty(self, orchestrator):
        """Test listing workflows when empty."""
        workflows = orchestrator.list_workflows()
        assert workflows == []

    def test_get_workflow_not_found(self, orchestrator):
        """Test getting nonexistent workflow."""
        result = orchestrator.get_workflow("nonexistent-id")
        assert result is None

    def test_get_active_agents_empty(self, orchestrator):
        """Test getting active agents when none."""
        agents = orchestrator.get_active_agents()
        assert agents == {}


class TestOrchestratorWorkflowExecution:
    """Tests for workflow execution functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return OrchestratorAgent()

    @pytest.fixture
    def mock_subagent(self):
        """Create a mock subagent."""
        from unittest.mock import AsyncMock, MagicMock

        from ptpd_calibration.agents.subagents.base import SubagentResult, SubagentStatus

        agent = MagicMock()
        agent.id = "mock-agent-1"
        agent.AGENT_TYPE = "planner"
        agent.status = SubagentStatus.IDLE

        # Mock successful run
        async def mock_run(task, _context=None):
            return SubagentResult(
                success=True,
                agent_id=agent.id,
                agent_type="planner",
                task=task,
                result={"plan": "test plan"},
            )

        agent.run = AsyncMock(side_effect=mock_run)
        return agent

    @pytest.mark.asyncio
    async def test_run_workflow_simple(self, orchestrator, mock_subagent):
        """Test running a simple workflow."""
        from unittest.mock import MagicMock

        # Mock the registry to return our mock agent
        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = mock_subagent

        # Create simple workflow
        task = WorkflowTask(
            name="Test Task",
            description="A test task",
            agent_type="planner",
        )
        workflow = Workflow(
            name="Test Workflow",
            description="Test",
            tasks=[task],
        )

        # Run workflow
        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.progress == 1.0
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_workflow_with_dependencies(self, orchestrator, mock_subagent):
        """Test running workflow with task dependencies."""
        from unittest.mock import MagicMock

        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = mock_subagent

        task1 = WorkflowTask(
            name="Task 1",
            description="First task",
            agent_type="planner",
        )
        task2 = WorkflowTask(
            name="Task 2",
            description="Second task",
            agent_type="planner",
            dependencies=[task1.id],
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task1, task2],
        )

        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_workflow_task_failure(self, orchestrator):
        """Test workflow with failing task."""
        from unittest.mock import AsyncMock, MagicMock

        from ptpd_calibration.agents.subagents.base import SubagentResult, SubagentStatus

        failing_agent = MagicMock()
        failing_agent.id = "failing-agent"
        failing_agent.AGENT_TYPE = "planner"
        failing_agent.status = SubagentStatus.IDLE
        failing_agent.run = AsyncMock(
            return_value=SubagentResult(
                success=False,
                agent_id="failing-agent",
                agent_type="planner",
                task="test",
                error="Task failed",
            )
        )

        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = failing_agent

        task = WorkflowTask(
            name="Failing Task",
            description="Will fail",
            agent_type="planner",
            max_retries=0,  # No retries
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task],
        )

        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.FAILED
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_workflow_task_retry(self, orchestrator):
        """Test workflow retries failing tasks."""
        from unittest.mock import AsyncMock, MagicMock

        from ptpd_calibration.agents.subagents.base import SubagentResult, SubagentStatus

        call_count = 0

        async def mock_run_with_retry(task, _context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SubagentResult(
                    success=False,
                    agent_id="retry-agent",
                    agent_type="planner",
                    task=task,
                    error="First attempt failed",
                )
            return SubagentResult(
                success=True,
                agent_id="retry-agent",
                agent_type="planner",
                task=task,
                result={"data": "success on retry"},
            )

        retry_agent = MagicMock()
        retry_agent.id = "retry-agent"
        retry_agent.AGENT_TYPE = "planner"
        retry_agent.status = SubagentStatus.IDLE
        retry_agent.run = AsyncMock(side_effect=mock_run_with_retry)

        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = retry_agent
        orchestrator.config.retry_failed_tasks = True

        task = WorkflowTask(
            name="Retry Task",
            description="Will retry",
            agent_type="planner",
            max_retries=3,
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task],
        )

        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_run_workflow_unknown_agent_type(self, orchestrator):
        """Test workflow fails gracefully for unknown agent type."""
        from unittest.mock import MagicMock

        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = None

        task = WorkflowTask(
            name="Task",
            description="Test",
            agent_type="unknown_type",
            max_retries=0,
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task],
        )

        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.FAILED


class TestOrchestratorParallelExecution:
    """Tests for parallel task execution."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with parallel execution enabled."""
        config = OrchestratorConfig(
            enable_parallel_execution=True,
            max_concurrent_tasks=2,
        )
        return OrchestratorAgent(config=config)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator):
        """Test parallel task execution."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from ptpd_calibration.agents.subagents.base import SubagentResult, SubagentStatus

        execution_order = []

        async def mock_run(task, _context=None):
            execution_order.append(f"start_{task[:10]}")
            await asyncio.sleep(0.01)  # Small delay
            execution_order.append(f"end_{task[:10]}")
            return SubagentResult(
                success=True,
                agent_id="parallel-agent",
                agent_type="planner",
                task=task,
                result={},
            )

        agent = MagicMock()
        agent.id = "parallel-agent"
        agent.AGENT_TYPE = "planner"
        agent.status = SubagentStatus.IDLE
        agent.run = AsyncMock(side_effect=mock_run)

        orchestrator._registry = MagicMock()
        orchestrator._registry.create_agent.return_value = agent

        # Two independent tasks (no dependencies)
        task1 = WorkflowTask(
            name="Task 1",
            description="Parallel 1",
            agent_type="planner",
        )
        task2 = WorkflowTask(
            name="Task 2",
            description="Parallel 2",
            agent_type="planner",
        )
        workflow = Workflow(
            name="Parallel Test",
            description="Test",
            tasks=[task1, task2],
        )

        result = await orchestrator.run_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        # Both tasks should complete
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED


class TestOrchestratorCancelWorkflow:
    """Tests for workflow cancellation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return OrchestratorAgent()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_workflow(self, orchestrator):
        """Test cancelling nonexistent workflow."""
        result = await orchestrator.cancel_workflow("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_non_running_workflow(self, orchestrator):
        """Test cancelling non-running workflow."""
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[],
        )
        workflow.status = WorkflowStatus.PENDING
        orchestrator._workflows[workflow.id] = workflow

        result = await orchestrator.cancel_workflow(workflow.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_running_workflow(self, orchestrator):
        """Test cancelling running workflow."""
        import asyncio

        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[],
        )
        workflow.status = WorkflowStatus.RUNNING
        orchestrator._workflows[workflow.id] = workflow

        # Add a mock running task
        mock_task = asyncio.create_task(asyncio.sleep(10))
        task_obj = WorkflowTask(
            name="Long Task",
            description="Test",
            agent_type="planner",
        )
        workflow.tasks.append(task_obj)
        orchestrator._running_tasks[task_obj.id] = mock_task

        result = await orchestrator.cancel_workflow(workflow.id)

        assert result is True
        assert workflow.status == WorkflowStatus.CANCELLED
        assert workflow.completed_at is not None


class TestOrchestratorListWorkflows:
    """Tests for listing workflows."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return OrchestratorAgent()

    def test_list_workflows_by_status(self, orchestrator):
        """Test listing workflows filtered by status."""
        w1 = Workflow(name="W1", description="Test")
        w1.status = WorkflowStatus.COMPLETED
        w2 = Workflow(name="W2", description="Test")
        w2.status = WorkflowStatus.RUNNING
        w3 = Workflow(name="W3", description="Test")
        w3.status = WorkflowStatus.COMPLETED

        orchestrator._workflows[w1.id] = w1
        orchestrator._workflows[w2.id] = w2
        orchestrator._workflows[w3.id] = w3

        completed = orchestrator.list_workflows(status=WorkflowStatus.COMPLETED)
        assert len(completed) == 2

        running = orchestrator.list_workflows(status=WorkflowStatus.RUNNING)
        assert len(running) == 1


class TestOrchestratorActiveAgents:
    """Tests for active agent management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return OrchestratorAgent()

    def test_get_active_agents_with_agents(self, orchestrator):
        """Test getting active agents when some exist."""
        from unittest.mock import MagicMock

        from ptpd_calibration.agents.subagents.base import SubagentStatus

        mock_agent = MagicMock()
        mock_agent.id = "agent-123"
        mock_agent.AGENT_TYPE = "planner"
        mock_agent.status = SubagentStatus.IDLE

        orchestrator._active_agents["planner"] = mock_agent

        agents = orchestrator.get_active_agents()

        assert "planner" in agents
        assert agents["planner"]["id"] == "agent-123"
        assert agents["planner"]["type"] == "planner"


class TestWorkflowReadyTasksAdvanced:
    """Additional tests for ready tasks logic."""

    def test_get_ready_tasks_all_deps_completed(self):
        """Test task becomes ready when all deps complete."""
        task1 = WorkflowTask(
            name="Task 1",
            description="T1",
            agent_type="planner",
        )
        task2 = WorkflowTask(
            name="Task 2",
            description="T2",
            agent_type="planner",
        )
        task3 = WorkflowTask(
            name="Task 3",
            description="T3",
            agent_type="planner",
            dependencies=[task1.id, task2.id],
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task1, task2, task3],
        )

        # Initially only task1 and task2 should be ready
        ready = workflow.get_ready_tasks()
        assert len(ready) == 2
        assert task3 not in ready

        # Complete task1 only
        task1.status = TaskStatus.COMPLETED
        ready = workflow.get_ready_tasks()
        assert task3 not in ready

        # Complete task2 as well
        task2.status = TaskStatus.COMPLETED
        ready = workflow.get_ready_tasks()
        assert task3 in ready

    def test_get_ready_tasks_skipped_deps_not_ready(self):
        """Test task NOT ready when deps are skipped (only completed counts)."""
        task1 = WorkflowTask(
            name="Task 1",
            description="T1",
            agent_type="planner",
        )
        task2 = WorkflowTask(
            name="Task 2",
            description="T2",
            agent_type="planner",
            dependencies=[task1.id],
        )
        workflow = Workflow(
            name="Test",
            description="Test",
            tasks=[task1, task2],
        )

        # Skip task1 - this should NOT satisfy the dependency
        task1.status = TaskStatus.SKIPPED
        ready = workflow.get_ready_tasks()
        # Task2 should NOT be ready because task1 is skipped (not completed)
        assert task2 not in ready
