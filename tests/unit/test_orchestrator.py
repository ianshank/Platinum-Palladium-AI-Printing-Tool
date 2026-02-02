"""
Unit tests for agents/orchestrator.py module.

Tests workflow management, task execution, and orchestrator functionality.
"""

from datetime import datetime

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
