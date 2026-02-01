"""
Unit tests for the subagent system.

Tests for BaseSubagent, SubagentRegistry, and specialized subagents.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
    get_subagent_registry,
    register_subagent,
)
from ptpd_calibration.agents.subagents.coder import (
    CoderAgent,
    CodeFile,
    CodeChange,
    ImplementationResult,
)
from ptpd_calibration.agents.subagents.planner import (
    AcceptanceCriteria,
    Epic,
    ImplementationPlan,
    Milestone,
    PlannerAgent,
    Story,
)
from ptpd_calibration.agents.subagents.reviewer import (
    CodeIssue,
    CodeReview,
    IssueSeverity,
    IssueCategory,
    ReviewerAgent,
    ReviewScore,
)
from ptpd_calibration.agents.subagents.sqa import (
    GeneratedTest,
    SQEAgent,
    TestCase,
    TestPlan,
    TestSuite,
)


# =============================================================================
# SubagentResult Tests
# =============================================================================


class TestSubagentResult:
    """Tests for SubagentResult model."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = SubagentResult(
            success=True,
            agent_id="test-agent",
            agent_type="planner",
            task="Create plan",
            result={"plan": "test"},
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = SubagentResult(
            success=False,
            agent_id="test-agent",
            agent_type="planner",
            task="Create plan",
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_with_artifacts(self):
        """Test result with artifacts."""
        result = SubagentResult(
            success=True,
            agent_id="test-agent",
            agent_type="coder",
            task="Generate code",
            artifacts=[
                {"type": "code_file", "filename": "test.py", "content": "print('hello')"}
            ],
        )
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["type"] == "code_file"


# =============================================================================
# SubagentRegistry Tests
# =============================================================================


class TestSubagentRegistry:
    """Tests for SubagentRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return SubagentRegistry()

    def test_register_agent(self, registry):
        """Test registering an agent class."""
        registry.register(PlannerAgent)
        assert "planner" in registry.list_agent_types()

    def test_get_agent_class(self, registry):
        """Test getting an agent class."""
        registry.register(PlannerAgent)
        agent_class = registry.get_agent_class("planner")
        assert agent_class == PlannerAgent

    def test_get_nonexistent_agent(self, registry):
        """Test getting a nonexistent agent class."""
        result = registry.get_agent_class("nonexistent")
        assert result is None

    def test_create_agent(self, registry):
        """Test creating an agent instance."""
        registry.register(PlannerAgent)
        agent = registry.create_agent("planner")
        assert agent is not None
        assert isinstance(agent, PlannerAgent)

    def test_create_unknown_agent(self, registry):
        """Test creating an unknown agent type."""
        agent = registry.create_agent("unknown")
        assert agent is None

    def test_find_by_capability(self, registry):
        """Test finding agents by capability."""
        registry.register(PlannerAgent)
        registry.register(CoderAgent)
        registry.register(ReviewerAgent)

        # Find agents with ANALYSIS capability
        analysis_agents = registry.find_by_capability(SubagentCapability.ANALYSIS)
        assert len(analysis_agents) >= 2  # Planner and Reviewer

    def test_get_agent_info(self, registry):
        """Test getting agent information."""
        registry.register(PlannerAgent)
        registry.register(SQEAgent)

        info = registry.get_agent_info()
        assert len(info) == 2
        types = [i["type"] for i in info]
        assert "planner" in types
        assert "sqa" in types

    def test_unregister(self, registry):
        """Test unregistering an agent."""
        registry.register(PlannerAgent)
        assert "planner" in registry.list_agent_types()

        result = registry.unregister("planner")
        assert result is True
        assert "planner" not in registry.list_agent_types()

    def test_get_instance(self, registry):
        """Test getting an existing instance."""
        registry.register(PlannerAgent)
        agent = registry.create_agent("planner")
        agent_id = agent.id

        retrieved = registry.get_instance(agent_id)
        assert retrieved is agent


# =============================================================================
# PlannerAgent Tests
# =============================================================================


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.fixture
    def planner(self):
        """Create a planner agent."""
        return PlannerAgent()

    def test_planner_initialization(self, planner):
        """Test planner initialization."""
        assert planner.AGENT_TYPE == "planner"
        assert SubagentCapability.PLANNING in planner.capabilities()

    def test_planner_capabilities(self, planner):
        """Test planner capabilities."""
        caps = planner.capabilities()
        assert SubagentCapability.PLANNING in caps
        assert SubagentCapability.ANALYSIS in caps

    @pytest.mark.asyncio
    async def test_planner_run_with_mock(self, planner):
        """Test planner run with mocked LLM."""
        mock_response = '''
        {
            "goal": "Test feature",
            "summary": "Test summary",
            "milestones": [
                {
                    "id": "M1",
                    "title": "Implementation",
                    "goal": "Build feature",
                    "epics": []
                }
            ],
            "architectural_decisions": [],
            "risks": [],
            "assumptions": []
        }
        '''
        planner._client = AsyncMock()
        planner._client.complete = AsyncMock(return_value=mock_response)

        result = await planner.run("Create a test feature")

        assert result.success is True
        assert result.agent_type == "planner"

    def test_fallback_plan_creation(self, planner):
        """Test fallback plan creation."""
        plan = planner._create_fallback_plan("Test task")
        assert plan.goal == "Test task"
        assert len(plan.milestones) > 0


# =============================================================================
# SQEAgent Tests
# =============================================================================


class TestSQEAgent:
    """Tests for SQEAgent."""

    @pytest.fixture
    def sqa(self):
        """Create an SQE agent."""
        return SQEAgent()

    def test_sqa_initialization(self, sqa):
        """Test SQE initialization."""
        assert sqa.AGENT_TYPE == "sqa"
        assert SubagentCapability.TESTING in sqa.capabilities()

    @pytest.mark.asyncio
    async def test_sqa_run_creates_test_plan(self, sqa):
        """Test SQE run creates test plan."""
        mock_response = '''
        {
            "title": "Test Plan",
            "scope": "Testing",
            "objectives": ["Verify functionality"],
            "test_suites": [],
            "entry_criteria": [],
            "exit_criteria": []
        }
        '''
        sqa._client = AsyncMock()
        sqa._client.complete = AsyncMock(return_value=mock_response)

        result = await sqa.run("Test the login feature")

        assert result.success is True
        assert result.agent_type == "sqa"

    def test_extract_imports(self, sqa):
        """Test extracting imports from code."""
        code = """
import pytest
from unittest.mock import MagicMock
from mymodule import MyClass

def test_something():
    pass
"""
        imports = sqa._extract_imports(code)
        assert "import pytest" in imports
        assert "from unittest.mock import MagicMock" in imports

    def test_extract_markers(self, sqa):
        """Test extracting pytest markers."""
        code = """
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass
"""
        markers = sqa._extract_markers(code)
        assert "unit" in markers
        assert "integration" in markers


# =============================================================================
# CoderAgent Tests
# =============================================================================


class TestCoderAgent:
    """Tests for CoderAgent."""

    @pytest.fixture
    def coder(self):
        """Create a coder agent."""
        return CoderAgent()

    def test_coder_initialization(self, coder):
        """Test coder initialization."""
        assert coder.AGENT_TYPE == "coder"
        assert SubagentCapability.CODING in coder.capabilities()

    @pytest.mark.asyncio
    async def test_coder_run_generates_code(self, coder):
        """Test coder run generates code."""
        mock_response = '''
```python
def hello_world():
    """Print hello world."""
    print("Hello, World!")
```
'''
        coder._client = AsyncMock()
        coder._client.complete = AsyncMock(return_value=mock_response)

        result = await coder.run("Create a hello world function")

        assert result.success is True
        assert result.agent_type == "coder"

    def test_extract_code_from_markdown(self, coder):
        """Test extracting code from markdown."""
        response = '''
Here's the code:
```python
def test():
    pass
```
'''
        code = coder._extract_code(response)
        assert "def test():" in code

    def test_extract_classes(self, coder):
        """Test extracting class names."""
        code = """
class MyClass:
    pass

class AnotherClass(BaseClass):
    pass
"""
        classes = coder._extract_classes(code)
        assert "MyClass" in classes
        assert "AnotherClass" in classes

    def test_extract_functions(self, coder):
        """Test extracting function names."""
        code = """
def my_function():
    pass

def another_function(arg):
    return arg
"""
        functions = coder._extract_functions(code)
        assert "my_function" in functions
        assert "another_function" in functions


# =============================================================================
# ReviewerAgent Tests
# =============================================================================


class TestReviewerAgent:
    """Tests for ReviewerAgent."""

    @pytest.fixture
    def reviewer(self):
        """Create a reviewer agent."""
        return ReviewerAgent()

    def test_reviewer_initialization(self, reviewer):
        """Test reviewer initialization."""
        assert reviewer.AGENT_TYPE == "reviewer"
        assert SubagentCapability.REVIEWING in reviewer.capabilities()

    @pytest.mark.asyncio
    async def test_reviewer_run(self, reviewer):
        """Test reviewer run."""
        mock_response = '''
        {
            "summary": "Good code overall",
            "overall_score": 8.0,
            "scores": [],
            "issues": [],
            "strengths": ["Clear naming"],
            "recommendations": [],
            "security_concerns": [],
            "approved": true
        }
        '''
        reviewer._client = AsyncMock()
        reviewer._client.complete = AsyncMock(return_value=mock_response)

        result = await reviewer.run("Review this code", context={"code": "def test(): pass"})

        assert result.success is True
        assert result.agent_type == "reviewer"

    def test_fallback_review(self, reviewer):
        """Test fallback review creation."""
        code = """
try:
    risky_operation()
except:
    pass  # TODO: fix this
"""
        review = reviewer._create_fallback_review(code)
        assert review.summary != ""
        assert len(review.issues) > 0  # Should detect bare except and TODO

    def test_fallback_review_detects_secrets(self, reviewer):
        """Test fallback review detects potential secrets."""
        code = 'password = "secret123"'
        review = reviewer._create_fallback_review(code)
        security_issues = [i for i in review.issues if i.category == IssueCategory.SECURITY]
        assert len(security_issues) > 0

    def test_get_issue_statistics(self, reviewer):
        """Test issue statistics."""
        # Add some reviews to history
        reviewer._review_history = [
            CodeReview(
                summary="Test",
                overall_score=7.0,
                issues=[
                    CodeIssue(
                        id="I1",
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.MAINTAINABILITY,
                        title="Test issue",
                        description="Test",
                    )
                ],
                approved=True,
            )
        ]

        stats = reviewer.get_issue_statistics()
        assert stats["total_reviews"] == 1
        assert stats["total_issues"] == 1
        assert stats["approval_rate"] == 1.0


# =============================================================================
# Model Tests
# =============================================================================


class TestPlannerModels:
    """Tests for Planner Pydantic models."""

    def test_acceptance_criteria(self):
        """Test AcceptanceCriteria model."""
        ac = AcceptanceCriteria(
            id="AC1",
            description="Feature works correctly",
            testable=True,
            automated=True,
        )
        assert ac.id == "AC1"
        assert ac.testable is True

    def test_story(self):
        """Test Story model."""
        story = Story(
            id="S1",
            title="Implement feature",
            description="Build the feature",
            acceptance_criteria=[
                AcceptanceCriteria(id="AC1", description="Works")
            ],
            estimated_complexity="medium",
        )
        assert story.id == "S1"
        assert len(story.acceptance_criteria) == 1

    def test_epic(self):
        """Test Epic model."""
        epic = Epic(
            id="E1",
            title="Core Feature",
            description="Main feature implementation",
            stories=[
                Story(id="S1", title="Task 1", description="Do task 1")
            ],
            priority=5,
        )
        assert epic.id == "E1"
        assert epic.priority == 5

    def test_milestone(self):
        """Test Milestone model."""
        milestone = Milestone(
            id="M1",
            title="Phase 1",
            goal="Complete foundation",
            epics=[
                Epic(id="E1", title="Setup", description="Initial setup")
            ],
            success_metrics=["All tests pass"],
        )
        assert milestone.id == "M1"
        assert len(milestone.success_metrics) == 1

    def test_implementation_plan(self):
        """Test ImplementationPlan model."""
        plan = ImplementationPlan(
            goal="Build new feature",
            summary="Implementation plan summary",
            milestones=[
                Milestone(id="M1", title="Phase 1", goal="Foundation")
            ],
            architectural_decisions=["Use microservices"],
            risks=["Tight timeline"],
            assumptions=["API is stable"],
        )
        assert plan.goal == "Build new feature"
        assert len(plan.milestones) == 1


class TestSQAModels:
    """Tests for SQA Pydantic models."""

    def test_test_case(self):
        """Test TestCase model."""
        tc = TestCase(
            id="TC1",
            name="test_login",
            description="Test login functionality",
            test_type="unit",
            steps=["Enter credentials", "Click login"],
            expected_result="User is logged in",
            priority="high",
        )
        assert tc.id == "TC1"
        assert tc.test_type == "unit"

    def test_test_suite(self):
        """Test TestSuite model."""
        suite = TestSuite(
            id="TS1",
            name="Auth Tests",
            description="Authentication test suite",
            test_cases=[
                TestCase(id="TC1", name="test_login", description="Login test")
            ],
            fixtures=["auth_client"],
        )
        assert suite.id == "TS1"
        assert len(suite.test_cases) == 1

    def test_test_plan(self):
        """Test TestPlan model."""
        plan = TestPlan(
            title="Feature Test Plan",
            scope="Login feature",
            objectives=["Verify login works"],
            test_suites=[
                TestSuite(id="TS1", name="Unit Tests", description="Unit tests")
            ],
            coverage_target=90.0,
        )
        assert plan.title == "Feature Test Plan"
        assert plan.coverage_target == 90.0


class TestCoderModels:
    """Tests for Coder Pydantic models."""

    def test_code_file(self):
        """Test CodeFile model."""
        code_file = CodeFile(
            filename="test.py",
            filepath="src/test.py",
            content="def hello(): pass",
            language="python",
            imports=["import os"],
            classes=[],
            functions=["hello"],
        )
        assert code_file.filename == "test.py"
        assert "hello" in code_file.functions

    def test_code_change(self):
        """Test CodeChange model."""
        change = CodeChange(
            file_path="src/test.py",
            change_type="modify",
            description="Added error handling",
            old_content="def hello(): pass",
            new_content="def hello():\n    try:\n        pass\n    except: pass",
        )
        assert change.change_type == "modify"
        assert change.old_content is not None

    def test_implementation_result(self):
        """Test ImplementationResult model."""
        result = ImplementationResult(
            success=True,
            files=[CodeFile(filename="test.py", filepath="src/test.py", content="")],
            notes=["Generated successfully"],
        )
        assert result.success is True
        assert len(result.files) == 1


class TestReviewerModels:
    """Tests for Reviewer Pydantic models."""

    def test_code_issue(self):
        """Test CodeIssue model."""
        issue = CodeIssue(
            id="I1",
            severity=IssueSeverity.HIGH,
            category=IssueCategory.SECURITY,
            title="SQL Injection Risk",
            description="Query uses string concatenation",
            file_path="src/db.py",
            line_number=42,
            suggestion="Use parameterized queries",
        )
        assert issue.severity == IssueSeverity.HIGH
        assert issue.category == IssueCategory.SECURITY

    def test_review_score(self):
        """Test ReviewScore model."""
        score = ReviewScore(
            dimension="Security",
            score=8.5,
            notes="Good security practices",
        )
        assert score.dimension == "Security"
        assert score.score == 8.5

    def test_code_review(self):
        """Test CodeReview model."""
        review = CodeReview(
            summary="Good code quality",
            overall_score=8.0,
            scores=[ReviewScore(dimension="Security", score=9.0)],
            issues=[],
            strengths=["Clear naming", "Good error handling"],
            recommendations=["Add more tests"],
            approved=True,
        )
        assert review.overall_score == 8.0
        assert review.approved is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestSubagentIntegration:
    """Integration tests for subagent system."""

    def test_registry_with_all_agents(self):
        """Test registry with all agent types."""
        registry = SubagentRegistry()
        registry.register(PlannerAgent)
        registry.register(SQEAgent)
        registry.register(CoderAgent)
        registry.register(ReviewerAgent)

        assert len(registry.list_agent_types()) == 4

        # Test creating each type
        for agent_type in ["planner", "sqa", "coder", "reviewer"]:
            agent = registry.create_agent(agent_type)
            assert agent is not None
            assert agent.status == SubagentStatus.IDLE

    def test_capability_coverage(self):
        """Test that all capabilities are covered."""
        registry = SubagentRegistry()
        registry.register(PlannerAgent)
        registry.register(SQEAgent)
        registry.register(CoderAgent)
        registry.register(ReviewerAgent)

        required_capabilities = [
            SubagentCapability.PLANNING,
            SubagentCapability.TESTING,
            SubagentCapability.CODING,
            SubagentCapability.REVIEWING,
        ]

        for cap in required_capabilities:
            agents = registry.find_by_capability(cap)
            assert len(agents) > 0, f"No agent for capability: {cap}"
