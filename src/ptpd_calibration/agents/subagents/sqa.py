"""
Software Quality Engineering (SQE) subagent for test generation and validation.

Creates test plans, generates test cases, and validates implementations.
"""


from pydantic import BaseModel, Field

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentResult,
    register_subagent,
)
from ptpd_calibration.agents.utils import (
    extract_code_block,
    extract_imports,
    parse_json_response,
)


class TestCase(BaseModel):
    """A single test case."""

    id: str
    name: str
    description: str
    test_type: str = "unit"  # unit, integration, e2e
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    expected_result: str = ""
    actual_result: str | None = None
    status: str = "pending"  # pending, passed, failed, skipped
    priority: str = "medium"  # critical, high, medium, low
    tags: list[str] = Field(default_factory=list)


class TestSuite(BaseModel):
    """Collection of related test cases."""

    id: str
    name: str
    description: str
    test_cases: list[TestCase] = Field(default_factory=list)
    setup_steps: list[str] = Field(default_factory=list)
    teardown_steps: list[str] = Field(default_factory=list)
    fixtures: list[str] = Field(default_factory=list)


class TestPlan(BaseModel):
    """Complete test plan for a feature or component."""

    title: str
    scope: str
    objectives: list[str] = Field(default_factory=list)
    test_suites: list[TestSuite] = Field(default_factory=list)
    test_environment: dict = Field(default_factory=dict)
    entry_criteria: list[str] = Field(default_factory=list)
    exit_criteria: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    coverage_target: float = 80.0


class GeneratedTest(BaseModel):
    """Generated pytest test code."""

    filename: str
    content: str
    imports: list[str] = Field(default_factory=list)
    fixtures_needed: list[str] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)


SQE_SYSTEM_PROMPT = """You are an expert Software Quality Engineer specializing in Python testing.

Your role is to create comprehensive test plans and generate high-quality pytest test code.

Follow these testing principles:
1. Test Pyramid: More unit tests, fewer integration/e2e tests
2. AAA Pattern: Arrange, Act, Assert in each test
3. Single Responsibility: Each test tests one thing
4. Isolation: Tests don't depend on each other
5. Determinism: Tests produce same results every time
6. Coverage: Aim for 80%+ code coverage
7. Edge Cases: Test boundaries, null/empty values, error conditions

Testing Best Practices:
- Use pytest fixtures for setup/teardown
- Use parametrize for multiple test cases
- Mock external dependencies
- Use meaningful test names (test_<what>_<condition>_<expected>)
- Include docstrings explaining the test purpose
- Use appropriate markers (@pytest.mark.unit, etc.)
- Handle async code with pytest-asyncio

When generating tests:
- Import from the actual module paths
- Use fixtures from conftest.py when available
- Include both happy path and error cases
- Test edge cases and boundary conditions
- Use type hints in test code

Output pytest code that can be directly executed."""


@register_subagent
class SQEAgent(BaseSubagent):
    """
    Software Quality Engineering subagent.

    Creates test plans, generates pytest test code, and validates
    implementations against acceptance criteria.
    """

    AGENT_TYPE = "sqa"
    CAPABILITIES = [SubagentCapability.TESTING, SubagentCapability.ANALYSIS]
    DESCRIPTION = "Creates test plans and generates pytest test code"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize the SQE agent."""
        super().__init__(config)
        self._test_cache: dict[str, GeneratedTest] = {}

    def capabilities(self) -> list[SubagentCapability]:
        """Return SQE capabilities."""
        return self.CAPABILITIES

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """
        Generate tests for the given task or code.

        Args:
            task: Description of what to test or code to test.
            context: Optional context including:
                - code: Source code to test
                - module_path: Import path for the module
                - existing_fixtures: Available fixtures
                - coverage_target: Target coverage percentage

        Returns:
            SubagentResult containing test plan and/or generated tests.
        """
        self._start_execution(task)

        try:
            ctx = context or {}

            if ctx.get("code"):
                # Generate tests for provided code
                tests = await self.generate_tests(
                    code=ctx["code"],
                    module_path=ctx.get("module_path", "module"),
                    fixtures=ctx.get("existing_fixtures", []),
                )
                result = SubagentResult(
                    success=True,
                    agent_id=self.id,
                    agent_type=self.AGENT_TYPE,
                    task=task,
                    result={"tests": [t.model_dump() for t in tests]},
                    metadata={"num_tests": len(tests)},
                    artifacts=[
                        {
                            "type": "test_file",
                            "filename": t.filename,
                            "content": t.content,
                        }
                        for t in tests
                    ],
                )
            else:
                # Create test plan
                plan = await self.create_test_plan(
                    feature_description=task,
                    acceptance_criteria=ctx.get("acceptance_criteria", []),
                )
                result = SubagentResult(
                    success=True,
                    agent_id=self.id,
                    agent_type=self.AGENT_TYPE,
                    task=task,
                    result=plan.model_dump(),
                    metadata={
                        "num_suites": len(plan.test_suites),
                        "num_cases": sum(len(s.test_cases) for s in plan.test_suites),
                    },
                    artifacts=[
                        {
                            "type": "test_plan",
                            "format": "json",
                            "content": plan.model_dump_json(indent=2),
                        }
                    ],
                )

        except Exception as e:
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                error=str(e),
            )

        return self._complete_execution(result)

    async def create_test_plan(
        self,
        feature_description: str,
        acceptance_criteria: list[str] | None = None,
    ) -> TestPlan:
        """
        Create a comprehensive test plan.

        Args:
            feature_description: Description of the feature to test.
            acceptance_criteria: List of acceptance criteria.

        Returns:
            TestPlan with test suites and cases.
        """
        ac_text = "\n".join(f"- {ac}" for ac in (acceptance_criteria or []))

        prompt = f"""Create a comprehensive test plan for:

Feature: {feature_description}

Acceptance Criteria:
{ac_text or "Not specified - derive from feature description"}

Include:
1. Test objectives
2. Unit test suite with 3-5 test cases
3. Integration test suite with 2-3 test cases
4. Edge cases and error conditions
5. Entry and exit criteria

Output as JSON matching this structure:
{{
    "title": "...",
    "scope": "...",
    "objectives": [...],
    "test_suites": [
        {{
            "id": "...",
            "name": "...",
            "description": "...",
            "test_cases": [
                {{
                    "id": "...",
                    "name": "...",
                    "description": "...",
                    "test_type": "unit",
                    "steps": [...],
                    "expected_result": "..."
                }}
            ]
        }}
    ],
    "entry_criteria": [...],
    "exit_criteria": [...]
}}"""

        response = await self._llm_complete(prompt, system=SQE_SYSTEM_PROMPT)

        result = parse_json_response(response, model_class=TestPlan)
        if isinstance(result, TestPlan):
            return result

        # Fallback plan
        return self._create_fallback_plan(feature_description)

    async def generate_tests(
        self,
        code: str,
        module_path: str,
        fixtures: list[str] | None = None,
    ) -> list[GeneratedTest]:
        """
        Generate pytest tests for given code.

        Args:
            code: Source code to test.
            module_path: Import path for the module.
            fixtures: Available pytest fixtures.

        Returns:
            List of generated test files.
        """
        fixtures_text = "\n".join(f"- {f}" for f in (fixtures or []))

        prompt = f"""Generate comprehensive pytest tests for this code:

```python
{code}
```

Module path for imports: {module_path}

Available fixtures:
{fixtures_text or "None specified"}

Requirements:
1. Use pytest and pytest-asyncio for async code
2. Include unit tests for each public function/method
3. Test happy path, edge cases, and error conditions
4. Use fixtures where appropriate
5. Include meaningful docstrings
6. Use parametrize for multiple test cases

Output the complete test file content as Python code.
Start with the imports and fixtures, then the test functions.
"""

        response = await self._llm_complete(prompt, system=SQE_SYSTEM_PROMPT)

        # Extract Python code from response
        test_content = self._extract_code(response)

        if not test_content:
            test_content = self._generate_fallback_tests(code, module_path)

        # Determine filename from module path
        filename = f"test_{module_path.split('.')[-1]}.py"

        test = GeneratedTest(
            filename=filename,
            content=test_content,
            imports=self._extract_imports(test_content),
            markers=self._extract_markers(test_content),
        )

        self._test_cache[module_path] = test
        return [test]

    async def validate_implementation(
        self,
        code: str,
        acceptance_criteria: list[str],
    ) -> dict:
        """
        Validate code against acceptance criteria.

        Args:
            code: Implementation code.
            acceptance_criteria: List of acceptance criteria.

        Returns:
            Validation results with pass/fail for each criterion.
        """
        ac_text = "\n".join(f"{i+1}. {ac}" for i, ac in enumerate(acceptance_criteria))

        prompt = f"""Analyze this code and validate against the acceptance criteria:

Code:
```python
{code}
```

Acceptance Criteria:
{ac_text}

For each criterion, determine:
1. Whether the code satisfies it (PASS/FAIL)
2. Evidence or explanation
3. Any concerns or suggestions

Output as JSON:
{{
    "overall_pass": true/false,
    "criteria": [
        {{
            "id": 1,
            "criterion": "...",
            "status": "PASS/FAIL",
            "evidence": "...",
            "suggestions": "..."
        }}
    ],
    "summary": "..."
}}"""

        response = await self._llm_complete(prompt, system=SQE_SYSTEM_PROMPT)

        result = parse_json_response(response)
        if isinstance(result, dict):
            return result

        return {
            "overall_pass": False,
            "criteria": [],
            "summary": "Unable to validate - parsing failed",
        }

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        return extract_code_block(
            response,
            language="python",
            fallback_keywords=["import", "def test_"],
        )

    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements from code."""
        return extract_imports(code)

    def _extract_markers(self, code: str) -> list[str]:
        """Extract pytest markers from code."""
        markers = []
        for line in code.split("\n"):
            if "@pytest.mark." in line:
                marker = line.strip().replace("@pytest.mark.", "").split("(")[0]
                if marker not in markers:
                    markers.append(marker)
        return markers

    def _generate_fallback_tests(self, _code: str, module_path: str) -> str:
        """Generate basic fallback tests."""
        return f'''"""
Auto-generated tests for {module_path}.
"""

import pytest

# TODO: Update import path
# from {module_path} import ...


class TestModule:
    """Test suite for {module_path}."""

    def test_import(self):
        """Test that module can be imported."""
        # TODO: Add import test
        pass

    def test_basic_functionality(self):
        """Test basic functionality."""
        # TODO: Add tests based on the code
        pass

    @pytest.mark.unit
    def test_edge_cases(self):
        """Test edge cases."""
        # TODO: Add edge case tests
        pass
'''

    def _create_fallback_plan(self, feature: str) -> TestPlan:
        """Create a basic fallback test plan."""
        return TestPlan(
            title=f"Test Plan: {feature[:50]}",
            scope=feature,
            objectives=[
                "Verify core functionality",
                "Test error handling",
                "Validate edge cases",
            ],
            test_suites=[
                TestSuite(
                    id="TS1",
                    name="Unit Tests",
                    description="Unit tests for core functionality",
                    test_cases=[
                        TestCase(
                            id="TC1.1",
                            name="test_basic_functionality",
                            description="Verify basic operation",
                            test_type="unit",
                            steps=["Call function with valid input", "Verify output"],
                            expected_result="Function returns expected result",
                        ),
                        TestCase(
                            id="TC1.2",
                            name="test_error_handling",
                            description="Verify error handling",
                            test_type="unit",
                            steps=["Call function with invalid input", "Verify error raised"],
                            expected_result="Appropriate error is raised",
                        ),
                    ],
                ),
                TestSuite(
                    id="TS2",
                    name="Integration Tests",
                    description="Integration tests",
                    test_cases=[
                        TestCase(
                            id="TC2.1",
                            name="test_integration",
                            description="Verify component integration",
                            test_type="integration",
                            steps=["Initialize components", "Test interaction"],
                            expected_result="Components work together correctly",
                        ),
                    ],
                ),
            ],
            entry_criteria=["Code complete", "Unit tests written"],
            exit_criteria=["All tests passing", "80% coverage achieved"],
        )
