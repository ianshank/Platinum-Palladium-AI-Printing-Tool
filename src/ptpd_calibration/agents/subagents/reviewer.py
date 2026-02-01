"""
Reviewer subagent for code review and quality analysis.

Reviews code for quality, security, and best practices.
"""

from enum import Enum

from pydantic import BaseModel, Field

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentResult,
    register_subagent,
)
from ptpd_calibration.agents.utils import parse_json_response


class IssueSeverity(str, Enum):
    """Severity level for code issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Category of code issue."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


class CodeIssue(BaseModel):
    """A single code issue found during review."""

    id: str
    severity: IssueSeverity
    category: IssueCategory
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    code_snippet: str | None = None
    suggestion: str | None = None
    auto_fixable: bool = False


class ReviewScore(BaseModel):
    """Score for a review dimension."""

    dimension: str
    score: float = Field(ge=0, le=10)
    max_score: float = 10
    notes: str = ""


class CodeReview(BaseModel):
    """Complete code review result."""

    summary: str
    overall_score: float = Field(ge=0, le=10)
    scores: list[ReviewScore] = Field(default_factory=list)
    issues: list[CodeIssue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    security_concerns: list[str] = Field(default_factory=list)
    approved: bool = False


REVIEWER_SYSTEM_PROMPT = """You are an expert code reviewer specializing in Python.

Your role is to review code for quality, security, and best practices.

Review Dimensions:
1. Security: OWASP top 10, injection vulnerabilities, secrets exposure
2. Performance: Algorithm efficiency, memory usage, I/O patterns
3. Maintainability: Code clarity, complexity, coupling
4. Reliability: Error handling, edge cases, data validation
5. Style: PEP 8 compliance, naming conventions, formatting
6. Documentation: Docstrings, comments, type hints
7. Testing: Testability, coverage considerations
8. Architecture: Design patterns, separation of concerns

Severity Levels:
- CRITICAL: Security vulnerabilities, data loss risks, crashes
- HIGH: Major bugs, significant performance issues
- MEDIUM: Code smells, maintainability concerns
- LOW: Minor style issues, small improvements
- INFO: Suggestions, best practices

Review Guidelines:
1. Be specific and actionable
2. Provide code examples for fixes
3. Acknowledge good practices
4. Consider context and trade-offs
5. Prioritize security issues

Output Format:
Provide a JSON review with:
- summary: Brief overview
- overall_score: 0-10
- scores: Per-dimension scores
- issues: List of issues found
- strengths: Good practices observed
- recommendations: Improvement suggestions
"""


@register_subagent
class ReviewerAgent(BaseSubagent):
    """
    Reviewer subagent for code quality analysis.

    Reviews code for security, performance, maintainability,
    and adherence to best practices.
    """

    AGENT_TYPE = "reviewer"
    CAPABILITIES = [SubagentCapability.REVIEWING, SubagentCapability.ANALYSIS]
    DESCRIPTION = "Reviews code for quality, security, and best practices"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize the reviewer agent."""
        super().__init__(config)
        self._review_history: list[CodeReview] = []

    def capabilities(self) -> list[SubagentCapability]:
        """Return reviewer capabilities."""
        return self.CAPABILITIES

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """
        Review code provided in task or context.

        Args:
            task: Description of what to review or the code itself.
            context: Optional context including:
                - code: Code to review
                - file_path: Path of the file being reviewed
                - focus_areas: Specific areas to focus on
                - ignore_rules: Rules to ignore

        Returns:
            SubagentResult containing the code review.
        """
        self._start_execution(task)

        try:
            ctx = context or {}
            code = ctx.get("code", task)

            review = await self.review_code(
                code=code,
                file_path=ctx.get("file_path"),
                focus_areas=ctx.get("focus_areas"),
                ignore_rules=ctx.get("ignore_rules"),
            )

            self._review_history.append(review)

            result = SubagentResult(
                success=True,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task[:200],
                result=review.model_dump(),
                metadata={
                    "overall_score": review.overall_score,
                    "num_issues": len(review.issues),
                    "critical_issues": len([i for i in review.issues if i.severity == IssueSeverity.CRITICAL]),
                    "approved": review.approved,
                },
                artifacts=[
                    {
                        "type": "code_review",
                        "format": "json",
                        "content": review.model_dump_json(indent=2),
                    }
                ],
            )

        except Exception as e:
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task[:200],
                error=str(e),
            )

        return self._complete_execution(result)

    async def review_code(
        self,
        code: str,
        file_path: str | None = None,
        focus_areas: list[str] | None = None,
        ignore_rules: list[str] | None = None,
    ) -> CodeReview:
        """
        Perform comprehensive code review.

        Args:
            code: Code to review.
            file_path: File path for context.
            focus_areas: Areas to focus on.
            ignore_rules: Rules to ignore.

        Returns:
            CodeReview with findings.
        """
        focus_text = "\n".join(f"- {f}" for f in (focus_areas or []))
        ignore_text = "\n".join(f"- {r}" for r in (ignore_rules or []))

        prompt = f"""Review the following code:

{"File: " + file_path if file_path else ""}

```python
{code}
```

{"Focus Areas:" + chr(10) + focus_text if focus_areas else ""}
{"Ignore Rules:" + chr(10) + ignore_text if ignore_rules else ""}

Provide a comprehensive review covering:
1. Security issues
2. Performance concerns
3. Maintainability
4. Code style
5. Documentation
6. Error handling

Output as JSON with this structure:
{{
    "summary": "...",
    "overall_score": 7.5,
    "scores": [
        {{"dimension": "Security", "score": 8, "notes": "..."}},
        ...
    ],
    "issues": [
        {{
            "id": "I1",
            "severity": "medium",
            "category": "maintainability",
            "title": "...",
            "description": "...",
            "line_number": 10,
            "suggestion": "..."
        }}
    ],
    "strengths": [...],
    "recommendations": [...],
    "security_concerns": [...],
    "approved": true/false
}}
"""

        response = await self._llm_complete(prompt, system=REVIEWER_SYSTEM_PROMPT)

        result = parse_json_response(response, model_class=CodeReview)
        if isinstance(result, CodeReview):
            return result

        # Fallback review
        return self._create_fallback_review(code)

    async def security_scan(self, code: str) -> list[CodeIssue]:
        """
        Perform focused security scan.

        Args:
            code: Code to scan.

        Returns:
            List of security issues.
        """
        prompt = f"""Perform a security-focused review of this code:

```python
{code}
```

Look for:
1. SQL/Command injection
2. XSS vulnerabilities
3. Hardcoded secrets/credentials
4. Insecure deserialization
5. Path traversal
6. Improper input validation
7. Sensitive data exposure

Output as JSON array of issues:
[
    {{
        "id": "SEC1",
        "severity": "critical",
        "category": "security",
        "title": "...",
        "description": "...",
        "line_number": 10,
        "suggestion": "..."
    }}
]
"""

        response = await self._llm_complete(prompt, system=REVIEWER_SYSTEM_PROMPT)

        result = parse_json_response(response, parse_array=True)
        if isinstance(result, list):
            return [CodeIssue(**issue) for issue in result]

        return []

    async def performance_analysis(self, code: str) -> list[CodeIssue]:
        """
        Analyze code for performance issues.

        Args:
            code: Code to analyze.

        Returns:
            List of performance issues.
        """
        prompt = f"""Analyze this code for performance issues:

```python
{code}
```

Look for:
1. O(n^2) or worse algorithms
2. Unnecessary loops or iterations
3. Memory inefficiencies
4. Blocking I/O in async code
5. Missing caching opportunities
6. Inefficient data structures

Output as JSON array of issues:
[
    {{
        "id": "PERF1",
        "severity": "medium",
        "category": "performance",
        "title": "...",
        "description": "...",
        "suggestion": "..."
    }}
]
"""

        response = await self._llm_complete(prompt, system=REVIEWER_SYSTEM_PROMPT)

        result = parse_json_response(response, parse_array=True)
        if isinstance(result, list):
            return [CodeIssue(**issue) for issue in result]

        return []

    async def compare_implementations(
        self,
        original: str,
        modified: str,
    ) -> dict:
        """
        Compare two code implementations.

        Args:
            original: Original code.
            modified: Modified code.

        Returns:
            Comparison analysis.
        """
        prompt = f"""Compare these two code implementations:

ORIGINAL:
```python
{original}
```

MODIFIED:
```python
{modified}
```

Analyze:
1. What changed?
2. Are changes improvements?
3. Any regressions introduced?
4. Backward compatibility?
5. Test impact?

Output as JSON:
{{
    "changes": [...],
    "improvements": [...],
    "regressions": [...],
    "backward_compatible": true/false,
    "test_changes_needed": [...],
    "recommendation": "approve/reject/revise"
}}
"""

        response = await self._llm_complete(prompt, system=REVIEWER_SYSTEM_PROMPT)

        result = parse_json_response(response)
        if isinstance(result, dict):
            return result

        return {
            "changes": ["Unable to parse changes"],
            "improvements": [],
            "regressions": [],
            "backward_compatible": None,
            "test_changes_needed": [],
            "recommendation": "revise",
        }

    async def suggest_improvements(self, code: str) -> list[str]:
        """
        Suggest code improvements.

        Args:
            code: Code to improve.

        Returns:
            List of improvement suggestions.
        """
        prompt = f"""Suggest improvements for this code:

```python
{code}
```

Provide specific, actionable suggestions for:
1. Code clarity
2. Performance
3. Error handling
4. Type safety
5. Documentation

Output as JSON array of strings:
["suggestion 1", "suggestion 2", ...]
"""

        response = await self._llm_complete(prompt, system=REVIEWER_SYSTEM_PROMPT)

        result = parse_json_response(response, parse_array=True)
        if isinstance(result, list):
            return result

        return []

    def _create_fallback_review(self, code: str) -> CodeReview:
        """Create a basic fallback review."""
        issues = []

        # Basic static checks
        if "except:" in code or "except Exception:" in code:
            issues.append(
                CodeIssue(
                    id="I1",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.RELIABILITY,
                    title="Broad exception handling",
                    description="Catching all exceptions may hide bugs",
                    suggestion="Catch specific exceptions",
                )
            )

        if "# TODO" in code or "# FIXME" in code:
            issues.append(
                CodeIssue(
                    id="I2",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.MAINTAINABILITY,
                    title="Unresolved TODOs",
                    description="Code contains TODO/FIXME comments",
                    suggestion="Address or document these items",
                )
            )

        if "password" in code.lower() or "secret" in code.lower():
            issues.append(
                CodeIssue(
                    id="I3",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    title="Potential hardcoded secrets",
                    description="Code may contain hardcoded credentials",
                    suggestion="Use environment variables or secrets manager",
                )
            )

        # Calculate score based on issues
        critical_count = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high_count = len([i for i in issues if i.severity == IssueSeverity.HIGH])
        medium_count = len([i for i in issues if i.severity == IssueSeverity.MEDIUM])

        score = 10 - (critical_count * 3) - (high_count * 1.5) - (medium_count * 0.5)
        score = max(0, min(10, score))

        return CodeReview(
            summary="Basic automated review (full analysis unavailable)",
            overall_score=score,
            scores=[
                ReviewScore(dimension="Security", score=score, notes="Basic check performed"),
                ReviewScore(dimension="Maintainability", score=score, notes="Basic check performed"),
            ],
            issues=issues,
            strengths=["Code submitted for review"],
            recommendations=["Perform detailed manual review"],
            approved=critical_count == 0 and high_count == 0,
        )

    def get_review_history(self) -> list[CodeReview]:
        """Get review history."""
        return self._review_history

    def get_issue_statistics(self) -> dict:
        """Get statistics from review history."""
        all_issues = [issue for review in self._review_history for issue in review.issues]

        by_severity = {}
        for severity in IssueSeverity:
            by_severity[severity.value] = len([i for i in all_issues if i.severity == severity])

        by_category = {}
        for category in IssueCategory:
            by_category[category.value] = len([i for i in all_issues if i.category == category])

        return {
            "total_reviews": len(self._review_history),
            "total_issues": len(all_issues),
            "by_severity": by_severity,
            "by_category": by_category,
            "approval_rate": (
                len([r for r in self._review_history if r.approved]) / len(self._review_history)
                if self._review_history
                else 0
            ),
        }
