"""
Planning system for agent task decomposition.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class PlanStatus(str, Enum):
    """Status of a plan or step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    ADAPTED = "adapted"


@dataclass
class PlanStep:
    """A single step in a plan."""

    id: UUID = field(default_factory=uuid4)
    description: str = ""
    status: PlanStatus = PlanStatus.PENDING
    tool_name: Optional[str] = None
    tool_args: dict = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: list[UUID] = field(default_factory=list)

    def can_execute(self, completed_steps: set[UUID]) -> bool:
        """Check if this step can execute based on dependencies."""
        return all(dep in completed_steps for dep in self.dependencies)

    def mark_started(self) -> None:
        """Mark step as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self, result: str) -> None:
        """Mark step as completed."""
        self.status = PlanStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark step as failed."""
        self.status = PlanStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()


@dataclass
class Plan:
    """A plan for achieving a goal."""

    id: UUID = field(default_factory=uuid4)
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    adapted_count: int = 0
    adaptation_history: list[str] = field(default_factory=list)

    @property
    def current_step_index(self) -> int:
        """Get index of current step (first non-completed)."""
        for i, step in enumerate(self.steps):
            if step.status in (PlanStatus.PENDING, PlanStatus.IN_PROGRESS):
                return i
        return len(self.steps)

    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get current step."""
        idx = self.current_step_index
        if idx < len(self.steps):
            return self.steps[idx]
        return None

    @property
    def progress(self) -> float:
        """Get plan progress (0-1)."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED)
        return completed / len(self.steps)

    @property
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(s.status == PlanStatus.COMPLETED for s in self.steps)

    @property
    def has_failed(self) -> bool:
        """Check if plan has failed."""
        return any(s.status == PlanStatus.FAILED for s in self.steps)

    def next_step(self) -> Optional[PlanStep]:
        """Get the next step to execute."""
        completed = {s.id for s in self.steps if s.status == PlanStatus.COMPLETED}

        for step in self.steps:
            if step.status == PlanStatus.PENDING and step.can_execute(completed):
                return step
        return None

    def advance(self) -> bool:
        """Advance to the next step."""
        current = self.current_step
        if current and current.status == PlanStatus.IN_PROGRESS:
            return False  # Current step still in progress

        next_step = self.next_step()
        if next_step:
            next_step.mark_started()
            self.status = PlanStatus.IN_PROGRESS
            return True
        return False

    def adapt(self, reason: str, new_steps: list[str]) -> None:
        """Adapt the plan by adding new steps."""
        self.adapted_count += 1
        self.adaptation_history.append(f"[{datetime.now().isoformat()}] {reason}")

        # Insert new steps after current step
        idx = self.current_step_index + 1
        for i, desc in enumerate(new_steps):
            new_step = PlanStep(description=desc)
            self.steps.insert(idx + i, new_step)

        self.status = PlanStatus.ADAPTED

    def summary(self) -> str:
        """Generate a plan summary."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == PlanStatus.FAILED)

        return (
            f"Plan: {self.goal}\n"
            f"Progress: {completed}/{total} steps ({self.progress:.0%})\n"
            f"Status: {self.status.value}\n"
            f"Adaptations: {self.adapted_count}"
        )


class Planner:
    """
    Planning engine for task decomposition.

    Generates and manages plans for calibration tasks.
    """

    def __init__(self, max_steps: int = 10):
        """
        Initialize planner.

        Args:
            max_steps: Maximum steps per plan.
        """
        self.max_steps = max_steps
        self._plan_templates: dict[str, list[str]] = self._default_templates()

    def create_plan(self, goal: str, context: Optional[dict] = None) -> Plan:
        """
        Create a plan for achieving a goal.

        Args:
            goal: The goal to achieve.
            context: Optional context for planning.

        Returns:
            A new Plan.
        """
        # Check for matching template
        for template_name, steps in self._plan_templates.items():
            if template_name.lower() in goal.lower():
                return self._create_from_template(goal, steps)

        # Create generic plan
        return self._create_generic_plan(goal, context)

    def _create_from_template(self, goal: str, step_descriptions: list[str]) -> Plan:
        """Create plan from template."""
        steps = [PlanStep(description=desc) for desc in step_descriptions]
        return Plan(goal=goal, steps=steps)

    def _create_generic_plan(self, goal: str, context: Optional[dict]) -> Plan:
        """Create a generic plan."""
        steps = [
            PlanStep(description="Analyze the request and gather requirements"),
            PlanStep(description="Search for relevant information"),
            PlanStep(description="Execute the main task"),
            PlanStep(description="Verify results and report findings"),
        ]
        return Plan(goal=goal, steps=steps)

    def suggest_adaptation(
        self, plan: Plan, observation: str
    ) -> Optional[list[str]]:
        """
        Suggest plan adaptation based on observation.

        Args:
            plan: Current plan.
            observation: What was observed.

        Returns:
            List of suggested new steps, or None.
        """
        observation_lower = observation.lower()

        # Check for common issues
        if "error" in observation_lower or "failed" in observation_lower:
            return [
                "Investigate the error",
                "Try alternative approach",
                "Report findings",
            ]

        if "unexpected" in observation_lower or "surprising" in observation_lower:
            return [
                "Analyze unexpected result",
                "Determine if approach needs adjustment",
            ]

        return None

    def _default_templates(self) -> dict[str, list[str]]:
        """Get default plan templates."""
        return {
            "calibrate": [
                "Load or scan the step tablet image",
                "Detect and extract patch densities",
                "Analyze density range and linearity",
                "Generate correction curve",
                "Export curve in desired format",
                "Document results",
            ],
            "analyze": [
                "Load the calibration data",
                "Calculate quality metrics",
                "Compare to target values",
                "Identify issues or improvements",
                "Generate recommendations",
            ],
            "troubleshoot": [
                "Understand the problem description",
                "Identify potential causes",
                "Search for similar past issues",
                "Suggest diagnostic steps",
                "Provide solutions",
            ],
            "recipe": [
                "Understand paper and desired characteristics",
                "Search for similar calibrations",
                "Calculate recommended parameters",
                "Generate coating recipe",
                "Suggest test procedure",
            ],
            "predict": [
                "Gather process parameters",
                "Validate inputs",
                "Run prediction model",
                "Estimate uncertainty",
                "Format and return results",
            ],
        }
