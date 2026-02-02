"""
Planner subagent for task decomposition and architecture planning.

Generates C4-aligned plans with epics, stories, and acceptance criteria.
"""


from pydantic import BaseModel, Field

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentResult,
    register_subagent,
)
from ptpd_calibration.agents.utils import parse_json_response


class AcceptanceCriteria(BaseModel):
    """Acceptance criteria for a story."""

    id: str
    description: str
    testable: bool = True
    automated: bool = False


class Story(BaseModel):
    """A user story within an epic."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[AcceptanceCriteria] = Field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high
    dependencies: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)


class Epic(BaseModel):
    """An epic containing multiple stories."""

    id: str
    title: str
    description: str
    stories: list[Story] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=10)


class Milestone(BaseModel):
    """A milestone containing multiple epics."""

    id: str
    title: str
    goal: str
    epics: list[Epic] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    """Complete implementation plan."""

    goal: str
    summary: str
    milestones: list[Milestone] = Field(default_factory=list)
    architectural_decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


PLANNER_SYSTEM_PROMPT = """You are an expert software architect and project planner.

Your role is to analyze requirements and create detailed, actionable implementation plans.

Follow these principles:
1. C4 Architecture: Consider Context, Containers, Components, and Code levels
2. Agile Methodology: Create epics, stories, and acceptance criteria
3. Backward Compatibility: Ensure changes don't break existing functionality
4. Testability: Every story should have testable acceptance criteria
5. Modularity: Design for reusable, loosely-coupled components
6. No Hardcoding: Use configuration, environment variables, and dependency injection

When creating plans:
- Break complex tasks into milestones
- Each milestone has a clear goal and success metrics
- Epics group related stories
- Stories are atomic, testable units of work
- Include Definition of Done for each story
- Identify dependencies between stories
- Estimate complexity (low/medium/high)
- Note risks and assumptions

Output Format:
Return a valid JSON object matching the ImplementationPlan schema:
{
    "goal": "...",
    "summary": "...",
    "milestones": [...],
    "architectural_decisions": [...],
    "risks": [...],
    "assumptions": [...]
}"""


@register_subagent
class PlannerAgent(BaseSubagent):
    """
    Planner subagent for creating implementation plans.

    Generates structured plans with milestones, epics, stories,
    and acceptance criteria following C4 architecture principles.
    """

    AGENT_TYPE = "planner"
    CAPABILITIES = [SubagentCapability.PLANNING, SubagentCapability.ANALYSIS]
    DESCRIPTION = "Creates C4-aligned implementation plans with epics and stories"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize the planner agent."""
        super().__init__(config)
        self._plan_cache: dict[str, ImplementationPlan] = {}

    def capabilities(self) -> list[SubagentCapability]:
        """Return planner capabilities."""
        return self.CAPABILITIES

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """
        Create an implementation plan for the given task.

        Args:
            task: Task or feature description to plan.
            context: Optional context including:
                - codebase_info: Information about existing code
                - constraints: Technical constraints
                - existing_patterns: Patterns to follow

        Returns:
            SubagentResult containing the implementation plan.
        """
        self._start_execution(task)

        try:
            # Build the prompt
            prompt = self._build_planning_prompt(task, context or {})

            # Get plan from LLM
            response = await self._execute_with_retry(
                "generate_plan",
                self._llm_complete,
                prompt,
                system=PLANNER_SYSTEM_PROMPT,
            )

            # Parse the response
            plan = self._parse_plan_response(response, task)

            # Cache the plan
            self._plan_cache[task[:50]] = plan

            result = SubagentResult(
                success=True,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                result=plan.model_dump(),
                metadata={
                    "num_milestones": len(plan.milestones),
                    "num_epics": sum(len(m.epics) for m in plan.milestones),
                    "num_stories": sum(
                        len(e.stories) for m in plan.milestones for e in m.epics
                    ),
                },
                artifacts=[
                    {
                        "type": "implementation_plan",
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

    def _build_planning_prompt(self, task: str, context: dict) -> str:
        """Build the prompt for plan generation."""
        prompt_parts = [f"Create an implementation plan for the following task:\n\n{task}"]

        if context.get("codebase_info"):
            prompt_parts.append(f"\n\nCodebase Information:\n{context['codebase_info']}")

        if context.get("constraints"):
            prompt_parts.append(f"\n\nConstraints:\n{context['constraints']}")

        if context.get("existing_patterns"):
            prompt_parts.append(f"\n\nExisting Patterns to Follow:\n{context['existing_patterns']}")

        prompt_parts.append(
            "\n\nCreate a detailed implementation plan with milestones, epics, "
            "and stories. Each story should have clear acceptance criteria and "
            "a definition of done. Output valid JSON matching the schema."
        )

        return "\n".join(prompt_parts)

    def _parse_plan_response(self, response: str, task: str) -> ImplementationPlan:
        """Parse LLM response into ImplementationPlan."""
        result = parse_json_response(response, model_class=ImplementationPlan)
        if isinstance(result, ImplementationPlan):
            return result
        return self._create_fallback_plan(task)

    def _create_fallback_plan(self, task: str) -> ImplementationPlan:
        """Create a basic fallback plan when parsing fails."""
        return ImplementationPlan(
            goal=task,
            summary="Auto-generated plan from task description",
            milestones=[
                Milestone(
                    id="M1",
                    title="Implementation",
                    goal="Complete the requested feature",
                    epics=[
                        Epic(
                            id="E1.1",
                            title="Core Implementation",
                            description="Implement the main functionality",
                            stories=[
                                Story(
                                    id="S1.1.1",
                                    title="Analyze requirements",
                                    description="Understand and document requirements",
                                    acceptance_criteria=[
                                        AcceptanceCriteria(
                                            id="AC1",
                                            description="Requirements documented",
                                        )
                                    ],
                                    definition_of_done=[
                                        "Requirements reviewed",
                                        "Approach approved",
                                    ],
                                ),
                                Story(
                                    id="S1.1.2",
                                    title="Implement solution",
                                    description="Build the required functionality",
                                    acceptance_criteria=[
                                        AcceptanceCriteria(
                                            id="AC1",
                                            description="Feature works as specified",
                                        )
                                    ],
                                    dependencies=["S1.1.1"],
                                    definition_of_done=[
                                        "Code implemented",
                                        "Tests passing",
                                        "Documentation updated",
                                    ],
                                ),
                            ],
                        ),
                        Epic(
                            id="E1.2",
                            title="Testing",
                            description="Validate the implementation",
                            stories=[
                                Story(
                                    id="S1.2.1",
                                    title="Write tests",
                                    description="Create comprehensive test suite",
                                    acceptance_criteria=[
                                        AcceptanceCriteria(
                                            id="AC1",
                                            description="80%+ test coverage",
                                            automated=True,
                                        )
                                    ],
                                    dependencies=["S1.1.2"],
                                    definition_of_done=[
                                        "Unit tests written",
                                        "Integration tests written",
                                        "All tests passing",
                                    ],
                                ),
                            ],
                        ),
                    ],
                    success_metrics=[
                        "All acceptance criteria met",
                        "Tests passing",
                        "No regressions",
                    ],
                )
            ],
            architectural_decisions=[
                "Follow existing codebase patterns",
                "Maintain backward compatibility",
            ],
            risks=["Requirements may be incomplete"],
            assumptions=["Existing architecture is suitable"],
        )

    async def create_epic_plan(
        self,
        epic_description: str,
        _context: dict | None = None,
    ) -> Epic:
        """
        Create a detailed epic with stories.

        Args:
            epic_description: Description of the epic.
            context: Optional context.

        Returns:
            Epic with stories and acceptance criteria.
        """
        prompt = f"""Create a detailed epic plan for:

{epic_description}

The epic should include:
1. Clear title and description
2. 3-5 stories with acceptance criteria
3. Dependencies between stories
4. Definition of done for each story

Output as JSON matching this structure:
{{
    "id": "E1",
    "title": "...",
    "description": "...",
    "stories": [...]
}}"""

        response = await self._llm_complete(prompt, system=PLANNER_SYSTEM_PROMPT)

        result = parse_json_response(response, model_class=Epic)
        if isinstance(result, Epic):
            return result

        # Fallback
        return Epic(
            id="E1",
            title=epic_description[:50],
            description=epic_description,
            stories=[
                Story(
                    id="S1",
                    title="Implement",
                    description=epic_description,
                    acceptance_criteria=[
                        AcceptanceCriteria(id="AC1", description="Feature complete")
                    ],
                )
            ],
        )

    async def decompose_task(
        self,
        task: str,
        _max_depth: int = 2,
    ) -> list[dict]:
        """
        Recursively decompose a task into subtasks.

        Args:
            task: Task to decompose.
            max_depth: Maximum decomposition depth.

        Returns:
            List of subtask dictionaries.
        """
        prompt = f"""Decompose this task into 3-5 subtasks:

{task}

For each subtask, provide:
- title: Brief title
- description: What needs to be done
- complexity: low/medium/high

Output as JSON array:
[{{"title": "...", "description": "...", "complexity": "..."}}]"""

        response = await self._llm_complete(prompt)

        result = parse_json_response(response, parse_array=True)
        if isinstance(result, list):
            return result

        return [{"title": task, "description": task, "complexity": "medium"}]
