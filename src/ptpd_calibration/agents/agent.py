"""
Main calibration agent with ReAct-style reasoning.
"""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptpd_calibration.agents.memory import AgentMemory
from ptpd_calibration.agents.planning import Plan, Planner
from ptpd_calibration.agents.tools import ToolResult, create_calibration_tools
from ptpd_calibration.config import AgentSettings, LLMSettings, get_settings
from ptpd_calibration.llm.client import LLMClient, create_client
from ptpd_calibration.llm.prompts import SYSTEM_PROMPT
from ptpd_calibration.ml.database import CalibrationDatabase


@dataclass
class AgentConfig:
    """Configuration for the calibration agent."""

    llm_settings: LLMSettings | None = None
    agent_settings: AgentSettings | None = None
    memory_path: Path | None = None
    database: CalibrationDatabase | None = None
    predictor: Any = None


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning trace."""

    step_type: str  # "thought", "action", "observation", "reflection"
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: ToolResult | None = None


class CalibrationAgent:
    """
    Autonomous calibration agent with ReAct-style reasoning.

    Uses tools, planning, and memory to assist with Pt/Pd calibration tasks.
    """

    def __init__(
        self,
        config: AgentConfig,
    ):
        """
        Initialize the calibration agent.

        Args:
            config: Agent configuration.
        """
        self.config = config
        self.settings = config.agent_settings or get_settings().agent
        self.llm_settings = config.llm_settings or get_settings().llm

        # Initialize components
        self.client: LLMClient | None = None
        self.tools = create_calibration_tools(config.database, config.predictor)
        self.memory = AgentMemory(
            storage_path=config.memory_path,
            max_items=self.settings.max_memory_items,
            working_memory_size=self.settings.working_memory_size,
        )
        self.planner = Planner(max_steps=self.settings.max_plan_steps)

        # State
        self._current_plan: Plan | None = None
        self._reasoning_trace: list[ReasoningStep] = []
        self._iteration_count = 0

    async def run(self, task: str) -> str:
        """
        Run the agent on a task.

        Args:
            task: The task to perform.

        Returns:
            Final response.
        """
        # Initialize client if needed
        if self.client is None:
            self.client = create_client(self.llm_settings)

        # Reset state
        self._reasoning_trace = []
        self._iteration_count = 0

        # Create initial plan
        self._current_plan = self.planner.create_plan(task)

        # Add task to working memory
        self.memory.add_to_working_memory(f"Task: {task}")

        # Main reasoning loop
        while self._iteration_count < self.settings.max_iterations:
            self._iteration_count += 1

            # Check if plan is complete
            if self._current_plan.is_complete:
                break

            # Get next action
            thought, action = await self._think_and_act()

            # Record thought
            self._reasoning_trace.append(
                ReasoningStep(step_type="thought", content=thought)
            )

            if action is None:
                # Agent decided to finish
                break

            # Execute action
            observation = await self._execute_action(action)

            # Record observation
            self._reasoning_trace.append(
                ReasoningStep(
                    step_type="observation",
                    content=observation.to_string() if observation else "No result",
                    tool_name=action.get("tool"),
                    tool_args=action.get("args"),
                    tool_result=observation,
                )
            )

            # Reflect if enabled
            if self.settings.enable_reflection:
                if self._iteration_count % self.settings.reflection_frequency == 0:
                    reflection = await self._reflect()
                    self._reasoning_trace.append(
                        ReasoningStep(step_type="reflection", content=reflection)
                    )

        # Generate final response
        return await self._generate_response(task)

    async def _think_and_act(self) -> tuple[str, dict | None]:
        """
        Generate thought and decide on action.

        Returns:
            Tuple of (thought, action_dict or None).
        """
        # Build context
        context = self._build_context()

        # Prompt for thought and action
        prompt = f"""{context}

Based on the current state, think step by step about what to do next.

Available tools:
{self._format_tools()}

Respond with:
1. THOUGHT: Your reasoning about what to do next
2. ACTION: Either a tool call in JSON format {{"tool": "name", "args": {{...}}}} or "FINISH" if done

Example:
THOUGHT: I need to analyze the density measurements to understand the calibration quality.
ACTION: {{"tool": "analyze_densities", "args": {{"densities": [0.1, 0.3, 0.5, 0.8, 1.2]}}}}"""

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        # Parse response
        thought = ""
        action = None

        lines = response.split("\n")
        for line in lines:
            if line.startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.startswith("ACTION:"):
                action_str = line[7:].strip()
                if action_str.upper() != "FINISH":
                    try:
                        action = json.loads(action_str)
                    except json.JSONDecodeError:
                        # Try to find JSON in the string
                        start = action_str.find("{")
                        end = action_str.rfind("}") + 1
                        if start >= 0 and end > start:
                            with contextlib.suppress(json.JSONDecodeError):
                                action = json.loads(action_str[start:end])

        return thought, action

    async def _execute_action(self, action: dict) -> ToolResult | None:
        """Execute a tool action."""
        tool_name = action.get("tool")
        tool_args = action.get("args", {})

        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        # Record action
        self._reasoning_trace.append(
            ReasoningStep(
                step_type="action",
                content=f"Calling {tool_name}",
                tool_name=tool_name,
                tool_args=tool_args,
            )
        )

        # Execute
        result = await tool.execute(**tool_args)

        # Add to working memory
        self.memory.add_to_working_memory(
            f"Tool {tool_name}: {result.to_string()[:200]}"
        )

        # Update plan
        if self._current_plan and self._current_plan.current_step:
            step = self._current_plan.current_step
            if result.success:
                step.mark_completed(result.to_string())
            else:
                step.mark_failed(result.error or "Unknown error")
            self._current_plan.advance()

        return result

    async def _reflect(self) -> str:
        """Reflect on progress and adapt if needed."""
        context = self._build_context()

        prompt = f"""{context}

Reflect on the progress so far:
1. What has been accomplished?
2. What challenges have been encountered?
3. Should the plan be adapted?
4. What insights should be remembered?

Respond with a brief reflection."""

        reflection = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            max_tokens=500,
        )

        # Check if plan needs adaptation
        if "adapt" in reflection.lower() or "change" in reflection.lower():
            adaptation = self.planner.suggest_adaptation(self._current_plan, reflection)
            if adaptation:
                self._current_plan.adapt(reflection, adaptation)

        return reflection

    async def _generate_response(self, original_task: str) -> str:
        """Generate final response from reasoning trace."""
        # Summarize trace
        trace_summary = self._summarize_trace()

        prompt = f"""Original task: {original_task}

Here's what I did:
{trace_summary}

Please provide a clear, helpful final response to the user summarizing the results and any recommendations."""

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        return response

    def _build_context(self) -> str:
        """Build context string for LLM."""
        parts = []

        # Current plan
        if self._current_plan:
            parts.append(f"Current plan:\n{self._current_plan.summary()}")

        # Working memory
        working = self.memory.get_working_memory()
        if working:
            parts.append("Working memory:\n" + "\n".join(f"- {w}" for w in working[-5:]))

        # Recent reasoning
        if self._reasoning_trace:
            recent = self._reasoning_trace[-3:]
            trace_str = "\n".join(
                f"[{s.step_type.upper()}] {s.content[:100]}" for s in recent
            )
            parts.append(f"Recent reasoning:\n{trace_str}")

        return "\n\n".join(parts)

    def _format_tools(self) -> str:
        """Format available tools for prompt."""
        tool_list = []
        for tool in self.tools.list_tools():
            params = ", ".join(
                f"{p.name}: {p.type}" for p in tool.parameters if p.required
            )
            tool_list.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(tool_list)

    def _summarize_trace(self) -> str:
        """Summarize the reasoning trace."""
        parts = []
        for step in self._reasoning_trace:
            if step.step_type == "thought":
                parts.append(f"Thought: {step.content}")
            elif step.step_type == "action":
                parts.append(f"Action: {step.content}")
            elif step.step_type == "observation":
                # Truncate long observations
                content = step.content[:200] + "..." if len(step.content) > 200 else step.content
                parts.append(f"Result: {content}")
            elif step.step_type == "reflection":
                parts.append(f"Reflection: {step.content}")
        return "\n".join(parts)

    def get_reasoning_trace(self) -> list[dict]:
        """Get the reasoning trace as dictionaries."""
        return [
            {
                "type": s.step_type,
                "content": s.content,
                "tool_name": s.tool_name,
                "tool_args": s.tool_args,
            }
            for s in self._reasoning_trace
        ]

    def get_plan_status(self) -> dict | None:
        """Get current plan status."""
        if not self._current_plan:
            return None

        return {
            "goal": self._current_plan.goal,
            "status": self._current_plan.status.value,
            "progress": self._current_plan.progress,
            "current_step": self._current_plan.current_step_index,
            "total_steps": len(self._current_plan.steps),
            "adaptations": self._current_plan.adapted_count,
        }


def create_agent(
    api_key: str | None = None,
    database: CalibrationDatabase | None = None,
    predictor: Any = None,
    memory_path: Path | None = None,
) -> CalibrationAgent:
    """
    Create a calibration agent.

    Args:
        api_key: LLM API key.
        database: Calibration database.
        predictor: ML predictor.
        memory_path: Path for persistent memory.

    Returns:
        Configured CalibrationAgent.
    """
    llm_settings = LLMSettings(api_key=api_key) if api_key else None

    config = AgentConfig(
        llm_settings=llm_settings,
        database=database,
        predictor=predictor,
        memory_path=memory_path,
    )

    return CalibrationAgent(config)
