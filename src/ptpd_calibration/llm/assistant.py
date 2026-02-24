"""
Calibration assistant powered by LLM.

Provides conversational AI assistance for Pt/Pd printing calibration.
"""

from collections.abc import AsyncIterator

from ptpd_calibration.config import LLMSettings, get_settings
from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.llm.client import LLMClient, create_client
from ptpd_calibration.llm.prompts import (
    SYSTEM_PROMPT,
    get_analysis_prompt,
    get_recipe_prompt,
    get_troubleshooting_prompt,
)
from ptpd_calibration.ml.database import CalibrationDatabase


class CalibrationAssistant:
    """
    AI-powered assistant for Pt/Pd calibration.

    Provides conversational interface for getting help with
    calibration, troubleshooting, and recipe development.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: LLMSettings | None = None,
        database: CalibrationDatabase | None = None,
    ):
        """
        Initialize the calibration assistant.

        Args:
            client: LLM client to use. Created automatically if not provided.
            settings: LLM settings.
            database: Optional calibration database for context.
        """
        self.settings = settings or get_settings().llm
        self.client = client or create_client(self.settings)
        self.database = database
        self.conversation_history: list[dict] = []

    async def chat(
        self,
        message: str,
        include_history: bool = True,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            message: User message.
            include_history: Whether to include conversation history.

        Returns:
            Assistant response.
        """
        # Build messages
        messages = []

        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": message})

        # Get response
        response = await self.client.complete(
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    async def chat_stream(
        self,
        message: str,
        include_history: bool = True,
    ) -> AsyncIterator[str]:
        """
        Send a message and stream the response.

        Args:
            message: User message.
            include_history: Whether to include conversation history.

        Yields:
            Response chunks as they arrive.
        """
        # Build messages
        messages = []

        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": message})

        # Stream response
        full_response = ""
        async for chunk in self.client.stream(
            messages=messages,
            system=SYSTEM_PROMPT,
        ):
            full_response += chunk
            yield chunk

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": full_response})

    async def analyze_calibration(
        self,
        record: CalibrationRecord,
    ) -> str:
        """
        Analyze a calibration record.

        Args:
            record: CalibrationRecord to analyze.

        Returns:
            Analysis and suggestions.
        """
        # Calculate metrics
        densities = record.measured_densities
        dmin = min(densities) if densities else 0
        dmax = max(densities) if densities else 0

        data = {
            "paper_type": record.paper_type,
            "chemistry_type": record.chemistry_type.value,
            "metal_ratio": record.metal_ratio,
            "contrast_agent": record.contrast_agent.value,
            "contrast_amount": record.contrast_amount,
            "developer": record.developer.value,
            "exposure_time": record.exposure_time,
            "dmin": dmin,
            "dmax": dmax,
            "density_range": dmax - dmin,
            "notes": record.notes or "None",
        }

        prompt = get_analysis_prompt(data)

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        return response

    async def suggest_recipe(
        self,
        paper_type: str,
        desired_characteristics: str,
    ) -> str:
        """
        Suggest a recipe for a paper and desired result.

        Args:
            paper_type: Target paper.
            desired_characteristics: What the user wants to achieve.

        Returns:
            Recipe suggestion with reasoning.
        """
        prompt = get_recipe_prompt(paper_type, desired_characteristics)

        # Add context from database if available
        if self.database:
            similar_records = self.database.query(paper_type=paper_type)
            if similar_records:
                context = "\n\nFor context, here are previous calibrations for this paper:\n"
                for i, rec in enumerate(similar_records[:3], 1):
                    densities = rec.measured_densities
                    dmax = max(densities) if densities else 0
                    context += (
                        f"{i}. Metal ratio: {rec.metal_ratio:.0%} Pt, "
                        f"Contrast: {rec.contrast_amount} drops, "
                        f"Exposure: {rec.exposure_time:.0f}s, "
                        f"Dmax: {dmax:.2f}\n"
                    )
                prompt += context

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        return response

    async def troubleshoot(self, problem: str) -> str:
        """
        Help troubleshoot a problem.

        Args:
            problem: Description of the problem.

        Returns:
            Troubleshooting guidance.
        """
        prompt = get_troubleshooting_prompt(problem)

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        return response

    async def explain_concept(self, concept: str) -> str:
        """
        Explain a Pt/Pd printing concept.

        Args:
            concept: The concept to explain.

        Returns:
            Explanation.
        """
        prompt = f"""Please explain this Pt/Pd printing concept in detail:

**Concept:** {concept}

Provide:
1. Clear explanation suitable for someone learning alternative processes
2. Why this concept matters for print quality
3. Practical examples
4. Common misconceptions
5. Related concepts to explore"""

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return self.conversation_history.copy()


def create_assistant(
    api_key: str | None = None,
    provider: str = "anthropic",
    database: CalibrationDatabase | None = None,
) -> CalibrationAssistant:
    """
    Create a calibration assistant with the specified configuration.

    Args:
        api_key: API key for the LLM provider.
        provider: LLM provider ("anthropic" or "openai").
        database: Optional calibration database for context.

    Returns:
        Configured CalibrationAssistant.
    """
    from ptpd_calibration.config import LLMProvider

    settings = LLMSettings(
        provider=LLMProvider(provider),
        api_key=api_key,
    )

    return CalibrationAssistant(settings=settings, database=database)
