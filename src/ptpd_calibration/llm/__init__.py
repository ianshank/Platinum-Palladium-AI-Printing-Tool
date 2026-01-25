"""
LLM integration for AI-powered calibration assistance.

Supports multiple providers:
- Anthropic Claude
- OpenAI GPT
- LM Studio (local)
- Ollama (local)
"""

from ptpd_calibration.llm.assistant import (
    CalibrationAssistant,
    create_assistant,
)
from ptpd_calibration.llm.client import (
    LLMClient,
    AnthropicClient,
    OpenAIClient,
    LocalLLMClient,
    LMStudioClient,
    OllamaClient,
    create_client,
    get_available_providers,
)
from ptpd_calibration.llm.prompts import (
    SYSTEM_PROMPT,
    get_analysis_prompt,
    get_recipe_prompt,
    get_troubleshooting_prompt,
)

__all__ = [
    # Assistant
    "CalibrationAssistant",
    "create_assistant",
    # Clients
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "LocalLLMClient",
    "LMStudioClient",
    "OllamaClient",
    "create_client",
    "get_available_providers",
    # Prompts
    "SYSTEM_PROMPT",
    "get_analysis_prompt",
    "get_recipe_prompt",
    "get_troubleshooting_prompt",
]
