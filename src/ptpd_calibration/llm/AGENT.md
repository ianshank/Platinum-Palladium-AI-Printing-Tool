# LLM Directory

## Purpose
Multi-provider LLM integration for AI-powered features: chat assistant, recipe suggestions, troubleshooting, and curve enhancement.

## Key Files
- `client.py` — Multi-provider LLM client supporting Anthropic Claude, OpenAI GPT, and Vertex AI (e.g., Gemini)
- `assistant.py` — Chat assistant orchestration: context building, conversation history, domain-aware responses
- `prompts.py` — Prompt templates for different use cases (calibration help, recipe suggestions, troubleshooting)
- `__init__.py` — Package exports

## Conventions
- **Provider-agnostic**: `client.py` abstracts provider differences — components use a unified interface
- **API keys from config**: Never hardcode keys. Load via `PTPD_LLM_ANTHROPIC_API_KEY` / `PTPD_LLM_OPENAI_API_KEY` (or a generic `PTPD_LLM_API_KEY`) env vars
- **Provider selection**: `PTPD_LLM_PROVIDER` env var (`anthropic`, `openai`, or `vertex`) — see CLAUDE.md "Environment Variables"
- **Prompt templates**: Structured templates in `prompts.py` — domain-specific context about Pt/Pd printing is embedded

## Key Features
- Chat: General Q&A about platinum/palladium printing
- Recipe: Chemistry recipe suggestions based on paper type and desired characteristics
- Troubleshooting: Problem diagnosis based on symptom description
- Curve enhancement: AI-powered curve optimization (called from `../curves/ai_enhance.py`)

## Testing
```bash
pytest tests/unit/ -v -k "llm or assistant or chat"
```
Tests should mock LLM API calls — never call real APIs in tests.

## Pitfalls
- Missing API keys should raise clear errors, not silently fail
- LLM responses are non-deterministic — test for structure, not exact content
- Token limits vary by provider/model — respect context windows
- Prompt injection: Sanitize user input before including in prompts

## Related
- `../curves/ai_enhance.py` — Uses LLM client for curve enhancement
- `../api/server.py` — `/api/chat/*` endpoints call assistant functions
- Frontend: `../../../frontend/src/components/assistant/` — Chat UI
- Frontend: `../../../frontend/src/hooks/useChat.ts` — Chat orchestration hook
