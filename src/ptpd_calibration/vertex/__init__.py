"""
Vertex AI integration for the Pt/Pd Calibration Studio.

This module provides:
- Vertex AI Search: Grounded Pt/Pd knowledge base (RAG)
- Gemini Vision: Multimodal print analysis (step tablets, print quality, defects)
- ADK Agents: Production multi-agent system replacing custom agent.py/multi_agent.py
- Memory Bank: Persistent user profiles across sessions

Requires: pip install ptpd-calibration[vertex]
"""

try:
    from ptpd_calibration.vertex.agents import (
        create_adk_agents,
        create_darkroom_coordinator,
    )
    from ptpd_calibration.vertex.corpus import (
        CorpusPreparator,
        prepare_and_upload_corpus,
    )
    from ptpd_calibration.vertex.memory import (
        MemoryBankClient,
        UserProfile,
    )
    from ptpd_calibration.vertex.search import (
        PtPdSearchClient,
        SearchResult,
    )
    from ptpd_calibration.vertex.vision import (
        GeminiVisionAnalyzer,
        analyze_step_tablet,
        diagnose_print_problem,
        evaluate_print_quality,
    )
except ImportError:
    # Vertex AI dependencies not installed
    pass

__all__ = [
    # Search (Layer 1)
    "PtPdSearchClient",
    "SearchResult",
    "CorpusPreparator",
    "prepare_and_upload_corpus",
    # Vision (Layer 2)
    "GeminiVisionAnalyzer",
    "analyze_step_tablet",
    "evaluate_print_quality",
    "diagnose_print_problem",
    # Agents (Layer 3)
    "create_adk_agents",
    "create_darkroom_coordinator",
    # Memory (Layer 4)
    "MemoryBankClient",
    "UserProfile",
]
