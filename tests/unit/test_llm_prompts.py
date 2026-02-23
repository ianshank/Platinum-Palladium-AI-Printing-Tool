"""
Unit tests for LLM prompt generators.
"""

from ptpd_calibration.llm.prompts import (
    SYSTEM_PROMPT,
    get_analysis_prompt,
    get_comparison_prompt,
    get_paper_recommendation_prompt,
    get_recipe_prompt,
    get_troubleshooting_prompt,
)


class TestSystemPrompt:
    """Verify the system prompt is well-formed."""

    def test_is_nonempty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_mentions_platinum_palladium(self) -> None:
        assert "platinum" in SYSTEM_PROMPT.lower()
        assert "palladium" in SYSTEM_PROMPT.lower()

    def test_covers_key_topics(self) -> None:
        topics = ["Chemistry", "Paper", "Exposure", "Calibration", "Troubleshooting"]
        for t in topics:
            assert t in SYSTEM_PROMPT, f"Missing topic: {t}"


class TestGetAnalysisPrompt:
    """Tests for get_analysis_prompt."""

    def test_includes_calibration_fields(self) -> None:
        data = {
            "paper_type": "Arches Platine",
            "metal_ratio": 0.7,
            "dmax": 1.65,
            "density_range": 1.20,
        }
        prompt = get_analysis_prompt(data)
        assert "Arches Platine" in prompt
        assert "70%" in prompt  # 0.7 -> 70%
        assert "1.65" in prompt
        assert "1.20" in prompt

    def test_handles_missing_keys_gracefully(self) -> None:
        prompt = get_analysis_prompt({})
        assert "Unknown" in prompt
        assert "0.00" in prompt  # default dmax


class TestGetRecipePrompt:
    """Tests for get_recipe_prompt."""

    def test_includes_paper_and_characteristics(self) -> None:
        prompt = get_recipe_prompt("Bergger COT320", "warm, deep blacks")
        assert "Bergger COT320" in prompt
        assert "warm, deep blacks" in prompt
        assert "Metal ratio" in prompt or "metal ratio" in prompt.lower()


class TestGetTroubleshootingPrompt:
    """Tests for get_troubleshooting_prompt."""

    def test_includes_problem_description(self) -> None:
        prompt = get_troubleshooting_prompt("Bronzing in shadow areas")
        assert "Bronzing in shadow areas" in prompt
        assert "troubleshoot" in prompt.lower() or "diagnos" in prompt.lower()


class TestGetComparisonPrompt:
    """Tests for get_comparison_prompt."""

    def test_includes_both_records(self) -> None:
        r1 = {"paper_type": "Arches", "dmax": 1.5, "density_range": 1.1}
        r2 = {"paper_type": "Bergger", "dmax": 1.7, "density_range": 1.3}
        prompt = get_comparison_prompt(r1, r2)
        assert "Arches" in prompt
        assert "Bergger" in prompt
        assert "1" in prompt  # Calibration 1


class TestGetPaperRecommendationPrompt:
    """Tests for get_paper_recommendation_prompt."""

    def test_includes_requirements(self) -> None:
        reqs = {
            "budget": "moderate",
            "print_size": "16x20",
            "tone": "warm neutral",
            "experience": "intermediate",
        }
        prompt = get_paper_recommendation_prompt(reqs)
        assert "moderate" in prompt
        assert "16x20" in prompt
        assert "warm neutral" in prompt

    def test_handles_missing_requirements(self) -> None:
        prompt = get_paper_recommendation_prompt({})
        assert "Not specified" in prompt
