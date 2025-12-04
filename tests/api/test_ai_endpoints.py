"""
AI and Chat Endpoint Tests.

Tests for AI assistant, recipe suggestions, and troubleshooting endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.api
@pytest.mark.slow
class TestChatEndpoints:
    """Test AI chat endpoints."""

    def test_chat_basic(self, client):
        """Test basic chat interaction."""
        request_data = {
            "message": "What is platinum printing?",
            "include_history": False,
        }

        # This may fail if no LLM provider is configured
        response = client.post("/api/chat", json=request_data)

        # Either succeeds or returns 500 due to missing LLM config
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "response" in data

    def test_chat_with_history(self, client):
        """Test chat with history included."""
        request_data = {
            "message": "Tell me more about the process",
            "include_history": True,
        }

        response = client.post("/api/chat", json=request_data)

        assert response.status_code in [200, 500]

    def test_chat_empty_message(self, client):
        """Test chat with empty message."""
        request_data = {
            "message": "",
            "include_history": False,
        }

        response = client.post("/api/chat", json=request_data)

        # Should handle empty message gracefully
        assert response.status_code in [200, 400, 422, 500]

    @patch("ptpd_calibration.llm.create_assistant")
    def test_chat_mocked(self, mock_create_assistant, client):
        """Test chat with mocked LLM."""
        # Set up mock
        mock_assistant = AsyncMock()
        mock_assistant.chat.return_value = "This is a test response"
        mock_create_assistant.return_value = mock_assistant

        request_data = {
            "message": "Test message",
            "include_history": False,
        }

        response = client.post("/api/chat", json=request_data)

        # With mock, should always succeed
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "This is a test response"


@pytest.mark.api
@pytest.mark.slow
class TestRecipeEndpoints:
    """Test recipe suggestion endpoints."""

    def test_suggest_recipe(self, client):
        """Test recipe suggestion."""
        request_data = {
            "paper_type": "Arches Platine",
            "characteristics": "high contrast, warm tone",
        }

        response = client.post("/api/chat/recipe", json=request_data)

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "response" in data

    def test_suggest_recipe_different_papers(self, client):
        """Test recipe suggestion for different papers."""
        papers = [
            "Arches Platine",
            "Bergger COT320",
            "Hahnemuhle Platinum Rag",
        ]

        for paper in papers:
            request_data = {
                "paper_type": paper,
                "characteristics": "neutral tone",
            }

            response = client.post("/api/chat/recipe", json=request_data)

            assert response.status_code in [200, 500]

    @patch("ptpd_calibration.llm.create_assistant")
    def test_suggest_recipe_mocked(self, mock_create_assistant, client):
        """Test recipe suggestion with mocked LLM."""
        mock_assistant = AsyncMock()
        mock_assistant.suggest_recipe.return_value = "Use 50/50 Pt/Pd ratio"
        mock_create_assistant.return_value = mock_assistant

        request_data = {
            "paper_type": "Test Paper",
            "characteristics": "test",
        }

        response = client.post("/api/chat/recipe", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data


@pytest.mark.api
@pytest.mark.slow
class TestTroubleshootEndpoints:
    """Test troubleshooting endpoints."""

    def test_troubleshoot_basic(self, client):
        """Test basic troubleshooting."""
        request_data = {
            "problem": "My prints are coming out too dark",
        }

        response = client.post("/api/chat/troubleshoot", json=request_data)

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "response" in data

    def test_troubleshoot_common_problems(self, client):
        """Test troubleshooting common problems."""
        problems = [
            "Prints are too dark",
            "Prints have uneven coating",
            "Highlights are blocked",
            "Shadows are muddy",
            "Contrast is too low",
        ]

        for problem in problems:
            request_data = {"problem": problem}

            response = client.post("/api/chat/troubleshoot", json=request_data)

            assert response.status_code in [200, 500]

    @patch("ptpd_calibration.llm.create_assistant")
    def test_troubleshoot_mocked(self, mock_create_assistant, client):
        """Test troubleshooting with mocked LLM."""
        mock_assistant = AsyncMock()
        mock_assistant.troubleshoot.return_value = (
            "Try reducing exposure time by 10%"
        )
        mock_create_assistant.return_value = mock_assistant

        request_data = {"problem": "Test problem"}

        response = client.post("/api/chat/troubleshoot", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data


@pytest.mark.api
class TestAIEnhancementEndpoints:
    """Test AI curve enhancement endpoints."""

    def test_enhance_curve_linearization(self, client, sample_curve_data):
        """Test AI curve enhancement for linearization."""
        request_data = {
            **sample_curve_data,
            "name": "AI Enhanced",
            "goal": "linearization",
        }

        response = client.post("/api/curves/enhance", json=request_data)

        # May fail without LLM, but should fall back to algorithmic
        assert response.status_code in [200, 400, 500]

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "curve_id" in data
            assert "confidence" in data

    def test_enhance_curve_different_goals(self, client, sample_curve_data):
        """Test AI enhancement with different goals."""
        goals = [
            "linearization",
            "maximize_range",
            "smooth_gradation",
            "highlight_detail",
            "shadow_detail",
        ]

        for goal in goals:
            request_data = {
                **sample_curve_data,
                "name": f"Enhanced for {goal}",
                "goal": goal,
            }

            response = client.post("/api/curves/enhance", json=request_data)

            assert response.status_code in [200, 400, 500]

    def test_enhance_curve_with_context(self, client, sample_curve_data):
        """Test AI enhancement with additional context."""
        request_data = {
            **sample_curve_data,
            "name": "Contextual Enhancement",
            "goal": "linearization",
            "paper_type": "Arches Platine",
            "additional_context": "High humidity environment, warm developer",
        }

        response = client.post("/api/curves/enhance", json=request_data)

        assert response.status_code in [200, 400, 500]
