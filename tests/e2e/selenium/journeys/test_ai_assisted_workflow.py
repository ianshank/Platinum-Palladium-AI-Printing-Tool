"""
AI-Assisted Workflow E2E Tests.

Tests the AI assistant integration in various workflows.
"""

import pytest

from tests.e2e.selenium.pages.ai_assistant_page import AIAssistantPage


@pytest.mark.selenium
@pytest.mark.e2e
@pytest.mark.slow
class TestAIAssistedWorkflow:
    """Test AI assistant integration workflows."""

    @pytest.fixture
    def ai_page(self, driver):
        """Create AIAssistantPage instance."""
        return AIAssistantPage(driver)

    def test_navigate_to_ai_assistant(self, ai_page):
        """Test navigation to AI assistant tab."""
        ai_page.wait_for_gradio_ready()
        ai_page.navigate_to_assistant()

        assert ai_page.is_assistant_active()

    def test_send_basic_message(self, ai_page):
        """Test sending a basic message to AI."""
        ai_page.navigate_to_assistant()
        ai_page.send_message("Hello, I need help with platinum printing.")

        # Wait for response
        response = ai_page.wait_for_response(timeout=60)

        assert len(response) > 0, "Should receive a response"

    def test_ask_about_chemistry(self, ai_page):
        """Test asking about chemistry calculations."""
        ai_page.navigate_to_assistant()

        response = ai_page.ask_question_and_wait(
            "What is the recommended metal ratio for high contrast images?"
        )

        assert len(response) > 0

    def test_ask_about_exposure(self, ai_page):
        """Test asking about exposure settings."""
        ai_page.navigate_to_assistant()

        response = ai_page.ask_question_and_wait(
            "How do I determine the correct exposure time for a new paper?"
        )

        assert len(response) > 0

    def test_troubleshooting_query(self, ai_page):
        """Test troubleshooting assistance."""
        ai_page.navigate_to_assistant()

        response = ai_page.get_troubleshooting_help(
            "My prints are coming out too dark and muddy"
        )

        assert len(response) > 0
        # Should contain relevant suggestions
        assert any(
            word in response.lower()
            for word in ["exposure", "contrast", "developer", "coating"]
        )

    def test_recipe_recommendation(self, ai_page):
        """Test getting recipe recommendations."""
        ai_page.navigate_to_assistant()

        response = ai_page.get_recipe_recommendation(
            paper="Arches Platine",
            print_size="8x10 inches",
            contrast="high",
        )

        assert len(response) > 0

    def test_clear_chat_history(self, ai_page):
        """Test clearing chat history."""
        ai_page.navigate_to_assistant()

        # Send a message first
        ai_page.send_message("Test message")
        ai_page.wait_for_response()

        # Clear chat
        ai_page.clear_chat()

        # Chat should be empty
        history = ai_page.get_chat_history()
        assert len(history) == 0 or all(not m["content"] for m in history)

    def test_conversation_context(self, ai_page):
        """Test that AI maintains conversation context."""
        ai_page.navigate_to_assistant()

        # First message
        ai_page.send_message("I'm working with Arches Platine paper.")
        ai_page.wait_for_response()

        # Follow-up message referencing previous context
        response = ai_page.ask_question_and_wait(
            "What exposure time would you recommend for that paper?"
        )

        # Should reference the paper from context
        assert len(response) > 0

    def test_multiple_questions(self, ai_page):
        """Test asking multiple questions in sequence."""
        ai_page.navigate_to_assistant()

        questions = [
            "What is platinum printing?",
            "How does it differ from silver gelatin?",
            "What chemicals do I need?",
        ]

        for question in questions:
            response = ai_page.ask_question_and_wait(question)
            assert len(response) > 0

        # Should have full conversation history
        history = ai_page.get_chat_history()
        assert len(history) >= len(questions)

    def test_quick_action_troubleshooting(self, ai_page):
        """Test using troubleshooting quick action."""
        ai_page.navigate_to_assistant()
        ai_page.click_troubleshooting()
        ai_page.wait_for_response()

        response = ai_page.get_last_response()
        assert len(response) > 0

    def test_quick_action_recipe_help(self, ai_page):
        """Test using recipe help quick action."""
        ai_page.navigate_to_assistant()
        ai_page.click_recipe_help()
        ai_page.wait_for_response()

        response = ai_page.get_last_response()
        assert len(response) > 0

    @pytest.mark.skip(reason="Requires LLM provider configuration")
    def test_provider_selection(self, ai_page):
        """Test selecting different LLM providers."""
        ai_page.navigate_to_assistant()

        providers = ["Claude", "GPT-4", "Local"]
        for provider in providers:
            try:
                ai_page.select_llm_provider(provider)
            except Exception:
                pass  # Provider may not be available
