"""
AI Assistant Page Object for PTPD Calibration UI.

Handles interactions with the AI chat assistant tab.
"""

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None
    Keys = None


class AIAssistantPage(BasePage):
    """Page Object for the AI Assistant tab."""

    TAB_NAME = "AI Assistant"

    def navigate_to_assistant(self) -> None:
        """Navigate to the AI Assistant tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_assistant_active(self) -> bool:
        """Check if AI Assistant tab is currently active."""
        return self.TAB_NAME.lower() in self.get_active_tab().lower()

    # --- Chat Interaction ---

    def send_message(self, message: str) -> None:
        """Send a message to the AI assistant."""
        textbox = self.wait_for_element(
            By.CSS_SELECTOR, "textarea, input[type='text']"
        )
        textbox.clear()
        textbox.send_keys(message)

        # Find and click send button
        self.click_button("Send")
        self.wait_for_gradio_ready()

    def send_message_with_enter(self, message: str) -> None:
        """Send a message using Enter key."""
        textbox = self.wait_for_element(
            By.CSS_SELECTOR, "textarea, input[type='text']"
        )
        textbox.clear()
        textbox.send_keys(message)
        textbox.send_keys(Keys.RETURN)
        self.wait_for_gradio_ready()

    def get_chat_history(self) -> list[dict]:
        """Get the chat history as a list of messages."""
        messages = []
        try:
            message_elements = self.find_elements(
                By.CSS_SELECTOR, ".message, .chat-message"
            )
            for msg in message_elements:
                role = "user" if "user" in msg.get_attribute("class") else "assistant"
                content = msg.text
                messages.append({"role": role, "content": content})
        except Exception:
            pass
        return messages

    def get_last_response(self) -> str:
        """Get the last assistant response."""
        try:
            responses = self.find_elements(
                By.CSS_SELECTOR, ".assistant-message, .bot-message, .message.assistant"
            )
            if responses:
                return responses[-1].text
        except Exception:
            pass
        return ""

    def wait_for_response(self, timeout: int = 60) -> str:
        """Wait for an AI response and return it."""
        # Wait for loading indicator to appear and disappear
        try:
            self.wait_for_element(
                By.CSS_SELECTOR, ".loading, .typing-indicator", timeout=5
            )
        except Exception:
            pass

        self.wait_for_invisible(
            By.CSS_SELECTOR, ".loading, .typing-indicator", timeout=timeout
        )
        return self.get_last_response()

    # --- Quick Actions ---

    def click_troubleshooting(self) -> None:
        """Click the troubleshooting quick action."""
        self.click_button("Troubleshooting")
        self.wait_for_gradio_ready()

    def click_recipe_help(self) -> None:
        """Click the recipe help quick action."""
        self.click_button("Recipe Help")
        self.wait_for_gradio_ready()

    def click_calibration_help(self) -> None:
        """Click the calibration help quick action."""
        self.click_button("Calibration Help")
        self.wait_for_gradio_ready()

    # --- Context Sharing ---

    def share_calibration_context(self) -> None:
        """Share current calibration context with AI."""
        self.click_button("Share Context")
        self.wait_for_gradio_ready()

    def clear_chat(self) -> None:
        """Clear the chat history."""
        self.click_button("Clear")
        self.wait_for_gradio_ready()

    # --- Provider Settings ---

    def select_llm_provider(self, provider: str) -> None:
        """Select the LLM provider (Claude, GPT, etc.)."""
        self.select_dropdown("Provider", provider)

    def get_current_provider(self) -> str:
        """Get the currently selected LLM provider."""
        try:
            return self.get_output_text("Current Provider")
        except Exception:
            return ""

    # --- Full Workflow ---

    def ask_question_and_wait(self, question: str, timeout: int = 60) -> str:
        """
        Ask a question and wait for the response.

        Returns the AI response.
        """
        self.send_message(question)
        return self.wait_for_response(timeout)

    def get_troubleshooting_help(self, issue: str) -> str:
        """
        Get troubleshooting help for an issue.

        Returns the AI response.
        """
        prompt = f"I'm having an issue: {issue}. What could be the problem and how can I fix it?"
        return self.ask_question_and_wait(prompt)

    def get_recipe_recommendation(
        self,
        paper: str,
        print_size: str,
        contrast: str = "normal",
    ) -> str:
        """
        Get a recipe recommendation from AI.

        Returns the AI response.
        """
        prompt = (
            f"Recommend a chemistry recipe for {paper} paper, "
            f"{print_size} print size, with {contrast} contrast."
        )
        return self.ask_question_and_wait(prompt)
