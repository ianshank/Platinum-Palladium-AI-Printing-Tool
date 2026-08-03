"""
Dashboard Page Object for PTPD Calibration UI.

Handles interactions with the main dashboard tab.
"""

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class DashboardPage(BasePage):
    """Page Object for the Dashboard tab."""

    # Locators
    TAB_NAME = "Dashboard"

    def navigate_to_dashboard(self) -> None:
        """Navigate to the Dashboard tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_dashboard_active(self) -> bool:
        """Check if Dashboard tab is currently active."""
        return self.TAB_NAME.lower() in self.get_active_tab().lower()

    def get_calibration_summary(self) -> dict:
        """Get the calibration summary information."""
        summary = {}
        try:
            # Look for summary sections
            sections = self.find_elements(By.CSS_SELECTOR, ".summary-section, .stat-card")
            for section in sections:
                label = section.find_element(By.CSS_SELECTOR, ".label, .title").text
                value = section.find_element(By.CSS_SELECTOR, ".value, .stat").text
                summary[label] = value
        except Exception:
            pass
        return summary

    def get_recent_calibrations(self) -> list[dict]:
        """Get list of recent calibrations from the dashboard."""
        calibrations = []
        try:
            rows = self.find_elements(By.CSS_SELECTOR, ".calibration-row, tr")
            for row in rows[1:]:  # Skip header row
                cells = row.find_elements(By.CSS_SELECTOR, "td")
                if len(cells) >= 3:
                    calibrations.append(
                        {
                            "paper": cells[0].text,
                            "date": cells[1].text,
                            "status": cells[2].text,
                        }
                    )
        except Exception:
            pass
        return calibrations

    def click_quick_action(self, action_name: str) -> None:
        """Click a quick action button on the dashboard."""
        buttons = self.find_elements(By.CSS_SELECTOR, ".quick-action, button")
        for btn in buttons:
            if action_name.lower() in btn.text.lower():
                btn.click()
                return
        raise ValueError(f"Quick action '{action_name}' not found")

    def get_system_status(self) -> dict:
        """Get the system status information."""
        status = {}
        try:
            indicators = self.find_elements(By.CSS_SELECTOR, ".status-indicator")
            for indicator in indicators:
                name = indicator.find_element(By.CSS_SELECTOR, ".name").text
                state = indicator.find_element(By.CSS_SELECTOR, ".state").text
                status[name] = state
        except Exception:
            pass
        return status

    def refresh_dashboard(self) -> None:
        """Refresh the dashboard data."""
        try:
            refresh_btn = self.wait_for_clickable(
                By.CSS_SELECTOR, "button[title='Refresh'], .refresh-button"
            )
            refresh_btn.click()
            self.wait_for_gradio_ready()
        except Exception:
            # Fallback to page refresh
            self.refresh()
