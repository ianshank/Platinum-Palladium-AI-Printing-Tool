"""
Chemistry Calculator Page Object for PTPD Calibration UI.

Handles interactions with the chemistry calculation tab.
"""

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class ChemistryCalculatorPage(BasePage):
    """Page Object for the Chemistry Calculator tab."""

    TAB_NAME = "Chemistry"

    def navigate_to_chemistry(self) -> None:
        """Navigate to the Chemistry Calculator tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_chemistry_active(self) -> bool:
        """Check if Chemistry Calculator tab is currently active."""
        return self.TAB_NAME.lower() in self.get_active_tab().lower()

    # --- Print Size Settings ---

    def set_print_width(self, width: float) -> None:
        """Set the print width in inches."""
        self.fill_number("Width", width)

    def set_print_height(self, height: float) -> None:
        """Set the print height in inches."""
        self.fill_number("Height", height)

    def set_print_dimensions(self, width: float, height: float) -> None:
        """Set both print dimensions."""
        self.set_print_width(width)
        self.set_print_height(height)

    # --- Chemistry Settings ---

    def select_chemistry_type(self, chemistry: str) -> None:
        """Select the chemistry type (Pt/Pd, Kallitype, etc.)."""
        self.select_dropdown("Chemistry Type", chemistry)

    def set_metal_ratio(self, ratio: float) -> None:
        """Set the platinum to palladium ratio (0-1)."""
        self.fill_number("Pt/Pd Ratio", ratio)

    def select_contrast_agent(self, agent: str) -> None:
        """Select the contrast agent."""
        self.select_dropdown("Contrast Agent", agent)

    def set_contrast_amount(self, drops: float) -> None:
        """Set the number of contrast agent drops."""
        self.fill_number("Contrast Drops", drops)

    def set_coating_factor(self, factor: float) -> None:
        """Set the coating factor multiplier."""
        self.fill_number("Coating Factor", factor)

    # --- Recipe Calculation ---

    def click_calculate(self) -> None:
        """Click to calculate the chemistry recipe."""
        self.click_button("Calculate")
        self.wait_for_gradio_ready()

    def get_recipe_results(self) -> dict:
        """Get the calculated recipe results."""
        results = {}
        try:
            # Look for result rows
            rows = self.find_elements(By.CSS_SELECTOR, ".recipe-row, .result-item, tr")
            for row in rows:
                try:
                    label = row.find_element(By.CSS_SELECTOR, ".label, td:first-child").text
                    value = row.find_element(By.CSS_SELECTOR, ".value, td:last-child").text
                    results[label.strip(":")] = value
                except Exception:
                    continue
        except Exception:
            pass
        return results

    def get_platinum_amount(self) -> str:
        """Get the calculated platinum solution amount."""
        try:
            return self.get_output_text("Platinum")
        except Exception:
            return ""

    def get_palladium_amount(self) -> str:
        """Get the calculated palladium solution amount."""
        try:
            return self.get_output_text("Palladium")
        except Exception:
            return ""

    def get_ferric_oxalate_amount(self) -> str:
        """Get the calculated ferric oxalate amount."""
        try:
            return self.get_output_text("Ferric Oxalate")
        except Exception:
            return ""

    def get_total_solution(self) -> str:
        """Get the total solution volume."""
        try:
            return self.get_output_text("Total")
        except Exception:
            return ""

    # --- Recipe Saving ---

    def save_recipe(self, name: str) -> None:
        """Save the current recipe with a name."""
        self.fill_textbox("Recipe Name", name)
        self.click_button("Save Recipe")
        self.wait_for_gradio_ready()

    def load_recipe(self, name: str) -> None:
        """Load a saved recipe by name."""
        self.select_dropdown("Saved Recipes", name)
        self.click_button("Load Recipe")
        self.wait_for_gradio_ready()

    # --- AI Assistant Integration ---

    def ask_ai_for_recommendation(self, prompt: str) -> str:
        """Ask the AI for chemistry recommendations."""
        self.fill_textbox("Ask AI", prompt)
        self.click_button("Get Recommendation")
        self.wait_for_gradio_ready()
        return self.get_output_text("AI Recommendation")

    # --- Full Workflow ---

    def calculate_recipe(
        self,
        width: float,
        height: float,
        chemistry_type: str = "Platinum/Palladium",
        metal_ratio: float = 0.5,
        contrast_agent: str = "Na2",
        contrast_drops: float = 5.0,
    ) -> dict:
        """
        Calculate a complete chemistry recipe.

        Returns the recipe results.
        """
        self.set_print_dimensions(width, height)
        self.select_chemistry_type(chemistry_type)
        self.set_metal_ratio(metal_ratio)
        self.select_contrast_agent(contrast_agent)
        self.set_contrast_amount(contrast_drops)
        self.click_calculate()
        return self.get_recipe_results()
