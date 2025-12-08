"""
Cyanotype Calculator Page Object for PTPD Calibration UI.

Handles interactions with the cyanotype chemistry and exposure calculation interface.
"""

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class CyanotypeCalculatorPage(BasePage):
    """Page Object for the Cyanotype Calculator tab."""

    TAB_NAME = "Cyanotype"
    PROCESS_TYPE = "cyanotype"

    def navigate_to_cyanotype(self) -> None:
        """Navigate to the Cyanotype Calculator tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_cyanotype_active(self) -> bool:
        """Check if Cyanotype Calculator tab is currently active."""
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

    # --- Formula Settings ---

    def select_formula(self, formula: str) -> None:
        """Select the cyanotype formula (Classic, New, Ware, Rex)."""
        self.select_dropdown("Formula", formula)

    def select_classic_formula(self) -> None:
        """Select the classic cyanotype formula."""
        self.select_formula("Classic")

    def select_new_formula(self) -> None:
        """Select the new cyanotype formula (Mike Ware)."""
        self.select_formula("New")

    # --- Paper Settings ---

    def select_paper_type(self, paper_type: str) -> None:
        """Select the paper type."""
        self.select_dropdown("Paper Type", paper_type)

    def select_cotton_rag(self) -> None:
        """Select cotton rag paper."""
        self.select_paper_type("Cotton Rag")

    def select_watercolor_paper(self) -> None:
        """Select watercolor paper."""
        self.select_paper_type("Watercolor")

    # --- Concentration Settings ---

    def set_concentration_factor(self, factor: float) -> None:
        """Set the concentration factor."""
        self.fill_number("Concentration", factor)

    def set_margin(self, margin_inches: float) -> None:
        """Set the margin in inches."""
        self.fill_number("Margin", margin_inches)

    # --- Chemistry Calculation ---

    def click_calculate_chemistry(self) -> None:
        """Click to calculate the chemistry recipe."""
        self.click_button("Calculate Chemistry")
        self.wait_for_gradio_ready()

    def get_chemistry_results(self) -> dict:
        """Get the calculated chemistry recipe results."""
        results = {}
        try:
            rows = self.find_elements(
                By.CSS_SELECTOR, ".recipe-row, .result-item, tr"
            )
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

    def get_solution_a_amount(self) -> str:
        """Get the calculated Solution A (FAC) amount."""
        try:
            return self.get_output_text("Solution A")
        except Exception:
            return ""

    def get_solution_b_amount(self) -> str:
        """Get the calculated Solution B (Potassium Ferricyanide) amount."""
        try:
            return self.get_output_text("Solution B")
        except Exception:
            return ""

    def get_total_volume(self) -> str:
        """Get the total sensitizer volume."""
        try:
            return self.get_output_text("Total Volume")
        except Exception:
            return ""

    # --- Exposure Settings ---

    def select_uv_source(self, source: str) -> None:
        """Select the UV light source."""
        self.select_dropdown("UV Source", source)

    def select_sunlight(self) -> None:
        """Select sunlight as UV source."""
        self.select_uv_source("Sunlight")

    def select_bl_tubes(self) -> None:
        """Select BL tubes as UV source."""
        self.select_uv_source("BL Tubes")

    def set_negative_density(self, density: float) -> None:
        """Set the negative density."""
        self.fill_number("Negative Density", density)

    def set_humidity(self, humidity: float) -> None:
        """Set the humidity percentage."""
        self.fill_number("Humidity", humidity)

    def set_uv_distance(self, distance: float) -> None:
        """Set the UV source distance in inches."""
        self.fill_number("Distance", distance)

    # --- Exposure Calculation ---

    def click_calculate_exposure(self) -> None:
        """Click to calculate exposure time."""
        self.click_button("Calculate Exposure")
        self.wait_for_gradio_ready()

    def get_exposure_time(self) -> str:
        """Get the calculated exposure time."""
        try:
            return self.get_output_text("Exposure Time")
        except Exception:
            return ""

    def get_exposure_results(self) -> dict:
        """Get all exposure calculation results."""
        results = {}
        try:
            exposure_time = self.get_exposure_time()
            if exposure_time:
                results["exposure_time"] = exposure_time
        except Exception:
            pass
        return results

    # --- Stock Solution Preparation ---

    def set_stock_volume(self, volume_ml: float) -> None:
        """Set the desired stock solution volume."""
        self.fill_number("Stock Volume", volume_ml)

    def click_calculate_stock(self) -> None:
        """Click to calculate stock solution preparation."""
        self.click_button("Calculate Stock")
        self.wait_for_gradio_ready()

    def get_stock_solution_instructions(self) -> str:
        """Get the stock solution preparation instructions."""
        try:
            return self.get_markdown_content()
        except Exception:
            return ""

    # --- Complete Workflows ---

    def calculate_chemistry_recipe(
        self,
        width: float,
        height: float,
        formula: str = "Classic",
        paper_type: str = "Cotton Rag",
        concentration: float = 1.0,
    ) -> dict:
        """
        Calculate a complete cyanotype chemistry recipe.

        Returns the recipe results.
        """
        self.set_print_dimensions(width, height)
        self.select_formula(formula)
        self.select_paper_type(paper_type)
        self.set_concentration_factor(concentration)
        self.click_calculate_chemistry()
        return self.get_chemistry_results()

    def calculate_exposure_time(
        self,
        negative_density: float = 1.6,
        uv_source: str = "BL Tubes",
        humidity: float = 50.0,
        distance: float = 4.0,
    ) -> dict:
        """
        Calculate exposure time for cyanotype printing.

        Returns exposure results.
        """
        self.set_negative_density(negative_density)
        self.select_uv_source(uv_source)
        self.set_humidity(humidity)
        self.set_uv_distance(distance)
        self.click_calculate_exposure()
        return self.get_exposure_results()

    def complete_cyanotype_workflow(
        self,
        width: float,
        height: float,
        formula: str = "Classic",
        paper_type: str = "Cotton Rag",
        negative_density: float = 1.6,
        uv_source: str = "BL Tubes",
    ) -> dict:
        """
        Complete cyanotype workflow: chemistry + exposure.

        Returns combined results.
        """
        chemistry = self.calculate_chemistry_recipe(
            width=width,
            height=height,
            formula=formula,
            paper_type=paper_type,
        )

        exposure = self.calculate_exposure_time(
            negative_density=negative_density,
            uv_source=uv_source,
        )

        return {
            "chemistry": chemistry,
            "exposure": exposure,
        }

    # --- Validation Helpers ---

    def is_error_displayed(self) -> bool:
        """Check if an error message is displayed."""
        try:
            errors = self.find_elements(By.CSS_SELECTOR, ".error, .error-message, [class*='error']")
            return len(errors) > 0 and any(e.is_displayed() for e in errors)
        except Exception:
            return False

    def get_error_message(self) -> str:
        """Get the displayed error message."""
        try:
            error = self.find_element(By.CSS_SELECTOR, ".error, .error-message, [class*='error']")
            return error.text
        except Exception:
            return ""

    def clear_all_inputs(self) -> None:
        """Clear all input fields."""
        try:
            inputs = self.find_elements(By.CSS_SELECTOR, "input[type='number'], input[type='text']")
            for inp in inputs:
                inp.clear()
        except Exception:
            pass
