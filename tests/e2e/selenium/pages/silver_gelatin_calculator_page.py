"""
Silver Gelatin Calculator Page Object for PTPD Calibration UI.

Handles interactions with the silver gelatin darkroom printing interface.
"""

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class SilverGelatinCalculatorPage(BasePage):
    """Page Object for the Silver Gelatin Calculator tab."""

    TAB_NAME = "Silver Gelatin"
    PROCESS_TYPE = "silver_gelatin"

    def navigate_to_silver_gelatin(self) -> None:
        """Navigate to the Silver Gelatin Calculator tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_silver_gelatin_active(self) -> bool:
        """Check if Silver Gelatin Calculator tab is currently active."""
        active = self.get_active_tab().lower()
        return "silver" in active or "gelatin" in active

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

    # --- Paper Settings ---

    def select_paper_base(self, base: str) -> None:
        """Select the paper base type (Fiber or RC)."""
        self.select_dropdown("Paper Base", base)

    def select_fiber_paper(self) -> None:
        """Select fiber-based paper."""
        self.select_paper_base("Fiber")

    def select_rc_paper(self) -> None:
        """Select resin-coated paper."""
        self.select_paper_base("RC")

    def select_paper_grade(self, grade: str) -> None:
        """Select the paper contrast grade."""
        self.select_dropdown("Paper Grade", grade)

    def select_grade_2(self) -> None:
        """Select Grade 2 (normal contrast)."""
        self.select_paper_grade("Grade 2")

    def select_variable_contrast(self) -> None:
        """Select variable contrast paper."""
        self.select_paper_grade("Variable Contrast")

    def set_paper_iso(self, iso: float) -> None:
        """Set the paper speed (ISO)."""
        self.fill_number("Paper ISO", iso)

    # --- Developer Settings ---

    def select_developer(self, developer: str) -> None:
        """Select the developer type."""
        self.select_dropdown("Developer", developer)

    def select_dektol(self) -> None:
        """Select Dektol developer."""
        self.select_developer("Dektol")

    def select_d76(self) -> None:
        """Select D-76 developer."""
        self.select_developer("D-76")

    def select_dilution(self, dilution: str) -> None:
        """Select the developer dilution."""
        self.select_dropdown("Dilution", dilution)

    def set_development_temperature(self, temp_c: float) -> None:
        """Set the development temperature in Celsius."""
        self.fill_number("Temperature", temp_c)

    # --- Fixer Settings ---

    def select_fixer(self, fixer: str) -> None:
        """Select the fixer type."""
        self.select_dropdown("Fixer", fixer)

    def check_include_hypo_clear(self, include: bool = True) -> None:
        """Enable or disable hypo clear step."""
        self.check_checkbox("Include Hypo Clear", include)

    # --- Tray Settings ---

    def select_tray_size(self, size: str) -> None:
        """Select the tray size."""
        self.select_dropdown("Tray Size", size)

    def set_num_prints(self, num: int) -> None:
        """Set the number of prints to process."""
        self.fill_number("Number of Prints", num)

    # --- Chemistry Calculation ---

    def click_calculate_chemistry(self) -> None:
        """Click to calculate processing chemistry."""
        self.click_button("Calculate Chemistry")
        self.wait_for_gradio_ready()

    def get_chemistry_results(self) -> dict:
        """Get the calculated chemistry results."""
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

    def get_developer_volume(self) -> str:
        """Get the calculated developer volume."""
        try:
            return self.get_output_text("Developer")
        except Exception:
            return ""

    def get_stop_bath_volume(self) -> str:
        """Get the calculated stop bath volume."""
        try:
            return self.get_output_text("Stop Bath")
        except Exception:
            return ""

    def get_fixer_volume(self) -> str:
        """Get the calculated fixer volume."""
        try:
            return self.get_output_text("Fixer")
        except Exception:
            return ""

    def get_processing_times(self) -> dict:
        """Get all processing times."""
        times = {}
        try:
            times["development"] = self.get_output_text("Development Time")
            times["stop"] = self.get_output_text("Stop Time")
            times["fix"] = self.get_output_text("Fix Time")
            times["wash"] = self.get_output_text("Wash Time")
        except Exception:
            pass
        return times

    # --- Exposure Settings ---

    def set_enlarger_height(self, height_cm: float) -> None:
        """Set the enlarger head height in cm."""
        self.fill_number("Enlarger Height", height_cm)

    def set_f_stop(self, f_stop: float) -> None:
        """Set the lens f-stop."""
        self.fill_number("F-Stop", f_stop)

    def select_light_source(self, source: str) -> None:
        """Select the enlarger light source type."""
        self.select_dropdown("Light Source", source)

    def set_filter_factor(self, factor: float) -> None:
        """Set the filter factor."""
        self.fill_number("Filter Factor", factor)

    def set_negative_density(self, density: float) -> None:
        """Set the negative density."""
        self.fill_number("Negative Density", density)

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
            results["exposure_time"] = self.get_exposure_time()
        except Exception:
            pass
        return results

    # --- Split Filter Settings ---

    def enable_split_filter_mode(self) -> None:
        """Enable split filter printing mode."""
        self.check_checkbox("Split Filter", True)

    def set_shadow_grade(self, grade: float) -> None:
        """Set the shadow (low) filter grade."""
        self.fill_number("Shadow Grade", grade)

    def set_highlight_grade(self, grade: float) -> None:
        """Set the highlight (high) filter grade."""
        self.fill_number("Highlight Grade", grade)

    def set_split_ratio(self, ratio: float) -> None:
        """Set the split ratio (0-1)."""
        self.fill_number("Split Ratio", ratio)

    def click_calculate_split(self) -> None:
        """Click to calculate split filter exposure."""
        self.click_button("Calculate Split")
        self.wait_for_gradio_ready()

    def get_split_filter_results(self) -> dict:
        """Get split filter calculation results."""
        results = {}
        try:
            results["shadow_exposure"] = self.get_output_text("Shadow Exposure")
            results["highlight_exposure"] = self.get_output_text("Highlight Exposure")
        except Exception:
            pass
        return results

    # --- Test Strip ---

    def set_base_exposure(self, seconds: float) -> None:
        """Set the base exposure for test strip."""
        self.fill_number("Base Exposure", seconds)

    def set_num_strips(self, num: int) -> None:
        """Set the number of test strips."""
        self.fill_number("Number of Strips", num)

    def set_increment_factor(self, factor: float) -> None:
        """Set the exposure increment factor."""
        self.fill_number("Increment Factor", factor)

    def click_generate_test_strip(self) -> None:
        """Click to generate test strip times."""
        self.click_button("Generate Test Strip")
        self.wait_for_gradio_ready()

    def get_test_strip_times(self) -> list:
        """Get the generated test strip times."""
        times = []
        try:
            output = self.get_output_text("Test Strip Times")
            # Parse comma-separated or list format
            if output:
                times = [t.strip() for t in output.split(",")]
        except Exception:
            pass
        return times

    # --- Complete Workflows ---

    def calculate_processing_chemistry(
        self,
        width: float,
        height: float,
        paper_base: str = "Fiber",
        developer: str = "Dektol",
        dilution: str = "1:2",
        temperature: float = 20.0,
        num_prints: int = 1,
    ) -> dict:
        """
        Calculate complete processing chemistry.

        Returns chemistry results.
        """
        self.set_print_dimensions(width, height)
        self.select_paper_base(paper_base)
        self.select_developer(developer)
        self.select_dilution(dilution)
        self.set_development_temperature(temperature)
        self.set_num_prints(num_prints)
        self.click_calculate_chemistry()
        return self.get_chemistry_results()

    def calculate_enlarger_exposure(
        self,
        enlarger_height: float = 30.0,
        f_stop: float = 8.0,
        paper_grade: str = "Grade 2",
        negative_density: float = 1.0,
    ) -> dict:
        """
        Calculate enlarger exposure time.

        Returns exposure results.
        """
        self.set_enlarger_height(enlarger_height)
        self.set_f_stop(f_stop)
        self.select_paper_grade(paper_grade)
        self.set_negative_density(negative_density)
        self.click_calculate_exposure()
        return self.get_exposure_results()

    def calculate_split_filter_print(
        self,
        base_exposure: float,
        shadow_grade: float = 5.0,
        highlight_grade: float = 0.0,
        split_ratio: float = 0.5,
    ) -> dict:
        """
        Calculate split filter printing exposures.

        Returns split filter results.
        """
        self.enable_split_filter_mode()
        self.set_base_exposure(base_exposure)
        self.set_shadow_grade(shadow_grade)
        self.set_highlight_grade(highlight_grade)
        self.set_split_ratio(split_ratio)
        self.click_calculate_split()
        return self.get_split_filter_results()

    def complete_darkroom_workflow(
        self,
        width: float,
        height: float,
        paper_base: str = "Fiber",
        developer: str = "Dektol",
        enlarger_height: float = 30.0,
        f_stop: float = 8.0,
    ) -> dict:
        """
        Complete darkroom workflow: chemistry + exposure.

        Returns combined results.
        """
        chemistry = self.calculate_processing_chemistry(
            width=width,
            height=height,
            paper_base=paper_base,
            developer=developer,
        )

        exposure = self.calculate_enlarger_exposure(
            enlarger_height=enlarger_height,
            f_stop=f_stop,
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

    def is_split_filter_enabled(self) -> bool:
        """Check if split filter mode is enabled."""
        try:
            checkbox = self.find_element(By.CSS_SELECTOR, "[aria-label*='Split']")
            return checkbox.get_attribute("checked") == "true"
        except Exception:
            return False
