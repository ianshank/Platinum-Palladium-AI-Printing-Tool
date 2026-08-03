"""
Calibration Wizard Page Object for PTPD Calibration UI.

Handles interactions with the calibration workflow tab.
"""

from pathlib import Path

from .base_page import BasePage

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class CalibrationWizardPage(BasePage):
    """Page Object for the Calibration Wizard tab."""

    TAB_NAME = "Calibration"

    def navigate_to_wizard(self) -> None:
        """Navigate to the Calibration Wizard tab."""
        self.click_tab(self.TAB_NAME)
        self.wait_for_gradio_ready()

    def is_wizard_active(self) -> bool:
        """Check if Calibration Wizard tab is currently active."""
        return self.TAB_NAME.lower() in self.get_active_tab().lower()

    # --- Step Tablet Upload ---

    def upload_step_tablet(self, file_path: str | Path) -> None:
        """Upload a step tablet image for analysis."""
        self.upload_file("Step Tablet", str(file_path))
        self.wait_for_gradio_ready()

    def get_upload_preview(self) -> bool:
        """Check if upload preview is visible."""
        try:
            self.wait_for_element(By.CSS_SELECTOR, ".image-preview, img", timeout=10)
            return True
        except Exception:
            return False

    # --- Step Tablet Analysis ---

    def click_analyze(self) -> None:
        """Click the Analyze button to process the step tablet."""
        self.click_button("Analyze")
        self.wait_for_gradio_ready()

    def get_detected_patches(self) -> int:
        """Get the number of detected patches from analysis."""
        try:
            text = self.get_output_text("Detected Patches")
            return int(text)
        except Exception:
            return 0

    def get_density_values(self) -> list[float]:
        """Get the extracted density values."""
        try:
            text = self.get_output_text("Density Values")
            # Parse comma-separated values
            values = [float(v.strip()) for v in text.split(",")]
            return values
        except Exception:
            return []

    # --- Paper and Chemistry Settings ---

    def select_paper_type(self, paper: str) -> None:
        """Select the paper type."""
        self.select_dropdown("Paper Type", paper)

    def select_chemistry_type(self, chemistry: str) -> None:
        """Select the chemistry type."""
        self.select_dropdown("Chemistry", chemistry)

    def set_metal_ratio(self, ratio: float) -> None:
        """Set the platinum/palladium metal ratio."""
        self.fill_number("Metal Ratio", ratio)

    def set_exposure_time(self, seconds: float) -> None:
        """Set the exposure time."""
        self.fill_number("Exposure Time", seconds)

    def set_contrast_agent(self, agent: str, amount: float) -> None:
        """Set the contrast agent and amount."""
        self.select_dropdown("Contrast Agent", agent)
        self.fill_number("Contrast Amount", amount)

    def select_developer(self, developer: str) -> None:
        """Select the developer type."""
        self.select_dropdown("Developer", developer)

    # --- Environment Settings ---

    def set_humidity(self, humidity: float) -> None:
        """Set the humidity value."""
        self.fill_number("Humidity", humidity)

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature value."""
        self.fill_number("Temperature", temperature)

    # --- Curve Generation ---

    def click_generate_curve(self) -> None:
        """Click to generate the calibration curve."""
        self.click_button("Generate Curve")
        self.wait_for_gradio_ready()

    def is_curve_displayed(self) -> bool:
        """Check if a curve is displayed."""
        try:
            self.wait_for_element(By.CSS_SELECTOR, "canvas, .plot-container, svg", timeout=10)
            return True
        except Exception:
            return False

    def get_curve_type(self) -> str:
        """Get the current curve type displayed."""
        try:
            return self.get_output_text("Curve Type")
        except Exception:
            return ""

    # --- Curve Export ---

    def click_export_quad(self) -> None:
        """Click to export as QTR .quad file."""
        self.click_button("Export .quad")
        self.wait_for_gradio_ready()

    def click_export_csv(self) -> None:
        """Click to export as CSV."""
        self.click_button("Export CSV")
        self.wait_for_gradio_ready()

    def get_export_download_link(self) -> str | None:
        """Get the download link for the exported file."""
        try:
            link = self.wait_for_element(By.CSS_SELECTOR, "a[download], .download-link")
            return link.get_attribute("href")
        except Exception:
            return None

    # --- Save Calibration ---

    def click_save_calibration(self) -> None:
        """Save the calibration to the database."""
        self.click_button("Save Calibration")
        self.wait_for_gradio_ready()

    def get_save_confirmation(self) -> str:
        """Get the save confirmation message."""
        try:
            return self.get_output_text("Status")
        except Exception:
            return ""

    # --- Complete Workflow ---

    def complete_calibration_workflow(
        self,
        step_tablet_path: str | Path,
        paper_type: str = "Arches Platine",
        chemistry_type: str = "Platinum/Palladium",
        metal_ratio: float = 0.5,
        exposure_time: float = 180.0,
    ) -> bool:
        """
        Complete the full calibration workflow.

        Returns True if workflow completed successfully.
        """
        try:
            # Step 1: Upload step tablet
            self.upload_step_tablet(step_tablet_path)

            # Step 2: Configure settings
            self.select_paper_type(paper_type)
            self.select_chemistry_type(chemistry_type)
            self.set_metal_ratio(metal_ratio)
            self.set_exposure_time(exposure_time)

            # Step 3: Analyze
            self.click_analyze()

            # Step 4: Generate curve
            self.click_generate_curve()

            # Verify curve is displayed
            return self.is_curve_displayed()
        except Exception:
            return False
