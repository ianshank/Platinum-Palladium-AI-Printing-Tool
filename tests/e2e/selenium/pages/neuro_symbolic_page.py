"""
Neuro-Symbolic Curve Generation Page Object for PTPD Calibration UI.

Handles interactions with the neuro-symbolic AI curve generation features.
"""

from typing import TYPE_CHECKING

from .base_page import BasePage

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

try:
    from selenium.webdriver.common.by import By

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None


class NeuroSymbolicPage(BasePage):
    """Page Object for neuro-symbolic curve generation features."""

    # Tab and section identifiers
    CALIBRATION_TAB = "Calibration"
    ADVANCED_SECTION = "Advanced"
    NEURO_SYMBOLIC_SECTION = "Neuro-Symbolic"

    def __init__(self, driver: "WebDriver"):
        """Initialize the page object."""
        super().__init__(driver)

    def navigate_to_neuro_symbolic(self) -> None:
        """Navigate to the neuro-symbolic curve generation section."""
        self.click_tab(self.CALIBRATION_TAB)
        self.wait_for_gradio_ready()
        # Look for advanced/neuro-symbolic section
        import contextlib

        with contextlib.suppress(Exception):
            self.click_button(self.ADVANCED_SECTION)
        with contextlib.suppress(Exception):
            self.click_button(self.NEURO_SYMBOLIC_SECTION)

    def is_neuro_symbolic_active(self) -> bool:
        """Check if neuro-symbolic features are visible."""
        try:
            # Look for neuro-symbolic specific elements
            self.wait_for_element(
                By.CSS_SELECTOR,
                "[data-testid='neuro-symbolic'], .neuro-symbolic-section, "
                "[class*='neuro'], [id*='neuro']",
                timeout=5,
            )
            return True
        except Exception:
            return False

    # --- Constraint Configuration ---

    def enable_monotonicity_constraint(self, enabled: bool = True) -> None:
        """Enable or disable monotonicity constraint."""
        self.check_checkbox("Monotonicity Constraint", enabled)

    def enable_density_bounds_constraint(self, enabled: bool = True) -> None:
        """Enable or disable density bounds constraint."""
        self.check_checkbox("Density Bounds Constraint", enabled)

    def enable_physics_constraint(self, enabled: bool = True) -> None:
        """Enable or disable physics-based constraint."""
        self.check_checkbox("Physics Constraint", enabled)

    def enable_smoothness_constraint(self, enabled: bool = True) -> None:
        """Enable or disable smoothness constraint."""
        self.check_checkbox("Smoothness Constraint", enabled)

    def set_constraint_weight(self, constraint_name: str, weight: float) -> None:
        """Set the weight for a specific constraint."""
        self.fill_number(f"{constraint_name} Weight", weight)

    def set_density_bounds(self, min_density: float, max_density: float) -> None:
        """Set the density bounds for constraint enforcement."""
        self.fill_number("Min Density", min_density)
        self.fill_number("Max Density", max_density)

    # --- Knowledge Graph Features ---

    def select_paper_from_knowledge_graph(self, paper_name: str) -> None:
        """Select a paper type from the knowledge graph."""
        self.select_dropdown("Paper (Knowledge Graph)", paper_name)

    def enable_analogical_reasoning(self, enabled: bool = True) -> None:
        """Enable or disable analogical reasoning for new materials."""
        self.check_checkbox("Analogical Reasoning", enabled)

    def get_inferred_settings(self) -> dict:
        """Get settings inferred from knowledge graph."""
        settings = {}
        try:
            # Look for inferred settings display
            elements = self.find_elements(By.CSS_SELECTOR, ".inferred-setting, .kg-inference")
            for elem in elements:
                try:
                    key = elem.find_element(By.CSS_SELECTOR, ".setting-name").text
                    value = elem.find_element(By.CSS_SELECTOR, ".setting-value").text
                    settings[key] = value
                except Exception:
                    continue
        except Exception:
            pass
        return settings

    def get_similar_papers(self) -> list[str]:
        """Get list of similar papers from knowledge graph."""
        papers = []
        try:
            elements = self.find_elements(By.CSS_SELECTOR, ".similar-paper, .kg-similar")
            papers = [elem.text for elem in elements]
        except Exception:
            pass
        return papers

    # --- Symbolic Regression Features ---

    def enable_formula_discovery(self, enabled: bool = True) -> None:
        """Enable or disable symbolic regression formula discovery."""
        self.check_checkbox("Discover Formula", enabled)

    def set_formula_complexity(self, complexity: str) -> None:
        """Set the formula complexity level (simple, moderate, complex)."""
        self.select_dropdown("Formula Complexity", complexity)

    def set_regression_generations(self, generations: int) -> None:
        """Set the number of generations for symbolic regression."""
        self.fill_number("Generations", generations)

    def set_population_size(self, size: int) -> None:
        """Set the population size for symbolic regression."""
        self.fill_number("Population Size", size)

    def get_discovered_formula(self) -> str:
        """Get the discovered formula from symbolic regression."""
        try:
            return self.get_output_text("Discovered Formula")
        except Exception:
            return ""

    def get_formula_r_squared(self) -> float:
        """Get the R-squared value of the discovered formula."""
        try:
            text = self.get_output_text("R-squared")
            return float(text.replace("R² = ", "").strip())
        except Exception:
            return 0.0

    # --- Curve Generation with Neuro-Symbolic ---

    def click_generate_neuro_symbolic_curve(self) -> None:
        """Generate curve using neuro-symbolic approach."""
        self.click_button("Generate Neuro-Symbolic Curve")
        self.wait_for_gradio_ready()

    def click_generate_with_constraints(self) -> None:
        """Generate curve with constraint enforcement."""
        self.click_button("Generate with Constraints")
        self.wait_for_gradio_ready()

    def is_constrained_curve_displayed(self) -> bool:
        """Check if a constrained curve is displayed."""
        try:
            self.wait_for_element(
                By.CSS_SELECTOR,
                "canvas, .plot-container, svg, .curve-display",
                timeout=10,
            )
            return True
        except Exception:
            return False

    def get_constraint_violations(self) -> list[str]:
        """Get list of constraint violations if any."""
        violations = []
        try:
            elements = self.find_elements(
                By.CSS_SELECTOR, ".constraint-violation, .violation-warning"
            )
            violations = [elem.text for elem in elements]
        except Exception:
            pass
        return violations

    def get_curve_quality_metrics(self) -> dict:
        """Get quality metrics for the generated curve."""
        metrics = {}
        try:
            elements = self.find_elements(By.CSS_SELECTOR, ".quality-metric, .curve-metric")
            for elem in elements:
                try:
                    name = elem.find_element(By.CSS_SELECTOR, ".metric-name").text
                    value = elem.find_element(By.CSS_SELECTOR, ".metric-value").text
                    metrics[name] = value
                except Exception:
                    continue
        except Exception:
            pass
        return metrics

    # --- Uncertainty Quantification ---

    def enable_uncertainty_quantification(self, enabled: bool = True) -> None:
        """Enable or disable uncertainty quantification."""
        self.check_checkbox("Uncertainty Quantification", enabled)

    def set_monte_carlo_samples(self, samples: int) -> None:
        """Set the number of Monte Carlo samples."""
        self.fill_number("MC Samples", samples)

    def get_confidence_interval(self) -> tuple[float, float]:
        """Get the confidence interval for the curve."""
        try:
            text = self.get_output_text("Confidence Interval")
            # Parse "95% CI: [0.1, 0.9]" format
            values = text.split("[")[1].split("]")[0].split(",")
            return (float(values[0].strip()), float(values[1].strip()))
        except Exception:
            return (0.0, 0.0)

    def is_confidence_band_displayed(self) -> bool:
        """Check if confidence band is displayed on curve plot."""
        try:
            self.wait_for_element(
                By.CSS_SELECTOR,
                ".confidence-band, .uncertainty-band, [class*='band']",
                timeout=5,
            )
            return True
        except Exception:
            return False

    # --- Explanation Features ---

    def get_curve_explanation(self) -> str:
        """Get the explanation for the generated curve."""
        try:
            return self.get_output_text("Explanation")
        except Exception:
            return ""

    def expand_technical_details(self) -> None:
        """Expand the technical details section."""
        import contextlib

        with contextlib.suppress(Exception):
            self.click_button("Technical Details")

    def get_technical_details(self) -> dict:
        """Get technical details about the curve generation."""
        details = {}
        try:
            elements = self.find_elements(By.CSS_SELECTOR, ".technical-detail")
            for elem in elements:
                try:
                    key = elem.find_element(By.CSS_SELECTOR, ".detail-key").text
                    value = elem.find_element(By.CSS_SELECTOR, ".detail-value").text
                    details[key] = value
                except Exception:
                    continue
        except Exception:
            pass
        return details

    # --- Complete Workflow ---

    def generate_neuro_symbolic_curve(
        self,
        enable_monotonicity: bool = True,
        enable_density_bounds: bool = True,
        enable_physics: bool = False,
        enable_smoothness: bool = True,
        enable_formula_discovery: bool = False,
        enable_uncertainty: bool = False,
    ) -> dict:
        """
        Generate a curve with neuro-symbolic features.

        Returns a dictionary with curve information and metrics.
        """
        result = {
            "success": False,
            "formula": "",
            "r_squared": 0.0,
            "violations": [],
            "metrics": {},
            "explanation": "",
        }

        try:
            # Configure constraints
            self.enable_monotonicity_constraint(enable_monotonicity)
            self.enable_density_bounds_constraint(enable_density_bounds)
            self.enable_physics_constraint(enable_physics)
            self.enable_smoothness_constraint(enable_smoothness)

            # Configure formula discovery
            self.enable_formula_discovery(enable_formula_discovery)

            # Configure uncertainty
            self.enable_uncertainty_quantification(enable_uncertainty)

            # Generate curve
            self.click_generate_neuro_symbolic_curve()

            # Collect results
            result["success"] = self.is_constrained_curve_displayed()
            result["violations"] = self.get_constraint_violations()
            result["metrics"] = self.get_curve_quality_metrics()
            result["explanation"] = self.get_curve_explanation()

            if enable_formula_discovery:
                result["formula"] = self.get_discovered_formula()
                result["r_squared"] = self.get_formula_r_squared()

        except Exception as e:
            result["error"] = str(e)

        return result

    def validate_curve_constraints(self) -> dict:
        """
        Validate that the generated curve satisfies all constraints.

        Returns validation results.
        """
        validation = {
            "monotonicity": True,
            "density_bounds": True,
            "physics": True,
            "smoothness": True,
            "overall": True,
        }

        violations = self.get_constraint_violations()

        for violation in violations:
            violation_lower = violation.lower()
            if "monoton" in violation_lower:
                validation["monotonicity"] = False
            if "density" in violation_lower or "bound" in violation_lower:
                validation["density_bounds"] = False
            if "physics" in violation_lower:
                validation["physics"] = False
            if "smooth" in violation_lower:
                validation["smoothness"] = False

        validation["overall"] = all(
            [
                validation["monotonicity"],
                validation["density_bounds"],
                validation["physics"],
                validation["smoothness"],
            ]
        )

        return validation
