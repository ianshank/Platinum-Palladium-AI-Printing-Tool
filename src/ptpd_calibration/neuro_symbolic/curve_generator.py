"""
Neuro-Symbolic Curve Generator.

This module provides an enhanced curve generator that integrates:
- Physics-informed symbolic constraints
- Knowledge graph for analogical reasoning
- Differentiable symbolic regression for formula discovery

The generator produces curves that:
1. Respect physical constraints (monotonicity, bounds, H&D characteristics)
2. Leverage knowledge from similar papers/chemistry
3. Can discover interpretable formulas for new paper types
4. Provide uncertainty quantification and explanations
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ptpd_calibration.config import (
    CurveSettings,
    NeuroSymbolicSettings,
    get_settings,
)
from ptpd_calibration.core.models import CurveData, ExtractionResult
from ptpd_calibration.core.types import CurveType
from ptpd_calibration.curves.generator import CurveGenerator, TargetCurve
from ptpd_calibration.neuro_symbolic.constraints import (
    ConstrainedCurveOptimizer,
    ConstraintSet,
    ConstraintViolation,
)
from ptpd_calibration.neuro_symbolic.knowledge_graph import (
    InferenceResult,
    PaperChemistryKnowledgeGraph,
)
from ptpd_calibration.neuro_symbolic.symbolic_regression import (
    CurveFormulaDiscovery,
)


class CurveGenerationResult(BaseModel):
    """Complete result from neuro-symbolic curve generation."""

    class Config:
        arbitrary_types_allowed = True

    # Core curve data
    curve: CurveData

    # Constraint information
    constraints_satisfied: bool = True
    constraint_violations: list[ConstraintViolation] = Field(default_factory=list)
    constraint_report: str = ""

    # Knowledge graph reasoning
    knowledge_inference: InferenceResult | None = None
    similar_papers: list[str] = Field(default_factory=list)
    inferred_settings: dict[str, Any] = Field(default_factory=dict)

    # Formula discovery
    discovered_formula: str | None = None
    formula_latex: str | None = None
    formula_r_squared: float | None = None
    formula_interpretation: str | None = None

    # Uncertainty quantification
    uncertainty: NDArray[np.float64] | None = None
    confidence_lower: list[float] | None = None
    confidence_upper: list[float] | None = None

    # Explanations
    explanation: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)


class NeuroSymbolicCurveGenerator:
    """Enhanced curve generator with neuro-symbolic AI capabilities.

    This generator extends the basic CurveGenerator with:
    - Physics-informed constraints that are enforced during generation
    - Knowledge graph reasoning for analogical transfer
    - Symbolic regression for formula discovery
    - Uncertainty quantification
    - Explainable outputs
    """

    def __init__(
        self,
        curve_settings: CurveSettings | None = None,
        neuro_settings: NeuroSymbolicSettings | None = None,
    ):
        """Initialize neuro-symbolic curve generator.

        Args:
            curve_settings: Standard curve generation settings
            neuro_settings: Neuro-symbolic AI settings
        """
        settings = get_settings()
        self.curve_settings = curve_settings or settings.curves
        self.neuro_settings = neuro_settings or settings.neuro_symbolic

        # Initialize components
        self._base_generator = CurveGenerator(self.curve_settings)
        self._constraint_set = ConstraintSet.default_set(self.neuro_settings)
        self._optimizer = ConstrainedCurveOptimizer(self._constraint_set, self.neuro_settings)
        self._knowledge_graph = PaperChemistryKnowledgeGraph(self.neuro_settings)
        self._formula_discovery = CurveFormulaDiscovery(self.neuro_settings)

    def generate(
        self,
        measured_densities: list[float],
        curve_type: CurveType = CurveType.LINEAR,
        target_curve: TargetCurve | None = None,
        name: str = "Neuro-Symbolic Calibration Curve",
        paper_type: str | None = None,
        chemistry: str | None = None,
        enforce_constraints: bool = True,
        discover_formula: bool = False,
        use_knowledge_graph: bool = True,
        quantify_uncertainty: bool = True,
    ) -> CurveGenerationResult:
        """Generate a calibration curve with neuro-symbolic enhancements.

        Args:
            measured_densities: List of measured densities from step tablet
            curve_type: Type of curve to generate
            target_curve: Optional custom target curve
            name: Curve name
            paper_type: Paper type for metadata and knowledge graph
            chemistry: Chemistry description for metadata
            enforce_constraints: Apply physics constraints
            discover_formula: Run symbolic regression to find formula
            use_knowledge_graph: Use knowledge graph for inference
            quantify_uncertainty: Calculate uncertainty estimates

        Returns:
            CurveGenerationResult with curve and all metadata
        """
        reasoning_steps = []
        reasoning_steps.append(
            f"Starting curve generation with {len(measured_densities)} measurements"
        )

        # Step 1: Generate base curve using standard method
        reasoning_steps.append("Generating base curve using classical interpolation")
        base_curve = self._base_generator.generate(
            measured_densities,
            curve_type=curve_type,
            target_curve=target_curve,
            name=name,
            paper_type=paper_type,
            chemistry=chemistry,
        )

        # Get curve values as numpy array
        curve_values = np.array(base_curve.output_values)

        # Step 2: Apply knowledge graph inference
        knowledge_inference = None
        similar_papers = []
        inferred_settings = {}

        if use_knowledge_graph and paper_type:
            reasoning_steps.append(f"Querying knowledge graph for '{paper_type}'")
            knowledge_inference = self._knowledge_graph.infer_settings_for_paper(paper_type)

            if knowledge_inference.confidence > 0.5:
                inferred_settings = knowledge_inference.result_values
                reasoning_steps.extend(knowledge_inference.reasoning_path)

                # Find similar papers
                paper_entity = self._knowledge_graph.get_entity_by_name(paper_type)
                if paper_entity:
                    similar = self._knowledge_graph.find_similar(paper_entity.id, top_k=3)
                    similar_papers = [s.entity_name for s in similar]
                    if similar_papers:
                        reasoning_steps.append(f"Found similar papers: {', '.join(similar_papers)}")
            else:
                reasoning_steps.append("Paper not found in knowledge graph, using defaults")

        # Step 3: Enforce symbolic constraints
        constraint_report = ""
        violations = []

        if enforce_constraints:
            reasoning_steps.append("Evaluating physics constraints")

            # Evaluate constraints on base curve
            is_valid, violations = self._optimizer.validate_curve(curve_values)

            if not is_valid:
                reasoning_steps.append(
                    f"Found {len(violations)} constraint violations, optimizing..."
                )

                # Optimize to satisfy constraints
                opt_result = self._optimizer.optimize(
                    initial_values=curve_values,
                    target_values=curve_values,  # Stay close to original
                )

                curve_values = opt_result["optimized_values"]
                violations = opt_result["violations"]
                constraint_report = opt_result["report"]

                reasoning_steps.append(
                    f"Optimization {'succeeded' if opt_result['all_constraints_satisfied'] else 'partially succeeded'}"
                )
            else:
                reasoning_steps.append("All physics constraints satisfied")
                constraint_report = self._constraint_set.generate_report(curve_values)

        # Step 4: Discover symbolic formula
        discovered_formula = None
        formula_latex = None
        formula_r_squared = None
        formula_interpretation = None

        if discover_formula:
            reasoning_steps.append("Discovering symbolic formula via genetic programming")

            formula_result = self._formula_discovery.discover_formula(
                measured_densities,
                paper_type=paper_type,
            )

            discovered_formula = formula_result["formula"]
            formula_latex = formula_result["latex"]
            formula_r_squared = formula_result["r_squared"]
            formula_interpretation = formula_result["interpretation"]

            reasoning_steps.append(f"Discovered formula: {discovered_formula}")
            reasoning_steps.append(f"R² = {formula_r_squared:.4f}")
            reasoning_steps.append(f"Interpretation: {formula_interpretation}")

        # Step 5: Quantify uncertainty
        uncertainty = None
        confidence_lower = None
        confidence_upper = None

        if quantify_uncertainty:
            reasoning_steps.append("Estimating uncertainty via Monte Carlo sampling")
            uncertainty_result = self._estimate_uncertainty(measured_densities, curve_values)
            uncertainty = uncertainty_result["uncertainty"]
            confidence_lower = uncertainty_result["lower"].tolist()
            confidence_upper = uncertainty_result["upper"].tolist()

            mean_uncertainty = np.mean(uncertainty)
            reasoning_steps.append(f"Mean uncertainty: {mean_uncertainty:.4f}")

        # Step 6: Create final curve data
        final_curve = CurveData(
            name=name,
            curve_type=curve_type,
            paper_type=paper_type,
            chemistry=chemistry,
            input_values=base_curve.input_values,
            output_values=curve_values.tolist(),
            target_curve_type=base_curve.target_curve_type,
            source_extraction_id=base_curve.source_extraction_id,
        )

        # Step 7: Generate explanation
        explanation = self._generate_explanation(
            curve_type=curve_type,
            paper_type=paper_type,
            constraints_satisfied=len(violations) == 0,
            num_violations=len(violations),
            formula=discovered_formula,
            formula_r_squared=formula_r_squared,
            similar_papers=similar_papers,
        )

        return CurveGenerationResult(
            curve=final_curve,
            constraints_satisfied=len(violations) == 0,
            constraint_violations=violations,
            constraint_report=constraint_report,
            knowledge_inference=knowledge_inference,
            similar_papers=similar_papers,
            inferred_settings=inferred_settings,
            discovered_formula=discovered_formula,
            formula_latex=formula_latex,
            formula_r_squared=formula_r_squared,
            formula_interpretation=formula_interpretation,
            uncertainty=uncertainty,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            explanation=explanation,
            reasoning_steps=reasoning_steps,
        )

    def generate_from_extraction(
        self,
        extraction: ExtractionResult,
        curve_type: CurveType = CurveType.LINEAR,
        target_curve: TargetCurve | None = None,
        name: str = "Neuro-Symbolic Calibration Curve",
        paper_type: str | None = None,
        chemistry: str | None = None,
        **kwargs,
    ) -> CurveGenerationResult:
        """Generate curve from extraction result.

        Args:
            extraction: Step tablet extraction result
            curve_type: Type of curve to generate
            target_curve: Optional custom target curve
            name: Curve name
            paper_type: Paper type for metadata
            chemistry: Chemistry description
            **kwargs: Additional arguments passed to generate()

        Returns:
            CurveGenerationResult with curve and metadata
        """
        densities = extraction.get_densities()
        if not densities:
            raise ValueError("No density measurements in extraction")

        result = self.generate(
            measured_densities=densities,
            curve_type=curve_type,
            target_curve=target_curve,
            name=name,
            paper_type=paper_type,
            chemistry=chemistry,
            **kwargs,
        )

        # Link to source extraction
        result.curve.source_extraction_id = extraction.id

        return result

    def generate_for_new_paper(
        self,
        paper_properties: dict[str, Any],
        measured_densities: list[float] | None = None,
        curve_type: CurveType = CurveType.LINEAR,
        name: str = "New Paper Calibration",
    ) -> CurveGenerationResult:
        """Generate curve for a new/unknown paper using analogical reasoning.

        Uses the knowledge graph to find similar papers and transfer
        curve generation parameters.

        Args:
            paper_properties: Properties of the new paper
            measured_densities: Optional measurements (if available)
            curve_type: Type of curve to generate
            name: Curve name

        Returns:
            CurveGenerationResult with inferred settings
        """
        reasoning_steps = ["Generating curve for unknown paper via analogical reasoning"]

        # Infer settings from knowledge graph
        inference = self._knowledge_graph.infer_for_new_paper(paper_properties)

        reasoning_steps.extend(inference.reasoning_path)
        reasoning_steps.append(f"Inference confidence: {inference.confidence:.2f}")

        if measured_densities:
            # Generate from actual measurements
            result = self.generate(
                measured_densities=measured_densities,
                curve_type=curve_type,
                name=name,
                paper_type=inference.result_values.get("paper", "Unknown"),
                use_knowledge_graph=False,  # Already did inference
            )
        else:
            # Generate synthetic curve based on inferred properties
            result = self._generate_synthetic_curve(
                inference.result_values,
                curve_type=curve_type,
                name=name,
            )

        result.knowledge_inference = inference
        result.inferred_settings = inference.result_values
        result.reasoning_steps = reasoning_steps + result.reasoning_steps

        return result

    def _generate_synthetic_curve(
        self,
        inferred_settings: dict[str, Any],
        curve_type: CurveType,
        name: str,
    ) -> CurveGenerationResult:
        """Generate synthetic curve from inferred settings."""
        # Extract inferred parameters
        expected_dmax = inferred_settings.get("expected_dmax", 2.0)
        coating_factor = inferred_settings.get("coating_factor", 1.0)
        metal_ratio = inferred_settings.get("metal_ratio", 0.5)

        # Generate synthetic densities based on typical H&D curve
        num_steps = 21
        x = np.linspace(0, 1, num_steps)

        # Model: D = Dmax * (1 - exp(-k * x^gamma)) + Dmin
        dmin = 0.08
        gamma = 0.8 + 0.2 * metal_ratio  # Pt makes curve slightly steeper
        k = 3.0 * coating_factor

        synthetic_densities = dmin + expected_dmax * (1 - np.exp(-k * np.power(x, gamma)))

        return self.generate(
            measured_densities=synthetic_densities.tolist(),
            curve_type=curve_type,
            name=name,
            paper_type=inferred_settings.get("paper"),
            enforce_constraints=True,
            discover_formula=False,
            use_knowledge_graph=False,
        )

    def _estimate_uncertainty(
        self,
        measured_densities: list[float],
        curve_values: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Estimate uncertainty via Monte Carlo sampling.

        Args:
            measured_densities: Original measurements
            curve_values: Generated curve values

        Returns:
            Dictionary with uncertainty, lower, and upper bounds
        """
        num_samples = self.neuro_settings.uncertainty_num_samples
        confidence = self.neuro_settings.uncertainty_confidence_level

        # Estimate measurement noise from data
        densities = np.array(measured_densities)
        # Use roughness of second derivative as noise estimate
        if len(densities) > 2:
            second_deriv = np.diff(np.diff(densities))
            noise_std = np.std(second_deriv) / np.sqrt(2)  # Approximate
        else:
            noise_std = 0.02  # Default

        noise_std = max(noise_std, 0.01)  # Minimum noise

        # Sample curves
        samples = []
        for _ in range(num_samples):
            # Add noise to measurements
            noisy_densities = densities + np.random.normal(0, noise_std, len(densities))

            # Generate curve from noisy measurements
            try:
                noisy_curve = self._base_generator.generate(noisy_densities.tolist())
                samples.append(np.array(noisy_curve.output_values))
            except Exception:
                continue

        if not samples:
            # Fallback: use constant uncertainty
            return {
                "uncertainty": np.full_like(curve_values, noise_std),
                "lower": curve_values - 2 * noise_std,
                "upper": curve_values + 2 * noise_std,
            }

        samples = np.array(samples)

        # Calculate statistics
        _mean = np.mean(samples, axis=0)  # Reserved for future mean curve output
        std = np.std(samples, axis=0)

        # Confidence interval
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(samples, lower_percentile, axis=0)
        upper = np.percentile(samples, upper_percentile, axis=0)

        return {
            "uncertainty": std,
            "lower": lower,
            "upper": upper,
        }

    def _generate_explanation(
        self,
        curve_type: CurveType,
        paper_type: str | None,
        constraints_satisfied: bool,
        num_violations: int,
        formula: str | None,
        formula_r_squared: float | None,
        similar_papers: list[str],
    ) -> str:
        """Generate human-readable explanation of curve generation."""
        parts = []

        # Curve type explanation
        curve_explanations = {
            CurveType.LINEAR: "This linearization curve corrects for the non-linear response of the paper/chemistry combination, producing a linear relationship between input and output density.",
            CurveType.PAPER_WHITE: "This curve preserves paper white in highlights while linearizing the rest of the tonal scale.",
            CurveType.AESTHETIC: "This aesthetic curve applies an S-curve characteristic for enhanced tonal separation.",
        }
        parts.append(curve_explanations.get(curve_type, "Standard calibration curve."))

        # Paper type
        if paper_type:
            parts.append(f"Calibrated for {paper_type}.")

        # Constraints
        if constraints_satisfied:
            parts.append(
                "All physics constraints satisfied: the curve is monotonically increasing, "
                "within valid density bounds, and follows expected H&D curve characteristics."
            )
        else:
            parts.append(
                f"Note: {num_violations} physics constraint(s) required adjustment. "
                "The curve was optimized to better match expected physical behavior."
            )

        # Formula
        if formula:
            parts.append(f"Discovered formula: {formula}")
            if formula_r_squared:
                parts.append(f"Formula fit quality: R² = {formula_r_squared:.4f}")

        # Similar papers
        if similar_papers:
            parts.append(
                f"Similar papers in knowledge base: {', '.join(similar_papers[:3])}. "
                "Calibration parameters may transfer well between these papers."
            )

        return " ".join(parts)

    def get_knowledge_graph(self) -> PaperChemistryKnowledgeGraph:
        """Get the knowledge graph for direct queries."""
        return self._knowledge_graph

    def get_constraint_set(self) -> ConstraintSet:
        """Get the constraint set for inspection or modification."""
        return self._constraint_set

    def learn_from_result(
        self,
        result: CurveGenerationResult,
        quality_metrics: dict[str, float],
    ) -> None:
        """Learn from a calibration result to improve future predictions.

        Args:
            result: Curve generation result
            quality_metrics: Quality metrics (dmax, linearity, etc.)
        """
        if result.curve.paper_type:
            # Extract chemistry settings from result
            chemistry_settings = {
                "curve_type": result.curve.curve_type.value,
                "discovered_formula": result.discovered_formula,
            }
            chemistry_settings.update(result.inferred_settings)

            # Update knowledge graph
            self._knowledge_graph.learn_from_calibration(
                paper_name=result.curve.paper_type,
                chemistry_settings=chemistry_settings,
                result_metrics=quality_metrics,
            )
