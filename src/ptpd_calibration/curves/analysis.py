"""
Curve analysis and diagnostic tools.

Provides analysis of calibration curves and measurement quality.
"""

from dataclasses import dataclass

import numpy as np

from ptpd_calibration.core.models import CurveData


@dataclass
class LinearityAnalysis:
    """Results of linearity analysis."""

    max_error: float
    rms_error: float
    mean_error: float
    is_monotonic: bool
    problem_regions: list[tuple[int, int]]  # (start_idx, end_idx)


@dataclass
class CurveComparison:
    """Results of comparing two curves."""

    delta_e_mean: float
    delta_e_max: float
    correlation: float
    significant_differences: list[tuple[float, float, float]]  # (input, delta, region)


class CurveAnalyzer:
    """
    Analyzer for calibration curves and measurements.

    Provides diagnostic information and adjustment suggestions.
    """

    @staticmethod
    def analyze_linearity(
        measured_densities: list[float],
        target_densities: list[float] | None = None,
    ) -> LinearityAnalysis:
        """
        Analyze linearity of measured densities.

        Args:
            measured_densities: Measured density values.
            target_densities: Optional target densities (default: linear).

        Returns:
            LinearityAnalysis with error metrics.
        """
        measured = np.array(measured_densities)
        n = len(measured)

        # Normalize to 0-1
        if measured.max() > measured.min():
            normalized = (measured - measured.min()) / (measured.max() - measured.min())
        else:
            normalized = np.zeros(n)

        # Target (linear if not provided)
        if target_densities is None:
            target = np.linspace(0, 1, n)
        else:
            target = np.array(target_densities)
            if target.max() > target.min():
                target = (target - target.min()) / (target.max() - target.min())
            else:
                target = np.zeros(n)

        # Calculate errors
        errors = normalized - target
        max_error = float(np.max(np.abs(errors)))
        rms_error = float(np.sqrt(np.mean(errors**2)))
        mean_error = float(np.mean(np.abs(errors)))

        # Check monotonicity
        diffs = np.diff(normalized)
        is_monotonic = bool(np.all(diffs >= -0.001))  # Small tolerance

        # Find problem regions (large deviations)
        threshold = max(2 * rms_error, 1e-4)
        problem_mask = np.abs(errors) > threshold
        problem_regions = []

        in_problem = False
        start_idx = 0

        for i, is_problem in enumerate(problem_mask):
            if is_problem and not in_problem:
                start_idx = i
                in_problem = True
            elif not is_problem and in_problem:
                problem_regions.append((start_idx, i - 1))
                in_problem = False

        if in_problem:
            problem_regions.append((start_idx, n - 1))

        return LinearityAnalysis(
            max_error=max_error,
            rms_error=rms_error,
            mean_error=mean_error,
            is_monotonic=is_monotonic,
            problem_regions=problem_regions,
        )

    @staticmethod
    def compare_curves(
        curve1: CurveData,
        curve2: CurveData,
        num_samples: int = 256,
    ) -> CurveComparison:
        """
        Compare two calibration curves.

        Args:
            curve1: First curve.
            curve2: Second curve.
            num_samples: Number of comparison points.

        Returns:
            CurveComparison with difference metrics.
        """
        # Resample both curves to same points
        x = np.linspace(0, 1, num_samples)

        y1 = np.interp(x, curve1.input_values, curve1.output_values)
        y2 = np.interp(x, curve2.input_values, curve2.output_values)

        # Calculate differences
        deltas = np.abs(y1 - y2)

        # Delta E approximation (simple Euclidean)
        delta_e_mean = float(np.mean(deltas) * 100)  # Scale to ~0-100
        delta_e_max = float(np.max(deltas) * 100)

        # Correlation
        correlation = float(np.corrcoef(y1, y2)[0, 1])

        # Find significant differences (> 5%)
        threshold = 0.05
        significant = []
        for _i, (xi, di) in enumerate(zip(x, deltas, strict=False)):
            if di > threshold:
                if xi < 0.2:
                    region = "highlights"
                elif xi < 0.5:
                    region = "midtones"
                elif xi < 0.8:
                    region = "shadows"
                else:
                    region = "deep shadows"
                significant.append((float(xi), float(di * 100), region))

        return CurveComparison(
            delta_e_mean=delta_e_mean,
            delta_e_max=delta_e_max,
            correlation=correlation,
            significant_differences=significant,
        )

    @staticmethod
    def suggest_adjustments(
        measured_densities: list[float],
        target_densities: list[float] | None = None,
    ) -> list[str]:
        """
        Generate suggestions for process adjustments.

        Args:
            measured_densities: Measured density values.
            target_densities: Optional target densities.

        Returns:
            List of adjustment suggestions.
        """
        suggestions = []
        measured = np.array(measured_densities)
        n = len(measured)

        # Analyze basic statistics
        dmin = measured.min()
        dmax = measured.max()
        density_range = dmax - dmin

        # Check density range
        if density_range < 1.5:
            suggestions.append(
                f"Low density range ({density_range:.2f}). Consider increasing exposure time "
                "or using more platinum for higher Dmax."
            )
        elif density_range > 2.5:
            suggestions.append(
                f"Very high density range ({density_range:.2f}). Print may be difficult to view. "
                "Consider reducing exposure or increasing contrast agent."
            )

        # Check Dmin (paper base + fog)
        if dmin > 0.15:
            suggestions.append(
                f"High Dmin ({dmin:.2f}). Check clearing procedure or consider "
                "different paper with less base tone."
            )

        # Analyze tonal distribution
        if n >= 5:
            highlight_density = measured[: n // 4].mean()
            midtone_density = measured[n // 4 : 3 * n // 4].mean()
            shadow_density = measured[3 * n // 4 :].mean()

            # Normalize
            if density_range > 0:
                h_norm = (highlight_density - dmin) / density_range
                m_norm = (midtone_density - dmin) / density_range
                s_norm = (shadow_density - dmin) / density_range

                # Check highlight separation
                if h_norm > 0.35:
                    suggestions.append(
                        "Highlights are too dense. Consider using paper white preservation "
                        "in your digital negative or reducing highlight exposure."
                    )

                # Check midtone contrast
                expected_mid = 0.5
                if abs(m_norm - expected_mid) > 0.15:
                    if m_norm > expected_mid:
                        suggestions.append(
                            "High midtone contrast. Consider reducing Na2 or other "
                            "contrast agent for smoother gradation."
                        )
                    else:
                        suggestions.append(
                            "Low midtone contrast. Consider adding more Na2 or "
                            "increasing Pd ratio for better separation."
                        )

                # Check shadow detail
                if s_norm < 0.85:
                    suggestions.append(
                        "Shadows may be blocking up. Consider reducing exposure or "
                        "increasing shadow separation in the digital negative."
                    )

        # Check for non-monotonicity (solarization)
        diffs = np.diff(measured)
        reversals = np.sum(diffs < -0.05)
        if reversals > 0:
            suggestions.append(
                f"Detected {reversals} density reversals which may indicate solarization. "
                "Consider reducing exposure or UV intensity."
            )

        if not suggestions:
            suggestions.append(
                "Calibration looks good! Density range and tonal distribution "
                "are within normal parameters."
            )

        return suggestions

    @staticmethod
    def analyze_curve(curve: CurveData) -> dict:
        """
        Comprehensive analysis of a calibration curve.

        Args:
            curve: Curve to analyze.

        Returns:
            Dictionary with analysis results.
        """
        inp = np.array(curve.input_values)
        out = np.array(curve.output_values)

        # Basic statistics
        analysis = {
            "name": curve.name,
            "num_points": len(inp),
            "input_range": (float(inp.min()), float(inp.max())),
            "output_range": (float(out.min()), float(out.max())),
        }

        # Monotonicity
        diffs = np.diff(out)
        analysis["is_monotonic"] = bool(np.all(diffs >= 0))
        analysis["num_reversals"] = int(np.sum(diffs < 0))

        # Slope analysis
        slopes = diffs / np.diff(inp)
        analysis["mean_slope"] = float(np.mean(slopes))
        analysis["max_slope"] = float(np.max(slopes))
        analysis["min_slope"] = float(np.min(slopes))

        # Deviation from linear
        linear = inp
        deviation = out - linear
        analysis["max_deviation"] = float(np.max(np.abs(deviation)))
        analysis["mean_deviation"] = float(np.mean(np.abs(deviation)))

        # Characterize curve shape
        mid_idx = len(inp) // 2
        mid_deviation = out[mid_idx] - inp[mid_idx]

        if mid_deviation > 0.1:
            analysis["shape"] = "convex (lifts midtones)"
        elif mid_deviation < -0.1:
            analysis["shape"] = "concave (darkens midtones)"
        else:
            analysis["shape"] = "approximately linear"

        return analysis

    @staticmethod
    def estimate_process_parameters(
        measured_densities: list[float],
        exposure_time: float,
        target_dmax: float = 2.0,
    ) -> dict:
        """
        Estimate process parameter adjustments.

        Args:
            measured_densities: Current measurements.
            exposure_time: Current exposure time in seconds.
            target_dmax: Desired maximum density.

        Returns:
            Dictionary with suggested parameters.
        """
        current_dmax = max(measured_densities)
        current_dmin = min(measured_densities)

        suggestions = {}

        # Estimate exposure adjustment for target Dmax
        if current_dmax > 0:
            # Rough H&D curve approximation
            exposure_factor = (target_dmax / current_dmax) ** 0.7
            suggested_exposure = exposure_time * exposure_factor

            suggestions["exposure_adjustment"] = {
                "current": exposure_time,
                "suggested": suggested_exposure,
                "factor": exposure_factor,
            }

        # Estimate contrast needs
        current_range = current_dmax - current_dmin
        target_range = target_dmax - 0.1  # Assume good dmin

        if current_range < target_range * 0.8:
            suggestions["contrast"] = "increase"
            suggestions["contrast_suggestion"] = (
                "Consider adding Na2 (sodium chloroplatinate) or increasing "
                "platinum ratio for higher contrast."
            )
        elif current_range > target_range * 1.2:
            suggestions["contrast"] = "decrease"
            suggestions["contrast_suggestion"] = (
                "Consider reducing Na2 or switching to pure palladium "
                "for lower contrast."
            )
        else:
            suggestions["contrast"] = "good"
            suggestions["contrast_suggestion"] = "Current contrast is appropriate."

        return suggestions
