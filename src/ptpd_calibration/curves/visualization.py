"""
Curve visualization module for PTPD Calibration System.

Provides comprehensive curve plotting, comparison, and statistics display.
"""

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ptpd_calibration.core.models import CurveData


class PlotStyle(str, Enum):
    """Available plot styles."""

    LINE = "line"
    LINE_MARKERS = "line_markers"
    SCATTER = "scatter"
    AREA = "area"
    STEP = "step"


class ColorScheme(str, Enum):
    """Predefined color schemes for curve visualization."""

    PLATINUM = "platinum"  # Warm metallic tones
    MONOCHROME = "monochrome"  # Grayscale
    VIBRANT = "vibrant"  # High contrast colors
    PASTEL = "pastel"  # Soft colors
    ACCESSIBLE = "accessible"  # Colorblind-friendly


@dataclass
class CurveStatistics:
    """Statistics computed from a curve."""

    name: str
    num_points: int
    input_min: float
    input_max: float
    output_min: float
    output_max: float
    gamma: float  # Approximate gamma value
    midpoint_value: float  # Output at input=0.5
    is_monotonic: bool
    max_slope: float
    min_slope: float
    average_slope: float
    linearity_error: float  # RMS deviation from linear

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "num_points": self.num_points,
            "input_range": (self.input_min, self.input_max),
            "output_range": (self.output_min, self.output_max),
            "gamma": round(self.gamma, 3),
            "midpoint_value": round(self.midpoint_value, 4),
            "is_monotonic": self.is_monotonic,
            "max_slope": round(self.max_slope, 4),
            "min_slope": round(self.min_slope, 4),
            "average_slope": round(self.average_slope, 4),
            "linearity_error": round(self.linearity_error, 4),
        }


@dataclass
class CurveComparisonResult:
    """Result of comparing two or more curves."""

    curve_names: list[str]
    max_difference: float
    average_difference: float
    rms_difference: float
    correlation: float
    difference_curve: Optional[tuple[list[float], list[float]]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "curves_compared": self.curve_names,
            "max_difference": round(self.max_difference, 4),
            "average_difference": round(self.average_difference, 4),
            "rms_difference": round(self.rms_difference, 4),
            "correlation": round(self.correlation, 4),
        }


@dataclass
class VisualizationConfig:
    """Configuration for curve visualization."""

    # Figure settings
    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 100
    background_color: str = "#FAF8F5"
    grid_alpha: float = 0.3

    # Line settings
    line_width: float = 2.0
    marker_size: float = 6.0
    reference_line_alpha: float = 0.5
    reference_line_style: str = "--"

    # Color settings
    color_scheme: ColorScheme = ColorScheme.PLATINUM
    custom_colors: list[str] = field(default_factory=list)

    # Labels and titles
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

    # Display options
    show_grid: bool = True
    show_legend: bool = True
    show_reference_line: bool = True
    show_statistics: bool = False
    show_difference: bool = False

    # Axis settings
    x_label: str = "Input"
    y_label: str = "Output"
    x_limits: Optional[tuple[float, float]] = None
    y_limits: Optional[tuple[float, float]] = None

    def get_color_palette(self, num_colors: int) -> list[str]:
        """Get color palette based on scheme."""
        if self.custom_colors:
            return self.custom_colors[:num_colors]

        palettes = {
            ColorScheme.PLATINUM: [
                "#8B7355",  # Platinum brown
                "#B8860B",  # Dark goldenrod
                "#CD853F",  # Peru
                "#D2691E",  # Chocolate
                "#8B4513",  # Saddle brown
                "#A0522D",  # Sienna
                "#6B4423",  # Dark brown
                "#DAA520",  # Goldenrod
            ],
            ColorScheme.MONOCHROME: [
                "#1a1a1a",
                "#4d4d4d",
                "#808080",
                "#999999",
                "#b3b3b3",
                "#333333",
                "#666666",
                "#cccccc",
            ],
            ColorScheme.VIBRANT: [
                "#E63946",
                "#457B9D",
                "#2A9D8F",
                "#E9C46A",
                "#F4A261",
                "#264653",
                "#A8DADC",
                "#1D3557",
            ],
            ColorScheme.PASTEL: [
                "#FFB5A7",
                "#A8DADC",
                "#B5E48C",
                "#E7C6FF",
                "#FFD6A5",
                "#CAFFBF",
                "#FDFFB6",
                "#BDE0FE",
            ],
            ColorScheme.ACCESSIBLE: [
                "#0072B2",  # Blue
                "#D55E00",  # Vermillion
                "#009E73",  # Bluish green
                "#CC79A7",  # Reddish purple
                "#F0E442",  # Yellow
                "#56B4E9",  # Sky blue
                "#E69F00",  # Orange
                "#000000",  # Black
            ],
        }

        colors = palettes.get(self.color_scheme, palettes[ColorScheme.PLATINUM])
        while len(colors) < num_colors:
            colors = colors + colors
        return colors[:num_colors]


class CurveVisualizer:
    """
    Comprehensive curve visualization system.

    Provides methods for plotting single curves, comparing multiple curves,
    and generating statistics displays.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses defaults if not provided.
        """
        self.config = config or VisualizationConfig()

    def compute_statistics(self, curve: CurveData) -> CurveStatistics:
        """
        Compute comprehensive statistics for a curve.

        Args:
            curve: The curve to analyze.

        Returns:
            CurveStatistics with computed metrics.
        """
        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)

        # Basic range
        input_min, input_max = float(np.min(inputs)), float(np.max(inputs))
        output_min, output_max = float(np.min(outputs)), float(np.max(outputs))

        # Compute slopes
        if len(inputs) > 1:
            dx = np.diff(inputs)
            dy = np.diff(outputs)
            # Avoid division by zero
            slopes = np.where(dx != 0, dy / dx, 0)
            max_slope = float(np.max(slopes))
            min_slope = float(np.min(slopes))
            average_slope = float(np.mean(slopes))
        else:
            max_slope = min_slope = average_slope = 0.0

        # Monotonicity check
        is_monotonic = bool(np.all(np.diff(outputs) >= -1e-10))

        # Gamma estimation (fit power law)
        gamma = self._estimate_gamma(inputs, outputs)

        # Midpoint value
        midpoint_idx = len(inputs) // 2
        midpoint_value = float(outputs[midpoint_idx])

        # Linearity error (RMS deviation from linear)
        linear_output = inputs * (output_max - output_min) + output_min
        linearity_error = float(np.sqrt(np.mean((outputs - linear_output) ** 2)))

        return CurveStatistics(
            name=curve.name,
            num_points=len(inputs),
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
            gamma=gamma,
            midpoint_value=midpoint_value,
            is_monotonic=is_monotonic,
            max_slope=max_slope,
            min_slope=min_slope,
            average_slope=average_slope,
            linearity_error=linearity_error,
        )

    def _estimate_gamma(self, inputs: np.ndarray, outputs: np.ndarray) -> float:
        """Estimate gamma value from curve data."""
        # Filter valid points for power law fit
        valid_mask = (inputs > 0.01) & (outputs > 0.01) & (inputs < 0.99) & (outputs < 0.99)
        if np.sum(valid_mask) < 2:
            return 1.0

        x_valid = inputs[valid_mask]
        y_valid = outputs[valid_mask]

        # Log-log linear regression to estimate gamma
        try:
            log_x = np.log(x_valid)
            log_y = np.log(y_valid)
            coeffs = np.polyfit(log_x, log_y, 1)
            gamma = float(coeffs[0])
            return max(0.1, min(10.0, gamma))
        except (ValueError, np.linalg.LinAlgError):
            return 1.0

    def compare_curves(
        self, curves: list[CurveData], reference_idx: int = 0
    ) -> CurveComparisonResult:
        """
        Compare multiple curves against a reference.

        Args:
            curves: List of curves to compare.
            reference_idx: Index of the reference curve.

        Returns:
            CurveComparisonResult with comparison metrics.
        """
        if len(curves) < 2:
            raise ValueError("At least 2 curves required for comparison")

        reference = curves[reference_idx]
        ref_inputs = np.array(reference.input_values)
        ref_outputs = np.array(reference.output_values)

        # Interpolate all curves to common input values
        all_outputs = [ref_outputs]
        for i, curve in enumerate(curves):
            if i != reference_idx:
                interp_outputs = np.interp(
                    ref_inputs, curve.input_values, curve.output_values
                )
                all_outputs.append(interp_outputs)

        # Compute differences from reference
        differences = []
        for outputs in all_outputs[1:]:
            diff = outputs - ref_outputs
            differences.append(diff)

        if differences:
            all_diff = np.concatenate(differences)
            max_difference = float(np.max(np.abs(all_diff)))
            average_difference = float(np.mean(np.abs(all_diff)))
            rms_difference = float(np.sqrt(np.mean(all_diff**2)))

            # Correlation with reference
            correlations = []
            for outputs in all_outputs[1:]:
                corr = np.corrcoef(ref_outputs, outputs)[0, 1]
                correlations.append(corr)
            correlation = float(np.mean(correlations))
        else:
            max_difference = average_difference = rms_difference = 0.0
            correlation = 1.0

        # Create difference curve if comparing two curves
        difference_curve = None
        if len(curves) == 2:
            other_idx = 1 if reference_idx == 0 else 0
            other_outputs = np.interp(
                ref_inputs, curves[other_idx].input_values, curves[other_idx].output_values
            )
            diff = other_outputs - ref_outputs
            difference_curve = (list(ref_inputs), list(diff))

        return CurveComparisonResult(
            curve_names=[c.name for c in curves],
            max_difference=max_difference,
            average_difference=average_difference,
            rms_difference=rms_difference,
            correlation=correlation,
            difference_curve=difference_curve,
        )

    def plot_single_curve(
        self,
        curve: CurveData,
        title: Optional[str] = None,
        style: PlotStyle = PlotStyle.LINE,
        color: Optional[str] = None,
        show_stats: Optional[bool] = None,
    ):
        """
        Plot a single curve.

        Args:
            curve: The curve to plot.
            title: Optional title override.
            style: Plot style to use.
            color: Optional color override.
            show_stats: Whether to show statistics (overrides config).

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        color = color or self.config.get_color_palette(1)[0]
        self._plot_curve_on_axis(ax, curve, style, color, curve.name)

        # Reference line
        if self.config.show_reference_line:
            ax.plot(
                [0, 1],
                [0, 1],
                self.config.reference_line_style,
                color="gray",
                alpha=self.config.reference_line_alpha,
                label="Linear Reference",
            )

        # Configure axes
        self._configure_axis(ax, title or f"Curve: {curve.name}")

        # Statistics annotation
        show_stats = show_stats if show_stats is not None else self.config.show_statistics
        if show_stats:
            stats = self.compute_statistics(curve)
            self._add_stats_annotation(ax, [stats])

        if self.config.show_legend:
            ax.legend(loc="lower right", fontsize=self.config.legend_fontsize)

        plt.tight_layout()
        return fig

    def plot_multiple_curves(
        self,
        curves: list[CurveData],
        title: str = "Curve Comparison",
        style: PlotStyle = PlotStyle.LINE,
        colors: Optional[list[str]] = None,
        show_difference: Optional[bool] = None,
        reference_idx: int = 0,
    ):
        """
        Plot multiple curves for comparison.

        Args:
            curves: List of curves to plot.
            title: Plot title.
            style: Plot style to use.
            colors: Optional color list.
            show_difference: Whether to show difference subplot.
            reference_idx: Reference curve index for difference.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        if not curves:
            raise ValueError("No curves provided")

        show_difference = (
            show_difference if show_difference is not None else self.config.show_difference
        )

        colors = colors or self.config.get_color_palette(len(curves))

        if show_difference and len(curves) >= 2:
            fig, (ax_main, ax_diff) = plt.subplots(
                2,
                1,
                figsize=(self.config.figure_width, self.config.figure_height * 1.5),
                dpi=self.config.dpi,
                height_ratios=[3, 1],
            )
        else:
            fig, ax_main = plt.subplots(
                figsize=(self.config.figure_width, self.config.figure_height),
                dpi=self.config.dpi,
            )
            ax_diff = None

        # Plot all curves
        for i, curve in enumerate(curves):
            self._plot_curve_on_axis(ax_main, curve, style, colors[i], curve.name)

        # Reference line
        if self.config.show_reference_line:
            ax_main.plot(
                [0, 1],
                [0, 1],
                self.config.reference_line_style,
                color="gray",
                alpha=self.config.reference_line_alpha,
                label="Linear Reference",
            )

        self._configure_axis(ax_main, title)

        if self.config.show_legend:
            ax_main.legend(loc="lower right", fontsize=self.config.legend_fontsize)

        # Difference plot
        if ax_diff is not None and len(curves) >= 2:
            comparison = self.compare_curves(curves, reference_idx)
            if comparison.difference_curve:
                inputs, diff = comparison.difference_curve
                ax_diff.fill_between(
                    inputs,
                    diff,
                    0,
                    alpha=0.3,
                    color=colors[1] if len(colors) > 1 else colors[0],
                )
                ax_diff.plot(inputs, diff, color=colors[1] if len(colors) > 1 else colors[0])
                ax_diff.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax_diff.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize)
                ax_diff.set_ylabel("Difference", fontsize=self.config.label_fontsize)
                ax_diff.set_title(
                    f"Difference from {curves[reference_idx].name}",
                    fontsize=self.config.title_fontsize - 2,
                )
                ax_diff.grid(True, alpha=self.config.grid_alpha)
                ax_diff.set_facecolor(self.config.background_color)
                ax_diff.set_xlim(0, 1)

        plt.tight_layout()
        return fig

    def plot_with_statistics(
        self,
        curves: list[CurveData],
        title: str = "Curve Analysis",
    ):
        """
        Plot curves with a statistics panel.

        Args:
            curves: Curves to analyze and plot.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(
            figsize=(self.config.figure_width * 1.4, self.config.figure_height),
            dpi=self.config.dpi,
        )
        gs = GridSpec(1, 5, figure=fig)

        # Main plot area
        ax_main = fig.add_subplot(gs[0, :3])
        colors = self.config.get_color_palette(len(curves))

        for i, curve in enumerate(curves):
            self._plot_curve_on_axis(ax_main, curve, PlotStyle.LINE, colors[i], curve.name)

        if self.config.show_reference_line:
            ax_main.plot(
                [0, 1],
                [0, 1],
                self.config.reference_line_style,
                color="gray",
                alpha=self.config.reference_line_alpha,
                label="Linear Reference",
            )

        self._configure_axis(ax_main, title)
        ax_main.legend(loc="lower right", fontsize=self.config.legend_fontsize)

        # Statistics panel
        ax_stats = fig.add_subplot(gs[0, 3:])
        ax_stats.axis("off")

        stats_list = [self.compute_statistics(curve) for curve in curves]
        self._render_stats_table(ax_stats, stats_list, colors)

        plt.tight_layout()
        return fig

    def plot_histogram(
        self,
        curve: CurveData,
        bins: int = 50,
        title: Optional[str] = None,
    ):
        """
        Plot histogram of curve output values.

        Args:
            curve: Curve to analyze.
            bins: Number of histogram bins.
            title: Optional title.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        color = self.config.get_color_palette(1)[0]
        ax.hist(curve.output_values, bins=bins, color=color, alpha=0.7, edgecolor="white")

        ax.set_xlabel("Output Value", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Frequency", fontsize=self.config.label_fontsize)
        ax.set_title(title or f"Output Distribution: {curve.name}", fontsize=self.config.title_fontsize)
        ax.set_facecolor(self.config.background_color)
        fig.patch.set_facecolor(self.config.background_color)
        ax.grid(True, alpha=self.config.grid_alpha)

        plt.tight_layout()
        return fig

    def plot_slope_analysis(
        self,
        curve: CurveData,
        title: Optional[str] = None,
    ):
        """
        Plot curve with slope analysis.

        Args:
            curve: Curve to analyze.
            title: Optional title.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, (ax_curve, ax_slope) = plt.subplots(
            2,
            1,
            figsize=(self.config.figure_width, self.config.figure_height * 1.3),
            dpi=self.config.dpi,
        )

        inputs = np.array(curve.input_values)
        outputs = np.array(curve.output_values)
        color = self.config.get_color_palette(1)[0]

        # Curve plot
        ax_curve.plot(inputs, outputs, color=color, linewidth=self.config.line_width, label=curve.name)
        if self.config.show_reference_line:
            ax_curve.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Linear")
        self._configure_axis(ax_curve, title or f"Curve: {curve.name}")
        ax_curve.legend(loc="lower right")

        # Slope plot
        if len(inputs) > 1:
            mid_points = (inputs[:-1] + inputs[1:]) / 2
            slopes = np.diff(outputs) / np.diff(inputs)
            ax_slope.plot(mid_points, slopes, color=color, linewidth=self.config.line_width)
            ax_slope.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Unity slope")
            ax_slope.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize)
            ax_slope.set_ylabel("Slope (dy/dx)", fontsize=self.config.label_fontsize)
            ax_slope.set_title("Local Slope", fontsize=self.config.title_fontsize - 2)
            ax_slope.grid(True, alpha=self.config.grid_alpha)
            ax_slope.set_facecolor(self.config.background_color)
            ax_slope.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_curve_on_axis(
        self,
        ax,
        curve: CurveData,
        style: PlotStyle,
        color: str,
        label: str,
    ) -> None:
        """Plot a curve on the given axis."""
        inputs = curve.input_values
        outputs = curve.output_values

        if style == PlotStyle.LINE:
            ax.plot(inputs, outputs, color=color, linewidth=self.config.line_width, label=label)
        elif style == PlotStyle.LINE_MARKERS:
            ax.plot(
                inputs,
                outputs,
                "o-",
                color=color,
                linewidth=self.config.line_width,
                markersize=self.config.marker_size,
                label=label,
            )
        elif style == PlotStyle.SCATTER:
            ax.scatter(inputs, outputs, color=color, s=self.config.marker_size**2, label=label)
        elif style == PlotStyle.AREA:
            ax.fill_between(inputs, 0, outputs, color=color, alpha=0.5, label=label)
            ax.plot(inputs, outputs, color=color, linewidth=self.config.line_width)
        elif style == PlotStyle.STEP:
            ax.step(inputs, outputs, where="mid", color=color, linewidth=self.config.line_width, label=label)

    def _configure_axis(self, ax, title: str) -> None:
        """Configure axis with standard settings."""
        ax.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha)

        ax.set_facecolor(self.config.background_color)
        ax.figure.patch.set_facecolor(self.config.background_color)

        x_limits = self.config.x_limits or (0, 1)
        y_limits = self.config.y_limits or (0, 1)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        ax.tick_params(labelsize=self.config.tick_fontsize)

    def _add_stats_annotation(self, ax, stats_list: list[CurveStatistics]) -> None:
        """Add statistics annotation to axis."""
        text_lines = []
        for stats in stats_list:
            text_lines.append(f"{stats.name}:")
            text_lines.append(f"  Gamma: {stats.gamma:.2f}")
            text_lines.append(f"  Midpoint: {stats.midpoint_value:.3f}")
            text_lines.append(f"  Lin. Error: {stats.linearity_error:.4f}")

        text = "\n".join(text_lines)
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=self.config.tick_fontsize,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _render_stats_table(
        self,
        ax,
        stats_list: list[CurveStatistics],
        colors: list[str],
    ) -> None:
        """Render statistics as a table."""
        y_pos = 0.95
        line_height = 0.08

        ax.text(0.5, y_pos, "Statistics", fontsize=self.config.title_fontsize, fontweight="bold", ha="center")
        y_pos -= line_height * 1.5

        for i, stats in enumerate(stats_list):
            color = colors[i] if i < len(colors) else "black"

            ax.text(0.05, y_pos, stats.name, fontsize=self.config.label_fontsize, color=color, fontweight="bold")
            y_pos -= line_height

            stat_items = [
                f"Points: {stats.num_points}",
                f"Gamma: {stats.gamma:.2f}",
                f"Midpoint: {stats.midpoint_value:.3f}",
                f"Monotonic: {'Yes' if stats.is_monotonic else 'No'}",
                f"Lin. Error: {stats.linearity_error:.4f}",
            ]

            for item in stat_items:
                ax.text(0.1, y_pos, item, fontsize=self.config.tick_fontsize, color="black")
                y_pos -= line_height * 0.7

            y_pos -= line_height * 0.5

    def save_figure(
        self,
        fig,
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Path:
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure.
            path: Output path.
            format: Output format (png, svg, pdf). Inferred from path if not provided.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        format = format or path.suffix.lstrip(".")

        fig.savefig(
            path,
            format=format,
            dpi=self.config.dpi,
            bbox_inches="tight",
            facecolor=self.config.background_color,
        )

        return path

    def figure_to_bytes(self, fig, format: str = "png") -> bytes:
        """
        Convert figure to bytes.

        Args:
            fig: Matplotlib figure.
            format: Output format.

        Returns:
            Image bytes.
        """
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format=format,
            dpi=self.config.dpi,
            bbox_inches="tight",
            facecolor=self.config.background_color,
        )
        buffer.seek(0)
        return buffer.getvalue()
