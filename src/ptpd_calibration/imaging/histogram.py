"""
Histogram analysis for image evaluation.

Provides comprehensive histogram analysis including:
- Tonal distribution visualization
- Zone-based analysis (Ansel Adams zones)
- Clipping detection
- Dynamic range assessment
- Contrast evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class HistogramScale(str, Enum):
    """Scale for histogram display."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


@dataclass
class HistogramStats:
    """Statistics computed from histogram data."""

    # Basic statistics
    mean: float
    median: float
    std_dev: float
    min_value: int
    max_value: int

    # Distribution
    mode: int  # Most common value
    percentile_5: float
    percentile_95: float

    # Range metrics
    dynamic_range: float  # In stops
    contrast: float  # Std dev normalized
    brightness: float  # Mean normalized to 0-1

    # Clipping detection
    shadow_clipping_percent: float  # % of pixels at 0-5
    highlight_clipping_percent: float  # % of pixels at 250-255

    # Zone distribution (0-10 mapping to Ansel Adams zones)
    zone_distribution: dict[int, float] = field(default_factory=dict)

    # Recommendations
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std_dev": round(self.std_dev, 2),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mode": self.mode,
            "percentile_5": round(self.percentile_5, 2),
            "percentile_95": round(self.percentile_95, 2),
            "dynamic_range_stops": round(self.dynamic_range, 2),
            "contrast": round(self.contrast, 3),
            "brightness": round(self.brightness, 3),
            "shadow_clipping": f"{self.shadow_clipping_percent:.1f}%",
            "highlight_clipping": f"{self.highlight_clipping_percent:.1f}%",
            "zone_distribution": {
                f"Zone {z}": f"{pct*100:.1f}%"
                for z, pct in self.zone_distribution.items()
            },
            "notes": self.notes,
        }


@dataclass
class HistogramResult:
    """Complete histogram analysis result."""

    # Raw histogram data (256 bins)
    histogram: np.ndarray

    # Statistics
    stats: HistogramStats

    # Original image info
    image_size: tuple[int, int]
    image_mode: str
    total_pixels: int

    # For RGB images, per-channel histograms
    red_histogram: np.ndarray | None = None
    green_histogram: np.ndarray | None = None
    blue_histogram: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
            "image_mode": self.image_mode,
            "total_pixels": self.total_pixels,
            "has_rgb_channels": self.red_histogram is not None,
            "statistics": self.stats.to_dict(),
        }


class HistogramAnalyzer:
    """Analyze image histograms for tone distribution and quality assessment.

    Provides detailed histogram analysis including zone-based distribution,
    dynamic range assessment, and clipping detection.
    """

    # Zone boundaries (0-255 mapped to zones 0-10)
    ZONE_BOUNDARIES = [
        (0, 12),     # Zone 0: Pure black
        (13, 38),    # Zone I: Near black
        (39, 63),    # Zone II: Dark with texture
        (64, 89),    # Zone III: Average dark
        (90, 114),   # Zone IV: Dark foliage
        (115, 140),  # Zone V: Middle gray (18%)
        (141, 165),  # Zone VI: Light skin
        (166, 191),  # Zone VII: Very light
        (192, 216),  # Zone VIII: Whites with texture
        (217, 242),  # Zone IX: Near white
        (243, 255),  # Zone X: Pure white
    ]

    def analyze(
        self,
        image: str | Path | Image.Image | np.ndarray,
        include_rgb: bool = True,
    ) -> HistogramResult:
        """Analyze image histogram.

        Args:
            image: Image to analyze (path, PIL Image, or numpy array)
            include_rgb: Include per-channel histograms for color images

        Returns:
            HistogramResult with complete analysis
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img = Image.fromarray(image.astype(np.uint8), mode="L")
            else:
                img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        image_mode = img.mode
        image_size = img.size
        total_pixels = image_size[0] * image_size[1]

        # Get grayscale version for main analysis
        if img.mode == "L":
            gray = img
        elif img.mode in ("LA", "RGBA"):
            gray = img.convert("L")
        else:
            gray = img.convert("L")

        # Compute main histogram
        gray_arr = np.array(gray)
        histogram = np.histogram(gray_arr, bins=256, range=(0, 255))[0]

        # Compute RGB histograms if applicable
        red_hist = None
        green_hist = None
        blue_hist = None

        if include_rgb and img.mode in ("RGB", "RGBA"):
            rgb = img.convert("RGB") if img.mode == "RGBA" else img
            rgb_arr = np.array(rgb)
            red_hist = np.histogram(rgb_arr[:, :, 0], bins=256, range=(0, 255))[0]
            green_hist = np.histogram(rgb_arr[:, :, 1], bins=256, range=(0, 255))[0]
            blue_hist = np.histogram(rgb_arr[:, :, 2], bins=256, range=(0, 255))[0]

        # Compute statistics
        stats = self._compute_stats(gray_arr, histogram, total_pixels)

        return HistogramResult(
            histogram=histogram,
            stats=stats,
            image_size=image_size,
            image_mode=image_mode,
            total_pixels=total_pixels,
            red_histogram=red_hist,
            green_histogram=green_hist,
            blue_histogram=blue_hist,
        )

    def compare_histograms(
        self,
        image1: str | Path | Image.Image | np.ndarray,
        image2: str | Path | Image.Image | np.ndarray,
    ) -> dict:
        """Compare histograms of two images.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Dictionary with comparison metrics
        """
        result1 = self.analyze(image1, include_rgb=False)
        result2 = self.analyze(image2, include_rgb=False)

        # Normalize histograms for comparison
        h1 = result1.histogram.astype(float) / result1.total_pixels
        h2 = result2.histogram.astype(float) / result2.total_pixels

        # Compute similarity metrics
        # Histogram intersection (higher = more similar, max 1.0)
        intersection = np.sum(np.minimum(h1, h2))

        # Chi-squared distance (lower = more similar)
        chi_squared = np.sum(
            np.where(h1 + h2 > 0, (h1 - h2) ** 2 / (h1 + h2 + 1e-10), 0)
        )

        # Bhattacharyya coefficient (higher = more similar, max 1.0)
        bhattacharyya = np.sum(np.sqrt(h1 * h2))

        # Mean shift
        mean_shift = result2.stats.mean - result1.stats.mean

        # Contrast change
        contrast_change = result2.stats.std_dev - result1.stats.std_dev

        return {
            "similarity": {
                "histogram_intersection": round(intersection, 4),
                "chi_squared_distance": round(chi_squared, 4),
                "bhattacharyya_coefficient": round(bhattacharyya, 4),
            },
            "changes": {
                "mean_shift": round(mean_shift, 2),
                "brightness_change": round(
                    result2.stats.brightness - result1.stats.brightness, 3
                ),
                "contrast_change": round(contrast_change, 2),
                "dynamic_range_change": round(
                    result2.stats.dynamic_range - result1.stats.dynamic_range, 2
                ),
            },
            "image1_stats": result1.stats.to_dict(),
            "image2_stats": result2.stats.to_dict(),
        }

    def _compute_stats(
        self,
        arr: np.ndarray,
        histogram: np.ndarray,
        total_pixels: int,
    ) -> HistogramStats:
        """Compute comprehensive statistics from image array and histogram.

        Args:
            arr: Grayscale image array
            histogram: Computed histogram
            total_pixels: Total number of pixels

        Returns:
            HistogramStats with all computed metrics
        """
        flat = arr.flatten()

        # Basic statistics
        mean_val = float(np.mean(flat))
        median_val = float(np.median(flat))
        std_val = float(np.std(flat))
        min_val = int(np.min(flat))
        max_val = int(np.max(flat))

        # Mode (most common value)
        mode_val = int(np.argmax(histogram))

        # Percentiles
        p5 = float(np.percentile(flat, 5))
        p95 = float(np.percentile(flat, 95))

        # Dynamic range in stops
        # Avoid division by zero
        if min_val > 0:
            dynamic_range = np.log2(max_val / min_val) if max_val > min_val else 0
        else:
            dynamic_range = np.log2(max_val + 1) if max_val > 0 else 0

        # Normalized metrics
        contrast = std_val / 128.0  # Normalize to ~0-1 range
        brightness = mean_val / 255.0

        # Clipping detection
        shadow_pixels = np.sum(histogram[:6])  # 0-5
        highlight_pixels = np.sum(histogram[250:])  # 250-255
        shadow_clip = (shadow_pixels / total_pixels) * 100
        highlight_clip = (highlight_pixels / total_pixels) * 100

        # Zone distribution
        zone_dist = {}
        for zone_num, (low, high) in enumerate(self.ZONE_BOUNDARIES):
            zone_pixels = np.sum(histogram[low : high + 1])
            zone_dist[zone_num] = zone_pixels / total_pixels

        # Generate notes/recommendations
        notes = []

        if shadow_clip > 5:
            notes.append(
                f"Shadow clipping detected ({shadow_clip:.1f}%). "
                "Consider reducing negative density."
            )
        if highlight_clip > 5:
            notes.append(
                f"Highlight clipping detected ({highlight_clip:.1f}%). "
                "Consider increasing exposure or reducing contrast."
            )
        if dynamic_range < 4:
            notes.append(
                "Low dynamic range. Image may benefit from contrast enhancement."
            )
        if dynamic_range > 7:
            notes.append(
                "High dynamic range. Consider N-1 or N-2 development for printing."
            )
        if brightness < 0.3:
            notes.append("Image is quite dark. Consider exposure adjustment.")
        if brightness > 0.7:
            notes.append("Image is quite bright. Consider exposure adjustment.")
        if contrast < 0.15:
            notes.append("Low contrast. May need contrast boost in printing.")
        if contrast > 0.35:
            notes.append("High contrast. Consider reduced development time.")

        # Zone-specific notes
        if zone_dist.get(5, 0) < 0.05:
            notes.append("Limited midtone content (Zone V).")
        if zone_dist.get(0, 0) + zone_dist.get(1, 0) > 0.2:
            notes.append("Heavy shadow content. Watch for blocked shadows.")
        if zone_dist.get(9, 0) + zone_dist.get(10, 0) > 0.2:
            notes.append("Heavy highlight content. Watch for blown highlights.")

        return HistogramStats(
            mean=mean_val,
            median=median_val,
            std_dev=std_val,
            min_value=min_val,
            max_value=max_val,
            mode=mode_val,
            percentile_5=p5,
            percentile_95=p95,
            dynamic_range=dynamic_range,
            contrast=contrast,
            brightness=brightness,
            shadow_clipping_percent=shadow_clip,
            highlight_clipping_percent=highlight_clip,
            zone_distribution=zone_dist,
            notes=notes,
        )

    def create_histogram_plot(
        self,
        result: HistogramResult,
        scale: HistogramScale = HistogramScale.LINEAR,
        show_zones: bool = True,
        show_rgb: bool = True,
    ):
        """Create a matplotlib figure for the histogram.

        Args:
            result: HistogramResult to visualize
            scale: Linear or logarithmic scale
            show_zones: Show zone boundaries
            show_rgb: Show RGB channels if available

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

        # Main histogram
        ax1 = axes[0]
        x = np.arange(256)

        # Plot luminance histogram
        if scale == HistogramScale.LOGARITHMIC:
            hist_data = np.log1p(result.histogram)
            ylabel = "Frequency (log)"
        else:
            hist_data = result.histogram
            ylabel = "Frequency"

        ax1.fill_between(x, hist_data, alpha=0.7, color="#4A4A4A", label="Luminance")
        ax1.plot(x, hist_data, color="#2A2A2A", linewidth=0.5)

        # Plot RGB if available
        if show_rgb and result.red_histogram is not None:
            if scale == HistogramScale.LOGARITHMIC:
                r_data = np.log1p(result.red_histogram)
                g_data = np.log1p(result.green_histogram)
                b_data = np.log1p(result.blue_histogram)
            else:
                r_data = result.red_histogram
                g_data = result.green_histogram
                b_data = result.blue_histogram

            ax1.plot(x, r_data, color="red", alpha=0.5, linewidth=1, label="Red")
            ax1.plot(x, g_data, color="green", alpha=0.5, linewidth=1, label="Green")
            ax1.plot(x, b_data, color="blue", alpha=0.5, linewidth=1, label="Blue")

        # Add zone boundaries
        if show_zones:
            zone_colors = [
                "#000000",  # Zone 0
                "#1A1A1A",  # Zone I
                "#333333",  # Zone II
                "#4D4D4D",  # Zone III
                "#666666",  # Zone IV
                "#808080",  # Zone V
                "#999999",  # Zone VI
                "#B3B3B3",  # Zone VII
                "#CCCCCC",  # Zone VIII
                "#E6E6E6",  # Zone IX
                "#FFFFFF",  # Zone X
            ]

            for zone_num, (low, high) in enumerate(self.ZONE_BOUNDARIES):
                ax1.axvspan(low, high, alpha=0.1, color=zone_colors[zone_num])

        ax1.set_xlabel("Pixel Value")
        ax1.set_ylabel(ylabel)
        ax1.set_title("Image Histogram")
        ax1.set_xlim(0, 255)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Zone distribution bar
        ax2 = axes[1]
        zone_pcts = [result.stats.zone_distribution.get(i, 0) * 100 for i in range(11)]
        zone_labels = [f"Z{i}" for i in range(11)]

        bars = ax2.bar(zone_labels, zone_pcts, color=[
            "#1A1A1A", "#333333", "#4D4D4D", "#666666", "#808080",
            "#999999", "#B3B3B3", "#CCCCCC", "#E6E6E6", "#F0F0F0", "#FFFFFF"
        ], edgecolor="black", linewidth=0.5)

        ax2.set_xlabel("Ansel Adams Zone")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Zone Distribution")
        ax2.set_ylim(0, max(zone_pcts) * 1.2 if max(zone_pcts) > 0 else 10)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add percentage labels on bars
        for bar, pct in zip(bars, zone_pcts, strict=False):
            if pct > 1:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        return fig

    @staticmethod
    def get_zone_descriptions() -> dict[int, str]:
        """Get descriptions for each zone.

        Returns:
            Dictionary mapping zone number to description
        """
        return {
            0: "Pure black, no texture",
            1: "Near black, slight tonality",
            2: "Very dark, first hint of texture",
            3: "Dark with full texture",
            4: "Dark foliage, open shadow",
            5: "Middle gray (18% gray card)",
            6: "Average skin, light stone",
            7: "Very light skin, light gray",
            8: "Whites with texture",
            9: "White without texture",
            10: "Pure white, paper base",
        }
