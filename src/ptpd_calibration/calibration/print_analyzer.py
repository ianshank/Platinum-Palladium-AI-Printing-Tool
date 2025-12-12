"""
Analyze scanned prints to provide curve adjustment recommendations.

This module processes scans of pt/pd prints to measure tonal characteristics
and generate recommendations for curve refinement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class PrintAnalysis:
    """Results from analyzing a scanned print."""

    # Identifiers
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Measured densities (converted from reflectance)
    highlight_density: float = 0.0      # Average density in highlight zones
    midtone_density: float = 0.0        # Average density in midtone zones
    shadow_density: float = 0.0         # Average density (Dmax)

    # Derived metrics
    tonal_range: float = 0.0           # Shadow - highlight density
    midtone_separation: float = 0.0    # Measure of midtone differentiation
    contrast_index: float = 0.0        # Overall contrast metric

    # Recommendations (as percentage adjustments)
    recommended_highlight_adj: float = 0.0
    recommended_midtone_adj: float = 0.0
    recommended_shadow_adj: float = 0.0

    # Qualitative assessment
    notes: List[str] = field(default_factory=list)
    zone_histogram: Optional[Dict[str, float]] = None

    # Reference to source
    source_image_path: Optional[str] = None
    paper_type: Optional[str] = None
    chemistry: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "PRINT ANALYSIS SUMMARY",
            "=" * 50,
            "",
            "MEASURED DENSITIES:",
            f"  Highlights (paper+slight tone): {self.highlight_density:.2f}",
            f"  Midtones (Zone V equivalent):   {self.midtone_density:.2f}",
            f"  Shadows (Dmax):                 {self.shadow_density:.2f}",
            "",
            f"  Tonal Range:        {self.tonal_range:.2f}",
            f"  Midtone Separation: {self.midtone_separation:.2f}",
            "",
            "RECOMMENDED ADJUSTMENTS:",
            f"  Highlights: {self.recommended_highlight_adj:+.1%}",
            f"  Midtones:   {self.recommended_midtone_adj:+.1%}",
            f"  Shadows:    {self.recommended_shadow_adj:+.1%}",
            "",
            "OBSERVATIONS:"
        ]
        for note in self.notes:
            lines.append(f"  - {note}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat(),
            'highlight_density': self.highlight_density,
            'midtone_density': self.midtone_density,
            'shadow_density': self.shadow_density,
            'tonal_range': self.tonal_range,
            'midtone_separation': self.midtone_separation,
            'contrast_index': self.contrast_index,
            'recommended_highlight_adj': self.recommended_highlight_adj,
            'recommended_midtone_adj': self.recommended_midtone_adj,
            'recommended_shadow_adj': self.recommended_shadow_adj,
            'notes': self.notes,
            'zone_histogram': self.zone_histogram,
            'source_image_path': self.source_image_path,
            'paper_type': self.paper_type,
            'chemistry': self.chemistry,
        }


@dataclass
class TargetDensities:
    """Target density values for well-calibrated prints."""

    highlight: float = 0.12   # Paper white + minimal tone
    midtone: float = 0.65     # Good separation, Zone V equivalent
    shadow: float = 1.55      # Typical pt/pd Dmax
    tonal_range: float = 1.40  # Minimum acceptable range

    @classmethod
    def for_platinum_palladium(cls) -> 'TargetDensities':
        """Target densities for pt/pd prints."""
        return cls(
            highlight=0.12,
            midtone=0.65,
            shadow=1.55,
            tonal_range=1.40
        )

    @classmethod
    def for_cyanotype(cls) -> 'TargetDensities':
        """Target densities for cyanotype prints."""
        return cls(
            highlight=0.15,
            midtone=0.80,
            shadow=1.90,
            tonal_range=1.70
        )

    @classmethod
    def for_silver_gelatin(cls) -> 'TargetDensities':
        """Target densities for silver gelatin prints."""
        return cls(
            highlight=0.08,
            midtone=0.55,
            shadow=2.10,
            tonal_range=1.95
        )


class PrintAnalyzer:
    """
    Analyze scanned prints for curve refinement feedback.

    This analyzer compares measured densities against target values
    for a well-calibrated platinum/palladium print and generates
    adjustment recommendations.
    """

    def __init__(self, targets: Optional[TargetDensities] = None):
        """
        Initialize analyzer with optional custom targets.

        Args:
            targets: Custom target densities, defaults to pt/pd targets
        """
        self.targets = targets or TargetDensities.for_platinum_palladium()

    def analyze_print_scan(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        exclude_borders: bool = True
    ) -> PrintAnalysis:
        """
        Analyze a scanned print to determine curve adjustments.

        Args:
            image: RGB or grayscale image array (0-255 or 0-1 range)
            mask: Optional binary mask for print area
            exclude_borders: Auto-detect and exclude brushed borders

        Returns:
            PrintAnalysis with measurements and recommendations
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Use luminance weighting for better perceptual accuracy
            gray = (
                0.299 * image[:, :, 0].astype(float) +
                0.587 * image[:, :, 1].astype(float) +
                0.114 * image[:, :, 2].astype(float)
            )
        else:
            gray = image.astype(float)

        # Normalize to 0-1 range
        if gray.max() > 1:
            gray = gray / 255.0

        # Auto-detect print region if requested
        if exclude_borders and mask is None:
            mask = self._detect_print_region(gray)

        # Apply mask
        if mask is not None:
            pixels = gray[mask > 0]
        else:
            pixels = gray.flatten()

        # Calculate zone-based density measurements
        densities = self._measure_zone_densities(pixels)

        # Generate recommendations
        recommendations = self._generate_recommendations(densities)

        # Build zone histogram
        zone_hist = self._build_zone_histogram(pixels)

        return PrintAnalysis(
            highlight_density=densities['highlight'],
            midtone_density=densities['midtone'],
            shadow_density=densities['shadow'],
            tonal_range=densities['shadow'] - densities['highlight'],
            midtone_separation=densities['midtone_separation'],
            contrast_index=densities['contrast_index'],
            recommended_highlight_adj=recommendations['highlight'],
            recommended_midtone_adj=recommendations['midtone'],
            recommended_shadow_adj=recommendations['shadow'],
            notes=recommendations['notes'],
            zone_histogram=zone_hist
        )

    def _reflectance_to_density(self, reflectance: np.ndarray) -> np.ndarray:
        """Convert reflectance values to density (D = -log10(R))."""
        # Clamp to avoid log(0)
        r = np.clip(reflectance, 0.01, 1.0)
        return -np.log10(r)

    def _measure_zone_densities(self, pixels: np.ndarray) -> Dict[str, float]:
        """
        Measure densities in different tonal zones.

        Uses percentile-based zone identification rather than fixed thresholds
        to handle varying exposure levels.
        """
        # Sort for percentile calculations
        sorted_pixels = np.sort(pixels)

        # Identify zones by percentile
        # Highlights: brightest 15% (but not pure paper white)
        highlight_start = int(len(sorted_pixels) * 0.80)
        highlight_end = int(len(sorted_pixels) * 0.95)
        highlights = sorted_pixels[highlight_start:highlight_end]

        # Shadows: darkest 15% (excluding possible clipping)
        shadow_start = int(len(sorted_pixels) * 0.02)
        shadow_end = int(len(sorted_pixels) * 0.15)
        shadows = sorted_pixels[shadow_start:shadow_end]

        # Midtones: middle 40%
        mid_start = int(len(sorted_pixels) * 0.30)
        mid_end = int(len(sorted_pixels) * 0.70)
        midtones = sorted_pixels[mid_start:mid_end]

        # Calculate mean densities
        highlight_density = float(
            np.mean(self._reflectance_to_density(highlights))
        ) if len(highlights) > 0 else 0.0

        shadow_density = float(
            np.mean(self._reflectance_to_density(shadows))
        ) if len(shadows) > 0 else 0.0

        midtone_density = float(
            np.mean(self._reflectance_to_density(midtones))
        ) if len(midtones) > 0 else 0.0

        # Calculate midtone separation (how much differentiation in the midrange)
        midtone_std = float(
            np.std(self._reflectance_to_density(midtones))
        ) if len(midtones) > 0 else 0.0

        # Overall contrast index
        tonal_range = shadow_density - highlight_density

        return {
            'highlight': highlight_density,
            'midtone': midtone_density,
            'shadow': shadow_density,
            'midtone_separation': midtone_std,
            'contrast_index': tonal_range
        }

    def _generate_recommendations(self, densities: Dict[str, float]) -> Dict:
        """
        Generate adjustment recommendations based on measured vs target densities.
        """
        notes: List[str] = []

        # Highlight analysis
        h_diff = densities['highlight'] - self.targets.highlight
        if abs(h_diff) < 0.06:
            h_adj = 0.0
            notes.append("Highlights look good - proper paper tone")
        elif h_diff > 0:
            # Highlights too dark
            h_adj = -min(0.06, h_diff * 0.5)
            notes.append(
                f"Highlights slightly dark (D={densities['highlight']:.2f}) - "
                "reduce highlight ink"
            )
        else:
            # Highlights too light (lost detail)
            h_adj = min(0.04, abs(h_diff) * 0.4)
            notes.append(
                "Highlights may lack subtle tone - slight increase may help"
            )

        # Midtone analysis - KEY ZONE for pt/pd
        m_diff = densities['midtone'] - self.targets.midtone

        # Also check midtone separation
        if densities['midtone_separation'] < 0.15:
            # Muted/flat midtones
            m_adj = min(0.12, max(0.06, abs(m_diff) * 0.8))
            notes.append(
                f"MIDTONES MUTED - separation={densities['midtone_separation']:.2f}"
            )
            notes.append("  -> Significant midtone boost recommended (+8-12%)")
        elif abs(m_diff) < 0.10:
            m_adj = 0.0
            notes.append("Midtones well-balanced with good separation")
        elif m_diff > 0:
            # Midtones too dark/muddy
            m_adj = -min(0.08, m_diff * 0.6)
            notes.append(
                f"Midtones slightly muddy (D={densities['midtone']:.2f}) - "
                "reduce ink"
            )
        else:
            # Midtones too light/flat
            m_adj = min(0.10, abs(m_diff) * 0.7)
            notes.append(
                f"Midtones light (D={densities['midtone']:.2f}) - "
                "boost midtone ink"
            )

        # Shadow analysis (Dmax)
        s_diff = densities['shadow'] - self.targets.shadow
        if abs(s_diff) < 0.12:
            s_adj = 0.0
            notes.append(
                f"Shadow density (Dmax={densities['shadow']:.2f}) is good for pt/pd"
            )
        elif s_diff > 0:
            # Shadows blocked up
            s_adj = -min(0.06, s_diff * 0.4)
            notes.append("Shadows may be blocking - reduce shadow ink slightly")
        else:
            # Dmax not reached
            s_adj = min(0.08, abs(s_diff) * 0.5)
            notes.append(
                f"Dmax low ({densities['shadow']:.2f}) - "
                "try longer exposure or more shadow ink"
            )

        # Overall tonal range check
        if densities['contrast_index'] < self.targets.tonal_range:
            notes.append(
                f"Overall tonal range ({densities['contrast_index']:.2f}) below target"
            )
            notes.append("  -> Consider both chemistry and curve adjustments")

        return {
            'highlight': h_adj,
            'midtone': m_adj,
            'shadow': s_adj,
            'notes': notes
        }

    def _build_zone_histogram(self, pixels: np.ndarray) -> Dict[str, float]:
        """Build Ansel Adams zone distribution histogram."""
        densities = self._reflectance_to_density(pixels)

        # Zone boundaries (approximate density values)
        zone_bounds = [
            (0.0, 0.10),   # Zone 0 - pure black
            (0.10, 0.25),  # Zone I
            (0.25, 0.40),  # Zone II
            (0.40, 0.55),  # Zone III
            (0.55, 0.70),  # Zone IV
            (0.70, 0.85),  # Zone V (middle gray)
            (0.85, 1.00),  # Zone VI
            (1.00, 1.20),  # Zone VII
            (1.20, 1.40),  # Zone VIII
            (1.40, 1.60),  # Zone IX
            (1.60, 2.00),  # Zone X - paper white
        ]

        histogram: Dict[str, float] = {}
        total = len(densities)

        for i, (low, high) in enumerate(zone_bounds):
            zone_name = f"Zone_{i}"
            count = np.sum((densities >= low) & (densities < high))
            histogram[zone_name] = float(count / total) if total > 0 else 0.0

        return histogram

    def _detect_print_region(self, gray: np.ndarray) -> np.ndarray:
        """
        Auto-detect the print area, excluding brushed borders and paper margin.

        Uses edge detection and morphological operations to find the
        main rectangular print region.
        """
        if not HAS_SCIPY:
            # Fallback: simple threshold-based detection
            threshold = np.percentile(gray, 92)
            mask = gray < threshold

            # Simple erosion approximation
            kernel_size = max(gray.shape) // 50
            if kernel_size > 1:
                # Basic erosion without scipy
                from_edge = kernel_size
                mask[:from_edge, :] = False
                mask[-from_edge:, :] = False
                mask[:, :from_edge] = False
                mask[:, -from_edge:] = False

            return mask.astype(np.uint8)

        # Full detection with scipy
        # 1. Find paper white threshold
        paper_white = np.percentile(gray, 95)

        # 2. Create binary mask of non-paper regions
        binary = gray < (paper_white * 0.95)

        # 3. Find edges of the brushed border
        # The border is typically irregular, so we look for the consistent inner region

        # Erode to get past the brushed edge
        erode_size = max(gray.shape) // 40
        structure = np.ones((erode_size, erode_size))
        eroded = ndimage.binary_erosion(binary, structure=structure, iterations=2)

        # Dilate back slightly to recover image area
        dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=1)

        # Fill holes
        filled = ndimage.binary_fill_holes(dilated)

        return filled.astype(np.uint8)

    def analyze_from_file(self, filepath: str) -> PrintAnalysis:
        """
        Convenience method to analyze directly from image file.

        Args:
            filepath: Path to image file (JPEG, TIFF, PNG)

        Returns:
            PrintAnalysis with measurements and recommendations

        Raises:
            ImportError: If PIL is not available
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for file loading")

        img = Image.open(filepath)
        img_array = np.array(img)

        analysis = self.analyze_print_scan(img_array)
        analysis.source_image_path = filepath

        return analysis


@dataclass
class CalibrationIteration:
    """Single iteration in a calibration session."""

    iteration_number: int
    exposure_time: str
    curve_version: str
    adjustments_applied: Dict[str, float] = field(default_factory=dict)
    analysis: Optional[PrintAnalysis] = None
    recommended_next: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class CalibrationSession:
    """
    Track a calibration session with multiple test prints.

    This class maintains state across multiple refinement iterations,
    tracking what adjustments have been made and their effects.
    """

    def __init__(
        self,
        paper_type: str,
        chemistry: str = "",
        notes: str = ""
    ):
        """
        Initialize a new calibration session.

        Args:
            paper_type: Paper being calibrated
            chemistry: Chemistry description (e.g., "6Pd:2Pt")
            notes: Additional session notes
        """
        self.id = uuid4()
        self.created_at = datetime.now()
        self.paper_type = paper_type
        self.chemistry = chemistry
        self.notes = notes
        self.iterations: List[CalibrationIteration] = []
        self.analyzer = PrintAnalyzer()

    def add_iteration(
        self,
        scan: np.ndarray,
        exposure_time: str,
        curve_version: str,
        adjustments_applied: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> PrintAnalysis:
        """
        Add a test print to the calibration session.

        Args:
            scan: Scanned print image array
            exposure_time: Exposure time used
            curve_version: Name/version of curve used
            adjustments_applied: Adjustments that were applied for this print
            notes: Notes about this iteration

        Returns:
            PrintAnalysis with measurements and recommendations
        """
        analysis = self.analyzer.analyze_print_scan(scan)
        analysis.paper_type = self.paper_type
        analysis.chemistry = self.chemistry

        iteration = CalibrationIteration(
            iteration_number=len(self.iterations) + 1,
            exposure_time=exposure_time,
            curve_version=curve_version,
            adjustments_applied=adjustments_applied or {},
            analysis=analysis,
            recommended_next={
                'highlight': analysis.recommended_highlight_adj,
                'midtone': analysis.recommended_midtone_adj,
                'shadow': analysis.recommended_shadow_adj
            },
            notes=notes
        )

        self.iterations.append(iteration)
        return analysis

    def add_iteration_from_file(
        self,
        filepath: str,
        exposure_time: str,
        curve_version: str,
        adjustments_applied: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> PrintAnalysis:
        """
        Add iteration from image file.

        Args:
            filepath: Path to scanned print image
            exposure_time: Exposure time used
            curve_version: Name/version of curve used
            adjustments_applied: Adjustments applied for this print
            notes: Notes about this iteration

        Returns:
            PrintAnalysis with measurements and recommendations
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for file loading")

        img = Image.open(filepath)
        scan = np.array(img)

        analysis = self.add_iteration(
            scan, exposure_time, curve_version, adjustments_applied, notes
        )
        analysis.source_image_path = filepath

        return analysis

    def get_cumulative_adjustment(self) -> Dict[str, float]:
        """Calculate total adjustments across all iterations."""
        cumulative = {'highlight': 0.0, 'midtone': 0.0, 'shadow': 0.0}

        for iteration in self.iterations:
            for zone in cumulative:
                cumulative[zone] += iteration.adjustments_applied.get(zone, 0.0)

        return cumulative

    def get_latest_recommendations(self) -> Optional[Dict[str, float]]:
        """Get recommendations from the most recent iteration."""
        if not self.iterations:
            return None
        return self.iterations[-1].recommended_next

    def summary(self) -> str:
        """Generate session summary."""
        lines = [
            "=" * 60,
            "CALIBRATION SESSION SUMMARY",
            "=" * 60,
            f"Session ID: {self.id}",
            f"Started: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Paper: {self.paper_type}",
            f"Chemistry: {self.chemistry}",
            f"Iterations: {len(self.iterations)}",
            ""
        ]

        for iteration in self.iterations:
            lines.append(f"--- Iteration {iteration.iteration_number} ---")
            lines.append(f"Exposure: {iteration.exposure_time}")
            lines.append(f"Curve: {iteration.curve_version}")

            adj = iteration.adjustments_applied
            if adj:
                lines.append(
                    f"Applied: H={adj.get('highlight', 0):+.1%}, "
                    f"M={adj.get('midtone', 0):+.1%}, "
                    f"S={adj.get('shadow', 0):+.1%}"
                )

            if iteration.analysis:
                analysis = iteration.analysis
                lines.append(
                    f"Measured: H={analysis.highlight_density:.2f}, "
                    f"M={analysis.midtone_density:.2f}, "
                    f"S={analysis.shadow_density:.2f}"
                )

            if iteration.notes:
                lines.append(f"Notes: {iteration.notes}")

            lines.append("")

        cumulative = self.get_cumulative_adjustment()
        lines.append("TOTAL ADJUSTMENTS:")
        lines.append(f"  Highlights: {cumulative['highlight']:+.1%}")
        lines.append(f"  Midtones:   {cumulative['midtone']:+.1%}")
        lines.append(f"  Shadows:    {cumulative['shadow']:+.1%}")

        latest = self.get_latest_recommendations()
        if latest:
            lines.append("")
            lines.append("NEXT RECOMMENDED:")
            lines.append(f"  Highlights: {latest['highlight']:+.1%}")
            lines.append(f"  Midtones:   {latest['midtone']:+.1%}")
            lines.append(f"  Shadows:    {latest['shadow']:+.1%}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'paper_type': self.paper_type,
            'chemistry': self.chemistry,
            'notes': self.notes,
            'iterations': [
                {
                    'iteration_number': it.iteration_number,
                    'exposure_time': it.exposure_time,
                    'curve_version': it.curve_version,
                    'adjustments_applied': it.adjustments_applied,
                    'analysis': it.analysis.to_dict() if it.analysis else None,
                    'recommended_next': it.recommended_next,
                    'notes': it.notes
                }
                for it in self.iterations
            ],
            'cumulative_adjustment': self.get_cumulative_adjustment()
        }
