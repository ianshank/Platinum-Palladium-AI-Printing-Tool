"""
Zone System mapping for Ansel Adams-style visualization.

Maps between Zone System values (0-X) and density measurements,
enabling photographers to translate visualization to Pt/Pd printing.

Zone System Reference:
- Zone 0: Pure black (no detail)
- Zone I: Near black with slight tonality
- Zone II-III: Dark shadows with texture
- Zone IV: Open shadow, dark foliage
- Zone V: Middle gray (18% gray card)
- Zone VI: Average Caucasian skin, light foliage
- Zone VII: Very light skin, bright objects
- Zone VIII: Whites with texture
- Zone IX: Near white
- Zone X: Pure white (paper base)

Each zone represents 1 stop (0.3 density units).
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from PIL import Image


class Zone(IntEnum):
    """Ansel Adams Zone System zones (0-10)."""

    ZONE_0 = 0  # Pure black
    ZONE_I = 1  # Near black
    ZONE_II = 2  # Dark shadows
    ZONE_III = 3  # Dark shadows with texture
    ZONE_IV = 4  # Open shadow
    ZONE_V = 5  # Middle gray
    ZONE_VI = 6  # Light skin
    ZONE_VII = 7  # Very light
    ZONE_VIII = 8  # Whites with texture
    ZONE_IX = 9  # Near white
    ZONE_X = 10  # Pure white


# Zone descriptions
ZONE_DESCRIPTIONS = {
    Zone.ZONE_0: "Pure black, no texture or detail",
    Zone.ZONE_I: "Near black, slight tonality but no texture",
    Zone.ZONE_II: "Very dark, first hint of texture",
    Zone.ZONE_III: "Dark with full texture, average dark materials",
    Zone.ZONE_IV: "Dark foliage, dark stone, open shadow",
    Zone.ZONE_V: "Middle gray, 18% gray card, dark skin, gray stone",
    Zone.ZONE_VI: "Average Caucasian skin, light stone, shadows on snow",
    Zone.ZONE_VII: "Very light skin, light gray objects",
    Zone.ZONE_VIII: "Whites with texture, textured snow",
    Zone.ZONE_IX: "White without texture, glaring white surfaces",
    Zone.ZONE_X: "Pure white, paper base, specular highlights",
}


@dataclass
class ZoneMapping:
    """Mapping between zones and density values."""

    # Target print density for each zone
    zone_densities: dict[Zone, float] = field(default_factory=dict)

    # Paper characteristics
    paper_dmax: float = 1.6
    paper_dmin: float = 0.08

    def __post_init__(self) -> None:
        """Initialize default zone densities if not provided."""
        if not self.zone_densities:
            self.zone_densities = self._calculate_default_densities()

    def _calculate_default_densities(self) -> dict[Zone, float]:
        """Calculate default zone densities based on paper characteristics.

        Maps zones linearly from Dmax (Zone 0) to Dmin (Zone X).
        """
        densities = {}
        range_density = self.paper_dmax - self.paper_dmin

        for zone in Zone:
            # Zone 0 = Dmax, Zone X = Dmin
            fraction = zone.value / 10.0
            density = self.paper_dmax - (fraction * range_density)
            densities[zone] = round(density, 3)

        return densities

    def get_density(self, zone: Zone) -> float:
        """Get print density for a zone.

        Args:
            zone: Zone value (0-10)

        Returns:
            Target print density
        """
        return self.zone_densities.get(zone, 0.0)

    def get_zone_for_density(self, density: float) -> Zone:
        """Get closest zone for a density value.

        Args:
            density: Print density value

        Returns:
            Closest Zone
        """
        closest_zone = Zone.ZONE_V
        min_diff = float("inf")

        for zone, zone_density in self.zone_densities.items():
            diff = abs(density - zone_density)
            if diff < min_diff:
                min_diff = diff
                closest_zone = zone

        return closest_zone

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paper_dmax": self.paper_dmax,
            "paper_dmin": self.paper_dmin,
            "zones": {
                zone.name: {
                    "value": zone.value,
                    "density": density,
                    "description": ZONE_DESCRIPTIONS[zone],
                }
                for zone, density in self.zone_densities.items()
            },
        }


@dataclass
class ZoneAnalysis:
    """Analysis of an image using the Zone System."""

    # Zone distribution
    zone_histogram: dict[Zone, float] = field(default_factory=dict)

    # Key zones
    shadow_zone: Zone = Zone.ZONE_III  # Placed shadow zone
    highlight_zone: Zone = Zone.ZONE_VII  # Placed highlight zone

    # Statistics
    average_zone: float = 5.0
    zone_range: int = 7  # Number of zones from darkest to lightest

    # Recommendations
    exposure_adjustment_stops: float = 0.0
    development_adjustment: str = "N"  # N, N+1, N+2, N-1, N-2

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "zone_histogram": {z.name: v for z, v in self.zone_histogram.items()},
            "shadow_zone": self.shadow_zone.name,
            "highlight_zone": self.highlight_zone.name,
            "average_zone": round(self.average_zone, 2),
            "zone_range": self.zone_range,
            "exposure_adjustment_stops": self.exposure_adjustment_stops,
            "development_adjustment": self.development_adjustment,
            "notes": self.notes,
        }


class ZoneMapper:
    """Map images and densities to the Zone System.

    Provides tools for visualizing and analyzing images using
    Ansel Adams' Zone System methodology.
    """

    def __init__(self, mapping: ZoneMapping | None = None):
        """Initialize zone mapper.

        Args:
            mapping: Zone mapping configuration. If None, uses defaults.
        """
        self.mapping = mapping or ZoneMapping()

    def analyze_image(
        self,
        image: Image.Image,
        placed_shadow: int | None = None,
        placed_highlight: int | None = None,
    ) -> ZoneAnalysis:
        """Analyze an image using the Zone System.

        Args:
            image: PIL Image to analyze
            placed_shadow: Zone to place shadows on (for exposure calculation)
            placed_highlight: Zone to place highlights on

        Returns:
            ZoneAnalysis with distribution and recommendations
        """
        # Convert to grayscale
        gray = image.convert("L") if image.mode != "L" else image

        # Get pixel values
        arr = np.array(gray)

        # Calculate zone histogram
        zone_histogram = {}
        for zone in Zone:
            # Map zone to 0-255 range
            zone_min = int((zone.value / 10.0) * 255 - 12.75)
            zone_max = int((zone.value / 10.0) * 255 + 12.75)
            zone_min = max(0, zone_min)
            zone_max = min(255, zone_max)

            # Count pixels in this zone
            mask = (arr >= zone_min) & (arr <= zone_max)
            count = np.sum(mask)
            zone_histogram[zone] = count / arr.size

        # Find actual shadow and highlight zones
        cumsum = 0
        actual_shadow = Zone.ZONE_0
        for zone in Zone:
            cumsum += zone_histogram[zone]
            if cumsum >= 0.02:  # 2% threshold for shadow
                actual_shadow = zone
                break

        cumsum = 0
        actual_highlight = Zone.ZONE_X
        for zone in reversed(Zone):
            cumsum += zone_histogram[zone]
            if cumsum >= 0.02:  # 2% threshold for highlight
                actual_highlight = zone
                break

        # Calculate average zone
        avg = sum(zone.value * pct for zone, pct in zone_histogram.items())

        # Calculate zone range
        zone_range = actual_highlight.value - actual_shadow.value

        # Calculate recommendations
        notes = []
        exposure_adj = 0.0
        dev_adj = "N"

        # If shadows are placed
        shadow_zone = Zone(placed_shadow) if placed_shadow is not None else actual_shadow
        highlight_zone = (
            Zone(placed_highlight) if placed_highlight is not None else actual_highlight
        )

        if placed_shadow is not None:
            exposure_adj = placed_shadow - actual_shadow.value
            if exposure_adj > 0:
                notes.append(
                    f"Open up {exposure_adj} stop(s) to place shadows on Zone {placed_shadow}"
                )
            elif exposure_adj < 0:
                notes.append(
                    f"Close down {-exposure_adj} stop(s) to place shadows on Zone {placed_shadow}"
                )

        # Development recommendation based on subject contrast
        if zone_range > 8:
            dev_adj = "N-1" if zone_range == 9 else "N-2"
            notes.append(
                f"High contrast scene ({zone_range} zones). Recommend {dev_adj} development."
            )
        elif zone_range < 6:
            dev_adj = "N+1" if zone_range == 5 else "N+2"
            notes.append(
                f"Low contrast scene ({zone_range} zones). Recommend {dev_adj} development."
            )
        else:
            dev_adj = "N"
            notes.append(f"Normal contrast scene ({zone_range} zones). Normal development.")

        return ZoneAnalysis(
            zone_histogram=zone_histogram,
            shadow_zone=shadow_zone,
            highlight_zone=highlight_zone,
            average_zone=avg,
            zone_range=zone_range,
            exposure_adjustment_stops=exposure_adj,
            development_adjustment=dev_adj,
            notes=notes,
        )

    def create_zone_scale(
        self,
        width: int = 500,
        height: int = 50,
    ) -> Image.Image:
        """Create a visual Zone System scale.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PIL Image of zone scale
        """
        # Create gradient
        arr = np.zeros((height, width), dtype=np.uint8)
        zone_width = width // 11

        for zone in Zone:
            start = zone.value * zone_width
            end = min(start + zone_width, width)
            # Map zone to 0-255
            value = int((1 - zone.value / 10.0) * 255)
            arr[:, start:end] = value

        return Image.fromarray(arr, mode="L")

    def visualize_zones(
        self,
        image: Image.Image,
        posterize: bool = True,
    ) -> Image.Image:
        """Visualize zones in an image.

        Args:
            image: Source image
            posterize: If True, posterize to 11 levels

        Returns:
            Image with zones visualized
        """
        # Convert to grayscale
        gray = image.convert("L") if image.mode != "L" else image.copy()

        if posterize:
            # Posterize to 11 levels
            arr = np.array(gray)
            # Map to zones (0-10), then back to 0-255
            zones = (arr / 255.0 * 10).astype(int)
            posterized = (zones / 10.0 * 255).astype(np.uint8)
            return Image.fromarray(posterized, mode="L")
        else:
            return gray

    def density_to_zone(self, density: float) -> Zone:
        """Convert a print density to Zone.

        Args:
            density: Print density value

        Returns:
            Corresponding Zone
        """
        return self.mapping.get_zone_for_density(density)

    def zone_to_density(self, zone: Zone) -> float:
        """Convert a Zone to target print density.

        Args:
            zone: Zone value

        Returns:
            Target print density
        """
        return self.mapping.get_density(zone)

    def get_exposure_scale(self) -> str:
        """Get the typical exposure scale for Pt/Pd.

        Returns:
            Description of exposure scale
        """
        dmax = self.mapping.paper_dmax
        dmin = self.mapping.paper_dmin
        range_d = dmax - dmin
        stops = range_d / 0.3

        return f"Paper range: {dmin:.2f} to {dmax:.2f} ({range_d:.2f} density, ~{stops:.1f} stops)"

    @staticmethod
    def get_zone_descriptions() -> dict[Zone, str]:
        """Get descriptions for all zones.

        Returns:
            Dictionary of zone descriptions
        """
        return ZONE_DESCRIPTIONS.copy()

    @staticmethod
    def get_development_adjustments() -> dict[str, str]:
        """Get development adjustment recommendations.

        Returns:
            Dictionary of adjustment descriptions
        """
        return {
            "N-2": "Reduce development 30-40%. For very high contrast scenes (10+ zones).",
            "N-1": "Reduce development 15-20%. For high contrast scenes (8-9 zones).",
            "N": "Normal development. For normal contrast scenes (6-7 zones).",
            "N+1": "Increase development 15-20%. For low contrast scenes (5 zones).",
            "N+2": "Increase development 30-40%. For very low contrast scenes (4 zones or less).",
        }
