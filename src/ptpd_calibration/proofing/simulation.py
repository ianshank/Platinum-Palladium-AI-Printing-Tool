"""
Soft proofing simulation for visualizing final print appearance.

Simulates how prints will look on paper, accounting for:
- Paper white point (base color)
- Maximum density (Dmax)
- Ink/metal tone characteristics
- Paper texture simulation
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image


class PaperSimulation(str, Enum):
    """Paper simulation presets."""

    ARCHES_PLATINE = "arches_platine"
    BERGGER_COT320 = "bergger_cot320"
    HAHNEMUHLE_PLATINUM = "hahnemuhle_platinum"
    REVERE_PLATINUM = "revere_platinum"
    STONEHENGE = "stonehenge"
    CUSTOM = "custom"


# Paper simulation presets
PAPER_PRESETS = {
    PaperSimulation.ARCHES_PLATINE: {
        "white_rgb": (250, 246, 238),
        "dmax": 1.6,
        "dmin": 0.07,
        "tone": "neutral",
    },
    PaperSimulation.BERGGER_COT320: {
        "white_rgb": (245, 240, 229),
        "dmax": 1.55,
        "dmin": 0.08,
        "tone": "warm",
    },
    PaperSimulation.HAHNEMUHLE_PLATINUM: {
        "white_rgb": (250, 250, 250),
        "dmax": 1.65,
        "dmin": 0.06,
        "tone": "neutral",
    },
    PaperSimulation.REVERE_PLATINUM: {
        "white_rgb": (248, 244, 234),
        "dmax": 1.5,
        "dmin": 0.09,
        "tone": "warm",
    },
    PaperSimulation.STONEHENGE: {
        "white_rgb": (245, 238, 224),
        "dmax": 1.4,
        "dmin": 0.10,
        "tone": "warm",
    },
}


@dataclass
class ProofSettings:
    """Settings for soft proofing simulation."""

    # Paper characteristics
    paper_white_rgb: tuple[int, int, int] = (250, 246, 238)
    paper_dmax: float = 1.6
    paper_dmin: float = 0.07

    # Tone characteristics
    shadow_tone_rgb: tuple[int, int, int] = (25, 22, 20)  # Warm black for Pd
    platinum_ratio: float = 0.0  # 0 = all Pd (warm), 1 = all Pt (cool)

    # Simulation quality
    add_paper_texture: bool = False
    texture_strength: float = 0.1  # 0-1

    # Viewing conditions
    ambient_light_temperature: int = 5500  # Kelvin
    viewing_brightness: float = 1.0  # 0.5-1.5

    @classmethod
    def from_paper_preset(cls, preset: PaperSimulation) -> "ProofSettings":
        """Create settings from a paper preset.

        Args:
            preset: Paper simulation preset

        Returns:
            ProofSettings configured for that paper
        """
        if preset == PaperSimulation.CUSTOM:
            return cls()

        preset_data = PAPER_PRESETS.get(preset, PAPER_PRESETS[PaperSimulation.ARCHES_PLATINE])

        return cls(
            paper_white_rgb=preset_data["white_rgb"],
            paper_dmax=preset_data["dmax"],
            paper_dmin=preset_data["dmin"],
        )


@dataclass
class ProofResult:
    """Result of soft proofing simulation."""

    image: Image.Image
    settings: ProofSettings
    original_size: tuple[int, int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding image)."""
        return {
            "size": f"{self.image.size[0]}x{self.image.size[1]}",
            "original_size": f"{self.original_size[0]}x{self.original_size[1]}",
            "paper_white": self.settings.paper_white_rgb,
            "dmax": self.settings.paper_dmax,
            "dmin": self.settings.paper_dmin,
            "platinum_ratio": f"{self.settings.platinum_ratio * 100:.0f}%",
            "notes": self.notes,
        }


class SoftProofer:
    """Simulate print appearance for soft proofing.

    Applies paper characteristics and tone rendering to preview
    how a print will look on actual paper.
    """

    def __init__(self, settings: ProofSettings | None = None):
        """Initialize soft proofer.

        Args:
            settings: Proofing settings. If None, uses defaults.
        """
        self.settings = settings or ProofSettings()

    def proof(
        self,
        image: Image.Image,
        settings: ProofSettings | None = None,
    ) -> ProofResult:
        """Generate soft proof of an image.

        Args:
            image: Source image to proof
            settings: Optional override settings

        Returns:
            ProofResult with simulated print appearance
        """
        settings = settings or self.settings
        notes = []

        # Convert to grayscale if needed
        if image.mode not in ("L", "LA"):
            gray = image.convert("L")
            notes.append("Converted to grayscale for Pt/Pd simulation")
        else:
            gray = image if image.mode == "L" else image.split()[0]

        original_size = image.size

        # Apply tone curve based on paper characteristics
        arr = np.array(gray, dtype=np.float32) / 255.0

        # Map to density range
        # Input 0 (black) = Dmax, Input 1 (white) = Dmin
        density_range = settings.paper_dmax - settings.paper_dmin
        density = settings.paper_dmax - (arr * density_range)

        # Convert density back to reflectance for display
        # Using approximate formula: density = -log10(reflectance)
        reflectance = np.power(10, -density)

        # Normalize to 0-1 range
        reflectance = np.clip(reflectance, 0, 1)

        # Apply paper white point and shadow tone
        result = self._apply_paper_toning(reflectance, settings)

        # Add paper texture if requested
        if settings.add_paper_texture:
            result = self._add_texture(result, settings.texture_strength)
            notes.append("Paper texture simulation applied")

        # Apply viewing conditions
        result = self._apply_viewing_conditions(result, settings)

        # Convert back to image
        result_arr = (result * 255).astype(np.uint8)
        proof_image = Image.fromarray(result_arr, mode="RGB")

        # Add notes
        notes.append(f"Simulating {settings.paper_dmax:.2f} Dmax, {settings.paper_dmin:.2f} Dmin")

        if settings.platinum_ratio > 0.5:
            notes.append("Cool tones (platinum-dominant)")
        else:
            notes.append("Warm tones (palladium-dominant)")

        return ProofResult(
            image=proof_image,
            settings=settings,
            original_size=original_size,
            notes=notes,
        )

    def compare(
        self,
        image: Image.Image,
        settings_list: list[ProofSettings],
    ) -> list[ProofResult]:
        """Compare proofs with different settings.

        Args:
            image: Source image
            settings_list: List of settings to compare

        Returns:
            List of ProofResults
        """
        return [self.proof(image, settings) for settings in settings_list]

    def _apply_paper_toning(
        self,
        reflectance: np.ndarray,
        settings: ProofSettings,
    ) -> np.ndarray:
        """Apply paper white and shadow tones.

        Args:
            reflectance: 2D array of reflectance values (0-1)
            settings: Proofing settings

        Returns:
            3D RGB array
        """
        # Paper white
        pw = np.array(settings.paper_white_rgb, dtype=np.float32) / 255.0

        # Shadow tone (interpolate between warm Pd and cool Pt)
        warm_shadow = np.array(settings.shadow_tone_rgb, dtype=np.float32) / 255.0
        cool_shadow = np.array([22, 22, 25], dtype=np.float32) / 255.0  # Cooler neutral
        shadow = warm_shadow * (1 - settings.platinum_ratio) + cool_shadow * settings.platinum_ratio

        # Create RGB output
        h, w = reflectance.shape
        result = np.zeros((h, w, 3), dtype=np.float32)

        # Interpolate between shadow and paper white based on reflectance
        for c in range(3):
            result[:, :, c] = shadow[c] + reflectance * (pw[c] - shadow[c])

        return result

    def _add_texture(
        self,
        image: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Add subtle paper texture.

        Args:
            image: RGB array
            strength: Texture strength (0-1)

        Returns:
            Textured RGB array
        """
        h, w = image.shape[:2]

        # Generate subtle noise for texture
        np.random.seed(42)  # Consistent texture
        noise = np.random.normal(0, strength * 0.02, (h, w))

        # Apply to all channels
        for c in range(3):
            image[:, :, c] = np.clip(image[:, :, c] + noise, 0, 1)

        return image

    def _apply_viewing_conditions(
        self,
        image: np.ndarray,
        settings: ProofSettings,
    ) -> np.ndarray:
        """Apply viewing condition adjustments.

        Args:
            image: RGB array
            settings: Proofing settings

        Returns:
            Adjusted RGB array
        """
        # Apply brightness adjustment
        if settings.viewing_brightness != 1.0:
            image = np.clip(image * settings.viewing_brightness, 0, 1)

        # Simple color temperature adjustment
        if settings.ambient_light_temperature != 5500:
            # Warmer light (lower K) = more yellow/red
            # Cooler light (higher K) = more blue
            temp_diff = (settings.ambient_light_temperature - 5500) / 5000

            if temp_diff > 0:  # Cooler/bluer
                image[:, :, 2] = np.clip(image[:, :, 2] * (1 + temp_diff * 0.1), 0, 1)
            else:  # Warmer/redder
                image[:, :, 0] = np.clip(image[:, :, 0] * (1 - temp_diff * 0.1), 0, 1)
                image[:, :, 1] = np.clip(image[:, :, 1] * (1 - temp_diff * 0.05), 0, 1)

        return image

    @staticmethod
    def get_paper_presets() -> list[tuple[str, str]]:
        """Get list of paper presets with descriptions.

        Returns:
            List of (value, description) tuples
        """
        return [
            (PaperSimulation.ARCHES_PLATINE.value, "Arches Platine - Natural white, neutral"),
            (PaperSimulation.BERGGER_COT320.value, "Bergger COT 320 - Cream base, warm"),
            (PaperSimulation.HAHNEMUHLE_PLATINUM.value, "Hahnemuhle Platinum Rag - Bright white"),
            (PaperSimulation.REVERE_PLATINUM.value, "Revere Platinum - Natural warm"),
            (PaperSimulation.STONEHENGE.value, "Stonehenge - Warm cream base"),
            (PaperSimulation.CUSTOM.value, "Custom paper settings"),
        ]

    @staticmethod
    def get_dmax_range() -> tuple[float, float]:
        """Get typical Dmax range for Pt/Pd.

        Returns:
            Tuple of (min, max) typical Dmax values
        """
        return (1.3, 1.8)
