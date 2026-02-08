"""
Scanner calibration and profiling for accurate measurements.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ChannelCurve:
    """Response curve for a single color channel."""

    input_values: list[float] = field(default_factory=lambda: list(np.linspace(0, 255, 256)))
    output_values: list[float] = field(default_factory=lambda: list(np.linspace(0, 255, 256)))

    def apply(self, value: float) -> float:
        """Apply curve to a single value."""
        # Linear interpolation
        idx = int(np.clip(value, 0, 255))
        if idx >= len(self.output_values) - 1:
            return self.output_values[-1]
        frac = value - idx
        return (1 - frac) * self.output_values[idx] + frac * self.output_values[idx + 1]


@dataclass
class ScannerProfile:
    """Scanner calibration profile."""

    name: str
    created_at: str = ""
    scanner_model: str | None = None
    resolution_dpi: int | None = None

    # Response curves per channel
    red_curve: ChannelCurve = field(default_factory=ChannelCurve)
    green_curve: ChannelCurve = field(default_factory=ChannelCurve)
    blue_curve: ChannelCurve = field(default_factory=ChannelCurve)

    # Uniformity correction
    uniformity_map: np.ndarray | None = None
    uniformity_map_size: tuple[int, int] | None = None


class ScannerCalibration:
    """
    Scanner calibration and profile management.

    Creates profiles from reference target scans and applies
    corrections to subsequent scans for accurate measurements.
    """

    def __init__(self, profile: ScannerProfile | None = None):
        """
        Initialize scanner calibration.

        Args:
            profile: Optional pre-loaded scanner profile.
        """
        self.profile = profile

    def calibrate_from_target(
        self,
        target_scan: np.ndarray | Image.Image | Path | str,
        reference_values: dict[str, tuple[float, float, float]],
        name: str = "Scanner Profile",
    ) -> ScannerProfile:
        """
        Create calibration profile from reference target scan.

        Args:
            target_scan: Scan of calibration target (e.g., IT8).
            reference_values: Dict mapping patch IDs to known RGB values.
            name: Profile name.

        Returns:
            ScannerProfile with calibration curves.
        """
        # Load image
        img_array = self._load_image(target_scan)

        # For now, create identity profile
        # Full implementation would extract patches and compute curves
        from datetime import datetime

        profile = ScannerProfile(
            name=name,
            created_at=datetime.now().isoformat(),
        )

        # Analyze uniformity
        profile.uniformity_map = self._analyze_uniformity(img_array)
        profile.uniformity_map_size = (img_array.shape[1], img_array.shape[0])

        self.profile = profile
        return profile

    def calibrate_simple(
        self,
        white_sample: tuple[float, float, float],
        black_sample: tuple[float, float, float],
        target_white: tuple[float, float, float] = (255.0, 255.0, 255.0),
        target_black: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "Simple Profile",
    ) -> ScannerProfile:
        """
        Create simple calibration from white and black point samples.

        Args:
            white_sample: Measured white point RGB.
            black_sample: Measured black point RGB.
            target_white: Target white point RGB.
            target_black: Target black point RGB.
            name: Profile name.

        Returns:
            ScannerProfile with linear correction curves.
        """
        from datetime import datetime

        profile = ScannerProfile(
            name=name,
            created_at=datetime.now().isoformat(),
        )

        # Create linear correction curves for each channel
        for _i, (curve, ws, bs, tw, tb) in enumerate(
            [
                (
                    profile.red_curve,
                    white_sample[0],
                    black_sample[0],
                    target_white[0],
                    target_black[0],
                ),
                (
                    profile.green_curve,
                    white_sample[1],
                    black_sample[1],
                    target_white[1],
                    target_black[1],
                ),
                (
                    profile.blue_curve,
                    white_sample[2],
                    black_sample[2],
                    target_white[2],
                    target_black[2],
                ),
            ]
        ):
            # Linear mapping from sample range to target range
            if ws != bs:
                scale = (tw - tb) / (ws - bs)
                offset = tb - bs * scale
                curve.output_values = [
                    float(np.clip(v * scale + offset, 0, 255)) for v in range(256)
                ]

        self.profile = profile
        return profile

    def apply_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply calibration correction to image.

        Args:
            image: Input image array.

        Returns:
            Corrected image array.
        """
        if self.profile is None:
            return image

        corrected = image.astype(np.float64).copy()

        # Apply channel curves
        if len(image.shape) == 3:
            for i, curve in enumerate(
                [self.profile.red_curve, self.profile.green_curve, self.profile.blue_curve]
            ):
                if i < image.shape[2]:
                    corrected[:, :, i] = np.vectorize(curve.apply)(image[:, :, i])

        # Apply uniformity correction
        if self.profile.uniformity_map is not None:
            uniformity = self._resize_uniformity_map(
                self.profile.uniformity_map, (image.shape[1], image.shape[0])
            )
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    corrected[:, :, i] *= uniformity
            else:
                corrected *= uniformity

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def save(self, path: Path) -> None:
        """Save profile to JSON file."""
        if self.profile is None:
            raise ValueError("No profile to save")

        data = {
            "name": self.profile.name,
            "created_at": self.profile.created_at,
            "scanner_model": self.profile.scanner_model,
            "resolution_dpi": self.profile.resolution_dpi,
            "red_curve": {
                "input_values": self.profile.red_curve.input_values,
                "output_values": self.profile.red_curve.output_values,
            },
            "green_curve": {
                "input_values": self.profile.green_curve.input_values,
                "output_values": self.profile.green_curve.output_values,
            },
            "blue_curve": {
                "input_values": self.profile.blue_curve.input_values,
                "output_values": self.profile.blue_curve.output_values,
            },
        }

        if self.profile.uniformity_map is not None:
            data["uniformity_map"] = self.profile.uniformity_map.tolist()
            data["uniformity_map_size"] = self.profile.uniformity_map_size

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ScannerCalibration":
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)

        profile = ScannerProfile(
            name=data["name"],
            created_at=data.get("created_at", ""),
            scanner_model=data.get("scanner_model"),
            resolution_dpi=data.get("resolution_dpi"),
        )

        # Load curves
        for curve_name, curve_obj in [
            ("red_curve", profile.red_curve),
            ("green_curve", profile.green_curve),
            ("blue_curve", profile.blue_curve),
        ]:
            if curve_name in data:
                curve_obj.input_values = data[curve_name]["input_values"]
                curve_obj.output_values = data[curve_name]["output_values"]

        # Load uniformity map
        if "uniformity_map" in data:
            profile.uniformity_map = np.array(data["uniformity_map"])
            profile.uniformity_map_size = tuple(data.get("uniformity_map_size", [0, 0]))

        return cls(profile)

    def _load_image(self, image: np.ndarray | Image.Image | Path | str) -> np.ndarray:
        """Load image from various sources."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            return np.array(image)
        if isinstance(image, (Path, str)):
            pil_img = Image.open(image)
            return np.array(pil_img)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _analyze_uniformity(self, image: np.ndarray) -> np.ndarray:
        """Analyze scanner field uniformity."""
        from scipy.ndimage import uniform_filter

        # Convert to grayscale if color
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Low-pass filter to find illumination pattern
        smoothed = uniform_filter(gray.astype(float), size=50)

        # Normalize to mean
        mean_val = np.mean(smoothed)
        if mean_val > 0:
            uniformity = mean_val / np.maximum(smoothed, 1e-6)
        else:
            uniformity = np.ones_like(smoothed)

        # Clip to reasonable range
        uniformity = np.clip(uniformity, 0.8, 1.2)

        return uniformity

    def _resize_uniformity_map(
        self, uniformity: np.ndarray, target_size: tuple[int, int]
    ) -> np.ndarray:
        """Resize uniformity map to target size.

        Uses cubic spline interpolation (order=3) for smooth uniformity
        correction maps. This provides better quality than bilinear (order=1)
        while avoiding potential ringing artifacts from higher orders.

        Args:
            uniformity: Source uniformity map array
            target_size: Target (width, height) in pixels

        Returns:
            Resized uniformity map matching target dimensions
        """
        from scipy.ndimage import zoom

        current_size = (uniformity.shape[1], uniformity.shape[0])
        if current_size == target_size:
            return uniformity

        zoom_factors = (target_size[1] / uniformity.shape[0], target_size[0] / uniformity.shape[1])
        # Use order=3 (cubic spline) for smooth correction maps
        # This is closer to Lanczos quality while avoiding potential issues
        return zoom(uniformity, zoom_factors, order=3, mode="nearest")
