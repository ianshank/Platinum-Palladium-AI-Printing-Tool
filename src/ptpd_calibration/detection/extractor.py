"""
Density and color extraction from step tablet patches.

Implements robust measurement with outlier rejection for Pt/Pd prints.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ptpd_calibration.config import ExtractionSettings, get_settings
from ptpd_calibration.core.models import ExtractionResult, PatchData
from ptpd_calibration.detection.detector import DetectionResult


@dataclass
class ColorMeasurement:
    """Color measurement from a patch."""

    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    lab_mean: tuple[float, float, float]
    density: float
    uniformity: float


class DensityExtractor:
    """
    Extracts density and color information from detected patches.

    Uses robust statistics with MAD-based outlier rejection to handle
    dust, scratches, and coating irregularities common in Pt/Pd prints.
    """

    def __init__(self, settings: ExtractionSettings | None = None):
        """
        Initialize the extractor.

        Args:
            settings: Extraction settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings().extraction

    def extract(
        self,
        image: np.ndarray | Image.Image | Path | str,
        detection: DetectionResult,
        reference_white: tuple[float, float, float] | None = None,
    ) -> ExtractionResult:
        """
        Extract density measurements from detected patches.

        Args:
            image: Input image.
            detection: Detection result with patch locations.
            reference_white: Optional RGB values for white reference.

        Returns:
            ExtractionResult with all measurements.
        """
        # Load image
        img_array = self._load_image(image)

        # Rotate image if needed
        if abs(detection.rotation) > 0.1:
            from scipy.ndimage import rotate

            img_array = rotate(img_array, -detection.rotation, reshape=False, order=1)

        # Detect paper base
        paper_rgb, paper_density = self._detect_paper_base(img_array, detection)
        if reference_white is not None:
            paper_rgb = reference_white

        # Extract patches
        patches = []
        for i, bounds in enumerate(detection.patch_bounds):
            patch_data = self._extract_patch(img_array, bounds, i, paper_rgb)
            patches.append(patch_data)

        # Calculate quality metrics
        overall_quality = self._calculate_quality(patches, detection)
        warnings = self._generate_warnings(patches, detection)

        # Get image info
        height, width = img_array.shape[:2]
        source_path = image if isinstance(image, (Path, str)) else None
        if isinstance(source_path, str):
            source_path = Path(source_path)

        return ExtractionResult(
            source_path=source_path,
            image_size=(width, height),
            tablet_bounds=detection.bounds,
            rotation_angle=detection.rotation,
            orientation=detection.orientation,
            patches=patches,
            num_patches=len(patches),
            paper_base_rgb=paper_rgb,
            paper_base_density=paper_density,
            overall_quality=overall_quality,
            warnings=warnings + detection.warnings,
        )

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

    def _detect_paper_base(
        self, image: np.ndarray, detection: DetectionResult
    ) -> tuple[tuple[float, float, float], float]:
        """Detect paper base color from margins around tablet."""
        height, width = image.shape[:2]
        x, y, w, h = detection.bounds

        # Sample from margins
        margin = self.settings.paper_margin_ratio
        sample_size = self.settings.paper_sample_size

        samples = []

        # Top margin
        if y > sample_size:
            top_region = image[
                max(0, y - sample_size) : y,
                x : x + w,
            ]
            samples.extend(self._sample_region(top_region))

        # Bottom margin
        if y + h + sample_size < height:
            bottom_region = image[
                y + h : min(height, y + h + sample_size),
                x : x + w,
            ]
            samples.extend(self._sample_region(bottom_region))

        # Left margin
        if x > sample_size:
            left_region = image[
                y : y + h,
                max(0, x - sample_size) : x,
            ]
            samples.extend(self._sample_region(left_region))

        # Right margin
        if x + w + sample_size < width:
            right_region = image[
                y : y + h,
                x + w : min(width, x + w + sample_size),
            ]
            samples.extend(self._sample_region(right_region))

        if not samples:
            # Fallback: use image corners
            corner_size = int(min(height, width) * margin)
            corners = [
                image[:corner_size, :corner_size],
                image[:corner_size, -corner_size:],
                image[-corner_size:, :corner_size],
                image[-corner_size:, -corner_size:],
            ]
            for corner in corners:
                samples.extend(self._sample_region(corner))

        if not samples:
            # Ultimate fallback
            return (255.0, 255.0, 255.0), 0.05

        # Robust mean using MAD
        samples_array = np.array(samples)
        rgb_mean = self._robust_mean(samples_array)

        # Calculate paper base density
        paper_density = self._rgb_to_density(rgb_mean)

        return tuple(rgb_mean), paper_density

    def _sample_region(self, region: np.ndarray) -> list[np.ndarray]:
        """Sample random pixels from a region."""
        if region.size == 0:
            return []

        # Flatten to list of RGB values
        if len(region.shape) == 3:
            pixels = region.reshape(-1, region.shape[-1])
        else:
            pixels = region.flatten()
            pixels = np.column_stack([pixels, pixels, pixels])

        # Random sample
        n_samples = min(100, len(pixels))
        indices = np.random.choice(len(pixels), n_samples, replace=False)

        return [pixels[i] for i in indices]

    def _extract_patch(
        self,
        image: np.ndarray,
        bounds: tuple[int, int, int, int],
        index: int,
        paper_rgb: tuple[float, float, float],
    ) -> PatchData:
        """Extract measurements from a single patch."""
        x, y, w, h = bounds

        # Apply margin to avoid edges
        margin = self.settings.sample_margin_ratio
        mx = int(w * margin)
        my = int(h * margin)

        # Extract center region
        x1, y1 = x + mx, y + my
        x2, y2 = x + w - mx, y + h - my

        # Ensure valid region
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(x1 + 1, min(x2, image.shape[1]))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(y1 + 1, min(y2, image.shape[0]))

        region = image[y1:y2, x1:x2]

        # Get RGB values
        if len(region.shape) == 3:
            pixels = region.reshape(-1, region.shape[-1])[:, :3]
        else:
            pixels = region.flatten()
            pixels = np.column_stack([pixels, pixels, pixels])

        if len(pixels) < self.settings.min_sample_pixels:
            # Region too small, use full patch
            region = image[y : y + h, x : x + w]
            if len(region.shape) == 3:
                pixels = region.reshape(-1, region.shape[-1])[:, :3]
            else:
                pixels = region.flatten()
                pixels = np.column_stack([pixels, pixels, pixels])

        # Robust mean with outlier rejection
        rgb_mean = self._robust_mean(pixels)
        rgb_std = self._robust_std(pixels)

        # Convert to Lab
        lab_mean = self._rgb_to_lab(rgb_mean)

        # Calculate density relative to paper
        density = self._rgb_to_density(rgb_mean, paper_rgb)

        # Calculate uniformity
        uniformity = self._calculate_uniformity(pixels)

        return PatchData(
            index=index,
            position=bounds,
            rgb_mean=tuple(rgb_mean),
            rgb_std=tuple(rgb_std),
            lab_mean=tuple(lab_mean),
            density=density,
            uniformity=uniformity,
        )

    def _robust_mean(self, pixels: np.ndarray) -> np.ndarray:
        """Calculate robust mean using MAD-based outlier rejection."""
        if len(pixels) == 0:
            return np.array([128.0, 128.0, 128.0])

        pixels = pixels.astype(np.float64)

        # Calculate median
        median = np.median(pixels, axis=0)

        # Calculate MAD (Median Absolute Deviation)
        mad = np.median(np.abs(pixels - median), axis=0)
        mad = np.maximum(mad, 1e-6)  # Avoid division by zero

        # Calculate modified z-scores
        z_scores = np.abs(pixels - median) / (mad * 1.4826)  # 1.4826 is the MAD constant

        # Reject outliers
        threshold = self.settings.mad_threshold
        mask = np.all(z_scores < threshold, axis=1)

        if np.sum(mask) < len(pixels) * 0.5:
            # Too many outliers, use median
            return median

        return np.mean(pixels[mask], axis=0)

    def _robust_std(self, pixels: np.ndarray) -> np.ndarray:
        """Calculate robust standard deviation."""
        if len(pixels) == 0:
            return np.array([0.0, 0.0, 0.0])

        # Use MAD as robust estimator
        mad = np.median(np.abs(pixels - np.median(pixels, axis=0)), axis=0)
        # Convert MAD to standard deviation estimate
        return mad * 1.4826

    def _rgb_to_density(
        self,
        rgb: np.ndarray,
        reference: tuple[float, float, float] | None = None,
    ) -> float:
        """
        Convert RGB to visual density using Status A weighting.

        Density = -log10(reflectance)
        where reflectance = sample / reference
        """
        # Status A weights (appropriate for warm-toned prints)
        weights = np.array(self.settings.status_a_weights)

        # Normalize to 0-1
        rgb_norm = np.array(rgb) / 255.0

        if reference is not None:
            ref_norm = np.array(reference) / 255.0
            # Ensure minimum reflectance
            ref_norm = np.maximum(ref_norm, 0.01)
        else:
            ref_norm = np.array([self.settings.reference_white_reflectance] * 3)

        # Calculate reflectance
        reflectance = np.clip(rgb_norm / ref_norm, 0.001, 1.0)

        # Weighted reflectance (Status A)
        weighted_reflectance = np.sum(reflectance * weights)

        # Convert to density
        density = -np.log10(weighted_reflectance)

        return float(max(0.0, density))

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to CIE L*a*b* color space."""
        # sRGB to XYZ
        rgb_linear = np.where(
            rgb / 255.0 <= 0.04045,
            rgb / 255.0 / 12.92,
            ((rgb / 255.0 + 0.055) / 1.055) ** 2.4,
        )

        # sRGB to XYZ matrix (D65 illuminant)
        matrix = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )

        xyz = np.dot(matrix, rgb_linear)

        # D65 white point
        white = np.array([0.95047, 1.0, 1.08883])

        # XYZ to Lab
        xyz_norm = xyz / white

        f = np.where(
            xyz_norm > 0.008856,
            xyz_norm ** (1 / 3),
            (903.3 * xyz_norm + 16) / 116,
        )

        L = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b = 200 * (f[1] - f[2])

        return np.array([L, a, b])

    def _calculate_uniformity(self, pixels: np.ndarray) -> float:
        """Calculate patch uniformity score."""
        if len(pixels) == 0:
            return 0.0

        # Use coefficient of variation
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)

        # Avoid division by zero
        mean = np.maximum(mean, 1e-6)

        cv = np.mean(std / mean)

        # Convert to 0-1 score (lower CV = higher uniformity)
        uniformity = 1.0 - min(1.0, cv)

        return float(uniformity)

    def _calculate_quality(self, patches: list[PatchData], detection: DetectionResult) -> float:
        """Calculate overall extraction quality score."""
        if not patches:
            return 0.0

        # Patch uniformity
        uniformities = [p.uniformity for p in patches]
        avg_uniformity = np.mean(uniformities)

        # Density monotonicity
        densities = [p.density for p in patches if p.density is not None]
        if len(densities) > 1:
            # Check if densities are monotonically increasing
            diffs = np.diff(densities)
            monotonicity = np.mean(diffs > 0)
        else:
            monotonicity = 1.0

        # Detection confidence
        detection_conf = detection.confidence

        # Combined quality
        quality = 0.4 * avg_uniformity + 0.3 * monotonicity + 0.3 * detection_conf

        return float(quality)

    def _generate_warnings(self, patches: list[PatchData], detection: DetectionResult) -> list[str]:
        """Generate warnings based on extraction quality."""
        warnings = []

        # Check for low uniformity patches
        low_uniformity = [p for p in patches if p.uniformity < 0.7]
        if low_uniformity:
            warnings.append(
                f"{len(low_uniformity)} patches have low uniformity (possible coating issues)"
            )

        # Check density range
        densities = [p.density for p in patches if p.density is not None]
        if densities:
            dmax = max(densities)
            dmin = min(densities)
            if dmax - dmin < 1.0:
                warnings.append(f"Low density range ({dmax - dmin:.2f}): consider longer exposure")
            elif dmax > 3.0:
                warnings.append(f"Very high Dmax ({dmax:.2f}): possible overexposure")

        # Check monotonicity
        if len(densities) > 1:
            diffs = np.diff(densities)
            reversals = np.sum(diffs < -0.05)
            if reversals > 0:
                warnings.append(f"{reversals} density reversals detected (possible solarization)")

        return warnings
