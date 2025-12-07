"""
Synthetic data generators for training deep learning models.

Key Design Principles:
1. NO exact input-output matching to prevent hallucinations
2. Realistic noise and variation in all data
3. Physically plausible values based on Pt/Pd printing domain
4. Configurable randomness for reproducibility
5. Support for train/val/test splits with no data leakage
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple
import json
import logging
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.

    All parameters are configurable - no hardcoded values in generators.
    """

    # Random seed for reproducibility
    seed: int = 42

    # Dataset sizes
    train_samples: int = 10000
    val_samples: int = 2000
    test_samples: int = 2000

    # Noise levels (prevent exact matching)
    input_noise_std: float = 0.05
    output_noise_std: float = 0.02
    label_noise_probability: float = 0.01

    # Image generation
    image_size: Tuple[int, int] = (640, 480)
    num_patches_range: Tuple[int, int] = (11, 41)

    # Curve generation
    curve_points: int = 256
    density_range: Tuple[float, float] = (0.1, 2.5)

    # Exposure generation
    exposure_range: Tuple[float, float] = (30.0, 600.0)
    humidity_range: Tuple[float, float] = (30.0, 80.0)
    temperature_range: Tuple[float, float] = (15.0, 30.0)

    # Defect generation
    defect_probability: float = 0.3
    max_defects_per_image: int = 10
    defect_size_range: Tuple[int, int] = (10, 100)

    # Augmentation
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.8, 1.2)

    # Output paths
    output_dir: Optional[Path] = None
    save_images: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.input_noise_std < 0:
            raise ValueError("input_noise_std must be non-negative")
        if self.output_noise_std < 0:
            raise ValueError("output_noise_std must be non-negative")
        if self.label_noise_probability < 0 or self.label_noise_probability > 1:
            raise ValueError("label_noise_probability must be in [0, 1]")


class PaperType(str, Enum):
    """Simulated paper types for training data."""
    ARCHES_PLATINE = "arches_platine"
    BERGGER_COT320 = "bergger_cot320"
    HAHNEMUHLE_PLATINUM = "hahnemuhle_platinum"
    REVERE_PLATINUM = "revere_platinum"
    STONEHENGE = "stonehenge"
    CANSON = "canson"
    RIVES_BFK = "rives_bfk"
    FABRIANO = "fabriano"


class UVSourceType(str, Enum):
    """Simulated UV sources for training data."""
    NUARC = "nuarc"
    FLUORESCENT_BL = "fluorescent_bl"
    LED_365NM = "led_365nm"
    LED_405NM = "led_405nm"
    MERCURY_VAPOR = "mercury_vapor"
    SUNLIGHT = "sunlight"


# =============================================================================
# Base Generator
# =============================================================================


class BaseDataGenerator:
    """Base class for all data generators."""

    def __init__(self, config: SyntheticDataConfig):
        """Initialize generator with configuration.

        Args:
            config: Generation configuration (no defaults, must be provided)
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._sample_count = 0

    def reset(self, seed: Optional[int] = None):
        """Reset generator state.

        Args:
            seed: Optional new seed
        """
        new_seed = seed if seed is not None else self.config.seed
        self.rng = np.random.default_rng(new_seed)
        self._sample_count = 0

    def _add_noise(self, value: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to prevent exact matching.

        Args:
            value: Input array
            std: Noise standard deviation

        Returns:
            Noised array (prevents input-output memorization)
        """
        noise = self.rng.normal(0, std, value.shape)
        return value + noise

    def _add_label_noise(self, label: int, num_classes: int) -> int:
        """Randomly flip labels to prevent overfitting.

        Args:
            label: Original label
            num_classes: Total number of classes

        Returns:
            Potentially noised label
        """
        if self.rng.random() < self.config.label_noise_probability:
            return self.rng.integers(0, num_classes)
        return label


# =============================================================================
# Step Tablet Detection Data Generator
# =============================================================================


class DetectionDataGenerator(BaseDataGenerator):
    """Generate synthetic step tablet images for detection training.

    Creates realistic step tablet images with:
    - Variable number of patches (11-41)
    - Realistic density gradients
    - Paper texture simulation
    - Rotation and perspective variations
    - Lighting variations
    - Noise and artifacts

    Output includes bounding boxes and segmentation masks.
    """

    def __init__(self, config: SyntheticDataConfig):
        """Initialize detection data generator."""
        super().__init__(config)
        self._paper_textures = self._generate_paper_textures()

    def _generate_paper_textures(self) -> list[np.ndarray]:
        """Pre-generate paper texture patterns."""
        textures = []
        for _ in range(10):
            texture = self.rng.normal(0.95, 0.03, (100, 100))
            texture = np.clip(texture, 0.8, 1.0)
            textures.append(texture.astype(np.float32))
        return textures

    def generate_sample(self) -> dict[str, Any]:
        """Generate a single training sample.

        Returns:
            Dictionary containing:
            - image: RGB image array (H, W, 3)
            - tablet_bbox: (x, y, w, h) bounding box
            - patch_bboxes: List of (x, y, w, h) for each patch
            - patch_masks: List of binary masks
            - num_patches: Number of patches
            - rotation: Applied rotation angle
            - metadata: Additional info
        """
        self._sample_count += 1

        # Randomize parameters
        num_patches = self.rng.integers(*self.config.num_patches_range)
        rotation = self.rng.uniform(*self.config.rotation_range)
        scale = self.rng.uniform(*self.config.scale_range)
        brightness = self.rng.uniform(*self.config.brightness_range)

        # Generate base image
        width, height = self.config.image_size
        image = self._generate_paper_background(width, height)

        # Calculate tablet dimensions
        margin = int(min(width, height) * 0.1)
        tablet_width = int((width - 2 * margin) * scale)
        tablet_height = int(tablet_width * 0.15)  # Typical step tablet ratio

        # Center the tablet
        tablet_x = (width - tablet_width) // 2
        tablet_y = (height - tablet_height) // 2

        # Generate patches
        patch_width = tablet_width // num_patches
        patches = []
        patch_bboxes = []
        patch_masks = []

        for i in range(num_patches):
            # Calculate density (lightest to darkest)
            density = self.config.density_range[0] + (
                (self.config.density_range[1] - self.config.density_range[0])
                * (i / (num_patches - 1))
            )

            # Add noise to density to prevent exact matching
            density = density + self.rng.normal(0, self.config.output_noise_std * 0.5)
            density = np.clip(density, 0.0, 3.0)

            # Convert density to grayscale (higher density = darker)
            gray_value = int(255 * (10 ** (-density)))
            gray_value = np.clip(gray_value, 5, 250)

            # Apply brightness variation
            gray_value = int(gray_value * brightness)
            gray_value = np.clip(gray_value, 0, 255)

            # Calculate patch position
            px = tablet_x + i * patch_width
            py = tablet_y
            pw = patch_width - 2  # Small gap between patches
            ph = tablet_height

            # Draw patch with realistic edges
            patch_bbox = (px, py, pw, ph)
            patch_bboxes.append(patch_bbox)

            # Create patch with slight edge variation
            edge_noise = self.rng.integers(-2, 3, 4)
            image[py + edge_noise[0]:py + ph + edge_noise[1],
                  px + edge_noise[2]:px + pw + edge_noise[3]] = gray_value

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[py:py + ph, px:px + pw] = 1
            patch_masks.append(mask)

            patches.append({
                "index": i,
                "density": density,
                "gray_value": gray_value,
                "bbox": patch_bbox,
            })

        # Apply rotation
        if abs(rotation) > 0.5:
            image, patch_bboxes = self._apply_rotation(
                image, patch_bboxes, rotation, (width // 2, height // 2)
            )

        # Add realistic artifacts
        image = self._add_artifacts(image)

        # Add input noise to prevent memorization
        noise = self.rng.normal(0, self.config.input_noise_std * 255, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Calculate overall tablet bbox
        if patch_bboxes:
            all_x = [b[0] for b in patch_bboxes]
            all_y = [b[1] for b in patch_bboxes]
            all_x2 = [b[0] + b[2] for b in patch_bboxes]
            all_y2 = [b[1] + b[3] for b in patch_bboxes]
            tablet_bbox = (
                min(all_x),
                min(all_y),
                max(all_x2) - min(all_x),
                max(all_y2) - min(all_y),
            )
        else:
            tablet_bbox = (tablet_x, tablet_y, tablet_width, tablet_height)

        return {
            "image": image,
            "tablet_bbox": tablet_bbox,
            "patch_bboxes": patch_bboxes,
            "patch_masks": patch_masks,
            "num_patches": num_patches,
            "rotation": rotation,
            "brightness": brightness,
            "scale": scale,
            "patches": patches,
            "sample_id": str(uuid4()),
        }

    def _generate_paper_background(self, width: int, height: int) -> np.ndarray:
        """Generate realistic paper background texture."""
        # Base cream/white color
        base_color = self.rng.integers(240, 255)
        image = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add paper texture
        texture = self._paper_textures[self.rng.integers(0, len(self._paper_textures))]
        texture_tiled = np.tile(
            texture,
            (height // texture.shape[0] + 1, width // texture.shape[1] + 1)
        )[:height, :width]

        # Apply texture
        for c in range(3):
            image[:, :, c] = (image[:, :, c] * texture_tiled).astype(np.uint8)

        return image

    def _apply_rotation(
        self,
        image: np.ndarray,
        bboxes: list[tuple],
        angle: float,
        center: tuple[int, int],
    ) -> tuple[np.ndarray, list[tuple]]:
        """Apply rotation to image and update bounding boxes."""
        from scipy.ndimage import rotate as scipy_rotate

        # Rotate image
        rotated = scipy_rotate(image, angle, reshape=False, order=1)

        # Update bboxes (simplified - in production would use proper transforms)
        # For training data, small rotation errors are acceptable
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))

        new_bboxes = []
        for x, y, w, h in bboxes:
            # Rotate center point
            cx, cy = x + w / 2, y + h / 2
            dx, dy = cx - center[0], cy - center[1]
            new_cx = center[0] + dx * cos_a - dy * sin_a
            new_cy = center[1] + dx * sin_a + dy * cos_a
            new_bboxes.append((
                int(new_cx - w / 2),
                int(new_cy - h / 2),
                w,
                h,
            ))

        return rotated, new_bboxes

    def _add_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic artifacts (dust, scratches, etc.)."""
        # Random dust spots
        num_dust = self.rng.integers(0, 20)
        for _ in range(num_dust):
            x = self.rng.integers(0, image.shape[1])
            y = self.rng.integers(0, image.shape[0])
            radius = self.rng.integers(1, 4)
            color = self.rng.integers(50, 150)

            y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
            x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)
            image[y_min:y_max, x_min:x_max] = color

        return image

    def generate_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Generate a batch of training samples.

        Args:
            batch_size: Number of samples to generate

        Returns:
            List of sample dictionaries
        """
        return [self.generate_sample() for _ in range(batch_size)]

    def generate_dataset(
        self,
        split: str = "train",
    ) -> Iterator[dict[str, Any]]:
        """Generate full dataset for specified split.

        Args:
            split: One of 'train', 'val', 'test'

        Yields:
            Sample dictionaries
        """
        if split == "train":
            n_samples = self.config.train_samples
            seed_offset = 0
        elif split == "val":
            n_samples = self.config.val_samples
            seed_offset = 100000
        elif split == "test":
            n_samples = self.config.test_samples
            seed_offset = 200000
        else:
            raise ValueError(f"Invalid split: {split}")

        # Reset with split-specific seed to ensure no overlap
        self.reset(self.config.seed + seed_offset)

        for _ in range(n_samples):
            yield self.generate_sample()


# =============================================================================
# Curve Prediction Data Generator
# =============================================================================


class CurveDataGenerator(BaseDataGenerator):
    """Generate synthetic curve data for neural curve prediction training.

    Creates realistic calibration curve data with:
    - Physically plausible density responses
    - Process parameter conditioning
    - Realistic measurement noise
    - Monotonicity constraints
    - Various curve shapes based on chemistry
    """

    # Paper speed factors (relative to baseline)
    PAPER_SPEED_FACTORS = {
        PaperType.ARCHES_PLATINE: 1.0,
        PaperType.BERGGER_COT320: 1.1,
        PaperType.HAHNEMUHLE_PLATINUM: 0.95,
        PaperType.REVERE_PLATINUM: 1.05,
        PaperType.STONEHENGE: 1.2,
        PaperType.CANSON: 1.15,
        PaperType.RIVES_BFK: 1.1,
        PaperType.FABRIANO: 1.0,
    }

    def generate_sample(self) -> dict[str, Any]:
        """Generate a single curve training sample.

        Returns:
            Dictionary containing:
            - input_densities: Measured step tablet densities
            - output_curve: Linearization curve points
            - conditioning: Process parameter dictionary
            - metadata: Additional info
        """
        self._sample_count += 1

        # Generate random process parameters
        paper_type = self.rng.choice(list(PaperType))
        metal_ratio = self.rng.uniform(0.0, 1.0)  # Pt/(Pt+Pd)
        contrast_amount = self.rng.uniform(0.0, 0.3)
        humidity = self.rng.uniform(*self.config.humidity_range)
        temperature = self.rng.uniform(*self.config.temperature_range)

        # Generate base curve shape based on parameters
        num_points = self.config.curve_points
        x = np.linspace(0, 1, num_points)

        # Gamma based on metal ratio (more Pt = higher contrast)
        gamma = 0.7 + metal_ratio * 0.4 + contrast_amount * 0.3

        # Generate characteristic curve with noise
        y = self._generate_characteristic_curve(x, gamma, humidity, temperature)

        # Add realistic measurement noise
        y = self._add_noise(y, self.config.output_noise_std)
        y = np.clip(y, 0, 1)

        # Ensure monotonicity (physical constraint)
        y = np.maximum.accumulate(y)

        # Generate corresponding measured densities (what we'd get from step tablet)
        num_steps = 21  # Typical step tablet
        step_x = np.linspace(0, 1, num_steps)
        measured_densities = np.interp(step_x, x, y)

        # Add input noise to densities (to prevent exact matching)
        measured_densities = self._add_noise(
            measured_densities, self.config.input_noise_std
        )
        measured_densities = np.clip(measured_densities, 0, 1)

        # Convert to actual density values
        dmin = self.rng.uniform(0.08, 0.15)
        dmax = self.rng.uniform(1.8, 2.4)
        measured_densities = dmin + measured_densities * (dmax - dmin)

        # Conditioning features
        conditioning = {
            "paper_type": paper_type.value,
            "paper_speed": self.PAPER_SPEED_FACTORS[paper_type],
            "metal_ratio": metal_ratio,
            "contrast_amount": contrast_amount,
            "humidity": humidity,
            "temperature": temperature,
            "dmin": dmin,
            "dmax": dmax,
        }

        return {
            "input_densities": measured_densities.tolist(),
            "output_curve_x": x.tolist(),
            "output_curve_y": y.tolist(),
            "conditioning": conditioning,
            "gamma": gamma,
            "sample_id": str(uuid4()),
        }

    def _generate_characteristic_curve(
        self,
        x: np.ndarray,
        gamma: float,
        humidity: float,
        temperature: float,
    ) -> np.ndarray:
        """Generate a characteristic curve with environmental factors.

        Args:
            x: Input values (0-1)
            gamma: Base gamma value
            humidity: Relative humidity (%)
            temperature: Temperature (°C)

        Returns:
            Output values with realistic curve shape
        """
        # Base power curve
        y = x ** gamma

        # Add shoulder and toe regions
        toe_strength = 0.1 + self.rng.uniform(-0.05, 0.05)
        shoulder_strength = 0.1 + self.rng.uniform(-0.05, 0.05)

        # Toe (shadow compression)
        toe = toe_strength * (1 - np.exp(-x * 5))
        y = np.where(x < 0.2, y + toe * (0.2 - x) * 5, y)

        # Shoulder (highlight compression)
        shoulder = shoulder_strength * (1 - np.exp(-(1 - x) * 5))
        y = np.where(x > 0.8, y - shoulder * (x - 0.8) * 5, y)

        # Environmental adjustments
        humidity_factor = 1 + (humidity - 50) * 0.002
        temp_factor = 1 + (temperature - 20) * 0.003

        y = y * humidity_factor * temp_factor

        # Normalize to 0-1
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        return y

    def generate_dataset(
        self,
        split: str = "train",
    ) -> Iterator[dict[str, Any]]:
        """Generate full dataset for specified split."""
        if split == "train":
            n_samples = self.config.train_samples
            seed_offset = 300000
        elif split == "val":
            n_samples = self.config.val_samples
            seed_offset = 400000
        else:
            n_samples = self.config.test_samples
            seed_offset = 500000

        self.reset(self.config.seed + seed_offset)

        for _ in range(n_samples):
            yield self.generate_sample()


# =============================================================================
# UV Exposure Data Generator
# =============================================================================


class ExposureDataGenerator(BaseDataGenerator):
    """Generate synthetic UV exposure data for training.

    Creates realistic exposure time data with:
    - Multiple UV source types
    - Paper speed factors
    - Environmental adjustments
    - Realistic variation and noise
    """

    # Base exposure times for different UV sources (seconds at 100% intensity)
    UV_SOURCE_BASE_TIMES = {
        UVSourceType.NUARC: 180.0,
        UVSourceType.FLUORESCENT_BL: 300.0,
        UVSourceType.LED_365NM: 150.0,
        UVSourceType.LED_405NM: 200.0,
        UVSourceType.MERCURY_VAPOR: 120.0,
        UVSourceType.SUNLIGHT: 600.0,
    }

    def generate_sample(self) -> dict[str, Any]:
        """Generate a single exposure training sample.

        Returns:
            Dictionary containing:
            - input_features: Feature dictionary
            - target_exposure: Target exposure time in seconds
            - confidence_interval: (lower, upper) bounds
        """
        self._sample_count += 1

        # Random parameters
        paper_type = self.rng.choice(list(PaperType))
        uv_source = self.rng.choice(list(UVSourceType))
        target_density = self.rng.uniform(1.5, 2.3)
        chemistry_ratio = self.rng.uniform(0.0, 1.0)  # Pt ratio
        humidity = self.rng.uniform(*self.config.humidity_range)
        temperature = self.rng.uniform(*self.config.temperature_range)
        coating_thickness = self.rng.uniform(0.8, 1.2)  # Relative to standard
        negative_dmax = self.rng.uniform(1.8, 2.5)

        # Calculate base exposure
        base_exposure = self.UV_SOURCE_BASE_TIMES[uv_source]

        # Apply paper speed factor
        paper_speed = CurveDataGenerator.PAPER_SPEED_FACTORS[paper_type]
        exposure = base_exposure * paper_speed

        # Chemistry adjustment (more Pt = longer exposure)
        chemistry_factor = 1.0 + chemistry_ratio * 0.2
        exposure *= chemistry_factor

        # Target density adjustment
        density_factor = target_density / 2.0
        exposure *= density_factor

        # Environmental adjustments
        humidity_factor = 1.0 + (humidity - 50) * 0.01
        temp_factor = 1.0 - (temperature - 20) * 0.02
        exposure *= humidity_factor * temp_factor

        # Coating thickness adjustment
        exposure *= coating_thickness

        # Negative density adjustment
        negative_factor = negative_dmax / 2.2
        exposure *= negative_factor

        # Add realistic variation (prevent exact mapping)
        exposure = exposure * self.rng.uniform(0.9, 1.1)
        exposure = self._add_noise(
            np.array([exposure]), self.config.output_noise_std * exposure
        )[0]

        # Confidence interval
        uncertainty = exposure * 0.15
        lower_bound = exposure - uncertainty
        upper_bound = exposure + uncertainty

        # Input features with noise
        features = {
            "target_density": float(target_density + self.rng.normal(0, 0.02)),
            "paper_type": paper_type.value,
            "paper_type_idx": list(PaperType).index(paper_type),
            "chemistry_ratio": float(chemistry_ratio + self.rng.normal(0, 0.02)),
            "uv_source": uv_source.value,
            "uv_source_idx": list(UVSourceType).index(uv_source),
            "humidity": float(humidity + self.rng.normal(0, 2)),
            "temperature": float(temperature + self.rng.normal(0, 0.5)),
            "coating_thickness": float(coating_thickness + self.rng.normal(0, 0.05)),
            "negative_dmax": float(negative_dmax + self.rng.normal(0, 0.05)),
        }

        return {
            "input_features": features,
            "target_exposure": float(np.clip(exposure, 30, 1200)),
            "lower_bound": float(max(30, lower_bound)),
            "upper_bound": float(min(1200, upper_bound)),
            "sample_id": str(uuid4()),
        }

    def generate_dataset(
        self,
        split: str = "train",
    ) -> Iterator[dict[str, Any]]:
        """Generate full dataset for specified split."""
        if split == "train":
            n_samples = self.config.train_samples
            seed_offset = 600000
        elif split == "val":
            n_samples = self.config.val_samples
            seed_offset = 700000
        else:
            n_samples = self.config.test_samples
            seed_offset = 800000

        self.reset(self.config.seed + seed_offset)

        for _ in range(n_samples):
            yield self.generate_sample()


# =============================================================================
# Defect Detection Data Generator
# =============================================================================


class DefectType(str, Enum):
    """Defect types for synthetic data."""
    BRUSH_MARK = "brush_mark"
    POOLING = "pooling"
    DUST = "dust"
    SCRATCH = "scratch"
    STAIN = "stain"
    UNEVEN_COATING = "uneven_coating"
    WATER_SPOT = "water_spot"


class DefectDataGenerator(BaseDataGenerator):
    """Generate synthetic defect detection training data.

    Creates realistic print images with:
    - Various defect types and severities
    - Accurate segmentation masks
    - Multiple defects per image
    - Realistic artifact appearance
    """

    def generate_sample(self) -> dict[str, Any]:
        """Generate a single defect detection sample.

        Returns:
            Dictionary containing:
            - image: Print image with defects
            - mask: Defect segmentation mask
            - defect_info: List of defect details
            - class_labels: Per-defect classification
        """
        self._sample_count += 1

        # Generate base print image
        width, height = self.config.image_size
        image = self._generate_print_image(width, height)

        # Initialize mask
        mask = np.zeros((height, width), dtype=np.uint8)
        defect_info = []

        # Decide if this image has defects
        if self.rng.random() < self.config.defect_probability:
            num_defects = self.rng.integers(1, self.config.max_defects_per_image + 1)

            for _ in range(num_defects):
                defect_type = self.rng.choice(list(DefectType))
                defect_data = self._add_defect(image, mask, defect_type)
                if defect_data:
                    defect_info.append(defect_data)

        # Add noise to prevent exact matching
        noise = self.rng.normal(0, self.config.input_noise_std * 255, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Add label noise
        for defect in defect_info:
            defect["class_idx"] = self._add_label_noise(
                defect["class_idx"], len(DefectType)
            )

        return {
            "image": image,
            "mask": mask,
            "defect_info": defect_info,
            "num_defects": len(defect_info),
            "has_defects": len(defect_info) > 0,
            "sample_id": str(uuid4()),
        }

    def _generate_print_image(self, width: int, height: int) -> np.ndarray:
        """Generate a base print image (without defects)."""
        # Gradient from light to dark (simulating tonal variation)
        gradient = np.linspace(0.9, 0.3, width)
        image = np.tile(gradient, (height, 1))

        # Add paper texture
        texture = self.rng.normal(1.0, 0.02, (height, width))
        image = image * texture

        # Add some local variation
        for _ in range(5):
            cx = self.rng.integers(0, width)
            cy = self.rng.integers(0, height)
            radius = self.rng.integers(50, 150)
            intensity = self.rng.uniform(-0.05, 0.05)

            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            falloff = np.exp(-dist / radius)
            image = image + intensity * falloff

        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return np.stack([image, image, image], axis=-1)

    def _add_defect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        defect_type: DefectType,
    ) -> Optional[dict[str, Any]]:
        """Add a defect to the image and mask."""
        height, width = image.shape[:2]

        # Random position
        cx = self.rng.integers(20, width - 20)
        cy = self.rng.integers(20, height - 20)
        size = self.rng.integers(*self.config.defect_size_range)

        # Severity (0-4 corresponding to negligible to critical)
        severity = self.rng.integers(0, 5)

        defect_mask = np.zeros((height, width), dtype=np.uint8)

        if defect_type == DefectType.BRUSH_MARK:
            # Elongated streak
            length = size * 2
            angle = self.rng.uniform(-30, 30)
            self._draw_streak(defect_mask, cx, cy, length, size // 4, angle)
            self._draw_streak(image[:, :, 0], cx, cy, length, size // 4, angle,
                              value=int(image[cy, cx, 0] * 0.9))

        elif defect_type == DefectType.DUST:
            # Small circular spot
            radius = min(size // 4, 10)
            self._draw_circle(defect_mask, cx, cy, radius)
            self._draw_circle(image[:, :, 0], cx, cy, radius,
                              value=self.rng.integers(30, 80))

        elif defect_type == DefectType.SCRATCH:
            # Thin line
            length = size
            angle = self.rng.uniform(0, 180)
            self._draw_line(defect_mask, cx, cy, length, 2, angle)
            self._draw_line(image[:, :, 0], cx, cy, length, 2, angle,
                            value=255)

        elif defect_type == DefectType.STAIN:
            # Irregular blob
            self._draw_blob(defect_mask, cx, cy, size)
            base_val = image[cy, cx, 0]
            self._draw_blob(image[:, :, 0], cx, cy, size,
                            value=int(base_val * 0.7))

        elif defect_type == DefectType.WATER_SPOT:
            # Circular with edge
            radius = size // 2
            self._draw_ring(defect_mask, cx, cy, radius, 3)
            self._draw_ring(image[:, :, 0], cx, cy, radius, 3,
                            value=int(image[cy, cx, 0] * 1.1))

        else:  # POOLING, UNEVEN_COATING
            # Large irregular area
            self._draw_blob(defect_mask, cx, cy, size * 2)
            self._draw_blob(image[:, :, 0], cx, cy, size * 2,
                            value=int(image[cy, cx, 0] * 0.85))

        # Update main mask
        mask |= defect_mask

        # Calculate bounding box
        coords = np.where(defect_mask > 0)
        if len(coords[0]) == 0:
            return None

        bbox = (
            int(coords[1].min()),
            int(coords[0].min()),
            int(coords[1].max() - coords[1].min()),
            int(coords[0].max() - coords[0].min()),
        )

        return {
            "defect_type": defect_type.value,
            "class_idx": list(DefectType).index(defect_type),
            "severity": severity,
            "bbox": bbox,
            "area": int(defect_mask.sum()),
            "center": (cx, cy),
        }

    def _draw_circle(
        self,
        arr: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        value: int = 1,
    ):
        """Draw a filled circle."""
        y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        arr[dist <= radius] = value

    def _draw_streak(
        self,
        arr: np.ndarray,
        cx: int,
        cy: int,
        length: int,
        width: int,
        angle: float,
        value: int = 1,
    ):
        """Draw an elongated streak."""
        angle_rad = np.radians(angle)
        for t in np.linspace(-length / 2, length / 2, length):
            px = int(cx + t * np.cos(angle_rad))
            py = int(cy + t * np.sin(angle_rad))
            if 0 <= px < arr.shape[1] and 0 <= py < arr.shape[0]:
                self._draw_circle(arr, px, py, width, value)

    def _draw_line(
        self,
        arr: np.ndarray,
        cx: int,
        cy: int,
        length: int,
        width: int,
        angle: float,
        value: int = 1,
    ):
        """Draw a thin line."""
        self._draw_streak(arr, cx, cy, length, 1, angle, value)

    def _draw_ring(
        self,
        arr: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        thickness: int,
        value: int = 1,
    ):
        """Draw a ring/circle outline."""
        y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        arr[(dist >= radius - thickness) & (dist <= radius + thickness)] = value

    def _draw_blob(
        self,
        arr: np.ndarray,
        cx: int,
        cy: int,
        size: int,
        value: int = 1,
    ):
        """Draw an irregular blob."""
        # Multiple overlapping circles
        num_circles = self.rng.integers(3, 8)
        for _ in range(num_circles):
            offset_x = self.rng.integers(-size // 3, size // 3)
            offset_y = self.rng.integers(-size // 3, size // 3)
            radius = self.rng.integers(size // 4, size // 2)
            self._draw_circle(arr, cx + offset_x, cy + offset_y, radius, value)

    def generate_dataset(
        self,
        split: str = "train",
    ) -> Iterator[dict[str, Any]]:
        """Generate full dataset for specified split."""
        if split == "train":
            n_samples = self.config.train_samples
            seed_offset = 900000
        elif split == "val":
            n_samples = self.config.val_samples
            seed_offset = 1000000
        else:
            n_samples = self.config.test_samples
            seed_offset = 1100000

        self.reset(self.config.seed + seed_offset)

        for _ in range(n_samples):
            yield self.generate_sample()


# =============================================================================
# Recipe Recommendation Data Generator
# =============================================================================


class RecipeDataGenerator(BaseDataGenerator):
    """Generate synthetic recipe data for recommendation training.

    Creates realistic recipe data with:
    - User interaction histories
    - Recipe metadata
    - Rating patterns
    - Collaborative filtering signals
    """

    def __init__(self, config: SyntheticDataConfig):
        """Initialize recipe data generator."""
        super().__init__(config)
        self.recipes = self._generate_recipes(1000)
        self.users = self._generate_users(500)

    def _generate_recipes(self, n_recipes: int) -> list[dict]:
        """Generate a set of recipes."""
        recipes = []
        for i in range(n_recipes):
            recipe = {
                "id": str(uuid4()),
                "name": f"Recipe_{i}",
                "paper_type": self.rng.choice(list(PaperType)).value,
                "metal_ratio": float(self.rng.uniform(0, 1)),
                "contrast_amount": float(self.rng.uniform(0, 0.3)),
                "exposure_time": float(self.rng.uniform(60, 600)),
                "developer": self.rng.choice(["potassium_oxalate", "ammonium_citrate"]),
                "tone": self.rng.choice(["warm", "neutral", "cool"]),
                "category": self.rng.choice(["portrait", "landscape", "abstract"]),
                "difficulty": self.rng.choice(["beginner", "intermediate", "advanced"]),
                "popularity": float(self.rng.beta(2, 5)),  # Skewed towards lower
            }
            recipes.append(recipe)
        return recipes

    def _generate_users(self, n_users: int) -> list[dict]:
        """Generate user profiles."""
        users = []
        for i in range(n_users):
            user = {
                "id": f"user_{i}",
                "preferred_tone": self.rng.choice(["warm", "neutral", "cool"]),
                "skill_level": self.rng.choice(["beginner", "intermediate", "advanced"]),
                "activity_level": float(self.rng.beta(1, 3)),
            }
            users.append(user)
        return users

    def generate_sample(self) -> dict[str, Any]:
        """Generate a single recommendation training sample.

        Returns:
            Dictionary containing:
            - user: User profile
            - positive_recipes: Recipes the user liked
            - negative_recipes: Recipes the user didn't like
            - target_recipe: Recipe to predict rating for
            - target_rating: True rating
        """
        self._sample_count += 1

        # Select random user
        user = self.rng.choice(self.users)

        # Generate interaction history
        num_interactions = self.rng.integers(5, 50)
        history_recipes = self.rng.choice(
            self.recipes, size=num_interactions, replace=False
        ).tolist()

        positive = []
        negative = []

        for recipe in history_recipes:
            # Rating based on preference match + noise
            base_rating = 3.0

            if recipe["tone"] == user["preferred_tone"]:
                base_rating += 1.0

            if recipe["difficulty"] == user["skill_level"]:
                base_rating += 0.5

            # Add noise to prevent exact matching
            rating = base_rating + self.rng.normal(0, 0.5)
            rating = np.clip(rating, 1, 5)

            if rating >= 3.5:
                positive.append({"recipe": recipe, "rating": float(rating)})
            else:
                negative.append({"recipe": recipe, "rating": float(rating)})

        # Select target recipe (not in history)
        available = [r for r in self.recipes if r not in history_recipes]
        target = self.rng.choice(available)

        # Generate target rating
        target_rating = 3.0
        if target["tone"] == user["preferred_tone"]:
            target_rating += 1.0
        if target["difficulty"] == user["skill_level"]:
            target_rating += 0.5
        target_rating += self.rng.normal(0, 0.5)
        target_rating = float(np.clip(target_rating, 1, 5))

        return {
            "user": user,
            "positive_recipes": positive,
            "negative_recipes": negative,
            "target_recipe": target,
            "target_rating": target_rating,
            "sample_id": str(uuid4()),
        }

    def generate_dataset(
        self,
        split: str = "train",
    ) -> Iterator[dict[str, Any]]:
        """Generate full dataset for specified split."""
        if split == "train":
            n_samples = self.config.train_samples
            seed_offset = 1200000
        elif split == "val":
            n_samples = self.config.val_samples
            seed_offset = 1300000
        else:
            n_samples = self.config.test_samples
            seed_offset = 1400000

        self.reset(self.config.seed + seed_offset)

        for _ in range(n_samples):
            yield self.generate_sample()


# =============================================================================
# Utility Functions
# =============================================================================


def create_generators(
    config: Optional[SyntheticDataConfig] = None,
) -> dict[str, BaseDataGenerator]:
    """Create all data generators with shared configuration.

    Args:
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Dictionary of generator name -> generator instance
    """
    if config is None:
        config = SyntheticDataConfig()

    return {
        "detection": DetectionDataGenerator(config),
        "curve": CurveDataGenerator(config),
        "exposure": ExposureDataGenerator(config),
        "defect": DefectDataGenerator(config),
        "recipe": RecipeDataGenerator(config),
    }


def save_dataset(
    generator: BaseDataGenerator,
    output_dir: Path,
    split: str = "train",
    format: str = "jsonl",
):
    """Save generated dataset to disk.

    Args:
        generator: Data generator instance
        output_dir: Output directory
        split: Dataset split (train, val, test)
        format: Output format (jsonl, npz)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{split}.{format}"

    logger.info(f"Generating {split} dataset to {output_file}")

    if format == "jsonl":
        with open(output_file, "w") as f:
            for sample in generator.generate_dataset(split):
                # Convert numpy arrays to lists for JSON
                serializable = {}
                for k, v in sample.items():
                    if isinstance(v, np.ndarray):
                        serializable[k] = v.tolist()
                    else:
                        serializable[k] = v
                f.write(json.dumps(serializable) + "\n")
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved {split} dataset to {output_file}")
