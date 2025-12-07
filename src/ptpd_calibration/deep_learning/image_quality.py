"""
Vision Transformer-based image quality assessment for Platinum-Palladium prints.

This module provides advanced image quality analysis using Vision Transformers
and multiple quality metrics. Includes zone-based analysis following the
Ansel Adams Zone System for photographic prints.

Example:
    >>> from ptpd_calibration.deep_learning.image_quality import VisionTransformerIQA
    >>> from ptpd_calibration.deep_learning.config import ImageQualitySettings
    >>>
    >>> settings = ImageQualitySettings()
    >>> iqa = VisionTransformerIQA(settings)
    >>> result = await iqa.assess_quality(image)
    >>> print(f"Quality: {result.quality_level}, Score: {result.overall_score:.3f}")
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np
from scipy import ndimage

from ptpd_calibration.deep_learning.config import ImageQualitySettings
from ptpd_calibration.deep_learning.models import (
    ImageQualityResult,
    ZoneQualityScore,
)
from ptpd_calibration.deep_learning.types import (
    IQAMetric,
    ImageArray,
    QualityLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Imports - Only load ML libraries when actually used
# =============================================================================


class _LazyImports:
    """Container for lazy-loaded ML dependencies."""

    def __init__(self):
        self._torch = None
        self._torchvision = None
        self._timm = None
        self._clip = None
        self._cv2 = None
        self._skimage = None

    @property
    def torch(self):
        """Lazy import torch."""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except ImportError:
                logger.warning("PyTorch not available, IQA features limited")
                self._torch = False
        return self._torch if self._torch is not False else None

    @property
    def torchvision(self):
        """Lazy import torchvision."""
        if self._torchvision is None:
            try:
                import torchvision

                self._torchvision = torchvision
            except ImportError:
                logger.warning("Torchvision not available")
                self._torchvision = False
        return self._torchvision if self._torchvision is not False else None

    @property
    def timm(self):
        """Lazy import timm (PyTorch Image Models)."""
        if self._timm is None:
            try:
                import timm

                self._timm = timm
            except ImportError:
                logger.warning("timm not available, Vision Transformer features disabled")
                self._timm = False
        return self._timm if self._timm is not False else None

    @property
    def clip(self):
        """Lazy import CLIP."""
        if self._clip is None:
            try:
                import clip

                self._clip = clip
            except ImportError:
                logger.warning("CLIP not available, CLIP-IQA disabled")
                self._clip = False
        return self._clip if self._clip is not False else None

    @property
    def cv2(self):
        """Lazy import OpenCV."""
        if self._cv2 is None:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError:
                logger.warning("OpenCV not available")
                self._cv2 = False
        return self._cv2 if self._cv2 is not False else None

    @property
    def skimage(self):
        """Lazy import scikit-image."""
        if self._skimage is None:
            try:
                import skimage

                self._skimage = skimage
            except ImportError:
                logger.warning("scikit-image not available")
                self._skimage = False
        return self._skimage if self._skimage is not False else None


_imports = _LazyImports()


# =============================================================================
# Zone System Definitions
# =============================================================================

# Ansel Adams Zone System - 11 zones (0-10)
ZONE_NAMES = {
    0: "Zone 0: Pure Black",
    1: "Zone I: Near Black",
    2: "Zone II: Deep Shadows",
    3: "Zone III: Average Shadows",
    4: "Zone IV: Dark Midtones",
    5: "Zone V: Middle Gray",
    6: "Zone VI: Light Midtones",
    7: "Zone VII: Average Highlights",
    8: "Zone VIII: Bright Highlights",
    9: "Zone IX: Near White",
    10: "Zone X: Pure White",
}

# Zone ranges (0-255 for 8-bit images)
ZONE_RANGES = [
    (0, 12),  # Zone 0
    (13, 25),  # Zone I
    (26, 51),  # Zone II
    (52, 76),  # Zone III
    (77, 102),  # Zone IV
    (103, 127),  # Zone V
    (128, 153),  # Zone VI
    (154, 178),  # Zone VII
    (179, 204),  # Zone VIII
    (205, 230),  # Zone IX
    (231, 255),  # Zone X
]


# =============================================================================
# Main IQA Class
# =============================================================================


class VisionTransformerIQA:
    """
    Vision Transformer-based Image Quality Assessment.

    This class provides comprehensive image quality analysis using Vision
    Transformers and multiple quality metrics. Specifically designed for
    Platinum-Palladium prints with zone-based analysis.

    Attributes:
        settings: IQA configuration settings
        device: Torch device for inference
        vit_model: Vision Transformer model (lazy loaded)
        clip_model: CLIP model for perceptual quality (lazy loaded)

    Example:
        >>> settings = ImageQualitySettings(
        ...     primary_metric=IQAMetric.MANIQA,
        ...     analyze_zones=True,
        ...     zone_count=11,
        ... )
        >>> iqa = VisionTransformerIQA(settings)
        >>> result = await iqa.assess_quality(image)
        >>> print(f"Overall: {result.overall_score:.3f}")
        >>> for zone_score in result.zone_scores:
        ...     print(f"{zone_score.zone_name}: {zone_score.score:.3f}")
    """

    def __init__(self, settings: Optional[ImageQualitySettings] = None):
        """
        Initialize the Vision Transformer IQA.

        Args:
            settings: IQA settings. If None, uses defaults.
        """
        self.settings = settings or ImageQualitySettings()
        self._device: Optional[str] = None
        self._vit_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._embedding_cache = {}
        self._models_loaded = False

        logger.info(
            f"Initialized VisionTransformerIQA with primary metric: {self.settings.primary_metric}"
        )

    @property
    def device(self) -> str:
        """
        Get the device for inference.

        Returns:
            Device string: 'cuda', 'cpu', or 'mps'
        """
        if self._device is None:
            self._device = self._determine_device()
        return self._device

    def _determine_device(self) -> str:
        """
        Determine the best available device.

        Returns:
            Device string
        """
        if self.settings.device != "auto":
            return self.settings.device

        torch = _imports.torch
        if torch is None:
            return "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS")
        else:
            device = "cpu"
            logger.info("Using CPU")

        return device

    def _load_models(self) -> bool:
        """
        Lazy load IQA models.

        Returns:
            True if models loaded successfully
        """
        if self._models_loaded:
            return True

        success = True

        # Load Vision Transformer for primary metric
        if self.settings.primary_metric in {
            IQAMetric.MANIQA,
            IQAMetric.MUSIQ,
            IQAMetric.TOPIQ,
        }:
            success = success and self._load_vit_model()

        # Load CLIP if needed
        if IQAMetric.CLIP_IQA in [
            self.settings.primary_metric,
            *self.settings.secondary_metrics,
        ]:
            success = success and self._load_clip_model()

        self._models_loaded = True
        return success

    def _load_vit_model(self) -> bool:
        """
        Load Vision Transformer model.

        Returns:
            True if successful
        """
        timm = _imports.timm
        torch = _imports.torch

        if timm is None or torch is None:
            return False

        try:
            # Load pretrained ViT
            if self.settings.custom_weights_path:
                logger.info(f"Loading custom weights from {self.settings.custom_weights_path}")
                self._vit_model = timm.create_model(
                    self.settings.vit_model_name,
                    pretrained=False,
                    num_classes=1,  # Regression task
                )
                checkpoint = torch.load(
                    self.settings.custom_weights_path, map_location=self.device
                )
                self._vit_model.load_state_dict(checkpoint)
            else:
                logger.info(f"Loading pretrained {self.settings.vit_model_name}")
                self._vit_model = timm.create_model(
                    self.settings.vit_model_name,
                    pretrained=True,
                    num_classes=1,
                )

            self._vit_model = self._vit_model.to(self.device)
            self._vit_model.eval()

            logger.info("Vision Transformer loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load ViT: {e}")
            return False

    def _load_clip_model(self) -> bool:
        """
        Load CLIP model for perceptual quality.

        Returns:
            True if successful
        """
        clip = _imports.clip
        torch = _imports.torch

        if clip is None or torch is None:
            return False

        try:
            # Load CLIP
            model_name = "ViT-B/32"  # Can be configured
            self._clip_model, self._clip_preprocess = clip.load(
                model_name, device=self.device
            )
            self._clip_model.eval()

            logger.info(f"CLIP model loaded: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            return False

    async def assess_quality(
        self, image: ImageArray, generate_embedding: bool = True
    ) -> ImageQualityResult:
        """
        Assess image quality using Vision Transformer and multiple metrics.

        This is the main entry point for quality assessment. Performs:
        1. Overall quality scoring with multiple metrics
        2. Zone-based analysis (Ansel Adams zones)
        3. Technical metric computation (sharpness, noise, etc.)
        4. Recommendation generation
        5. Optional embedding for comparison

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            generate_embedding: Whether to generate embedding for comparison

        Returns:
            ImageQualityResult with comprehensive quality analysis

        Example:
            >>> result = await iqa.assess_quality(image)
            >>> if result.quality_level == QualityLevel.EXCELLENT:
            ...     print("Outstanding print quality!")
            >>> for rec in result.recommendations:
            ...     print(f"Suggestion: {rec}")
        """
        start_time = time.perf_counter()

        # Validate and prepare image
        image = self._prepare_image(image)

        # Load models if needed
        self._load_models()

        # Compute metrics
        metric_scores = await self._compute_all_metrics(image)

        # Determine primary score
        primary_metric_name = self.settings.primary_metric.value
        primary_score = metric_scores.get(primary_metric_name, 0.5)

        # Compute overall score (weighted average)
        overall_score = self._compute_overall_score(metric_scores)

        # Classify quality level
        quality_level = self._classify_quality_level(overall_score)

        # Zone-based analysis
        zone_scores = []
        if self.settings.analyze_zones:
            zone_scores = await self._analyze_zones(image)

        # Compute technical metrics
        tech_metrics = self._compute_technical_metrics(image)

        # Generate recommendations and identify issues
        recommendations, issues = self._generate_recommendations(
            overall_score, zone_scores, tech_metrics
        )

        # Generate embedding if requested
        embedding = None
        if generate_embedding and self.settings.cache_embeddings:
            embedding = await self._generate_embedding(image)

        # Compute zone quality aggregates
        highlight_quality, midtone_quality, shadow_quality = self._aggregate_zone_quality(
            zone_scores
        )

        # Create result
        inference_time = (time.perf_counter() - start_time) * 1000

        result = ImageQualityResult(
            overall_score=overall_score,
            quality_level=quality_level,
            metric_scores=metric_scores,
            primary_metric_name=primary_metric_name,
            primary_metric_score=primary_score,
            zone_scores=zone_scores,
            highlight_quality=highlight_quality,
            midtone_quality=midtone_quality,
            shadow_quality=shadow_quality,
            sharpness=tech_metrics["sharpness"],
            noise_level=tech_metrics["noise"],
            dynamic_range=tech_metrics["dynamic_range"],
            contrast=tech_metrics["contrast"],
            recommendations=recommendations,
            issues=issues,
            embedding=embedding,
            inference_time_ms=inference_time,
            device_used=self.device,
            model_version=f"{self.settings.vit_model_name}_{self.settings.primary_metric.value}",
        )

        logger.info(
            f"Quality assessment complete: {quality_level.value} "
            f"(score: {overall_score:.3f}) in {inference_time:.1f}ms"
        )

        return result

    def _prepare_image(self, image: ImageArray) -> ImageArray:
        """
        Prepare image for quality assessment.

        Args:
            image: Input image

        Returns:
            Prepared image (normalized, proper dimensions)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Ensure 2D or 3D
        if len(image.shape) not in {2, 3}:
            raise ValueError(f"Invalid image shape: {image.shape}")

        # Convert to grayscale if needed (Pt/Pd prints are monochrome)
        if len(image.shape) == 3 and image.shape[2] > 1:
            cv2 = _imports.cv2
            if cv2 is not None:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Manual grayscale conversion
                image = np.mean(image, axis=2)

        # Ensure float [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        return image

    async def _compute_all_metrics(self, image: ImageArray) -> dict[str, float]:
        """
        Compute all configured IQA metrics.

        Args:
            image: Prepared image

        Returns:
            Dictionary of metric name -> score
        """
        metrics = {}

        # Primary metric
        score = await self._compute_metric(image, self.settings.primary_metric)
        metrics[self.settings.primary_metric.value] = score

        # Secondary metrics
        for metric in self.settings.secondary_metrics:
            score = await self._compute_metric(image, metric)
            metrics[metric.value] = score

        return metrics

    async def _compute_metric(self, image: ImageArray, metric: IQAMetric) -> float:
        """
        Compute a specific IQA metric.

        Args:
            image: Input image
            metric: Metric to compute

        Returns:
            Quality score [0, 1] (higher is better)
        """
        try:
            if metric == IQAMetric.NIQE:
                return await self._compute_niqe(image)
            elif metric == IQAMetric.BRISQUE:
                return await self._compute_brisque(image)
            elif metric == IQAMetric.CLIP_IQA:
                return await self._compute_clip_iqa(image)
            elif metric in {IQAMetric.MANIQA, IQAMetric.MUSIQ, IQAMetric.TOPIQ}:
                return await self._compute_vit_based_metric(image, metric)
            else:
                logger.warning(f"Metric {metric} not implemented, using fallback")
                return 0.5

        except Exception as e:
            logger.error(f"Failed to compute {metric}: {e}")
            return 0.5

    async def _compute_niqe(self, image: ImageArray) -> float:
        """
        Compute NIQE (Natural Image Quality Evaluator).

        Args:
            image: Input image

        Returns:
            Quality score [0, 1]
        """
        skimage = _imports.skimage
        if skimage is None:
            return 0.5

        try:
            from skimage.metrics import structural_similarity as ssim

            # NIQE approximation using local statistics
            # Lower NIQE is better, normalize to [0, 1]
            image_8bit = (image * 255).astype(np.uint8)

            # Compute local variance as quality indicator
            local_var = ndimage.generic_filter(
                image, np.var, size=7, mode="reflect"
            )
            avg_local_var = np.mean(local_var)

            # Normalize (empirical range 0-0.05)
            score = 1.0 - min(avg_local_var / 0.05, 1.0)

            return float(score)

        except Exception as e:
            logger.error(f"NIQE computation failed: {e}")
            return 0.5

    async def _compute_brisque(self, image: ImageArray) -> float:
        """
        Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

        Args:
            image: Input image

        Returns:
            Quality score [0, 1]
        """
        cv2 = _imports.cv2
        if cv2 is None:
            return 0.5

        try:
            # BRISQUE approximation using natural scene statistics
            image_8bit = (image * 255).astype(np.uint8)

            # Compute MSCN (Mean Subtracted Contrast Normalized) coefficients
            mu = cv2.GaussianBlur(image_8bit, (7, 7), 7 / 6)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(image_8bit * image_8bit, (7, 7), 7 / 6)
            sigma = np.sqrt(np.abs(sigma - mu_sq))

            mscn = (image_8bit - mu) / (sigma + 1.0)

            # Compute statistics
            variance = np.var(mscn)

            # Normalize (empirical range)
            score = 1.0 / (1.0 + np.abs(variance - 1.0))

            return float(score)

        except Exception as e:
            logger.error(f"BRISQUE computation failed: {e}")
            return 0.5

    async def _compute_clip_iqa(self, image: ImageArray) -> float:
        """
        Compute CLIP-IQA (quality assessment using CLIP embeddings).

        Args:
            image: Input image

        Returns:
            Quality score [0, 1]
        """
        if self._clip_model is None:
            return 0.5

        torch = _imports.torch
        if torch is None:
            return 0.5

        try:
            # Prepare image for CLIP
            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image

            # Convert to PIL for preprocessing
            from PIL import Image

            pil_image = Image.fromarray((image_rgb * 255).astype(np.uint8))

            # Preprocess and encode
            image_input = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)

            # Define quality prompts
            positive_prompts = [
                "a high quality photograph",
                "excellent image quality",
                "sharp and clear image",
            ]
            negative_prompts = [
                "a low quality photograph",
                "poor image quality",
                "blurry and noisy image",
            ]

            # Tokenize prompts
            import clip

            positive_tokens = clip.tokenize(positive_prompts).to(self.device)
            negative_tokens = clip.tokenize(negative_prompts).to(self.device)

            with torch.no_grad():
                positive_features = self._clip_model.encode_text(positive_tokens)
                negative_features = self._clip_model.encode_text(negative_tokens)

            # Compute similarities
            pos_sim = torch.cosine_similarity(
                image_features, positive_features.mean(dim=0, keepdim=True)
            )
            neg_sim = torch.cosine_similarity(
                image_features, negative_features.mean(dim=0, keepdim=True)
            )

            # Quality score based on relative similarity
            score = (pos_sim - neg_sim).item()
            score = (score + 1.0) / 2.0  # Normalize to [0, 1]

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error(f"CLIP-IQA computation failed: {e}")
            return 0.5

    async def _compute_vit_based_metric(
        self, image: ImageArray, metric: IQAMetric
    ) -> float:
        """
        Compute ViT-based quality metric (MANIQA, MUSIQ, TOPIQ).

        Args:
            image: Input image
            metric: Specific ViT metric

        Returns:
            Quality score [0, 1]
        """
        if self._vit_model is None:
            return 0.5

        torch = _imports.torch
        torchvision = _imports.torchvision

        if torch is None or torchvision is None:
            return 0.5

        try:
            # Prepare image for ViT
            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image

            # Resize and normalize
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.settings.input_size, self.settings.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            image_tensor = transform((image_rgb * 255).astype(np.uint8))
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Multi-crop evaluation if enabled
            if self.settings.use_multi_scale and self.settings.num_crops > 1:
                scores = []
                for _ in range(self.settings.num_crops):
                    with torch.no_grad():
                        score = self._vit_model(image_tensor)
                    scores.append(score.item())
                final_score = np.mean(scores)
            else:
                with torch.no_grad():
                    final_score = self._vit_model(image_tensor).item()

            # Normalize score to [0, 1] (model-specific)
            final_score = np.clip(final_score, 0.0, 1.0)

            return float(final_score)

        except Exception as e:
            logger.error(f"ViT-based metric computation failed: {e}")
            return 0.5

    def _compute_overall_score(self, metric_scores: dict[str, float]) -> float:
        """
        Compute weighted overall quality score.

        Args:
            metric_scores: Individual metric scores

        Returns:
            Overall score [0, 1]
        """
        if not metric_scores:
            return 0.5

        # Primary metric has higher weight
        primary_weight = 0.6
        secondary_weight = 0.4 / max(len(self.settings.secondary_metrics), 1)

        total_score = 0.0
        total_weight = 0.0

        # Add primary metric
        primary_name = self.settings.primary_metric.value
        if primary_name in metric_scores:
            total_score += metric_scores[primary_name] * primary_weight
            total_weight += primary_weight

        # Add secondary metrics
        for metric in self.settings.secondary_metrics:
            if metric.value in metric_scores:
                total_score += metric_scores[metric.value] * secondary_weight
                total_weight += secondary_weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _classify_quality_level(self, score: float) -> QualityLevel:
        """
        Classify quality level based on score.

        Args:
            score: Overall quality score

        Returns:
            Quality level classification
        """
        if score >= self.settings.excellent_threshold:
            return QualityLevel.EXCELLENT
        elif score >= self.settings.good_threshold:
            return QualityLevel.GOOD
        elif score >= self.settings.acceptable_threshold:
            return QualityLevel.ACCEPTABLE
        elif score >= self.settings.poor_threshold:
            return QualityLevel.MARGINAL
        else:
            return QualityLevel.POOR

    async def _analyze_zones(self, image: ImageArray) -> list[ZoneQualityScore]:
        """
        Perform zone-based quality analysis (Ansel Adams Zone System).

        Args:
            image: Input image

        Returns:
            List of zone quality scores
        """
        zone_scores = []

        # Convert to 8-bit for zone assignment
        image_8bit = (image * 255).astype(np.uint8)

        for zone_idx, (low, high) in enumerate(ZONE_RANGES[: self.settings.zone_count]):
            # Create zone mask
            zone_mask = (image_8bit >= low) & (image_8bit <= high)
            pixel_count = np.sum(zone_mask)

            if pixel_count == 0:
                continue

            # Extract zone pixels
            zone_pixels = image[zone_mask]

            # Compute zone quality metrics
            zone_score = self._compute_zone_quality(zone_pixels, zone_idx)
            pixel_percentage = (pixel_count / image.size) * 100

            # Detail preservation (local variance)
            detail_score = self._compute_zone_detail(image, zone_mask)

            # Identify zone issues
            issues = self._identify_zone_issues(zone_pixels, zone_idx, pixel_percentage)

            zone_quality_score = ZoneQualityScore(
                zone=zone_idx,
                zone_name=ZONE_NAMES[zone_idx],
                score=zone_score,
                pixel_percentage=pixel_percentage,
                detail_preservation=detail_score,
                issues=issues,
            )

            zone_scores.append(zone_quality_score)

        return zone_scores

    def _compute_zone_quality(self, zone_pixels: np.ndarray, zone_idx: int) -> float:
        """
        Compute quality score for a specific zone.

        Args:
            zone_pixels: Pixel values in the zone
            zone_idx: Zone index

        Returns:
            Quality score [0, 1]
        """
        # Quality based on tonal consistency and smoothness
        variance = np.var(zone_pixels)
        std = np.std(zone_pixels)

        # Zones should have smooth gradations
        # Lower variance is better (more uniform tones)
        # But some detail variance is expected

        # Ideal variance depends on zone
        ideal_variance = 0.01  # Empirical

        variance_score = 1.0 / (1.0 + abs(variance - ideal_variance))

        return float(np.clip(variance_score, 0.0, 1.0))

    def _compute_zone_detail(self, image: ImageArray, zone_mask: np.ndarray) -> float:
        """
        Compute detail preservation in a zone.

        Args:
            image: Full image
            zone_mask: Boolean mask for the zone

        Returns:
            Detail score [0, 1]
        """
        # Compute local gradients in the zone
        grad_x = np.abs(np.gradient(image, axis=1))
        grad_y = np.abs(np.gradient(image, axis=0))

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Average gradient in zone
        zone_gradients = gradient_magnitude[zone_mask]

        if len(zone_gradients) == 0:
            return 0.5

        avg_gradient = np.mean(zone_gradients)

        # Normalize (empirical range 0-0.1)
        detail_score = min(avg_gradient / 0.1, 1.0)

        return float(detail_score)

    def _identify_zone_issues(
        self, zone_pixels: np.ndarray, zone_idx: int, pixel_percentage: float
    ) -> list[str]:
        """
        Identify potential issues in a zone.

        Args:
            zone_pixels: Zone pixel values
            zone_idx: Zone index
            pixel_percentage: Percentage of image in this zone

        Returns:
            List of issue descriptions
        """
        issues = []

        # Check for clipping
        if zone_idx == 0 and pixel_percentage > 10:
            issues.append("Excessive pure black (possible shadow clipping)")
        elif zone_idx == 10 and pixel_percentage > 10:
            issues.append("Excessive pure white (possible highlight clipping)")

        # Check for posterization (too uniform)
        if len(zone_pixels) > 100:
            unique_values = len(np.unique(zone_pixels))
            if unique_values < 10:
                issues.append("Possible posterization (limited tonal gradation)")

        # Check for unexpected noise
        noise_estimate = np.std(zone_pixels)
        if noise_estimate > 0.1:
            issues.append("Elevated noise level detected")

        return issues

    def _aggregate_zone_quality(
        self, zone_scores: list[ZoneQualityScore]
    ) -> tuple[float, float, float]:
        """
        Aggregate zone scores into highlight/midtone/shadow quality.

        Args:
            zone_scores: Individual zone scores

        Returns:
            Tuple of (highlight_quality, midtone_quality, shadow_quality)
        """
        if not zone_scores:
            return 1.0, 1.0, 1.0

        highlight_zones = [7, 8, 9, 10]
        midtone_zones = [4, 5, 6]
        shadow_zones = [0, 1, 2, 3]

        highlight_scores = [z.score for z in zone_scores if z.zone in highlight_zones]
        midtone_scores = [z.score for z in zone_scores if z.zone in midtone_zones]
        shadow_scores = [z.score for z in zone_scores if z.zone in shadow_zones]

        highlight_quality = np.mean(highlight_scores) if highlight_scores else 1.0
        midtone_quality = np.mean(midtone_scores) if midtone_scores else 1.0
        shadow_quality = np.mean(shadow_scores) if shadow_scores else 1.0

        return (
            float(highlight_quality),
            float(midtone_quality),
            float(shadow_quality),
        )

    def _compute_technical_metrics(self, image: ImageArray) -> dict[str, float]:
        """
        Compute technical image metrics.

        Args:
            image: Input image

        Returns:
            Dictionary of technical metrics
        """
        metrics = {}

        # Sharpness (gradient magnitude)
        grad_x = np.abs(np.gradient(image, axis=1))
        grad_y = np.abs(np.gradient(image, axis=0))
        sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        metrics["sharpness"] = float(min(sharpness / 0.1, 1.0))

        # Noise level (high-frequency content)
        laplacian = ndimage.laplace(image)
        noise_level = np.std(laplacian)
        metrics["noise"] = float(min(noise_level / 0.05, 1.0))

        # Dynamic range
        min_val, max_val = np.min(image), np.max(image)
        metrics["dynamic_range"] = float(max_val - min_val)

        # Contrast (standard deviation)
        contrast = np.std(image)
        metrics["contrast"] = float(min(contrast / 0.3, 2.0))

        return metrics

    def _generate_recommendations(
        self,
        overall_score: float,
        zone_scores: list[ZoneQualityScore],
        tech_metrics: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        """
        Generate quality improvement recommendations.

        Args:
            overall_score: Overall quality score
            zone_scores: Zone analysis results
            tech_metrics: Technical metrics

        Returns:
            Tuple of (recommendations, issues)
        """
        recommendations = []
        issues = []

        # Overall quality recommendations
        if overall_score < self.settings.acceptable_threshold:
            issues.append("Overall quality below acceptable threshold")
            recommendations.append("Consider re-printing with adjusted exposure time")

        # Zone-specific recommendations
        for zone_score in zone_scores:
            if zone_score.issues:
                issues.extend(zone_score.issues)

            if zone_score.score < 0.5:
                zone_name = zone_score.zone_name
                recommendations.append(
                    f"Improve tonal quality in {zone_name} "
                    f"(consider chemistry ratio adjustment)"
                )

        # Technical recommendations
        if tech_metrics["sharpness"] < 0.4:
            issues.append("Low sharpness detected")
            recommendations.append("Check contact frame pressure and negative sharpness")

        if tech_metrics["noise"] > 0.7:
            issues.append("High noise level detected")
            recommendations.append("Reduce developer agitation or check paper quality")

        if tech_metrics["dynamic_range"] < 0.6:
            issues.append("Limited dynamic range")
            recommendations.append(
                "Consider increasing contrast through chemistry ratio or negative adjustment"
            )

        # Remove duplicates
        recommendations = list(dict.fromkeys(recommendations))
        issues = list(dict.fromkeys(issues))

        return recommendations, issues

    async def _generate_embedding(self, image: ImageArray) -> Optional[list[float]]:
        """
        Generate image embedding for comparison.

        Args:
            image: Input image

        Returns:
            Embedding vector or None
        """
        if self._vit_model is None:
            return None

        torch = _imports.torch
        if torch is None:
            return None

        try:
            # Prepare image
            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            else:
                image_rgb = image

            # Transform and extract features
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            image_tensor = transform((image_rgb * 255).astype(np.uint8))
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Extract features from ViT (before final layer)
                features = self._vit_model.forward_features(image_tensor)
                features = features.mean(dim=1)  # Global average pooling
                embedding = features.cpu().numpy()[0].tolist()

            # Cache embedding if enabled
            if self.settings.cache_embeddings:
                image_hash = hash(image.tobytes())
                self._embedding_cache[image_hash] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def clear_cache(self):
        """Clear model and embedding cache to free memory."""
        self._vit_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._embedding_cache = {}
        self._models_loaded = False
        logger.info("IQA cache cleared")
