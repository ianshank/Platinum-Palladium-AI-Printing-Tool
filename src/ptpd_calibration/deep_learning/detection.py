"""
Deep learning-based step tablet detection using YOLOv8 + SAM.

This module provides advanced detection using state-of-the-art object detection
and instance segmentation models. Falls back gracefully to classical CV when
deep learning dependencies are unavailable.

Example:
    >>> from ptpd_calibration.deep_learning.detection import DeepTabletDetector
    >>> from ptpd_calibration.deep_learning.config import DetectionModelSettings
    >>>
    >>> settings = DetectionModelSettings()
    >>> detector = DeepTabletDetector(settings)
    >>> result = await detector.detect(image)
    >>> print(f"Detected {result.num_patches} patches with confidence {result.tablet_confidence}")
"""

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from ptpd_calibration.deep_learning.config import DetectionModelSettings
from ptpd_calibration.deep_learning.models import (
    DeepDetectionResult,
    DetectedPatch,
)
from ptpd_calibration.deep_learning.types import (
    BoundingBox,
    DetectionBackend,
    DetectionConfidence,
    ImageArray,
    SegmentationBackend,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Imports - Only load ML libraries when actually used
# =============================================================================


class _LazyImports:
    """Container for lazy-loaded ML dependencies."""

    def __init__(self):
        self._torch = None
        self._ultralytics = None
        self._sam = None
        self._cv2 = None

    @property
    def torch(self):
        """Lazy import torch."""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except ImportError:
                logger.warning("PyTorch not available, deep learning features disabled")
                self._torch = False
        return self._torch if self._torch is not False else None

    @property
    def ultralytics(self):
        """Lazy import ultralytics (YOLOv8)."""
        if self._ultralytics is None:
            try:
                import ultralytics

                self._ultralytics = ultralytics
            except ImportError:
                logger.warning("Ultralytics not available, YOLO detection disabled")
                self._ultralytics = False
        return self._ultralytics if self._ultralytics is not False else None

    @property
    def sam(self):
        """Lazy import segment-anything."""
        if self._sam is None:
            try:
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

                self._sam = {"registry": sam_model_registry, "generator": SamAutomaticMaskGenerator}
            except ImportError:
                logger.warning("Segment Anything not available, SAM segmentation disabled")
                self._sam = False
        return self._sam if self._sam is not False else None

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


_imports = _LazyImports()


# =============================================================================
# Main Detector Class
# =============================================================================


class DeepTabletDetector:
    """
    Deep learning-based step tablet detector using YOLOv8 + SAM.

    This class combines object detection (YOLOv8) for tablet localization with
    instance segmentation (SAM) for precise patch extraction. Automatically
    falls back to classical computer vision when deep learning is unavailable.

    Attributes:
        settings: Detection model configuration settings
        device: Torch device for inference (cuda/cpu/mps)
        yolo_model: YOLOv8 detection model (lazy loaded)
        sam_model: SAM segmentation model (lazy loaded)

    Example:
        >>> settings = DetectionModelSettings(
        ...     detection_backend=DetectionBackend.YOLOV8,
        ...     segmentation_backend=SegmentationBackend.SAM,
        ...     yolo_confidence_threshold=0.3,
        ... )
        >>> detector = DeepTabletDetector(settings)
        >>> result = await detector.detect(image)
    """

    def __init__(self, settings: DetectionModelSettings | None = None):
        """
        Initialize the deep tablet detector.

        Args:
            settings: Detection model settings. If None, uses defaults.
        """
        self.settings = settings or DetectionModelSettings()
        self._device: str | None = None
        self._yolo_model = None
        self._sam_model = None
        self._sam_generator = None
        self._models_loaded = False

        logger.info(
            f"Initialized DeepTabletDetector with backend: {self.settings.detection_backend}, "
            f"segmentation: {self.settings.segmentation_backend}"
        )

    @property
    def device(self) -> str:
        """
        Get the device for inference.

        Returns:
            Device string: 'cuda', 'cuda:0', 'cpu', or 'mps'
        """
        if self._device is None:
            self._device = self._determine_device()
        return self._device

    def _determine_device(self) -> str:
        """
        Determine the best available device based on settings.

        Returns:
            Device string
        """
        if self.settings.device != "auto":
            return self.settings.device

        torch = _imports.torch
        if torch is None:
            return "cpu"

        # Auto-detect best device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")

        return device

    def _load_models(self) -> bool:
        """
        Lazy load detection and segmentation models.

        Returns:
            True if models loaded successfully, False otherwise
        """
        if self._models_loaded:
            return True

        # Try loading YOLO model
        if (
            self.settings.detection_backend == DetectionBackend.YOLOV8
            and not self._load_yolo_model()
        ):
            logger.warning("Failed to load YOLO model")
            return False

        # Try loading SAM model if requested
        if (
            self.settings.segmentation_backend
            in {SegmentationBackend.SAM, SegmentationBackend.SAM2}
            and not self._load_sam_model()
        ):
            logger.warning("SAM model not available, using bounding boxes only")

        self._models_loaded = True
        return True

    def _load_yolo_model(self) -> bool:
        """
        Load YOLOv8 detection model.

        Returns:
            True if successful, False otherwise
        """
        ultralytics = _imports.ultralytics
        if ultralytics is None:
            return False

        try:
            # Use custom weights if provided, otherwise pretrained
            if self.settings.custom_yolo_weights:
                model_path = self.settings.custom_yolo_weights
                logger.info(f"Loading custom YOLO weights from {model_path}")
            else:
                model_name = f"yolov8{self.settings.yolo_model_size}.pt"
                model_path = model_name
                logger.info(f"Loading pretrained YOLO model: {model_name}")

            from ultralytics import YOLO

            self._yolo_model = YOLO(str(model_path))

            # Move to device
            self._yolo_model.to(self.device)

            logger.info(f"YOLO model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _load_sam_model(self) -> bool:
        """
        Load Segment Anything Model (SAM).

        Returns:
            True if successful, False otherwise
        """
        sam = _imports.sam
        if sam is None:
            return False

        try:
            # Determine model checkpoint
            if self.settings.custom_sam_checkpoint:
                checkpoint_path = self.settings.custom_sam_checkpoint
            else:
                # Try to find SAM checkpoint in cache
                checkpoint_path = self._find_sam_checkpoint()
                if checkpoint_path is None:
                    logger.warning("SAM checkpoint not found, segmentation unavailable")
                    return False

            # Load SAM model
            sam_registry = sam["registry"]
            sam_generator_class = sam["generator"]

            model_type = self.settings.sam_model_type
            logger.info(f"Loading SAM model: {model_type} from {checkpoint_path}")

            self._sam_model = sam_registry[model_type](checkpoint=str(checkpoint_path))
            self._sam_model.to(device=self.device)

            # Create mask generator
            self._sam_generator = sam_generator_class(
                model=self._sam_model,
                points_per_side=self.settings.sam_points_per_side,
                pred_iou_thresh=self.settings.sam_pred_iou_threshold,
                stability_score_thresh=self.settings.sam_stability_score_threshold,
            )

            logger.info("SAM model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            return False

    def _find_sam_checkpoint(self) -> Path | None:
        """
        Find SAM checkpoint in common locations.

        Returns:
            Path to checkpoint if found, None otherwise
        """
        model_type = self.settings.sam_model_type
        checkpoint_name = f"sam_{model_type}.pth"

        # Check common locations
        search_paths = [
            Path.home() / ".ptpd" / "models" / checkpoint_name,
            Path.home() / ".cache" / "segment_anything" / checkpoint_name,
            Path("models") / checkpoint_name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    async def detect(self, image: ImageArray) -> DeepDetectionResult:
        """
        Detect step tablet and patches in image using deep learning.

        This is the main entry point for detection. Handles the full pipeline:
        1. Load models if not already loaded
        2. Detect tablet with YOLO
        3. Segment patches with SAM
        4. Post-process and validate results
        5. Fall back to classical if needed

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            DeepDetectionResult with detected patches and metadata

        Example:
            >>> image = load_image("step_tablet.jpg")
            >>> result = await detector.detect(image)
            >>> for patch in result.patches:
            ...     print(f"Patch {patch.index}: confidence {patch.confidence:.3f}")
        """
        start_time = time.perf_counter()

        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Ensure grayscale or RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Drop alpha channel

        # Try loading models
        if not self._load_models():
            return await self._fallback_classical_detection(
                image, "Models not available", start_time
            )

        # Run detection pipeline
        try:
            # Step 1: Detect tablet with YOLO
            tablet_bbox, tablet_conf = await self._detect_tablet_yolo(image)

            if tablet_bbox is None:
                return await self._fallback_classical_detection(
                    image, "Tablet not detected by YOLO", start_time
                )

            # Step 2: Crop to tablet region
            x, y, w, h = tablet_bbox
            tablet_crop = image[y : y + h, x : x + w]

            # Step 3: Segment patches
            patches = await self._segment_patches(tablet_crop, offset=(x, y))

            if not patches or len(patches) < 5:  # Minimum reasonable patch count
                return await self._fallback_classical_detection(
                    image, f"Too few patches detected: {len(patches)}", start_time
                )

            # Step 4: Post-process and validate
            patches = self._post_process_patches(patches, image.shape[:2])

            # Step 5: Create result
            inference_time = (time.perf_counter() - start_time) * 1000

            result = DeepDetectionResult(
                tablet_bbox=tablet_bbox,
                tablet_confidence=tablet_conf,
                patches=patches,
                num_patches=len(patches),
                detection_quality=self._compute_detection_quality(patches, tablet_conf),
                confidence_level=self._classify_confidence(tablet_conf, len(patches)),
                used_fallback=False,
                inference_time_ms=inference_time,
                device_used=self.device,
                model_version=f"yolo_{self.settings.yolo_model_size}_sam_{self.settings.sam_model_type}",
            )

            logger.info(
                f"Detection completed: {len(patches)} patches in {inference_time:.1f}ms "
                f"(confidence: {tablet_conf:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return await self._fallback_classical_detection(image, str(e), start_time)

    async def _detect_tablet_yolo(self, image: ImageArray) -> tuple[BoundingBox | None, float]:
        """
        Detect step tablet using YOLOv8.

        Args:
            image: Input image

        Returns:
            Tuple of (bounding_box, confidence) or (None, 0.0) if not detected
        """
        if self._yolo_model is None:
            return None, 0.0

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._yolo_model.predict(
                    image,
                    conf=self.settings.yolo_confidence_threshold,
                    iou=self.settings.yolo_iou_threshold,
                    max_det=self.settings.yolo_max_detections,
                    verbose=False,
                ),
            )

            # Process results
            if not results or len(results) == 0:
                return None, 0.0

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return None, 0.0

            # Get highest confidence detection
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)

            box = boxes.xyxy[best_idx].cpu().numpy()
            conf = float(confidences[best_idx])

            # Convert from xyxy to xywh
            x1, y1, x2, y2 = box
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            return bbox, conf

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return None, 0.0

    async def _segment_patches(
        self, tablet_crop: ImageArray, offset: tuple[int, int] = (0, 0)
    ) -> list[DetectedPatch]:
        """
        Segment individual patches using SAM.

        Args:
            tablet_crop: Cropped tablet region
            offset: Offset to add to coordinates (x, y)

        Returns:
            List of detected patches
        """
        if self._sam_generator is None:
            # Fall back to grid-based detection
            return self._segment_patches_grid(tablet_crop, offset)

        try:
            # Run SAM in thread pool
            loop = asyncio.get_event_loop()
            masks = await loop.run_in_executor(
                None, lambda: self._sam_generator.generate(tablet_crop)
            )

            # Convert SAM masks to patches
            patches = []
            for idx, mask_data in enumerate(masks):
                mask = mask_data["segmentation"]
                bbox_xyxy = mask_data["bbox"]
                area = mask_data["area"]
                predicted_iou = mask_data["predicted_iou"]

                # Filter by area
                h, w = tablet_crop.shape[:2]
                area_ratio = area / (h * w)

                if (
                    area_ratio < self.settings.min_patch_area_ratio
                    or area_ratio > self.settings.max_patch_area_ratio
                ):
                    continue

                # Convert bbox
                x, y, w_box, h_box = (
                    int(bbox_xyxy[0]),
                    int(bbox_xyxy[1]),
                    int(bbox_xyxy[2]),
                    int(bbox_xyxy[3]),
                )
                bbox = (x + offset[0], y + offset[1], w_box, h_box)

                # Compute centroid
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0:
                    centroid = (
                        float(np.mean(x_coords)) + offset[0],
                        float(np.mean(y_coords)) + offset[1],
                    )
                else:
                    centroid = (
                        float(x + w_box / 2) + offset[0],
                        float(y + h_box / 2) + offset[1],
                    )

                patch = DetectedPatch(
                    index=idx,  # Will be reordered later
                    bbox=bbox,
                    confidence=float(predicted_iou),
                    mask=mask,
                    mask_area=int(area),
                    centroid=centroid,
                )
                patches.append(patch)

            # Sort patches by position (left to right or top to bottom)
            patches = self._sort_patches(patches)

            # Reassign indices
            for idx, patch in enumerate(patches):
                patch.index = idx

            return patches

        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return self._segment_patches_grid(tablet_crop, offset)

    def _segment_patches_grid(
        self, tablet_crop: ImageArray, offset: tuple[int, int] = (0, 0)
    ) -> list[DetectedPatch]:
        """
        Fallback grid-based patch detection.

        Args:
            tablet_crop: Cropped tablet region
            offset: Coordinate offset

        Returns:
            List of detected patches
        """
        # Simple grid-based approach - assume 21 patches in a row
        h, w = tablet_crop.shape[:2]
        num_patches = 21
        patch_width = w // num_patches

        patches = []
        for i in range(num_patches):
            x = i * patch_width
            bbox = (x + offset[0], offset[1], patch_width, h)
            centroid = (x + patch_width / 2 + offset[0], h / 2 + offset[1])

            patch = DetectedPatch(
                index=i,
                bbox=bbox,
                confidence=0.5,  # Low confidence for grid-based
                mask=None,
                mask_area=patch_width * h,
                centroid=centroid,
            )
            patches.append(patch)

        return patches

    def _sort_patches(self, patches: list[DetectedPatch]) -> list[DetectedPatch]:
        """
        Sort patches by spatial position.

        Args:
            patches: Unsorted patches

        Returns:
            Sorted patches (left-to-right or top-to-bottom)
        """
        if not patches:
            return patches

        # Check if patches are arranged horizontally or vertically
        centroids = np.array([p.centroid for p in patches])
        x_range = centroids[:, 0].max() - centroids[:, 0].min()
        y_range = centroids[:, 1].max() - centroids[:, 1].min()

        if x_range > y_range:
            # Horizontal arrangement - sort by x
            return sorted(patches, key=lambda p: p.centroid[0])
        else:
            # Vertical arrangement - sort by y
            return sorted(patches, key=lambda p: p.centroid[1])

    def _post_process_patches(
        self, patches: list[DetectedPatch], image_shape: tuple[int, int]
    ) -> list[DetectedPatch]:
        """
        Post-process patches: merge overlaps, filter outliers.

        Args:
            patches: Raw detected patches
            image_shape: Image shape (H, W)

        Returns:
            Cleaned patches
        """
        if not patches:
            return patches

        # Merge overlapping patches if enabled
        if self.settings.merge_overlapping_masks:
            patches = self._merge_overlapping_patches(patches)

        # Filter patches that are clearly outliers
        patches = self._filter_outlier_patches(patches, image_shape)

        return patches

    def _merge_overlapping_patches(self, patches: list[DetectedPatch]) -> list[DetectedPatch]:
        """
        Merge patches with significant overlap.

        Args:
            patches: Input patches

        Returns:
            Merged patches
        """
        if len(patches) <= 1:
            return patches

        # Simple IoU-based merging
        merged = []
        used = set()

        for i, p1 in enumerate(patches):
            if i in used:
                continue

            # Check for overlaps
            to_merge = [p1]
            for j, p2 in enumerate(patches[i + 1 :], start=i + 1):
                if j in used:
                    continue

                iou = self._compute_iou(p1.bbox, p2.bbox)
                if iou > 0.3:  # Significant overlap
                    to_merge.append(p2)
                    used.add(j)

            # Merge if needed
            if len(to_merge) == 1:
                merged.append(p1)
            else:
                merged_patch = self._merge_patch_group(to_merge)
                merged.append(merged_patch)

            used.add(i)

        return merged

    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Compute IoU between two bounding boxes.

        Args:
            bbox1: First box (x, y, w, h)
            bbox2: Second box (x, y, w, h)

        Returns:
            IoU score [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Compute intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_patch_group(self, patches: list[DetectedPatch]) -> DetectedPatch:
        """
        Merge a group of patches into one.

        Args:
            patches: Patches to merge

        Returns:
            Merged patch
        """
        # Compute bounding box encompassing all patches
        x_min = min(p.bbox[0] for p in patches)
        y_min = min(p.bbox[1] for p in patches)
        x_max = max(p.bbox[0] + p.bbox[2] for p in patches)
        y_max = max(p.bbox[1] + p.bbox[3] for p in patches)

        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Average confidence
        avg_conf = np.mean([p.confidence for p in patches])

        # Compute merged centroid
        centroids = np.array([p.centroid for p in patches])
        merged_centroid = tuple(centroids.mean(axis=0))

        # Merge masks if available
        merged_mask = None
        total_area = sum(p.mask_area for p in patches)

        return DetectedPatch(
            index=patches[0].index,
            bbox=merged_bbox,
            confidence=float(avg_conf),
            mask=merged_mask,
            mask_area=total_area,
            centroid=merged_centroid,
        )

    def _filter_outlier_patches(
        self, patches: list[DetectedPatch], image_shape: tuple[int, int]
    ) -> list[DetectedPatch]:
        """
        Filter outlier patches based on size and position.

        Args:
            patches: Input patches
            image_shape: Image dimensions (H, W)

        Returns:
            Filtered patches
        """
        if len(patches) < 3:
            return patches

        # Compute median area
        areas = np.array([p.mask_area for p in patches])
        median_area = np.median(areas)

        # Filter patches with area too different from median
        filtered = []
        for patch in patches:
            area_ratio = patch.mask_area / median_area
            if 0.3 <= area_ratio <= 3.0:  # Within 3x of median
                filtered.append(patch)

        return filtered if len(filtered) >= 3 else patches

    def _compute_detection_quality(
        self, patches: list[DetectedPatch], tablet_confidence: float
    ) -> float:
        """
        Compute overall detection quality score.

        Args:
            patches: Detected patches
            tablet_confidence: Tablet detection confidence

        Returns:
            Quality score [0, 1]
        """
        if not patches:
            return 0.0

        # Factors: tablet confidence, patch count, patch confidences
        avg_patch_conf = np.mean([p.confidence for p in patches])
        patch_count_factor = min(len(patches) / 21, 1.0)  # Ideal is 21

        quality = 0.4 * tablet_confidence + 0.4 * avg_patch_conf + 0.2 * patch_count_factor

        return float(np.clip(quality, 0.0, 1.0))

    def _classify_confidence(
        self, tablet_confidence: float, num_patches: int
    ) -> DetectionConfidence:
        """
        Classify overall confidence level.

        Args:
            tablet_confidence: Tablet detection confidence
            num_patches: Number of detected patches

        Returns:
            Confidence level classification
        """
        # Consider both tablet confidence and patch count
        expected_patches = 21
        patch_ratio = min(num_patches / expected_patches, 1.0)

        combined_score = 0.6 * tablet_confidence + 0.4 * patch_ratio

        if combined_score >= 0.9:
            return DetectionConfidence.VERY_HIGH
        elif combined_score >= 0.7:
            return DetectionConfidence.HIGH
        elif combined_score >= 0.5:
            return DetectionConfidence.MEDIUM
        else:
            return DetectionConfidence.LOW

    async def _fallback_classical_detection(
        self, image: ImageArray, reason: str, start_time: float
    ) -> DeepDetectionResult:
        """
        Fall back to classical computer vision detection.

        Args:
            image: Input image
            reason: Reason for fallback
            start_time: Start time for timing

        Returns:
            Detection result using classical methods
        """
        if not self.settings.fallback_to_classical:
            raise RuntimeError(f"Deep learning detection failed: {reason}")

        logger.warning(f"Falling back to classical detection: {reason}")

        try:
            # Import classical detector
            from ptpd_calibration.detection.classical import ClassicalTabletDetector

            classical_detector = ClassicalTabletDetector()
            classical_result = await classical_detector.detect(image)

            # Convert to deep detection result
            inference_time = (time.perf_counter() - start_time) * 1000

            result = DeepDetectionResult(
                tablet_bbox=classical_result.tablet_bbox,
                tablet_confidence=0.7,  # Moderate confidence for classical
                patches=classical_result.patches,
                num_patches=len(classical_result.patches),
                detection_quality=0.6,
                confidence_level=DetectionConfidence.MEDIUM,
                used_fallback=True,
                fallback_reason=reason,
                warnings=[f"Used classical fallback: {reason}"],
                inference_time_ms=inference_time,
                device_used="cpu",
                model_version="classical_cv",
            )

            return result

        except Exception as e:
            logger.error(f"Classical fallback also failed: {e}")

            # Return empty result
            inference_time = (time.perf_counter() - start_time) * 1000

            return DeepDetectionResult(
                tablet_bbox=(0, 0, image.shape[1], image.shape[0]),
                tablet_confidence=0.0,
                patches=[],
                num_patches=0,
                detection_quality=0.0,
                confidence_level=DetectionConfidence.LOW,
                used_fallback=True,
                fallback_reason=f"Both deep and classical detection failed: {reason}, {e}",
                warnings=[
                    f"Deep learning failed: {reason}",
                    f"Classical fallback failed: {e}",
                ],
                inference_time_ms=inference_time,
                device_used="cpu",
                model_version="none",
            )

    def clear_cache(self):
        """Clear loaded models from memory to free resources."""
        self._yolo_model = None
        self._sam_model = None
        self._sam_generator = None
        self._models_loaded = False
        logger.info("Model cache cleared")
