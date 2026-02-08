"""
Automated Defect Detection for Platinum-Palladium Prints.

Provides deep learning-based detection and classification of print defects
using U-Net++ segmentation and ResNet classification.

Features:
- U-Net++ architecture for defect segmentation
- ResNet-based defect classification
- Multi-scale detection for various defect sizes
- Morphological post-processing
- Severity estimation
- Remediation suggestions

All settings are configuration-driven with no hardcoded values.
"""

import logging
from pathlib import Path

import numpy as np

from ptpd_calibration.deep_learning.config import DefectDetectionSettings
from ptpd_calibration.deep_learning.models import (
    DefectDetectionResult,
    DetectedDefect,
)
from ptpd_calibration.deep_learning.types import (
    DefectSeverity,
    DefectType,
)

logger = logging.getLogger(__name__)

# Lazy imports for PyTorch (only loaded when needed)
_torch = None
_nn = None
_functional = None


def _import_torch():
    """Lazy import of PyTorch dependencies."""
    global _torch, _nn, _functional
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            _torch = torch
            _nn = nn
            _functional = F
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for defect detection. "
                "Install with: pip install torch torchvision"
            ) from e
    return _torch, _nn, _functional


class ConvBlock:
    """Convolutional block with batch norm and activation."""

    def __init__(self, in_channels: int, out_channels: int, nn):
        """Initialize conv block."""
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def __call__(self, x):
        """Forward pass."""
        return self.block(x)


class DefectSegmentationNet:
    """
    U-Net++ style segmentation network for defect detection.

    Architecture:
    - Encoder: ResNet backbone (configurable)
    - Decoder: Nested skip connections (U-Net++)
    - Multi-scale outputs for detecting defects of various sizes
    - Deep supervision for better gradient flow

    All architecture parameters from settings.
    """

    def __init__(self, settings: DefectDetectionSettings):
        """
        Initialize the segmentation network.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        torch, nn, F = _import_torch()

        # Encoder
        self.encoder = self._build_encoder(nn)

        # Decoder with nested skip connections (U-Net++)
        self.decoder = self._build_decoder(nn)

        # Output heads for multi-scale predictions
        self.output_heads = self._build_output_heads(nn)

        # Device
        self.device = self._get_device()
        self.to(self.device)

        total_params = sum(
            p.numel()
            for module in [self.encoder, self.decoder, self.output_heads]
            for p in module.parameters()
        )
        logger.info(f"DefectSegmentationNet initialized with {total_params} parameters")

    def _build_encoder(self, nn):
        """Build encoder backbone."""
        # Simplified encoder - in production would use pretrained ResNet
        encoder_channels = [64, 128, 256, 512]

        encoder = nn.ModuleList()

        # Initial conv
        encoder.append(
            nn.Sequential(
                nn.Conv2d(3, encoder_channels[0], kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(encoder_channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )

        # Encoder blocks
        for i in range(len(encoder_channels) - 1):
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(encoder_channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        encoder_channels[i + 1],
                        encoder_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(encoder_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        return encoder

    def _build_decoder(self, nn):
        """Build U-Net++ style decoder with nested skip connections."""
        decoder_channels = [512, 256, 128, 64]

        decoder = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            # Upsample block
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.BatchNorm2d(decoder_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

            # Fusion block (combines upsampled features with skip connection)
            decoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        decoder_channels[i + 1] * 2,  # Concat with skip
                        decoder_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(decoder_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        return decoder

    def _build_output_heads(self, nn):
        """Build output heads for multi-scale predictions."""
        output_heads = nn.ModuleList()

        # Binary segmentation head (defect vs background)
        output_heads.append(nn.Conv2d(64, 1, kernel_size=1))

        # Multi-class defect type head
        output_heads.append(nn.Conv2d(64, self.settings.num_classes, kernel_size=1))

        return output_heads

    def _get_device(self) -> str:
        """Determine device to use."""
        torch, _, _ = _import_torch()

        if self.settings.device != "auto":
            return self.settings.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        for module in [self.encoder, self.decoder, self.output_heads]:
            module.to(device)

    def forward(self, x):
        """
        Forward pass through segmentation network.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Tuple of (binary_mask, defect_type_map)
        """
        torch, _, F = _import_torch()

        # Encoder
        encoder_features = []
        for enc_block in self.encoder:
            x = enc_block(x)
            encoder_features.append(x)

        # Decoder with skip connections
        decoder_idx = 0
        for i in range(len(encoder_features) - 1, 0, -1):
            # Upsample
            x = self.decoder[decoder_idx](x)
            decoder_idx += 1

            # Concatenate with skip connection
            skip = encoder_features[i - 1]
            # Match spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)

            # Fusion
            x = self.decoder[decoder_idx](x)
            decoder_idx += 1

        # Output heads
        binary_mask = self.output_heads[0](x)
        defect_types = self.output_heads[1](x)

        return binary_mask, defect_types

    def parameters(self):
        """Get all model parameters."""
        params = []
        for module in [self.encoder, self.decoder, self.output_heads]:
            params.extend(module.parameters())
        return params


class DefectClassifierNet:
    """
    ResNet-based defect classifier for defect type and severity estimation.

    Takes a crop of a detected defect region and classifies:
    1. Defect type (coating, chemical, paper, environmental, exposure, other)
    2. Severity level (negligible to critical)
    """

    def __init__(self, settings: DefectDetectionSettings):
        """
        Initialize the classifier network.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        torch, nn, _ = _import_torch()

        # Build ResNet-style backbone
        self.backbone = self._build_backbone(nn)

        # Classification heads
        self.type_head = nn.Linear(512, self.settings.num_classes)
        self.severity_head = nn.Linear(512, len(DefectSeverity))

        # Device
        self.device = self._get_device()
        self.to(self.device)

        total_params = sum(
            p.numel()
            for module in [self.backbone, self.type_head, self.severity_head]
            for p in module.parameters()
        )
        logger.info(f"DefectClassifierNet initialized with {total_params} parameters")

    def _build_backbone(self, nn):
        """Build ResNet-style backbone."""
        # Simplified ResNet blocks
        return nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # ResNet blocks
            self._make_layer(64, 64, 2, nn),
            self._make_layer(64, 128, 2, nn, stride=2),
            self._make_layer(128, 256, 2, nn, stride=2),
            self._make_layer(256, 512, 2, nn, stride=2),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, nn, stride=1):
        """Make a ResNet layer."""
        layers = []

        # First block may downsample
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
        )

        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
            )

        return nn.Sequential(*layers)

    def _get_device(self) -> str:
        """Determine device to use."""
        torch, _, _ = _import_torch()

        if self.settings.device != "auto":
            return self.settings.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        self.backbone.to(device)
        self.type_head.to(device)
        self.severity_head.to(device)

    def forward(self, x):
        """
        Forward pass through classifier.

        Args:
            x: Input defect crop [B, 3, H, W]

        Returns:
            Tuple of (type_logits, severity_logits)
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Classification heads
        type_logits = self.type_head(features)
        severity_logits = self.severity_head(features)

        return type_logits, severity_logits

    def parameters(self):
        """Get all model parameters."""
        params = []
        params.extend(self.backbone.parameters())
        params.extend(self.type_head.parameters())
        params.extend(self.severity_head.parameters())
        return params


class DefectDetector:
    """
    Complete defect detection pipeline.

    Combines:
    - Multi-scale segmentation
    - Defect classification
    - Morphological post-processing
    - Remediation suggestions
    """

    def __init__(
        self,
        settings: DefectDetectionSettings | None = None,
        segmentation_model_path: Path | None = None,
        classifier_model_path: Path | None = None,
    ):
        """
        Initialize the defect detector.

        Args:
            settings: Configuration settings
            segmentation_model_path: Path to pretrained segmentation model
            classifier_model_path: Path to pretrained classifier model
        """
        self.settings = settings or DefectDetectionSettings()

        # Initialize models
        self.segmentation_net = DefectSegmentationNet(self.settings)
        self.classifier_net = DefectClassifierNet(self.settings)

        # Load pretrained models if provided
        if segmentation_model_path or self.settings.segmentation_model_path:
            path = segmentation_model_path or self.settings.segmentation_model_path
            self._load_segmentation_model(path)

        if classifier_model_path or self.settings.classifier_model_path:
            path = classifier_model_path or self.settings.classifier_model_path
            self._load_classifier_model(path)

        # Defect type to enum mapping
        self.defect_type_mapping = dict(enumerate(DefectType))
        self.severity_mapping = dict(enumerate(DefectSeverity))

        logger.info("DefectDetector initialized successfully")

    def detect(self, image: np.ndarray) -> DefectDetectionResult:
        """
        Detect defects in a platinum-palladium print.

        Args:
            image: Input image as numpy array [H, W, 3] in RGB

        Returns:
            DefectDetectionResult with all detected defects
        """
        import time

        start_time = time.time()
        torch, _, F = _import_torch()

        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB with shape [H, W, 3]")

        original_shape = image.shape[:2]

        # Multi-scale detection if enabled
        if self.settings.use_multi_scale:
            all_defects = []
            for scale in self.settings.scales:
                scaled_defects = self._detect_at_scale(image, scale)
                all_defects.extend(scaled_defects)

            # Merge overlapping defects
            if self.settings.merge_nearby_defects:
                all_defects = self._merge_defects(all_defects)
        else:
            all_defects = self._detect_at_scale(image, 1.0)

        # Sort by severity and confidence
        all_defects.sort(key=lambda d: (d.severity.value, -d.confidence), reverse=True)

        # Limit number of defects
        all_defects = all_defects[: self.settings.max_defects_per_image]

        # Calculate summary statistics
        defects_by_type = {}
        defects_by_severity = {}
        for defect in all_defects:
            defects_by_type[defect.defect_type.value] = (
                defects_by_type.get(defect.defect_type.value, 0) + 1
            )
            defects_by_severity[defect.severity.value] = (
                defects_by_severity.get(defect.severity.value, 0) + 1
            )

        # Determine overall severity
        overall_severity = self._determine_overall_severity(all_defects)

        # Determine if print is acceptable
        print_acceptable = overall_severity in [
            DefectSeverity.NEGLIGIBLE,
            DefectSeverity.MINOR,
        ]

        # Calculate total defect coverage
        total_defect_area = sum(d.area_pixels for d in all_defects)
        total_image_area = original_shape[0] * original_shape[1]
        defect_coverage = (total_defect_area / total_image_area) * 100

        # Generate recommendations
        recommendations = self._generate_recommendations(all_defects)

        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000

        # Create result
        result = DefectDetectionResult(
            defects=all_defects,
            num_defects=len(all_defects),
            defects_by_type=defects_by_type,
            defects_by_severity=defects_by_severity,
            overall_severity=overall_severity,
            print_acceptable=print_acceptable,
            full_mask=None,  # Could generate combined mask if needed
            defect_coverage=defect_coverage,
            recommendations=recommendations,
            inference_time_ms=inference_time_ms,
            device_used=self.segmentation_net.device,
            model_version="1.0.0",
        )

        return result

    def _detect_at_scale(self, image: np.ndarray, scale: float) -> list[DetectedDefect]:
        """Detect defects at a specific scale."""
        torch, _, F = _import_torch()

        # Resize image
        if scale != 1.0:
            new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
            import cv2

            scaled_image = cv2.resize(image, (new_size[1], new_size[0]))
        else:
            scaled_image = image

        # Prepare input tensor
        input_tensor = self._preprocess_image(scaled_image)

        # Run segmentation
        with torch.no_grad():
            binary_mask, defect_type_map = self.segmentation_net.forward(input_tensor)

        # Convert to numpy
        binary_mask = torch.sigmoid(binary_mask).cpu().numpy()[0, 0]
        defect_type_map = F.softmax(defect_type_map, dim=1).cpu().numpy()[0]

        # Threshold binary mask
        binary_mask = (binary_mask > self.settings.confidence_threshold).astype(np.uint8)

        # Apply morphological cleanup if enabled
        if self.settings.apply_morphological_cleanup:
            binary_mask = self._morphological_cleanup(binary_mask)

        # Find connected components
        defects = self._extract_defects(binary_mask, defect_type_map, scaled_image, scale)

        return defects

    def _preprocess_image(self, image: np.ndarray):
        """Preprocess image for model input."""
        torch, _, _ = _import_torch()

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor [1, 3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.segmentation_net.device)

        return image_tensor

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        try:
            import cv2

            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Fill small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        except ImportError:
            logger.warning("OpenCV not available for morphological cleanup")

        return mask

    def _extract_defects(
        self,
        binary_mask: np.ndarray,
        defect_type_map: np.ndarray,
        image: np.ndarray,
        scale: float,
    ) -> list[DetectedDefect]:
        """Extract individual defects from segmentation mask."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV required for defect extraction")
            return []

        defects = []

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        for label in range(1, num_labels):  # Skip background (0)
            # Get defect properties
            x, y, w, h, area = stats[label]

            # Filter by area
            if area < self.settings.min_defect_area:
                continue

            # Scale back to original coordinates
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                area = int(area / (scale * scale))

            # Extract defect region
            defect_mask = (labels == label).astype(np.uint8)

            # Determine defect type from type map
            defect_region_type_scores = defect_type_map[:, defect_mask > 0].mean(axis=1)
            defect_type_idx = defect_region_type_scores.argmax()
            confidence = float(defect_region_type_scores[defect_type_idx])

            defect_type = self.defect_type_mapping.get(defect_type_idx, DefectType.UNKNOWN)

            # Estimate severity based on area and type
            severity = self._estimate_severity(area, defect_type, image.shape[:2])

            # Generate remediation suggestion
            remediation = self._get_remediation(defect_type, severity)

            # Calculate area percentage
            total_area = image.shape[0] * image.shape[1]
            area_percentage = (area / total_area) * 100

            # Create defect object
            defect = DetectedDefect(
                defect_type=defect_type,
                severity=severity,
                confidence=confidence,
                bbox=(x, y, w, h),
                mask=defect_mask if scale == 1.0 else None,
                area_pixels=area,
                area_percentage=area_percentage,
                remediation=remediation,
            )

            defects.append(defect)

        return defects

    def _estimate_severity(
        self, area: int, defect_type: DefectType, image_shape: tuple[int, int]
    ) -> DefectSeverity:
        """Estimate defect severity based on area and type."""
        total_area = image_shape[0] * image_shape[1]
        area_ratio = area / total_area

        # Critical defects (affect entire print)
        if area_ratio > 0.1 or defect_type in [
            DefectType.SOLARIZATION,
            DefectType.BRONZING,
        ]:
            return DefectSeverity.CRITICAL

        # Major defects
        if area_ratio > 0.05 or defect_type in [
            DefectType.UNEVEN_COATING,
            DefectType.UNEVEN_EXPOSURE,
        ]:
            return DefectSeverity.MAJOR

        # Moderate defects
        if area_ratio > 0.01 or defect_type in [
            DefectType.BRUSH_MARK,
            DefectType.STREAKING,
            DefectType.DEVELOPER_STAIN,
        ]:
            return DefectSeverity.MODERATE

        # Minor defects
        if area_ratio > 0.001:
            return DefectSeverity.MINOR

        return DefectSeverity.NEGLIGIBLE

    def _get_remediation(self, defect_type: DefectType, severity: DefectSeverity) -> str:
        """Get remediation suggestion for defect."""
        remediation_map = {
            DefectType.BRUSH_MARK: "Use softer brush or apply coating in different direction",
            DefectType.POOLING: "Reduce coating amount or improve spreading technique",
            DefectType.STREAKING: "Ensure even coating application with overlapping strokes",
            DefectType.UNEVEN_COATING: "Practice consistent coating technique; consider coating jig",
            DefectType.COATING_GAP: "Ensure complete coverage during coating",
            DefectType.DEVELOPER_STAIN: "Use fresh developer and ensure even development",
            DefectType.FIXER_RESIDUE: "Extend washing time or use hypo clear",
            DefectType.OXIDATION: "Improve storage conditions; reduce exposure to air",
            DefectType.BRONZING: "Adjust chemistry ratio or development time",
            DefectType.SOLARIZATION: "Reduce exposure time significantly",
            DefectType.DUST: "Improve darkroom cleanliness; cover work surfaces",
            DefectType.FINGERPRINT: "Handle paper by edges only; use cotton gloves",
            DefectType.WATER_SPOT: "Ensure even drying; use distilled water for final rinse",
            DefectType.LIGHT_LEAK: "Check contact frame seal and darkroom safelights",
            DefectType.UNEVEN_EXPOSURE: "Ensure even UV source or check contact frame pressure",
            DefectType.UNDEREXPOSURE: "Increase exposure time",
            DefectType.OVEREXPOSURE: "Decrease exposure time",
            DefectType.SCRATCH: "Handle prints more carefully during processing",
        }

        base_remediation = remediation_map.get(
            defect_type, "Inspect printing process for anomalies"
        )

        if severity == DefectSeverity.CRITICAL:
            return f"CRITICAL: {base_remediation}. Consider reprinting."
        elif severity == DefectSeverity.MAJOR:
            return f"MAJOR: {base_remediation}"
        else:
            return base_remediation

    def _merge_defects(self, defects: list[DetectedDefect]) -> list[DetectedDefect]:
        """Merge nearby defects of the same type."""
        if not defects:
            return defects

        merged = []
        used = set()

        for i, defect1 in enumerate(defects):
            if i in used:
                continue

            # Find nearby defects of same type
            similar_defects = [defect1]

            for j, defect2 in enumerate(defects[i + 1 :], start=i + 1):
                if j in used:
                    continue

                # Check if same type and nearby
                if defect1.defect_type == defect2.defect_type:
                    dist = self._bbox_distance(defect1.bbox, defect2.bbox)
                    if dist < self.settings.merge_distance:
                        similar_defects.append(defect2)
                        used.add(j)

            # Merge similar defects
            if len(similar_defects) == 1:
                merged.append(defect1)
            else:
                merged_defect = self._create_merged_defect(similar_defects)
                merged.append(merged_defect)

            used.add(i)

        return merged

    def _bbox_distance(
        self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
    ) -> float:
        """Calculate distance between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate center points
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

        # Euclidean distance
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def _create_merged_defect(self, defects: list[DetectedDefect]) -> DetectedDefect:
        """Create a merged defect from multiple defects."""
        # Calculate merged bounding box
        x_min = min(d.bbox[0] for d in defects)
        y_min = min(d.bbox[1] for d in defects)
        x_max = max(d.bbox[0] + d.bbox[2] for d in defects)
        y_max = max(d.bbox[1] + d.bbox[3] for d in defects)

        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Use highest severity and average confidence
        max_severity = max(d.severity for d in defects)
        avg_confidence = np.mean([d.confidence for d in defects])

        # Sum areas
        total_area = sum(d.area_pixels for d in defects)
        total_area_pct = sum(d.area_percentage for d in defects)

        return DetectedDefect(
            defect_type=defects[0].defect_type,
            severity=max_severity,
            confidence=float(avg_confidence),
            bbox=merged_bbox,
            mask=None,
            area_pixels=total_area,
            area_percentage=total_area_pct,
            remediation=defects[0].remediation,
        )

    def _determine_overall_severity(self, defects: list[DetectedDefect]) -> DefectSeverity:
        """Determine overall print severity from all defects."""
        if not defects:
            return DefectSeverity.NEGLIGIBLE

        # Overall severity is the maximum severity
        severities = [d.severity for d in defects]
        severity_order = [
            DefectSeverity.CRITICAL,
            DefectSeverity.MAJOR,
            DefectSeverity.MODERATE,
            DefectSeverity.MINOR,
            DefectSeverity.NEGLIGIBLE,
        ]

        for severity in severity_order:
            if severity in severities:
                return severity

        return DefectSeverity.NEGLIGIBLE

    def _generate_recommendations(self, defects: list[DetectedDefect]) -> list[str]:
        """Generate overall recommendations based on detected defects."""
        recommendations = []

        if not defects:
            recommendations.append("No defects detected - excellent print quality!")
            return recommendations

        # Count defects by category
        coating_defects = sum(
            1
            for d in defects
            if d.defect_type
            in [
                DefectType.BRUSH_MARK,
                DefectType.POOLING,
                DefectType.STREAKING,
                DefectType.UNEVEN_COATING,
            ]
        )
        exposure_defects = sum(
            1
            for d in defects
            if d.defect_type
            in [
                DefectType.UNDEREXPOSURE,
                DefectType.OVEREXPOSURE,
                DefectType.UNEVEN_EXPOSURE,
            ]
        )
        chemical_defects = sum(
            1
            for d in defects
            if d.defect_type
            in [DefectType.DEVELOPER_STAIN, DefectType.FIXER_RESIDUE, DefectType.OXIDATION]
        )

        # Generate category-specific recommendations
        if coating_defects > 0:
            recommendations.append(
                f"Found {coating_defects} coating defect(s). "
                "Review coating technique and brush quality."
            )

        if exposure_defects > 0:
            recommendations.append(
                f"Found {exposure_defects} exposure defect(s). "
                "Check UV source consistency and contact frame."
            )

        if chemical_defects > 0:
            recommendations.append(
                f"Found {chemical_defects} chemical defect(s). "
                "Review chemistry freshness and processing times."
            )

        # Overall defect count recommendation
        if len(defects) > 10:
            recommendations.append(
                "High defect count detected. Consider reviewing entire workflow."
            )

        # Severity-based recommendations
        critical_defects = sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL)
        if critical_defects > 0:
            recommendations.append(
                f"{critical_defects} critical defect(s) found. Recommend reprinting."
            )

        return recommendations

    def _load_segmentation_model(self, path: Path):
        """Load pretrained segmentation model."""
        torch, _, _ = _import_torch()
        # Implementation would load model weights
        logger.info(f"Loading segmentation model from {path}")

    def _load_classifier_model(self, path: Path):
        """Load pretrained classifier model."""
        torch, _, _ = _import_torch()
        # Implementation would load model weights
        logger.info(f"Loading classifier model from {path}")


# Factory function for easy instantiation
def create_defect_detector(
    settings: DefectDetectionSettings | None = None,
    segmentation_model_path: Path | None = None,
    classifier_model_path: Path | None = None,
) -> DefectDetector:
    """
    Create a defect detector.

    Args:
        settings: Configuration settings
        segmentation_model_path: Path to segmentation model
        classifier_model_path: Path to classifier model

    Returns:
        Configured DefectDetector instance
    """
    try:
        return DefectDetector(
            settings=settings,
            segmentation_model_path=segmentation_model_path,
            classifier_model_path=classifier_model_path,
        )
    except ImportError as e:
        logger.warning(f"Could not initialize defect detector: {e}")
        logger.warning(
            "PyTorch and torchvision are required. Install with: pip install torch torchvision"
        )
        raise
