"""
Step tablet detection and localization.

Uses classical computer vision techniques for reliable patch detection.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ptpd_calibration.config import DetectionSettings, get_settings


@dataclass
class DetectionResult:
    """Result of step tablet detection."""

    bounds: tuple[int, int, int, int]  # x, y, width, height
    rotation: float  # degrees
    orientation: str  # "horizontal" or "vertical"
    patch_bounds: list[tuple[int, int, int, int]]
    confidence: float
    warnings: list[str]


class StepTabletDetector:
    """
    Detects and localizes step tablets in scanned images.

    Uses edge detection, morphological operations, and contour analysis
    to find step tablets and segment individual patches.
    """

    def __init__(self, settings: DetectionSettings | None = None):
        """
        Initialize the detector.

        Args:
            settings: Detection settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings().detection

    def detect(
        self,
        image: np.ndarray | Image.Image | Path | str,
        num_patches: int | None = None,
    ) -> DetectionResult:
        """
        Detect step tablet in image.

        Args:
            image: Input image as numpy array, PIL Image, or path.
            num_patches: Expected number of patches (for validation).

        Returns:
            DetectionResult with bounds and patch locations.
        """
        # Load image
        img_array = self._load_image(image)
        height, width = img_array.shape[:2]

        # Convert to grayscale if needed
        gray = self._to_grayscale(img_array) if len(img_array.shape) == 3 else img_array

        warnings: list[str] = []

        # Detect tablet region
        tablet_bounds, orientation = self._detect_tablet_region(gray)

        # Calculate rotation and correct
        rotation = self._detect_rotation(gray, tablet_bounds)
        if abs(rotation) > self.settings.rotation_threshold:
            gray = self._rotate_image(gray, -rotation)
            img_array = self._rotate_image(img_array, -rotation)
            # Re-detect after rotation
            tablet_bounds, orientation = self._detect_tablet_region(gray)

        # Segment patches
        patch_bounds = self._segment_patches(gray, tablet_bounds, orientation, num_patches)

        # Validate detection
        if num_patches and len(patch_bounds) != num_patches:
            warnings.append(
                f"Expected {num_patches} patches, detected {len(patch_bounds)}. "
                "Using uniform segmentation."
            )
            patch_bounds = self._uniform_segmentation(tablet_bounds, num_patches, orientation)

        # Calculate confidence
        confidence = self._calculate_confidence(patch_bounds, tablet_bounds, orientation)

        return DetectionResult(
            bounds=tablet_bounds,
            rotation=rotation,
            orientation=orientation,
            patch_bounds=patch_bounds,
            confidence=confidence,
            warnings=warnings,
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

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using luminosity method."""
        if len(image.shape) == 2:
            return image
        # Use Rec. 709 coefficients for better perceptual accuracy
        return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)

    def _detect_tablet_region(
        self, gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int], str]:
        """Detect the main tablet region using edge detection and contours."""
        height, width = gray.shape

        # Apply edge detection
        edges = self._canny_edge_detection(gray)

        # Morphological closing to connect edges
        kernel_size = self.settings.morph_kernel_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        closed = self._morphological_close(edges, kernel, self.settings.morph_iterations)

        # Find contours
        contours = self._find_contours(closed)

        # Filter contours by area
        min_area = height * width * self.settings.min_contour_area_ratio
        max_area = height * width * self.settings.max_contour_area_ratio

        valid_contours = [c for c in contours if min_area < self._contour_area(c) < max_area]

        if not valid_contours:
            # Fallback: use center region
            margin = int(min(height, width) * 0.1)
            bounds = (margin, margin, width - 2 * margin, height - 2 * margin)
            orientation = "horizontal" if width > height else "vertical"
            return bounds, orientation

        # Find largest rectangular contour
        best_contour = max(valid_contours, key=lambda c: self._contour_area(c))
        bounds = self._bounding_rect(best_contour)

        # Determine orientation
        x, y, w, h = bounds
        orientation = "horizontal" if w > h else "vertical"

        return bounds, orientation

    def _canny_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Simple Canny-like edge detection without OpenCV."""
        # Compute gradients using Sobel-like kernels
        gx = np.zeros_like(gray, dtype=np.float64)
        gy = np.zeros_like(gray, dtype=np.float64)

        # Sobel kernels
        gray_f = gray.astype(np.float64)
        gx[:, 1:-1] = gray_f[:, 2:] - gray_f[:, :-2]
        gy[1:-1, :] = gray_f[2:, :] - gray_f[:-2, :]

        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Thresholding
        low = self.settings.canny_low_threshold
        high = self.settings.canny_high_threshold

        edges = np.zeros_like(gray, dtype=np.uint8)
        strong = magnitude > high
        weak = (magnitude >= low) & (magnitude <= high)

        # Simple hysteresis: connect weak edges to strong ones
        edges[strong] = 255

        # Dilate strong edges and keep weak edges connected to them
        for _ in range(2):
            dilated = self._dilate(edges, 3)
            edges[weak & (dilated > 0)] = 255

        return edges

    def _dilate(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Simple dilation operation."""
        result = np.zeros_like(image)
        half = kernel_size // 2

        for i in range(half, image.shape[0] - half):
            for j in range(half, image.shape[1] - half):
                result[i, j] = np.max(image[i - half : i + half + 1, j - half : j + half + 1])

        return result

    def _morphological_close(
        self, image: np.ndarray, kernel: np.ndarray, iterations: int
    ) -> np.ndarray:
        """Morphological closing (dilation followed by erosion)."""
        result = image.copy()
        kh, kw = kernel.shape
        half_h, half_w = kh // 2, kw // 2

        for _ in range(iterations):
            # Dilation
            dilated = np.zeros_like(result)
            for i in range(half_h, result.shape[0] - half_h):
                for j in range(half_w, result.shape[1] - half_w):
                    region = result[i - half_h : i + half_h + 1, j - half_w : j + half_w + 1]
                    dilated[i, j] = np.max(region * kernel)
            result = dilated

        for _ in range(iterations):
            # Erosion
            eroded = np.zeros_like(result)
            for i in range(half_h, result.shape[0] - half_h):
                for j in range(half_w, result.shape[1] - half_w):
                    region = result[i - half_h : i + half_h + 1, j - half_w : j + half_w + 1]
                    eroded[i, j] = np.min(region + (1 - kernel) * 255)
            result = eroded

        return result

    def _find_contours(self, binary: np.ndarray) -> list[np.ndarray]:
        """Simple contour finding algorithm."""
        # Find connected components
        labeled, num_features = self._label_components(binary)
        contours = []

        for label in range(1, num_features + 1):
            mask = (labeled == label).astype(np.uint8)
            # Find boundary pixels
            boundary = self._find_boundary(mask)
            if len(boundary) > 10:
                contours.append(np.array(boundary))

        return contours

    def _label_components(self, binary: np.ndarray) -> tuple[np.ndarray, int]:
        """Label connected components."""
        from scipy.ndimage import label

        return label(binary > 0)

    def _find_boundary(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Find boundary pixels of a binary mask."""
        boundary = []
        h, w = mask.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if mask[i, j] > 0:
                    # Check if any neighbor is background
                    neighbors = mask[i - 1 : i + 2, j - 1 : j + 2]
                    if np.any(neighbors == 0):
                        boundary.append((j, i))  # (x, y)

        return boundary

    def _contour_area(self, contour: np.ndarray) -> float:
        """Calculate contour area using shoelace formula."""
        if len(contour) < 3:
            return 0.0

        x = contour[:, 0]
        y = contour[:, 1]
        return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    def _bounding_rect(self, contour: np.ndarray) -> tuple[int, int, int, int]:
        """Get bounding rectangle of contour."""
        x_min, y_min = np.min(contour, axis=0)
        x_max, y_max = np.max(contour, axis=0)
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _detect_rotation(
        self, gray: np.ndarray, bounds: tuple[int, int, int, int]
    ) -> float:
        """Detect rotation angle of tablet."""
        x, y, w, h = bounds

        # Extract tablet region
        region = gray[y : y + h, x : x + w]

        # Compute gradients
        gy = np.gradient(region.astype(float), axis=0)
        gx = np.gradient(region.astype(float), axis=1)

        # Find dominant angle
        angles = np.arctan2(gy.flatten(), gx.flatten())
        angles = np.degrees(angles)

        # Find angles near horizontal (±20 degrees)
        mask = (np.abs(angles) < 20) | (np.abs(angles - 180) < 20) | (np.abs(angles + 180) < 20)
        if np.sum(mask) > 0:
            relevant_angles = angles[mask]
            # Correct angles near ±180
            relevant_angles = np.where(
                relevant_angles > 90, relevant_angles - 180, relevant_angles
            )
            relevant_angles = np.where(
                relevant_angles < -90, relevant_angles + 180, relevant_angles
            )
            rotation = float(np.median(relevant_angles))
        else:
            rotation = 0.0

        return np.clip(rotation, -self.settings.max_rotation_angle, self.settings.max_rotation_angle)

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)."""
        from scipy.ndimage import rotate

        return rotate(image, angle, reshape=False, order=1)

    def _segment_patches(
        self,
        gray: np.ndarray,
        bounds: tuple[int, int, int, int],
        orientation: str,
        num_patches: int | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Segment patches within tablet region."""
        x, y, w, h = bounds
        region = gray[y : y + h, x : x + w]

        # Use gradient analysis to find patch boundaries
        if orientation == "horizontal":
            profile = np.mean(region, axis=0)
        else:
            profile = np.mean(region, axis=1)

        # Compute gradient
        gradient = np.abs(np.gradient(profile))

        # Find peaks (patch boundaries)
        threshold = np.mean(gradient) + self.settings.gradient_threshold * np.std(gradient)
        peaks = self._find_peaks(gradient, threshold)

        if len(peaks) < 2:
            # Fallback to uniform segmentation
            if num_patches:
                return self._uniform_segmentation(bounds, num_patches, orientation)
            return [bounds]

        # Create patch bounds from peaks
        patches = []
        min_width = int((w if orientation == "horizontal" else h) * self.settings.min_patch_width_ratio)

        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]

            if end - start >= min_width:
                if orientation == "horizontal":
                    patches.append((x + start, y, end - start, h))
                else:
                    patches.append((x, y + start, w, end - start))

        return patches if patches else self._uniform_segmentation(bounds, num_patches or 21, orientation)

    def _find_peaks(self, data: np.ndarray, threshold: float) -> list[int]:
        """Find peaks in 1D data above threshold."""
        peaks = [0]  # Start with first position

        above_threshold = data > threshold
        in_peak = False
        peak_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_peak:
                peak_start = i
                in_peak = True
            elif not is_above and in_peak:
                peak_center = (peak_start + i) // 2
                peaks.append(peak_center)
                in_peak = False

        peaks.append(len(data))  # End with last position
        return peaks

    def _uniform_segmentation(
        self,
        bounds: tuple[int, int, int, int],
        num_patches: int,
        orientation: str,
    ) -> list[tuple[int, int, int, int]]:
        """Create uniform patch segmentation."""
        x, y, w, h = bounds
        patches = []

        if orientation == "horizontal":
            patch_width = w / num_patches
            for i in range(num_patches):
                px = int(x + i * patch_width)
                pw = int(patch_width)
                patches.append((px, y, pw, h))
        else:
            patch_height = h / num_patches
            for i in range(num_patches):
                py = int(y + i * patch_height)
                ph = int(patch_height)
                patches.append((x, py, w, ph))

        return patches

    def _calculate_confidence(
        self,
        patch_bounds: list[tuple[int, int, int, int]],
        tablet_bounds: tuple[int, int, int, int],
        orientation: str,
    ) -> float:
        """Calculate confidence score for detection."""
        if not patch_bounds:
            return 0.0

        # Check patch size uniformity
        if orientation == "horizontal":
            sizes = [b[2] for b in patch_bounds]  # widths
        else:
            sizes = [b[3] for b in patch_bounds]  # heights

        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        uniformity = 1.0 - min(1.0, std_size / mean_size) if mean_size > 0 else 0.0

        # Check coverage
        total_patch_size = sum(sizes)
        tablet_size = tablet_bounds[2] if orientation == "horizontal" else tablet_bounds[3]
        coverage = total_patch_size / tablet_size if tablet_size > 0 else 0.0

        # Combined confidence
        confidence = 0.6 * uniformity + 0.4 * min(1.0, coverage)

        return confidence
