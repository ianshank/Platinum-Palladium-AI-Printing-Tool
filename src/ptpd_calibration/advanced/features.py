"""
Advanced features for Platinum/Palladium printing workflow.

Provides sophisticated tools for:
- Alternative process simulation (cyanotype, Van Dyke, kallitype, etc.)
- Advanced negative blending and masking
- QR code metadata generation for archival labeling
- Historic style transfer based on master printers
- Print comparison and quality analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import qrcode
    from qrcode.image.pil import PilImage
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False


def _get_truetype_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a TrueType font with cross-platform support.

    Tries multiple common font paths for Windows, macOS, and Linux.
    Falls back to default font if none are found.

    Args:
        size: Font size in points
        bold: Whether to use bold variant

    Returns:
        Font object (TrueType if available, default otherwise)
    """
    # Font candidates to try
    font_names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "Helvetica-Bold.ttf" if bold else "Helvetica.ttf",
    ]

    # Common font directories by platform
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/",  # Linux (Debian/Ubuntu)
        "/usr/share/fonts/dejavu/",            # Linux (other distros)
        "/System/Library/Fonts/",              # macOS
        "/Library/Fonts/",                     # macOS
        "C:/Windows/Fonts/",                   # Windows
        "C:\\Windows\\Fonts\\",                # Windows (alternative)
    ]

    # Try each combination of path and font name
    for path in font_paths:
        for name in font_names:
            try:
                return ImageFont.truetype(path + name, size)
            except OSError:
                continue

    # Fall back to default font if nothing works
    return ImageFont.load_default()


class BlendMode(str, Enum):
    """Blending modes for negative composition."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"


class HistoricStyle(str, Enum):
    """Historic platinum/palladium printing styles."""

    PICTORIALIST_1890S = "1890s_pictorialist"
    EDWARD_WESTON = "edward_weston"
    IRVING_PENN = "irving_penn"
    SALLY_MANN = "sally_mann"
    FREDERICK_EVANS = "frederick_evans"
    PAUL_STRAND = "paul_strand"


@dataclass
class AlternativeProcessParams:
    """Parameters for alternative process simulation."""

    # Tone curve characteristics
    gamma: float = 1.0
    contrast: float = 1.0

    # Color characteristics (RGB)
    shadow_color: tuple[int, int, int] = (0, 0, 0)
    midtone_color: tuple[int, int, int] = (128, 128, 128)
    highlight_color: tuple[int, int, int] = (255, 255, 255)

    # Process-specific
    dmax: float = 1.8
    dmin: float = 0.1
    stain_level: float = 0.0  # Paper staining (0-1)


@dataclass
class StyleParameters:
    """Parameters defining a historic printing style."""

    name: str
    description: str

    # Tone curve
    gamma: float = 1.0
    contrast: float = 1.0
    toe_contrast: float = 1.0  # Shadow contrast
    shoulder_contrast: float = 1.0  # Highlight contrast

    # Tonal characteristics
    shadow_tone: tuple[int, int, int] = (20, 18, 16)
    highlight_tone: tuple[int, int, int] = (245, 240, 235)

    # Density range
    dmax: float = 1.7
    dmin: float = 0.08

    # Paper characteristics
    paper_warmth: float = 0.5  # 0 = cool, 1 = warm
    texture_strength: float = 0.0


@dataclass
class PrintMetadata:
    """Metadata for archival print labeling."""

    # Image info
    title: str = ""
    artist: str = ""
    date: str = ""
    edition: str = ""

    # Technical details
    paper: str = ""
    chemistry: str = ""
    exposure_time: str = ""
    developer: str = ""

    # Calibration info
    curve_name: str = ""
    dmax: float = 0.0
    dmin: float = 0.0

    # Additional notes
    notes: str = ""


class AlternativeProcessSimulator:
    """Simulate alternative photographic processes.

    Provides accurate simulations of historic and alternative printing
    processes with characteristic tone curves and color transformations.
    """

    def __init__(self):
        """Initialize the alternative process simulator."""
        self._process_presets = self._create_process_presets()

    def simulate_cyanotype(
        self,
        image: Union[Image.Image, np.ndarray],
        params: Optional[AlternativeProcessParams] = None,
    ) -> Image.Image:
        """Simulate cyanotype (iron-based blue) print.

        Cyanotypes are characterized by deep cyan-blue tones with
        relatively high contrast and unique color characteristics.

        Args:
            image: Input image to simulate
            params: Optional custom parameters

        Returns:
            Simulated cyanotype image
        """
        if params is None:
            params = AlternativeProcessParams(
                gamma=1.2,
                contrast=1.15,
                shadow_color=(0, 20, 60),
                midtone_color=(40, 100, 160),
                highlight_color=(160, 200, 240),
                dmax=1.9,
                dmin=0.12,
                stain_level=0.05,
            )

        return self._apply_process_simulation(image, params, "Cyanotype")

    def simulate_vandyke(
        self,
        image: Union[Image.Image, np.ndarray],
        params: Optional[AlternativeProcessParams] = None,
    ) -> Image.Image:
        """Simulate Van Dyke brown print.

        Van Dyke prints feature rich brown tones with smooth gradation
        and warm, earthy characteristics.

        Args:
            image: Input image to simulate
            params: Optional custom parameters

        Returns:
            Simulated Van Dyke brown image
        """
        if params is None:
            params = AlternativeProcessParams(
                gamma=1.1,
                contrast=1.05,
                shadow_color=(25, 15, 8),
                midtone_color=(120, 80, 50),
                highlight_color=(230, 210, 180),
                dmax=1.8,
                dmin=0.15,
                stain_level=0.08,
            )

        return self._apply_process_simulation(image, params, "Van Dyke Brown")

    def simulate_kallitype(
        self,
        image: Union[Image.Image, np.ndarray],
        params: Optional[AlternativeProcessParams] = None,
    ) -> Image.Image:
        """Simulate Kallitype print.

        Kallitypes can range from warm browns to cool neutral tones
        depending on chemistry, with excellent tonal range.

        Args:
            image: Input image to simulate
            params: Optional custom parameters

        Returns:
            Simulated kallitype image
        """
        if params is None:
            params = AlternativeProcessParams(
                gamma=1.15,
                contrast=1.1,
                shadow_color=(20, 18, 15),
                midtone_color=(110, 95, 80),
                highlight_color=(240, 230, 215),
                dmax=1.85,
                dmin=0.1,
                stain_level=0.03,
            )

        return self._apply_process_simulation(image, params, "Kallitype")

    def simulate_gum_bichromate(
        self,
        image: Union[Image.Image, np.ndarray],
        pigment_color: tuple[int, int, int] = (100, 70, 50),
        params: Optional[AlternativeProcessParams] = None,
    ) -> Image.Image:
        """Simulate gum bichromate print with custom pigment.

        Gum printing allows for custom pigment colors with painterly
        characteristics and multi-layer possibilities.

        Args:
            image: Input image to simulate
            pigment_color: RGB color of pigment
            params: Optional custom parameters

        Returns:
            Simulated gum bichromate image
        """
        if params is None:
            # Gum prints have softer contrast and can have custom colors
            params = AlternativeProcessParams(
                gamma=0.95,
                contrast=0.9,
                shadow_color=pigment_color,
                midtone_color=tuple(int(c * 1.8) for c in pigment_color),
                highlight_color=(245, 240, 232),
                dmax=1.4,
                dmin=0.08,
                stain_level=0.02,
            )

        return self._apply_process_simulation(image, params, "Gum Bichromate")

    def simulate_salt_print(
        self,
        image: Union[Image.Image, np.ndarray],
        params: Optional[AlternativeProcessParams] = None,
    ) -> Image.Image:
        """Simulate salt print (oldest photographic process).

        Salt prints have characteristic warm tones, lower contrast,
        and delicate highlight rendering.

        Args:
            image: Input image to simulate
            params: Optional custom parameters

        Returns:
            Simulated salt print image
        """
        if params is None:
            params = AlternativeProcessParams(
                gamma=0.9,
                contrast=0.85,
                shadow_color=(40, 30, 20),
                midtone_color=(140, 120, 95),
                highlight_color=(240, 230, 210),
                dmax=1.3,
                dmin=0.18,
                stain_level=0.12,
            )

        return self._apply_process_simulation(image, params, "Salt Print")

    def _apply_process_simulation(
        self,
        image: Union[Image.Image, np.ndarray],
        params: AlternativeProcessParams,
        process_name: str,
    ) -> Image.Image:
        """Apply process simulation with given parameters.

        Args:
            image: Input image
            params: Process parameters
            process_name: Name of process for metadata

        Returns:
            Simulated process image
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image.astype(np.uint8), mode='L')
            else:
                pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
        else:
            pil_img = image

        # Convert to grayscale for processing
        if pil_img.mode != 'L':
            gray = pil_img.convert('L')
        else:
            gray = pil_img

        # Convert to normalized array
        arr = np.array(gray, dtype=np.float32) / 255.0

        # Apply gamma correction
        arr = np.power(arr, 1.0 / params.gamma)

        # Apply contrast
        arr = (arr - 0.5) * params.contrast + 0.5
        arr = np.clip(arr, 0, 1)

        # Map to density
        density_range = params.dmax - params.dmin
        density = params.dmax - (arr * density_range)

        # Convert back to reflectance
        reflectance = np.power(10, -density)
        reflectance = np.clip(reflectance, 0, 1)

        # Apply color mapping
        h, w = reflectance.shape
        rgb_output = np.zeros((h, w, 3), dtype=np.float32)

        shadow = np.array(params.shadow_color, dtype=np.float32) / 255.0
        midtone = np.array(params.midtone_color, dtype=np.float32) / 255.0
        highlight = np.array(params.highlight_color, dtype=np.float32) / 255.0

        # Three-point interpolation: shadow -> midtone -> highlight
        for c in range(3):
            mask_low = reflectance < 0.5
            mask_high = reflectance >= 0.5

            # Shadow to midtone
            t_low = reflectance * 2  # 0->0.5 maps to 0->1
            rgb_output[mask_low, c] = shadow[c] + t_low[mask_low] * (midtone[c] - shadow[c])

            # Midtone to highlight
            t_high = (reflectance - 0.5) * 2  # 0.5->1 maps to 0->1
            rgb_output[mask_high, c] = midtone[c] + t_high[mask_high] * (highlight[c] - midtone[c])

        # Add paper staining if present
        if params.stain_level > 0:
            stain_color = np.array(params.highlight_color, dtype=np.float32) / 255.0
            rgb_output = rgb_output * (1 - params.stain_level) + stain_color * params.stain_level

        # Convert to 8-bit
        rgb_output = np.clip(rgb_output * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(rgb_output, mode='RGB')

        # Store process info in image metadata
        result.info['process'] = process_name

        return result

    def _create_process_presets(self) -> dict[str, AlternativeProcessParams]:
        """Create preset parameters for various processes."""
        return {
            'cyanotype': AlternativeProcessParams(
                gamma=1.2, contrast=1.15,
                shadow_color=(0, 20, 60),
                highlight_color=(160, 200, 240),
            ),
            'vandyke': AlternativeProcessParams(
                gamma=1.1, contrast=1.05,
                shadow_color=(25, 15, 8),
                highlight_color=(230, 210, 180),
            ),
        }


class NegativeBlender:
    """Advanced negative blending and masking tools.

    Provides sophisticated blending operations for creating complex
    multi-negative composites and local adjustments.
    """

    def __init__(self):
        """Initialize the negative blender."""
        pass

    def blend_negatives(
        self,
        negatives: list[Union[Image.Image, np.ndarray]],
        masks: Optional[list[Union[Image.Image, np.ndarray]]] = None,
        blend_modes: Optional[list[BlendMode]] = None,
    ) -> Image.Image:
        """Blend multiple negatives with optional masks and blend modes.

        Args:
            negatives: List of negative images to blend
            masks: Optional list of masks (one per negative)
            blend_modes: Optional list of blend modes

        Returns:
            Blended negative image
        """
        if not negatives:
            raise ValueError("At least one negative is required")

        # Convert all to numpy arrays
        neg_arrays = []
        for neg in negatives:
            if isinstance(neg, Image.Image):
                if neg.mode != 'L':
                    neg = neg.convert('L')
                arr = np.array(neg, dtype=np.float32) / 255.0
            else:
                arr = neg.astype(np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
            neg_arrays.append(arr)

        # Ensure all same size
        target_size = neg_arrays[0].shape
        for i, arr in enumerate(neg_arrays[1:], 1):
            if arr.shape != target_size:
                raise ValueError(f"Negative {i} size {arr.shape} doesn't match {target_size}")

        # Process masks
        mask_arrays = []
        if masks:
            for mask in masks:
                if mask is None:
                    mask_arrays.append(np.ones(target_size, dtype=np.float32))
                    continue
                if isinstance(mask, Image.Image):
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    m_arr = np.array(mask, dtype=np.float32) / 255.0
                else:
                    m_arr = mask.astype(np.float32)
                    if m_arr.max() > 1.0:
                        m_arr = m_arr / 255.0
                mask_arrays.append(m_arr)
        else:
            # Create uniform masks
            mask_arrays = [np.ones(target_size, dtype=np.float32) for _ in negatives]

        # Default blend modes
        if blend_modes is None:
            blend_modes = [BlendMode.NORMAL] * len(negatives)

        # Start with first negative
        result = neg_arrays[0].copy()

        # Blend remaining negatives
        for i in range(1, len(neg_arrays)):
            neg = neg_arrays[i]
            mask = mask_arrays[i] if i < len(mask_arrays) else np.ones(target_size)
            mode = blend_modes[i] if i < len(blend_modes) else BlendMode.NORMAL

            blended = self._apply_blend_mode(result, neg, mode)
            result = result * (1 - mask) + blended * mask

        # Convert back to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode='L')

    def create_contrast_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: float = 0.5,
    ) -> Image.Image:
        """Create a contrast reduction mask.

        Masks high-contrast areas to reduce their density range.

        Args:
            image: Input image
            threshold: Contrast threshold (0-1)

        Returns:
            Contrast mask image
        """
        # Convert to array
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            arr = np.array(image, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

        # Calculate local contrast using gradient magnitude
        from scipy.ndimage import sobel

        dx = sobel(arr, axis=0)
        dy = sobel(arr, axis=1)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        # Normalize and threshold
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

        # Create mask (high where contrast is high)
        mask = np.clip(gradient_magnitude / threshold, 0, 1)

        # Smooth the mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))

        return mask_img

    def create_highlight_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: float = 0.7,
    ) -> Image.Image:
        """Create a highlight mask for burning highlights.

        Args:
            image: Input image
            threshold: Highlight threshold (0-1)

        Returns:
            Highlight mask (white in highlights)
        """
        # Convert to array
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            arr = np.array(image, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

        # Create mask: 0 below threshold, smooth ramp above
        mask = np.clip((arr - threshold) / (1 - threshold), 0, 1)

        # Smooth transitions
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))

        return mask_img

    def create_shadow_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: float = 0.3,
    ) -> Image.Image:
        """Create a shadow mask for dodging shadows.

        Args:
            image: Input image
            threshold: Shadow threshold (0-1)

        Returns:
            Shadow mask (white in shadows)
        """
        # Convert to array
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            arr = np.array(image, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

        # Create mask: 1 below threshold, smooth ramp to 0 above
        mask = 1.0 - np.clip((arr - 0) / threshold, 0, 1)

        # Smooth transitions
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))

        return mask_img

    def apply_dodge_burn(
        self,
        image: Union[Image.Image, np.ndarray],
        dodge_mask: Optional[Union[Image.Image, np.ndarray]] = None,
        burn_mask: Optional[Union[Image.Image, np.ndarray]] = None,
        dodge_amount: float = 0.3,
        burn_amount: float = 0.3,
    ) -> Image.Image:
        """Apply dodge and burn adjustments using masks.

        Args:
            image: Input image
            dodge_mask: Mask for dodging (lighten)
            burn_mask: Mask for burning (darken)
            dodge_amount: Dodge strength (0-1)
            burn_amount: Burn strength (0-1)

        Returns:
            Adjusted image
        """
        # Convert image to array
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            arr = np.array(image, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

        result = arr.copy()

        # Apply dodging (lighten)
        if dodge_mask is not None:
            if isinstance(dodge_mask, Image.Image):
                dodge_arr = np.array(dodge_mask.convert('L'), dtype=np.float32) / 255.0
            else:
                dodge_arr = dodge_mask.astype(np.float32)
                if dodge_arr.max() > 1.0:
                    dodge_arr = dodge_arr / 255.0

            # Dodging: move toward 1 (white)
            dodge_effect = result + (1 - result) * dodge_amount
            result = result * (1 - dodge_arr) + dodge_effect * dodge_arr

        # Apply burning (darken)
        if burn_mask is not None:
            if isinstance(burn_mask, Image.Image):
                burn_arr = np.array(burn_mask.convert('L'), dtype=np.float32) / 255.0
            else:
                burn_arr = burn_mask.astype(np.float32)
                if burn_arr.max() > 1.0:
                    burn_arr = burn_arr / 255.0

            # Burning: move toward 0 (black)
            burn_effect = result * (1 - burn_amount)
            result = result * (1 - burn_arr) + burn_effect * burn_arr

        # Convert back to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode='L')

    def create_multi_layer_mask(
        self,
        layers: list[Union[Image.Image, np.ndarray]],
        blend_modes: Optional[list[str]] = None,
    ) -> Image.Image:
        """Create complex multi-layer mask by combining multiple masks.

        Args:
            layers: List of mask layers
            blend_modes: Blend modes for each layer ('add', 'multiply', 'max', 'min')

        Returns:
            Combined mask
        """
        if not layers:
            raise ValueError("At least one layer is required")

        if blend_modes is None:
            blend_modes = ['multiply'] * len(layers)

        # Convert first layer
        if isinstance(layers[0], Image.Image):
            result = np.array(layers[0].convert('L'), dtype=np.float32) / 255.0
        else:
            result = layers[0].astype(np.float32)
            if result.max() > 1.0:
                result = result / 255.0

        # Blend remaining layers
        for i in range(1, len(layers)):
            if isinstance(layers[i], Image.Image):
                layer = np.array(layers[i].convert('L'), dtype=np.float32) / 255.0
            else:
                layer = layers[i].astype(np.float32)
                if layer.max() > 1.0:
                    layer = layer / 255.0

            mode = blend_modes[i] if i < len(blend_modes) else 'multiply'

            if mode == 'add':
                result = np.clip(result + layer, 0, 1)
            elif mode == 'multiply':
                result = result * layer
            elif mode == 'max':
                result = np.maximum(result, layer)
            elif mode == 'min':
                result = np.minimum(result, layer)
            else:
                result = result * layer  # Default to multiply

        # Convert to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode='L')

    def _apply_blend_mode(
        self,
        base: np.ndarray,
        blend: np.ndarray,
        mode: BlendMode,
    ) -> np.ndarray:
        """Apply blend mode between two images.

        Args:
            base: Base image array (0-1)
            blend: Blend image array (0-1)
            mode: Blend mode

        Returns:
            Blended result
        """
        if mode == BlendMode.NORMAL:
            return blend
        elif mode == BlendMode.MULTIPLY:
            return base * blend
        elif mode == BlendMode.SCREEN:
            return 1 - (1 - base) * (1 - blend)
        elif mode == BlendMode.OVERLAY:
            mask = base < 0.5
            result = np.zeros_like(base)
            result[mask] = 2 * base[mask] * blend[mask]
            result[~mask] = 1 - 2 * (1 - base[~mask]) * (1 - blend[~mask])
            return result
        elif mode == BlendMode.SOFT_LIGHT:
            return (1 - 2 * blend) * base**2 + 2 * blend * base
        elif mode == BlendMode.HARD_LIGHT:
            mask = blend < 0.5
            result = np.zeros_like(base)
            result[mask] = 2 * base[mask] * blend[mask]
            result[~mask] = 1 - 2 * (1 - base[~mask]) * (1 - blend[~mask])
            return result
        elif mode == BlendMode.LINEAR_DODGE:
            return np.clip(base + blend, 0, 1)
        elif mode == BlendMode.LINEAR_BURN:
            return np.clip(base + blend - 1, 0, 1)
        else:
            return blend


class QRMetadataGenerator:
    """Generate QR codes with print metadata for archival labeling.

    Creates QR codes containing complete print metadata and archival
    labels suitable for attachment to finished prints.
    """

    def __init__(self):
        """Initialize the QR metadata generator."""
        if not HAS_QRCODE:
            raise ImportError(
                "qrcode library is required for QR code generation. "
                "Install with: pip install qrcode[pil]"
            )

    def generate_print_qr(
        self,
        print_data: PrintMetadata,
        size: int = 200,
        error_correction: str = 'H',
    ) -> Image.Image:
        """Generate QR code with print metadata.

        Args:
            print_data: Print metadata to encode
            size: QR code size in pixels
            error_correction: Error correction level (L, M, Q, H)

        Returns:
            QR code image
        """
        # Encode metadata as string
        encoded = self.encode_recipe(print_data)

        # Create QR code
        qr = qrcode.QRCode(
            version=None,  # Auto-detect
            error_correction={
                'L': qrcode.constants.ERROR_CORRECT_L,
                'M': qrcode.constants.ERROR_CORRECT_M,
                'Q': qrcode.constants.ERROR_CORRECT_Q,
                'H': qrcode.constants.ERROR_CORRECT_H,
            }.get(error_correction, qrcode.constants.ERROR_CORRECT_H),
            box_size=10,
            border=2,
        )

        qr.add_data(encoded)
        qr.make(fit=True)

        # Create image
        qr_img = qr.make_image(fill_color="black", back_color="white")

        # Resize to requested size
        qr_img = qr_img.resize((size, size), Image.Resampling.NEAREST)

        return qr_img

    def encode_recipe(self, recipe: Union[PrintMetadata, dict]) -> str:
        """Encode recipe/metadata into string format.

        Args:
            recipe: PrintMetadata or dict to encode

        Returns:
            Encoded string
        """
        if isinstance(recipe, PrintMetadata):
            data = {
                'title': recipe.title,
                'artist': recipe.artist,
                'date': recipe.date,
                'edition': recipe.edition,
                'paper': recipe.paper,
                'chemistry': recipe.chemistry,
                'exposure': recipe.exposure_time,
                'developer': recipe.developer,
                'curve': recipe.curve_name,
                'dmax': recipe.dmax,
                'dmin': recipe.dmin,
                'notes': recipe.notes,
            }
        else:
            data = recipe

        # Create compact encoding
        parts = []
        for key, value in data.items():
            if value:  # Only include non-empty values
                parts.append(f"{key}:{value}")

        return "|".join(parts)

    def decode_qr(self, qr_image: Union[Image.Image, str, Path]) -> dict:
        """Decode QR code back to metadata.

        Args:
            qr_image: QR code image or path

        Returns:
            Decoded metadata dictionary
        """
        try:
            from pyzbar.pyzbar import decode as decode_qr
        except ImportError:
            raise ImportError(
                "pyzbar library is required for QR decoding. "
                "Install with: pip install pyzbar"
            )

        # Load image if path
        if isinstance(qr_image, (str, Path)):
            qr_image = Image.open(qr_image)

        # Decode QR code
        decoded = decode_qr(qr_image)

        if not decoded:
            raise ValueError("No QR code found in image")

        # Parse the data
        data_str = decoded[0].data.decode('utf-8')
        return self._parse_encoded_data(data_str)

    def create_archival_label(
        self,
        print_info: PrintMetadata,
        label_size: tuple[int, int] = (600, 300),
        qr_size: int = 180,
    ) -> Image.Image:
        """Create printable archival label with QR code and text.

        Args:
            print_info: Print metadata
            label_size: Label dimensions (width, height)
            qr_size: QR code size

        Returns:
            Label image ready for printing
        """
        # Create white background
        label = Image.new('RGB', label_size, color='white')
        draw = ImageDraw.Draw(label)

        # Generate QR code
        qr_img = self.generate_print_qr(print_info, size=qr_size)

        # Paste QR code (left side)
        qr_x = 20
        qr_y = (label_size[1] - qr_size) // 2
        label.paste(qr_img, (qr_x, qr_y))

        # Add text information (right side)
        text_x = qr_x + qr_size + 30
        text_y = 20
        line_height = 25

        # Load fonts with cross-platform support
        font_large = _get_truetype_font(16, bold=True)
        font_small = _get_truetype_font(12, bold=False)

        # Draw text fields
        y = text_y
        if print_info.title:
            draw.text((text_x, y), f"Title: {print_info.title}", fill='black', font=font_large)
            y += line_height

        if print_info.artist:
            draw.text((text_x, y), f"Artist: {print_info.artist}", fill='black', font=font_small)
            y += line_height

        if print_info.date:
            draw.text((text_x, y), f"Date: {print_info.date}", fill='black', font=font_small)
            y += line_height

        if print_info.edition:
            draw.text((text_x, y), f"Edition: {print_info.edition}", fill='black', font=font_small)
            y += line_height

        y += 10  # Extra spacing

        if print_info.paper:
            draw.text((text_x, y), f"Paper: {print_info.paper}", fill='black', font=font_small)
            y += line_height

        if print_info.chemistry:
            draw.text((text_x, y), f"Chemistry: {print_info.chemistry}", fill='black', font=font_small)
            y += line_height

        if print_info.exposure_time:
            draw.text((text_x, y), f"Exposure: {print_info.exposure_time}", fill='black', font=font_small)
            y += line_height

        if print_info.dmax > 0:
            draw.text((text_x, y), f"Dmax: {print_info.dmax:.2f}", fill='black', font=font_small)
            y += line_height

        return label

    def _parse_encoded_data(self, data_str: str) -> dict:
        """Parse encoded metadata string.

        Args:
            data_str: Encoded string

        Returns:
            Parsed dictionary
        """
        result = {}
        parts = data_str.split('|')

        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                result[key] = value

        return result


class StyleTransfer:
    """Apply historic platinum/palladium printing styles.

    Transfers the characteristic appearance of master printers
    and historic periods to contemporary images.
    """

    def __init__(self):
        """Initialize the style transfer system."""
        self.styles = self.load_historic_styles()

    def load_historic_styles(self) -> dict[str, StyleParameters]:
        """Load database of historic Pt/Pd print styles.

        Returns:
            Dictionary of style name to parameters
        """
        styles = {
            HistoricStyle.PICTORIALIST_1890S: StyleParameters(
                name="1890s Pictorialist",
                description="Soft-focus pictorialist style with gentle tones",
                gamma=0.9,
                contrast=0.85,
                toe_contrast=0.7,
                shoulder_contrast=0.8,
                shadow_tone=(30, 28, 26),
                highlight_tone=(240, 235, 225),
                dmax=1.5,
                dmin=0.12,
                paper_warmth=0.7,
                texture_strength=0.05,
            ),
            HistoricStyle.EDWARD_WESTON: StyleParameters(
                name="Edward Weston",
                description="High contrast, brilliant highlights, deep blacks",
                gamma=1.2,
                contrast=1.3,
                toe_contrast=1.4,
                shoulder_contrast=1.2,
                shadow_tone=(15, 14, 12),
                highlight_tone=(250, 248, 245),
                dmax=1.9,
                dmin=0.06,
                paper_warmth=0.3,
                texture_strength=0.0,
            ),
            HistoricStyle.IRVING_PENN: StyleParameters(
                name="Irving Penn",
                description="Clean, precise tones with subtle warmth",
                gamma=1.1,
                contrast=1.15,
                toe_contrast=1.1,
                shoulder_contrast=1.0,
                shadow_tone=(22, 20, 18),
                highlight_tone=(248, 244, 238),
                dmax=1.75,
                dmin=0.08,
                paper_warmth=0.5,
                texture_strength=0.0,
            ),
            HistoricStyle.SALLY_MANN: StyleParameters(
                name="Sally Mann",
                description="Rich, atmospheric with expanded tonal range",
                gamma=1.0,
                contrast=1.05,
                toe_contrast=0.9,
                shoulder_contrast=1.1,
                shadow_tone=(25, 22, 19),
                highlight_tone=(245, 238, 228),
                dmax=1.7,
                dmin=0.10,
                paper_warmth=0.6,
                texture_strength=0.03,
            ),
            HistoricStyle.FREDERICK_EVANS: StyleParameters(
                name="Frederick Evans",
                description="Delicate highlights, architectural precision",
                gamma=1.05,
                contrast=1.0,
                toe_contrast=1.0,
                shoulder_contrast=0.85,
                shadow_tone=(28, 26, 24),
                highlight_tone=(250, 248, 244),
                dmax=1.65,
                dmin=0.07,
                paper_warmth=0.4,
                texture_strength=0.0,
            ),
            HistoricStyle.PAUL_STRAND: StyleParameters(
                name="Paul Strand",
                description="Full tonal range, modernist clarity",
                gamma=1.15,
                contrast=1.2,
                toe_contrast=1.15,
                shoulder_contrast=1.1,
                shadow_tone=(18, 17, 15),
                highlight_tone=(248, 245, 240),
                dmax=1.85,
                dmin=0.07,
                paper_warmth=0.35,
                texture_strength=0.0,
            ),
        }

        return styles

    def analyze_style(
        self,
        reference_image: Union[Image.Image, np.ndarray],
    ) -> StyleParameters:
        """Extract style characteristics from reference image.

        Analyzes a reference print to extract its tonal characteristics,
        which can then be applied to other images.

        Args:
            reference_image: Reference print to analyze

        Returns:
            Extracted style parameters
        """
        # Convert to grayscale array
        if isinstance(reference_image, Image.Image):
            if reference_image.mode != 'L':
                gray = reference_image.convert('L')
            else:
                gray = reference_image
            arr = np.array(gray, dtype=np.float32) / 255.0
        else:
            arr = reference_image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

        # Analyze histogram
        hist, bins = np.histogram(arr, bins=256, range=(0, 1))

        # Find shadow and highlight tones (5th and 95th percentiles)
        cumsum = np.cumsum(hist)
        total = cumsum[-1]

        shadow_val = bins[np.searchsorted(cumsum, total * 0.05)]
        highlight_val = bins[np.searchsorted(cumsum, total * 0.95)]

        # Estimate gamma from median
        median_val = bins[np.searchsorted(cumsum, total * 0.5)]
        gamma = np.log(0.5) / np.log(median_val + 0.001)

        # Estimate contrast from histogram spread
        std_dev = np.std(arr)
        contrast = np.clip(std_dev * 3, 0.5, 1.5)

        # Estimate color characteristics from RGB if available
        if isinstance(reference_image, Image.Image) and reference_image.mode == 'RGB':
            rgb_arr = np.array(reference_image, dtype=np.float32) / 255.0

            # Sample shadow and highlight colors
            shadow_mask = arr < shadow_val + 0.05
            highlight_mask = arr > highlight_val - 0.05

            if shadow_mask.any():
                shadow_color = tuple(int(c * 255) for c in rgb_arr[shadow_mask].mean(axis=0))
            else:
                shadow_color = (20, 18, 16)

            if highlight_mask.any():
                highlight_color = tuple(int(c * 255) for c in rgb_arr[highlight_mask].mean(axis=0))
            else:
                highlight_color = (245, 240, 235)
        else:
            shadow_color = (20, 18, 16)
            highlight_color = (245, 240, 235)

        return StyleParameters(
            name="Analyzed Style",
            description="Extracted from reference image",
            gamma=float(gamma),
            contrast=float(contrast),
            toe_contrast=1.0,
            shoulder_contrast=1.0,
            shadow_tone=shadow_color,
            highlight_tone=highlight_color,
            dmax=float(2.0 - shadow_val * 0.5),
            dmin=float(0.05 + highlight_val * 0.1),
            paper_warmth=0.5,
            texture_strength=0.0,
        )

    def apply_style(
        self,
        image: Union[Image.Image, np.ndarray],
        style_name: Union[HistoricStyle, str],
    ) -> Image.Image:
        """Apply named historic style to image.

        Args:
            image: Input image
            style_name: Name of style to apply

        Returns:
            Styled image
        """
        # Get style parameters
        if isinstance(style_name, str):
            # Try to find matching style
            style_params = None
            for key, params in self.styles.items():
                if key == style_name or key.value == style_name:
                    style_params = params
                    break
            if style_params is None:
                raise ValueError(f"Unknown style: {style_name}")
        else:
            style_params = self.styles.get(style_name)
            if style_params is None:
                raise ValueError(f"Unknown style: {style_name}")

        return self._apply_style_params(image, style_params)

    def create_custom_style(
        self,
        name: str,
        parameters: dict[str, Any],
    ) -> StyleParameters:
        """Define a new custom style.

        Args:
            name: Style name
            parameters: Style parameters as dict

        Returns:
            Created style parameters
        """
        # Create style with defaults
        style = StyleParameters(
            name=name,
            description=parameters.get('description', 'Custom style'),
            gamma=parameters.get('gamma', 1.0),
            contrast=parameters.get('contrast', 1.0),
            toe_contrast=parameters.get('toe_contrast', 1.0),
            shoulder_contrast=parameters.get('shoulder_contrast', 1.0),
            shadow_tone=parameters.get('shadow_tone', (20, 18, 16)),
            highlight_tone=parameters.get('highlight_tone', (245, 240, 235)),
            dmax=parameters.get('dmax', 1.7),
            dmin=parameters.get('dmin', 0.08),
            paper_warmth=parameters.get('paper_warmth', 0.5),
            texture_strength=parameters.get('texture_strength', 0.0),
        )

        # Add to styles dictionary
        self.styles[name] = style

        return style

    def _apply_style_params(
        self,
        image: Union[Image.Image, np.ndarray],
        params: StyleParameters,
    ) -> Image.Image:
        """Apply style parameters to image.

        Args:
            image: Input image
            params: Style parameters

        Returns:
            Styled image
        """
        # Convert to PIL and grayscale
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='L')
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
        else:
            pil_img = image

        if pil_img.mode != 'L':
            gray = pil_img.convert('L')
        else:
            gray = pil_img

        # Convert to array
        arr = np.array(gray, dtype=np.float32) / 255.0

        # Apply gamma
        arr = np.power(arr, 1.0 / params.gamma)

        # Apply global contrast
        arr = (arr - 0.5) * params.contrast + 0.5

        # Apply split-tone contrast (toe and shoulder)
        shadow_mask = arr < 0.3
        highlight_mask = arr > 0.7

        arr[shadow_mask] = (arr[shadow_mask] - 0.15) * params.toe_contrast + 0.15
        arr[highlight_mask] = (arr[highlight_mask] - 0.85) * params.shoulder_contrast + 0.85

        arr = np.clip(arr, 0, 1)

        # Map to density range
        density_range = params.dmax - params.dmin
        density = params.dmax - (arr * density_range)

        # Convert to reflectance
        reflectance = np.power(10, -density)
        reflectance = np.clip(reflectance, 0, 1)

        # Apply color toning
        h, w = reflectance.shape
        rgb_output = np.zeros((h, w, 3), dtype=np.float32)

        shadow = np.array(params.shadow_tone, dtype=np.float32) / 255.0
        highlight = np.array(params.highlight_tone, dtype=np.float32) / 255.0

        for c in range(3):
            rgb_output[:, :, c] = shadow[c] + reflectance * (highlight[c] - shadow[c])

        # Add texture if requested
        if params.texture_strength > 0:
            np.random.seed(42)
            noise = np.random.normal(0, params.texture_strength * 0.02, (h, w))
            for c in range(3):
                rgb_output[:, :, c] = np.clip(rgb_output[:, :, c] + noise, 0, 1)

        # Convert to image
        rgb_output = (rgb_output * 255).astype(np.uint8)
        result = Image.fromarray(rgb_output, mode='RGB')

        # Store style info
        result.info['style'] = params.name

        return result


class PrintComparison:
    """Compare original images with scanned prints.

    Provides tools for analyzing print quality and consistency
    by comparing digital files with scanned prints.
    """

    def __init__(self):
        """Initialize the print comparison system."""
        pass

    def compare_before_after(
        self,
        original: Union[Image.Image, np.ndarray],
        print_scan: Union[Image.Image, np.ndarray],
    ) -> dict:
        """Compare original digital file with scanned print.

        Args:
            original: Original digital image
            print_scan: Scanned print image

        Returns:
            Comparison metrics dictionary
        """
        # Convert both to grayscale arrays
        orig_arr = self._to_gray_array(original)
        scan_arr = self._to_gray_array(print_scan)

        # Resize to match if needed
        if orig_arr.shape != scan_arr.shape:
            from scipy.ndimage import zoom
            scale_y = orig_arr.shape[0] / scan_arr.shape[0]
            scale_x = orig_arr.shape[1] / scan_arr.shape[1]
            scan_arr = zoom(scan_arr, (scale_y, scale_x), order=1)

        # Calculate metrics
        mse = np.mean((orig_arr - scan_arr) ** 2)
        rmse = np.sqrt(mse)

        # Peak signal-to-noise ratio
        max_val = 1.0
        psnr = 20 * np.log10(max_val / (rmse + 1e-10))

        # Structural similarity (simplified)
        similarity = float(np.clip(1.0 - rmse, 0.0, 1.0))

        # Histogram comparison
        orig_hist = np.histogram(orig_arr, bins=50, range=(0, 1))[0]
        scan_hist = np.histogram(scan_arr, bins=50, range=(0, 1))[0]
        hist_correlation = float(np.clip(np.corrcoef(orig_hist, scan_hist)[0, 1], -1.0, 1.0))

        # Tonal range comparison
        orig_range = float(orig_arr.max() - orig_arr.min())
        scan_range = float(scan_arr.max() - scan_arr.min())
        max_range = max(scan_range, orig_range)
        if max_range == 0:
            range_preservation = 1.0
        else:
            range_preservation = float(np.clip(min(scan_range, orig_range) / max_range, 0.0, 1.0))

        return {
            'rmse': float(rmse),
            'psnr': float(psnr),
            'similarity_score': float(similarity),
            'histogram_correlation': float(hist_correlation),
            'tonal_range_preservation': float(range_preservation),
            'original_range': orig_range,
            'print_range': scan_range,
        }

    def generate_difference_map(
        self,
        image1: Union[Image.Image, np.ndarray],
        image2: Union[Image.Image, np.ndarray],
        colorize: bool = True,
    ) -> Image.Image:
        """Generate visual difference map between two images.

        Args:
            image1: First image
            image2: Second image
            colorize: Whether to colorize differences (red/blue)

        Returns:
            Difference map image
        """
        # Convert to arrays
        arr1 = self._to_gray_array(image1)
        arr2 = self._to_gray_array(image2)

        # Resize to match if needed
        if arr1.shape != arr2.shape:
            from scipy.ndimage import zoom
            scale_y = arr2.shape[0] / arr1.shape[0]
            scale_x = arr2.shape[1] / arr1.shape[1]
            arr2 = zoom(arr2, (scale_y, scale_x), order=1)

        # Calculate difference
        diff = arr1 - arr2

        if colorize:
            # Create RGB difference map
            h, w = diff.shape
            rgb_diff = np.zeros((h, w, 3), dtype=np.float32)

            # Positive differences (image1 lighter) = red
            # Negative differences (image2 lighter) = blue
            # No difference = gray

            pos_mask = diff > 0
            neg_mask = diff < 0

            # Gray base
            rgb_diff[:, :, 0] = 0.5
            rgb_diff[:, :, 1] = 0.5
            rgb_diff[:, :, 2] = 0.5

            # Add red for positive diff
            rgb_diff[pos_mask, 0] = 0.5 + np.abs(diff[pos_mask]) * 0.5
            rgb_diff[pos_mask, 1] = 0.5 - np.abs(diff[pos_mask]) * 0.5
            rgb_diff[pos_mask, 2] = 0.5 - np.abs(diff[pos_mask]) * 0.5

            # Add blue for negative diff
            rgb_diff[neg_mask, 0] = 0.5 - np.abs(diff[neg_mask]) * 0.5
            rgb_diff[neg_mask, 1] = 0.5 - np.abs(diff[neg_mask]) * 0.5
            rgb_diff[neg_mask, 2] = 0.5 + np.abs(diff[neg_mask]) * 0.5

            rgb_diff = np.clip(rgb_diff * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb_diff, mode='RGB')
        else:
            # Grayscale difference
            diff_normalized = (np.abs(diff) * 255).astype(np.uint8)
            return Image.fromarray(diff_normalized, mode='L')

    def calculate_similarity_score(
        self,
        image1: Union[Image.Image, np.ndarray],
        image2: Union[Image.Image, np.ndarray],
        method: str = 'ssim',
    ) -> float:
        """Calculate numerical similarity score between images.

        Args:
            image1: First image
            image2: Second image
            method: Similarity method ('ssim', 'mse', 'correlation')

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Convert to arrays
        arr1 = self._to_gray_array(image1)
        arr2 = self._to_gray_array(image2)

        # Resize to match if needed
        if arr1.shape != arr2.shape:
            from scipy.ndimage import zoom
            scale_y = arr2.shape[0] / arr1.shape[0]
            scale_x = arr2.shape[1] / arr1.shape[1]
            arr2 = zoom(arr2, (scale_y, scale_x), order=1)

        if method == 'mse':
            mse = np.mean((arr1 - arr2) ** 2)
            return float(np.clip(1.0 - np.sqrt(mse), 0.0, 1.0))

        elif method == 'correlation':
            corr = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
            return float(np.clip((corr + 1) / 2, 0.0, 1.0))  # Map from [-1, 1] to [0, 1]

        elif method == 'ssim':
            # Simplified SSIM calculation
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2

            mu1 = arr1.mean()
            mu2 = arr2.mean()

            sigma1 = arr1.std()
            sigma2 = arr2.std()

            sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))

            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

            return float(np.clip((ssim + 1) / 2, 0.0, 1.0))  # Normalize to 0-1

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def generate_comparison_report(
        self,
        images: dict[str, Union[Image.Image, np.ndarray]],
        reference_key: Optional[str] = None,
    ) -> dict:
        """Generate comprehensive comparison report for multiple images.

        Args:
            images: Dictionary of image_name -> image
            reference_key: Key of reference image (if None, uses first)

        Returns:
            Detailed comparison report
        """
        if not images:
            raise ValueError("At least one image is required")

        # Get reference image
        if reference_key is None:
            reference_key = list(images.keys())[0]

        if reference_key not in images:
            raise ValueError(f"Reference key {reference_key} not in images")

        reference = images[reference_key]
        ref_arr = self._to_gray_array(reference)

        # Compare all images to reference
        comparisons = {}

        for name, img in images.items():
            if name == reference_key:
                continue

            # Calculate various metrics
            comparison = self.compare_before_after(reference, img)
            comparison['similarity_ssim'] = self.calculate_similarity_score(
                reference, img, method='ssim'
            )
            comparison['similarity_correlation'] = self.calculate_similarity_score(
                reference, img, method='correlation'
            )

            comparisons[name] = comparison

        # Generate summary statistics
        if comparisons:
            avg_similarity = np.mean([c['similarity_ssim'] for c in comparisons.values()])
            avg_psnr = np.mean([c['psnr'] for c in comparisons.values()])
        else:
            avg_similarity = 1.0
            avg_psnr = float('inf')

        return {
            'reference': reference_key,
            'num_images': len(images),
            'comparisons': comparisons,
            'summary': {
                'average_similarity': float(avg_similarity),
                'average_psnr': float(avg_psnr),
            },
        }

    def _to_gray_array(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Convert image to grayscale float array.

        Args:
            image: Input image

        Returns:
            Normalized grayscale array (0-1)
        """
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            arr = np.array(gray, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.ndim == 3:
                # Convert RGB to grayscale
                arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            if arr.max() > 1.0:
                arr = arr / 255.0

        return arr
