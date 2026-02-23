"""
ICC profile management for color-managed workflow.

Provides tools for loading, applying, creating, and managing ICC color profiles
for accurate color reproduction in platinum/palladium printing workflow.
"""

import io
import logging
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ColorSpace(str, Enum):
    """Color space types."""

    RGB = "RGB"
    CMYK = "CMYK"
    LAB = "LAB"
    GRAY = "GRAY"
    XYZ = "XYZ"


class RenderingIntent(str, Enum):
    """ICC rendering intents."""

    PERCEPTUAL = "perceptual"
    RELATIVE_COLORIMETRIC = "relative"
    SATURATION = "saturation"
    ABSOLUTE_COLORIMETRIC = "absolute"


class ProfileClass(str, Enum):
    """ICC profile classes."""

    INPUT = "input"  # Scanner, camera
    DISPLAY = "display"  # Monitor
    OUTPUT = "output"  # Printer
    DEVICE_LINK = "device_link"
    COLOR_SPACE = "color_space"
    ABSTRACT = "abstract"
    NAMED_COLOR = "named_color"


@dataclass
class ProfileInfo:
    """ICC profile metadata."""

    path: Path
    description: str
    copyright: str
    color_space: ColorSpace
    profile_class: ProfileClass
    size_bytes: int
    creation_date: str | None = None
    manufacturer: str | None = None
    model: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "description": self.description,
            "copyright": self.copyright,
            "color_space": self.color_space.value,
            "profile_class": self.profile_class.value,
            "size_bytes": self.size_bytes,
            "creation_date": self.creation_date,
            "manufacturer": self.manufacturer,
            "model": self.model,
        }


class ProfileValidation(BaseModel):
    """Profile validation result."""

    is_valid: bool
    profile_path: Path
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    info: ProfileInfo | None = None

    class Config:
        arbitrary_types_allowed = True


class ICCProfileManager:
    """
    ICC profile management system.

    Handles loading, applying, creating, and validating ICC color profiles
    for accurate color workflow in printing.
    """

    # Standard ICC profile directories by platform
    SYSTEM_PROFILE_DIRS = {
        "Darwin": [  # macOS
            Path("/Library/ColorSync/Profiles"),
            Path("/System/Library/ColorSync/Profiles"),
            Path.home() / "Library/ColorSync/Profiles",
        ],
        "Linux": [
            Path("/usr/share/color/icc"),
            Path("/usr/local/share/color/icc"),
            Path.home() / ".color/icc",
        ],
        "Windows": [Path("C:/Windows/System32/spool/drivers/color")],
    }

    def __init__(self, custom_profile_dir: Path | None = None):
        """
        Initialize ICC profile manager.

        Args:
            custom_profile_dir: Additional directory to search for profiles
        """
        self.custom_profile_dir = Path(custom_profile_dir) if custom_profile_dir else None
        self._profile_cache: dict[str, ImageCms.ImageCmsProfile] = {}

    def load_profile(self, path: Path) -> ImageCms.ImageCmsProfile:
        """
        Load ICC profile from file.

        Args:
            path: Path to ICC profile

        Returns:
            ImageCmsProfile object

        Raises:
            FileNotFoundError: If profile file not found
            ValueError: If profile is invalid
        """
        path = Path(path)

        # Check cache
        cache_key = str(path.resolve())
        if cache_key in self._profile_cache:
            logger.debug(f"Returning cached profile: {path.name}")
            return self._profile_cache[cache_key]

        # Load profile
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        try:
            # Read profile bytes into memory so Windows can release the file handle
            profile_bytes = path.read_bytes()
            profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
            # Keep a reference to the bytes to prevent garbage collection while cached
            profile._profile_bytes = profile_bytes  # type: ignore[attr-defined]
            self._profile_cache[cache_key] = profile
            logger.info(f"Loaded ICC profile: {path.name}")
            return profile

        except Exception as e:
            raise ValueError(f"Failed to load profile {path}: {e}") from e

    def apply_profile(
        self,
        image: Image.Image,
        profile: ImageCms.ImageCmsProfile,
        rendering_intent: RenderingIntent = RenderingIntent.PERCEPTUAL,
    ) -> Image.Image:
        """
        Apply ICC profile to image.

        Args:
            image: PIL Image
            profile: ICC profile to apply
            rendering_intent: Rendering intent to use

        Returns:
            Image with profile applied
        """
        # Map our enum to PIL constants
        intent_map = {
            RenderingIntent.PERCEPTUAL: ImageCms.Intent.PERCEPTUAL,
            RenderingIntent.RELATIVE_COLORIMETRIC: ImageCms.Intent.RELATIVE_COLORIMETRIC,
            RenderingIntent.SATURATION: ImageCms.Intent.SATURATION,
            RenderingIntent.ABSOLUTE_COLORIMETRIC: ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
        }

        intent = intent_map[rendering_intent]

        # Get source profile (embedded or sRGB default)
        try:
            if "icc_profile" in image.info:
                import io

                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"]))
            else:
                # Use sRGB as default source
                source_profile = ImageCms.createProfile("sRGB")

            # Create transform
            transform = ImageCms.buildTransform(
                source_profile, profile, image.mode, image.mode, renderingIntent=intent, flags=0
            )

            # Apply transform
            result = ImageCms.applyTransform(image, transform)

            logger.debug(f"Applied ICC profile with {rendering_intent.value} intent")
            return result

        except Exception as e:
            logger.error(f"Failed to apply profile: {e}")
            logger.warning("Returning original image")
            return image

    def create_paper_profile(
        self,
        measurements: list[tuple[np.ndarray, np.ndarray]],
        profile_path: Path,
        paper_name: str = "Custom Paper",
        white_point: tuple[float, float, float] | None = None,
    ) -> Path:
        """
        Create custom ICC profile from measurement data.

        Args:
            measurements: List of (RGB input, LAB measured) tuples
            profile_path: Path to save profile
            paper_name: Descriptive name for profile
            white_point: Paper white point in XYZ (or None for D50)

        Returns:
            Path to created profile

        Note:
            This is a simplified profile creation. For production use,
            consider using dedicated profiling tools like Argyll CMS.
        """
        logger.info(f"Creating custom ICC profile: {paper_name}")

        if len(measurements) < 24:
            logger.warning(
                f"Only {len(measurements)} patches - recommend at least 24 for accurate profile"
            )

        # Use default D50 white point if not provided
        if white_point is None:
            white_point = (0.9642, 1.0000, 0.8251)  # D50

        try:
            # For a real implementation, would need to:
            # 1. Build lookup tables from measurements
            # 2. Create proper ICC profile structure
            # 3. Write binary ICC format

            # Simplified approach: create a basic profile
            # In production, use Argyll CMS or similar

            # Create a basic sRGB profile as template
            _profile = ImageCms.createProfile("sRGB")  # Template for future customization

            # Save with custom description
            # Note: PIL doesn't provide full ICC profile creation,
            # so this is a placeholder for the concept

            profile_path = Path(profile_path)
            profile_path.parent.mkdir(parents=True, exist_ok=True)

            # For now, save the template profile
            # Real implementation would build custom profile from measurements
            logger.warning(
                "Full profile creation requires external tools (e.g., Argyll CMS). "
                "Creating template profile."
            )

            # Write a marker file indicating this is a template
            with open(profile_path, "wb") as f:
                # This is a placeholder - real ICC profile creation
                # requires proper binary structure per ICC spec
                f.write(b"ICC_PROFILE_PLACEHOLDER\n")
                f.write(f"Paper: {paper_name}\n".encode())
                f.write(f"Patches: {len(measurements)}\n".encode())

            logger.info(f"Profile template created: {profile_path}")
            logger.info("For production use, process measurements with Argyll CMS or similar tool")

            return profile_path

        except Exception as e:
            logger.error(f"Failed to create profile: {e}")
            raise

    def list_installed_profiles(
        self,
        color_space: ColorSpace | None = None,
        profile_class: ProfileClass | None = None,
    ) -> list[ProfileInfo]:
        """
        List installed ICC profiles on the system.

        Args:
            color_space: Filter by color space (optional)
            profile_class: Filter by profile class (optional)

        Returns:
            List of ProfileInfo objects
        """
        profiles = []

        # Get profile directories for current platform
        system = platform.system()
        profile_dirs = self.SYSTEM_PROFILE_DIRS.get(system, [])

        # Add custom directory if provided
        if self.custom_profile_dir:
            profile_dirs.append(self.custom_profile_dir)

        # Search for .icc and .icm files
        for directory in profile_dirs:
            if not directory.exists():
                continue

            for pattern in ["*.icc", "*.icm", "*.ICC", "*.ICM"]:
                for profile_path in directory.glob(pattern):
                    try:
                        info = self._get_profile_info(profile_path)

                        # Apply filters
                        if color_space and info.color_space != color_space:
                            continue
                        if profile_class and info.profile_class != profile_class:
                            continue

                        profiles.append(info)

                    except Exception as e:
                        logger.debug(f"Skipping invalid profile {profile_path}: {e}")

        logger.info(f"Found {len(profiles)} ICC profiles")
        return profiles

    def _get_profile_info(self, path: Path) -> ProfileInfo:
        """Extract information from ICC profile."""
        try:
            profile = self.load_profile(path)

            # Get profile description
            description = ImageCms.getProfileDescription(profile) or "Unknown"
            copyright_info = ImageCms.getProfileCopyright(profile) or "Unknown"

            # Get color space
            color_space_str = ImageCms.getProfileColorSpace(profile)
            color_space_map = {
                "RGB": ColorSpace.RGB,
                "CMYK": ColorSpace.CMYK,
                "LAB": ColorSpace.LAB,
                "GRAY": ColorSpace.GRAY,
                "XYZ": ColorSpace.XYZ,
            }
            color_space = color_space_map.get(color_space_str, ColorSpace.RGB)

            # Read profile class from ICC header (byte offset 12, 4-byte signature)
            profile_class = self._read_profile_class_from_header(path)

            # Fall back to description-based detection if header read fails
            if profile_class is None:
                logger.debug(f"Falling back to description-based class detection for {path.name}")
                desc_lower = description.lower()
                if any(x in desc_lower for x in ["printer", "output"]):
                    profile_class = ProfileClass.OUTPUT
                elif any(x in desc_lower for x in ["display", "monitor"]):
                    profile_class = ProfileClass.DISPLAY
                elif any(x in desc_lower for x in ["scanner", "camera", "input"]):
                    profile_class = ProfileClass.INPUT
                else:
                    profile_class = ProfileClass.COLOR_SPACE

            return ProfileInfo(
                path=path,
                description=description,
                copyright=copyright_info,
                color_space=color_space,
                profile_class=profile_class,
                size_bytes=path.stat().st_size,
            )

        except Exception as e:
            raise ValueError(f"Failed to read profile info: {e}") from e

    def _read_profile_class_from_header(self, path: Path) -> ProfileClass | None:
        """
        Read ICC profile class directly from header.

        The ICC specification defines a Profile/Device Class field at byte offset 12
        containing a 4-byte signature.

        Args:
            path: Path to ICC profile file

        Returns:
            ProfileClass enum value, or None if reading fails
        """
        try:
            with open(path, "rb") as f:
                # Skip to byte offset 12 (profile class signature)
                f.seek(12)
                class_signature = f.read(4)

                if len(class_signature) != 4:
                    logger.warning(f"Could not read profile class signature from {path.name}")
                    return None

                # Map ICC profile class signatures to ProfileClass enum
                signature_map = {
                    b"scnr": ProfileClass.INPUT,  # Scanner/Input
                    b"mntr": ProfileClass.DISPLAY,  # Display/Monitor
                    b"prtr": ProfileClass.OUTPUT,  # Printer/Output
                    b"link": ProfileClass.DEVICE_LINK,  # Device Link
                    b"spac": ProfileClass.COLOR_SPACE,  # Color Space
                    b"abst": ProfileClass.ABSTRACT,  # Abstract
                    b"nmcl": ProfileClass.NAMED_COLOR,  # Named Color
                }

                profile_class = signature_map.get(class_signature)

                if profile_class:
                    logger.debug(
                        f"Read profile class from header: {class_signature.decode('ascii', errors='ignore')} "
                        f"-> {profile_class.value}"
                    )
                else:
                    logger.warning(
                        f"Unknown profile class signature: {class_signature.hex()} in {path.name}"
                    )

                return profile_class

        except Exception as e:
            logger.debug(f"Failed to read profile class from header: {e}")
            return None

    def convert_colorspace(
        self,
        image: Image.Image,
        source_profile: ImageCms.ImageCmsProfile,
        target_profile: ImageCms.ImageCmsProfile,
        rendering_intent: RenderingIntent = RenderingIntent.PERCEPTUAL,
    ) -> Image.Image:
        """
        Convert image between color spaces using ICC profiles.

        Args:
            image: Source image
            source_profile: Source ICC profile
            target_profile: Target ICC profile
            rendering_intent: Rendering intent

        Returns:
            Converted image
        """
        # Map rendering intent
        intent_map = {
            RenderingIntent.PERCEPTUAL: ImageCms.Intent.PERCEPTUAL,
            RenderingIntent.RELATIVE_COLORIMETRIC: ImageCms.Intent.RELATIVE_COLORIMETRIC,
            RenderingIntent.SATURATION: ImageCms.Intent.SATURATION,
            RenderingIntent.ABSOLUTE_COLORIMETRIC: ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
        }

        intent = intent_map[rendering_intent]

        try:
            # Determine output mode based on target profile color space
            source_mode = image.mode
            target_space = ImageCms.getProfileColorSpace(target_profile)

            mode_map = {
                "RGB": "RGB",
                "CMYK": "CMYK",
                "LAB": "LAB",
                "GRAY": "L",
            }
            target_mode = mode_map.get(target_space, source_mode)

            # Create transform
            transform = ImageCms.buildTransform(
                source_profile,
                target_profile,
                source_mode,
                target_mode,
                renderingIntent=intent,
                flags=0,
            )

            # Apply transform
            result = ImageCms.applyTransform(image, transform)

            logger.info(
                f"Converted color space: {source_mode} -> {target_mode} ({rendering_intent.value})"
            )

            return result

        except Exception as e:
            logger.error(f"Color space conversion failed: {e}")
            raise

    def validate_profile(self, profile_path: Path) -> ProfileValidation:
        """
        Validate ICC profile structure and content.

        Args:
            profile_path: Path to profile to validate

        Returns:
            ProfileValidation with validation results
        """
        profile_path = Path(profile_path)
        errors = []
        warnings = []
        info = None

        # Check file exists
        if not profile_path.exists():
            errors.append(f"File not found: {profile_path}")
            return ProfileValidation(is_valid=False, profile_path=profile_path, errors=errors)

        # Check file size
        size = profile_path.stat().st_size
        if size < 128:
            errors.append(f"File too small ({size} bytes) - not a valid ICC profile")
        elif size > 10 * 1024 * 1024:  # 10MB
            warnings.append(f"Large profile file ({size / 1024 / 1024:.1f} MB)")

        # Check ICC signature
        try:
            with open(profile_path, "rb") as f:
                # ICC profiles start with size (4 bytes) and signature
                header = f.read(128)

                if len(header) < 128:
                    errors.append("File too short to be valid ICC profile")
                else:
                    # Check profile signature at offset 36-40
                    signature = header[36:40]
                    if signature != b"acsp":
                        errors.append(f"Invalid ICC signature: {signature!r} (expected b'acsp')")

        except Exception as e:
            errors.append(f"Failed to read file: {e}")

        # Try to load with PIL
        try:
            _profile = self.load_profile(profile_path)  # Validates profile can be loaded
            info = self._get_profile_info(profile_path)

            # Additional validation
            if info.size_bytes != size:
                warnings.append("Profile size mismatch")

        except Exception as e:
            errors.append(f"Failed to load as ICC profile: {e}")

        is_valid = len(errors) == 0

        return ProfileValidation(
            is_valid=is_valid,
            profile_path=profile_path,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    def embed_profile(self, image: Image.Image, profile: ImageCms.ImageCmsProfile) -> Image.Image:
        """
        Embed ICC profile in image.

        Args:
            image: Image to embed profile in
            profile: ICC profile to embed

        Returns:
            Image with embedded profile
        """
        try:
            # Get profile data
            profile_bytes = profile.tobytes()

            # Create new image with profile embedded
            result = image.copy()
            result.info["icc_profile"] = profile_bytes

            logger.debug("Embedded ICC profile in image")
            return result

        except Exception as e:
            logger.error(f"Failed to embed profile: {e}")
            return image

    def extract_profile(self, image: Image.Image) -> ImageCms.ImageCmsProfile | None:
        """
        Extract embedded ICC profile from image.

        Args:
            image: Image to extract profile from

        Returns:
            ImageCmsProfile if found, None otherwise
        """
        try:
            if "icc_profile" in image.info:
                import io

                profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"]))
                logger.debug("Extracted embedded ICC profile")
                return profile
            else:
                logger.debug("No embedded ICC profile found")
                return None

        except Exception as e:
            logger.error(f"Failed to extract profile: {e}")
            return None

    def get_default_rgb_profile(self) -> ImageCms.ImageCmsProfile:
        """Get default sRGB profile."""
        return ImageCms.createProfile("sRGB")

    def get_default_gray_profile(self) -> ImageCms.ImageCmsProfile:
        """Get default grayscale profile (Gray Gamma 2.2)."""
        # Create a grayscale profile with 2.2 gamma
        # This is appropriate for digital negatives
        return ImageCms.createProfile("sGRAY")

    def get_default_lab_profile(self) -> ImageCms.ImageCmsProfile:
        """Get default L*a*b* profile."""
        return ImageCms.createProfile("LAB")
