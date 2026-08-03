"""
Base validation utilities for UI components.

Provides validation patterns for density values, file uploads, and parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class DensityValidator:
    """Validates density measurements and curve data."""

    def __init__(self, min_value: float = 0.0, max_value: float = 3.0):
        """Initialize density validator.

        Args:
            min_value: Minimum valid density value.
            max_value: Maximum valid density value.
        """
        self.min_value = min_value
        self.max_value = max_value

    def validate_single(self, value: Union[float, str]) -> float:
        """Validate a single density value.

        Args:
            value: Value to validate (float or string representation).

        Returns:
            Validated density value.

        Raises:
            ValidationError: If value is invalid.
        """
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid density value: {value}") from e

        if not self.min_value <= float_value <= self.max_value:
            raise ValidationError(
                f"Density {float_value} out of range [{self.min_value}, {self.max_value}]"
            )

        return float_value

    def validate_curve(self, values: List[Union[float, str]]) -> List[float]:
        """Validate a list of density values.

        Args:
            values: List of values to validate.

        Returns:
            List of validated density values.

        Raises:
            ValidationError: If any value is invalid.
        """
        validated = []
        for i, v in enumerate(values):
            try:
                validated.append(self.validate_single(v))
            except ValidationError as e:
                raise ValidationError(f"Value at index {i}: {str(e)}") from e

        return validated


class FileValidator:
    """Validates file uploads."""

    ALLOWED_EXTENSIONS = {".quad", ".txt", ".csv", ".json", ".acv"}

    @staticmethod
    def validate_extension(file_path: Union[str, Path]) -> bool:
        """Check if file extension is allowed.

        Args:
            file_path: Path to file.

        Returns:
            True if extension is allowed.
        """
        path = Path(file_path)
        return path.suffix.lower() in FileValidator.ALLOWED_EXTENSIONS

    @staticmethod
    def validate_readable(file_path: Union[str, Path]) -> bool:
        """Check if file is readable.

        Args:
            file_path: Path to file.

        Returns:
            True if file is readable.
        """
        path = Path(file_path)
        return path.exists() and path.is_file()

    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate a file completely.

        Args:
            file_path: Path to file.

        Returns:
            Tuple of (is_valid, message).
        """
        if not FileValidator.validate_readable(file_path):
            return False, "File not found or not readable"

        if not FileValidator.validate_extension(file_path):
            path = Path(file_path)
            return False, f"Unsupported file type: {path.suffix}"

        return True, "File is valid"


class CurveValidator:
    """Validates curve data and parameters."""

    def __init__(self, min_points: int = 2, max_points: int = 1000):
        """Initialize curve validator.

        Args:
            min_points: Minimum number of points required.
            max_points: Maximum number of points allowed.
        """
        self.min_points = min_points
        self.max_points = max_points

    def validate_curve_data(
        self,
        inputs: List[float],
        outputs: List[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate curve input/output data.

        Args:
            inputs: Input values (typically 0-1 range).
            outputs: Output values (density or tone values).
            context: Optional context for error logging.

        Returns:
            True if curve data is valid.

        Raises:
            ValidationError: If data is invalid.
        """
        if len(inputs) != len(outputs):
            raise ValidationError("Input and output length mismatch")

        if not self.min_points <= len(inputs) <= self.max_points:
            raise ValidationError(
                f"Points {len(inputs)} out of range [{self.min_points}, {self.max_points}]"
            )

        # Check monotonicity
        for i in range(len(inputs) - 1):
            if inputs[i] >= inputs[i + 1]:
                raise ValidationError(f"Inputs not strictly increasing at index {i}")

        logger.debug(f"Curve validation passed: {len(inputs)} points")
        return True


__all__ = [
    "ValidationError",
    "DensityValidator",
    "FileValidator",
    "CurveValidator",
]
