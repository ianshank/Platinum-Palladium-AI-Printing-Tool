"""
Tests for UI validators module.
"""

import pytest
from pathlib import Path

from ptpd_calibration.ui.validators import (
    CurveValidator,
    DensityValidator,
    FileValidator,
    ValidationError,
)


class TestDensityValidator:
    """Test DensityValidator."""

    def test_density_validator_creation(self) -> None:
        """Test density validator initialization."""
        validator = DensityValidator()
        assert validator.min_value == 0.0
        assert validator.max_value == 3.0

    def test_density_validator_custom_range(self) -> None:
        """Test density validator with custom range."""
        validator = DensityValidator(min_value=0.1, max_value=2.5)
        assert validator.min_value == 0.1
        assert validator.max_value == 2.5

    def test_validate_single_valid(self) -> None:
        """Test validating a single valid density value."""
        validator = DensityValidator()
        result = validator.validate_single(1.5)
        assert result == 1.5

    def test_validate_single_string(self) -> None:
        """Test validating a string density value."""
        validator = DensityValidator()
        result = validator.validate_single("1.5")
        assert result == 1.5

    def test_validate_single_boundary_values(self) -> None:
        """Test boundary values."""
        validator = DensityValidator()
        assert validator.validate_single(0.0) == 0.0
        assert validator.validate_single(3.0) == 3.0

    def test_validate_single_below_min(self) -> None:
        """Test value below minimum."""
        validator = DensityValidator()
        with pytest.raises(ValidationError):
            validator.validate_single(-0.1)

    def test_validate_single_above_max(self) -> None:
        """Test value above maximum."""
        validator = DensityValidator()
        with pytest.raises(ValidationError):
            validator.validate_single(3.1)

    def test_validate_single_invalid_string(self) -> None:
        """Test invalid string value."""
        validator = DensityValidator()
        with pytest.raises(ValidationError):
            validator.validate_single("not_a_number")

    def test_validate_curve_valid(self) -> None:
        """Test validating a valid curve."""
        validator = DensityValidator()
        values = [0.0, 0.5, 1.0, 1.5, 2.0]
        result = validator.validate_curve(values)
        assert result == values
        assert len(result) == 5

    def test_validate_curve_with_strings(self) -> None:
        """Test validating curve with string values."""
        validator = DensityValidator()
        values = ["0.0", "0.5", "1.0"]
        result = validator.validate_curve(values)
        assert result == [0.0, 0.5, 1.0]

    def test_validate_curve_with_invalid_value(self) -> None:
        """Test curve validation with one invalid value."""
        validator = DensityValidator()
        values = [0.0, 0.5, 5.0]  # 5.0 is out of range
        with pytest.raises(ValidationError):
            validator.validate_curve(values)


class TestFileValidator:
    """Test FileValidator."""

    def test_validate_extension_allowed(self) -> None:
        """Test validation of allowed file extensions."""
        assert FileValidator.validate_extension("curve.quad")
        assert FileValidator.validate_extension("data.csv")
        assert FileValidator.validate_extension("export.json")
        assert FileValidator.validate_extension("profile.txt")

    def test_validate_extension_case_insensitive(self) -> None:
        """Test that extension validation is case-insensitive."""
        assert FileValidator.validate_extension("curve.QUAD")
        assert FileValidator.validate_extension("data.CSV")
        assert FileValidator.validate_extension("export.JSON")

    def test_validate_extension_not_allowed(self) -> None:
        """Test validation of disallowed extensions."""
        assert not FileValidator.validate_extension("image.png")
        assert not FileValidator.validate_extension("document.pdf")
        assert not FileValidator.validate_extension("script.py")

    def test_validate_file(self, tmp_path: Path) -> None:
        """Test complete file validation."""
        # Create a temporary file
        test_file = tmp_path / "test.quad"
        test_file.write_text("test content")

        is_valid, message = FileValidator.validate_file(test_file)
        assert is_valid
        assert message == "File is valid"

    def test_validate_file_not_found(self) -> None:
        """Test validation of non-existent file."""
        is_valid, message = FileValidator.validate_file("/nonexistent/file.quad")
        assert not is_valid
        assert "not found" in message.lower() or "not readable" in message.lower()

    def test_validate_file_wrong_extension(self, tmp_path: Path) -> None:
        """Test validation of file with wrong extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        is_valid, message = FileValidator.validate_file(test_file)
        # .txt is allowed
        assert is_valid

    def test_validate_file_unsupported_extension(self, tmp_path: Path) -> None:
        """Test validation of file with unsupported extension."""
        test_file = tmp_path / "test.png"
        test_file.write_text("test content")

        is_valid, message = FileValidator.validate_file(test_file)
        assert not is_valid
        assert "unsupported" in message.lower()


class TestCurveValidator:
    """Test CurveValidator."""

    def test_curve_validator_creation(self) -> None:
        """Test curve validator initialization."""
        validator = CurveValidator()
        assert validator.min_points == 2
        assert validator.max_points == 1000

    def test_curve_validator_custom_limits(self) -> None:
        """Test curve validator with custom limits."""
        validator = CurveValidator(min_points=5, max_points=500)
        assert validator.min_points == 5
        assert validator.max_points == 500

    def test_validate_curve_data_valid(self) -> None:
        """Test validating valid curve data."""
        validator = CurveValidator()
        inputs = [0.0, 0.25, 0.5, 0.75, 1.0]
        outputs = [0.0, 0.3, 0.6, 0.85, 1.0]

        result = validator.validate_curve_data(inputs, outputs)
        assert result is True

    def test_validate_curve_data_length_mismatch(self) -> None:
        """Test validation with mismatched input/output lengths."""
        validator = CurveValidator()
        inputs = [0.0, 0.5, 1.0]
        outputs = [0.0, 0.5]  # Wrong length

        with pytest.raises(ValidationError):
            validator.validate_curve_data(inputs, outputs)

    def test_validate_curve_data_too_few_points(self) -> None:
        """Test validation with too few points."""
        validator = CurveValidator(min_points=5)
        inputs = [0.0, 1.0]  # Only 2 points
        outputs = [0.0, 1.0]

        with pytest.raises(ValidationError):
            validator.validate_curve_data(inputs, outputs)

    def test_validate_curve_data_too_many_points(self) -> None:
        """Test validation with too many points."""
        validator = CurveValidator(max_points=10)
        inputs = list(range(20))
        outputs = list(range(20))

        with pytest.raises(ValidationError):
            validator.validate_curve_data(inputs, outputs)

    def test_validate_curve_data_non_monotonic(self) -> None:
        """Test validation with non-monotonic input."""
        validator = CurveValidator()
        inputs = [0.0, 0.5, 0.3, 1.0]  # Not strictly increasing
        outputs = [0.0, 0.5, 0.6, 1.0]

        with pytest.raises(ValidationError):
            validator.validate_curve_data(inputs, outputs)

    def test_validate_curve_data_with_context(self) -> None:
        """Test validation with context for logging."""
        validator = CurveValidator()
        inputs = [0.0, 0.5, 1.0]
        outputs = [0.0, 0.5, 1.0]
        context = {"curve_name": "test_curve", "source": "manual"}

        result = validator.validate_curve_data(inputs, outputs, context)
        assert result is True
