"""
ACV (Adobe Curve) file export for AlphaZero results.

Exports optimized linearization curves to Photoshop .acv format,
which can be loaded in Photoshop, Lightroom, and other software.
"""

import struct
from pathlib import Path

import numpy as np


class ACVExporter:
    """
    Exporter for Adobe Photoshop Curve (.acv) files.

    ACV files contain one or more curves that map input values
    to output values. Each curve has up to 16 control points.
    """

    # ACV file format constants
    ACV_VERSION = 4
    MAX_POINTS = 16
    NUM_CHANNELS = 5  # Master + RGBK

    def __init__(self):
        """Initialize the ACV exporter."""
        pass

    def export(
        self,
        densities: np.ndarray,
        output_path: Path,
        curve_name: str | None = None,  # noqa: ARG002 - reserved for future metadata
    ) -> Path:
        """
        Export density values to an ACV file.

        The densities are converted to a Photoshop-compatible curve
        that linearizes the printing process.

        Args:
            densities: Predicted density values (21 steps)
            output_path: Path for the output .acv file
            curve_name: Optional name for the curve (reserved for future use)

        Returns:
            Path to the created file
        """
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".acv":
            output_path = output_path.with_suffix(".acv")

        # Generate correction curve from densities
        curve_points = self._densities_to_curve(densities)

        # Write ACV file
        with open(output_path, "wb") as f:
            self._write_acv(f, curve_points)

        return output_path

    def _densities_to_curve(
        self,
        densities: np.ndarray,
        num_points: int = 16,
    ) -> list[tuple[int, int]]:
        """
        Convert density values to ACV curve control points.

        Creates a correction curve that linearizes the response.
        The curve maps what the printer produces (actual) to what
        we want it to produce (target).

        Args:
            densities: Measured/predicted density values
            num_points: Number of control points (max 16)

        Returns:
            List of (input, output) control points (0-255 range)
        """
        num_steps = len(densities)

        # Normalize densities to 0-1
        dmin = densities.min()
        dmax = densities.max()
        if dmax - dmin > 0.01:
            normalized = (densities - dmin) / (dmax - dmin)
        else:
            normalized = np.linspace(0, 1, num_steps)

        # Target is linear response
        target = np.linspace(0, 1, num_steps)

        # Create correction: for each target, find what input gives that response
        # This inverts the characteristic curve
        correction_curve = []
        sample_indices = np.linspace(0, num_steps - 1, num_points, dtype=int)

        for idx in sample_indices:
            target_val = target[idx]

            # Find input that produces this target
            # We want output = target when we apply the curve
            # So correction(normalized) = target
            # This means for normalized[idx], output should be target[idx]

            # The correction adjusts the input to achieve linear output
            input_val = int(np.clip(normalized[idx] * 255, 0, 255))
            output_val = int(np.clip(target_val * 255, 0, 255))

            correction_curve.append((input_val, output_val))

        # Ensure endpoints
        if correction_curve[0] != (0, 0):
            correction_curve[0] = (0, 0)
        if correction_curve[-1] != (255, 255):
            correction_curve[-1] = (255, 255)

        # Sort and remove duplicates
        correction_curve = sorted(set(correction_curve), key=lambda x: x[0])

        # Ensure we have at least 2 points
        if len(correction_curve) < 2:
            correction_curve = [(0, 0), (255, 255)]

        # Limit to max points
        while len(correction_curve) > self.MAX_POINTS:
            # Remove points that cause least curvature change
            min_impact = float("inf")
            min_idx = 1

            for i in range(1, len(correction_curve) - 1):
                # Calculate impact of removing this point
                prev_point = correction_curve[i - 1]
                curr_point = correction_curve[i]
                next_point = correction_curve[i + 1]

                # Linear interpolation without this point
                t = (curr_point[0] - prev_point[0]) / (next_point[0] - prev_point[0] + 1e-6)
                expected_y = prev_point[1] + t * (next_point[1] - prev_point[1])
                impact = abs(curr_point[1] - expected_y)

                if impact < min_impact:
                    min_impact = impact
                    min_idx = i

            correction_curve.pop(min_idx)

        return correction_curve

    def _write_acv(
        self,
        file,
        curve_points: list[tuple[int, int]],
    ) -> None:
        """
        Write ACV file format.

        ACV format:
        - 2 bytes: version (4)
        - 2 bytes: number of curves
        - For each curve:
            - 2 bytes: number of points
            - For each point:
                - 2 bytes: output value (0-255, big-endian)
                - 2 bytes: input value (0-255, big-endian)

        Args:
            file: File object to write to
            curve_points: List of (input, output) control points
        """
        # Version
        file.write(struct.pack(">H", self.ACV_VERSION))

        # Number of curves (Master + RGBK = 5)
        file.write(struct.pack(">H", self.NUM_CHANNELS))

        # Write each channel curve
        for channel in range(self.NUM_CHANNELS):
            # Master channel uses correction; others get identity curve
            points = curve_points if channel == 0 else [(0, 0), (255, 255)]

            # Number of points
            file.write(struct.pack(">H", len(points)))

            # Write points (output first, then input - ACV quirk)
            for input_val, output_val in points:
                file.write(struct.pack(">H", output_val))
                file.write(struct.pack(">H", input_val))

    def create_linear_curve(self) -> list[tuple[int, int]]:
        """Create a linear (identity) curve."""
        return [(0, 0), (255, 255)]

    def create_contrast_curve(
        self,
        amount: float = 0.2,
    ) -> list[tuple[int, int]]:
        """
        Create an S-curve for contrast adjustment.

        Args:
            amount: Contrast amount (-1 to 1, positive increases contrast)

        Returns:
            Curve control points
        """
        # S-curve using cubic function
        x = np.linspace(0, 255, 16)
        y = x.copy()

        # Apply S-curve adjustment
        normalized = (x / 255.0 - 0.5) * 2  # -1 to 1
        adjusted = normalized + amount * (normalized ** 3 - normalized)
        y = (adjusted / 2 + 0.5) * 255
        y = np.clip(y, 0, 255)

        return [(int(xi), int(yi)) for xi, yi in zip(x, y, strict=True)]


def export_to_acv(
    densities: np.ndarray,
    output_path: str | Path,
    curve_name: str | None = None,
) -> Path:
    """
    Convenience function to export densities to ACV file.

    Args:
        densities: Predicted density values
        output_path: Path for output file
        curve_name: Optional curve name

    Returns:
        Path to created file
    """
    exporter = ACVExporter()
    return exporter.export(densities, Path(output_path), curve_name)


def check_acv_export() -> bool:
    """
    Verify ACV export works correctly.

    Returns:
        True if export succeeds
    """
    import tempfile

    # Create sample density curve (non-linear)
    steps = np.linspace(0, 1, 21)
    # Simulate typical Pt/Pd response (high gamma)
    densities = 0.1 + 1.9 * np.power(steps, 1.8)

    # Export to temp file
    with tempfile.NamedTemporaryFile(suffix=".acv", delete=False) as f:
        output_path = Path(f.name)

    try:
        result_path = export_to_acv(densities, output_path)

        # Verify file exists and has content
        if not result_path.exists():
            print("ERROR: Output file not created")
            return False

        file_size = result_path.stat().st_size
        if file_size < 20:  # Minimum valid ACV size
            print(f"ERROR: File too small: {file_size} bytes")
            return False

        # Read and verify header
        with open(result_path, "rb") as f:
            version = struct.unpack(">H", f.read(2))[0]
            num_curves = struct.unpack(">H", f.read(2))[0]

        if version != 4:
            print(f"ERROR: Wrong version: {version}")
            return False

        if num_curves != 5:
            print(f"ERROR: Wrong number of curves: {num_curves}")
            return False

        print("ACV export check passed!")
        print(f"  Output file: {result_path}")
        print(f"  File size: {file_size} bytes")
        print(f"  Version: {version}")
        print(f"  Curves: {num_curves}")

        return True

    finally:
        # Cleanup
        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    check_acv_export()
