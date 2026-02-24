"""
MCTS result export to various calibration formats.

Converts SearchResult from MCTS optimization to formats compatible with
the existing calibration infrastructure (CurveData, CalibrationRecord,
QuadTone RIP, CSV, JSON).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES, MCTSSettings
from ptpd_calibration.mcts.types import SearchResult

if TYPE_CHECKING:
    from ptpd_calibration.core.models import CalibrationRecord, CurveData

logger = logging.getLogger(__name__)


class MCTSResultExporter:
    """Converts MCTS SearchResult to various export formats.

    Bridges MCTS output to the existing export infrastructure
    (CurveData, CalibrationRecord, QTR, ICC profiles).
    """

    def __init__(self, settings: MCTSSettings | None = None):
        """
        Initialize MCTS result exporter.

        Args:
            settings: MCTS settings for export configuration
        """
        self.settings = settings or MCTSSettings()
        logger.debug("Initialized MCTSResultExporter")

    def to_curve_data(self, result: SearchResult) -> CurveData:
        """Convert SearchResult to CurveData for existing export pipeline.

        Maps the predicted density curve to CurveData format.

        Args:
            result: MCTS search result

        Returns:
            CurveData with predicted curve

        Raises:
            ImportError: If core.models is not available
        """
        try:
            from ptpd_calibration.core.models import CurveData
            from ptpd_calibration.core.types import CurveType
        except ImportError as e:
            logger.error("Failed to import CurveData - core.models not available")
            raise ImportError("CurveData requires ptpd_calibration.core.models") from e

        logger.debug(f"Converting SearchResult {result.id} to CurveData")

        # Generate input values (0-1 normalized)
        num_points = len(result.predicted_curve)
        input_values = np.linspace(0.0, 1.0, num_points).tolist()
        output_values = result.predicted_curve

        # Generate descriptive name
        name = self._generate_curve_name(result)

        # Extract chemistry info
        chemistry = self._format_chemistry_info(result.best_parameters)

        curve_data = CurveData(
            id=result.id,
            name=name,
            created_at=result.timestamp,
            curve_type=CurveType.MCTS_OPTIMIZED,
            paper_type=result.paper_type,
            chemistry=chemistry,
            notes=self._generate_notes(result),
            input_values=input_values,
            output_values=output_values,
            source_extraction_id=None,
            target_curve_type="mcts_density_curve",
        )

        logger.info(
            f"Created CurveData: {name}, {num_points} points, "
            f"quality={result.quality_score:.3f}"
        )

        return curve_data

    def to_calibration_record(self, result: SearchResult) -> CalibrationRecord:
        """Convert SearchResult to CalibrationRecord for database storage.

        Maps MCTS parameters to CalibrationRecord fields.

        Args:
            result: MCTS search result

        Returns:
            CalibrationRecord with MCTS parameters

        Raises:
            ImportError: If core.models is not available
        """
        try:
            from ptpd_calibration.core.models import CalibrationRecord
            from ptpd_calibration.core.types import (
                ChemistryType,
                ContrastAgent,
                DeveloperType,
                PaperSizing,
            )
        except ImportError as e:
            logger.error("Failed to import CalibrationRecord - core.models not available")
            raise ImportError("CalibrationRecord requires ptpd_calibration.core.models") from e

        logger.debug(f"Converting SearchResult {result.id} to CalibrationRecord")

        params = result.best_parameters

        # Extract parameters with fallback to defaults
        metal_ratio = params.get("metal_ratio", 0.5)
        exposure_time = params.get(
            "exposure_time",
            DEFAULT_PARAMETER_RANGES["exposure_time"].default_value,
        )
        humidity = params.get(
            "humidity",
            DEFAULT_PARAMETER_RANGES["humidity"].default_value,
        )
        developer_temp = params.get(
            "developer_temp",
            DEFAULT_PARAMETER_RANGES["developer_temp"].default_value,
        )

        # Infer chemistry type from metal ratio
        if metal_ratio > 0.9:
            chemistry_type = ChemistryType.PLATINUM
        elif metal_ratio < 0.1:
            chemistry_type = ChemistryType.PALLADIUM
        else:
            chemistry_type = ChemistryType.PLATINUM_PALLADIUM

        # Generate measured densities from predicted curve
        measured_densities = self._generate_density_measurements(result.predicted_curve)

        record = CalibrationRecord(
            id=result.id,
            timestamp=result.timestamp,
            name=self._generate_curve_name(result),
            paper_type=result.paper_type or "Unknown",
            paper_weight=None,
            paper_sizing=PaperSizing.INTERNAL,
            chemistry_type=chemistry_type,
            metal_ratio=metal_ratio,
            contrast_agent=ContrastAgent.NONE,
            contrast_amount=0.0,
            developer=DeveloperType.POTASSIUM_OXALATE,
            exposure_time=exposure_time,
            uv_source=result.uv_source,
            humidity=humidity,
            temperature=developer_temp,
            measured_densities=measured_densities,
            extraction_id=None,
            curve_id=result.id,
            notes=self._generate_notes(result),
            tags=["mcts", "optimized", f"quality_{int(result.quality_score * 100)}"],
        )

        logger.info(
            f"Created CalibrationRecord: {record.name}, "
            f"chemistry={chemistry_type.value}, "
            f"metal_ratio={metal_ratio:.2f}"
        )

        return record

    def to_recipe_json(self, result: SearchResult) -> dict:
        """Export complete recipe as JSON (chemistry + process + curve).

        Returns:
            Dictionary with:
            - parameters: all calibration parameters
            - predicted_curve: density curve
            - quality_score: overall quality
            - alternatives: top alternative parameter sets
            - metadata: search metadata (timestamp, num_simulations, etc.)

        Args:
            result: MCTS search result
        """
        logger.debug(f"Exporting SearchResult {result.id} to recipe JSON")

        # Generate input values for curve
        num_points = len(result.predicted_curve)
        input_values = np.linspace(0.0, 1.0, num_points).tolist()

        recipe = {
            "id": str(result.id),
            "timestamp": result.timestamp.isoformat(),
            "name": self._generate_curve_name(result),
            "parameters": {
                **result.best_parameters,
                "quality_score": result.quality_score,
            },
            "predicted_curve": {
                "input_values": input_values,
                "output_values": result.predicted_curve,
                "num_points": num_points,
            },
            "quality_score": result.quality_score,
            "alternatives": [
                {
                    **alt,
                    "rank": i + 1,
                }
                for i, alt in enumerate(result.alternatives)
            ],
            "metadata": {
                "num_simulations": result.num_simulations,
                "search_time_seconds": result.search_time_seconds,
                "paper_type": result.paper_type,
                "uv_source": result.uv_source,
                "constraint_violations": result.constraint_violations,
                "visit_distribution": result.visit_distribution,
            },
            "chemistry_info": self._format_chemistry_info(result.best_parameters),
            "process_summary": self._generate_process_summary(result.best_parameters),
        }

        logger.info(
            f"Exported recipe JSON: {num_points} curve points, "
            f"{len(result.alternatives)} alternatives"
        )

        return recipe

    def to_csv(self, result: SearchResult) -> str:
        """Export density curve as CSV string.

        Columns: step, exposure, density

        Args:
            result: MCTS search result

        Returns:
            CSV string
        """
        logger.debug(f"Exporting SearchResult {result.id} to CSV")

        lines = ["step,exposure,density"]

        num_points = len(result.predicted_curve)
        exposures = np.linspace(0.0, 1.0, num_points)

        for step, (exposure, density) in enumerate(
            zip(exposures, result.predicted_curve, strict=True)
        ):
            lines.append(f"{step},{exposure:.6f},{density:.6f}")

        csv_str = "\n".join(lines)
        logger.debug(f"Generated CSV with {num_points} rows")

        return csv_str

    def to_qtr_curve(self, result: SearchResult) -> list[int]:
        """Convert to QuadTone RIP curve values (0-255).

        Maps density curve to ink values suitable for QTR.

        Args:
            result: MCTS search result

        Returns:
            List of 256 QTR curve values (0-255)
        """
        logger.debug(f"Converting SearchResult {result.id} to QTR curve")

        # Interpolate to 256 points
        num_points = len(result.predicted_curve)
        x_original = np.linspace(0.0, 1.0, num_points)
        x_qtr = np.linspace(0.0, 1.0, 256)

        # Interpolate density curve
        densities_256 = np.interp(x_qtr, x_original, result.predicted_curve)

        # Normalize to 0-255 range
        # Higher density = more ink
        dmin = densities_256.min()
        dmax = densities_256.max()

        if dmax > dmin:
            # Normalize to 0-1, then scale to 0-255
            normalized = (densities_256 - dmin) / (dmax - dmin)
            qtr_values = (normalized * 255).astype(int)
        else:
            # Flat curve - use midpoint
            qtr_values = np.full(256, 128, dtype=int)

        # Ensure values are in range
        qtr_values = np.clip(qtr_values, 0, 255)

        logger.debug(
            f"Generated QTR curve: min={qtr_values.min()}, "
            f"max={qtr_values.max()}, "
            f"mean={qtr_values.mean():.1f}"
        )

        qtr_list: list[int] = qtr_values.tolist()
        return qtr_list

    def export_to_file(
        self,
        result: SearchResult,
        path: str,
        format: str = "json",
    ) -> str:
        """Export result to file in specified format.

        Args:
            result: MCTS search result
            path: Output directory path
            format: Export format ("json", "csv", "recipe")

        Returns:
            Path to exported file
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"mcts_result_{timestamp}_{result.id.hex[:8]}"

        if format == "json":
            file_path = output_dir / f"{base_name}.json"
            recipe = self.to_recipe_json(result)
            with open(file_path, "w") as f:
                json.dump(recipe, f, indent=2, default=str)
            logger.info(f"Exported JSON to {file_path}")

        elif format == "csv":
            file_path = output_dir / f"{base_name}.csv"
            csv_content = self.to_csv(result)
            with open(file_path, "w") as f:
                f.write(csv_content)
            logger.info(f"Exported CSV to {file_path}")

        elif format == "recipe":
            # Export as comprehensive recipe with all formats
            base_path = output_dir / base_name

            # JSON recipe
            json_path = f"{base_path}_recipe.json"
            recipe = self.to_recipe_json(result)
            with open(json_path, "w") as f:
                json.dump(recipe, f, indent=2, default=str)

            # CSV curve
            csv_path = f"{base_path}_curve.csv"
            csv_content = self.to_csv(result)
            with open(csv_path, "w") as f:
                f.write(csv_content)

            # QTR curve
            qtr_path = f"{base_path}_qtr.txt"
            qtr_values = self.to_qtr_curve(result)
            with open(qtr_path, "w") as f:
                f.write("# QuadTone RIP Curve\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Quality Score: {result.quality_score:.3f}\n\n")
                for value in qtr_values:
                    f.write(f"{value}\n")

            file_path = base_path
            logger.info(f"Exported recipe bundle to {base_path}_*")

        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(file_path)

    def _generate_curve_name(self, result: SearchResult) -> str:
        """Generate descriptive name for curve."""
        params = result.best_parameters
        metal_ratio = params.get("metal_ratio", 0.5)

        # Chemistry label
        if metal_ratio > 0.9:
            chem = "Pt"
        elif metal_ratio < 0.1:
            chem = "Pd"
        else:
            chem = f"Pt{int(metal_ratio * 100)}Pd{int((1 - metal_ratio) * 100)}"

        # Paper type
        paper = result.paper_type or "Unknown"

        # Quality indicator
        quality = int(result.quality_score * 100)

        # Timestamp
        timestamp = result.timestamp.strftime("%Y%m%d")

        return f"{chem}_{paper}_Q{quality}_{timestamp}"

    def _format_chemistry_info(self, parameters: dict[str, float]) -> str:
        """Format chemistry parameters as readable string."""
        metal_ratio = parameters.get("metal_ratio", 0.5)
        coating_weight = parameters.get("coating_weight", 1.5)
        ferric_oxalate = parameters.get("ferric_oxalate_pct", 20.0)

        pt_pct = int(metal_ratio * 100)
        pd_pct = 100 - pt_pct

        return (
            f"Pt:{pt_pct}% Pd:{pd_pct}%, "
            f"Coating:{coating_weight:.2f}ml/sq-in, "
            f"FO:{ferric_oxalate:.1f}%"
        )

    def _generate_notes(self, result: SearchResult) -> str:
        """Generate notes for calibration record."""
        lines = [
            "MCTS-optimized calibration parameters",
            f"Quality Score: {result.quality_score:.3f}",
            f"Simulations: {result.num_simulations}",
            f"Search Time: {result.search_time_seconds:.1f}s",
            f"Alternatives: {len(result.alternatives)}",
        ]

        if result.constraint_violations:
            lines.append("Constraint Violations:")
            for violation in result.constraint_violations:
                lines.append(f"  - {violation}")

        return "\n".join(lines)

    def _generate_process_summary(self, parameters: dict[str, float]) -> dict:
        """Generate process summary from parameters."""
        exposure_time = parameters.get(
            "exposure_time",
            DEFAULT_PARAMETER_RANGES["exposure_time"].default_value,
        )
        developer_temp = parameters.get(
            "developer_temp",
            DEFAULT_PARAMETER_RANGES["developer_temp"].default_value,
        )
        humidity = parameters.get(
            "humidity",
            DEFAULT_PARAMETER_RANGES["humidity"].default_value,
        )

        return {
            "exposure_time_seconds": exposure_time,
            "developer_temperature_celsius": developer_temp,
            "ambient_humidity_percent": humidity,
            "developer_type": "Potassium Oxalate",
        }

    def _generate_density_measurements(self, curve: list[float]) -> list[float]:
        """Generate density measurements from predicted curve.

        Samples the curve at regular intervals to create a realistic
        measurement list.

        Args:
            curve: Predicted density curve

        Returns:
            List of density measurements
        """
        # Sample at 21 evenly-spaced points (standard step wedge)
        num_samples = min(21, len(curve))
        indices = np.linspace(0, len(curve) - 1, num_samples).astype(int)
        measurements = [curve[i] for i in indices]

        return measurements
