"""
Transfer learning for new papers and chemistries.

Enables bootstrapping calibrations from similar known materials.
"""

import numpy as np

from ptpd_calibration.core.models import CalibrationRecord
from ptpd_calibration.ml.database import CalibrationDatabase

# Paper characteristic database for similarity matching
PAPER_CHARACTERISTICS = {
    # Cotton papers
    "arches platine": {
        "fiber": "cotton",
        "sizing": "internal",
        "weight_range": (280, 320),
        "characteristics": ["smooth", "bright_white", "good_dmax"],
    },
    "bergger cot320": {
        "fiber": "cotton",
        "sizing": "internal",
        "weight_range": (300, 320),
        "characteristics": ["textured", "warm_white", "high_dmax"],
    },
    "hahnemuhle platinum rag": {
        "fiber": "cotton",
        "sizing": "internal",
        "weight_range": (280, 320),
        "characteristics": ["smooth", "bright_white", "very_high_dmax"],
    },
    # Japanese papers
    "gampi torinoko": {
        "fiber": "gampi",
        "sizing": "none",
        "weight_range": (100, 200),
        "characteristics": ["thin", "translucent", "warm_tone"],
    },
    "kozo": {
        "fiber": "kozo",
        "sizing": "none",
        "weight_range": (40, 100),
        "characteristics": ["fibrous", "absorbent", "soft_tones"],
    },
    # Mixed papers
    "revere platinum": {
        "fiber": "cotton_alpha",
        "sizing": "gelatin",
        "weight_range": (250, 300),
        "characteristics": ["smooth", "neutral_white", "good_dmax"],
    },
}


class TransferLearner:
    """
    Transfer learning for calibration across papers and chemistries.

    Uses similarity metrics to transfer calibration knowledge from
    known papers to new, similar papers.
    """

    def __init__(self, database: CalibrationDatabase):
        """
        Initialize transfer learner.

        Args:
            database: CalibrationDatabase with known calibrations.
        """
        self.database = database

    def find_similar_papers(
        self,
        paper_type: str,
        paper_weight: int | None = None,
        paper_sizing: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find papers similar to the query paper.

        Args:
            paper_type: Name of paper to find matches for.
            paper_weight: Weight in GSM (optional).
            paper_sizing: Sizing type (optional).
            top_k: Number of results to return.

        Returns:
            List of (paper_name, similarity_score) tuples.
        """
        query_lower = paper_type.lower()

        # Get query characteristics
        query_chars = PAPER_CHARACTERISTICS.get(query_lower, {})

        # Get all known papers from database
        known_papers = set()
        for record in self.database.get_all_records():
            known_papers.add(record.paper_type)

        similarities = []

        for known_paper in known_papers:
            known_lower = known_paper.lower()
            known_chars = PAPER_CHARACTERISTICS.get(known_lower, {})

            # Calculate similarity
            score = self._calculate_paper_similarity(
                query_chars, known_chars, query_lower, known_lower
            )

            # Adjust for weight if provided
            if paper_weight and known_chars.get("weight_range"):
                w_min, w_max = known_chars["weight_range"]
                if w_min <= paper_weight <= w_max:
                    score += 0.1
                elif abs(paper_weight - (w_min + w_max) / 2) < 50:
                    score += 0.05

            similarities.append((known_paper, score))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def transfer_curve(
        self,
        source_records: list[CalibrationRecord],
        target_paper: str,
        adjustment_factor: float = 1.0,
    ) -> list[float]:
        """
        Transfer calibration curve from source records to new paper.

        Args:
            source_records: Calibration records from similar paper.
            target_paper: Name of target paper.
            adjustment_factor: Scaling factor for transfer (0.5-1.5).

        Returns:
            Predicted density values for target paper.
        """
        if not source_records:
            raise ValueError("No source records provided")

        # Average density curves from source records
        all_densities = [r.measured_densities for r in source_records if r.measured_densities]

        if not all_densities:
            raise ValueError("Source records have no density measurements")

        # Normalize to common length
        normalized = []
        target_length = 21

        for densities in all_densities:
            if len(densities) != target_length:
                x_old = np.linspace(0, 1, len(densities))
                x_new = np.linspace(0, 1, target_length)
                densities = list(np.interp(x_new, x_old, densities))
            normalized.append(densities)

        # Average
        averaged = np.mean(normalized, axis=0)

        # Apply adjustment factor
        dmin = averaged.min()
        adjusted = dmin + (averaged - dmin) * adjustment_factor

        return list(adjusted)

    def suggest_starting_parameters(
        self,
        target_paper: str,
        target_chemistry: str = "platinum_palladium",
    ) -> dict:
        """
        Suggest starting parameters for a new paper.

        Args:
            target_paper: Name of target paper.
            target_chemistry: Type of chemistry to use.

        Returns:
            Dictionary with suggested starting parameters.
        """
        # Find similar papers
        similar = self.find_similar_papers(target_paper, top_k=3)

        if not similar:
            # Use default parameters
            return self._default_parameters()

        # Get records from most similar paper
        best_paper, _ = similar[0]
        records = self.database.get_records_for_paper(best_paper)

        if not records:
            return self._default_parameters()

        # Average parameters from similar records
        exposures = [r.exposure_time for r in records]
        ratios = [r.metal_ratio for r in records]
        contrasts = [r.contrast_amount for r in records]

        return {
            "paper_type": target_paper,
            "source_paper": best_paper,
            "exposure_time": float(np.median(exposures)),
            "metal_ratio": float(np.median(ratios)),
            "contrast_amount": float(np.median(contrasts)),
            "notes": f"Parameters based on {len(records)} calibrations for similar paper: {best_paper}",
        }

    def estimate_exposure_adjustment(
        self,
        source_paper: str,
        target_paper: str,
        source_exposure: float,
    ) -> float:
        """
        Estimate exposure adjustment between papers.

        Args:
            source_paper: Known paper type.
            target_paper: Target paper type.
            source_exposure: Exposure time for source paper.

        Returns:
            Estimated exposure time for target paper.
        """
        source_chars = PAPER_CHARACTERISTICS.get(source_paper.lower(), {})
        target_chars = PAPER_CHARACTERISTICS.get(target_paper.lower(), {})

        # Base adjustment factors
        adjustment = 1.0

        # Sizing affects absorption
        source_sizing = source_chars.get("sizing", "internal")
        target_sizing = target_chars.get("sizing", "internal")

        if source_sizing == "none" and target_sizing == "internal":
            adjustment *= 1.2  # Internal sizing needs more exposure
        elif source_sizing == "internal" and target_sizing == "none":
            adjustment *= 0.85  # Unsized paper more sensitive

        # Fiber type affects Dmax potential
        source_fiber = source_chars.get("fiber", "cotton")
        target_fiber = target_chars.get("fiber", "cotton")

        fiber_factors = {
            "cotton": 1.0,
            "cotton_alpha": 1.0,
            "gampi": 0.9,
            "kozo": 0.85,
            "mitsumata": 0.88,
        }

        source_factor = fiber_factors.get(source_fiber, 1.0)
        target_factor = fiber_factors.get(target_fiber, 1.0)

        if source_factor > 0:
            adjustment *= target_factor / source_factor

        return source_exposure * adjustment

    def _calculate_paper_similarity(
        self,
        query_chars: dict,
        known_chars: dict,
        query_name: str,
        known_name: str,
    ) -> float:
        """Calculate similarity between two papers."""
        score = 0.0

        # Name similarity (basic)
        if query_name == known_name:
            return 1.0

        # Check if one contains the other
        if query_name in known_name or known_name in query_name:
            score += 0.3

        # Fiber type
        if query_chars.get("fiber") and known_chars.get("fiber"):
            if query_chars["fiber"] == known_chars["fiber"]:
                score += 0.3
            elif query_chars["fiber"].startswith(known_chars["fiber"][:3]):
                score += 0.15

        # Sizing
        if (
            query_chars.get("sizing")
            and known_chars.get("sizing")
            and query_chars["sizing"] == known_chars["sizing"]
        ):
            score += 0.2

        # Characteristics overlap
        query_chars_set = set(query_chars.get("characteristics", []))
        known_chars_set = set(known_chars.get("characteristics", []))

        if query_chars_set and known_chars_set:
            overlap = len(query_chars_set & known_chars_set)
            total = len(query_chars_set | known_chars_set)
            if total > 0:
                score += 0.2 * (overlap / total)

        return min(1.0, score)

    def _default_parameters(self) -> dict:
        """Return default starting parameters."""
        return {
            "paper_type": "unknown",
            "source_paper": None,
            "exposure_time": 180.0,  # 3 minutes
            "metal_ratio": 0.5,  # 50/50 Pt/Pd
            "contrast_amount": 5.0,  # 5 drops Na2
            "notes": "Default parameters for unknown paper. Start with test strips.",
        }
