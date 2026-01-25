"""Predictive quality assessment for Pt/Pd printing.

This module implements quality prediction that:
- Predicts print quality before printing
- Learns from historical print data
- Identifies risk factors
- Provides confidence intervals
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QualityGrade(str, Enum):
    """Quality grade for prints."""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 75-89%
    ACCEPTABLE = "acceptable"  # 60-74%
    POOR = "poor"  # 40-59%
    UNACCEPTABLE = "unacceptable"  # <40%


class RiskLevel(str, Enum):
    """Risk level for potential issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityFeatures:
    """Features used for quality prediction."""

    # Calibration features
    dmin: float = 0.0
    dmax: float = 0.0
    contrast: float = 0.0
    curve_smoothness: float = 0.0

    # Chemistry features
    metal_ratio: float = 0.0  # Pt to Pd ratio
    coating_amount: float = 0.0  # ml per square inch
    sensitizer_ratio: float = 0.0

    # Environmental features
    temperature: float = 21.0
    humidity: float = 50.0

    # Process features
    exposure_time: float = 0.0
    paper_type: str = ""
    coating_age_hours: float = 0.0

    def to_vector(self) -> list[float]:
        """Convert to numerical feature vector."""
        return [
            self.dmin,
            self.dmax,
            self.contrast,
            self.curve_smoothness,
            self.metal_ratio,
            self.coating_amount,
            self.sensitizer_ratio,
            self.temperature,
            self.humidity,
            self.exposure_time,
            self.coating_age_hours,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get names of numerical features."""
        return [
            "dmin", "dmax", "contrast", "curve_smoothness",
            "metal_ratio", "coating_amount", "sensitizer_ratio",
            "temperature", "humidity", "exposure_time", "coating_age_hours"
        ]


@dataclass
class HistoricalData:
    """Historical print data for learning."""

    features: QualityFeatures
    actual_quality: float  # 0-100
    quality_grade: QualityGrade
    timestamp: datetime
    issues: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": {
                "dmin": self.features.dmin,
                "dmax": self.features.dmax,
                "contrast": self.features.contrast,
                "curve_smoothness": self.features.curve_smoothness,
                "metal_ratio": self.features.metal_ratio,
                "coating_amount": self.features.coating_amount,
                "sensitizer_ratio": self.features.sensitizer_ratio,
                "temperature": self.features.temperature,
                "humidity": self.features.humidity,
                "exposure_time": self.features.exposure_time,
                "coating_age_hours": self.features.coating_age_hours,
                "paper_type": self.features.paper_type,
            },
            "actual_quality": self.actual_quality,
            "quality_grade": self.quality_grade.value,
            "timestamp": self.timestamp.isoformat(),
            "issues": self.issues,
            "notes": self.notes,
        }


class PredictionSettings(BaseSettings):
    """Settings for quality prediction."""

    model_config = SettingsConfigDict(
        env_prefix="PTPD_PREDICTION_",
        env_file=".env",
        extra="ignore",
    )

    # Minimum data points for prediction
    min_data_points: int = Field(
        default=10,
        description="Minimum historical data points before making predictions"
    )

    # Feature weights for rule-based prediction
    dmax_weight: float = Field(
        default=0.20,
        description="Weight for Dmax in quality calculation"
    )

    contrast_weight: float = Field(
        default=0.15,
        description="Weight for contrast in quality calculation"
    )

    smoothness_weight: float = Field(
        default=0.10,
        description="Weight for curve smoothness in quality calculation"
    )

    environment_weight: float = Field(
        default=0.15,
        description="Weight for environmental factors"
    )

    chemistry_weight: float = Field(
        default=0.20,
        description="Weight for chemistry factors"
    )

    process_weight: float = Field(
        default=0.20,
        description="Weight for process factors"
    )

    # Target values
    target_dmax: float = Field(
        default=2.0,
        description="Target maximum density"
    )

    target_dmin: float = Field(
        default=0.1,
        description="Target minimum density"
    )

    target_contrast: float = Field(
        default=1.8,
        description="Target contrast range"
    )

    # Risk thresholds
    high_risk_threshold: float = Field(
        default=0.7,
        description="Risk score threshold for high risk"
    )

    critical_risk_threshold: float = Field(
        default=0.85,
        description="Risk score threshold for critical risk"
    )


@dataclass
class PredictionResult:
    """Result of quality prediction."""

    predicted_quality: float  # 0-100
    quality_grade: QualityGrade
    confidence: float  # 0-1
    confidence_interval: tuple[float, float]  # (low, high)
    risk_factors: list[dict[str, Any]]
    recommendations: list[str]
    feature_contributions: dict[str, float]
    data_points_used: int

    def get_highest_risk(self) -> Optional[dict[str, Any]]:
        """Get highest risk factor."""
        if not self.risk_factors:
            return None
        return max(self.risk_factors, key=lambda r: r.get("severity", 0))

    def is_acceptable(self) -> bool:
        """Check if predicted quality is acceptable."""
        return self.quality_grade in [
            QualityGrade.EXCELLENT,
            QualityGrade.GOOD,
            QualityGrade.ACCEPTABLE
        ]


class QualityPredictor:
    """Main quality prediction system."""

    def __init__(self, settings: Optional[PredictionSettings] = None):
        """Initialize quality predictor.

        Args:
            settings: Prediction settings
        """
        self.settings = settings or PredictionSettings()

        # Historical data storage
        self._history: list[HistoricalData] = []

        # Feature statistics for normalization
        self._feature_stats: dict[str, dict[str, float]] = {}

    def predict(self, features: QualityFeatures) -> PredictionResult:
        """Predict quality for given features.

        Args:
            features: Quality features to evaluate

        Returns:
            Prediction result
        """
        # Calculate component scores
        calibration_score = self._score_calibration(features)
        chemistry_score = self._score_chemistry(features)
        environment_score = self._score_environment(features)
        process_score = self._score_process(features)

        # Calculate weighted quality score
        weighted_score = (
            calibration_score * (self.settings.dmax_weight +
                                 self.settings.contrast_weight +
                                 self.settings.smoothness_weight)
            + chemistry_score * self.settings.chemistry_weight
            + environment_score * self.settings.environment_weight
            + process_score * self.settings.process_weight
        )

        # Normalize to 0-100
        predicted_quality = max(0, min(100, weighted_score * 100))

        # Determine grade
        quality_grade = self._quality_to_grade(predicted_quality)

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(features)

        # Calculate confidence interval
        margin = (1 - confidence) * 20  # Up to 20% margin with low confidence
        confidence_interval = (
            max(0, predicted_quality - margin),
            min(100, predicted_quality + margin)
        )

        # Identify risk factors
        risk_factors = self._identify_risks(features)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_factors)

        # Feature contributions
        feature_contributions = {
            "calibration": calibration_score,
            "chemistry": chemistry_score,
            "environment": environment_score,
            "process": process_score,
        }

        return PredictionResult(
            predicted_quality=predicted_quality,
            quality_grade=quality_grade,
            confidence=confidence,
            confidence_interval=confidence_interval,
            risk_factors=risk_factors,
            recommendations=recommendations,
            feature_contributions=feature_contributions,
            data_points_used=len(self._history),
        )

    def add_historical_data(
        self,
        features: QualityFeatures,
        actual_quality: float,
        issues: Optional[list[str]] = None,
        notes: str = ""
    ) -> HistoricalData:
        """Add historical print data.

        Args:
            features: Features of the print
            actual_quality: Actual quality achieved (0-100)
            issues: Any issues encountered
            notes: Additional notes

        Returns:
            Created historical data record
        """
        record = HistoricalData(
            features=features,
            actual_quality=actual_quality,
            quality_grade=self._quality_to_grade(actual_quality),
            timestamp=datetime.now(timezone.utc),
            issues=issues or [],
            notes=notes,
        )

        self._history.append(record)

        # Update feature statistics
        self._update_statistics(features)

        return record

    def get_similar_prints(
        self,
        features: QualityFeatures,
        limit: int = 5
    ) -> list[tuple[HistoricalData, float]]:
        """Find similar historical prints.

        Args:
            features: Features to match against
            limit: Maximum number of results

        Returns:
            List of (historical_data, similarity_score) tuples
        """
        if not self._history:
            return []

        target_vector = features.to_vector()
        similarities = []

        for record in self._history:
            record_vector = record.features.to_vector()
            similarity = self._calculate_similarity(target_vector, record_vector)
            similarities.append((record, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about historical data.

        Returns:
            Statistics dictionary
        """
        if not self._history:
            return {
                "total_prints": 0,
                "has_sufficient_data": False,
            }

        qualities = [h.actual_quality for h in self._history]
        grades = [h.quality_grade.value for h in self._history]

        grade_counts = {}
        for grade in grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        return {
            "total_prints": len(self._history),
            "has_sufficient_data": len(self._history) >= self.settings.min_data_points,
            "quality": {
                "mean": sum(qualities) / len(qualities),
                "min": min(qualities),
                "max": max(qualities),
                "std_dev": self._std_dev(qualities),
            },
            "grade_distribution": grade_counts,
            "common_issues": self._get_common_issues(),
        }

    def _score_calibration(self, features: QualityFeatures) -> float:
        """Score calibration features."""
        scores = []

        # Dmax score (closer to target is better)
        if features.dmax > 0:
            dmax_ratio = features.dmax / self.settings.target_dmax
            dmax_score = 1.0 - abs(1.0 - dmax_ratio)
            scores.append(max(0, dmax_score))

        # Dmin score (lower is better)
        if features.dmin >= 0:
            dmin_score = 1.0 - min(1.0, features.dmin / self.settings.target_dmin)
            scores.append(max(0, dmin_score))

        # Contrast score
        if features.contrast > 0:
            contrast_ratio = features.contrast / self.settings.target_contrast
            contrast_score = 1.0 - abs(1.0 - contrast_ratio)
            scores.append(max(0, contrast_score))

        # Smoothness score (higher is better)
        if features.curve_smoothness > 0:
            scores.append(min(1.0, features.curve_smoothness))

        return sum(scores) / len(scores) if scores else 0.5

    def _score_chemistry(self, features: QualityFeatures) -> float:
        """Score chemistry features."""
        scores = []

        # Metal ratio score (optimal around 1.0-1.5)
        if features.metal_ratio > 0:
            optimal_ratio = 1.2
            ratio_diff = abs(features.metal_ratio - optimal_ratio) / optimal_ratio
            ratio_score = 1.0 - min(1.0, ratio_diff)
            scores.append(ratio_score)

        # Coating amount score (optimal around 1.5-2.0 ml/sq inch)
        if features.coating_amount > 0:
            optimal_coating = 1.75
            coating_diff = abs(features.coating_amount - optimal_coating) / optimal_coating
            coating_score = 1.0 - min(1.0, coating_diff)
            scores.append(coating_score)

        # Sensitizer ratio score
        if features.sensitizer_ratio > 0:
            # Optimal around 0.15-0.25
            optimal_sensitizer = 0.2
            sens_diff = abs(features.sensitizer_ratio - optimal_sensitizer) / optimal_sensitizer
            sens_score = 1.0 - min(1.0, sens_diff)
            scores.append(sens_score)

        return sum(scores) / len(scores) if scores else 0.5

    def _score_environment(self, features: QualityFeatures) -> float:
        """Score environmental features."""
        scores = []

        # Temperature score (optimal 20-22°C)
        optimal_temp = 21.0
        temp_diff = abs(features.temperature - optimal_temp)
        temp_score = 1.0 - min(1.0, temp_diff / 10)  # 10°C deviation = 0 score
        scores.append(temp_score)

        # Humidity score (optimal 45-55%)
        optimal_humidity = 50.0
        humidity_diff = abs(features.humidity - optimal_humidity)
        humidity_score = 1.0 - min(1.0, humidity_diff / 30)  # 30% deviation = 0 score
        scores.append(humidity_score)

        return sum(scores) / len(scores) if scores else 0.5

    def _score_process(self, features: QualityFeatures) -> float:
        """Score process features."""
        scores = []

        # Exposure time score (based on typical values)
        if features.exposure_time > 0:
            # Typical range 120-300 seconds
            if 120 <= features.exposure_time <= 300:
                scores.append(1.0)
            elif features.exposure_time < 60 or features.exposure_time > 600:
                scores.append(0.3)
            else:
                scores.append(0.7)

        # Coating age score (fresher is better)
        if features.coating_age_hours >= 0:
            # Best within 2 hours, usable up to 24 hours
            if features.coating_age_hours <= 2:
                scores.append(1.0)
            elif features.coating_age_hours <= 6:
                scores.append(0.8)
            elif features.coating_age_hours <= 24:
                scores.append(0.5)
            else:
                scores.append(0.2)

        return sum(scores) / len(scores) if scores else 0.5

    def _calculate_confidence(self, features: QualityFeatures) -> float:
        """Calculate prediction confidence."""
        # Base confidence from data availability
        data_confidence = min(1.0, len(self._history) / self.settings.min_data_points)

        # Feature completeness
        vector = features.to_vector()
        non_zero = sum(1 for v in vector if v != 0)
        feature_confidence = non_zero / len(vector)

        # Similar data availability
        similar = self.get_similar_prints(features, limit=5)
        if similar:
            avg_similarity = sum(s for _, s in similar) / len(similar)
            similarity_confidence = avg_similarity
        else:
            similarity_confidence = 0.3

        # Combined confidence
        confidence = (
            data_confidence * 0.4 +
            feature_confidence * 0.3 +
            similarity_confidence * 0.3
        )

        return min(1.0, confidence)

    def _identify_risks(self, features: QualityFeatures) -> list[dict[str, Any]]:
        """Identify risk factors for the print."""
        risks = []

        # Environmental risks
        if features.temperature < 18:
            risks.append({
                "factor": "low_temperature",
                "description": f"Temperature ({features.temperature:.1f}°C) is below optimal range",
                "severity": 0.6,
                "level": RiskLevel.MEDIUM.value,
            })
        elif features.temperature > 26:
            risks.append({
                "factor": "high_temperature",
                "description": f"Temperature ({features.temperature:.1f}°C) is above optimal range",
                "severity": 0.7,
                "level": RiskLevel.HIGH.value,
            })

        if features.humidity < 35:
            risks.append({
                "factor": "low_humidity",
                "description": f"Humidity ({features.humidity:.0f}%) is very low",
                "severity": 0.5,
                "level": RiskLevel.MEDIUM.value,
            })
        elif features.humidity > 65:
            risks.append({
                "factor": "high_humidity",
                "description": f"Humidity ({features.humidity:.0f}%) is very high",
                "severity": 0.6,
                "level": RiskLevel.MEDIUM.value,
            })

        # Chemistry risks
        if features.coating_age_hours > 12:
            severity = min(1.0, features.coating_age_hours / 24)
            level = (RiskLevel.CRITICAL if severity > 0.85
                     else RiskLevel.HIGH if severity > 0.7
                     else RiskLevel.MEDIUM)
            risks.append({
                "factor": "old_coating",
                "description": f"Coating is {features.coating_age_hours:.1f} hours old",
                "severity": severity,
                "level": level.value,
            })

        # Calibration risks
        if features.dmax < 1.5:
            risks.append({
                "factor": "low_dmax",
                "description": f"Dmax ({features.dmax:.2f}) is below target",
                "severity": 0.6,
                "level": RiskLevel.MEDIUM.value,
            })

        if features.contrast < 1.2:
            risks.append({
                "factor": "low_contrast",
                "description": f"Contrast ({features.contrast:.2f}) may be insufficient",
                "severity": 0.5,
                "level": RiskLevel.MEDIUM.value,
            })

        # Process risks
        if features.exposure_time < 90:
            risks.append({
                "factor": "short_exposure",
                "description": f"Exposure ({features.exposure_time:.0f}s) may be too short",
                "severity": 0.7,
                "level": RiskLevel.HIGH.value,
            })
        elif features.exposure_time > 400:
            risks.append({
                "factor": "long_exposure",
                "description": f"Exposure ({features.exposure_time:.0f}s) may be too long",
                "severity": 0.5,
                "level": RiskLevel.MEDIUM.value,
            })

        # Sort by severity
        risks.sort(key=lambda r: r["severity"], reverse=True)
        return risks

    def _generate_recommendations(
        self,
        features: QualityFeatures,
        risks: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on features and risks."""
        recommendations = []

        for risk in risks:
            factor = risk["factor"]

            if factor == "low_temperature":
                recommendations.append(
                    "Warm your workspace to 20-22°C before printing. "
                    "Allow chemistry to reach room temperature."
                )
            elif factor == "high_temperature":
                recommendations.append(
                    "Cool your workspace or print during cooler hours. "
                    "Chemistry may be more reactive at high temperatures."
                )
            elif factor == "low_humidity":
                recommendations.append(
                    "Consider using a humidifier or misting paper lightly. "
                    "Low humidity can cause uneven coating absorption."
                )
            elif factor == "high_humidity":
                recommendations.append(
                    "Allow extra drying time before exposure. "
                    "Consider using a dehumidifier."
                )
            elif factor == "old_coating":
                recommendations.append(
                    "Mix fresh chemistry if possible. "
                    "Old coating may produce inconsistent results."
                )
            elif factor == "low_dmax":
                recommendations.append(
                    "Increase exposure time or adjust metal ratio. "
                    "Consider using more platinum in the mix."
                )
            elif factor == "low_contrast":
                recommendations.append(
                    "Review curve adjustment settings. "
                    "May need to adjust highlights and shadows separately."
                )
            elif factor == "short_exposure":
                recommendations.append(
                    "Increase exposure time. Test with step wedge to verify."
                )
            elif factor == "long_exposure":
                recommendations.append(
                    "Reduce exposure time to avoid solarization. "
                    "Check UV source intensity."
                )

        # Add general recommendations based on similar prints
        similar = self.get_similar_prints(features, limit=3)
        if similar:
            successful = [h for h, s in similar if h.actual_quality >= 75]
            if successful and successful[0].notes:
                recommendations.append(
                    f"Note from similar successful print: {successful[0].notes}"
                )

        return recommendations

    def _quality_to_grade(self, quality: float) -> QualityGrade:
        """Convert quality score to grade."""
        if quality >= 90:
            return QualityGrade.EXCELLENT
        elif quality >= 75:
            return QualityGrade.GOOD
        elif quality >= 60:
            return QualityGrade.ACCEPTABLE
        elif quality >= 40:
            return QualityGrade.POOR
        else:
            return QualityGrade.UNACCEPTABLE

    def _calculate_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate similarity between two feature vectors."""
        if len(v1) != len(v2):
            return 0.0

        # Normalized Euclidean distance converted to similarity
        squared_diff = sum((a - b) ** 2 for a, b in zip(v1, v2))
        distance = math.sqrt(squared_diff)

        # Convert to similarity (0-1)
        max_distance = math.sqrt(len(v1) * 100)  # Assuming max value ~10 per feature
        similarity = 1.0 - min(1.0, distance / max_distance)

        return similarity

    def _update_statistics(self, features: QualityFeatures) -> None:
        """Update feature statistics with new data point."""
        vector = features.to_vector()
        names = QualityFeatures.feature_names()

        for name, value in zip(names, vector):
            if name not in self._feature_stats:
                self._feature_stats[name] = {
                    "count": 0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }

            stats = self._feature_stats[name]
            stats["count"] += 1
            stats["sum"] += value
            stats["sum_sq"] += value ** 2
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)

    def _std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _get_common_issues(self) -> list[tuple[str, int]]:
        """Get most common issues from history."""
        issue_counts: dict[str, int] = {}

        for record in self._history:
            for issue in record.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_issues[:5]
