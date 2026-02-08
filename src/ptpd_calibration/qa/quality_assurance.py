"""
Quality Assurance module for platinum/palladium printing.

Provides comprehensive QA tools for:
- Negative density validation
- Chemistry freshness tracking
- Paper humidity monitoring
- UV light meter integration
- Quality reporting
- Alert system

All thresholds are configurable via QASettings.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ptpd_calibration.config import QASettings

# ============================================================================
# Configuration
# ============================================================================


class SolutionType(str, Enum):
    """Types of chemistry solutions."""

    FERRIC_OXALATE_1 = "ferric_oxalate_1"
    FERRIC_OXALATE_2 = "ferric_oxalate_2"
    PALLADIUM = "palladium"
    PLATINUM = "platinum"
    NA2 = "na2"
    DEVELOPER = "developer"
    CLEARING_BATH = "clearing_bath"
    EDTA = "edta"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of quality alerts."""

    DENSITY = "density"
    CHEMISTRY = "chemistry"
    HUMIDITY = "humidity"
    UV_LIGHT = "uv_light"
    GENERAL = "general"


class ReportFormat(str, Enum):
    """Supported report export formats."""

    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DensityAnalysis:
    """Results of negative density analysis."""

    min_density: float
    max_density: float
    mean_density: float
    density_range: float
    highlight_blocked: bool
    shadow_blocked: bool
    histogram: np.ndarray
    zone_distribution: dict[int, float]
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_density": round(self.min_density, 3),
            "max_density": round(self.max_density, 3),
            "mean_density": round(self.mean_density, 3),
            "density_range": round(self.density_range, 3),
            "highlight_blocked": self.highlight_blocked,
            "shadow_blocked": self.shadow_blocked,
            "zone_distribution": {
                f"Zone {z}": f"{pct * 100:.1f}%" for z, pct in self.zone_distribution.items()
            },
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


@dataclass
class ChemistrySolution:
    """Chemistry solution tracking record."""

    solution_id: str
    solution_type: SolutionType
    date_mixed: datetime
    initial_volume_ml: float
    current_volume_ml: float
    shelf_life_days: int
    usage_log: list[tuple[datetime, float]] = field(default_factory=list)
    notes: str = ""

    @property
    def expiration_date(self) -> datetime:
        """Calculate expiration date."""
        return self.date_mixed + timedelta(days=self.shelf_life_days)

    @property
    def days_until_expiration(self) -> int:
        """Days until solution expires."""
        delta = self.expiration_date - datetime.now()
        return delta.days

    @property
    def is_expired(self) -> bool:
        """Check if solution is expired."""
        return self.days_until_expiration < 0

    @property
    def volume_percent_remaining(self) -> float:
        """Percentage of volume remaining."""
        if self.initial_volume_ml == 0:
            return 0.0
        return (self.current_volume_ml / self.initial_volume_ml) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "solution_id": self.solution_id,
            "solution_type": self.solution_type.value,
            "date_mixed": self.date_mixed.isoformat(),
            "expiration_date": self.expiration_date.isoformat(),
            "days_until_expiration": self.days_until_expiration,
            "is_expired": self.is_expired,
            "initial_volume_ml": self.initial_volume_ml,
            "current_volume_ml": self.current_volume_ml,
            "volume_percent_remaining": round(self.volume_percent_remaining, 1),
            "shelf_life_days": self.shelf_life_days,
            "total_usage_ml": sum(amount for _, amount in self.usage_log),
            "notes": self.notes,
        }


@dataclass
class HumidityReading:
    """Paper humidity measurement."""

    timestamp: datetime
    humidity_percent: float
    temperature_celsius: float | None = None
    paper_type: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "humidity_percent": round(self.humidity_percent, 1),
            "temperature_celsius": (
                round(self.temperature_celsius, 1)
                if self.temperature_celsius is not None
                else None
            ),
            "paper_type": self.paper_type,
            "notes": self.notes,
        }


@dataclass
class UVReading:
    """UV light meter reading."""

    timestamp: datetime
    intensity: float
    wavelength: float | None = None
    bulb_hours: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "intensity": round(self.intensity, 2),
            "wavelength": (
                round(self.wavelength, 1) if self.wavelength is not None else None
            ),
            "bulb_hours": (
                round(self.bulb_hours, 1) if self.bulb_hours is not None else None
            ),
            "notes": self.notes,
        }


@dataclass
class Alert:
    """Quality assurance alert."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    dismissed: bool = False
    dismissed_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "dismissed": self.dismissed,
            "dismissed_at": self.dismissed_at.isoformat() if self.dismissed_at else None,
        }


# ============================================================================
# Negative Density Validator
# ============================================================================


class NegativeDensityValidator:
    """
    Validates negative density for platinum/palladium printing.

    Checks if density values are in printable range, warns about blocked
    highlights/shadows, and suggests corrections.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize validator.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()

    def validate_density_range(self, image: np.ndarray | Image.Image) -> DensityAnalysis:
        """
        Check if image density is in printable range.

        Args:
            image: Image array or PIL Image

        Returns:
            DensityAnalysis with validation results
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("L"))
        elif len(image.shape) == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2)

        # Convert pixel values (0-255) to density
        # Density = -log10(transmittance) = -log10(pixel/255)
        # Avoid log(0) by adding small epsilon
        pixel_norm = np.clip(image / 255.0, 1e-6, 1.0)
        density = -np.log10(pixel_norm)

        # Calculate statistics
        min_density = float(np.min(density))
        max_density = float(np.max(density))
        mean_density = float(np.mean(density))
        density_range = max_density - min_density

        # Check for blocked highlights/shadows
        highlight_blocked = min_density < self.settings.highlight_warning_threshold
        shadow_blocked = max_density > self.settings.shadow_warning_threshold

        # Get histogram
        histogram = self._get_density_histogram(density)

        # Get zone distribution
        zone_distribution = self._calculate_zone_distribution(density)

        # Collect warnings
        warnings = []
        if min_density < self.settings.min_density:
            warnings.append(f"Minimum density ({min_density:.3f}) below printable range")
        if max_density > self.settings.max_density:
            warnings.append(f"Maximum density ({max_density:.3f}) exceeds printable range")
        if highlight_blocked:
            warnings.append(f"Highlights may be blocked (Dmin={min_density:.3f})")
        if shadow_blocked:
            warnings.append(f"Shadows may be blocked (Dmax={max_density:.3f})")
        if density_range < 1.5:
            warnings.append(f"Low density range ({density_range:.2f}) - print may lack contrast")

        # Generate suggestions
        suggestions = self.suggest_corrections(
            min_density, max_density, density_range, highlight_blocked, shadow_blocked
        )

        return DensityAnalysis(
            min_density=min_density,
            max_density=max_density,
            mean_density=mean_density,
            density_range=density_range,
            highlight_blocked=highlight_blocked,
            shadow_blocked=shadow_blocked,
            histogram=histogram,
            zone_distribution=zone_distribution,
            warnings=warnings,
            suggestions=suggestions,
        )

    def check_highlight_detail(self, image: np.ndarray | Image.Image) -> tuple[bool, str]:
        """
        Check if highlights have detail or are blocked.

        Args:
            image: Image array or PIL Image

        Returns:
            Tuple of (has_detail, message)
        """
        analysis = self.validate_density_range(image)

        if analysis.highlight_blocked:
            return False, f"Highlights blocked at Dmin={analysis.min_density:.3f}"

        # Check if we have good highlight detail
        highlight_zone_percent = analysis.zone_distribution.get(
            1, 0
        ) + analysis.zone_distribution.get(2, 0)
        if highlight_zone_percent < 0.05:
            return False, "Very little highlight detail present"

        return True, f"Good highlight detail (Dmin={analysis.min_density:.3f})"

    def check_shadow_detail(self, image: np.ndarray | Image.Image) -> tuple[bool, str]:
        """
        Check if shadows have detail or are blocked.

        Args:
            image: Image array or PIL Image

        Returns:
            Tuple of (has_detail, message)
        """
        analysis = self.validate_density_range(image)

        if analysis.shadow_blocked:
            return False, f"Shadows blocked at Dmax={analysis.max_density:.3f}"

        # Check if we have good shadow detail
        shadow_zone_percent = analysis.zone_distribution.get(8, 0) + analysis.zone_distribution.get(
            9, 0
        )
        if shadow_zone_percent < 0.05:
            return False, "Very little shadow detail present"

        return True, f"Good shadow detail (Dmax={analysis.max_density:.3f})"

    def get_density_histogram(
        self, image: np.ndarray | Image.Image, bins: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return density histogram with zones.

        Args:
            image: Image array or PIL Image
            bins: Number of histogram bins

        Returns:
            Tuple of (histogram counts, bin edges)
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("L"))
        elif len(image.shape) == 3:
            image = np.mean(image, axis=2)

        # Convert to density
        pixel_norm = np.clip(image / 255.0, 1e-6, 1.0)
        density = -np.log10(pixel_norm)

        # Calculate histogram
        hist, edges = np.histogram(density, bins=bins, range=(0, 4.0))

        return hist, edges

    def suggest_corrections(
        self,
        min_density: float,
        max_density: float,
        density_range: float,
        highlight_blocked: bool,
        shadow_blocked: bool,
    ) -> list[str]:
        """
        Suggest exposure/contrast adjustments.

        Args:
            min_density: Minimum density value
            max_density: Maximum density value
            density_range: Density range
            highlight_blocked: Whether highlights are blocked
            shadow_blocked: Whether shadows are blocked

        Returns:
            List of suggestions
        """
        suggestions = []

        # Overall density adjustment
        if max_density < 1.8:
            suggestions.append("Increase overall negative density for deeper blacks")
        elif min_density > 0.15:
            suggestions.append("Decrease overall negative density for better highlight detail")

        # Contrast adjustments
        if density_range < 1.5:
            suggestions.append(
                "Increase negative contrast - consider using Grade 3+ paper or longer development"
            )
        elif density_range > 3.0:
            suggestions.append("Reduce negative contrast - may be too contrasty for Pt/Pd process")

        # Specific corrections
        if highlight_blocked:
            suggestions.append("Reduce exposure or development time to recover highlight detail")
        if shadow_blocked:
            suggestions.append(
                "Increase exposure to add shadow detail, or use higher platinum ratio for deeper blacks"
            )

        # Curve adjustments
        if not highlight_blocked and not shadow_blocked and density_range < 2.0:
            suggestions.append("Apply gentle S-curve to boost midtone contrast")

        return suggestions

    def _get_density_histogram(self, density: np.ndarray, bins: int = 100) -> np.ndarray:
        """Calculate density histogram."""
        hist, _ = np.histogram(density, bins=bins, range=(0, 4.0))
        return hist

    def _calculate_zone_distribution(self, density: np.ndarray) -> dict[int, float]:
        """
        Calculate distribution across Ansel Adams zones.

        Zones 0-10 mapped from density values.
        """
        # Zone mapping (approximate)
        # Zone 0: Pure black (D > 3.0)
        # Zone 1-2: Deep shadows (D 2.0-3.0)
        # Zone 3-4: Shadows (D 1.5-2.0)
        # Zone 5: Middle gray (D 1.0-1.5)
        # Zone 6-7: Highlights (D 0.5-1.0)
        # Zone 8-9: Bright highlights (D 0.1-0.5)
        # Zone 10: Pure white (D < 0.1)

        total_pixels = density.size
        zones = {}

        zones[0] = np.sum(density > 3.0) / total_pixels
        zones[1] = np.sum((density > 2.5) & (density <= 3.0)) / total_pixels
        zones[2] = np.sum((density > 2.0) & (density <= 2.5)) / total_pixels
        zones[3] = np.sum((density > 1.75) & (density <= 2.0)) / total_pixels
        zones[4] = np.sum((density > 1.5) & (density <= 1.75)) / total_pixels
        zones[5] = np.sum((density > 1.0) & (density <= 1.5)) / total_pixels
        zones[6] = np.sum((density > 0.75) & (density <= 1.0)) / total_pixels
        zones[7] = np.sum((density > 0.5) & (density <= 0.75)) / total_pixels
        zones[8] = np.sum((density > 0.25) & (density <= 0.5)) / total_pixels
        zones[9] = np.sum((density > 0.1) & (density <= 0.25)) / total_pixels
        zones[10] = np.sum(density <= 0.1) / total_pixels

        return zones


# ============================================================================
# Chemistry Freshness Tracker
# ============================================================================


class ChemistryFreshnessTracker:
    """
    Tracks chemistry solution freshness and usage.

    Monitors expiration dates, volume remaining, and usage patterns.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize tracker.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()
        self.solutions: dict[str, ChemistrySolution] = {}
        self._shelf_life_map = {
            SolutionType.FERRIC_OXALATE_1: self.settings.ferric_oxalate_shelf_life,
            SolutionType.FERRIC_OXALATE_2: self.settings.ferric_oxalate_shelf_life,
            SolutionType.PALLADIUM: self.settings.palladium_shelf_life,
            SolutionType.PLATINUM: self.settings.platinum_shelf_life,
            SolutionType.NA2: self.settings.na2_shelf_life,
            SolutionType.DEVELOPER: self.settings.developer_shelf_life,
            SolutionType.CLEARING_BATH: self.settings.clearing_bath_shelf_life,
            SolutionType.EDTA: self.settings.edta_shelf_life,
        }

    def register_solution(
        self,
        solution_type: SolutionType,
        date_mixed: datetime,
        volume_ml: float,
        notes: str = "",
        solution_id: str | None = None,
    ) -> str:
        """
        Register a new chemistry solution.

        Args:
            solution_type: Type of solution
            date_mixed: Date solution was mixed
            volume_ml: Initial volume in milliliters
            notes: Optional notes
            solution_id: Optional custom ID (auto-generated if None)

        Returns:
            Solution ID
        """
        if solution_id is None:
            solution_id = f"{solution_type.value}_{uuid4().hex[:8]}"

        shelf_life = self._shelf_life_map[solution_type]

        solution = ChemistrySolution(
            solution_id=solution_id,
            solution_type=solution_type,
            date_mixed=date_mixed,
            initial_volume_ml=volume_ml,
            current_volume_ml=volume_ml,
            shelf_life_days=shelf_life,
            notes=notes,
        )

        self.solutions[solution_id] = solution
        return solution_id

    def check_freshness(self, solution_id: str) -> tuple[bool, str]:
        """
        Check if solution is still fresh.

        Args:
            solution_id: Solution identifier

        Returns:
            Tuple of (is_fresh, message)
        """
        if solution_id not in self.solutions:
            return False, "Solution not found"

        solution = self.solutions[solution_id]

        if solution.is_expired:
            days_expired = abs(solution.days_until_expiration)
            return False, f"Expired {days_expired} days ago"

        days_left = solution.days_until_expiration
        if days_left < self.settings.expiration_critical_days:
            return True, f"Critical: Expires in {days_left} days"
        elif days_left < self.settings.expiration_warning_days:
            return True, f"Warning: Expires in {days_left} days"
        else:
            return True, f"Fresh ({days_left} days remaining)"

    def get_expiration_date(self, solution_id: str) -> datetime | None:
        """
        Get solution expiration date.

        Args:
            solution_id: Solution identifier

        Returns:
            Expiration date or None if not found
        """
        if solution_id not in self.solutions:
            return None
        return self.solutions[solution_id].expiration_date

    def log_usage(
        self, solution_id: str, amount_ml: float, timestamp: datetime | None = None
    ) -> bool:
        """
        Log solution usage.

        Args:
            solution_id: Solution identifier
            amount_ml: Amount used in milliliters
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if successful, False if solution not found or insufficient volume
        """
        if solution_id not in self.solutions:
            return False

        solution = self.solutions[solution_id]

        if amount_ml > solution.current_volume_ml:
            return False

        if timestamp is None:
            timestamp = datetime.now()

        solution.usage_log.append((timestamp, amount_ml))
        solution.current_volume_ml -= amount_ml

        return True

    def get_remaining_volume(self, solution_id: str) -> float | None:
        """
        Get remaining volume of solution.

        Args:
            solution_id: Solution identifier

        Returns:
            Remaining volume in ml or None if not found
        """
        if solution_id not in self.solutions:
            return None
        return self.solutions[solution_id].current_volume_ml

    def get_alerts(self) -> list[dict[str, str]]:
        """
        Get list of expiring/expired solutions.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        for solution in self.solutions.values():
            # Check expiration
            if solution.is_expired:
                alerts.append(
                    {
                        "solution_id": solution.solution_id,
                        "type": "expired",
                        "severity": "critical",
                        "message": f"{solution.solution_type.value} expired {abs(solution.days_until_expiration)} days ago",
                    }
                )
            elif solution.days_until_expiration < self.settings.expiration_critical_days:
                alerts.append(
                    {
                        "solution_id": solution.solution_id,
                        "type": "expiring_soon",
                        "severity": "error",
                        "message": f"{solution.solution_type.value} expires in {solution.days_until_expiration} days",
                    }
                )
            elif solution.days_until_expiration < self.settings.expiration_warning_days:
                alerts.append(
                    {
                        "solution_id": solution.solution_id,
                        "type": "expiring",
                        "severity": "warning",
                        "message": f"{solution.solution_type.value} expires in {solution.days_until_expiration} days",
                    }
                )

            # Check volume
            volume_pct = solution.volume_percent_remaining
            if volume_pct < self.settings.critical_volume_percent:
                alerts.append(
                    {
                        "solution_id": solution.solution_id,
                        "type": "low_volume",
                        "severity": "error",
                        "message": f"{solution.solution_type.value} critically low ({volume_pct:.1f}% remaining)",
                    }
                )
            elif volume_pct < self.settings.low_volume_warning_percent:
                alerts.append(
                    {
                        "solution_id": solution.solution_id,
                        "type": "low_volume",
                        "severity": "warning",
                        "message": f"{solution.solution_type.value} running low ({volume_pct:.1f}% remaining)",
                    }
                )

        return alerts

    def recommend_replenishment(self, solution_id: str) -> str | None:
        """
        Suggest when to replenish solution.

        Args:
            solution_id: Solution identifier

        Returns:
            Recommendation string or None if not found
        """
        if solution_id not in self.solutions:
            return None

        solution = self.solutions[solution_id]

        # Calculate average daily usage
        if len(solution.usage_log) < 2:
            return "Insufficient usage data for recommendation"

        total_used = sum(amount for _, amount in solution.usage_log)
        first_use = solution.usage_log[0][0]
        last_use = solution.usage_log[-1][0]
        days_active = (last_use - first_use).days or 1
        daily_usage = total_used / days_active

        # Estimate days remaining
        if daily_usage > 0:
            days_of_supply = solution.current_volume_ml / daily_usage
            return (
                f"Approximately {days_of_supply:.0f} days of supply remaining at current usage rate"
            )
        else:
            return "No recent usage detected"

    def get_solution_info(self, solution_id: str) -> dict | None:
        """Get complete solution information."""
        if solution_id not in self.solutions:
            return None
        return self.solutions[solution_id].to_dict()

    def list_all_solutions(self) -> list[dict]:
        """Get list of all tracked solutions."""
        return [sol.to_dict() for sol in self.solutions.values()]


# ============================================================================
# Paper Humidity Checker
# ============================================================================


class PaperHumidityChecker:
    """
    Monitors paper humidity for coating readiness.

    Tracks humidity readings and provides recommendations for paper conditioning.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize checker.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()
        self.readings: list[HumidityReading] = []
        self.ambient_conditions: list[tuple[datetime, float, float]] = []

    def measure_paper_humidity(
        self,
        humidity_percent: float,
        temperature_celsius: float | None = None,
        paper_type: str | None = None,
        notes: str = "",
        timestamp: datetime | None = None,
    ) -> HumidityReading:
        """
        Log a paper humidity reading.

        Args:
            humidity_percent: Humidity percentage (0-100)
            temperature_celsius: Optional temperature
            paper_type: Optional paper type
            notes: Optional notes
            timestamp: Optional timestamp (defaults to now)

        Returns:
            HumidityReading object
        """
        if timestamp is None:
            timestamp = datetime.now()

        reading = HumidityReading(
            timestamp=timestamp,
            humidity_percent=humidity_percent,
            temperature_celsius=temperature_celsius,
            paper_type=paper_type,
            notes=notes,
        )

        self.readings.append(reading)
        return reading

    def is_paper_ready(
        self,
        target_humidity_min: float | None = None,
        target_humidity_max: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check if paper is ready to coat.

        Args:
            target_humidity_min: Minimum target humidity (defaults to settings)
            target_humidity_max: Maximum target humidity (defaults to settings)

        Returns:
            Tuple of (is_ready, message)
        """
        if not self.readings:
            return False, "No humidity readings available"

        # Use latest reading
        latest = self.readings[-1]

        target_min = target_humidity_min or self.settings.ideal_humidity_min
        target_max = target_humidity_max or self.settings.ideal_humidity_max

        humidity = latest.humidity_percent

        if humidity < target_min - self.settings.humidity_tolerance:
            return False, f"Paper too dry ({humidity:.1f}% RH) - humidify before coating"
        elif humidity > target_max + self.settings.humidity_tolerance:
            return False, f"Paper too humid ({humidity:.1f}% RH) - allow to dry before coating"
        elif target_min <= humidity <= target_max:
            return True, f"Paper ready ({humidity:.1f}% RH in ideal range)"
        else:
            return True, f"Paper acceptable ({humidity:.1f}% RH within tolerance)"

    def estimate_drying_time(
        self,
        current_humidity: float,
        target_humidity: float | None = None,
        room_temperature: float | None = None,
    ) -> tuple[float, str]:
        """
        Estimate time needed to reach target humidity.

        Args:
            current_humidity: Current paper humidity percentage
            target_humidity: Target humidity (defaults to ideal max)
            room_temperature: Room temperature in Celsius

        Returns:
            Tuple of (hours, message)
        """
        if target_humidity is None:
            target_humidity = self.settings.ideal_humidity_max

        humidity_diff = abs(current_humidity - target_humidity)

        # Base estimation on humidity difference
        hours = humidity_diff * self.settings.drying_time_factor

        # Adjust for temperature if provided
        if room_temperature is not None:
            # Higher temp = faster drying
            temp_factor = 1.0 - ((room_temperature - 20) * 0.05)
            hours *= max(0.5, min(2.0, temp_factor))

        action = "drying" if current_humidity > target_humidity else "humidifying"

        message = f"Approximately {hours:.1f} hours of {action} needed"
        return hours, message

    def log_ambient_conditions(
        self,
        humidity_percent: float,
        temperature_celsius: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Track ambient room conditions.

        Args:
            humidity_percent: Room humidity percentage
            temperature_celsius: Room temperature
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.ambient_conditions.append((timestamp, humidity_percent, temperature_celsius))

    def recommend_humidity_adjustment(self) -> str | None:
        """
        Suggest humidifying or dehumidifying.

        Returns:
            Recommendation string or None if no data
        """
        if not self.readings:
            return "No humidity readings available"

        latest = self.readings[-1]
        humidity = latest.humidity_percent

        target_mid = (self.settings.ideal_humidity_min + self.settings.ideal_humidity_max) / 2

        if humidity < self.settings.ideal_humidity_min - self.settings.humidity_tolerance:
            diff = target_mid - humidity
            return f"Humidify paper: Need to increase humidity by {diff:.1f}%. Use damp blotters or humid box."
        elif humidity > self.settings.ideal_humidity_max + self.settings.humidity_tolerance:
            diff = humidity - target_mid
            return f"Dehumidify paper: Need to decrease humidity by {diff:.1f}%. Use dry environment or desiccant."
        else:
            return "Paper humidity in acceptable range"

    def get_latest_reading(self) -> HumidityReading | None:
        """Get the most recent humidity reading."""
        if not self.readings:
            return None
        return self.readings[-1]

    def get_readings_history(self, hours: int = 24) -> list[HumidityReading]:
        """Get humidity readings from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [r for r in self.readings if r.timestamp >= cutoff]


# ============================================================================
# UV Light Meter Integration
# ============================================================================


class UVLightMeterIntegration:
    """
    Integrates UV light meter for exposure control.

    Tracks UV intensity, bulb degradation, and provides exposure adjustments.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize UV meter integration.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()
        self.readings: list[UVReading] = []
        self.calibration_date: datetime | None = None
        self.calibration_factor: float = 1.0

    def calibrate_meter(self, reference_intensity: float | None = None) -> str:
        """
        Calibrate UV meter.

        Args:
            reference_intensity: Reference intensity value

        Returns:
            Calibration status message
        """
        self.calibration_date = datetime.now()

        if reference_intensity is not None:
            # If we have recent readings, calculate calibration factor
            if self.readings:
                latest = self.readings[-1]
                if latest.intensity > 0:
                    self.calibration_factor = reference_intensity / latest.intensity

        return f"Meter calibrated at {self.calibration_date.strftime('%Y-%m-%d %H:%M')}"

    def read_intensity(
        self,
        intensity: float,
        wavelength: float | None = None,
        bulb_hours: float | None = None,
        notes: str = "",
        timestamp: datetime | None = None,
    ) -> UVReading:
        """
        Record a UV intensity reading.

        Args:
            intensity: UV intensity measurement
            wavelength: Optional wavelength in nm
            bulb_hours: Optional cumulative bulb hours
            notes: Optional notes
            timestamp: Optional timestamp (defaults to now)

        Returns:
            UVReading object
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Apply calibration factor
        calibrated_intensity = intensity * self.calibration_factor

        reading = UVReading(
            timestamp=timestamp,
            intensity=calibrated_intensity,
            wavelength=wavelength,
            bulb_hours=bulb_hours,
            notes=notes,
        )

        self.readings.append(reading)
        return reading

    def log_reading(
        self,
        intensity: float,
        wavelength: float | None = None,
        timestamp: datetime | None = None,
    ) -> UVReading:
        """
        Log a UV reading (alias for read_intensity).

        Args:
            intensity: UV intensity
            wavelength: Optional wavelength
            timestamp: Optional timestamp

        Returns:
            UVReading object
        """
        return self.read_intensity(intensity, wavelength=wavelength, timestamp=timestamp)

    def calculate_exposure_adjustment(
        self,
        target_intensity: float | None = None,
        actual_intensity: float | None = None,
    ) -> tuple[float, str]:
        """
        Calculate exposure time adjustment based on UV intensity.

        Args:
            target_intensity: Target UV intensity (defaults to settings)
            actual_intensity: Actual measured intensity (defaults to latest reading)

        Returns:
            Tuple of (adjustment_factor, message)
        """
        if target_intensity is None:
            target_intensity = self.settings.uv_intensity_target

        if actual_intensity is None:
            if not self.readings:
                return 1.0, "No UV readings available"
            actual_intensity = self.readings[-1].intensity

        if actual_intensity == 0:
            return 1.0, "Invalid UV reading (zero intensity)"

        # Adjustment factor: if actual is lower than target, need longer exposure
        adjustment = target_intensity / actual_intensity

        if adjustment > 1.2:
            message = f"Increase exposure by {(adjustment - 1) * 100:.0f}% (low UV intensity)"
        elif adjustment < 0.8:
            message = f"Decrease exposure by {(1 - adjustment) * 100:.0f}% (high UV intensity)"
        else:
            message = "UV intensity within acceptable range"

        return adjustment, message

    def check_bulb_degradation(
        self,
        readings_window_hours: int = 168,  # 1 week
    ) -> tuple[bool, str]:
        """
        Detect bulb aging based on intensity trends.

        Args:
            readings_window_hours: Time window to analyze

        Returns:
            Tuple of (needs_replacement, message)
        """
        cutoff = datetime.now() - timedelta(hours=readings_window_hours)
        recent_readings = [r for r in self.readings if r.timestamp >= cutoff]

        if len(recent_readings) < 3:
            return False, "Insufficient data for degradation analysis"

        # Calculate trend (simple linear regression)
        intensities = [r.intensity for r in recent_readings]
        initial_avg = np.mean(intensities[: len(intensities) // 3])
        recent_avg = np.mean(intensities[-len(intensities) // 3 :])

        if initial_avg == 0:
            return False, "Invalid baseline intensity"

        degradation_pct = ((initial_avg - recent_avg) / initial_avg) * 100

        if degradation_pct > self.settings.bulb_degradation_threshold:
            return True, f"Significant bulb degradation detected ({degradation_pct:.1f}% decline)"
        elif degradation_pct > self.settings.bulb_degradation_threshold / 2:
            return False, f"Moderate bulb aging observed ({degradation_pct:.1f}% decline)"
        else:
            return False, "Bulb performing normally"

    def recommend_bulb_replacement(self) -> str:
        """
        Suggest bulb replacement timing.

        Returns:
            Recommendation string
        """
        # Check if we have bulb hours data
        hours_readings = [r for r in self.readings if r.bulb_hours is not None]

        if hours_readings:
            latest_hours = hours_readings[-1].bulb_hours
            if latest_hours >= self.settings.bulb_replacement_hours:
                return f"Replace bulb: {latest_hours:.0f} hours (exceeds {self.settings.bulb_replacement_hours} hour limit)"
            else:
                remaining = self.settings.bulb_replacement_hours - latest_hours
                return f"Bulb OK: {remaining:.0f} hours remaining before recommended replacement"

        # Check degradation
        degraded, msg = self.check_bulb_degradation()
        if degraded:
            return f"Replace bulb: {msg}"

        return "Bulb replacement not currently needed based on available data"

    def get_latest_reading(self) -> UVReading | None:
        """Get the most recent UV reading."""
        if not self.readings:
            return None
        return self.readings[-1]

    def get_readings_history(self, hours: int = 24) -> list[UVReading]:
        """Get UV readings from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [r for r in self.readings if r.timestamp >= cutoff]


# ============================================================================
# Quality Report Generator
# ============================================================================


class QualityReport:
    """
    Generates comprehensive quality assurance reports.

    Combines multiple QA checks into pre-print checklists and post-print analysis.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize report generator.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()

    def generate_pre_print_checklist(
        self,
        image: np.ndarray | Image.Image | None = None,
        chemistry_tracker: ChemistryFreshnessTracker | None = None,
        humidity_checker: PaperHumidityChecker | None = None,
        uv_meter: UVLightMeterIntegration | None = None,
    ) -> dict:
        """
        Generate pre-print checklist.

        Args:
            image: Optional negative image to validate
            chemistry_tracker: Optional chemistry tracker instance
            humidity_checker: Optional humidity checker instance
            uv_meter: Optional UV meter instance

        Returns:
            Checklist dictionary
        """
        checklist = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "ready_to_print": True,
        }

        # Image density check
        if image is not None:
            validator = NegativeDensityValidator(self.settings)
            density_analysis = validator.validate_density_range(image)

            checklist["checks"]["negative_density"] = {
                "status": "pass" if not density_analysis.warnings else "warning",
                "density_range": f"{density_analysis.density_range:.2f}",
                "min_density": f"{density_analysis.min_density:.3f}",
                "max_density": f"{density_analysis.max_density:.3f}",
                "warnings": density_analysis.warnings,
                "suggestions": density_analysis.suggestions,
            }

            if density_analysis.warnings:
                checklist["warnings"].extend(density_analysis.warnings)
            if density_analysis.highlight_blocked or density_analysis.shadow_blocked:
                checklist["ready_to_print"] = False
                checklist["errors"].append("Negative has blocked highlights or shadows")

        # Chemistry freshness check
        if chemistry_tracker is not None:
            alerts = chemistry_tracker.get_alerts()
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]

            checklist["checks"]["chemistry"] = {
                "status": "fail" if critical_alerts else ("warning" if alerts else "pass"),
                "alerts": alerts,
            }

            if critical_alerts:
                checklist["ready_to_print"] = False
                checklist["errors"].extend([a["message"] for a in critical_alerts])
            elif alerts:
                checklist["warnings"].extend(
                    [a["message"] for a in alerts if a["severity"] != "critical"]
                )

        # Paper humidity check
        if humidity_checker is not None:
            ready, msg = humidity_checker.is_paper_ready()

            checklist["checks"]["paper_humidity"] = {
                "status": "pass" if ready else "fail",
                "message": msg,
            }

            if not ready:
                checklist["ready_to_print"] = False
                checklist["errors"].append(msg)

        # UV light check
        if uv_meter is not None:
            adjustment, msg = uv_meter.calculate_exposure_adjustment()
            degraded, degrad_msg = uv_meter.check_bulb_degradation()

            checklist["checks"]["uv_light"] = {
                "status": "warning" if degraded else "pass",
                "exposure_adjustment": f"{adjustment:.2f}x",
                "message": msg,
                "bulb_status": degrad_msg,
            }

            if degraded:
                checklist["warnings"].append(degrad_msg)

        return checklist

    def generate_post_print_analysis(
        self,
        scan: np.ndarray | Image.Image,
        expected_density_range: tuple[float, float] | None = None,
    ) -> dict:
        """
        Generate post-print analysis report.

        Args:
            scan: Scanned print image
            expected_density_range: Expected (min, max) density range

        Returns:
            Analysis dictionary
        """
        validator = NegativeDensityValidator(self.settings)
        analysis = validator.validate_density_range(scan)

        report = {
            "timestamp": datetime.now().isoformat(),
            "density_analysis": analysis.to_dict(),
            "quality_assessment": {},
            "recommendations": [],
        }

        # Quality assessment
        quality_score = 100.0

        # Check density range
        if analysis.density_range < 1.5:
            quality_score -= 20
            report["quality_assessment"]["contrast"] = "Low"
            report["recommendations"].append("Increase negative contrast or development time")
        elif analysis.density_range > 3.0:
            quality_score -= 10
            report["quality_assessment"]["contrast"] = "High"
            report["recommendations"].append("Reduce negative contrast for this process")
        else:
            report["quality_assessment"]["contrast"] = "Good"

        # Check for blocked areas
        if analysis.highlight_blocked:
            quality_score -= 25
            report["quality_assessment"]["highlights"] = "Blocked"
            report["recommendations"].append("Reduce exposure to recover highlight detail")
        else:
            report["quality_assessment"]["highlights"] = "Clear"

        if analysis.shadow_blocked:
            quality_score -= 15
            report["quality_assessment"]["shadows"] = "Blocked"
            report["recommendations"].append("Increase exposure or use higher platinum ratio")
        else:
            report["quality_assessment"]["shadows"] = "Good detail"

        # Compare to expected range if provided
        if expected_density_range is not None:
            exp_min, exp_max = expected_density_range
            if abs(analysis.min_density - exp_min) > 0.2:
                quality_score -= 10
                report["recommendations"].append(
                    f"Dmin deviation: expected {exp_min:.2f}, got {analysis.min_density:.2f}"
                )
            if abs(analysis.max_density - exp_max) > 0.2:
                quality_score -= 10
                report["recommendations"].append(
                    f"Dmax deviation: expected {exp_max:.2f}, got {analysis.max_density:.2f}"
                )

        report["quality_score"] = max(0, quality_score)
        report["grade"] = self._get_grade(quality_score)

        return report

    def export_report(
        self,
        report_data: dict,
        format: ReportFormat,
        output_path: Path | None = None,
    ) -> str:
        """
        Export report to specified format.

        Args:
            report_data: Report dictionary
            format: Export format
            output_path: Optional output file path

        Returns:
            Formatted report string or file path
        """
        if format == ReportFormat.JSON:
            import json

            content = json.dumps(report_data, indent=2)

        elif format == ReportFormat.MARKDOWN:
            content = self._format_as_markdown(report_data)

        elif format == ReportFormat.HTML:
            content = self._format_as_html(report_data)

        elif format == ReportFormat.PDF:
            # PDF export would require additional libraries
            # For now, generate markdown and note PDF conversion needed
            content = self._format_as_markdown(report_data)
            content = "PDF export requires conversion tool\n\n" + content
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write to file if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            return str(output_path)

        return content

    def _get_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _format_as_markdown(self, data: dict) -> str:
        """Format report as Markdown."""
        lines = [
            "# Quality Assurance Report",
            "",
            f"**Generated:** {data.get('timestamp', 'N/A')}",
            "",
        ]

        if "checks" in data:
            lines.extend(["## Pre-Print Checklist", ""])
            for check_name, check_data in data["checks"].items():
                lines.append(f"### {check_name.replace('_', ' ').title()}")
                lines.append(f"**Status:** {check_data.get('status', 'unknown').upper()}")
                lines.append("")

        if "quality_score" in data:
            lines.extend(
                [
                    "## Quality Assessment",
                    "",
                    f"**Score:** {data['quality_score']:.1f}/100 (Grade: {data['grade']})",
                    "",
                ]
            )

        if "recommendations" in data and data["recommendations"]:
            lines.extend(["## Recommendations", ""])
            for rec in data["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    def _format_as_html(self, data: dict) -> str:
        """Format report as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Assurance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .status-pass {{ color: green; }}
        .status-warning {{ color: orange; }}
        .status-fail {{ color: red; }}
        .score {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Quality Assurance Report</h1>
    <p><strong>Generated:</strong> {data.get("timestamp", "N/A")}</p>
"""

        if "quality_score" in data:
            html += f"""
    <h2>Quality Assessment</h2>
    <p class="score">Score: {data["quality_score"]:.1f}/100 (Grade: {data["grade"]})</p>
"""

        if "recommendations" in data and data["recommendations"]:
            html += "<h2>Recommendations</h2><ul>"
            for rec in data["recommendations"]:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        html += """
</body>
</html>
"""
        return html


# ============================================================================
# Alert System
# ============================================================================


class AlertSystem:
    """
    Manages quality assurance alerts.

    Tracks active alerts, dismissals, and maintains alert history.
    """

    def __init__(self, settings: QASettings | None = None):
        """
        Initialize alert system.

        Args:
            settings: QA settings. If None, uses defaults.
        """
        self.settings = settings or QASettings()
        self.alerts: dict[str, Alert] = {}

    def add_alert(
        self,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity,
        alert_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Add a new alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            alert_id: Optional custom ID (auto-generated if None)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Alert ID
        """
        if alert_id is None:
            alert_id = f"{alert_type.value}_{uuid4().hex[:8]}"

        if timestamp is None:
            timestamp = datetime.now()

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=timestamp,
        )

        self.alerts[alert_id] = alert
        return alert_id

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """
        Get list of active (non-dismissed) alerts.

        Args:
            severity: Optional filter by severity
            alert_type: Optional filter by type

        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts.values() if not a.dismissed]

        if severity is not None:
            active = [a for a in active if a.severity == severity]

        if alert_type is not None:
            active = [a for a in active if a.alert_type == alert_type]

        # Sort by severity (critical first) then timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        active.sort(key=lambda a: (severity_order[a.severity], a.timestamp))

        return active

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful, False if alert not found
        """
        if alert_id not in self.alerts:
            return False

        self.alerts[alert_id].dismissed = True
        self.alerts[alert_id].dismissed_at = datetime.now()
        return True

    def get_alert_history(
        self,
        hours: int | None = None,
        include_dismissed: bool = True,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            hours: Optional filter for alerts from last N hours
            include_dismissed: Whether to include dismissed alerts

        Returns:
            List of alerts
        """
        alerts = list(self.alerts.values())

        if not include_dismissed:
            alerts = [a for a in alerts if not a.dismissed]

        if hours is not None:
            cutoff = datetime.now() - timedelta(hours=hours)
            alerts = [a for a in alerts if a.timestamp >= cutoff]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts

    def clear_old_alerts(self) -> int:
        """
        Clear alerts older than retention period.

        Returns:
            Number of alerts cleared
        """
        cutoff = datetime.now() - timedelta(days=self.settings.alert_history_days)

        old_alerts = [
            alert_id
            for alert_id, alert in self.alerts.items()
            if alert.dismissed and alert.dismissed_at and alert.dismissed_at < cutoff
        ]

        for alert_id in old_alerts:
            del self.alerts[alert_id]

        return len(old_alerts)

    def get_alert_summary(self) -> dict[str, int]:
        """
        Get summary of active alerts by severity.

        Returns:
            Dictionary with counts by severity
        """
        active = self.get_active_alerts()

        summary = {
            "critical": 0,
            "error": 0,
            "warning": 0,
            "info": 0,
            "total": len(active),
        }

        for alert in active:
            summary[alert.severity.value] += 1

        return summary

    def get_alert(self, alert_id: str) -> Alert | None:
        """Get a specific alert by ID."""
        return self.alerts.get(alert_id)
