"""
Weather API integration for environmental conditions.

Provides weather data to help optimize platinum/palladium printing workflow,
including humidity and temperature monitoring for coating and drying times.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class WeatherCondition(str, Enum):
    """Weather condition types."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    UNKNOWN = "unknown"


class PaperType(str, Enum):
    """Paper types for drying time calculations."""

    HOT_PRESS = "hot_press"  # Smooth, low absorbency
    COLD_PRESS = "cold_press"  # Textured, high absorbency
    ROUGH = "rough"  # Very textured, very high absorbency
    SIZED = "sized"  # Hardened surface


@dataclass
class CurrentConditions:
    """Current weather conditions."""

    temperature_c: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_ms: float
    condition: WeatherCondition
    description: str
    timestamp: datetime

    @property
    def temperature_f(self) -> float:
        """Temperature in Fahrenheit."""
        return self.temperature_c * 9/5 + 32

    @property
    def is_suitable_for_coating(self) -> bool:
        """Check if conditions are suitable for coating paper."""
        # Ideal: 18-24°C, 40-60% humidity
        # Acceptable: 15-28°C, 30-70% humidity
        temp_ok = 15 <= self.temperature_c <= 28
        humidity_ok = 30 <= self.humidity_percent <= 70
        return temp_ok and humidity_ok

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "temperature_c": self.temperature_c,
            "temperature_f": self.temperature_f,
            "humidity_percent": self.humidity_percent,
            "pressure_hpa": self.pressure_hpa,
            "wind_speed_ms": self.wind_speed_ms,
            "condition": self.condition.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "suitable_for_coating": self.is_suitable_for_coating
        }


@dataclass
class ForecastPeriod:
    """Weather forecast for a specific period."""

    timestamp: datetime
    temperature_c: float
    humidity_percent: float
    condition: WeatherCondition
    description: str
    precipitation_probability: float  # 0-100

    @property
    def temperature_f(self) -> float:
        """Temperature in Fahrenheit."""
        return self.temperature_c * 9/5 + 32

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "temperature_c": self.temperature_c,
            "temperature_f": self.temperature_f,
            "humidity_percent": self.humidity_percent,
            "condition": self.condition.value,
            "description": self.description,
            "precipitation_probability": self.precipitation_probability
        }


class DryingTimeEstimate(BaseModel):
    """Estimate for paper drying time."""

    paper_type: PaperType
    estimated_hours: float = Field(description="Estimated drying time in hours")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in estimate")
    conditions: CurrentConditions = Field(description="Weather conditions used")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CoatingRecommendation(BaseModel):
    """Recommendation for when to coat paper."""

    best_time: datetime = Field(description="Recommended coating time")
    forecast: ForecastPeriod = Field(description="Forecast for recommended time")
    reason: str = Field(description="Reason for recommendation")
    alternative_times: list[datetime] = Field(default_factory=list, description="Alternative times")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WeatherProvider(ABC):
    """
    Abstract base class for weather data providers.

    Concrete implementations should handle API-specific requests and parsing.
    """

    def __init__(self, api_key: str | None = None, units: str = "metric"):
        """
        Initialize weather provider.

        Args:
            api_key: API key for the weather service
            units: Unit system (metric, imperial)
        """
        self.api_key = api_key
        self.units = units
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_duration = timedelta(minutes=10)

    @abstractmethod
    async def get_current_conditions(
        self,
        location: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> CurrentConditions:
        """
        Get current weather conditions.

        Args:
            location: Location name (e.g., "Portland, OR")
            latitude: Latitude (alternative to location)
            longitude: Longitude (alternative to location)

        Returns:
            CurrentConditions object
        """
        pass

    @abstractmethod
    async def get_forecast(
        self,
        location: str,
        hours: int = 24,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[ForecastPeriod]:
        """
        Get weather forecast.

        Args:
            location: Location name
            hours: Number of hours to forecast
            latitude: Latitude (alternative to location)
            longitude: Longitude (alternative to location)

        Returns:
            List of ForecastPeriod objects
        """
        pass

    def calculate_drying_time(
        self,
        conditions: CurrentConditions,
        paper_type: PaperType = PaperType.COLD_PRESS,
    ) -> DryingTimeEstimate:
        """
        Calculate estimated paper drying time based on conditions.

        Args:
            conditions: Current weather conditions
            paper_type: Type of paper

        Returns:
            DryingTimeEstimate with time and recommendations
        """
        # Base drying times at ideal conditions (20°C, 50% humidity)
        base_times = {
            PaperType.HOT_PRESS: 2.0,  # hours
            PaperType.COLD_PRESS: 3.0,
            PaperType.ROUGH: 4.0,
            PaperType.SIZED: 1.5,
        }

        base_time = base_times.get(paper_type, 3.0)

        # Temperature adjustment
        # Warmer = faster drying, cooler = slower
        temp_factor = 1.0
        if conditions.temperature_c < 15:
            temp_factor = 1.5
        elif conditions.temperature_c < 18:
            temp_factor = 1.2
        elif conditions.temperature_c > 25:
            temp_factor = 0.8
        elif conditions.temperature_c > 22:
            temp_factor = 0.9

        # Humidity adjustment
        # Higher humidity = slower drying
        humidity_factor = 1.0
        if conditions.humidity_percent > 70:
            humidity_factor = 1.5
        elif conditions.humidity_percent > 60:
            humidity_factor = 1.2
        elif conditions.humidity_percent < 40:
            humidity_factor = 0.8
        elif conditions.humidity_percent < 50:
            humidity_factor = 0.9

        # Calculate final estimate
        estimated_hours = base_time * temp_factor * humidity_factor

        # Generate recommendations
        recommendations = []

        if conditions.humidity_percent > 70:
            recommendations.append(
                "High humidity - consider using a dehumidifier or fan to speed drying"
            )
        elif conditions.humidity_percent < 30:
            recommendations.append(
                "Low humidity - monitor paper carefully to prevent over-drying"
            )

        if conditions.temperature_c < 15:
            recommendations.append(
                "Low temperature - consider moving to a warmer location"
            )
        elif conditions.temperature_c > 28:
            recommendations.append(
                "High temperature - ensure good ventilation to prevent uneven drying"
            )

        if conditions.is_suitable_for_coating:
            recommendations.append("Conditions are ideal for coating")
        else:
            recommendations.append("Conditions are suboptimal - consider waiting")

        # Confidence based on how far from ideal conditions
        temp_diff = abs(conditions.temperature_c - 20)
        humidity_diff = abs(conditions.humidity_percent - 50)
        confidence = max(0.5, 1.0 - (temp_diff / 20 + humidity_diff / 50) / 2)

        return DryingTimeEstimate(
            paper_type=paper_type,
            estimated_hours=round(estimated_hours, 1),
            confidence=round(confidence, 2),
            conditions=conditions,
            recommendations=recommendations
        )

    async def recommend_coating_time(
        self,
        location: str,
        forecast_hours: int = 48,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> CoatingRecommendation:
        """
        Recommend best time to coat paper based on forecast.

        Args:
            location: Location name
            forecast_hours: Hours to look ahead
            latitude: Latitude (alternative to location)
            longitude: Longitude (alternative to location)

        Returns:
            CoatingRecommendation with best time and alternatives
        """
        forecast = await self.get_forecast(
            location=location,
            hours=forecast_hours,
            latitude=latitude,
            longitude=longitude
        )

        # Score each forecast period
        scored_periods = []
        for period in forecast:
            # Ideal: 18-24°C, 40-60% humidity, no precipitation
            temp_score = 1.0 - abs(period.temperature_c - 21) / 21
            humidity_score = 1.0 - abs(period.humidity_percent - 50) / 50
            precip_score = 1.0 - period.precipitation_probability / 100

            # Weight the scores
            total_score = (
                temp_score * 0.4 +
                humidity_score * 0.4 +
                precip_score * 0.2
            )

            scored_periods.append((total_score, period))

        # Sort by score
        scored_periods.sort(key=lambda x: x[0], reverse=True)

        # Best time is highest score
        best_score, best_period = scored_periods[0]

        # Alternative times are next 2-3 best
        alternatives = [p[1].timestamp for p in scored_periods[1:4]]

        # Generate reason
        reasons = []
        if best_period.temperature_c >= 18 and best_period.temperature_c <= 24:
            reasons.append(f"ideal temperature ({best_period.temperature_c:.1f}°C)")
        if best_period.humidity_percent >= 40 and best_period.humidity_percent <= 60:
            reasons.append(f"optimal humidity ({best_period.humidity_percent:.0f}%)")
        if best_period.precipitation_probability < 20:
            reasons.append("low precipitation risk")

        reason = "Best conditions: " + ", ".join(reasons) if reasons else "Most suitable time in forecast"

        return CoatingRecommendation(
            best_time=best_period.timestamp,
            forecast=best_period,
            reason=reason,
            alternative_times=alternatives
        )

    def _get_from_cache(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < self._cache_duration:
                return value
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (datetime.now(), value)


class OpenWeatherMapProvider(WeatherProvider):
    """
    OpenWeatherMap API integration.

    Requires API key from https://openweathermap.org/api
    Free tier includes current conditions and 5-day forecast.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str | None = None, units: str = "metric"):
        """
        Initialize OpenWeatherMap provider.

        Args:
            api_key: OpenWeatherMap API key
            units: Unit system (metric, imperial)
        """
        super().__init__(api_key, units)
        self.base_url = self.BASE_URL

    async def get_current_conditions(
        self,
        location: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> CurrentConditions:
        """Get current weather from OpenWeatherMap."""
        if not self.api_key:
            logger.warning("No API key provided, returning simulated data")
            return self._simulate_current_conditions()

        # Check cache
        cache_key = f"current:{location}:{latitude}:{longitude}"
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug("Returning cached current conditions")
            from typing import cast
            return cast(CurrentConditions, cached)

        # Build request parameters
        params: dict[str, Any] = {
            "appid": self.api_key,
            "units": self.units,
        }

        if latitude is not None and longitude is not None:
            params["lat"] = latitude
            params["lon"] = longitude
        else:
            params["q"] = location

        # Make API request
        url = f"{self.base_url}/weather"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            conditions = self._parse_current_conditions(data)
            self._set_cache(cache_key, conditions)

            logger.info(f"Retrieved current conditions for {location}")
            return conditions

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            logger.warning("Falling back to simulated data")
            return self._simulate_current_conditions()

    async def get_forecast(
        self,
        location: str,
        hours: int = 24,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[ForecastPeriod]:
        """Get weather forecast from OpenWeatherMap."""
        if not self.api_key:
            logger.warning("No API key provided, returning simulated data")
            return self._simulate_forecast(hours)

        # Check cache
        cache_key = f"forecast:{location}:{latitude}:{longitude}:{hours}"
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug("Returning cached forecast")
            return cached

        # Build request parameters
        params: dict[str, Any] = {
            "appid": self.api_key,
            "units": self.units,
        }

        if latitude is not None and longitude is not None:
            params["lat"] = latitude
            params["lon"] = longitude
        else:
            params["q"] = location

        # Make API request (5-day forecast with 3-hour intervals)
        url = f"{self.base_url}/forecast"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            forecast = self._parse_forecast(data, hours)
            self._set_cache(cache_key, forecast)

            logger.info(f"Retrieved forecast for {location}: {len(forecast)} periods")
            return forecast

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            logger.warning("Falling back to simulated data")
            return self._simulate_forecast(hours)

    def _parse_current_conditions(self, data: dict) -> CurrentConditions:
        """Parse OpenWeatherMap current weather response."""
        main = data["main"]
        weather = data["weather"][0]
        wind = data["wind"]

        # Map OpenWeatherMap condition codes to our enum
        condition_map = {
            "Clear": WeatherCondition.CLEAR,
            "Clouds": WeatherCondition.CLOUDY,
            "Rain": WeatherCondition.RAIN,
            "Drizzle": WeatherCondition.RAIN,
            "Snow": WeatherCondition.SNOW,
            "Mist": WeatherCondition.FOG,
            "Fog": WeatherCondition.FOG,
        }

        condition = condition_map.get(weather["main"], WeatherCondition.UNKNOWN)

        return CurrentConditions(
            temperature_c=main["temp"],
            humidity_percent=main["humidity"],
            pressure_hpa=main["pressure"],
            wind_speed_ms=wind.get("speed", 0.0),
            condition=condition,
            description=weather["description"],
            timestamp=datetime.fromtimestamp(data["dt"])
        )

    def _parse_forecast(self, data: dict, hours: int) -> list[ForecastPeriod]:
        """Parse OpenWeatherMap forecast response."""
        forecast_list = data["list"]
        periods = []

        # OpenWeatherMap returns 3-hour intervals
        # Calculate how many periods we need
        num_periods = min(len(forecast_list), (hours + 2) // 3)

        for item in forecast_list[:num_periods]:
            main = item["main"]
            weather = item["weather"][0]

            # Map condition
            condition_map = {
                "Clear": WeatherCondition.CLEAR,
                "Clouds": WeatherCondition.CLOUDY,
                "Rain": WeatherCondition.RAIN,
                "Drizzle": WeatherCondition.RAIN,
                "Snow": WeatherCondition.SNOW,
                "Mist": WeatherCondition.FOG,
                "Fog": WeatherCondition.FOG,
            }

            condition = condition_map.get(weather["main"], WeatherCondition.UNKNOWN)

            # Get precipitation probability (if available in 3-hour forecast)
            precip_prob = item.get("pop", 0.0) * 100  # Convert to percentage

            period = ForecastPeriod(
                timestamp=datetime.fromtimestamp(item["dt"]),
                temperature_c=main["temp"],
                humidity_percent=main["humidity"],
                condition=condition,
                description=weather["description"],
                precipitation_probability=precip_prob
            )
            periods.append(period)

        return periods

    def _simulate_current_conditions(self) -> CurrentConditions:
        """Generate simulated current conditions for testing."""
        import random

        return CurrentConditions(
            temperature_c=random.uniform(15, 25),
            humidity_percent=random.uniform(40, 70),
            pressure_hpa=random.uniform(1000, 1020),
            wind_speed_ms=random.uniform(0, 5),
            condition=WeatherCondition.CLEAR,
            description="simulated clear sky",
            timestamp=datetime.now()
        )

    def _simulate_forecast(self, hours: int) -> list[ForecastPeriod]:
        """Generate simulated forecast for testing."""
        import random

        periods = []
        base_time = datetime.now()

        for i in range(0, hours, 3):  # 3-hour intervals
            periods.append(ForecastPeriod(
                timestamp=base_time + timedelta(hours=i),
                temperature_c=random.uniform(15, 25),
                humidity_percent=random.uniform(40, 70),
                condition=WeatherCondition.CLEAR,
                description="simulated conditions",
                precipitation_probability=random.uniform(0, 30)
            ))

        return periods
