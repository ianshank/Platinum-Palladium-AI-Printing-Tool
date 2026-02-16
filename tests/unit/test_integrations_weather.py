"""Unit tests for weather integration and protocols modules.

Tests verify weather API integration, caching, parsing, and device protocols.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ptpd_calibration.integrations.protocols import (
    DensityMeasurement,
    DeviceInfo,
    DeviceManager,
    DeviceStatus,
    PrintJob,
    PrintResult,
    SpectralData,
)
from ptpd_calibration.integrations.weather import (
    CoatingRecommendation,
    CurrentConditions,
    DryingTimeEstimate,
    ForecastPeriod,
    OpenWeatherMapProvider,
    PaperType,
    WeatherCondition,
    WeatherProvider,
)


# ============================================================================
# WEATHER MODULE TESTS
# ============================================================================


class TestWeatherCondition:
    """Test WeatherCondition enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert WeatherCondition.CLEAR == "clear"
        assert WeatherCondition.CLOUDY == "cloudy"
        assert WeatherCondition.RAIN == "rain"
        assert WeatherCondition.SNOW == "snow"
        assert WeatherCondition.FOG == "fog"
        assert WeatherCondition.UNKNOWN == "unknown"


class TestPaperType:
    """Test PaperType enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert PaperType.HOT_PRESS == "hot_press"
        assert PaperType.COLD_PRESS == "cold_press"
        assert PaperType.ROUGH == "rough"
        assert PaperType.SIZED == "sized"


class TestCurrentConditions:
    """Test CurrentConditions dataclass."""

    def test_temperature_f_conversion(self) -> None:
        """Test Celsius to Fahrenheit conversion."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        # 20°C = 68°F
        assert abs(conditions.temperature_f - 68.0) < 0.1

    def test_temperature_f_freezing(self) -> None:
        """Test freezing point conversion."""
        conditions = CurrentConditions(
            temperature_c=0.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert abs(conditions.temperature_f - 32.0) < 0.1

    def test_is_suitable_for_coating_ideal(self) -> None:
        """Test ideal coating conditions."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is True

    def test_is_suitable_for_coating_too_cold(self) -> None:
        """Test unsuitable conditions - too cold."""
        conditions = CurrentConditions(
            temperature_c=10.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is False

    def test_is_suitable_for_coating_too_hot(self) -> None:
        """Test unsuitable conditions - too hot."""
        conditions = CurrentConditions(
            temperature_c=30.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is False

    def test_is_suitable_for_coating_too_dry(self) -> None:
        """Test unsuitable conditions - too dry."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=20.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is False

    def test_is_suitable_for_coating_too_humid(self) -> None:
        """Test unsuitable conditions - too humid."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=80.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is False

    def test_is_suitable_for_coating_boundary_temp(self) -> None:
        """Test boundary conditions for temperature."""
        # Lower boundary - acceptable
        conditions = CurrentConditions(
            temperature_c=15.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is True

        # Upper boundary - acceptable
        conditions.temperature_c = 28.0
        assert conditions.is_suitable_for_coating is True

    def test_is_suitable_for_coating_boundary_humidity(self) -> None:
        """Test boundary conditions for humidity."""
        # Lower boundary - acceptable
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=30.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        assert conditions.is_suitable_for_coating is True

        # Upper boundary - acceptable
        conditions.humidity_percent = 70.0
        assert conditions.is_suitable_for_coating is True

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=timestamp,
        )
        result = conditions.to_dict()

        assert result["temperature_c"] == 20.0
        assert abs(result["temperature_f"] - 68.0) < 0.1
        assert result["humidity_percent"] == 50.0
        assert result["pressure_hpa"] == 1013.25
        assert result["wind_speed_ms"] == 5.0
        assert result["condition"] == "clear"
        assert result["description"] == "clear sky"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["suitable_for_coating"] is True


class TestForecastPeriod:
    """Test ForecastPeriod dataclass."""

    def test_temperature_f_conversion(self) -> None:
        """Test Celsius to Fahrenheit conversion."""
        period = ForecastPeriod(
            timestamp=datetime.now(),
            temperature_c=25.0,
            humidity_percent=60.0,
            condition=WeatherCondition.CLOUDY,
            description="partly cloudy",
            precipitation_probability=30.0,
        )
        # 25°C = 77°F
        assert abs(period.temperature_f - 77.0) < 0.1

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        period = ForecastPeriod(
            timestamp=timestamp,
            temperature_c=25.0,
            humidity_percent=60.0,
            condition=WeatherCondition.CLOUDY,
            description="partly cloudy",
            precipitation_probability=30.0,
        )
        result = period.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert result["temperature_c"] == 25.0
        assert abs(result["temperature_f"] - 77.0) < 0.1
        assert result["humidity_percent"] == 60.0
        assert result["condition"] == "cloudy"
        assert result["description"] == "partly cloudy"
        assert result["precipitation_probability"] == 30.0


class TestDryingTimeEstimate:
    """Test DryingTimeEstimate model."""

    def test_model_creation(self) -> None:
        """Test creating a drying time estimate."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )
        estimate = DryingTimeEstimate(
            paper_type=PaperType.COLD_PRESS,
            estimated_hours=3.0,
            confidence=0.95,
            conditions=conditions,
            recommendations=["Test recommendation"],
        )

        assert estimate.paper_type == PaperType.COLD_PRESS
        assert estimate.estimated_hours == 3.0
        assert estimate.confidence == 0.95
        assert estimate.conditions == conditions
        assert estimate.recommendations == ["Test recommendation"]

    def test_model_validation_confidence_range(self) -> None:
        """Test confidence validation (0-1 range)."""
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        # Valid confidence
        estimate = DryingTimeEstimate(
            paper_type=PaperType.COLD_PRESS,
            estimated_hours=3.0,
            confidence=0.5,
            conditions=conditions,
        )
        assert estimate.confidence == 0.5

        # Invalid confidence > 1
        with pytest.raises(Exception):  # Pydantic validation error
            DryingTimeEstimate(
                paper_type=PaperType.COLD_PRESS,
                estimated_hours=3.0,
                confidence=1.5,
                conditions=conditions,
            )

        # Invalid confidence < 0
        with pytest.raises(Exception):  # Pydantic validation error
            DryingTimeEstimate(
                paper_type=PaperType.COLD_PRESS,
                estimated_hours=3.0,
                confidence=-0.5,
                conditions=conditions,
            )


class TestCoatingRecommendation:
    """Test CoatingRecommendation model."""

    def test_model_creation(self) -> None:
        """Test creating a coating recommendation."""
        timestamp = datetime.now()
        forecast = ForecastPeriod(
            timestamp=timestamp,
            temperature_c=20.0,
            humidity_percent=50.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            precipitation_probability=10.0,
        )
        recommendation = CoatingRecommendation(
            best_time=timestamp,
            forecast=forecast,
            reason="Ideal conditions",
            alternative_times=[timestamp + timedelta(hours=3)],
        )

        assert recommendation.best_time == timestamp
        assert recommendation.forecast == forecast
        assert recommendation.reason == "Ideal conditions"
        assert len(recommendation.alternative_times) == 1


class TestWeatherProvider:
    """Test WeatherProvider abstract base class."""

    class ConcreteProvider(WeatherProvider):
        """Concrete implementation for testing."""

        async def get_current_conditions(
            self,
            location: str,
            latitude: float | None = None,
            longitude: float | None = None,
        ) -> CurrentConditions:
            return CurrentConditions(
                temperature_c=20.0,
                humidity_percent=50.0,
                pressure_hpa=1013.25,
                wind_speed_ms=5.0,
                condition=WeatherCondition.CLEAR,
                description="test",
                timestamp=datetime.now(),
            )

        async def get_forecast(
            self,
            location: str,
            hours: int = 24,
            latitude: float | None = None,
            longitude: float | None = None,
        ) -> list[ForecastPeriod]:
            return [
                ForecastPeriod(
                    timestamp=datetime.now() + timedelta(hours=i),
                    temperature_c=20.0 + i,
                    humidity_percent=50.0,
                    condition=WeatherCondition.CLEAR,
                    description="test",
                    precipitation_probability=10.0,
                )
                for i in range(hours // 3)
            ]

    def test_initialization(self) -> None:
        """Test provider initialization."""
        provider = self.ConcreteProvider(api_key="test_key", units="metric")
        assert provider.api_key == "test_key"
        assert provider.units == "metric"

    def test_cache_get_set(self) -> None:
        """Test cache get and set operations."""
        provider = self.ConcreteProvider()

        # Initially empty
        assert provider._get_from_cache("test_key") is None

        # Set value
        provider._set_cache("test_key", "test_value")
        assert provider._get_from_cache("test_key") == "test_value"

    def test_cache_expiration(self) -> None:
        """Test cache expiration."""
        provider = self.ConcreteProvider()
        provider._cache_duration = timedelta(seconds=0.1)

        provider._set_cache("test_key", "test_value")
        assert provider._get_from_cache("test_key") == "test_value"

        # Wait for cache to expire
        import time
        time.sleep(0.2)

        assert provider._get_from_cache("test_key") is None

    def test_calculate_drying_time_hot_press_ideal(self) -> None:
        """Test drying time calculation for hot press in ideal conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.HOT_PRESS)

        assert estimate.paper_type == PaperType.HOT_PRESS
        assert estimate.estimated_hours == 2.0  # Base time for hot press
        assert estimate.confidence > 0.9  # High confidence for ideal conditions
        assert "ideal for coating" in " ".join(estimate.recommendations).lower()

    def test_calculate_drying_time_cold_press_ideal(self) -> None:
        """Test drying time calculation for cold press in ideal conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        assert estimate.paper_type == PaperType.COLD_PRESS
        assert estimate.estimated_hours == 3.0  # Base time for cold press

    def test_calculate_drying_time_rough_ideal(self) -> None:
        """Test drying time calculation for rough in ideal conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.ROUGH)

        assert estimate.paper_type == PaperType.ROUGH
        assert estimate.estimated_hours == 4.0  # Base time for rough

    def test_calculate_drying_time_sized_ideal(self) -> None:
        """Test drying time calculation for sized in ideal conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.SIZED)

        assert estimate.paper_type == PaperType.SIZED
        assert estimate.estimated_hours == 1.5  # Base time for sized

    def test_calculate_drying_time_cold_temperature(self) -> None:
        """Test drying time increases with cold temperature."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=10.0,  # Cold
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        # Should be longer than base time (3.0 * 1.5 = 4.5)
        assert estimate.estimated_hours > 4.0
        assert any("low temperature" in rec.lower() for rec in estimate.recommendations)

    def test_calculate_drying_time_warm_temperature(self) -> None:
        """Test drying time decreases with warm temperature."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=26.0,  # Warm
            humidity_percent=50.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        # Should be shorter than base time (3.0 * 0.8 = 2.4)
        assert estimate.estimated_hours < 3.0

    def test_calculate_drying_time_high_humidity(self) -> None:
        """Test drying time increases with high humidity."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=75.0,  # High humidity
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        # Should be longer than base time (3.0 * 1.5 = 4.5)
        assert estimate.estimated_hours > 4.0
        assert any("high humidity" in rec.lower() for rec in estimate.recommendations)

    def test_calculate_drying_time_low_humidity(self) -> None:
        """Test drying time decreases with low humidity."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=20.0,
            humidity_percent=25.0,  # Very low humidity (<30)
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        # Should be shorter than base time (3.0 * 0.8 = 2.4)
        assert estimate.estimated_hours < 3.0
        assert any("low humidity" in rec.lower() for rec in estimate.recommendations)

    def test_calculate_drying_time_extreme_conditions(self) -> None:
        """Test drying time with extreme conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=35.0,  # Very hot
            humidity_percent=20.0,  # Very dry
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        # Confidence should be lower for extreme conditions
        assert estimate.confidence < 0.8
        assert any("temperature" in rec.lower() for rec in estimate.recommendations)

    def test_calculate_drying_time_suboptimal_conditions(self) -> None:
        """Test recommendations for suboptimal conditions."""
        provider = self.ConcreteProvider()
        conditions = CurrentConditions(
            temperature_c=10.0,  # Too cold
            humidity_percent=80.0,  # Too humid
            pressure_hpa=1013.25,
            wind_speed_ms=5.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            timestamp=datetime.now(),
        )

        estimate = provider.calculate_drying_time(conditions, PaperType.COLD_PRESS)

        assert any("suboptimal" in rec.lower() or "waiting" in rec.lower()
                  for rec in estimate.recommendations)

    @pytest.mark.asyncio
    async def test_recommend_coating_time(self) -> None:
        """Test coating time recommendation."""
        provider = self.ConcreteProvider()

        recommendation = await provider.recommend_coating_time(
            location="Portland, OR",
            forecast_hours=24
        )

        assert recommendation.best_time is not None
        assert recommendation.forecast is not None
        assert recommendation.reason != ""
        assert len(recommendation.alternative_times) > 0

    @pytest.mark.asyncio
    async def test_recommend_coating_time_with_coordinates(self) -> None:
        """Test coating time recommendation with lat/lon."""
        provider = self.ConcreteProvider()

        recommendation = await provider.recommend_coating_time(
            location="",
            forecast_hours=48,
            latitude=45.5,
            longitude=-122.6
        )

        assert recommendation.best_time is not None
        assert len(recommendation.alternative_times) <= 3

    @pytest.mark.asyncio
    async def test_recommend_coating_time_scoring(self) -> None:
        """Test that recommendation selects best time based on scoring."""
        provider = self.ConcreteProvider()

        recommendation = await provider.recommend_coating_time(
            location="Portland, OR",
            forecast_hours=24
        )

        # Best time should have reasonable temperature and humidity
        best_forecast = recommendation.forecast
        assert 15.0 <= best_forecast.temperature_c <= 30.0
        assert 30.0 <= best_forecast.humidity_percent <= 80.0


class TestOpenWeatherMapProvider:
    """Test OpenWeatherMapProvider implementation."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        provider = OpenWeatherMapProvider(api_key="test_key", units="metric")
        assert provider.api_key == "test_key"
        assert provider.units == "metric"
        assert provider.base_url == "https://api.openweathermap.org/data/2.5"

    @pytest.mark.asyncio
    async def test_get_current_conditions_no_api_key(self) -> None:
        """Test that provider returns simulated data when no API key."""
        provider = OpenWeatherMapProvider(api_key=None)

        conditions = await provider.get_current_conditions(location="Portland, OR")

        assert conditions is not None
        assert conditions.description == "simulated clear sky"
        assert 15.0 <= conditions.temperature_c <= 25.0

    @pytest.mark.asyncio
    async def test_get_forecast_no_api_key(self) -> None:
        """Test that provider returns simulated forecast when no API key."""
        provider = OpenWeatherMapProvider(api_key=None)

        forecast = await provider.get_forecast(location="Portland, OR", hours=24)

        assert len(forecast) > 0
        assert all(period.description == "simulated conditions" for period in forecast)

    @pytest.mark.asyncio
    async def test_get_current_conditions_with_location(self) -> None:
        """Test getting current conditions with location name."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        mock_response_data = {
            "main": {
                "temp": 20.5,
                "humidity": 65,
                "pressure": 1013
            },
            "weather": [
                {
                    "main": "Clear",
                    "description": "clear sky"
                }
            ],
            "wind": {
                "speed": 3.5
            },
            "dt": int(datetime.now().timestamp())
        }

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            conditions = await provider.get_current_conditions(location="Portland, OR")

            assert conditions.temperature_c == 20.5
            assert conditions.humidity_percent == 65
            assert conditions.pressure_hpa == 1013
            assert conditions.wind_speed_ms == 3.5
            assert conditions.condition == WeatherCondition.CLEAR
            assert conditions.description == "clear sky"

    @pytest.mark.asyncio
    async def test_get_current_conditions_with_coordinates(self) -> None:
        """Test getting current conditions with lat/lon."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        mock_response_data = {
            "main": {
                "temp": 18.0,
                "humidity": 70,
                "pressure": 1015
            },
            "weather": [
                {
                    "main": "Clouds",
                    "description": "broken clouds"
                }
            ],
            "wind": {
                "speed": 5.0
            },
            "dt": int(datetime.now().timestamp())
        }

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            conditions = await provider.get_current_conditions(
                location="",
                latitude=45.5,
                longitude=-122.6
            )

            assert conditions.condition == WeatherCondition.CLOUDY

    @pytest.mark.asyncio
    async def test_get_current_conditions_caching(self) -> None:
        """Test that results are cached."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        mock_response_data = {
            "main": {"temp": 20.0, "humidity": 50, "pressure": 1013},
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "wind": {"speed": 2.0},
            "dt": int(datetime.now().timestamp())
        }

        call_count = 0

        async def mock_get(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # First call
            conditions1 = await provider.get_current_conditions(location="Portland, OR")
            assert call_count == 1

            # Second call should use cache
            conditions2 = await provider.get_current_conditions(location="Portland, OR")
            assert call_count == 1  # No additional API call

            assert conditions1.temperature_c == conditions2.temperature_c

    @pytest.mark.asyncio
    async def test_get_current_conditions_api_error(self) -> None:
        """Test fallback to simulated data on API error."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        async def mock_get_error(*args: Any, **kwargs: Any) -> Any:
            raise httpx.HTTPError("API error")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            conditions = await provider.get_current_conditions(location="Portland, OR")

            # Should return simulated data
            assert conditions is not None
            assert conditions.description == "simulated clear sky"

    @pytest.mark.asyncio
    async def test_get_forecast_with_location(self) -> None:
        """Test getting forecast with location name."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        mock_response_data = {
            "list": [
                {
                    "main": {"temp": 20.0 + i, "humidity": 50 + i, "pressure": 1013},
                    "weather": [{"main": "Clear", "description": "clear sky"}],
                    "pop": 0.1,
                    "dt": int((datetime.now() + timedelta(hours=i * 3)).timestamp())
                }
                for i in range(8)  # 24 hours / 3-hour intervals
            ]
        }

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            forecast = await provider.get_forecast(location="Portland, OR", hours=24)

            assert len(forecast) == 8
            assert forecast[0].temperature_c == 20.0
            assert forecast[0].precipitation_probability == 10.0

    @pytest.mark.asyncio
    async def test_get_forecast_caching(self) -> None:
        """Test that forecast is cached."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        mock_response_data = {
            "list": [
                {
                    "main": {"temp": 20.0, "humidity": 50, "pressure": 1013},
                    "weather": [{"main": "Clear", "description": "clear sky"}],
                    "pop": 0.1,
                    "dt": int(datetime.now().timestamp())
                }
            ]
        }

        call_count = 0

        async def mock_get(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=mock_response_data)
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # First call
            forecast1 = await provider.get_forecast(location="Portland, OR", hours=24)
            assert call_count == 1

            # Second call should use cache
            forecast2 = await provider.get_forecast(location="Portland, OR", hours=24)
            assert call_count == 1  # No additional API call

            assert len(forecast1) == len(forecast2)

    @pytest.mark.asyncio
    async def test_get_forecast_api_error(self) -> None:
        """Test fallback to simulated forecast on API error."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        async def mock_get_error(*args: Any, **kwargs: Any) -> Any:
            raise httpx.HTTPError("API error")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            forecast = await provider.get_forecast(location="Portland, OR", hours=24)

            # Should return simulated data
            assert len(forecast) > 0
            assert all(period.description == "simulated conditions" for period in forecast)

    def test_parse_current_conditions_clear(self) -> None:
        """Test parsing clear weather conditions."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": 22.0, "humidity": 55, "pressure": 1015},
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "wind": {"speed": 3.0},
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.condition == WeatherCondition.CLEAR
        assert conditions.temperature_c == 22.0

    def test_parse_current_conditions_rain(self) -> None:
        """Test parsing rainy weather conditions."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": 15.0, "humidity": 80, "pressure": 1005},
            "weather": [{"main": "Rain", "description": "light rain"}],
            "wind": {"speed": 5.0},
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.condition == WeatherCondition.RAIN

    def test_parse_current_conditions_snow(self) -> None:
        """Test parsing snowy weather conditions."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": -2.0, "humidity": 90, "pressure": 1010},
            "weather": [{"main": "Snow", "description": "light snow"}],
            "wind": {"speed": 2.0},
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.condition == WeatherCondition.SNOW

    def test_parse_current_conditions_fog(self) -> None:
        """Test parsing foggy weather conditions."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": 10.0, "humidity": 95, "pressure": 1013},
            "weather": [{"main": "Fog", "description": "fog"}],
            "wind": {"speed": 1.0},
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.condition == WeatherCondition.FOG

    def test_parse_current_conditions_unknown(self) -> None:
        """Test parsing unknown weather conditions."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": 18.0, "humidity": 60, "pressure": 1013},
            "weather": [{"main": "Tornado", "description": "tornado"}],
            "wind": {"speed": 50.0},
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.condition == WeatherCondition.UNKNOWN

    def test_parse_current_conditions_missing_wind(self) -> None:
        """Test parsing conditions with missing wind data."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "main": {"temp": 20.0, "humidity": 50, "pressure": 1013},
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "wind": {},  # No speed
            "dt": int(datetime.now().timestamp())
        }

        conditions = provider._parse_current_conditions(data)

        assert conditions.wind_speed_ms == 0.0

    def test_parse_forecast(self) -> None:
        """Test parsing forecast response."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        data = {
            "list": [
                {
                    "main": {"temp": 20.0 + i, "humidity": 50 + i, "pressure": 1013},
                    "weather": [{"main": "Clear", "description": "clear sky"}],
                    "pop": 0.1 * i,
                    "dt": int((datetime.now() + timedelta(hours=i * 3)).timestamp())
                }
                for i in range(5)
            ]
        }

        forecast = provider._parse_forecast(data, hours=12)

        assert len(forecast) == 4  # 12 hours / 3-hour intervals (rounded down)
        assert forecast[0].temperature_c == 20.0
        assert forecast[1].temperature_c == 21.0

    def test_parse_forecast_limits_periods(self) -> None:
        """Test that forecast parsing limits number of periods."""
        provider = OpenWeatherMapProvider(api_key="test_key")

        # Provide more data than requested
        data = {
            "list": [
                {
                    "main": {"temp": 20.0, "humidity": 50, "pressure": 1013},
                    "weather": [{"main": "Clear", "description": "clear sky"}],
                    "pop": 0.1,
                    "dt": int((datetime.now() + timedelta(hours=i * 3)).timestamp())
                }
                for i in range(40)  # 5 days of data
            ]
        }

        forecast = provider._parse_forecast(data, hours=24)

        # Should limit to 24 hours
        assert len(forecast) <= 8  # 24 hours / 3-hour intervals

    def test_simulate_current_conditions(self) -> None:
        """Test simulated current conditions generation."""
        provider = OpenWeatherMapProvider(api_key=None)

        conditions = provider._simulate_current_conditions()

        assert 15.0 <= conditions.temperature_c <= 25.0
        assert 40.0 <= conditions.humidity_percent <= 70.0
        assert 1000.0 <= conditions.pressure_hpa <= 1020.0
        assert conditions.condition == WeatherCondition.CLEAR
        assert conditions.description == "simulated clear sky"

    def test_simulate_forecast(self) -> None:
        """Test simulated forecast generation."""
        provider = OpenWeatherMapProvider(api_key=None)

        forecast = provider._simulate_forecast(hours=24)

        # Should have approximately 24/3 = 8 periods
        assert len(forecast) == 8
        assert all(15.0 <= period.temperature_c <= 25.0 for period in forecast)
        assert all(period.condition == WeatherCondition.CLEAR for period in forecast)


# ============================================================================
# PROTOCOLS MODULE TESTS
# ============================================================================


class TestDeviceStatus:
    """Test DeviceStatus enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert DeviceStatus.DISCONNECTED == "disconnected"
        assert DeviceStatus.CONNECTING == "connecting"
        assert DeviceStatus.CONNECTED == "connected"
        assert DeviceStatus.CALIBRATING == "calibrating"
        assert DeviceStatus.MEASURING == "measuring"
        assert DeviceStatus.ERROR == "error"
        assert DeviceStatus.BUSY == "busy"


class TestDeviceInfo:
    """Test DeviceInfo model."""

    def test_basic_creation(self) -> None:
        """Test creating basic device info."""
        info = DeviceInfo(vendor="Test Vendor", model="Test Model")

        assert info.vendor == "Test Vendor"
        assert info.model == "Test Model"
        assert info.serial_number is None
        assert info.firmware_version is None
        assert info.capabilities == []

    def test_full_creation(self) -> None:
        """Test creating device info with all fields."""
        info = DeviceInfo(
            vendor="X-Rite",
            model="i1Pro 3",
            serial_number="SN123456",
            firmware_version="1.2.3",
            capabilities=["reflectance", "transmission", "spectral"]
        )

        assert info.vendor == "X-Rite"
        assert info.model == "i1Pro 3"
        assert info.serial_number == "SN123456"
        assert info.firmware_version == "1.2.3"
        assert len(info.capabilities) == 3

    def test_str_representation(self) -> None:
        """Test string representation."""
        info = DeviceInfo(vendor="X-Rite", model="i1Pro 3")

        assert str(info) == "X-Rite i1Pro 3"


class TestDensityMeasurement:
    """Test DensityMeasurement model."""

    def test_basic_creation(self) -> None:
        """Test creating basic density measurement."""
        measurement = DensityMeasurement(
            density=1.5,
            lab_l=50.0,
            lab_a=0.0,
            lab_b=0.0
        )

        assert measurement.density == 1.5
        assert measurement.lab_l == 50.0
        assert measurement.lab_a == 0.0
        assert measurement.lab_b == 0.0
        assert measurement.measurement_mode == "reflection"

    def test_full_creation(self) -> None:
        """Test creating measurement with all fields."""
        timestamp = datetime.now(timezone.utc)
        measurement = DensityMeasurement(
            density=2.0,
            lab_l=30.0,
            lab_a=5.0,
            lab_b=-3.0,
            status_a_density=1.95,
            timestamp=timestamp,
            aperture_size="4mm",
            measurement_mode="transmission"
        )

        assert measurement.status_a_density == 1.95
        assert measurement.timestamp == timestamp
        assert measurement.aperture_size == "4mm"
        assert measurement.measurement_mode == "transmission"

    def test_validation_density_range(self) -> None:
        """Test density value validation."""
        # Valid density
        measurement = DensityMeasurement(
            density=2.5, lab_l=50.0, lab_a=0.0, lab_b=0.0
        )
        assert measurement.density == 2.5

        # Invalid density > 5.0
        with pytest.raises(Exception):  # Pydantic validation error
            DensityMeasurement(density=6.0, lab_l=50.0, lab_a=0.0, lab_b=0.0)

        # Invalid density < 0.0
        with pytest.raises(Exception):  # Pydantic validation error
            DensityMeasurement(density=-1.0, lab_l=50.0, lab_a=0.0, lab_b=0.0)

    def test_validation_lab_l_range(self) -> None:
        """Test L* value validation."""
        # Valid L*
        measurement = DensityMeasurement(
            density=1.0, lab_l=75.0, lab_a=0.0, lab_b=0.0
        )
        assert measurement.lab_l == 75.0

        # Invalid L* > 100
        with pytest.raises(Exception):  # Pydantic validation error
            DensityMeasurement(density=1.0, lab_l=150.0, lab_a=0.0, lab_b=0.0)

        # Invalid L* < 0
        with pytest.raises(Exception):  # Pydantic validation error
            DensityMeasurement(density=1.0, lab_l=-10.0, lab_a=0.0, lab_b=0.0)

    def test_validation_lab_a_range(self) -> None:
        """Test a* value validation."""
        # Valid a* (positive and negative)
        measurement = DensityMeasurement(
            density=1.0, lab_l=50.0, lab_a=50.0, lab_b=0.0
        )
        assert measurement.lab_a == 50.0

        measurement.lab_a = -50.0
        assert measurement.lab_a == -50.0

    def test_validation_lab_b_range(self) -> None:
        """Test b* value validation."""
        # Valid b* (positive and negative)
        measurement = DensityMeasurement(
            density=1.0, lab_l=50.0, lab_a=0.0, lab_b=60.0
        )
        assert measurement.lab_b == 60.0

        measurement.lab_b = -60.0
        assert measurement.lab_b == -60.0

    def test_xyz_conversion_white(self) -> None:
        """Test Lab to XYZ conversion for white."""
        # L*=100, a*=0, b*=0 should be close to white point
        measurement = DensityMeasurement(
            density=0.0, lab_l=100.0, lab_a=0.0, lab_b=0.0
        )

        x, y, z = measurement.xyz

        # Should be close to D50 white point
        assert abs(x - 96.422) < 1.0
        assert abs(y - 100.0) < 1.0
        assert abs(z - 82.521) < 1.0

    def test_xyz_conversion_black(self) -> None:
        """Test Lab to XYZ conversion for black."""
        # L*=0, a*=0, b*=0 should be close to black
        measurement = DensityMeasurement(
            density=5.0, lab_l=0.0, lab_a=0.0, lab_b=0.0
        )

        x, y, z = measurement.xyz

        # Should be close to zero
        assert x < 1.0
        assert y < 1.0
        assert z < 1.0

    def test_xyz_conversion_gray(self) -> None:
        """Test Lab to XYZ conversion for neutral gray."""
        # L*=50, a*=0, b*=0 should be neutral gray
        measurement = DensityMeasurement(
            density=1.0, lab_l=50.0, lab_a=0.0, lab_b=0.0
        )

        x, y, z = measurement.xyz

        # All values should be positive and proportional
        assert x > 0
        assert y > 0
        assert z > 0


class TestSpectralData:
    """Test SpectralData model."""

    def test_basic_creation(self) -> None:
        """Test creating spectral data."""
        wavelengths = list(range(380, 731, 10))
        values = [0.5] * len(wavelengths)

        data = SpectralData(wavelengths=wavelengths, values=values)

        assert len(data.wavelengths) == len(wavelengths)
        assert len(data.values) == len(values)
        assert data.start_nm == 380.0
        assert data.end_nm == 730.0
        assert data.interval_nm == 10.0

    def test_custom_range(self) -> None:
        """Test creating spectral data with custom range."""
        wavelengths = list(range(400, 701, 5))
        values = [0.3] * len(wavelengths)

        data = SpectralData(
            wavelengths=wavelengths,
            values=values,
            start_nm=400.0,
            end_nm=700.0,
            interval_nm=5.0
        )

        assert data.start_nm == 400.0
        assert data.end_nm == 700.0
        assert data.interval_nm == 5.0

    def test_len(self) -> None:
        """Test __len__ method."""
        wavelengths = list(range(380, 731, 10))
        values = [0.5] * len(wavelengths)

        data = SpectralData(wavelengths=wavelengths, values=values)

        assert len(data) == len(values)


class TestPrintJob:
    """Test PrintJob model."""

    def test_minimal_creation(self) -> None:
        """Test creating print job with minimal fields."""
        job = PrintJob(name="Test Job", image_path="/path/to/image.tiff")

        assert job.name == "Test Job"
        assert job.image_path == "/path/to/image.tiff"
        assert job.paper_size == "8x10"
        assert job.resolution_dpi == 2880
        assert job.copies == 1
        assert job.color_profile is None
        assert job.paper_type is None

    def test_full_creation(self) -> None:
        """Test creating print job with all fields."""
        job = PrintJob(
            name="Negative Print",
            image_path="/path/to/negative.tiff",
            paper_size="letter",
            resolution_dpi=3600,
            copies=3,
            color_profile="Gray-Gamma-2.2.icc",
            paper_type="Premium Glossy"
        )

        assert job.name == "Negative Print"
        assert job.resolution_dpi == 3600
        assert job.copies == 3
        assert job.color_profile == "Gray-Gamma-2.2.icc"

    def test_resolution_validation(self) -> None:
        """Test resolution DPI validation."""
        # Valid resolution
        job = PrintJob(
            name="Test", image_path="/path", resolution_dpi=1440
        )
        assert job.resolution_dpi == 1440

        # Too low
        with pytest.raises(Exception):  # Pydantic validation error
            PrintJob(name="Test", image_path="/path", resolution_dpi=100)

        # Too high
        with pytest.raises(Exception):  # Pydantic validation error
            PrintJob(name="Test", image_path="/path", resolution_dpi=10000)

    def test_copies_validation(self) -> None:
        """Test copies validation."""
        # Valid copies
        job = PrintJob(name="Test", image_path="/path", copies=5)
        assert job.copies == 5

        # Invalid (< 1)
        with pytest.raises(Exception):  # Pydantic validation error
            PrintJob(name="Test", image_path="/path", copies=0)


class TestPrintResult:
    """Test PrintResult model."""

    def test_success_result(self) -> None:
        """Test successful print result."""
        result = PrintResult(
            success=True,
            job_id="job_12345",
            pages_printed=1,
            duration_seconds=45.2
        )

        assert result.success is True
        assert result.job_id == "job_12345"
        assert result.pages_printed == 1
        assert result.error is None
        assert result.duration_seconds == 45.2

    def test_failure_result(self) -> None:
        """Test failed print result."""
        result = PrintResult(
            success=False,
            error="Printer out of paper"
        )

        assert result.success is False
        assert result.error == "Printer out of paper"
        assert result.job_id is None
        assert result.pages_printed == 0


class TestDeviceManager:
    """Test DeviceManager class."""

    class MockDevice:
        """Mock device for testing."""

        def __init__(self) -> None:
            self.status = DeviceStatus.DISCONNECTED
            self.connected = False

        def connect(self) -> bool:
            self.connected = True
            self.status = DeviceStatus.CONNECTED
            return True

        def disconnect(self) -> None:
            self.connected = False
            self.status = DeviceStatus.DISCONNECTED

    def test_initialization(self) -> None:
        """Test manager initialization."""
        device = self.MockDevice()
        manager = DeviceManager(device)

        assert manager.device == device
        assert manager.auto_reconnect is True
        assert manager.max_reconnect_attempts == 3

    def test_context_manager_connect(self) -> None:
        """Test context manager connects device."""
        device = self.MockDevice()
        manager = DeviceManager(device)

        with manager:
            assert device.connected is True
            assert device.status == DeviceStatus.CONNECTED

    def test_context_manager_disconnect(self) -> None:
        """Test context manager disconnects device."""
        device = self.MockDevice()
        manager = DeviceManager(device)

        with manager:
            pass

        assert device.connected is False
        assert device.status == DeviceStatus.DISCONNECTED

    def test_ensure_connected_when_connected(self) -> None:
        """Test ensure_connected when already connected."""
        device = self.MockDevice()
        device.status = DeviceStatus.CONNECTED
        manager = DeviceManager(device)

        result = manager.ensure_connected()

        assert result is True

    def test_ensure_connected_reconnect(self) -> None:
        """Test ensure_connected triggers reconnect."""
        device = self.MockDevice()
        manager = DeviceManager(device, auto_reconnect=True)

        result = manager.ensure_connected()

        assert result is True
        assert device.connected is True

    def test_ensure_connected_no_auto_reconnect(self) -> None:
        """Test ensure_connected without auto_reconnect."""
        device = self.MockDevice()
        manager = DeviceManager(device, auto_reconnect=False)

        result = manager.ensure_connected()

        assert result is False

    def test_device_without_protocol_methods(self) -> None:
        """Test manager with device lacking connect/disconnect."""
        simple_device = object()
        manager = DeviceManager(simple_device)

        # Should not raise errors
        with manager:
            assert manager.device == simple_device


class TestSpectrophotometerProtocol:
    """Test SpectrophotometerProtocol."""

    class MockSpectro:
        """Mock spectrophotometer implementation."""

        def __init__(self) -> None:
            self._status = DeviceStatus.DISCONNECTED
            self._device_info: DeviceInfo | None = None

        @property
        def status(self) -> DeviceStatus:
            return self._status

        @property
        def device_info(self) -> DeviceInfo | None:
            return self._device_info

        def connect(self, port: str | None = None, timeout: float = 5.0) -> bool:
            self._status = DeviceStatus.CONNECTED
            self._device_info = DeviceInfo(
                vendor="Test", model="Spectro", capabilities=["density"]
            )
            return True

        def disconnect(self) -> None:
            self._status = DeviceStatus.DISCONNECTED
            self._device_info = None

        def calibrate_white(self) -> bool:
            return True

        def calibrate_black(self) -> bool:
            return True

        def read_density(self) -> DensityMeasurement:
            return DensityMeasurement(
                density=1.5, lab_l=50.0, lab_a=0.0, lab_b=0.0
            )

        def read_spectral(self) -> SpectralData:
            wavelengths = list(range(380, 731, 10))
            return SpectralData(
                wavelengths=wavelengths,
                values=[0.5] * len(wavelengths)
            )

    def test_protocol_implementation(self) -> None:
        """Test that mock implements protocol correctly."""
        spectro = self.MockSpectro()

        # Test status
        assert spectro.status == DeviceStatus.DISCONNECTED

        # Test connect
        result = spectro.connect()
        assert result is True
        assert spectro.status == DeviceStatus.CONNECTED
        assert spectro.device_info is not None

        # Test calibration
        assert spectro.calibrate_white() is True
        assert spectro.calibrate_black() is True

        # Test measurement
        density = spectro.read_density()
        assert density.density == 1.5

        spectral = spectro.read_spectral()
        assert len(spectral) > 0

        # Test disconnect
        spectro.disconnect()
        assert spectro.status == DeviceStatus.DISCONNECTED


class TestPrinterProtocol:
    """Test PrinterProtocol."""

    class MockPrinter:
        """Mock printer implementation."""

        def __init__(self) -> None:
            self._status = DeviceStatus.DISCONNECTED
            self._device_info: DeviceInfo | None = None

        @property
        def status(self) -> DeviceStatus:
            return self._status

        @property
        def device_info(self) -> DeviceInfo | None:
            return self._device_info

        def connect(self, printer_name: str | None = None) -> bool:
            self._status = DeviceStatus.CONNECTED
            self._device_info = DeviceInfo(
                vendor="Test", model="Printer", capabilities=["color"]
            )
            return True

        def disconnect(self) -> None:
            self._status = DeviceStatus.DISCONNECTED
            self._device_info = None

        def print_image(self, job: PrintJob) -> PrintResult:
            return PrintResult(
                success=True,
                job_id="test_job_123",
                pages_printed=1,
                duration_seconds=30.0
            )

        def get_paper_sizes(self) -> list[str]:
            return ["8x10", "letter", "a4"]

        def get_resolutions(self) -> list[int]:
            return [1440, 2880, 5760]

    def test_protocol_implementation(self) -> None:
        """Test that mock implements protocol correctly."""
        printer = self.MockPrinter()

        # Test status
        assert printer.status == DeviceStatus.DISCONNECTED

        # Test connect
        result = printer.connect()
        assert result is True
        assert printer.status == DeviceStatus.CONNECTED
        assert printer.device_info is not None

        # Test capabilities
        sizes = printer.get_paper_sizes()
        assert "letter" in sizes

        resolutions = printer.get_resolutions()
        assert 2880 in resolutions

        # Test printing
        job = PrintJob(name="Test", image_path="/test.tiff")
        result = printer.print_image(job)
        assert result.success is True
        assert result.pages_printed == 1

        # Test disconnect
        printer.disconnect()
        assert printer.status == DeviceStatus.DISCONNECTED
