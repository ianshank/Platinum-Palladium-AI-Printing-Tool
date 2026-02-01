"""
Mock object factories for PTPD Calibration tests.

Provides factories for creating mock objects and responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch


class MockLLMResponse:
    """Factory for mock LLM responses."""

    @staticmethod
    def chat_response(message: str = "This is a mock response.") -> str:
        """Create a mock chat response."""
        return message

    @staticmethod
    def recipe_response(
        paper: str = "Arches Platine",
        platinum_ml: float = 1.2,
        palladium_ml: float = 1.2,
        fo_ml: float = 2.4,
    ) -> str:
        """Create a mock recipe recommendation response."""
        return f"""Based on your requirements for {paper}, I recommend:

- Platinum solution: {platinum_ml} ml
- Palladium solution: {palladium_ml} ml
- Ferric oxalate (FO#1): {fo_ml} ml
- Contrast agent: 4-5 drops Na2

This will give you a good starting point for an 8x10 print."""

    @staticmethod
    def troubleshoot_response(problem: str = "dark prints") -> str:
        """Create a mock troubleshooting response."""
        return f"""For {problem}, here are some suggestions:

1. Check your exposure time - try reducing by 10-15%
2. Verify your negative density range
3. Check developer temperature
4. Inspect coating evenness

If the issue persists, try a new calibration test strip."""

    @staticmethod
    def error_response() -> str:
        """Create a mock error response."""
        return "I'm sorry, I encountered an error processing your request."


class MockDatabaseFactory:
    """Factory for mock database instances."""

    @staticmethod
    def empty_database() -> MagicMock:
        """Create a mock empty database."""
        mock = MagicMock()
        mock.query.return_value = []
        mock.get_statistics.return_value = {"total_records": 0}
        return mock

    @staticmethod
    def populated_database(num_records: int = 10) -> MagicMock:
        """Create a mock populated database."""
        from ptpd_calibration.core.models import CalibrationRecord
        from ptpd_calibration.core.types import (
            ChemistryType,
            ContrastAgent,
            DeveloperType,
        )

        records = []
        for i in range(num_records):
            record = CalibrationRecord(
                paper_type=f"Paper_{i}",
                exposure_time=180.0 + i * 10,
                metal_ratio=0.5,
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                contrast_agent=ContrastAgent.NA2,
                contrast_amount=5.0,
                developer=DeveloperType.POTASSIUM_OXALATE,
                measured_densities=[0.1 + j * 0.1 for j in range(21)],
            )
            records.append(record)

        mock = MagicMock()
        mock.query.return_value = records
        mock.get_statistics.return_value = {"total_records": num_records}
        mock.get_record.side_effect = lambda id: next(
            (r for r in records if str(r.id) == str(id)), None
        )
        return mock


class MockAssistantFactory:
    """Factory for mock AI assistant instances."""

    @staticmethod
    def basic_assistant() -> MagicMock:
        """Create a basic mock assistant."""
        mock = MagicMock()
        mock.chat = AsyncMock(return_value=MockLLMResponse.chat_response())
        mock.suggest_recipe = AsyncMock(return_value=MockLLMResponse.recipe_response())
        mock.troubleshoot = AsyncMock(return_value=MockLLMResponse.troubleshoot_response())
        return mock

    @staticmethod
    def failing_assistant() -> MagicMock:
        """Create an assistant that raises exceptions."""
        mock = MagicMock()
        mock.chat = AsyncMock(side_effect=Exception("LLM connection failed"))
        mock.suggest_recipe = AsyncMock(side_effect=Exception("LLM connection failed"))
        mock.troubleshoot = AsyncMock(side_effect=Exception("LLM connection failed"))
        return mock

    @staticmethod
    def slow_assistant(delay: float = 2.0) -> MagicMock:
        """Create an assistant with delayed responses."""
        import asyncio

        async def delayed_response(_message: str) -> str:
            await asyncio.sleep(delay)
            return MockLLMResponse.chat_response()

        mock = MagicMock()
        mock.chat = AsyncMock(side_effect=delayed_response)
        return mock


class MockHTTPResponseFactory:
    """Factory for mock HTTP responses."""

    @staticmethod
    def success(data: dict | None = None) -> MagicMock:
        """Create a successful HTTP response."""
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = data or {"success": True}
        mock.text = str(data or {"success": True})
        mock.ok = True
        return mock

    @staticmethod
    def not_found() -> MagicMock:
        """Create a 404 response."""
        mock = MagicMock()
        mock.status_code = 404
        mock.json.return_value = {"detail": "Not found"}
        mock.ok = False
        return mock

    @staticmethod
    def server_error() -> MagicMock:
        """Create a 500 response."""
        mock = MagicMock()
        mock.status_code = 500
        mock.json.return_value = {"detail": "Internal server error"}
        mock.ok = False
        return mock

    @staticmethod
    def validation_error(errors: list | None = None) -> MagicMock:
        """Create a 422 validation error response."""
        mock = MagicMock()
        mock.status_code = 422
        mock.json.return_value = {"detail": errors or [{"msg": "Validation error"}]}
        mock.ok = False
        return mock


class MockImageFactory:
    """Factory for mock image objects."""

    @staticmethod
    def grayscale_image(width: int = 100, height: int = 100) -> MagicMock:
        """Create a mock grayscale image."""
        mock = MagicMock()
        mock.size = (width, height)
        mock.mode = "L"
        mock.width = width
        mock.height = height
        return mock

    @staticmethod
    def rgb_image(width: int = 100, height: int = 100) -> MagicMock:
        """Create a mock RGB image."""
        mock = MagicMock()
        mock.size = (width, height)
        mock.mode = "RGB"
        mock.width = width
        mock.height = height
        return mock


class MockWeatherAPIFactory:
    """Factory for mock weather API responses."""

    @staticmethod
    def sunny(temperature: float = 22.0, humidity: float = 50.0) -> dict:
        """Create a sunny weather response."""
        return {
            "temperature": temperature,
            "humidity": humidity,
            "conditions": "sunny",
            "uv_index": 6,
        }

    @staticmethod
    def rainy(temperature: float = 18.0, humidity: float = 85.0) -> dict:
        """Create a rainy weather response."""
        return {
            "temperature": temperature,
            "humidity": humidity,
            "conditions": "rainy",
            "uv_index": 1,
        }

    @staticmethod
    def humid(temperature: float = 28.0, humidity: float = 90.0) -> dict:
        """Create a humid weather response."""
        return {
            "temperature": temperature,
            "humidity": humidity,
            "conditions": "partly_cloudy",
            "uv_index": 4,
        }


def patch_llm_client(response: str = "Mock response"):
    """Context manager to patch LLM client with mock response."""
    return patch(
        "ptpd_calibration.llm.client.LLMClient.chat",
        new=AsyncMock(return_value=response),
    )


def patch_weather_api(response: dict | None = None):
    """Context manager to patch weather API."""
    response = response or MockWeatherAPIFactory.sunny()
    return patch(
        "ptpd_calibration.integrations.weather.WeatherAPI.get_current",
        new=AsyncMock(return_value=response),
    )
