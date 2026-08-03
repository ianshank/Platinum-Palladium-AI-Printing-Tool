"""
Base service pattern for PTPD Calibration System.

Provides a reusable base class for implementing service components
with consistent validation, logging, and error handling.

Usage:
    from ptpd_calibration.core.base_service import BaseService, ServiceResult

    class CurveService(BaseService[CurveInput, CurveOutput]):
        def validate_input(self, data: CurveInput) -> ValidationResult:
            errors = []
            if len(data.densities) < 3:
                errors.append("At least 3 density points required")
            return ValidationResult(is_valid=len(errors) == 0, errors=errors)

        def process(self, data: CurveInput) -> CurveOutput:
            # Implementation...
            return CurveOutput(curve=generated_curve)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import LogContext, get_logger

# Type variables for input and output
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


@dataclass
class ValidationResult:
    """Result of input validation.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of validation error messages.
        warnings: List of validation warnings (non-blocking).
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid


@dataclass
class ServiceResult(Generic[TOutput]):
    """Result wrapper for service operations.

    Provides a consistent structure for returning results with
    metadata about the operation.

    Attributes:
        success: Whether the operation succeeded.
        data: The output data (if successful).
        error: Error message (if failed).
        error_type: Type of error (if failed).
        duration_seconds: Operation duration.
        timestamp: When the operation completed.
        metadata: Additional metadata about the operation.
    """

    success: bool
    data: TOutput | None = None
    error: str | None = None
    error_type: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        data: TOutput,
        duration: float = 0.0,
        **metadata: Any,
    ) -> "ServiceResult[TOutput]":
        """Create a successful result.

        Args:
            data: The output data.
            duration: Operation duration in seconds.
            **metadata: Additional metadata.

        Returns:
            ServiceResult with success=True.
        """
        return cls(
            success=True,
            data=data,
            duration_seconds=duration,
            metadata=metadata,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        error_type: str | None = None,
        duration: float = 0.0,
        **metadata: Any,
    ) -> "ServiceResult[TOutput]":
        """Create a failed result.

        Args:
            error: Error message.
            error_type: Type of error (e.g., "ValidationError").
            duration: Operation duration in seconds.
            **metadata: Additional metadata.

        Returns:
            ServiceResult with success=False.
        """
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            duration_seconds=duration,
            metadata=metadata,
        )

    def unwrap(self) -> TOutput:
        """Get the data, raising an exception if failed.

        Returns:
            The output data.

        Raises:
            RuntimeError: If the operation failed.
        """
        if not self.success or self.data is None:
            raise RuntimeError(f"Operation failed: {self.error}")
        return self.data

    def unwrap_or(self, default: TOutput) -> TOutput:
        """Get the data or return a default value.

        Args:
            default: Value to return if operation failed.

        Returns:
            The output data or the default.
        """
        return self.data if self.success and self.data is not None else default


class BaseService(ABC, Generic[TInput, TOutput]):
    """Abstract base class for service components.

    Provides consistent patterns for:
    - Input validation
    - Logging with context
    - Error handling
    - Performance measurement
    - Configuration access

    Subclasses must implement:
    - validate_input(): Validate input data
    - process(): Perform the actual operation

    Example:
        class CurveGeneratorService(BaseService[DensityInput, CurveOutput]):
            config_key = "curves"

            def validate_input(self, data: DensityInput) -> ValidationResult:
                if len(data.densities) < 3:
                    return ValidationResult(False, ["Need at least 3 points"])
                return ValidationResult(True)

            def process(self, data: DensityInput) -> CurveOutput:
                curve = self._generate_curve(data.densities)
                return CurveOutput(curve=curve)
    """

    # Override in subclass to specify which config section to use
    config_key: str | None = None

    def __init__(self, config_key: str | None = None):
        """Initialize the service.

        Args:
            config_key: Optional override for configuration key.
        """
        self._config_key = config_key or self.config_key
        self._settings = get_settings()
        self._logger = get_logger(self.__class__.__module__)

        # Get specific config section if specified
        self._config = (
            getattr(self._settings, self._config_key)
            if self._config_key and hasattr(self._settings, self._config_key)
            else self._settings
        )

    @property
    def settings(self) -> Any:
        """Access to full application settings."""
        return self._settings

    @property
    def config(self) -> Any:
        """Access to service-specific configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Access to service logger."""
        return self._logger

    @abstractmethod
    def validate_input(self, data: TInput) -> ValidationResult:
        """Validate input data before processing.

        Args:
            data: Input data to validate.

        Returns:
            ValidationResult indicating success/failure with errors.
        """
        ...

    @abstractmethod
    def process(self, data: TInput) -> TOutput:
        """Process the validated input data.

        This method is called only after validation passes.

        Args:
            data: Validated input data.

        Returns:
            Processed output data.

        Raises:
            Exception: Any exceptions during processing.
        """
        ...

    def execute(
        self,
        data: TInput,
        *,
        skip_validation: bool = False,
        context: dict[str, Any] | None = None,
    ) -> ServiceResult[TOutput]:
        """Execute the service operation with full lifecycle management.

        This is the main entry point for using the service. It handles:
        1. Input validation
        2. Logging with context
        3. Error handling
        4. Performance measurement

        Args:
            data: Input data to process.
            skip_validation: Skip validation (use with caution).
            context: Additional context for logging.

        Returns:
            ServiceResult containing the output or error information.
        """
        start_time = time.perf_counter()
        context = context or {}
        context["service"] = self.__class__.__name__
        context["input_type"] = type(data).__name__

        with LogContext(**context):
            self.logger.debug(f"Executing {self.__class__.__name__}")

            # Validation
            if not skip_validation:
                validation = self.validate_input(data)

                # Log warnings even if valid
                for warning in validation.warnings:
                    self.logger.warning(f"Validation warning: {warning}")

                if not validation.is_valid:
                    duration = time.perf_counter() - start_time
                    error_msg = "; ".join(validation.errors)
                    self.logger.error(f"Validation failed: {error_msg}")
                    return ServiceResult.fail(
                        error=error_msg,
                        error_type="ValidationError",
                        duration=duration,
                        validation_errors=validation.errors,
                    )

            # Processing
            try:
                result = self.process(data)
                duration = time.perf_counter() - start_time

                self.logger.info(
                    f"Successfully processed {type(data).__name__}",
                    extra={"duration_seconds": duration},
                )

                return ServiceResult.ok(
                    data=result,
                    duration=duration,
                )

            except Exception as e:
                duration = time.perf_counter() - start_time
                self.logger.exception(
                    f"Processing failed: {e}", extra={"duration_seconds": duration}
                )

                return ServiceResult.fail(
                    error=str(e),
                    error_type=type(e).__name__,
                    duration=duration,
                )

    def __call__(self, data: TInput) -> TOutput:
        """Allow calling service directly like a function.

        Raises exception on failure instead of returning ServiceResult.

        Args:
            data: Input data to process.

        Returns:
            Output data.

        Raises:
            ValueError: If validation fails.
            Exception: If processing fails.
        """
        result = self.execute(data)
        if not result.success:
            if result.error_type == "ValidationError":
                raise ValueError(result.error)
            raise RuntimeError(result.error)
        return result.unwrap()


class AsyncBaseService(ABC, Generic[TInput, TOutput]):
    """Async version of BaseService for I/O-bound operations.

    Use this when your service needs to perform async operations
    (network calls, file I/O, etc.).

    Example:
        class ExternalAPIService(AsyncBaseService[Request, Response]):
            async def validate_input(self, data: Request) -> ValidationResult:
                return ValidationResult(True)

            async def process(self, data: Request) -> Response:
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.api_url, json=data.dict())
                return Response.parse_obj(response.json())
    """

    config_key: str | None = None

    def __init__(self, config_key: str | None = None):
        """Initialize the async service."""
        self._config_key = config_key or self.config_key
        self._settings = get_settings()
        self._logger = get_logger(self.__class__.__module__)
        self._config = (
            getattr(self._settings, self._config_key)
            if self._config_key and hasattr(self._settings, self._config_key)
            else self._settings
        )

    @property
    def settings(self) -> Any:
        return self._settings

    @property
    def config(self) -> Any:
        return self._config

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @abstractmethod
    async def validate_input(self, data: TInput) -> ValidationResult:
        """Async validation of input data."""
        ...

    @abstractmethod
    async def process(self, data: TInput) -> TOutput:
        """Async processing of input data."""
        ...

    async def execute(
        self,
        data: TInput,
        *,
        skip_validation: bool = False,
        context: dict[str, Any] | None = None,
    ) -> ServiceResult[TOutput]:
        """Execute the async service operation."""
        start_time = time.perf_counter()
        context = context or {}
        context["service"] = self.__class__.__name__

        with LogContext(**context):
            self.logger.debug(f"Executing async {self.__class__.__name__}")

            if not skip_validation:
                validation = await self.validate_input(data)

                for warning in validation.warnings:
                    self.logger.warning(f"Validation warning: {warning}")

                if not validation.is_valid:
                    duration = time.perf_counter() - start_time
                    return ServiceResult.fail(
                        error="; ".join(validation.errors),
                        error_type="ValidationError",
                        duration=duration,
                    )

            try:
                result = await self.process(data)
                duration = time.perf_counter() - start_time

                self.logger.info(
                    f"Successfully processed {type(data).__name__}",
                    extra={"duration_seconds": duration},
                )

                return ServiceResult.ok(data=result, duration=duration)

            except Exception as e:
                duration = time.perf_counter() - start_time
                self.logger.exception(f"Processing failed: {e}")

                return ServiceResult.fail(
                    error=str(e),
                    error_type=type(e).__name__,
                    duration=duration,
                )

    async def __call__(self, data: TInput) -> TOutput:
        """Allow calling async service directly."""
        result = await self.execute(data)
        if not result.success:
            if result.error_type == "ValidationError":
                raise ValueError(result.error)
            raise RuntimeError(result.error)
        return result.unwrap()
