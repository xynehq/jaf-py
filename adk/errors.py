"""
ADK Error Handling - Production-Ready Error Management

This module provides comprehensive error handling for the ADK layer,
including circuit breakers, retry logic, and error classification.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import logging

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# ========== Error Classification ==========


class AdkErrorType(str, Enum):
    """ADK error types for classification."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONTENT_FILTER = "content_filter"
    MODEL_UNAVAILABLE = "model_unavailable"
    NETWORK = "network"
    VALIDATION = "validation"
    INTERNAL = "internal"
    CIRCUIT_BREAKER = "circuit_breaker"
    UNKNOWN = "unknown"


class AdkErrorSeverity(str, Enum):
    """ADK error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ========== Base Error Classes ==========


class AdkError(Exception):
    """Base ADK error with production features."""

    def __init__(
        self,
        message: str,
        error_type: AdkErrorType = AdkErrorType.UNKNOWN,
        severity: AdkErrorSeverity = AdkErrorSeverity.MEDIUM,
        retryable: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.retryable = retryable
        self.metadata = metadata or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        return f"[{self.error_type.value.upper()}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "cause": str(self.cause) if self.cause else None,
        }


class AdkLLMError(AdkError):
    """LLM-specific error."""

    def __init__(
        self, message: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        if provider:
            self.metadata["provider"] = provider
        if model:
            self.metadata["model"] = model


class AdkSessionError(AdkError):
    """Session-specific error."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.session_id = session_id
        self.user_id = user_id
        if session_id:
            self.metadata["session_id"] = session_id
        if user_id:
            self.metadata["user_id"] = user_id


class AdkConfigError(AdkError):
    """Configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message, error_type=AdkErrorType.VALIDATION, severity=AdkErrorSeverity.HIGH, **kwargs
        )
        self.config_key = config_key
        if config_key:
            self.metadata["config_key"] = config_key


class AdkCircuitBreakerError(AdkError):
    """Circuit breaker error."""

    def __init__(self, message: str = "Circuit breaker is open", **kwargs):
        super().__init__(
            message,
            error_type=AdkErrorType.CIRCUIT_BREAKER,
            severity=AdkErrorSeverity.HIGH,
            retryable=False,
            **kwargs,
        )


# ========== Error Factory Functions ==========


def create_adk_timeout_error(
    timeout_seconds: float, operation: str = "LLM request", **kwargs
) -> AdkLLMError:
    """Create a timeout error."""
    return AdkLLMError(
        f"{operation} timed out after {timeout_seconds}s",
        error_type=AdkErrorType.TIMEOUT,
        severity=AdkErrorSeverity.MEDIUM,
        retryable=True,
        **kwargs,
    )


def create_adk_rate_limit_error(retry_after: Optional[int] = None, **kwargs) -> AdkLLMError:
    """Create a rate limit error."""
    message = "Rate limit exceeded"
    if retry_after:
        message += f", retry after {retry_after}s"

    metadata = kwargs.get("metadata", {})
    if retry_after:
        metadata["retry_after"] = retry_after

    return AdkLLMError(
        message,
        error_type=AdkErrorType.RATE_LIMIT,
        severity=AdkErrorSeverity.HIGH,
        retryable=True,
        metadata=metadata,
        **kwargs,
    )


def create_adk_quota_error(**kwargs) -> AdkLLMError:
    """Create a quota exceeded error."""
    return AdkLLMError(
        "API quota exceeded",
        error_type=AdkErrorType.QUOTA_EXCEEDED,
        severity=AdkErrorSeverity.CRITICAL,
        retryable=False,
        **kwargs,
    )


def create_adk_content_filter_error(content: str = "", **kwargs) -> AdkLLMError:
    """Create a content filter error."""
    message = "Content filtered by provider"
    if content:
        message += f": {content[:100]}..."

    return AdkLLMError(
        message,
        error_type=AdkErrorType.CONTENT_FILTER,
        severity=AdkErrorSeverity.MEDIUM,
        retryable=False,
        **kwargs,
    )


# ========== Error Classification ==========


def classify_error(exception: Exception) -> AdkErrorType:
    """Classify an exception into an ADK error type."""
    error_message = str(exception).lower()

    # Timeout errors
    if "timeout" in error_message or "timed out" in error_message:
        return AdkErrorType.TIMEOUT

    # Rate limit errors
    if "rate limit" in error_message or "too many requests" in error_message:
        return AdkErrorType.RATE_LIMIT

    # Quota errors
    if "quota" in error_message or "exceeded" in error_message:
        return AdkErrorType.QUOTA_EXCEEDED

    # Authentication errors
    if "authentication" in error_message or "unauthorized" in error_message:
        return AdkErrorType.AUTHENTICATION

    # Authorization errors
    if "authorization" in error_message or "forbidden" in error_message:
        return AdkErrorType.AUTHORIZATION

    # Content filter errors
    if "content" in error_message and "filter" in error_message:
        return AdkErrorType.CONTENT_FILTER

    # Network errors
    if any(term in error_message for term in ["connection", "network", "dns", "resolve"]):
        return AdkErrorType.NETWORK

    return AdkErrorType.UNKNOWN


def should_retry_error(error: Union[AdkError, Exception]) -> bool:
    """Determine if an error should be retried."""
    if isinstance(error, AdkError):
        return error.retryable

    error_type = classify_error(error)
    return error_type in [AdkErrorType.TIMEOUT, AdkErrorType.RATE_LIMIT, AdkErrorType.NETWORK]


# ========== Retry Logic ==========


def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate retry delay with exponential backoff and jitter."""
    import random

    delay = min(base_delay * (exponential_base**attempt), max_delay)

    if jitter:
        delay += random.uniform(0, delay * 0.1)  # Add 10% jitter

    return delay


async def with_adk_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: Optional[Callable[[Exception], bool]] = None,
    logger: Optional[logging.Logger] = None,
) -> Callable[..., T]:
    """
    Decorator/wrapper for adding retry logic to async functions.

    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        retry_on: Function to determine if error should be retried
        logger: Logger for retry attempts
    """
    if retry_on is None:
        retry_on = should_retry_error

    if logger is None:
        logger = logging.getLogger(__name__)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries or not retry_on(e):
                    raise

                delay = calculate_retry_delay(attempt, base_delay, max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    return wrapper


# ========== Timeout Wrapper ==========


async def with_adk_timeout(
    func: Callable[..., T], timeout_seconds: float, operation_name: str = "operation"
) -> T:
    """
    Wrapper for adding timeout to async functions.

    Args:
        func: Function to wrap
        timeout_seconds: Timeout in seconds
        operation_name: Name for error messages
    """
    try:
        return await asyncio.wait_for(func, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise create_adk_timeout_error(timeout_seconds, operation_name)


# ========== Circuit Breaker ==========


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""

    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0


class CircuitBreaker:
    """Production-ready circuit breaker implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name
        self.state = CircuitBreakerState()
        self.logger = logging.getLogger(f"adk.circuit_breaker.{name}")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.state.state != "open":
            return False

        if not self.state.last_failure_time:
            return True

        time_since_failure = datetime.now() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        # Check if circuit is open and should stay open
        if self.state.state == "open" and not self._should_attempt_reset():
            raise AdkCircuitBreakerError(f"Circuit breaker '{self.name}' is open")

        # If we should attempt reset, move to half-open
        if self.state.state == "open" and self._should_attempt_reset():
            self.state.state = "half_open"
            self.state.success_count = 0
            self.logger.info(f"Circuit breaker '{self.name}' moved to half-open")

        try:
            result = await func(*args, **kwargs)

            # Success handling
            if self.state.state == "half_open":
                self.state.success_count += 1
                if self.state.success_count >= self.success_threshold:
                    self.state.state = "closed"
                    self.state.failure_count = 0
                    self.logger.info(f"Circuit breaker '{self.name}' reset to closed")
            elif self.state.state == "closed":
                self.state.failure_count = 0

            return result

        except Exception as e:
            # Failure handling
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()

            if (
                self.state.state in ["closed", "half_open"]
                and self.state.failure_count >= self.failure_threshold
            ):
                self.state.state = "open"
                self.logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.state.failure_count} failures"
                )

            raise e


def create_circuit_breaker(
    name: str, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 3
) -> CircuitBreaker:
    """Create a circuit breaker with the given parameters."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        name=name,
    )


# ========== Error Handler ==========


class AdkErrorHandler:
    """Centralized error handling for ADK operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> AdkError:
        """Handle an error and convert to ADK error if needed."""
        if isinstance(error, AdkError):
            self.logger.error(f"ADK Error: {error}", extra={"context": context})
            return error

        # Convert to ADK error
        error_type = classify_error(error)
        adk_error = AdkError(
            message=str(error),
            error_type=error_type,
            retryable=should_retry_error(error),
            cause=error,
            metadata=context or {},
        )

        self.logger.error(f"Converted Error: {adk_error}", extra={"context": context})
        return adk_error


def create_adk_error_handler(logger: Optional[logging.Logger] = None) -> AdkErrorHandler:
    """Create an ADK error handler."""
    return AdkErrorHandler(logger)
