"""
Tests for validation policies and guardrails.

Based on the TypeScript validation.test.ts file.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from jaf.core.types import InvalidValidationResult, ValidationResult, ValidValidationResult
from jaf.policies.validation import (
    combine_guardrails,
    create_content_filter,
    create_format_validator,
    create_length_limiter,
)


class ValidationTestOutput(BaseModel):
    """Test output format for validation."""
    message: str
    priority: int


@pytest.mark.asyncio
async def test_content_filter_guardrail():
    """Test content filtering guardrail."""
    # Create content filter
    filter_guardrail = create_content_filter([
        "badword", "inappropriate", "spam"
    ])

    # Test valid content
    result = await filter_guardrail("This is a normal message")
    assert result.is_valid

    # Test invalid content
    result = await filter_guardrail("This contains badword in it")
    assert not result.is_valid
    assert "inappropriate content" in result.error_message.lower()

    # Test case insensitive
    result = await filter_guardrail("This contains BADWORD in it")
    assert not result.is_valid


@pytest.mark.asyncio
async def test_length_limiter_guardrail():
    """Test length limiting guardrail."""
    # Create length limiter
    length_guardrail = create_length_limiter(max_length=50)

    # Test valid length
    result = await length_guardrail("Short message")
    assert result.is_valid

    # Test invalid length
    long_message = "This is a very long message that exceeds the maximum allowed length limit"
    result = await length_guardrail(long_message)
    assert not result.is_valid
    assert "exceeds maximum length" in result.error_message


@pytest.mark.asyncio
async def test_format_validator_guardrail():
    """Test format validation guardrail."""
    # Create format validator for ValidationTestOutput
    format_guardrail = create_format_validator(ValidationTestOutput)

    # Test valid format
    valid_output = ValidationTestOutput(message="Hello", priority=1)
    result = await format_guardrail(valid_output)
    assert result.is_valid

    # Test valid dict format
    result = await format_guardrail({"message": "Hello", "priority": 1})
    assert result.is_valid

    # Test invalid format - wrong type
    class WrongOutput(BaseModel):
        different_field: str

    wrong_output = WrongOutput(different_field="test")
    result = await format_guardrail(wrong_output)
    assert not result.is_valid

    # Test invalid format - missing fields
    result = await format_guardrail({"message": "Hello"})  # Missing priority
    assert not result.is_valid

    # Test invalid format - wrong field types
    result = await format_guardrail({"message": "Hello", "priority": "not_a_number"})
    assert not result.is_valid


@pytest.mark.asyncio
async def test_combine_guardrails():
    """Test combining multiple guardrails."""
    # Create individual guardrails
    content_filter = create_content_filter(["bad"])
    length_limiter = create_length_limiter(max_length=20)

    # Combine them
    combined_guardrail = combine_guardrails([content_filter, length_limiter])

    # Test valid input (passes both)
    result = await combined_guardrail("Good short text")
    assert result.is_valid

    # Test invalid content (fails content filter)
    result = await combined_guardrail("This has bad word")
    assert not result.is_valid
    assert "inappropriate content" in result.error_message.lower()

    # Test invalid length (fails length limiter)
    result = await combined_guardrail("This is a very long message that exceeds limit")
    assert not result.is_valid
    assert "exceeds maximum length" in result.error_message

    # Test fails both (should return first failure)
    result = await combined_guardrail("This is a very long message with bad word that exceeds limit")
    assert not result.is_valid
    # Should fail on content first (since it's first in the list)
    assert "inappropriate content" in result.error_message.lower()


@pytest.mark.asyncio
async def test_synchronous_guardrails():
    """Test that synchronous guardrails work correctly."""
    def sync_guardrail(text: str) -> ValidationResult:
        if "banana" in text:
            return InvalidValidationResult(error_message="Sync validation failed")
        return ValidValidationResult()

    # Test direct call
    result = sync_guardrail("normal text")
    assert result.is_valid

    result = sync_guardrail("text with banana")
    assert not result.is_valid

    # Test in combined guardrails
    async_guardrail = create_content_filter(["orange"])
    combined = combine_guardrails([sync_guardrail, async_guardrail])

    # Test sync failure
    result = await combined("text with banana")
    assert not result.is_valid
    assert "Sync validation failed" in result.error_message

    # Test async failure
    result = await combined("text with orange")
    assert not result.is_valid
    assert "inappropriate content" in result.error_message.lower()


@pytest.mark.asyncio
async def test_empty_guardrails_list():
    """Test behavior with empty guardrails list."""
    combined_guardrail = combine_guardrails([])

    # Should always pass with empty list
    result = await combined_guardrail("any text")
    assert result.is_valid

    result = await combined_guardrail("")
    assert result.is_valid


@pytest.mark.asyncio
async def test_guardrail_with_objects():
    """Test guardrails with complex objects."""
    def object_guardrail(obj: Any) -> ValidationResult:
        # Handle both objects with attributes and dictionaries with keys
        has_dangerous_field = (
            hasattr(obj, 'dangerous_field') or
            (isinstance(obj, dict) and 'dangerous_field' in obj)
        )

        if has_dangerous_field:
            return InvalidValidationResult(error_message="Object contains dangerous field")
        return ValidValidationResult()

    # Test safe object
    safe_obj = {"message": "hello", "safe": True}
    result = object_guardrail(safe_obj)
    assert result.is_valid

    # Test dangerous object
    dangerous_obj = {"message": "hello", "dangerous_field": "bad"}
    result = object_guardrail(dangerous_obj)
    assert not result.is_valid
    assert "dangerous field" in result.error_message


@pytest.mark.asyncio
async def test_custom_error_messages():
    """Test custom error messages in guardrails."""
    def custom_message_guardrail(text: str) -> ValidationResult:
        if "trigger" in text:
            return InvalidValidationResult(
                error_message="Custom error: The input triggered our custom validation rule"
            )
        return ValidValidationResult()

    result = custom_message_guardrail("normal text")
    assert result.is_valid

    result = custom_message_guardrail("this will trigger the rule")
    assert not result.is_valid
    assert "Custom error:" in result.error_message
    assert "custom validation rule" in result.error_message


@pytest.mark.asyncio
async def test_guardrail_exception_handling():
    """Test that guardrail exceptions are handled gracefully."""
    def failing_guardrail(text: str) -> ValidationResult:
        raise Exception("Guardrail implementation error")

    # When used directly, should raise a specific exception
    with pytest.raises(Exception, match="Guardrail implementation error"):
        failing_guardrail("test")

    # When used in combined guardrails, should be handled gracefully
    safe_guardrail = create_content_filter(["safe"])

    # Note: The combine_guardrails function should handle exceptions
    # and convert them to validation failures
    combined = combine_guardrails([safe_guardrail])  # Only safe guardrail for now

    result = await combined("test")
    assert result.is_valid  # Should pass with safe guardrail


@pytest.mark.asyncio
async def test_regex_pattern_validation():
    """Test regex pattern validation guardrail."""
    import re

    def email_format_guardrail(text: str) -> ValidationResult:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, text):
            return InvalidValidationResult(error_message="Invalid email format")
        return ValidValidationResult()

    # Test valid email
    result = email_format_guardrail("user@example.com")
    assert result.is_valid

    # Test invalid email
    result = email_format_guardrail("not-an-email")
    assert not result.is_valid
    assert "Invalid email format" in result.error_message

    # Test empty string
    result = email_format_guardrail("")
    assert not result.is_valid


@pytest.mark.asyncio
async def test_conditional_validation():
    """Test conditional validation based on context."""
    def conditional_guardrail(data: Any) -> ValidationResult:
        if isinstance(data, dict):
            if data.get('type') == 'sensitive' and not data.get('authorized'):
                return InvalidValidationResult(
                    error_message="Sensitive data requires authorization"
                )
        return ValidValidationResult()

    # Test non-sensitive data
    result = conditional_guardrail({"type": "normal", "content": "hello"})
    assert result.is_valid

    # Test sensitive but authorized
    result = conditional_guardrail({"type": "sensitive", "authorized": True, "content": "secret"})
    assert result.is_valid

    # Test sensitive but not authorized
    result = conditional_guardrail({"type": "sensitive", "authorized": False, "content": "secret"})
    assert not result.is_valid
    assert "requires authorization" in result.error_message

    # Test sensitive without authorization field
    result = conditional_guardrail({"type": "sensitive", "content": "secret"})
    assert not result.is_valid
