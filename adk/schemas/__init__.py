"""
ADK Schemas Module

Comprehensive JSON Schema validation implementation with full support for:
- String validation: minLength, maxLength, pattern, format (email, URI, date, UUID)
- Number validation: minimum, maximum, exclusiveMin/Max, multipleOf, integer check
- Array validation: minItems, maxItems, uniqueItems

This module provides enterprise-grade validation capabilities for tool parameters
and agent configurations.
"""

from .validation import (
    validate_schema,
    validate_string,
    validate_number,
    validate_array,
    validate_object,
    ValidationResult,
)
from .types import JsonSchema

__all__ = [
    "validate_schema",
    "validate_string",
    "validate_number",
    "validate_array",
    "validate_object",
    "ValidationResult",
    "JsonSchema",
]
