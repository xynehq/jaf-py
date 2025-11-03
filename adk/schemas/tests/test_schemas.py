"""
Comprehensive test suite for JSON Schema validation.

Tests all enhanced validation capabilities including string, number,
and array validations with edge cases and error conditions.
"""

import pytest
from typing import Any, Dict

from adk.schemas.validation import (
    validate_schema,
    validate_string,
    validate_number,
    validate_array,
    validate_object,
)
from adk.schemas.types import JsonSchema, ValidationResult


class TestStringValidation:
    """Test enhanced string validation capabilities."""

    def test_min_length_validation(self):
        """Test minLength constraint validation."""
        schema: JsonSchema = {"type": "string", "minLength": 5}

        # Valid case
        result = validate_string("hello", schema)
        assert result.success
        assert len(result.errors) == 0

        # Invalid case
        result = validate_string("hi", schema)
        assert not result.success
        assert "String length must be at least 5" in result.errors

    def test_max_length_validation(self):
        """Test maxLength constraint validation."""
        schema: JsonSchema = {"type": "string", "maxLength": 10}

        # Valid case
        result = validate_string("short", schema)
        assert result.success

        # Invalid case
        result = validate_string("this is too long", schema)
        assert not result.success
        assert "String length must be at most 10" in result.errors

    def test_pattern_validation(self):
        """Test regex pattern validation."""
        schema: JsonSchema = {"type": "string", "pattern": r"^[A-Z][a-z]+$"}

        # Valid case
        result = validate_string("Hello", schema)
        assert result.success

        # Invalid case
        result = validate_string("hello", schema)
        assert not result.success
        assert "String does not match pattern" in result.errors[0]

    def test_invalid_pattern_handling(self):
        """Test handling of invalid regex patterns."""
        schema: JsonSchema = {"type": "string", "pattern": "["}

        result = validate_string("test", schema)
        assert not result.success
        assert "Invalid regex pattern" in result.errors[0]

    def test_email_format_validation(self):
        """Test email format validation."""
        schema: JsonSchema = {"type": "string", "format": "email"}

        # Valid emails
        valid_emails = ["test@example.com", "user.name@domain.co.uk", "a@b.c"]
        for email in valid_emails:
            result = validate_string(email, schema)
            assert result.success, f"Expected {email} to be valid"

        # Invalid emails
        invalid_emails = ["invalid", "@domain.com", "user@", "user@domain"]
        for email in invalid_emails:
            result = validate_string(email, schema)
            assert not result.success, f"Expected {email} to be invalid"
            assert "Invalid email format" in result.errors

    def test_url_format_validation(self):
        """Test URL format validation."""
        schema: JsonSchema = {"type": "string", "format": "url"}

        # Valid URLs
        valid_urls = ["https://example.com", "http://localhost:8080", "ftp://files.example.com"]
        for url in valid_urls:
            result = validate_string(url, schema)
            assert result.success, f"Expected {url} to be valid"

        # Invalid URLs
        invalid_urls = ["not-a-url", "://missing-scheme", "http://"]
        for url in invalid_urls:
            result = validate_string(url, schema)
            assert not result.success, f"Expected {url} to be invalid"

    def test_date_format_validation(self):
        """Test date format validation."""
        schema: JsonSchema = {"type": "string", "format": "date"}

        # Valid dates
        valid_dates = ["2023-01-01", "2023-12-31", "2024-02-29"]  # leap year
        for date in valid_dates:
            result = validate_string(date, schema)
            assert result.success, f"Expected {date} to be valid"

        # Invalid dates
        invalid_dates = ["2023-13-01", "2023-01-32", "23-01-01", "2023/01/01"]
        for date in invalid_dates:
            result = validate_string(date, schema)
            assert not result.success, f"Expected {date} to be invalid"

    def test_datetime_format_validation(self):
        """Test date-time format validation."""
        schema: JsonSchema = {"type": "string", "format": "date-time"}

        # Valid date-times
        valid_datetimes = [
            "2023-01-01T10:00:00Z",
            "2023-01-01T10:00:00+05:30",
            "2023-01-01T10:00:00",
        ]
        for dt in valid_datetimes:
            result = validate_string(dt, schema)
            assert result.success, f"Expected {dt} to be valid"

        # Invalid date-times
        invalid_datetimes = ["not-a-datetime", "2023-01-01", "10:00:00"]
        for dt in invalid_datetimes:
            result = validate_string(dt, schema)
            assert not result.success, f"Expected {dt} to be invalid"

    def test_uuid_format_validation(self):
        """Test UUID format validation."""
        schema: JsonSchema = {"type": "string", "format": "uuid"}

        # Valid UUIDs
        valid_uuids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "550e8400-e29b-41d4-a716-446655440000",
        ]
        for uuid in valid_uuids:
            result = validate_string(uuid, schema)
            assert result.success, f"Expected {uuid} to be valid"

        # Invalid UUIDs
        invalid_uuids = [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456",
            "123e4567-e89b-12d3-a456-42661417400g",
        ]
        for uuid in invalid_uuids:
            result = validate_string(uuid, schema)
            assert not result.success, f"Expected {uuid} to be invalid"


class TestNumberValidation:
    """Test enhanced number validation capabilities."""

    def test_minimum_validation(self):
        """Test minimum constraint validation."""
        schema: JsonSchema = {"type": "number", "minimum": 10}

        # Valid cases
        assert validate_number(10, schema).success
        assert validate_number(15, schema).success

        # Invalid case
        result = validate_number(5, schema)
        assert not result.success
        assert "Number must be at least 10" in result.errors

    def test_maximum_validation(self):
        """Test maximum constraint validation."""
        schema: JsonSchema = {"type": "number", "maximum": 100}

        # Valid cases
        assert validate_number(100, schema).success
        assert validate_number(50, schema).success

        # Invalid case
        result = validate_number(150, schema)
        assert not result.success
        assert "Number must be at most 100" in result.errors

    def test_exclusive_minimum_validation(self):
        """Test exclusiveMinimum constraint validation."""
        schema: JsonSchema = {"type": "number", "minimum": 10, "exclusiveMinimum": True}

        # Valid case
        assert validate_number(11, schema).success

        # Invalid cases
        result = validate_number(10, schema)
        assert not result.success
        assert "Number must be greater than 10" in result.errors

        result = validate_number(5, schema)
        assert not result.success

    def test_exclusive_maximum_validation(self):
        """Test exclusiveMaximum constraint validation."""
        schema: JsonSchema = {"type": "number", "maximum": 100, "exclusiveMaximum": True}

        # Valid case
        assert validate_number(99, schema).success

        # Invalid cases
        result = validate_number(100, schema)
        assert not result.success
        assert "Number must be less than 100" in result.errors

    def test_multiple_of_validation(self):
        """Test multipleOf constraint validation."""
        schema: JsonSchema = {"type": "number", "multipleOf": 3}

        # Valid cases
        assert validate_number(6, schema).success
        assert validate_number(9, schema).success
        assert validate_number(0, schema).success

        # Invalid case
        result = validate_number(7, schema)
        assert not result.success
        assert "Number must be a multiple of 3" in result.errors

    def test_multiple_of_float_precision(self):
        """Test multipleOf with floating point precision handling."""
        schema: JsonSchema = {"type": "number", "multipleOf": 0.1}

        # This should be valid despite floating point precision issues
        assert validate_number(0.3, schema).success
        assert validate_number(1.1, schema).success

    def test_integer_type_validation(self):
        """Test integer type constraint validation."""
        schema: JsonSchema = {"type": "integer"}

        # Valid cases
        assert validate_number(42, schema).success
        assert validate_number(-10, schema).success
        assert validate_number(0, schema).success

        # Invalid case (float)
        result = validate_number(3.14, schema)
        assert not result.success
        assert "Number must be an integer" in result.errors


class TestArrayValidation:
    """Test enhanced array validation capabilities."""

    def test_min_items_validation(self):
        """Test minItems constraint validation."""
        schema: JsonSchema = {"type": "array", "minItems": 2}

        # Valid case
        assert validate_array([1, 2, 3], schema).success

        # Invalid case
        result = validate_array([1], schema)
        assert not result.success
        assert "Array must have at least 2 items" in result.errors

    def test_max_items_validation(self):
        """Test maxItems constraint validation."""
        schema: JsonSchema = {"type": "array", "maxItems": 3}

        # Valid case
        assert validate_array([1, 2], schema).success

        # Invalid case
        result = validate_array([1, 2, 3, 4], schema)
        assert not result.success
        assert "Array must have at most 3 items" in result.errors

    def test_unique_items_validation(self):
        """Test uniqueItems constraint validation."""
        schema: JsonSchema = {"type": "array", "uniqueItems": True}

        # Valid cases
        assert validate_array([1, 2, 3], schema).success
        assert validate_array(["a", "b", "c"], schema).success
        assert validate_array([{"id": 1}, {"id": 2}], schema).success

        # Invalid cases
        result = validate_array([1, 2, 2], schema)
        assert not result.success
        assert "Array must contain unique items" in result.errors

        result = validate_array([{"id": 1}, {"id": 1}], schema)
        assert not result.success
        assert "Array must contain unique items" in result.errors

    def test_items_validation(self):
        """Test array items schema validation."""
        schema: JsonSchema = {"type": "array", "items": {"type": "string", "minLength": 2}}

        # Valid case
        assert validate_array(["hello", "world"], schema).success

        # Invalid case
        result = validate_array(["hello", "a"], schema)
        assert not result.success
        assert "Item 1:" in result.errors[0]
        assert "String length must be at least 2" in result.errors[0]


class TestObjectValidation:
    """Test enhanced object validation capabilities."""

    def test_required_properties_validation(self):
        """Test required properties validation."""
        schema: JsonSchema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }

        # Valid case
        assert validate_object({"name": "John", "age": 30}, schema).success

        # Invalid case - missing required property
        result = validate_object({"name": "John"}, schema)
        assert not result.success
        assert "Missing required property: age" in result.errors

    def test_additional_properties_false(self):
        """Test additionalProperties: false validation."""
        schema: JsonSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }

        # Valid case
        assert validate_object({"name": "John"}, schema).success

        # Invalid case - additional property not allowed
        result = validate_object({"name": "John", "age": 30}, schema)
        assert not result.success
        assert "Additional property not allowed: age" in result.errors

    def test_additional_properties_schema(self):
        """Test additionalProperties with schema validation."""
        schema: JsonSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "number"},
        }

        # Valid case
        assert validate_object({"name": "John", "age": 30}, schema).success

        # Invalid case - additional property doesn't match schema
        result = validate_object({"name": "John", "age": "thirty"}, schema)
        assert not result.success
        assert "Additional property 'age'" in result.errors[0]


class TestSchemaValidationEdgeCases:
    """Test edge cases and error conditions."""

    def test_null_value_handling(self):
        """Test null value validation."""
        # Schema allows null
        schema: JsonSchema = {"type": "string"}
        result = validate_schema(None, schema)
        assert not result.success

        # No type specified (should allow null)
        schema = {}
        result = validate_schema(None, schema)
        assert result.success

    def test_enum_validation(self):
        """Test enum constraint validation."""
        schema: JsonSchema = {"enum": ["red", "green", "blue"]}

        # Valid case
        assert validate_schema("red", schema).success

        # Invalid case
        result = validate_schema("yellow", schema)
        assert not result.success
        assert "Value must be one of:" in result.errors[0]

    def test_invalid_schema_format(self):
        """Test handling of invalid schema formats."""
        result = validate_schema("test", "invalid_schema")  # type: ignore
        assert not result.success
        assert "Invalid schema format" in result.errors

    def test_type_mismatch_handling(self):
        """Test type mismatch error handling."""
        schema: JsonSchema = {"type": "string"}

        result = validate_schema(123, schema)
        assert not result.success
        assert "Expected string, got int" in result.errors

    def test_complex_nested_validation(self):
        """Test complex nested object validation."""
        schema: JsonSchema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "email"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "email": {"type": "string", "format": "email"},
                            "age": {"type": "integer", "minimum": 0, "maximum": 150},
                        },
                    },
                }
            },
        }

        # Valid case
        data = {
            "users": [
                {"name": "John", "email": "john@example.com", "age": 30},
                {"name": "Jane", "email": "jane@example.com"},
            ]
        }
        assert validate_schema(data, schema).success

        # Invalid case - invalid email
        data = {"users": [{"name": "John", "email": "invalid-email", "age": 30}]}
        result = validate_schema(data, schema)
        assert not result.success
        assert "email" in str(result.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
