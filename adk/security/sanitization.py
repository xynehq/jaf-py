"""
ADK Input Sanitization - Production-Ready Input Validation

This module provides comprehensive input sanitization to prevent injection attacks,
XSS, and other security vulnerabilities in user-provided data.
"""

import re
import html
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SanitizationLevel(str, Enum):
    """Levels of input sanitization."""

    STRICT = "strict"  # Maximum security, limited functionality
    MODERATE = "moderate"  # Balanced security and usability
    PERMISSIVE = "permissive"  # Minimal security, maximum functionality


@dataclass(frozen=True)
class SanitizationResult:
    """Result of input sanitization."""

    sanitized_input: str
    is_safe: bool
    detected_issues: List[str]
    original_length: int
    sanitized_length: int
    security_level: SanitizationLevel


class AdkInputSanitizer:
    """
    Production-ready input sanitizer with configurable security levels.

    This sanitizer protects against various injection attacks including:
    - SQL injection patterns
    - XSS (Cross-Site Scripting)
    - Command injection
    - LLM prompt injection
    - Path traversal attacks
    """

    # Dangerous patterns that indicate potential injection attacks
    INJECTION_PATTERNS = {
        "sql_injection": [
            r"(?i)\b(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(?i)\b(exec|execute|sp_|xp_)\w*\s*\(",
            r"(?i)(\;|\||\&|\$|\`)",
            r"(?i)(\'|\")\s*(or|and)\s*\1\s*=\s*\1",
            r"(?i)\b(or|and)\s+\d+\s*=\s*\d+",
        ],
        "xss_injection": [
            r"(?i)<\s*script[^>]*>",
            r"(?i)<\s*iframe[^>]*>",
            r"(?i)<\s*object[^>]*>",
            r"(?i)<\s*embed[^>]*>",
            r"(?i)javascript\s*:",
            r"(?i)vbscript\s*:",
            r"(?i)data\s*:",
            r"(?i)on\w+\s*=",
        ],
        "command_injection": [
            r"(?i)\b(rm|del|format|fdisk|shutdown|reboot)\s+",
            r"(?i)\b(curl|wget|nc|netcat|telnet)\s+",
            r"(?i)(\;|\||\&)\s*(rm|del|cat|ls|dir|type)\s+",
            r"(?i)\$\(.*\)",
            r"(?i)\`.*\`",
            r"(?i)>\s*/dev/",
        ],
        "llm_injection": [
            r"(?i)ignore\s+(previous|all)\s+(instructions|prompts?)",
            r"(?i)(forget|disregard)\s+(everything|all|that)",
            r"(?i)you\s+are\s+now\s+(a|an)\s+",
            r"(?i)new\s+(role|character|persona)",
            r"(?i)act\s+as\s+(if|though)\s+you\s+(are|were)",
            r"(?i)pretend\s+(you\s+are|to\s+be)",
            r"(?i)roleplay\s+as",
            r"(?i)system\s*:\s*",
            r"(?i)assistant\s*:\s*",
            r"(?i)\[SYSTEM\]",
            r"(?i)\[ASSISTANT\]",
            r"(?i)<\|.*\|>",
        ],
        "path_traversal": [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"(?i)\\x2e\\x2e[\\/]",
            r"(?i)%2e%2e[\\/]",
            r"(?i)\.\.%2f",
            r"(?i)\.\.%5c",
        ],
    }

    # Characters that are commonly dangerous in various contexts
    DANGEROUS_CHARS = {
        SanitizationLevel.STRICT: "<>\"';(){}[]|&$`\\",
        SanitizationLevel.MODERATE: "<>\"';(){}|&$`",
        SanitizationLevel.PERMISSIVE: "<>\"'",
    }

    def __init__(self, level: SanitizationLevel = SanitizationLevel.MODERATE):
        """
        Initialize the sanitizer with a specific security level.

        Args:
            level: Security level for sanitization
        """
        self.level = level
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for better performance."""
        compiled = {}
        for category, patterns in self.INJECTION_PATTERNS.items():
            compiled[category] = [re.compile(pattern) for pattern in patterns]
        return compiled

    def sanitize(self, input_text: str, max_length: int = 10000) -> SanitizationResult:
        """
        Sanitize input text according to the configured security level.

        Args:
            input_text: Text to sanitize
            max_length: Maximum allowed length

        Returns:
            SanitizationResult with sanitized text and security info
        """
        if not input_text:
            return SanitizationResult(
                sanitized_input="",
                is_safe=True,
                detected_issues=[],
                original_length=0,
                sanitized_length=0,
                security_level=self.level,
            )

        original_length = len(input_text)
        detected_issues = []

        # Length validation
        if original_length > max_length:
            detected_issues.append(f"Input too long: {original_length} > {max_length}")
            input_text = input_text[:max_length]

        # Detect injection patterns
        injection_issues = self._detect_injection_patterns(input_text)
        detected_issues.extend(injection_issues)

        # Apply sanitization based on security level
        sanitized = self._apply_sanitization(input_text)

        return SanitizationResult(
            sanitized_input=sanitized,
            is_safe=len(detected_issues) == 0,
            detected_issues=detected_issues,
            original_length=original_length,
            sanitized_length=len(sanitized),
            security_level=self.level,
        )

    def _detect_injection_patterns(self, text: str) -> List[str]:
        """Detect potential injection patterns in text."""
        issues = []

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    issues.append(f"Potential {category.replace('_', ' ')} detected")
                    break  # Only report once per category

        return issues

    def _apply_sanitization(self, text: str) -> str:
        """Apply sanitization based on security level."""
        sanitized = text

        # HTML escape for XSS prevention
        sanitized = html.escape(sanitized, quote=True)

        # Remove or escape dangerous characters based on level
        dangerous_chars = self.DANGEROUS_CHARS[self.level]

        if self.level == SanitizationLevel.STRICT:
            # Remove dangerous characters entirely
            sanitized = "".join(c for c in sanitized if c not in dangerous_chars)
        elif self.level == SanitizationLevel.MODERATE:
            # Escape dangerous characters
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, f"\\{char}")
        # PERMISSIVE level only does HTML escaping

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        return sanitized


# Convenience functions for common sanitization tasks


def sanitize_llm_prompt(prompt: str, max_length: int = 10000) -> str:
    """
    Sanitize an LLM prompt to prevent injection attacks.

    Args:
        prompt: Prompt text to sanitize
        max_length: Maximum allowed prompt length

    Returns:
        Sanitized prompt text

    Raises:
        ValueError: If the prompt contains dangerous injection patterns
    """
    sanitizer = AdkInputSanitizer(SanitizationLevel.STRICT)
    result = sanitizer.sanitize(prompt, max_length)

    if not result.is_safe:
        raise ValueError(f"Unsafe prompt detected: {', '.join(result.detected_issues)}")

    return result.sanitized_input


def sanitize_user_input(
    user_input: str, level: SanitizationLevel = SanitizationLevel.MODERATE
) -> str:
    """
    Sanitize general user input.

    Args:
        user_input: User input to sanitize
        level: Security level for sanitization

    Returns:
        Sanitized user input
    """
    sanitizer = AdkInputSanitizer(level)
    result = sanitizer.sanitize(user_input)
    return result.sanitized_input


def validate_input_length(text: str, max_length: int, field_name: str = "input") -> None:
    """
    Validate input length and raise an error if too long.

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        field_name: Name of the field for error messages

    Raises:
        ValueError: If text exceeds maximum length
    """
    if len(text) > max_length:
        raise ValueError(f"{field_name} too long: {len(text)} > {max_length} characters")


def detect_injection_patterns(text: str) -> List[str]:
    """
    Detect potential injection patterns in text.

    Args:
        text: Text to analyze

    Returns:
        List of detected injection pattern types
    """
    sanitizer = AdkInputSanitizer()
    issues = sanitizer._detect_injection_patterns(text)
    return issues


def is_safe_input(text: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> bool:
    """
    Check if input is safe according to the specified security level.

    Args:
        text: Text to check
        level: Security level for validation

    Returns:
        True if input is considered safe
    """
    sanitizer = AdkInputSanitizer(level)
    result = sanitizer.sanitize(text)
    return result.is_safe


# Content filtering for specific domains


def sanitize_file_path(path: str) -> str:
    """
    Sanitize a file path to prevent path traversal attacks.

    Args:
        path: File path to sanitize

    Returns:
        Sanitized file path

    Raises:
        ValueError: If path contains dangerous patterns
    """
    if not path:
        raise ValueError("Empty path not allowed")

    # Check for path traversal patterns
    dangerous_patterns = ["..", "\\", "//", "%2e%2e", "\\x2e\\x2e"]

    path_lower = path.lower()
    for pattern in dangerous_patterns:
        if pattern in path_lower:
            raise ValueError(f"Dangerous path pattern detected: {pattern}")

    # Normalize path separators and remove dangerous characters
    sanitized = path.replace("\\", "/").replace("//", "/")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ".-_/")

    return sanitized


def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitize an SQL identifier (table name, column name, etc.).

    Args:
        identifier: SQL identifier to sanitize

    Returns:
        Sanitized SQL identifier

    Raises:
        ValueError: If identifier is invalid
    """
    if not identifier:
        raise ValueError("Empty identifier not allowed")

    # Only allow alphanumeric characters and underscores
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError("Invalid SQL identifier format")

    # Check for SQL keywords (basic list)
    sql_keywords = {
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "table",
        "index",
        "view",
        "trigger",
        "procedure",
        "function",
    }

    if identifier.lower() in sql_keywords:
        raise ValueError(f"SQL keyword not allowed as identifier: {identifier}")

    return identifier


def sanitize_email(email: str) -> str:
    """
    Sanitize and validate an email address.

    Args:
        email: Email address to sanitize

    Returns:
        Sanitized email address

    Raises:
        ValueError: If email format is invalid
    """
    if not email:
        raise ValueError("Empty email not allowed")

    # Basic email format validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email format")

    return email.lower().strip()


# Security monitoring helpers


def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Log a security event for monitoring and alerting.

    Args:
        event_type: Type of security event
        details: Event details
    """
    # In a production system, this would integrate with logging infrastructure
    import json
    from datetime import datetime

    event = {"timestamp": datetime.now().isoformat(), "event_type": event_type, "details": details}

    # For now, just print to stderr (in production, use proper logging)
    import sys

    print(f"[SECURITY EVENT] {json.dumps(event)}", file=sys.stderr)
