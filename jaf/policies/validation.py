"""
Validation policies for the JAF framework.

This module provides guardrails and validation functions to ensure
safe and reliable agent behavior.
"""

import re
import json
import time
from typing import Any, Dict, List, Optional, Set, Callable, Union, Awaitable
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError

from ..core.types import ValidationResult, Guardrail

@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    enabled: bool = True
    strict_mode: bool = False
    custom_message: Optional[str] = None

def create_length_guardrail(
    max_length: int,
    min_length: int = 0,
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """
    Create a guardrail that validates text length.
    
    Args:
        max_length: Maximum allowed length
        min_length: Minimum required length
        config: Optional guardrail configuration
        
    Returns:
        Guardrail function
    """
    config = config or GuardrailConfig()
    
    def length_guardrail(text: str) -> ValidationResult:
        if not config.enabled:
            return ValidationResult(is_valid=True)
        
        text_length = len(text)
        
        if text_length > max_length:
            message = (config.custom_message or 
                      f"Text length {text_length} exceeds maximum {max_length}")
            return ValidationResult(is_valid=False, error_message=message)
        
        if text_length < min_length:
            message = (config.custom_message or 
                      f"Text length {text_length} below minimum {min_length}")
            return ValidationResult(is_valid=False, error_message=message)
        
        return ValidationResult(is_valid=True)
    
    return length_guardrail

def create_content_filter_guardrail(
    blocked_patterns: List[str],
    allowed_patterns: Optional[List[str]] = None,
    case_sensitive: bool = False,
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """
    Create a guardrail that filters content based on patterns.
    
    Args:
        blocked_patterns: Regex patterns that should be blocked
        allowed_patterns: Regex patterns that override blocks (allowlist)
        case_sensitive: Whether pattern matching is case sensitive
        config: Optional guardrail configuration
        
    Returns:
        Guardrail function
    """
    config = config or GuardrailConfig()
    
    # Compile patterns for efficiency
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_blocked = [re.compile(pattern, flags) for pattern in blocked_patterns]
    compiled_allowed = ([re.compile(pattern, flags) for pattern in allowed_patterns] 
                       if allowed_patterns else [])
    
    def content_filter_guardrail(text: str) -> ValidationResult:
        if not config.enabled:
            return ValidationResult(is_valid=True)
        
        # Check blocked patterns
        for pattern in compiled_blocked:
            if pattern.search(text):
                # Check if allowed patterns override the block
                is_allowed = any(allowed.search(text) for allowed in compiled_allowed)
                
                if not is_allowed:
                    message = (config.custom_message or 
                              f"Content blocked by pattern: {pattern.pattern}")
                    return ValidationResult(is_valid=False, error_message=message)
        
        return ValidationResult(is_valid=True)
    
    return content_filter_guardrail

def create_json_validation_guardrail(
    schema_class: type[BaseModel],
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """
    Create a guardrail that validates JSON against a Pydantic schema.
    
    Args:
        schema_class: Pydantic model class for validation
        config: Optional guardrail configuration
        
    Returns:
        Guardrail function
    """
    config = config or GuardrailConfig()
    
    def json_validation_guardrail(data: Any) -> ValidationResult:
        if not config.enabled:
            return ValidationResult(is_valid=True)
        
        try:
            # If data is a string, try to parse as JSON
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    message = (config.custom_message or 
                              f"Invalid JSON format: {str(e)}")
                    return ValidationResult(is_valid=False, error_message=message)
            
            # Validate against schema
            schema_class.model_validate(data)
            return ValidationResult(is_valid=True)
            
        except ValidationError as e:
            message = (config.custom_message or 
                      f"Schema validation failed: {str(e)}")
            return ValidationResult(is_valid=False, error_message=message)
        except Exception as e:
            message = (config.custom_message or 
                      f"Validation error: {str(e)}")
            return ValidationResult(is_valid=False, error_message=message)
    
    return json_validation_guardrail

@dataclass
class RateLimitState:
    """State for rate limiting."""
    calls: List[float] = field(default_factory=list)
    window_size: float = 60.0  # seconds
    max_calls: int = 10

def create_rate_limit_guardrail(
    max_calls: int = 10,
    window_size: float = 60.0,  # seconds
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """
    Create a guardrail that implements rate limiting.
    
    Args:
        max_calls: Maximum number of calls allowed in the window
        window_size: Time window in seconds
        config: Optional guardrail configuration
        
    Returns:
        Guardrail function
    """
    config = config or GuardrailConfig()
    state = RateLimitState(window_size=window_size, max_calls=max_calls)
    
    def rate_limit_guardrail(data: Any) -> ValidationResult:
        if not config.enabled:
            return ValidationResult(is_valid=True)
        
        current_time = time.time()
        
        # Remove old calls outside the window
        cutoff_time = current_time - state.window_size
        state.calls = [call_time for call_time in state.calls if call_time > cutoff_time]
        
        # Check if we're at the limit
        if len(state.calls) >= state.max_calls:
            message = (config.custom_message or 
                      f"Rate limit exceeded: {len(state.calls)}/{state.max_calls} calls in {state.window_size}s")
            return ValidationResult(is_valid=False, error_message=message)
        
        # Record this call
        state.calls.append(current_time)
        return ValidationResult(is_valid=True)
    
    return rate_limit_guardrail

def combine_guardrails(
    guardrails: List[Guardrail],
    require_all: bool = True,
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """
    Combine multiple guardrails into a single guardrail.
    
    Args:
        guardrails: List of guardrails to combine
        require_all: If True, all guardrails must pass; if False, at least one must pass
        config: Optional guardrail configuration
        
    Returns:
        Combined guardrail function
    """
    config = config or GuardrailConfig()
    
    async def combined_guardrail(data: Any) -> ValidationResult:
        if not config.enabled:
            return ValidationResult(is_valid=True)
        
        results = []
        
        for guardrail in guardrails:
            try:
                # Handle both sync and async guardrails
                if callable(guardrail):
                    result = guardrail(data)
                    if hasattr(result, '__await__'):
                        result = await result
                else:
                    continue
                
                results.append(result)
                
                if require_all and not result.is_valid:
                    # Fail fast if requiring all and one fails
                    return result
                elif not require_all and result.is_valid:
                    # Success fast if requiring any and one passes
                    return result
                    
            except Exception as e:
                error_result = ValidationResult(
                    is_valid=False,
                    error_message=f"Guardrail execution error: {str(e)}"
                )
                
                if require_all:
                    return error_result
                else:
                    results.append(error_result)
        
        if require_all:
            # All passed
            return ValidationResult(is_valid=True)
        else:
            # None passed
            failed_messages = [r.error_message for r in results if not r.is_valid]
            combined_message = "; ".join(failed_messages)
            return ValidationResult(
                is_valid=False,
                error_message=f"All guardrails failed: {combined_message}"
            )
    
    return combined_guardrail

# Common guardrail presets
def create_safe_text_guardrail(config: Optional[GuardrailConfig] = None) -> Guardrail:
    """Create a guardrail for safe text content."""
    return combine_guardrails([
        create_length_guardrail(max_length=10000, min_length=1),
        create_content_filter_guardrail([
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:.*base64',  # Base64 data URLs
        ], case_sensitive=False)
    ], config=config)

def create_api_input_guardrail(config: Optional[GuardrailConfig] = None) -> Guardrail:
    """Create a guardrail for API input validation."""
    return combine_guardrails([
        create_length_guardrail(max_length=50000),
        create_rate_limit_guardrail(max_calls=100, window_size=60),
    ], config=config)