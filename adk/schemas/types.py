"""
JSON Schema type definitions for comprehensive validation.

Provides TypedDict and dataclass definitions that support the full
JSON Schema Draft 7 specification for validation keywords.
"""

from typing import TypedDict, Optional, Any, Dict, List, Union
from dataclasses import dataclass


class JsonSchema(TypedDict, total=False):
    """
    Complete JSON Schema definition supporting all validation keywords.
    
    This extends the basic schema with comprehensive validation properties
    for strings, numbers, arrays, and objects.
    """
    # Core schema properties
    type: Optional[str]
    properties: Optional[Dict[str, 'JsonSchema']]
    required: Optional[List[str]]
    items: Optional['JsonSchema']
    enum: Optional[List[Any]]
    description: Optional[str]
    default: Optional[Any]
    
    # String validation properties
    minLength: Optional[int]
    maxLength: Optional[int]
    pattern: Optional[str]
    format: Optional[str]  # email, uri, url, date, date-time, uuid, etc.
    
    # Number validation properties
    minimum: Optional[Union[int, float]]
    maximum: Optional[Union[int, float]]
    exclusiveMinimum: Optional[bool]
    exclusiveMaximum: Optional[bool]
    multipleOf: Optional[Union[int, float]]
    
    # Array validation properties
    minItems: Optional[int]
    maxItems: Optional[int]
    uniqueItems: Optional[bool]
    
    # Object validation properties
    additionalProperties: Optional[Union[bool, 'JsonSchema']]
    minProperties: Optional[int]
    maxProperties: Optional[int]


@dataclass
class ValidationResult:
    """
    Result of a schema validation operation.
    
    Provides detailed information about validation success or failure,
    including specific error messages for debugging.
    """
    success: bool
    errors: List[str]
    data: Optional[Any] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if validation was successful."""
        return self.success and len(self.errors) == 0
    
    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        self.success = False


# Common format validation patterns
FORMAT_PATTERNS = {
    'email': r'^[^\s@]+@[^\s@]+\.[^\s@]+$',
    'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    'date': r'^\d{4}-\d{2}-\d{2}$',
    'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
    'ipv6': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
}

# Common stop words for keyword extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
    'should', 'may', 'might', 'must', 'can', 'what', 'how', 'when', 
    'where', 'why', 'who', 'this', 'that', 'these', 'those', 'i', 'you', 
    'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'like'  # Add 'like' to stop words
}