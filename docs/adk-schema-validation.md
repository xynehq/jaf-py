# ADK Schema Validation System

The Agent Development Kit (ADK) provides enterprise-grade JSON Schema validation for tool parameters, API inputs, and data validation. This system implements the full JSON Schema Draft 7 specification with advanced validation features for production applications.

## Overview

The ADK schema validation system offers:

- **Complete JSON Schema Support**: Full Draft 7 specification compliance
- **Advanced Type Validation**: Strings, numbers, arrays, objects, and more
- **Format Validation**: Email, URI, UUID, dates, IP addresses
- **Business Rule Validation**: Custom constraints and complex validations
- **Performance Optimized**: Efficient validation with detailed error reporting
- **Production Ready**: Enterprise security and reliability features

## Core Components

### JsonSchema Type

The `JsonSchema` type provides comprehensive schema definition capabilities:

```python
from adk.schemas import JsonSchema, validate_schema

# Complete schema definition
user_schema: JsonSchema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 2,
            "maxLength": 50,
            "pattern": r"^[A-Za-z\s]+$"
        },
        "email": {
            "type": "string",
            "format": "email"
        },
        "age": {
            "type": "integer",
            "minimum": 18,
            "maximum": 120
        },
        "preferences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True
        }
    },
    "required": ["name", "email"],
    "additionalProperties": False
}
```

### ValidationResult

The `ValidationResult` provides detailed validation feedback:

```python
from adk.schemas import ValidationResult

# Validate data
result = validate_schema(user_data, user_schema)

if result.is_valid:
    print(f"✅ Validation successful: {result.data}")
else:
    print("❌ Validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

## Validation Types

### String Validation

Comprehensive string validation with multiple constraint types:

```python
from adk.schemas import validate_schema

# Advanced string schema
password_schema = {
    "type": "string",
    "minLength": 8,
    "maxLength": 128,
    "pattern": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$",
    "description": "Password must contain uppercase, lowercase, digit, and special character"
}

# Format validation
email_schema = {
    "type": "string",
    "format": "email",
    "maxLength": 254
}

url_schema = {
    "type": "string",
    "format": "uri",
    "pattern": r"^https://"  # Require HTTPS
}

# Validation examples
password_result = validate_schema("SecurePass123!", password_schema)
email_result = validate_schema("user@example.com", email_schema)
url_result = validate_schema("https://api.example.com", url_schema)

print(f"Password valid: {password_result.is_valid}")
print(f"Email valid: {email_result.is_valid}")
print(f"URL valid: {url_result.is_valid}")
```

### Number Validation

Precise numeric validation with range and precision constraints:

```python
# Integer validation
age_schema = {
    "type": "integer",
    "minimum": 0,
    "maximum": 150,
    "description": "Age in years"
}

# Float validation with precision
price_schema = {
    "type": "number",
    "minimum": 0,
    "exclusiveMinimum": True,  # Must be > 0
    "multipleOf": 0.01,        # Currency precision
    "maximum": 1000000
}

# Percentage validation
percentage_schema = {
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 0.1
}

# Examples
age_result = validate_schema(25, age_schema)
price_result = validate_schema(29.99, price_schema)
percentage_result = validate_schema(85.5, percentage_schema)
```

### Array Validation

Advanced array validation with item constraints:

```python
# Homogeneous array
tags_schema = {
    "type": "array",
    "items": {
        "type": "string",
        "minLength": 1,
        "maxLength": 20
    },
    "minItems": 1,
    "maxItems": 10,
    "uniqueItems": True
}

# Complex nested array
coordinates_schema = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 3  # 2D or 3D coordinates
    },
    "minItems": 1
}

# Array of objects
users_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "name": {"type": "string", "minLength": 1}
        },
        "required": ["id", "name"]
    }
}

# Examples
tags_result = validate_schema(["python", "json", "validation"], tags_schema)
coords_result = validate_schema([[0, 0], [1, 1], [2, 2]], coordinates_schema)
users_result = validate_schema([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
], users_schema)
```

### Object Validation

Comprehensive object validation with property constraints:

```python
# Strict object schema
api_request_schema = {
    "type": "object",
    "properties": {
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE"]
        },
        "url": {
            "type": "string",
            "format": "uri"
        },
        "headers": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "body": {
            "type": "string"
        }
    },
    "required": ["method", "url"],
    "additionalProperties": False,
    "minProperties": 2,
    "maxProperties": 10
}

# Flexible configuration object
config_schema = {
    "type": "object",
    "properties": {
        "timeout": {"type": "integer", "minimum": 1, "default": 30},
        "retries": {"type": "integer", "minimum": 0, "maximum": 5, "default": 3}
    },
    "additionalProperties": {
        "type": "string"  # Allow additional string properties
    }
}

# Examples
request_data = {
    "method": "POST",
    "url": "https://api.example.com/users",
    "headers": {"Content-Type": "application/json"},
    "body": '{"name": "Alice"}'
}

request_result = validate_schema(request_data, api_request_schema)
```

## Format Validation

Built-in format validators for common data types:

```python
# Email validation
email_result = validate_schema("user@example.com", {
    "type": "string",
    "format": "email"
})

# UUID validation
uuid_result = validate_schema("550e8400-e29b-41d4-a716-446655440000", {
    "type": "string",
    "format": "uuid"
})

# Date validation
date_result = validate_schema("2024-03-15", {
    "type": "string",
    "format": "date"
})

# DateTime validation
datetime_result = validate_schema("2024-03-15T14:30:00Z", {
    "type": "string",
    "format": "date-time"
})

# URL validation
url_result = validate_schema("https://www.example.com/path?query=value", {
    "type": "string",
    "format": "uri"
})

# IP address validation
ipv4_result = validate_schema("192.168.1.1", {
    "type": "string",
    "format": "ipv4"
})

ipv6_result = validate_schema("2001:db8::1", {
    "type": "string",
    "format": "ipv6"
})
```

## Advanced Validation Patterns

### Conditional Validation

Implement business rules with conditional logic:

```python
def validate_user_with_business_rules(user_data):
    """Custom validation with business logic"""
    
    # Basic schema validation
    result = validate_schema(user_data, user_schema)
    if not result.is_valid:
        return result
    
    # Business rule: Premium users must have valid payment method
    if user_data.get("plan") == "premium":
        if not user_data.get("payment_method"):
            result.add_error("Premium users must provide payment method")
    
    # Business rule: Admin users must have strong passwords
    if user_data.get("role") == "admin":
        password = user_data.get("password", "")
        if len(password) < 12:
            result.add_error("Admin passwords must be at least 12 characters")
    
    return result

# Usage
user_data = {
    "name": "Alice Admin",
    "email": "alice@example.com",
    "role": "admin",
    "password": "short"
}

result = validate_user_with_business_rules(user_data)
```

### Multi-Schema Validation

Validate against multiple schemas:

```python
def validate_api_endpoint(data, endpoint_type):
    """Validate API endpoint data against appropriate schema"""
    
    schemas = {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        },
        "product": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number", "minimum": 0}
            },
            "required": ["title", "price"]
        }
    }
    
    if endpoint_type not in schemas:
        return ValidationResult(
            success=False,
            errors=[f"Unknown endpoint type: {endpoint_type}"]
        )
    
    return validate_schema(data, schemas[endpoint_type])

# Usage
user_result = validate_api_endpoint(
    {"name": "Alice", "email": "alice@example.com"},
    "user"
)

product_result = validate_api_endpoint(
    {"title": "Widget", "price": 19.99},
    "product"
)
```

### Recursive Schema Validation

Handle deeply nested data structures:

```python
# Tree structure schema
tree_schema = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "children": {
            "type": "array",
            "items": {"$ref": "#"}  # Self-reference
        }
    },
    "required": ["value"]
}

# Note: JSON Schema $ref requires special handling
# For now, implement custom recursive validation

def validate_tree(data, depth=0, max_depth=10):
    """Validate tree structure with depth limit"""
    
    if depth > max_depth:
        return ValidationResult(
            success=False,
            errors=["Tree depth exceeds maximum allowed"]
        )
    
    # Validate current node
    node_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "minLength": 1},
            "children": {"type": "array"}
        },
        "required": ["value"]
    }
    
    result = validate_schema(data, node_schema)
    if not result.is_valid:
        return result
    
    # Recursively validate children
    for i, child in enumerate(data.get("children", [])):
        child_result = validate_tree(child, depth + 1, max_depth)
        if not child_result.is_valid:
            result.errors.extend([
                f"Child {i}: {error}" for error in child_result.errors
            ])
            result.success = False
    
    return result

# Usage
tree_data = {
    "value": "root",
    "children": [
        {
            "value": "child1",
            "children": [
                {"value": "grandchild1"}
            ]
        },
        {"value": "child2"}
    ]
}

tree_result = validate_tree(tree_data)
```

## Integration Patterns

### Tool Parameter Validation

Integrate with JAF tool creation:

```python
from jaf import create_function_tool
from adk.schemas import validate_schema
from pydantic import BaseModel

class CalculateArgs(BaseModel):
    expression: str
    precision: int = 2

# Define validation schema
calculate_schema = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "pattern": r"^[0-9+\-*/().\\s]+$"  # Safe math expressions only
        },
        "precision": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "default": 2
        }
    },
    "required": ["expression"]
}

async def safe_calculate(args: CalculateArgs, context) -> str:
    """Calculator with schema validation"""
    
    # Validate with schema
    args_dict = args.dict()
    result = validate_schema(args_dict, calculate_schema)
    
    if not result.is_valid:
        return f"Validation error: {'; '.join(result.errors)}"
    
    # Proceed with calculation
    try:
        value = eval(args.expression)
        return f"{args.expression} = {round(value, args.precision)}"
    except Exception as e:
        return f"Calculation error: {e}"

# Create tool with validation
calculator_tool = create_function_tool({
    "name": "safe_calculate",
    "description": "Perform safe mathematical calculations",
    "execute": safe_calculate,
    "parameters": CalculateArgs
})
```

### API Request Validation

Validate API requests and responses:

```python
import httpx
from adk.schemas import validate_schema

class APIClient:
    def __init__(self):
        self.request_schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                },
                "json": {"type": "object"},
                "timeout": {"type": "number", "minimum": 0.1, "maximum": 300}
            },
            "required": ["url", "method"]
        }
    
    async def make_request(self, request_config):
        """Make HTTP request with validation"""
        
        # Validate request configuration
        result = validate_schema(request_config, self.request_schema)
        if not result.is_valid:
            raise ValueError(f"Invalid request config: {result.errors}")
        
        # Make validated request
        async with httpx.AsyncClient() as client:
            response = await client.request(**request_config)
            return response

# Usage
client = APIClient()

request_config = {
    "url": "https://api.example.com/users",
    "method": "POST",
    "headers": {"Content-Type": "application/json"},
    "json": {"name": "Alice", "email": "alice@example.com"},
    "timeout": 30.0
}

try:
    response = await client.make_request(request_config)
    print(f"Request successful: {response.status_code}")
except ValueError as e:
    print(f"Validation error: {e}")
```

### Configuration Validation

Validate application configuration:

```python
from adk.schemas import validate_schema
import os
import json

# Application configuration schema
app_config_schema = {
    "type": "object",
    "properties": {
        "database": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "format": "ipv4"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "name": {"type": "string", "minLength": 1},
                "ssl": {"type": "boolean"}
            },
            "required": ["host", "port", "name"]
        },
        "api": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "cors_origins": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"}
                }
            },
            "required": ["host", "port"]
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "format": {"type": "string"}
            }
        }
    },
    "required": ["database", "api"]
}

def load_and_validate_config(config_path: str):
    """Load and validate application configuration"""
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load config: {e}")
    
    # Validate configuration
    result = validate_schema(config_data, app_config_schema)
    if not result.is_valid:
        raise ValueError(f"Invalid configuration: {result.errors}")
    
    return result.data

# Usage
try:
    config = load_and_validate_config("app_config.json")
    print("✅ Configuration loaded and validated successfully")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

## Error Handling and Debugging

### Detailed Error Analysis

```python
from adk.schemas import validate_schema

def analyze_validation_errors(data, schema):
    """Provide detailed error analysis"""
    
    result = validate_schema(data, schema)
    
    if result.is_valid:
        print("✅ Validation successful")
        return result
    
    print("❌ Validation failed:")
    print(f"Data type: {type(data).__name__}")
    print(f"Schema type: {schema.get('type', 'unspecified')}")
    print("\nErrors:")
    
    for i, error in enumerate(result.errors, 1):
        print(f"  {i}. {error}")
    
    # Provide suggestions
    print("\nSuggestions:")
    for error in result.errors:
        if "minimum" in error.lower():
            print("  - Increase the value to meet minimum requirements")
        elif "maximum" in error.lower():
            print("  - Decrease the value to meet maximum requirements")
        elif "required" in error.lower():
            print("  - Add the missing required properties")
        elif "format" in error.lower():
            print("  - Check the format specification and examples")
    
    return result

# Usage
invalid_data = {
    "name": "A",  # Too short
    "email": "invalid-email",  # Invalid format
    "age": -5  # Below minimum
}

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "email": {"type": "string", "format": "email"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name", "email"]
}

analyze_validation_errors(invalid_data, schema)
```

### Custom Validation Functions

```python
from adk.schemas import ValidationResult

def create_custom_validator(validation_func, error_message):
    """Create custom validation function"""
    
    def validator(data, schema):
        base_result = validate_schema(data, schema)
        
        if base_result.is_valid:
            try:
                if not validation_func(data):
                    base_result.add_error(error_message)
            except Exception as e:
                base_result.add_error(f"Custom validation error: {e}")
        
        return base_result
    
    return validator

# Example: Credit card number validation
def is_valid_credit_card(number_str):
    """Luhn algorithm for credit card validation"""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Luhn algorithm
    checksum = 0
    is_even = False
    for digit in reversed(digits):
        if is_even:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
        is_even = not is_even
    
    return checksum % 10 == 0

# Create custom validator
credit_card_validator = create_custom_validator(
    is_valid_credit_card,
    "Invalid credit card number (fails Luhn check)"
)

# Usage
card_schema = {
    "type": "string",
    "pattern": r"^\d{13,19}$"
}

result = credit_card_validator("4532015112830366", card_schema)
```

## Performance and Best Practices

### Performance Optimization

```python
import time
from functools import lru_cache
from adk.schemas import validate_schema

# Cache compiled schemas for better performance
@lru_cache(maxsize=128)
def get_compiled_schema(schema_key):
    """Get cached schema definition"""
    schemas = {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        },
        "product": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number", "minimum": 0}
            },
            "required": ["title", "price"]
        }
    }
    return schemas.get(schema_key)

def benchmark_validation(data, schema, iterations=1000):
    """Benchmark validation performance"""
    
    start_time = time.time()
    
    for _ in range(iterations):
        result = validate_schema(data, schema)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Validated {iterations} times in {duration:.4f}s")
    print(f"Average: {duration/iterations*1000:.2f}ms per validation")
    
    return result

# Usage
user_data = {"name": "Alice", "email": "alice@example.com"}
user_schema = get_compiled_schema("user")

benchmark_validation(user_data, user_schema)
```

### Best Practices

1. **Schema Design**:
   ```python
   # ✅ Good: Clear, specific constraints
   good_schema = {
       "type": "object",
       "properties": {
           "email": {
               "type": "string",
               "format": "email",
               "maxLength": 254
           },
           "age": {
               "type": "integer",
               "minimum": 0,
               "maximum": 150
           }
       },
       "required": ["email"],
       "additionalProperties": False
   }
   
   # ❌ Avoid: Overly permissive
   bad_schema = {
       "type": "object",
       "additionalProperties": True  # Too permissive
   }
   ```

2. **Error Handling**:
   ```python
   def safe_validate(data, schema):
       """Safely validate with error handling"""
       try:
           result = validate_schema(data, schema)
           return result
       except Exception as e:
           return ValidationResult(
               success=False,
               errors=[f"Validation exception: {e}"]
           )
   ```

3. **Schema Reuse**:
   ```python
   # Define reusable schema components
   common_schemas = {
       "email": {"type": "string", "format": "email"},
       "positive_integer": {"type": "integer", "minimum": 1},
       "uuid": {"type": "string", "format": "uuid"}
   }
   
   def create_user_schema():
       return {
           "type": "object",
           "properties": {
               "id": common_schemas["uuid"],
               "email": common_schemas["email"],
               "age": common_schemas["positive_integer"]
           }
       }
   ```

## Testing Schema Validation

```python
import pytest
from adk.schemas import validate_schema

def test_string_validation():
    """Test string validation edge cases"""
    
    schema = {
        "type": "string",
        "minLength": 3,
        "maxLength": 10,
        "pattern": r"^[A-Za-z]+$"
    }
    
    # Valid cases
    assert validate_schema("abc", schema).is_valid
    assert validate_schema("Hello", schema).is_valid
    assert validate_schema("abcdefghij", schema).is_valid
    
    # Invalid cases
    assert not validate_schema("ab", schema).is_valid  # Too short
    assert not validate_schema("abcdefghijk", schema).is_valid  # Too long
    assert not validate_schema("abc123", schema).is_valid  # Invalid pattern

def test_number_validation():
    """Test number validation edge cases"""
    
    schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 0.5
    }
    
    # Valid cases
    assert validate_schema(0, schema).is_valid
    assert validate_schema(50.5, schema).is_valid
    assert validate_schema(100, schema).is_valid
    
    # Invalid cases
    assert not validate_schema(-0.1, schema).is_valid  # Below minimum
    assert not validate_schema(100.1, schema).is_valid  # Above maximum
    assert not validate_schema(50.3, schema).is_valid  # Not multiple of 0.5

@pytest.mark.parametrize("email,expected", [
    ("user@example.com", True),
    ("invalid-email", False),
    ("user@", False),
    ("@example.com", False),
])
def test_email_format(email, expected):
    """Test email format validation"""
    
    schema = {"type": "string", "format": "email"}
    result = validate_schema(email, schema)
    assert result.is_valid == expected
```

The ADK schema validation system provides comprehensive, production-ready validation capabilities that integrate seamlessly with JAF agents and tools. Use these patterns to ensure data integrity and provide clear error feedback in your applications.