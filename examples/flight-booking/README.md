# Flight Booking Example

This comprehensive example demonstrates the new JAF Python API with object-based tool configuration, functional composition patterns, and multi-agent coordination for a realistic flight booking system.

## 🎯 What This Example Demonstrates

### 1. **New Object-Based Tool API**
All tools are created using the new `create_function_tool` API with object configuration:

```python
search_flights_tool = create_function_tool({
    'name': 'search_flights',
    'description': 'Search for available flights between origin and destination',
    'execute': search_flights_execute,
    'parameters': FlightSearchArgs,
    'metadata': {'category': 'flight_search', 'priority': 'high'},
    'source': ToolSource.NATIVE
})
```

### 2. **Functional Composition Patterns**
Higher-order functions for composing tool behavior:

- **Logging**: `with_logging(tool)` - Adds execution logging
- **Retry Logic**: `with_retry(tool, max_retries=3)` - Adds retry on failure
- **Caching**: `with_cache(tool)` - Adds result caching
- **Validator Composition**: `compose_validators()` - Combines validation functions

### 3. **Multi-Agent Coordination**
Four specialized agents working together:

- **Coordinator**: Routes requests to appropriate specialists
- **SearchSpecialist**: Handles flight search and comparisons
- **BookingSpecialist**: Manages reservations and cancellations
- **PricingSpecialist**: Explains fares and pricing policies

### 4. **Type Safety & Enums**
Uses the new enum system for better type safety:

```python
from jaf import ContentRole, ToolSource, Model

role=ContentRole.USER          # Instead of 'user'
source=ToolSource.NATIVE       # Instead of 'native'
model=Model.GEMINI_2_0_FLASH   # Instead of 'gemini-2.0-flash'
```

## 📁 Files Overview

### `index.py` - Core Flight Booking System
- **Flight booking tools**: Search, availability, booking, status, cancellation
- **Data models**: Flight, Booking, and argument classes
- **Single comprehensive agent**: Complete flight booking assistant
- **Mock data**: Realistic flight and booking information

### `multi_agent.py` - Multi-Agent Coordination
- **Specialized agents**: Each agent has specific expertise
- **Handoff coordination**: Agents can transfer conversations
- **Functional composition**: Higher-order functions for tool enhancement
- **Validator composition**: Composable validation functions

### `jaf_server.py` - HTTP Server Integration
- **JAF server setup**: Exposes agents via HTTP endpoints
- **Development mode**: Mock providers for local testing
- **Production config**: Real LLM integration via LiteLLM
- **API documentation**: Auto-generated OpenAPI docs

## 🚀 Getting Started

### Prerequisites

```bash
# Install JAF with server dependencies
pip install -e ".[server]"

# For production (LiteLLM integration)
pip install litellm fastapi uvicorn
```

### Running the Examples

#### 1. Test Core Functionality
```bash
cd examples/flight-booking
python index.py
```

#### 2. Test Multi-Agent Coordination
```bash
python multi_agent.py
```

#### 3. Run HTTP Server (Development Mode)
```bash
python jaf_server.py --dev
```

#### 4. Run HTTP Server (Production Mode)
```bash
# First, start LiteLLM proxy
litellm --model gemini-2.0-flash --port 4000

# Then start JAF server
python jaf_server.py
```

## 🌐 HTTP API Usage

Once the server is running, you can interact with it via HTTP:

### Chat with Flight Coordinator
```bash
curl -X POST http://localhost:3000/agents/FlightCoordinator/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to book a flight from LAX to JFK tomorrow"}'
```

### Search Flights Directly
```bash
curl -X POST http://localhost:3000/agents/SearchSpecialist/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find flights from LAX to JFK departing January 15th"}'
```

### Complete a Booking
```bash
curl -X POST http://localhost:3000/agents/BookingSpecialist/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Book flight AA101 for John Doe with window seat preference"}'
```

### Get API Documentation
Visit: http://localhost:3000/docs

## 🔧 Functional Composition Examples

### Tool Enhancement
```python
# Compose multiple enhancements
enhanced_tool = with_logging(
    with_cache(
        with_retry(base_tool, max_retries=3)
    )
)

# Use the enhanced tool
result = await enhanced_tool.execute(args, context)
```

### Validator Composition
```python
# Combine validators
combined_validator = compose_validators(
    validate_airport_code,
    validate_passenger_count,
    validate_date_format
)

# Use composed validator
validation_result = combined_validator(flight_data)
```

## 🏗️ Architecture Benefits

### Type Safety
- **Enum usage**: Eliminates magic strings
- **Pydantic models**: Runtime type validation
- **TypedDict configs**: IDE support for tool configuration

### Extensibility
- **Object-based API**: Easy to add new configuration options
- **Functional composition**: Mix and match behaviors
- **Plugin architecture**: Tools can be sourced from different providers

### Developer Experience
- **Self-documenting**: Tool configuration is explicit
- **IDE support**: Full autocomplete and type checking
- **Error prevention**: Compile-time error detection

### Maintainability
- **Separation of concerns**: Each agent has specific responsibilities
- **Immutable state**: Functional programming principles
- **Testable**: Easy to mock and test individual components

## 🧪 Testing

The example includes comprehensive testing capabilities:

```python
# Test individual tools
search_result = await search_flights_execute(
    FlightSearchArgs(origin="LAX", destination="JFK", departure_date="2024-01-15"),
    {}
)

# Test agent coordination
result = await run(initial_state, config)

# Test functional composition
enhanced_search = with_logging(with_cache(search_tool))
```

## 📚 Learning Objectives

After studying this example, you'll understand:

1. **Modern JAF API**: Object-based tool configuration vs. positional arguments
2. **Functional Programming**: Higher-order functions and composition patterns
3. **Multi-Agent Systems**: Coordination and handoff patterns
4. **Type Safety**: Using enums and typed configurations
5. **Server Integration**: Exposing agents via HTTP APIs
6. **Error Handling**: Retry patterns and graceful degradation
7. **Caching Strategies**: Performance optimization techniques
8. **Validation Composition**: Building complex validation from simple functions

## 🔄 Migration Guide

### From Old API
```python
# Old positional API
tool = create_function_tool(
    'search_flights',
    'Search for flights',
    search_execute,
    FlightSearchArgs
)
```

### To New Object API
```python
# New object-based API
tool = create_function_tool({
    'name': 'search_flights',
    'description': 'Search for flights',
    'execute': search_execute,
    'parameters': FlightSearchArgs,
    'metadata': {'category': 'search'},
    'source': ToolSource.NATIVE
})
```

## 🎉 Next Steps

1. **Extend the example**: Add more flight booking features
2. **Custom composition**: Create your own higher-order functions
3. **Real integrations**: Connect to actual flight APIs
4. **Advanced patterns**: Implement circuit breakers, rate limiting
5. **Testing**: Add comprehensive test suites
6. **Deployment**: Deploy to production with monitoring

This example showcases the power and flexibility of the new JAF Python API while demonstrating real-world patterns for building sophisticated AI agents.