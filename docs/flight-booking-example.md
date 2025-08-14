# Flight Booking System: Production Agent Architecture

This comprehensive example demonstrates a production-grade flight booking system built on JAF's modern architecture. It showcases enterprise-level patterns including multi-agent coordination, functional composition, type safety, and scalable server integration.

## System Overview

### Core Architectural Demonstrations

- **Modern Object-Based Tool Creation**: Implementation using the advanced `create_function_tool` API with comprehensive type safety
- **Multi-Agent Coordination Patterns**: Specialized agent roles with intelligent handoff mechanisms and context preservation
- **Functional Composition Architecture**: Higher-order functions enabling tool enhancement, caching strategies, and retry logic
- **Enterprise Type Safety**: Comprehensive enum usage and typed configurations for runtime safety
- **Production Business Logic**: Real-world flight booking operations with error handling and validation
- **Scalable HTTP Server Integration**: FastAPI-based server with auto-documentation and monitoring endpoints

### Target Audience

This example is designed for:
- **Enterprise developers** building production agent systems
- **System architects** designing multi-agent workflows  
- **DevOps engineers** deploying agent-based services
- **Security engineers** implementing secure agent interactions

## Architecture

The flight booking system consists of four main components:

### 1. Core Tools (`index.py`)

Five main tools handle all flight operations:

- **`search_flights`**: Find available flights between airports
- **`check_seat_availability`**: Verify seat availability for specific flights
- **`book_flight`**: Reserve flights for passengers
- **`check_flight_status`**: Get current flight status information
- **`cancel_booking`**: Cancel existing reservations

### 2. Multi-Agent System (`multi_agent.py`)

Four specialized agents work together:

- **`Coordinator`**: Entry point that routes requests to specialists
- **`SearchSpecialist`**: Handles flight search and comparisons
- **`BookingSpecialist`**: Manages reservations and cancellations
- **`PricingSpecialist`**: Explains fares and pricing policies

### 3. Server Integration (`jaf_server.py`)

HTTP server that exposes agents via REST API:

- **Development mode**: Mock providers for local testing
- **Production mode**: Real LLM integration via LiteLLM
- **Health checks**: Monitoring and status endpoints
- **Auto-documentation**: OpenAPI/Swagger UI

## Key Features

### Object-Based Tool Creation

All tools use the new object-based API for better type safety and developer experience:

```python
from jaf import create_function_tool, ToolSource
from pydantic import BaseModel, Field

class FlightSearchArgs(BaseModel):
    origin: str = Field(description="Origin airport code (e.g., 'LAX')")
    destination: str = Field(description="Destination airport code (e.g., 'JFK')")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    passengers: int = Field(default=1, description="Number of passengers")

async def search_flights_execute(args: FlightSearchArgs, context) -> ToolResult:
    # Implementation here...
    return ToolResponse.success(results)

# Create tool with object-based configuration
search_flights_tool = create_function_tool({
    'name': 'search_flights',
    'description': 'Search for available flights between origin and destination',
    'execute': search_flights_execute,
    'parameters': FlightSearchArgs,
    'metadata': {'category': 'flight_search', 'priority': 'high'},
    'source': ToolSource.NATIVE
})
```

### Multi-Agent Coordination

Agents can hand off conversations to specialists:

```python
# Handoff tool for agent coordination
async def handoff_execute(args: HandoffArgs, context) -> ToolResult:
    return ToolResponse.success({
        "handoff_to": args.target_agent,
        "context": args.context,
        "reason": args.reason
    })

handoff_tool = create_function_tool({
    'name': 'handoff',
    'description': 'Hand off conversation to a specialized agent',
    'execute': handoff_execute,
    'parameters': HandoffArgs,
    'metadata': {'category': 'coordination'},
    'source': ToolSource.NATIVE
})

# Agent with handoff capabilities
search_specialist_agent = Agent(
    name="SearchSpecialist",
    instructions=search_specialist_instructions,
    tools=[search_flights_tool, handoff_tool],
    handoffs=["BookingSpecialist", "PricingSpecialist"]
)
```

### Functional Composition

Higher-order functions enhance tool behavior:

```python
def with_cache(tool_func):
    """Add caching to tool execution."""
    cache = {}
    async def cached_execute(args, context):
        cache_key = str(args)
        if cache_key in cache:
            return cache[cache_key]
        result = await tool_func(args, context)
        if result.status == "success":
            cache[cache_key] = result
        return result
    return cached_execute

def with_retry(tool_func, max_retries=3):
    """Add retry logic to tool execution."""
    async def retry_execute(args, context):
        for attempt in range(max_retries):
            try:
                result = await tool_func(args, context)
                if result.status == "success":
                    return result
            except Exception:
                if attempt == max_retries - 1:
                    raise
        return result
    return retry_execute

# Compose enhancements
enhanced_search = create_function_tool({
    'name': 'enhanced_search',
    'description': 'Search with caching and retry',
    'execute': with_cache(with_retry(search_flights_execute)),
    'parameters': FlightSearchArgs,
    'source': ToolSource.NATIVE
})
```

## Running the Example

### Prerequisites

```bash
# Install JAF with server dependencies
pip install jaf-py[server]

# For production LLM integration
pip install litellm
```

### Basic Flight Booking System

```bash
cd examples/flight-booking
python index.py
```

**Expected Output:**
```
Flight Booking System Demonstration
==================================================
Agent Execution Status: COMPLETED
Final Response: I can help you search for flights between any airports. 
Please provide your origin, destination, departure date, and number of passengers.

Tool Validation Results:
- Flight Search Tool: OPERATIONAL
- Seat Availability Tool: OPERATIONAL  
- Booking Management Tool: OPERATIONAL
- Status Information Tool: OPERATIONAL
- Cancellation Tool: OPERATIONAL

System Status: All components initialized successfully
Demo Execution: COMPLETED SUCCESSFULLY
```

### Multi-Agent Coordination

```bash
python multi_agent.py
```

**Features demonstrated:**
- Agent handoffs between specialists
- Tool composition with caching and retry
- Validator composition for data validation
- Functional programming patterns

### HTTP Server

#### Development Mode (No External Dependencies)

```bash
python jaf_server.py --dev
```

#### Production Mode (Requires LiteLLM)

```bash
# Start LiteLLM proxy
litellm --model gemini-2.0-flash --port 4000

# Start JAF server
python jaf_server.py
```

## API Endpoints

Once the server is running, you can interact via HTTP:

### Search for Flights

```bash
curl -X POST http://localhost:3000/agents/SearchSpecialist/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find flights from LAX to JFK departing January 15th for 2 passengers"
  }'
```

### Book a Flight

```bash
curl -X POST http://localhost:3000/agents/BookingSpecialist/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Book flight AA101 for John Doe with window seat preference"
  }'
```

### Get Pricing Information

```bash
curl -X POST http://localhost:3000/agents/PricingSpecialist/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the fare rules and baggage fees for flight AA101"
  }'
```

### Coordinate Through Main Agent

```bash
curl -X POST http://localhost:3000/agents/FlightCoordinator/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to book a flight from Los Angeles to New York tomorrow"
  }'
```

## Code Architecture

### Data Models

The example uses Pydantic models for type safety:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Flight:
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    airline: str
    seats_available: int
    aircraft_type: str

@dataclass
class Booking:
    booking_id: str
    flight: Flight
    passenger_name: str
    seat_number: str
    booking_status: str
    total_cost: float
```

### Error Handling

Comprehensive error handling with appropriate responses:

```python
async def book_flight_execute(args: BookFlightArgs, context) -> ToolResult:
    try:
        flight = find_flight(args.flight_number)
        if not flight:
            return ToolResponse.validation_error(f"Flight {args.flight_number} not found.")
        
        if flight.seats_available < 1:
            return ToolResponse.validation_error(f"No seats available on flight {args.flight_number}.")
        
        # Process booking...
        return ToolResponse.success(booking_result)
        
    except Exception as e:
        return ToolResponse.error(f"Error booking flight: {str(e)}")
```

### Agent Instructions

Dynamic instructions based on agent role:

```python
def search_specialist_instructions(state: RunState) -> str:
    return """You are a flight search specialist. Your job is to:

1. Help users find the best flights based on their criteria
2. Provide detailed flight information including prices, times, and availability
3. Compare different options and make recommendations
4. Hand off to the booking specialist when user is ready to book

When users want to proceed with booking, use the handoff tool to transfer them to the BookingSpecialist."""
```

## Type Safety Features

### Enums for Magic Strings

```python
from jaf import ContentRole, ToolSource, Model

# Instead of magic strings
role = ContentRole.USER          # vs 'user'
source = ToolSource.NATIVE       # vs 'native'
model = Model.GEMINI_2_0_FLASH   # vs 'gemini-2.0-flash'
```

### Typed Tool Configuration

```python
from jaf.core.types import FunctionToolConfig

# Type-safe configuration
config: FunctionToolConfig = {
    'name': 'search_flights',
    'description': 'Search for flights',
    'execute': search_execute,
    'parameters': FlightSearchArgs,
    'metadata': {'category': 'search'},
    'source': ToolSource.NATIVE
}
```

## Testing

The example includes comprehensive testing:

```python
# Test individual tools
search_result = await search_flights_execute(
    FlightSearchArgs(origin="LAX", destination="JFK", departure_date="2024-01-15"),
    {}
)
assert search_result.status == "success"

# Test agent coordination
result = await run(initial_state, config)
assert result.outcome.status == "completed"

# Test functional composition
enhanced_search = with_cache(with_retry(search_flights_execute))
result = await enhanced_search(args, context)
```

## Performance Considerations

### Caching Strategy

```python
# L1: In-memory cache for frequent queries
# L2: Redis cache for session persistence
# L3: Database for permanent storage

layered_cache = create_layered_cache(
    in_memory_cache,
    redis_cache,
    database_store
)
```

### Connection Pooling

```python
# HTTP client with connection pooling
async_client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

### Rate Limiting

```python
# Per-user rate limiting
rate_limiter = with_rate_limit(
    search_flights_execute,
    max_calls=100,
    time_window=3600,
    key_func=lambda args, ctx: ctx.get('user_id', 'anonymous')
)
```

## Security Features

### Input Validation

All inputs are validated using Pydantic models:

```python
class BookFlightArgs(BaseModel):
    flight_number: str = Field(regex=r'^[A-Z]{2}\d{3,4}$', description="Flight number")
    passenger_name: str = Field(min_length=2, max_length=100, description="Passenger name")
    seat_preference: Optional[str] = Field(regex=r'^(window|aisle|middle)$', description="Seat preference")
```

### Safe Expression Evaluation

Calculator tool uses safe evaluation:

```python
async def calculator_execute(args: CalculateArgs, context) -> ToolResult:
    # Sanitize input - only allow safe characters
    safe_chars = '0123456789+-*/(). '
    sanitized = ''.join(c for c in args.expression if c in safe_chars)
    
    if sanitized != args.expression:
        return ToolResponse.validation_error("Invalid characters in expression")
    
    # Use safe evaluation
    try:
        result = eval(sanitized, {"__builtins__": {}}, {})
        return ToolResponse.success(result)
    except Exception as e:
        return ToolResponse.error(f"Calculation error: {str(e)}")
```

### Permission Checks

Context-based permissions:

```python
async def book_flight_execute(args: BookFlightArgs, context) -> ToolResult:
    user_permissions = context.get('permissions', [])
    if 'booking' not in user_permissions:
        return ToolResponse.error("Insufficient permissions to book flights")
    
    # Proceed with booking...
```

## Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 3000

CMD ["python", "jaf_server.py"]
```

### Environment Configuration

```bash
# Production environment variables
export JAF_HOST=0.0.0.0
export JAF_PORT=3000
export LITELLM_BASE_URL=https://api.example.com
export LITELLM_API_KEY=your-api-key
export REDIS_URL=redis://localhost:6379
```

## Next Steps

1. **Extend functionality**: Add more flight operations (seat selection, meal preferences)
2. **Real integrations**: Connect to actual airline APIs
3. **Advanced patterns**: Implement circuit breakers, bulkheads
4. **Monitoring**: Add metrics and observability
5. **Testing**: Comprehensive test suite with mocks
6. **Documentation**: API documentation and user guides

This example demonstrates the power and flexibility of JAF's new API while providing a realistic foundation for building production agent systems.