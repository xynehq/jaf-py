"""
Comprehensive example of using the @function_tool decorator.

This example demonstrates various ways to use the decorator including timeout
configuration and shows how it automatically extracts type information and creates tools.
"""

import asyncio
from jaf import function_tool, ToolSource

# Example 1: Simple decorator usage (uses global default timeout)
@function_tool
async def fetch_weather(location: str, context) -> str:
    """Fetch the weather for a given location.
    
    Args:
        location: The location to fetch the weather for.
    """
    # Simulate API call delay
    await asyncio.sleep(1)
    # In real life, we'd fetch the weather from a weather API
    return f"The weather in {location} is sunny and 75°F"

# Example 2: Function with multiple parameters and defaults
@function_tool(timeout=10.0)  # 10 second timeout for tax calculations
def calculate_tax(amount: float, tax_rate: float = 0.1, context=None) -> str:
    """Calculate tax for a given amount.
    
    Args:
        amount: The base amount to calculate tax for.
        tax_rate: The tax rate to apply (default 0.1 for 10%).
    """
    tax = amount * tax_rate
    total = amount + tax
    return f"Amount: ${amount:.2f}, Tax: ${tax:.2f}, Total: ${total:.2f}"

# Example 3: Decorator with custom parameters including timeout
@function_tool(
    name="custom_greet",
    description="A custom greeting tool with enhanced features",
    metadata={"category": "social", "version": "1.0", "author": "JAF Team"},
    timeout=5.0  # Quick 5 second timeout for greetings
)
def greet_user(name: str, greeting: str = "Hello", context=None) -> str:
    """Greet a user with a customizable greeting."""
    return f"{greeting}, {name}! Welcome to JAF!"

# Example 4: Function with various parameter types and medium timeout
@function_tool(timeout=30.0)  # 30 second timeout for data processing
def process_data(
    text: str,
    count: int,
    enabled: bool = True,
    multiplier: float = 1.0,
    context=None
) -> str:
    """Process data with various parameter types.
    
    Args:
        text: The text to process.
        count: Number of times to repeat.
        enabled: Whether processing is enabled.
        multiplier: Factor to multiply count by.
    """
    if not enabled:
        return "Processing disabled"
    
    result_count = int(count * multiplier)
    return f"Processed '{text}' {result_count} times"

# Example 5: Tool with external source and API timeout
@function_tool(
    metadata={"integration": "weather_api"},
    source=ToolSource.EXTERNAL,
    timeout=45.0  # 45 second timeout for external API calls
)
async def get_forecast(city: str, days: int = 5, context=None) -> str:
    """Get weather forecast for multiple days.
    
    Args:
        city: The city to get forecast for.
        days: Number of days to forecast (default 5).
    """
    # Simulate longer API call
    await asyncio.sleep(2)
    return f"5-day forecast for {city}: Mostly sunny with temperatures 70-80°F"

# Example 6: Heavy computation with long timeout
@function_tool(timeout=300.0)  # 5 minute timeout for heavy operations
async def heavy_computation(
    dataset_size: int,
    algorithm: str = "standard",
    context=None
) -> str:
    """Perform heavy computational task.
    
    Args:
        dataset_size: Size of dataset to process.
        algorithm: Algorithm to use (standard, optimized, experimental).
    """
    # Simulate heavy computation delay
    computation_time = min(dataset_size / 1000, 10)  # Cap at 10 seconds for demo
    await asyncio.sleep(computation_time)
    
    return f"Processed {dataset_size} items using {algorithm} algorithm in {computation_time:.1f}s"

# Example 7: Interactive tool with no timeout
@function_tool(timeout=None)  # No timeout - wait indefinitely (use with caution)
async def interactive_prompt(prompt: str, context=None) -> str:
    """Interactive tool that waits for user input.
    
    Args:
        prompt: The prompt to show to the user.
    """
    # In a real implementation, this would wait for user input
    # For demo purposes, we'll just simulate a quick response
    await asyncio.sleep(0.5)
    return f"User responded to prompt: '{prompt}'"

# Example 8: Database operation with appropriate timeout
@function_tool(timeout=120.0)  # 2 minute timeout for database operations
async def database_query(
    query: str,
    table: str,
    limit: int = 100,
    context=None
) -> str:
    """Execute database query with appropriate timeout.
    
    Args:
        query: SQL query to execute.
        table: Table to query.
        limit: Maximum number of results.
    """
    # Simulate database query delay
    await asyncio.sleep(1.5)
    return f"Executed query '{query}' on table '{table}' with limit {limit}."

def demonstrate_tools():
    """Demonstrate the created tools and their properties."""
    tools = [
        fetch_weather, calculate_tax, greet_user, process_data, 
        get_forecast, heavy_computation, interactive_prompt, database_query
    ]
    
    print("=== JAF @function_tool Decorator Examples with Timeout Configuration ===\n")
    
    for tool in tools:
        print(f"Tool: {tool.schema.name}")
        print(f"Description: {tool.schema.description}")
        print(f"Timeout: {getattr(tool.schema, 'timeout', 'Uses default (30s)')}")
        print(f"Source: {tool.source}")
        print(f"Metadata: {tool.metadata}")
        print(f"Parameters: {tool.schema.parameters.__name__ if hasattr(tool.schema.parameters, '__name__') else 'Dynamic'}")
        print("-" * 50)

async def demonstrate_timeout_behaviors():
    """Demonstrate different timeout behaviors."""
    print("\n=== Timeout Behavior Demonstration ===\n")
    
    # Example of different timeout scenarios
    timeout_examples = [
        ("Quick operation", fetch_weather, {"location": "New York"}, "Should complete quickly"),
        ("Tax calculation", calculate_tax, {"amount": 100.0}, "10 second timeout"),
        ("Greeting", greet_user, {"name": "Alice"}, "5 second timeout"),
        ("Data processing", process_data, {"text": "test", "count": 5}, "30 second timeout"),
        ("Weather forecast", get_forecast, {"city": "London"}, "45 second timeout for API"),
        ("Heavy computation", heavy_computation, {"dataset_size": 1000}, "5 minute timeout"),
        ("Database query", database_query, {"query": "SELECT * FROM users", "table": "users"}, "2 minute timeout"),
    ]
    
    for name, tool, args, description in timeout_examples:
        print(f"🔧 {name}: {description}")
        
        # Create a mock context
        class MockContext:
            def __init__(self):
                pass
        
        context = MockContext()
        
        try:
            # This would normally be called by the JAF engine with proper argument handling
            print(f"   Tool timeout: {getattr(tool.schema, 'timeout', 'default (30s)')}")
            print(f"   Expected behavior: {description}")
            print()
        except Exception as e:
            print(f"   Error demonstrating tool: {e}")
            print()

if __name__ == "__main__":
    demonstrate_tools()
    
    # Run async demonstration
    print("\nRunning timeout behavior demonstration...")
    asyncio.run(demonstrate_timeout_behaviors())
    
    print("\n=== Summary ===")
    print("Timeout Configuration Options:")
    print("1. @function_tool                    - Uses global default (30s)")
    print("2. @function_tool(timeout=15.0)      - Specific timeout (15s)")
    print("3. @function_tool(timeout=None)      - No timeout (infinite)")
    print("4. RunConfig(default_tool_timeout=60.0) - Override defaults")
    print("\nTimeout Resolution Priority:")
    print("1. Tool-specific timeout (highest)")
    print("2. RunConfig default_tool_timeout")
    print("3. Global default 30 seconds (lowest)")
