"""
Comprehensive example of using the @function_tool decorator.

This example demonstrates various ways to use the decorator and shows
how it automatically extracts type information and creates tools.
"""

from jaf import function_tool, ToolSource

# Example 1: Simple decorator usage
@function_tool
async def fetch_weather(location: str, context) -> str:
    """Fetch the weather for a given location.
    
    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return f"The weather in {location} is sunny and 75°F"

# Example 2: Function with multiple parameters and defaults
@function_tool
def calculate_tax(amount: float, tax_rate: float = 0.1, context=None) -> str:
    """Calculate tax for a given amount.
    
    Args:
        amount: The base amount to calculate tax for.
        tax_rate: The tax rate to apply (default 0.1 for 10%).
    """
    tax = amount * tax_rate
    total = amount + tax
    return f"Amount: ${amount:.2f}, Tax: ${tax:.2f}, Total: ${total:.2f}"

# Example 3: Decorator with custom parameters
@function_tool(
    name="custom_greet",
    description="A custom greeting tool with enhanced features",
    metadata={"category": "social", "version": "1.0", "author": "JAF Team"}
)
def greet_user(name: str, greeting: str = "Hello", context=None) -> str:
    """Greet a user with a customizable greeting."""
    return f"{greeting}, {name}! Welcome to JAF!"

# Example 4: Function with various parameter types
@function_tool
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

# Example 5: Tool with external source
@function_tool(
    metadata={"integration": "weather_api"},
    source=ToolSource.EXTERNAL
)
async def get_forecast(city: str, days: int = 5, context=None) -> str:
    """Get weather forecast for multiple days.
    
    Args:
        city: The city to get forecast for.
        days: Number of days to forecast (default 5).
    """
    return f"5-day forecast for {city}: Mostly sunny with temperatures 70-80°F"

def demonstrate_tools():
    """Demonstrate the created tools and their properties."""
    tools = [fetch_weather, calculate_tax, greet_user, process_data, get_forecast]
    
    print("=== JAF @function_tool Decorator Examples ===\n")
    
    for tool in tools:
        print(f"Tool: {tool.schema.name}")
        print(f"Description: {tool.schema.description}")
        print(f"Source: {tool.source}")
        print(f"Metadata: {tool.metadata}")
        print(f"Parameters: {tool.schema.parameters}")
        print("-" * 50)

if __name__ == "__main__":
    demonstrate_tools()
