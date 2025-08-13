# Getting Started with JAF

Welcome to **JAF (Juspay Agent Framework)**! This guide will walk you through everything you need to build your first AI agent with JAF's functional architecture.

!!! tip "What You'll Learn"
    By the end of this guide, you'll have:
    
    - ✅ JAF installed and running
    - ✅ Your first functional agent
    - ✅ Understanding of core concepts
    - ✅ A working example you can extend

## Prerequisites

!!! info "System Requirements"
    - **Python 3.9+** (3.11+ recommended for best performance)
    - **LiteLLM proxy** (for LLM integration) or direct LLM API access
    - Basic knowledge of Python async/await and type hints

## Installation

=== "PyPI (Recommended)"

    ```bash
    # Basic installation
    pip install jaf-python

    # With all optional dependencies (recommended for development)
    pip install "jaf-python[all]"
    ```

=== "Feature-Specific"

    ```bash
    # Install specific feature sets
    pip install "jaf-python[server]"        # FastAPI server support
    pip install "jaf-python[memory]"        # Redis/PostgreSQL memory providers
    pip install "jaf-python[visualization]" # Graphviz visualization
    pip install "jaf-python[dev]"           # Development tools
    ```

=== "Development"

    ```bash
    git clone https://github.com/juspay/jaf-python
    cd jaf-python
    pip install -e ".[dev]"

    # Verify installation
    python -c "import jaf; print('JAF imported successfully!')"
    ```

=== "Docker"

    ```bash
    # Pull the official image
    docker pull juspay/jaf-python:latest
    
    # Run with port mapping
    docker run -p 8000:8000 juspay/jaf-python
    ```

## Setting Up LiteLLM (Model Provider)

JAF works with 100+ LLM models through LiteLLM. You'll need a running LiteLLM proxy:

### Quick LiteLLM Setup

```bash
# Install LiteLLM
pip install litellm[proxy]

# Create config file
cat > litellm_config.yaml << EOF
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: your-openai-api-key
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: your-anthropic-api-key
EOF

# Start LiteLLM proxy
litellm --config litellm_config.yaml --port 4000
```

### Environment Variables

Create a `.env` file in your project:

```bash
# LiteLLM Configuration
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your-api-key  # Optional, depending on your setup

# Memory Provider (Optional)
MEMORY_PROVIDER=in_memory  # Options: in_memory, redis, postgres
REDIS_URL=redis://localhost:6379  # If using Redis
DATABASE_URL=postgresql://user:pass@localhost/jaf  # If using PostgreSQL

# Tracing (Optional)
JAF_TRACE_ENABLED=true
JAF_TRACE_LEVEL=INFO
```

## Your First Agent

Let's build a simple calculator agent to understand JAF's core concepts:

### Step 1: Define Your Context Type

```python
# calculator_agent.py
from dataclasses import dataclass
from typing import List

@dataclass
class CalculatorContext:
    """Context for our calculator agent."""
    user_id: str
    allowed_operations: List[str]
    max_result: float = 1000000.0
```

### Step 2: Create a Tool

```python
from pydantic import BaseModel, Field
from jaf.core.tool_results import ToolSuccess, ToolError

class CalculateArgs(BaseModel):
    """Arguments for the calculate tool."""
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')")

class CalculatorTool:
    """A safe calculator tool with validation."""
    
    @property
    def schema(self):
        """Tool schema for LLM integration."""
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Safely evaluate mathematical expressions',
            'parameters': CalculateArgs
        })()
    
    async def execute(self, args: CalculateArgs, context: CalculatorContext) -> str:
        """Execute the calculation with safety checks."""
        try:
            # Simple whitelist validation
            allowed_chars = set('0123456789+-*/.() ')
            if not all(char in allowed_chars for char in args.expression):
                return ToolError("Expression contains invalid characters").format()
            
            # Evaluate safely (in production, use a proper math parser)
            result = eval(args.expression)
            
            # Check context limits
            if abs(result) > context.max_result:
                return ToolError(f"Result {result} exceeds maximum allowed value").format()
            
            return ToolSuccess(f"Result: {args.expression} = {result}").format()
            
        except Exception as e:
            return ToolError(f"Calculation error: {str(e)}").format()
```

### Step 3: Define Your Agent

```python
from jaf import Agent

def create_calculator_agent() -> Agent[CalculatorContext]:
    """Create a calculator agent with mathematical capabilities."""
    
    def instructions(state):
        """Dynamic instructions based on current state."""
        calc_count = len([m for m in state.messages if 'calculate' in m.content.lower()])
        
        base_instruction = """You are a helpful calculator assistant. You can perform mathematical calculations safely.
        
Available operations: addition (+), subtraction (-), multiplication (*), division (/), parentheses ()

Rules:
- Always use the calculate tool for mathematical expressions
- Explain your calculations clearly
- Results are limited to values under 1,000,000"""
        
        if calc_count > 3:
            base_instruction += "\n\nNote: You've performed several calculations. Consider summarizing results if helpful."
        
        return base_instruction
    
    return Agent(
        name='Calculator',
        instructions=instructions,
        tools=[CalculatorTool()]
    )
```

### Step 4: Run Your Agent

```python
import asyncio
from jaf import run_agent
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message
from jaf.core.engine import generate_run_id, generate_trace_id

async def main():
    """Main function to run the calculator agent."""
    
    # Set up model provider
    model_provider = make_litellm_provider('http://localhost:4000')
    
    # Create the agent
    calculator_agent = create_calculator_agent()
    
    # Configure the run
    config = RunConfig(
        agent_registry={'Calculator': calculator_agent},
        model_provider=model_provider,
        max_turns=10,
        on_event=lambda event: print(f"[{event.type}] {event.data}"),  # Simple tracing
    )
    
    # Set up initial state
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role='user', content='What is 15 * 8 + 32?')],
        current_agent_name='Calculator',
        context=CalculatorContext(
            user_id='demo_user',
            allowed_operations=['add', 'subtract', 'multiply', 'divide']
        ),
        turn_count=0,
    )
    
    # Run the agent
    print("🤖 Running Calculator Agent...")
    result = await run_agent(initial_state, config)
    
    # Handle the result
    if result.outcome.status == 'completed':
        print(f"\n✅ Success! Final output:\n{result.outcome.output}")
    else:
        print(f"\n❌ Error: {result.outcome.error}")
    
    return result

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

### Step 5: Test Your Agent

Save the code above as `calculator_agent.py` and run it:

```bash
python calculator_agent.py
```

Expected output:
```
🤖 Running Calculator Agent...
[agent_start] {'agent_name': 'Calculator', 'run_id': 'run_...'}
[tool_call] {'tool_name': 'calculate', 'args': {'expression': '15 * 8 + 32'}}
[tool_result] {'success': True, 'result': 'Result: 15 * 8 + 32 = 152'}

✅ Success! Final output:
The calculation is: 15 × 8 + 32 = 152

First, I multiply 15 by 8 to get 120, then add 32 to get a final result of 152.
```

## Interactive Chat Mode

For a more interactive experience, let's create a chat loop:

```python
async def interactive_calculator():
    """Interactive calculator chat session."""
    model_provider = make_litellm_provider('http://localhost:4000')
    calculator_agent = create_calculator_agent()
    
    config = RunConfig(
        agent_registry={'Calculator': calculator_agent},
        model_provider=model_provider,
        max_turns=20,
    )
    
    print("🧮 Calculator Agent - Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Create new state for each interaction
        state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role='user', content=user_input)],
            current_agent_name='Calculator',
            context=CalculatorContext(user_id='interactive_user', allowed_operations=['*']),
            turn_count=0,
        )
        
        result = await run_agent(state, config)
        
        if result.outcome.status == 'completed':
            print(f"Agent: {result.outcome.output}\n")
        else:
            print(f"Error: {result.outcome.error}\n")

# Run interactive mode
# asyncio.run(interactive_calculator())
```

## CLI Usage

JAF provides a CLI for common tasks:

```bash
# Initialize a new JAF project
jaf init my-calculator-project
cd my-calculator-project

# Run development server (if you have server components)
jaf server --host 0.0.0.0 --port 8000

# Show version and help
jaf --version
jaf --help
```

## Next Steps

Now that you have a working agent, explore these topics:

1. **[Core Concepts](core-concepts.md)** - Understand JAF's functional architecture
2. **[Tools Guide](tools.md)** - Build more sophisticated tools
3. **[Memory System](memory-system.md)** - Add conversation persistence
4. **[Server API](server-api.md)** - Expose your agent via HTTP API
5. **[Examples](examples.md)** - Study advanced examples

## Troubleshooting

### Common Issues

**Import Error**: If you get `ModuleNotFoundError: No module named 'jaf'`:
```bash
pip install jaf-python
# Or for development:
pip install -e .
```

**LiteLLM Connection Error**: Ensure your LiteLLM proxy is running:
```bash
# Check if proxy is accessible
curl http://localhost:4000/health
```

**Type Checking Issues**: JAF is fully typed. If you see mypy errors:
```bash
pip install mypy
mypy your_agent.py --ignore-missing-imports
```

**Performance Issues**: For high-throughput scenarios:
- Use connection pooling with your model provider
- Consider caching with Redis memory provider
- Enable performance tracing to identify bottlenecks

See [Troubleshooting](troubleshooting.md) for more detailed solutions.

## What's Next?

You've successfully created your first JAF agent! The calculator example demonstrates JAF's core principles:

- **Immutable State**: All data flows through immutable state objects
- **Pure Functions**: Business logic is predictable and testable
- **Type Safety**: Full typing with runtime validation
- **Composability**: Tools and agents can be easily combined

Ready to build more complex agents? Check out our [Examples](examples.md) for multi-agent systems, RAG implementations, and production deployments.