# Getting Started with JAF

Welcome to **JAF (Juspay Agent Framework)** - a production-ready, functionally pure framework for building AI agents with immutable state and composable architecture. This comprehensive guide provides everything you need to build sophisticated AI agent systems.

## Learning Objectives

By completing this guide, you will have:

- **Installed and configured** JAF with all necessary dependencies
- **Built your first functional agent** using modern object-based APIs
- **Mastered core architectural concepts** including immutable state and pure functions
- **Implemented a complete working example** ready for production extension
- **Understanding of best practices** for scalable agent development

## Prerequisites and System Requirements

### Technical Requirements

- **Python 3.9 or higher** (Python 3.11+ recommended for optimal performance and latest features)
- **LiteLLM proxy server** for LLM integration, or direct access to LLM APIs
- **Development environment** with package management (pip, conda, or poetry)
- **Basic understanding** of Python asyncio, type hints, and functional programming concepts

### Knowledge Prerequisites

This guide assumes familiarity with:

- **Python programming** including classes, decorators, and async/await patterns
- **Type hints and annotations** using typing module and Pydantic
- **REST API concepts** for server integration scenarios
- **Basic understanding of AI/LLM concepts** such as prompts, tools, and agent workflows

## Installation and Setup

### Production Installation

For production environments, install JAF with all dependencies:

```bash
# Complete installation with all features
pip install "git+https://github.com/xynehq/jaf-py.git[all]"

# Verify installation
python -c "import jaf; print('JAF installed successfully')"
```

### Feature-Specific Installation

Install only the components you need for optimized deployments:

```bash
# Core framework only
pip install git+https://github.com/xynehq/jaf-py.git

# Server capabilities (FastAPI, uvicorn)
pip install "git+https://github.com/xynehq/jaf-py.git[server]"

# Memory providers (Redis, PostgreSQL)
pip install "git+https://github.com/xynehq/jaf-py.git[memory]"

# Visualization tools (Graphviz, diagrams)
pip install "git+https://github.com/xynehq/jaf-py.git[visualization]"

# Development tools (testing, linting, type checking)
pip install "git+https://github.com/xynehq/jaf-py.git[dev]"

# Combine multiple feature sets
pip install "git+https://github.com/xynehq/jaf-py.git[server,memory,visualization]"
```

### Development Environment Setup

For contributors and advanced development:

```bash
# Clone the repository
git clone https://github.com/xynehq/jaf-py.git
cd jaf-py

# Install in development mode with all dependencies
pip install -e ".[dev,server,memory,visualization]"

# Install pre-commit hooks
pre-commit install

# Verify development setup
python -m pytest tests/ --tb=short
```

### Container Deployment

For containerized deployments:

```bash
# Use official image
docker pull xynehq/jaf-py:latest

# Run with configuration
docker run -d \
  --name jaf-agent \
  -p 8000:8000 \
  -e LITELLM_BASE_URL=http://your-llm-server:4000 \
  -e JAF_LOG_LEVEL=INFO \
  xynehq/jaf-py:latest

# Custom build with your agents
FROM xynehq/jaf-py:latest
COPY your_agents/ /app/agents/
CMD ["python", "-m", "your_agents.main"]
```

## Model Provider Configuration

JAF integrates with 100+ LLM models through LiteLLM, providing a unified interface for OpenAI, Anthropic, Google, and other providers. This section covers both development and production configurations.

### LiteLLM Proxy Setup

#### Development Configuration

```bash
# Install LiteLLM with proxy support
pip install litellm[proxy]

# Create development configuration
cat > litellm_config.yaml << EOF
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: ${OPENAI_API_KEY}
      max_tokens: 4096
      temperature: 0.1
  
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: ${ANTHROPIC_API_KEY}
      max_tokens: 4096
      temperature: 0.1
  
  - model_name: gemini-pro
    litellm_params:
      model: google/gemini-pro
      api_key: ${GOOGLE_API_KEY}

general_settings:
  master_key: "your-proxy-master-key"
  database_url: "sqlite:///litellm_proxy.db"
  
router_settings:
  routing_strategy: "least-busy"
  model_group_alias:
    "gpt-4": ["gpt-4o", "gpt-4-turbo"]
    "claude": ["claude-3-sonnet", "claude-3-haiku"]
EOF

# Start LiteLLM proxy with enhanced configuration
litellm --config litellm_config.yaml --port 4000 --num_workers 4
```

#### Production Configuration

For production deployments, consider these additional configurations:

```yaml
# litellm_production.yaml
model_list:
  # Load balanced OpenAI endpoints
  - model_name: gpt-4o-primary
    litellm_params:
      model: openai/gpt-4o
      api_key: ${OPENAI_PRIMARY_KEY}
      api_base: ${OPENAI_PRIMARY_BASE}
  
  - model_name: gpt-4o-fallback
    litellm_params:
      model: openai/gpt-4o
      api_key: ${OPENAI_FALLBACK_KEY}
      api_base: ${OPENAI_FALLBACK_BASE}

general_settings:
  master_key: ${LITELLM_MASTER_KEY}
  database_url: ${DATABASE_URL}
  redis_url: ${REDIS_URL}
  
  # Security settings
  enforce_user_param: true
  allowed_ips: ["10.0.0.0/8", "172.16.0.0/12"]
  
  # Rate limiting
  global_max_parallel_requests: 1000
  rpm_limit: 10000
  tpm_limit: 1000000

router_settings:
  routing_strategy: "least-busy"
  fallback_models:
    - "gpt-4o-fallback"
  
  retry_policy:
    max_retries: 3
    retry_delay: 1.0
    backoff_factor: 2.0

litellm_settings:
  telemetry: false
  success_callback: ["prometheus", "langfuse"]
  failure_callback: ["slack", "prometheus"]
```

### Environment Configuration

#### Development Environment

Create a `.env` file for local development:

```bash
# Core Configuration
JAF_ENV=development
JAF_LOG_LEVEL=DEBUG
JAF_DEBUG=true

# LiteLLM Integration
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your-proxy-master-key
LITELLM_TIMEOUT=60
LITELLM_MAX_RETRIES=3

# Model Provider API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Memory Provider Configuration
JAF_MEMORY_TYPE=in_memory
JAF_MEMORY_MAX_CONVERSATIONS=1000
JAF_MEMORY_MAX_MESSAGES=10000

# Optional: Redis for distributed memory
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password
REDIS_MAX_CONNECTIONS=20

# Optional: PostgreSQL for persistent memory
DATABASE_URL=postgresql://user:password@localhost:5432/jaf_dev
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Observability and Monitoring
JAF_TRACE_ENABLED=true
JAF_TRACE_LEVEL=INFO
JAF_METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Security Configuration
JAF_CORS_ENABLED=true
JAF_CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
JAF_API_KEY_REQUIRED=false  # Set to true in production
```

#### Production Environment

For production deployments:

```bash
# Core Configuration
JAF_ENV=production
JAF_LOG_LEVEL=INFO
JAF_DEBUG=false

# LiteLLM Configuration
LITELLM_BASE_URL=https://api.your-company.com/llm
LITELLM_API_KEY=${LITELLM_MASTER_KEY}
LITELLM_TIMEOUT=120
LITELLM_MAX_RETRIES=5

# Memory Provider (Production Redis)
JAF_MEMORY_TYPE=redis
REDIS_URL=redis://redis-cluster.internal:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_MAX_CONNECTIONS=100
REDIS_SSL=true

# Database Configuration
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_SSL_MODE=require

# Security Configuration
JAF_API_KEY_REQUIRED=true
JAF_API_KEYS=${ALLOWED_API_KEYS}  # Comma-separated list
JAF_CORS_ENABLED=true
JAF_CORS_ORIGINS=${ALLOWED_ORIGINS}  # JSON array string
JAF_RATE_LIMIT_ENABLED=true
JAF_RATE_LIMIT_REQUESTS=1000
JAF_RATE_LIMIT_WINDOW=3600

# Monitoring and Observability
JAF_TRACE_ENABLED=true
JAF_TRACE_LEVEL=WARN
JAF_METRICS_ENABLED=true
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=${JAEGER_COLLECTOR_ENDPOINT}
SENTRY_DSN=${SENTRY_DSN}

# Performance Tuning
JAF_WORKER_PROCESSES=4
JAF_MAX_CONCURRENT_REQUESTS=1000
JAF_REQUEST_TIMEOUT=300
JAF_MEMORY_LIMIT=2048  # MB
```

## Building Your First Production Agent

This section demonstrates JAF's core concepts through a comprehensive calculator agent example. You'll learn about context definition, tool creation using the modern object-based API, agent configuration, and execution patterns.

### Step 1: Context Definition and Type Safety

Context objects in JAF are immutable data structures that carry state throughout the agent execution lifecycle. They provide type safety and ensure predictable behavior.

```python
# calculator_agent.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass(frozen=True)  # Immutable context
class CalculatorContext:
    """
    Immutable context for calculator agent operations.
    
    This context carries user-specific configuration and permissions
    throughout the agent execution lifecycle.
    """
    user_id: str
    session_id: str
    allowed_operations: List[str]
    max_result: float = 1000000.0
    precision: int = 10
    user_permissions: List[str] = None
    session_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.user_permissions is None:
            object.__setattr__(self, 'user_permissions', ['basic_math'])
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())
    
    def has_permission(self, operation: str) -> bool:
        """Check if user has permission for specific operation."""
        return operation in self.user_permissions
    
    def can_perform_operation(self, operation: str) -> bool:
        """Check if operation is allowed in current context."""
        return operation in self.allowed_operations
```

### Step 2: Tool Implementation with @function_tool Decorator

JAF's modern tool creation uses the `@function_tool` decorator for clean, type-safe tool definitions. This approach automatically extracts type information and docstrings to create properly configured tools.

#### @function_tool Decorator (Production Recommended)

The `@function_tool` decorator provides the cleanest way to create tools with automatic schema generation:

```python
from jaf import function_tool
import re
import ast
import operator

@function_tool
async def calculate(expression: str, context) -> str:
    """
    Execute mathematical calculation with comprehensive safety checks.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')
    
    This function implements secure expression evaluation using AST parsing
    instead of direct eval() to prevent code injection attacks.
    """
    try:
        # Input validation
        if not expression or len(expression.strip()) == 0:
            return "Error: Expression cannot be empty"
        
        if len(expression) > 200:
            return "Error: Expression too long (max 200 characters)"
        
        # Check for dangerous patterns
        dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file']
        for pattern in dangerous_patterns:
            if pattern in expression.lower():
                return f"Error: Expression contains prohibited pattern: {pattern}"
        
        # Parse expression safely using AST
        try:
            tree = ast.parse(expression, mode='eval')
            result = _safe_eval(tree.body, context)
        except (SyntaxError, ValueError) as e:
            return f"Error: Invalid mathematical expression: {str(e)}"
        
        # Format result
        if isinstance(result, float):
            result = round(result, 2)
        
        return f"Result: {expression} = {result}"
        
    except Exception as e:
        return f"Error: Failed to evaluate expression: {str(e)}"

def _safe_eval(node, context: CalculatorContext):
    """Safely evaluate AST node with limited operations."""
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in safe_operators:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        left = _safe_eval(node.left, context)
        right = _safe_eval(node.right, context)
        return safe_operators[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in safe_operators:
            raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
        operand = _safe_eval(node.operand, context)
        return safe_operators[type(node.op)](operand)
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

# The decorator automatically creates the tool with these properties:
# - Tool name: 'calculate' (from function name)
# - Description: extracted from docstring
# - Parameters: auto-generated from function signature
# - Source: defaults to ToolSource.NATIVE

# For advanced configuration, you can use decorator parameters:
@function_tool(
    name="advanced_calculate",
    description="Enhanced calculator with metadata",
    metadata={
        'category': 'mathematical_operations',
        'safety_level': 'high',
        'version': '1.0.0'
    }
)
async def advanced_calculate(expression: str, precision: int = 2, context=None) -> str:
    """Advanced calculator with configurable precision.
    
    Args:
        expression: Mathematical expression to evaluate
        precision: Number of decimal places for results
    """
    # Implementation would be similar to above
    return f"Advanced calculation: {expression}"
```

#### Class-Based API (Legacy Support)

While the object-based API is recommended for new development, JAF maintains full backward compatibility with the traditional class-based approach:

    ```python
    from pydantic import BaseModel, Field
    from jaf import ToolResponse

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
                    return ToolResponse.validation_error("Expression contains invalid characters")
                
                # Evaluate safely (in production, use a proper math parser)
                result = eval(args.expression)
                
                # Check context limits
                if abs(result) > context.max_result:
                    return ToolResponse.validation_error(f"Result {result} exceeds maximum allowed value")
                
                return ToolResponse.success(f"Result: {args.expression} = {result}")
                
            except Exception as e:
                return ToolResponse.error(f"Calculation error: {str(e)}")
    
    calculator_tool = CalculatorTool()
    ```

**Key Advantages of Object-Based API:**

- **Enhanced Type Safety**: Complete TypedDict support with full IDE autocomplete and static analysis
- **Superior Extensibility**: Seamless addition of metadata, source tracking, versioning, and custom configurations
- **Functional Composition**: Native integration with higher-order functions and composition patterns
- **Future-Proof Architecture**: Primary target for new features, optimizations, and enhancements
- **Production Readiness**: Designed for enterprise-scale deployments with comprehensive error handling

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
from adk.runners import run_agent
from jaf import make_litellm_provider
from jaf import RunState, RunConfig, Message, generate_run_id, generate_trace_id

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
        print(f"\nSuccess! Final output:\n{result.outcome.output}")
    else:
        print(f"\nError: {result.outcome.error}")
    
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

Success! Final output:
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
pip install git+https://github.com/xynehq/jaf-py.git
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