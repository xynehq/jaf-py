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

- **Python 3.10 or higher** (Python 3.11+ recommended for optimal performance and latest features)
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
pip install "jaf-py[all] @ git+https://github.com/xynehq/jaf-py.git"

# Verify installation
python -c "import jaf; print('JAF installed successfully')"
```

### Feature-Specific Installation

Install only the components you need for optimized deployments:

```bash
# Core framework only
pip install git+https://github.com/xynehq/jaf-py.git

# Server capabilities (FastAPI, uvicorn)
pip install "jaf-py[server] @ git+https://github.com/xynehq/jaf-py.git"

# Memory providers (Redis, PostgreSQL)
pip install "jaf-py[memory] @ git+https://github.com/xynehq/jaf-py.git"

# Visualization tools (Graphviz, diagrams)
pip install "jaf-py[visualization] @ git+https://github.com/xynehq/jaf-py.git"

# Tracing and observability (OpenTelemetry, Langfuse)
pip install "jaf-py[tracing] @ git+https://github.com/xynehq/jaf-py.git"

# Development tools (testing, linting, type checking)
pip install "jaf-py[dev] @ git+https://github.com/xynehq/jaf-py.git"

# Combine multiple feature sets
pip install "jaf-py[server,memory,visualization,tracing] @ git+https://github.com/xynehq/jaf-py.git"
```

### Development Environment Setup

For contributors and advanced development:

```bash
# Clone the repository
git clone https://github.com/xynehq/jaf-py.git
cd jaf-py

# Make virtual environment
python -m venv .venv
source .venv/bin/activate

# Rename .env.default to .env and update the file with your api's.

# Install in development mode with all dependencies
pip install -e ".[dev,server,memory,visualization,tracing]"

# Verify development setup
python -m pytest tests/ --tb=short

# Note: Some tests require external services:
# - Redis tests will be automatically skipped if Redis is not running locally
# - To run Redis tests, install and start Redis: brew install redis && brew services start redis
# - To manually skip Redis tests: python -m pytest tests/ -k "not redis" --tb=short
```

### Container Deployment

For containerized deployments, create your own Docker image:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install JAF
RUN pip install git+https://github.com/xynehq/jaf-py.git

# Copy your agent code
COPY . .

# Install additional dependencies if needed
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV JAF_LOG_LEVEL=INFO

# Expose port for server applications
EXPOSE 8000

# Run your agent
CMD ["python", "your_agent.py"]
```

```bash
# Build and run your containerized agent
docker build -t my-jaf-agent .

docker run -d \
  --name jaf-agent \
  -p 8000:8000 \
  -e LITELLM_URL=http://your-llm-server:4000 \
  -e LITELLM_API_KEY=your-api-key \
  my-jaf-agent
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
# LiteLLM Provider Configuration (Required)
LITELLM_URL=http://localhost:4000/
LITELLM_API_KEY=your-litellm-api-key
LITELLM_MODEL=gpt-4
PORT=3000
HOST=127.0.0.1
DEMO_MODE=development
VERBOSE_LOGGING=true

# Model Provider API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Memory Provider Configuration
# Options: memory, redis, postgres
JAF_MEMORY_TYPE=memory

# In-Memory Provider Configuration (default)
JAF_MEMORY_MAX_CONVERSATIONS=1000
JAF_MEMORY_MAX_MESSAGES=1000

# Redis Provider Configuration
# Uncomment and configure when using JAF_MEMORY_TYPE=redis
JAF_REDIS_HOST=localhost
JAF_REDIS_PORT=6379
JAF_REDIS_PASSWORD=your-redis-password
JAF_REDIS_DB=0
JAF_REDIS_PREFIX=JAF:memory:
JAF_REDIS_TTL=86400

# Alternative Redis URL (overrides individual settings)
JAF_REDIS_URL=redis://localhost:6379/0

# PostgreSQL Provider Configuration  
# Uncomment and configure when using JAF_MEMORY_TYPE=postgres
JAF_POSTGRES_HOST=localhost
JAF_POSTGRES_PORT=5432
JAF_POSTGRES_DB=jaf_test
JAF_POSTGRES_USER=postgres
JAF_POSTGRES_PASSWORD=your-postgres-password
JAF_POSTGRES_SSL=false
JAF_POSTGRES_TABLE=conversations
JAF_POSTGRES_MAX_CONNECTIONS=10

# Alternative PostgreSQL connection string (overrides individual settings)
# JAF_POSTGRES_CONNECTION_STRING=postgresql://postgres:your-postgres-password@localhost:5432/jaf_test
```

#### Production Environment

For production deployments:

```bash
# LiteLLM Provider Configuration
LITELLM_URL=https://api.your-company.com/llm/
LITELLM_API_KEY=${LITELLM_MASTER_KEY}
LITELLM_MODEL=gpt-4o
PORT=8000
HOST=0.0.0.0
DEMO_MODE=production
VERBOSE_LOGGING=false

# Memory Provider (Production Redis)
JAF_MEMORY_TYPE=redis
JAF_REDIS_URL=redis://redis-cluster.internal:6379/0
JAF_REDIS_PASSWORD=${REDIS_PASSWORD}
JAF_REDIS_PREFIX=JAF:memory:prod:
JAF_REDIS_TTL=604800  # 7 days

# Alternative: Individual Redis settings
# JAF_REDIS_HOST=redis-cluster.internal
# JAF_REDIS_PORT=6379
# JAF_REDIS_DB=0

# Alternative: PostgreSQL for persistent memory
# JAF_MEMORY_TYPE=postgres
# JAF_POSTGRES_CONNECTION_STRING=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
# JAF_POSTGRES_SSL=true
# JAF_POSTGRES_MAX_CONNECTIONS=50

# A2A (Agent-to-Agent) Configuration (if using multi-agent systems)
JAF_A2A_MEMORY_TYPE=redis
JAF_A2A_KEY_PREFIX=JAF:a2a:prod:
JAF_A2A_DEFAULT_TTL=86400
JAF_A2A_CLEANUP_ENABLED=true
JAF_A2A_CLEANUP_INTERVAL=3600
JAF_A2A_MAX_TASKS=10000

# Performance and Cleanup Settings
JAF_A2A_CLEANUP_MAX_AGE=604800  # 7 days
JAF_A2A_CLEANUP_MAX_COMPLETED=1000
JAF_A2A_CLEANUP_MAX_FAILED=500
JAF_A2A_CLEANUP_BATCH_SIZE=100
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

### Step 2: Tool Implementation with Modern Object-Based API

JAF's modern tool creation API prioritizes type safety, functional composition, and developer experience. This section demonstrates both the recommended object-based approach and the traditional class-based approach for comparison.

#### Object-Based API (Production Recommended)

The object-based API leverages TypedDict configurations and functional programming principles for superior maintainability and extensibility:

```python
from jaf import function_tool
import ast
import operator

def _safe_eval(node, context):
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

@function_tool
async def calculate(expression: str, context: 'CalculatorContext') -> str:
    """Safely evaluate mathematical expressions using AST parsing.
    
    This function implements secure expression evaluation using AST parsing
    instead of direct eval() to prevent code injection attacks.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')
    """
    try:
        # Input validation
        if not expression or len(expression.strip()) == 0:
            return "Error: Expression cannot be empty"
        
        if len(expression) > 200:
            return "Error: Expression too long (max 200 characters)"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            '__', 'import', 'exec', 'eval', 'open', 'file',
            'input', 'raw_input', 'compile', 'globals', 'locals'
        ]
        
        cleaned = expression.replace(' ', '')
        for pattern in dangerous_patterns:
            if pattern in cleaned.lower():
                return f"Error: Expression contains prohibited pattern: {pattern}"
        
        # Only allow safe mathematical characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        # Permission check
        if not context.has_permission('basic_math'):
            return "Error: Mathematical operations require basic_math permission"
        
        # Parse expression safely using AST
        try:
            tree = ast.parse(expression, mode='eval')
            result = _safe_eval(tree.body, context)
        except (SyntaxError, ValueError) as e:
            return f"Error: Invalid mathematical expression: {str(e)}"
        
        # Apply context limits
        if abs(result) > context.max_result:
            return f"Error: Result {result} exceeds maximum allowed value ({context.max_result})"
        
        # Format result with context precision
        if isinstance(result, float):
            result = round(result, context.precision)
        
        return f"Result: {expression} = {result}"
        
    except Exception as e:
        return f"Error: Failed to evaluate expression: {str(e)}"
```

#### Class-Based API (Legacy Support)

While the modern `@function_tool` decorator is recommended for new development, JAF maintains full backward compatibility with the traditional class-based approach for existing codebases:

```python
from jaf import function_tool

@function_tool
async def calculate_legacy(expression: str, context: 'CalculatorContext') -> str:
    """Execute calculation with safety checks (legacy API pattern).
    
    This demonstrates the same functionality using the modern decorator
    while maintaining backward compatibility for existing systems.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')
    """
    try:
        # Simple whitelist validation
        allowed_chars = set('0123456789+-*/.() ')
        if not all(char in allowed_chars for char in expression):
            return "Error: Expression contains invalid characters"
        
        # Check context limits (demonstration of context usage)
        if not context.can_perform_operation('multiply') and '*' in expression:
            return "Error: Multiplication not allowed in current context"
        
        # Evaluate safely (in production, use a proper math parser)
        result = eval(expression)
        
        # Check context limits
        if abs(result) > context.max_result:
            return f"Error: Result {result} exceeds maximum allowed value"
        
        return f"Result: {expression} = {result}"
        
    except Exception as e:
        return f"Error: Calculation error: {str(e)}"
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

def create_calculator_agent() -> Agent:
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
        tools=[calculate]
    )
```

### Step 4: Run Your Agent

```python
import asyncio
from jaf import run, make_litellm_provider
from jaf import RunState, RunConfig, Message, generate_run_id, generate_trace_id

async def main():
    """Main function to run the calculator agent."""
    
    # Set up model provider
    import os
    litellm_url = os.getenv('LITELLM_URL', 'http://localhost:4000/')
    litellm_api_key = os.getenv('LITELLM_API_KEY', 'anything')
    model_provider = make_litellm_provider(litellm_url, litellm_api_key)
    
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
            session_id='demo_session',
            allowed_operations=['add', 'subtract', 'multiply', 'divide']
        ),
        turn_count=0,
    )
    
    # Run the agent
    print("🤖 Running Calculator Agent...")
    result = await run(initial_state, config)
    
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
    import os
    litellm_url = os.getenv('LITELLM_URL', 'http://localhost:4000/')
    litellm_api_key = os.getenv('LITELLM_API_KEY', 'anything')
    model_provider = make_litellm_provider(litellm_url, litellm_api_key)
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
            context=CalculatorContext(
                user_id='interactive_user', 
                session_id='interactive_session',
                allowed_operations=['add', 'subtract', 'multiply', 'divide']
            ),
            turn_count=0,
        )
        
        result = await run(state, config)
        
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

## Adding Observability and Tracing

JAF provides comprehensive observability through its tracing system. This enables monitoring, debugging, and performance analysis of your agents.

### Basic Console Tracing

The simplest way to add observability is with console tracing:

```python
from jaf.core.tracing import ConsoleTraceCollector

# Create console trace collector
trace_collector = ConsoleTraceCollector()

# Update your calculator example with tracing
config = RunConfig(
    agent_registry={'Calculator': calculator_agent},
    model_provider=model_provider,
    max_turns=10,
    on_event=trace_collector.collect,  # Enable detailed tracing
)

# Run your agent - you'll see detailed execution logs
result = await run(initial_state, config)
```

This provides detailed execution logs including:
- Agent execution start/end
- LLM calls and responses
- Tool executions
- Performance timing
- Error conditions

### Advanced Tracing with OpenTelemetry

For production environments, use OpenTelemetry for industry-standard observability:

```python
import os
from jaf.core.tracing import create_composite_trace_collector, ConsoleTraceCollector

# Configure OpenTelemetry (requires running OTLP collector like Jaeger)
os.environ["TRACE_COLLECTOR_URL"] = "http://localhost:4318/v1/traces"

# Auto-configured tracing with multiple backends
trace_collector = create_composite_trace_collector(
    ConsoleTraceCollector()  # Console output for development
    # OpenTelemetry automatically added if TRACE_COLLECTOR_URL is set
)

config = RunConfig(
    agent_registry={'Calculator': calculator_agent},
    model_provider=model_provider,
    max_turns=10,
    on_event=trace_collector.collect,
)
```

To view OpenTelemetry traces, start Jaeger:

```bash
# Start Jaeger for trace visualization
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# View traces at http://localhost:16686
```

### Langfuse Integration

For AI-specific observability, JAF integrates with Langfuse:

```bash
# Set environment variables for Langfuse
export LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
export LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted instance
```

```python
# Langfuse tracing is automatically enabled when keys are configured
trace_collector = create_composite_trace_collector(
    ConsoleTraceCollector()
    # Langfuse automatically added if LANGFUSE_* environment variables are set
)

config = RunConfig(
    agent_registry={'Calculator': calculator_agent},
    model_provider=model_provider,
    max_turns=10,
    on_event=trace_collector.collect,
)
```

### File-Based Tracing

For debugging and analysis, save traces to files:

```python
from jaf.core.tracing import FileTraceCollector

file_collector = FileTraceCollector("traces/calculator_traces.jsonl")

config = RunConfig(
    agent_registry={'Calculator': calculator_agent},
    model_provider=model_provider,
    max_turns=10,
    on_event=file_collector.collect,
)
```

Traces are saved as JSON Lines format for easy analysis with tools like `jq`:

```bash
# Analyze trace files
cat traces/calculator_traces.jsonl | jq '.type' | sort | uniq -c
cat traces/calculator_traces.jsonl | jq 'select(.type == "tool_call_start")'
```

### Next Steps with Observability

- **[Tracing Guide](tracing.md)** - Comprehensive tracing documentation
- **[Performance Monitoring](performance-monitoring.md)** - Optimize agent performance
- **[Production Deployment](deployment.md)** - Deploy with observability

## Next Steps

Now that you have a working agent, explore these topics:

1. **[Core Concepts](core-concepts.md)** - Understand JAF's functional architecture
2. **[Tools Guide](tools.md)** - Build more sophisticated tools
3. **[Agent-as-Tool](agent-as-tool.md)** - Create hierarchical multi-agent systems
4. **[Tracing Guide](tracing.md)** - Comprehensive observability and monitoring
5. **[Memory System](memory-system.md)** - Add conversation persistence
6. **[Server API](server-api.md)** - Expose your agent via HTTP API
7. **[Examples](examples.md)** - Study advanced examples

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
