# Troubleshooting

This guide helps you resolve common issues when working with JAF Python.

!!! warning "Before You Start"
    Make sure you're using a supported Python version (3.9+) and have properly installed JAF with `pip install jaf-py`.

## Common Issues

### Installation Problems

#### Issue: `pip install jaf-py` fails

=== "Solution 1: Update pip"
    ```bash
    # Update pip to the latest version
    python -m pip install --upgrade pip
    
    # Then try installing JAF again
    pip install jaf-py
    ```

=== "Solution 2: Use virtual environment"
    ```bash
    # Create a fresh virtual environment
    python -m venv jaf-env
    source jaf-env/bin/activate  # On Windows: jaf-env\Scripts\activate
    
    # Install JAF
    pip install jaf-py
    ```

=== "Solution 3: Clear cache"
    ```bash
    # Clear pip cache and reinstall
    pip cache purge
    pip install --no-cache-dir jaf-py
    ```

#### Issue: Import errors after installation

!!! danger "Import Error"
    ```
    ImportError: No module named 'jaf'
    ```

**Solution**: Verify you're in the correct Python environment:

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Check installed packages
pip list | grep jaf

# Reinstall if necessary
pip uninstall jaf-py
pip install jaf-py
```

### Model Provider Issues

#### Issue: LiteLLM connection failed

!!! error "Connection Error"
    ```
    ConnectionError: Failed to connect to LiteLLM proxy at http://localhost:4000
    ```

=== "Check LiteLLM Status"
    ```bash
    # Verify LiteLLM is running
    curl http://localhost:4000/health
    
    # Should return: {"status": "healthy"}
    ```

=== "Start LiteLLM"
    ```bash
    # Install LiteLLM if not installed
    pip install litellm[proxy]
    
    # Start LiteLLM proxy
    litellm --model gpt-3.5-turbo --port 4000
    ```

=== "Alternative Provider"
    ```python
    # Use direct OpenAI if LiteLLM unavailable
    from jaf.providers.model import make_openai_provider
    
    provider = make_openai_provider(
        api_key="your-openai-key",
        model="gpt-3.5-turbo"
    )
    ```

#### Issue: Authentication errors

!!! error "Auth Error"
    ```
    AuthenticationError: Incorrect API key provided
    ```

**Solution**: Check your API key configuration:

```python
# For LiteLLM
provider = make_litellm_provider(
    'http://localhost:4000',
    api_key='your-api-key'  # Make sure this is correct
)

# For direct OpenAI
provider = make_openai_provider(
    api_key=os.getenv('OPENAI_API_KEY'),  # Use environment variable
    model='gpt-3.5-turbo'
)
```

### Agent Execution Issues

#### Issue: Agent not responding

!!! warning "Timeout or No Response"
    Your agent runs but doesn't generate any output.

=== "Check Instructions"
    ```python
    def instructions(state):
        # Make sure this returns a string
        return "You are a helpful assistant."  # ✅ Good
        # return None  # ❌ Bad - will cause issues
    ```

=== "Verify Tool Schema"
    ```python
    class MyTool:
        @property
        def schema(self):
            return ToolSchema(
                name='my_tool',
                description='A helpful tool',  # ✅ Add description
                parameters=MyArgs
            )
    ```

=== "Check Model Provider"
    ```python
    # Test your model provider directly
    async def test_provider():
        response = await model_provider.get_completion(
            test_state, test_agent, test_config
        )
        print(response)  # Should see model response
    ```

#### Issue: Tool execution fails

!!! error "Tool Error"
    ```
    ToolExecutionError: Tool 'my_tool' failed to execute
    ```

**Common causes and solutions**:

=== "Missing Parameters"
    ```python
    # Make sure all required parameters are defined
    class ToolArgs(BaseModel):
        required_param: str = Field(description="This is required")
        optional_param: str = Field(default="default", description="Optional")
    ```

=== "Async/Await Issues"
    ```python
    class MyTool:
        async def execute(self, args, context):  # ✅ Use async
            result = await some_async_operation()
            return result
        
        # Not this:
        def execute(self, args, context):  # ❌ Missing async
            return "result"
    ```

=== "Context Type Mismatch"
    ```python
    # Make sure your context type matches
    @dataclass
    class MyContext:
        user_id: str
    
    # Tool should expect the same type
    async def execute(self, args, context: MyContext):
        user_id = context.user_id  # ✅ Correct type
    ```

### Memory Provider Issues

#### Issue: Redis connection failed

!!! error "Redis Error"
    ```
    ConnectionError: Redis connection failed
    ```

=== "Check Redis Status"
    ```bash
    # Test Redis connection
    redis-cli ping
    # Should return: PONG
    ```

=== "Start Redis"
    ```bash
    # Install and start Redis (macOS)
    brew install redis
    brew services start redis
    
    # Install and start Redis (Ubuntu)
    sudo apt install redis-server
    sudo systemctl start redis
    ```

=== "Use In-Memory Provider"
    ```python
    # Fallback to in-memory if Redis unavailable
    from jaf.memory import create_in_memory_provider, InMemoryConfig
    
    memory_provider = create_in_memory_provider(
        InMemoryConfig(max_conversations=100)
    )
    ```

#### Issue: PostgreSQL connection failed

!!! error "PostgreSQL Error"
    ```
    OperationalError: could not connect to server
    ```

**Solution**: Check your PostgreSQL configuration:

```python
from jaf.memory import create_postgres_provider, PostgresConfig

# Make sure connection details are correct
memory_provider = create_postgres_provider(
    PostgresConfig(
        host='localhost',        # Correct host
        port=5432,              # Correct port  
        database='jaf_memory',   # Database exists
        username='your_user',    # User has permissions
        password='your_pass'     # Correct password
    )
)
```

## Performance Issues

### Issue: Slow agent responses

!!! warning "Performance"
    Your agents are taking too long to respond.

=== "Optimize Instructions"
    ```python
    def instructions(state):
        # Keep instructions concise and focused
        return "You are a helpful math tutor. Be concise."
        
        # Avoid overly long instructions:
        # return "You are a helpful assistant who..." (500+ words)
    ```

=== "Limit Tool Count"
    ```python
    # Use fewer, more focused tools
    agent = Agent(
        name='MathAgent',
        instructions=math_instructions,
        tools=[calculator_tool, graphing_tool]  # 2-5 tools optimal
        # tools=[tool1, tool2, ..., tool20]  # Too many tools
    )
    ```

=== "Use Async Properly"
    ```python
    # Make sure all I/O operations are async
    async def tool_execute(self, args, context):
        # Good - async database call
        result = await database.query(args.sql)
        
        # Bad - blocking call
        # result = requests.get(args.url)  # Use httpx instead
        
        return result
    ```

### Issue: Memory usage growing

!!! warning "Memory Leak"
    Memory usage keeps increasing over time.

**Solution**: Configure memory limits:

```python
from jaf.memory import InMemoryConfig

# Set reasonable limits
config = InMemoryConfig(
    max_conversations=1000,        # Limit stored conversations
    max_messages_per_conversation=100,  # Limit messages per conversation
    cleanup_interval=3600          # Cleanup every hour
)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging

# Enable debug logging for JAF
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('jaf')
logger.setLevel(logging.DEBUG)

# Your JAF code here...
```

### Use Tracing

```python
from jaf.core.tracing import ConsoleTraceCollector

# Add tracing to see what's happening
tracer = ConsoleTraceCollector()

config = RunConfig(
    # ... other config
    on_event=tracer.collect  # This will print events to console
)
```

### Test Individual Components

```python
# Test your tools separately
async def test_tool():
    tool = MyTool()
    args = MyArgs(param="test")
    context = MyContext(user_id="test")
    
    result = await tool.execute(args, context)
    print(f"Tool result: {result}")

# Test your model provider
async def test_provider():
    response = await model_provider.get_completion(state, agent, config)
    print(f"Model response: {response}")
```

## FAQ

### Q: Can I use JAF without LiteLLM?

**A**: Yes! You can use the direct OpenAI provider or implement your own model provider:

```python
from jaf.providers.model import make_openai_provider

# Direct OpenAI
provider = make_openai_provider(
    api_key="your-key",
    model="gpt-3.5-turbo"
)
```

### Q: How do I handle errors in tools?

**A**: Use try-catch blocks and return appropriate error messages:

```python
async def execute(self, args, context):
    try:
        result = await risky_operation(args.input)
        return f"Success: {result}"
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        return f"Operation failed: {str(e)}"
```

### Q: Can I run JAF in production?

**A**: Absolutely! JAF is designed for production use. See our [Deployment Guide](deployment.md) for best practices.

### Q: How do I contribute to JAF?

**A**: We welcome contributions! See the [Contributing section](https://github.com/xynehq/jaf-py#contributing) in our README.

## Getting Help

If you're still having issues:

1. **Check the [Examples](examples.md)** - Working code you can reference
2. **Review the [API Reference](api-reference.md)** - Detailed function documentation  
3. **Search [GitHub Issues](https://github.com/xynehq/jaf-py/issues)** - Someone might have solved your issue
4. **Open a new issue** - Provide error messages, code samples, and environment details

!!! tip "When Reporting Issues"
    Please include:
    
    - JAF version: `pip show jaf-py`
    - Python version: `python --version`
    - Operating system
    - Complete error traceback
    - Minimal code example that reproduces the issue