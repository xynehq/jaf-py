# Function Composition Patterns

JAF's architecture is built on functional programming principles, enabling sophisticated composition patterns that promote code reusability, testability, and maintainability. This comprehensive guide demonstrates how to leverage these patterns for building production-grade agent systems.

## Architectural Overview

Function composition in JAF enables several key architectural patterns:

### Core Composition Benefits

- **Cross-Cutting Concern Integration**: Seamlessly add logging, caching, retry logic, and monitoring to any component
- **Validation Pipeline Construction**: Build complex validation rules from simple, testable predicates  
- **Middleware-Style Agent Enhancement**: Layer agent behaviors using composable instruction modifiers
- **Stream Processing Pipelines**: Construct data processing workflows from individual transformation steps
- **Memory Strategy Composition**: Combine multiple memory providers for sophisticated storage patterns

### Design Principles

1. **Pure Function Priority**: Maintain functional purity wherever possible for predictable behavior
2. **Immutable Data Flow**: Ensure data transformations don't mutate original inputs
3. **Type Safety Throughout**: Leverage Python's type system for compile-time composition validation
4. **Error Boundary Management**: Handle failures gracefully without breaking composition chains
5. **Performance Optimization**: Enable optimizations like memoization and lazy evaluation

## Tool Composition

### Higher-Order Functions for Tools

Higher-order functions are the foundation of tool composition. They take a function as input and return an enhanced version:

```python
from jaf import create_function_tool, ToolSource
from jaf import ToolResponse

# Base tool function
async def search_execute(args, context):
    """Basic search functionality."""
    results = await perform_search(args.query)
    return ToolResponse.success(results)

# Higher-order function for caching
def with_cache(tool_func, cache_ttl=300):
    """Add caching to any tool function."""
    cache = {}
    
    async def cached_execute(args, context):
        cache_key = str(args)
        current_time = time.time()
        
        # Check cache
        if cache_key in cache:
            cached_result, timestamp = cache[cache_key]
            if current_time - timestamp < cache_ttl:
                logger.debug(f"Cache hit for {tool_func.__name__}", extra={'cache_key': cache_key})
                return cached_result
        
        # Execute and cache
        result = await tool_func(args, context)
        if result.status == "success":
            cache[cache_key] = (result, current_time)
            logger.debug(f"Cached result for {tool_func.__name__}", extra={'cache_key': cache_key})
        
        return result
    
    return cached_execute

# Create enhanced tool
search_tool = create_function_tool({
    'name': 'cached_search',
    'description': 'Search with caching',
    'execute': with_cache(search_execute),
    'parameters': SearchArgs,
    'metadata': {'enhanced': True, 'features': ['caching']},
    'source': ToolSource.NATIVE
})
```

### Retry Logic

Add robust retry logic to handle transient failures:

```python
import asyncio
from typing import Optional

def with_retry(tool_func, max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    """Add exponential backoff retry to tool functions."""
    
    async def retry_execute(args, context):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = await tool_func(args, context)
                if result.status == "success":
                    return result
                elif attempt == max_retries - 1:
                    return result  # Return last result on final attempt
                    
            except exceptions as e:
                last_exception = e
                if attempt == max_retries - 1:
                    return ToolResponse.error(f"Failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...", 
                             extra={'attempt': attempt + 1, 'wait_time': wait_time})
                await asyncio.sleep(wait_time)
        
        return ToolResponse.error(f"Max retries exceeded: {str(last_exception)}")
    
    return retry_execute

# Usage
reliable_search = create_function_tool({
    'name': 'reliable_search',
    'description': 'Search with retry logic',
    'execute': with_retry(search_execute, max_retries=3),
    'parameters': SearchArgs,
    'source': ToolSource.NATIVE
})
```

### Logging and Observability

Add comprehensive logging to any tool:

```python
import functools
import time
from typing import Any

def with_logging(tool_func, logger=None):
    """Add detailed logging to tool execution."""
    if logger is None:
        import logging
        logger = logging.getLogger(f"jaf.tool.{tool_func.__name__}")
    
    @functools.wraps(tool_func)
    async def logged_execute(args, context):
        start_time = time.time()
        tool_name = getattr(tool_func, '__name__', 'unknown')
        
        logger.info(f"Starting tool execution: {tool_name}", extra={
            'tool_name': tool_name,
            'args': str(args),
            'context_keys': list(context.keys()) if isinstance(context, dict) else None
        })
        
        try:
            result = await tool_func(args, context)
            duration = time.time() - start_time
            
            logger.info(f"Tool execution completed: {tool_name} ({duration:.3f}s)", extra={
                'tool_name': tool_name,
                'duration_ms': duration * 1000,
                'status': result.status,
                'success': result.status == 'success'
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Tool execution failed: {tool_name} after {duration:.3f}s - {str(e)}", extra={
                'tool_name': tool_name,
                'duration_ms': duration * 1000,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
    
    return logged_execute
```

### Rate Limiting

Implement rate limiting for external API calls:

```python
import asyncio
from collections import defaultdict, deque
import time

def with_rate_limit(tool_func, max_calls=10, time_window=60, key_func=None):
    """Add rate limiting to tool functions."""
    call_history = defaultdict(deque)
    
    if key_func is None:
        key_func = lambda args, context: context.get('user_id', 'global')
    
    async def rate_limited_execute(args, context):
        key = key_func(args, context)
        now = time.time()
        
        # Clean old calls
        while call_history[key] and call_history[key][0] < now - time_window:
            call_history[key].popleft()
        
        # Check rate limit
        if len(call_history[key]) >= max_calls:
            return ToolResponse.error(
                f"Rate limit exceeded: {max_calls} calls per {time_window}s"
            )
        
        # Record call and execute
        call_history[key].append(now)
        return await tool_func(args, context)
    
    return rate_limited_execute
```

### Composing Multiple Enhancements

Chain multiple enhancements together:

```python
# Compose multiple enhancements
enhanced_search = create_function_tool({
    'name': 'enhanced_search',
    'description': 'Search with caching, retry, logging, and rate limiting',
    'execute': with_logging(
        with_rate_limit(
            with_cache(
                with_retry(search_execute, max_retries=3),
                cache_ttl=300
            ),
            max_calls=100,
            time_window=3600
        )
    ),
    'parameters': SearchArgs,
    'metadata': {
        'enhanced': True,
        'features': ['caching', 'retry', 'logging', 'rate_limiting']
    },
    'source': ToolSource.NATIVE
})
```

## Validator Composition

Build complex validation logic from simple, testable functions:

```python
from jaf import ValidationResult, ValidValidationResult, InvalidValidationResult

def compose_validators(*validators):
    """Compose multiple validation functions into one."""
    def composed_validator(data):
        for validator in validators:
            result = validator(data)
            if not result.get("is_valid", False):
                return result
        return {"is_valid": True}
    return composed_validator

# Individual validators
def validate_required_fields(required_fields):
    """Create a validator for required fields."""
    def validator(data):
        for field in required_fields:
            if not hasattr(data, field) or not getattr(data, field):
                return {"is_valid": False, "error": f"Missing required field: {field}"}
        return {"is_valid": True}
    return validator

def validate_string_length(field, min_length=None, max_length=None):
    """Create a validator for string length."""
    def validator(data):
        if not hasattr(data, field):
            return {"is_valid": True}  # Skip if field doesn't exist
        
        value = getattr(data, field)
        if not isinstance(value, str):
            return {"is_valid": False, "error": f"{field} must be a string"}
        
        if min_length and len(value) < min_length:
            return {"is_valid": False, "error": f"{field} must be at least {min_length} characters"}
        
        if max_length and len(value) > max_length:
            return {"is_valid": False, "error": f"{field} must be no more than {max_length} characters"}
        
        return {"is_valid": True}
    return validator

def validate_email_format(field):
    """Create an email format validator."""
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    def validator(data):
        if not hasattr(data, field):
            return {"is_valid": True}
        
        email = getattr(data, field)
        if not re.match(email_pattern, email):
            return {"is_valid": False, "error": f"Invalid email format: {email}"}
        
        return {"is_valid": True}
    return validator

# Compose user validation
user_validator = compose_validators(
    validate_required_fields(['name', 'email']),
    validate_string_length('name', min_length=2, max_length=50),
    validate_string_length('email', max_length=254),
    validate_email_format('email')
)

# Use in tool
async def create_user_execute(args, context):
    validation = user_validator(args)
    if not validation.get("is_valid", False):
        return ToolResponse.validation_error(validation.get("error", "Validation failed"))
    
    # Proceed with user creation
    user = await create_user_in_database(args)
    return ToolResponse.success(user)
```

## Agent Behavior Composition

Layer agent functionality using middleware-style patterns:

```python
def with_context_enhancement(agent_func):
    """Enhance agent with additional context information."""
    def enhanced_agent(state):
        # Add helpful context
        enhanced_instructions = agent_func(state)
        
        # Add current time and user context
        context_info = f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if hasattr(state.context, 'user_id'):
            context_info += f"\nUser ID: {state.context.user_id}"
        
        return enhanced_instructions + context_info
    
    return enhanced_agent

def with_safety_guidelines(agent_func):
    """Add safety guidelines to agent instructions."""
    def safe_agent(state):
        base_instructions = agent_func(state)
        
        safety_guidelines = """
        
SAFETY GUIDELINES:
- Never provide harmful, illegal, or unethical information
- Protect user privacy and confidentiality
- If uncertain about a request, ask for clarification
- Escalate concerning requests to human oversight
        """
        
        return base_instructions + safety_guidelines
    
    return safe_agent

def with_conversation_memory(agent_func, max_history=5):
    """Add conversation context to agent instructions."""
    def memory_enhanced_agent(state):
        base_instructions = agent_func(state)
        
        # Add recent conversation context
        recent_messages = state.messages[-max_history:] if len(state.messages) > 1 else []
        if recent_messages:
            context = "\n\nRECENT CONVERSATION:\n"
            for msg in recent_messages:
                context += f"{msg.role}: {msg.content[:100]}...\n"
            return base_instructions + context
        
        return base_instructions
    
    return memory_enhanced_agent

# Compose agent behaviors
def create_enhanced_instructions(state):
    base_instructions = "You are a helpful AI assistant."
    
    return with_context_enhancement(
        with_safety_guidelines(
            with_conversation_memory(
                lambda s: base_instructions
            )
        )
    )(state)
```

## Memory Provider Composition

Create sophisticated memory strategies by combining providers:

```python
from jaf.memory import create_in_memory_provider, create_redis_provider

def create_tiered_memory(fast_provider, persistent_provider):
    """Create a two-tier memory system with fast cache and persistent storage."""
    
    class TieredMemoryProvider:
        def __init__(self):
            self.fast = fast_provider
            self.persistent = persistent_provider
        
        async def get_conversation(self, conversation_id):
            # Try fast cache first
            result = await self.fast.get_conversation(conversation_id)
            if result.data:
                return result
            
            # Fall back to persistent storage
            result = await self.persistent.get_conversation(conversation_id)
            if result.data:
                # Warm the fast cache
                await self.fast.store_messages(
                    conversation_id,
                    result.data.messages,
                    result.data.metadata
                )
            
            return result
        
        async def store_messages(self, conversation_id, messages, metadata=None):
            # Store in both tiers
            results = await asyncio.gather(
                self.fast.store_messages(conversation_id, messages, metadata),
                self.persistent.store_messages(conversation_id, messages, metadata),
                return_exceptions=True
            )
            
            # Return persistent storage result (more authoritative)
            return results[1] if not isinstance(results[1], Exception) else results[0]
        
        async def delete_conversation(self, conversation_id):
            # Delete from both tiers
            await asyncio.gather(
                self.fast.delete_conversation(conversation_id),
                self.persistent.delete_conversation(conversation_id),
                return_exceptions=True
            )
        
        async def health_check(self):
            fast_health, persistent_health = await asyncio.gather(
                self.fast.health_check(),
                self.persistent.health_check(),
                return_exceptions=True
            )
            
            return {
                "healthy": (
                    fast_health.get("healthy", False) and 
                    persistent_health.get("healthy", False)
                ),
                "tiers": {
                    "fast": fast_health,
                    "persistent": persistent_health
                }
            }
    
    return TieredMemoryProvider()

# Usage
fast_cache = create_in_memory_provider(InMemoryConfig(max_conversations=1000))
persistent_store = create_redis_provider(RedisConfig(host="localhost"))
tiered_memory = create_tiered_memory(fast_cache, persistent_store)
```

## Pipeline Composition

Build processing pipelines for complex workflows:

```python
def create_pipeline(*steps):
    """Create a processing pipeline from multiple steps."""
    
    async def pipeline_execute(args, context):
        data = args
        step_results = []
        
        for i, step in enumerate(steps):
            try:
                result = await step(data, context)
                step_results.append({
                    'step': i,
                    'name': step.__name__,
                    'success': True,
                    'result': result
                })
                
                # Check for pipeline termination
                if hasattr(result, 'status') and result.status != 'success':
                    return ToolResponse.error(
                        f"Pipeline failed at step {i} ({step.__name__}): {result.error}"
                    )
                
                # Update data for next step
                data = result.data if hasattr(result, 'data') else result
                
            except Exception as e:
                step_results.append({
                    'step': i,
                    'name': step.__name__,
                    'success': False,
                    'error': str(e)
                })
                return ToolResponse.error(f"Pipeline failed at step {i}: {str(e)}")
        
        return ToolResponse.success({
            'final_result': data,
            'pipeline_steps': step_results
        })
    
    return pipeline_execute

# Example: NLP Pipeline
async def extract_entities(text, context):
    # Extract named entities
    entities = await nlp_service.extract_entities(text)
    return ToolResponse.success(entities)

async def classify_intent(entities, context):
    # Classify intent from entities
    intent = await intent_classifier.predict(entities)
    return ToolResponse.success(intent)

async def generate_response(intent, context):
    # Generate appropriate response
    response = await response_generator.generate(intent)
    return ToolResponse.success(response)

# Create NLP pipeline tool
nlp_pipeline = create_function_tool({
    'name': 'nlp_pipeline',
    'description': 'Process text through complete NLP pipeline',
    'execute': create_pipeline(extract_entities, classify_intent, generate_response),
    'parameters': TextProcessingArgs,
    'metadata': {'type': 'pipeline', 'steps': 3},
    'source': ToolSource.NATIVE
})
```

## Best Practices

### 1. Keep Functions Pure

```python
# Good: Pure function
async def search_api(query: str) -> List[Dict]:
    response = await http_client.get(f"/search?q={query}")
    return response.json()

# Better: Composed with side effects isolated
search_tool = create_function_tool({
    'name': 'search',
    'execute': with_logging(with_cache(search_api)),
    'parameters': SearchArgs,
    'source': ToolSource.NATIVE
})
```

### 2. Use Type Hints

```python
from typing import Callable, Awaitable, TypeVar, Generic

T = TypeVar('T')
R = TypeVar('R')

def with_cache(
    func: Callable[[T], Awaitable[R]], 
    cache_ttl: int = 300
) -> Callable[[T], Awaitable[R]]:
    """Type-safe caching decorator."""
    # Implementation...
```

### 3. Handle Errors Gracefully

```python
def with_fallback(primary_func, fallback_func):
    """Execute fallback function if primary fails."""
    
    async def fallback_execute(args, context):
        try:
            return await primary_func(args, context)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}, using fallback")
            return await fallback_func(args, context)
    
    return fallback_execute
```

### 4. Make Composition Explicit

```python
# Good: Clear composition
enhanced_tool = create_function_tool({
    'name': 'robust_search',
    'execute': with_logging(
        with_fallback(
            with_retry(primary_search),
            fallback_search
        )
    ),
    'metadata': {'composition': ['logging', 'fallback', 'retry']},
    'source': ToolSource.NATIVE
})
```

## Benefits

1. **Reusability**: Write cross-cutting concerns once, apply everywhere
2. **Testability**: Test each function in isolation
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Mix and match behaviors as needed
5. **Type Safety**: Full type checking with composition
6. **Performance**: Optimize individual pieces independently

Function composition in JAF enables you to build sophisticated, maintainable systems from simple, reusable building blocks while maintaining the functional programming principles that make JAF robust and predictable.