# Examples and Tutorials

This guide provides comprehensive walkthroughs of JAF example applications, demonstrating real-world usage patterns and best practices for building AI agent systems.

## Overview

JAF includes several example applications that showcase different aspects of the framework:

1. **Server Demo** - Multi-agent HTTP server with tools and memory
2. **RAG Example** - Retrieval-Augmented Generation with knowledge base
3. **Custom Tools** - Advanced tool implementation patterns
4. **Memory Integration** - Persistent conversation examples

## Server Demo Walkthrough

The server demo (`examples/server_demo.py`) demonstrates a complete production-ready JAF server with multiple specialized agents, custom tools, and memory persistence.

### Architecture Overview

```python
# Three specialized agents
MathTutor    # Mathematical calculations and explanations
ChatBot      # Friendly conversation and greetings  
Assistant    # General-purpose with all tools

# Two custom tools
Calculator   # Safe mathematical expression evaluation
Greeting     # Personalized greeting generation

# Memory support
InMemory     # Development
Redis        # Production caching
PostgreSQL   # Production persistence
```

### Key Components

#### 1. Context Definition

```python
@dataclass
class MyContext:
    user_id: str
    permissions: list[str]
```

The context provides user information and permissions to agents and tools, enabling security and personalization.

#### 2. Tool Implementation

**Calculator Tool with Security**:
```python
class CalculatorTool:
    async def execute(self, args: CalculateArgs, context: MyContext) -> Any:
        # Input sanitization - only allow safe characters
        sanitized = ''.join(c for c in args.expression if c in '0123456789+-*/(). ')
        if sanitized != args.expression:
            return ToolResponse.validation_error(
                "Invalid characters in expression. Only numbers, +, -, *, /, and () are allowed.",
                {'original_expression': args.expression, 'sanitized_expression': sanitized}
            )
        
        try:
            expression_for_eval = sanitized.replace(' ', '')
            result = eval(expression_for_eval)  # Safe due to sanitization
            return ToolResponse.success(
                f"{args.expression} = {result}",
                {'original_expression': args.expression, 'result': result}
            )
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Failed to evaluate expression: {str(e)}"
            )
```

**Greeting Tool with Validation**:
```python
class GreetingTool:
    async def execute(self, args: GreetArgs, context: MyContext) -> Any:
        # Input validation
        if not args.name or args.name.strip() == "":
            return ToolResponse.validation_error(
                "Name cannot be empty",
                {'provided_name': args.name}
            )
        
        # Length validation
        if len(args.name) > 100:
            return ToolResponse.validation_error(
                "Name is too long (max 100 characters)",
                {'name_length': len(args.name), 'max_length': 100}
            )
        
        greeting = f"Hello, {args.name.strip()}! Nice to meet you. I'm a helpful AI assistant running on the JAF framework."
        
        return ToolResponse.success(
            greeting,
            {'greeted_name': args.name.strip(), 'greeting_type': 'personal'}
        )
```

#### 3. Agent Specialization

**Math Tutor Agent**:
```python
def create_math_agent() -> Agent[MyContext, str]:
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a helpful math tutor. Use the calculator tool to perform calculations and explain math concepts clearly.'
    
    return Agent(
        name='MathTutor',
        instructions=instructions,
        tools=[calculator_tool]  # Only calculator access
    )
```

**Chat Bot Agent**:
```python
def create_chat_agent() -> Agent[MyContext, str]:
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a friendly chatbot. Use the greeting tool when meeting new people, and engage in helpful conversation.'
    
    return Agent(
        name='ChatBot',
        instructions=instructions,
        tools=[greeting_tool]  # Only greeting access
    )
```

**General Assistant Agent**:
```python
def create_assistant_agent() -> Agent[MyContext, str]:
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a general-purpose assistant. You can help with math calculations and provide greetings.'
    
    return Agent(
        name='Assistant',
        instructions=instructions,
        tools=[calculator_tool, greeting_tool]  # Access to all tools
    )
```

#### 4. Memory Integration

```python
# Environment-based memory configuration
memory_type = os.getenv("JAF_MEMORY_TYPE", "memory").lower()

if memory_type == "redis":
    # Redis configuration
    redis_client = redis.Redis(
        host=os.getenv("JAF_REDIS_HOST", "localhost"),
        port=int(os.getenv("JAF_REDIS_PORT", "6379")),
        password=os.getenv("JAF_REDIS_PASSWORD"),
        db=int(os.getenv("JAF_REDIS_DB", "0"))
    )
    external_clients["redis"] = redis_client

elif memory_type == "postgres":
    # PostgreSQL configuration
    postgres_client = await asyncpg.connect(
        host=os.getenv("JAF_POSTGRES_HOST", "localhost"),
        port=int(os.getenv("JAF_POSTGRES_PORT", "5432")),
        database=os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
        user=os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
        password=os.getenv("JAF_POSTGRES_PASSWORD")
    )
    external_clients["postgres"] = postgres_client

# Create memory provider
memory_provider = await create_memory_provider_from_env(external_clients)
memory_config = MemoryConfig(
    provider=memory_provider,
    auto_store=True,
    max_messages=1000
)
```

### Running the Server Demo

#### 1. Basic Setup

```bash
# Install dependencies
pip install jaf-python litellm redis asyncpg

# Set environment variables
export LITELLM_URL=http://localhost:4000
export LITELLM_API_KEY=your-api-key
export LITELLM_MODEL=gemini-2.5-pro
export PORT=3000

# Optional: Memory configuration
export JAF_MEMORY_TYPE=memory  # or redis, postgres
```

#### 2. Run the Server

```bash
python examples/server_demo.py
```

**Expected Output**:
```
🚀 Starting JAF Development Server...

📡 LiteLLM URL: http://localhost:4000
🔑 API Key: Set
⚠️  Note: Chat endpoints will fail without a running LiteLLM server

🧠 Memory Type: memory
✅ Memory provider created: InMemoryMemoryProvider
🔧 Creating server...

📚 Try these example requests:

1. Health Check:
   curl http://localhost:3000/health

2. List Agents:
   curl http://localhost:3000/agents

3. Chat with Math Tutor:
   curl -X POST http://localhost:3000/chat \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"What is 15 * 7?"}],"agent_name":"MathTutor","context":{"userId":"demo","permissions":["user"]}}'

🚀 Starting server...
```

#### 3. Test the Agents

**Math Calculations**:
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 15 * 7?"}],
    "agent_name": "MathTutor",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

**Friendly Greetings**:
```bash
curl -X POST http://localhost:3000/agents/ChatBot/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, my name is Alice"}],
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

**Multi-Tool Assistant**:
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Calculate 25 + 17 and then greet me as Bob"}],
    "agent_name": "Assistant",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

#### 4. Persistent Conversations

```bash
# Start a conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, I am starting a new conversation"}],
    "agent_name": "ChatBot",
    "conversation_id": "my-conversation",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'

# Continue the conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Do you remember me?"}],
    "agent_name": "ChatBot",
    "conversation_id": "my-conversation",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'

# Get conversation history
curl http://localhost:3000/conversations/my-conversation
```

## RAG Example Walkthrough

The RAG example (`examples/server_example.py`) demonstrates Retrieval-Augmented Generation with a knowledge base and LiteLLM integration.

### Architecture Overview

```python
# RAG Components
Knowledge Base  # Mock documents with metadata
Semantic Search # Keyword-based retrieval
LiteLLM Agent  # Gemini-powered responses
RAG Tool       # Integration layer
```

### Key Components

#### 1. Knowledge Base Structure

```python
knowledge_base = [
    {
        "id": "doc1",
        "title": "Python Programming Basics",
        "content": "Python is a high-level, interpreted programming language...",
        "metadata": {"category": "programming", "level": "beginner"}
    },
    {
        "id": "doc2",
        "title": "Machine Learning with Python", 
        "content": "Python is the leading language for machine learning...",
        "metadata": {"category": "ml", "level": "intermediate"}
    }
    # ... more documents
]
```

#### 2. RAG Tool Implementation

```python
class LiteLLMRAGTool:
    async def execute(self, args: RAGQueryArgs, context: Any) -> ToolResult:
        # Step 1: Retrieve relevant documents
        relevant_docs = self._semantic_search(args.query, args.max_results)
        
        if not relevant_docs:
            return ToolResponse.success(
                "I couldn't find any relevant information in the knowledge base for your query.",
                {"query": args.query, "results_count": 0}
            )
        
        # Step 2: Format the retrieved information
        formatted_response = self._format_retrieved_docs(relevant_docs, args.query)
        
        return ToolResponse.success(
            formatted_response,
            {
                "query": args.query,
                "results_count": len(relevant_docs),
                "sources": [{"title": doc["title"], "category": doc["metadata"]["category"]} for doc in relevant_docs]
            }
        )
```

#### 3. Semantic Search Algorithm

```python
def _semantic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    scored_docs = []
    
    for doc in self.knowledge_base:
        score = 0
        
        # Title and content matching
        title_matches = sum(1 for word in query_lower.split() if word in doc["title"].lower())
        content_matches = sum(1 for word in query_lower.split() if word in doc["content"].lower())
        
        # Category-specific keywords
        category_keywords = {
            "programming": ["python", "code", "programming", "language", "syntax"],
            "ml": ["machine learning", "ai", "model", "training", "neural"],
            "web": ["web", "api", "fastapi", "server", "http"],
            "ai": ["ai", "agent", "framework", "intelligent", "litellm", "gemini"]
        }
        
        category = doc["metadata"]["category"]
        if category in category_keywords:
            category_matches = sum(1 for keyword in category_keywords[category] if keyword in query_lower)
            score += category_matches * 1.5
        
        score += title_matches * 3 + content_matches
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by relevance score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_results]]
```

#### 4. RAG Agent Configuration

```python
def create_litellm_rag_agent() -> Agent:
    def rag_instructions(state: RunState) -> str:
        return """You are a knowledgeable AI assistant with access to a specialized knowledge base through the LiteLLM proxy.

When users ask questions, you should:
1. Use the litellm_rag tool to search for relevant information in the knowledge base
2. Provide comprehensive answers based on the retrieved information
3. Always cite your sources when providing information from the knowledge base
4. Be specific and detailed in your responses
5. If the knowledge base doesn't contain relevant information, be honest about the limitations

You have access to information about programming, machine learning, web development, data science, AI frameworks, and LiteLLM proxy configuration."""
    
    rag_tool = LiteLLMRAGTool()
    
    return Agent(
        name="litellm_rag_assistant",
        instructions=rag_instructions,
        tools=[rag_tool]
    )
```

### Running the RAG Example

#### 1. Setup

```bash
# Install dependencies
pip install jaf-python litellm python-dotenv

# Configure environment
export LITELLM_URL=http://localhost:4000
export LITELLM_API_KEY=your-gemini-api-key
export LITELLM_MODEL=gemini-2.5-pro
```

#### 2. Run the Example

```bash
python examples/server_example.py
```

#### 3. Demo Modes

**Automated Demo**:
```
Choose demo mode:
1. Automated demo with sample questions
2. Interactive chat
Enter 1 or 2: 1

🔍 Demo Question 1: What is Python and why is it popular for programming?
------------------------------------------------------------
🤖 Assistant: Based on the knowledge base information, Python is a high-level, interpreted programming language that has gained popularity for several key reasons:

**What Python Is:**
Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

**Why It's Popular:**
1. **Clean and Expressive Syntax** - Python's syntax is clean and expressive, making it accessible to both beginners and experienced developers
2. **Readability** - The language emphasizes code readability, which reduces development time and maintenance costs
3. **Versatility** - Python supports multiple programming paradigms, making it suitable for various types of projects

[Source 1] Python Programming Basics
Category: programming | Level: beginner
```

**Interactive Mode**:
```
Choose demo mode:
1. Automated demo with sample questions  
2. Interactive chat
Enter 1 or 2: 2

🤖 Interactive JAF LiteLLM RAG Demo
Type your questions and get answers from the knowledge base!
Type 'quit' or 'exit' to stop.

👤 You: How do I use Python for machine learning?
🔍 Searching knowledge base and generating response...
🤖 Assistant: Python is the leading language for machine learning due to its rich ecosystem of specialized libraries. Here's how you can use Python for ML:

**Key Libraries:**
1. **Scikit-learn** - Provides simple and efficient tools for data mining and analysis
2. **TensorFlow and PyTorch** - Enable deep learning and neural network development
3. **NumPy and Pandas** - Handle numerical computations and data manipulation

These libraries make Python an excellent choice for machine learning projects, from basic data analysis to advanced deep learning applications.

[Source 1] Machine Learning with Python
Category: ml | Level: intermediate
```

## Custom Tool Examples

### Advanced Calculator Tool

```python
class AdvancedCalculatorTool:
    """Calculator with advanced mathematical functions."""
    
    def __init__(self):
        self.safe_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'abs': abs, 'round': round, 'max': max, 'min': min
        }
    
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'advanced_calculate',
            'description': 'Perform advanced mathematical calculations including trigonometry',
            'parameters': AdvancedCalculateArgs
        })()
    
    async def execute(self, args: AdvancedCalculateArgs, context) -> ToolResponse:
        try:
            # Parse and validate expression
            parsed_expr = self._parse_expression(args.expression)
            
            # Evaluate with safe functions
            result = self._safe_evaluate(parsed_expr)
            
            # Format result based on precision
            if args.precision:
                result = round(result, args.precision)
            
            return ToolResponse.success(
                f"{args.expression} = {result}",
                {
                    'expression': args.expression,
                    'result': result,
                    'precision': args.precision,
                    'functions_used': self._extract_functions(parsed_expr)
                }
            )
            
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Advanced calculation failed: {str(e)}"
            )
    
    def _parse_expression(self, expression: str) -> str:
        """Parse and validate mathematical expression."""
        # Remove spaces and validate characters
        expr = expression.replace(' ', '')
        
        # Allow mathematical operators, numbers, and safe functions
        allowed_pattern = r'^[0-9+\-*/().a-z_]+$'
        if not re.match(allowed_pattern, expr):
            raise ValueError("Expression contains invalid characters")
        
        # Replace function names with safe implementations
        for func_name in self.safe_functions:
            expr = expr.replace(func_name, f'self.safe_functions["{func_name}"]')
        
        return expr
```

### Database Query Tool

```python
class DatabaseQueryTool:
    """Safe database query tool with prepared statements."""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
        self.allowed_tables = {'users', 'products', 'orders', 'analytics'}
        self.allowed_columns = {
            'users': ['id', 'name', 'email', 'created_at'],
            'products': ['id', 'name', 'price', 'category'],
            'orders': ['id', 'user_id', 'product_id', 'quantity', 'total']
        }
    
    async def execute(self, args: DatabaseQueryArgs, context) -> ToolResponse:
        # Validate permissions
        if 'database_read' not in context.permissions:
            return ToolResponse.error(
                ToolErrorCodes.PERMISSION_DENIED,
                "Database read permission required"
            )
        
        # Validate table access
        if args.table not in self.allowed_tables:
            return ToolResponse.validation_error(
                f"Table '{args.table}' not accessible",
                {'allowed_tables': list(self.allowed_tables)}
            )
        
        try:
            async with self.pool.acquire() as conn:
                # Build safe parameterized query
                query, params = self._build_safe_query(args)
                
                # Execute query
                rows = await conn.fetch(query, *params)
                results = [dict(row) for row in rows]
                
                return ToolResponse.success(
                    f"Found {len(results)} records from {args.table}",
                    {
                        'table': args.table,
                        'count': len(results),
                        'results': results[:args.limit],  # Respect limit
                        'query_info': {
                            'filters': args.filters,
                            'limit': args.limit
                        }
                    }
                )
                
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Database query failed: {str(e)}"
            )
```

### HTTP API Tool

```python
class APIClientTool:
    """Tool for making HTTP API requests with rate limiting."""
    
    def __init__(self, rate_limiter=None):
        self.rate_limiter = rate_limiter or TokenBucket(rate=10, capacity=50)
        self.session = httpx.AsyncClient(timeout=30.0)
    
    async def execute(self, args: APIRequestArgs, context) -> ToolResponse:
        # Check rate limit
        if not self.rate_limiter.consume():
            return ToolResponse.error(
                ToolErrorCodes.RATE_LIMITED,
                "API rate limit exceeded",
                {'retry_after': 60}
            )
        
        # Validate URL
        if not self._is_allowed_url(args.url):
            return ToolResponse.validation_error(
                "URL not in allowed list",
                {'allowed_domains': self.allowed_domains}
            )
        
        try:
            # Prepare request
            request_kwargs = {
                'method': args.method,
                'url': args.url,
                'headers': args.headers or {},
                'timeout': args.timeout or 30.0
            }
            
            if args.data:
                request_kwargs['json'] = args.data
            
            # Make request
            response = await self.session.request(**request_kwargs)
            
            # Process response
            result_data = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'url': str(response.url)
            }
            
            # Parse response body
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                result_data['data'] = response.json()
            else:
                result_data['text'] = response.text[:1000]  # Limit response size
            
            return ToolResponse.success(
                f"API request completed with status {response.status_code}",
                result_data
            )
            
        except httpx.TimeoutException:
            return ToolResponse.error(
                ToolErrorCodes.TIMEOUT,
                "API request timed out"
            )
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"API request failed: {str(e)}"
            )
```

## Memory Integration Examples

### Redis Memory Example

```python
async def setup_redis_memory():
    """Configure Redis memory provider."""
    import redis.asyncio as redis
    
    # Create Redis client
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        password="your-password",
        db=0,
        decode_responses=False
    )
    
    # Test connection
    await redis_client.ping()
    
    # Create memory provider
    from jaf.memory import create_redis_provider, RedisConfig
    
    config = RedisConfig(
        host="localhost",
        port=6379,
        password="your-password",
        db=0,
        key_prefix="jaf:conversations:",
        ttl=86400  # 24 hours
    )
    
    provider = await create_redis_provider(config, redis_client)
    
    return MemoryConfig(
        provider=provider,
        auto_store=True,
        max_messages=1000
    )

# Usage in server
memory_config = await setup_redis_memory()
run_config = RunConfig(
    agent_registry=agents,
    model_provider=model_provider,
    memory=memory_config
)
```

### PostgreSQL Memory Example

```python
async def setup_postgres_memory():
    """Configure PostgreSQL memory provider."""
    import asyncpg
    
    # Create connection
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="jaf_memory",
        user="postgres",
        password="your-password"
    )
    
    # Create memory provider
    from jaf.memory import create_postgres_provider, PostgresConfig
    
    config = PostgresConfig(
        host="localhost",
        port=5432,
        database="jaf_memory",
        username="postgres",
        password="your-password",
        table_name="conversations"
    )
    
    provider = await create_postgres_provider(config, conn)
    
    return MemoryConfig(
        provider=provider,
        auto_store=True,
        max_messages=1000
    )

# Database schema setup
async def setup_postgres_schema(conn):
    """Set up PostgreSQL schema for JAF memory."""
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) UNIQUE NOT NULL,
            user_id VARCHAR(255),
            messages JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
        ON conversations(user_id);
        
        CREATE INDEX IF NOT EXISTS idx_conversations_created_at 
        ON conversations(created_at);
    ''')
```

## Best Practices from Examples

### 1. Security Patterns

```python
# Input sanitization
def sanitize_expression(expr: str) -> str:
    """Remove potentially dangerous characters."""
    allowed = set('0123456789+-*/(). ')
    return ''.join(c for c in expr if c in allowed)

# Permission checking
def check_permissions(context, required_permissions):
    """Verify user has required permissions."""
    user_permissions = set(context.permissions)
    required = set(required_permissions)
    return required.issubset(user_permissions)

# Rate limiting
class RateLimiter:
    def __init__(self, rate: int, window: int = 60):
        self.rate = rate
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        cutoff = now - self.window
        self.requests[user_id] = [req for req in user_requests if req > cutoff]
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.rate:
            return False
        
        self.requests[user_id].append(now)
        return True
```

### 2. Error Handling Patterns

```python
# Comprehensive error handling
async def safe_tool_execution(tool, args, context):
    """Execute tool with comprehensive error handling."""
    try:
        # Validate inputs
        if not hasattr(args, 'validate'):
            raise ValueError("Invalid arguments object")
        
        # Check permissions
        if not check_permissions(context, tool.required_permissions):
            return ToolResponse.error(
                ToolErrorCodes.PERMISSION_DENIED,
                "Insufficient permissions"
            )
        
        # Execute with timeout
        result = await asyncio.wait_for(
            tool.execute(args, context),
            timeout=30.0
        )
        
        return result
        
    except asyncio.TimeoutError:
        return ToolResponse.error(
            ToolErrorCodes.TIMEOUT,
            "Tool execution timed out"
        )
    except ValidationError as e:
        return ToolResponse.validation_error(
            str(e),
            {'validation_errors': e.errors()}
        )
    except Exception as e:
        logger.exception(f"Tool execution failed: {e}")
        return ToolResponse.error(
            ToolErrorCodes.EXECUTION_FAILED,
            "Internal tool error"
        )
```

### 3. Performance Patterns

```python
# Connection pooling
async def create_optimized_client():
    """Create HTTP client with connection pooling."""
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        ),
        timeout=httpx.Timeout(30.0)
    )

# Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input_data: str) -> str:
    """Cache expensive computations."""
    # Expensive operation here
    return result

# Batch processing
async def process_batch(items: List[Any], batch_size: int = 10):
    """Process items in batches to avoid overwhelming resources."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_item(item) for item in batch
        ])
        results.extend(batch_results)
    return results
```

## Next Steps

- Learn about [Deployment](deployment.md) for production setup
- Review [Troubleshooting](troubleshooting.md) for common issues
- Explore [API Reference](api-reference.md) for complete documentation
- Check [Tools Guide](tools.md) for advanced tool patterns