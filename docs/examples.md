# Examples and Tutorials

This guide provides comprehensive walkthroughs of JAF example applications, demonstrating real-world usage patterns and best practices for building AI agent systems.

## Overview

JAF includes several example applications that showcase different aspects of the framework:

1. **Server Demo** - Multi-agent HTTP server with tools and memory
2. **RAG Example** - Retrieval-Augmented Generation with knowledge base
3. **Iterative Search Agent** - Advanced callback system showcase with ReAct patterns
4. **Custom Tools** - Advanced tool implementation patterns
5. **Memory Integration** - Persistent conversation examples

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
@function_tool
async def calculator(expression: str, context=None) -> str:
    """Safely evaluate mathematical expressions with input sanitization.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3", "15 * 7")
    """
    # Input sanitization - only allow safe characters
    sanitized = ''.join(c for c in expression if c in '0123456789+-*/(). ')
    if sanitized != expression:
        return f"Error: Invalid characters in expression. Only numbers, +, -, *, /, and () are allowed."
    
    try:
        expression_for_eval = sanitized.replace(' ', '')
        result = eval(expression_for_eval)  # Safe due to sanitization
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: Failed to evaluate expression: {str(e)}"
```

**Greeting Tool with Validation**:
```python
@function_tool
async def greeting(name: str, context=None) -> str:
    """Generate a personalized greeting with input validation.
    
    Args:
        name: Name of the person to greet
    """
    # Input validation
    if not name or name.strip() == "":
        return "Error: Name cannot be empty"
    
    # Length validation
    if len(name) > 100:
        return f"Error: Name is too long (max 100 characters, got {len(name)})"
    
    greeting = f"Hello, {name.strip()}! Nice to meet you. I'm a helpful AI assistant running on the JAF framework."
    return greeting
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
        tools=[calculator]  # Only calculator access
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
        tools=[greeting]  # Only greeting access
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
        tools=[calculator, greeting]  # Access to all tools
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
pip install jaf-py litellm redis asyncpg

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
 Starting JAF Development Server...

📡 LiteLLM URL: http://localhost:4000
🔑 API Key: Set
⚠️  Note: Chat endpoints will fail without a running LiteLLM server

 Memory Type: memory
 Memory provider created: InMemoryMemoryProvider
 Creating server...

 Try these example requests:

1. Health Check:
   curl http://localhost:3000/health

2. List Agents:
   curl http://localhost:3000/agents

3. Chat with Math Tutor:
   curl -X POST http://localhost:3000/chat \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"What is 15 * 7?"}],"agent_name":"MathTutor","context":{"userId":"demo","permissions":["user"]}}'

 Starting server...
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
@function_tool
async def litellm_rag_search(query: str, max_results: int = 3, context=None) -> str:
    """Search the knowledge base and format retrieved information for LLM processing.
    
    Args:
        query: Search query for the knowledge base
        max_results: Maximum number of documents to retrieve (default: 3)
    """
    # Step 1: Retrieve relevant documents using semantic search
    relevant_docs = _semantic_search(query, max_results)
    
    if not relevant_docs:
        return f"I couldn't find any relevant information in the knowledge base for your query: '{query}'"
    
    # Step 2: Format the retrieved information
    formatted_response = _format_retrieved_docs(relevant_docs, query)
    
    # Include source information
    sources = [f"[{doc['title']}] - {doc['metadata']['category']}" for doc in relevant_docs]
    source_info = "\n\nSources: " + ", ".join(sources)
    
    return formatted_response + source_info
```

#### 3. Semantic Search Algorithm

```python
def _semantic_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Perform semantic search on the knowledge base using keyword matching and scoring."""
    query_lower = query.lower()
    scored_docs = []
    
    for doc in knowledge_base:  # Access global knowledge_base
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

def _format_retrieved_docs(docs: List[Dict[str, Any]], query: str) -> str:
    """Format retrieved documents for presentation."""
    if not docs:
        return "No relevant documents found."
    
    formatted_sections = []
    for i, doc in enumerate(docs, 1):
        section = f"**{i}. {doc['title']}**\n{doc['content'][:300]}..."
        if doc['metadata']:
            section += f"\n*Category: {doc['metadata']['category']}*"
        formatted_sections.append(section)
    
    return "\n\n".join(formatted_sections)
```

#### 4. RAG Agent Configuration

```python
def create_litellm_rag_agent() -> Agent:
    def rag_instructions(state: RunState) -> str:
        return """You are a knowledgeable AI assistant with access to a specialized knowledge base through the LiteLLM proxy.

When users ask questions, you should:
1. Use the litellm_rag_search tool to search for relevant information in the knowledge base
2. Provide comprehensive answers based on the retrieved information
3. Always cite your sources when providing information from the knowledge base
4. Be specific and detailed in your responses
5. If the knowledge base doesn't contain relevant information, be honest about the limitations

You have access to information about programming, machine learning, web development, data science, AI frameworks, and LiteLLM proxy configuration."""
    
    return Agent(
        name="litellm_rag_assistant",
        instructions=rag_instructions,
        tools=[litellm_rag_search]
    )
```

### Running the RAG Example

#### 1. Setup

```bash
# Install dependencies
pip install jaf-py litellm python-dotenv

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

 Demo Question 1: What is Python and why is it popular for programming?
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
 Searching knowledge base and generating response...
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
import math
import re
import ast
import operator

# Safe mathematical functions registry
SAFE_MATH_FUNCTIONS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
    'abs': abs, 'round': round, 'max': max, 'min': min,
    'pi': math.pi, 'e': math.e
}

@function_tool
async def advanced_calculator(expression: str, precision: int = 6, context=None) -> str:
    """Perform advanced mathematical calculations including trigonometry and scientific functions.
    
    Args:
        expression: Mathematical expression with functions like sin(x), sqrt(x), log(x)
        precision: Number of decimal places for result formatting (default: 6)
    """
    try:
        # Input validation
        if not expression or len(expression.strip()) == 0:
            return "Error: Expression cannot be empty"
        
        if len(expression) > 500:
            return f"Error: Expression too long (max 500 characters, got {len(expression)})"
        
        # Security validation - only allow safe mathematical characters and functions
        allowed_pattern = r'^[0-9+\-*/().a-z_\s]+$'
        if not re.match(allowed_pattern, expression.lower()):
            return "Error: Expression contains invalid characters. Only numbers, operators, parentheses, and mathematical functions are allowed."
        
        # Parse expression safely using AST
        try:
            tree = ast.parse(expression, mode='eval')
            result = _safe_eval_advanced(tree.body, precision)
        except SyntaxError:
            return f"Error: Invalid mathematical syntax in expression: {expression}"
        except ValueError as e:
            return f"Error: {str(e)}"
        
        # Format result with specified precision
        if isinstance(result, float):
            result = round(result, precision)
            # Remove trailing zeros for cleaner display
            if result == int(result):
                result = int(result)
        
        # Extract used functions for informational purposes
        used_functions = _extract_functions_from_expression(expression)
        function_info = f" (using: {', '.join(used_functions)})" if used_functions else ""
        
        return f"Result: {expression} = {result}{function_info}"
        
    except Exception as e:
        return f"Error: Advanced calculation failed: {str(e)}"

def _safe_eval_advanced(node, precision: int):
    """Safely evaluate AST node with advanced mathematical functions."""
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return node.n
    elif isinstance(node, ast.Name):
        # Handle mathematical constants
        if node.id in SAFE_MATH_FUNCTIONS:
            return SAFE_MATH_FUNCTIONS[node.id]
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in safe_operators:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        left = _safe_eval_advanced(node.left, precision)
        right = _safe_eval_advanced(node.right, precision)
        return safe_operators[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in safe_operators:
            raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
        operand = _safe_eval_advanced(node.operand, precision)
        return safe_operators[type(node.op)](operand)
    elif isinstance(node, ast.Call):
        # Handle function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in SAFE_MATH_FUNCTIONS:
                func = SAFE_MATH_FUNCTIONS[func_name]
                args = [_safe_eval_advanced(arg, precision) for arg in node.args]
                try:
                    return func(*args)
                except Exception as e:
                    raise ValueError(f"Error in function {func_name}: {str(e)}")
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError("Complex function calls not supported")
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

def _extract_functions_from_expression(expression: str) -> list:
    """Extract mathematical function names from expression."""
    functions_used = []
    for func_name in SAFE_MATH_FUNCTIONS:
        if isinstance(SAFE_MATH_FUNCTIONS[func_name], type(math.sin)):  # Callable functions only
            if func_name + '(' in expression:
                functions_used.append(func_name)
    return functions_used
```

### Database Query Tool

```python
import asyncpg
from typing import Dict, List, Any, Optional

# Global configuration for database security
ALLOWED_TABLES = {'users', 'products', 'orders', 'analytics'}
ALLOWED_COLUMNS = {
    'users': ['id', 'name', 'email', 'created_at'],
    'products': ['id', 'name', 'price', 'category'],
    'orders': ['id', 'user_id', 'product_id', 'quantity', 'total'],
    'analytics': ['id', 'metric_name', 'value', 'timestamp']
}

@function_tool
async def database_query(
    table: str,
    columns: str = "*",
    where_clause: str = "",
    limit: int = 100,
    context=None
) -> str:
    """Execute safe database queries with prepared statements and access control.
    
    Args:
        table: Table name to query (must be in allowed list)
        columns: Comma-separated column names or "*" for all (default: "*")
        where_clause: SQL WHERE conditions using safe syntax (optional)
        limit: Maximum number of rows to return (default: 100, max: 1000)
    """
    try:
        # Permission validation
        if not context or not hasattr(context, 'permissions'):
            return "Error: Context with permissions required"
        
        if 'database_read' not in context.permissions:
            return "Error: Database read permission required"
        
        # Table validation
        if table not in ALLOWED_TABLES:
            available_tables = ', '.join(sorted(ALLOWED_TABLES))
            return f"Error: Table '{table}' not accessible. Available tables: {available_tables}"
        
        # Limit validation
        if limit > 1000:
            return "Error: Limit cannot exceed 1000 rows"
        if limit < 1:
            return "Error: Limit must be at least 1"
        
        # Column validation
        if columns != "*":
            requested_columns = [col.strip() for col in columns.split(',')]
            allowed_for_table = ALLOWED_COLUMNS.get(table, [])
            invalid_columns = [col for col in requested_columns if col not in allowed_for_table]
            if invalid_columns:
                available_cols = ', '.join(sorted(allowed_for_table))
                return f"Error: Invalid columns {invalid_columns} for table '{table}'. Available: {available_cols}"
            columns_sql = ', '.join(requested_columns)
        else:
            columns_sql = ', '.join(ALLOWED_COLUMNS.get(table, ['*']))
        
        # Build safe query with parameterized statements
        query_parts = [f"SELECT {columns_sql} FROM {table}"]
        params = []
        
        if where_clause:
            # Basic WHERE clause validation (prevent SQL injection)
            if _validate_where_clause(where_clause):
                query_parts.append(f"WHERE {where_clause}")
            else:
                return "Error: Invalid WHERE clause. Use simple conditions like 'column = value' or 'column > value'"
        
        query_parts.append(f"LIMIT ${len(params) + 1}")
        params.append(limit)
        
        final_query = ' '.join(query_parts)
        
        # Get database connection (in real implementation, this would come from context)
        connection_pool = getattr(context, 'db_pool', None)
        if not connection_pool:
            return "Error: Database connection not available in context"
        
        # Execute query safely
        async with connection_pool.acquire() as conn:
            rows = await conn.fetch(final_query, *params)
            results = [dict(row) for row in rows]
            
            # Format response
            if not results:
                return f"No records found in table '{table}'"
            
            # Create formatted response
            response_lines = [
                f"Database Query Results:",
                f"Table: {table}",
                f"Columns: {columns}",
                f"Records found: {len(results)}",
                f"Limit applied: {limit}",
                ""
            ]
            
            # Add sample of results (first 5 rows)
            if results:
                response_lines.append("Sample Results:")
                for i, row in enumerate(results[:5], 1):
                    row_str = ', '.join([f"{k}: {v}" for k, v in row.items()])
                    response_lines.append(f"  {i}. {row_str}")
                
                if len(results) > 5:
                    response_lines.append(f"  ... and {len(results) - 5} more records")
            
            return '\n'.join(response_lines)
            
    except asyncpg.PostgresError as e:
        return f"Error: Database error: {str(e)}"
    except Exception as e:
        return f"Error: Query execution failed: {str(e)}"

def _validate_where_clause(where_clause: str) -> bool:
    """Validate WHERE clause for basic SQL injection protection."""
    if not where_clause:
        return True
    
    # Convert to lowercase for analysis
    clause_lower = where_clause.lower()
    
    # Block dangerous SQL keywords
    dangerous_keywords = [
        'drop', 'delete', 'insert', 'update', 'create', 'alter',
        'exec', 'execute', 'union', 'select', 'script', '--', ';'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in clause_lower:
            return False
    
    # Only allow basic comparison operators and logical operators
    allowed_operators = ['=', '>', '<', '>=', '<=', '!=', 'and', 'or', 'like', 'in', 'between']
    
    # Simple validation - this is basic and should be enhanced for production
    # In production, use proper SQL parsing or ORM query builders
    return True

# Usage example for creating database-enabled agent
def create_database_agent(db_pool) -> Agent:
    """Create an agent with database query capabilities."""
    def instructions(state):
        return """You are a data analyst assistant with access to a company database.
        
You can query the following tables:
- users: Customer information (id, name, email, created_at)
- products: Product catalog (id, name, price, category)  
- orders: Order history (id, user_id, product_id, quantity, total)
- analytics: Business metrics (id, metric_name, value, timestamp)

Always use the database_query tool to retrieve information. Be specific about which columns you need and apply appropriate filters and limits."""
    
    return Agent(
        name="DatabaseAnalyst",
        instructions=instructions,
        tools=[database_query]
    )
```

### HTTP API Tool

```python
import httpx
import time
from urllib.parse import urlparse
from collections import defaultdict
from typing import Dict, List, Optional, Any

# Global configuration for API security
ALLOWED_DOMAINS = [
    'api.github.com',
    'jsonplaceholder.typicode.com',
    'httpbin.org',
    'api.openweathermap.org'
]

# Simple rate limiter implementation
class SimpleRateLimiter:
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str = "default") -> bool:
        now = time.time()
        # Clean old requests
        cutoff = now - self.time_window
        self.requests[identifier] = [req_time for req_time in self.requests[identifier] if req_time > cutoff]
        
        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Record this request
        self.requests[identifier].append(now)
        return True

# Global rate limiter instance
_rate_limiter = SimpleRateLimiter(max_requests=10, time_window=60)

@function_tool
async def http_api_request(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    data: Optional[str] = None,
    timeout: int = 30,
    context=None
) -> str:
    """Make HTTP API requests with security controls and rate limiting.
    
    Args:
        url: Target URL (must be from allowed domains)
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: JSON string of headers (optional)
        data: JSON string of request body data (optional)
        timeout: Request timeout in seconds (default: 30, max: 60)
    """
    try:
        # Input validation
        if not url or not url.startswith(('http://', 'https://')):
            return "Error: Invalid URL. Must start with http:// or https://"
        
        # Rate limiting
        user_id = getattr(context, 'user_id', 'anonymous') if context else 'anonymous'
        if not _rate_limiter.is_allowed(user_id):
            return "Error: Rate limit exceeded. Please wait before making another request."
        
        # Domain validation
        parsed_url = urlparse(url)
        if parsed_url.hostname not in ALLOWED_DOMAINS:
            allowed_list = ', '.join(ALLOWED_DOMAINS)
            return f"Error: Domain '{parsed_url.hostname}' not in allowed list. Allowed domains: {allowed_list}"
        
        # Method validation
        allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        method = method.upper()
        if method not in allowed_methods:
            return f"Error: HTTP method '{method}' not allowed. Use: {', '.join(allowed_methods)}"
        
        # Timeout validation
        if timeout > 60:
            timeout = 60
        if timeout < 1:
            timeout = 1
        
        # Parse headers
        parsed_headers = {}
        if headers:
            try:
                import json
                parsed_headers = json.loads(headers)
                if not isinstance(parsed_headers, dict):
                    return "Error: Headers must be a JSON object"
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for headers"
        
        # Parse data
        parsed_data = None
        if data:
            try:
                import json
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for request data"
        
        # Add default headers
        final_headers = {
            'User-Agent': 'JAF-HTTP-Tool/1.0',
            **parsed_headers
        }
        
        # Make HTTP request
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_kwargs = {
                'method': method,
                'url': url,
                'headers': final_headers
            }
            
            if parsed_data and method in ['POST', 'PUT', 'PATCH']:
                request_kwargs['json'] = parsed_data
            
            response = await client.request(**request_kwargs)
            
            # Process response
            response_info = [
                f"HTTP {method} Request to {url}",
                f"Status: {response.status_code} {response.reason_phrase}",
                f"Response Time: {response.elapsed.total_seconds():.2f}s" if hasattr(response, 'elapsed') else "",
                ""
            ]
            
            # Add response headers (filtered)
            important_headers = ['content-type', 'content-length', 'server', 'date']
            response_info.append("Response Headers:")
            for header in important_headers:
                if header in response.headers:
                    response_info.append(f"  {header}: {response.headers[header]}")
            response_info.append("")
            
            # Process response body
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                try:
                    json_data = response.json()
                    # Limit JSON response size for readability
                    json_str = str(json_data)
                    if len(json_str) > 2000:
                        response_info.append("JSON Response (truncated):")
                        response_info.append(json_str[:2000] + "...")
                    else:
                        response_info.append("JSON Response:")
                        response_info.append(json_str)
                except Exception:
                    response_info.append("Response Body (invalid JSON):")
                    response_info.append(response.text[:1000])
            elif 'text/' in content_type:
                response_info.append("Text Response:")
                text_content = response.text
                if len(text_content) > 1500:
                    response_info.append(text_content[:1500] + "...")
                else:
                    response_info.append(text_content)
            else:
                response_info.append(f"Binary Response ({len(response.content)} bytes)")
                response_info.append("Content type: " + content_type)
            
            # Add status assessment
            if 200 <= response.status_code < 300:
                response_info.insert(1, "✅ Request successful")
            elif 400 <= response.status_code < 500:
                response_info.insert(1, "⚠️ Client error")
            elif 500 <= response.status_code < 600:
                response_info.insert(1, "❌ Server error")
            
            return '\n'.join(response_info)
            
    except httpx.TimeoutException:
        return f"Error: Request to {url} timed out after {timeout} seconds"
    except httpx.ConnectError:
        return f"Error: Could not connect to {url}. Check if the server is running."
    except httpx.HTTPError as e:
        return f"Error: HTTP error occurred: {str(e)}"
    except Exception as e:
        return f"Error: Request failed: {str(e)}"

# Usage example for creating API-enabled agent
def create_api_agent() -> Agent:
    """Create an agent with HTTP API capabilities."""
    def instructions(state):
        return f"""You are an API integration assistant that can make HTTP requests to external services.
        
Available domains for API calls:
{chr(10).join([f'- {domain}' for domain in ALLOWED_DOMAINS])}

You can use GET requests to fetch data and POST/PUT requests to send data.
Always explain what API you're calling and what data you're requesting or sending.

Rate limit: 10 requests per minute per user.
Timeout: Maximum 60 seconds per request.

Use the http_api_request tool for all external API calls."""
    
    return Agent(
        name="APIAssistant",
        instructions=instructions,
        tools=[http_api_request]
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

## Iterative Search Agent - Callback System Showcase

The iterative search agent (`examples/iterative_search_agent.py`) demonstrates the full power of JAF's advanced callback system by implementing a sophisticated ReAct-style agent that can iteratively gather information, check for synthesis completion, and provide comprehensive answers.

### Key Features Demonstrated

This example showcases how the callback system enables complex agent behaviors:

- **🔄 Iterative Information Gathering** - Agent searches across multiple iterations
- ** Synthesis Checking** - Automatically determines when enough information is gathered
- ** Dynamic Query Refinement** - Refines search queries based on previous results
- **🚫 Loop Detection** - Prevents repetitive searches
- ** Context Management** - Intelligently accumulates and filters information
- ** Performance Monitoring** - Tracks metrics and execution statistics

### Architecture Overview

```python
from adk.runners import RunnerConfig, execute_agent

# Comprehensive callback implementation
class IterativeSearchCallbacks:
    async def on_start(self, context, message, session_state):
        """Initialize tracking for iterative search."""
        self.original_query = message.content
        print(f" Starting search for: '{self.original_query}'")
    
    async def on_check_synthesis(self, session_state, context_data):
        """Determine if enough information has been gathered."""
        if len(context_data) >= self.synthesis_threshold:
            confidence = self._calculate_confidence(context_data)
            if confidence >= 0.75:
                return {
                    'complete': True,
                    'answer': self._generate_synthesis_prompt(context_data),
                    'confidence': confidence
                }
        return None
    
    async def on_query_rewrite(self, original_query, context_data):
        """Refine queries based on accumulated context."""
        gaps = self._identify_knowledge_gaps(context_data)
        if gaps:
            return f"{original_query} focusing on {', '.join(gaps)}"
        return None
    
    async def on_loop_detection(self, tool_history, current_tool):
        """Prevent repetitive searches."""
        recent_queries = [item['query'] for item in tool_history[-3:]]
        return self._detect_similarity(recent_queries) > 0.7

# Configure agent with callbacks
config = RunnerConfig(
    agent=search_agent,
    callbacks=IterativeSearchCallbacks(max_iterations=5, synthesis_threshold=4),
    enable_context_accumulation=True,
    enable_loop_detection=True
)

# Execute with full instrumentation
result = await execute_agent(config, session_state, message, context, model_provider)
```

### Example Execution Flow

When you run the iterative search agent, you'll see output like this:

```
 ITERATIVE SEARCH AGENT DEMONSTRATION
============================================================
 Starting iterative search for: 'What are the applications of machine learning?'

🔄 ITERATION 1/4
 Executing search: 'machine learning applications in different industries'
 Adding 3 new context items...
   Total context items: 3

🔄 ITERATION 2/4
 Query refined: 'machine learning applications in finance and trading'
 Executing search: 'machine learning applications in finance and trading'
 Adding 2 new context items...
   Total context items: 5

🧮 Evaluating synthesis readiness with 5 context items...
   Coverage: 0.85
   Quality: 0.90
   Completeness: 0.50
   Overall confidence: 0.75

 Synthesis complete! Confidence: 0.85

 ITERATIVE SEARCH COMPLETED
   Total iterations: 2
   Context items gathered: 5
   Searches performed: 2
   Final confidence: 0.85
   Execution time: 1247ms
============================================================
```

### Key Implementation Patterns

#### 1. Context Accumulation

```python
async def on_context_update(self, current_context, new_items):
    """Manage context with deduplication and relevance filtering."""
    # Deduplicate based on content similarity
    filtered_items = self._deduplicate_and_filter(new_items)
    
    # Merge and sort by relevance
    self.context_accumulator.extend(filtered_items)
    self.context_accumulator.sort(key=lambda x: x.get('relevance', 0), reverse=True)
    
    # Keep top items within limits
    return self.context_accumulator[:20]
```

#### 2. Intelligent Query Refinement

```python
async def on_query_rewrite(self, original_query, context_data):
    """Analyze context gaps and refine search queries."""
    topics_covered = self._analyze_topic_coverage(context_data)
    
    if 'healthcare' in topics_covered and 'finance' not in topics_covered:
        return f"{original_query} applications in finance and trading"
    elif len(topics_covered) >= 2:
        return f"{original_query} future trends and emerging applications"
    
    return None
```

#### 3. Synthesis Quality Assessment

```python
async def on_check_synthesis(self, session_state, context_data):
    """Multi-factor synthesis readiness assessment."""
    coverage_score = self._analyze_coverage(context_data)
    quality_score = self._analyze_quality(context_data)
    completeness_score = min(len(context_data) / 10.0, 1.0)
    
    confidence = (coverage_score + quality_score + completeness_score) / 3.0
    
    if confidence >= 0.75:
        return {
            'complete': True,
            'answer': self._create_comprehensive_synthesis(context_data),
            'confidence': confidence
        }
    return None
```

### Running the Example

#### 1. Basic Execution

```bash
python examples/iterative_search_agent.py
```

#### 2. Custom Configuration

```python
from examples.iterative_search_agent import IterativeSearchCallbacks

# Configure for different behavior
callbacks = IterativeSearchCallbacks(
    max_iterations=10,        # More thorough search
    synthesis_threshold=8     # Require more information
)

config = RunnerConfig(
    agent=search_agent,
    callbacks=callbacks,
    enable_context_accumulation=True,
    max_context_items=50      # Larger context window
)
```

### Advanced Patterns Demonstrated

#### ReAct (Reasoning + Acting) Pattern

The example implements a full ReAct pattern where the agent:

1. **Reasons** about what information it needs
2. **Acts** by searching for that information  
3. **Observes** the results and their relevance
4. **Reasons** about gaps and next steps
5. **Repeats** until synthesis is complete

#### Dynamic Behavior Adaptation

```python
class AdaptiveCallbacks(IterativeSearchCallbacks):
    async def on_iteration_complete(self, iteration, has_tool_calls):
        """Adapt behavior based on progress."""
        if not has_tool_calls:
            # No tools called, likely finished
            return {'should_stop': True}
        
        if self._making_progress():
            # Continue if making good progress
            return {'should_continue': True}
        else:
            # Try different approach
            return {'should_stop': True}
```

#### Performance Monitoring

```python
async def on_complete(self, response):
    """Comprehensive execution analytics."""
    print(f" Performance Metrics:")
    print(f"   Iterations: {self.iteration_count}")
    print(f"   Context Quality: {self.final_quality_score:.2f}")
    print(f"   Search Efficiency: {len(self.context_accumulator)/self.iteration_count:.1f} items/iteration")
    print(f"   Synthesis Confidence: {self.synthesis_confidence:.2f}")
```

This example demonstrates how the callback system transforms JAF from a simple agent executor into a sophisticated reasoning engine capable of complex, adaptive behaviors that would be impossible with traditional fixed execution patterns.

## Next Steps

- Learn about [Deployment](deployment.md) for production setup
- Review [Troubleshooting](troubleshooting.md) for common issues
- Explore [API Reference](api-reference.md) for complete documentation
- Check [Tools Guide](tools.md) for advanced tool patterns
- **[Callback System](callback-system.md)** - Deep dive into advanced agent instrumentation