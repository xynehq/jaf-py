# A2A Protocol Examples

This guide provides comprehensive examples of using the A2A (Agent-to-Agent) protocol for building distributed agent systems. From simple client-server interactions to complex multi-agent coordination patterns.

## Quick Start Examples

### Basic Client Connection

```python
import asyncio
from jaf.a2a import connect_to_a2a_agent, send_message_to_agent

async def simple_client_example():
    """Connect to an A2A agent and send a message"""
    
    # Connect to A2A server
    client = await connect_to_a2a_agent("http://localhost:3000")
    
    # Send a simple message
    response = await send_message_to_agent(
        client,
        agent_name="MathTutor",
        message="What is 15 * 7?"
    )
    
    print(f"Agent response: {response}")

# Run the example
asyncio.run(simple_client_example())
```

### Basic Server Setup

```python
import asyncio
from jaf.a2a import (
    create_a2a_agent, create_a2a_tool, 
    create_server_config, start_a2a_server
)

def create_calculator_tool():
    """Create a safe calculator tool"""
    
    def calculate(expression: str) -> str:
        # Basic validation for safety
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return 'Error: Invalid characters in expression'
        
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error: {e}"
    
    return create_a2a_tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        execute_func=calculate
    )

async def basic_server_example():
    """Create and start a basic A2A server"""
    
    # Create calculator tool
    calc_tool = create_calculator_tool()
    
    # Create math tutor agent
    math_agent = create_a2a_agent(
        name="MathTutor",
        description="A helpful math tutor that can perform calculations",
        instruction="You are a math tutor. Use the calculate tool for math problems.",
        tools=[calc_tool]
    )
    
    # Create server configuration
    server_config = create_server_config(
        agents={"MathTutor": math_agent},
        name="Math Server",
        description="Server with math calculation capabilities",
        port=3000,
        cors=True
    )
    
    # Start the server
    print("Starting A2A server on http://localhost:3000")
    server = await start_a2a_server(server_config)
    
    # Server endpoints are automatically available:
    # GET  /.well-known/agent-card     # Agent discovery
    # POST /a2a                        # Main A2A endpoint
    # POST /a2a/agents/MathTutor       # Agent-specific endpoint
    # GET  /a2a/health                 # Health check
    
    print("Server started successfully!")
    return server

# Run the server
asyncio.run(basic_server_example())
```

## Agent Creation Examples

### Multi-Tool Agent

```python
from jaf.a2a import create_a2a_agent, create_a2a_tool

def create_research_agent():
    """Create an agent with multiple research tools"""
    
    # Web search tool
    def web_search(query: str, max_results: int = 5) -> str:
        # Mock implementation - replace with real search API
        results = [
            f"Search result {i+1} for '{query}'"
            for i in range(min(max_results, 3))
        ]
        return "\n".join(results)
    
    search_tool = create_a2a_tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
            },
            "required": ["query"]
        },
        execute_func=web_search
    )
    
    # Summarization tool
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        # Simple summarization - replace with real summarization
        sentences = text.split('. ')
        summary = '. '.join(sentences[:max_sentences])
        return f"Summary: {summary}"
    
    summary_tool = create_a2a_tool(
        name="summarize_text",
        description="Summarize long text content",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_sentences": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}
            },
            "required": ["text"]
        },
        execute_func=summarize_text
    )
    
    # Create research agent with multiple tools
    return create_a2a_agent(
        name="ResearchAgent",
        description="An intelligent research assistant that can search and summarize information",
        instruction=(
            "You are a research assistant. Use web_search to find information "
            "and summarize_text to create concise summaries. Always provide "
            "comprehensive research with multiple sources."
        ),
        tools=[search_tool, summary_tool]
    )

# Usage
research_agent = create_research_agent()
```

### Specialized Domain Agent

```python
def create_financial_advisor_agent():
    """Create a specialized financial advisory agent"""
    
    # Stock price lookup tool
    def get_stock_price(symbol: str) -> str:
        # Mock implementation - integrate with real financial API
        mock_prices = {
            "AAPL": "$175.43",
            "GOOGL": "$142.56", 
            "MSFT": "$378.85",
            "TSLA": "$248.50"
        }
        price = mock_prices.get(symbol.upper(), "Unknown")
        return f"Current price of {symbol.upper()}: {price}"
    
    stock_tool = create_a2a_tool(
        name="get_stock_price",
        description="Get current stock price for a given symbol",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL, GOOGL)",
                    "pattern": "^[A-Z]{1,5}$"
                }
            },
            "required": ["symbol"]
        },
        execute_func=get_stock_price
    )
    
    # Portfolio analysis tool
    def analyze_portfolio(holdings: list) -> str:
        total_value = sum(holding.get("value", 0) for holding in holdings)
        risk_score = min(len(holdings) * 10, 100)  # Simple diversification score
        
        return f"""
Portfolio Analysis:
- Total Value: ${total_value:,.2f}
- Number of Holdings: {len(holdings)}
- Diversification Score: {risk_score}/100
- Recommendation: {"Well diversified" if risk_score > 50 else "Consider diversifying"}
"""
    
    portfolio_tool = create_a2a_tool(
        name="analyze_portfolio",
        description="Analyze investment portfolio risk and diversification",
        parameters={
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "shares": {"type": "number"},
                            "value": {"type": "number"}
                        },
                        "required": ["symbol", "shares", "value"]
                    }
                }
            },
            "required": ["holdings"]
        },
        execute_func=analyze_portfolio
    )
    
    return create_a2a_agent(
        name="FinancialAdvisor",
        description="Expert financial advisor for investment guidance",
        instruction=(
            "You are a professional financial advisor. Use get_stock_price to "
            "check current market values and analyze_portfolio to assess investment "
            "portfolios. Always provide balanced, risk-aware advice."
        ),
        tools=[stock_tool, portfolio_tool]
    )

# Usage
financial_agent = create_financial_advisor_agent()
```

## Client Examples

### Streaming Responses

```python
import asyncio
from jaf.a2a import stream_message_to_agent, create_a2a_client

async def streaming_client_example():
    """Example of streaming responses from an A2A agent"""
    
    client = create_a2a_client("http://localhost:3000")
    
    print("🔄 Streaming response from agent...")
    
    async for event in stream_message_to_agent(
        client,
        agent_name="ResearchAgent",
        message="Research the latest developments in artificial intelligence"
    ):
        if event.get("kind") == "message":
            content = event["message"]["content"]
            print(f"📝 Chunk: {content}")
        elif event.get("kind") == "status-update":
            status = event["status"]["state"]
            print(f"📊 Status: {status}")
        elif event.get("kind") == "tool-call":
            tool_name = event.get("tool", {}).get("name", "unknown")
            print(f"🔧 Tool called: {tool_name}")

asyncio.run(streaming_client_example())
```

### Batch Operations

```python
import asyncio
from jaf.a2a import create_a2a_client, send_message_to_agent

async def batch_client_example():
    """Send multiple requests to different agents"""
    
    client = create_a2a_client("http://localhost:3000")
    
    # Define multiple tasks
    tasks = [
        ("MathTutor", "What is 25 * 17?"),
        ("ResearchAgent", "Find information about Python programming"),
        ("FinancialAdvisor", "What are the risks of investing in tech stocks?")
    ]
    
    # Create concurrent requests
    async def send_request(agent_name, message):
        try:
            response = await send_message_to_agent(client, agent_name, message)
            return {"agent": agent_name, "response": response, "error": None}
        except Exception as e:
            return {"agent": agent_name, "response": None, "error": str(e)}
    
    # Execute all requests concurrently
    print("🚀 Sending batch requests...")
    results = await asyncio.gather(*[
        send_request(agent, message) for agent, message in tasks
    ])
    
    # Process results
    for result in results:
        agent = result["agent"]
        if result["error"]:
            print(f"❌ {agent}: Error - {result['error']}")
        else:
            print(f"✅ {agent}: {result['response'][:100]}...")

asyncio.run(batch_client_example())
```

### Agent Discovery

```python
import asyncio
from jaf.a2a import discover_agents, get_agent_card

async def discovery_example():
    """Discover available agents and their capabilities"""
    
    server_url = "http://localhost:3000"
    
    # Get overall agent card
    print("🔍 Discovering agents...")
    agent_card = await get_agent_card(server_url)
    
    print(f"Server: {agent_card['name']}")
    print(f"Description: {agent_card['description']}")
    print(f"Protocol Version: {agent_card['protocolVersion']}")
    print(f"Available Skills: {len(agent_card['skills'])}")
    
    # List individual skills
    print("\n📋 Available Skills:")
    for skill in agent_card['skills']:
        print(f"  • {skill['name']}: {skill['description']}")
        if skill.get('tags'):
            print(f"    Tags: {', '.join(skill['tags'])}")
    
    # Check capabilities
    capabilities = agent_card.get('capabilities', {})
    print(f"\n⚙️ Capabilities:")
    for cap, enabled in capabilities.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {cap}")

asyncio.run(discovery_example())
```

## Server Examples

### Multi-Agent Server

```python
import asyncio
from jaf.a2a import (
    create_a2a_agent, create_a2a_tool,
    create_server_config, start_a2a_server
)

def create_customer_service_tools():
    """Create tools for customer service agent"""
    
    def lookup_order(order_id: str) -> str:
        # Mock order lookup
        return f"Order {order_id}: Status - Shipped, Expected delivery: 2 days"
    
    def process_refund(order_id: str, reason: str) -> str:
        # Mock refund processing
        return f"Refund initiated for order {order_id}. Reason: {reason}. Expected processing: 3-5 business days"
    
    return [
        create_a2a_tool(
            name="lookup_order",
            description="Look up order status and details",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID to lookup"}
                },
                "required": ["order_id"]
            },
            execute_func=lookup_order
        ),
        create_a2a_tool(
            name="process_refund",
            description="Process customer refund request",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID for refund"},
                    "reason": {"type": "string", "description": "Reason for refund"}
                },
                "required": ["order_id", "reason"]
            },
            execute_func=process_refund
        )
    ]

def create_technical_support_tools():
    """Create tools for technical support agent"""
    
    def diagnose_issue(symptoms: list) -> str:
        # Mock diagnostic logic
        if "slow" in ' '.join(symptoms).lower():
            return "Likely performance issue. Try clearing cache and restarting application."
        elif "error" in ' '.join(symptoms).lower():
            return "Error detected. Please check logs and verify configuration."
        else:
            return "Unable to diagnose. Please provide more detailed symptoms."
    
    def create_ticket(title: str, description: str, priority: str = "medium") -> str:
        # Mock ticket creation
        ticket_id = f"TECH-{hash(title) % 10000:04d}"
        return f"Ticket {ticket_id} created. Priority: {priority}. We'll respond within 24 hours."
    
    return [
        create_a2a_tool(
            name="diagnose_issue",
            description="Diagnose technical issues based on symptoms",
            parameters={
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symptoms or issues"
                    }
                },
                "required": ["symptoms"]
            },
            execute_func=diagnose_issue
        ),
        create_a2a_tool(
            name="create_ticket",
            description="Create technical support ticket",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Issue title"},
                    "description": {"type": "string", "description": "Detailed description"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "default": "medium"
                    }
                },
                "required": ["title", "description"]
            },
            execute_func=create_ticket
        )
    ]

async def multi_agent_server_example():
    """Create server with multiple specialized agents"""
    
    # Create customer service agent
    customer_agent = create_a2a_agent(
        name="CustomerService",
        description="Customer service agent for order inquiries and returns",
        instruction=(
            "You are a friendly customer service representative. "
            "Help customers with order status, returns, and general inquiries. "
            "Use lookup_order to check order status and process_refund for returns."
        ),
        tools=create_customer_service_tools()
    )
    
    # Create technical support agent
    tech_agent = create_a2a_agent(
        name="TechnicalSupport",
        description="Technical support agent for troubleshooting and issue resolution",
        instruction=(
            "You are a technical support specialist. "
            "Help users diagnose and resolve technical issues. "
            "Use diagnose_issue for troubleshooting and create_ticket for complex problems."
        ),
        tools=create_technical_support_tools()
    )
    
    # Create general assistant
    general_agent = create_a2a_agent(
        name="GeneralAssistant",
        description="General purpose assistant for information and guidance",
        instruction=(
            "You are a helpful general assistant. "
            "Provide information, answer questions, and guide users to appropriate specialists. "
            "Route customers to CustomerService for orders and TechnicalSupport for tech issues."
        ),
        tools=[]
    )
    
    # Create server with all agents
    agents = {
        "CustomerService": customer_agent,
        "TechnicalSupport": tech_agent,
        "GeneralAssistant": general_agent
    }
    
    server_config = create_server_config(
        agents=agents,
        name="Customer Support Server",
        description="Multi-agent customer support system",
        port=3000,
        cors=True
    )
    
    print("🚀 Starting multi-agent customer support server...")
    server = await start_a2a_server(server_config)
    print("✅ Server running with agents:", list(agents.keys()))
    
    return server

asyncio.run(multi_agent_server_example())
```

### Server with Memory and Configuration

```python
import asyncio
import os
from jaf.a2a import (
    create_a2a_server_config, start_a2a_server,
    create_a2a_agent, create_a2a_tool
)
from jaf.a2a.memory import create_a2a_in_memory_task_provider, A2AInMemoryTaskConfig

async def advanced_server_example():
    """Create server with advanced configuration"""
    
    # Create a conversational agent
    def remember_conversation(user_message: str, context_id: str) -> str:
        # Mock conversation memory
        return f"I remember our conversation about: {user_message[:50]}..."
    
    memory_tool = create_a2a_tool(
        name="remember_conversation",
        description="Remember important parts of the conversation",
        parameters={
            "type": "object",
            "properties": {
                "user_message": {"type": "string"},
                "context_id": {"type": "string"}
            },
            "required": ["user_message", "context_id"]
        },
        execute_func=remember_conversation
    )
    
    conversational_agent = create_a2a_agent(
        name="ConversationalAgent",
        description="Friendly conversational agent with memory",
        instruction=(
            "You are a friendly, conversational agent. "
            "Remember important details from conversations using remember_conversation. "
            "Be personable and maintain context across interactions."
        ),
        tools=[memory_tool]
    )
    
    # Configure task memory
    memory_config = A2AInMemoryTaskConfig(
        max_tasks=1000,
        max_tasks_per_context=50,
        task_ttl_seconds=3600  # 1 hour
    )
    
    task_provider = create_a2a_in_memory_task_provider(memory_config)
    
    # Advanced server configuration
    config = create_a2a_server_config(
        agents={"ConversationalAgent": conversational_agent},
        server_info={
            "name": "Advanced A2A Server",
            "description": "Production-ready A2A server with memory and monitoring",
            "version": "1.0.0",
            "contact": {"email": "support@example.com"},
            "capabilities": {
                "streaming": True,
                "taskManagement": True,
                "conversationMemory": True
            }
        },
        network_config={
            "host": "0.0.0.0",
            "port": int(os.getenv("A2A_PORT", "3000")),
            "cors": {
                "allow_origins": ["http://localhost:3000", "https://app.example.com"],
                "allow_credentials": True
            }
        },
        memory_config={
            "task_provider": task_provider,
            "conversation_ttl": 7200  # 2 hours
        }
    )
    
    print("🏗️ Starting advanced A2A server...")
    server = await start_a2a_server(config)
    print("✅ Advanced server running with full configuration")
    
    return server

asyncio.run(advanced_server_example())
```

## Integration Examples

### JAF Core Integration

```python
import asyncio
from jaf import Agent, run, RunState, RunConfig, Message, generate_run_id, generate_trace_id
from jaf.a2a import create_a2a_client, transform_a2a_agent_to_jaf, connect_to_a2a_agent

async def hybrid_local_remote_example():
    """Use both local and remote agents in a single workflow"""
    
    # Local JAF agent
    def local_instructions(state):
        return (
            "You are a local data processor. Process data and hand off "
            "to RemoteAnalyzer for complex analysis when needed."
        )
    
    local_agent = Agent(
        name="LocalProcessor",
        instructions=local_instructions,
        tools=[],
        handoffs=["RemoteAnalyzer"]  # Can hand off to remote agent
    )
    
    # Connect to remote A2A agent
    a2a_connection = await connect_to_a2a_agent("http://localhost:3000")
    
    # Transform remote agent for local use
    remote_agent = transform_a2a_agent_to_jaf(
        await a2a_connection.get_agent("ResearchAgent")
    )
    
    # Create hybrid configuration
    config = RunConfig(
        agent_registry={
            "LocalProcessor": local_agent,
            "RemoteAnalyzer": remote_agent
        },
        model_provider=make_litellm_provider("http://localhost:4000"),
        max_turns=5
    )
    
    # Run with hybrid agents
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role="user", content="Analyze this complex dataset")],
        current_agent_name="LocalProcessor",
        context={"dataset": "complex_data.csv"},
        turn_count=0
    )
    
    result = await run(initial_state, config)
    print(f"Hybrid execution result: {result.outcome}")

asyncio.run(hybrid_local_remote_example())
```

### Load Balancing Example

```python
import asyncio
import random
from jaf.a2a import create_a2a_client, send_message_to_agent

class A2ALoadBalancer:
    """Simple load balancer for A2A agents"""
    
    def __init__(self, server_urls):
        self.server_urls = server_urls
        self.clients = {}
        self.request_counts = {url: 0 for url in server_urls}
    
    async def get_client(self, strategy="round_robin"):
        """Get client based on load balancing strategy"""
        
        if strategy == "round_robin":
            # Find server with minimum requests
            selected_url = min(self.request_counts, key=self.request_counts.get)
        elif strategy == "random":
            selected_url = random.choice(self.server_urls)
        else:
            selected_url = self.server_urls[0]  # Default to first
        
        # Create client if not exists
        if selected_url not in self.clients:
            self.clients[selected_url] = create_a2a_client(selected_url)
        
        self.request_counts[selected_url] += 1
        return self.clients[selected_url], selected_url
    
    async def send_message(self, agent_name, message, strategy="round_robin"):
        """Send message with load balancing"""
        
        client, server_url = await self.get_client(strategy)
        
        try:
            response = await send_message_to_agent(client, agent_name, message)
            print(f"✅ Request sent to {server_url}")
            return response
        except Exception as e:
            print(f"❌ Request to {server_url} failed: {e}")
            # Try next server
            remaining_urls = [url for url in self.server_urls if url != server_url]
            if remaining_urls:
                backup_client = create_a2a_client(remaining_urls[0])
                return await send_message_to_agent(backup_client, agent_name, message)
            raise

async def load_balancing_example():
    """Example of load balancing across multiple A2A servers"""
    
    # Multiple server URLs (in practice, these would be different servers)
    server_urls = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002"
    ]
    
    # Create load balancer
    balancer = A2ALoadBalancer(server_urls)
    
    # Send multiple requests
    tasks = []
    for i in range(10):
        task = balancer.send_message(
            "MathTutor",
            f"What is {i} * {i}?",
            strategy="round_robin"
        )
        tasks.append(task)
    
    # Execute all requests
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Show distribution
    print("\n📊 Request Distribution:")
    for url, count in balancer.request_counts.items():
        print(f"  {url}: {count} requests")

# Note: This example assumes multiple servers are running
# asyncio.run(load_balancing_example())
```

## Error Handling Examples

### Robust Client

```python
import asyncio
import logging
from jaf.a2a import create_a2a_client, send_message_to_agent, A2AError

class RobustA2AClient:
    """A2A client with comprehensive error handling"""
    
    def __init__(self, base_url, max_retries=3, timeout=30):
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = create_a2a_client(base_url, {"timeout": timeout})
        self.logger = logging.getLogger(__name__)
    
    async def send_message_with_retry(self, agent_name, message):
        """Send message with retry logic"""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for {agent_name}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                response = await send_message_to_agent(
                    self.client, agent_name, message
                )
                
                self.logger.info(f"✅ Message sent successfully to {agent_name}")
                return response
                
            except A2AError as e:
                last_error = e
                self.logger.warning(f"A2A error on attempt {attempt + 1}: {e}")
                
                # Don't retry certain errors
                if e.code in ["AGENT_NOT_FOUND", "INVALID_REQUEST"]:
                    break
                    
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError("Request timed out")
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        # All retries failed
        self.logger.error(f"❌ All retry attempts failed for {agent_name}")
        raise last_error
    
    async def health_check(self):
        """Check if the A2A server is healthy"""
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/a2a/health",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    return health_data.get("healthy", False)
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

async def robust_client_example():
    """Example of robust A2A client usage"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create robust client
    client = RobustA2AClient("http://localhost:3000", max_retries=3)
    
    # Check server health first
    is_healthy = await client.health_check()
    if not is_healthy:
        print("❌ Server is not healthy, aborting")
        return
    
    print("✅ Server is healthy, proceeding with requests")
    
    # Send messages with error handling
    messages = [
        ("MathTutor", "What is 5 + 3?"),
        ("NonExistentAgent", "This should fail"),  # Will fail
        ("MathTutor", "What is 10 * 7?")
    ]
    
    for agent_name, message in messages:
        try:
            response = await client.send_message_with_retry(agent_name, message)
            print(f"✅ {agent_name}: {response}")
        except Exception as e:
            print(f"❌ {agent_name}: Failed after retries - {e}")

asyncio.run(robust_client_example())
```

## Testing Examples

### Unit Tests for A2A Components

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from jaf.a2a import create_a2a_agent, create_a2a_tool, create_a2a_client

class TestA2AAgent:
    """Test A2A agent functionality"""
    
    def test_agent_creation(self):
        """Test basic agent creation"""
        
        agent = create_a2a_agent(
            name="TestAgent",
            description="A test agent",
            instruction="You are a test agent",
            tools=[]
        )
        
        assert agent.name == "TestAgent"
        assert agent.description == "A test agent"
        assert agent.instruction == "You are a test agent"
        assert len(agent.tools) == 0
    
    def test_agent_with_tools(self):
        """Test agent creation with tools"""
        
        def test_func(value: str) -> str:
            return f"Processed: {value}"
        
        tool = create_a2a_tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            },
            execute_func=test_func
        )
        
        agent = create_a2a_agent(
            name="ToolAgent",
            description="Agent with tools",
            instruction="Use tools to help users",
            tools=[tool]
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"

class TestA2AClient:
    """Test A2A client functionality"""
    
    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Test A2A client creation"""
        
        client = create_a2a_client("http://localhost:3000")
        assert client.base_url == "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_mock_message_sending(self):
        """Test message sending with mocked response"""
        
        # Mock the HTTP client
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from mock agent!"
                    }
                }
            }
            mock_response.status_code = 200
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            from jaf.a2a import send_message_to_agent
            
            client = create_a2a_client("http://localhost:3000")
            response = await send_message_to_agent(
                client, "TestAgent", "Hello"
            )
            
            assert "Hello from mock agent!" in str(response)

class TestA2ATool:
    """Test A2A tool functionality"""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution"""
        
        def calculator(expression: str) -> str:
            try:
                result = eval(expression)
                return str(result)
            except:
                return "Error"
        
        tool = create_a2a_tool(
            name="calculator",
            description="Basic calculator",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            },
            execute_func=calculator
        )
        
        # Test tool execution
        result = await tool.execute_func("2 + 2")
        assert result == "4"
        
        result = await tool.execute_func("invalid")
        assert result == "Error"

# Integration tests
@pytest.mark.integration
class TestA2AIntegration:
    """Integration tests for A2A system"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete A2A workflow"""
        
        # This test assumes a test server is running
        # In practice, you might start a test server here
        
        from jaf.a2a import create_a2a_client, send_message_to_agent
        
        try:
            client = create_a2a_client("http://localhost:3001")  # Test server
            response = await send_message_to_agent(
                client, "TestAgent", "Hello, test!"
            )
            assert response is not None
        except Exception:
            pytest.skip("Test server not available")

# Run tests with: python -m pytest test_a2a_examples.py -v
```

### Load Testing

```python
import asyncio
import time
import statistics
from jaf.a2a import create_a2a_client, send_message_to_agent

async def load_test_a2a_server():
    """Load test an A2A server"""
    
    client = create_a2a_client("http://localhost:3000")
    
    # Test configuration
    num_concurrent = 10
    num_requests_per_client = 20
    
    async def client_worker(worker_id):
        """Individual client worker"""
        
        response_times = []
        errors = 0
        
        for i in range(num_requests_per_client):
            start_time = time.time()
            
            try:
                response = await send_message_to_agent(
                    client,
                    "MathTutor",
                    f"What is {i} + {worker_id}?"
                )
                
                end_time = time.time()
                response_times.append(end_time - start_time)
                
            except Exception as e:
                errors += 1
                print(f"Worker {worker_id}, Request {i}: Error - {e}")
        
        return {
            "worker_id": worker_id,
            "response_times": response_times,
            "errors": errors,
            "success_rate": (num_requests_per_client - errors) / num_requests_per_client
        }
    
    # Run load test
    print(f"🚀 Starting load test: {num_concurrent} clients, {num_requests_per_client} requests each")
    start_time = time.time()
    
    # Create concurrent workers
    workers = [client_worker(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*workers)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Aggregate results
    all_response_times = []
    total_errors = 0
    total_requests = 0
    
    for result in results:
        all_response_times.extend(result["response_times"])
        total_errors += result["errors"]
        total_requests += num_requests_per_client
    
    # Calculate statistics
    if all_response_times:
        avg_response_time = statistics.mean(all_response_times)
        median_response_time = statistics.median(all_response_times)
        p95_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.95)]
        requests_per_second = len(all_response_times) / total_duration
    else:
        avg_response_time = median_response_time = p95_response_time = 0
        requests_per_second = 0
    
    # Print results
    print(f"\n📊 Load Test Results:")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {total_requests - total_errors}")
    print(f"Failed Requests: {total_errors}")
    print(f"Success Rate: {(total_requests - total_errors) / total_requests * 100:.1f}%")
    print(f"Requests/Second: {requests_per_second:.2f}")
    print(f"Average Response Time: {avg_response_time * 1000:.2f}ms")
    print(f"Median Response Time: {median_response_time * 1000:.2f}ms")
    print(f"95th Percentile: {p95_response_time * 1000:.2f}ms")

# Run load test
# asyncio.run(load_test_a2a_server())
```

## Production Deployment Examples

### Docker Deployment

```dockerfile
# Dockerfile for A2A server
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/a2a/health || exit 1

# Run application
CMD ["python", "-m", "jaf.a2a.examples.production_server"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  a2a-server:
    build: .
    ports:
      - "3000:3000"
    environment:
      - A2A_HOST=0.0.0.0
      - A2A_PORT=3000
      - A2A_LOG_LEVEL=INFO
      - A2A_CORS_ORIGINS=https://app.example.com
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - a2a-server
    restart: unless-stopped
```

### Production Server Configuration

```python
import os
import logging
import asyncio
from jaf.a2a import (
    create_a2a_server_config, start_a2a_server,
    create_a2a_agent, create_a2a_tool
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("A2A_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def create_production_agents():
    """Create production-ready agents"""
    
    # Create robust tools with error handling
    def safe_calculator(expression: str) -> str:
        try:
            # Validate expression for security
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            # Limit expression length
            if len(expression) > 100:
                return "Error: Expression too long"
            
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            logging.error(f"Calculator error: {e}")
            return f"Error: {str(e)}"
    
    calc_tool = create_a2a_tool(
        name="calculate",
        description="Perform safe mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": r"^[0-9+\-*/().\\s]+$"
                }
            },
            "required": ["expression"]
        },
        execute_func=safe_calculator
    )
    
    # Production math agent
    math_agent = create_a2a_agent(
        name="MathTutor",
        description="Production math tutor with safety features",
        instruction=(
            "You are a professional math tutor. Use the calculate tool for "
            "mathematical computations. Always validate inputs and provide "
            "clear explanations. Handle errors gracefully."
        ),
        tools=[calc_tool]
    )
    
    return {"MathTutor": math_agent}

async def main():
    """Production server main function"""
    
    # Environment configuration
    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "3000"))
    cors_origins = os.getenv("A2A_CORS_ORIGINS", "").split(",")
    
    # Create agents
    agents = create_production_agents()
    
    # Production server configuration
    config = create_a2a_server_config(
        agents=agents,
        server_info={
            "name": "Production A2A Server",
            "description": "Production-ready A2A agent server",
            "version": "1.0.0",
            "contact": {"email": "support@example.com"},
            "capabilities": {
                "streaming": True,
                "taskManagement": True,
                "healthChecks": True
            }
        },
        network_config={
            "host": host,
            "port": port,
            "cors": {
                "allow_origins": cors_origins if cors_origins != [''] else ["*"],
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["*"]
            }
        }
    )
    
    # Start server with graceful shutdown
    logging.info(f"Starting A2A server on {host}:{port}")
    server = await start_a2a_server(config)
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down A2A server...")
    finally:
        if hasattr(server, 'shutdown'):
            await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

These examples provide comprehensive coverage of A2A protocol usage, from simple client-server interactions to complex production deployments. Use them as starting points for building your own distributed agent systems.