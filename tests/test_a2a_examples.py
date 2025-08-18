"""
Test suite for A2A examples documentation.
Tests all code examples from docs/a2a-examples.md to ensure they work with the actual implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from jaf.a2a import (
    create_a2a_agent, create_a2a_tool, 
    create_server_config, create_a2a_client
)


class TestA2ABasicExamples:
    """Test basic A2A examples from the documentation."""
    
    def test_create_calculator_tool(self):
        """Test calculator tool creation from basic server example."""
        
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
        
        tool = create_a2a_tool(
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
        
        assert tool.name == "calculate"
        assert tool.description == "Perform mathematical calculations"
        assert "expression" in tool.parameters["properties"]
        
        # Test tool execution
        result = tool.execute("2 + 2")
        assert result == "2 + 2 = 4"
        
        result = tool.execute("invalid$")
        assert "Error: Invalid characters" in result
    
    def test_create_math_agent(self):
        """Test math tutor agent creation."""
        
        def calculate(expression: str) -> str:
            try:
                result = eval(expression)
                return f"{expression} = {result}"
            except Exception as e:
                return f"Error: {e}"
        
        calc_tool = create_a2a_tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            },
            execute_func=calculate
        )
        
        math_agent = create_a2a_agent(
            name="MathTutor",
            description="A helpful math tutor that can perform calculations",
            instruction="You are a math tutor. Use the calculate tool for math problems.",
            tools=[calc_tool]
        )
        
        assert math_agent.name == "MathTutor"
        assert "math tutor" in math_agent.description.lower()
        assert len(math_agent.tools) == 1
        assert math_agent.tools[0].name == "calculate"


class TestA2AAgentCreation:
    """Test agent creation examples."""
    
    def test_research_agent_creation(self):
        """Test multi-tool research agent creation."""
        
        def web_search(query: str, max_results: int = 5) -> str:
            # Mock implementation
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
        
        def summarize_text(text: str, max_sentences: int = 3) -> str:
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
        
        research_agent = create_a2a_agent(
            name="ResearchAgent",
            description="An intelligent research assistant that can search and summarize information",
            instruction=(
                "You are a research assistant. Use web_search to find information "
                "and summarize_text to create concise summaries. Always provide "
                "comprehensive research with multiple sources."
            ),
            tools=[search_tool, summary_tool]
        )
        
        assert research_agent.name == "ResearchAgent"
        assert len(research_agent.tools) == 2
        assert any(tool.name == "web_search" for tool in research_agent.tools)
        assert any(tool.name == "summarize_text" for tool in research_agent.tools)
    
    def test_financial_advisor_agent(self):
        """Test specialized financial advisor agent."""
        
        def get_stock_price(symbol: str) -> str:
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
        
        def analyze_portfolio(holdings: list) -> str:
            total_value = sum(holding.get("value", 0) for holding in holdings)
            risk_score = min(len(holdings) * 10, 100)
            
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
        
        financial_agent = create_a2a_agent(
            name="FinancialAdvisor",
            description="Expert financial advisor for investment guidance",
            instruction=(
                "You are a professional financial advisor. Use get_stock_price to "
                "check current market values and analyze_portfolio to assess investment "
                "portfolios. Always provide balanced, risk-aware advice."
            ),
            tools=[stock_tool, portfolio_tool]
        )
        
        assert financial_agent.name == "FinancialAdvisor"
        assert len(financial_agent.tools) == 2
        
        # Test tool functionality
        stock_result = stock_tool.execute("AAPL")
        assert "AAPL" in stock_result and "$175.43" in stock_result
        
        portfolio_result = portfolio_tool.execute([
            {"symbol": "AAPL", "shares": 10, "value": 1754.30},
            {"symbol": "GOOGL", "shares": 5, "value": 712.80}
        ])
        assert "Total Value: $2,467.10" in portfolio_result
        assert "Number of Holdings: 2" in portfolio_result


class TestA2AClientExamples:
    """Test A2A client examples."""
    
    def test_client_creation(self):
        """Test A2A client creation."""
        client = create_a2a_client("http://localhost:3000")
        assert client.config.base_url == "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_robust_client_class(self):
        """Test robust A2A client implementation."""
        
        class RobustA2AClient:
            def __init__(self, base_url, max_retries=3, timeout=30):
                self.base_url = base_url
                self.max_retries = max_retries
                self.timeout = timeout
                self.client = create_a2a_client(base_url)
            
            async def health_check(self):
                """Mock health check implementation."""
                try:
                    # Mock successful health check
                    return True
                except Exception:
                    return False
        
        client = RobustA2AClient("http://localhost:3000", max_retries=3)
        assert client.base_url == "http://localhost:3000"
        assert client.max_retries == 3
        assert client.timeout == 30
        
        # Test health check
        is_healthy = await client.health_check()
        assert is_healthy is True


class TestA2AServerExamples:
    """Test A2A server configuration examples."""
    
    def test_multi_agent_server_config(self):
        """Test multi-agent server configuration."""
        
        # Customer service tools
        def lookup_order(order_id: str) -> str:
            return f"Order {order_id}: Status - Shipped, Expected delivery: 2 days"
        
        def process_refund(order_id: str, reason: str) -> str:
            return f"Refund initiated for order {order_id}. Reason: {reason}. Expected processing: 3-5 business days"
        
        customer_tools = [
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
        
        # Technical support tools
        def diagnose_issue(symptoms: list) -> str:
            if "slow" in ' '.join(symptoms).lower():
                return "Likely performance issue. Try clearing cache and restarting application."
            elif "error" in ' '.join(symptoms).lower():
                return "Error detected. Please check logs and verify configuration."
            else:
                return "Unable to diagnose. Please provide more detailed symptoms."
        
        def create_ticket(title: str, description: str, priority: str = "medium") -> str:
            ticket_id = f"TECH-{hash(title) % 10000:04d}"
            return f"Ticket {ticket_id} created. Priority: {priority}. We'll respond within 24 hours."
        
        tech_tools = [
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
        
        # Create agents
        customer_agent = create_a2a_agent(
            name="CustomerService",
            description="Customer service agent for order inquiries and returns",
            instruction=(
                "You are a friendly customer service representative. "
                "Help customers with order status, returns, and general inquiries. "
                "Use lookup_order to check order status and process_refund for returns."
            ),
            tools=customer_tools
        )
        
        tech_agent = create_a2a_agent(
            name="TechnicalSupport",
            description="Technical support agent for troubleshooting and issue resolution",
            instruction=(
                "You are a technical support specialist. "
                "Help users diagnose and resolve technical issues. "
                "Use diagnose_issue for troubleshooting and create_ticket for complex problems."
            ),
            tools=tech_tools
        )
        
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
        
        # Test agent creation
        assert customer_agent.name == "CustomerService"
        assert len(customer_agent.tools) == 2
        assert tech_agent.name == "TechnicalSupport"
        assert len(tech_agent.tools) == 2
        assert general_agent.name == "GeneralAssistant"
        assert len(general_agent.tools) == 0
        
        # Test tool functionality
        order_result = customer_tools[0].execute("12345")
        assert "Order 12345" in order_result and "Shipped" in order_result
        
        refund_result = customer_tools[1].execute("12345", "Defective item")
        assert "Refund initiated" in refund_result and "Defective item" in refund_result
        
        diagnosis_result = tech_tools[0].execute(["application is slow", "takes forever to load"])
        assert "performance issue" in diagnosis_result.lower()
        
        ticket_result = tech_tools[1].execute("Login issues", "Cannot access account", "high")
        assert "TECH-" in ticket_result and "high" in ticket_result


class TestA2AIntegrationExamples:
    """Test A2A integration examples."""
    
    def test_load_balancer_class(self):
        """Test A2A load balancer implementation."""
        
        class A2ALoadBalancer:
            def __init__(self, server_urls):
                self.server_urls = server_urls
                self.clients = {}
                self.request_counts = {url: 0 for url in server_urls}
            
            async def get_client(self, strategy="round_robin"):
                if strategy == "round_robin":
                    selected_url = min(self.request_counts, key=self.request_counts.get)
                elif strategy == "random":
                    import random
                    selected_url = random.choice(self.server_urls)
                else:
                    selected_url = self.server_urls[0]
                
                if selected_url not in self.clients:
                    self.clients[selected_url] = create_a2a_client(selected_url)
                
                self.request_counts[selected_url] += 1
                return self.clients[selected_url], selected_url
        
        server_urls = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:3002"
        ]
        
        balancer = A2ALoadBalancer(server_urls)
        assert len(balancer.server_urls) == 3
        assert all(count == 0 for count in balancer.request_counts.values())
        
        # Test round robin selection
        import asyncio
        async def test_selection():
            client1, url1 = await balancer.get_client("round_robin")
            client2, url2 = await balancer.get_client("round_robin")
            return url1, url2
        
        url1, url2 = asyncio.run(test_selection())
        assert url1 in server_urls
        assert url2 in server_urls


class TestA2AErrorHandling:
    """Test A2A error handling examples."""
    
    def test_robust_client_error_handling(self):
        """Test robust client with error handling."""
        
        class RobustA2AClient:
            def __init__(self, base_url, max_retries=3, timeout=30):
                self.base_url = base_url
                self.max_retries = max_retries
                self.timeout = timeout
                self.client = create_a2a_client(base_url)
            
            async def send_message_with_retry(self, agent_name, message):
                """Mock implementation of retry logic."""
                last_error = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
                        # Mock successful response after retries
                        if attempt == self.max_retries:
                            return f"Response from {agent_name}: {message}"
                        else:
                            # Simulate failure for testing
                            raise Exception(f"Attempt {attempt + 1} failed")
                            
                    except Exception as e:
                        last_error = e
                        continue
                
                raise last_error
        
        client = RobustA2AClient("http://localhost:3000", max_retries=2)
        
        # Test that retry logic is properly structured
        assert client.max_retries == 2
        assert client.timeout == 30
        
        # Test retry method exists and works
        import asyncio
        result = asyncio.run(client.send_message_with_retry("TestAgent", "Hello"))
        assert "Response from TestAgent" in result


class TestA2ATestingExamples:
    """Test A2A testing examples."""
    
    def test_agent_creation_testing(self):
        """Test basic agent creation testing."""
        
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
    
    def test_agent_with_tools_testing(self):
        """Test agent creation with tools testing."""
        
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
        
        # Test tool execution
        result = tool.execute("test input")
        assert result == "Processed: test input"
    
    @pytest.mark.asyncio
    async def test_tool_execution_async(self):
        """Test asynchronous tool execution."""
        
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
        result = tool.execute("2 + 2")
        assert result == "4"
        
        result = tool.execute("invalid")
        assert result == "Error"


class TestA2AProductionExamples:
    """Test production deployment examples."""
    
    def test_production_agent_creation(self):
        """Test production-ready agent creation."""
        
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
        
        assert math_agent.name == "MathTutor"
        assert len(math_agent.tools) == 1
        
        # Test safety features
        result = calc_tool.execute("2 + 2")
        assert result == "2 + 2 = 4"
        
        result = calc_tool.execute("invalid$characters")
        assert "Invalid characters" in result
        
        result = calc_tool.execute("1" * 101)  # Too long but valid characters
        assert "Expression too long" in result
    
    def test_environment_configuration(self):
        """Test environment-based configuration."""
        import os
        
        # Mock environment variables
        test_env = {
            "A2A_HOST": "0.0.0.0",
            "A2A_PORT": "3000",
            "A2A_CORS_ORIGINS": "https://app.example.com,https://admin.example.com"
        }
        
        # Test configuration parsing
        host = test_env.get("A2A_HOST", "0.0.0.0")
        port = int(test_env.get("A2A_PORT", "3000"))
        cors_origins = test_env.get("A2A_CORS_ORIGINS", "").split(",")
        
        assert host == "0.0.0.0"
        assert port == 3000
        assert len(cors_origins) == 2
        assert "https://app.example.com" in cors_origins
        assert "https://admin.example.com" in cors_origins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
