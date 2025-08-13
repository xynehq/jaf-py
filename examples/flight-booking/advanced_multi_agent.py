#!/usr/bin/env python3
"""
Advanced Multi-Agent Flight Booking System with Intelligent Coordination

This example demonstrates the advanced ADK capabilities including:
- Intelligent agent selection based on keyword matching
- Sophisticated response merging for parallel execution
- Hierarchical coordination with delegation decision extraction
- Rule-based coordination with conditional/parallel/sequential actions
- Comprehensive schema validation for tool parameters
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from jaf import (
    Agent,
    ContentRole,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    ToolSource,
    create_function_tool,
    generate_run_id,
    generate_trace_id,
    run,
    Tool
)
from jaf.providers.model import make_litellm_provider
from jaf.core.tool_results import ToolResponse, ToolResult

# Import new ADK capabilities (using relative imports for the example)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adk.runners.multi_agent import (
    execute_multi_agent,
    select_best_agent,
    merge_parallel_responses,
    extract_keywords
)
from adk.runners.types import (
    AgentConfig,
    MultiAgentConfig,
    DelegationStrategy,
    RunContext,
    SimpleCoordinationRule,
    CoordinationAction
)
from adk.schemas.validation import validate_schema
from adk.schemas.types import JsonSchema

# Import basic tools from index
from index import (
    search_flights_tool,
    check_seat_availability_tool,
    book_flight_tool,
    check_flight_status_tool,
    cancel_booking_tool,
    FlightSearchArgs,
    BookFlightArgs,
)


# ========== Enhanced Tools with Validation ==========

class WeatherCheckArgs(BaseModel):
    """Arguments for weather checking tool."""
    city: str = Field(description="City name to check weather for", min_length=2, max_length=50)
    date: str = Field(description="Date to check weather (YYYY-MM-DD format)")


async def check_weather_execute(args: WeatherCheckArgs, context: Any) -> ToolResult:
    """Check weather conditions for travel planning."""
    # Validate arguments using enhanced schema validation
    schema: JsonSchema = {
        'type': 'object',
        'properties': {
            'city': {
                'type': 'string',
                'minLength': 2,
                'maxLength': 50,
                'pattern': r'^[a-zA-Z\s\-\.]+$'
            },
            'date': {
                'type': 'string',
                'format': 'date'
            }
        },
        'required': ['city', 'date']
    }
    
    validation_result = validate_schema(args.dict(), schema)
    if not validation_result.success:
        return ToolResponse.validation_error(f"Invalid arguments: {validation_result.errors}")
    
    # Mock weather data
    weather_info = {
        "city": args.city,
        "date": args.date,
        "temperature": "72°F",
        "conditions": "Partly cloudy",
        "travel_advisory": "Good conditions for travel"
    }
    
    return ToolResponse.success(weather_info)


check_weather_tool = create_function_tool({
    'name': 'check_weather',
    'description': 'Check weather conditions for travel planning',
    'execute': check_weather_execute,
    'parameters': WeatherCheckArgs,
    'metadata': {'category': 'travel', 'provider': 'weather_service'},
    'source': ToolSource.EXTERNAL
})


class LoyaltyPointsArgs(BaseModel):
    """Arguments for loyalty points management."""
    customer_id: str = Field(description="Customer ID", pattern=r'^[A-Z0-9]{6,12}$')
    action: str = Field(description="Action: check, redeem, or earn")
    points: Optional[int] = Field(default=None, description="Points amount", ge=0, le=100000)


async def manage_loyalty_points_execute(args: LoyaltyPointsArgs, context: Any) -> ToolResult:
    """Manage customer loyalty points."""
    # Enhanced validation with multiple constraints
    schema: JsonSchema = {
        'type': 'object',
        'properties': {
            'customer_id': {
                'type': 'string',
                'pattern': r'^[A-Z0-9]{6,12}$',
                'description': 'Customer ID must be 6-12 alphanumeric characters'
            },
            'action': {
                'type': 'string',
                'enum': ['check', 'redeem', 'earn']
            },
            'points': {
                'type': 'integer',
                'minimum': 0,
                'maximum': 100000
            }
        },
        'required': ['customer_id', 'action']
    }
    
    validation_result = validate_schema(args.dict(), schema)
    if not validation_result.success:
        return ToolResponse.validation_error(f"Validation failed: {validation_result.errors}")
    
    # Mock loyalty points operations
    if args.action == "check":
        return ToolResponse.success({
            "customer_id": args.customer_id,
            "current_balance": 15750,
            "tier_status": "Gold",
            "next_tier_requirements": 5000
        })
    elif args.action == "redeem":
        if not args.points:
            return ToolResponse.error("Points amount required for redemption")
        return ToolResponse.success({
            "redeemed": args.points,
            "remaining_balance": 15750 - args.points,
            "confirmation": f"Redeemed {args.points} points successfully"
        })
    elif args.action == "earn":
        return ToolResponse.success({
            "earned": args.points or 0,
            "new_balance": 15750 + (args.points or 0),
            "message": "Points earned successfully"
        })


loyalty_points_tool = create_function_tool({
    'name': 'manage_loyalty_points',
    'description': 'Check, redeem, or earn customer loyalty points',
    'execute': manage_loyalty_points_execute,
    'parameters': LoyaltyPointsArgs,
    'metadata': {'category': 'loyalty', 'security_level': 'authenticated'},
    'source': ToolSource.NATIVE
})


class TravelInsuranceArgs(BaseModel):
    """Arguments for travel insurance quotes."""
    trip_cost: float = Field(description="Total trip cost", gt=0, le=50000)
    destination: str = Field(description="Travel destination", min_length=2)
    duration_days: int = Field(description="Trip duration in days", ge=1, le=365)
    traveler_age: int = Field(description="Primary traveler age", ge=18, le=100)


async def get_travel_insurance_execute(args: TravelInsuranceArgs, context: Any) -> ToolResult:
    """Get travel insurance quotes."""
    # Comprehensive validation with business rules
    schema: JsonSchema = {
        'type': 'object',
        'properties': {
            'trip_cost': {
                'type': 'number',
                'minimum': 0,
                'exclusiveMinimum': True,
                'maximum': 50000
            },
            'destination': {
                'type': 'string',
                'minLength': 2,
                'maxLength': 100
            },
            'duration_days': {
                'type': 'integer',
                'minimum': 1,
                'maximum': 365
            },
            'traveler_age': {
                'type': 'integer',
                'minimum': 18,
                'maximum': 100
            }
        },
        'required': ['trip_cost', 'destination', 'duration_days', 'traveler_age']
    }
    
    validation_result = validate_schema(args.dict(), schema)
    if not validation_result.success:
        return ToolResponse.validation_error(f"Invalid insurance request: {validation_result.errors}")
    
    # Calculate insurance premium (simplified)
    base_rate = 0.05
    age_multiplier = 1.2 if args.traveler_age > 65 else 1.0
    duration_multiplier = 1.1 if args.duration_days > 14 else 1.0
    
    premium = args.trip_cost * base_rate * age_multiplier * duration_multiplier
    
    return ToolResponse.success({
        "trip_cost": args.trip_cost,
        "destination": args.destination,
        "duration_days": args.duration_days,
        "premium_quote": round(premium, 2),
        "coverage_details": {
            "trip_cancellation": args.trip_cost,
            "medical_emergency": 100000,
            "baggage_loss": 2500
        },
        "quote_valid_until": (datetime.now() + timedelta(days=30)).isoformat()
    })


travel_insurance_tool = create_function_tool({
    'name': 'get_travel_insurance',
    'description': 'Get travel insurance quotes and coverage options',
    'execute': get_travel_insurance_execute,
    'parameters': TravelInsuranceArgs,
    'metadata': {'category': 'insurance', 'partner': 'travel_protect'},
    'source': ToolSource.EXTERNAL
})


# ========== Advanced Agent Configurations ==========

def create_search_agent() -> AgentConfig:
    """Create intelligent flight search agent."""
    return AgentConfig(
        name="SearchAgent",
        instruction="""You are an expert flight search specialist with advanced capabilities.
        
Your expertise includes:
- Finding optimal flight combinations and routes
- Comparing prices across multiple airlines and booking classes
- Checking real-time availability and seat options
- Providing weather information for travel planning
- Understanding complex routing requirements

Use your tools to provide comprehensive flight search results. When customers are ready to book,
guide them toward the booking process or delegate to the BookingAgent.""",
        tools=[search_flights_tool, check_seat_availability_tool, check_flight_status_tool, check_weather_tool],
        metadata={
            "specialization": ["flight_search", "route_planning", "weather"],
            "keywords": ["flight", "search", "find", "compare", "weather", "availability"]
        }
    )


def create_booking_agent() -> AgentConfig:
    """Create intelligent booking agent."""
    return AgentConfig(
        name="BookingAgent",
        instruction="""You are a specialized flight booking agent focused on completing reservations.

Your responsibilities:
- Processing flight bookings with accuracy and care
- Managing booking confirmations and documentation
- Handling booking modifications and cancellations
- Integrating loyalty points and rewards programs
- Coordinating travel insurance options

Always confirm all booking details before finalizing. Ensure customers understand terms and conditions.""",
        tools=[book_flight_tool, cancel_booking_tool, check_flight_status_tool, loyalty_points_tool],
        metadata={
            "specialization": ["booking", "reservations", "loyalty", "cancellation"],
            "keywords": ["book", "reserve", "confirm", "cancel", "loyalty", "points"]
        }
    )


def create_pricing_agent() -> AgentConfig:
    """Create intelligent pricing and fare specialist."""
    return AgentConfig(
        name="PricingAgent",
        instruction="""You are a flight pricing expert with deep knowledge of fare structures.

Your expertise covers:
- Explaining different fare classes and restrictions
- Analyzing price trends and finding best deals
- Understanding airline pricing strategies
- Providing cost optimization recommendations
- Explaining baggage fees, change fees, and policies

Help customers understand pricing complexity and make informed decisions.""",
        tools=[search_flights_tool, check_seat_availability_tool],
        metadata={
            "specialization": ["pricing", "fares", "policies", "optimization"],
            "keywords": ["price", "cost", "fare", "fee", "cheap", "expensive", "deal"]
        }
    )


def create_support_agent() -> AgentConfig:
    """Create customer support and insurance specialist."""
    return AgentConfig(
        name="SupportAgent",
        instruction="""You are a comprehensive travel support specialist.

Your services include:
- Providing travel insurance quotes and information
- Handling general customer service inquiries
- Managing loyalty program questions and transactions
- Assisting with travel planning and recommendations
- Resolving booking issues and concerns

You're the go-to agent for insurance, loyalty programs, and general support.""",
        tools=[travel_insurance_tool, loyalty_points_tool, check_flight_status_tool],
        metadata={
            "specialization": ["support", "insurance", "loyalty", "assistance"],
            "keywords": ["insurance", "help", "support", "problem", "issue", "loyalty", "assistance"]
        }
    )


def create_coordinator_agent() -> AgentConfig:
    """Create intelligent coordinator agent."""
    return AgentConfig(
        name="CoordinatorAgent",
        instruction="""You are the main travel coordination specialist. 

Your role is to:
- Understand customer travel needs comprehensively
- Route customers to the most appropriate specialist
- Coordinate complex multi-step travel planning
- Ensure smooth handoffs between specialists

Available specialists:
- SearchAgent: Flight search, availability, weather
- BookingAgent: Reservations, confirmations, cancellations
- PricingAgent: Fare analysis, pricing optimization
- SupportAgent: Insurance, loyalty programs, general support

Analyze each request and delegate to the best specialist for the customer's needs.""",
        tools=[],  # Coordinator primarily delegates
        metadata={
            "role": "coordinator",
            "delegates_to": ["SearchAgent", "BookingAgent", "PricingAgent", "SupportAgent"]
        }
    )


# ========== Advanced Coordination Rules ==========

def create_coordination_rules() -> List[SimpleCoordinationRule]:
    """Create intelligent coordination rules for agent selection."""
    
    # Rule 1: Insurance inquiries go to SupportAgent
    def insurance_condition(message: Message, context: RunContext) -> bool:
        text = str(message.content).lower()
        insurance_keywords = ["insurance", "coverage", "protect", "policy", "claim"]
        return any(keyword in text for keyword in insurance_keywords)
    
    insurance_rule = SimpleCoordinationRule(
        condition_func=insurance_condition,
        action_type=CoordinationAction.DELEGATE,
        target_agent_names=["SupportAgent"]
    )
    
    # Rule 2: Pricing/cost questions to PricingAgent
    def pricing_condition(message: Message, context: RunContext) -> bool:
        text = str(message.content).lower()
        pricing_keywords = ["price", "cost", "cheap", "expensive", "fare", "fee", "deal"]
        return any(keyword in text for keyword in pricing_keywords)
    
    pricing_rule = SimpleCoordinationRule(
        condition_func=pricing_condition,
        action_type=CoordinationAction.DELEGATE,
        target_agent_names=["PricingAgent"]
    )
    
    # Rule 3: Booking/reservation requests to BookingAgent
    def booking_condition(message: Message, context: RunContext) -> bool:
        text = str(message.content).lower()
        booking_keywords = ["book", "reserve", "confirm", "purchase", "buy"]
        return any(keyword in text for keyword in booking_keywords)
    
    booking_rule = SimpleCoordinationRule(
        condition_func=booking_condition,
        action_type=CoordinationAction.DELEGATE,
        target_agent_names=["BookingAgent"]
    )
    
    # Rule 4: Complex travel planning - parallel execution
    def complex_condition(message: Message, context: RunContext) -> bool:
        text = str(message.content).lower()
        complex_indicators = ["plan", "organize", "comprehensive", "everything", "all"]
        multi_service = sum(1 for word in ["flight", "insurance", "weather", "loyalty"] if word in text)
        return any(indicator in text for indicator in complex_indicators) or multi_service >= 2
    
    complex_rule = SimpleCoordinationRule(
        condition_func=complex_condition,
        action_type=CoordinationAction.PARALLEL,
        target_agent_names=["SearchAgent", "SupportAgent"]
    )
    
    # Rule 5: Admin users get parallel execution for efficiency
    def admin_condition(message: Message, context: RunContext) -> bool:
        permissions = context.get('permissions', [])
        return 'admin' in permissions
    
    admin_rule = SimpleCoordinationRule(
        condition_func=admin_condition,
        action_type=CoordinationAction.PARALLEL,
        target_agent_names=["SearchAgent", "BookingAgent", "PricingAgent"]
    )
    
    return [insurance_rule, pricing_rule, booking_rule, complex_rule, admin_rule]


# ========== Advanced Multi-Agent Orchestration ==========

class AdvancedFlightBookingSystem:
    """Advanced flight booking system with intelligent multi-agent coordination."""
    
    def __init__(self):
        self.agents = {
            "SearchAgent": create_search_agent(),
            "BookingAgent": create_booking_agent(),
            "PricingAgent": create_pricing_agent(),
            "SupportAgent": create_support_agent(),
            "CoordinatorAgent": create_coordinator_agent()
        }
        
        self.coordination_rules = create_coordination_rules()
    
    def create_conditional_config(self) -> MultiAgentConfig:
        """Create configuration for conditional agent selection."""
        return MultiAgentConfig(
            delegation_strategy=DelegationStrategy.CONDITIONAL,
            sub_agents=list(self.agents.values()),
            coordination_rules=self.coordination_rules
        )
    
    def create_hierarchical_config(self) -> MultiAgentConfig:
        """Create configuration for hierarchical coordination."""
        # Coordinator first, then specialists
        hierarchical_agents = [
            self.agents["CoordinatorAgent"],
            self.agents["SearchAgent"],
            self.agents["BookingAgent"],
            self.agents["PricingAgent"],
            self.agents["SupportAgent"]
        ]
        
        return MultiAgentConfig(
            delegation_strategy=DelegationStrategy.HIERARCHICAL,
            sub_agents=hierarchical_agents
        )
    
    def create_parallel_config(self) -> MultiAgentConfig:
        """Create configuration for parallel execution."""
        # Execute search and support in parallel for comprehensive results
        return MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL,
            sub_agents=[
                self.agents["SearchAgent"],
                self.agents["SupportAgent"]
            ]
        )
    
    async def demonstrate_conditional_coordination(self):
        """Demonstrate intelligent conditional agent selection."""
        print("\n🎯 Conditional Coordination Demo")
        print("=" * 50)
        
        test_messages = [
            "I need travel insurance for my trip to Europe",
            "What's the cheapest flight to Miami?", 
            "I want to book this flight AA123",
            "Help me plan a comprehensive trip with flights and insurance",
            "Check my loyalty points balance"
        ]
        
        config = self.create_conditional_config()
        
        for message_text in test_messages:
            print(f"\n📝 User: {message_text}")
            
            # Extract keywords for analysis
            keywords = extract_keywords(message_text.lower())
            print(f"🔍 Keywords: {keywords}")
            
            # Select best agent
            message = Message(role=ContentRole.USER, content=message_text)
            context = RunContext(user_id="demo_user")
            
            selected_agent = select_best_agent(list(self.agents.values()), message, context)
            print(f"🎯 Selected Agent: {selected_agent.name}")
            print(f"📋 Agent Specialization: {selected_agent.metadata.get('specialization', [])}")
    
    async def demonstrate_parallel_coordination(self):
        """Demonstrate parallel agent execution and response merging."""
        print("\n⚡ Parallel Coordination Demo")
        print("=" * 50)
        
        # Mock responses for demonstration
        search_response = {
            "content": Message(role="assistant", content="Found 5 great flight options to your destination"),
            "session_state": {"search_results": ["flight1", "flight2"]},
            "artifacts": {"flights": ["AA123", "UA456"]},
            "execution_time_ms": 150.0,
            "metadata": {}
        }
        
        support_response = {
            "content": Message(role="assistant", content="Travel insurance quote: $89 for comprehensive coverage"),
            "session_state": {"insurance_quote": "$89"},
            "artifacts": {"quote_id": "INS-789", "coverage": "comprehensive"},
            "execution_time_ms": 200.0,
            "metadata": {}
        }
        
        responses = [
            type('AgentResponse', (), search_response)(),
            type('AgentResponse', (), support_response)()
        ]
        
        config = self.create_parallel_config()
        merged_response = merge_parallel_responses(responses, config)
        
        print("📋 Merged Response Content:")
        print(merged_response.content.content)
        print(f"\n📊 Merged Artifacts: {merged_response.artifacts}")
        print(f"⏱️ Total Execution Time: {merged_response.execution_time_ms}ms")
    
    async def demonstrate_schema_validation(self):
        """Demonstrate enhanced schema validation capabilities."""
        print("\n✅ Schema Validation Demo")
        print("=" * 50)
        
        # Test weather tool validation
        print("🌤️ Testing Weather Tool Validation:")
        
        # Valid request
        valid_args = WeatherCheckArgs(city="New York", date="2024-01-15")
        print(f"✅ Valid: {valid_args}")
        
        # Invalid requests
        try:
            invalid_city = WeatherCheckArgs(city="X", date="2024-01-15")  # Too short
        except Exception as e:
            print(f"❌ Invalid city (too short): {e}")
        
        try:
            invalid_date = WeatherCheckArgs(city="Boston", date="invalid-date")
        except Exception as e:
            print(f"❌ Invalid date format: {e}")
        
        # Test loyalty points validation
        print("\n💳 Testing Loyalty Points Validation:")
        
        valid_loyalty = LoyaltyPointsArgs(customer_id="ABC123XYZ", action="check")
        print(f"✅ Valid: {valid_loyalty}")
        
        try:
            invalid_customer = LoyaltyPointsArgs(customer_id="invalid", action="check")
        except Exception as e:
            print(f"❌ Invalid customer ID: {e}")


# ========== Mock Model Provider for Demo ==========

class MockAdvancedModelProvider:
    """Advanced mock model provider with intelligent responses."""
    
    def __init__(self):
        self.agent_responses = {
            "SearchAgent": "I found several excellent flight options matching your criteria. Here are the top recommendations with pricing and availability details.",
            "BookingAgent": "I can help you complete this booking. Let me confirm all the details and process your reservation securely.",
            "PricingAgent": "Based on current market analysis, here are the best fare options and pricing strategies for your travel dates.",
            "SupportAgent": "I can assist with travel insurance, loyalty programs, and any other support needs you may have.",
            "CoordinatorAgent": "I understand your travel needs. Let me connect you with the most appropriate specialist for your request."
        }
    
    async def get_completion(self, state, agent, config):
        agent_name = agent.name
        response_text = self.agent_responses.get(agent_name, "I'm here to help with your travel needs!")
        
        return {
            'message': {
                'content': response_text,
                'tool_calls': None
            }
        }


# ========== Main Demonstration ==========

async def main():
    """Demonstrate advanced multi-agent coordination capabilities."""
    print("🚀 Advanced Multi-Agent Flight Booking System")
    print("=" * 60)
    print("Demonstrating ADK's intelligent coordination capabilities:")
    print("• Keyword-based agent selection")
    print("• Rule-based coordination")  
    print("• Parallel execution and response merging")
    print("• Enhanced schema validation")
    print("• Hierarchical coordination")
    
    system = AdvancedFlightBookingSystem()
    
    # Demonstrate different coordination strategies
    await system.demonstrate_conditional_coordination()
    await system.demonstrate_parallel_coordination()
    await system.demonstrate_schema_validation()
    
    print("\n🎯 Live Multi-Agent Execution Demo")
    print("=" * 50)
    
    # Create a sample configuration for live demo
    config = system.create_conditional_config()
    model_provider = make_litellm_provider(
        base_url=os.getenv("LITELLM_URL"),
        api_key=os.getenv("LITELLM_API_KEY")
    )
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Complex travel planning request",
            "message": "I need comprehensive help planning a business trip - flights, insurance, and loyalty point redemption",
            "expected_agent": "SupportAgent"
        },
        {
            "name": "Simple flight search",
            "message": "Find me a flight to London",
            "expected_agent": "SearchAgent"
        },
        {
            "name": "Booking request",
            "message": "I want to book flight BA249",
            "expected_agent": "BookingAgent"
        },
        {
            "name": "Pricing question",
            "message": "What is the price of a flight to New York?",
            "expected_agent": "PricingAgent"
        }
    ]

    for scenario in test_scenarios:
        print(f"\n--- Testing Scenario: {scenario['name']} ---")
        context = RunContext(
            user_id="demo_user_123",
            session_id="advanced_demo",
            metadata={"demo": True},
            permissions=["user"]
        )
        
        message = Message(
            role=ContentRole.USER,
            content=scenario["message"]
        )
        
        print(f"📝 User Request: {message.content}")
        
        result = await execute_multi_agent(
            config=config,
            message=message,
            context=context,
            session_state={},
            model_provider=model_provider
        )
        
        print(f"🎯 Intelligent Selection Result: {result.content.content}")
        assert result.content.content is not None
    
    print("\n🎉 Advanced Multi-Agent System Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Intelligent agent selection with keyword matching")
    print("✅ Comprehensive schema validation with business rules")
    print("✅ Parallel execution with intelligent response merging")
    print("✅ Rule-based coordination with flexible conditions")
    print("✅ Enhanced tool ecosystem with validation")


if __name__ == "__main__":
    asyncio.run(main())
