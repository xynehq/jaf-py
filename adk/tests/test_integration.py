"""
Comprehensive integration tests for ADK enhanced capabilities.

Tests the integration between schema validation, multi-agent coordination,
and the existing JAF framework to ensure production readiness.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from jaf.core.types import Message, Tool, ToolSchema, ContentRole
from adk.schemas.validation import validate_schema
from adk.schemas.types import JsonSchema
from adk.runners.multi_agent import (
    execute_multi_agent,
    select_best_agent,
    merge_parallel_responses
)
from adk.runners.types import (
    AgentConfig,
    MultiAgentConfig,
    DelegationStrategy,
    RunContext,
    SimpleCoordinationRule,
    CoordinationAction
)


class TestSchemaIntegration:
    """Test integration of enhanced schema validation with JAF tools."""
    
    def test_complex_nested_schema_validation(self):
        """Test validation of complex nested schemas."""
        schema: JsonSchema = {
            'type': 'object',
            'properties': {
                'user_profile': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'minLength': 2,
                            'maxLength': 50,
                            'pattern': r'^[A-Za-z\s]+$'
                        },
                        'email': {
                            'type': 'string',
                            'format': 'email'
                        },
                        'age': {
                            'type': 'integer',
                            'minimum': 18,
                            'maximum': 120
                        }
                    },
                    'required': ['name', 'email']
                },
                'preferences': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'minLength': 1
                    },
                    'minItems': 1,
                    'maxItems': 10,
                    'uniqueItems': True
                },
                'settings': {
                    'type': 'object',
                    'properties': {
                        'notifications': {'type': 'boolean'},
                        'theme': {
                            'type': 'string',
                            'enum': ['light', 'dark', 'auto']
                        }
                    },
                    'additionalProperties': False
                }
            },
            'required': ['user_profile']
        }
        
        # Valid complex data
        valid_data = {
            'user_profile': {
                'name': 'John Doe',
                'email': 'john.doe@example.com',
                'age': 30
            },
            'preferences': ['email_notifications', 'sms_alerts'],
            'settings': {
                'notifications': True,
                'theme': 'dark'
            }
        }
        
        result = validate_schema(valid_data, schema)
        assert result.success
        assert len(result.errors) == 0
        
        # Invalid nested data
        invalid_data = {
            'user_profile': {
                'name': 'J',  # Too short
                'email': 'invalid-email',  # Invalid format
                'age': 150  # Too old
            },
            'preferences': [],  # Too few items
            'settings': {
                'notifications': True,
                'theme': 'invalid',  # Not in enum
                'extra_field': 'not_allowed'  # Additional property
            }
        }
        
        result = validate_schema(invalid_data, schema)
        assert not result.success
        assert len(result.errors) > 0
        
        # Check that all validation errors are caught
        error_text = ' '.join(result.errors).lower()
        assert 'length' in error_text  # Name too short
        assert 'email' in error_text  # Invalid email
        assert 'maximum' in error_text  # Age too high
        assert 'items' in error_text  # Preferences array issues
        assert 'enum' in error_text  # Invalid theme
    
    def test_format_validation_edge_cases(self):
        """Test edge cases in format validation."""
        # Test email format variations
        email_schema: JsonSchema = {'type': 'string', 'format': 'email'}
        
        valid_emails = [
            'test@example.com',
            'user.name+tag@domain.co.uk',
            'test.email-with-dash@example.org',
            'x@y.z'
        ]
        
        for email in valid_emails:
            result = validate_schema(email, email_schema)
            assert result.success, f"Email {email} should be valid"
        
        invalid_emails = [
            'plainaddress',
            '@missingdomain.com',
            'missing@.com',
            'missing@domain',
            'spaces @domain.com',
            'double@@domain.com'
        ]
        
        for email in invalid_emails:
            result = validate_schema(email, email_schema)
            assert not result.success, f"Email {email} should be invalid"
        
        # Test date format validation
        date_schema: JsonSchema = {'type': 'string', 'format': 'date'}
        
        valid_dates = ['2024-01-01', '2024-12-31', '2024-02-29']  # Leap year
        for date in valid_dates:
            result = validate_schema(date, date_schema)
            assert result.success, f"Date {date} should be valid"
        
        invalid_dates = ['2024-13-01', '2024-02-30', '24-01-01', '2024/01/01']
        for date in invalid_dates:
            result = validate_schema(date, date_schema)
            assert not result.success, f"Date {date} should be invalid"


class TestMultiAgentIntegration:
    """Test integration of multi-agent coordination with JAF framework."""
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        class MockTool:
            def __init__(self, name, description, category="general"):
                self.name = name
                self.description = description
                self.category = category
            
            @property
            def schema(self):
                return ToolSchema(name=self.name, description=self.description, parameters={})
            
            async def execute(self, args, context):
                return {"status": "success", "tool": self.name, "result": f"Executed {self.name}"}
        
        return {
            'search': MockTool('search_flights', 'Search for available flights', 'search'),
            'book': MockTool('book_flight', 'Book a flight reservation', 'booking'),
            'weather': MockTool('check_weather', 'Check weather conditions', 'weather'),
            'loyalty': MockTool('manage_loyalty', 'Manage loyalty points', 'loyalty')
        }
    
    @pytest.fixture
    def test_agents(self, mock_tools):
        """Create test agent configurations."""
        return [
            AgentConfig(
                name="FlightSearchAgent",
                instruction="I help find and compare flights",
                tools=[mock_tools['search'], mock_tools['weather']],
                metadata={
                    "specialization": ["search", "weather"],
                    "keywords": ["flight", "search", "find", "weather"]
                }
            ),
            AgentConfig(
                name="BookingAgent",
                instruction="I handle flight bookings and reservations",
                tools=[mock_tools['book'], mock_tools['loyalty']],
                metadata={
                    "specialization": ["booking", "loyalty"],
                    "keywords": ["book", "reserve", "confirm", "loyalty"]
                }
            ),
            AgentConfig(
                name="WeatherAgent", 
                instruction="I provide weather information",
                tools=[mock_tools['weather']],
                metadata={
                    "specialization": ["weather"],
                    "keywords": ["weather", "forecast", "temperature"]
                }
            )
        ]
    
    def test_intelligent_agent_selection_integration(self, test_agents):
        """Test intelligent agent selection with realistic scenarios."""
        context = RunContext(user_id="test_user", session_id="test_session")
        
        # Test flight search scenario
        flight_message = Message(
            role=ContentRole.USER,
            content="I need to find flights from LAX to JFK for next week"
        )
        
        selected = select_best_agent(test_agents, flight_message, context)
        assert selected.name == "FlightSearchAgent"
        
        # Test booking scenario
        booking_message = Message(
            role=ContentRole.USER,
            content="I want to book flight AA123 and use my loyalty points"
        )
        
        selected = select_best_agent(test_agents, booking_message, context)
        assert selected.name == "BookingAgent"
        
        # Test weather scenario
        weather_message = Message(
            role=ContentRole.USER,
            content="What's the weather forecast for my destination?"
        )
        
        selected = select_best_agent(test_agents, weather_message, context)
        assert selected.name == "WeatherAgent"
        
        # Test ambiguous scenario (should use scoring)
        ambiguous_message = Message(
            role=ContentRole.USER,
            content="Can you help me with my trip planning?"
        )
        
        selected = select_best_agent(test_agents, ambiguous_message, context)
        # Should fall back to first agent when no clear winner
        assert selected.name == "FlightSearchAgent"
    
    def test_metadata_based_agent_selection(self, test_agents):
        """Test agent selection using metadata and keywords."""
        context = RunContext(
            user_id="test_user",
            preferences={"preferred_agents": ["BookingAgent"]}
        )
        
        # Test preference-based selection
        general_message = Message(
            role=ContentRole.USER,
            content="I need some help with travel"
        )
        
        selected = select_best_agent(test_agents, general_message, context)
        # Should prefer BookingAgent due to context preferences
        assert selected.name == "BookingAgent"
        
        # Test keyword override of preferences
        specific_message = Message(
            role=ContentRole.USER,
            content="What's the weather like today?"
        )
        
        selected = select_best_agent(test_agents, specific_message, context)
        # Specific weather keywords should override preferences
        assert selected.name == "WeatherAgent"
    
    def test_coordination_rules_integration(self, test_agents):
        """Test coordination rules with realistic business logic."""
        
        # Rule 1: VIP customers get priority booking agent
        def vip_condition(message: Message, context: RunContext) -> bool:
            user_metadata = context.get('metadata', {})
            return user_metadata.get('vip_status') == True
        
        vip_rule = SimpleCoordinationRule(
            condition_func=vip_condition,
            action_type=CoordinationAction.DELEGATE,
            target_agent_names=["BookingAgent"]
        )
        
        # Rule 2: Weather inquiries during travel season go to weather specialist
        def weather_season_condition(message: Message, context: RunContext) -> bool:
            text = str(message.content).lower()
            is_weather_query = "weather" in text or "forecast" in text
            season_metadata = context.get('metadata', {})
            is_travel_season = season_metadata.get('travel_season') == True
            return is_weather_query and is_travel_season
        
        weather_rule = SimpleCoordinationRule(
            condition_func=weather_season_condition,
            action_type=CoordinationAction.DELEGATE,
            target_agent_names=["WeatherAgent"]
        )
        
        # Test VIP rule
        vip_context = RunContext(
            user_id="vip_user",
            metadata={"vip_status": True}
        )
        
        message = Message(role=ContentRole.USER, content="I need help with anything")
        assert vip_rule.condition(message, vip_context) == True
        assert vip_rule.target_agents == ["BookingAgent"]
        
        # Test weather season rule
        season_context = RunContext(
            user_id="regular_user",
            metadata={"travel_season": True}
        )
        
        weather_message = Message(role=ContentRole.USER, content="What's the weather forecast?")
        assert weather_rule.condition(weather_message, season_context) == True
        assert weather_rule.target_agents == ["WeatherAgent"]
        
        # Test rule that doesn't match
        off_season_context = RunContext(
            user_id="regular_user",
            metadata={"travel_season": False}
        )
        
        assert weather_rule.condition(weather_message, off_season_context) == False
    
    def test_response_merging_integration(self, test_agents):
        """Test response merging with realistic agent responses."""
        from adk.runners.types import AgentResponse
        
        # Mock realistic agent responses
        search_response = AgentResponse(
            content=Message(role="assistant", content="Found 3 flights: AA123 ($299), UA456 ($325), DL789 ($310)"),
            session_state={"search_results": ["AA123", "UA456", "DL789"]},
            artifacts={
                "flights": [
                    {"flight": "AA123", "price": 299, "airline": "American"},
                    {"flight": "UA456", "price": 325, "airline": "United"},
                    {"flight": "DL789", "price": 310, "airline": "Delta"}
                ]
            },
            execution_time_ms=150.0
        )
        
        weather_response = AgentResponse(
            content=Message(role="assistant", content="Weather in destination: 75°F, partly cloudy, good for travel"),
            session_state={"weather_data": {"temp": 75, "conditions": "partly cloudy"}},
            artifacts={
                "forecast": {
                    "temperature": "75°F",
                    "conditions": "partly cloudy", 
                    "travel_advisory": "good"
                }
            },
            execution_time_ms=100.0
        )
        
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL,
            sub_agents=[test_agents[0], test_agents[2]]  # Search and Weather agents
        )
        
        merged = merge_parallel_responses([search_response, weather_response], config)
        
        # Check content attribution
        content = merged.content.content
        assert "[FlightSearchAgent]:" in content
        assert "[WeatherAgent]:" in content
        assert "Found 3 flights" in content
        assert "75°F, partly cloudy" in content
        
        # Check artifact merging
        assert "FlightSearchAgent_flights" in merged.artifacts
        assert "WeatherAgent_forecast" in merged.artifacts
        assert len(merged.artifacts["FlightSearchAgent_flights"]) == 3
        assert merged.artifacts["WeatherAgent_forecast"]["temperature"] == "75°F"
        
        # Check execution time aggregation
        assert merged.execution_time_ms == 250.0
        assert merged.metadata["total_execution_time_ms"] == 250.0
        assert merged.metadata["agent_count"] == 2


class TestCompleteWorkflowIntegration:
    """Test complete end-to-end workflows combining all ADK enhancements."""
    
    @pytest.mark.asyncio
    async def test_flight_booking_workflow_integration(self):
        """Test complete flight booking workflow with validation and coordination."""
        
        # Step 1: Schema validation for flight search
        search_schema: JsonSchema = {
            'type': 'object',
            'properties': {
                'origin': {
                    'type': 'string',
                    'pattern': r'^[A-Z]{3}$',
                    'description': 'Origin airport code (3 letters)'
                },
                'destination': {
                    'type': 'string', 
                    'pattern': r'^[A-Z]{3}$',
                    'description': 'Destination airport code (3 letters)'
                },
                'departure_date': {
                    'type': 'string',
                    'format': 'date'
                },
                'passengers': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 9
                }
            },
            'required': ['origin', 'destination', 'departure_date']
        }
        
        # Valid search data
        search_data = {
            'origin': 'LAX',
            'destination': 'JFK',
            'departure_date': '2024-01-15',
            'passengers': 2
        }
        
        validation_result = validate_schema(search_data, search_schema)
        assert validation_result.success
        
        # Step 2: Agent selection for search request
        search_agent = AgentConfig(
            name="SearchAgent",
            instruction="Find flights",
            tools=[],
            metadata={"keywords": ["flight", "search", "find"]}
        )
        
        booking_agent = AgentConfig(
            name="BookingAgent", 
            instruction="Book flights",
            tools=[],
            metadata={"keywords": ["book", "reserve"]}
        )
        
        message = Message(
            role=ContentRole.USER,
            content="I need to find flights from LAX to JFK"
        )
        
        context = RunContext(user_id="workflow_test")
        selected = select_best_agent([search_agent, booking_agent], message, context)
        assert selected.name == "SearchAgent"
        
        # Step 3: Booking validation
        booking_schema: JsonSchema = {
            'type': 'object',
            'properties': {
                'flight_id': {
                    'type': 'string',
                    'pattern': r'^[A-Z]{2}\d{3,4}$'
                },
                'passenger_details': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'minLength': 2,
                                'maxLength': 50
                            },
                            'email': {
                                'type': 'string',
                                'format': 'email'
                            }
                        },
                        'required': ['name', 'email']
                    },
                    'minItems': 1,
                    'maxItems': 9
                }
            },
            'required': ['flight_id', 'passenger_details']
        }
        
        booking_data = {
            'flight_id': 'AA123',
            'passenger_details': [
                {
                    'name': 'John Doe',
                    'email': 'john.doe@example.com'
                },
                {
                    'name': 'Jane Doe', 
                    'email': 'jane.doe@example.com'
                }
            ]
        }
        
        booking_validation = validate_schema(booking_data, booking_schema)
        assert booking_validation.success
        
        # Step 4: Booking agent selection
        booking_message = Message(
            role=ContentRole.USER,
            content="I want to book flight AA123"
        )
        
        selected = select_best_agent([search_agent, booking_agent], booking_message, context)
        assert selected.name == "BookingAgent"
    
    def test_error_handling_integration(self):
        """Test error handling across validation and coordination."""
        
        # Test schema validation errors
        invalid_schema: JsonSchema = {
            'type': 'object',
            'properties': {
                'email': {'type': 'string', 'format': 'email'},
                'age': {'type': 'integer', 'minimum': 0, 'maximum': 150}
            },
            'required': ['email']
        }
        
        invalid_data = {
            'email': 'not-an-email',
            'age': 200
        }
        
        result = validate_schema(invalid_data, invalid_schema)
        assert not result.success
        assert len(result.errors) >= 2  # Email format and age maximum
        
        # Test agent selection with empty list
        context = RunContext(user_id="error_test")
        message = Message(role=ContentRole.USER, content="test")
        
        with pytest.raises(ValueError, match="No sub-agents available"):
            select_best_agent([], message, context)
        
        # Test response merging with empty list
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL,
            sub_agents=[]
        )
        
        with pytest.raises(ValueError, match="No responses to merge"):
            merge_parallel_responses([], config)
    
    def test_performance_and_scalability_integration(self):
        """Test performance characteristics of integrated system."""
        import time
        
        # Test schema validation performance with large data
        large_schema: JsonSchema = {
            'type': 'object',
            'properties': {
                'users': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string', 'pattern': r'^user_\d+$'},
                            'email': {'type': 'string', 'format': 'email'},
                            'preferences': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'uniqueItems': True
                            }
                        },
                        'required': ['id', 'email']
                    },
                    'minItems': 1,
                    'maxItems': 1000
                }
            }
        }
        
        # Generate large but valid data
        large_data = {
            'users': [
                {
                    'id': f'user_{i}',
                    'email': f'user{i}@example.com',
                    'preferences': [f'pref_{i}_1', f'pref_{i}_2']
                }
                for i in range(100)
            ]
        }
        
        start_time = time.time()
        result = validate_schema(large_data, large_schema)
        validation_time = time.time() - start_time
        
        assert result.success
        assert validation_time < 1.0  # Should validate 100 users in under 1 second
        
        # Test agent selection performance with many agents
        many_agents = [
            AgentConfig(
                name=f"Agent{i}",
                instruction=f"I am agent number {i}",
                tools=[],
                metadata={"keywords": [f"keyword{i}", f"task{i}"]}
            )
            for i in range(50)
        ]
        
        context = RunContext(user_id="performance_test")
        message = Message(role=ContentRole.USER, content="keyword25 task25")
        
        start_time = time.time()
        selected = select_best_agent(many_agents, message, context)
        selection_time = time.time() - start_time
        
        assert selected.name == "Agent25"  # Should find the matching agent
        assert selection_time < 0.1  # Should select from 50 agents in under 100ms


if __name__ == '__main__':
    pytest.main([__file__, '-v'])