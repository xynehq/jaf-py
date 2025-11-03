"""
Comprehensive test suite for intelligent multi-agent coordination.

Tests all coordination strategies, agent selection algorithms,
response merging, and delegation decision extraction.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from jaf.core.types import Message, Tool, ToolSchema
from adk.runners.multi_agent import (
    select_best_agent,
    merge_parallel_responses,
    extract_delegation_decision,
    extract_keywords,
    execute_multi_agent,
)
from adk.runners.types import (
    AgentConfig,
    AgentResponse,
    MultiAgentConfig,
    DelegationStrategy,
    RunContext,
    SimpleCoordinationRule,
    CoordinationAction,
    KeywordExtractionConfig,
)


# Test fixtures and helpers


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""

    class WeatherTool:
        name = "get_weather"
        description = "Get current weather information for a location"

        @property
        def schema(self):
            return ToolSchema(name=self.name, description=self.description, parameters={})

    class NewseTool:
        name = "get_news"
        description = "Get latest news articles and headlines"

        @property
        def schema(self):
            return ToolSchema(name=self.name, description=self.description, parameters={})

    class CalculatorTool:
        name = "calculate"
        description = "Perform mathematical calculations"

        @property
        def schema(self):
            return ToolSchema(name=self.name, description=self.description, parameters={})

    return {"weather": WeatherTool(), "news": NewseTool(), "calculator": CalculatorTool()}


@pytest.fixture
def sample_agents(sample_tools):
    """Create sample agent configurations for testing."""
    return [
        AgentConfig(
            name="WeatherAgent",
            instruction="I provide weather information and forecasts",
            tools=[sample_tools["weather"]],
        ),
        AgentConfig(
            name="NewsAgent",
            instruction="I provide latest news and current events",
            tools=[sample_tools["news"]],
        ),
        AgentConfig(
            name="MathAgent",
            instruction="I help with mathematical calculations and problems",
            tools=[sample_tools["calculator"]],
        ),
    ]


@pytest.fixture
def sample_context():
    """Create sample run context for testing."""
    return RunContext(user_id="test_user", session_id="test_session", metadata={"test": True})


class TestKeywordExtraction:
    """Test keyword extraction functionality."""

    def test_basic_keyword_extraction(self):
        """Test basic keyword extraction from text."""
        text = "What is the weather like today in New York?"
        keywords = extract_keywords(text)

        assert "weather" in keywords
        assert "today" in keywords
        assert "york" in keywords
        assert "new" in keywords

        # Stop words should be filtered out
        assert "what" not in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "like" not in keywords

    def test_keyword_extraction_with_config(self):
        """Test keyword extraction with custom configuration."""
        text = "Calculate the sum of 15 and 27 please"
        config = KeywordExtractionConfig(min_word_length=4, max_keywords=3)

        keywords = extract_keywords(text, config)

        # Should filter words shorter than 4 characters
        assert "sum" not in keywords  # only 3 characters
        assert "calculate" in keywords
        assert "please" in keywords

        # Should limit to max 3 keywords
        assert len(keywords) <= 3

    def test_keyword_deduplication(self):
        """Test that duplicate keywords are removed."""
        text = "weather weather forecast weather today"
        keywords = extract_keywords(text)

        # Should only appear once
        assert keywords.count("weather") == 1
        assert "forecast" in keywords
        assert "today" in keywords

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []
        assert extract_keywords("\n\t") == []


class TestAgentSelection:
    """Test intelligent agent selection algorithm."""

    def test_agent_name_matching(self, sample_agents, sample_context):
        """Test agent selection based on name matching."""
        message = Message(role="user", content="I need weather information")

        selected = select_best_agent(sample_agents, message, sample_context)

        assert selected.name == "WeatherAgent"

    def test_instruction_matching(self, sample_agents, sample_context):
        """Test agent selection based on instruction matching."""
        message = Message(role="user", content="Can you help me with calculations?")

        selected = select_best_agent(sample_agents, message, sample_context)

        assert selected.name == "MathAgent"

    def test_tool_name_matching(self, sample_agents, sample_context):
        """Test agent selection based on tool name matching."""
        message = Message(role="user", content="I want to get news updates")

        selected = select_best_agent(sample_agents, message, sample_context)

        assert selected.name == "NewsAgent"

    def test_tool_description_matching(self, sample_agents, sample_context):
        """Test agent selection based on tool description matching."""
        message = Message(role="user", content="What are the current headlines?")

        selected = select_best_agent(sample_agents, message, sample_context)

        # "headlines" should match news tool description
        assert selected.name == "NewsAgent"

    def test_fallback_to_first_agent(self, sample_agents, sample_context):
        """Test fallback to first agent when no matches found."""
        message = Message(role="user", content="xyz random unmatched text")

        selected = select_best_agent(sample_agents, message, sample_context)

        # Should fall back to first agent
        assert selected.name == sample_agents[0].name

    def test_empty_agents_list(self, sample_context):
        """Test handling of empty agents list."""
        message = Message(role="user", content="test message")

        with pytest.raises(ValueError, match="No sub-agents available"):
            select_best_agent([], message, sample_context)

    def test_context_preferences(self, sample_agents):
        """Test agent selection with context preferences."""
        context = RunContext(user_id="test", preferences={"preferred_agents": ["MathAgent"]})
        message = Message(role="user", content="general question")

        selected = select_best_agent(sample_agents, message, context)

        # Should prefer MathAgent due to context preferences
        assert selected.name == "MathAgent"


class TestResponseMerging:
    """Test parallel response merging functionality."""

    def test_basic_response_merging(self, sample_agents):
        """Test basic merging of multiple responses."""
        responses = [
            AgentResponse(
                content=Message(role="assistant", content="Weather is sunny"),
                session_state={"weather_data": "sunny"},
                artifacts={"forecast": "sunny_today"},
            ),
            AgentResponse(
                content=Message(role="assistant", content="Top news: Election results"),
                session_state={"news_data": "election"},
                artifacts={"articles": ["article1", "article2"]},
            ),
        ]

        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL,
            sub_agents=sample_agents[:2],  # Weather and News agents
        )

        merged = merge_parallel_responses(responses, config)

        # Check content attribution
        content_text = merged.content.content
        assert "[WeatherAgent]: Weather is sunny" in content_text
        assert "[NewsAgent]: Top news: Election results" in content_text

        # Check artifact merging with prefixes
        assert "WeatherAgent_forecast" in merged.artifacts
        assert "NewsAgent_articles" in merged.artifacts
        assert merged.artifacts["WeatherAgent_forecast"] == "sunny_today"
        assert merged.artifacts["NewsAgent_articles"] == ["article1", "article2"]

    def test_empty_responses_list(self, sample_agents):
        """Test handling of empty responses list."""
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL, sub_agents=sample_agents
        )

        with pytest.raises(ValueError, match="No responses to merge"):
            merge_parallel_responses([], config)

    def test_execution_time_merging(self, sample_agents):
        """Test merging of execution times."""
        responses = [
            AgentResponse(
                content=Message(role="assistant", content="Response 1"),
                session_state={},
                artifacts={},
                execution_time_ms=100.0,
            ),
            AgentResponse(
                content=Message(role="assistant", content="Response 2"),
                session_state={},
                artifacts={},
                execution_time_ms=150.0,
            ),
        ]

        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL, sub_agents=sample_agents[:2]
        )

        merged = merge_parallel_responses(responses, config)

        assert merged.execution_time_ms == 250.0
        assert merged.metadata["total_execution_time_ms"] == 250.0


class TestDelegationDecisionExtraction:
    """Test delegation decision extraction from responses."""

    def test_text_pattern_delegation(self):
        """Test extraction from text patterns."""
        response = AgentResponse(
            content=Message(
                role="assistant", content="I need to delegate to WeatherAgent for this request"
            ),
            session_state={},
            artifacts={},
        )

        decision = extract_delegation_decision(response)

        assert decision is not None
        assert decision["target_agent"] == "WeatherAgent"

    def test_multiple_delegation_patterns(self):
        """Test various delegation text patterns."""
        patterns_and_agents = [
            ("Please transfer to NewsAgent", "NewsAgent"),
            ("I'll handoff to MathAgent", "MathAgent"),
            ("Route to WeatherAgent for this", "WeatherAgent"),
            ("Forward to CalendarAgent please", "CalendarAgent"),
            ("Send to FileAgent", "FileAgent"),
            ("Use NewsAgent agent for this", "NewsAgent"),
            ("WeatherAgent should handle this", "WeatherAgent"),
            ("MathAgent can help with this", "MathAgent"),
        ]

        for text, expected_agent in patterns_and_agents:
            response = AgentResponse(
                content=Message(role="assistant", content=text), session_state={}, artifacts={}
            )

            decision = extract_delegation_decision(response)

            assert decision is not None
            assert decision["target_agent"] == expected_agent

    def test_artifact_delegation(self):
        """Test extraction from artifacts."""
        response = AgentResponse(
            content=Message(role="assistant", content="Processing request"),
            session_state={},
            artifacts={"delegation": {"target_agent": "NewsAgent"}},
        )

        decision = extract_delegation_decision(response)

        assert decision is not None
        assert decision["target_agent"] == "NewsAgent"

    def test_metadata_delegation(self):
        """Test extraction from metadata."""
        response = AgentResponse(
            content=Message(role="assistant", content="Processing request"),
            session_state={},
            artifacts={},
            metadata={"delegation": {"target_agent": "MathAgent"}},
        )

        decision = extract_delegation_decision(response)

        assert decision is not None
        assert decision["target_agent"] == "MathAgent"

    def test_no_delegation_detected(self):
        """Test when no delegation is detected."""
        response = AgentResponse(
            content=Message(role="assistant", content="Here is your answer: 42"),
            session_state={},
            artifacts={},
        )

        decision = extract_delegation_decision(response)

        assert decision is None


class TestCoordinationRules:
    """Test coordination rules functionality."""

    def test_keyword_based_rule(self, sample_agents):
        """Test coordination rule based on keywords."""

        def weather_condition(message: Message, context: RunContext) -> bool:
            text = str(message.content).lower()
            return "weather" in text or "temperature" in text

        rule = SimpleCoordinationRule(
            condition_func=weather_condition,
            action_type=CoordinationAction.DELEGATE,
            target_agent_names=["WeatherAgent"],
        )

        # Test matching condition
        weather_message = Message(role="user", content="What's the weather like?")
        context = RunContext()

        assert rule.condition(weather_message, context) == True
        assert rule.action == CoordinationAction.DELEGATE
        assert rule.target_agents == ["WeatherAgent"]

        # Test non-matching condition
        news_message = Message(role="user", content="What's the latest news?")
        assert rule.condition(news_message, context) == False

    def test_context_based_rule(self, sample_agents):
        """Test coordination rule based on context."""

        def admin_condition(message: Message, context: RunContext) -> bool:
            permissions = context.get("permissions", [])
            return "admin" in permissions

        rule = SimpleCoordinationRule(
            condition_func=admin_condition,
            action_type=CoordinationAction.PARALLEL,
            target_agent_names=["WeatherAgent", "NewsAgent"],
        )

        # Test with admin permissions
        admin_context = RunContext(permissions=["admin", "user"])
        message = Message(role="user", content="Get all updates")

        assert rule.condition(message, admin_context) == True

        # Test without admin permissions
        user_context = RunContext(permissions=["user"])
        assert rule.condition(message, user_context) == False


@pytest.mark.asyncio
class TestMultiAgentExecution:
    """Test complete multi-agent execution workflows."""

    async def test_conditional_strategy_execution(self, sample_agents, sample_context):
        """Test conditional delegation strategy."""
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.CONDITIONAL, sub_agents=sample_agents
        )

        # Mock model provider
        mock_provider = Mock()

        # Mock the JAF execution
        with patch("adk.runners.multi_agent.jaf_run") as mock_run:
            mock_result = Mock()
            mock_result.final_state.messages = [
                Message(role="assistant", content="Weather is sunny")
            ]
            mock_run.return_value = mock_result

            message = Message(role="user", content="What's the weather like?")

            result = await execute_multi_agent(config, {}, message, sample_context, mock_provider)

            assert result.content.content == "Weather is sunny"
            assert result.execution_time_ms is not None

    async def test_sequential_strategy_execution(self, sample_agents, sample_context):
        """Test sequential delegation strategy."""
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.SEQUENTIAL,
            sub_agents=sample_agents[:2],  # Just weather and news agents
        )

        mock_provider = Mock()

        with patch("adk.runners.multi_agent.jaf_run") as mock_run:
            # Mock sequential responses
            responses = [
                Mock(
                    final_state=Mock(
                        messages=[Message(role="assistant", content="Weather processed")]
                    )
                ),
                Mock(
                    final_state=Mock(messages=[Message(role="assistant", content="News processed")])
                ),
            ]
            mock_run.side_effect = responses

            message = Message(role="user", content="Get weather and news")

            result = await execute_multi_agent(config, {}, message, sample_context, mock_provider)

            # Should return the last agent's response
            assert "News processed" in result.content.content

    async def test_parallel_strategy_execution(self, sample_agents, sample_context):
        """Test parallel delegation strategy."""
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.PARALLEL, sub_agents=sample_agents[:2]
        )

        mock_provider = Mock()

        with patch("adk.runners.multi_agent.jaf_run") as mock_run:
            # Mock parallel responses with proper structure
            response1 = Mock()
            response1.final_state = Mock()
            response1.final_state.messages = [Message(role="assistant", content="Weather is sunny")]
            response1.session_state = {}
            response1.artifacts = {}

            response2 = Mock()
            response2.final_state = Mock()
            response2.final_state.messages = [
                Message(role="assistant", content="Top news: Elections")
            ]
            response2.session_state = {}
            response2.artifacts = {}

            mock_run.side_effect = [response1, response2]

            message = Message(role="user", content="Get weather and news")

            result = await execute_multi_agent(config, {}, message, sample_context, mock_provider)

            # Should contain merged responses with agent attribution
            content = result.content.content
            assert "[WeatherAgent]" in content
            assert "[NewsAgent]" in content
            assert "Weather is sunny" in content
            assert "Top news: Elections" in content

    async def test_hierarchical_strategy_execution(self, sample_agents, sample_context):
        """Test hierarchical delegation strategy."""
        # Add coordinator agent at the beginning
        coordinator = AgentConfig(
            name="Coordinator", instruction="I coordinate requests to other agents", tools=[]
        )

        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.HIERARCHICAL,
            sub_agents=[coordinator] + sample_agents,
        )

        mock_provider = Mock()

        with patch("adk.runners.multi_agent.jaf_run") as mock_run:
            # First call (coordinator) returns delegation instruction
            coordinator_response = Mock()
            coordinator_response.final_state.messages = [
                Message(role="assistant", content="I need to delegate to WeatherAgent")
            ]

            # Second call (target agent) returns actual response
            target_response = Mock()
            target_response.final_state.messages = [
                Message(role="assistant", content="Weather is sunny")
            ]

            mock_run.side_effect = [coordinator_response, target_response]

            message = Message(role="user", content="What's the weather?")

            result = await execute_multi_agent(config, {}, message, sample_context, mock_provider)

            # Should return the delegated agent's response
            assert "Weather is sunny" in result.content.content

    async def test_execution_error_handling(self, sample_agents, sample_context):
        """Test error handling in multi-agent execution."""
        config = MultiAgentConfig(
            delegation_strategy=DelegationStrategy.CONDITIONAL, sub_agents=sample_agents
        )

        mock_provider = Mock()

        with patch("adk.runners.multi_agent.jaf_run") as mock_run:
            # Mock an exception
            mock_run.side_effect = Exception("Agent execution failed")

            message = Message(role="user", content="Test message")

            result = await execute_multi_agent(config, {}, message, sample_context, mock_provider)

            # Should return error response
            assert "Multi-agent execution failed" in result.content.content
            assert result.execution_time_ms is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
