"""
Intelligent multi-agent coordination and execution.

This module implements sophisticated multi-agent orchestration including:
- Intelligent agent selection based on keyword matching
- Advanced response merging for parallel execution  
- Hierarchical coordination with delegation decision extraction
- Rule-based coordination with flexible condition evaluation
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

# Import JAF core types
from jaf.core.types import Message, Agent, RunState, RunConfig, generate_run_id, generate_trace_id
from jaf.core.engine import run as jaf_run

from .types import (
    MultiAgentConfig,
    AgentConfig,
    AgentResponse,
    AgentSelectionScore,
    RunContext,
    DelegationStrategy,
    CoordinationAction,
    KeywordExtractionConfig
)


async def execute_multi_agent(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """
    Main entry point for multi-agent execution.
    
    Dispatches to the appropriate execution strategy based on the
    delegation strategy specified in the configuration.
    
    Args:
        config: Multi-agent configuration
        session_state: Current session state
        message: User message to process
        context: Execution context
        model_provider: Model provider for agent execution
        
    Returns:
        AgentResponse with the result of multi-agent execution
    """
    start_time = time.time()
    
    try:
        if config.delegation_strategy == DelegationStrategy.SEQUENTIAL:
            result = await _execute_sequential_agents(
                config, session_state, message, context, model_provider
            )
        elif config.delegation_strategy == DelegationStrategy.PARALLEL:
            result = await _execute_parallel_agents(
                config, session_state, message, context, model_provider
            )
        elif config.delegation_strategy == DelegationStrategy.CONDITIONAL:
            result = await _execute_conditional_agents(
                config, session_state, message, context, model_provider
            )
        elif config.delegation_strategy == DelegationStrategy.HIERARCHICAL:
            result = await _execute_hierarchical_agents(
                config, session_state, message, context, model_provider
            )
        else:
            raise ValueError(f"Unknown delegation strategy: {config.delegation_strategy}")
        
        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time
        
        return result
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        error_message = Message(
            role='assistant',
            content=f"Multi-agent execution failed: {str(e)}"
        )
        
        return AgentResponse(
            content=error_message,
            session_state=session_state,
            artifacts={},
            execution_time_ms=execution_time
        )


def select_best_agent(
    sub_agents: List[AgentConfig],
    message: Message,
    context: RunContext,
    extraction_config: Optional[KeywordExtractionConfig] = None
) -> AgentConfig:
    """
    Intelligently select the best agent based on message content and context.
    
    Uses keyword matching, tool relevance, and context analysis to score
    each agent and select the most appropriate one for the given message.
    
    Args:
        sub_agents: List of available agent configurations
        message: User message to analyze
        context: Execution context
        extraction_config: Configuration for keyword extraction
        
    Returns:
        The agent configuration with the highest relevance score
    """
    if not sub_agents:
        raise ValueError("No sub-agents available for selection")
    
    # Extract message text for analysis
    message_text = _extract_message_text(message).lower()
    
    # Extract keywords from the message
    keywords = extract_keywords(message_text, extraction_config)
    
    # Score each agent
    agent_scores: List[AgentSelectionScore] = []
    
    for agent in sub_agents:
        score = AgentSelectionScore(agent=agent, score=0.0, reasoning=[])
        
        # Score based on agent name relevance
        agent_name_lower = agent.name.lower()
        for keyword in keywords:
            if keyword in agent_name_lower:
                score.add_score(3.0, f"Agent name matches keyword '{keyword}'")
        
        # Score based on instruction relevance
        instruction_lower = agent.instruction.lower()
        for keyword in keywords:
            if keyword in instruction_lower:
                score.add_score(2.0, f"Instruction matches keyword '{keyword}'")
        
        # Score based on tool relevance
        for tool in agent.tools:
            tool_name_lower = getattr(tool, 'name', '').lower()
            tool_desc_lower = getattr(tool, 'description', '').lower()
            
            for keyword in keywords:
                if keyword in tool_name_lower:
                    score.add_score(2.0, f"Tool name '{tool.name}' matches keyword '{keyword}'")
                elif keyword in tool_desc_lower:
                    score.add_score(1.0, f"Tool description matches keyword '{keyword}'")
        
        # Score based on context (e.g., user permissions, preferences)
        if context.get('preferences'):
            preferences = context['preferences']
            if 'preferred_agents' in preferences and agent.name in preferences['preferred_agents']:
                score.add_score(1.5, "Agent is in user's preferred agents")
        
        # Score based on metadata relevance
        if agent.metadata:
            for keyword in keywords:
                metadata_text = str(agent.metadata).lower()
                if keyword in metadata_text:
                    score.add_score(0.5, f"Metadata matches keyword '{keyword}'")
        
        agent_scores.append(score)
    
    # Sort by score and return the best agent
    agent_scores.sort(key=lambda x: x.score, reverse=True)
    
    # If no agent has a positive score, return the first agent as fallback
    best_score = agent_scores[0]
    if best_score.score <= 0:
        return sub_agents[0]
    
    return best_score.agent


def merge_parallel_responses(
    responses: List[AgentResponse],
    config: MultiAgentConfig
) -> AgentResponse:
    """
    Intelligently merge responses from parallel agent execution.
    
    Combines content from multiple agents with proper attribution,
    merges artifacts with conflict resolution, and preserves metadata.
    
    Args:
        responses: List of agent responses to merge
        config: Multi-agent configuration containing agent information
        
    Returns:
        Single merged AgentResponse
    """
    if not responses:
        raise ValueError("No responses to merge from parallel execution")
    
    # Merge content with agent attribution
    merged_content_parts = []
    merged_artifacts = {}
    merged_metadata = {}
    total_execution_time = 0.0
    
    for i, response in enumerate(responses):
        # Get agent name for attribution
        agent_name = config.sub_agents[i].name if i < len(config.sub_agents) else f"agent_{i}"
        
        # Add agent attribution to content
        response_text = response.content.content if hasattr(response.content, 'content') else str(response.content)
        attributed_content = f"[{agent_name}]: {response_text}"
        merged_content_parts.append(attributed_content)
        
        # Merge artifacts with agent prefixes to avoid conflicts
        for key, value in response.artifacts.items():
            prefixed_key = f"{agent_name}_{key}"
            merged_artifacts[prefixed_key] = value
        
        # Merge metadata
        if response.metadata:
            merged_metadata[agent_name] = response.metadata
        
        # Sum execution times
        if response.execution_time_ms:
            total_execution_time += response.execution_time_ms
    
    # Create merged content
    merged_content_text = "\n\n".join(merged_content_parts)
    merged_content = Message(role='assistant', content=merged_content_text)
    
    # Use the session state from the first response as base
    base_session_state = responses[0].session_state
    
    return AgentResponse(
        content=merged_content,
        session_state=base_session_state,
        artifacts=merged_artifacts,
        metadata={
            'parallel_execution': True,
            'agent_count': len(responses),
            'agent_metadata': merged_metadata,
            'total_execution_time_ms': total_execution_time
        },
        execution_time_ms=total_execution_time
    )


def extract_delegation_decision(response: AgentResponse) -> Optional[Dict[str, str]]:
    """
    Extract delegation decision from coordinator agent response.
    
    Analyzes the response content and metadata to determine if the
    coordinator agent wants to delegate to another agent.
    
    Args:
        response: Response from coordinator agent
        
    Returns:
        Dictionary with target agent name if delegation detected, None otherwise
    """
    # Extract response text for analysis
    response_text = ""
    if hasattr(response.content, 'content'):
        response_text = response.content.content
    else:
        response_text = str(response.content)
    
    response_text = response_text.lower()
    
    # Look for delegation patterns in text
    delegation_patterns = [
        r'delegate to (\w+)',
        r'transfer to (\w+)',
        r'handoff to (\w+)',
        r'route to (\w+)',
        r'forward to (\w+)',
        r'send to (\w+)',
        r'use (\w+) agent',
        r'(\w+) should handle',
        r'(\w+) can help'
    ]
    
    for pattern in delegation_patterns:
        match = re.search(pattern, response_text)
        if match:
            target_agent = match.group(1)
            if target_agent.endswith('agent'):
                base_name = target_agent[:-5]
                target_agent = base_name.capitalize() + 'Agent'
            else:
                target_agent = target_agent.capitalize()
            return {'target_agent': target_agent}
    
    # Check artifacts for delegation metadata
    if 'delegation' in response.artifacts:
        delegation_data = response.artifacts['delegation']
        if isinstance(delegation_data, dict) and 'target_agent' in delegation_data:
            return {'target_agent': delegation_data['target_agent']}
    
    # Check response metadata
    if response.metadata and 'delegation' in response.metadata:
        delegation_data = response.metadata['delegation']
        if isinstance(delegation_data, dict) and 'target_agent' in delegation_data:
            return {'target_agent': delegation_data['target_agent']}
    
    return None


async def execute_with_coordination_rules(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """
    Execute agents using coordination rules.
    
    Evaluates coordination rules in order and executes the action
    specified by the first matching rule.
    
    Args:
        config: Multi-agent configuration with coordination rules
        session_state: Current session state
        message: User message to process
        context: Execution context
        model_provider: Model provider for agent execution
        
    Returns:
        AgentResponse from rule-based execution
    """
    if not config.coordination_rules:
        raise ValueError("No coordination rules defined")
    
    # Evaluate rules in order
    for rule in config.coordination_rules:
        if rule.condition(message, context):
            # Execute the rule's action
            if rule.action == CoordinationAction.DELEGATE:
                # Delegate to specific agent
                target_agents = rule.target_agents or []
                if target_agents:
                    target_agent_name = target_agents[0]
                    agent_config = config.get_agent_by_name(target_agent_name)
                    if agent_config:
                        return await _execute_single_agent(
                            agent_config, session_state, message, context, model_provider
                        )
            
            elif rule.action == CoordinationAction.PARALLEL:
                # Execute specified agents in parallel
                target_configs = []
                if rule.target_agents:
                    target_configs = [
                        config.get_agent_by_name(name)
                        for name in rule.target_agents
                        if config.get_agent_by_name(name) is not None
                    ]
                else:
                    target_configs = config.sub_agents
                
                return await _execute_parallel_specific_agents(
                    target_configs, session_state, message, context, model_provider, config
                )
            
            elif rule.action == CoordinationAction.SEQUENTIAL:
                # Execute specified agents sequentially
                target_configs = []
                if rule.target_agents:
                    target_configs = [
                        config.get_agent_by_name(name)
                        for name in rule.target_agents
                        if config.get_agent_by_name(name) is not None
                    ]
                else:
                    target_configs = config.sub_agents
                
                return await _execute_sequential_specific_agents(
                    target_configs, session_state, message, context, model_provider
                )
    
    # If no rules match, fall back to intelligent selection
    selected_agent = select_best_agent(config.sub_agents, message, context)
    return await _execute_single_agent(
        selected_agent, session_state, message, context, model_provider
    )


def extract_keywords(
    text: str,
    config: Optional[KeywordExtractionConfig] = None
) -> List[str]:
    """
    Extract meaningful keywords from text for agent selection.
    
    Filters out common stop words and extracts relevant terms
    that can be used for keyword matching against agent capabilities.
    
    Args:
        text: Text to extract keywords from
        config: Configuration for keyword extraction
        
    Returns:
        List of extracted keywords
    """
    if config is None:
        config = KeywordExtractionConfig()
    
    # Normalize text
    text = text.lower().strip()
    
    # Split into words and filter
    words = re.findall(r'\b\w+\b', text)
    
    # Get stop words
    stop_words = config.get_stop_words()
    
    # Filter words
    keywords = []
    for word in words:
        if (len(word) >= config.min_word_length and 
            word not in stop_words and
            not word.isdigit()):
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    # Limit number of keywords
    if len(unique_keywords) > config.max_keywords:
        unique_keywords = unique_keywords[:config.max_keywords]
    
    return unique_keywords


# Private helper functions

async def _execute_sequential_agents(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute agents sequentially, passing output to next agent."""
    current_session_state = session_state.copy()
    current_message = message
    final_response = None
    
    for agent_config in config.sub_agents:
        response = await _execute_single_agent(
            agent_config, current_session_state, current_message, context, model_provider
        )
        
        # Update session state and message for next agent
        current_session_state = response.session_state
        current_message = response.content
        final_response = response
    
    return final_response or AgentResponse(
        content=Message(role='assistant', content='No agents executed'),
        session_state=session_state,
        artifacts={}
    )


async def _execute_parallel_agents(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute all agents in parallel and merge responses."""
    return await _execute_parallel_specific_agents(
        config.sub_agents, session_state, message, context, model_provider, config
    )


async def _execute_conditional_agents(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute agents using conditional logic."""
    # Use coordination rules if provided
    if config.coordination_rules:
        return await execute_with_coordination_rules(
            config, session_state, message, context, model_provider
        )
    
    # Otherwise use intelligent agent selection
    selected_agent = select_best_agent(config.sub_agents, message, context)
    return await _execute_single_agent(
        selected_agent, session_state, message, context, model_provider
    )


async def _execute_hierarchical_agents(
    config: MultiAgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute hierarchical coordination with coordinator and delegation."""
    if not config.sub_agents:
        raise ValueError("No sub-agents configured for hierarchical execution")
    
    # Execute coordinator agent first (first agent in list)
    coordinator_config = config.sub_agents[0]
    coordinator_response = await _execute_single_agent(
        coordinator_config, session_state, message, context, model_provider
    )
    
    # Extract delegation decision
    delegation_decision = extract_delegation_decision(coordinator_response)
    
    if delegation_decision and delegation_decision.get('target_agent'):
        # Find and execute target agent
        target_agent_name = delegation_decision['target_agent']
        target_agent_config = config.get_agent_by_name(target_agent_name)
        
        if target_agent_config:
            return await _execute_single_agent(
                target_agent_config,
                coordinator_response.session_state,
                coordinator_response.content,
                context,
                model_provider
            )
    
    # If no delegation, return coordinator response
    return coordinator_response


async def _execute_single_agent(
    agent_config: AgentConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute a single agent with the given configuration."""
    # Convert agent config to JAF Agent
    agent = agent_config.to_agent()
    
    # Create initial state
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[message],
        current_agent_name=agent.name,
        context=context,
        turn_count=0
    )
    
    # Create run config
    run_config = RunConfig(
        agent_registry={agent.name: agent},
        model_provider=model_provider,
        max_turns=5  # Default max turns
    )
    
    # Execute the agent
    result = await jaf_run(initial_state, run_config)
    
    # Convert result to AgentResponse
    return AgentResponse.from_jaf_result(result)


async def _execute_parallel_specific_agents(
    agent_configs: List[AgentConfig],
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any,
    multi_config: MultiAgentConfig
) -> AgentResponse:
    """Execute specific agents in parallel."""
    if not agent_configs:
        raise ValueError("No agents to execute in parallel")
    
    # Create tasks for parallel execution
    tasks = [
        _execute_single_agent(agent_config, session_state, message, context, model_provider)
        for agent_config in agent_configs
    ]
    
    # Execute in parallel
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and convert to AgentResponse list
    valid_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            # Create error response
            error_content = Message(
                role='assistant',
                content=f"Agent {agent_configs[i].name} failed: {str(response)}"
            )
            error_response = AgentResponse(
                content=error_content,
                session_state=session_state,
                artifacts={}
            )
            valid_responses.append(error_response)
        else:
            valid_responses.append(response)
    
    # Merge responses
    return merge_parallel_responses(valid_responses, multi_config)


async def _execute_sequential_specific_agents(
    agent_configs: List[AgentConfig],
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """Execute specific agents sequentially."""
    current_session_state = session_state.copy()
    current_message = message
    final_response = None
    
    for agent_config in agent_configs:
        response = await _execute_single_agent(
            agent_config, current_session_state, current_message, context, model_provider
        )
        
        current_session_state = response.session_state
        current_message = response.content
        final_response = response
    
    return final_response or AgentResponse(
        content=Message(role='assistant', content='No agents executed'),
        session_state=session_state,
        artifacts={}
    )


def _extract_message_text(message: Message) -> str:
    """Extract text content from a message."""
    if hasattr(message, 'content'):
        return str(message.content)
    else:
        return str(message)