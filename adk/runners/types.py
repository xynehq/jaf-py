"""
Type definitions for multi-agent coordination and execution.

Provides comprehensive type definitions for different agent coordination
strategies, configuration options, and execution contexts.
"""

from typing import Dict, List, Any, Optional, Callable, TypedDict, Literal, Protocol, Awaitable, Union
from dataclasses import dataclass
from enum import Enum

# Import core JAF types
from jaf.core.types import Agent, Message, RunState, Tool


class DelegationStrategy(Enum):
    """
    Multi-agent delegation strategies.
    
    Defines different approaches for coordinating multiple agents:
    - SEQUENTIAL: Execute agents one after another in order
    - PARALLEL: Execute multiple agents concurrently and merge responses
    - CONDITIONAL: Select agent based on intelligent analysis of input
    - HIERARCHICAL: Use coordinator agent to decide delegation
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    HIERARCHICAL = "hierarchical"


class CoordinationAction(Enum):
    """Actions that can be taken by coordination rules."""
    DELEGATE = "delegate"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class AgentConfig:
    """
    Configuration for a single agent in multi-agent setup.
    
    Contains all necessary information to instantiate and execute
    an agent including its tools, instructions, and metadata.
    """
    name: str
    instruction: str
    tools: List[Tool]
    model_config: Optional[Dict[str, Any]] = None
    handoffs: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_agent(self) -> Agent:
        """Convert config to JAF Agent instance."""
        def instructions(state: RunState) -> str:
            return self.instruction
        
        return Agent(
            name=self.name,
            instructions=instructions,
            tools=self.tools,
            handoffs=self.handoffs or []
        )


class RunContext(TypedDict, total=False):
    """
    Execution context for multi-agent coordination.
    
    Provides additional context information that can be used
    by coordination rules and agent selection algorithms.
    """
    user_id: Optional[str]
    session_id: Optional[str]
    conversation_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    permissions: Optional[List[str]]
    preferences: Optional[Dict[str, Any]]


class CoordinationRule(Protocol):
    """
    Protocol for coordination rules that determine agent execution flow.
    
    Rules evaluate conditions and specify actions to take when conditions are met.
    """
    
    def condition(self, message: Message, context: RunContext) -> bool:
        """
        Evaluate if this rule's condition is met.
        
        Args:
            message: The user message being processed
            context: Additional execution context
            
        Returns:
            True if the condition is satisfied, False otherwise
        """
        ...
    
    @property
    def action(self) -> CoordinationAction:
        """The action to take when condition is met."""
        ...
    
    @property
    def target_agents(self) -> Optional[List[str]]:
        """Target agent names for this rule's action."""
        ...


@dataclass
class SimpleCoordinationRule:
    """
    Simple implementation of CoordinationRule for common cases.
    """
    condition_func: Callable[[Message, RunContext], bool]
    action_type: CoordinationAction
    target_agent_names: Optional[List[str]] = None
    
    def condition(self, message: Message, context: RunContext) -> bool:
        return self.condition_func(message, context)
    
    @property
    def action(self) -> CoordinationAction:
        return self.action_type
    
    @property
    def target_agents(self) -> Optional[List[str]]:
        return self.target_agent_names


@dataclass
class MultiAgentConfig:
    """
    Configuration for multi-agent execution strategies.
    
    Defines how multiple agents should be coordinated, including
    the delegation strategy, agent configurations, and coordination rules.
    """
    delegation_strategy: DelegationStrategy
    sub_agents: List[AgentConfig]
    coordination_rules: Optional[List[CoordinationRule]] = None
    parallel_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    
    def get_agent_by_name(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        return next((agent for agent in self.sub_agents if agent.name == name), None)


@dataclass
class AgentResponse:
    """
    Response from an agent execution.
    
    Contains the agent's response content, updated session state,
    and any metadata about the execution.
    """
    content: Message
    session_state: Dict[str, Any]
    artifacts: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    
    @classmethod
    def from_jaf_result(cls, result: Any, execution_time: Optional[float] = None) -> 'AgentResponse':
        """Create AgentResponse from JAF execution result."""
        # Extract content from JAF result
        if hasattr(result, 'final_state') and hasattr(result.final_state, 'messages'):
            # Get the last message as the response
            messages = result.final_state.messages
            if messages and len(messages) > 0:
                content = messages[-1] if hasattr(messages[-1], 'role') else Message(role='assistant', content=str(messages[-1]))
            else:
                content = Message(role='assistant', content='No response')
        else:
            content = Message(role='assistant', content=str(result))
        
        return cls(
            content=content,
            session_state=getattr(result, 'session_state', {}),
            artifacts=getattr(result, 'artifacts', {}),
            execution_time_ms=execution_time
        )


@dataclass
class AgentSelectionScore:
    """
    Score for agent selection algorithm.
    
    Contains the agent configuration and its relevance score
    based on keyword matching and other criteria.
    """
    agent: AgentConfig
    score: float
    reasoning: List[str]  # Explanation of why this score was assigned
    
    def add_score(self, points: float, reason: str) -> None:
        """Add points to the score with reasoning."""
        self.score += points
        self.reasoning.append(f"+{points}: {reason}")


# Keyword extraction configuration
@dataclass
class KeywordExtractionConfig:
    """
    Configuration for keyword extraction from user messages.
    """
    min_word_length: int = 3
    max_keywords: int = 20
    stop_words: Optional[set] = None
    custom_patterns: Optional[List[str]] = None
    
    def get_stop_words(self) -> set:
        """Get the set of stop words to filter out."""
        if self.stop_words is not None:
            return self.stop_words
        
        # Default stop words from schema types
        from adk.schemas.types import STOP_WORDS
        return STOP_WORDS


# ========== Runner Callback System ==========

class LLMControlResult(TypedDict, total=False):
    """Control object returned by onBeforeLLMCall to modify LLM behavior."""
    message: Optional[Message]  # Allow message modification
    skip: Optional[bool]        # Skip this LLM call
    response: Optional[Message] # Provide custom response instead


class ToolSelectionControlResult(TypedDict, total=False):
    """Control object returned by onBeforeToolSelection to modify tool selection."""
    tools: Optional[List[Tool]]                           # Modify available tools
    custom_selection: Optional[Dict[str, Any]]            # Force specific tool: {'tool': str, 'params': Any}


class ToolExecutionControlResult(TypedDict, total=False):
    """Control object returned by onBeforeToolExecution to modify tool execution."""
    params: Optional[Any]       # Modify parameters
    skip: Optional[bool]        # Skip execution
    result: Optional[Any]       # Provide custom result


class IterationControlResult(TypedDict, total=False):
    """Control object returned by iteration hooks to control loop flow."""
    continue_iteration: Optional[bool]      # Continue iterating
    max_iterations: Optional[int]           # Modify max iterations


class IterationCompleteResult(TypedDict, total=False):
    """Control object returned by onIterationComplete to control continuation."""
    should_continue: Optional[bool]         # Force another iteration
    should_stop: Optional[bool]             # Force stop


class SynthesisCheckResult(TypedDict, total=False):
    """Result from synthesis check callback."""
    complete: Optional[bool]                # Is synthesis complete?
    answer: Optional[str]                   # Synthesis answer
    confidence: Optional[float]             # Confidence score (0.0-1.0)


class FallbackCheckResult(TypedDict, total=False):
    """Result from fallback requirement check."""
    required: bool                          # Is fallback required?
    strategy: Optional[str]                 # Fallback strategy to use


class RunnerCallbacks(Protocol):
    """
    Comprehensive callback system for instrumenting agent execution.
    
    This protocol defines hooks that can be injected at every critical stage
    of the agent's execution lifecycle, enabling:
    - Custom iteration control (ReAct patterns)
    - Tool selection and execution modification
    - LLM call interception and modification
    - Synthesis checking for iterative refinement
    - Loop detection and prevention
    - Context management and accumulation
    """
    
    # ========== Lifecycle Hooks ==========
    
    async def on_start(
        self, 
        context: RunContext, 
        message: Message, 
        session_state: Dict[str, Any]
    ) -> None:
        """Called at the start of agent execution."""
        ...
    
    async def on_complete(self, response: AgentResponse) -> None:
        """Called when agent execution completes successfully."""
        ...
    
    async def on_error(self, error: Exception, context: RunContext) -> None:
        """Called when agent execution encounters an error."""
        ...
    
    # ========== LLM Interaction Hooks ==========
    
    async def on_before_llm_call(
        self, 
        agent: Agent, 
        message: Message, 
        session_state: Dict[str, Any]
    ) -> Optional[LLMControlResult]:
        """
        Called before making an LLM call.
        
        Can modify the message, skip the call entirely, or provide a custom response.
        """
        ...
    
    async def on_after_llm_call(
        self, 
        response: Message, 
        session_state: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Called after receiving an LLM response.
        
        Can modify the response before it's processed.
        """
        ...
    
    # ========== Tool Execution Hooks ==========
    
    async def on_before_tool_selection(
        self, 
        tools: List[Tool], 
        context_data: List[Any]
    ) -> Optional[ToolSelectionControlResult]:
        """
        Called before tool selection from LLM response.
        
        Can modify available tools or force a specific tool selection.
        """
        ...
    
    async def on_tool_selected(
        self, 
        tool_name: Optional[str], 
        params: Any
    ) -> None:
        """Called when a tool is selected for execution."""
        ...
    
    async def on_before_tool_execution(
        self, 
        tool: Tool, 
        params: Any
    ) -> Optional[ToolExecutionControlResult]:
        """
        Called before executing a tool.
        
        Can modify parameters, skip execution, or provide a custom result.
        """
        ...
    
    async def on_after_tool_execution(
        self, 
        tool: Tool, 
        result: Any, 
        error: Optional[Exception] = None
    ) -> Optional[Any]:
        """
        Called after tool execution (successful or failed).
        
        Can modify the result before it's processed.
        """
        ...
    
    # ========== Iteration Control Hooks ==========
    
    async def on_iteration_start(
        self, 
        iteration: int
    ) -> Optional[IterationControlResult]:
        """
        Called at the start of each iteration in the main loop.
        
        Can control whether to continue iterating or modify max iterations.
        """
        ...
    
    async def on_iteration_complete(
        self, 
        iteration: int, 
        has_tool_calls: bool
    ) -> Optional[IterationCompleteResult]:
        """
        Called at the end of each iteration.
        
        Can force continuation or termination of the iteration loop.
        """
        ...
    
    # ========== Custom Logic Injection Points ==========
    
    async def on_check_synthesis(
        self, 
        session_state: Dict[str, Any], 
        context_data: List[Any]
    ) -> Optional[SynthesisCheckResult]:
        """
        Called to check if synthesis is complete for iterative patterns.
        
        Used for ReAct-style patterns where the agent accumulates information
        across multiple iterations and needs to determine when enough context
        has been gathered to provide a final answer.
        """
        ...
    
    async def on_query_rewrite(
        self, 
        original_query: str, 
        context_data: List[Any]
    ) -> Optional[str]:
        """
        Called to rewrite the query for refined iterations.
        
        Allows for dynamic query modification based on accumulated context,
        enabling more focused subsequent tool calls.
        """
        ...
    
    async def on_loop_detection(
        self, 
        tool_history: List[Dict[str, Any]], 
        current_tool: str
    ) -> bool:
        """
        Called to detect if the agent is stuck in a loop.
        
        Returns True if the current tool call should be skipped to prevent loops.
        """
        ...
    
    async def on_fallback_required(
        self, 
        context_data: List[Any]
    ) -> Optional[FallbackCheckResult]:
        """
        Called to check if fallback strategy is needed.
        
        Used when the agent fails to make progress through normal means.
        """
        ...
    
    # ========== Context Management Hooks ==========
    
    async def on_context_update(
        self, 
        current_context: List[Any], 
        new_items: List[Any]
    ) -> Optional[List[Any]]:
        """
        Called when new context items are available.
        
        Can filter, merge, or reorganize the context data.
        """
        ...
    
    async def on_excluded_ids_update(
        self, 
        excluded_ids: List[str], 
        new_ids: List[str]
    ) -> Optional[List[str]]:
        """
        Called to update excluded IDs list.
        
        Used for tracking items that should not be processed again.
        """
        ...


@dataclass
class RunnerConfig:
    """
    Enhanced configuration for agent runners with callback support.
    
    Includes all necessary configuration for agent execution plus
    optional callbacks for advanced behavior customization.
    """
    agent: Agent
    session_provider: Any  # Session provider interface
    max_llm_calls: Optional[int] = 10
    timeout_seconds: Optional[int] = None
    callbacks: Optional[RunnerCallbacks] = None
    guardrails: Optional[List[Callable]] = None
    
    # Context accumulation settings
    enable_context_accumulation: bool = True
    max_context_items: int = 100
    
    # Loop detection settings
    enable_loop_detection: bool = True
    max_repeated_tools: int = 3