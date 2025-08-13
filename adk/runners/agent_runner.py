"""
Advanced Agent Runner with Comprehensive Callback System

This module implements the core agent execution engine with full lifecycle
instrumentation through callbacks. It transforms the JAF runner from a 
simple execution function into a sophisticated, observable state machine
that supports complex agent patterns like ReAct, iterative synthesis,
and custom tool selection strategies.

Key Features:
- Complete lifecycle instrumentation with 14+ callback hooks
- Iterative execution loops with synthesis checking
- Tool selection and execution modification
- LLM call interception and customization
- Loop detection and prevention
- Context accumulation and management
- Backward compatibility when callbacks are not provided
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import JAF core types
from jaf.core.types import Message, Agent, Tool, generate_run_id
from jaf.core.engine import run as jaf_run
from jaf import RunState, RunConfig, generate_trace_id

# Import ADK types
from .types import (
    RunnerConfig, 
    RunnerCallbacks,
    AgentResponse,
    RunContext,
    LLMControlResult,
    ToolSelectionControlResult,
    ToolExecutionControlResult,
    IterationControlResult,
    IterationCompleteResult,
    SynthesisCheckResult,
    FallbackCheckResult
)


def get_message_text(message: Message) -> str:
    """Extract text content from a message."""
    if hasattr(message, 'content') and isinstance(message.content, str):
        return message.content
    return str(message)


def create_user_message(text: str) -> Message:
    """Create a user message with the given text."""
    return Message(role='user', content=text)


def create_assistant_message(text: str) -> Message:
    """Create an assistant message with the given text."""
    return Message(role='assistant', content=text)


def create_tool_context(agent: Agent, session_state: Dict[str, Any], message: Message) -> Dict[str, Any]:
    """Create context for tool execution."""
    return {
        'agent': agent,
        'session_state': session_state,
        'message': message,
        'actions': {
            'transfer_to_agent': None,
            'add_artifact': None
        }
    }


def get_function_calls(message: Message) -> List[Dict[str, Any]]:
    """Extract function calls from LLM response."""
    # This would integrate with the LLM response parsing
    # For now, return empty list as placeholder
    return []


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{int(time.time() * 1000)}_{generate_run_id()[:8]}"


async def execute_tool(tool: Tool, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool with the given parameters."""
    try:
        # Execute the tool (this would integrate with JAF's tool execution)
        result = await tool.execute(params, context)
        return {
            'success': True,
            'data': result,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': str(e)
        }


async def call_real_llm(
    agent: Agent, 
    message: Message, 
    session_state: Dict[str, Any],
    model_provider: Any
) -> Message:
    """
    Call the real LLM service using JAF's engine.
    
    This integrates with the JAF core engine to make actual LLM calls.
    """
    try:
        # Create a minimal run state for the LLM call
        run_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[message],
            current_agent_name=agent.name,
            context=session_state,
            turn_count=0
        )
        
        # Create run config with the model provider
        run_config = RunConfig(
            agent_registry={agent.name: agent},
            model_provider=model_provider,
            max_turns=1  # Single LLM call
        )
        
        # Execute through JAF
        result = await jaf_run(run_state, run_config)
        
        # Extract the response from the result
        if result.final_state.messages:
            return result.final_state.messages[-1]
        else:
            return create_assistant_message("I couldn't generate a response.")
            
    except Exception as e:
        return create_assistant_message(f"Error calling LLM: {str(e)}")


async def execute_agent(
    config: RunnerConfig,
    session_state: Dict[str, Any],
    message: Message,
    context: RunContext,
    model_provider: Any
) -> AgentResponse:
    """
    Execute an agent with comprehensive callback instrumentation.
    
    This is the core function that implements the callback-driven execution loop,
    transforming the simple JAF execution into a fully observable and customizable
    state machine that supports advanced agent patterns.
    
    Args:
        config: Runner configuration including agent and callbacks
        session_state: Current session state dictionary
        message: Input message to process
        context: Execution context with user info, permissions, etc.
        model_provider: LLM provider for making model calls
        
    Returns:
        AgentResponse with content, session state, and execution metadata
    """
    agent = config.agent
    callbacks = config.callbacks
    current_session_state = session_state.copy()
    tool_calls: List[Dict[str, Any]] = []
    tool_responses: List[Dict[str, Any]] = []
    iteration_count = 0
    max_iterations = config.max_llm_calls or 10
    should_continue = True
    tool_history: List[Dict[str, Any]] = []
    context_data: List[Any] = []
    llm_response: Optional[Message] = None
    execution_start_time = time.time()
    
    # ========== Lifecycle: on_start ==========
    if callbacks and hasattr(callbacks, 'on_start'):
        try:
            await callbacks.on_start(context, message, current_session_state)
        except Exception as e:
            if callbacks and hasattr(callbacks, 'on_error'):
                await callbacks.on_error(e, context)
            raise
    
    try:
        # Check if this is a multi-agent scenario
        if hasattr(agent, 'sub_agents') and agent.sub_agents:
            # Delegate to multi-agent execution
            from .multi_agent import execute_multi_agent
            return await execute_multi_agent(config, current_session_state, message, context)
        
        # ========== Main Iteration Loop for Synthesis-Based Execution ==========
        while should_continue and iteration_count < max_iterations:
            iteration_count += 1
            
            # ========== Iteration Control: on_iteration_start ==========
            if callbacks and hasattr(callbacks, 'on_iteration_start'):
                try:
                    iteration_control = await callbacks.on_iteration_start(iteration_count)
                    if iteration_control:
                        if iteration_control.get('continue_iteration') is False:
                            should_continue = False
                            break
                        if iteration_control.get('max_iterations'):
                            max_iterations = iteration_control['max_iterations']
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    # Continue execution despite callback error
            
            # ========== Synthesis Check ==========
            if callbacks and hasattr(callbacks, 'on_check_synthesis') and context_data:
                try:
                    synthesis_result = await callbacks.on_check_synthesis(current_session_state, context_data)
                    if synthesis_result and synthesis_result.get('complete'):
                        # Synthesis complete, generate final answer
                        final_message_text = synthesis_result.get('answer', 'Please provide a final answer based on the context.')
                        final_message = create_user_message(final_message_text)
                        final_response = await call_real_llm(agent, final_message, current_session_state, model_provider)
                        
                        # Update session state
                        current_session_state['messages'] = current_session_state.get('messages', []) + [final_response]
                        
                        # Create final response
                        response = AgentResponse(
                            content=final_response,
                            session_state=current_session_state,
                            artifacts={},
                            metadata={
                                'request_id': generate_request_id(),
                                'agent_id': agent.name,
                                'llm_calls': iteration_count,
                                'timestamp': datetime.now(),
                                'synthesis_complete': True,
                                'confidence': synthesis_result.get('confidence', 1.0)
                            },
                            execution_time_ms=(time.time() - execution_start_time) * 1000
                        )
                        
                        if callbacks and hasattr(callbacks, 'on_complete'):
                            await callbacks.on_complete(response)
                        
                        return response
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    # Continue execution despite callback error
            
            # ========== Query Rewriting ==========
            current_message = message
            if callbacks and hasattr(callbacks, 'on_query_rewrite'):
                try:
                    rewritten_query = await callbacks.on_query_rewrite(
                        get_message_text(message),
                        context_data
                    )
                    if rewritten_query:
                        current_message = create_user_message(rewritten_query)
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    # Continue with original message
            
            # ========== LLM Call with Callbacks ==========
            llm_response_temp: Optional[Message] = None
            
            if callbacks and hasattr(callbacks, 'on_before_llm_call'):
                try:
                    llm_control = await callbacks.on_before_llm_call(agent, current_message, current_session_state)
                    if llm_control:
                        if llm_control.get('skip'):
                            # Skip LLM call, use provided response or continue
                            if llm_control.get('response'):
                                llm_response_temp = llm_control['response']
                            else:
                                continue
                        else:
                            if llm_control.get('message'):
                                current_message = llm_control['message']
                            llm_response_temp = await call_real_llm(agent, current_message, current_session_state, model_provider)
                    else:
                        llm_response_temp = await call_real_llm(agent, current_message, current_session_state, model_provider)
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    llm_response_temp = await call_real_llm(agent, current_message, current_session_state, model_provider)
            else:
                llm_response_temp = await call_real_llm(agent, current_message, current_session_state, model_provider)
            
            if callbacks and hasattr(callbacks, 'on_after_llm_call'):
                try:
                    modified_response = await callbacks.on_after_llm_call(llm_response_temp, current_session_state)
                    if modified_response:
                        llm_response_temp = modified_response
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    # Continue with original response
            
            llm_response = llm_response_temp
            
            # ========== Check for Function Calls ==========
            function_calls = get_function_calls(llm_response)
            
            if function_calls:
                # ========== Tool Selection Callbacks ==========
                available_tools = agent.tools if agent.tools else []
                
                if callbacks and hasattr(callbacks, 'on_before_tool_selection'):
                    try:
                        tool_selection_control = await callbacks.on_before_tool_selection(available_tools, context_data)
                        if tool_selection_control:
                            if tool_selection_control.get('tools'):
                                available_tools = tool_selection_control['tools']
                            if tool_selection_control.get('custom_selection'):
                                # Force a specific tool selection
                                custom_sel = tool_selection_control['custom_selection']
                                function_calls = [{
                                    'id': generate_request_id(),
                                    'name': custom_sel['tool'],
                                    'args': custom_sel['params']
                                }]
                    except Exception as e:
                        if callbacks and hasattr(callbacks, 'on_error'):
                            await callbacks.on_error(e, context)
                        # Continue with original tools
                
                # ========== Execute Tools ==========
                tool_context = create_tool_context(agent, current_session_state, current_message)
                
                for function_call in function_calls:
                    # ========== Loop Detection ==========
                    if (config.enable_loop_detection and callbacks and 
                        hasattr(callbacks, 'on_loop_detection')):
                        try:
                            should_skip = await callbacks.on_loop_detection(tool_history, function_call['name'])
                            if should_skip:
                                continue
                        except Exception as e:
                            if callbacks and hasattr(callbacks, 'on_error'):
                                await callbacks.on_error(e, context)
                            # Continue without skipping
                    
                    # Track tool in history
                    tool_history.append({
                        'tool': function_call['name'],
                        'params': function_call.get('args', {}),
                        'timestamp': time.time()
                    })
                    
                    if callbacks and hasattr(callbacks, 'on_tool_selected'):
                        try:
                            await callbacks.on_tool_selected(function_call['name'], function_call.get('args'))
                        except Exception as e:
                            if callbacks and hasattr(callbacks, 'on_error'):
                                await callbacks.on_error(e, context)
                            # Continue execution
                    
                    # Find the tool
                    tool = next((t for t in available_tools if t.schema.name == function_call['name']), None)
                    
                    if tool:
                        try:
                            tool_params = function_call.get('args', {})
                            skip_execution = False
                            custom_result = None
                            
                            # ========== Before Tool Execution Callback ==========
                            if callbacks and hasattr(callbacks, 'on_before_tool_execution'):
                                try:
                                    tool_control = await callbacks.on_before_tool_execution(tool, tool_params)
                                    if tool_control:
                                        if tool_control.get('params'):
                                            tool_params = tool_control['params']
                                        if tool_control.get('skip'):
                                            skip_execution = True
                                        if tool_control.get('result'):
                                            custom_result = tool_control['result']
                                except Exception as e:
                                    if callbacks and hasattr(callbacks, 'on_error'):
                                        await callbacks.on_error(e, context)
                                    # Continue with original parameters
                            
                            # Execute tool or use custom result
                            if skip_execution:
                                tool_result = custom_result or {'success': False, 'data': None}
                            else:
                                tool_result = await execute_tool(tool, tool_params, tool_context)
                            
                            # ========== After Tool Execution Callback ==========
                            if callbacks and hasattr(callbacks, 'on_after_tool_execution'):
                                try:
                                    modified_result = await callbacks.on_after_tool_execution(tool, tool_result)
                                    if modified_result:
                                        tool_result = modified_result
                                except Exception as e:
                                    if callbacks and hasattr(callbacks, 'on_error'):
                                        await callbacks.on_error(e, context)
                                    # Continue with original result
                            
                            # ========== Update Context Data ==========
                            if (config.enable_context_accumulation and 
                                tool_result.get('data') and 
                                isinstance(tool_result['data'], dict) and 
                                'contexts' in tool_result['data']):
                                
                                new_context_items = tool_result['data']['contexts']
                                
                                if callbacks and hasattr(callbacks, 'on_context_update'):
                                    try:
                                        updated_context = await callbacks.on_context_update(context_data, new_context_items)
                                        if updated_context is not None:
                                            context_data = updated_context
                                        else:
                                            context_data.extend(new_context_items)
                                    except Exception as e:
                                        if callbacks and hasattr(callbacks, 'on_error'):
                                            await callbacks.on_error(e, context)
                                        context_data.extend(new_context_items)
                                else:
                                    context_data.extend(new_context_items)
                                
                                # Limit context size
                                if len(context_data) > config.max_context_items:
                                    context_data = context_data[-config.max_context_items:]
                            
                            # Create function response
                            function_response = {
                                'id': function_call['id'],
                                'name': function_call['name'],
                                'response': tool_result.get('data'),
                                'success': tool_result.get('success', False),
                                'error': tool_result.get('error')
                            }
                            
                            tool_responses.append(function_response)
                            
                            # Handle tool actions (agent transfer, etc.)
                            if tool_context['actions'].get('transfer_to_agent'):
                                # Handle agent transfer - would need to implement this
                                pass
                            
                        except Exception as error:
                            function_response = {
                                'id': function_call['id'],
                                'name': function_call['name'],
                                'response': None,
                                'success': False,
                                'error': str(error)
                            }
                            
                            tool_responses.append(function_response)
                            
                            # Callback for tool error
                            if callbacks and hasattr(callbacks, 'on_after_tool_execution'):
                                try:
                                    await callbacks.on_after_tool_execution(tool, None, error)
                                except Exception as callback_error:
                                    if callbacks and hasattr(callbacks, 'on_error'):
                                        await callbacks.on_error(callback_error, context)
                    
                    tool_calls.append(function_call)
            
            # ========== Iteration Complete Callback ==========
            if callbacks and hasattr(callbacks, 'on_iteration_complete'):
                try:
                    iteration_result = await callbacks.on_iteration_complete(iteration_count, len(function_calls) > 0)
                    if iteration_result:
                        if iteration_result.get('should_stop'):
                            should_continue = False
                        elif iteration_result.get('should_continue'):
                            should_continue = True
                except Exception as e:
                    if callbacks and hasattr(callbacks, 'on_error'):
                        await callbacks.on_error(e, context)
                    # Continue with current state
            
            # Update session state after iteration
            current_session_state['messages'] = current_session_state.get('messages', []) + [llm_response]
            
            # Check if we should continue iterating
            if not should_continue or (config.enable_context_accumulation and not context_data):
                break
        
        # ========== Fallback Check ==========
        if callbacks and hasattr(callbacks, 'on_fallback_required'):
            try:
                fallback_check = await callbacks.on_fallback_required(context_data)
                if fallback_check and fallback_check.get('required'):
                    # Execute fallback strategy - would implement based on strategy
                    pass
            except Exception as e:
                if callbacks and hasattr(callbacks, 'on_error'):
                    await callbacks.on_error(e, context)
                # Continue to generate final response
        
        # ========== Generate Final Response ==========
        if not llm_response:
            llm_response = create_assistant_message('I was unable to find relevant information to answer your question.')
        
        # Create final response
        final_response = AgentResponse(
            content=llm_response,
            session_state=current_session_state,
            artifacts={},
            metadata={
                'request_id': generate_request_id(),
                'agent_id': agent.name,
                'llm_calls': iteration_count,
                'timestamp': datetime.now(),
                'tool_calls_count': len(tool_calls),
                'context_items_collected': len(context_data)
            },
            execution_time_ms=(time.time() - execution_start_time) * 1000
        )
        
        # ========== Lifecycle: on_complete ==========
        if callbacks and hasattr(callbacks, 'on_complete'):
            try:
                await callbacks.on_complete(final_response)
            except Exception as e:
                if callbacks and hasattr(callbacks, 'on_error'):
                    await callbacks.on_error(e, context)
                # Don't fail the overall execution for callback errors
        
        return final_response
        
    except Exception as error:
        # ========== Lifecycle: on_error ==========
        if callbacks and hasattr(callbacks, 'on_error'):
            try:
                await callbacks.on_error(error, context)
            except Exception:
                pass  # Don't fail on callback error during error handling
        
        # Create error response
        error_response = AgentResponse(
            content=create_assistant_message(f"Agent execution failed: {str(error)}"),
            session_state=current_session_state,
            artifacts={},
            metadata={
                'request_id': generate_request_id(),
                'agent_id': agent.name,
                'llm_calls': iteration_count,
                'timestamp': datetime.now(),
                'error': str(error)
            },
            execution_time_ms=(time.time() - execution_start_time) * 1000
        )
        
        return error_response


async def run_agent(
    config: RunnerConfig,
    message: Message,
    context: Optional[RunContext] = None,
    session_state: Optional[Dict[str, Any]] = None,
    model_provider: Optional[Any] = None
) -> AgentResponse:
    """
    High-level function to run an agent with callback support.
    
    This is the main entry point for executing agents with the new callback system.
    It provides a clean interface while maintaining backward compatibility.
    
    Args:
        config: Runner configuration with agent and optional callbacks
        message: Input message to process
        context: Optional execution context
        session_state: Optional initial session state
        model_provider: Optional LLM provider (uses default if not provided)
        
    Returns:
        AgentResponse with execution results
    """
    # Set defaults
    if context is None:
        context = {}
    if session_state is None:
        session_state = {}
    if model_provider is None:
        # Would use a default model provider here
        # For now, this is a placeholder
        model_provider = None
    
    return await execute_agent(config, session_state, message, context, model_provider)