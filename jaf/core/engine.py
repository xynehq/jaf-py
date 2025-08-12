"""
Core execution engine for the JAF framework.

This module implements the main run function that orchestrates agent execution,
tool calling, and state management while maintaining functional purity.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from dataclasses import replace

from .types import (
    RunState, RunConfig, RunResult, JAFError, Message, TraceEvent,
    Agent, Tool, CompletedOutcome, ErrorOutcome,
    MaxTurnsExceeded, ModelBehaviorError, DecodeError, InputGuardrailTripwire,
    OutputGuardrailTripwire, ToolCallError, HandoffError, AgentNotFound,
    RunStartEvent, RunEndEvent, LLMCallStartEvent, LLMCallEndEvent,
    ToolCallStartEvent, ToolCallEndEvent, HandoffEvent, ToolCall, ToolCallFunction
)
from .tool_results import ToolResult, tool_result_to_string
from pydantic import ValidationError

Ctx = TypeVar('Ctx')
Out = TypeVar('Out')

async def run(
    initial_state: RunState[Ctx],
    config: RunConfig[Ctx]
) -> RunResult[Out]:
    """
    Main execution function for running agents.
    
    Args:
        initial_state: Initial state of the run
        config: Configuration for the run
        
    Returns:
        RunResult containing final state and outcome
    """
    try:
        if config.on_event:
            config.on_event(RunStartEvent(data={
                'run_id': initial_state.run_id,
                'trace_id': initial_state.trace_id
            }))

        result = await _run_internal(initial_state, config)
        
        if config.on_event:
            config.on_event(RunEndEvent(data={'outcome': result.outcome}))

        return result
    except Exception as error:
        error_result = RunResult(
            final_state=initial_state,
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=str(error) if isinstance(error, Exception) else str(error)
            ))
        )

        if config.on_event:
            config.on_event(RunEndEvent(data={'outcome': error_result.outcome}))

        return error_result

async def _run_internal(
    state: RunState[Ctx],
    config: RunConfig[Ctx]
) -> RunResult[Out]:
    """Internal run function with recursive execution logic."""
    
    # Check initial input guardrails on first turn
    if state.turn_count == 0:
        first_user_message = next((m for m in state.messages if m.role == 'user'), None)
        if first_user_message and config.initial_input_guardrails:
            for guardrail in config.initial_input_guardrails:
                if asyncio.iscoroutinefunction(guardrail):
                    result = await guardrail(first_user_message.content)
                else:
                    result = guardrail(first_user_message.content)
                
                if not result.is_valid:
                    return RunResult(
                        final_state=state,
                        outcome=ErrorOutcome(error=InputGuardrailTripwire(
                            reason=result.error_message or "Input guardrail failed"
                        ))
                    )
    
    # Check max turns
    max_turns = config.max_turns or 50
    if state.turn_count >= max_turns:
        return RunResult(
            final_state=state,
            outcome=ErrorOutcome(error=MaxTurnsExceeded(turns=state.turn_count))
        )
    
    # Get current agent
    current_agent = config.agent_registry.get(state.current_agent_name)
    if not current_agent:
        return RunResult(
            final_state=state,
            outcome=ErrorOutcome(error=AgentNotFound(agent_name=state.current_agent_name))
        )
    
    print(f"[JAF:ENGINE] Using agent: {current_agent.name}")
    print(f"[JAF:ENGINE] Agent has {len(current_agent.tools or [])} tools available")
    if current_agent.tools:
        print(f"[JAF:ENGINE] Available tools: {[t.schema.name for t in current_agent.tools]}")
    
    # Get model name
    model = (
        config.model_override or 
        (current_agent.model_config.name if current_agent.model_config else None) or 
        "gpt-4o"
    )
    
    # Emit LLM call start event
    if config.on_event:
        config.on_event(LLMCallStartEvent(data={
            'agent_name': current_agent.name,
            'model': model
        }))
    
    # Get completion from model provider
    llm_response = await config.model_provider.get_completion(state, current_agent, config)
    
    # Emit LLM call end event
    if config.on_event:
        config.on_event(LLMCallEndEvent(data={'choice': llm_response}))
    
    # Check if response has message
    if not llm_response.get('message'):
        return RunResult(
            final_state=state,
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail='No message in model response'
            ))
        )
    
    # Create assistant message
    assistant_message = Message(
        role='assistant',
        content=llm_response['message'].get('content') or '',
        tool_calls=_convert_tool_calls(llm_response['message'].get('tool_calls'))
    )
    
    new_messages = list(state.messages) + [assistant_message]
    
    # Handle tool calls
    if assistant_message.tool_calls:
        print(f"[JAF:ENGINE] Processing {len(assistant_message.tool_calls)} tool calls")
        print(f"[JAF:ENGINE] Tool calls: {assistant_message.tool_calls}")
        
        tool_results = await _execute_tool_calls(
            assistant_message.tool_calls,
            current_agent,
            state,
            config
        )
        
        print(f"[JAF:ENGINE] Tool execution completed. Results count: {len(tool_results)}")
        
        # Check for handoffs
        handoff_result = next((r for r in tool_results if r.get('is_handoff')), None)
        if handoff_result:
            target_agent = handoff_result['target_agent']
            
            # Validate handoff permission
            if not current_agent.handoffs or target_agent not in current_agent.handoffs:
                return RunResult(
                    final_state=replace(state, messages=new_messages),
                    outcome=ErrorOutcome(error=HandoffError(
                        detail=f"Agent {current_agent.name} cannot handoff to {target_agent}"
                    ))
                )
            
            # Emit handoff event
            if config.on_event:
                config.on_event(HandoffEvent(data={
                    'from': current_agent.name,
                    'to': target_agent
                }))
            
            # Continue with new agent
            next_state = replace(
                state,
                messages=new_messages + [r['message'] for r in tool_results],
                current_agent_name=target_agent,
                turn_count=state.turn_count + 1
            )
            
            return await _run_internal(next_state, config)
        
        # Continue with tool results
        next_state = replace(
            state,
            messages=new_messages + [r['message'] for r in tool_results],
            turn_count=state.turn_count + 1
        )
        
        return await _run_internal(next_state, config)
    
    # Handle text completion
    if assistant_message.content:
        if current_agent.output_codec:
            # Parse with output codec
            try:
                parsed_content = _try_parse_json(assistant_message.content)
                output_data = current_agent.output_codec.model_validate(parsed_content)
                
                # Check final output guardrails
                if config.final_output_guardrails:
                    for guardrail in config.final_output_guardrails:
                        if asyncio.iscoroutinefunction(guardrail):
                            result = await guardrail(output_data)
                        else:
                            result = guardrail(output_data)
                        
                        if not result.is_valid:
                            return RunResult(
                                final_state=replace(state, messages=new_messages),
                                outcome=ErrorOutcome(error=OutputGuardrailTripwire(
                                    reason=result.error_message or "Output guardrail failed"
                                ))
                            )
                
                return RunResult(
                    final_state=replace(state, messages=new_messages),
                    outcome=CompletedOutcome(output=output_data)
                )
                
            except ValidationError as e:
                return RunResult(
                    final_state=replace(state, messages=new_messages),
                    outcome=ErrorOutcome(error=DecodeError(
                        errors=[{'message': str(e), 'details': e.errors()}]
                    ))
                )
        else:
            # No output codec, return content as string
            if config.final_output_guardrails:
                for guardrail in config.final_output_guardrails:
                    if asyncio.iscoroutinefunction(guardrail):
                        result = await guardrail(assistant_message.content)
                    else:
                        result = guardrail(assistant_message.content)
                    
                    if not result.is_valid:
                        return RunResult(
                            final_state=replace(state, messages=new_messages),
                            outcome=ErrorOutcome(error=OutputGuardrailTripwire(
                                reason=result.error_message or "Output guardrail failed"
                            ))
                        )
            
            return RunResult(
                final_state=replace(state, messages=new_messages),
                outcome=CompletedOutcome(output=assistant_message.content)
            )
    
    # Model produced neither content nor tool calls
    return RunResult(
        final_state=replace(state, messages=new_messages),
        outcome=ErrorOutcome(error=ModelBehaviorError(
            detail='Model produced neither content nor tool calls'
        ))
    )

def _convert_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolCall]]:
    """Convert API tool calls to internal format."""
    if not tool_calls:
        return None
    
    return [
        ToolCall(
            id=tc['id'],
            type='function',
            function=ToolCallFunction(
                name=tc['function']['name'],
                arguments=tc['function']['arguments']
            )
        )
        for tc in tool_calls
    ]

async def _execute_tool_calls(
    tool_calls: List[ToolCall],
    agent: Agent[Ctx, Any],
    state: RunState[Ctx],
    config: RunConfig[Ctx]
) -> List[Dict[str, Any]]:
    """Execute tool calls and return results."""
    
    async def execute_single_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
        if config.on_event:
            config.on_event(ToolCallStartEvent(data={
                'tool_name': tool_call.function.name,
                'args': _try_parse_json(tool_call.function.arguments)
            }))
        
        try:
            # Find the tool
            tool = None
            if agent.tools:
                for t in agent.tools:
                    if t.schema.name == tool_call.function.name:
                        tool = t
                        break
            
            if not tool:
                error_result = json.dumps({
                    'error': 'tool_not_found',
                    'message': f'Tool {tool_call.function.name} not found',
                    'tool_name': tool_call.function.name,
                })
                
                if config.on_event:
                    config.on_event(ToolCallEndEvent(data={
                        'tool_name': tool_call.function.name,
                        'result': error_result
                    }))
                
                return {
                    'message': Message(
                        role='tool',
                        content=error_result,
                        tool_call_id=tool_call.id
                    )
                }
            
            # Parse and validate arguments
            raw_args = _try_parse_json(tool_call.function.arguments)
            try:
                # Assuming the tool schema parameters is a Pydantic model
                if hasattr(tool.schema.parameters, 'model_validate'):
                    validated_args = tool.schema.parameters.model_validate(raw_args)
                else:
                    validated_args = raw_args
            except ValidationError as e:
                error_result = json.dumps({
                    'error': 'validation_error',
                    'message': f'Invalid arguments for {tool_call.function.name}: {str(e)}',
                    'tool_name': tool_call.function.name,
                    'validation_errors': e.errors()
                })
                
                if config.on_event:
                    config.on_event(ToolCallEndEvent(data={
                        'tool_name': tool_call.function.name,
                        'result': error_result
                    }))
                
                return {
                    'message': Message(
                        role='tool',
                        content=error_result,
                        tool_call_id=tool_call.id
                    )
                }
            
            print(f"[JAF:ENGINE] About to execute tool: {tool_call.function.name}")
            print(f"[JAF:ENGINE] Tool args: {validated_args}")
            print(f"[JAF:ENGINE] Tool context: {state.context}")
            
            # Execute the tool
            tool_result = await tool.execute(validated_args, state.context)
            
            # Handle both string and ToolResult formats
            if isinstance(tool_result, str):
                result_string = tool_result
                tool_result_obj = None
                print(f"[JAF:ENGINE] Tool {tool_call.function.name} returned string: {result_string}")
            else:
                # It's a ToolResult object
                tool_result_obj = tool_result
                result_string = tool_result_to_string(tool_result)
                print(f"[JAF:ENGINE] Tool {tool_call.function.name} returned ToolResult: {tool_result}")
                print(f"[JAF:ENGINE] Converted to string: {result_string}")
            
            if config.on_event:
                config.on_event(ToolCallEndEvent(data={
                    'tool_name': tool_call.function.name,
                    'result': result_string,
                    'tool_result': tool_result_obj,
                    'status': tool_result_obj.status if tool_result_obj else 'success'
                }))
            
            # Check for handoff
            handoff_check = _try_parse_json(result_string)
            if (isinstance(handoff_check, dict) and 
                'handoff_to' in handoff_check):
                return {
                    'message': Message(
                        role='tool',
                        content=result_string,
                        tool_call_id=tool_call.id
                    ),
                    'is_handoff': True,
                    'target_agent': handoff_check['handoff_to']
                }
            
            return {
                'message': Message(
                    role='tool',
                    content=result_string,
                    tool_call_id=tool_call.id
                )
            }
            
        except Exception as error:
            error_result = json.dumps({
                'error': 'execution_error',
                'message': str(error),
                'tool_name': tool_call.function.name,
            })
            
            if config.on_event:
                config.on_event(ToolCallEndEvent(data={
                    'tool_name': tool_call.function.name,
                    'result': error_result
                }))
            
            return {
                'message': Message(
                    role='tool',
                    content=error_result,
                    tool_call_id=tool_call.id
                )
            }
    
    # Execute all tool calls in parallel
    results = await asyncio.gather(*[
        execute_single_tool_call(tc) for tc in tool_calls
    ])
    
    return results

def _try_parse_json(text: str) -> Any:
    """Try to parse JSON, return original string if it fails."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text