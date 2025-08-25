"""
Core execution engine for the JAF framework.

This module implements the main run function that orchestrates agent execution,
tool calling, and state management while maintaining functional purity.
"""

import asyncio
import json
from dataclasses import replace, asdict, is_dataclass
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import ValidationError, BaseModel

from ..memory.types import Failure
from .tool_results import tool_result_to_string
from .types import (
    Agent,
    AgentNotFound,
    CompletedOutcome,
    ContentRole,
    DecodeError,
    ErrorOutcome,
    HandoffError,
    HandoffEvent,
    HandoffEventData,
    InputGuardrailTripwire,
    LLMCallEndEvent,
    LLMCallEndEventData,
    LLMCallStartEvent,
    LLMCallStartEventData,
    MaxTurnsExceeded,
    Message,
    ModelBehaviorError,
    OutputGuardrailTripwire,
    RunConfig,
    RunEndEvent,
    RunEndEventData,
    RunResult,
    RunStartEvent,
    RunStartEventData,
    RunState,
    ToolCall,
    ToolCallEndEvent,
    ToolCallEndEventData,
    ToolCallFunction,
    ToolCallStartEvent,
    ToolCallStartEventData,
)


def to_event_data(value: Any) -> Any:
    """
    Resilient serializer helper for event payloads.
    
    Converts various types to event-compatible data:
    - dataclasses: uses asdict()
    - Pydantic BaseModel: uses model_dump()
    - other types: returns as-is
    
    This prevents TypeError when serializing nested Pydantic models or non-dataclass types.
    """
    if is_dataclass(value):
        return asdict(value)
    elif isinstance(value, BaseModel):
        return value.model_dump()
    else:
        return value


Ctx = TypeVar('Ctx')
Out = TypeVar('Out')

async def run(
    initial_state: RunState[Ctx],
    config: RunConfig[Ctx]
) -> RunResult[Out]:
    """
    Main execution function for running agents.
    """
    try:
        if config.on_event:
            config.on_event(RunStartEvent(data=to_event_data(RunStartEventData(run_id=initial_state.run_id, trace_id=initial_state.trace_id))))

        state_with_memory = await _load_conversation_history(initial_state, config)
        result = await _run_internal(state_with_memory, config)

        await _store_conversation_history(result.final_state, config)

        if config.on_event:
            config.on_event(RunEndEvent(data=to_event_data(RunEndEventData(outcome=result.outcome))))

        return result
    except Exception as error:
        error_result = RunResult(
            final_state=initial_state,
            outcome=ErrorOutcome(error=ModelBehaviorError(detail=str(error)))
        )
        if config.on_event:
            config.on_event(RunEndEvent(data=to_event_data(RunEndEventData(outcome=error_result.outcome))))
        return error_result

async def _load_conversation_history(state: RunState[Ctx], config: RunConfig[Ctx]) -> RunState[Ctx]:
    """Load conversation history from memory provider."""
    if not (config.memory and config.memory.provider and config.conversation_id):
        return state

    result = await config.memory.provider.get_conversation(config.conversation_id)
    if isinstance(result, Failure):
        print(f"[JAF:ENGINE] Warning: Failed to load conversation: {result.error}")
        return state

    conversation_data = result.data
    if conversation_data:
        max_messages = config.memory.max_messages or len(conversation_data.messages)
        memory_messages = conversation_data.messages[-max_messages:]

        print(f"[JAF:ENGINE] Loaded {len(memory_messages)} messages from memory for conversation {config.conversation_id}")

        # Calculate turn count based on assistant messages in memory + current state
        memory_assistant_count = len([msg for msg in memory_messages if msg.role == ContentRole.ASSISTANT or msg.role == 'assistant'])
        current_assistant_count = len([msg for msg in state.messages if msg.role == ContentRole.ASSISTANT or msg.role == 'assistant'])
        calculated_turn_count = memory_assistant_count + current_assistant_count

        # Use metadata turn_count if available, otherwise calculate from messages
        turn_count = calculated_turn_count
        if conversation_data.metadata and "turn_count" in conversation_data.metadata:
            metadata_turn_count = conversation_data.metadata["turn_count"]
            # Use the higher of the two to handle edge cases
            turn_count = max(metadata_turn_count, calculated_turn_count)
            print(f"[JAF:ENGINE] Metadata turn_count: {metadata_turn_count}, calculated: {calculated_turn_count}, using: {turn_count}")
        else:
            print(f"[JAF:ENGINE] No metadata turn_count, calculated from messages: {turn_count}")

        return replace(
            state,
            messages=list(memory_messages) + list(state.messages),
            turn_count=turn_count
        )
    return state

async def _store_conversation_history(state: RunState[Ctx], config: RunConfig[Ctx]):
    """Store conversation history to memory provider."""
    if not (config.memory and config.memory.provider and config.conversation_id and config.memory.auto_store):
        return

    messages_to_store = list(state.messages)
    if config.memory.compression_threshold and len(messages_to_store) > config.memory.compression_threshold:
        keep_first = int(config.memory.compression_threshold * 0.2)
        keep_recent = config.memory.compression_threshold - keep_first
        messages_to_store = messages_to_store[:keep_first] + messages_to_store[-keep_recent:]
        print(f"[JAF:ENGINE] Compressed conversation from {len(state.messages)} to {len(messages_to_store)} messages")

    metadata = {
        "user_id": getattr(state.context, 'user_id', None),
        "trace_id": str(state.trace_id),
        "run_id": str(state.run_id),
        "agent_name": state.current_agent_name,
        "turn_count": state.turn_count
    }

    result = await config.memory.provider.store_messages(config.conversation_id, messages_to_store, metadata)
    if isinstance(result, Failure):
        print(f"[JAF:ENGINE] Warning: Failed to store conversation: {result.error}")
    else:
        print(f"[JAF:ENGINE] Stored {len(messages_to_store)} messages for conversation {config.conversation_id}")

async def _run_internal(
    state: RunState[Ctx],
    config: RunConfig[Ctx]
) -> RunResult[Out]:
    """Internal run function with recursive execution logic."""
    # Check initial input guardrails on first turn
    if state.turn_count == 0:
        first_user_message = next((m for m in state.messages if m.role == ContentRole.USER or m.role == 'user'), None)
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
        config.on_event(LLMCallStartEvent(data=to_event_data(LLMCallStartEventData(
            agent_name=current_agent.name,
            model=model
        ))))

    # Get completion from model provider
    llm_response = await config.model_provider.get_completion(state, current_agent, config)

    # Emit LLM call end event
    if config.on_event:
        config.on_event(LLMCallEndEvent(data=to_event_data(LLMCallEndEventData(choice=llm_response))))

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
        role=ContentRole.ASSISTANT,
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
                config.on_event(HandoffEvent(data=to_event_data(HandoffEventData(
                    from_=current_agent.name,
                    to=target_agent
                ))))

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
                    final_state=replace(state, messages=new_messages, turn_count=state.turn_count + 1),
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
                final_state=replace(state, messages=new_messages, turn_count=state.turn_count + 1),
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
            config.on_event(ToolCallStartEvent(data=to_event_data(ToolCallStartEventData(
                tool_name=tool_call.function.name,
                args=_try_parse_json(tool_call.function.arguments)
            ))))

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
                    config.on_event(ToolCallEndEvent(data=to_event_data(ToolCallEndEventData(
                        tool_name=tool_call.function.name,
                        result=error_result,
                        status='error'
                    ))))

                return {
                    'message': Message(
                        role=ContentRole.TOOL,
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
                    'message': f'Invalid arguments for {tool_call.function.name}: {e!s}',
                    'tool_name': tool_call.function.name,
                    'validation_errors': e.errors()
                })

                if config.on_event:
                    config.on_event(ToolCallEndEvent(data=to_event_data(ToolCallEndEventData(
                        tool_name=tool_call.function.name,
                        result=error_result,
                        status='error'
                    ))))

                return {
                    'message': Message(
                        role=ContentRole.TOOL,
                        content=error_result,
                        tool_call_id=tool_call.id
                    )
                }

            print(f"[JAF:ENGINE] About to execute tool: {tool_call.function.name}")
            print(f"[JAF:ENGINE] Tool args: {validated_args}")
            print(f"[JAF:ENGINE] Tool context: {state.context}")

            # Determine timeout for this tool
            # Priority: tool-specific timeout > RunConfig default > 30 seconds global default
            timeout = getattr(tool.schema, 'timeout', None)
            if timeout is None:
                timeout = config.default_tool_timeout if config.default_tool_timeout is not None else 30.0

            print(f"[JAF:ENGINE] Using timeout: {timeout} seconds for tool {tool_call.function.name}")

            # Execute the tool with timeout
            try:
                tool_result = await asyncio.wait_for(
                    tool.execute(validated_args, state.context),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                timeout_error_result = json.dumps({
                    'error': 'timeout_error',
                    'message': f'Tool {tool_call.function.name} timed out after {timeout} seconds',
                    'tool_name': tool_call.function.name,
                    'timeout_seconds': timeout
                })

                if config.on_event:
                    config.on_event(ToolCallEndEvent(data=to_event_data(ToolCallEndEventData(
                        tool_name=tool_call.function.name,
                        result=timeout_error_result,
                        status='timeout'
                    ))))

                return {
                    'message': Message(
                        role=ContentRole.TOOL,
                        content=timeout_error_result,
                        tool_call_id=tool_call.id
                    )
                }

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
                config.on_event(ToolCallEndEvent(data=to_event_data(ToolCallEndEventData(
                    tool_name=tool_call.function.name,
                    result=result_string,
                    status='success'
                ))))

            # Check for handoff
            handoff_check = _try_parse_json(result_string)
            if (isinstance(handoff_check, dict) and
                'handoff_to' in handoff_check):
                return {
                    'message': Message(
                        role=ContentRole.TOOL,
                        content=result_string,
                        tool_call_id=tool_call.id
                    ),
                    'is_handoff': True,
                    'target_agent': handoff_check['handoff_to']
                }

            return {
                'message': Message(
                    role=ContentRole.TOOL,
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
                config.on_event(ToolCallEndEvent(data=to_event_data(ToolCallEndEventData(
                    tool_name=tool_call.function.name,
                    result=error_result,
                    status='error'
                ))))

            return {
                'message': Message(
                    role=ContentRole.TOOL,
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
