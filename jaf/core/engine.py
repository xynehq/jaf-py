"""
Core execution engine for the JAF framework.

This module implements the main run function that orchestrates agent execution,
tool calling, and state management while maintaining functional purity.
"""

import asyncio
import json
import os
import time
from dataclasses import replace, asdict, is_dataclass
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import ValidationError, BaseModel

from ..memory.types import Failure
from .tool_results import tool_result_to_string
from .types import (
    Agent,
    AgentNotFound,
    ApprovalValue,
    CompletedOutcome,
    ContentRole,
    DecodeError,
    ErrorOutcome,
    HandoffError,
    HandoffEvent,
    HandoffEventData,
    InputGuardrailTripwire,
    InterruptedOutcome,
    Interruption,
    GuardrailEvent,
    GuardrailEventData,
    GuardrailViolationEvent,
    GuardrailViolationEventData,
    MemoryEvent,
    MemoryEventData,
    OutputParseEvent,
    OutputParseEventData,
    LLMCallEndEvent,
    LLMCallEndEventData,
    LLMCallStartEvent,
    LLMCallStartEventData,
    AssistantMessageEvent,
    AssistantMessageEventData,
    MaxTurnsExceeded,
    Message,
    get_text_content,
    ModelBehaviorError,
    OutputGuardrailTripwire,
    RunConfig,
    RunEndEvent,
    RunEndEventData,
    RunResult,
    RunStartEvent,
    RunStartEventData,
    RunState,
    ToolApprovalInterruption,
    ToolCall,
    ToolCallEndEvent,
    ToolCallEndEventData,
    ToolCallFunction,
    ToolCallStartEvent,
    ToolCallStartEventData,
    Guardrail,
    ValidValidationResult,
    InvalidValidationResult,
)
from .guardrails import (
    build_effective_guardrails,
    execute_input_guardrails_sequential,
    execute_input_guardrails_parallel,
    execute_output_guardrails,
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


Ctx = TypeVar("Ctx")
Out = TypeVar("Out")


async def try_resume_pending_tool_calls(
    state: RunState[Ctx], config: RunConfig[Ctx]
) -> Optional[RunResult[Out]]:
    """
    Try to resume pending tool calls if the last assistant message contained tool_calls
    and some of those calls have not yet produced tool results.
    """
    try:
        messages = state.messages
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            # Handle both string and enum roles
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role_str == "assistant" and msg.tool_calls:
                tool_call_ids = {tc.id for tc in msg.tool_calls}

                # Scan forward for tool results tied to these ids
                executed_ids = set()
                for j in range(i + 1, len(messages)):
                    m = messages[j]
                    # Handle both string and enum roles
                    m_role_str = m.role.value if hasattr(m.role, "value") else str(m.role)
                    if m_role_str == "tool" and m.tool_call_id and m.tool_call_id in tool_call_ids:
                        executed_ids.add(m.tool_call_id)

                pending_tool_calls = [tc for tc in msg.tool_calls if tc.id not in executed_ids]

                if not pending_tool_calls:
                    continue  # Continue checking other assistant messages

                current_agent = config.agent_registry.get(state.current_agent_name)
                if not current_agent:
                    return RunResult(
                        final_state=state,
                        outcome=ErrorOutcome(
                            error=AgentNotFound(agent_name=state.current_agent_name)
                        ),
                    )

                # Execute pending tool calls
                tool_results = await _execute_tool_calls(
                    pending_tool_calls, current_agent, state, config
                )

                # Check for interruptions
                interruptions = [
                    r.get("interruption") for r in tool_results if r.get("interruption")
                ]
                if interruptions:
                    completed_results = [r for r in tool_results if not r.get("interruption")]
                    interrupted_state = replace(
                        state,
                        messages=list(state.messages) + [r["message"] for r in completed_results],
                        turn_count=state.turn_count,
                        approvals=state.approvals,
                    )
                    return RunResult(
                        final_state=interrupted_state,
                        outcome=InterruptedOutcome(interruptions=interruptions),
                    )

                # Continue with normal execution
                next_state = replace(
                    state,
                    messages=list(state.messages) + [r["message"] for r in tool_results],
                    turn_count=state.turn_count,
                    approvals=state.approvals,
                )
                return await _run_internal(next_state, config)

    except Exception as e:
        # Best-effort resume; ignore and continue normal flow
        pass

    return None


async def run(initial_state: RunState[Ctx], config: RunConfig[Ctx]) -> RunResult[Out]:
    """
    Main execution function for running agents.
    """
    try:
        # Set the current RunConfig in context for agent tools
        from .agent_tool import set_current_run_config

        set_current_run_config(config)

        state_with_memory = await _load_conversation_history(initial_state, config)

        # Emit RunStartEvent AFTER loading conversation history so we have complete context
        if config.on_event:
            config.on_event(
                RunStartEvent(
                    data=to_event_data(
                        RunStartEventData(
                            run_id=initial_state.run_id,
                            trace_id=initial_state.trace_id,
                            session_id=config.conversation_id,
                            context=state_with_memory.context,
                            messages=state_with_memory.messages,  # Now includes full conversation history
                            agent_name=state_with_memory.current_agent_name,
                        )
                    )
                )
            )

        # Load approvals from storage if configured
        if config.approval_storage:
            print(f"[JAF:ENGINE] Loading approvals for runId {state_with_memory.run_id}")
            from .state import load_approvals_into_state

            state_with_memory = await load_approvals_into_state(state_with_memory, config)

        result = await _run_internal(state_with_memory, config)

        # Store conversation history only if this is a final completion of the entire conversation
        # For HITL scenarios, storage happens on interruption to allow resumption
        # We only store on completion if explicitly indicated this is the end of the conversation
        if (
            config.memory
            and config.memory.auto_store
            and config.conversation_id
            and result.outcome.status == "completed"
            and getattr(config.memory, "store_on_completion", True)
        ):
            print(f"[JAF:ENGINE] Storing final completed conversation for {config.conversation_id}")
            await _store_conversation_history(result.final_state, config)
        elif result.outcome.status == "interrupted":
            print(
                "[JAF:ENGINE] Conversation interrupted - storage already handled during interruption"
            )
        else:
            print(
                f"[JAF:ENGINE] Skipping memory store - status: {result.outcome.status}, store_on_completion: {getattr(config.memory, 'store_on_completion', True) if config.memory else 'N/A'}"
            )

        if config.on_event:
            config.on_event(
                RunEndEvent(
                    data=to_event_data(
                        RunEndEventData(
                            outcome=result.outcome,
                            trace_id=initial_state.trace_id,
                            run_id=initial_state.run_id,
                        )
                    )
                )
            )

        return result
    except Exception as error:
        error_result = RunResult(
            final_state=initial_state,
            outcome=ErrorOutcome(error=ModelBehaviorError(detail=str(error))),
        )
        if config.on_event:
            config.on_event(
                RunEndEvent(
                    data=to_event_data(
                        RunEndEventData(
                            outcome=error_result.outcome,
                            trace_id=initial_state.trace_id,
                            run_id=initial_state.run_id,
                        )
                    )
                )
            )
        return error_result


async def _load_conversation_history(state: RunState[Ctx], config: RunConfig[Ctx]) -> RunState[Ctx]:
    """Load conversation history from memory provider."""
    if not (config.memory and config.memory.provider and config.conversation_id):
        return state

    if config.on_event:
        config.on_event(
            MemoryEvent(
                data=MemoryEventData(
                    operation="load", conversation_id=config.conversation_id, status="start"
                )
            )
        )

    result = await config.memory.provider.get_conversation(config.conversation_id)
    if isinstance(result, Failure):
        print(f"[JAF:ENGINE] Warning: Failed to load conversation: {result.error}")
        if config.on_event:
            config.on_event(
                MemoryEvent(
                    data=MemoryEventData(
                        operation="load",
                        conversation_id=config.conversation_id,
                        status="fail",
                        error=str(result.error),
                    )
                )
            )
        return state

    conversation_data = result.data
    if conversation_data:
        max_messages = config.memory.max_messages or len(conversation_data.messages)
        all_memory_messages = conversation_data.messages[-max_messages:]

        # Filter out halted messages - they're for audit/database only, not for LLM context
        memory_messages = []
        filtered_count = 0

        for msg in all_memory_messages:
            if msg.role not in (ContentRole.TOOL, "tool"):
                memory_messages.append(msg)
            else:
                try:
                    content = json.loads(msg.content)
                    status = content.get("status")
                    hitl_status = content.get("hitl_status")
                    # Filter out ALL halted/pending approval messages (they're for audit only)
                    if status == "halted" or hitl_status == "pending_approval":
                        filtered_count += 1
                        continue  # Skip this halted message
                    else:
                        memory_messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    # Keep non-JSON tool messages
                    memory_messages.append(msg)

        # For HITL scenarios, append new messages to memory messages
        # This prevents duplication when resuming from interruptions
        if memory_messages:
            combined_messages = memory_messages + list(state.messages)
        else:
            combined_messages = list(state.messages)

        # Approvals will be loaded separately via approval storage system
        approvals_map = state.approvals

        # Calculate turn count efficiently
        memory_assistant_count = sum(
            1 for msg in memory_messages if msg.role in (ContentRole.ASSISTANT, "assistant")
        )
        current_assistant_count = sum(
            1 for msg in state.messages if msg.role in (ContentRole.ASSISTANT, "assistant")
        )
        calculated_turn_count = memory_assistant_count + current_assistant_count

        # Use metadata turn_count if available, otherwise calculate from messages
        turn_count = calculated_turn_count
        if conversation_data.metadata and "turn_count" in conversation_data.metadata:
            metadata_turn_count = conversation_data.metadata["turn_count"]
            turn_count = max(metadata_turn_count, calculated_turn_count)

        if config.on_event:
            config.on_event(
                MemoryEvent(
                    data=MemoryEventData(
                        operation="load",
                        conversation_id=config.conversation_id,
                        status="end",
                        message_count=len(memory_messages),
                    )
                )
            )

        if filtered_count > 0:
            print(
                f"[JAF:MEMORY] Loaded {len(all_memory_messages)} messages from memory, filtered to {len(memory_messages)} for LLM context (removed {filtered_count} halted messages)"
            )
        else:
            print(f"[JAF:MEMORY] Loaded {len(all_memory_messages)} messages from memory")

        return replace(
            state, messages=combined_messages, turn_count=turn_count, approvals=approvals_map
        )
    return state


async def _store_conversation_history(state: RunState[Ctx], config: RunConfig[Ctx]):
    """Store conversation history to memory provider."""
    if not (
        config.memory
        and config.memory.provider
        and config.conversation_id
        and config.memory.auto_store
    ):
        return

    if config.on_event:
        config.on_event(
            MemoryEvent(
                data=MemoryEventData(
                    operation="store", conversation_id=config.conversation_id, status="start"
                )
            )
        )

    messages_to_store = list(state.messages)
    if (
        config.memory.compression_threshold
        and len(messages_to_store) > config.memory.compression_threshold
    ):
        keep_first = int(config.memory.compression_threshold * 0.2)
        keep_recent = config.memory.compression_threshold - keep_first
        messages_to_store = messages_to_store[:keep_first] + messages_to_store[-keep_recent:]

    # Store approval information if any approvals were made
    approval_metadata = {}
    if state.approvals:
        approval_metadata = {
            "approval_count": len(state.approvals),
            "approved_tools": [
                tool_id for tool_id, approval in state.approvals.items() if approval.approved
            ],
            "rejected_tools": [
                tool_id for tool_id, approval in state.approvals.items() if not approval.approved
            ],
            "has_approvals": True,
        }

    metadata = {
        "user_id": getattr(state.context, "user_id", None),
        "trace_id": str(state.trace_id),
        "run_id": str(state.run_id),
        "agent_name": state.current_agent_name,
        "turn_count": state.turn_count,
        **approval_metadata,
    }

    result = await config.memory.provider.store_messages(
        config.conversation_id, messages_to_store, metadata
    )

    if isinstance(result, Failure):
        print(f"[JAF:ENGINE] Warning: Failed to store conversation: {result.error}")
        if config.on_event:
            config.on_event(
                MemoryEvent(
                    data=MemoryEventData(
                        operation="store",
                        conversation_id=config.conversation_id,
                        status="fail",
                        error=str(result.error),
                    )
                )
            )
    else:
        print(
            f"[JAF:ENGINE] Stored {len(messages_to_store)} messages for conversation {config.conversation_id}"
        )
        if config.on_event:
            config.on_event(
                MemoryEvent(
                    data=MemoryEventData(
                        operation="store",
                        conversation_id=config.conversation_id,
                        status="end",
                        message_count=len(messages_to_store),
                    )
                )
            )

    # Removed verbose logging for performance


async def _run_internal(state: RunState[Ctx], config: RunConfig[Ctx]) -> RunResult[Out]:
    """Internal run function with recursive execution logic."""
    # Try to resume pending tool calls first
    resumed = await try_resume_pending_tool_calls(state, config)
    if resumed:
        return resumed

    # Check max turns
    max_turns = config.max_turns or 50
    if state.turn_count >= max_turns:
        return RunResult(
            final_state=state, outcome=ErrorOutcome(error=MaxTurnsExceeded(turns=state.turn_count))
        )

    # Get current agent
    current_agent = config.agent_registry.get(state.current_agent_name)
    if not current_agent:
        return RunResult(
            final_state=state,
            outcome=ErrorOutcome(error=AgentNotFound(agent_name=state.current_agent_name)),
        )

    # Determine if agent has advanced guardrails configuration
    has_advanced_guardrails = bool(
        current_agent.advanced_config
        and current_agent.advanced_config.guardrails
        and (
            current_agent.advanced_config.guardrails.input_prompt
            or current_agent.advanced_config.guardrails.output_prompt
            or current_agent.advanced_config.guardrails.require_citations
        )
    )

    print(
        "[JAF:ENGINE] Debug guardrails setup:",
        {
            "agent_name": current_agent.name,
            "has_advanced_config": bool(current_agent.advanced_config),
            "has_advanced_guardrails": has_advanced_guardrails,
            "initial_input_guardrails": len(config.initial_input_guardrails or []),
            "final_output_guardrails": len(config.final_output_guardrails or []),
        },
    )

    # Build effective guardrails
    effective_input_guardrails: List[Guardrail] = []
    effective_output_guardrails: List[Guardrail] = []

    if has_advanced_guardrails:
        result = await build_effective_guardrails(current_agent, config)
        effective_input_guardrails, effective_output_guardrails = result
    else:
        effective_input_guardrails = list(config.initial_input_guardrails or [])
        effective_output_guardrails = list(config.final_output_guardrails or [])

    # Execute input guardrails on first turn
    input_guardrails_to_run = (
        effective_input_guardrails if state.turn_count == 0 and effective_input_guardrails else []
    )

    print(
        "[JAF:ENGINE] Input guardrails to run:",
        {
            "turn_count": state.turn_count,
            "effective_input_length": len(effective_input_guardrails),
            "input_guardrails_to_run_length": len(input_guardrails_to_run),
            "has_advanced_guardrails": has_advanced_guardrails,
        },
    )

    if input_guardrails_to_run and state.turn_count == 0:
        first_user_message = next(
            (m for m in state.messages if m.role == ContentRole.USER or m.role == "user"), None
        )
        if first_user_message:
            if has_advanced_guardrails:
                execution_mode = (
                    current_agent.advanced_config.guardrails.execution_mode
                    if current_agent.advanced_config and current_agent.advanced_config.guardrails
                    else "parallel"
                )

                if execution_mode == "sequential":
                    guardrail_result = await execute_input_guardrails_sequential(
                        input_guardrails_to_run, first_user_message, config
                    )
                    if not guardrail_result.is_valid:
                        return RunResult(
                            final_state=state,
                            outcome=ErrorOutcome(
                                error=InputGuardrailTripwire(
                                    reason=getattr(
                                        guardrail_result,
                                        "error_message",
                                        "Input guardrail violation",
                                    )
                                )
                            ),
                        )
                else:
                    # Parallel execution with LLM call overlap
                    guardrail_result = await execute_input_guardrails_parallel(
                        input_guardrails_to_run, first_user_message, config
                    )
                    if not guardrail_result.is_valid:
                        print(
                            f"🚨 Input guardrail violation: {getattr(guardrail_result, 'error_message', 'Unknown violation')}"
                        )
                        return RunResult(
                            final_state=state,
                            outcome=ErrorOutcome(
                                error=InputGuardrailTripwire(
                                    reason=getattr(
                                        guardrail_result,
                                        "error_message",
                                        "Input guardrail violation",
                                    )
                                )
                            ),
                        )
            else:
                # Legacy guardrails path
                print(
                    "[JAF:ENGINE] Using LEGACY guardrails path with",
                    len(input_guardrails_to_run),
                    "guardrails",
                )
                for guardrail in input_guardrails_to_run:
                    if config.on_event:
                        config.on_event(
                            GuardrailEvent(
                                data=GuardrailEventData(
                                    guardrail_name=getattr(
                                        guardrail, "__name__", "unknown_guardrail"
                                    ),
                                    content=get_text_content(first_user_message.content),
                                )
                            )
                        )
                    if asyncio.iscoroutinefunction(guardrail):
                        result = await guardrail(get_text_content(first_user_message.content))
                    else:
                        result = guardrail(get_text_content(first_user_message.content))

                    if not result.is_valid:
                        if config.on_event:
                            config.on_event(
                                GuardrailViolationEvent(
                                    data=GuardrailViolationEventData(
                                        stage="input",
                                        reason=getattr(
                                            result, "error_message", "Input guardrail failed"
                                        ),
                                    )
                                )
                            )
                        return RunResult(
                            final_state=state,
                            outcome=ErrorOutcome(
                                error=InputGuardrailTripwire(
                                    reason=getattr(
                                        result, "error_message", "Input guardrail failed"
                                    )
                                )
                            ),
                        )

    # Agent debugging logs removed for performance

    # Get model name
    model = (
        config.model_override
        or (current_agent.model_config.name if current_agent.model_config else None)
        or "gpt-4o"
    )

    # Apply before_llm_call callback if provided
    if config.before_llm_call:
        if asyncio.iscoroutinefunction(config.before_llm_call):
            state = await config.before_llm_call(state, current_agent)
        else:
            result = config.before_llm_call(state, current_agent)
            if asyncio.iscoroutine(result):
                state = await result
            else:
                state = result

    # Emit LLM call start event
    if config.on_event:
        config.on_event(
            LLMCallStartEvent(
                data=to_event_data(
                    LLMCallStartEventData(
                        agent_name=current_agent.name,
                        model=model,
                        trace_id=state.trace_id,
                        run_id=state.run_id,
                        context=state.context,
                        messages=state.messages,
                    )
                )
            )
        )

    # Retry logic for empty LLM responses
    llm_response: Dict[str, Any]
    assistant_event_streamed = False

    for retry_attempt in range(config.max_empty_response_retries + 1):
        # Get completion from model provider
        # Check if streaming should be used based on configuration and availability
        get_stream = getattr(config.model_provider, "get_completion_stream", None)
        use_streaming = config.prefer_streaming != False and callable(get_stream)

        if use_streaming:
            try:
                aggregated_text = ""
                # Working array of partial tool calls
                partial_tool_calls: List[Dict[str, Any]] = []

                async for chunk in get_stream(state, current_agent, config):  # type: ignore[arg-type]
                    # Text deltas
                    delta_text = getattr(chunk, "delta", None)
                    if delta_text:
                        aggregated_text += delta_text

                    # Tool call deltas
                    tcd = getattr(chunk, "tool_call_delta", None)
                    if tcd is not None:
                        idx = getattr(tcd, "index", 0) or 0
                        # Ensure slot exists
                        while len(partial_tool_calls) <= idx:
                            partial_tool_calls.append(
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                }
                            )
                        target = partial_tool_calls[idx]
                        # id
                        tc_id = getattr(tcd, "id", None)
                        if tc_id:
                            target["id"] = tc_id
                        # function fields
                        fn = getattr(tcd, "function", None)
                        if fn is not None:
                            fn_name = getattr(fn, "name", None)
                            if fn_name:
                                target["function"]["name"] = fn_name
                            args_delta = getattr(fn, "arguments_delta", None)
                            if args_delta:
                                target["function"]["arguments"] += args_delta

                    # Emit partial assistant message when something changed
                    if delta_text or tcd is not None:
                        assistant_event_streamed = True
                        # Normalize tool_calls for message
                        message_tool_calls = None
                        if len(partial_tool_calls) > 0:
                            message_tool_calls = []
                            for i, tc in enumerate(partial_tool_calls):
                                arguments = tc["function"]["arguments"]
                                if isinstance(arguments, str):
                                    arguments = _normalize_tool_call_arguments(arguments)
                                message_tool_calls.append(
                                    {
                                        "id": tc["id"] or f"call_{i}",
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"] or "",
                                            "arguments": arguments,
                                        },
                                    }
                                )

                        partial_msg = Message(
                            role=ContentRole.ASSISTANT,
                            content=aggregated_text or "",
                            tool_calls=None
                            if not message_tool_calls
                            else [
                                ToolCall(
                                    id=mc["id"],
                                    type="function",
                                    function=ToolCallFunction(
                                        name=mc["function"]["name"],
                                        arguments=_normalize_tool_call_arguments(
                                            mc["function"]["arguments"]
                                        ),
                                    ),
                                )
                                for mc in message_tool_calls
                            ],
                        )
                        try:
                            if config.on_event:
                                config.on_event(
                                    AssistantMessageEvent(
                                        data=to_event_data(
                                            AssistantMessageEventData(message=partial_msg)
                                        )
                                    )
                                )
                        except Exception as _e:
                            # Do not fail the run on callback errors
                            pass

                # Build final response object compatible with downstream logic
                final_tool_calls = None
                if len(partial_tool_calls) > 0:
                    final_tool_calls = []
                    for i, tc in enumerate(partial_tool_calls):
                        arguments = tc["function"]["arguments"]
                        if isinstance(arguments, str):
                            arguments = _normalize_tool_call_arguments(arguments)
                        final_tool_calls.append(
                            {
                                "id": tc["id"] or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"] or "",
                                    "arguments": arguments,
                                },
                            }
                        )

                llm_response = {
                    "message": {"content": aggregated_text or None, "tool_calls": final_tool_calls}
                }
            except Exception:
                # Fallback to non-streaming on error
                assistant_event_streamed = False
                llm_response = await config.model_provider.get_completion(
                    state, current_agent, config
                )
        else:
            llm_response = await config.model_provider.get_completion(state, current_agent, config)

        # Check if response has meaningful content
        has_content = llm_response.get("message", {}).get("content")
        has_tool_calls = llm_response.get("message", {}).get("tool_calls")

        # If we got a valid response, break out of retry loop
        if has_content or has_tool_calls:
            break

        # If this is not the last attempt, retry with exponential backoff
        if retry_attempt < config.max_empty_response_retries:
            delay = config.empty_response_retry_delay * (2**retry_attempt)
            if config.log_empty_responses:
                print(
                    f"[JAF:ENGINE] Empty LLM response on attempt {retry_attempt + 1}/{config.max_empty_response_retries + 1}, retrying in {delay:.1f}s..."
                )
                print(
                    f"[JAF:ENGINE] Response had message: {bool(llm_response.get('message'))}, content: {bool(has_content)}, tool_calls: {bool(has_tool_calls)}"
                )
            await asyncio.sleep(delay)
        else:
            # Last attempt failed, log detailed diagnostic info
            if config.log_empty_responses:
                print(
                    f"[JAF:ENGINE] Empty LLM response after {config.max_empty_response_retries + 1} attempts"
                )
                print(f"[JAF:ENGINE] Agent: {current_agent.name}, Model: {model}")
                print(
                    f"[JAF:ENGINE] Message count: {len(state.messages)}, Turn: {state.turn_count}"
                )
                print(
                    f"[JAF:ENGINE] Response structure: {json.dumps(llm_response, indent=2)[:1000]}"
                )

    # Apply after_llm_call callback if provided
    if config.after_llm_call:
        if asyncio.iscoroutinefunction(config.after_llm_call):
            llm_response = await config.after_llm_call(state, llm_response)
        else:
            result = config.after_llm_call(state, llm_response)
            if asyncio.iscoroutine(result):
                llm_response = await result
            else:
                llm_response = result

    # Emit LLM call end event
    if config.on_event:
        config.on_event(
            LLMCallEndEvent(
                data=to_event_data(
                    LLMCallEndEventData(
                        choice=llm_response,
                        trace_id=state.trace_id,
                        run_id=state.run_id,
                        usage=llm_response.get("usage"),
                    )
                )
            )
        )

    # Check if response has message
    if not llm_response.get("message"):
        if config.log_empty_responses:
            print(f"[JAF:ENGINE] ERROR: No message in LLM response")
            print(f"[JAF:ENGINE] Response structure: {json.dumps(llm_response, indent=2)[:500]}")
        return RunResult(
            final_state=state,
            outcome=ErrorOutcome(error=ModelBehaviorError(detail="No message in model response")),
        )

    # Create assistant message
    assistant_message = Message(
        role=ContentRole.ASSISTANT,
        content=llm_response["message"].get("content") or "",
        tool_calls=_convert_tool_calls(llm_response["message"].get("tool_calls")),
    )

    new_messages = list(state.messages) + [assistant_message]

    # Handle tool calls
    if assistant_message.tool_calls:
        tool_results = await _execute_tool_calls(
            assistant_message.tool_calls, current_agent, state, config
        )

        # Check for interruptions
        interruptions = [r.get("interruption") for r in tool_results if r.get("interruption")]
        if interruptions:
            # Separate completed tool results from interrupted ones
            completed_results = [r for r in tool_results if not r.get("interruption")]
            approval_required_results = [r for r in tool_results if r.get("interruption")]

            # Add pending approvals to state.approvals
            updated_approvals = dict(state.approvals)
            for interruption in interruptions:
                if interruption.type == "tool_approval":
                    updated_approvals[interruption.tool_call.id] = ApprovalValue(
                        status="pending",
                        approved=False,
                        additional_context={
                            "status": "pending",
                            "timestamp": str(int(time.time() * 1000)),
                        },
                    )

            # Create state with only completed tool results (for LLM context)
            interrupted_state = replace(
                state,
                messages=new_messages + [r["message"] for r in completed_results],
                turn_count=state.turn_count + 1,
                approvals=updated_approvals,
            )

            # Store conversation state with ALL messages including approval-required (for database records)
            if config.memory and config.memory.auto_store and config.conversation_id:
                print(
                    f"[JAF:ENGINE] Storing conversation state due to interruption for {config.conversation_id}"
                )
                state_for_storage = replace(
                    interrupted_state,
                    messages=interrupted_state.messages
                    + [r["message"] for r in approval_required_results],
                )
                await _store_conversation_history(state_for_storage, config)

            return RunResult(
                final_state=interrupted_state,
                outcome=InterruptedOutcome(interruptions=interruptions),
            )

        # Check for handoffs
        handoff_result = next((r for r in tool_results if r.get("is_handoff")), None)
        if handoff_result:
            target_agent = handoff_result["target_agent"]

            # Validate handoff permission
            if not current_agent.handoffs or target_agent not in current_agent.handoffs:
                return RunResult(
                    final_state=replace(state, messages=new_messages),
                    outcome=ErrorOutcome(
                        error=HandoffError(
                            detail=f"Agent {current_agent.name} cannot handoff to {target_agent}"
                        )
                    ),
                )

            # Emit handoff event
            if config.on_event:
                config.on_event(
                    HandoffEvent(
                        data=to_event_data(
                            HandoffEventData(from_=current_agent.name, to=target_agent)
                        )
                    )
                )

            # Remove any halted messages that are being replaced by actual execution results
            cleaned_new_messages = []
            for msg in new_messages:
                if msg.role not in (ContentRole.TOOL, "tool"):
                    cleaned_new_messages.append(msg)
                else:
                    try:
                        content = json.loads(msg.content)
                        if (
                            content.get("status") == "halted"
                            or content.get("hitl_status") == "pending_approval"
                        ):
                            # Remove this halted message if we have a new result for the same tool_call_id
                            if not any(
                                result["message"].tool_call_id == msg.tool_call_id
                                for result in tool_results
                            ):
                                cleaned_new_messages.append(msg)
                        else:
                            cleaned_new_messages.append(msg)
                    except (json.JSONDecodeError, TypeError):
                        cleaned_new_messages.append(msg)

            # Continue with new agent
            next_state = replace(
                state,
                messages=cleaned_new_messages + [r["message"] for r in tool_results],
                current_agent_name=target_agent,
                turn_count=state.turn_count + 1,
                approvals=state.approvals,
            )

            return await _run_internal(next_state, config)

        # Remove any halted messages that are being replaced by actual execution results
        cleaned_new_messages = []
        for msg in new_messages:
            if msg.role not in (ContentRole.TOOL, "tool"):
                cleaned_new_messages.append(msg)
            else:
                try:
                    content = json.loads(msg.content)
                    if (
                        content.get("status") == "halted"
                        or content.get("hitl_status") == "pending_approval"
                    ):
                        # Remove this halted message if we have a new result for the same tool_call_id
                        if not any(
                            result["message"].tool_call_id == msg.tool_call_id
                            for result in tool_results
                        ):
                            cleaned_new_messages.append(msg)
                    else:
                        cleaned_new_messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    cleaned_new_messages.append(msg)

        # Continue with tool results
        next_state = replace(
            state,
            messages=cleaned_new_messages + [r["message"] for r in tool_results],
            turn_count=state.turn_count + 1,
            approvals=state.approvals,
        )

        return await _run_internal(next_state, config)

    # Handle text completion
    if get_text_content(assistant_message.content):
        if current_agent.output_codec:
            # Parse with output codec
            if config.on_event:
                config.on_event(
                    OutputParseEvent(
                        data=OutputParseEventData(
                            content=get_text_content(assistant_message.content), status="start"
                        )
                    )
                )
            try:
                parsed_content = _try_parse_json(get_text_content(assistant_message.content))
                output_data = current_agent.output_codec.model_validate(parsed_content)
                if config.on_event:
                    config.on_event(
                        OutputParseEvent(
                            data=OutputParseEventData(
                                content=get_text_content(assistant_message.content),
                                status="end",
                                parsed_output=output_data,
                            )
                        )
                    )

                # Check final output guardrails
                if has_advanced_guardrails:
                    # Use new advanced system
                    output_guardrail_result = await execute_output_guardrails(
                        effective_output_guardrails, output_data, config
                    )
                    if not output_guardrail_result.is_valid:
                        return RunResult(
                            final_state=replace(state, messages=new_messages),
                            outcome=ErrorOutcome(
                                error=OutputGuardrailTripwire(
                                    reason=getattr(
                                        output_guardrail_result,
                                        "error_message",
                                        "Output guardrail violation",
                                    )
                                )
                            ),
                        )
                else:
                    # Legacy system
                    if effective_output_guardrails:
                        for guardrail in effective_output_guardrails:
                            if config.on_event:
                                config.on_event(
                                    GuardrailEvent(
                                        data=GuardrailEventData(
                                            guardrail_name=getattr(
                                                guardrail, "__name__", "unknown_guardrail"
                                            ),
                                            content=output_data,
                                        )
                                    )
                                )
                        if asyncio.iscoroutinefunction(guardrail):
                            result = await guardrail(output_data)
                        else:
                            result = guardrail(output_data)

                        if not result.is_valid:
                            if config.on_event:
                                config.on_event(
                                    GuardrailViolationEvent(
                                        data=GuardrailViolationEventData(
                                            stage="output",
                                            reason=getattr(
                                                result, "error_message", "Output guardrail failed"
                                            ),
                                        )
                                    )
                                )
                            return RunResult(
                                final_state=replace(
                                    state, messages=new_messages, approvals=state.approvals
                                ),
                                outcome=ErrorOutcome(
                                    error=OutputGuardrailTripwire(
                                        reason=getattr(
                                            result, "error_message", "Output guardrail failed"
                                        )
                                    )
                                ),
                            )

                return RunResult(
                    final_state=replace(
                        state,
                        messages=new_messages,
                        turn_count=state.turn_count + 1,
                        approvals=state.approvals,
                    ),
                    outcome=CompletedOutcome(output=output_data),
                )

            except ValidationError as e:
                if config.on_event:
                    config.on_event(
                        OutputParseEvent(
                            data=OutputParseEventData(
                                content=get_text_content(assistant_message.content),
                                status="fail",
                                error=str(e),
                            )
                        )
                    )
                return RunResult(
                    final_state=replace(state, messages=new_messages, approvals=state.approvals),
                    outcome=ErrorOutcome(
                        error=DecodeError(errors=[{"message": str(e), "details": e.errors()}])
                    ),
                )
        else:
            # No output codec, return content as string
            if has_advanced_guardrails:
                # Use new advanced system
                output_guardrail_result = await execute_output_guardrails(
                    effective_output_guardrails, get_text_content(assistant_message.content), config
                )
                if not output_guardrail_result.is_valid:
                    return RunResult(
                        final_state=replace(state, messages=new_messages),
                        outcome=ErrorOutcome(
                            error=OutputGuardrailTripwire(
                                reason=getattr(
                                    output_guardrail_result,
                                    "error_message",
                                    "Output guardrail violation",
                                )
                            )
                        ),
                    )
            else:
                # Legacy system
                if effective_output_guardrails:
                    for guardrail in effective_output_guardrails:
                        if config.on_event:
                            config.on_event(
                                GuardrailEvent(
                                    data=GuardrailEventData(
                                        guardrail_name=getattr(
                                            guardrail, "__name__", "unknown_guardrail"
                                        ),
                                        content=get_text_content(assistant_message.content),
                                    )
                                )
                            )
                        if asyncio.iscoroutinefunction(guardrail):
                            result = await guardrail(get_text_content(assistant_message.content))
                        else:
                            result = guardrail(get_text_content(assistant_message.content))

                        if not result.is_valid:
                            if config.on_event:
                                config.on_event(
                                    GuardrailViolationEvent(
                                        data=GuardrailViolationEventData(
                                            stage="output",
                                            reason=getattr(
                                                result, "error_message", "Output guardrail failed"
                                            ),
                                        )
                                    )
                                )
                            return RunResult(
                                final_state=replace(
                                    state, messages=new_messages, approvals=state.approvals
                                ),
                                outcome=ErrorOutcome(
                                    error=OutputGuardrailTripwire(
                                        reason=getattr(
                                            result, "error_message", "Output guardrail failed"
                                        )
                                    )
                                ),
                            )

            return RunResult(
                final_state=replace(
                    state,
                    messages=new_messages,
                    turn_count=state.turn_count + 1,
                    approvals=state.approvals,
                ),
                outcome=CompletedOutcome(output=get_text_content(assistant_message.content)),
            )

    # Model produced neither content nor tool calls
    return RunResult(
        final_state=replace(state, messages=new_messages, approvals=state.approvals),
        outcome=ErrorOutcome(
            error=ModelBehaviorError(detail="Model produced neither content nor tool calls")
        ),
    )


def _convert_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolCall]]:
    """Convert API tool calls to internal format."""
    if not tool_calls:
        return None

    return [
        ToolCall(
            id=tc["id"],
            type="function",
            function=ToolCallFunction(
                name=tc["function"]["name"],
                arguments=_normalize_tool_call_arguments(tc["function"]["arguments"]),
            ),
        )
        for tc in tool_calls
    ]


def _normalize_tool_call_arguments(arguments: Any) -> Any:
    """Strip trailing streaming artifacts so arguments remain valid JSON strings."""
    if not arguments or not isinstance(arguments, str):
        return arguments

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(arguments)
    except json.JSONDecodeError:
        return arguments

    remainder = arguments[end:].strip()
    if remainder:
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            return arguments

    return arguments


async def _execute_tool_calls(
    tool_calls: List[ToolCall], agent: Agent[Ctx, Any], state: RunState[Ctx], config: RunConfig[Ctx]
) -> List[Dict[str, Any]]:
    """Execute tool calls and return results."""

    async def execute_single_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
        print(f"[JAF:TOOL-EXEC] Starting execute_single_tool_call for {tool_call.function.name}")
        if config.on_event:
            config.on_event(
                ToolCallStartEvent(
                    data=to_event_data(
                        ToolCallStartEventData(
                            tool_name=tool_call.function.name,
                            args=_try_parse_json(tool_call.function.arguments),
                            trace_id=state.trace_id,
                            run_id=state.run_id,
                            call_id=tool_call.id,
                        )
                    )
                )
            )

        try:
            # Find the tool
            tool = None
            if agent.tools:
                for t in agent.tools:
                    if t.schema.name == tool_call.function.name:
                        tool = t
                        break

            if not tool:
                error_result = json.dumps(
                    {
                        "hitl_status": "tool_not_found",  # HITL workflow status
                        "message": f"Tool {tool_call.function.name} not found",
                        "tool_name": tool_call.function.name,
                    }
                )

                if config.on_event:
                    config.on_event(
                        ToolCallEndEvent(
                            data=to_event_data(
                                ToolCallEndEventData(
                                    tool_name=tool_call.function.name,
                                    result=error_result,
                                    trace_id=state.trace_id,
                                    run_id=state.run_id,
                                    execution_status="error",  # Tool execution failed
                                    tool_result={"error": "tool_not_found"},
                                    call_id=tool_call.id,
                                )
                            )
                        )
                    )

                return {
                    "message": Message(
                        role=ContentRole.TOOL, content=error_result, tool_call_id=tool_call.id
                    )
                }

            # Parse and validate arguments
            raw_args = _try_parse_json(tool_call.function.arguments)
            try:
                # Assuming the tool schema parameters is a Pydantic model
                if hasattr(tool.schema.parameters, "model_validate"):
                    validated_args = tool.schema.parameters.model_validate(raw_args)
                else:
                    validated_args = raw_args
            except ValidationError as e:
                error_result = json.dumps(
                    {
                        "hitl_status": "validation_error",  # HITL workflow status
                        "message": f"Invalid arguments for {tool_call.function.name}: {e!s}",
                        "tool_name": tool_call.function.name,
                        "validation_errors": e.errors(),
                    }
                )

                if config.on_event:
                    config.on_event(
                        ToolCallEndEvent(
                            data=to_event_data(
                                ToolCallEndEventData(
                                    tool_name=tool_call.function.name,
                                    result=error_result,
                                    trace_id=state.trace_id,
                                    run_id=state.run_id,
                                    execution_status="error",  # Tool execution failed due to validation
                                    tool_result={
                                        "error": "validation_error",
                                        "details": e.errors(),
                                    },
                                    call_id=tool_call.id,
                                )
                            )
                        )
                    )

                return {
                    "message": Message(
                        role=ContentRole.TOOL, content=error_result, tool_call_id=tool_call.id
                    )
                }

            # Check if tool needs approval
            needs_approval = False
            approval_func = getattr(tool, "needs_approval", False)
            if callable(approval_func):
                needs_approval = await approval_func(state.context, validated_args)
            else:
                needs_approval = bool(approval_func)

            # Check approval status - first by ID, then by signature for cross-session matching
            approval_status = state.approvals.get(tool_call.id)
            if not approval_status:
                signature = f"{tool_call.function.name}:{tool_call.function.arguments}"
                for _, approval in state.approvals.items():
                    if (
                        approval.additional_context
                        and approval.additional_context.get("signature") == signature
                    ):
                        approval_status = approval
                        break

            derived_status = None
            if approval_status:
                # Use explicit status if available
                if approval_status.status:
                    derived_status = approval_status.status
                # Fall back to approved boolean if status not set
                elif approval_status.approved is True:
                    derived_status = "approved"
                elif approval_status.approved is False:
                    if (
                        approval_status.additional_context
                        and approval_status.additional_context.get("status") == "pending"
                    ):
                        derived_status = "pending"
                    else:
                        derived_status = "rejected"

            is_pending = derived_status == "pending"

            # If approval needed and not yet decided, create interruption
            if needs_approval and (approval_status is None or is_pending):
                interruption = ToolApprovalInterruption(
                    type="tool_approval",
                    tool_call=tool_call,
                    agent=agent,
                    session_id=str(state.run_id),
                )

                # Return interrupted result with halted message
                halted_result = json.dumps(
                    {
                        "hitl_status": "pending_approval",  # HITL workflow status: waiting for approval
                        "message": f"Tool {tool_call.function.name} requires approval.",
                    }
                )

                return {
                    "message": Message(
                        role=ContentRole.TOOL, content=halted_result, tool_call_id=tool_call.id
                    ),
                    "interruption": interruption,
                }

            # If approval was explicitly rejected, return rejection message
            if derived_status == "rejected":
                rejection_reason = (
                    approval_status.additional_context.get(
                        "rejection_reason", "User declined the action"
                    )
                    if approval_status.additional_context
                    else "User declined the action"
                )
                rejection_result = json.dumps(
                    {
                        "hitl_status": "rejected",  # HITL workflow status: user rejected the action
                        "message": f"Action was not approved. {rejection_reason}. Please ask if you can help with something else or suggest an alternative approach.",
                        "tool_name": tool_call.function.name,
                        "rejection_reason": rejection_reason,
                        "additional_context": approval_status.additional_context
                        if approval_status
                        else None,
                    }
                )

                return {
                    "message": Message(
                        role=ContentRole.TOOL, content=rejection_result, tool_call_id=tool_call.id
                    )
                }

            # Determine timeout for this tool
            # Priority: tool-specific timeout > RunConfig default > 30 seconds global default
            if tool and hasattr(tool, "schema"):
                timeout = getattr(tool.schema, "timeout", None)
            else:
                timeout = None
            if timeout is None:
                timeout = (
                    config.default_tool_timeout
                    if config.default_tool_timeout is not None
                    else 300.0
                )

            # Merge additional context if provided through approval
            additional_context = approval_status.additional_context if approval_status else None
            context_with_additional = state.context
            if additional_context:
                # Create a copy of context with additional fields from approval
                if hasattr(state.context, "__dict__"):
                    # For dataclass contexts, add additional context as attributes
                    context_dict = {**state.context.__dict__, **additional_context}
                    context_with_additional = type(state.context)(
                        **{k: v for k, v in context_dict.items() if k in state.context.__dict__}
                    )
                    # Add any extra fields as attributes
                    for key, value in additional_context.items():
                        if not hasattr(context_with_additional, key):
                            setattr(context_with_additional, key, value)
                else:
                    # For dict contexts, merge normally
                    context_with_additional = {**state.context, **additional_context}

            print(f"[JAF:ENGINE] About to execute tool: {tool_call.function.name}")
            print(f"[JAF:ENGINE] Tool args:", validated_args)
            print(f"[JAF:ENGINE] Tool context:", state.context)

            # Execute the tool with timeout
            try:
                tool_result = await asyncio.wait_for(
                    tool.execute(validated_args, context_with_additional), timeout=timeout
                )
            except asyncio.TimeoutError:
                timeout_error_result = json.dumps(
                    {
                        "hitl_status": "execution_timeout",  # HITL workflow status
                        "message": f"Tool {tool_call.function.name} timed out after {timeout} seconds",
                        "tool_name": tool_call.function.name,
                        "timeout_seconds": timeout,
                    }
                )

                if config.on_event:
                    config.on_event(
                        ToolCallEndEvent(
                            data=to_event_data(
                                ToolCallEndEventData(
                                    tool_name=tool_call.function.name,
                                    result=timeout_error_result,
                                    trace_id=state.trace_id,
                                    run_id=state.run_id,
                                    execution_status="timeout",  # Tool execution timed out
                                    tool_result={"error": "timeout"},
                                    call_id=tool_call.id,
                                )
                            )
                        )
                    )

                return {
                    "message": Message(
                        role=ContentRole.TOOL,
                        content=timeout_error_result,
                        tool_call_id=tool_call.id,
                    )
                }

            # Handle both string and ToolResult formats
            if isinstance(tool_result, str):
                result_string = tool_result
                print(
                    f"[JAF:ENGINE] Tool {tool_call.function.name} returned string:", result_string
                )
            else:
                # It's a ToolResult object
                result_string = tool_result_to_string(tool_result)
                print(
                    f"[JAF:ENGINE] Tool {tool_call.function.name} returned ToolResult:", tool_result
                )
                print(f"[JAF:ENGINE] Converted to string:", result_string)

            # Wrap tool result with status information for approval context
            if approval_status and approval_status.additional_context:
                final_content = json.dumps(
                    {
                        "hitl_status": "approved_and_executed",  # HITL workflow status: approved by user and executed
                        "result": result_string,
                        "tool_name": tool_call.function.name,
                        "approval_context": approval_status.additional_context,
                        "message": "Tool was approved and executed successfully with additional context.",
                    }
                )
            elif needs_approval:
                final_content = json.dumps(
                    {
                        "hitl_status": "approved_and_executed",  # HITL workflow status: approved by user and executed
                        "result": result_string,
                        "tool_name": tool_call.function.name,
                        "message": "Tool was approved and executed successfully.",
                    }
                )
            else:
                final_content = json.dumps(
                    {
                        "hitl_status": "executed",  # HITL workflow status: executed normally (no approval needed)
                        "result": result_string,
                        "tool_name": tool_call.function.name,
                        "message": "Tool executed successfully.",
                    }
                )

            if config.on_event:
                config.on_event(
                    ToolCallEndEvent(
                        data=to_event_data(
                            ToolCallEndEventData(
                                tool_name=tool_call.function.name,
                                result=final_content,
                                trace_id=state.trace_id,
                                run_id=state.run_id,
                                tool_result=tool_result,
                                execution_status="success",  # Tool execution succeeded
                                call_id=tool_call.id,
                            )
                        )
                    )
                )

            # Check for handoff
            handoff_check = _try_parse_json(result_string)
            if isinstance(handoff_check, dict) and "handoff_to" in handoff_check:
                return {
                    "message": Message(
                        role=ContentRole.TOOL, content=final_content, tool_call_id=tool_call.id
                    ),
                    "is_handoff": True,
                    "target_agent": handoff_check["handoff_to"],
                }

            return {
                "message": Message(
                    role=ContentRole.TOOL, content=final_content, tool_call_id=tool_call.id
                )
            }

        except Exception as error:
            error_result = json.dumps(
                {
                    "hitl_status": "execution_error",  # HITL workflow status
                    "message": str(error),
                    "tool_name": tool_call.function.name,
                }
            )

            if config.on_event:
                config.on_event(
                    ToolCallEndEvent(
                        data=to_event_data(
                            ToolCallEndEventData(
                                tool_name=tool_call.function.name,
                                result=error_result,
                                trace_id=state.trace_id,
                                run_id=state.run_id,
                                execution_status="error",  # Tool execution failed with exception
                                tool_result={"error": "execution_error", "detail": str(error)},
                                call_id=tool_call.id,
                            )
                        )
                    )
                )

            return {
                "message": Message(
                    role=ContentRole.TOOL, content=error_result, tool_call_id=tool_call.id
                )
            }

    # Execute all tool calls in parallel
    results = await asyncio.gather(*[execute_single_tool_call(tc) for tc in tool_calls])

    return results


def _try_parse_json(text: str) -> Any:
    """Try to parse JSON, return original string if it fails."""
    if not text or not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
