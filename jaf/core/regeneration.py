"""
Regeneration functionality for the JAF framework.

This module implements conversation regeneration where a specific message can be
regenerated, removing all subsequent messages and creating a new conversation path.
"""

import time
from dataclasses import replace
from typing import Any, TypeVar, Optional

from .types import (
    RunState, RunConfig, RunResult, 
    RegenerationRequest, RegenerationContext,
    MessageId, Message, ErrorOutcome, ModelBehaviorError,
    find_message_index, truncate_messages_after, get_message_by_id,
    generate_run_id, generate_trace_id
)
from .engine import run as engine_run
from ..memory.types import Success, Failure

Ctx = TypeVar('Ctx')
Out = TypeVar('Out')

async def regenerate_conversation(
    regeneration_request: RegenerationRequest,
    config: RunConfig[Ctx],
    context: Ctx,
    agent_name: str
) -> RunResult[Out]:
    """
    Regenerate a conversation from a specific message ID.
    
    This function:
    1. Loads the full conversation from memory
    2. Finds the message to regenerate from
    3. Truncates the conversation at that point
    4. Creates a new RunState with truncated conversation
    5. Executes the regeneration through the normal engine flow
    6. Updates memory with the new conversation path
    
    Args:
        regeneration_request: The regeneration request containing conversation_id and message_id
        config: The run configuration
        context: The context for the regeneration
        agent_name: The name of the agent to use for regeneration
        
    Returns:
        RunResult with the regenerated conversation outcome
    """
    if not config.memory or not config.memory.provider or not config.conversation_id:
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=[],
                current_agent_name=agent_name,
                context=context,
                turn_count=0
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail="Regeneration requires memory provider and conversation_id to be configured"
            ))
        )

    # Load the conversation from memory
    conversation_result = await config.memory.provider.get_conversation(regeneration_request.conversation_id)
    if isinstance(conversation_result, Failure):
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=[],
                current_agent_name=agent_name,
                context=context,
                turn_count=0
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=f"Failed to load conversation: {conversation_result.error}"
            ))
        )

    conversation_memory = conversation_result.data
    if not conversation_memory:
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=[],
                current_agent_name=agent_name,
                context=context,
                turn_count=0
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=f"Conversation {regeneration_request.conversation_id} not found"
            ))
        )

    # Convert tuple back to list for processing
    original_messages = list(conversation_memory.messages)
    
    # Find the message to regenerate from
    regenerate_message = get_message_by_id(original_messages, regeneration_request.message_id)
    if not regenerate_message:
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=original_messages,
                current_agent_name=agent_name,
                context=context,
                turn_count=len([m for m in original_messages if m.role.value == 'assistant'])
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=f"Message {regeneration_request.message_id} not found in conversation"
            ))
        )

    # Get the index of the message to regenerate
    regenerate_index = find_message_index(original_messages, regeneration_request.message_id)
    if regenerate_index is None:
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=original_messages,
                current_agent_name=agent_name,
                context=context,
                turn_count=len([m for m in original_messages if m.role.value == 'assistant'])
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=f"Failed to find index for message {regeneration_request.message_id}"
            ))
        )

    # Truncate messages after (and including) the regeneration point
    truncated_messages = original_messages[:regenerate_index]
    
    # CRITICAL: Update the conversation in memory with truncated messages BEFORE running the engine
    # This ensures the engine starts with the correct message history
    store_result = await config.memory.provider.store_messages(
        regeneration_request.conversation_id,
        truncated_messages,
        {
            **conversation_memory.metadata,
            "regeneration_truncated": True,
            "regeneration_point": regeneration_request.message_id,
            "original_message_count": len(original_messages),
            "truncated_at_index": regenerate_index,
            "turn_count": len([m for m in truncated_messages if m.role.value == 'assistant'])
        }
    )
    
    if isinstance(store_result, Failure):
        return RunResult(
            final_state=RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=original_messages,
                current_agent_name=agent_name,
                context=context,
                turn_count=len([m for m in original_messages if m.role.value == 'assistant'])
            ),
            outcome=ErrorOutcome(error=ModelBehaviorError(
                detail=f"Failed to store truncated conversation: {store_result.error}"
            ))
        )

    # Create regeneration context for later use
    regeneration_context = RegenerationContext(
        original_message_count=len(original_messages),
        truncated_at_index=regenerate_index,
        regenerated_message_id=regeneration_request.message_id,
        regeneration_id=f"regen_{int(time.time() * 1000)}_{regeneration_request.message_id}",
        timestamp=int(time.time() * 1000)
    )

    # Calculate turn count from truncated messages
    truncated_turn_count = len([m for m in truncated_messages if m.role.value == 'assistant'])
    
    # Merge context if provided in regeneration request
    final_context = context
    if regeneration_request.context:
        if hasattr(context, '__dict__'):
            # For dataclass contexts, merge the additional context
            context_dict = {**context.__dict__, **regeneration_request.context}
            final_context = type(context)(**{k: v for k, v in context_dict.items() if k in context.__dict__})
            # Add any extra fields as attributes
            for key, value in regeneration_request.context.items():
                if not hasattr(final_context, key):
                    setattr(final_context, key, value)
        else:
            # For dict contexts, merge normally
            final_context = {**context, **regeneration_request.context}

    # Create initial state for regeneration with truncated conversation
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=truncated_messages,
        current_agent_name=agent_name,
        context=final_context,
        turn_count=truncated_turn_count,
        approvals={}  # Reset approvals for regeneration
    )

    print(f"[JAF:REGENERATION] Starting regeneration from message {regeneration_request.message_id}")
    print(f"[JAF:REGENERATION] Original messages: {len(original_messages)}, Truncated to: {len(truncated_messages)}")
    print(f"[JAF:REGENERATION] Regeneration context: {regeneration_context}")

    # Create a modified config for regeneration that ensures memory storage
    regeneration_config = replace(
        config,
        conversation_id=regeneration_request.conversation_id,
        memory=replace(config.memory, auto_store=True, store_on_completion=True) if config.memory else None
    )

    # Execute the regeneration through the normal engine flow
    result = await engine_run(initial_state, regeneration_config)

    print(f"[JAF:REGENERATION] Regeneration completed with status: {result.outcome.status}")
    
    # After successful regeneration, mark the regeneration point and preserve metadata
    if result.outcome.status == 'completed' and config.memory and config.memory.provider:
        try:
            print(f"[JAF:REGENERATION] Marking regeneration point after successful regeneration")
            
            # Get the current conversation to preserve regeneration metadata
            current_conv_result = await config.memory.provider.get_conversation(regeneration_request.conversation_id)
            print(f"[JAF:REGENERATION] Retrieved conversation for preservation: {hasattr(current_conv_result, 'data') and current_conv_result.data is not None}")
            
            if hasattr(current_conv_result, 'data') and current_conv_result.data:
                current_metadata = current_conv_result.data.metadata
                regeneration_points = current_metadata.get('regeneration_points', [])
                print(f"[JAF:REGENERATION] Found {len(regeneration_points)} regeneration points in metadata before marking")
                
                # Mark the regeneration point by calling the provider method directly
                mark_result = await config.memory.provider.mark_regeneration_point(
                    regeneration_request.conversation_id,
                    regeneration_request.message_id,
                    {
                        "regeneration_id": regeneration_context.regeneration_id,
                        "original_message_count": len(original_messages),
                        "truncated_at_index": regenerate_index,
                        "timestamp": regeneration_context.timestamp
                    }
                )
                
                if isinstance(mark_result, Failure):
                    print(f"[JAF:REGENERATION] Warning: Failed to mark regeneration point: {mark_result.error}")
                else:
                    print(f"[JAF:REGENERATION] Successfully marked regeneration point")
                
                # Get the updated conversation with the new regeneration point
                updated_conv_result = await config.memory.provider.get_conversation(regeneration_request.conversation_id)
                if hasattr(updated_conv_result, 'data') and updated_conv_result.data:
                    updated_metadata = updated_conv_result.data.metadata
                    updated_regeneration_points = updated_metadata.get('regeneration_points', [])
                    print(f"[JAF:REGENERATION] Found {len(updated_regeneration_points)} regeneration points after marking")
                    
                    # Ensure final metadata includes the regeneration points
                    final_metadata = {
                        **updated_metadata,
                        'regeneration_points': updated_regeneration_points,
                        'regeneration_count': len(updated_regeneration_points),
                        'last_regeneration': updated_regeneration_points[-1] if updated_regeneration_points else None,
                        'regeneration_preserved': True,
                        'final_preservation_timestamp': int(time.time() * 1000)
                    }
                    
                    # Store the final conversation with preserved regeneration metadata
                    await config.memory.provider.store_messages(
                        regeneration_request.conversation_id,
                        result.final_state.messages,
                        final_metadata
                    )
                    print(f"[JAF:REGENERATION] Final preservation completed with {len(updated_regeneration_points)} regeneration points")
            else:
                print(f"[JAF:REGENERATION] No conversation data found for preservation")
                    
        except Exception as e:
            print(f"[JAF:REGENERATION] Warning: Failed to preserve regeneration points: {e}")
            import traceback
            traceback.print_exc()
    
    return result


async def get_regeneration_points(
    conversation_id: str,
    config: RunConfig[Ctx]
) -> Optional[list]:
    """
    Get all regeneration points for a conversation.
    
    Args:
        conversation_id: The conversation ID
        config: The run configuration
        
    Returns:
        List of regeneration points or None if not available
    """
    if not config.memory or not config.memory.provider:
        return None
        
    try:
        conversation_result = await config.memory.provider.get_conversation(conversation_id)
        if hasattr(conversation_result, 'data') and conversation_result.data:
            metadata = conversation_result.data.metadata
            regeneration_points = metadata.get('regeneration_points', [])
            print(f"[JAF:REGENERATION] Retrieved {len(regeneration_points)} regeneration points for {conversation_id}")
            return regeneration_points
        else:
            print(f"[JAF:REGENERATION] No conversation data found for {conversation_id}")
            return []
    except Exception as e:
        print(f"[JAF:REGENERATION] Failed to get regeneration points: {e}")
    
    return []
