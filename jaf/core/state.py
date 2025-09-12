"""
State management functions for approval handling in HITL scenarios.

This module provides functions to manage approval state transitions
and integrate with approval storage systems.
"""

from typing import Dict, Any, Optional
from dataclasses import replace

from .types import RunState, RunConfig, Interruption, ApprovalValue


async def approve(
    state: RunState[Any],
    interruption: Interruption,
    additional_context: Optional[Dict[str, Any]] = None,
    config: Optional[RunConfig[Any]] = None
) -> RunState[Any]:
    """
    Approve a tool call interruption and update the run state.
    
    Args:
        state: Current run state
        interruption: The interruption to approve
        additional_context: Optional additional context for the approval
        config: Optional run configuration for approval storage
        
    Returns:
        Updated run state with approval recorded
    """
    if interruption.type == 'tool_approval':
        approval_value = ApprovalValue(
            status='approved',
            approved=True,
            additional_context={
                **(additional_context or {}), 
                'status': 'approved'
            }
        )
        
        # Store in approval storage if available
        if config and config.approval_storage:
            try:
                print(f"[JAF:APPROVAL] Storing approval for tool_call_id {interruption.tool_call.id}: {approval_value}")
                result = await config.approval_storage.store_approval(
                    state.run_id,
                    interruption.tool_call.id,
                    approval_value
                )
                if not result.success:
                    print(f"[JAF:APPROVAL] Failed to store approval: {result.error}")
                    # Continue with in-memory fallback
                else:
                    print(f"[JAF:APPROVAL] Successfully stored approval in storage")
            except Exception as e:
                print(f"[JAF:APPROVAL] Approval storage error: {e}")
                # Continue with in-memory fallback
        
        # Update in-memory state
        new_approvals = {**state.approvals}
        new_approvals[interruption.tool_call.id] = approval_value
        
        return replace(state, approvals=new_approvals)
    
    return state


async def reject(
    state: RunState[Any],
    interruption: Interruption,
    additional_context: Optional[Dict[str, Any]] = None,
    config: Optional[RunConfig[Any]] = None
) -> RunState[Any]:
    """
    Reject a tool call interruption and update the run state.
    
    Args:
        state: Current run state
        interruption: The interruption to reject
        additional_context: Optional additional context for the rejection
        config: Optional run configuration for approval storage
        
    Returns:
        Updated run state with rejection recorded
    """
    if interruption.type == 'tool_approval':
        approval_value = ApprovalValue(
            status='rejected',
            approved=False,
            additional_context={
                **(additional_context or {}), 
                'status': 'rejected'
            }
        )
        
        # Store in approval storage if available
        if config and config.approval_storage:
            try:
                print(f"[JAF:APPROVAL] Storing approval for tool_call_id {interruption.tool_call.id}: {approval_value}")
                result = await config.approval_storage.store_approval(
                    state.run_id,
                    interruption.tool_call.id,
                    approval_value
                )
                if not result.success:
                    print(f"[JAF:APPROVAL] Failed to store approval: {result.error}")
                    # Continue with in-memory fallback
                else:
                    print(f"[JAF:APPROVAL] Successfully stored approval in storage")
            except Exception as e:
                print(f"[JAF:APPROVAL] Approval storage error: {e}")
                # Continue with in-memory fallback
        
        # Update in-memory state
        new_approvals = {**state.approvals}
        new_approvals[interruption.tool_call.id] = approval_value
        
        return replace(state, approvals=new_approvals)
    
    return state


async def load_approvals_into_state(
    state: RunState[Any],
    config: Optional[RunConfig[Any]] = None
) -> RunState[Any]:
    """
    Load approvals from storage into the run state.
    
    Args:
        state: Current run state
        config: Optional run configuration with approval storage
        
    Returns:
        Updated run state with loaded approvals
    """
    if not config or not config.approval_storage:
        print(f"[JAF:APPROVAL] No approval storage configured, using existing approvals: {state.approvals}")
        return state
    
    try:
        print(f"[JAF:APPROVAL] Loading approvals from storage for run_id: {state.run_id}")
        result = await config.approval_storage.get_run_approvals(state.run_id)
        if result.success and result.data:
            print(f"[JAF:APPROVAL] Loaded {len(result.data)} approvals from storage: {result.data}")
            return replace(state, approvals=result.data)
        else:
            if not result.success:
                print(f"[JAF:APPROVAL] Failed to load approvals: {result.error}")
            else:
                print(f"[JAF:APPROVAL] No approvals found in storage for run_id: {state.run_id}")
            return state
    except Exception as e:
        print(f"[JAF:APPROVAL] Approval loading error: {e}")
        return state