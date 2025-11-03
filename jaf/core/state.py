"""
State management functions for approval handling in HITL scenarios.

This module provides functions to manage approval state transitions
and integrate with approval storage systems.
"""

from typing import Dict, Any, Optional, List
from dataclasses import replace

from .types import (
    RunState,
    RunConfig,
    Interruption,
    ApprovalValue,
    Message,
    ContentRole,
    Attachment,
)


def _extract_attachments_from_messages(messages: List[Dict[str, Any]]) -> List[Attachment]:
    """Extract attachment objects from message data."""
    attachments = []

    for msg in messages:
        msg_attachments = msg.get("attachments", [])
        for att in msg_attachments:
            try:
                # Convert dict to Attachment object
                attachment = Attachment(
                    kind=att.get("kind", "image"),
                    mime_type=att.get("mime_type"),
                    name=att.get("name"),
                    url=att.get("url"),
                    data=att.get("data"),
                    format=att.get("format"),
                    use_litellm_format=att.get("use_litellm_format"),
                )
                attachments.append(attachment)
            except Exception as e:
                print(f"[JAF:APPROVAL] Failed to process attachment: {e}")

    return attachments


def _process_additional_context_images(
    additional_context: Optional[Dict[str, Any]],
) -> List[Attachment]:
    """Process additional context and extract any image attachments."""
    if not additional_context:
        return []

    attachments = []

    # Handle messages with attachments
    messages = additional_context.get("messages", [])
    if messages:
        attachments.extend(_extract_attachments_from_messages(messages))

    # Handle legacy image_context format
    image_context = additional_context.get("image_context")
    if image_context and image_context.get("type") == "image_url":
        try:
            image_url = image_context.get("image_url", {})
            url = image_url.get("url", "")

            if url.startswith("data:"):
                # Parse data URL: data:image/png;base64,iVBORw0KGgo...
                header, data = url.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]

                attachment = Attachment(
                    kind="image",
                    mime_type=mime_type,
                    data=data,
                    name=f"approval_image.{mime_type.split('/')[-1]}",
                )
                attachments.append(attachment)
        except Exception as e:
            print(f"[JAF:APPROVAL] Failed to process image_context: {e}")

    return attachments


def _add_approval_context_to_conversation(
    state: RunState[Any], additional_context: Optional[Dict[str, Any]]
) -> RunState[Any]:
    """Add approval context including images to the conversation."""
    if not additional_context:
        return state

    # Extract image attachments
    attachments = _process_additional_context_images(additional_context)

    if not attachments:
        return state

    # Create approval context message
    approval_message = "Additional context provided during approval process."

    # Check if there are text messages to include
    messages = additional_context.get("messages", [])
    if messages:
        text_content = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                text_content.append(content)

        if text_content:
            approval_message = f"User provided additional context: {' '.join(text_content)}"

    # Create user message with attachments (using USER role for better compatibility)
    context_message = Message(
        role=ContentRole.USER, content=approval_message, attachments=attachments
    )

    # Add to conversation
    new_messages = state.messages + [context_message]
    return replace(state, messages=new_messages)


async def approve(
    state: RunState[Any],
    interruption: Interruption,
    additional_context: Optional[Dict[str, Any]] = None,
    config: Optional[RunConfig[Any]] = None,
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
    if interruption.type == "tool_approval":
        approval_value = ApprovalValue(
            status="approved",
            approved=True,
            additional_context={**(additional_context or {}), "status": "approved"},
        )

        # Store in approval storage if available
        if config and config.approval_storage:
            try:
                print(
                    f"[JAF:APPROVAL] Storing approval for tool_call_id {interruption.tool_call.id}: {approval_value}"
                )
                result = await config.approval_storage.store_approval(
                    state.run_id, interruption.tool_call.id, approval_value
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

        # Process any image context and add to conversation
        updated_state = replace(state, approvals=new_approvals)
        updated_state = _add_approval_context_to_conversation(updated_state, additional_context)

        return updated_state

    return state


async def reject(
    state: RunState[Any],
    interruption: Interruption,
    additional_context: Optional[Dict[str, Any]] = None,
    config: Optional[RunConfig[Any]] = None,
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
    if interruption.type == "tool_approval":
        approval_value = ApprovalValue(
            status="rejected",
            approved=False,
            additional_context={**(additional_context or {}), "status": "rejected"},
        )

        # Store in approval storage if available
        if config and config.approval_storage:
            try:
                print(
                    f"[JAF:APPROVAL] Storing approval for tool_call_id {interruption.tool_call.id}: {approval_value}"
                )
                result = await config.approval_storage.store_approval(
                    state.run_id, interruption.tool_call.id, approval_value
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

        # Process any image context and add to conversation
        updated_state = replace(state, approvals=new_approvals)
        updated_state = _add_approval_context_to_conversation(updated_state, additional_context)

        return updated_state

    return state


async def load_approvals_into_state(
    state: RunState[Any], config: Optional[RunConfig[Any]] = None
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
        print(
            f"[JAF:APPROVAL] No approval storage configured, using existing approvals: {state.approvals}"
        )
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
