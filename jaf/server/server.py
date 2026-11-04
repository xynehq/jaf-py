"""
FastAPI-based HTTP server for the JAF framework.

This module implements a complete HTTP server that exposes JAF agents
via REST API endpoints with proper error handling and validation.
"""

import logging
import time
import uuid
import asyncio
import json
from dataclasses import asdict, replace
from typing import Any, Dict, Optional, Set, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..policies.validation import rate_limit_key
from ..core.engine import run
from ..core.streaming import run_streaming
from ..core.regeneration import regenerate_conversation, get_regeneration_points
from ..core.checkpoint import checkpoint_conversation, get_checkpoint_history
from ..core.types import (
    ApprovalValue,
    CompletedOutcome,
    ErrorOutcome,
    InterruptedOutcome,
    Message,
    MessageContentPart,
    Attachment,
    RunState,
    create_run_id,
    create_trace_id,
    create_message_id,
    RegenerationRequest,
    CheckpointRequest,
)
from ..memory.types import MemoryConfig
from .types import (
    AgentInfo,
    AgentListData,
    AgentListResponse,
    ApprovalMessage,
    BaseOutcomeData,
    ChatRequest,
    ChatResponse,
    CompletedChatData,
    ConversationData,
    ConversationResponse,
    DeleteConversationData,
    DeleteConversationResponse,
    HealthResponse,
    HttpMessage,
    InterruptedOutcomeData,
    InterruptionData,
    MemoryHealthResponse,
    PendingApprovalData,
    PendingApprovalsData,
    PendingApprovalsResponse,
    RegenerationHttpRequest,
    RegenerationData,
    RegenerationResponse,
    RegenerationPointData,
    RegenerationHistoryData,
    RegenerationHistoryResponse,
    CheckpointHttpRequest,
    CheckpointData,
    CheckpointResponse,
    CheckpointPointData,
    CheckpointHistoryData,
    CheckpointHistoryResponse,
    ServerConfig,
    ToolCallInterruption,
    validate_regeneration_request,
)

logger = logging.getLogger(__name__)

Ctx = TypeVar("Ctx")


# Helper functions for HITL (moved outside like TypeScript)
def stable_stringify(value) -> str:
    """Create deterministic JSON string for tool call signatures."""
    try:
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def try_parse_json(s: str):
    """Try to parse JSON, return original string if it fails."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def compute_tool_call_signature(tool_call) -> str:
    """Compute deterministic signature for tool call matching."""
    try:
        args = try_parse_json(tool_call.function.arguments)
        return f"{tool_call.function.name}:{stable_stringify(args)}"
    except Exception:
        return f"{tool_call.function.name}:unknown"


def _convert_http_message_to_core(http_msg: HttpMessage) -> Message:
    """Convert HTTP message format to core Message format."""
    # Convert content
    if isinstance(http_msg.content, str):
        content = http_msg.content
    else:
        # Convert list of content parts
        content_parts = []
        for i, part in enumerate(http_msg.content):
            if part.type == "text":
                content_parts.append(
                    MessageContentPart(type="text", text=part.text, image_url=None, file=None)
                )
            elif part.type == "image_url":
                content_parts.append(
                    MessageContentPart(
                        type="image_url", text=None, image_url=part.image_url, file=None
                    )
                )
            elif part.type == "file":
                content_parts.append(
                    MessageContentPart(type="file", text=None, image_url=None, file=part.file)
                )
            else:
                # Raise explicit error for unrecognized part types
                raise ValueError(
                    f"Unrecognized message content part type: '{part.type}' at index {i}. "
                    f"Supported types are: 'text', 'image_url', 'file'"
                )
        content = content_parts

    # Convert attachments
    attachments = None
    if http_msg.attachments:
        attachments = [
            Attachment(
                kind=att.kind,
                mime_type=att.mime_type,
                name=att.name,
                url=att.url,
                data=att.data,
                format=att.format,
                use_litellm_format=att.use_litellm_format,
            )
            for att in http_msg.attachments
        ]

    return Message(
        role=http_msg.role,
        content=content,
        attachments=attachments,
        tool_call_id=http_msg.tool_call_id,
        tool_calls=http_msg.tool_calls,
    )


def _convert_core_message_to_http(core_msg: Message) -> HttpMessage:
    """Convert core Message format to HTTP message format."""
    from .types import HttpAttachment, HttpMessageContentPart
    from ..core.types import get_text_content

    # Convert content
    if isinstance(core_msg.content, str):
        content = core_msg.content
    elif isinstance(core_msg.content, list):
        # Convert content parts to HTTP format
        http_parts = []
        for i, part in enumerate(core_msg.content):
            if part.type == "text":
                http_parts.append(
                    HttpMessageContentPart(type="text", text=part.text, image_url=None, file=None)
                )
            elif part.type == "image_url":
                http_parts.append(
                    HttpMessageContentPart(
                        type="image_url", text=None, image_url=part.image_url, file=None
                    )
                )
            elif part.type == "file":
                http_parts.append(
                    HttpMessageContentPart(type="file", text=None, image_url=None, file=part.file)
                )
            else:
                # Raise explicit error for unrecognized part types
                message_info = f"role={core_msg.role}"
                raise ValueError(
                    f"Unrecognized core message content part type: '{part.type}' at index {i}. "
                    f"Message info: {message_info}. "
                    f"Supported types are: 'text', 'image_url', 'file'"
                )
        content = http_parts
    else:
        content = get_text_content(core_msg.content)

    # Convert attachments
    attachments = None
    if core_msg.attachments:
        attachments = [
            HttpAttachment(
                kind=att.kind,
                mime_type=att.mime_type,
                name=att.name,
                url=att.url,
                data=att.data,
                format=att.format,
                use_litellm_format=att.use_litellm_format,
            )
            for att in core_msg.attachments
        ]

    return HttpMessage(
        role=core_msg.role,
        content=content,
        attachments=attachments,
        tool_call_id=core_msg.tool_call_id,
        tool_calls=[asdict(tc) for tc in core_msg.tool_calls] if core_msg.tool_calls else None,
    )


def create_jaf_server(config: ServerConfig[Ctx]) -> FastAPI:
    """Create and configure a JAF server instance."""

    start_time = time.time()

    # SSE subscribers for approval-related events (matching TypeScript)
    approval_subscribers = set()

    def sse_send(response, event: str, data: dict):
        """Send SSE event to client."""
        try:
            response.write(f"event: {event}\n")
            response.write(f"data: {json.dumps(data)}\n\n")
        except Exception:
            pass  # ignore connection errors

    def broadcast_approval_required(payload: dict):
        """Broadcast approval_required event to SSE clients."""
        for client in approval_subscribers.copy():  # copy to avoid modification during iteration
            filter_conv_id = client.get("filter_conversation_id")
            if filter_conv_id and filter_conv_id != payload.get("conversationId"):
                continue

            payload_with_timestamp = {
                **payload,
                "timestamp": payload.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
            }
            sse_send(client["response"], "approval_required", payload_with_timestamp)

    def broadcast_approval_decision(payload: dict):
        """Broadcast approval_decision event to SSE clients."""
        for client in approval_subscribers.copy():  # copy to avoid modification during iteration
            filter_conv_id = client.get("filter_conversation_id")
            if filter_conv_id and filter_conv_id != payload.get("conversationId"):
                continue

            payload_with_timestamp = {
                **payload,
                "timestamp": payload.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
            }
            sse_send(client["response"], "approval_decision", payload_with_timestamp)

    app = FastAPI(
        title="JAF Agent Framework Server",
        description="HTTP API for JAF agents with HITL support",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Setup middleware
    if config.cors is not False:
        cors_options: Dict[str, Any] = {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        if isinstance(config.cors, dict):
            cors_options.update(config.cors)
        # Credentials must never be combined with a wildcard origin
        if "*" in cors_options["allow_origins"]:
            cors_options["allow_credentials"] = False
        app.add_middleware(CORSMiddleware, **cors_options)

    async def _require_user(request: Request) -> Optional[str]:
        """Resolve the caller; fail closed unless unauthenticated access is explicitly allowed."""
        if config.authenticate is None:
            if config.allow_unauthenticated_memory_access:
                return None
            raise HTTPException(status_code=401, detail="Authentication required")
        caller_id = await config.authenticate(request)
        if not caller_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        return caller_id

    async def _authorize_conversation(
        request: Request, conversation_id: str, allow_new: bool = False
    ) -> Optional[str]:
        """Require an authenticated caller who owns the conversation."""
        caller_id = await _require_user(request)
        if caller_id is None or not config.default_memory_provider:
            return caller_id

        existing = await config.default_memory_provider.get_conversation(conversation_id)
        if hasattr(existing, "error"):
            raise HTTPException(status_code=500, detail="Failed to load conversation")
        if not existing.data:
            if allow_new:
                return caller_id
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Same response for missing and foreign conversations to avoid ID enumeration
        if existing.data.user_id != caller_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return caller_id

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            version="2.0.0",
            uptime=int((time.time() - start_time) * 1000),
        )

    @app.get("/agents", response_model=AgentListResponse)
    async def list_agents():
        try:
            agents = [
                AgentInfo(
                    name=name,
                    description=agent.instructions(None) if agent.instructions else "",
                    tools=[tool.schema.name for tool in agent.tools or []],
                )
                for name, agent in config.agent_registry.items()
            ]
            return AgentListResponse(success=True, data=AgentListData(agents=agents))
        except Exception as e:
            logger.exception("[JAF:SERVER] Unexpected error handling request")
            return AgentListResponse(success=False, error="Internal server error")

    @app.post("/chat", response_model=ChatResponse)
    async def chat_completion(request: ChatRequest, http_request: Request):
        request_start_time = time.time()

        # Authentication is opt-in for /chat; when configured it also binds the conversation owner
        caller_id = None
        if config.authenticate is not None:
            caller_id = await _require_user(http_request)
            if request.conversation_id:
                await _authorize_conversation(http_request, request.conversation_id, allow_new=True)

        try:
            # Validate request (matching TypeScript approach)
            validated_request = (
                request  # Already validated by FastAPI, but keeping TypeScript structure
            )

            # Check if agent exists (matching TypeScript response pattern)
            if validated_request.agent_name not in config.agent_registry:
                logger.warning(f"[JAF:SERVER] Unknown agent requested: {validated_request.agent_name!r}")
                return ChatResponse(success=False, error="Invalid agent specified")

            # Convert HTTP messages to JAF messages (matching TypeScript)
            jaf_messages = [
                Message(role="user" if msg.role == "system" else msg.role, content=msg.content)
                for msg in validated_request.messages
            ]

            # Create initial state (matching TypeScript)
            run_id = create_run_id(str(uuid.uuid4()))
            trace_id = create_trace_id(str(uuid.uuid4()))

            # Generate conversationId if not provided (matching TypeScript)
            conversation_id = validated_request.conversation_id or f"conv-{uuid.uuid4()}"

            # Never trust a client-supplied identity when the server authenticates callers
            if caller_id is not None and isinstance(validated_request.context, dict):
                validated_request.context["user_id"] = caller_id
                rate_limit_key.set(caller_id)  # per-user quotas for rate-limit guardrails
        except Exception as e:
            return ChatResponse(success=False, error=f"Invalid request: {str(e)}")

        # Load conversation history to get correct turn count
        initial_turn_count = 0
        if config.default_memory_provider and conversation_id:
            try:
                conversation_result = await config.default_memory_provider.get_conversation(
                    conversation_id
                )
                if hasattr(conversation_result, "data") and conversation_result.data:
                    conversation_data = conversation_result.data
                    if conversation_data.metadata and "turn_count" in conversation_data.metadata:
                        initial_turn_count = conversation_data.metadata["turn_count"]
                        logger.debug(f"[JAF:SERVER] Loaded initial turn_count: {initial_turn_count}")
            except Exception as e:
                logger.warning(f"[JAF:SERVER] Warning: Failed to load conversation history: {e}")

        # Handle approval message(s) if present (matching TypeScript approach)
        initial_approvals = {}  # Will act like TypeScript's Map
        initial_state_messages = jaf_messages

        approvals_list = validated_request.approvals or []

        def _log_if_failed(result: Any, action: str) -> None:
            error = getattr(result, "error", None)
            if error is not None and not hasattr(result, "data"):
                logger.warning(f"[JAF:SERVER] Failed to {action}: {error}")

        async def persist_approval(conv_id: str, appr: ApprovalMessage):
            """Persist approval to memory provider with metadata (matching TypeScript)."""
            if not config.default_memory_provider:
                return

            provider = config.default_memory_provider
            # Keyed by previous run/session id + toolCallId for uniqueness (matching TypeScript)
            approval_key = f"{appr.session_id}:{appr.tool_call_id}"
            base_entry = {
                "approved": appr.approved,
                "status": "approved" if appr.approved else "rejected",
                "additionalContext": appr.additional_context,
                "sessionId": appr.session_id,
                "toolCallId": appr.tool_call_id,
            }

            try:
                existing = await provider.get_conversation(conv_id)
                existing_data = getattr(existing, "data", None)
                if existing_data:
                    # Try to enrich entry with tool name and signature for robust matching (exactly matching TypeScript)
                    try:
                        msgs = existing_data.messages
                        for i in range(len(msgs) - 1, -1, -1):
                            m = msgs[i]
                            if m.role == "assistant" and hasattr(m, "tool_calls") and m.tool_calls:
                                match = next(
                                    (tc for tc in m.tool_calls if tc.id == appr.tool_call_id), None
                                )
                                if match:
                                    base_entry["toolName"] = match.function.name
                                    base_entry["signature"] = compute_tool_call_signature(match)
                                    break
                    except Exception:
                        pass  # best-effort

                    existing_approvals = (
                        existing_data.metadata.get("toolApprovals")
                        if existing_data.metadata
                        else {}
                    ) or {}
                    prev = existing_approvals.get(approval_key)

                    # Merge additionalContext shallowly and avoid regressions (exactly matching TypeScript)
                    merged_additional = {
                        **(prev.get("additionalContext") if prev else {}),
                        **(base_entry.get("additionalContext") or {}),
                    }

                    next_entry = {
                        **(prev or {}),
                        **base_entry,
                        "additionalContext": merged_additional,
                        # Preserve earliest timestamp if no effective change; else update (exactly matching TypeScript)
                        "timestamp": (
                            prev.get("timestamp")
                            if prev
                            and (
                                prev.get("status") == base_entry["status"]
                                and stable_stringify(prev.get("additionalContext"))
                                == stable_stringify(merged_additional)
                            )
                            else time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        ),
                        # Epoch seconds; the engine uses it to expire signature-matched approvals
                        "approvedAt": time.time(),
                    }

                    # Check if there's actually a change (exactly matching TypeScript)
                    no_change = prev and (
                        prev.get("status") == next_entry["status"]
                        and stable_stringify(prev.get("additionalContext"))
                        == stable_stringify(next_entry["additionalContext"])
                        and (prev.get("toolName") or None) == (next_entry.get("toolName") or None)
                        and (prev.get("signature") or None) == (next_entry.get("signature") or None)
                    )

                    if not no_change:
                        merged_approvals = {**existing_approvals, approval_key: next_entry}
                        saved = await provider.append_messages(
                            conv_id, [], {"toolApprovals": merged_approvals, "traceId": trace_id}
                        )
                        _log_if_failed(saved, "save approval")

                elif existing is not None and not existing_data and hasattr(existing, "data"):
                    # Create conversation shell with just metadata if not present (exactly matching TypeScript)
                    entry = {
                        **base_entry,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "approvedAt": time.time(),
                    }
                    saved = await provider.store_messages(
                        conv_id, [], {"toolApprovals": {approval_key: entry}, "traceId": trace_id}
                    )
                    _log_if_failed(saved, "create conversation for approval")
                # If provider call failed, we intentionally do not throw; run will proceed
            except Exception as e:
                # Do not fail the request, but make the failure visible instead of hiding it
                logger.warning(f"[JAF:SERVER] Failed to persist approval: {e}")

            # Broadcast decision to approvals SSE (exactly matching TypeScript)
            try:
                broadcast_approval_decision(
                    {
                        "conversationId": conv_id,
                        "sessionId": appr.session_id,
                        "toolCallId": appr.tool_call_id,
                        "status": "approved" if appr.approved else "rejected",
                        "additionalContext": appr.additional_context,
                    }
                )
            except Exception:
                pass  # ignore

        if len(approvals_list) > 0:
            for approval in approvals_list:
                if approval.session_id:  # Matching TypeScript condition
                    initial_approvals[approval.tool_call_id] = {
                        "status": "approved" if approval.approved else "rejected",
                        "approved": approval.approved,
                        "additionalContext": approval.additional_context,
                    }
                await persist_approval(conversation_id, approval)

        # Seed approvals from persisted conversation metadata
        if config.default_memory_provider:
            try:
                conv_result = await config.default_memory_provider.get_conversation(conversation_id)
                if hasattr(conv_result, "data") and conv_result.data:
                    conversation_data = conv_result.data
                    tool_approvals = (
                        conversation_data.metadata.get("toolApprovals")
                        if isinstance(conversation_data.metadata, dict)
                        else None
                    ) or {}

                    if tool_approvals:
                        # Find latest assistant message with tool calls for matching
                        assistant_msg = None
                        for msg in reversed(conversation_data.messages):
                            if (
                                hasattr(msg, "role")
                                and msg.role == "assistant"
                                and hasattr(msg, "tool_calls")
                                and msg.tool_calls
                            ):
                                assistant_msg = msg
                                break

                        if assistant_msg:
                            candidate_ids = {tc.id for tc in assistant_msg.tool_calls}
                            candidate_signatures = {
                                tc.id: compute_tool_call_signature(tc)
                                for tc in assistant_msg.tool_calls
                            }

                            # Load persisted approvals that aren't already in initial_approvals
                            for approval_entry in tool_approvals.values():
                                if not isinstance(approval_entry, dict):
                                    continue

                                persisted_tool_call_id = approval_entry.get(
                                    "toolCallId"
                                ) or approval_entry.get("tool_call_id")
                                persisted_signature = approval_entry.get("signature")

                                # Try direct ID match first
                                target_id = None
                                if (
                                    persisted_tool_call_id
                                    and persisted_tool_call_id in candidate_ids
                                ):
                                    target_id = persisted_tool_call_id
                                elif persisted_signature:
                                    # Signature fallback
                                    for tc_id, sig in candidate_signatures.items():
                                        if sig == persisted_signature:
                                            target_id = tc_id
                                            break

                                if target_id and target_id not in initial_approvals:
                                    status = approval_entry.get("status", "pending")
                                    if approval_entry.get("approved") is True:
                                        status = "approved"
                                    elif approval_entry.get("approved") is False:
                                        status = "rejected"

                                    context_data = dict(
                                        approval_entry.get("additionalContext")
                                        or approval_entry.get("additional_context")
                                        or {}
                                    )
                                    if isinstance(approval_entry.get("approvedAt"), (int, float)):
                                        context_data["approved_at"] = approval_entry["approvedAt"]

                                    initial_approvals[target_id] = ApprovalValue(
                                        status=status,
                                        approved=approval_entry.get("approved", False),
                                        additional_context=context_data or None,
                                    )

            except Exception as e:
                logger.warning(f"[JAF:SERVER] Warning: Failed to seed approvals from metadata: {e}")

        initial_state = RunState(
            run_id=create_run_id(str(uuid.uuid4())),
            trace_id=create_trace_id(str(uuid.uuid4())),
            messages=[
                _convert_http_message_to_core(
                    msg.model_copy(update={"role": "user"})
                    if msg.role == "system" and not config.allow_client_system_messages
                    else msg
                )
                for msg in request.messages
            ],
            current_agent_name=request.agent_name,
            context=request.context or {},
            turn_count=initial_turn_count,  # Use loaded turn count instead of always 0
            approvals=initial_approvals,
        )

        run_config_with_memory = config.run_config
        if config.default_memory_provider:
            # Handle memory configuration with request overrides (matching TypeScript)
            memory_config = MemoryConfig(
                provider=config.default_memory_provider,
                auto_store=request.memory.get("auto_store", True) if request.memory else True,
                max_messages=request.memory.get("max_messages") if request.memory else None,
                compression_threshold=request.memory.get("compression_threshold")
                if request.memory
                else None,
                store_on_completion=request.store_on_completion
                if request.store_on_completion is not None
                else True,
            )
            run_config_with_memory = replace(
                run_config_with_memory, memory=memory_config, conversation_id=conversation_id
            )

        if request.max_turns is not None:
            run_config_with_memory = replace(run_config_with_memory, max_turns=request.max_turns)

        # Handle streaming vs non-streaming (matching TypeScript)
        if request.stream:

            async def event_stream():
                try:
                    # Send initial metadata
                    yield f"""event: stream_start data: {
                        json.dumps(
                            {
                                "runId": str(initial_state.run_id),
                                "traceId": str(initial_state.trace_id),
                                "conversationId": conversation_id,
                                "agent": request.agent_name,
                            }
                        )
                    }"""

                    # Stream events from the engine
                    async for event in run_streaming(initial_state, run_config_with_memory):
                        yield f"event: {event.type}\ndata: {json.dumps(asdict(event))}\n\n"

                        # Check for run end and handle approval broadcasts
                        if event.type == "complete" and hasattr(event, "data"):
                            outcome = getattr(event.data, "outcome", None)
                            if outcome and getattr(outcome, "status", None) == "interrupted":
                                interruptions = getattr(outcome, "interruptions", [])
                                for intr in interruptions:
                                    if getattr(intr, "type", None) == "tool_approval":
                                        tool_call = getattr(intr, "tool_call", None)
                                        if tool_call:
                                            broadcast_approval_required(
                                                {
                                                    "conversationId": conversation_id,
                                                    "sessionId": getattr(intr, "session_id", None)
                                                    or str(initial_state.run_id),
                                                    "toolCallId": tool_call.id,
                                                    "toolName": tool_call.function.name,
                                                    "args": try_parse_json(
                                                        tool_call.function.arguments
                                                    ),
                                                    "signature": compute_tool_call_signature(
                                                        tool_call
                                                    ),
                                                }
                                            )
                            break

                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
                finally:
                    yield f"event: stream_end\ndata: {json.dumps({'ended': True})}\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming execution
        result = await run(initial_state, run_config_with_memory)

        http_messages = [_convert_core_message_to_http(msg) for msg in result.final_state.messages]

        # Create proper outcome object
        if isinstance(result.outcome, CompletedOutcome):
            outcome_data = BaseOutcomeData(status="completed", output=result.outcome.output)
        elif isinstance(result.outcome, ErrorOutcome):
            error_info = result.outcome.error
            outcome_data = BaseOutcomeData(
                status="error",
                error={"type": error_info.__class__.__name__, "message": str(error_info)},
            )
        elif isinstance(result.outcome, InterruptedOutcome):
            # Convert interruptions to response format
            interruptions = []
            for interruption in result.outcome.interruptions:
                if hasattr(interruption, "tool_call") and hasattr(interruption, "type"):
                    tool_call_data = ToolCallInterruption(
                        id=interruption.tool_call.id,
                        function={
                            "name": interruption.tool_call.function.name,
                            "arguments": interruption.tool_call.function.arguments,
                        },
                    )
                    interruptions.append(
                        InterruptionData(
                            type="tool_approval",
                            tool_call=tool_call_data,
                            session_id=interruption.session_id or str(result.final_state.run_id),
                        )
                    )

                    # Broadcast approval request via SSE
                    broadcast_approval_required(
                        {
                            "conversationId": conversation_id,
                            "sessionId": interruption.session_id or str(result.final_state.run_id),
                            "toolCallId": interruption.tool_call.id,
                            "toolName": interruption.tool_call.function.name,
                            "args": try_parse_json(interruption.tool_call.function.arguments),
                            "signature": compute_tool_call_signature(interruption.tool_call),
                        }
                    )

            outcome_data = InterruptedOutcomeData(status="interrupted", interruptions=interruptions)
        else:
            outcome_data = BaseOutcomeData(status="error", error="Unknown outcome type")

        return ChatResponse(
            success=True,
            data=CompletedChatData(
                run_id=str(result.final_state.run_id),
                trace_id=str(result.final_state.trace_id),
                messages=http_messages,
                outcome=outcome_data,
                turn_count=result.final_state.turn_count,
                execution_time_ms=int(
                    (time.time() - request_start_time) * 1000
                ),  # Use request start time
                conversation_id=conversation_id,
            ),
        )

    # Memory endpoints
    if config.default_memory_provider:

        @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
        async def get_conversation(conversation_id: str, request: Request):
            await _authorize_conversation(request, conversation_id)
            result = await config.default_memory_provider.get_conversation(conversation_id)

            # Handle Result type properly
            if hasattr(result, "error"):  # Failure case
                logger.error(f"[JAF:SERVER] Memory provider error: {result.error.message}")
                raise HTTPException(status_code=500, detail="Memory provider error")

            conversation = result.data
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Convert ConversationMemory to ConversationData
            conversation_data = ConversationData(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=[asdict(msg) for msg in conversation.messages],
                metadata=conversation.metadata,
            )

            return ConversationResponse(success=True, data=conversation_data)

        @app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
        async def delete_conversation(conversation_id: str, request: Request):
            await _authorize_conversation(request, conversation_id)
            result = await config.default_memory_provider.delete_conversation(conversation_id)

            # Handle Result type properly
            if hasattr(result, "error"):  # Failure case
                logger.error(f"[JAF:SERVER] Memory provider error: {result.error.message}")
                raise HTTPException(status_code=500, detail="Memory provider error")

            return DeleteConversationResponse(
                success=True,
                data=DeleteConversationData(conversation_id=conversation_id, deleted=result.data),
            )

        @app.get("/memory/health", response_model=MemoryHealthResponse)
        async def memory_health():
            result = await config.default_memory_provider.health_check()

            # Handle Result type properly
            if hasattr(result, "error"):  # Failure case
                logger.error(f"[JAF:SERVER] Memory provider error: {result.error.message}")
                raise HTTPException(status_code=500, detail="Memory provider error")

            return MemoryHealthResponse(success=True, data=result.data)

    # Approval endpoints for HITL functionality
    @app.get("/approvals/pending", response_model=PendingApprovalsResponse)
    async def get_pending_approvals(request: Request, conversation_id: str = None):
        """Get pending approvals for a conversation."""
        try:
            if not conversation_id:
                raise HTTPException(status_code=400, detail="conversation_id is required")
            await _authorize_conversation(request, conversation_id)

            if not config.default_memory_provider:
                return PendingApprovalsResponse(
                    success=False, error="Memory provider not configured"
                )

            # Get conversation to analyze pending approvals
            conv_result = await config.default_memory_provider.get_conversation(conversation_id)
            if hasattr(conv_result, "error"):
                logger.error(f"[JAF:SERVER] Memory provider error: {conv_result.error.message}")
                return PendingApprovalsResponse(success=False, error="Memory provider error")

            if not conv_result.data:
                return PendingApprovalsResponse(success=True, data=PendingApprovalsData(pending=[]))

            conversation = conv_result.data
            messages = conversation.messages
            approvals_meta = (
                getattr(conversation.metadata, "tool_approvals", {})
                if conversation.metadata
                else {}
            )

            # Find most recent assistant message with tool calls
            assistant_msg = None
            assistant_index = -1
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if (
                    hasattr(msg, "role")
                    and msg.role == "assistant"
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    assistant_msg = msg
                    assistant_index = i
                    break

            if not assistant_msg:
                return PendingApprovalsResponse(success=True, data=PendingApprovalsData(pending=[]))

            # Check which tool calls have already been executed
            tool_ids = {tc.id for tc in assistant_msg.tool_calls}
            executed = set()
            for j in range(assistant_index + 1, len(messages)):
                msg = messages[j]
                if hasattr(msg, "role") and msg.role == "tool" and hasattr(msg, "tool_call_id"):
                    if msg.tool_call_id in tool_ids:
                        executed.add(msg.tool_call_id)

            # Build pending approvals list
            pending_approvals = []
            for tc in assistant_msg.tool_calls:
                if tc.id in executed:
                    continue  # Already executed

                # Check approval status
                approval_key = f"{conversation.conversation_id}:{tc.id}"
                approval_entry = approvals_meta.get(approval_key)

                status = "pending"
                if approval_entry:
                    status = approval_entry.get("status", "pending")
                    if approval_entry.get("approved") is True:
                        status = "approved"
                    elif approval_entry.get("approved") is False:
                        status = "rejected"

                if status == "pending":
                    pending_approvals.append(
                        PendingApprovalData(
                            conversation_id=conversation_id,
                            tool_call_id=tc.id,
                            tool_name=tc.function.name,
                            args=try_parse_json(tc.function.arguments),
                            signature=compute_tool_call_signature(tc),
                            status="pending",
                            session_id=getattr(conversation.metadata, "run_id", None)
                            if conversation.metadata
                            else None,
                        )
                    )

            return PendingApprovalsResponse(
                success=True, data=PendingApprovalsData(pending=pending_approvals)
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[JAF:SERVER] Unexpected error handling request")
            return PendingApprovalsResponse(success=False, error="Internal server error")

    # Agent-specific chat endpoint (convenience - matching TypeScript)
    @app.post("/agents/{agent_name}/chat", response_model=ChatResponse)
    async def agent_chat_completion(
        agent_name: str, request_body: ChatRequest, http_request: Request
    ):
        """Agent-specific chat endpoint for convenience."""
        # Create modified request with agent name
        modified_request = ChatRequest(
            messages=request_body.messages,
            agent_name=agent_name,
            context=request_body.context,
            max_turns=request_body.max_turns,
            stream=request_body.stream,
            conversation_id=request_body.conversation_id,
            memory=request_body.memory,
            approvals=request_body.approvals,
        )

        # Delegate to main chat endpoint logic
        return await chat_completion(modified_request, http_request)

    # Approvals SSE stream endpoint (matching TypeScript placement inside start function)
    @app.get("/approvals/stream")
    async def stream_approval_updates(request: Request, conversation_id: str = None):
        """Stream real-time approval updates via Server-Sent Events."""
        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required")
        await _authorize_conversation(request, conversation_id)

        async def event_stream():
            # Simple client structure matching TypeScript
            client = {
                "response": request,  # Store request for disconnection check
                "filter_conversation_id": conversation_id,
            }
            approval_subscribers.add(client)

            try:
                # Initial greeting (matching TypeScript)
                yield f"event: stream_start\ndata: {json.dumps({'conversationId': conversation_id})}\n\n"

                # Heartbeat like TypeScript (15 second interval)
                last_heartbeat = time.time()

                while True:
                    # Check client disconnection
                    if await request.is_disconnected():
                        break

                    # Send heartbeat every 15 seconds
                    current_time = time.time()
                    if current_time - last_heartbeat >= 15:
                        yield f"event: ping\ndata: {json.dumps({'ts': int(current_time * 1000)})}\n\n"
                        last_heartbeat = current_time

                    await asyncio.sleep(1)

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            finally:
                approval_subscribers.discard(client)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Regeneration endpoints
    if config.default_memory_provider:

        @app.post(
            "/conversations/{conversation_id}/regenerate", response_model=RegenerationResponse
        )
        async def regenerate_conversation_endpoint(
            conversation_id: str, request: RegenerationHttpRequest, http_request: Request
        ):
            """Regenerate conversation from a specific message."""
            await _authorize_conversation(http_request, conversation_id)
            request_start_time = time.time()

            try:
                # Validate agent exists
                if request.agent_name not in config.agent_registry:
                    logger.warning(f"[JAF:SERVER] Unknown agent requested: {request.agent_name!r}")
                    return RegenerationResponse(success=False, error="Invalid agent specified")

                # Create regeneration request
                regen_request = RegenerationRequest(
                    conversation_id=conversation_id,
                    message_id=create_message_id(request.message_id),
                    context=request.context,
                )

                # Create run config with memory
                memory_config = MemoryConfig(
                    provider=config.default_memory_provider,
                    auto_store=True,
                    store_on_completion=True,
                )

                run_config_with_memory = replace(
                    config.run_config,
                    memory=memory_config,
                    conversation_id=conversation_id,
                    max_turns=request.max_turns or 10,
                )

                # Execute regeneration
                result = await regenerate_conversation(
                    regen_request, run_config_with_memory, request.context or {}, request.agent_name
                )

                # Convert result to HTTP format
                http_messages = [
                    _convert_core_message_to_http(msg) for msg in result.final_state.messages
                ]

                # Create outcome data
                if isinstance(result.outcome, CompletedOutcome):
                    outcome_data = BaseOutcomeData(status="completed", output=result.outcome.output)
                elif isinstance(result.outcome, ErrorOutcome):
                    error_info = result.outcome.error
                    outcome_data = BaseOutcomeData(
                        status="error",
                        error={"type": error_info.__class__.__name__, "message": str(error_info)},
                    )
                elif isinstance(result.outcome, InterruptedOutcome):
                    interruptions = []
                    for interruption in result.outcome.interruptions:
                        if hasattr(interruption, "tool_call") and hasattr(interruption, "type"):
                            tool_call_data = ToolCallInterruption(
                                id=interruption.tool_call.id,
                                function={
                                    "name": interruption.tool_call.function.name,
                                    "arguments": interruption.tool_call.function.arguments,
                                },
                            )
                            interruptions.append(
                                InterruptionData(
                                    type="tool_approval",
                                    tool_call=tool_call_data,
                                    session_id=interruption.session_id
                                    or str(result.final_state.run_id),
                                )
                            )

                    outcome_data = InterruptedOutcomeData(
                        status="interrupted", interruptions=interruptions
                    )
                else:
                    outcome_data = BaseOutcomeData(status="error", error="Unknown outcome type")

                # Get regeneration metadata from conversation
                conversation_result = await config.default_memory_provider.get_conversation(
                    conversation_id
                )
                regeneration_id = f"regen_{int(time.time() * 1000)}_{request.message_id}"
                original_message_count = 0
                truncated_at_index = 0

                if hasattr(conversation_result, "data") and conversation_result.data:
                    conversation_data = conversation_result.data
                    regeneration_points = (
                        conversation_data.metadata.get("regeneration_points", [])
                        if conversation_data.metadata
                        else []
                    )
                    if regeneration_points:
                        latest_regen = regeneration_points[-1]
                        original_message_count = latest_regen.get(
                            "original_message_count", len(conversation_data.messages)
                        )
                        truncated_at_index = latest_regen.get("truncated_at_index", 0)
                        regeneration_id = latest_regen.get("regeneration_id", regeneration_id)

                return RegenerationResponse(
                    success=True,
                    data=RegenerationData(
                        regeneration_id=regeneration_id,
                        conversation_id=conversation_id,
                        original_message_count=original_message_count,
                        truncated_at_index=truncated_at_index,
                        regenerated_message_id=request.message_id,
                        messages=http_messages,
                        outcome=outcome_data,
                        turn_count=result.final_state.turn_count,
                        execution_time_ms=int((time.time() - request_start_time) * 1000),
                    ),
                )

            except Exception as e:
                logger.exception("[JAF:SERVER] Unexpected error handling request")
                return RegenerationResponse(success=False, error="Internal server error")

        @app.get(
            "/conversations/{conversation_id}/regeneration-history",
            response_model=RegenerationHistoryResponse,
        )
        async def get_regeneration_history(conversation_id: str, http_request: Request):
            """Get regeneration history for a conversation."""
            await _authorize_conversation(http_request, conversation_id)
            try:
                regeneration_points = await get_regeneration_points(
                    conversation_id, config.run_config
                )

                if regeneration_points is None:
                    return RegenerationHistoryResponse(
                        success=False, error="Failed to get regeneration history"
                    )

                # Convert to response format
                regeneration_data = []
                for point in regeneration_points:
                    regeneration_data.append(
                        RegenerationPointData(
                            regeneration_id=point.get("regeneration_id", ""),
                            message_id=point.get("message_id", ""),
                            timestamp=point.get("timestamp", 0),
                            original_message_count=point.get("original_message_count", 0),
                            truncated_at_index=point.get("truncated_at_index", 0),
                        )
                    )

                return RegenerationHistoryResponse(
                    success=True,
                    data=RegenerationHistoryData(
                        conversation_id=conversation_id, regeneration_points=regeneration_data
                    ),
                )

            except Exception as e:
                logger.exception("[JAF:SERVER] Unexpected error handling request")
                return RegenerationHistoryResponse(success=False, error="Internal server error")

        # Checkpoint endpoints
        @app.post("/conversations/{conversation_id}/checkpoint", response_model=CheckpointResponse)
        async def checkpoint_conversation_endpoint(
            conversation_id: str, request: CheckpointHttpRequest, http_request: Request
        ):
            """Checkpoint conversation after a specific message."""
            await _authorize_conversation(http_request, conversation_id)
            request_start_time = time.time()

            try:
                # Create checkpoint request
                chk_request = CheckpointRequest(
                    conversation_id=conversation_id,
                    message_id=create_message_id(request.message_id),
                    context=request.context,
                )

                # Create run config with memory
                memory_config = MemoryConfig(
                    provider=config.default_memory_provider,
                    auto_store=True,
                    store_on_completion=True,
                )

                run_config_with_memory = replace(
                    config.run_config, memory=memory_config, conversation_id=conversation_id
                )

                # Execute checkpoint
                result = await checkpoint_conversation(chk_request, run_config_with_memory)

                # Convert result to HTTP format
                http_messages = [_convert_core_message_to_http(msg) for msg in result.messages]

                return CheckpointResponse(
                    success=True,
                    data=CheckpointData(
                        checkpoint_id=result.checkpoint_id,
                        conversation_id=result.conversation_id,
                        original_message_count=result.original_message_count,
                        checkpointed_at_index=result.checkpointed_at_index,
                        checkpointed_message_id=str(result.checkpointed_message_id),
                        messages=http_messages,
                        execution_time_ms=result.execution_time_ms,
                    ),
                )

            except Exception as e:
                logger.exception("[JAF:SERVER] Unexpected error handling request")
                return CheckpointResponse(success=False, error="Internal server error")

        @app.get(
            "/conversations/{conversation_id}/checkpoint-history",
            response_model=CheckpointHistoryResponse,
        )
        async def get_checkpoint_history_endpoint(conversation_id: str, http_request: Request):
            """Get checkpoint history for a conversation."""
            await _authorize_conversation(http_request, conversation_id)
            try:
                checkpoint_points = await get_checkpoint_history(conversation_id, config.run_config)

                if checkpoint_points is None:
                    return CheckpointHistoryResponse(
                        success=False, error="Failed to get checkpoint history"
                    )

                # Convert to response format
                checkpoint_data = []
                for point in checkpoint_points:
                    checkpoint_data.append(
                        CheckpointPointData(
                            checkpoint_id=point.get("checkpoint_id", ""),
                            checkpoint_point=point.get("checkpoint_point", ""),
                            timestamp=point.get("timestamp", 0),
                            original_message_count=point.get("original_message_count", 0),
                            checkpointed_at_index=point.get("checkpointed_at_index", 0),
                            checkpointed_messages=point.get("checkpointed_messages", 0),
                        )
                    )

                return CheckpointHistoryResponse(
                    success=True,
                    data=CheckpointHistoryData(
                        conversation_id=conversation_id, checkpoint_points=checkpoint_data
                    ),
                )

            except Exception as e:
                logger.exception("[JAF:SERVER] Unexpected error handling request")
                return CheckpointHistoryResponse(success=False, error="Internal server error")

    return app
