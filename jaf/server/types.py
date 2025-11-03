"""
Server types for the JAF framework HTTP API.

This module defines the request/response types and configuration
for the JAF HTTP server implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
import base64

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.types import (
    Agent,
    RunConfig,
    Attachment,
    MessageContentPart,
    get_text_content,
    MessageId,
    RegenerationRequest,
)
from ..memory.types import MemoryProvider

Ctx = TypeVar("Ctx")


# Pydantic models for attachments to work with HTTP API
class HttpAttachment(BaseModel):
    """HTTP attachment format for API requests."""

    kind: Literal["image", "document", "file"]
    mime_type: Optional[str] = None
    name: Optional[str] = None
    url: Optional[str] = None
    data: Optional[str] = None  # Base64 encoded data
    format: Optional[str] = None
    use_litellm_format: Optional[bool] = None

    @model_validator(mode="after")
    def validate_url_or_data_present(self) -> "HttpAttachment":
        """Validate that at least one of url or data is present."""
        if self.url is None and self.data is None:
            raise ValueError("At least one of 'url' or 'data' must be provided")
        return self

    @field_validator("data")
    @classmethod
    def validate_base64_data(cls, v: Optional[str]) -> Optional[str]:
        """Validate that data is proper base64 encoded."""
        if v is not None:
            try:
                # Try to decode the base64 data to verify it's valid
                decoded = base64.b64decode(v)
                # Check if it's empty
                if len(decoded) == 0:
                    raise ValueError("Base64 data decodes to empty content")
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding: {str(e)}")
        return v

    @model_validator(mode="after")
    def validate_mime_type_consistency(self) -> "HttpAttachment":
        """Validate that mime_type is consistent with kind."""
        if self.mime_type is not None and self.kind is not None:
            if self.kind == "image" and not self.mime_type.startswith("image/"):
                raise ValueError(
                    f"For kind='image', mime_type must start with 'image/'. Got: {self.mime_type}"
                )

            elif self.kind == "document" and not (
                self.mime_type.startswith("application/")
                or self.mime_type.startswith("text/")
                or self.mime_type.startswith("document/")
            ):
                raise ValueError(
                    f"For kind='document', mime_type must start with 'application/', 'text/', "
                    f"or 'document/'. Got: {self.mime_type}"
                )
        return self


class HttpMessageContentPart(BaseModel):
    """HTTP message content part for multi-part messages."""

    type: Literal["text", "image_url", "file"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    file: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_content_consistency(self) -> "HttpMessageContentPart":
        """Validate that exactly one field is populated and it matches the declared type."""
        # Count non-None content fields
        populated_fields = []
        if self.text is not None:
            populated_fields.append("text")
        if self.image_url is not None:
            populated_fields.append("image_url")
        if self.file is not None:
            populated_fields.append("file")

        # Check if exactly one field is populated
        if len(populated_fields) != 1:
            raise ValueError(
                f"Exactly one content field must be populated. Found {len(populated_fields)}: {populated_fields}"
            )

        # Check that the populated field matches the declared type
        populated_field = populated_fields[0]
        if self.type == "text" and populated_field != "text":
            raise ValueError(
                f"For type='text', the 'text' field must be populated, but found '{populated_field}' instead"
            )
        elif self.type == "image_url" and populated_field != "image_url":
            raise ValueError(
                f"For type='image_url', the 'image_url' field must be populated, but found '{populated_field}' instead"
            )
        elif self.type == "file" and populated_field != "file":
            raise ValueError(
                f"For type='file', the 'file' field must be populated, but found '{populated_field}' instead"
            )

        return self


# HTTP Message types
class HttpMessage(BaseModel):
    """HTTP message format for API requests."""

    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, List[HttpMessageContentPart]]
    attachments: Optional[List[HttpAttachment]] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


# Approval types for HITL
class ApprovalMessage(BaseModel):
    """Approval message for tool execution."""

    type: Literal["approval"] = "approval"
    session_id: str = Field(..., description="Session ID for the approval")
    tool_call_id: str = Field(..., description="ID of the tool call being approved")
    approved: bool = Field(..., description="Whether the tool execution is approved")
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for the approval"
    )


# Request types
class ChatRequest(BaseModel):
    """Request format for chat endpoints."""

    messages: List[HttpMessage]
    agent_name: str = Field(..., description="Name of the agent to use")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Context data for the agent"
    )
    max_turns: Optional[int] = Field(default=10, description="Maximum number of turns")
    stream: bool = Field(default=False, description="Whether to stream the response")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for memory persistence"
    )
    memory: Optional[Dict[str, Any]] = Field(
        default=None, description="Memory configuration override"
    )
    store_on_completion: Optional[bool] = Field(
        default=None, description="Whether to store conversation on completion"
    )
    approvals: Optional[List[ApprovalMessage]] = Field(
        default=None, description="Approval decisions for tool calls"
    )


# Interruption types for HITL
class ToolCallInterruption(BaseModel):
    """Tool call interruption data."""

    id: str
    type: Literal["function"] = "function"
    function: Dict[str, str]  # name and arguments


class InterruptionData(BaseModel):
    """Interruption information."""

    type: Literal["tool_approval"] = "tool_approval"
    tool_call: Optional[ToolCallInterruption]
    session_id: str


# Base outcome types
class BaseOutcomeData(BaseModel):
    """Base outcome data."""

    status: Literal["completed", "error", "max_turns", "interrupted"]
    output: Optional[str] = None
    error: Optional[Any] = None


class InterruptedOutcomeData(BaseOutcomeData):
    """Outcome data for interrupted runs."""

    status: Literal["interrupted"] = "interrupted"
    interruptions: Optional[List[InterruptionData]] = None


# Response types
class CompletedChatData(BaseModel):
    """Data for successful chat completion."""

    run_id: str
    trace_id: str
    messages: List[HttpMessage]
    outcome: BaseOutcomeData
    turn_count: int
    execution_time_ms: int
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response format for chat endpoints."""

    success: bool
    data: Optional[CompletedChatData] = None
    error: Optional[str] = None


class AgentInfo(BaseModel):
    """Information about an available agent."""

    name: str
    description: str
    tools: List[str]


class AgentListData(BaseModel):
    """Data for agent list response."""

    agents: List[AgentInfo]


class AgentListResponse(BaseModel):
    """Response format for agent list endpoint."""

    success: bool
    data: Optional[AgentListData] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response format for health check endpoint."""

    status: Literal["healthy", "unhealthy"]
    timestamp: str
    version: str
    uptime: int  # milliseconds


# Memory-specific response types
class ConversationData(BaseModel):
    """Data for a conversation."""

    conversation_id: str
    user_id: Optional[str] = None
    messages: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response format for conversation endpoints."""

    success: bool
    data: Optional[ConversationData] = None
    error: Optional[str] = None


class MemoryHealthData(BaseModel):
    """Data for memory health check."""

    healthy: bool
    provider: str
    latency_ms: float
    details: Optional[Dict[str, Any]] = None


class MemoryHealthResponse(BaseModel):
    """Response format for memory health endpoint."""

    success: bool
    data: Optional[MemoryHealthData] = None
    error: Optional[str] = None


class DeleteConversationData(BaseModel):
    """Data for delete conversation response."""

    conversation_id: str
    deleted: bool


class DeleteConversationResponse(BaseModel):
    """Response format for delete conversation endpoint."""

    success: bool
    data: Optional[DeleteConversationData] = None
    error: Optional[str] = None


# Server configuration
@dataclass
class ServerConfig(Generic[Ctx]):
    """Configuration for the JAF HTTP server."""

    agent_registry: Dict[str, Agent[Ctx, Any]]
    run_config: RunConfig[Ctx]
    host: str = "127.0.0.1"
    port: int = 3000
    cors: Union[bool, Dict[str, Any]] = True
    default_memory_provider: Optional[MemoryProvider] = None


# Approval response types
class PendingApprovalData(BaseModel):
    """Data for a pending approval."""

    conversation_id: str
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    signature: Optional[str] = None
    status: Literal["pending"] = "pending"
    session_id: Optional[str] = None


class PendingApprovalsData(BaseModel):
    """Data for pending approvals response."""

    pending: List[PendingApprovalData]


class PendingApprovalsResponse(BaseModel):
    """Response format for pending approvals endpoint."""

    success: bool
    data: Optional[PendingApprovalsData] = None
    error: Optional[str] = None


# Regeneration types
class RegenerationHttpRequest(BaseModel):
    """HTTP request format for conversation regeneration."""

    message_id: str = Field(..., description="ID of the message to regenerate from")
    agent_name: str = Field(..., description="Name of the agent to use for regeneration")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context override for regeneration"
    )
    max_turns: Optional[int] = Field(
        default=10, description="Maximum number of turns for regeneration"
    )


class RegenerationData(BaseModel):
    """Data for successful regeneration response."""

    regeneration_id: str
    conversation_id: str
    original_message_count: int
    truncated_at_index: int
    regenerated_message_id: str
    messages: List[HttpMessage]
    outcome: BaseOutcomeData
    turn_count: int
    execution_time_ms: int


class RegenerationResponse(BaseModel):
    """Response format for regeneration endpoints."""

    success: bool
    data: Optional[RegenerationData] = None
    error: Optional[str] = None


class RegenerationPointData(BaseModel):
    """Data for a regeneration point."""

    regeneration_id: str
    message_id: str
    timestamp: int
    original_message_count: int
    truncated_at_index: int


class RegenerationHistoryData(BaseModel):
    """Data for regeneration history response."""

    conversation_id: str
    regeneration_points: List[RegenerationPointData]


class RegenerationHistoryResponse(BaseModel):
    """Response format for regeneration history endpoint."""

    success: bool
    data: Optional[RegenerationHistoryData] = None
    error: Optional[str] = None


# Checkpoint types
class CheckpointHttpRequest(BaseModel):
    """HTTP request format for conversation checkpoint."""

    message_id: str = Field(
        ..., description="ID of the message to checkpoint after (this message is kept)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context for the checkpoint"
    )


class CheckpointData(BaseModel):
    """Data for successful checkpoint response."""

    checkpoint_id: str
    conversation_id: str
    original_message_count: int
    checkpointed_at_index: int
    checkpointed_message_id: str
    messages: List[HttpMessage]
    execution_time_ms: int


class CheckpointResponse(BaseModel):
    """Response format for checkpoint endpoints."""

    success: bool
    data: Optional[CheckpointData] = None
    error: Optional[str] = None


class CheckpointPointData(BaseModel):
    """Data for a checkpoint point."""

    checkpoint_id: str
    checkpoint_point: str
    timestamp: int
    original_message_count: int
    checkpointed_at_index: int
    checkpointed_messages: int


class CheckpointHistoryData(BaseModel):
    """Data for checkpoint history response."""

    conversation_id: str
    checkpoint_points: List[CheckpointPointData]


class CheckpointHistoryResponse(BaseModel):
    """Response format for checkpoint history endpoint."""

    success: bool
    data: Optional[CheckpointHistoryData] = None
    error: Optional[str] = None


# Validation schemas
def validate_chat_request(data: Dict[str, Any]) -> ChatRequest:
    """Validate and parse a chat request."""
    return ChatRequest.model_validate(data)


def validate_regeneration_request(data: Dict[str, Any]) -> RegenerationHttpRequest:
    """Validate and parse a regeneration request."""
    return RegenerationHttpRequest.model_validate(data)
