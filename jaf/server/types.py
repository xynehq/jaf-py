"""
Server types for the JAF framework HTTP API.

This module defines the request/response types and configuration
for the JAF HTTP server implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from ..core.types import Agent, RunConfig
from ..memory.types import MemoryProvider

Ctx = TypeVar('Ctx')

# HTTP Message types
class HttpMessage(BaseModel):
    """HTTP message format for API requests."""
    role: Literal['user', 'assistant', 'system', 'tool']
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

# Request types
class ChatRequest(BaseModel):
    """Request format for chat endpoints."""
    messages: List[HttpMessage]
    agent_name: str = Field(..., description="Name of the agent to use")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context data for the agent")
    max_turns: Optional[int] = Field(default=10, description="Maximum number of turns")
    stream: bool = Field(default=False, description="Whether to stream the response")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for memory persistence")
    memory: Optional[Dict[str, Any]] = Field(default=None, description="Memory configuration override")

# Response types
class CompletedChatData(BaseModel):
    """Data for successful chat completion."""
    run_id: str
    trace_id: str
    messages: List[HttpMessage]
    outcome: Dict[str, Any]
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
    status: Literal['healthy', 'unhealthy']
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
    host: str = '127.0.0.1'
    port: int = 3000
    cors: Union[bool, Dict[str, Any]] = True
    default_memory_provider: Optional[MemoryProvider] = None

# Validation schemas
def validate_chat_request(data: Dict[str, Any]) -> ChatRequest:
    """Validate and parse a chat request."""
    return ChatRequest.model_validate(data)
