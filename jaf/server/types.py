"""
Server types for the JAF framework HTTP API.

This module defines the request/response types and configuration
for the JAF HTTP server implementation.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..core.types import Agent, RunConfig, TraceEvent

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

# Response types
class CompletedChatData(BaseModel):
    """Data for successful chat completion."""
    run_id: str
    trace_id: str
    messages: List[Dict[str, Any]]
    outcome: Dict[str, Any]
    turn_count: int
    execution_time_ms: int

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

# Server configuration
@dataclass
class ServerConfig:
    """Configuration for the JAF HTTP server."""
    agent_registry: Dict[str, Agent]
    run_config: RunConfig
    host: str = 'localhost'
    port: int = 3000
    cors: Union[bool, Dict[str, Any]] = True

# Validation schemas
def validate_chat_request(data: Dict[str, Any]) -> ChatRequest:
    """Validate and parse a chat request."""
    return ChatRequest.model_validate(data)