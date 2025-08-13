"""
Pure functional A2A types for JAF Python implementation
Maintains immutability and type safety through Pydantic
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class A2AErrorCodes(Enum):
    """A2A protocol error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
    UNSUPPORTED_OPERATION = -32004
    CONTENT_TYPE_NOT_SUPPORTED = -32005
    INVALID_AGENT_RESPONSE = -32006


class A2AFile(BaseModel):
    """A2A file representation"""
    model_config = {"frozen": True}

    bytes: Optional[str] = None
    uri: Optional[str] = None
    name: Optional[str] = None
    mime_type: Optional[str] = Field(None, alias="mimeType")


class A2ATextPart(BaseModel):
    """A2A text part"""
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["text"]
    text: str
    metadata: Optional[Dict[str, Any]] = Field(None, exclude=True)


class A2ADataPart(BaseModel):
    """A2A data part"""
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["data"]
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(None, exclude=True)


class A2AFilePart(BaseModel):
    """A2A file part"""
    model_config = {"frozen": True}

    kind: Literal["file"]
    file: A2AFile
    metadata: Optional[Dict[str, Any]] = None


A2APart = Union[A2ATextPart, A2ADataPart, A2AFilePart]


class A2AMessage(BaseModel):
    """Core A2A message type"""
    model_config = {"frozen": True}

    role: Literal["user", "agent"]
    parts: List[A2APart]
    message_id: str = Field(alias="messageId")
    context_id: Optional[str] = Field(None, alias="contextId")
    task_id: Optional[str] = Field(None, alias="taskId")
    kind: Literal["message"]
    metadata: Optional[Dict[str, Any]] = None
    extensions: Optional[List[str]] = None
    reference_task_ids: Optional[List[str]] = Field(None, alias="referenceTaskIds")


class TaskState(str, Enum):
    """Task state enumeration"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


class A2ATaskStatus(BaseModel):
    """Task status information"""
    model_config = {"frozen": True}

    state: TaskState
    message: Optional[A2AMessage] = None
    timestamp: Optional[str] = None


class A2AArtifact(BaseModel):
    """A2A artifact type"""
    model_config = {"frozen": True}

    artifact_id: str = Field(alias="artifactId")
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None
    extensions: Optional[List[str]] = None


class A2ATask(BaseModel):
    """A2A task representation"""
    model_config = {"frozen": True}

    id: str
    context_id: str = Field(alias="contextId")
    status: A2ATaskStatus
    history: Optional[List[A2AMessage]] = None
    artifacts: Optional[List[A2AArtifact]] = None
    metadata: Optional[Dict[str, Any]] = None
    kind: Literal["task"]


class AgentSkill(BaseModel):
    """Agent skill definition"""
    model_config = {"frozen": True}

    id: str
    name: str
    description: str
    tags: List[str]
    examples: Optional[List[str]] = None
    input_modes: Optional[List[str]] = Field(None, alias="inputModes")
    output_modes: Optional[List[str]] = Field(None, alias="outputModes")


class AgentCapabilities(BaseModel):
    """Agent capabilities definition"""
    model_config = {"frozen": True}

    streaming: Optional[bool] = None
    push_notifications: Optional[bool] = Field(None, alias="pushNotifications")
    state_transition_history: Optional[bool] = Field(None, alias="stateTransitionHistory")


class AgentProvider(BaseModel):
    """Agent provider information"""
    model_config = {"frozen": True}

    organization: str
    url: str


class AgentCard(BaseModel):
    """A2A agent card for discovery"""
    model_config = {"frozen": True}

    protocol_version: str = Field(alias="protocolVersion")
    name: str
    description: str
    url: str
    preferred_transport: Optional[str] = Field(None, alias="preferredTransport")
    version: str
    provider: Optional[AgentProvider] = None
    capabilities: AgentCapabilities
    default_input_modes: List[str] = Field(alias="defaultInputModes")
    default_output_modes: List[str] = Field(alias="defaultOutputModes")
    skills: List[AgentSkill]
    security_schemes: Optional[Dict[str, Any]] = Field(None, alias="securitySchemes")
    security: Optional[List[Dict[str, List[str]]]] = None


class JSONRPCError(BaseModel):
    """JSON-RPC error format"""
    model_config = {"frozen": True}

    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC request format"""
    model_config = {"frozen": True}

    jsonrpc: Literal["2.0"]
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCSuccessResponse(BaseModel):
    """JSON-RPC success response"""
    model_config = {"frozen": True}

    jsonrpc: Literal["2.0"]
    id: Union[str, int, None]
    result: Any


class JSONRPCErrorResponse(BaseModel):
    """JSON-RPC error response"""
    model_config = {"frozen": True}

    jsonrpc: Literal["2.0"]
    id: Union[str, int, None]
    error: JSONRPCError


JSONRPCResponse = Union[JSONRPCSuccessResponse, JSONRPCErrorResponse]


class MessageSendConfiguration(BaseModel):
    """Configuration for message sending"""
    model_config = {"frozen": True}

    model: Optional[str] = None
    temperature: Optional[float] = None
    accepted_output_modes: Optional[List[str]] = Field(None, alias="acceptedOutputModes")
    history_length: Optional[int] = Field(None, alias="historyLength")
    blocking: Optional[bool] = None


class SendMessageParams(BaseModel):
    """Parameters for message/send method"""
    model_config = {"frozen": True}

    message: A2AMessage
    configuration: Optional[MessageSendConfiguration] = None
    metadata: Optional[Dict[str, Any]] = None


class SendMessageRequest(JSONRPCRequest):
    """A2A message/send request"""
    model_config = {"frozen": True}

    method: Literal["message/send"]
    params: SendMessageParams


class SendStreamingMessageRequest(JSONRPCRequest):
    """A2A message/stream request"""
    model_config = {"frozen": True}

    method: Literal["message/stream"]
    params: SendMessageParams


class GetTaskParams(BaseModel):
    """Parameters for tasks/get method"""
    model_config = {"frozen": True}

    id: str
    history_length: Optional[int] = Field(None, alias="historyLength")
    metadata: Optional[Dict[str, Any]] = None


class GetTaskRequest(JSONRPCRequest):
    """A2A tasks/get request"""
    model_config = {"frozen": True}

    method: Literal["tasks/get"]
    params: GetTaskParams


class A2AError(BaseModel):
    """A2A error format"""
    model_config = {"frozen": True}

    code: int
    message: str
    data: Optional[Any] = None


class StreamEvent(BaseModel):
    """Base stream event"""
    model_config = {"frozen": True}

    isTaskComplete: bool
    content: Optional[Any] = None
    updates: Optional[str] = None
    new_state: Optional[Dict[str, Any]] = Field(None, alias="newState")
    timestamp: str


class A2AStatusUpdateEvent(BaseModel):
    """A2A status update stream event"""
    model_config = {"frozen": True}

    kind: Literal["status-update"]
    task_id: str = Field(alias="taskId")
    context_id: str = Field(alias="contextId")
    status: A2ATaskStatus
    final: bool


class A2AArtifactUpdateEvent(BaseModel):
    """A2A artifact update stream event"""
    model_config = {"frozen": True}

    kind: Literal["artifact-update"]
    task_id: str = Field(alias="taskId")
    context_id: str = Field(alias="contextId")
    artifact: A2AArtifact
    append: Optional[bool] = None
    last_chunk: Optional[bool] = Field(None, alias="lastChunk")


class A2AMessageEvent(BaseModel):
    """A2A message stream event"""
    model_config = {"frozen": True}

    kind: Literal["message"]
    message: A2AMessage


A2AStreamEvent = Union[A2AStatusUpdateEvent, A2AArtifactUpdateEvent, A2AMessageEvent]


class AgentState(BaseModel):
    """Agent state representation"""
    model_config = {"frozen": True}

    sessionId: str
    messages: List[Any]
    context: Dict[str, Any]
    artifacts: List[Any]
    timestamp: str


class ToolContext(BaseModel):
    """Tool execution context"""
    model_config = {"frozen": True}

    actions: Dict[str, bool]
    metadata: Dict[str, Any]


class A2AToolResult(BaseModel):
    """A2A tool execution result"""
    model_config = {"frozen": True}

    status: str
    result: Any
    data: Optional[Any] = None
    error: Optional[A2AError] = None
    context: Optional[ToolContext] = None


class A2AAgentTool(BaseModel):
    """A2A agent tool definition"""
    model_config = {"frozen": True}

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    execute: Any  # Function - will be set dynamically


class A2AAgent(BaseModel):
    """A2A agent definition"""
    model_config = {"frozen": True}

    name: str
    description: str
    supported_content_types: List[str] = Field(alias="supportedContentTypes")
    instruction: str
    tools: List[A2AAgentTool]


class A2AServerConfig(BaseModel):
    """A2A server configuration"""
    model_config = {"frozen": True}

    agents: Dict[str, A2AAgent]
    agent_card: Dict[str, Any] = Field(alias="agentCard")
    port: int
    host: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None


class A2AClientConfig(BaseModel):
    """A2A client configuration"""
    model_config = {"frozen": True}

    base_url: str = Field(alias="baseUrl")
    timeout: Optional[int] = None


class A2AClientState(BaseModel):
    """A2A client state"""
    model_config = {"frozen": True}

    config: A2AClientConfig
    session_id: str = Field(alias="sessionId")


# Factory functions for creating instances

def create_a2a_text_part(text: str, metadata: Optional[Dict[str, Any]] = None) -> A2ATextPart:
    """Create an A2A text part"""
    if metadata is None:
        return A2ATextPart(kind="text", text=text)
    return A2ATextPart(kind="text", text=text, metadata=metadata)


def create_a2a_data_part(data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> A2ADataPart:
    """Create an A2A data part"""
    if metadata is None:
        return A2ADataPart(kind="data", data=data)
    return A2ADataPart(kind="data", data=data, metadata=metadata)


def create_a2a_file_part(file: A2AFile, metadata: Optional[Dict[str, Any]] = None) -> A2AFilePart:
    """Create an A2A file part"""
    return A2AFilePart(kind="file", file=file, metadata=metadata)


def create_jsonrpc_success_response(id: Union[str, int, None], result: Any) -> JSONRPCSuccessResponse:
    """Create a JSON-RPC success response"""
    return JSONRPCSuccessResponse(jsonrpc="2.0", id=id, result=result)


def create_jsonrpc_error_response(id: Union[str, int, None], error: Union[JSONRPCError, A2AError]) -> JSONRPCErrorResponse:
    """Create a JSON-RPC error response"""
    if isinstance(error, A2AError):
        jsonrpc_error = JSONRPCError(code=error.code, message=error.message, data=error.data)
    else:
        jsonrpc_error = error
    return JSONRPCErrorResponse(jsonrpc="2.0", id=id, error=jsonrpc_error)


def create_a2a_error(code: A2AErrorCodes, message: str, data: Optional[Any] = None) -> A2AError:
    """Create an A2A error"""
    return A2AError(code=code.value, message=message, data=data)


def create_a2a_message(
    role: Literal["user", "agent"],
    parts: List[A2APart],
    context_id: str,
    task_id: Optional[str] = None
) -> A2AMessage:
    """Create an A2A message"""
    import time
    import uuid

    return A2AMessage(
        role=role,
        parts=parts,
        messageId=f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        contextId=context_id,
        taskId=task_id,
        kind="message"
    )


def create_a2a_task(
    initial_message: A2AMessage,
    context_id: str
) -> A2ATask:
    """Create an A2A task"""
    import time
    import uuid
    from datetime import datetime

    return A2ATask(
        id=f"task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        contextId=context_id,
        status=A2ATaskStatus(
            state="submitted",
            timestamp=datetime.now().isoformat()
        ),
        history=[initial_message],
        artifacts=[],
        kind="task"
    )


def create_a2a_artifact(
    name: str,
    parts: List[A2APart]
) -> A2AArtifact:
    """Create an A2A artifact"""
    import time
    import uuid
    from datetime import datetime

    return A2AArtifact(
        artifactId=f"artifact_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        name=name,
        parts=parts,
        timestamp=datetime.now().isoformat()
    )


def create_a2a_agent_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    execute_func: Any
) -> A2AAgentTool:
    """Create an A2A agent tool"""
    return A2AAgentTool(
        name=name,
        description=description,
        parameters=parameters,
        execute=execute_func
    )


def create_jsonrpc_request(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    request_id: Optional[Union[str, int]] = None
) -> JSONRPCRequest:
    """Create a JSON-RPC request"""
    import time
    import uuid

    if request_id is None:
        request_id = f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    return JSONRPCRequest(
        jsonrpc="2.0",
        method=method,
        params=params,
        id=request_id
    )
