"""
Pure functional JSON-RPC protocol handlers for A2A
All handlers are pure functions with no side effects
"""

from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, Optional, Union

from .types import A2AAgent, A2AErrorCodes


def validate_jsonrpc_request(request: Dict[str, Any]) -> bool:
    """Pure function to validate JSON-RPC request"""
    return (
        isinstance(request, dict) and
        request.get("jsonrpc") == "2.0" and
        "id" in request and
        isinstance(request.get("method"), str)
    )


def create_jsonrpc_success_response_dict(id: Union[str, int, None], result: Any) -> Dict[str, Any]:
    """Pure function to create JSON-RPC success response as dict"""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }


def create_jsonrpc_error_response_dict(id: Union[str, int, None], error: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to create JSON-RPC error response as dict"""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": error
    }


def map_error_to_a2a_error(error: Exception) -> Dict[str, Any]:
    """Pure function to map Python exceptions to A2A errors"""
    if isinstance(error, Exception):
        return {
            "code": A2AErrorCodes.INTERNAL_ERROR.value,
            "message": str(error),
            "data": {"type": type(error).__name__}
        }

    return {
        "code": A2AErrorCodes.INTERNAL_ERROR.value,
        "message": "Unknown error occurred"
    }


def validate_send_message_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to validate send message request"""
    try:
        # Basic validation
        if not validate_jsonrpc_request(request):
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_REQUEST.value,
                    "message": "Invalid JSON-RPC request"
                }
            }

        if request.get("method") not in ["message/send", "message/stream"]:
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.METHOD_NOT_FOUND.value,
                    "message": f"Method {request.get('method')} not supported"
                }
            }

        params = request.get("params", {})
        if not isinstance(params, dict):
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Invalid params format - must be an object",
                    "data": {"expected": "object", "received": type(params).__name__}
                }
            }

        message = params.get("message")
        if not isinstance(message, dict):
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Invalid message format - message must be an object",
                    "data": {"expected": "object", "received": type(message).__name__ if message is not None else "null"}
                }
            }

        # Validate required message fields
        required_fields = ["role", "parts", "messageId", "contextId", "kind"]
        missing_fields = [field for field in required_fields if field not in message]
        if missing_fields:
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": f"Missing required message fields: {', '.join(missing_fields)}",
                    "data": {"missing_fields": missing_fields}
                }
            }

        # Validate message structure
        if message.get("kind") != "message":
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Message kind must be 'message'",
                    "data": {"expected": "message", "received": message.get("kind")}
                }
            }

        if message.get("role") not in ["user", "agent"]:
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Message role must be 'user' or 'agent'",
                    "data": {"expected": ["user", "agent"], "received": message.get("role")}
                }
            }

        if not isinstance(message.get("parts"), list):
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Message parts must be a list",
                    "data": {"expected": "array", "received": type(message.get("parts")).__name__}
                }
            }

        if len(message.get("parts", [])) == 0:
            return {
                "is_valid": False,
                "error": {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Message parts cannot be empty",
                    "data": {"minimum_parts": 1}
                }
            }

        # Validate parts structure
        for i, part in enumerate(message.get("parts", [])):
            if not isinstance(part, dict):
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"Message part {i} must be an object",
                        "data": {"part_index": i, "expected": "object", "received": type(part).__name__}
                    }
                }

            if "kind" not in part:
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"Message part {i} missing 'kind' field",
                        "data": {"part_index": i, "missing_field": "kind"}
                    }
                }

            kind = part.get("kind")
            if kind == "text" and "text" not in part:
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"Text part {i} missing 'text' field",
                        "data": {"part_index": i, "part_kind": "text", "missing_field": "text"}
                    }
                }
            elif kind == "data" and "data" not in part:
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"Data part {i} missing 'data' field",
                        "data": {"part_index": i, "part_kind": "data", "missing_field": "data"}
                    }
                }
            elif kind == "file" and "file" not in part:
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"File part {i} missing 'file' field",
                        "data": {"part_index": i, "part_kind": "file", "missing_field": "file"}
                    }
                }
            elif kind not in ["text", "data", "file"]:
                return {
                    "is_valid": False,
                    "error": {
                        "code": A2AErrorCodes.INVALID_PARAMS.value,
                        "message": f"Unknown part kind '{kind}' in part {i}",
                        "data": {"part_index": i, "supported_kinds": ["text", "data", "file"], "received": kind}
                    }
                }

        return {
            "is_valid": True,
            "data": request
        }

    except Exception as e:
        return {
            "is_valid": False,
            "error": {
                "code": A2AErrorCodes.INTERNAL_ERROR.value,
                "message": f"Request validation failed: {e!s}",
                "data": {"error_type": type(e).__name__}
            }
        }


async def handle_message_send(
    request: Dict[str, Any],
    agent: 'A2AAgent',  # Forward reference to avoid circular imports
    model_provider: Any,
    executor_func: Callable
) -> Dict[str, Any]:
    """Pure function to handle message/send method"""
    try:
        params = request.get("params", {})
        message = params.get("message", {})

        context = {
            "message": message,
            "session_id": message.get("contextId", f"session_{id(request)}"),
            "metadata": params.get("metadata")
        }

        result = await executor_func(context, agent, model_provider)

        if result.get("error"):
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.INTERNAL_ERROR.value,
                    "message": result["error"]
                }
            )

        return create_jsonrpc_success_response_dict(
            request.get("id"),
            result.get("final_task", {"message": "No result available"})
        )

    except Exception as error:
        return create_jsonrpc_error_response_dict(
            request.get("id"),
            map_error_to_a2a_error(error)
        )


async def handle_message_stream(
    request: Dict[str, Any],
    agent: 'A2AAgent',  # Forward reference
    model_provider: Any,
    executor_func: Callable
) -> AsyncGenerator[Dict[str, Any], None]:
    """Pure function to handle message/stream method"""
    try:
        params = request.get("params", {})
        message = params.get("message", {})

        context = {
            "message": message,
            "session_id": message.get("contextId", f"session_{id(request)}"),
            "metadata": params.get("metadata")
        }

        async for event in executor_func(context, agent, model_provider):
            yield create_jsonrpc_success_response_dict(request.get("id"), event)

    except Exception as error:
        yield create_jsonrpc_error_response_dict(
            request.get("id"),
            map_error_to_a2a_error(error)
        )


async def handle_tasks_get(
    request: Dict[str, Any],
    task_storage: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Pure function to handle tasks/get method"""
    try:
        params = request.get("params", {})
        task_id = params.get("id")

        if not task_id:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Task ID is required"
                }
            )

        task = task_storage.get(task_id)

        if not task:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.TASK_NOT_FOUND.value,
                    "message": f"Task with id {task_id} not found"
                }
            )

        # Apply history length limit if specified
        result_task = task.copy()
        history_length = params.get("historyLength")
        if history_length and "history" in task and task["history"]:
            result_task["history"] = task["history"][-history_length:]

        return create_jsonrpc_success_response_dict(request.get("id"), result_task)

    except Exception as error:
        return create_jsonrpc_error_response_dict(
            request.get("id"),
            map_error_to_a2a_error(error)
        )


async def handle_tasks_cancel(
    request: Dict[str, Any],
    task_storage: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Pure function to handle tasks/cancel method"""
    try:
        params = request.get("params", {})
        task_id = params.get("id")

        if not task_id:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": "Task ID is required"
                }
            )

        task = task_storage.get(task_id)

        if not task:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.TASK_NOT_FOUND.value,
                    "message": f"Task with id {task_id} not found"
                }
            )

        # Check if task can be canceled
        current_state = task.get("status", {}).get("state")
        if current_state in ["completed", "failed", "canceled"]:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.TASK_NOT_CANCELABLE.value,
                    "message": f"Task {task_id} cannot be canceled in state {current_state}"
                }
            )

        # Create canceled task
        canceled_task = task.copy()
        canceled_task["status"] = {
            "state": "canceled",
            "timestamp": None  # Would be set by the system
        }

        return create_jsonrpc_success_response_dict(request.get("id"), canceled_task)

    except Exception as error:
        return create_jsonrpc_error_response_dict(
            request.get("id"),
            map_error_to_a2a_error(error)
        )


async def handle_get_authenticated_extended_card(
    request: Dict[str, Any],
    agent_card: Dict[str, Any]
) -> Dict[str, Any]:
    """Pure function to handle agent/getAuthenticatedExtendedCard method"""
    try:
        # In a real implementation, this would check authentication
        # For now, return the standard agent card
        return create_jsonrpc_success_response_dict(request.get("id"), agent_card)

    except Exception as error:
        return create_jsonrpc_error_response_dict(
            request.get("id"),
            map_error_to_a2a_error(error)
        )


def route_a2a_request(
    request: Dict[str, Any],
    agent: 'A2AAgent',  # Forward reference
    model_provider: Any,
    task_storage: Dict[str, Dict[str, Any]],
    agent_card: Dict[str, Any],
    executor_func: Callable,
    streaming_executor_func: Callable
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Pure function to route A2A requests"""

    async def _route_async():
        if not validate_jsonrpc_request(request):
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.INVALID_REQUEST.value,
                    "message": "Invalid JSON-RPC request"
                }
            )

        method = request.get("method")

        if method == "message/send":
            validation = validate_send_message_request(request)
            if not validation["is_valid"]:
                return create_jsonrpc_error_response_dict(
                    request.get("id"),
                    validation["error"]
                )
            return await handle_message_send(request, agent, model_provider, executor_func)

        elif method == "message/stream":
            validation = validate_send_message_request(request)
            if not validation["is_valid"]:
                async def error_generator():
                    yield create_jsonrpc_error_response_dict(request.get("id"), validation["error"])
                return error_generator()
            # Return the async generator directly
            return handle_message_stream(request, agent, model_provider, streaming_executor_func)

        elif method == "tasks/get":
            return await handle_tasks_get(request, task_storage)

        elif method == "tasks/cancel":
            return await handle_tasks_cancel(request, task_storage)

        elif method == "agent/getAuthenticatedExtendedCard":
            return await handle_get_authenticated_extended_card(request, agent_card)

        else:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.METHOD_NOT_FOUND.value,
                    "message": f"Method {method} not found"
                }
            )

    # Handle streaming vs non-streaming
    method = request.get("method")
    if method == "message/stream":
        # For streaming, return the async generator directly
        return _route_async()
    else:
        return _route_async()


def create_protocol_handler_config(
    agents: Dict[str, 'A2AAgent'],  # Forward reference
    model_provider: Any,
    agent_card: Dict[str, Any],
    executor_func: Callable,
    streaming_executor_func: Callable
) -> Dict[str, Any]:
    """Pure function to create protocol handler configuration"""

    task_storage: Dict[str, Dict[str, Any]] = {}  # In real implementation, this would be persistent

    async def handle_request(request: Dict[str, Any], agent_name: Optional[str] = None):
        """Pure function to handle any A2A request"""
        if agent_name:
            agent = agents.get(agent_name)
        else:
            # Use first available agent
            agent = next(iter(agents.values()), None)

        if not agent:
            return create_jsonrpc_error_response_dict(
                request.get("id"),
                {
                    "code": A2AErrorCodes.INVALID_PARAMS.value,
                    "message": f"Agent {agent_name or 'default'} not found"
                }
            )

        return route_a2a_request(
            request,
            agent,
            model_provider,
            task_storage,
            agent_card,
            executor_func,
            streaming_executor_func
        )

    return {
        "agents": agents,
        "model_provider": model_provider,
        "agent_card": agent_card,
        "task_storage": task_storage,
        "handle_request": handle_request
    }
