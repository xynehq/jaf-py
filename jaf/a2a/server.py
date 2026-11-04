"""
Pure functional A2A server integration with JAF
Extends JAF server with A2A protocol support using FastAPI
"""

import json
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import execute_a2a_agent, execute_a2a_agent_with_streaming
from .agent_card import generate_agent_card
from .protocol import route_a2a_request, validate_jsonrpc_request
from .types import A2AAgent

logger = logging.getLogger(__name__)


class _BoundedTaskStore(OrderedDict):
    """Task store that drops the oldest tasks beyond ``max_tasks`` so it cannot grow forever."""

    def __init__(self, max_tasks: int = 1000):
        super().__init__()
        self.max_tasks = max_tasks

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_tasks:
            self.popitem(last=False)


def create_a2a_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to create A2A server configuration"""
    host = config.get("host", "localhost")
    capabilities = config.get(
        "capabilities",
        {"streaming": True, "pushNotifications": False, "stateTransitionHistory": True},
    )

    agent_card = generate_agent_card(
        config["agentCard"],
        config["agents"],
        # Behind TLS termination set "publicUrl" (e.g. https://agents.example.com) so the card is not http://
        (config.get("publicUrl") or f"http://{host}:{config['port']}").rstrip("/"),
    )

    # Override the capabilities in the generated agent card
    updated_agent_card = {**agent_card, "capabilities": capabilities}

    # When the server authenticates callers, stop advertising anonymous access in the card
    if config.get("authenticate") and updated_agent_card.get("security"):
        updated_agent_card["security"] = [req for req in updated_agent_card["security"] if req]

    # One store for the whole server (not per request), so tasks can be fetched and cancelled later
    task_storage = config.get("taskStorage")
    if task_storage is None:
        task_storage = _BoundedTaskStore(int(config.get("maxStoredTasks", 1000)))

    return {
        **config,
        "host": host,
        "capabilities": capabilities,
        "agentCard": updated_agent_card,
        "taskStorage": task_storage,
    }


def create_fastapi_app(cors_origins: Optional[List[str]] = None) -> FastAPI:
    """Pure function to create FastAPI app instance"""
    app = FastAPI(
        title="JAF A2A Server",
        description="Agent-to-Agent protocol server for JAF",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Credentials are only allowed with an explicit origin list, never with a wildcard
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def setup_a2a_routes(app: FastAPI, config: Dict[str, Any]) -> None:
    """Pure function to setup A2A routes"""

    # async callable(request) -> truthy caller identity; falsy rejects the request.
    # Fails closed: without it every agent endpoint returns 401 unless allowUnauthenticated is set.
    authenticate = config.get("authenticate")
    allow_unauthenticated = bool(config.get("allowUnauthenticated"))
    if authenticate is None and allow_unauthenticated:
        logger.warning(
            "[JAF:A2A] allowUnauthenticated is set: agent endpoints accept anonymous requests"
        )

    async def _authenticate(request: Request) -> Tuple[Optional[str], Optional[JSONResponse]]:
        """Returns (caller_id, None) when allowed, or (None, 401 response) when rejected."""
        if authenticate is None and allow_unauthenticated:
            return None, None
        identity = await authenticate(request) if authenticate is not None else None
        if not identity:
            return None, JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": "Authentication required"},
                },
            )
        # Tasks are bound to this identity, so authenticate() should return a stable user id
        return str(identity), None

    # Agent Card endpoint (A2A discovery)
    @app.get("/.well-known/agent-card")
    async def get_agent_card():
        return config["agentCard"]

    # Main A2A JSON-RPC endpoint
    @app.post("/a2a")
    async def handle_a2a_request(request: Request):
        request_id = None

        caller_id, denied = await _authenticate(request)
        if denied:
            return denied

        try:
            # Parse JSON with proper error handling
            try:
                body = await request.json()
                request_id = body.get("id") if isinstance(body, dict) else None
            except json.JSONDecodeError as e:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,  # Parse Error
                        "message": "Parse error",
                        "data": {"details": str(e)},
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,  # Parse Error
                        "message": "Parse error",
                        "data": {"details": str(e)},
                    },
                }

            # Handle batch requests
            if isinstance(body, list):
                if len(body) == 0:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,  # Invalid Request
                            "message": "Invalid Request",
                            "data": {"details": "Batch request cannot be empty"},
                        },
                    }

                max_batch_size = int(config.get("maxBatchSize", 50))
                if len(body) > max_batch_size:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,  # Invalid Request
                            "message": "Invalid Request",
                            "data": {"details": f"Batch exceeds {max_batch_size} requests"},
                        },
                    }

                # Process batch requests functionally
                async def process_batch_item(single_request):
                    if not isinstance(single_request, dict):
                        return {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32600,  # Invalid Request
                                "message": "Invalid Request",
                                "data": {"details": "Each batch item must be an object"},
                            },
                        }

                    single_result = await handle_a2a_request_internal(
                        config, single_request, caller_id
                    )
                    if not hasattr(single_result, "__aiter__"):
                        return single_result
                    else:
                        # Streaming not supported in batch
                        return {
                            "jsonrpc": "2.0",
                            "id": single_request.get("id"),
                            "error": {
                                "code": -32005,  # Content Type Not Supported
                                "message": "Streaming not supported in batch requests",
                            },
                        }

                # Process all batch items
                results = [await process_batch_item(item) for item in body]

                return results

            # Validate request structure
            if not isinstance(body, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,  # Invalid Request
                        "message": "Invalid Request",
                        "data": {"details": "Request must be an object"},
                    },
                }

            result = await handle_a2a_request_internal(config, body, caller_id)

            # Check if this is a streaming request
            method = body.get("method")
            if method == "message/stream":
                # For streaming, result should be an async generator
                if hasattr(result, "__aiter__"):

                    async def generate_sse():
                        async for chunk in result:
                            yield f"data: {json.dumps(chunk)}\n\n"

                    return StreamingResponse(
                        generate_sse(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                    )
                else:
                    # If it's not iterable, wrap single response
                    async def generate_sse():
                        yield f"data: {json.dumps(result)}\n\n"

                    return StreamingResponse(
                        generate_sse(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                    )

            return result

        except Exception as error:
            # Log the error for debugging but don't expose internal details
            import logging

            logging.error(f"Internal server error: {error!s}")

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,  # Internal Error
                    "message": "Internal error",
                    "data": {"type": "server_error"},
                },
            }

    # Agent-specific endpoints
    for agent_name, agent in config["agents"].items():
        # Agent-specific JSON-RPC endpoint
        @app.post(f"/a2a/agents/{agent_name}")
        async def handle_agent_request(request: Request, agent_name: str = agent_name):
            request_id = None

            caller_id, denied = await _authenticate(request)
            if denied:
                return denied

            try:
                # Parse JSON with proper error handling
                try:
                    body = await request.json()
                    request_id = body.get("id") if isinstance(body, dict) else None
                except json.JSONDecodeError as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,  # Parse Error
                            "message": "Parse error",
                            "data": {"details": str(e)},
                        },
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,  # Parse Error
                            "message": "Parse error",
                            "data": {"details": str(e)},
                        },
                    }

                # Validate request structure
                if not isinstance(body, dict):
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,  # Invalid Request
                            "message": "Invalid Request",
                            "data": {"details": "Request must be an object"},
                        },
                    }

                result = await handle_a2a_request_for_agent(config, body, agent_name, caller_id)

                # Check if this is a streaming request
                method = body.get("method")
                if method == "message/stream":
                    # For streaming, result should be an async generator
                    if hasattr(result, "__aiter__"):

                        async def generate_sse():
                            async for chunk in result:
                                yield f"data: {json.dumps(chunk)}\n\n"

                        return StreamingResponse(
                            generate_sse(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                        )
                    else:
                        # If it's not iterable, wrap single response
                        async def generate_sse():
                            yield f"data: {json.dumps(result)}\n\n"

                        return StreamingResponse(
                            generate_sse(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                        )

                return result

            except Exception as error:
                # Log the error for debugging but don't expose internal details
                import logging

                logging.error(f"Internal server error: {error!s}")

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,  # Internal Error
                        "message": "Internal error",
                        "data": {"type": "server_error"},
                    },
                }

        # Agent-specific card endpoint - fix closure issue
        def create_agent_card_endpoint(current_agent_name: str, current_agent):
            @app.get(f"/a2a/agents/{current_agent_name}/card")
            async def get_agent_card_specific():
                agent_card = generate_agent_card(
                    {
                        "name": current_agent.name,
                        "description": current_agent.description,
                        "version": "1.0.0",
                        "provider": config["agentCard"].get(
                            "provider", {"organization": "Unknown", "url": ""}
                        ),
                    },
                    {current_agent_name: current_agent},
                    (
                        config.get("publicUrl")
                        or f"http://{config.get('host', 'localhost')}:{config['port']}"
                    ).rstrip("/"),
                )

                return agent_card

            return get_agent_card_specific

        # Create the endpoint with proper closure
        create_agent_card_endpoint(agent_name, agent)

    # Health check for A2A
    @app.get("/a2a/health")
    async def get_a2a_health():
        return {
            "status": "healthy",
            "protocol": "A2A",
            "version": "0.3.0",
            "agents": list(config["agents"].keys()),
            "timestamp": None,  # Would be set by the system
        }

    # A2A capabilities endpoint
    @app.get("/a2a/capabilities")
    async def get_a2a_capabilities():
        return {
            "supportedMethods": [
                "message/send",
                "message/stream",
                "tasks/get",
                "tasks/cancel",
                "agent/getAuthenticatedExtendedCard",
            ],
            "supportedTransports": ["JSONRPC"],
            "capabilities": config["agentCard"]["capabilities"],
            "inputModes": config["agentCard"]["defaultInputModes"],
            "outputModes": config["agentCard"]["defaultOutputModes"],
        }


async def handle_a2a_request_internal(
    config: Dict[str, Any], request: Dict[str, Any], caller_id: Optional[str] = None
) -> Any:
    """Pure function to handle A2A requests"""
    # Validate JSON-RPC request structure first
    if not validate_jsonrpc_request(request):
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32600,  # Invalid Request
                "message": "Invalid Request",
                "data": {"details": "Missing required JSON-RPC fields"},
            },
        }

    # Check if method is supported
    method = request.get("method")
    supported_methods = [
        "message/send",
        "message/stream",
        "tasks/get",
        "tasks/cancel",
        "agent/getAuthenticatedExtendedCard",
    ]

    if method not in supported_methods:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32601,  # Method Not Found
                "message": f"Method '{method}' not found",
                "data": {"supported_methods": supported_methods},
            },
        }

    # Use the first available agent by default
    agents = config["agents"]
    if not agents:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32001, "message": "No agents available"},
        }

    first_agent = next(iter(agents.values()))
    return await route_a2a_request_wrapper(config, request, first_agent, caller_id)


async def handle_a2a_request_for_agent(
    config: Dict[str, Any],
    request: Dict[str, Any],
    agent_name: str,
    caller_id: Optional[str] = None,
) -> Any:
    """Pure function to handle agent-specific A2A requests"""
    # Validate JSON-RPC request structure first
    if not validate_jsonrpc_request(request):
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32600,  # Invalid Request
                "message": "Invalid Request",
                "data": {"details": "Missing required JSON-RPC fields"},
            },
        }

    # Check if method is supported
    method = request.get("method")
    supported_methods = [
        "message/send",
        "message/stream",
        "tasks/get",
        "tasks/cancel",
        "agent/getAuthenticatedExtendedCard",
    ]

    if method not in supported_methods:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32601,  # Method Not Found
                "message": f"Method '{method}' not found",
                "data": {"supported_methods": supported_methods},
            },
        }

    agent = config["agents"].get(agent_name)
    if not agent:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32001, "message": f"Agent {agent_name} not found"},
        }

    return await route_a2a_request_wrapper(config, request, agent, caller_id)


async def route_a2a_request_wrapper(
    config: Dict[str, Any],
    request: Dict[str, Any],
    agent: A2AAgent,
    caller_id: Optional[str] = None,
) -> Any:
    """Wrapper for route_a2a_request to provide required dependencies"""
    try:
        # Get model provider from config
        model_provider = config.get("model_provider")
        task_storage = config.get("taskStorage")
        if task_storage is None:
            task_storage = config["taskStorage"] = _BoundedTaskStore()
        agent_card = config["agentCard"]

        result = route_a2a_request(
            request,
            agent,
            model_provider,
            task_storage,
            agent_card,
            execute_a2a_agent,
            execute_a2a_agent_with_streaming,
            caller_id,
        )

        # Check if result is a coroutine/awaitable
        if hasattr(result, "__aiter__"):
            # It's an async generator for streaming
            return result
        elif hasattr(result, "__await__"):
            # It's a coroutine
            return await result
        else:
            # It's a regular value
            return result

    except Exception as e:
        # Return proper JSON-RPC error
        import logging

        logging.error(f"Route wrapper error: {e!s}")

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32603,  # Internal Error
                "message": "Internal error",
                "data": {"type": "route_error"},
            },
        }


def create_a2a_server(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to create A2A server instance"""
    server_config = create_a2a_server_config(config)
    app = create_fastapi_app(server_config.get("corsOrigins"))
    setup_a2a_routes(app, server_config)

    # Create the server instance here to be shared by start/stop
    uvicorn_config = uvicorn.Config(
        app,
        host=server_config.get("host", "localhost"),
        port=server_config["port"],
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)

    async def start_server():
        """Start the A2A server"""
        host = server_config.get("host", "localhost")
        port = server_config["port"]

        logger.debug(f"🔧 Starting A2A-enabled JAF server on {host}:{port}...")
        logger.debug(f"🚀 A2A Server running on http://{host}:{port}")
        logger.debug(f"🤖 Available agents: {', '.join(server_config['agents'].keys())}")
        logger.debug(f"📋 Agent Card: http://{host}:{port}/.well-known/agent-card")
        logger.debug(f"🔗 A2A Endpoint: http://{host}:{port}/a2a")
        logger.debug(f"🏥 A2A Health: http://{host}:{port}/a2a/health")
        logger.debug(f"⚡ A2A Capabilities: http://{host}:{port}/a2a/capabilities")

        for agent_name in server_config["agents"].keys():
            logger.debug(f"🎯 Agent {agent_name}: http://{host}:{port}/a2a/agents/{agent_name}")

        await server.serve()

    async def stop_server():
        """Stop the A2A server"""
        if server:
            server.should_exit = True
            logger.debug("🛑 A2A Server stopped")

    def add_agent(name: str, agent: A2AAgent) -> Dict[str, Any]:
        """Pure function to add agent to server"""
        new_agents = {**server_config["agents"], name: agent}

        return {
            **server_config,
            "agents": new_agents,
            "agentCard": generate_agent_card(
                {
                    **server_config["agentCard"],
                    "provider": server_config["agentCard"].get(
                        "provider", {"organization": "Unknown", "url": ""}
                    ),
                },
                new_agents,
                server_config["agentCard"]["url"].replace("/a2a", ""),
            ),
        }

    def remove_agent(name: str) -> Dict[str, Any]:
        """Pure function to remove agent from server"""
        new_agents = {k: v for k, v in server_config["agents"].items() if k != name}

        return {
            **server_config,
            "agents": new_agents,
            "agentCard": generate_agent_card(
                {
                    **server_config["agentCard"],
                    "provider": server_config["agentCard"].get(
                        "provider", {"organization": "Unknown", "url": ""}
                    ),
                },
                new_agents,
                server_config["agentCard"]["url"].replace("/a2a", ""),
            ),
        }

    return {
        "app": app,
        "config": server_config,
        "start": start_server,
        "stop": stop_server,
        "add_agent": add_agent,
        "remove_agent": remove_agent,
        "get_agent_card": lambda: server_config["agentCard"],
    }


async def start_a2a_server(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function for one-line server creation and startup"""
    server = create_a2a_server(config)
    await server["start"]()
    return server


# Utility functions for configuration


def create_server_config(
    agents: Dict[str, A2AAgent],
    name: str,
    description: str,
    port: int,
    host: str = "localhost",
    version: str = "1.0.0",
    provider: Optional[Dict[str, str]] = None,
    authenticate: Optional[Callable[[Request], Awaitable[Any]]] = None,
    allow_unauthenticated: bool = False,
    cors_origins: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Utility function to create server configuration.

    Agent endpoints require ``authenticate`` (async callable taking the request and returning a
    truthy caller identity). Pass ``allow_unauthenticated=True`` only for local development.
    """
    return {
        "authenticate": authenticate,
        "allowUnauthenticated": allow_unauthenticated,
        "corsOrigins": cors_origins,
        "agents": agents,
        "agentCard": {
            "name": name,
            "description": description,
            "version": version,
            "provider": provider
            or {"organization": "JAF Framework", "url": "https://functional-agent-framework.com"},
        },
        "port": port,
        "host": host,
    }
