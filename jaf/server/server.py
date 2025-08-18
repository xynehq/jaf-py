"""
FastAPI-based HTTP server for the JAF framework.

This module implements a complete HTTP server that exposes JAF agents
via REST API endpoints with proper error handling and validation.
"""

import time
import uuid
from dataclasses import asdict, replace
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.engine import run
from ..core.types import (
    CompletedOutcome,
    ErrorOutcome,
    Message,
    RunState,
    create_run_id,
    create_trace_id,
)
from ..memory.types import MemoryConfig
from .types import (
    AgentInfo,
    AgentListData,
    AgentListResponse,
    ChatRequest,
    ChatResponse,
    CompletedChatData,
    ConversationData,
    ConversationResponse,
    DeleteConversationData,
    DeleteConversationResponse,
    HealthResponse,
    HttpMessage,
    MemoryHealthResponse,
    ServerConfig,
)

Ctx = TypeVar('Ctx')

def create_jaf_server(config: ServerConfig[Ctx]) -> FastAPI:
    """Create and configure a JAF server instance."""

    start_time = time.time()

    app = FastAPI(
        title="JAF Agent Framework Server",
        description="HTTP API for JAF agents",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Setup middleware
    if config.cors is not False:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            version="2.0.0",
            uptime=int((time.time() - start_time) * 1000)
        )

    @app.get("/agents", response_model=AgentListResponse)
    async def list_agents():
        try:
            agents = [
                AgentInfo(
                    name=name,
                    description=agent.instructions(None) if agent.instructions else "",
                    tools=[tool.schema.name for tool in agent.tools or []]
                )
                for name, agent in config.agent_registry.items()
            ]
            return AgentListResponse(success=True, data=AgentListData(agents=agents))
        except Exception as e:
            return AgentListResponse(success=False, error=str(e))

    @app.post("/chat", response_model=ChatResponse)
    async def chat_completion(request: ChatRequest):
        request_start_time = time.time()  # Track request start time, not server start time
        
        if request.agent_name not in config.agent_registry:
            raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")

        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Load conversation history to get correct turn count
        initial_turn_count = 0
        if config.default_memory_provider and conversation_id:
            try:
                conversation_result = await config.default_memory_provider.get_conversation(conversation_id)
                if hasattr(conversation_result, 'data') and conversation_result.data:
                    conversation_data = conversation_result.data
                    if conversation_data.metadata and "turn_count" in conversation_data.metadata:
                        initial_turn_count = conversation_data.metadata["turn_count"]
                        print(f"[JAF:SERVER] Loaded initial turn_count: {initial_turn_count}")
            except Exception as e:
                print(f"[JAF:SERVER] Warning: Failed to load conversation history: {e}")

        initial_state = RunState(
            run_id=create_run_id(str(uuid.uuid4())),
            trace_id=create_trace_id(str(uuid.uuid4())),
            messages=[Message(**msg.model_dump()) for msg in request.messages],
            current_agent_name=request.agent_name,
            context=request.context or {},
            turn_count=initial_turn_count  # Use loaded turn count instead of always 0
        )

        run_config_with_memory = config.run_config
        if config.default_memory_provider:
            run_config_with_memory = replace(
                run_config_with_memory,
                memory=MemoryConfig(provider=config.default_memory_provider, auto_store=True),
                conversation_id=conversation_id
            )

        if request.max_turns is not None:
            run_config_with_memory = replace(run_config_with_memory, max_turns=request.max_turns)

        result = await run(initial_state, run_config_with_memory)

        http_messages = [HttpMessage.model_validate(asdict(msg)) for msg in result.final_state.messages]

        outcome_dict = {}
        if isinstance(result.outcome, CompletedOutcome):
            outcome_dict = {
                'status': 'completed',
                'output': result.outcome.output
            }
        elif isinstance(result.outcome, ErrorOutcome):
            error_info = result.outcome.error
            outcome_dict = {
                'status': 'error',
                'error': {
                    'type': error_info.__class__.__name__,
                    'message': str(error_info)
                }
            }

        return ChatResponse(
            success=True,
            data=CompletedChatData(
                run_id=str(result.final_state.run_id),
                trace_id=str(result.final_state.trace_id),
                messages=http_messages,
                outcome=outcome_dict,
                turn_count=result.final_state.turn_count,
                execution_time_ms=int((time.time() - request_start_time) * 1000),  # Use request start time
                conversation_id=conversation_id
            )
        )

    # Memory endpoints
    if config.default_memory_provider:
        @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
        async def get_conversation(conversation_id: str):
            result = await config.default_memory_provider.get_conversation(conversation_id)

            # Handle Result type properly
            if hasattr(result, 'error'):  # Failure case
                raise HTTPException(status_code=500, detail=str(result.error.message))

            conversation = result.data
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Convert ConversationMemory to ConversationData
            conversation_data = ConversationData(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=[asdict(msg) for msg in conversation.messages],
                metadata=conversation.metadata
            )

            return ConversationResponse(success=True, data=conversation_data)

        @app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
        async def delete_conversation(conversation_id: str):
            result = await config.default_memory_provider.delete_conversation(conversation_id)

            # Handle Result type properly
            if hasattr(result, 'error'):  # Failure case
                raise HTTPException(status_code=500, detail=str(result.error.message))

            return DeleteConversationResponse(
                success=True,
                data=DeleteConversationData(conversation_id=conversation_id, deleted=result.data)
            )

        @app.get("/memory/health", response_model=MemoryHealthResponse)
        async def memory_health():
            result = await config.default_memory_provider.health_check()

            # Handle Result type properly
            if hasattr(result, 'error'):  # Failure case
                raise HTTPException(status_code=500, detail=str(result.error.message))

            return MemoryHealthResponse(success=True, data=result.data)

    return app
