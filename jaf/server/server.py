"""
FastAPI-based HTTP server for the JAF framework.

This module implements a complete HTTP server that exposes JAF agents
via REST API endpoints with proper error handling and validation.
"""

import time
import uuid
import json
from typing import Any, Dict, List, Optional, TypeVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from .types import (
    ServerConfig, ChatRequest, ChatResponse, AgentListResponse, 
    HealthResponse, HttpMessage, CompletedChatData, AgentInfo, AgentListData,
    ConversationResponse, ConversationData, MemoryHealthResponse, MemoryHealthData,
    DeleteConversationResponse, DeleteConversationData
)
from ..core.engine import run
from ..core.types import (
    RunState, Message, create_run_id, create_trace_id, 
    CompletedOutcome, ErrorOutcome
)

Ctx = TypeVar('Ctx')

class JAFServer:
    """FastAPI-based server for JAF agents."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.start_time = time.time()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="JAF Agent Framework Server",
            description="HTTP API for JAF agents",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Set up FastAPI middleware."""
        # CORS middleware
        if self.config.cors is not False:
            cors_config = {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }
            
            if isinstance(self.config.cors, dict):
                cors_config.update(self.config.cors)
            
            self.app.add_middleware(CORSMiddleware, **cors_config)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            print(f"[JAF:SERVER] {request.method} {request.url.path} - "
                  f"{response.status_code} - {process_time:.3f}s")
            
            return response
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                version="2.0.0",
                uptime=int((time.time() - self.start_time) * 1000)
            )

        @self.app.get("/metrics")
        async def metrics():
            """Basic metrics endpoint."""
            return JSONResponse(content={"status": "ok"})
        
        @self.app.get("/agents", response_model=AgentListResponse)
        async def list_agents():
            """List available agents."""
            try:
                agents = []
                for name, agent in self.config.agent_registry.items():
                    # Get a safe description
                    try:
                        # Create a dummy state to get instructions
                        dummy_state = RunState(
                            run_id=create_run_id("dummy"),
                            trace_id=create_trace_id("dummy"),
                            messages=[],
                            current_agent_name=name,
                            context={},
                            turn_count=0
                        )
                        description = agent.instructions(dummy_state)[:200]  # Truncate
                    except Exception:
                        description = f"Agent: {name}"
                    
                    agents.append(AgentInfo(
                        name=name,
                        description=description,
                        tools=[tool.schema.name for tool in (agent.tools or [])]
                    ))
                
                return AgentListResponse(
                    success=True,
                    data=AgentListData(agents=agents)
                )
                
            except Exception as e:
                return AgentListResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_completion(request: ChatRequest):
            """Main chat completion endpoint."""
            return await self._handle_chat(request)
        
        @self.app.post("/agents/{agent_name}/chat", response_model=ChatResponse)
        async def agent_chat(agent_name: str, request: Request):
            """Agent-specific chat endpoint."""
            try:
                body = await request.json()
                chat_request = ChatRequest(
                    agent_name=agent_name,
                    **body
                )
                return await self._handle_chat(chat_request)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Memory endpoints
        @self.app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
        async def get_conversation(conversation_id: str):
            """Get conversation history."""
            try:
                # Check if memory is configured
                if not self.config.run_config.memory:
                    return ConversationResponse(
                        success=False,
                        error="Memory not configured for this server"
                    )
                
                conversation = await self.config.run_config.memory.provider.get_conversation(conversation_id)
                if not conversation:
                    return ConversationResponse(
                        success=False,
                        error=f"Conversation {conversation_id} not found"
                    )
                
                # Convert messages to HTTP format
                http_messages = []
                for msg in conversation.messages:
                    if msg.role == 'tool':
                        http_messages.append({
                            'role': 'tool',
                            'content': msg.content,
                            'tool_call_id': msg.tool_call_id
                        })
                    elif msg.role == 'assistant' and msg.tool_calls:
                        http_messages.append({
                            'role': msg.role,
                            'content': msg.content or '',
                            'tool_calls': [
                                {
                                    'id': tc.id,
                                    'type': tc.type,
                                    'function': {
                                        'name': tc.function.name,
                                        'arguments': tc.function.arguments
                                    }
                                }
                                for tc in msg.tool_calls
                            ]
                        })
                    else:
                        http_messages.append({
                            'role': msg.role,
                            'content': msg.content
                        })
                
                return ConversationResponse(
                    success=True,
                    data=ConversationData(
                        conversation_id=conversation.conversation_id,
                        user_id=conversation.user_id,
                        messages=http_messages,
                        metadata=conversation.metadata
                    )
                )
                
            except Exception as e:
                return ConversationResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
        async def delete_conversation(conversation_id: str):
            """Delete conversation."""
            try:
                # Check if memory is configured
                if not self.config.run_config.memory:
                    return DeleteConversationResponse(
                        success=False,
                        error="Memory not configured for this server"
                    )
                
                deleted = await self.config.run_config.memory.provider.delete_conversation(conversation_id)
                
                return DeleteConversationResponse(
                    success=True,
                    data=DeleteConversationData(
                        conversation_id=conversation_id,
                        deleted=deleted
                    )
                )
                
            except Exception as e:
                return DeleteConversationResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/memory/health", response_model=MemoryHealthResponse)
        async def memory_health_check():
            """Check memory provider health."""
            try:
                # Check if memory is configured
                if not self.config.run_config.memory:
                    return MemoryHealthResponse(
                        success=False,
                        error="Memory not configured for this server"
                    )
                
                health_data = await self.config.run_config.memory.provider.health_check()
                
                return MemoryHealthResponse(
                    success=True,
                    data=MemoryHealthData(
                        healthy=health_data.get('healthy', False),
                        provider=health_data.get('provider', 'Unknown'),
                        latency_ms=health_data.get('latency_ms', 0),
                        details=health_data
                    )
                )
                
            except Exception as e:
                return MemoryHealthResponse(
                    success=False,
                    error=str(e)
                )
    
    async def _handle_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat completion logic."""
        start_time = time.time()
        
        try:
            # Validate agent exists
            if request.agent_name not in self.config.agent_registry:
                available_agents = list(self.config.agent_registry.keys())
                return ChatResponse(
                    success=False,
                    error=f"Agent '{request.agent_name}' not found. Available agents: {', '.join(available_agents)}"
                )
            
            # Check for streaming (not implemented)
            if request.stream:
                return ChatResponse(
                    success=False,
                    error="Streaming is not yet implemented. Set stream: false."
                )
            
            # Convert HTTP messages to JAF messages
            jaf_messages = []
            for msg in request.messages:
                jaf_msg = Message(
                    role='user' if msg.role == 'system' else msg.role,  # Convert system to user
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=None  # Will be set by engine
                )
                jaf_messages.append(jaf_msg)
            
            # Generate or use provided conversation_id
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Create initial state
            run_id = create_run_id(str(uuid.uuid4()))
            trace_id = create_trace_id(str(uuid.uuid4()))
            
            initial_state = RunState(
                run_id=run_id,
                trace_id=trace_id,
                messages=jaf_messages,
                current_agent_name=request.agent_name,
                context=request.context or {},
                turn_count=0
            )
            
            # Create run config with overrides
            run_config = self.config.run_config
            from dataclasses import replace
            
            # Apply overrides
            if request.max_turns is not None or conversation_id:
                config_updates = {}
                if request.max_turns is not None:
                    config_updates['max_turns'] = request.max_turns
                if conversation_id:
                    config_updates['conversation_id'] = conversation_id
                
                run_config = replace(run_config, **config_updates)
            
            # Run the agent
            result = await run(initial_state, run_config)
            execution_time = int((time.time() - start_time) * 1000)
            
            # Convert JAF messages back to HTTP format
            http_messages = []
            for msg in result.final_state.messages:
                if msg.role == 'tool':
                    http_messages.append({
                        'role': 'tool',
                        'content': msg.content,
                        'tool_call_id': msg.tool_call_id
                    })
                elif msg.role == 'assistant' and msg.tool_calls:
                    http_messages.append({
                        'role': msg.role,
                        'content': msg.content or '',
                        'tool_calls': [
                            {
                                'id': tc.id,
                                'type': tc.type,
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': tc.function.arguments
                                }
                            }
                            for tc in msg.tool_calls
                        ]
                    })
                else:
                    http_messages.append({
                        'role': msg.role,
                        'content': msg.content
                    })
            
            # Format outcome
            outcome_dict = {}
            if isinstance(result.outcome, CompletedOutcome):
                outcome_dict = {
                    'status': 'completed',
                    'output': str(result.outcome.output) if result.outcome.output is not None else None
                }
            elif isinstance(result.outcome, ErrorOutcome):
                error_info = result.outcome.error
                outcome_dict = {
                    'status': 'error',
                    'error': {
                        'type': error_info._tag if hasattr(error_info, '_tag') else 'UnknownError',
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
                    execution_time_ms=execution_time,
                    conversation_id=conversation_id
                )
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            print(f"[JAF:SERVER] Chat endpoint error after {execution_time}ms: {e}")
            
            return ChatResponse(
                success=False,
                error=str(e)
            )
    
    async def start(self) -> None:
        """Start the server."""
        host = self.config.host
        port = self.config.port
        
        print(f"🔧 Starting FastAPI server on {host}:{port}...")
        print(f"🚀 JAF Server running on http://{host}:{port}")
        print(f"📋 Available agents: {', '.join(self.config.agent_registry.keys())}")
        print(f"🏥 Health check: http://{host}:{port}/health")
        print(f"🤖 Agents list: http://{host}:{port}/agents")
        print(f"💬 Chat endpoint: http://{host}:{port}/chat")
        
        # Memory endpoints info
        if self.config.run_config.memory:
            print(f"🧠 Memory provider: {type(self.config.run_config.memory.provider).__name__}")
            print(f"📚 Conversation API: http://{host}:{port}/conversations/{{id}}")
            print(f"🗑️  Delete conversation: DELETE http://{host}:{port}/conversations/{{id}}")
            print(f"💾 Memory health: http://{host}:{port}/memory/health")
        else:
            print("⚠️  Memory: Not configured (conversations will not persist)")
            
        print(f"📖 API docs: http://{host}:{port}/docs")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self) -> None:
        """Stop the server (placeholder for compatibility)."""
        print("🛑 Server stop requested")
    
    @property
    def server(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app
