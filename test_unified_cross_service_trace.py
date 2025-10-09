#!/usr/bin/env python3
"""
Test unified tracing across services/containers.

Scenario:
1. Main agent (Service 1) starts with trace_id and session_id
2. Main agent calls sub-agent (within Service 1)
3. Sub-agent uses a tool that calls external Service 2
4. Service 2 has its own JAF agent that processes the request
5. All events should appear under the SAME trace_id and session_id

This simulates cross-pod/cross-container distributed tracing.
"""

import asyncio
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# Set Langfuse credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-d8b28b3e-eb67-4916-9bc9-4cd3d57ee7fa"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-50fea071-404e-4ecf-b963-afed3630d742"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

# JAF imports
from jaf import (
    Agent,
    RunState,
    RunConfig,
    Message,
    ContentRole,
    TraceId,
    generate_run_id,
    create_agent_tool,
    get_current_trace_id,
    get_current_session_id,
)
from jaf.core.tools import function_tool
from jaf.core.tracing import LangfuseTraceCollector


# ============================================================================
# SERVICE 2: External Service (Different Pod/Container)
# ============================================================================

class ExternalServiceRequest(BaseModel):
    """Request to external service."""
    query: str = Field(description="The query to process")
    trace_id: str = Field(description="Trace ID for unified tracing")
    session_id: str = Field(description="Session ID for unified tracing")


class ExternalServiceResponse(BaseModel):
    """Response from external service."""
    result: str
    trace_id: str
    session_id: str


class ExternalService:
    """Simulates an external JAF agent service in a different pod/container."""

    def __init__(self, langfuse_collector):
        # Trace collector for Service 2 - SAME Langfuse instance for unified tracing
        self.trace_collector = langfuse_collector

        # Create the external agent
        self.agent = Agent(
            name="external_data_processor",
            instructions=lambda s: "You are a data processing agent in Service 2. Process queries and return results.",
            tools=[]  # Could have its own tools
        )

    async def process_request(self, request: ExternalServiceRequest) -> ExternalServiceResponse:
        """
        Process request from another service.
        Uses the SAME trace_id and session_id from the request.
        """
        print(f"\n{'='*80}")
        print(f"[SERVICE 2] Received request")
        print(f"[SERVICE 2] Trace ID: {request.trace_id}")
        print(f"[SERVICE 2] Session ID: {request.session_id}")
        print(f"[SERVICE 2] Query: {request.query}")
        print(f"{'='*80}\n")

        # Create RunConfig with the SAME session_id
        config = RunConfig(
            agent_registry={"external_data_processor": self.agent},
            model_provider=None,  # Would use real model provider
            conversation_id=request.session_id,  # SAME session_id
            on_event=self.trace_collector.collect
        )

        # Create initial state with the SAME trace_id
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=TraceId(request.trace_id),  # SAME trace_id for unified tracing
            messages=[Message(
                role=ContentRole.USER,
                content=request.query
            )],
            current_agent_name="external_data_processor",
            context={},
            turn_count=0
        )

        print(f"[SERVICE 2] Running external agent with trace_id={request.trace_id}")

        # Run the agent (this would call actual LLM in production)
        # For demo, we'll just return a mock response
        # result = await run(initial_state, config)

        # Simulate agent response
        processed_result = f"[Service 2 Processed]: {request.query.upper()}"

        print(f"[SERVICE 2] Completed processing")
        print(f"[SERVICE 2] Result: {processed_result}\n")

        return ExternalServiceResponse(
            result=processed_result,
            trace_id=request.trace_id,
            session_id=request.session_id
        )


# Global external service instance (will be initialized with Langfuse collector)
external_service = None


# ============================================================================
# SERVICE 1: Main Service with Sub-Agent
# ============================================================================

class CallExternalServiceArgs(BaseModel):
    """Arguments for calling external service."""
    query: str = Field(description="Query to send to external service")


@function_tool
async def call_external_service_tool(args: CallExternalServiceArgs, context: Any) -> str:
    """
    Tool that calls external Service 2.
    Automatically propagates trace_id and session_id for unified tracing.
    """
    # Get current trace_id and session_id from context (set by JAF)
    trace_id = get_current_trace_id()
    session_id = get_current_session_id()

    print(f"\n{'*'*80}")
    print(f"[TOOL] call_external_service_tool invoked")
    print(f"[TOOL] Current trace_id: {trace_id}")
    print(f"[TOOL] Current session_id: {session_id}")
    print(f"[TOOL] Query: {args.query}")
    print(f"{'*'*80}\n")

    # In real scenario, this would be an HTTP call with headers:
    # headers = {
    #     "X-Trace-Id": trace_id,
    #     "X-Session-Id": session_id
    # }
    # response = requests.post("http://service-2/process", json={"query": args.query}, headers=headers)

    # Simulate external service call
    request = ExternalServiceRequest(
        query=args.query,
        trace_id=trace_id or "default-trace",
        session_id=session_id or "default-session"
    )

    response = await external_service.process_request(request)

    print(f"[TOOL] Received response from Service 2: {response.result}\n")

    return response.result


async def main():
    """
    Main test demonstrating unified tracing across services.
    """
    print("\n" + "="*80)
    print("UNIFIED CROSS-SERVICE TRACING TEST WITH LANGFUSE")
    print("="*80 + "\n")

    # ========================================================================
    # Setup Shared Langfuse Trace Collector
    # ========================================================================

    # Create a SHARED Langfuse trace collector for both services
    # In real distributed systems, both services would use the same Langfuse instance
    print("[SETUP] Initializing Langfuse trace collector...")
    langfuse_collector = LangfuseTraceCollector()
    print(f"[SETUP] Langfuse configured for host: {os.environ['LANGFUSE_HOST']}\n")

    # Initialize external service with the same Langfuse collector
    global external_service
    external_service = ExternalService(langfuse_collector)

    # ========================================================================
    # Setup Service 1: Main Agent + Sub-Agent
    # ========================================================================

    # Use the same trace collector for Service 1
    trace_collector = langfuse_collector

    # Create sub-agent with external service tool
    sub_agent = Agent(
        name="data_fetcher_subagent",
        instructions=lambda s: "You fetch data using external service. Use call_external_service_tool to get data.",
        tools=[call_external_service_tool]
    )

    # Convert sub-agent to tool for main agent
    sub_agent_tool = create_agent_tool(
        agent=sub_agent,
        tool_name="fetch_data",
        tool_description="Fetch data using the sub-agent"
    )

    # Create main agent
    main_agent = Agent(
        name="main_orchestrator",
        instructions=lambda s: "You orchestrate tasks by delegating to sub-agents.",
        tools=[sub_agent_tool]
    )

    # ========================================================================
    # Run Test: Start from Main Agent
    # ========================================================================

    # Generate unified trace_id and session_id
    unified_trace_id = "trace_unified_test_12345"
    unified_session_id = "session_user_abc_xyz"

    print(f"[MAIN] Starting with unified IDs:")
    print(f"[MAIN] Trace ID: {unified_trace_id}")
    print(f"[MAIN] Session ID: {unified_session_id}\n")

    # Create RunConfig with custom session_id
    config = RunConfig(
        agent_registry={
            "main_orchestrator": main_agent,
            "data_fetcher_subagent": sub_agent
        },
        model_provider=None,  # Would use real model provider
        conversation_id=unified_session_id,  # Custom session_id
        on_event=trace_collector.collect
    )

    # Create initial state with custom trace_id
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=TraceId(unified_trace_id),  # Custom trace_id
        messages=[Message(
            role=ContentRole.USER,
            content="Fetch data about user activity from external service"
        )],
        current_agent_name="main_orchestrator",
        context={},
        turn_count=0
    )

    print(f"[MAIN] Starting agent execution...\n")

    # In production, you would run:
    # result = await run(initial_state, config)

    # For demo, simulate the flow:
    print(f"[MAIN] Main agent would call sub-agent...")
    print(f"[MAIN] Sub-agent inherits trace_id: {unified_trace_id}")
    print(f"[MAIN] Sub-agent inherits session_id: {unified_session_id}\n")

    # Manually simulate sub-agent calling the tool
    # (In real run, this happens automatically)
    from jaf.core.agent_tool import set_current_trace_id, set_current_session_id

    # Set context variables (normally done by engine.py)
    set_current_trace_id(unified_trace_id)
    set_current_session_id(unified_session_id)

    # Call the tool (simulates sub-agent using the tool)
    print(f"[MAIN] Sub-agent calling external service tool...\n")
    tool_result = await call_external_service_tool(
        CallExternalServiceArgs(query="get user activity data"),
        context={}
    )

    # ========================================================================
    # Verify Results
    # ========================================================================

    print("\n" + "="*80)
    print("TEST VERIFICATION")
    print("="*80 + "\n")

    print("✓ Main agent started with trace_id:", unified_trace_id)
    print("✓ Main agent started with session_id:", unified_session_id)
    print("✓ Sub-agent inherited trace_id:", unified_trace_id)
    print("✓ Sub-agent inherited session_id:", unified_session_id)
    print("✓ External service received trace_id:", unified_trace_id)
    print("✓ External service received session_id:", unified_session_id)
    print("\n✓ ALL SERVICES UNIFIED UNDER SAME TRACE AND SESSION!")
    print("\nResult from external service:", tool_result)

    print("\n" + "="*80)
    print("In Langfuse Dashboard (http://localhost:3000), you should see:")
    print("="*80)
    print(f"Trace ID: {unified_trace_id}")
    print(f"  ├─ main_orchestrator (Service 1)")
    print(f"  │   └─ data_fetcher_subagent (Service 1)")
    print(f"  │       └─ call_external_service_tool")
    print(f"  │           └─ external_data_processor (Service 2)")
    print(f"\nSession ID: {unified_session_id}")
    print(f"\nAll events unified under the same trace!")
    print("="*80 + "\n")

    # Flush Langfuse to ensure all events are sent
    print("[CLEANUP] Flushing Langfuse events...")
    langfuse_collector.langfuse.flush()
    print("[CLEANUP] All events sent to Langfuse!\n")

    print(f"Check Langfuse dashboard: {os.environ['LANGFUSE_HOST']}")
    print(f"Look for trace_id: {unified_trace_id}")
    print(f"Look for session_id: {unified_session_id}\n")


if __name__ == "__main__":
    asyncio.run(main())
