#!/usr/bin/env python3
"""
Test that actually generates Langfuse traces for cross-service tracing.
This manually emits trace events to verify Langfuse integration.
"""

import asyncio
import os
from datetime import datetime

# Set Langfuse credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-d8b28b3e-eb67-4916-9bc9-4cd3d57ee7fa"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-50fea071-404e-4ecf-b963-afed3630d742"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

from jaf.core.tracing import LangfuseTraceCollector
from jaf.core.types import (
    RunStartEvent,
    RunStartEventData,
    RunEndEvent,
    RunEndEventData,
    RunId,
    TraceId,
    Message,
    ContentRole,
    CompletedOutcome,
)
from jaf import generate_run_id


async def main():
    """Generate actual trace events to Langfuse."""

    print("\n" + "="*80)
    print("GENERATING ACTUAL LANGFUSE TRACES")
    print("="*80 + "\n")

    # Initialize Langfuse collector
    print("[SETUP] Initializing Langfuse...")
    collector = LangfuseTraceCollector()
    print(f"[SETUP] Connected to: {os.environ['LANGFUSE_HOST']}\n")

    # Unified IDs for cross-service tracing
    unified_trace_id = "trace_unified_test_12345"
    unified_session_id = "session_user_abc_xyz"

    print(f"[TRACE] Creating unified trace:")
    print(f"  Trace ID: {unified_trace_id}")
    print(f"  Session ID: {unified_session_id}\n")

    # ========================================================================
    # SERVICE 1: Main Agent - Emit run_start event
    # ========================================================================

    print("[SERVICE 1] Emitting run_start event for main_orchestrator...")

    main_run_id = generate_run_id()

    run_start_event = RunStartEvent(
        data=RunStartEventData(
            run_id=main_run_id,
            trace_id=TraceId(unified_trace_id),
            session_id=unified_session_id,
            agent_name="main_orchestrator",
            messages=[
                Message(
                    role=ContentRole.USER,
                    content="Fetch data about user activity from external service"
                )
            ],
            context=type('obj', (object,), {
                'query': 'Fetch data about user activity from external service',
                'user_info': {'email': 'test@example.com'}
            })()
        )
    )

    collector.collect(run_start_event)
    print("[SERVICE 1] ✓ run_start event sent\n")

    await asyncio.sleep(0.5)  # Simulate processing time

    # ========================================================================
    # SERVICE 1: Sub-Agent - Emit run_start event (inherits same trace_id)
    # ========================================================================

    print("[SERVICE 1 - SUB-AGENT] Emitting run_start event for data_fetcher_subagent...")

    sub_run_id = generate_run_id()

    sub_agent_start_event = RunStartEvent(
        data=RunStartEventData(
            run_id=sub_run_id,
            trace_id=TraceId(unified_trace_id),  # SAME trace_id
            session_id=unified_session_id,  # SAME session_id
            agent_name="data_fetcher_subagent",
            messages=[
                Message(
                    role=ContentRole.USER,
                    content="Use external service to fetch user activity data"
                )
            ],
            context=type('obj', (object,), {
                'query': 'Use external service to fetch user activity data'
            })()
        )
    )

    collector.collect(sub_agent_start_event)
    print("[SERVICE 1 - SUB-AGENT] ✓ run_start event sent\n")

    await asyncio.sleep(0.5)

    # ========================================================================
    # SERVICE 2: External Agent - Emit run_start event (same trace_id!)
    # ========================================================================

    print("[SERVICE 2] Emitting run_start event for external_data_processor...")

    external_run_id = generate_run_id()

    external_start_event = RunStartEvent(
        data=RunStartEventData(
            run_id=external_run_id,
            trace_id=TraceId(unified_trace_id),  # SAME trace_id for unified tracing!
            session_id=unified_session_id,  # SAME session_id!
            agent_name="external_data_processor",
            messages=[
                Message(
                    role=ContentRole.USER,
                    content="Process: get user activity data"
                )
            ],
            context=type('obj', (object,), {
                'query': 'Process: get user activity data'
            })()
        )
    )

    collector.collect(external_start_event)
    print("[SERVICE 2] ✓ run_start event sent\n")

    await asyncio.sleep(0.5)

    # ========================================================================
    # SERVICE 2: External Agent - Emit run_end event
    # ========================================================================

    print("[SERVICE 2] Emitting run_end event for external_data_processor...")

    external_end_event = RunEndEvent(
        data=RunEndEventData(
            outcome=CompletedOutcome(output="[Service 2 Processed]: GET USER ACTIVITY DATA"),
            trace_id=TraceId(unified_trace_id),
            run_id=external_run_id
        )
    )

    collector.collect(external_end_event)
    print("[SERVICE 2] ✓ run_end event sent\n")

    await asyncio.sleep(0.5)

    # ========================================================================
    # SERVICE 1: Sub-Agent - Emit run_end event
    # ========================================================================

    print("[SERVICE 1 - SUB-AGENT] Emitting run_end event for data_fetcher_subagent...")

    sub_agent_end_event = RunEndEvent(
        data=RunEndEventData(
            outcome=CompletedOutcome(output="Successfully fetched data from external service"),
            trace_id=TraceId(unified_trace_id),
            run_id=sub_run_id
        )
    )

    collector.collect(sub_agent_end_event)
    print("[SERVICE 1 - SUB-AGENT] ✓ run_end event sent\n")

    await asyncio.sleep(0.5)

    # ========================================================================
    # SERVICE 1: Main Agent - Emit run_end event
    # ========================================================================

    print("[SERVICE 1] Emitting run_end event for main_orchestrator...")

    run_end_event = RunEndEvent(
        data=RunEndEventData(
            outcome=CompletedOutcome(output="Task completed successfully"),
            trace_id=TraceId(unified_trace_id),
            run_id=main_run_id
        )
    )

    collector.collect(run_end_event)
    print("[SERVICE 1] ✓ run_end event sent\n")

    # ========================================================================
    # Flush to Langfuse
    # ========================================================================

    print("="*80)
    print("[FLUSH] Sending all events to Langfuse...")
    collector.langfuse.flush()
    print("[FLUSH] ✓ All events flushed to Langfuse!")
    print("="*80 + "\n")

    print("✅ SUCCESS! Check Langfuse dashboard:\n")
    print(f"   URL: {os.environ['LANGFUSE_HOST']}")
    print(f"   Trace ID: {unified_trace_id}")
    print(f"   Session ID: {unified_session_id}\n")
    print("You should see a unified trace with all three agents:")
    print("   - main_orchestrator (Service 1)")
    print("   - data_fetcher_subagent (Service 1)")
    print("   - external_data_processor (Service 2)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
