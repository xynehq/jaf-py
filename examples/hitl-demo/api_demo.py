#!/usr/bin/env python3

"""
File System HITL API Demo - With HTTP endpoints for approval

This demo extends the file system HITL demo with HTTP API endpoints
for remote approval/rejection via curl commands:
- All file operations from the main demo
- HTTP API server for approval management
- curl-based approval/rejection support
- Real-time coordination between terminal and API

Usage: python examples/hitl-demo/api_demo.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import concurrent.futures
import threading

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from jaf.core.types import RunState, RunConfig, create_run_id, create_trace_id, Message, ContentRole
from jaf.core.engine import run
from jaf.core.state import approve, reject
from jaf.providers.model import make_litellm_provider
from jaf.memory.approval_storage import create_in_memory_approval_storage

from shared.agent import file_system_agent, LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL
from shared.tools import FileSystemContext, DEMO_DIR
from shared.memory import setup_memory_provider, Colors


# Configuration
API_PORT = int(os.getenv("API_PORT", "3001"))

# Global state for pending approvals
pending_approvals: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API requests
class ApprovalRequest(BaseModel):
    additionalContext: Optional[Dict[str, Any]] = None


class RejectionRequest(BaseModel):
    reason: Optional[str] = "Rejected via API"
    additionalContext: Optional[Dict[str, Any]] = None


def create_model_provider():
    """Create model provider - requires LiteLLM configuration."""
    # Check if we have environment variables set (not using defaults)
    has_env_config = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_URL")
    has_api_key = os.getenv("LITELLM_API_KEY")

    if not has_env_config or not has_api_key:
        print(Colors.yellow("❌ No LiteLLM configuration found"))
        print(
            Colors.yellow(
                "   Please set LITELLM_BASE_URL and LITELLM_API_KEY environment variables"
            )
        )
        print(
            Colors.yellow(
                "   Example: LITELLM_BASE_URL=http://localhost:4000 LITELLM_API_KEY=your-key python examples/hitl-demo/api_demo.py"
            )
        )
        print(
            Colors.dim(
                "   Or copy examples/hitl-demo/.env.example to .env and configure your LiteLLM server"
            )
        )
        sys.exit(1)

    print(Colors.green(f"🤖 Using LiteLLM: {LITELLM_BASE_URL} ({LITELLM_MODEL})"))
    return make_litellm_provider(LITELLM_BASE_URL, LITELLM_API_KEY)


def setup_sandbox():
    """Setup demo sandbox directory."""
    try:
        DEMO_DIR.mkdir(parents=True, exist_ok=True)

        demo_files = [
            {
                "name": "README.txt",
                "content": "Welcome to the File System HITL API Demo!\\nThis is a sample file for testing.",
            },
            {
                "name": "config.json",
                "content": '{\\n  "app": "filesystem-api-demo",\\n  "version": "1.0.0",\\n  "api": true\\n}',
            },
            {
                "name": "notes.md",
                "content": "# API Demo Notes\\n\\n- This is a markdown file\\n- You can edit or delete it via terminal or API\\n- Operations require approval",
            },
        ]

        for file_info in demo_files:
            file_path = DEMO_DIR / file_info["name"]
            if not file_path.exists():
                file_path.write_text(file_info["content"], encoding="utf-8")

        print(Colors.green(f"📁 Sandbox directory ready: {DEMO_DIR}"))

    except Exception as e:
        print(Colors.yellow(f"Failed to setup sandbox: {e}"))
        sys.exit(1)


def display_welcome():
    """Display welcome message."""
    os.system("clear" if os.name == "posix" else "cls")
    print(Colors.cyan("🌐 JAF File System HITL API Demo"))
    print(Colors.cyan("===================================="))
    print()

    print(Colors.green("This demo showcases HITL with curl-based approval only:"))
    print(Colors.green("• Safe operations: listFiles, readFile (no approval)"))
    print(Colors.green("• Dangerous operations: deleteFile, editFile (require approval)"))
    print(Colors.green("• Approve/reject ONLY via curl commands"))
    print(Colors.green("• No terminal approval - must use API endpoints"))
    print()

    print(Colors.cyan("Try these commands:"))
    print('• "list files in the current directory"')
    print('• "read the README file"')
    print('• "edit the config file to add api: true"')
    print('• "delete the notes file"')
    print()

    print(Colors.yellow("API Endpoints:"))
    print(f"• GET http://localhost:{API_PORT}/pending - List pending approvals")
    print(f"• POST http://localhost:{API_PORT}/approve/:sessionId/:toolCallId - Approve")
    print(f"• POST http://localhost:{API_PORT}/reject/:sessionId/:toolCallId - Reject")
    print()

    print(Colors.dim('Commands: type "exit" to quit, "clear" to clear screen'))
    print()


def get_additional_context(tool_name: str) -> Dict[str, Any]:
    """Get additional context based on tool."""
    if tool_name == "deleteFile":
        return {
            "deletion_confirmed": {
                "confirmed_by": "demo-user",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "backup_created": True,
            }
        }
    elif tool_name == "editFile":
        return {
            "editing_approved": {
                "approved_by": "demo-user",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "safety_level": "standard",
            }
        }
    return {}


def setup_api_server():
    """Setup HTTP API server."""
    app = FastAPI(
        title="JAF HITL API Demo", description="File System HITL Demo with FastAPI", version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "pending_approvals": len(pending_approvals),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }

    @app.get("/pending")
    def list_pending_approvals():
        pending_list = []
        for key, data in pending_approvals.items():
            pending_list.append({"key": key, **data["metadata"]})
        return pending_list

    @app.post("/approve/{session_id}/{tool_call_id}")
    def approve_tool_call(
        session_id: str,
        tool_call_id: str,
        request: ApprovalRequest = Body(default=ApprovalRequest()),
    ):
        approval_key = f"{session_id}-{tool_call_id}"

        pending = pending_approvals.get(approval_key)
        if not pending:
            raise HTTPException(status_code=404, detail="Approval request not found")

        additional_context = request.additionalContext or {}

        result = {
            "approved": True,
            "source": "API",
            "additional_context": {
                **get_additional_context(pending["metadata"]["tool_name"]),
                **additional_context,
                "approved_via_api": True,
            },
        }

        # Use the concurrent.futures approach - thread-safe
        future = pending["future"]
        if not future.done():
            future.set_result(result)
            print(f"[API] Approval set for {approval_key}")
        else:
            print(f"[API] Future already done for {approval_key}")

        return {
            "message": "Approval recorded",
            "session_id": session_id,
            "tool_call_id": tool_call_id,
        }

    @app.post("/reject/{session_id}/{tool_call_id}")
    def reject_tool_call(
        session_id: str,
        tool_call_id: str,
        request: RejectionRequest = Body(default=RejectionRequest()),
    ):
        approval_key = f"{session_id}-{tool_call_id}"

        pending = pending_approvals.get(approval_key)
        if not pending:
            raise HTTPException(status_code=404, detail="Approval request not found")

        result = {
            "approved": False,
            "source": "API",
            "additional_context": {
                "rejection_reason": request.reason,
                "rejected_by": "api-user",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "rejected_via_api": True,
                **(request.additionalContext or {}),
            },
        }

        # Use the concurrent.futures approach - thread-safe
        future = pending["future"]
        if not future.done():
            future.set_result(result)
            print(f"[API] Rejection set for {approval_key}")
        else:
            print(f"[API] Future already done for {approval_key}")

        return {
            "message": "Rejection recorded",
            "session_id": session_id,
            "tool_call_id": tool_call_id,
        }

    return app


async def handle_approval(interruption: Any) -> Dict[str, Any]:
    """Handle approval request (curl-only)."""
    tool_call = interruption.tool_call

    # Parse arguments safely
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        args = {"arguments": tool_call.function.arguments}

    approval_key = f"{interruption.session_id}-{tool_call.id}"

    print(Colors.yellow("🛑 APPROVAL REQUIRED"))
    print()
    print(Colors.yellow(f"Tool: {tool_call.function.name}"))
    print(Colors.yellow("Arguments:"))
    for key, value in args.items():
        print(Colors.yellow(f"  {key}: {value}"))
    print(Colors.yellow(f"Session ID: {interruption.session_id}"))
    print(Colors.yellow(f"Tool Call ID: {tool_call.id}"))
    print()

    print(Colors.cyan("💡 Use curl to approve/reject:"))
    print(
        f"   Approve:  curl -X POST http://localhost:{API_PORT}/approve/{interruption.session_id}/{tool_call.id}"
    )
    print()
    print(
        f"   Approve with context: curl -X POST http://localhost:{API_PORT}/approve/{interruption.session_id}/{tool_call.id} \\"
    )
    print('              -H "Content-Type: application/json" \\')
    print('              -d \'{"additionalContext": {"message": "your-additional-context"}}\'')
    print()
    print(Colors.green("   📎 Approve with image (base64):"))
    print(
        f"   curl -X POST http://localhost:{API_PORT}/approve/{interruption.session_id}/{tool_call.id} \\"
    )
    print('        -H "Content-Type: application/json" \\')
    print(
        '        -d \'{"additionalContext": {"messages": [{"role": "user", "content": "Here is visual context", "attachments": [{"kind": "image", "mime_type": "image/png", "name": "test.png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAE..."}]}]}}\''
    )
    print()
    print(Colors.green("   📎 Approve with image (URL):"))
    print(
        f"   curl -X POST http://localhost:{API_PORT}/approve/{interruption.session_id}/{tool_call.id} \\"
    )
    print('        -H "Content-Type: application/json" \\')
    print(
        '        -d \'{"additionalContext": {"messages": [{"role": "user", "content": "Image for context", "attachments": [{"kind": "image", "mime_type": "image/jpeg", "name": "photo.jpg", "url": "https://example.com/image.jpg"}]}]}}\''
    )
    print()
    print(
        f"   Reject: curl -X POST http://localhost:{API_PORT}/reject/{interruption.session_id}/{tool_call.id}"
    )
    print()
    print(
        f"   Reject with context:  curl -X POST http://localhost:{API_PORT}/reject/{interruption.session_id}/{tool_call.id} \\"
    )
    print('              -H "Content-Type: application/json" \\')
    print(
        '              -d \'{"reason": "not authorized", "additionalContext": {"rejectedBy": "your-name"}}\''
    )
    print()
    print(f"   Check:   curl http://localhost:{API_PORT}/pending")
    print()

    # Store pending approval for API access only - use concurrent.futures for thread safety
    future = concurrent.futures.Future()
    pending_approvals[approval_key] = {
        "interruption": interruption,
        "future": future,
        "metadata": {
            "session_id": interruption.session_id,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.function.name,
            "arguments": args,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        },
    }

    print(Colors.dim("⏳ Waiting for curl approval/rejection..."))
    print()

    # Wait for API call only - use asyncio.run_in_executor for blocking concurrent.futures.Future
    print(f"[DEBUG] Waiting for future {approval_key}...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, future.result)
    print(f"[DEBUG] Future resolved with result: {result}")

    # Clean up pending approval
    pending_approvals.pop(approval_key, None)

    if result["approved"]:
        print(Colors.green("\\n✅ Approved via curl! Providing additional context...\\n"))
    else:
        print(Colors.yellow("\\n❌ Rejected via curl!\\n"))

    return result


async def process_conversation(
    user_input: str,
    conversation_history: List[Dict[str, str]],
    config: RunConfig[FileSystemContext],
) -> tuple[List[Dict[str, str]], bool]:
    """Process a single conversation turn."""

    # Add user message to conversation
    new_history = conversation_history + [{"role": "user", "content": user_input}]

    context = FileSystemContext(
        user_id="api-demo-user",
        working_directory=str(DEMO_DIR),
        permissions=["read", "write", "delete"],
    )

    # Convert history to Message objects
    messages = [
        Message(role=ContentRole(msg["role"]), content=msg["content"]) for msg in new_history
    ]

    state = RunState(
        run_id=create_run_id("filesystem-api-demo"),
        trace_id=create_trace_id("fs-api-trace"),
        messages=messages,
        current_agent_name="FileSystemAgent",
        context=context,
        turn_count=0,
        approvals={},
    )

    print(Colors.dim("⏳ Processing...\\n"))

    # Process with the engine
    while True:
        result = await run(state, config)

        if result.outcome.status == "interrupted":
            interruption = result.outcome.interruptions[0]

            if interruption.type == "tool_approval":
                approval_result = await handle_approval(interruption)

                if approval_result["approved"]:
                    state = await approve(
                        state, interruption, approval_result.get("additional_context"), config
                    )
                else:
                    state = await reject(
                        state, interruption, approval_result.get("additional_context"), config
                    )

                # Continue processing with the approval decision
                continue

        elif result.outcome.status == "completed":
            # Add assistant response to conversation history
            final_history = new_history + [{"role": "assistant", "content": result.outcome.output}]

            print(Colors.cyan("Assistant: ") + str(result.outcome.output) + "\\n")
            return final_history, True

        elif result.outcome.status == "error":
            print(Colors.yellow(f"❌ Error: {result.outcome.error}\\n"))
            return new_history, True


async def conversation_loop(
    conversation_history: List[Dict[str, str]], config: RunConfig[FileSystemContext]
):
    """Main conversation loop (recursive pattern)."""
    try:
        user_input = input(Colors.green("You: ")).strip()

        if user_input.lower() == "exit":
            print(Colors.cyan("👋 Goodbye!"))
            return

        if user_input.lower() == "clear":
            display_welcome()
            return await conversation_loop(conversation_history, config)

        if not user_input:
            return await conversation_loop(conversation_history, config)

        # Process the conversation turn
        conversation_history, should_continue = await process_conversation(
            user_input, conversation_history, config
        )

        if should_continue:
            # Recursive call to continue the conversation
            return await conversation_loop(conversation_history, config)

    except KeyboardInterrupt:
        print(Colors.cyan("\\n👋 Goodbye!"))
        return
    except EOFError:
        print(Colors.cyan("\\n👋 Goodbye!"))
        return


async def main():
    """Main demo function."""
    display_welcome()
    setup_sandbox()

    # Setup API server
    app = setup_api_server()

    # Generate session ID for this demo run
    session_id = f"api-demo-{int(time.time() * 1000)}"
    print(Colors.cyan(f"🔗 Session ID: {session_id}"))
    print()

    model_provider = create_model_provider()

    # Setup memory and approval storage
    memory_provider = await setup_memory_provider()

    print(Colors.cyan("🔐 Setting up approval storage..."))
    approval_storage = create_in_memory_approval_storage()
    print(Colors.green("✅ Approval storage initialized"))
    print()

    from jaf.memory.types import MemoryConfig

    memory_config = MemoryConfig(
        provider=memory_provider, auto_store=True, max_messages=50, store_on_completion=True
    )

    config = RunConfig(
        agent_registry={"FileSystemAgent": file_system_agent},
        model_provider=model_provider,
        memory=memory_config,
        conversation_id=f"filesystem-api-demo-{int(time.time() * 1000)}",
        approval_storage=approval_storage,
    )

    # Start API server in background
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="error")
    )
    server_thread.daemon = True
    server_thread.start()

    print(Colors.green(f"🌐 API server running on http://localhost:{API_PORT}"))
    print(Colors.dim(f"   Health: http://localhost:{API_PORT}/health"))
    print(Colors.dim(f"   Pending: http://localhost:{API_PORT}/pending"))
    print()

    try:
        # Start the recursive conversation loop
        await conversation_loop([], config)
    except Exception as e:
        print(Colors.yellow(f"Error: {e}"))
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Colors.cyan("\\n👋 Goodbye!"))
    except Exception as e:
        print(Colors.yellow(f"Error: {e}"))
        import traceback

        traceback.print_exc()
