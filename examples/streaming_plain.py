import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel

from jaf.core.types import (
    Agent,
    RunState,
    Message,
    ContentRole,
    RunConfig,
    generate_run_id,
    generate_trace_id,
)
from jaf.core.engine import run
from jaf.core.tools import create_function_tool
from jaf.providers.model import make_litellm_provider


# Tool argument schemas
class GreetArgs(BaseModel):
    name: str


class TimeArgs(BaseModel):
    pass


# Tool implementations
async def greet_execute(args: GreetArgs, context: Dict[str, Any]) -> str:
    return f"Hello {args.name}!"


async def time_execute(args: TimeArgs, context: Dict[str, Any]) -> str:
    return json.dumps({"now": datetime.utcnow().isoformat() + "Z"})


greet_tool = create_function_tool(
    {
        "name": "greet",
        "description": "Greet the user by name",
        "execute": greet_execute,
        "parameters": GreetArgs,
    }
)

time_tool = create_function_tool(
    {
        "name": "get_current_time",
        "description": "Get the current server time",
        "execute": time_execute,
        "parameters": TimeArgs,
    }
)


def make_agent() -> Agent[Dict[str, Any], str]:
    def instructions(state: RunState[Dict[str, Any]]) -> str:
        return "\n".join(
            [
                "You are StreamerBot. You can greet users and tell the current time. Along with your response give the list of PMs of India with their detailed brief",
                "- If user tells their name, use the greet tool.",
                "- If user asks for time, use the get_current_time tool.",
                "- Otherwise, reply helpfully.",
            ]
        )

    return Agent[Dict[str, Any], str](
        name="StreamerPlainPy",
        instructions=instructions,
        tools=[greet_tool, time_tool],
    )


async def main():
    litellm_url = os.getenv("LITELLM_URL", "http://localhost:4000")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "anything")
    model = os.getenv("LITELLM_MODEL", "gpt-3.5-turbo")

    user_message = " ".join(sys.argv[1:]) or "Hi, I am Alice. What time is it?"

    print("📡 Using LiteLLM:", litellm_url)
    print("💬 Model:", model)
    print("📝 Prompt:", user_message)

    provider = make_litellm_provider(litellm_url, litellm_api_key)

    agent = make_agent()

    initial_state = RunState[Dict[str, Any]](
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content=user_message)],
        current_agent_name=agent.name,
        context={"user_id": "demo"},
        turn_count=0,
    )

    last_len = 0

    def on_event(event):
        nonlocal last_len
        etype = getattr(event, "type", None)
        if etype == "assistant_message":
            data = getattr(event, "data", {}) or {}
            msg = data.get("message", {}) if isinstance(data, dict) else {}
            content = ""
            if isinstance(msg, dict):
                content = msg.get("content") or ""
            else:
                # dataclass passthrough (unlikely, but safe)
                content = getattr(msg, "content", "") or ""
            if len(content) > last_len:
                delta = content[last_len:]
                sys.stdout.write(delta)
                sys.stdout.flush()
                last_len = len(content)
        elif etype == "run_end":
            sys.stdout.write("\n")
            sys.stdout.flush()

    run_config = RunConfig[Dict[str, Any]](
        agent_registry={agent.name: agent},
        model_provider=provider,
        model_override=model,
        max_turns=5,
        on_event=on_event,
    )

    print("\n🔴 Plain streaming below:\n")
    await run(initial_state, run_config)


if __name__ == "__main__":
    asyncio.run(main())
