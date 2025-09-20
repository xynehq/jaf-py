import os
import asyncio
from typing import Dict, Any

from pydantic import BaseModel, Field

from jaf.core.types import (
    Agent,
    RunState,
    Message,
    ContentRole,
    RunConfig,
    ModelConfig,
    generate_run_id,
    generate_trace_id,
)
from jaf.core.engine import run
from jaf.core.tools import create_function_tool
from jaf.core.tracing import ConsoleTraceCollector, create_composite_trace_collector
from jaf.providers.model import make_litellm_provider


class SecretBalanceArgs(BaseModel):
    account_id: str = Field(..., description="The account identifier to fetch the secret balance for")


class PublicEchoArgs(BaseModel):
    text: str = Field(..., description="Text to echo back")


# Sensitive tool: simulate fetching a secret value
async def get_secret_balance(args: SecretBalanceArgs, context: Dict[str, Any]) -> str:
    # Simulate some PII/secret handling; this value must not be traced
    secret_value = f"balance_for_{args.account_id}=INR 3,21,900.55"
    # Return raw string; engine wraps it into a JSON envelope and adds 'sensitive': true for redaction
    return secret_value


# Non-sensitive tool: safe to trace
async def public_echo(args: PublicEchoArgs, context: Dict[str, Any]) -> str:
    return f"ECHO: {args.text}"


def build_agent() -> Agent[Dict[str, Any], str]:
    # Mark the secret tool as sensitive at schema level
    secret_tool = create_function_tool({
        "name": "get_secret_balance",
        "description": "Fetch the sensitive balance for an account_id",
        "execute": get_secret_balance,
        "parameters": SecretBalanceArgs,
        "sensitive": True,  # This flag makes tracing automatically redact inputs/outputs
    })

    echo_tool = create_function_tool({
        "name": "public_echo",
        "description": "Echo text back (non-sensitive)",
        "execute": public_echo,
        "parameters": PublicEchoArgs,
    })

    def instructions(state: RunState[Dict[str, Any]]) -> str:
        return "\n".join([
            "You are a function-calling assistant.",
            "Rules:",
            "1. If the user asks about an account balance, call get_secret_balance exactly once.",
            "2. Otherwise, you may call public_echo.",
            "3. After calling a tool, return a concise final answer to the user.",
        ])

    return Agent[Dict[str, Any], str](
        name="SensitiveToolsAgent",
        instructions=instructions,
        tools=[secret_tool, echo_tool],
        model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.5-pro")),
    )


async def main():
    print("=== Sensitive Tools + Tracing (LiteLLM) Demo ===")
    litellm_url = os.getenv("LITELLM_URL", "URL not set")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "anything")
    model = os.getenv("LITELLM_MODEL", "gemini-2.5-pro")

    print(f"LiteLLM URL: {litellm_url}")
    print(f"Model: {model}")

    provider = make_litellm_provider(base_url=litellm_url, api_key=litellm_api_key)

    # Enable composite tracing; child collectors (console, file, OTEL, Langfuse) all receive sanitized events
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

    agent = build_agent()

    # Sample user asks about a specific account balance (sensitive)
    user_prompt = "What's the balance for account id acc-9988?"

    initial_state = RunState[Dict[str, Any]](
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content=user_prompt)],
        current_agent_name=agent.name,
        context={"user_id": "demo-user"},
        turn_count=0,
    )

    # Redaction is automatic when a tool is marked sensitive. The run-level flag is optional and defaults to False.
    # Tracing will never store tool args/results for sensitive tools, nor include them in traced LLM histories.
    config = RunConfig[Dict[str, Any]](
        agent_registry={agent.name: agent},
        model_provider=provider,
        model_override=model,
        max_turns=5,
        on_event=trace_collector.collect,
        # Optional run-level preference (default False). Redaction still happens automatically based on tool sensitivity.
        # redact_sensitive_tools_in_traces=True,
    )

    result = await run(initial_state, config)

    print("\n=== Run Complete ===")
    if result.outcome.status == "completed":
        print(f"Final Answer: {result.outcome.output}")
    else:
        print(f"Run Error: {getattr(result.outcome.error, 'detail', str(result.outcome.error))}")

    print("\nNote:")
    print("- Sensitive tool inputs/outputs are redacted in tracing (console/file/OTEL/Langfuse).")
    print("- The LLM still receives the sensitive tool result within the runtime state to proceed normally.")
    print("- In traces, tool_call_start args and tool_call_end result/tool_result are replaced with [REDACTED]/None.")
    print("- Any sensitive tool messages in conversation history are redacted before being traced.")


if __name__ == "__main__":
    asyncio.run(main())