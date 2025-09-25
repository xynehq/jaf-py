"""
Enhanced Sensitive Tools Demo with Automatic Detection

This demo shows both manual and automatic sensitive content detection using
LLM Guard integration for sophisticated sensitivity detection.
"""

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
from jaf.core.sensitive import SensitiveContentConfig
from jaf.providers.model import make_litellm_provider


class UserInfoArgs(BaseModel):
    user_id: str = Field(..., description="User ID to fetch information for")


class DatabaseQueryArgs(BaseModel):
    query: str = Field(..., description="SQL query to execute")


class ApiKeyArgs(BaseModel):
    service_name: str = Field(..., description="Name of the service to get API key for")


# Tool that handles user information (automatically detected as sensitive due to PII patterns)
async def get_user_info(args: UserInfoArgs, context: Dict[str, Any]) -> str:
    # Simulate retrieving user information that contains PII
    user_info = {
        "user_id": args.user_id,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "555-123-4567",
        "ssn": "123-45-6789",  # This will trigger sensitivity detection
        "address": "123 Main St, Anytown, USA"
    }
    return f"User information: {user_info}"


# Tool that executes database queries (can contain sensitive data)
async def execute_database_query(args: DatabaseQueryArgs, context: Dict[str, Any]) -> str:
    # This would normally execute against a real database
    if "password" in args.query.lower() or "secret" in args.query.lower():
        return f"Query result: Found 3 records with encrypted passwords and API tokens"
    return f"Query executed: {args.query}. Result: 42 rows affected."


# Tool that retrieves API keys (manually marked as sensitive)
async def get_api_key(args: ApiKeyArgs, context: Dict[str, Any]) -> str:
    # Simulate retrieving an API key
    fake_key = f"sk-{args.service_name}_1234567890abcdef1234567890abcdef"
    return f"API key for {args.service_name}: {fake_key}"


# Regular tool that shouldn't be sensitive
async def get_public_info(args: BaseModel, context: Dict[str, Any]) -> str:
    return "This is public information that can be safely traced."


def build_agent() -> Agent[Dict[str, Any], str]:
    # Create tools with different sensitivity configurations
    
    # Tool that will be automatically detected as sensitive due to output content
    user_info_tool = create_function_tool({
        "name": "get_user_info",
        "description": "Retrieve user information (may contain PII)",
        "execute": get_user_info,
        "parameters": UserInfoArgs,
        # Not explicitly marked as sensitive - will be auto-detected
    })
    
    # Tool that may contain sensitive content depending on query
    db_query_tool = create_function_tool({
        "name": "execute_database_query", 
        "description": "Execute a database query",
        "execute": execute_database_query,
        "parameters": DatabaseQueryArgs,
        # Not explicitly marked as sensitive - will be auto-detected based on content
    })
    
    # Explicitly sensitive tool
    api_key_tool = create_function_tool({
        "name": "get_api_key",
        "description": "Retrieve API key for a service",
        "execute": get_api_key,
        "parameters": ApiKeyArgs,
        "sensitive": True,  # Explicitly marked as sensitive
    })
    
    # Regular non-sensitive tool
    public_tool = create_function_tool({
        "name": "get_public_info",
        "description": "Get public information",
        "execute": get_public_info,
        "parameters": BaseModel,
    })

    def instructions(state: RunState[Dict[str, Any]]) -> str:
        return "\n".join([
            "You are an assistant with access to various tools including sensitive ones.",
            "You can:",
            "1. Fetch user information (may contain PII)",
            "2. Execute database queries (may return sensitive data)",
            "3. Retrieve API keys (always sensitive)",
            "4. Get public information (never sensitive)",
            "Use the appropriate tool based on the user's request.",
        ])

    return Agent[Dict[str, Any], str](
        name="EnhancedSensitiveAgent",
        instructions=instructions,
        tools=[user_info_tool, db_query_tool, api_key_tool, public_tool],
        model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.5-pro")),
    )


async def main():
    print("=== Enhanced Sensitive Tools Demo ===")
    print("Features:")
    print("- Manual sensitivity marking")
    print("- Automatic PII detection")
    print("- Automatic secrets detection")
    print("- Content-based sensitivity detection")
    print("- Comprehensive tracing redaction")
    print()
    
    litellm_url = os.getenv("LITELLM_URL", "URL not set")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "anything")
    model = os.getenv("LITELLM_MODEL", "gemini-2.5-pro")

    print(f"LiteLLM URL: {litellm_url}")
    print(f"Model: {model}")

    provider = make_litellm_provider(base_url=litellm_url, api_key=litellm_api_key)

    # Configure enhanced sensitivity detection
    sensitive_config = SensitiveContentConfig(
        auto_detect_sensitive=True,
        enable_secrets_detection=True,
        enable_pii_detection=True,
        enable_code_detection=False,  # Disable for this demo
        sensitivity_threshold=0.6,  # Lower threshold for demo purposes
        custom_patterns=[
            r'\bapi[_-]?key\b',  # Custom pattern for API keys
            r'\btoken\b',        # Custom pattern for tokens
        ]
    )

    # Enable composite tracing with redaction
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

    agent = build_agent()

    # Test cases to demonstrate different sensitivity scenarios
    test_cases = [
        "Get user information for user ID user123",
        "Execute this database query: SELECT * FROM users WHERE password IS NOT NULL",
        "Get the API key for the OpenAI service",
        "Get some public information",
    ]

    for i, user_prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {user_prompt}")
        print('='*60)
        
        initial_state = RunState[Dict[str, Any]](
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role=ContentRole.USER, content=user_prompt)],
            current_agent_name=agent.name,
            context={"user_id": "demo-user"},
            turn_count=0,
        )

        config = RunConfig[Dict[str, Any]](
            agent_registry={agent.name: agent},
            model_provider=provider,
            model_override=model,
            max_turns=5,
            on_event=trace_collector.collect,
            redact_sensitive_tools_in_traces=True,
            sensitive_content_config=sensitive_config,  # Enable enhanced detection
        )

        result = await run(initial_state, config)

        print(f"\nResult for Test Case {i}:")
        if result.outcome.status == "completed":
            print(f"Final Answer: {result.outcome.output}")
        else:
            print(f"Run Error: {getattr(result.outcome.error, 'detail', str(result.outcome.error))}")
        
        print(f"\nSensitivity Detection Notes:")
        print("- Tools marked with [REDACTED] had sensitive content")
        print("- Automatic detection works on both inputs and outputs")
        print("- Manual marking (get_api_key) always takes precedence")

    print(f"\n{'='*60}")
    print("Demo Complete!")
    print("Key Features Demonstrated:")
    print("✓ Manual sensitivity marking via ToolSchema.sensitive=True")
    print("✓ Automatic PII detection in outputs")
    print("✓ Automatic secrets detection in tool names/content")
    print("✓ Custom pattern matching for domain-specific sensitivity")
    print("✓ Comprehensive tracing redaction")
    print("✓ LLM continues to work normally with sensitive data")


if __name__ == "__main__":
    asyncio.run(main())