import asyncio
import os
import sys
from pydantic import BaseModel

# Add the project root to Python path
sys.path.insert(0, '.')

from jaf.core.tools import create_function_tool
from jaf.core.types import Agent, Message, RunState, RunConfig, ModelConfig
from jaf.core.engine import run
from jaf.providers.model import make_litellm_provider

class MathArgs(BaseModel):
    """Arguments for math operations."""
    a: float
    b: float

async def calculator_function(args: MathArgs, context) -> str:
    """Simple calculator function."""
    result = args.a + args.b
    # Return a unique string to prove this function was called
    return f"MAGIC_TOOL_RESULT: {result}"

def test_instructions(state):
    return 'You are a math assistant with a calculator tool.'

async def test_tool_integration():
    """Test that the create_function_tool works with the JAF engine."""
    
    print("🧪 Testing create_function_tool integration with JAF engine...")
    print("=" * 65)
    
    # Create tool using create_function_tool
    calculator_tool = create_function_tool({
        'name': 'calculator',
        'description': 'Performs addition of two numbers',
        'execute': calculator_function,
        'parameters': MathArgs,
    })
    
    print(f"✅ Calculator tool created: {calculator_tool.schema.name}")
    
    # Create agent with the tool
    agent = Agent(
        name='MathAgent',
        instructions=test_instructions,
        tools=[calculator_tool],
        model_config=ModelConfig(name="gemini-2.5-pro")
    )
    
    print(f"✅ Agent created with {len(agent.tools)} tool(s)")
    
    # Create initial state
    initial_state = RunState(
        run_id='run-123',
        trace_id='trace-456',
        messages=[Message(role='user', content='What is 15 + 7?')],
        current_agent_name='MathAgent',
        context={},
        turn_count=0
    )
    
    # Create config
    litellm_url = os.environ.get("LITELLM_URL", "http://0.0.0.0:4000")
    litellm_api_key = os.environ.get("LITELLM_API_KEY", "anything")
    
    config = RunConfig(
        agent_registry={'MathAgent': agent},
        model_provider=make_litellm_provider(
            base_url=litellm_url,
            api_key=litellm_api_key
        ),
        max_turns=3
    )
    
    print("🚀 Running JAF engine with calculator tool...")
    
    try:
        result = await run(initial_state, config)
        print(f"✅ Engine run completed successfully")
        
        print(f"DEBUG: result.outcome = {result.outcome}")
        
        # Assertions
        assert result.outcome.status == 'completed'
        assert len(result.final_state.messages) == 4  # user, assistant (tool call), tool, assistant (final)
        
        tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
        assert len(tool_messages) == 1
        assert "MAGIC_TOOL_RESULT: 22.0" in tool_messages[0].content
        
        print(f"✅ Tool execution successful!")
        print(f"   Tool result: {tool_messages[0].content}")
        
    except Exception as e:
        print(f"❌ Engine run failed: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to fail the test
        raise
    
    print("\n" + "=" * 65)
    print("🎯 Integration Test Summary:")
    print("   - create_function_tool creates valid FunctionTool objects")
    print("   - Tools integrate correctly with JAF agents")
    print("   - JAF engine can execute tools created with create_function_tool")
    print("   - Mock model provider can trigger tool calls")
    print("✅ All integration tests passed!")

if __name__ == "__main__":
    asyncio.run(test_tool_integration())
