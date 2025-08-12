"""
JAF Server Demo - Python Implementation

This demonstrates a complete JAF server with multiple agents and tools,
directly converted from the TypeScript server demo.
"""

import os
import asyncio
from typing import Any, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import JAF components
from jaf import (
    Tool, Agent, make_litellm_provider, ConsoleTraceCollector,
    ToolResponse, ToolErrorCodes, with_error_handling, run_server,
    create_run_id, create_trace_id, RunState, Message
)

# Define context type
@dataclass
class MyContext:
    user_id: str
    permissions: list[str]

# Define tool schemas using Pydantic
class CalculateArgs(BaseModel):
    expression: str = Field(description="Math expression to evaluate (e.g., '2 + 2', '10 * 5')")

class GreetArgs(BaseModel):
    name: str = Field(description="Name of the person to greet")

# Create tools with standardized error handling
class CalculatorTool:
    """Calculator tool that performs mathematical calculations."""
    
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Perform mathematical calculations',
            'parameters': CalculateArgs
        })()
    
    async def execute(self, args: CalculateArgs, context: MyContext) -> Any:
        """Execute calculator tool with error handling."""
        # Basic safety check - only allow simple math expressions (including spaces)
        sanitized = ''.join(c for c in args.expression if c in '0123456789+-*/(). ')
        if sanitized != args.expression:
            return ToolResponse.validation_error(
                "Invalid characters in expression. Only numbers, +, -, *, /, and () are allowed.",
                {
                    'original_expression': args.expression,
                    'sanitized_expression': sanitized,
                    'invalid_characters': ''.join(c for c in args.expression if c not in '0123456789+-*/(). ')
                }
            )
        
        try:
            # Remove spaces for evaluation
            expression_for_eval = sanitized.replace(' ', '')
            result = eval(expression_for_eval)  # Note: In production, use a safe math evaluator
            return ToolResponse.success(
                f"{args.expression} = {result}",
                {
                    'original_expression': args.expression,
                    'result': result,
                    'calculation_type': 'arithmetic'
                }
            )
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Failed to evaluate expression: {str(e)}",
                {
                    'expression': args.expression,
                    'error': str(e)
                }
            )

class GreetingTool:
    """Greeting tool that generates personalized greetings."""
    
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'greet',
            'description': 'Generate a personalized greeting',
            'parameters': GreetArgs
        })()
    
    async def execute(self, args: GreetArgs, context: MyContext) -> Any:
        """Execute greeting tool with error handling."""
        # Validate name input
        if not args.name or args.name.strip() == "":
            return ToolResponse.validation_error(
                "Name cannot be empty",
                {'provided_name': args.name}
            )
        
        # Check for extremely long names (potential abuse)
        if len(args.name) > 100:
            return ToolResponse.validation_error(
                "Name is too long (max 100 characters)",
                {
                    'name_length': len(args.name),
                    'max_length': 100
                }
            )
        
        greeting = f"Hello, {args.name.strip()}! Nice to meet you. I'm a helpful AI assistant running on the JAF framework."
        
        return ToolResponse.success(
            greeting,
            {
                'greeted_name': args.name.strip(),
                'greeting_type': 'personal'
            }
        )

# Create tool instances
calculator_tool = CalculatorTool()
greeting_tool = GreetingTool()

# Define agents
def create_math_agent() -> Agent[MyContext, str]:
    """Create a math tutor agent."""
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a helpful math tutor. Use the calculator tool to perform calculations and explain math concepts clearly.'
    
    return Agent(
        name='MathTutor',
        instructions=instructions,
        tools=[calculator_tool]
    )

def create_chat_agent() -> Agent[MyContext, str]:
    """Create a friendly chatbot agent."""
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a friendly chatbot. Use the greeting tool when meeting new people, and engage in helpful conversation.'
    
    return Agent(
        name='ChatBot',
        instructions=instructions,
        tools=[greeting_tool]
    )

def create_assistant_agent() -> Agent[MyContext, str]:
    """Create a general-purpose assistant agent."""
    def instructions(state: RunState[MyContext]) -> str:
        return 'You are a general-purpose assistant. You can help with math calculations and provide greetings.'
    
    return Agent(
        name='Assistant',
        instructions=instructions,
        tools=[calculator_tool, greeting_tool]
    )

async def start_server():
    """Start the JAF development server."""
    load_dotenv()
    print('🚀 Starting JAF Development Server...\n')
    
    # Check if LiteLLM configuration is provided
    litellm_url = os.getenv('LITELLM_URL', 'http://localhost:4000')
    litellm_api_key = os.getenv('LITELLM_API_KEY')
    
    print(f'📡 LiteLLM URL: {litellm_url}')
    print(f'🔑 API Key: {"Set" if litellm_api_key else "Not set"}')
    print('⚠️  Note: Chat endpoints will fail without a running LiteLLM server\n')
    
    # Set up model provider
    model_provider = make_litellm_provider(litellm_url, litellm_api_key)
    
    # Set up tracing
    trace_collector = ConsoleTraceCollector()
    
    # Set up memory provider based on environment
    memory_provider = None
    memory_config = None
    
    try:
        from jaf.memory import create_memory_provider_from_env, MemoryConfig
        
        # Create external clients if needed (for database connections)
        external_clients = {}
        
        memory_type = os.getenv("JAF_MEMORY_TYPE", "memory").lower()
        print(f'🧠 Memory Type: {memory_type}')
        
        if memory_type == "redis":
            try:
                import redis.asyncio as redis
                
                # Create Redis client based on environment
                redis_url = os.getenv("JAF_REDIS_URL")
                if redis_url:
                    redis_client = redis.from_url(redis_url)
                else:
                    redis_client = redis.Redis(
                        host=os.getenv("JAF_REDIS_HOST", "localhost"),
                        port=int(os.getenv("JAF_REDIS_PORT", "6379")),
                        password=os.getenv("JAF_REDIS_PASSWORD"),
                        db=int(os.getenv("JAF_REDIS_DB", "0"))
                    )
                
                external_clients["redis"] = redis_client
                print(f'🔗 Redis connection configured')
                
            except ImportError:
                print('⚠️  Redis library not installed. Run: pip install redis')
                print('   Using in-memory storage instead')
                
        elif memory_type == "postgres":
            try:
                import asyncpg
                
                # Create PostgreSQL connection
                connection_string = os.getenv("JAF_POSTGRES_CONNECTION_STRING")
                if connection_string:
                    postgres_client = await asyncpg.connect(connection_string)
                else:
                    postgres_client = await asyncpg.connect(
                        host=os.getenv("JAF_POSTGRES_HOST", "localhost"),
                        port=int(os.getenv("JAF_POSTGRES_PORT", "5432")),
                        database=os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
                        user=os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
                        password=os.getenv("JAF_POSTGRES_PASSWORD"),
                        ssl=os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true"
                    )
                
                external_clients["postgres"] = postgres_client
                print(f'🔗 PostgreSQL connection configured')
                
            except ImportError:
                print('⚠️  asyncpg library not installed. Run: pip install asyncpg')
                print('   Using in-memory storage instead')
            except Exception as e:
                print(f'⚠️  Failed to connect to PostgreSQL: {e}')
                print('   Using in-memory storage instead')
        
        # Create memory provider
        memory_provider = await create_memory_provider_from_env(external_clients)
        memory_config = MemoryConfig(
            provider=memory_provider,
            auto_store=True,
            max_messages=int(os.getenv("JAF_MEMORY_MAX_MESSAGES", "1000"))
        )
        
        print(f'✅ Memory provider created: {type(memory_provider).__name__}')
        
    except Exception as e:
        print(f'⚠️  Failed to set up memory provider: {e}')
        print('   Conversations will not persist between sessions')
    
    try:
        print('🔧 Creating server...')
        
        # Create agents
        math_agent = create_math_agent()
        chat_agent = create_chat_agent()
        assistant_agent = create_assistant_agent()
        
        # Start the server
        from jaf.core.types import RunConfig
        
        run_config = RunConfig(
            agent_registry={
                'MathTutor': math_agent,
                'ChatBot': chat_agent,
                'Assistant': assistant_agent
            },
            model_provider=model_provider,
            max_turns=5,
            model_override=os.getenv('LITELLM_MODEL', 'gemini-2.5-pro'),
            on_event=trace_collector.collect,
            memory=memory_config
        )
        
        server_options = {
            'port': int(os.getenv('PORT', '3000')),
            'host': '127.0.0.1',
            'cors': False
        }
        
        # Create server config
        from jaf.server.types import ServerConfig
        
        server_config = ServerConfig(
            host=server_options['host'],
            port=server_options['port'],
            agent_registry=run_config.agent_registry,
            run_config=run_config,
            cors=server_options['cors']
        )
        
        print('\n📚 Try these example requests:')
        print('')
        print('1. Health Check:')
        print('   curl http://localhost:3000/health')
        print('')
        print('2. List Agents:')
        print('   curl http://localhost:3000/agents')
        print('')
        print('3. Chat with Math Tutor:')
        print('   curl -X POST http://localhost:3000/chat \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"messages":[{"role":"user","content":"What is 15 * 7?"}],"agent_name":"MathTutor","context":{"userId":"demo","permissions":["user"]}}\'')
        print('')
        print('4. Chat with ChatBot:')
        print('   curl -X POST http://localhost:3000/agents/ChatBot/chat \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"messages":[{"role":"user","content":"Hi, my name is Alice"}],"context":{"userId":"demo","permissions":["user"]}}\'')
        print('')
        print('5. Chat with Assistant:')
        print('   curl -X POST http://localhost:3000/chat \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"messages":[{"role":"user","content":"Calculate 25 + 17 and then greet me as Bob"}],"agent_name":"Assistant","context":{"userId":"demo","permissions":["user"]}}\'')
        print('')
        
        if memory_config:
            print('6. Start a persistent conversation:')
            print('   curl -X POST http://localhost:3000/chat \\')
            print('     -H "Content-Type: application/json" \\')
            print('     -d \'{"messages":[{"role":"user","content":"Hello, I am starting a new conversation"}],"agent_name":"ChatBot","conversation_id":"my-conversation","context":{"userId":"demo","permissions":["user"]}}\'')
            print('')
            print('7. Continue the conversation:')
            print('   curl -X POST http://localhost:3000/chat \\')
            print('     -H "Content-Type: application/json" \\')
            print('     -d \'{"messages":[{"role":"user","content":"Do you remember me?"}],"agent_name":"ChatBot","conversation_id":"my-conversation","context":{"userId":"demo","permissions":["user"]}}\'')
            print('')
            print('8. Get conversation history:')
            print('   curl http://localhost:3000/conversations/my-conversation')
            print('')
            print('9. Delete conversation:')
            print('   curl -X DELETE http://localhost:3000/conversations/my-conversation')
            print('')
            print('10. Memory health check:')
            print('    curl http://localhost:3000/memory/health')
            print('')
        print('🚀 Starting server...')
        
        # Start the server (this will block until server stops)
        await run_server(server_config)
        
    except Exception as error:
        print(f'❌ Failed to start server: {error}')
        import traceback
        traceback.print_exc()

async def main():
    """Main entry point."""
    try:
        await start_server()
    except KeyboardInterrupt:
        print('\n🛑 Received interrupt, shutting down gracefully...')
    except Exception as error:
        print(f'❌ Unhandled error in main: {error}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
