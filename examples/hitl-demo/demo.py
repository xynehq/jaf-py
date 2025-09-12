#!/usr/bin/env python3

"""
File System HITL Demo - Recursive conversation pattern

This demo showcases the HITL (Human-in-the-Loop) system with file operations:
- listFiles, readFile: No approval required
- deleteFile, editFile: Require approval
- Uses memory providers from environment
- Uses approval storage for persistence
- Recursive conversation pattern (no while loops)

Usage: python examples/hitl-demo/demo.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jaf.core.types import RunState, RunConfig, create_run_id, create_trace_id, Message, ContentRole
from jaf.core.engine import run
from jaf.core.state import approve, reject
from jaf.providers.model import make_litellm_provider
from jaf.memory.approval_storage import create_in_memory_approval_storage
from jaf.core.tracing import create_composite_trace_collector, ConsoleTraceCollector

from shared.agent import file_system_agent, LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL
from shared.tools import FileSystemContext, DEMO_DIR
from shared.memory import setup_memory_provider, Colors


def setup_sandbox():
    """Setup demo sandbox directory."""
    try:
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create some demo files
        demo_files = [
            {
                'name': 'README.txt',
                'content': 'Welcome to the File System HITL Demo!\nThis is a sample file for testing.'
            },
            {
                'name': 'config.json',
                'content': '{\n  "app": "filesystem-demo",\n  "version": "1.0.0"\n}'
            },
            {
                'name': 'notes.md',
                'content': '# Demo Notes\n\n- This is a markdown file\n- You can edit or delete it\n- Operations require approval'
            }
        ]
        
        for file_info in demo_files:
            file_path = DEMO_DIR / file_info['name']
            if not file_path.exists():
                file_path.write_text(file_info['content'], encoding='utf-8')
        
        print(Colors.green(f'📁 Sandbox directory ready: {DEMO_DIR}'))
        
    except Exception as e:
        print(Colors.yellow(f'Failed to setup sandbox: {e}'))
        sys.exit(1)


def display_welcome():
    """Display welcome message."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print(Colors.cyan('🗂️  JAF File System Human-in-the-Loop Demo'))
    print(Colors.cyan('=' * 48))
    print()
    
    print(Colors.green('This demo showcases HITL approval for file operations:'))
    print(Colors.green('• Safe operations: listFiles, readFile (no approval)'))
    print(Colors.green('• Dangerous operations: deleteFile, editFile (require approval)'))
    print(Colors.green('• Approval state persists using memory providers'))
    print(Colors.green('• Conversation history is maintained across sessions'))
    print()
    
    print(Colors.cyan('Try these commands:'))
    print('• "list files in the current directory"')
    print('• "read the README file"')
    print('• "edit the config file to add a new field"')
    print('• "delete the notes file"')
    print()
    
    print(Colors.dim('Commands: type "exit" to quit, "clear" to clear screen'))
    print()


def create_model_provider():
    """Create model provider - requires LiteLLM configuration."""
    # Check if we have environment variables set (not using defaults)
    has_env_config = os.getenv('LITELLM_BASE_URL') or os.getenv('LITELLM_URL')
    has_api_key = os.getenv('LITELLM_API_KEY')
    
    if not has_env_config or not has_api_key:
        print(Colors.yellow('❌ No LiteLLM configuration found'))
        print(Colors.yellow('   Please set LITELLM_BASE_URL and LITELLM_API_KEY environment variables'))
        print(Colors.yellow('   Example: LITELLM_BASE_URL=http://localhost:4000 LITELLM_API_KEY=your-key python examples/hitl-demo/demo.py'))
        print(Colors.dim('   Or copy examples/hitl-demo/.env.example to .env and configure your LiteLLM server'))
        sys.exit(1)
    
    print(Colors.green(f'🤖 Using LiteLLM: {LITELLM_BASE_URL} ({LITELLM_MODEL})'))
    return make_litellm_provider(LITELLM_BASE_URL, LITELLM_API_KEY)


async def handle_approval(interruption: Any) -> Dict[str, Any]:
    """Handle approval request interactively."""
    tool_call = interruption.tool_call
    
    # Parse arguments safely
    try:
        import json
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        args = {"arguments": tool_call.function.arguments}
    
    print(Colors.yellow('🛑 APPROVAL REQUIRED'))
    print()
    print(Colors.yellow(f'Tool: {tool_call.function.name}'))
    print(Colors.yellow('Arguments:'))
    for key, value in args.items():
        print(Colors.yellow(f'  {key}: {value}'))
    print(Colors.yellow(f'Session ID: {interruption.session_id}'))
    print()
    
    while True:
        approval = input(Colors.cyan('Do you approve this action? (y/n): ')).strip().lower()
        
        if approval in ['y', 'yes']:
            print(Colors.green('\n✅ Approved! Providing additional context...\n'))
            
            # Provide additional context based on the tool
            additional_context = {}
            
            if tool_call.function.name == 'deleteFile':
                additional_context = {
                    'deletion_confirmed': {
                        'confirmed_by': 'demo-user',
                        'timestamp': int(time.time() * 1000),
                        'backup_created': True
                    }
                }
            elif tool_call.function.name == 'editFile':
                additional_context = {
                    'editing_approved': {
                        'approved_by': 'demo-user', 
                        'timestamp': int(time.time() * 1000),
                        'safety_level': 'standard'
                    }
                }
            
            return {'approved': True, 'additional_context': additional_context}
            
        elif approval in ['n', 'no']:
            print(Colors.yellow('\n❌ Rejected!\n'))
            return {
                'approved': False,
                'additional_context': {
                    'rejection_reason': 'User declined the action',
                    'rejected_by': 'demo-user',
                    'timestamp': int(time.time() * 1000)
                }
            }
        else:
            print(Colors.yellow('Please enter "y" for yes or "n" for no.'))


async def process_conversation(
    user_input: str,
    conversation_history: List[Dict[str, str]],
    config: RunConfig[FileSystemContext]
) -> tuple[List[Dict[str, str]], bool]:
    """Process a single conversation turn."""
    
    # Add user message to conversation
    new_history = conversation_history + [{'role': 'user', 'content': user_input}]
    
    context = FileSystemContext(
        user_id='demo-user',
        working_directory=str(DEMO_DIR),
        permissions=['read', 'write', 'delete']
    )
    
    # Convert history to Message objects
    messages = [
        Message(role=ContentRole(msg['role']), content=msg['content'])
        for msg in new_history
    ]
    
    state = RunState(
        run_id=create_run_id('filesystem-demo'),
        trace_id=create_trace_id('fs-trace'),
        messages=messages,
        current_agent_name='FileSystemAgent',
        context=context,
        turn_count=0,
        approvals={}
    )
    
    print(Colors.dim('⏳ Processing...\n'))
    
    # Process with the engine
    while True:
        result = await run(state, config)
        
        if result.outcome.status == 'interrupted':
            interruption = result.outcome.interruptions[0]
            
            if interruption.type == 'tool_approval':
                approval_result = await handle_approval(interruption)
                
                if approval_result['approved']:
                    state = await approve(state, interruption, approval_result.get('additional_context'), config)
                else:
                    state = await reject(state, interruption, approval_result.get('additional_context'), config)
                
                # Continue processing with the approval decision
                continue
                
        elif result.outcome.status == 'completed':
            # Add assistant response to conversation history
            final_history = new_history + [{'role': 'assistant', 'content': result.outcome.output}]
            
            print(Colors.cyan('Assistant: ') + str(result.outcome.output) + '\n')
            return final_history, True
            
        elif result.outcome.status == 'error':
            print(Colors.yellow(f'❌ Error: {result.outcome.error}\n'))
            return new_history, True


async def conversation_loop(
    conversation_history: List[Dict[str, str]],
    config: RunConfig[FileSystemContext]
):
    """Main conversation loop (recursive pattern)."""
    try:
        user_input = input(Colors.green('You: ')).strip()
        
        if user_input.lower() == 'exit':
            print(Colors.cyan('👋 Goodbye!'))
            return
        
        if user_input.lower() == 'clear':
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
        print(Colors.cyan('\n👋 Goodbye!'))
        return
    except EOFError:
        print(Colors.cyan('\n👋 Goodbye!'))
        return


async def main():
    """Main demo function."""
    display_welcome()
    setup_sandbox()
    
    model_provider = create_model_provider()
    
    # Set up memory provider from environment
    memory_provider = await setup_memory_provider()
    
    # Set up approval storage
    print(Colors.cyan('🔐 Setting up approval storage...'))
    approval_storage = create_in_memory_approval_storage()
    print(Colors.green('✅ Approval storage initialized'))
    
    # Set up tracing
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())
    
    from jaf.memory.types import MemoryConfig
    memory_config = MemoryConfig(
        provider=memory_provider,
        auto_store=True,
        max_messages=50,
        store_on_completion=True
    )
    
    config = RunConfig(
        agent_registry={'FileSystemAgent': file_system_agent},
        model_provider=model_provider,
        memory=memory_config,
        conversation_id=f'filesystem-demo-{int(time.time() * 1000)}',
        approval_storage=approval_storage,
        on_event=trace_collector.collect
    )
    
    try:
        # Start the recursive conversation loop
        await conversation_loop([], config)
    except Exception as e:
        print(Colors.yellow(f'Error: {e}'))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())