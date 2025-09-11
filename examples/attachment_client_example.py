"""
JAF Attachment Client Example

This example demonstrates how to create and send attachments to JAF agents
using the attachment utilities and server.

Run with: python examples/attachment_client_example.py
"""

import asyncio
import base64
import json
from typing import Any

import httpx

from jaf import Agent, RunState, run, RunConfig, make_litellm_provider
from jaf.core.types import Message, ModelConfig
from jaf.utils.attachments import (
    make_image_attachment, 
    make_document_attachment,
    make_file_attachment,
    ATTACHMENT_LIMITS
)


# Sample base64 data for testing
SAMPLE_TEXT_B64 = base64.b64encode(b"This is a sample text file for testing.").decode('ascii')
SAMPLE_JSON_B64 = base64.b64encode(b'{"name": "test", "value": 42}').decode('ascii') 
SAMPLE_CSV_B64 = base64.b64encode(b"Name,Age,City\nJohn,30,NYC\nJane,25,LA").decode('ascii')

# Tiny 1x1 PNG pixel for testing
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


def create_attachment_agent() -> Agent[Any, str]:
    """Create an agent that can analyze attachments."""
    
    def instructions(state: RunState[Any]) -> str:
        return """You are an AI assistant that can analyze various types of attachments.
Provide helpful analysis and summaries of any attachments sent to you.
Be detailed but concise in your responses."""

    return Agent(
        name="attachment-analyst",
        instructions=instructions,
        model_config=ModelConfig(
            name="gpt-4o",
            temperature=0.7,
            max_tokens=500
        )
    )


async def test_attachments_with_agent():
    """Test attachments directly with an agent (no server)."""
    print("=== Testing Attachments with Agent (Direct) ===\n")
    
    # Create agent and provider
    agent = create_attachment_agent()
    model_provider = make_litellm_provider("https://api.openai.com/v1", "your-api-key")
    
    # Create run config
    config = RunConfig(
        agent_registry={"attachment-analyst": agent},
        model_provider=model_provider,
        max_turns=3
    )
    
    # Test 1: Text file attachment
    print("Test 1: Text file attachment")
    try:
        text_attachment = make_file_attachment(
            data=SAMPLE_TEXT_B64,
            mime_type="text/plain",
            name="sample.txt"
        )
        
        message = Message(
            role="user",
            content="Please analyze this text file",
            attachments=[text_attachment]
        )
        
        initial_state = RunState(
            run_id="test-1",
            trace_id="trace-1", 
            messages=[message],
            current_agent_name="attachment-analyst",
            context={},
            turn_count=0,
            approvals={}
        )
        
        print(f"Created text attachment: {text_attachment.name} ({text_attachment.mime_type})")
        print("✓ Text attachment created successfully\n")
        
    except Exception as e:
        print(f"✗ Error with text attachment: {e}\n")
    
    # Test 2: Image attachment (base64 data)
    print("Test 2: Image attachment from base64 data")
    try:
        image_attachment = make_image_attachment(
            data=TINY_PNG_B64,
            mime_type="image/png",
            name="sample-image.png"
        )
        
        print(f"Created image attachment: {image_attachment.name} from base64 data")
        print("✓ Image attachment from base64 data created successfully\n")
        
    except Exception as e:
        print(f"✗ Error with image attachment: {e}\n")
    
    # Test 3: Image attachment (base64 data)
    print("Test 3: Image attachment from base64 data")
    try:
        image_attachment = make_image_attachment(
            data=TINY_PNG_B64,
            mime_type="image/png", 
            name="tiny-pixel.png"
        )
        
        print(f"Created image attachment: {image_attachment.name} from base64 data")
        print("✓ Image attachment from base64 created successfully\n")
        
    except Exception as e:
        print(f"✗ Error with image attachment: {e}\n")
    
    # Test 4: Document attachment with LiteLLM format
    print("Test 4: Document attachment with LiteLLM format")
    try:
        doc_attachment = make_document_attachment(
            data=SAMPLE_JSON_B64,
            mime_type="application/json",
            name="config.json",
            use_litellm_format=True
        )
        
        print(f"Created document attachment: {doc_attachment.name} with LiteLLM format")
        print("✓ Document attachment with LiteLLM format created successfully\n")
        
    except Exception as e:
        print(f"✗ Error with document attachment: {e}\n")
    
    # Test 5: CSV file attachment
    print("Test 5: CSV file attachment")
    try:
        csv_attachment = make_file_attachment(
            data=SAMPLE_CSV_B64,
            mime_type="text/csv",
            name="data.csv"
        )
        
        print(f"Created CSV attachment: {csv_attachment.name}")
        print("✓ CSV attachment created successfully\n")
        
    except Exception as e:
        print(f"✗ Error with CSV attachment: {e}\n")


async def test_attachments_with_server():
    """Test attachments via HTTP server."""
    print("=== Testing Attachments with Server (HTTP) ===\n")
    
    server_url = "http://localhost:3002"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test server health
        print("Checking server health...")
        try:
            health_response = await client.get(f"{server_url}/health")
            if health_response.status_code == 200:
                print("✓ Server is running\n")
            else:
                print(f"✗ Server health check failed: {health_response.status_code}")
                return
        except Exception as e:
            print(f"✗ Cannot connect to server: {e}")
            print("Make sure to run: python examples/attachment_demo_server.py")
            return
        
        # Test 1: Simple text message
        print("Test 1: Simple text message")
        try:
            response = await client.post(f"{server_url}/chat", json={
                "agentName": "attachment-analyst",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello! Can you help me analyze attachments?"
                    }
                ]
            })
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Simple message test passed")
                print(f"Response: {result['data']['messages'][-1]['content'][:100]}...\n")
            else:
                print(f"✗ Simple message test failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error with simple message: {e}\n")
        
        # Test 2: Image attachment
        print("Test 2: Image attachment via HTTP")
        try:
            response = await client.post(f"{server_url}/chat", json={
                "agentName": "attachment-analyst",
                "messages": [
                    {
                        "role": "user",
                        "content": "What do you see in this image?",
                        "attachments": [
                            {
                                "kind": "image",
                                "mime_type": "image/png", 
                                "name": "test-pixel.png",
                                "data": TINY_PNG_B64
                            }
                        ]
                    }
                ]
            })
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Image attachment test passed")
                print(f"Response: {result['data']['messages'][-1]['content'][:100]}...\n")
            else:
                print(f"✗ Image attachment test failed: {response.status_code}")
                print(f"Error: {response.text}\n")
                
        except Exception as e:
            print(f"✗ Error with image attachment: {e}\n")
        
        # Test 3: Document attachment
        print("Test 3: Document attachment via HTTP")
        try:
            response = await client.post(f"{server_url}/chat", json={
                "agentName": "attachment-analyst", 
                "messages": [
                    {
                        "role": "user",
                        "content": "Please analyze this JSON file",
                        "attachments": [
                            {
                                "kind": "file",
                                "mime_type": "application/json",
                                "name": "config.json", 
                                "data": SAMPLE_JSON_B64
                            }
                        ]
                    }
                ]
            })
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Document attachment test passed")
                print(f"Response: {result['data']['messages'][-1]['content'][:100]}...\n")
            else:
                print(f"✗ Document attachment test failed: {response.status_code}")
                print(f"Error: {response.text}\n")
                
        except Exception as e:
            print(f"✗ Error with document attachment: {e}\n")


def print_attachment_info():
    """Print information about attachment capabilities."""
    print("=== JAF Attachment System Information ===\n")
    
    print("Supported attachment kinds:")
    print("- image: Images in various formats (JPEG, PNG, GIF, etc.)")
    print("- document: Structured documents (PDF, DOCX, XLSX)")  
    print("- file: General files (TXT, CSV, JSON, ZIP, etc.)")
    print()
    
    print("Attachment limits:")
    for key, value in ATTACHMENT_LIMITS.items():
        if key == 'MAX_SIZE':
            print(f"- {key}: {value // 1024 // 1024}MB")
        elif key in ['ALLOWED_IMAGE_MIME_TYPES', 'ALLOWED_DOCUMENT_MIME_TYPES']:
            print(f"- {key}: {len(value)} types supported")
        else:
            print(f"- {key}: {value}")
    print()
    
    print("Features:")
    print("- Base64 data support for any content type")
    print("- Remote URL support for images and documents")
    print("- LiteLLM format for efficient large document processing")
    print("- Automatic content extraction for supported document types")
    print("- Security validation for filenames and content")
    print("- Multi-part message content support")
    print()


async def main():
    """Main function to run attachment examples."""
    print("JAF Attachment Client Example\n")
    
    # Print attachment system information
    print_attachment_info()
    
    # Test attachments directly with agent
    await test_attachments_with_agent()
    
    # Test attachments via HTTP server
    await test_attachments_with_server()
    
    print("=== Example Complete ===")
    print("\nTo run the attachment demo server:")
    print("python examples/attachment_demo_server.py")
    print("\nTo install attachment processing dependencies:")
    print("pip install 'jaf-py[attachments]'")


if __name__ == "__main__":
    asyncio.run(main())