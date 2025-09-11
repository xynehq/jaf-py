"""
JAF Attachment Demo Server

This example demonstrates how to create a JAF server that supports various types of attachments
including images, documents, and files. The server can process and analyze different content types.

Run with: python examples/attachment_demo_server.py
"""

import os
from typing import Any

from jaf import Agent, RunState, RunConfig, make_litellm_provider
from jaf.server import create_jaf_server, ServerConfig
from jaf.core.types import ModelConfig


def create_attachment_agent() -> Agent[Any, str]:
    """Create an agent that can analyze attachments."""
    
    def instructions(state: RunState[Any]) -> str:
        return """You are an AI assistant that can analyze various types of attachments including images and documents.
When users send you attachments, analyze them carefully and provide helpful, detailed responses about their content.

For images: Describe what you see in detail.
For documents: Analyze and summarize the content, structure, or data as appropriate.
Supported document types: PDF, DOCX, XLSX, CSV, TXT, JSON, ZIP files."""

    return Agent(
        name="attachment-analyst",
        instructions=instructions,
        model_config=ModelConfig(
            name="claude-sonnet-4",
            temperature=0.7,
            max_tokens=1000
        )
    )


def main():
    """Main function to run the attachment demo server."""
    
    # Configuration
    DEFAULT_BASE_URL = 'https://api.openai.com/v1'
    DEFAULT_API_KEY = 'your-api-key-here'
    DEFAULT_PORT = 3002
    DEFAULT_HOST = 'localhost'
    DEFAULT_MAX_TURNS = 5

    # Get environment variables with defaults
    litellm_base_url = os.getenv('LITELLM_BASE_URL', DEFAULT_BASE_URL)
    litellm_api_key = os.getenv('LITELLM_API_KEY', DEFAULT_API_KEY)

    if litellm_api_key == DEFAULT_API_KEY:
        print("Warning: LITELLM_API_KEY not set. Server will start but may not function correctly.")

    # Create model provider
    model_provider = make_litellm_provider(litellm_base_url, litellm_api_key)

    # Create agent
    attachment_agent = create_attachment_agent()

    # Create agent registry
    agent_registry = {
        'attachment-analyst': attachment_agent
    }

    # Create run configuration
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=DEFAULT_MAX_TURNS
    )

    # Create server configuration
    server_config = ServerConfig(
        port=DEFAULT_PORT,
        host=DEFAULT_HOST,
        agent_registry=agent_registry,
        run_config=run_config
    )

    # Create and start server
    app = create_jaf_server(server_config)
    
    print(f"JAF Attachment Demo Server starting on http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print("\nTesting Attachment Support - Use these curl commands:\n")
    
    print("1. Simple text message:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user", 
        "content": "Hello! Can you help me analyze attachments?"
      }}
    ]
  }}'
""")
    
    print("2. Image with URL attachment:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "What do you see in this image?",
        "attachments": [
          {{
            "kind": "image",
            "mime_type": "image/jpeg",
            "name": "random-image.jpg",
            "url": "https://picsum.photos/400/300"
          }}
        ]
      }}
    ]
  }}'
""")

    print("3. Image attachment with base64 data:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst", 
    "messages": [
      {{
        "role": "user",
        "content": "Analyze this small test image",
        "attachments": [
          {{
            "kind": "image",
            "mime_type": "image/png",
            "name": "test-pixel.png",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
          }}
        ]
      }}
    ]
  }}'
""")

    print("4. PDF document:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "What can you tell me about this PDF document?",
        "attachments": [
          {{
            "kind": "document",
            "mime_type": "application/pdf",
            "name": "sample.pdf",
            "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
          }}
        ]
      }}
    ]
  }}'
""")

    print("5. Text file:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "Please analyze this text file",
        "attachments": [
          {{
            "kind": "file",
            "mime_type": "text/plain",
            "name": "sample.txt",
            "data": "VGhpcyBpcyBhIHNhbXBsZSB0ZXh0IGZpbGUgZm9yIHRlc3RpbmcgYXR0YWNobWVudCBmdW5jdGlvbmFsaXR5LgoKSXQgY29udGFpbnMgbXVsdGlwbGUgbGluZXMgb2YgdGV4dCB0byBkZW1vbnN0cmF0ZSBob3cgdGV4dCBmaWxlcyBhcmUgcHJvY2Vzc2VkLgoKS2V5IHBvaW50czoKLSBUZXh0IGZpbGUgcHJvY2Vzc2luZyB3b3JrcwotIE11bHRpcGxlIGxpbmVzIGFyZSBzdXBwb3J0ZWQKLSBTcGVjaWFsIGNoYXJhY3RlcnMgYW5kIGZvcm1hdHRpbmcgYXJlIHByZXNlcnZlZA=="
          }}
        ]
      }}
    ]
  }}'
""")

    print("6. CSV data:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "Analyze this CSV data",
        "attachments": [
          {{
            "kind": "file",
            "mime_type": "text/csv",
            "name": "sales_data.csv",
            "data": "TmFtZSxBZ2UsRGVwYXJ0bWVudCxTYWxhcnkKSm9obiBEb2UsMzUsRW5naW5lZXJpbmcsNzUwMDAKSmFuZSBTbWl0aCwyOCxNYXJrZXRpbmcsNjUwMDAKTWlrZSBKb2huc29uLDQyLFNhbGVzLDcwMDAwCkFubmEgTGVlLDI2LEhSLDU1MDAwClJvYmVydCBXaWxzb24sMzksRmluYW5jZSw4MDAwMA=="
          }}
        ]
      }}
    ]
  }}'
""")

    print("7. JSON file:")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "What is in this JSON file?",
        "attachments": [
          {{
            "kind": "file",
            "mime_type": "application/json",
            "name": "config.json",
            "data": "ewogICJuYW1lIjogIlNhbXBsZSBBcHAiLAogICJ2ZXJzaW9uIjogIjEuMC4wIiwKICAiZGVzY3JpcHRpb24iOiAiQSBzYW1wbGUgYXBwbGljYXRpb24gZm9yIGRlbW9uc3RyYXRpb24gcHVycG9zZXMiLAogICJhdXRob3IiOiAiSm9obiBEb2UiLAogICJsaWNlbnNlIjogIk1JVCIsCiAgImRlcGVuZGVuY2llcyI6IHsKICAgICJleHByZXNzIjogIl40LjE4LjIiLAogICAgImxvZGFzaCI6ICJeNC4xNy4yMSIsCiAgICAiYXhpb3MiOiAiXjEuNi4yIgogIH0sCiAgInNjcmlwdHMiOiB7CiAgICAic3RhcnQiOiAibm9kZSBpbmRleC5qcyIsCiAgICAidGVzdCI6ICJucG0gcnVuIGplc3QiCiAgfQp9"
          }}
        ]
      }}
    ]
  }}'
""")

    print("8. LiteLLM format - Large PDF via URL (efficient):")
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "Analyze this large PDF using LiteLLM format",
        "attachments": [
          {{
            "kind": "document",
            "mime_type": "application/pdf",
            "name": "large-pdf.pdf",
            "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "use_litellm_format": true
          }}
        ]
      }}
    ]
  }}'
""")

    print("9. LiteLLM format - Base64 document (native processing):")
    pdf_b64 = "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCgoyIDAgb2JqCjw8Ci9UeXBlIC9QYWdlcwovS2lkcyBbMyAwIFJdCi9Db3VudCAxCj4+CmVuZG9iagoKMyAwIG9iago8PAovVHlwZSAvUGFnZQovUGFyZW50IDIgMCBSCi9NZWRpYUJveCBbMCAwIDYxMiA3OTJdCi9Db250ZW50cyA0IDAgUgo+PgplbmRvYmoKCjQgMCBvYmoKPDwKL0xlbmd0aCA0NAo+PgpzdHJlYW0KQlQKL0YxIDEyIFRmCjEwMCA3MDAgVGQKKEhlbGxvIFdvcmxkKSBUagpFVAplbmRzdHJlYW0KZW5kb2JqCgp4cmVmCjAgNQowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMTAgMDAwMDAgbiAKMDAwMDAwMDA1MyAwMDAwMCBuIAowMDAwMDAwMTI1IDAwMDAwIG4gCjAwMDAwMDAyMTAgMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSA1Ci9Sb290IDEgMCBSCj4+CnN0YXJ0eHJlZgoyNjQKJSVFT0Y="
    print(f"""curl -X POST http://{DEFAULT_HOST}:{DEFAULT_PORT}/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "agent_name": "attachment-analyst",
    "messages": [
      {{
        "role": "user",
        "content": "Process this document using native model capabilities",
        "attachments": [
          {{
            "kind": "document",
            "mime_type": "application/pdf",
            "name": "document.pdf",
            "data": "{pdf_b64}",
            "use_litellm_format": true
          }}
        ]
      }}
    ]
  }}'
""")

    print("\nConfiguration:")
    print("- Use Ctrl+C to stop the server")
    print("- Image attachments: Full visual analysis")
    print("- Document attachments: Text extraction and analysis for PDF, DOCX, XLSX, CSV, TXT, JSON, ZIP")
    print("- LiteLLM format: Use 'use_litellm_format': true for efficient large file processing")
    print("  * Large PDFs: No context window waste, native model processing")
    print("  * Better layout understanding, tables, images preserved")  
    print("  * Automatic provider optimization (Bedrock, Gemini, OpenAI)")
    print("- URL support: Both remote URLs and base64 data supported")
    print("- Base64 strings in examples contain real document content")
    print("- Security validations will reject malicious inputs")
    print("- Max attachment size: 10MB per attachment (25MB with LiteLLM format)")
    print("- Max body size: 25MB total per request")
    print("\nNote: To use document processing features, install with:")
    print("pip install 'jaf-py[attachments]'")
    print()

    # Import uvicorn and run server
    try:
        import uvicorn
        uvicorn.run(
            app,
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down attachment demo server...")
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install 'jaf-py[server]'")


if __name__ == "__main__":
    main()