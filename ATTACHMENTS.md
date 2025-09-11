# JAF Attachment Support

This document describes the comprehensive attachment support in the JAF Python framework, allowing agents to process images, documents, and other file types.

## Overview

JAF supports rich multimodal interactions through a robust attachment system that handles:

- **Images**: JPEG, PNG, GIF, WebP, BMP, SVG with visual analysis
- **Documents**: PDF, Word (DOCX), Excel (XLSX) with content extraction  
- **Files**: Plain text, CSV, JSON, ZIP and other formats
- **URLs**: Remote content via HTTP/HTTPS
- **Base64**: Inline data encoding for any content type

## Quick Start

### Basic Usage

```python
from jaf import (
    make_image_attachment, 
    make_document_attachment,
    Message, 
    Agent
)

# Create an image attachment
image_attachment = make_image_attachment(
    url="https://example.com/image.jpg",
    mime_type="image/jpeg",
    name="example.jpg"
)

# Create a message with attachment
message = Message(
    role="user",
    content="What do you see in this image?",
    attachments=[image_attachment]
)
```

### Document Processing

```python
import base64
from jaf import make_document_attachment

# Create document from base64 data
pdf_data = base64.b64encode(pdf_bytes).decode('ascii')
doc_attachment = make_document_attachment(
    data=pdf_data,
    mime_type="application/pdf", 
    name="report.pdf"
)

# Use LiteLLM format for large documents
large_doc = make_document_attachment(
    url="https://example.com/large-report.pdf",
    mime_type="application/pdf",
    use_litellm_format=True  # Efficient processing
)
```

## Installation

Install with attachment processing support:

```bash
# Basic installation
pip install jaf-py

# With document processing dependencies
pip install 'jaf-py[attachments]'

# All features
pip install 'jaf-py[all]'
```

### Document Processing Dependencies

The attachment system has optional dependencies for document processing:

- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing  
- **openpyxl**: Excel spreadsheet processing
- **Pillow**: Image processing and validation
- **python-magic**: MIME type detection
- **aiofiles**: Async file operations

Check dependency status:

```python
from jaf.utils.document_processor import check_dependencies, get_missing_dependencies

# Check what's available
deps = check_dependencies()
print(f"PDF support: {deps['pdf']}")
print(f"Excel support: {deps['excel']}")

# Get list of missing dependencies
missing = get_missing_dependencies()
if missing:
    print(f"Install missing: {', '.join(missing)}")
```

## Attachment Types

### Images (`kind="image"`)

```python
from jaf import make_image_attachment

# From URL
image = make_image_attachment(
    url="https://picsum.photos/400/300",
    mime_type="image/jpeg",
    name="random-image.jpg"
)

# From base64 data
import base64
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('ascii')

image = make_image_attachment(
    data=image_data,
    mime_type="image/png",
    name="my-image.png"
)
```

**Supported formats**: JPEG, PNG, GIF, WebP, BMP, SVG

### Documents (`kind="document"`)

```python
from jaf import make_document_attachment

# PDF with content extraction
pdf_doc = make_document_attachment(
    url="https://example.com/report.pdf",
    mime_type="application/pdf",
    name="report.pdf"
)

# Large PDF with LiteLLM format (no extraction, native model processing)
large_pdf = make_document_attachment(
    url="https://example.com/large-report.pdf", 
    mime_type="application/pdf",
    name="large-report.pdf",
    use_litellm_format=True
)

# Word document
docx_doc = make_document_attachment(
    data=base64_docx_data,
    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    name="document.docx"
)
```

**Supported formats**: PDF, DOCX, XLSX

### Files (`kind="file"`)

```python
from jaf import make_file_attachment

# Text file
text_file = make_file_attachment(
    data=base64.b64encode(b"Hello, world!").decode('ascii'),
    mime_type="text/plain",
    name="hello.txt",
    format="txt"
)

# CSV file (with structured analysis)
csv_file = make_file_attachment(
    data=base64_csv_data,
    mime_type="text/csv", 
    name="data.csv"
)

# JSON file
json_file = make_file_attachment(
    data=base64_json_data,
    mime_type="application/json",
    name="config.json"
)
```

**Supported formats**: TXT, CSV, JSON, ZIP, and more

## Document Processing

### Automatic Content Extraction

When documents are sent to agents, JAF automatically extracts text content:

```python
# This PDF will have its text content extracted and sent to the model
pdf_attachment = make_document_attachment(
    url="https://example.com/document.pdf",
    mime_type="application/pdf",
    name="document.pdf"
    # use_litellm_format=False (default)
)
```

### LiteLLM Format

For large documents or when you want native model processing:

```python
# This PDF will be sent directly to the model without extraction
pdf_attachment = make_document_attachment(
    url="https://example.com/large-document.pdf",
    mime_type="application/pdf", 
    name="large-document.pdf",
    use_litellm_format=True  # Native model processing
)
```

**Benefits of LiteLLM format:**
- No context window waste from text extraction
- Better layout understanding  
- Tables and images preserved
- Automatic provider optimization
- Efficient for large files

### Supported Processing

| Format | Extension | Content Extraction | LiteLLM Format |
|--------|-----------|-------------------|----------------|
| PDF | .pdf | ✅ Text only | ✅ Full document |
| Word | .docx | ✅ Text + structure | ✅ Full document |
| Excel | .xlsx | ✅ Data + metadata | ✅ Full spreadsheet |
| Text | .txt | ✅ Full content | ❌ |
| CSV | .csv | ✅ Structure + preview | ❌ |
| JSON | .json | ✅ Pretty printed | ❌ |
| ZIP | .zip | ✅ File listing | ❌ |

## Server Integration

### HTTP API

The JAF server accepts attachments in HTTP requests:

```bash
curl -X POST http://localhost:3002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agentName": "my-agent",
    "messages": [
      {
        "role": "user",
        "content": "Analyze this image",
        "attachments": [
          {
            "kind": "image",
            "mime_type": "image/jpeg",
            "name": "photo.jpg", 
            "url": "https://example.com/photo.jpg"
          }
        ]
      }
    ]
  }'
```

### Multi-part Content

JAF supports OpenAI-style multi-part message content:

```python
from jaf.core.types import Message, MessageContentPart

message = Message(
    role="user",
    content=[
        MessageContentPart(type="text", text="Look at this image: "),
        MessageContentPart(
            type="image_url", 
            image_url={"url": "https://example.com/image.jpg"}
        )
    ]
)
```

## Security & Validation

### Automatic Validation

All attachments are automatically validated:

```python
from jaf import make_image_attachment, AttachmentValidationError

try:
    # This will fail - dangerous filename
    attachment = make_image_attachment(
        data="...",
        name="../../../etc/passwd"
    )
except AttachmentValidationError as e:
    print(f"Validation failed: {e}")
```

### Security Features

- **Filename validation**: Prevents path traversal attacks
- **MIME type validation**: Ensures correct content types
- **Size limits**: 10MB per attachment (25MB with LiteLLM)
- **Base64 validation**: Ensures proper encoding
- **URL validation**: Restricts to safe protocols

### Limits

```python
from jaf.utils.attachments import ATTACHMENT_LIMITS

print(f"Max size: {ATTACHMENT_LIMITS['MAX_SIZE'] // 1024 // 1024}MB")
print(f"Max filename: {ATTACHMENT_LIMITS['MAX_FILENAME_LENGTH']} chars")
print(f"Image types: {len(ATTACHMENT_LIMITS['ALLOWED_IMAGE_MIME_TYPES'])}")
print(f"Document types: {len(ATTACHMENT_LIMITS['ALLOWED_DOCUMENT_MIME_TYPES'])}")
```

## Examples

### Running the Demo Server

```bash
# Start the attachment demo server
python examples/attachment_demo_server.py

# Test with curl commands (provided in output)
curl -X POST http://localhost:3002/chat \
  -H "Content-Type: application/json" \
  -d '{"agentName": "attachment-analyst", "messages": [...]}'
```

### Client Examples

```bash
# Run the client examples
python examples/attachment_client_example.py
```

### Complete Agent Example

```python
import asyncio
from jaf import Agent, RunState, run, RunConfig, make_litellm_provider, Message
from jaf import make_image_attachment

def create_vision_agent():
    def instructions(state: RunState) -> str:
        return "You are a helpful assistant that can analyze images and documents."
    
    return Agent(
        name="vision-agent",
        instructions=instructions,
        model_config={"name": "gpt-4o", "temperature": 0.7}
    )

async def main():
    # Create agent and provider
    agent = create_vision_agent()
    provider = make_litellm_provider("https://api.openai.com/v1", "your-api-key")
    
    # Create message with image
    image_attachment = make_image_attachment(
        url="https://picsum.photos/400/300",
        mime_type="image/jpeg",
        name="random.jpg"
    )
    
    message = Message(
        role="user",
        content="What do you see in this image?",
        attachments=[image_attachment]
    )
    
    # Run agent
    config = RunConfig(
        agent_registry={"vision-agent": agent},
        model_provider=provider
    )
    
    initial_state = RunState(
        run_id="demo",
        trace_id="demo",
        messages=[message],
        current_agent_name="vision-agent",
        context={},
        turn_count=0,
        approvals={}
    )
    
    result = await run(initial_state, config)
    print(f"Response: {result.final_state.messages[-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install with `pip install 'jaf-py[attachments]'`
2. **Large files**: Use `use_litellm_format=True` for documents > 10MB
3. **MIME type errors**: Ensure correct MIME types for attachments
4. **Vision model errors**: Use vision-capable models (gpt-4o, claude-sonnet-4, etc.)

### Debug Information

```python
from jaf.utils.document_processor import check_dependencies

# Check what processing is available
deps = check_dependencies()
for feature, available in deps.items():
    print(f"{feature}: {'✅' if available else '❌'}")
```

### Model Compatibility

| Model | Image Support | Document Support | LiteLLM Format |
|-------|---------------|------------------|----------------|
| GPT-4o | ✅ | ✅ | ✅ |
| GPT-4 Vision | ✅ | ✅ | ✅ | 
| Claude Sonnet 4 | ✅ | ✅ | ✅ |
| Gemini 2.5 Pro | ✅ | ✅ | ✅ |
| GPT-3.5 Turbo | ❌ | ✅ | ✅ |

## Best Practices

1. **Use appropriate formats**: Images for visual content, documents for text/data
2. **Leverage LiteLLM format**: For large documents and better processing
3. **Validate early**: Use attachment utilities to catch issues early
4. **Handle errors gracefully**: Always catch `AttachmentValidationError`
5. **Choose right models**: Use vision-capable models for image analysis
6. **Monitor sizes**: Stay within attachment size limits
7. **Security first**: Never disable validation in production

## API Reference

See the full API documentation for detailed information about all attachment functions, types, and configuration options.