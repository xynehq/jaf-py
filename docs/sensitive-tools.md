# Enhanced Sensitive Tools Support

JAF now provides comprehensive support for detecting and handling sensitive content in tool inputs and outputs, with automatic tracing redaction to prevent sensitive data from being logged.

## Features

### Automatic Sensitivity Detection
- **PII Detection**: Social Security Numbers, email addresses, phone numbers  
- **Secrets Detection**: API keys, passwords, tokens
- **Custom Patterns**: Configurable regex patterns for domain-specific sensitivity
- **LLM Guard Integration**: Optional advanced detection using the llm-guard library

### Manual Sensitivity Marking
- **ToolSchema.sensitive**: Explicit marking of tools as sensitive
- **Tool Name Heuristics**: Automatic detection based on tool names containing keywords like "secret", "password", etc.

### Tracing Redaction
- **Automatic Redaction**: Sensitive tool inputs/outputs automatically replaced with `[REDACTED]` in traces
- **Multiple Collectors**: Works with all tracing collectors (Console, File, OTEL, Langfuse)
- **Conversation History**: Sensitive content redacted from LLM conversation history

## Quick Start

### Basic Usage

```python
from jaf.core.tools import create_function_tool
from jaf.core.sensitive import SensitiveContentConfig

# Manually mark a tool as sensitive
sensitive_tool = create_function_tool({
    "name": "get_api_key",
    "description": "Retrieve API key", 
    "execute": my_function,
    "parameters": MyArgs,
    "sensitive": True  # Explicit marking
})

# Configure automatic detection
config = SensitiveContentConfig(
    auto_detect_sensitive=True,
    enable_secrets_detection=True,
    sensitivity_threshold=0.7
)
```

### RunConfig Integration

```python
from jaf.core.types import RunConfig
from jaf.core.tracing import create_composite_trace_collector, ConsoleTraceCollector

run_config = RunConfig(
    agent_registry={"my_agent": agent},
    model_provider=provider,
    sensitive_content_config=config,  # Enable automatic detection
    redact_sensitive_tools_in_traces=True,  # Enable tracing redaction
    on_event=create_composite_trace_collector(ConsoleTraceCollector()).collect
)
```

## Configuration Options

### SensitiveContentConfig

```python
@dataclass
class SensitiveContentConfig:
    # Enable automatic sensitivity detection
    auto_detect_sensitive: bool = True
    
    # LLM Guard scanner configurations  
    enable_secrets_detection: bool = True
    enable_pii_detection: bool = False  # Requires model downloads
    enable_code_detection: bool = False  # Requires model downloads
    
    # Custom regex patterns
    custom_patterns: List[str] = None
    
    # Sensitivity score threshold (0.0 - 1.0)
    sensitivity_threshold: float = 0.7
```

### Installation

For basic functionality (heuristic detection):
```bash
pip install jaf-py
```

For advanced LLM Guard integration:
```bash  
pip install jaf-py[sensitive]
```

## How It Works

1. **Tool Execution**: When a tool is called, both input arguments and output are scanned
2. **Sensitivity Detection**: Content is checked using:
   - Explicit `ToolSchema.sensitive=True` marking
   - Tool name heuristics (names containing "secret", "password", etc.)
   - Heuristic regex patterns (SSN, credit cards, emails, API keys)
   - Optional LLM Guard scanners (when available)
3. **Tracing Redaction**: If sensitive content is detected:
   - Tool inputs/outputs are replaced with `[REDACTED]` in trace events
   - LLM conversation history is sanitized before being traced
   - Original content remains available to the LLM for continued execution

## Examples

See `examples/enhanced_sensitive_tools_demo.py` for a comprehensive demonstration of all features.

## Best Practices

1. **Explicit Marking**: Always explicitly mark tools as sensitive when handling known sensitive data
2. **Custom Patterns**: Add domain-specific regex patterns for your use case
3. **Threshold Tuning**: Adjust `sensitivity_threshold` based on your false positive tolerance
4. **Testing**: Test sensitivity detection with your actual tool inputs/outputs
5. **Documentation**: Document which tools handle sensitive data for your team

## Compatibility

- **Existing Tools**: All existing tools continue to work unchanged
- **Tracing**: Compatible with all existing trace collectors
- **LLM Guard**: Optional dependency - graceful fallback to heuristic detection
- **Offline**: Fully functional without internet connectivity (heuristic mode)