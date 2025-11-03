"""
JAF Agent Development Kit (ADK) - Production-Ready Layer with Intelligence

The ADK provides a production-ready abstraction layer on top of the JAF Core,
featuring real LLM integration, production-grade session providers,
comprehensive error handling, intelligent multi-agent coordination, and
advanced schema validation.

Key Features:
- Real LLM integration with production providers
- Advanced multi-agent coordination with intelligent selection
- Comprehensive JSON Schema validation with business rules
- Production-grade session and memory providers
- Enhanced security and input sanitization
- Sophisticated response merging and delegation strategies

This layer transforms the JAF framework into a production-ready system with
intelligent orchestration capabilities.
"""

from .config import (
    create_adk_llm_config,
    create_default_adk_llm_config,
    create_adk_llm_config_from_environment,
    validate_adk_llm_config,
    debug_adk_llm_config,
    AdkLLMConfig,
    AdkProviderConfig,
    AdkModelConfig,
    AdkEnvironmentConfig,
)

from .providers import (
    create_adk_llm_service,
    create_default_adk_llm_service,
    AdkLLMService,
    AdkLLMServiceConfig,
    AdkLLMStreamChunk,
)

from .sessions import (
    create_redis_session_provider,
    create_postgres_session_provider,
    create_in_memory_session_provider,
    AdkSessionProvider,
    AdkSessionConfig,
    AdkRedisSessionConfig,
    AdkPostgresSessionConfig,
    AdkSession,
)

from .errors import (
    AdkError,
    AdkLLMError,
    AdkSessionError,
    AdkConfigError,
    AdkCircuitBreakerError,
    create_adk_error_handler,
    create_circuit_breaker,
    with_adk_retry,
    with_adk_timeout,
)

from .types import (
    AdkAgent,
    AdkMessage,
    AdkTool,
    AdkContext,
    AdkResult,
    AdkSuccess,
    AdkFailure,
    AdkModelType,
    AdkProviderType,
    create_user_message,
    create_assistant_message,
    create_system_message,
    create_adk_context,
)

from .utils import SafeMathEvaluator, safe_calculate

from .security import (
    AdkInputSanitizer,
    sanitize_llm_prompt,
    sanitize_user_input,
    AdkSecurityConfig,
    AdkSecurityValidator,
    validate_api_key,
    validate_session_token,
)

from .types import ImmutableAdkSession, create_immutable_session, add_message_to_session

# Enhanced Capabilities - Schemas and Multi-Agent Coordination
from .schemas import (
    validate_schema,
    validate_string,
    validate_number,
    validate_array,
    validate_object,
    ValidationResult,
    JsonSchema,
)

from .runners import (
    execute_multi_agent,
    select_best_agent,
    merge_parallel_responses,
    extract_delegation_decision,
    execute_with_coordination_rules,
    extract_keywords,
    execute_agent,
    run_agent,
    MultiAgentConfig,
    AgentConfig,
    CoordinationRule,
    DelegationStrategy,
    RunnerCallbacks,
    RunnerConfig,
    LLMControlResult,
    ToolSelectionControlResult,
    ToolExecutionControlResult,
    IterationControlResult,
    IterationCompleteResult,
    SynthesisCheckResult,
    FallbackCheckResult,
)

__all__ = [
    # Config
    "create_adk_llm_config",
    "create_default_adk_llm_config",
    "create_adk_llm_config_from_environment",
    "validate_adk_llm_config",
    "debug_adk_llm_config",
    "AdkLLMConfig",
    "AdkProviderConfig",
    "AdkModelConfig",
    "AdkEnvironmentConfig",
    # Providers
    "create_adk_llm_service",
    "create_default_adk_llm_service",
    "AdkLLMService",
    "AdkLLMServiceConfig",
    "AdkLLMStreamChunk",
    # Sessions
    "create_redis_session_provider",
    "create_postgres_session_provider",
    "create_in_memory_session_provider",
    "AdkSessionProvider",
    "AdkSessionConfig",
    "AdkRedisSessionConfig",
    "AdkPostgresSessionConfig",
    "AdkSession",
    # Errors
    "AdkError",
    "AdkLLMError",
    "AdkSessionError",
    "AdkConfigError",
    "AdkCircuitBreakerError",
    "create_adk_error_handler",
    "create_circuit_breaker",
    "with_adk_retry",
    "with_adk_timeout",
    # Types
    "AdkAgent",
    "AdkMessage",
    "AdkTool",
    "AdkContext",
    "AdkResult",
    "AdkSuccess",
    "AdkFailure",
    "AdkModelType",
    "AdkProviderType",
    "create_user_message",
    "create_assistant_message",
    "create_system_message",
    "create_adk_context",
    # Utils
    "SafeMathEvaluator",
    "safe_calculate",
    # Security
    "AdkInputSanitizer",
    "sanitize_llm_prompt",
    "sanitize_user_input",
    "AdkSecurityConfig",
    "AdkSecurityValidator",
    "validate_api_key",
    "validate_session_token",
    # Immutable Types
    "ImmutableAdkSession",
    "create_immutable_session",
    "add_message_to_session",
    # Enhanced Schema Validation
    "validate_schema",
    "validate_string",
    "validate_number",
    "validate_array",
    "validate_object",
    "ValidationResult",
    "JsonSchema",
    # Intelligent Multi-Agent Coordination
    "execute_multi_agent",
    "select_best_agent",
    "merge_parallel_responses",
    "extract_delegation_decision",
    "execute_with_coordination_rules",
    "extract_keywords",
    "MultiAgentConfig",
    "AgentConfig",
    "CoordinationRule",
    "DelegationStrategy",
    # Advanced Runner with Callback System
    "execute_agent",
    "run_agent",
    "RunnerCallbacks",
    "RunnerConfig",
    "LLMControlResult",
    "ToolSelectionControlResult",
    "ToolExecutionControlResult",
    "IterationControlResult",
    "IterationCompleteResult",
    "SynthesisCheckResult",
    "FallbackCheckResult",
]
