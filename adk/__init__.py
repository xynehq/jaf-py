"""
JAF Agent Development Kit (ADK) - Production-Ready Layer

The ADK provides a production-ready abstraction layer on top of the JAF Core,
featuring real LLM integration, production-grade session providers, and
comprehensive error handling.

This layer transforms the JAF framework from a sophisticated mock-up into
a production-ready system with actual database storage and real LLM providers.
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
    AdkEnvironmentConfig
)

from .providers import (
    create_adk_llm_service,
    create_default_adk_llm_service,
    AdkLLMService,
    AdkLLMServiceConfig,
    AdkLLMStreamChunk
)

from .sessions import (
    create_redis_session_provider,
    create_postgres_session_provider,
    create_in_memory_session_provider,
    AdkSessionProvider,
    AdkSessionConfig,
    AdkRedisSessionConfig,
    AdkPostgresSessionConfig,
    AdkSession
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
    with_adk_timeout
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
    create_adk_context
)

from .utils import (
    SafeMathEvaluator,
    safe_calculate
)

from .security import (
    AdkInputSanitizer,
    sanitize_llm_prompt,
    sanitize_user_input,
    AdkSecurityConfig,
    AdkSecurityValidator,
    validate_api_key,
    validate_session_token
)

from .types import (
    ImmutableAdkSession,
    create_immutable_session,
    add_message_to_session
)

__all__ = [
    # Config
    'create_adk_llm_config',
    'create_default_adk_llm_config', 
    'create_adk_llm_config_from_environment',
    'validate_adk_llm_config',
    'debug_adk_llm_config',
    'AdkLLMConfig',
    'AdkProviderConfig',
    'AdkModelConfig',
    'AdkEnvironmentConfig',
    
    # Providers
    'create_adk_llm_service',
    'create_default_adk_llm_service',
    'AdkLLMService',
    'AdkLLMServiceConfig',
    'AdkLLMStreamChunk',
    
    # Sessions
    'create_redis_session_provider',
    'create_postgres_session_provider',
    'create_in_memory_session_provider',
    'AdkSessionProvider',
    'AdkSessionConfig',
    'AdkRedisSessionConfig',
    'AdkPostgresSessionConfig',
    'AdkSession',
    
    # Errors
    'AdkError',
    'AdkLLMError',
    'AdkSessionError',
    'AdkConfigError',
    'AdkCircuitBreakerError',
    'create_adk_error_handler',
    'create_circuit_breaker',
    'with_adk_retry',
    'with_adk_timeout',
    
    # Types
    'AdkAgent',
    'AdkMessage',
    'AdkTool',
    'AdkContext',
    'AdkResult',
    'AdkSuccess',
    'AdkFailure',
    'AdkModelType',
    'AdkProviderType',
    'create_user_message',
    'create_assistant_message',
    'create_system_message',
    'create_adk_context',
    
    # Utils
    'SafeMathEvaluator',
    'safe_calculate',
    
    # Security
    'AdkInputSanitizer',
    'sanitize_llm_prompt',
    'sanitize_user_input',
    'AdkSecurityConfig',
    'AdkSecurityValidator',
    'validate_api_key',
    'validate_session_token',
    
    # Immutable Types
    'ImmutableAdkSession',
    'create_immutable_session',
    'add_message_to_session'
]