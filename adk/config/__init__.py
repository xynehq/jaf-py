"""
ADK Configuration System - Production-Ready Configuration

This module provides a flexible configuration system for the ADK layer,
supporting environment variables, provider-specific settings, and validation.
"""

from .llm_config import (
    create_adk_llm_config,
    create_default_adk_llm_config,
    create_adk_llm_config_from_environment,
    validate_adk_llm_config,
    debug_adk_llm_config,
    load_environment_config,
    get_model_config,
    get_models_for_provider,
    get_all_available_models,
    get_provider_for_model,
    map_adk_model_to_provider_model,
    AdkLLMConfig,
    AdkProviderConfig,
    AdkModelConfig,
    AdkRateLimitConfig,
    AdkProviderFeatures,
    AdkEnvironmentConfig,
    DEFAULT_PROVIDER_CONFIGS,
)

# Import types from parent types module
from ..types import AdkProviderType

__all__ = [
    "create_adk_llm_config",
    "create_default_adk_llm_config",
    "create_adk_llm_config_from_environment",
    "validate_adk_llm_config",
    "debug_adk_llm_config",
    "load_environment_config",
    "get_model_config",
    "get_models_for_provider",
    "get_all_available_models",
    "get_provider_for_model",
    "map_adk_model_to_provider_model",
    "AdkLLMConfig",
    "AdkProviderConfig",
    "AdkModelConfig",
    "AdkRateLimitConfig",
    "AdkProviderFeatures",
    "AdkEnvironmentConfig",
    "DEFAULT_PROVIDER_CONFIGS",
    "AdkProviderType",
]
