"""
ADK LLM Configuration - Production-Ready LLM Configuration System

This module provides comprehensive LLM configuration following functional patterns,
supporting multiple providers with environment variable integration and validation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from ..types import AdkModelType, AdkProviderType

# ========== Configuration Types ==========


@dataclass
class AdkLLMConfig:
    """ADK LLM configuration with production features."""

    provider: AdkProviderType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[Union[AdkModelType, str]] = None
    timeout: int = 30000  # milliseconds
    retries: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    streaming: bool = True


@dataclass
class AdkModelConfig:
    """Model configuration with capabilities and limits."""

    name: str
    display_name: str
    context_window: int
    max_tokens: int
    supports_function_calling: bool
    supports_streaming: bool
    cost_per_1k_tokens: Optional[Dict[str, float]] = None


@dataclass
class AdkRateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    concurrent: Optional[int] = None


@dataclass
class AdkProviderFeatures:
    """Provider feature support."""

    streaming: bool = True
    function_calling: bool = True
    multimodal: bool = False
    vision: bool = False
    json_mode: bool = False


@dataclass
class AdkProviderConfig:
    """Provider configuration with models and features."""

    name: str
    base_url: str
    api_key: str
    models: List[AdkModelConfig]
    rate_limits: Optional[AdkRateLimitConfig] = None
    features: Optional[AdkProviderFeatures] = None


@dataclass
class AdkEnvironmentConfig:
    """Environment configuration variables."""

    litellm_url: Optional[str] = None
    litellm_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    default_provider: Optional[str] = None
    default_model: Optional[str] = None


# ========== Default Provider Configurations ==========

DEFAULT_PROVIDER_CONFIGS: Dict[str, AdkProviderConfig] = {
    "litellm": AdkProviderConfig(
        name="LiteLLM",
        base_url="http://localhost:4000",
        api_key="anything",
        models=[
            AdkModelConfig(
                name="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.005, "output": 0.015},
            ),
            AdkModelConfig(
                name="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.01, "output": 0.03},
            ),
            AdkModelConfig(
                name="claude-3-sonnet",
                display_name="Claude 3 Sonnet",
                context_window=200000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.003, "output": 0.015},
            ),
            AdkModelConfig(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                context_window=1048576,
                max_tokens=8192,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.001, "output": 0.003},
            ),
        ],
        features=AdkProviderFeatures(
            streaming=True, function_calling=True, multimodal=True, vision=True, json_mode=True
        ),
    ),
    "openai": AdkProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key="",
        models=[
            AdkModelConfig(
                name="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.005, "output": 0.015},
            ),
            AdkModelConfig(
                name="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.01, "output": 0.03},
            ),
            AdkModelConfig(
                name="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                context_window=16385,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.0005, "output": 0.0015},
            ),
        ],
        rate_limits=AdkRateLimitConfig(
            requests_per_minute=3500, tokens_per_minute=90000, concurrent=20
        ),
        features=AdkProviderFeatures(
            streaming=True, function_calling=True, multimodal=True, vision=True, json_mode=True
        ),
    ),
    "anthropic": AdkProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        api_key="",
        models=[
            AdkModelConfig(
                name="claude-3-sonnet",
                display_name="Claude 3 Sonnet",
                context_window=200000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.003, "output": 0.015},
            ),
            AdkModelConfig(
                name="claude-3-haiku",
                display_name="Claude 3 Haiku",
                context_window=200000,
                max_tokens=4096,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.00025, "output": 0.00125},
            ),
        ],
        rate_limits=AdkRateLimitConfig(
            requests_per_minute=1000, tokens_per_minute=40000, concurrent=5
        ),
        features=AdkProviderFeatures(
            streaming=True, function_calling=True, multimodal=True, vision=True, json_mode=False
        ),
    ),
    "google": AdkProviderConfig(
        name="Google AI",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="",
        models=[
            AdkModelConfig(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                context_window=1048576,  # 1M tokens
                max_tokens=8192,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.001, "output": 0.003},
            ),
            AdkModelConfig(
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                context_window=1048576,
                max_tokens=8192,
                supports_function_calling=True,
                supports_streaming=True,
                cost_per_1k_tokens={"input": 0.0001, "output": 0.0003},
            ),
        ],
        rate_limits=AdkRateLimitConfig(
            requests_per_minute=60, tokens_per_minute=32000, concurrent=2
        ),
        features=AdkProviderFeatures(
            streaming=True, function_calling=True, multimodal=True, vision=True, json_mode=False
        ),
    ),
}

# ========== Configuration Factory Functions ==========


def create_adk_llm_config(provider: Union[str, AdkProviderType]) -> AdkLLMConfig:
    """
    Create an ADK LLM configuration for the specified provider.

    Args:
        provider: Provider type ('openai', 'anthropic', 'google', 'litellm')

    Returns:
        AdkLLMConfig with default settings for the provider
    """
    if isinstance(provider, str):
        provider = AdkProviderType(provider)

    provider_config = DEFAULT_PROVIDER_CONFIGS.get(provider.value)
    if not provider_config:
        raise ValueError(f"Unknown provider: {provider}")

    return AdkLLMConfig(
        provider=provider,
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        default_model=provider_config.models[0].name if provider_config.models else None,
        timeout=30000,
        retries=3,
        streaming=True,
    )


def create_default_adk_llm_config() -> AdkLLMConfig:
    """Create a default ADK LLM configuration using LiteLLM."""
    return create_adk_llm_config(AdkProviderType.LITELLM)


def load_environment_config() -> AdkEnvironmentConfig:
    """Load configuration from environment variables."""
    return AdkEnvironmentConfig(
        litellm_url=os.getenv("LITELLM_URL"),
        litellm_api_key=os.getenv("LITELLM_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        azure_api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        default_provider=os.getenv("LLM_PROVIDER"),
        default_model=os.getenv("LLM_MODEL"),
    )


def create_adk_llm_config_from_environment() -> AdkLLMConfig:
    """
    Create ADK LLM configuration from environment variables.

    Returns:
        AdkLLMConfig configured from environment variables
    """
    env_config = load_environment_config()

    # Determine provider from environment
    provider = AdkProviderType.LITELLM  # Default

    if env_config.default_provider:
        try:
            provider = AdkProviderType(env_config.default_provider.lower())
        except ValueError:
            print(f"[ADK:CONFIG] Unknown provider '{env_config.default_provider}', using litellm")

    elif env_config.openai_api_key:
        provider = AdkProviderType.OPENAI
    elif env_config.anthropic_api_key:
        provider = AdkProviderType.ANTHROPIC
    elif env_config.google_api_key:
        provider = AdkProviderType.GOOGLE

    # Create base config
    config = create_adk_llm_config(provider)

    # Override with environment values
    if provider == AdkProviderType.OPENAI and env_config.openai_api_key:
        config.api_key = env_config.openai_api_key
    elif provider == AdkProviderType.ANTHROPIC and env_config.anthropic_api_key:
        config.api_key = env_config.anthropic_api_key
    elif provider == AdkProviderType.GOOGLE and env_config.google_api_key:
        config.api_key = env_config.google_api_key
    elif provider == AdkProviderType.LITELLM:
        if env_config.litellm_url:
            config.base_url = env_config.litellm_url
        if env_config.litellm_api_key:
            config.api_key = env_config.litellm_api_key

    if env_config.default_model:
        config.default_model = env_config.default_model

    return config


# ========== Validation Functions ==========


def validate_adk_llm_config(config: AdkLLMConfig) -> List[str]:
    """
    Validate an ADK LLM configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages
    """
    errors = []

    # Provider validation
    if config.provider not in [p.value for p in AdkProviderType]:
        errors.append(f"Invalid provider: {config.provider}")

    # API key validation for direct providers
    if config.provider in [
        AdkProviderType.OPENAI,
        AdkProviderType.ANTHROPIC,
        AdkProviderType.GOOGLE,
    ]:
        if not config.api_key or config.api_key == "":
            errors.append(f"API key required for {config.provider} provider")

    # URL validation
    if config.base_url and not config.base_url.startswith(("http://", "https://")):
        errors.append(f"Base URL must start with http:// or https://")

    # Timeout validation
    if config.timeout <= 0:
        errors.append("Timeout must be positive")

    # Retries validation
    if config.retries < 0:
        errors.append("Retries must be non-negative")

    # Temperature validation
    if config.temperature is not None and (config.temperature < 0 or config.temperature > 2):
        errors.append("Temperature must be between 0 and 2")

    # Max tokens validation
    if config.max_tokens is not None and config.max_tokens <= 0:
        errors.append("Max tokens must be positive")

    return errors


def debug_adk_llm_config(config: AdkLLMConfig) -> None:
    """Debug print an ADK LLM configuration."""
    print("[ADK:CONFIG] LLM Configuration:")
    print(f"  Provider: {config.provider}")
    print(f"  Base URL: {config.base_url}")
    print(f"  API Key: {'*' * len(config.api_key) if config.api_key else 'None'}")
    print(f"  Default Model: {config.default_model}")
    print(f"  Timeout: {config.timeout}ms")
    print(f"  Retries: {config.retries}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Streaming: {config.streaming}")

    # Validate and show issues
    errors = validate_adk_llm_config(config)
    if errors:
        print(f"[ADK:CONFIG] Configuration issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("[ADK:CONFIG] Configuration is valid")


# ========== Model Utility Functions ==========


def get_models_for_provider(provider: str) -> List[AdkModelConfig]:
    """Get available models for a provider."""
    provider_config = DEFAULT_PROVIDER_CONFIGS.get(provider, None)
    return provider_config.models if provider_config else []


def get_all_available_models() -> Dict[str, List[AdkModelConfig]]:
    """Get all available models grouped by provider."""
    return {provider: config.models for provider, config in DEFAULT_PROVIDER_CONFIGS.items()}


def get_provider_for_model(model_name: str) -> Optional[str]:
    """Get the provider that supports a specific model."""
    for provider, config in DEFAULT_PROVIDER_CONFIGS.items():
        if any(model.name == model_name for model in config.models):
            return provider
    return None


def get_model_config(model_name: str) -> Optional[AdkModelConfig]:
    """Get configuration for a specific model."""
    for config in DEFAULT_PROVIDER_CONFIGS.values():
        for model in config.models:
            if model.name == model_name:
                return model
    return None


def map_adk_model_to_provider_model(model: Optional[Union[AdkModelType, str]]) -> str:
    """Map ADK model enum to provider model string."""
    if model is None:
        return "gpt-4o"  # Default

    if isinstance(model, AdkModelType):
        return model.value

    return str(model)
