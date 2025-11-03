"""
Helper functions for creating JAF core components with proxy support.

This module provides convenience functions to easily create model providers
with proxy configuration.
"""

from typing import Optional, Dict, Any
from .proxy import ProxyConfig, get_default_proxy_config
from ..providers.model import make_litellm_provider


def make_proxy_enabled_litellm_provider(
    base_url: str,
    api_key: str = "anything",
    default_timeout: Optional[float] = None,
    proxy_url: Optional[str] = None,
    proxy_username: Optional[str] = None,
    proxy_password: Optional[str] = None,
    use_env_proxy: bool = True,
):
    """
    Create a LiteLLM provider with proxy support.

    Args:
        base_url: Base URL for the LiteLLM server
        api_key: API key (defaults to "anything" for local servers)
        default_timeout: Default timeout for model API calls in seconds
        proxy_url: Proxy URL (if not using environment variables)
        proxy_username: Proxy username for authentication
        proxy_password: Proxy password for authentication
        use_env_proxy: Whether to use proxy settings from environment variables

    Returns:
        ModelProvider with proxy configuration
    """
    proxy_config = None

    if proxy_url:
        # Use explicitly provided proxy settings
        from .proxy import ProxyAuth, create_proxy_config

        proxy_config = create_proxy_config(
            proxy_url=proxy_url, username=proxy_username, password=proxy_password
        )
    elif use_env_proxy:
        # Use environment-based proxy settings
        proxy_config = get_default_proxy_config()
        # Only use proxy if actually configured in environment
        if not proxy_config.http_proxy and not proxy_config.https_proxy:
            proxy_config = None

    return make_litellm_provider(
        base_url=base_url,
        api_key=api_key,
        default_timeout=default_timeout,
        proxy_config=proxy_config,
    )


def get_proxy_info() -> Dict[str, Any]:
    """
    Get information about current proxy configuration from environment.

    Returns:
        Dictionary with proxy configuration details
    """
    proxy_config = get_default_proxy_config()

    return {
        "http_proxy": proxy_config.http_proxy,
        "https_proxy": proxy_config.https_proxy,
        "no_proxy": proxy_config.no_proxy,
        "has_auth": proxy_config.auth is not None,
        "auth_username": proxy_config.auth.username if proxy_config.auth else None,
        "is_configured": bool(proxy_config.http_proxy or proxy_config.https_proxy),
    }


def validate_proxy_config(proxy_config: ProxyConfig) -> Dict[str, Any]:
    """
    Validate proxy configuration and return validation results.

    Args:
        proxy_config: Proxy configuration to validate

    Returns:
        Dictionary with validation results
    """
    results = {"valid": True, "warnings": [], "errors": []}

    # Check if at least one proxy is configured
    if not proxy_config.http_proxy and not proxy_config.https_proxy:
        results["warnings"].append("No proxy URLs configured")

    # Validate proxy URLs if configured
    for proxy_type, proxy_url in [
        ("HTTP", proxy_config.http_proxy),
        ("HTTPS", proxy_config.https_proxy),
    ]:
        if proxy_url:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(proxy_url)
                if not parsed.scheme:
                    results["errors"].append(f"{proxy_type} proxy URL missing scheme: {proxy_url}")
                    results["valid"] = False
                if not parsed.netloc:
                    results["errors"].append(f"{proxy_type} proxy URL missing host: {proxy_url}")
                    results["valid"] = False
            except Exception as e:
                results["errors"].append(f"Invalid {proxy_type} proxy URL: {e}")
                results["valid"] = False

    # Check authentication consistency
    if proxy_config.auth:
        if not proxy_config.auth.username:
            results["warnings"].append("Proxy authentication configured but username is empty")
        if not proxy_config.auth.password:
            results["warnings"].append("Proxy authentication configured but password is empty")

    return results
