"""
Example demonstrating proxy support in JAF agents.

This example shows how to configure proxy settings for both LLM providers
and A2A client communications.
"""

import asyncio
import os
from jaf.core.proxy import ProxyConfig, ProxyAuth
from jaf.providers.model import make_litellm_provider
from jaf.a2a.client import create_a2a_client, send_message


async def main():
    """Demonstrate proxy configuration usage."""

    print("=== JAF Proxy Support Demo ===\n")

    # Example 1: Creating proxy config from environment variables
    print("1. Using proxy configuration from environment:")
    env_proxy = ProxyConfig.from_environment()
    print(f"   HTTP Proxy: {env_proxy.http_proxy}")
    print(f"   HTTPS Proxy: {env_proxy.https_proxy}")
    print(f"   No Proxy: {env_proxy.no_proxy}")
    print(f"   Has Auth: {env_proxy.auth is not None}")
    print()

    # Example 2: Creating proxy config manually
    print("2. Creating proxy configuration manually:")
    manual_proxy = ProxyConfig(
        http_proxy="http://proxy.company.com:8080",
        https_proxy="http://proxy.company.com:8080",
        no_proxy="localhost,127.0.0.1,*.local",
        auth=ProxyAuth(username="user", password="pass"),
    )
    print(f"   HTTP Proxy: {manual_proxy.http_proxy}")
    print(f"   HTTPS Proxy: {manual_proxy.https_proxy}")
    print(f"   No Proxy: {manual_proxy.no_proxy}")
    print(f"   Auth User: {manual_proxy.auth.username if manual_proxy.auth else None}")
    print()

    # Example 3: Converting to HTTP client formats
    print("3. Converting proxy config for HTTP clients:")
    httpx_proxies = manual_proxy.to_httpx_proxies()
    print(f"   HTTPX format: {httpx_proxies}")
    openai_proxies = manual_proxy.to_openai_proxies()
    print(f"   OpenAI format: {openai_proxies}")
    print()

    # Example 4: Testing proxy bypass
    print("4. Testing proxy bypass logic:")
    test_hosts = ["localhost", "internal.company.com", "api.openai.com", "127.0.0.1"]
    for host in test_hosts:
        should_bypass = manual_proxy.should_bypass_proxy(host)
        print(f"   {host}: {'bypass proxy' if should_bypass else 'use proxy'}")
    print()

    # Example 5: Using proxy with LiteLLM provider
    print("5. Creating LiteLLM provider with proxy (demo only - won't make actual requests):")
    try:
        proxy_config = ProxyConfig.from_url("http://proxy.example.com:8080")
        llm_provider = make_litellm_provider(
            base_url="http://localhost:4000", api_key="test-key", proxy_config=proxy_config
        )
        print("   ✓ LiteLLM provider created successfully with proxy support")
    except Exception as e:
        print(f"   ✗ Error creating LiteLLM provider: {e}")
    print()

    # Example 6: Using proxy with A2A client
    print("6. Creating A2A client with proxy (demo only - won't make actual requests):")
    try:
        proxy_config = ProxyConfig.from_url("http://proxy.example.com:8080")
        a2a_client = create_a2a_client(base_url="http://localhost:8000", proxy_config=proxy_config)
        print("   ✓ A2A client created successfully with proxy support")
        print(f"   Client session ID: {a2a_client.session_id}")
        print(f"   Base URL: {a2a_client.config.base_url}")
        print(f"   Has proxy config: {a2a_client.config.proxy_config is not None}")
    except Exception as e:
        print(f"   ✗ Error creating A2A client: {e}")
    print()

    # Example 7: Environment variable configuration
    print("7. Environment variables for proxy configuration:")
    env_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "PROXY_USERNAME",
        "PROXY_PASSWORD",
    ]

    print("   Set these environment variables to configure proxy:")
    for var in env_vars:
        value = os.getenv(var, "not set")
        print(f"   {var}: {value}")
    print()

    print("=== Demo Complete ===")
    print("\nTo use proxy support in your JAF applications:")
    print("1. Set environment variables (HTTP_PROXY, HTTPS_PROXY, etc.)")
    print("2. Or create ProxyConfig manually and pass to providers/clients")
    print("3. The proxy config will be automatically used for all HTTP requests")


if __name__ == "__main__":
    asyncio.run(main())
