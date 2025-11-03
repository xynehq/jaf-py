"""
Simple proxy configuration test for JAF.

This script tests basic proxy configuration without making actual LLM calls.
Use this to verify proxy settings are loaded correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Add the jaf package to the path
sys.path.insert(0, "/Users/yash.gupta/check/jaf-py")

from jaf.core import ProxyConfig, ProxyAuth, get_default_proxy_config
from jaf.core.proxy_helpers import get_proxy_info, validate_proxy_config


def main():
    """Test proxy configuration loading and validation."""
    print("=== JAF Proxy Configuration Test ===\n")

    # Load .env if it exists
    if os.path.exists(".env"):
        load_dotenv(".env")
        print("✓ Loaded .env file\n")
    else:
        print("ℹ No .env file found, using system environment\n")

    # Test 1: Get proxy info
    print("1. Current proxy configuration:")
    proxy_info = get_proxy_info()
    for key, value in proxy_info.items():
        print(f"   {key}: {value}")
    print()

    # Test 2: Get default proxy config
    print("2. Default proxy config from environment:")
    proxy_config = get_default_proxy_config()
    print(f"   HTTP Proxy: {proxy_config.http_proxy}")
    print(f"   HTTPS Proxy: {proxy_config.https_proxy}")
    print(f"   No Proxy: {proxy_config.no_proxy}")
    print(f"   Has Auth: {proxy_config.auth is not None}")
    if proxy_config.auth:
        print(f"   Auth Username: {proxy_config.auth.username}")
        print(
            f"   Auth Password: {'*' * len(proxy_config.auth.password) if proxy_config.auth.password else 'None'}"
        )
    print()

    # Test 3: Validate proxy config
    print("3. Proxy configuration validation:")
    validation = validate_proxy_config(proxy_config)
    print(f"   Valid: {validation['valid']}")
    if validation["warnings"]:
        print("   Warnings:")
        for warning in validation["warnings"]:
            print(f"     - {warning}")
    if validation["errors"]:
        print("   Errors:")
        for error in validation["errors"]:
            print(f"     - {error}")
    print()

    # Test 4: HTTP client format conversion
    print("4. HTTP client format conversion:")
    if proxy_config.http_proxy or proxy_config.https_proxy:
        httpx_proxies = proxy_config.to_httpx_proxies()
        openai_proxies = proxy_config.to_openai_proxies()
        print(f"   HTTPX format: {httpx_proxies}")
        print(f"   OpenAI format: {openai_proxies}")
    else:
        print("   No proxy configured for conversion")
    print()

    # Test 5: Proxy bypass logic
    print("5. Proxy bypass test:")
    test_hosts = ["localhost", "127.0.0.1", "api.openai.com", "internal.local"]
    for host in test_hosts:
        bypass = proxy_config.should_bypass_proxy(host)
        status = "BYPASS" if bypass else "USE PROXY"
        print(f"   {host}: {status}")
    print()

    # Test 6: Manual proxy creation
    print("6. Manual proxy configuration test:")
    try:
        manual_proxy = ProxyConfig(
            http_proxy="http://test-proxy.example.com:8080",
            https_proxy="http://test-proxy.example.com:8080",
            no_proxy="localhost,*.local",
            auth=ProxyAuth(username="testuser", password="testpass"),
        )
        manual_validation = validate_proxy_config(manual_proxy)
        print(f"   Manual config valid: {manual_validation['valid']}")
        print(f"   Manual config HTTPX: {manual_proxy.to_httpx_proxies()}")
    except Exception as e:
        print(f"   Error creating manual config: {e}")
    print()

    print("=== Configuration Summary ===")
    if proxy_info["is_configured"]:
        print("✓ Proxy is configured and ready to use")
        print(f"  HTTP: {proxy_info['http_proxy']}")
        print(f"  HTTPS: {proxy_info['https_proxy']}")
        print(f"  Auth: {'Yes' if proxy_info['has_auth'] else 'No'}")
    else:
        print("ℹ No proxy configured (will use direct connections)")

    print("\nTo configure proxy, set environment variables or create .env file:")
    print("  HTTP_PROXY=http://proxy.example.com:8080")
    print("  HTTPS_PROXY=http://proxy.example.com:8080")
    print("  PROXY_USERNAME=username (optional)")
    print("  PROXY_PASSWORD=password (optional)")
    print("  NO_PROXY=localhost,127.0.0.1,*.local (optional)")


if __name__ == "__main__":
    main()
