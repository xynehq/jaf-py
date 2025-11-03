"""
Proxy configuration for JAF agents and HTTP clients.

This module provides unified proxy configuration that can be used across
different HTTP clients (httpx, OpenAI, etc.) in the JAF framework.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse


@dataclass
class ProxyAuth:
    """Proxy authentication configuration."""

    username: str
    password: str


@dataclass
class ProxyConfig:
    """Proxy configuration for HTTP clients."""

    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    auth: Optional[ProxyAuth] = None

    @classmethod
    def from_environment(cls) -> "ProxyConfig":
        """Create proxy configuration from environment variables."""
        return cls(
            http_proxy=os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
            https_proxy=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
            no_proxy=os.getenv("NO_PROXY") or os.getenv("no_proxy"),
            auth=ProxyAuth(
                username=os.getenv("PROXY_USERNAME", ""), password=os.getenv("PROXY_PASSWORD", "")
            )
            if os.getenv("PROXY_USERNAME")
            else None,
        )

    @classmethod
    def from_url(cls, proxy_url: str, auth: Optional[ProxyAuth] = None) -> "ProxyConfig":
        """Create proxy configuration from a single URL."""
        return cls(http_proxy=proxy_url, https_proxy=proxy_url, auth=auth)

    def to_httpx_proxies(self) -> Optional[Dict[str, str]]:
        """Convert to httpx proxies format."""
        if not self.http_proxy and not self.https_proxy:
            return None

        proxies = {}

        if self.http_proxy:
            proxies["http://"] = self._add_auth_to_url(self.http_proxy)

        if self.https_proxy:
            proxies["https://"] = self._add_auth_to_url(self.https_proxy)

        return proxies if proxies else None

    def to_openai_proxies(self) -> Optional[Dict[str, str]]:
        """Convert to OpenAI client proxies format."""
        # OpenAI client supports httpx-style proxy configuration
        return self.to_httpx_proxies()

    def _add_auth_to_url(self, url: str) -> str:
        """Add authentication to proxy URL if configured."""
        if not self.auth or not self.auth.username:
            return url

        parsed = urlparse(url)

        # If URL already has auth, don't override
        if "@" in parsed.netloc:
            return url

        auth_string = f"{self.auth.username}:{self.auth.password}"

        # Reconstruct URL with auth
        if parsed.port:
            netloc = f"{auth_string}@{parsed.hostname}:{parsed.port}"
        else:
            netloc = f"{auth_string}@{parsed.hostname}"

        return f"{parsed.scheme}://{netloc}{parsed.path}"

    def should_bypass_proxy(self, host: str) -> bool:
        """Check if a host should bypass the proxy based on no_proxy settings."""
        if not self.no_proxy:
            return False

        no_proxy_hosts = [h.strip() for h in self.no_proxy.split(",")]

        for no_proxy_host in no_proxy_hosts:
            if not no_proxy_host:
                continue

            # Exact match
            if host == no_proxy_host:
                return True

            # Wildcard match (e.g., *.example.com)
            if no_proxy_host.startswith("*"):
                suffix = no_proxy_host[1:]
                if host.endswith(suffix):
                    return True

            # Domain suffix match
            if no_proxy_host.startswith("."):
                if host.endswith(no_proxy_host) or host == no_proxy_host[1:]:
                    return True

        return False


def create_proxy_config(
    proxy_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    no_proxy: Optional[str] = None,
) -> ProxyConfig:
    """Create a proxy configuration with optional parameters."""
    auth = ProxyAuth(username, password) if username else None

    if proxy_url:
        config = ProxyConfig.from_url(proxy_url, auth)
        if no_proxy:
            config.no_proxy = no_proxy
        return config

    return ProxyConfig.from_environment()


def get_default_proxy_config() -> ProxyConfig:
    """Get the default proxy configuration from environment variables."""
    return ProxyConfig.from_environment()
