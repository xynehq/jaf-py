"""
Web Search tools for JAF.

Capabilities:
- web_search: Query the web for fresh information using a pluggable strategy:
  1) If context provides a search client (context.web_search/search_client/retriever with .search/.query), use it.
  2) If provider='tavily' and API key is available (arg or env TAVILY_API_KEY), use Tavily API.
  3) If provider='serpapi' and API key is available (arg or env SERPAPI_API_KEY), use SerpAPI (Google).
  4) Fallback to DuckDuckGo Instant Answer API (limited results).

- fetch_url: Fetch a web page (http/https) and return status, content-type, and truncated text content.

All tools return JSON strings for structured consumption.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

from ..core.tools import function_tool


def _get_context_search(context: Any) -> Optional[Any]:
    """Return a context-provided web search client if available."""
    if context is None:
        return None
    for attr in ("web_search", "search_client", "searcher", "retriever"):
        if hasattr(context, attr):
            return getattr(context, attr)
    return None


def _normalize_result_item(
    title: Optional[str],
    url: Optional[str],
    snippet: Optional[str],
    score: Optional[float] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "title": title or "",
        "url": url or "",
        "snippet": snippet or "",
        **({"score": score} if score is not None else {}),
        **({"source": source} if source is not None else {}),
    }


async def _search_via_context(
    query: str,
    max_results: int,
    context_client: Any
) -> List[Dict[str, Any]]:
    """Try standard methods on a context-provided search client."""
    # Common method names
    for method_name, kwargs in [
        ("search", {"max_results": max_results}),
        ("query", {"max_results": max_results}),
        ("search", {"k": max_results}),
        ("query", {"k": max_results}),
        ("search", {}),
        ("query", {}),
    ]:
        if hasattr(context_client, method_name):
            try:
                method = getattr(context_client, method_name)
                res = method(query, **kwargs) if kwargs else method(query)
                # Support async/sync
                if hasattr(res, "__await__"):
                    res = await res
                # Normalize
                items: List[Dict[str, Any]] = []
                if isinstance(res, dict) and "results" in res and isinstance(res["results"], list):
                    iterable = res["results"]
                else:
                    iterable = res if isinstance(res, list) else [res]

                for it in iterable[:max_results]:
                    # Try common shapes
                    title = None
                    url = None
                    snippet = None
                    score = None
                    if isinstance(it, dict):
                        title = it.get("title") or it.get("name") or it.get("headline")
                        url = it.get("url") or it.get("link")
                        snippet = it.get("snippet") or it.get("content") or it.get("description")
                        s = it.get("score") or it.get("similarity")
                        try:
                            score = float(s) if s is not None else None
                        except Exception:
                            score = None
                    else:
                        # Object with attributes?
                        title = getattr(it, "title", None) or getattr(it, "name", None)
                        url = getattr(it, "url", None) or getattr(it, "link", None)
                        snippet = getattr(it, "snippet", None) or getattr(it, "content", None) or getattr(it, "description", None)
                        s = getattr(it, "score", None) or getattr(it, "similarity", None)
                        try:
                            score = float(s) if s is not None else None
                        except Exception:
                            score = None
                    items.append(_normalize_result_item(title, url, snippet, score, source="context"))
                return items
            except Exception:
                pass
    raise RuntimeError("Context search client has no compatible 'search' or 'query' method")


async def _search_tavily(
    query: str,
    max_results: int,
    api_key: str,
    include_raw_content: bool = False,
    timeout: float = 10.0
) -> List[Dict[str, Any]]:
    import httpx
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": include_raw_content,
        "search_depth": "advanced",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        items: List[Dict[str, Any]] = []
        for it in results[:max_results]:
            title = it.get("title")
            url = it.get("url")
            snippet = it.get("content")
            items.append(_normalize_result_item(title, url, snippet, source="tavily"))
        return items


async def _search_serpapi(
    query: str,
    max_results: int,
    api_key: str,
    timeout: float = 10.0
) -> List[Dict[str, Any]]:
    import httpx
    params = {
        "engine": "google",
        "q": query,
        "num": max(1, min(max_results, 10)),
        "api_key": api_key,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get("https://serpapi.com/search.json", params=params)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results", []) or []
        items: List[Dict[str, Any]] = []
        for it in organic[:max_results]:
            title = it.get("title")
            url = it.get("link")
            snippet = it.get("snippet")
            items.append(_normalize_result_item(title, url, snippet, source="serpapi"))
        return items


async def _search_duckduckgo(
    query: str,
    max_results: int,
    lang: Optional[str] = None,
    timeout: float = 8.0
) -> List[Dict[str, Any]]:
    import httpx
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
    }
    if lang:
        params["kl"] = lang  # DDG region/language hint, e.g., 'us-en'
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "JAF-WebSearch/1.0"}) as client:
        r = await client.get("https://api.duckduckgo.com/", params=params)
        r.raise_for_status()
        data = r.json()
        results = []
        # The instant answer API doesn't return typical web results, but RelatedTopics and Abstract
        if data.get("AbstractURL"):
            results.append(_normalize_result_item(
                data.get("Heading"), data.get("AbstractURL"), data.get("AbstractText"), source="duckduckgo"
            ))
        for rt in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(rt, dict):
                # Nested topics can occur
                if "Topics" in rt and isinstance(rt["Topics"], list):
                    for sub in rt["Topics"]:
                        if "FirstURL" in sub or "Text" in sub:
                            results.append(_normalize_result_item(sub.get("Text"), sub.get("FirstURL"), sub.get("Text"), source="duckduckgo"))
                            if len(results) >= max_results:
                                break
                else:
                    if "FirstURL" in rt or "Text" in rt:
                        results.append(_normalize_result_item(rt.get("Text"), rt.get("FirstURL"), rt.get("Text"), source="duckduckgo"))
            if len(results) >= max_results:
                break
        return results[:max_results]


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v and str(v).strip() else None


@function_tool(timeout=30.0)
async def web_search(
    query: str,
    provider: str = "auto",
    max_results: int = 5,
    lang: Optional[str] = None,
    api_key: Optional[str] = None,
    include_raw_content: bool = False,
    context=None,
) -> str:
    """Search the web for fresh information.

    Strategy (in order):
    1) Use context-provided search client if available (context.web_search/search_client/retriever) via .search/.query
    2) Provider 'tavily' if API key available (arg or env TAVILY_API_KEY)
    3) Provider 'serpapi' if API key available (arg or env SERPAPI_API_KEY)
    4) Fallback to duckduckgo instant answer API (limited results)

    Args:
        query: Search query
        provider: 'auto' (default), 'tavily', 'serpapi', or 'duckduckgo'
        max_results: Maximum results to return (default 5)
        lang: Optional language/region hint (e.g., 'us-en') for duckduckgo fallback
        api_key: Optional API key override for specific providers
        include_raw_content: If supported by provider (e.g., tavily), include raw content

    Returns:
        JSON: {"type":"web_search","provider":"...","query":"...","results":[{title,url,snippet,score?,source?}, ...]}
    """
    try:
        # 1) Context-provided client
        client = _get_context_search(context)
        if client is not None and (provider == "auto" or provider == "context"):
            try:
                items = await _search_via_context(query, max_results, client)
                return json.dumps({"type": "web_search", "provider": "context", "query": query, "results": items}, ensure_ascii=False)
            except Exception:
                # Fallback to next providers
                pass

        # 2) Tavily
        if provider in ("auto", "tavily"):
            key = api_key or _get_env("TAVILY_API_KEY")
            if key:
                try:
                    items = await _search_tavily(query, max_results, key, include_raw_content=include_raw_content)
                    return json.dumps({"type": "web_search", "provider": "tavily", "query": query, "results": items}, ensure_ascii=False)
                except Exception as e:
                    if provider == "tavily":
                        return json.dumps({"error": f"Tavily search failed: {str(e)}"})

        # 3) SerpAPI (Google)
        if provider in ("auto", "serpapi"):
            key = api_key or _get_env("SERPAPI_API_KEY")
            if key:
                try:
                    items = await _search_serpapi(query, max_results, key)
                    return json.dumps({"type": "web_search", "provider": "serpapi", "query": query, "results": items}, ensure_ascii=False)
                except Exception as e:
                    if provider == "serpapi":
                        return json.dumps({"error": f"SerpAPI search failed: {str(e)}"})

        # 4) DuckDuckGo fallback
        try:
            items = await _search_duckduckgo(query, max_results, lang=lang)
            return json.dumps({"type": "web_search", "provider": "duckduckgo", "query": query, "results": items}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"DuckDuckGo search failed: {str(e)}"})

    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}"})


def _is_http_url(url: str) -> bool:
    if not url or len(url) > 2000:
        return False
    return bool(re.match(r"^https?://", url.strip(), flags=re.IGNORECASE))


@function_tool(timeout=20.0)
async def fetch_url(
    url: str,
    max_bytes: int = 500000,
    timeout: float = 10.0,
    context=None,
) -> str:
    """Fetch a URL and return status, content-type, and (truncated) text content.

    Args:
        url: HTTP/HTTPS URL to fetch
        max_bytes: Maximum number of bytes of response content to include (default 500 KB)
        timeout: HTTP client timeout in seconds

    Returns:
        JSON: {"type":"fetch_url","url":"...","status":...,"content_type":"...","text":"..."} or {"error":"..."}
    """
    try:
        if not _is_http_url(url):
            return json.dumps({"error": "Only http/https URLs are allowed and must be under 2000 characters"})

        import httpx
        headers = {"User-Agent": "JAF-WebFetch/1.0"}
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            r = await client.get(url)
            content_type = r.headers.get("content-type", "")
            text = ""
            # Best-effort decode with apparent encoding
            try:
                r.encoding = r.encoding or "utf-8"
                text = r.text[:max_bytes]
            except Exception:
                # Fallback to raw bytes truncated and decoded ignoring errors
                text = r.content[:max_bytes].decode("utf-8", errors="ignore")

            return json.dumps({
                "type": "fetch_url",
                "url": url,
                "status": r.status_code,
                "content_type": content_type,
                "text": text
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Fetch failed: {str(e)}"})


def create_web_search_tools():
    """Return list of Web Search tools for easy agent registration."""
    return [web_search, fetch_url]