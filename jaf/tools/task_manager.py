"""
Task Manager tools for JAF.

Capabilities:
- Jira:
  - create_jira_issue: Create an issue in Jira Cloud/Server
  - update_jira_issue: Update fields on an existing Jira issue
- Trello:
  - create_trello_card: Create a card in a Trello list
  - update_trello_card: Update fields on an existing Trello card

These tools prefer a context-provided client when available (jira/trello/task client), and otherwise
use direct HTTP APIs with credentials provided via arguments or environment variables.

Environment variables supported:
  Jira:
    JIRA_BASE_URL, JIRA_EMAIL (or JIRA_USERNAME), JIRA_API_TOKEN
  Trello:
    TRELLO_KEY, TRELLO_TOKEN

All tools return JSON strings with clear "error" messages on failure.
"""

import os
import json
from typing import Any, Dict, Optional

from ..core.tools import function_tool


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v and str(v).strip() else None


# ----------------------------
# Context helpers
# ----------------------------

def _get_context_jira(context: Any) -> Optional[Any]:
    if context is None:
        return None
    # Common attributes for Jira client
    for attr in ("jira", "jira_client", "task_client", "issue_client"):
        if hasattr(context, attr):
            return getattr(context, attr)
    return None


def _get_context_trello(context: Any) -> Optional[Any]:
    if context is None:
        return None
    # Common attributes for Trello client
    for attr in ("trello", "trello_client", "board_client", "task_client"):
        if hasattr(context, attr):
            return getattr(context, attr)
    return None


# ----------------------------
# Jira
# ----------------------------

@function_tool(timeout=30.0)
async def create_jira_issue(
    summary: str,
    description: Optional[str] = None,
    project_key: Optional[str] = None,
    issue_type: str = "Task",
    base_url: Optional[str] = None,
    email_or_username: Optional[str] = None,
    api_token: Optional[str] = None,
    fields_json: Optional[str] = None,
    context=None,
) -> str:
    """Create a Jira issue.

    Strategy:
    1) Try context-provided Jira client:
       context.jira|jira_client|task_client with a method 'create_issue(fields=...)' or similar
    2) Fallback to Jira REST API (POST /rest/api/3/issue for Cloud or /rest/api/2/issue for Server)
       Requires base_url, email/username, api_token (Jira Cloud uses email + API token)

    Args:
        summary: Issue summary (title)
        description: Issue description (optional)
        project_key: Jira project key (e.g., "ENG")
        issue_type: Issue type name (default "Task")
        base_url: Jira base URL (e.g., "https://your-org.atlassian.net")
        email_or_username: Jira account email/username (Cloud typically uses email)
        api_token: Jira API token
        fields_json: Optional JSON string for additional/override fields (merged into fields)
    """
    try:
        if not summary or not project_key:
            return json.dumps({"error": "summary and project_key are required"})

        # Assemble fields
        fields: Dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description:
            fields["description"] = description

        if fields_json:
            try:
                extra = json.loads(fields_json)
                if not isinstance(extra, dict):
                    return json.dumps({"error": "fields_json must be a JSON object"})
                # Shallow merge (extra overrides base)
                fields.update(extra)
            except Exception as e:
                return json.dumps({"error": f"Invalid fields_json: {str(e)}"})

        # 1) Context Jira client
        client = _get_context_jira(context)
        if client is not None:
            for method_name in ("create_issue", "create"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        res = method(fields=fields)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        # Normalize response
                        out = {"type": "jira_create", "fields": fields}
                        if isinstance(res, dict):
                            out["result"] = res
                        else:
                            # Try common attributes
                            key = getattr(res, "key", None)
                            iid = getattr(res, "id", None)
                            out["result"] = {"key": key, "id": iid, "repr": str(res)}
                        return json.dumps(out, ensure_ascii=False)
                    except Exception as e:
                        # Fall back to REST if context client fails
                        pass

        # 2) Direct REST API
        b = base_url or _get_env("JIRA_BASE_URL")
        user = email_or_username or _get_env("JIRA_EMAIL") or _get_env("JIRA_USERNAME")
        token = api_token or _get_env("JIRA_API_TOKEN")
        if not b or not user or not token:
            return json.dumps({"error": "Missing Jira credentials: base_url, email/username, api_token"})

        # Jira Cloud typically uses v3; server might be v2. We'll try v3 first, then v2 fallback.
        import httpx
        payload = {"fields": fields}

        async with httpx.AsyncClient(timeout=15.0, auth=(user, token)) as client_http:
            # Try API v3
            url_v3 = f"{b.rstrip('/')}/rest/api/3/issue"
            r = await client_http.post(url_v3, json=payload)
            if r.status_code in (200, 201):
                data = r.json()
                return json.dumps({"type": "jira_create", "fields": fields, "result": data}, ensure_ascii=False)

            # Try API v2 fallback
            url_v2 = f"{b.rstrip('/')}/rest/api/2/issue"
            r2 = await client_http.post(url_v2, json=payload)
            if r2.status_code in (200, 201):
                data = r2.json()
                return json.dumps({"type": "jira_create", "fields": fields, "result": data}, ensure_ascii=False)

            return json.dumps({"error": f"Jira create failed: {r.status_code} {r.text[:500]} | {r2.status_code} {r2.text[:500]}"})

    except Exception as e:
        return json.dumps({"error": f"Jira create failed: {str(e)}"})


@function_tool(timeout=30.0)
async def update_jira_issue(
    issue_key: str,
    fields_json: str,
    base_url: Optional[str] = None,
    email_or_username: Optional[str] = None,
    api_token: Optional[str] = None,
    context=None,
) -> str:
    """Update fields on a Jira issue.

    Strategy:
    1) Try context-provided Jira client with method 'update_issue(issue_key, fields=...)' or 'update(...)'
    2) Fallback to Jira REST API (PUT /rest/api/3/issue/{issueIdOrKey} with {"fields": {...}})

    Args:
        issue_key: Jira issue key (e.g., "ENG-123")
        fields_json: JSON object with fields to update
    """
    try:
        if not issue_key:
            return json.dumps({"error": "issue_key is required"})
        try:
            fields = json.loads(fields_json)
            if not isinstance(fields, dict):
                return json.dumps({"error": "fields_json must be a JSON object"})
        except Exception as e:
            return json.dumps({"error": f"Invalid fields_json: {str(e)}"})

        # 1) Context client
        client = _get_context_jira(context)
        if client is not None:
            for method_name in ("update_issue", "update"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        res = method(issue_key, fields=fields)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        return json.dumps({"type": "jira_update", "issue": issue_key, "fields": fields, "result": str(res)}, ensure_ascii=False)
                    except Exception as e:
                        # fallback to REST
                        pass

        # 2) REST API
        b = base_url or _get_env("JIRA_BASE_URL")
        user = email_or_username or _get_env("JIRA_EMAIL") or _get_env("JIRA_USERNAME")
        token = api_token or _get_env("JIRA_API_TOKEN")
        if not b or not user or not token:
            return json.dumps({"error": "Missing Jira credentials: base_url, email/username, api_token"})

        import httpx
        payload = {"fields": fields}
        async with httpx.AsyncClient(timeout=15.0, auth=(user, token)) as client_http:
            url_v3 = f"{b.rstrip('/')}/rest/api/3/issue/{issue_key}"
            r = await client_http.put(url_v3, json=payload)
            if r.status_code in (200, 204):
                return json.dumps({"type": "jira_update", "issue": issue_key, "fields": fields, "result": "OK"}, ensure_ascii=False)
            url_v2 = f"{b.rstrip('/')}/rest/api/2/issue/{issue_key}"
            r2 = await client_http.put(url_v2, json=payload)
            if r2.status_code in (200, 204):
                return json.dumps({"type": "jira_update", "issue": issue_key, "fields": fields, "result": "OK"}, ensure_ascii=False)
            return json.dumps({"error": f"Jira update failed: {r.status_code} {r.text[:500]} | {r2.status_code} {r2.text[:500]}"})
    except Exception as e:
        return json.dumps({"error": f"Jira update failed: {str(e)}"})


# ----------------------------
# Trello
# ----------------------------

@function_tool(timeout=20.0)
async def create_trello_card(
    name: str,
    list_id: str,
    desc: Optional[str] = None,
    labels_csv: Optional[str] = None,
    due: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    fields_json: Optional[str] = None,
    context=None,
) -> str:
    """Create a Trello card via API or context client.

    Strategy:
    1) Try context-provided Trello client with method 'create_card' or 'add_card'
    2) Fallback to Trello REST API: POST https://api.trello.com/1/cards
       Required: key, token, idList; Optional: name, desc, labels, due
    """
    try:
        if not name or not list_id:
            return json.dumps({"error": "name and list_id are required"})

        # 1) Context client
        client = _get_context_trello(context)
        if client is not None:
            for method_name in ("create_card", "add_card", "create"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        kwargs = {"name": name, "list_id": list_id, "desc": desc, "labels_csv": labels_csv, "due": due}
                        # merge any fields_json extras
                        if fields_json:
                            extra = json.loads(fields_json)
                            if isinstance(extra, dict):
                                kwargs.update(extra)
                        res = method(**{k: v for k, v in kwargs.items() if v is not None})
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        return json.dumps({"type": "trello_create", "result": str(res)}, ensure_ascii=False)
                    except Exception:
                        # fallback to REST
                        pass

        # 2) REST API
        key = api_key or _get_env("TRELLO_KEY")
        tok = token or _get_env("TRELLO_TOKEN")
        if not key or not tok:
            return json.dumps({"error": "Missing Trello credentials: api_key/token or TRELLO_KEY/TRELLO_TOKEN"})

        import httpx
        url = "https://api.trello.com/1/cards"
        params = {
            "key": key,
            "token": tok,
            "idList": list_id,
            "name": name,
        }
        if desc:
            params["desc"] = desc
        if labels_csv:
            params["idLabels"] = labels_csv
        if due:
            params["due"] = due

        # Accept additional fields (e.g., pos, idMembers) via fields_json
        if fields_json:
            try:
                extra = json.loads(fields_json)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        params[str(k)] = v
            except Exception:
                pass

        async with httpx.AsyncClient(timeout=15.0) as client_http:
            r = await client_http.post(url, params=params)
            if 200 <= r.status_code < 300:
                return json.dumps({"type": "trello_create", "result": r.json()}, ensure_ascii=False)
            return json.dumps({"error": f"Trello create failed: {r.status_code} {r.text[:500]}"})
    except Exception as e:
        return json.dumps({"error": f"Trello create failed: {str(e)}"})


@function_tool(timeout=20.0)
async def update_trello_card(
    card_id: str,
    fields_json: str,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    context=None,
) -> str:
    """Update a Trello card.

    Strategy:
    1) Try context-provided Trello client with method 'update_card' or 'update'
    2) Fallback to Trello REST API: PUT https://api.trello.com/1/cards/{id}
       Requires key and token
    """
    try:
        if not card_id:
            return json.dumps({"error": "card_id is required"})
        try:
            fields = json.loads(fields_json)
            if not isinstance(fields, dict):
                return json.dumps({"error": "fields_json must be a JSON object"})
        except Exception as e:
            return json.dumps({"error": f"Invalid fields_json: {str(e)}"})

        # 1) Context client
        client = _get_context_trello(context)
        if client is not None:
            for method_name in ("update_card", "update"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        res = method(card_id, **fields)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        return json.dumps({"type": "trello_update", "card": card_id, "fields": fields, "result": str(res)}, ensure_ascii=False)
                    except Exception:
                        # fallback to REST
                        pass

        # 2) REST API
        key = api_key or _get_env("TRELLO_KEY")
        tok = token or _get_env("TRELLO_TOKEN")
        if not key or not tok:
            return json.dumps({"error": "Missing Trello credentials: api_key/token or TRELLO_KEY/TRELLO_TOKEN"})

        import httpx
        url = f"https://api.trello.com/1/cards/{card_id}"
        params = {
            "key": key,
            "token": tok,
        }
        # Trello updates via query params or JSON body; we'll use query params for simplicity
        for k, v in fields.items():
            params[str(k)] = v

        async with httpx.AsyncClient(timeout=15.0) as client_http:
            r = await client_http.put(url, params=params)
            if 200 <= r.status_code < 300:
                return json.dumps({"type": "trello_update", "card": card_id, "fields": fields, "result": r.json()}, ensure_ascii=False)
            return json.dumps({"error": f"Trello update failed: {r.status_code} {r.text[:500]}"})
    except Exception as e:
        return json.dumps({"error": f"Trello update failed: {str(e)}"})


def create_task_manager_tools():
    """Return list of Task Manager tools for easy agent registration."""
    return [create_jira_issue, update_jira_issue, create_trello_card, update_trello_card]