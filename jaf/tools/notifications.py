"""
Email / Notification tools for JAF.

Capabilities:
- send_email: Send email via context-provided client or SMTP (env/args).
- send_slack_message: Send Slack notifications via Incoming Webhook URL.
- send_webhook: Send generic HTTP webhook notifications (POST/PUT/PATCH/GET).

All tools return JSON strings for structured consumption.
Environment variables supported:
  SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_USE_TLS, SMTP_USE_SSL, SMTP_FROM
  SLACK_WEBHOOK_URL
"""

import os
import ssl
import json
import re
from typing import Any, Dict, Optional

from ..core.tools import function_tool


def _get_env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _is_http_url(url: str) -> bool:
    if not url or len(url) > 4096:
        return False
    return bool(re.match(r"^https?://", url.strip(), flags=re.IGNORECASE))


async def _send_email_via_context(context: Any, to: str, subject: str, body: str, from_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Try to locate a mailer/email client on context and send:
    - context.email_client.send(to, subject, body, from_email=?)
    - context.mailer.send(...)
    - context.mailer.send_email(...)
    """
    if context is None:
        raise RuntimeError("No context available for context-based email")
    for attr in ("email_client", "mailer"):
        client = getattr(context, attr, None)
        if client is None:
            continue
        for method_name in ("send_email", "send"):
            method = getattr(client, method_name, None)
            if callable(method):
                res = method(to, subject, body, from_email=from_email) if "from_email" in getattr(method, "__code__", ().__dir__) else method(to, subject, body)
                if hasattr(res, "__await__"):
                    res = await res  # type: ignore[func-returns-value]
                return {"provider": "context", "to": to, "subject": subject, "status": "sent", "result": str(res)}
    raise RuntimeError("No compatible email sender found on context (email_client/mailer)")


def _send_email_via_smtp(
    to: str,
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: Optional[bool] = True,
    use_ssl: Optional[bool] = None,
    from_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send email via SMTP using Python smtplib. Synchronous; intended to be called in a thread or quickly.
    """
    import smtplib
    from email.mime.text import MIMEText

    if not from_email:
        from_email = username or "no-reply@example.com"

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            if username and password:
                server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            if username and password:
                server.login(username, password)
            server.send_message(msg)

    return {"provider": "smtp", "to": to, "subject": subject, "status": "sent"}


@function_tool(timeout=30.0)
async def send_email(
    to: str,
    subject: str,
    body: str,
    use_context_mailer: bool = True,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_username: Optional[str] = None,
    smtp_password: Optional[str] = None,
    smtp_use_tls: Optional[bool] = None,
    smtp_use_ssl: Optional[bool] = None,
    from_email: Optional[str] = None,
    context=None,
) -> str:
    """Send an email using context-provided client or SMTP.

    Args:
        to: Recipient email
        subject: Subject line
        body: Body text (UTF-8)
        use_context_mailer: Attempt to use context.mailer/email_client first (default True)
        smtp_*: SMTP settings (fallback to env if omitted)
        from_email: Optional 'From' address (defaults to SMTP_USERNAME or no-reply)

    Env fallbacks:
        SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_USE_TLS, SMTP_USE_SSL, SMTP_FROM

    Returns:
        JSON: {"type":"email","provider":"context|smtp","status":"sent",...} or {"error":"..."}
    """
    try:
        # 1) Try context mailer
        if use_context_mailer:
            try:
                sent = await _send_email_via_context(context, to, subject, body, from_email)
                return json.dumps({"type": "email", **sent}, ensure_ascii=False)
            except Exception:
                # fallback to SMTP
                pass

        # 2) SMTP
        host = smtp_host or os.getenv("SMTP_HOST")
        port_val = smtp_port if smtp_port is not None else (int(os.getenv("SMTP_PORT")) if os.getenv("SMTP_PORT") else None)
        user = smtp_username or os.getenv("SMTP_USERNAME")
        pwd = smtp_password or os.getenv("SMTP_PASSWORD")
        tls = smtp_use_tls if smtp_use_tls is not None else _get_env_bool("SMTP_USE_TLS", True)
        ssl_on = smtp_use_ssl if smtp_use_ssl is not None else _get_env_bool("SMTP_USE_SSL", False)
        from_addr = from_email or os.getenv("SMTP_FROM") or user

        if not host or not port_val:
            return json.dumps({"error": "SMTP configuration missing: SMTP_HOST/SMTP_PORT"})

        sent = _send_email_via_smtp(
            to=to,
            subject=subject,
            body=body,
            smtp_host=host,
            smtp_port=int(port_val),
            username=user,
            password=pwd,
            use_tls=tls,
            use_ssl=ssl_on,
            from_email=from_addr,
        )
        return json.dumps({"type": "email", **sent}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Email send failed: {str(e)}"})


@function_tool(timeout=15.0)
async def send_slack_message(
    text: str,
    webhook_url: Optional[str] = None,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    channel: Optional[str] = None,
    context=None,
) -> str:
    """Send a Slack notification via Incoming Webhook URL.

    Args:
        text: Message text
        webhook_url: Slack Incoming Webhook URL (falls back to env SLACK_WEBHOOK_URL)
        username: Optional bot username override
        icon_emoji: Optional emoji (e.g., ":robot_face:")
        channel: Optional channel override (if webhook allows it)

    Returns:
        JSON: {"type":"slack","status":"sent"} or {"error":"..."}
    """
    try:
        url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not url or not _is_http_url(url):
            return json.dumps({"error": "Missing or invalid Slack webhook URL (provide arg or SLACK_WEBHOOK_URL)"})

        import httpx
        payload: Dict[str, Any] = {"text": text}
        if username:
            payload["username"] = username
        if icon_emoji:
            payload["icon_emoji"] = icon_emoji
        if channel:
            payload["channel"] = channel

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            ok = 200 <= r.status_code < 300
            if not ok:
                return json.dumps({"error": f"Slack webhook failed with status {r.status_code}: {r.text[:200]}"} )
        return json.dumps({"type": "slack", "status": "sent"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Slack send failed: {str(e)}"})


@function_tool(timeout=20.0)
async def send_webhook(
    url: str,
    method: str = "POST",
    headers_json: Optional[str] = None,
    body_json: Optional[str] = None,
    timeout: float = 10.0,
    context=None,
) -> str:
    """Send a generic HTTP webhook.

    Args:
        url: Target HTTP/HTTPS URL
        method: HTTP method (POST, PUT, PATCH, GET)
        headers_json: Optional JSON map of headers
        body_json: Optional JSON string body (for GET, body is ignored)
        timeout: Client timeout in seconds

    Returns:
        JSON: {"type":"webhook","url":"...","status":...,"response":"..."} or {"error":"..."}
    """
    try:
        if not _is_http_url(url):
            return json.dumps({"error": "Only http/https URLs allowed for webhook"})

        headers = {}
        if headers_json:
            try:
                h = json.loads(headers_json)
                if not isinstance(h, dict):
                    return json.dumps({"error": "headers_json must be a JSON object"})
                headers = {str(k): str(v) for k, v in h.items()}
            except Exception as e:
                return json.dumps({"error": f"Invalid headers_json: {str(e)}"})

        data = None
        if body_json and method.upper() in ("POST", "PUT", "PATCH"):
            try:
                data = json.loads(body_json)
            except Exception as e:
                return json.dumps({"error": f"Invalid body_json: {str(e)}"})

        import httpx
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            m = method.upper()
            if m == "GET":
                r = await client.get(url)
            elif m == "POST":
                r = await client.post(url, json=data)
            elif m == "PUT":
                r = await client.put(url, json=data)
            elif m == "PATCH":
                r = await client.patch(url, json=data)
            else:
                return json.dumps({"error": f"Unsupported method '{method}'. Use GET/POST/PUT/PATCH."})

            content_type = r.headers.get("content-type", "")
            # Safe short response preview
            preview = ""
            try:
                preview = r.text[:500]
            except Exception:
                try:
                    preview = r.content[:500].decode("utf-8", errors="ignore")
                except Exception:
                    preview = ""

            return json.dumps({
                "type": "webhook",
                "url": url,
                "status": r.status_code,
                "content_type": content_type,
                "response": preview
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Webhook send failed: {str(e)}"})


def create_notification_tools():
    """Return list of Notification tools for easy agent registration."""
    return [send_email, send_slack_message, send_webhook]