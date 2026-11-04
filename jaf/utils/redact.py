"""Helpers for keeping secrets out of error text that leaves the process (events, tool output).

Only strings that clearly look like credentials are masked, so error messages stay readable for
the model and for developers (field names, validation messages, status codes are untouched).
"""

import re

_SECRET_IN_TEXT = re.compile(
    r"(://)[^/\s:@]+:[^/\s@]+@"  # user:password@host in a URL
    r"|\b(?:sk|rk)-[A-Za-z0-9_\-]{16,}"  # OpenAI/Anthropic style secret keys
    r"|(Bearer\s+)[A-Za-z0-9._\-]{8,}",  # Authorization header values
    re.IGNORECASE,
)


def redact_error_text(text: str, max_length: int = 1000) -> str:
    """Mask URL passwords, secret-looking keys and bearer tokens, and cap the length."""
    def _mask(match: "re.Match[str]") -> str:
        if match.group(1):  # keep the URL shape: ://[redacted]@host
            return "://[redacted]@"
        return (match.group(2) or "") + "[redacted]"

    redacted = _SECRET_IN_TEXT.sub(_mask, text)
    return redacted[:max_length]
