"""Validation for caller-supplied session ids."""

import re

# Long, unguessable ids only: a short id picked by a client can be pre-registered by an attacker
# and later reused after the victim authenticates (session fixation).
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def is_acceptable_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_PATTERN.match(session_id))
