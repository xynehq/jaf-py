"""
Data Cleaning tools for JAF.

Capabilities:
- normalize_email: Normalize/validate email addresses (lowercasing, optional +tag removal).
- normalize_phone: Normalize phone numbers to international +E.164 style with basic heuristics.
- normalize_csv_rows: Normalize common fields across CSV-like rows (trim/collapse whitespace,
  lowercase emails, standardize booleans, strip non-digits on specified fields).

All tools return JSON strings for structured consumption.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from ..core.tools import function_tool
from adk.security.sanitization import sanitize_email as _sanitize_email


def _lowercase_email(email: str) -> str:
    return email.strip().lower()


def _remove_plus_tag(email: str) -> str:
    """
    Remove +tag from local part: local+tag@example.com -> local@example.com
    """
    email = email.strip()
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if "+" in local:
        local = local.split("+", 1)[0]
    return f"{local}@{domain}"


@function_tool(timeout=10.0)
async def normalize_email(
    email: str,
    lowercase: bool = True,
    remove_plus_tag: bool = False,
    context=None,
) -> str:
    """Normalize and validate an email address.

    Args:
        email: Input email address
        lowercase: Lowercase the entire email (default True)
        remove_plus_tag: Remove '+tag' from the local part (e.g., name+tag@x.com -> name@x.com)

    Returns:
        JSON: {"type":"email","input":"...","normalized":"..."} or {"error":"..."}
    """
    try:
        e = (email or "").strip()
        if not e:
            return json.dumps({"error": "Empty email"})
        # Basic sanitization/validation
        try:
            e = _sanitize_email(e)
        except Exception as se:
            return json.dumps({"error": f"Invalid email: {str(se)}"})
        if lowercase:
            e = _lowercase_email(e)
        if remove_plus_tag:
            e = _remove_plus_tag(e)
        return json.dumps({"type": "email", "input": email, "normalized": e}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Email normalization failed: {str(e)}"})


# ----------------------------
# Phone normalization (basic heuristics)
# ----------------------------

# Minimal country calling code mapping for common regions
_COUNTRY_CALLING_CODES: Dict[str, str] = {
    # North America
    "US": "1", "CA": "1",
    # Europe
    "GB": "44", "DE": "49", "FR": "33", "ES": "34", "IT": "39", "NL": "31",
    # APAC
    "IN": "91", "SG": "65", "JP": "81", "CN": "86", "AU": "61",
    # LATAM
    "BR": "55", "MX": "52",
    # Africa
    "ZA": "27",
}

_DIGITS_RE = re.compile(r"\d+")


def _strip_to_digits(s: str) -> str:
    return "".join(re.findall(r"\d", s))


def _normalize_to_e164(phone: str, country_code: Optional[str]) -> Optional[str]:
    """
    Normalize to +E.164 style with simple heuristics:
    - Accepts leading '+' or '00' international prefix
    - If no '+' and country_code provided (ISO 2-letter), prepend calling code
    - Validates length between 8 and 15 digits total (excluding '+')
    Returns normalized string (e.g., '+14155552671') or None if cannot normalize.
    """
    p = (phone or "").strip()
    if not p:
        return None

    # Convert leading 00 to +, or keep + if present
    if p.startswith("00"):
        p = "+" + p[2:]
    # If starts with +, keep only digits after it
    if p.startswith("+"):
        digits = _strip_to_digits(p)
        # p like +<digits> becomes '+' + digits (digits already excludes '+')
        norm = "+" + digits
    else:
        # No explicit +, keep digits and apply country calling code if provided
        digits = _strip_to_digits(p)
        if not digits:
            return None
        if country_code:
            cc = _COUNTRY_CALLING_CODES.get(country_code.strip().upper())
            if not cc:
                # Unknown country; cannot infer safely
                return None
            norm = f"+{cc}{digits}"
        else:
            # No country provided; cannot infer safely
            return None

    # Validate plausible E.164 length (min 8, max 15 digits)
    just_digits = _strip_to_digits(norm)
    if not (8 <= len(just_digits) <= 15):
        return None

    return norm


@function_tool(timeout=10.0)
async def normalize_phone(
    phone: str,
    country_code: Optional[str] = None,
    context=None,
) -> str:
    """Normalize a phone number into +E.164 style with basic heuristics.

    Args:
        phone: Phone number string (may include spaces, hyphens, parentheses)
        country_code: Optional ISO 2-letter country code to infer calling code when number lacks '+'

    Returns:
        JSON: {"type":"phone","input":"...","normalized":"...","country_code":"..."} or {"error":"..."}
    """
    try:
        normalized = _normalize_to_e164(phone, country_code)
        if not normalized:
            hint = " Provide country_code (ISO alpha-2) if the number lacks an international prefix."
            return json.dumps({"error": ("Unable to normalize phone number." + hint).strip()})
        return json.dumps({
            "type": "phone",
            "input": phone,
            "normalized": normalized,
            "country_code": country_code
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Phone normalization failed: {str(e)}"})


# ----------------------------
# CSV rows normalization
# ----------------------------

_EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)

_BOOLEAN_TRUE = {"true", "t", "1", "yes", "y", "on"}
_BOOLEAN_FALSE = {"false", "f", "0", "no", "n", "off"}


def _standardize_boolean(v: str) -> Optional[bool]:
    s = v.strip().lower()
    if s in _BOOLEAN_TRUE:
        return True
    if s in _BOOLEAN_FALSE:
        return False
    return None


def _normalize_field_value(
    field_name: str,
    value: Any,
    trim_whitespace: bool,
    collapse_spaces: bool,
    lowercase_emails: bool,
    standardize_booleans: bool,
    strip_non_digits_fields: Optional[set],
) -> Any:
    # Non-string types are returned as-is
    if not isinstance(value, str):
        return value

    v = value
    if trim_whitespace:
        v = v.strip()
    if collapse_spaces:
        v = re.sub(r"\s+", " ", v)

    # Email handling
    if lowercase_emails:
        is_email_field = "email" in (field_name or "").lower()
        looks_like_email = bool(_EMAIL_RE.match(v))
        if is_email_field or looks_like_email:
            v = v.lower()

    # Booleans
    if standardize_booleans:
        b = _standardize_boolean(v)
        if b is not None:
            return b

    # Strip non-digits for specified fields
    if strip_non_digits_fields and field_name in strip_non_digits_fields:
        v = "".join(ch for ch in v if ch.isdigit() or (ch == "+" and v.startswith("+")))

    return v


def _as_dict_rows(rows: Union[List[Dict[str, Any]], List[List[Any]]], headers: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Ensure we have list of dict rows; if given list-of-lists, headers must be provided."""
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return rows  # type: ignore[return-value]
    if not headers:
        raise ValueError("headers must be provided when rows are a list of lists")
    out: List[Dict[str, Any]] = []
    for r in rows:  # type: ignore[assignment]
        d = {}
        for i, h in enumerate(headers):
            d[h] = r[i] if i < len(r) else None
        out.append(d)
    return out


@function_tool(timeout=30.0)
async def normalize_csv_rows(
    rows_json: str,
    headers: Optional[List[str]] = None,
    trim_whitespace: bool = True,
    collapse_spaces: bool = True,
    lowercase_emails: bool = True,
    standardize_booleans: bool = True,
    strip_non_digits_fields_json: Optional[str] = None,
    context=None,
) -> str:
    """Normalize common issues in CSV-like rows.

    Args:
        rows_json: JSON string: either List[Dict[str, Any]] or List[List[Any]]
        headers: Required when rows_json is list-of-lists (defines field names)
        trim_whitespace: Trim leading/trailing whitespace in string fields (default True)
        collapse_spaces: Collapse internal whitespace sequences to single space (default True)
        lowercase_emails: Lowercase email fields or values that look like emails (default True)
        standardize_booleans: Convert yes/no/true/false/1/0 into booleans (default True)
        strip_non_digits_fields_json: Optional JSON list of field names to strip non-digits (e.g., ["phone","zip"])

    Returns:
        JSON: {"type":"csv_rows","count": n, "rows": [...]} or {"error":"..."}
    """
    try:
        # Parse input rows
        try:
            rows = json.loads(rows_json)
            if not isinstance(rows, list):
                return json.dumps({"error": "rows_json must be a JSON list"})
        except Exception as e:
            return json.dumps({"error": f"Invalid rows_json: {str(e)}"})

        # Parse strip fields
        strip_set: Optional[set] = None
        if strip_non_digits_fields_json:
            try:
                fields = json.loads(strip_non_digits_fields_json)
                if not isinstance(fields, list):
                    return json.dumps({"error": "strip_non_digits_fields_json must be a JSON list"})
                strip_set = {str(f) for f in fields}
            except Exception as e:
                return json.dumps({"error": f"Invalid strip_non_digits_fields_json: {str(e)}"})

        dict_rows = _as_dict_rows(rows, headers)

        normalized_rows: List[Dict[str, Any]] = []
        for row in dict_rows:
            new_row: Dict[str, Any] = {}
            for k, v in row.items():
                new_row[k] = _normalize_field_value(
                    field_name=str(k),
                    value=v,
                    trim_whitespace=trim_whitespace,
                    collapse_spaces=collapse_spaces,
                    lowercase_emails=lowercase_emails,
                    standardize_booleans=standardize_booleans,
                    strip_non_digits_fields=strip_set
                )
            normalized_rows.append(new_row)

        return json.dumps({"type": "csv_rows", "count": len(normalized_rows), "rows": normalized_rows}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"CSV rows normalization failed: {str(e)}"})


def create_data_cleaning_tools():
    """Return list of Data Cleaning tools for easy agent registration."""
    return [normalize_email, normalize_phone, normalize_csv_rows]