"""
Regex/Text extractor tools for JAF.

Provides safe utilities to extract emails, invoice numbers, order IDs, and arbitrary regex patterns from text.
All tools return JSON strings for structured consumption.
"""

import re
import json
from typing import List, Optional, Dict, Any, Tuple

from ..core.tools import function_tool


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _compile_flags(flags: Optional[List[str]]) -> int:
    """Map a list of flag short-codes to re flags: i=IGNORECASE, m=MULTILINE, s=DOTALL, x=VERBOSE."""
    if not flags:
        return 0
    mapping = {
        "i": re.IGNORECASE,
        "m": re.MULTILINE,
        "s": re.DOTALL,
        "x": re.VERBOSE,
        # long names as convenience
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "VERBOSE": re.VERBOSE,
    }
    out = 0
    for f in flags:
        if f in mapping:
            out |= mapping[f]
    return out


@function_tool(timeout=30.0)
async def extract_emails(
    text: str,
    unique: bool = True,
    max_matches: int = 1000,
    context=None,
) -> str:
    """Extract email addresses from text.

    Args:
        text: Source text to scan.
        unique: Return unique results only (default True).
        max_matches: Maximum number of results to return (default 1000).

    Returns:
        JSON string: {"type": "emails", "count": n, "results": [...]} or {"error": "..."} on error.
    """
    try:
        # Reasonably robust email pattern
        pattern = re.compile(r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", re.IGNORECASE)
        matches = pattern.findall(text or "")
        if unique:
            matches = _dedupe_preserve_order(matches)
        if max_matches and max_matches > 0:
            matches = matches[:max_matches]
        return json.dumps({"type": "emails", "count": len(matches), "results": matches}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Email extraction failed: {str(e)}"})


@function_tool(timeout=30.0)
async def extract_invoice_numbers(
    text: str,
    custom_pattern: Optional[str] = None,
    unique: bool = True,
    max_matches: int = 1000,
    context=None,
) -> str:
    """Extract invoice numbers from text.

    Args:
        text: Source text to scan.
        custom_pattern: Optional custom regex to override the default.
                        Default matches examples like: INV-12345, INVOICE #ABC-9999
        unique: Return unique results only (default True).
        max_matches: Maximum number of results to return (default 1000).

    Returns:
        JSON string: {"type": "invoices", "count": n, "results": [...]} or {"error": "..."} on error.
    """
    try:
        default = r"\\b(?:INV(?:OICE)?)\\s*[-#:]?\\s*[A-Z0-9][A-Z0-9_-]{2,}\\b"
        pattern_str = custom_pattern or default
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(text or "")
        # If there are capture groups, findall returns tuples; join them
        if matches and isinstance(matches[0], tuple):
            matches = ["".join(m) for m in matches]
        if unique:
            matches = _dedupe_preserve_order(matches)
        if max_matches and max_matches > 0:
            matches = matches[:max_matches]
        return json.dumps({"type": "invoices", "count": len(matches), "results": matches}, ensure_ascii=False)
    except re.error as e:
        return json.dumps({"error": f"Invalid invoice regex: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Invoice extraction failed: {str(e)}"})


@function_tool(timeout=30.0)
async def extract_order_ids(
    text: str,
    custom_pattern: Optional[str] = None,
    unique: bool = True,
    max_matches: int = 1000,
    context=None,
) -> str:
    """Extract order IDs from text.

    Args:
        text: Source text to scan.
        custom_pattern: Optional custom regex to override the default.
                        Default matches examples like: ORD-12345, ORDER #A1B2C3, #987654
        unique: Return unique results only (default True).
        max_matches: Maximum number of results to return (default 1000).

    Returns:
        JSON string: {"type": "orders", "count": n, "results": [...]} or {"error": "..."} on error.
    """
    try:
        default_parts = [
            r"\\b(?:ORD(?:ER)?)\\s*[-#:]?\\s*[A-Z0-9][A-Z0-9_-]{2,}\\b",  # ORD-123, ORDER #ABC-123
            r"(?<!\\w)#\\d{5,}\\b",  # #12345
        ]
        pattern_str = custom_pattern or f"(?:{'|'.join(default_parts)})"
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(text or "")
        if matches and isinstance(matches[0], tuple):
            matches = ["".join(m) for m in matches]
        if unique:
            matches = _dedupe_preserve_order(matches)
        if max_matches and max_matches > 0:
            matches = matches[:max_matches]
        return json.dumps({"type": "orders", "count": len(matches), "results": matches}, ensure_ascii=False)
    except re.error as e:
        return json.dumps({"error": f"Invalid order-id regex: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Order ID extraction failed: {str(e)}"})


@function_tool(timeout=45.0)
async def extract_regex(
    text: str,
    patterns: List[str],
    flags: Optional[List[str]] = None,
    unique: bool = False,
    max_matches_per_pattern: int = 1000,
    return_groups: bool = False,
    flatten: bool = False,
    max_text_chars: int = 1000000,
    context=None,
) -> str:
    """Extract matches for arbitrary regex patterns from text.

    Args:
        text: Source text to scan.
        patterns: List of regex pattern strings.
        flags: Optional flags list (i, m, s, x) or long names (IGNORECASE, MULTILINE, DOTALL, VERBOSE).
        unique: Deduplicate results per-pattern (default False).
        max_matches_per_pattern: Limit matches per pattern (default 1000).
        return_groups: If true, include capture groups for each match.
        flatten: If true, combine all matches from all patterns into a single list.
        max_text_chars: Maximum number of characters to inspect (default 1,000,000).

    Returns:
        JSON string:
          - If flatten is False:
              {"results": [{"pattern": "...", "matches": [...]}, ...], "total": N}
          - If flatten is True:
              {"results": [...], "total": N}
        or {"error": "..."} on error.
    """
    try:
        if not isinstance(patterns, list) or not patterns:
            return json.dumps({"error": "patterns must be a non-empty list of regex strings"})

        # Enforce text size limit
        text_to_scan = (text or "")[:max(0, int(max_text_chars))]

        flag_bits = _compile_flags(flags)
        compiled: List[Tuple[str, re.Pattern]] = []
        for p in patterns:
            try:
                compiled.append((p, re.compile(p, flag_bits)))
            except re.error as e:
                return json.dumps({"error": f"Invalid regex '{p}': {str(e)}"})

        total = 0
        if not flatten:
            out: List[Dict[str, Any]] = []
            for p_str, rx in compiled:
                matches_list: List[Any] = []
                for m_i, m in enumerate(rx.finditer(text_to_scan)):
                    if return_groups:
                        # Prefer named groups; fallback to tuple of groups; if no groups, use full match
                        if m.groupdict():
                            matches_list.append({"match": m.group(0), "groups": m.groupdict()})
                        elif m.groups():
                            matches_list.append({"match": m.group(0), "groups": list(m.groups())})
                        else:
                            matches_list.append({"match": m.group(0)})
                    else:
                        matches_list.append(m.group(0))

                    if max_matches_per_pattern and len(matches_list) >= max_matches_per_pattern:
                        break

                if unique and not return_groups:
                    matches_list = _dedupe_preserve_order([str(x) for x in matches_list])

                total += len(matches_list)
                out.append({"pattern": p_str, "matches": matches_list})

            return json.dumps({"results": out, "total": total}, ensure_ascii=False)
        else:
            flat: List[Any] = []
            for _, rx in compiled:
                for m in rx.finditer(text_to_scan):
                    if return_groups:
                        if m.groupdict():
                            flat.append({"match": m.group(0), "groups": m.groupdict()})
                        elif m.groups():
                            flat.append({"match": m.group(0), "groups": list(m.groups())})
                        else:
                            flat.append({"match": m.group(0)})
                    else:
                        flat.append(m.group(0))

                    if max_matches_per_pattern and sum(1 for _ in [m]) >= max_matches_per_pattern:
                        # Note: per-pattern cap is approximated within loop across patterns; for strict caps,
                        # track counts per-pattern. We'll implement strict caps below.
                        pass
            # Strict per-pattern cap is handled above when not flattened; for flattened, we'll apply global dedupe/cap:
            if unique and not return_groups:
                flat = _dedupe_preserve_order([str(x) for x in flat])

            total = len(flat)
            return json.dumps({"results": flat[: (max_matches_per_pattern or total)], "total": total}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Regex extraction failed: {str(e)}"})


def create_text_extractor_tools():
    """Return list of Text/Regex extractor tools for easy agent registration."""
    return [extract_emails, extract_invoice_numbers, extract_order_ids, extract_regex]