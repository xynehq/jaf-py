"""
Date & Time tools for JAF.

Capabilities:
- Get current date/time (timezone-aware, multiple output formats)
- Parse date/time strings using ISO or custom formats
- Adjust date/time by adding/subtracting days/hours/minutes/seconds
- Format date/time to custom strftime formats or standard presets
- Compute differences between two date/times in various units

All tools return JSON strings for structured consumption.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from ..core.tools import function_tool

# Python 3.9+ zoneinfo (stdlib)
try:
    from zoneinfo import ZoneInfo
    _ZONEINFO_AVAILABLE = True
except Exception:
    _ZONEINFO_AVAILABLE = False


def _get_zoneinfo(tz_name: Optional[str]) -> Optional[Any]:
    """Return ZoneInfo for tz_name, or UTC if None. Raises ValueError if invalid."""
    if tz_name is None or not tz_name.strip():
        return timezone.utc
    if not _ZONEINFO_AVAILABLE:
        raise ValueError("ZoneInfo not available in this environment")
    try:
        return ZoneInfo(tz_name.strip())
    except Exception as e:
        raise ValueError(f"Invalid timezone '{tz_name}': {str(e)}")


def _to_iso(dt: datetime) -> str:
    """Return RFC3339/ISO8601 string with offset (e.g., 2025-09-22T10:00:00+05:30)."""
    # Ensure aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _to_rfc3339(dt: datetime) -> str:
    """Return RFC3339-compatible string (same as ISO with offset)."""
    return _to_iso(dt)


def _parse_iso_with_z(value: str) -> Optional[datetime]:
    """Try parsing ISO strings that may end with 'Z' or include offsets."""
    try:
        v = value.strip()
        # Support 'Z' for UTC
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        # datetime.fromisoformat supports offsets like +05:30
        dt = datetime.fromisoformat(v)
        return dt
    except Exception:
        return None


def _parse_with_formats(value: str, formats: List[str]) -> Optional[datetime]:
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt)
        except Exception:
            continue
    return None


def _parse_datetime(
    value: str,
    input_formats: Optional[List[str]] = None,
) -> Optional[datetime]:
    """Parse datetime from string using ISO first, then provided formats, then common fallbacks."""
    if not value or not value.strip():
        return None

    # Try ISO (supports timezone offsets)
    dt = _parse_iso_with_z(value)
    if dt:
        return dt

    # Try provided formats (naive)
    if input_formats:
        dt = _parse_with_formats(value, input_formats)
        if dt:
            return dt

    # Common fallbacks (naive)
    common_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",  # naive ISO without offset
    ]
    return _parse_with_formats(value, common_formats)


def _apply_tz(dt: datetime, tz: Optional[str]) -> datetime:
    """Attach or convert timezone. Naive -> assume tz; aware -> convert to tz."""
    zone = _get_zoneinfo(tz)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=zone)  # assume local as tz
    return dt.astimezone(zone)


def _format_output(
    dt: datetime,
    output_format: Optional[str] = "iso",
) -> str:
    """Format datetime to desired output."""
    if output_format is None or output_format.lower() == "iso":
        return _to_iso(dt)
    if output_format.lower() in ("rfc3339", "rfc"):
        return _to_rfc3339(dt)
    if output_format.lower() in ("epoch", "unix", "timestamp"):
        # Seconds since epoch
        return str(int(dt.timestamp()))
    # Custom strftime
    try:
        return dt.strftime(output_format)
    except Exception as e:
        raise ValueError(f"Invalid output_format '{output_format}': {str(e)}")


@function_tool(timeout=10.0)
async def get_current_datetime(
    tz: Optional[str] = None,
    output_format: Optional[str] = "iso",
    include_unix: bool = True,
    context=None,
) -> str:
    """Get current date/time with optional timezone and output format.

    Args:
        tz: IANA timezone (e.g., 'UTC', 'Asia/Calcutta'). Defaults to UTC if omitted.
        output_format: 'iso' (default), 'rfc3339', 'epoch', or custom strftime pattern.
        include_unix: Include unix epoch seconds in response.

    Returns:
        JSON string: {"type":"current_datetime","tz":"...","iso":"...","formatted":"...","unix": 169...}
    """
    try:
        zone = _get_zoneinfo(tz)
        now = datetime.now(tz=zone)
        formatted = _format_output(now, output_format)
        result = {
            "type": "current_datetime",
            "tz": tz or "UTC",
            "iso": _to_iso(now),
            "formatted": formatted,
        }
        if include_unix:
            result["unix"] = int(now.timestamp())
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to get current datetime: {str(e)}"})


@function_tool(timeout=10.0)
async def parse_datetime(
    value: str,
    input_formats: Optional[List[str]] = None,
    tz: Optional[str] = None,
    output_format: Optional[str] = "iso",
    context=None,
) -> str:
    """Parse a date/time string using ISO or custom formats.

    Args:
        value: Input date/time string (supports ISO with 'Z' or offsets).
        input_formats: Optional list of strptime formats to try.
        tz: Optional timezone to apply/convert to.
        output_format: 'iso' (default), 'rfc3339', 'epoch', or custom strftime pattern.

    Returns:
        JSON: {"type":"parsed_datetime","input":"...","tz":"...","iso":"...","formatted":"..."}
    """
    try:
        dt = _parse_datetime(value, input_formats)
        if not dt:
            return json.dumps({"error": f"Failed to parse datetime from '{value}'"})

        if tz is not None:
            dt = _apply_tz(dt, tz)

        return json.dumps({
            "type": "parsed_datetime",
            "input": value,
            "tz": tz or ("UTC" if dt.tzinfo else None),
            "iso": _to_iso(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)),
            "formatted": _format_output(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc), output_format),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse datetime: {str(e)}"})


@function_tool(timeout=10.0)
async def adjust_datetime(
    value: str,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    tz: Optional[str] = None,
    output_format: Optional[str] = "iso",
    input_formats: Optional[List[str]] = None,
    context=None,
) -> str:
    """Adjust a date/time by adding/subtracting components.

    Args:
        value: Input date/time string
        days/hours/minutes/seconds: Adjustment values (negative to subtract)
        tz: Optional timezone to apply/convert after adjustment
        output_format: Desired output format ('iso' default)
        input_formats: Optional list of strptime formats

    Returns:
        JSON: {"type":"adjusted_datetime","input":"...","delta":{"days":...,"hours":...,"minutes":...,"seconds":...},"tz":"...","iso":"...","formatted":"..."}
    """
    try:
        dt = _parse_datetime(value, input_formats)
        if not dt:
            return json.dumps({"error": f"Failed to parse datetime from '{value}'"})

        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        dt2 = dt + delta

        if tz is not None:
            dt2 = _apply_tz(dt2, tz)

        iso_out = _to_iso(dt2 if dt2.tzinfo else dt2.replace(tzinfo=timezone.utc))
        formatted_out = _format_output(dt2 if dt2.tzinfo else dt2.replace(tzinfo=timezone.utc), output_format)

        return json.dumps({
            "type": "adjusted_datetime",
            "input": value,
            "delta": {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds},
            "tz": tz or ("UTC" if dt2.tzinfo else None),
            "iso": iso_out,
            "formatted": formatted_out,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to adjust datetime: {str(e)}"})


@function_tool(timeout=10.0)
async def format_datetime(
    value: str,
    output_format: str,
    tz: Optional[str] = None,
    input_formats: Optional[List[str]] = None,
    context=None,
) -> str:
    """Format a date/time string using a custom or preset format.

    Args:
        value: Input date/time string
        output_format: strftime format or presets ('iso', 'rfc3339', 'epoch')
        tz: Optional timezone to apply/convert
        input_formats: Optional list of strptime formats

    Returns:
        JSON: {"type":"formatted_datetime","input":"...","tz":"...","formatted":"..."}
    """
    try:
        dt = _parse_datetime(value, input_formats)
        if not dt:
            return json.dumps({"error": f"Failed to parse datetime from '{value}'"})

        if tz is not None:
            dt = _apply_tz(dt, tz)

        formatted = _format_output(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc), output_format)

        return json.dumps({
            "type": "formatted_datetime",
            "input": value,
            "tz": tz or ("UTC" if dt.tzinfo else None),
            "formatted": formatted,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to format datetime: {str(e)}"})


@function_tool(timeout=10.0)
async def diff_datetimes(
    start: str,
    end: str,
    units: str = "seconds",
    tz: Optional[str] = None,
    input_formats: Optional[List[str]] = None,
    context=None,
) -> str:
    """Compute difference between two dates/times.

    Args:
        start: Start datetime string
        end: End datetime string
        units: 'seconds' (default), 'minutes', 'hours', 'days'
        tz: Optional timezone to apply/convert to before diff (useful when inputs are naive)
        input_formats: Optional list of strptime formats

    Returns:
        JSON: {"type":"datetime_diff","units":"...","value":...,"seconds":...,"minutes":...,"hours":...,"days":...}
    """
    try:
        dt_start = _parse_datetime(start, input_formats)
        dt_end = _parse_datetime(end, input_formats)
        if not dt_start or not dt_end:
            return json.dumps({"error": f"Failed to parse start/end datetimes: start='{start}', end='{end}'"})

        if tz is not None:
            dt_start = _apply_tz(dt_start, tz)
            dt_end = _apply_tz(dt_end, tz)

        # Ensure aware for reliable timestamp
        if dt_start.tzinfo is None:
            dt_start = dt_start.replace(tzinfo=timezone.utc)
        if dt_end.tzinfo is None:
            dt_end = dt_end.replace(tzinfo=timezone.utc)

        delta_sec = dt_end.timestamp() - dt_start.timestamp()
        seconds = delta_sec
        minutes = seconds / 60.0
        hours = minutes / 60.0
        days = hours / 24.0

        units_norm = (units or "seconds").lower()
        if units_norm == "seconds":
            value = seconds
        elif units_norm == "minutes":
            value = minutes
        elif units_norm == "hours":
            value = hours
        elif units_norm == "days":
            value = days
        else:
            return json.dumps({"error": f"Unsupported units '{units}'. Use seconds|minutes|hours|days."})

        return json.dumps({
            "type": "datetime_diff",
            "units": units_norm,
            "value": value,
            "seconds": seconds,
            "minutes": minutes,
            "hours": hours,
            "days": days,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to compute datetime difference: {str(e)}"})


def create_date_time_tools():
    """Return list of Date & Time tools for easy agent registration."""
    return [get_current_datetime, parse_datetime, adjust_datetime, format_datetime, diff_datetimes]