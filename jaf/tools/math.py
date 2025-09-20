"""
Prebuilt Math Tools for JAF.

This module provides safe, production-ready math tools:
- calculate: safely evaluate arithmetic expressions (uses ADK SafeMathEvaluator)
- percent_of: compute 'percent of base'
- percentage: compute 'value as a percent of total'
- ratio: compute ratio (decimal, optional simplified fraction when possible)
- date_diff: compute difference between two dates/times in chosen units

All tools return standardized ToolResult with rich metadata via ToolResponse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import gcd, isfinite
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

# JAF core imports
from jaf.core.tools import create_function_tool
from jaf.core.types import Tool, ToolSource
from jaf.core.tool_results import with_error_handling

# Security: AST-based safe evaluator from ADK (already part of this repo)
from adk.utils.safe_evaluator import safe_calculate


# ========== Argument Models ==========

class CalculateArgs(BaseModel):
    """Arguments for arithmetic expression evaluation."""
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate. Supports +, -, *, /, %, **, parentheses. Example: '(25 + 5) * 0.2'"
    )


class PercentOfArgs(BaseModel):
    """Arguments for computing 'percent of base'."""
    base: float = Field(..., description="Base number (e.g., 200)")
    percent: float = Field(..., description="Percent to apply (e.g., 15 for 15%)")
    precision: int = Field(6, ge=0, le=12, description="Decimal places for rounding")


class PercentageArgs(BaseModel):
    """Arguments for computing 'value as a percent of total'."""
    value: float = Field(..., description="Value (e.g., 30)")
    total: float = Field(..., description="Total (e.g., 200)")
    precision: int = Field(6, ge=0, le=12, description="Decimal places for rounding")

    @field_validator("total")
    @classmethod
    def _non_zero_total(cls, v: float) -> float:
        if v == 0:
            raise ValueError("total must be non-zero")
        return v


class RatioArgs(BaseModel):
    """Arguments for computing ratio."""
    numerator: float = Field(..., description="Numerator")
    denominator: float = Field(..., description="Denominator")
    precision: int = Field(6, ge=0, le=12, description="Decimal places for decimal form")
    simplify_fraction: bool = Field(True, description="Attempt to simplify fraction when inputs are integral")


class DateDiffArgs(BaseModel):
    """Arguments for computing date/time differences."""
    start: str = Field(..., description="Start date/time (ISO-8601 or 'YYYY-MM-DD')")
    end: str = Field(..., description="End date/time (ISO-8601 or 'YYYY-MM-DD')")
    unit: str = Field(
        "days",
        description="Primary unit for result: one of 'seconds','minutes','hours','days','weeks'"
    )

    @field_validator("unit")
    @classmethod
    def _unit_valid(cls, v: str) -> str:
        allowed = {"seconds", "minutes", "hours", "days", "weeks"}
        lv = v.lower()
        if lv not in allowed:
            raise ValueError(f"unit must be one of {sorted(allowed)}")
        return lv


# ========== Helpers ==========

def _parse_datetime(value: str) -> datetime:
    """
    Parse a datetime from ISO-8601 or 'YYYY-MM-DD' formats.
    Interprets date-only values as midnight in UTC for determinism.
    """
    s = value.strip()
    # Try strict ISO first
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Try date-only
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    # Fallback: try common datetime formats
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    raise ValueError(f"Unsupported datetime format: {value!r}")


def _round(v: float, precision: int) -> float:
    try:
        if not isfinite(v):
            return v
    except Exception:
        pass
    return float(f"{v:.{precision}f}")


def _simplify_fraction(n: int, d: int) -> Tuple[int, int]:
    if d == 0:
        return (n, d)
    g = gcd(n, d)
    n_s = n // g
    d_s = d // g
    # keep denominator positive
    if d_s < 0:
        n_s, d_s = -n_s, -d_s
    return n_s, d_s


# ========== Executors (pure logic) ==========

async def _calculate_impl(args: CalculateArgs, _context: Any) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression using ADK's safe evaluator.
    Returns a structured result.
    """
    result = safe_calculate(args.expression)
    return {
        "expression": args.expression,
        "result": result
    }


async def _percent_of_impl(args: PercentOfArgs, _context: Any) -> Dict[str, Any]:
    value = (args.base * args.percent) / 100.0
    return {
        "base": args.base,
        "percent": args.percent,
        "result": _round(value, args.precision),
        "explanation": f"{args.percent}% of {args.base} = {value}"
    }


async def _percentage_impl(args: PercentageArgs, _context: Any) -> Dict[str, Any]:
    # Extra guard (in addition to validator)
    if args.total == 0:
        raise ValueError("total must be non-zero")
    pct = (args.value / args.total) * 100.0
    return {
        "value": args.value,
        "total": args.total,
        "percent": _round(pct, args.precision),
        "explanation": f"{args.value} is {pct}% of {args.total}"
    }


async def _ratio_impl(args: RatioArgs, _context: Any) -> Dict[str, Any]:
    if args.denominator == 0:
        raise ValueError("denominator must be non-zero")

    decimal = args.numerator / args.denominator
    out: Dict[str, Any] = {
        "numerator": args.numerator,
        "denominator": args.denominator,
        "decimal": _round(decimal, args.precision),
    }

    # Only attempt simplification if both look integral within a tolerance
    num_int = int(round(args.numerator))
    den_int = int(round(args.denominator))
    if args.simplify_fraction and abs(args.numerator - num_int) < 1e-9 and abs(args.denominator - den_int) < 1e-9:
        sn, sd = _simplify_fraction(num_int, den_int)
        out["fraction"] = f"{sn}:{sd}"
    else:
        out["fraction"] = f"{args.numerator}:{args.denominator}"

    return out


async def _date_diff_impl(args: DateDiffArgs, _context: Any) -> Dict[str, Any]:
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)

    delta = end - start
    seconds = delta.total_seconds()
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    weeks = days / 7.0

    unit_map = {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "days": days,
        "weeks": weeks,
    }

    primary_value = unit_map[args.unit.lower()]

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "unit": args.unit.lower(),
        "value": primary_value,
        "all_units": {
            "seconds": seconds,
            "minutes": minutes,
            "hours": hours,
            "days": days,
            "weeks": weeks
        }
    }


# ========== Wrapped Executors with Standardized Error Handling ==========

calculate_execute = with_error_handling("calculate", _calculate_impl)
percent_of_execute = with_error_handling("percent_of", _percent_of_impl)
percentage_execute = with_error_handling("percentage", _percentage_impl)
ratio_execute = with_error_handling("ratio", _ratio_impl)
date_diff_execute = with_error_handling("date_diff", _date_diff_impl)


# ========== Tool Instances ==========

calculate_tool: Tool[Any, Any] = create_function_tool({
    "name": "calculate",
    "description": "Safely evaluate mathematical expressions (arithmetic, parentheses, exponents). Uses secure AST evaluation.",
    "execute": calculate_execute,
    "parameters": CalculateArgs,
    "metadata": {"category": "math", "version": "1.0"},
    "source": ToolSource.NATIVE,
    "timeout": 15.0
})

percent_of_tool: Tool[Any, Any] = create_function_tool({
    "name": "percent_of",
    "description": "Compute X% of a base value. Example: 15% of 200 = 30",
    "execute": percent_of_execute,
    "parameters": PercentOfArgs,
    "metadata": {"category": "math", "version": "1.0"},
    "source": ToolSource.NATIVE,
    "timeout": 10.0
})

percentage_tool: Tool[Any, Any] = create_function_tool({
    "name": "percentage",
    "description": "Compute 'value as a percent of total'. Example: 30 of 200 = 15%",
    "execute": percentage_execute,
    "parameters": PercentageArgs,
    "metadata": {"category": "math", "version": "1.0"},
    "source": ToolSource.NATIVE,
    "timeout": 10.0
})

ratio_tool: Tool[Any, Any] = create_function_tool({
    "name": "ratio",
    "description": "Compute ratio as decimal and fraction (simplified for integral inputs).",
    "execute": ratio_execute,
    "parameters": RatioArgs,
    "metadata": {"category": "math", "version": "1.0"},
    "source": ToolSource.NATIVE,
    "timeout": 10.0
})

date_diff_tool: Tool[Any, Any] = create_function_tool({
    "name": "date_diff",
    "description": "Compute difference between two dates/times in chosen unit and provide all units.",
    "execute": date_diff_execute,
    "parameters": DateDiffArgs,
    "metadata": {"category": "math", "version": "1.0"},
    "source": ToolSource.NATIVE,
    "timeout": 10.0
})


# ========== Factory for Batch Access ==========

def create_math_tools() -> List[Tool[Any, Any]]:
    """
    Create and return the suite of prebuilt math tools.
    - calculate
    - percent_of
    - percentage
    - ratio
    - date_diff
    """
    return [
        calculate_tool,
        percent_of_tool,
        percentage_tool,
        ratio_tool,
        date_diff_tool,
    ]


# Backward/explicit exports
__all__ = [
    "CalculateArgs",
    "PercentOfArgs",
    "PercentageArgs",
    "RatioArgs",
    "DateDiffArgs",
    "calculate_tool",
    "percent_of_tool",
    "percentage_tool",
    "ratio_tool",
    "date_diff_tool",
    "create_math_tools",
]