"""
Unit Conversion tools for JAF.

Provides:
- Measurement conversion across common categories (length, mass, volume, area, time, data, temperature)
- Currency conversion using provided exchange rates (via JSON input or context)
- Listing supported units

All tools return JSON strings for structured consumption.
"""

import json
from typing import Optional, Dict, Any, List, Tuple

from ..core.tools import function_tool


# ----------------------------
# Measurement conversion setup
# ----------------------------

# Base units per category:
# - length: meter (m)
# - mass: kilogram (kg)
# - volume: liter (L) except cubic meter noted with factor
# - area: square meter (m2)
# - time: second (s)
# - data: byte (B) with decimal (SI) and binary (IEC) prefixes distinguished
# - temperature: special handling (C, F, K) via formulas

_LENGTH_FACTORS: Dict[str, float] = {
    # Base m
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    # Imperial
    "mi": 1609.344,
    "yard": 0.9144,
    "yd": 0.9144,
    "ft": 0.3048,
    "inch": 0.0254,
    "in": 0.0254,
}

_MASS_FACTORS: Dict[str, float] = {
    # Base kg
    "kg": 1.0,
    "g": 0.001,
    "mg": 0.000001,
    "tonne": 1000.0,
    "t": 1000.0,
    # Imperial
    "lb": 0.45359237,
    "ounce": 0.028349523125,
    "oz": 0.028349523125,
}

_VOLUME_FACTORS: Dict[str, float] = {
    # Base L
    "l": 1.0,
    "L": 1.0,
    "liter": 1.0,
    "ml": 0.001,
    "mL": 0.001,
    # Metric cubic
    "m3": 1000.0,  # 1 cubic meter = 1000 liters
    # US customary
    "gallon_us": 3.785411784,
    "quart_us": 0.946352946,
    "pint_us": 0.473176473,
    "cup_us": 0.24,  # cooking cup approximation (exact US legal cup is 240 mL)
    "fl_oz_us": 0.0295735295625,
}

_AREA_FACTORS: Dict[str, float] = {
    # Base m2
    "m2": 1.0,
    "m^2": 1.0,
    "km2": 1_000_000.0,
    "km^2": 1_000_000.0,
    "cm2": 0.0001,
    "cm^2": 0.0001,
    "ft2": 0.09290304,
    "ft^2": 0.09290304,
    "in2": 0.00064516,
    "in^2": 0.00064516,
    "acre": 4046.8564224,
    "hectare": 10_000.0,
    "ha": 10_000.0,
}

_TIME_FACTORS: Dict[str, float] = {
    # Base s
    "s": 1.0,
    "sec": 1.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "day": 86400.0,
}

# Data sizes: byte base
_DATA_FACTORS: Dict[str, float] = {
    # Decimal SI units (1000-based)
    "B": 1.0,
    "KB": 1000.0,
    "MB": 1000.0 ** 2,
    "GB": 1000.0 ** 3,
    "TB": 1000.0 ** 4,
    # Binary IEC units (1024-based)
    "KiB": 1024.0,
    "MiB": 1024.0 ** 2,
    "GiB": 1024.0 ** 3,
    "TiB": 1024.0 ** 4,
}

# Supported categories to factor maps
_CATEGORY_MAPS: Dict[str, Dict[str, float]] = {
    "length": _LENGTH_FACTORS,
    "mass": _MASS_FACTORS,
    "volume": _VOLUME_FACTORS,
    "area": _AREA_FACTORS,
    "time": _TIME_FACTORS,
    "data": _DATA_FACTORS,
}

# Temperature units handled separately
_TEMPERATURE_UNITS = {"C", "F", "K", "celsius", "fahrenheit", "kelvin"}


def _normalize_unit(u: str) -> str:
    return u.strip()


def _detect_category(from_unit: str, to_unit: str) -> Optional[str]:
    fu = _normalize_unit(from_unit)
    tu = _normalize_unit(to_unit)
    # Temperature special-case
    if fu in _TEMPERATURE_UNITS or tu in _TEMPERATURE_UNITS:
        return "temperature"
    # Otherwise search factor maps
    for cat, factors in _CATEGORY_MAPS.items():
        if fu in factors and tu in factors:
            return cat
    return None


def _convert_via_factor(value: float, from_unit: str, to_unit: str, factors: Dict[str, float]) -> float:
    # value_in_base = value * factor_from
    # value_in_target = value_in_base / factor_to
    factor_from = factors[from_unit]
    factor_to = factors[to_unit]
    return (value * factor_from) / factor_to


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    f = from_unit.lower()
    t = to_unit.lower()
    # Normalize names
    if f == "celsius":
        f = "c"
    if t == "celsius":
        t = "c"
    if f == "fahrenheit":
        f = "f"
    if t == "fahrenheit":
        t = "f"
    if f == "kelvin":
        f = "k"
    if t == "kelvin":
        t = "k"

    # Convert from source to Celsius first
    if f == "c":
        c = value
    elif f == "f":
        c = (value - 32.0) * 5.0 / 9.0
    elif f == "k":
        c = value - 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {from_unit}")

    # Convert Celsius to target
    if t == "c":
        return c
    elif t == "f":
        return c * 9.0 / 5.0 + 32.0
    elif t == "k":
        return c + 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {to_unit}")


@function_tool(timeout=30.0)
async def convert_measurement(
    value: float,
    from_unit: str,
    to_unit: str,
    category: Optional[str] = None,
    context=None,
) -> str:
    """Convert measurement between units.

    Args:
        value: Numeric value to convert
        from_unit: Source unit string (e.g., 'km', 'm', 'lb', 'C', 'F', 'K', 'GB', 'MiB')
        to_unit: Target unit string
        category: Optional category hint ('length', 'mass', 'volume', 'area', 'time', 'data', 'temperature').
                  If omitted, category is inferred from units.

    Returns:
        JSON string: {"type":"unit_conversion","category":"...", "from_unit":"...", "to_unit":"...", "input": ..., "output": ...}
        or {"error":"..."} on error.
    """
    try:
        fu = _normalize_unit(from_unit)
        tu = _normalize_unit(to_unit)

        cat = category or _detect_category(fu, tu)
        if not cat:
            # Provide guidance if inference failed
            return json.dumps({
                "error": f"Could not infer category for units '{from_unit}' -> '{to_unit}'. "
                         f"Provide 'category' or ensure units belong to same category."
            })

        if cat == "temperature":
            out = _convert_temperature(value, fu, tu)
        else:
            factors = _CATEGORY_MAPS.get(cat)
            if not factors:
                return json.dumps({"error": f"Unsupported category '{cat}'"})
            if fu not in factors or tu not in factors:
                return json.dumps({
                    "error": f"Unsupported units for category '{cat}': from='{from_unit}', to='{to_unit}'",
                    "supported_units": sorted(list(factors.keys()))
                })
            out = _convert_via_factor(value, fu, tu, factors)

        return json.dumps({
            "type": "unit_conversion",
            "category": cat,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "input": value,
            "output": out
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Measurement conversion failed: {str(e)}"})


@function_tool(timeout=15.0)
async def list_supported_units(
    category: Optional[str] = None,
    context=None,
) -> str:
    """List supported measurement units (optionally by category).

    Args:
        category: Optional category filter

    Returns:
        JSON string:
          - If category provided: {"category":"...", "units":[...]}
          - Else: {"categories":[...], "units_by_category": {"length":[...], ...}}
    """
    try:
        if category:
            cat = category.strip().lower()
            if cat == "temperature":
                return json.dumps({"category": "temperature", "units": sorted(list(_TEMPERATURE_UNITS))})
            factors = _CATEGORY_MAPS.get(cat)
            if not factors:
                return json.dumps({"error": f"Unsupported category '{category}'"})
            return json.dumps({"category": cat, "units": sorted(list(factors.keys()))})
        else:
            units_by_category = {
                k: sorted(list(v.keys()))
                for k, v in _CATEGORY_MAPS.items()
            }
            units_by_category["temperature"] = sorted(list(_TEMPERATURE_UNITS))
            return json.dumps({
                "categories": sorted(list(units_by_category.keys())),
                "units_by_category": units_by_category
            })
    except Exception as e:
        return json.dumps({"error": f"Failed to list units: {str(e)}"})


# ----------------------------
# Currency conversion
# ----------------------------

def _load_rates(rates_json: Optional[str], context) -> Optional[Dict[str, float]]:
    # Try JSON first
    if rates_json:
        try:
            data = json.loads(rates_json)
            if not isinstance(data, dict):
                return None
            # Normalize keys to upper-case
            return {str(k).upper(): float(v) for k, v in data.items()}
        except Exception:
            return None
    # Try context (e.g., context.exchange_rates)
    if context is not None:
        # Support dict or callable on context
        if hasattr(context, "exchange_rates"):
            rates = getattr(context, "exchange_rates")
            if callable(rates):
                try:
                    rates = rates()
                except Exception:
                    rates = None
            if isinstance(rates, dict):
                return {str(k).upper(): float(v) for k, v in rates.items()}
    return None


@function_tool(timeout=15.0)
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    base_currency: Optional[str] = "USD",
    rates_json: Optional[str] = None,
    context=None,
) -> str:
    """Convert currency using provided exchange rates relative to a base currency.

    Args:
        amount: Amount to convert
        from_currency: ISO currency code (e.g., 'USD', 'INR', 'EUR')
        to_currency: ISO currency code
        base_currency: Base currency for rates (default 'USD')
        rates_json: Optional JSON mapping {currency_code: rate_vs_base}. If omitted, tool will look for context.exchange_rates.

    Notes:
        - Rates must be provided either via rates_json or context.exchange_rates
        - Example rates_json: {"USD": 1.0, "EUR": 0.92, "INR": 83.0}

    Returns:
        JSON string: {"type":"currency_conversion","base":"USD","from":"USD","to":"INR","amount":..., "result": ...}
        or {"error":"..."} on error.
    """
    try:
        fr = from_currency.strip().upper()
        to = to_currency.strip().upper()
        base = (base_currency or "USD").strip().upper()

        rates = _load_rates(rates_json, context)
        if not rates or base not in rates:
            return json.dumps({
                "error": "Exchange rates not provided or missing base currency.",
                "hint": "Provide rates_json (e.g., {\"USD\":1.0,\"EUR\":0.92,\"INR\":83.0}) or context.exchange_rates",
                "required_base": base
            })

        if fr not in rates or to not in rates:
            return json.dumps({
                "error": f"Missing currency in rates: from='{fr}' or to='{to}'",
                "available": sorted(list(rates.keys()))
            })

        # Convert amount from 'fr' to base, then to 'to'
        # amount_in_base = amount / rate(fr)
        # result = amount_in_base * rate(to)
        rate_fr = float(rates[fr])
        rate_to = float(rates[to])
        if rate_fr == 0.0:
            return json.dumps({"error": f"Invalid zero rate for '{fr}'"})
        result = (amount / rate_fr) * rate_to

        return json.dumps({
            "type": "currency_conversion",
            "base": base,
            "from": fr,
            "to": to,
            "amount": amount,
            "result": result
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Currency conversion failed: {str(e)}"})


def create_unit_conversion_tools():
    """Return list of Unit Conversion tools for easy agent registration."""
    return [convert_measurement, list_supported_units, convert_currency]