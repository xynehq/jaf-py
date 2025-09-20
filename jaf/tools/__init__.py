"""
Prebuilt tools namespace for JAF.

Usage:
- Import the full math tool suite:
    from jaf.tools import create_math_tools
    tools = create_math_tools()

- Or import individual tools / argument models:
    from jaf.tools import (
        calculate_tool, percent_of_tool, percentage_tool, ratio_tool, date_diff_tool,
        CalculateArgs, PercentOfArgs, PercentageArgs, RatioArgs, DateDiffArgs
    )
"""

from .math import (
    CalculateArgs,
    PercentOfArgs,
    PercentageArgs,
    RatioArgs,
    DateDiffArgs,
    calculate_tool,
    percent_of_tool,
    percentage_tool,
    ratio_tool,
    date_diff_tool,
    create_math_tools,
)

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