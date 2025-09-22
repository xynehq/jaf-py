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

- File I/O tools:
    from jaf.tools import read_text, write_text, read_json, write_json, read_csv, write_csv
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

from .file_io import (
    read_text,
    write_text,
    read_json,
    write_json,
    read_csv,
    write_csv,
)

from .text_extractor import (
    extract_emails,
    extract_invoice_numbers,
    extract_order_ids,
    extract_regex,
    create_text_extractor_tools,
)

from .table_chart import (
    table_from_csv,
    chart_from_csv,
    create_table_chart_tools,
)

# (duplicate import block removed)

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
    "read_text",
    "write_text",
    "read_json",
    "write_json",
    "read_csv",
    "write_csv",
    # Text extractor tools
    "extract_emails",
    "extract_invoice_numbers",
    "extract_order_ids",
    "extract_regex",
    "create_text_extractor_tools",
    # Table & chart tools
    "table_from_csv",
    "chart_from_csv",
    "create_table_chart_tools",
]