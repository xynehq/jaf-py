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

from .unit_conversion import (
    convert_measurement,
    list_supported_units,
    convert_currency,
    create_unit_conversion_tools,
)

from .date_time import (
    get_current_datetime,
    parse_datetime,
    adjust_datetime,
    format_datetime,
    diff_datetimes,
    create_date_time_tools,
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
    # Unit conversion tools
    "convert_measurement",
    "list_supported_units",
    "convert_currency",
    "create_unit_conversion_tools",
    # Date & time tools
    "get_current_datetime",
    "parse_datetime",
    "adjust_datetime",
    "format_datetime",
    "diff_datetimes",
    "create_date_time_tools",
]