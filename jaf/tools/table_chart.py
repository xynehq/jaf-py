"""
Table & Chart Generator tools for JAF.

Convert structured data (primarily CSV) into:
- Markdown/HTML tables
- Charts (line/bar) rendered to PNG and returned as base64 JSON

Safe path handling is enforced and outputs are JSON strings for structured consumption.
"""

import os
import csv
import json
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

from ..core.tools import function_tool
from adk.security.sanitization import sanitize_file_path


def _validate_relative_path(path: Optional[str]) -> Optional[str]:
    """Ensure the path is relative and non-empty."""
    if not path:
        return "Error: Empty file path"
    if os.path.isabs(path) or "://" in path:
        return "Error: Absolute or URL-like paths are not allowed. Use a relative path within the project."
    return None


def _ensure_parent_dirs(file_path: str) -> Optional[str]:
    """Ensure parent directories exist for a path."""
    try:
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return None
    except Exception as e:
        return f"Error: Failed to prepare directories for '{file_path}': {str(e)}"


def _read_csv_dicts(
    file_path: str,
    delimiter: str = ",",
    has_header: bool = True,
    max_rows: int = 10000,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read CSV rows as list[dict] if has_header, else synthesize headers and return dict rows.
    Returns (rows, headers). Raises exceptions on failures.
    """
    safe_path = sanitize_file_path(file_path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"File not found: {safe_path}")

    rows: List[Dict[str, Any]] = []
    headers: List[str] = []

    with open(safe_path, "r", encoding="utf-8", newline="") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(dict(row))
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append({f"col_{j+1}": value for j, value in enumerate(row)})
            # Determine headers from first row
            if rows:
                headers = list(rows[0].keys())

    return rows, headers


def _render_markdown_table(rows: List[Dict[str, Any]], headers: List[str], max_cols: int = 50, max_rows: int = 1000) -> str:
    """Convert dict rows to Markdown table."""
    if not headers and rows:
        headers = list(rows[0].keys())
    headers = headers[:max_cols]

    md_lines = []
    # Header
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Rows
    count = 0
    for row in rows:
        md_lines.append("| " + " | ".join([str(row.get(h, ""))[:1000] for h in headers]) + " |")
        count += 1
        if count >= max_rows:
            break
    return "\n".join(md_lines)


def _render_html_table(rows: List[Dict[str, Any]], headers: List[str], max_cols: int = 50, max_rows: int = 1000) -> str:
    """Convert dict rows to HTML table."""
    if not headers and rows:
        headers = list(rows[0].keys())
    headers = headers[:max_cols]

    html = []
    html.append("<table>")
    html.append("<thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>")
    html.append("<tbody>")
    count = 0
    for row in rows:
        html.append("<tr>" + "".join([f"<td>{str(row.get(h, ''))[:1000]}</td>" for h in headers]) + "</tr>")
        count += 1
        if count >= max_rows:
            break
    html.append("</tbody>")
    html.append("</table>")
    return "\n".join(html)


def _to_float_safe(value: Any) -> Optional[float]:
    """Convert to float if possible, else None."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        v = str(value).strip()
        if v == "":
            return None
        return float(v)
    except Exception:
        return None


def _prepare_xy_series(rows: List[Dict[str, Any]], x_column: Optional[str], y_columns: List[str]) -> Tuple[List[Any], Dict[str, List[Optional[float]]]]:
    """
    Create x sequence and y series dict from rows.
    If x_column is None or non-numeric, uses row index for x (0..n-1).
    """
    # X values
    x_vals: List[Any] = []
    if x_column and rows and (x_column in rows[0]):
        # try numeric x
        numeric_x = []
        all_numeric = True
        for r in rows:
            val = _to_float_safe(r.get(x_column))
            numeric_x.append(val)
            if val is None:
                all_numeric = False
        if all_numeric:
            x_vals = [float(v) for v in numeric_x if v is not None]
        else:
            # Use raw values; if some None, fall back to index
            raw = [r.get(x_column) for r in rows]
            if any(v is None for v in raw):
                x_vals = list(range(len(rows)))
            else:
                x_vals = raw
    else:
        x_vals = list(range(len(rows)))

    # Y series
    y_series: Dict[str, List[Optional[float]]] = {}
    for y in y_columns:
        series: List[Optional[float]] = []
        for r in rows:
            series.append(_to_float_safe(r.get(y)))
        y_series[y] = series
    return x_vals, y_series


def _render_chart_png(
    x_vals: List[Any],
    y_series: Dict[str, List[Optional[float]]],
    chart_type: str,
    title: Optional[str],
    width: int,
    height: int,
    dpi: int,
) -> bytes:
    """Render chart to PNG bytes using matplotlib; raises if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Matplotlib not available: {str(e)}")

    # Prepare figure
    fig, ax = plt.subplots(figsize=(max(1, width) / dpi, max(1, height) / dpi), dpi=dpi)

    # Plot series
    for label, series in y_series.items():
        # Align length with x_vals
        n = min(len(x_vals), len(series))
        xv = x_vals[:n]
        yv = series[:n]
        # Replace None with NaN for plotting gaps
        import math
        yv_num = [v if v is not None else float("nan") for v in yv]

        if chart_type.lower() == "line":
            ax.plot(xv, yv_num, label=label)
        elif chart_type.lower() == "bar":
            # For multi-series bar, use offsets
            import numpy as np
            indices = np.arange(n)
            width_bar = 0.8 / max(1, len(y_series))
            # Determine position index for label
            labels_list = list(y_series.keys())
            pos_idx = labels_list.index(label)
            ax.bar(indices + pos_idx * width_bar, yv_num, width_bar, label=label)
            ax.set_xticks(indices + (len(y_series) - 1) * width_bar / 2)
            ax.set_xticklabels([str(x) for x in xv])
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}. Use 'line' or 'bar'.")

    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Render to PNG
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@function_tool(timeout=60.0)
async def table_from_csv(
    file_path: str,
    delimiter: str = ",",
    has_header: bool = True,
    max_rows: int = 1000,
    format: str = "markdown",  # "markdown" or "html"
    context=None,
) -> str:
    """Generate a table (Markdown/HTML) from CSV.

    Args:
        file_path: Relative path to CSV file (sanitized; absolute paths disallowed)
        delimiter: CSV delimiter (default ',')
        has_header: Whether the first row is header (default True)
        max_rows: Maximum number of rows to render (default 1000)
        format: 'markdown' or 'html' (default 'markdown')

    Returns:
        JSON string: {"type":"table","format":"markdown|html","content":"..."} or {"error":"..."}
    """
    # Validate path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return json.dumps({"error": path_error})
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return json.dumps({"error": f"Unsafe file path '{file_path}': {str(e)}"})

    try:
        rows, headers = _read_csv_dicts(safe_path, delimiter=delimiter, has_header=has_header, max_rows=max_rows)
        if format.lower() == "html":
            content = _render_html_table(rows, headers, max_rows=max_rows)
            return json.dumps({"type": "table", "format": "html", "content": content}, ensure_ascii=False)
        elif format.lower() == "markdown":
            content = _render_markdown_table(rows, headers, max_rows=max_rows)
            return json.dumps({"type": "table", "format": "markdown", "content": content}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Unsupported format '{format}'. Use 'markdown' or 'html'."})
    except Exception as e:
        return json.dumps({"error": f"Failed to generate table: {str(e)}"})


@function_tool(timeout=90.0)
async def chart_from_csv(
    file_path: str,
    x_column: Optional[str],
    y_columns: List[str],
    chart_type: str = "line",  # "line" or "bar"
    delimiter: str = ",",
    has_header: bool = True,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    dpi: int = 100,
    output_image_path: Optional[str] = None,
    context=None,
) -> str:
    """Generate a chart (line/bar) from CSV and return base64 PNG in JSON.

    Args:
        file_path: Relative path to CSV file (sanitized; absolute paths disallowed)
        x_column: Column to use for X axis (optional; index used if None or non-numeric)
        y_columns: List of columns to plot as series on Y axis (at least one required)
        chart_type: 'line' or 'bar'
        delimiter: CSV delimiter
        has_header: Whether the first row is header
        title: Optional chart title
        width: Image width in pixels (default 800)
        height: Image height in pixels (default 600)
        dpi: Rendering DPI (default 100)
        output_image_path: Optional relative path to save PNG to disk in addition to returning base64

    Returns:
        JSON string:
            {"type":"chart","chart_type":"line|bar","mime":"image/png",
             "image_base64":"...","width":..., "height":..., "x_column": "...", "y_columns":[...], "saved_path": Optional[str]}
        or {"error":"..."} on error
    """
    # Validate paths
    path_error = _validate_relative_path(file_path)
    if path_error:
        return json.dumps({"error": path_error})
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return json.dumps({"error": f"Unsafe file path '{file_path}': {str(e)}"})

    if output_image_path:
        out_err = _validate_relative_path(output_image_path)
        if out_err:
            return json.dumps({"error": out_err})
        try:
            output_image_path = sanitize_file_path(output_image_path)
        except Exception as e:
            return json.dumps({"error": f"Unsafe output path '{output_image_path}': {str(e)}"})

    # Validate y_columns
    if not isinstance(y_columns, list) or not y_columns:
        return json.dumps({"error": "y_columns must be a non-empty list of column names"})

    try:
        # Load data
        rows, headers = _read_csv_dicts(safe_path, delimiter=delimiter, has_header=has_header, max_rows=100000)

        # Validate columns
        missing = [c for c in ([x_column] if x_column else []) + y_columns if c and c not in headers]
        # Allow non-header CSV; in that case, headers may be synthesized (col_1, col_2, ...)
        if missing:
            return json.dumps({"error": f"Missing columns: {', '.join(missing)}. Available: {headers}"})

        # Prepare series
        x_vals, y_series = _prepare_xy_series(rows, x_column, y_columns)

        # Render PNG
        png_bytes = _render_chart_png(x_vals, y_series, chart_type=chart_type, title=title, width=width, height=height, dpi=dpi)
        b64 = base64.b64encode(png_bytes).decode("ascii")

        # Optionally save to disk
        saved_path = None
        if output_image_path:
            dir_error = _ensure_parent_dirs(output_image_path)
            if dir_error:
                return json.dumps({"error": dir_error})
            try:
                with open(output_image_path, "wb") as f:
                    f.write(png_bytes)
                saved_path = output_image_path
            except Exception as e:
                return json.dumps({"error": f"Failed to save PNG to '{output_image_path}': {str(e)}"})

        return json.dumps(
            {
                "type": "chart",
                "chart_type": chart_type,
                "mime": "image/png",
                "image_base64": b64,
                "width": width,
                "height": height,
                "x_column": x_column,
                "y_columns": y_columns,
                "saved_path": saved_path,
            },
            ensure_ascii=False,
        )

    except RuntimeError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to generate chart: {str(e)}"})


def create_table_chart_tools():
    """Return list of Table & Chart tools for easy agent registration."""
    return [table_from_csv, chart_from_csv]