"""
Built-in File I/O tools for JAF.

This module provides safe, convenience tools for reading and writing text, JSON, and CSV files
with input sanitization to prevent path traversal and other injection risks.
"""

import os
import json
import csv
from typing import Optional, List, Dict, Any

# Use the JAF function tool decorator for schema generation and timeouts
from ..core.tools import function_tool

# Production-grade path sanitization
from adk.security.sanitization import sanitize_file_path


def _validate_relative_path(path: str) -> Optional[str]:
    """
    Validate that the path is relative (to prevent absolute path writes/reads).
    Returns an error string if invalid, else None.
    """
    if not path:
        return "Error: Empty file path"
    # Disallow absolute paths (macOS/Linux style) and protocol specifiers
    if os.path.isabs(path) or "://" in path:
        return "Error: Absolute or URL-like paths are not allowed. Use a relative path within the project."
    return None


def _ensure_parent_dirs(file_path: str) -> Optional[str]:
    """
    Ensure that parent directories exist for a target file path.
    Returns error string on failure, else None.
    """
    try:
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return None
    except Exception as e:
        return f"Error: Failed to prepare directories for '{file_path}': {str(e)}"


@function_tool(timeout=60.0)
async def read_text(
    file_path: str,
    encoding: str = "utf-8",
    max_bytes: int = 1048576,  # 1 MB default
    context=None,
) -> str:
    """Read a text file safely.

    Args:
        file_path: Relative path to the file (sanitized; absolute paths disallowed)
        encoding: Text encoding (default utf-8)
        max_bytes: Maximum allowed file size to read (default 1 MB)

    Returns:
        File contents as a string, or an error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    try:
        if not os.path.exists(safe_path):
            return f"Error: File not found: {safe_path}"
        size = os.path.getsize(safe_path)
        if size > max_bytes:
            return f"Error: File too large ({size} bytes). Max allowed is {max_bytes} bytes"
        with open(safe_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        return f"Error: Failed to read file '{safe_path}': {str(e)}"


@function_tool(timeout=60.0)
async def write_text(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    overwrite: bool = True,
    append: bool = False,
    context=None,
) -> str:
    """Write text to a file safely.

    Args:
        file_path: Relative path to the file (sanitized; absolute paths disallowed)
        content: Text content to write
        encoding: Text encoding (default utf-8)
        overwrite: Overwrite existing file (default True)
        append: Append to existing file instead of overwriting (default False)

    Returns:
        Confirmation string, or an error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    # Mode resolution: append wins if true
    mode = "a" if append else "w"
    if not append and not overwrite and os.path.exists(safe_path):
        return f"Error: File '{safe_path}' exists and overwrite is False"

    dir_error = _ensure_parent_dirs(safe_path)
    if dir_error:
        return dir_error

    try:
        with open(safe_path, mode, encoding=encoding) as f:
            f.write(content)
        action = "appended to" if append else ("overwritten" if os.path.exists(safe_path) else "created")
        return f"Text {action} '{safe_path}' ({len(content)} bytes)"
    except Exception as e:
        return f"Error: Failed to write file '{safe_path}': {str(e)}"


@function_tool(timeout=60.0)
async def read_json(
    file_path: str,
    pretty: bool = True,
    context=None,
) -> str:
    """Read a JSON file and return its content as JSON string.

    Args:
        file_path: Relative path to JSON file (sanitized; absolute paths disallowed)
        pretty: Pretty-print output JSON (default True)

    Returns:
        JSON string content, or an error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    try:
        if not os.path.exists(safe_path):
            return f"Error: File not found: {safe_path}"
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
    except Exception as e:
        return f"Error: Failed to read JSON from '{safe_path}': {str(e)}"


@function_tool(timeout=60.0)
async def write_json(
    file_path: str,
    data_json: str,
    pretty: bool = True,
    overwrite: bool = True,
    context=None,
) -> str:
    """Write JSON string to a file safely.

    Args:
        file_path: Relative path to JSON file (sanitized; absolute paths disallowed)
        data_json: JSON string to write
        pretty: Pretty-print output JSON (default True)
        overwrite: Overwrite existing file (default True)

    Returns:
        Confirmation string, or an error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    if not overwrite and os.path.exists(safe_path):
        return f"Error: File '{safe_path}' exists and overwrite is False"

    dir_error = _ensure_parent_dirs(safe_path)
    if dir_error:
        return dir_error

    try:
        # Validate input JSON
        data = json.loads(data_json)
    except Exception as e:
        return f"Error: Invalid input JSON: {str(e)}"

    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)
        return f"JSON written to '{safe_path}'"
    except Exception as e:
        return f"Error: Failed to write JSON to '{safe_path}': {str(e)}"


@function_tool(timeout=60.0)
async def read_csv(
    file_path: str,
    delimiter: str = ",",
    has_header: bool = True,
    max_rows: int = 10000,
    context=None,
) -> str:
    """Read a CSV file and return rows as JSON string.

    Args:
        file_path: Relative path to CSV file (sanitized; absolute paths disallowed)
        delimiter: CSV delimiter (default ',')
        has_header: Whether the first row is header (default True)
        max_rows: Maximum number of rows to read (default 10,000)

    Returns:
        JSON string: list of objects (if has_header) or list of lists, or error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    try:
        if not os.path.exists(safe_path):
            return f"Error: File not found: {safe_path}"

        rows_out: List[Any] = []
        with open(safe_path, "r", encoding="utf-8", newline="") as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows_out.append(dict(row))
            else:
                reader = csv.reader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows_out.append(list(row))

        return json.dumps(rows_out, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: Failed to read CSV from '{safe_path}': {str(e)}"


@function_tool(timeout=60.0)
async def write_csv(
    file_path: str,
    rows_json: str,
    delimiter: str = ",",
    headers: Optional[List[str]] = None,
    overwrite: bool = True,
    context=None,
) -> str:
    """Write rows to a CSV file from JSON.

    Args:
        file_path: Relative path to CSV file (sanitized; absolute paths disallowed)
        rows_json: JSON string representing rows. Either:
                   - List[Dict[str, Any]] for header-based writing
                   - List[List[str]] (requires 'headers' for the first row)
        delimiter: CSV delimiter (default ',')
        headers: Optional explicit headers for List[List] input
        overwrite: Overwrite existing file (default True)

    Returns:
        Confirmation string, or an error string starting with "Error: "
    """
    # Validate and sanitize path
    path_error = _validate_relative_path(file_path)
    if path_error:
        return path_error
    try:
        safe_path = sanitize_file_path(file_path)
    except Exception as e:
        return f"Error: Unsafe file path '{file_path}': {str(e)}"

    if not overwrite and os.path.exists(safe_path):
        return f"Error: File '{safe_path}' exists and overwrite is False"

    dir_error = _ensure_parent_dirs(safe_path)
    if dir_error:
        return dir_error

    # Parse input JSON
    try:
        rows = json.loads(rows_json)
    except Exception as e:
        return f"Error: Invalid input JSON: {str(e)}"

    try:
        with open(safe_path, "w", encoding="utf-8", newline="") as f:
            # Determine rows type
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                # Dict rows -> use DictWriter
                fieldnames: List[str] = list(rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(rows)  # type: ignore[arg-type]
            elif isinstance(rows, list) and (not rows or isinstance(rows[0], list)):
                # List rows -> need headers provided
                if headers is None:
                    return "Error: 'headers' must be provided when rows_json is a list of lists"
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(headers)
                for row in rows:
                    if not isinstance(row, list):
                        return "Error: Mixed row types; expected list-of-lists"
                    writer.writerow(row)
            else:
                return "Error: rows_json must be a JSON list of objects or list of lists"

        return f"CSV written to '{safe_path}'"
    except Exception as e:
        return f"Error: Failed to write CSV to '{safe_path}': {str(e)}"

def create_file_io_tools():
    """Return list of File I/O tools for easy registration with agents."""
    return [read_text, write_text, read_json, write_json, read_csv, write_csv]