"""Output formatting utilities.

This module provides functions for formatting output data
in various styles. Demonstrates:
- Multiple output format support
- Conditional formatting logic
- Table formatting
- Error message formatting

Ground Truth Call Graph:
- format_output: leaf function (may call json.dumps internally)
- format_table -> format_output (for cell values)
- format_error -> format_output (wraps error data)
"""

import json
from typing import Any


def format_output(
    data: Any,
    style: str = "default",
    indent: int = 2,
    sort_keys: bool = False,
) -> str:
    """Format data for output in various styles.

    This is a primary formatting function used throughout the
    codebase. Demonstrates multiple output formats and
    conditional logic.

    Args:
        data: The data to format (any JSON-serializable type).
        style: Output style - 'default', 'json', 'compact', 'pretty'.
        indent: Indentation level for pretty printing (default: 2).
        sort_keys: Whether to sort dictionary keys (default: False).

    Returns:
        Formatted string representation of the data.

    Examples:
        >>> format_output({"name": "test"}, style="json")
        '{"name": "test"}'
        >>> format_output({"a": 1}, style="pretty", indent=4)
        '{\\n    "a": 1\\n}'

    Note:
        This is a leaf function with no calls to other
        functions in the sample codebase.
    """
    if style == "compact":
        return json.dumps(data, separators=(",", ":"), sort_keys=sort_keys)

    elif style == "json":
        return json.dumps(data, sort_keys=sort_keys)

    elif style == "pretty":
        return json.dumps(data, indent=indent, sort_keys=sort_keys)

    else:  # default
        if isinstance(data, dict):
            pairs = [f"{k}={v!r}" for k, v in data.items()]
            return f"{{{', '.join(pairs)}}}"
        elif isinstance(data, list):
            return f"[{', '.join(repr(item) for item in data)}]"
        else:
            return repr(data)


def format_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    header: bool = True,
    separator: str = "|",
) -> str:
    """Format data as an ASCII table.

    This function demonstrates:
    - Complex string formatting
    - Calls to format_output for cell values
    - Multi-step processing

    Args:
        rows: List of row dictionaries.
        columns: Column names to include (default: all from first row).
        header: Whether to include a header row (default: True).
        separator: Column separator character (default: "|").

    Returns:
        ASCII table string.

    Examples:
        >>> format_table([{"a": 1, "b": 2}])
        '| a | b |\\n|---|---|\\n| 1 | 2 |'

    Call Graph Edges:
        - format_table -> format_output (for complex cell values)
    """
    if not rows:
        return ""

    # Determine columns
    if columns is None:
        columns = list(rows[0].keys())

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            value = row.get(col, "")
            # Use format_output for complex values
            if isinstance(value, (dict, list)):
                formatted = format_output(value, style="compact")
            else:
                formatted = str(value)
            widths[col] = max(widths[col], len(formatted))

    # Build table
    lines = []

    # Header
    if header:
        header_line = separator + separator.join(
            f" {col.center(widths[col])} " for col in columns
        ) + separator
        lines.append(header_line)

        # Separator line
        sep_line = separator + separator.join(
            "-" * (widths[col] + 2) for col in columns
        ) + separator
        lines.append(sep_line)

    # Data rows
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (dict, list)):
                formatted = format_output(value, style="compact")
            else:
                formatted = str(value)
            cells.append(f" {formatted.ljust(widths[col])} ")
        lines.append(separator + separator.join(cells) + separator)

    return "\n".join(lines)


def format_error(
    error: Exception,
    include_traceback: bool = False,
    context: dict[str, Any] | None = None,
) -> str:
    """Format an error for user display.

    Demonstrates:
    - Exception handling and formatting
    - Optional context inclusion
    - Calls to format_output for context data

    Args:
        error: The exception to format.
        include_traceback: Include full traceback (default: False).
        context: Additional context data to include.

    Returns:
        Formatted error string.

    Call Graph Edges:
        - format_error -> format_output (for context data)
    """
    parts = [f"Error: {type(error).__name__}: {error!s}"]

    if include_traceback:
        import traceback
        tb = traceback.format_exc()
        if tb and tb != "NoneType: None\n":
            parts.append(f"\nTraceback:\n{tb}")

    if context:
        # Use format_output for context data
        formatted_context = format_output(context, style="pretty")
        parts.append(f"\nContext:\n{formatted_context}")

    return "\n".join(parts)


def format_duration(seconds: float, precision: int = 2) -> str:
    """Format a duration in seconds to human-readable string.

    Simple leaf function demonstrating time formatting.

    Args:
        seconds: Duration in seconds.
        precision: Decimal places for sub-second values.

    Returns:
        Human-readable duration string.

    Examples:
        >>> format_duration(3661.5)
        '1h 1m 1.50s'
        >>> format_duration(0.123, precision=3)
        '123.000ms'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.{precision}f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.{precision}f}ms"
    elif seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.{precision}f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.{precision}f}s"
