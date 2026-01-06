"""Utility functions for the sample codebase.

This module provides helper utilities used across the application:
- helpers.py: General-purpose helper functions
- formatters.py: Output formatting utilities

Call Graph Patterns:
- helpers.py functions are called by core.processor
- formatters.py functions are called by core.processor
- Some internal calls between helpers
"""

from tests.fixtures.sample_codebase.utils.formatters import (
    format_error,
    format_output,
    format_table,
)
from tests.fixtures.sample_codebase.utils.helpers import (
    deep_merge,
    flatten_dict,
    normalize_string,
    retry_operation,
)

__all__ = [
    "normalize_string",
    "flatten_dict",
    "deep_merge",
    "retry_operation",
    "format_output",
    "format_table",
    "format_error",
]
