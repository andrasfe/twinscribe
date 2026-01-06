"""General-purpose helper functions.

This module provides utility functions that are used throughout the
sample codebase. Demonstrates:
- Simple utility functions (leaf nodes)
- Functions calling other functions
- Recursive functions
- Decorator patterns
- Exception handling

Ground Truth Call Graph:
- normalize_string: leaf function
- flatten_dict -> flatten_dict (recursive)
- deep_merge -> deep_merge (recursive)
- retry_operation: higher-order function (calls provided callable)
"""

import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def normalize_string(
    value: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    remove_special: bool = False,
) -> str:
    """Normalize a string value using configurable rules.

    This is a leaf function with no outgoing calls to other
    functions in the codebase. Used to test basic function
    documentation.

    Args:
        value: The string to normalize.
        lowercase: Convert to lowercase (default: True).
        strip_whitespace: Strip leading/trailing whitespace (default: True).
        remove_special: Remove special characters (default: False).

    Returns:
        The normalized string.

    Examples:
        >>> normalize_string("  HELLO World  ")
        'hello world'
        >>> normalize_string("Test@123", remove_special=True)
        'test123'
    """
    result = value

    if strip_whitespace:
        result = result.strip()

    if lowercase:
        result = result.lower()

    if remove_special:
        result = re.sub(r"[^a-zA-Z0-9\s]", "", result)

    return result


def flatten_dict(
    nested: dict[str, Any],
    separator: str = ".",
    parent_key: str = "",
) -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    This function demonstrates recursive self-calls for processing
    nested structures.

    Args:
        nested: The nested dictionary to flatten.
        separator: String to join nested keys (default: ".").
        parent_key: Prefix for keys (used in recursion).

    Returns:
        Flattened dictionary with compound keys.

    Examples:
        >>> flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
        {'a.b': 1, 'a.c.d': 2}

    Call Graph Edges:
        - flatten_dict -> flatten_dict (recursive for nested dicts)
    """
    items: list[tuple[str, Any]] = []

    for key, value in nested.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
    merge_lists: bool = False,
) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    This function demonstrates:
    - Recursive merging of nested structures
    - Multiple conditional branches
    - Type checking at runtime

    Args:
        base: The base dictionary.
        override: Dictionary with values to merge/override.
        merge_lists: If True, concatenate lists instead of replacing.

    Returns:
        New dictionary with merged values.

    Examples:
        >>> deep_merge({"a": 1}, {"b": 2})
        {'a': 1, 'b': 2}
        >>> deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
        {'a': {'x': 1, 'y': 2}}

    Call Graph Edges:
        - deep_merge -> deep_merge (recursive for nested dicts)
    """
    result = base.copy()

    for key, value in override.items():
        if key in result:
            base_value = result[key]

            # Both are dicts: recursive merge
            if isinstance(base_value, dict) and isinstance(value, dict):
                result[key] = deep_merge(base_value, value, merge_lists)

            # Both are lists and merge_lists is True: concatenate
            elif isinstance(base_value, list) and isinstance(value, list) and merge_lists:
                result[key] = base_value + value

            # Otherwise: override takes precedence
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def retry_operation(
    operation: Callable[[], T],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry an operation with exponential backoff.

    This is a higher-order function that accepts a callable and
    executes it with retry logic. Demonstrates:
    - Higher-order function patterns
    - Exception handling with multiple exception types
    - Configurable retry behavior

    Args:
        operation: Callable to execute (no arguments).
        max_retries: Maximum number of retry attempts (default: 3).
        delay: Initial delay between retries in seconds (default: 1.0).
        backoff: Multiplier for delay after each retry (default: 2.0).
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        The result of the operation if successful.

    Raises:
        Exception: The last exception if all retries fail.

    Examples:
        >>> retry_operation(lambda: api_call(), max_retries=5)
        # Will retry api_call up to 5 times on failure

    Note:
        The operation callable is not part of the static call graph
        since it's provided at runtime.
    """
    last_exception: Exception | None = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(current_delay)
                current_delay *= backoff

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed with no exception")


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of specified size.

    Simple utility function demonstrating list processing
    without external calls.

    Args:
        items: List to split into chunks.
        chunk_size: Maximum size of each chunk.

    Returns:
        List of chunks (sub-lists).

    Raises:
        ValueError: If chunk_size is less than 1.

    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
