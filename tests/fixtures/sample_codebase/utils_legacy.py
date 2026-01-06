"""
Utility functions for the sample codebase.

This module provides helper functions used throughout the sample codebase.
It demonstrates:
- Simple functions with type hints
- Functions with default arguments
- Functions that raise exceptions
"""

from typing import Any


def helper_function(value: Any) -> str:
    """
    Convert any value to its string representation.

    Args:
        value: Any value to convert to string

    Returns:
        String representation of the value

    Examples:
        >>> helper_function(42)
        '42'
        >>> helper_function([1, 2, 3])
        '[1, 2, 3]'
    """
    return str(value)


def validate_input(data: dict, required_keys: list[str] | None = None) -> bool:
    """
    Validate that input data contains required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of keys that must be present.
            If None, only checks that data is a non-empty dict.

    Returns:
        True if validation passes

    Raises:
        TypeError: If data is not a dictionary
        ValueError: If data is empty or missing required keys

    Examples:
        >>> validate_input({'name': 'test'}, ['name'])
        True
        >>> validate_input({})
        Traceback (most recent call last):
            ...
        ValueError: Data cannot be empty
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    if not data:
        raise ValueError("Data cannot be empty")

    if required_keys:
        missing = set(required_keys) - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

    return True


def format_output(result: Any, precision: int = 2) -> str:
    """
    Format a result value for display.

    Args:
        result: Value to format
        precision: Decimal precision for floats (default: 2)

    Returns:
        Formatted string representation
    """
    if isinstance(result, float):
        return f"{result:.{precision}f}"
    return helper_function(result)


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: The dividend
        denominator: The divisor

    Returns:
        Result of division

    Raises:
        ZeroDivisionError: If denominator is zero
    """
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return numerator / denominator
