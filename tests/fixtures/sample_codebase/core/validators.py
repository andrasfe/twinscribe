"""Input validation functions for the core processing module.

This module provides validation utilities that are called by the processor
and other components. Demonstrates:
- Functions with multiple parameters
- Exception raising patterns
- Conditional validation logic
- Default parameter values

Ground Truth Call Graph:
- validate_input -> validate_range (conditional)
- validate_config -> validate_input (loop)
- validate_range: no outgoing calls
"""

from typing import Any


class ValidationError(Exception):
    """Custom exception for validation failures.

    Attributes:
        field: The name of the field that failed validation.
        message: Human-readable error description.
        value: The invalid value that was provided.
    """

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        """Initialize a ValidationError.

        Args:
            field: The name of the field that failed validation.
            message: Human-readable error description.
            value: The invalid value that was provided.
        """
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


def validate_range(
    value: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
    inclusive: bool = True,
) -> bool:
    """Validate that a numeric value falls within a specified range.

    This is a leaf function with no outgoing calls, used to test
    simple validation logic without dependencies.

    Args:
        value: The numeric value to validate.
        min_value: Minimum allowed value (default: 0.0).
        max_value: Maximum allowed value (default: 100.0).
        inclusive: Whether range bounds are inclusive (default: True).

    Returns:
        True if the value is within the valid range.

    Raises:
        ValidationError: If the value is outside the allowed range.

    Examples:
        >>> validate_range(50)  # Valid
        True
        >>> validate_range(150)  # Raises ValidationError
    """
    if inclusive:
        if min_value <= value <= max_value:
            return True
    else:
        if min_value < value < max_value:
            return True

    raise ValidationError(
        field="value",
        message=f"Value {value} not in range [{min_value}, {max_value}]",
        value=value,
    )


def validate_input(
    data: dict[str, Any],
    required_fields: list[str] | None = None,
    validate_types: bool = True,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Validate input data dictionary with configurable rules.

    This function demonstrates:
    - Multiple validation paths (conditional calls)
    - Optional parameter handling
    - Cross-function calls to validate_range

    Args:
        data: The input dictionary to validate.
        required_fields: List of field names that must be present.
        validate_types: Whether to perform type checking.
        numeric_ranges: Dict mapping field names to (min, max) tuples.

    Returns:
        The validated data dictionary (unchanged if valid).

    Raises:
        ValidationError: If any validation check fails.
        TypeError: If data is not a dictionary.

    Call Graph Edges:
        - validate_input -> validate_range (when numeric_ranges provided)
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    field=field,
                    message="Required field is missing",
                )

    # Validate numeric ranges (calls validate_range)
    if numeric_ranges:
        for field, (min_val, max_val) in numeric_ranges.items():
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    validate_range(value, min_val, max_val)

    return data


def validate_config(
    config: dict[str, Any],
    schema: dict[str, type] | None = None,
    strict: bool = False,
) -> bool:
    """Validate a configuration dictionary against an optional schema.

    Demonstrates:
    - Loop-based validation (iterates over schema)
    - Calls to validate_input within loop
    - Nested data structure handling

    Args:
        config: The configuration dictionary to validate.
        schema: Optional type schema mapping field names to expected types.
        strict: If True, reject unknown fields not in schema.

    Returns:
        True if configuration is valid.

    Raises:
        ValidationError: If configuration is invalid.

    Call Graph Edges:
        - validate_config -> validate_input (in loop for each section)
    """
    # First validate config is dict
    validate_input(config, required_fields=None)

    if schema:
        for field_name, expected_type in schema.items():
            if field_name not in config:
                if strict:
                    raise ValidationError(
                        field=field_name,
                        message="Required by schema",
                    )
                continue

            value = config[field_name]
            if not isinstance(value, expected_type):
                raise ValidationError(
                    field=field_name,
                    message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                    value=value,
                )

        # Check for unknown fields in strict mode
        if strict:
            unknown = set(config.keys()) - set(schema.keys())
            if unknown:
                raise ValidationError(
                    field=", ".join(unknown),
                    message="Unknown fields not allowed in strict mode",
                )

    return True
