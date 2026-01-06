"""
Calculator classes for the sample codebase.

This module demonstrates:
- Classes with __init__ methods
- Instance methods with type hints
- Class inheritance
- Methods that call other methods (internal calls)
- Methods that call functions from other modules (external calls)
"""

from collections.abc import Sequence

from tests.fixtures.sample_codebase.utils_legacy import format_output, helper_function, safe_divide


class Calculator:
    """
    Simple calculator with basic arithmetic operations.

    This class provides basic mathematical operations with
    configurable precision for results.

    Attributes:
        precision: Number of decimal places for results
    """

    def __init__(self, precision: int = 2) -> None:
        """
        Initialize the calculator.

        Args:
            precision: Decimal precision for results (default: 2)
        """
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b, rounded to precision
        """
        result = a + b
        return round(result, self.precision)

    def subtract(self, a: float, b: float) -> float:
        """
        Subtract b from a.

        Args:
            a: Number to subtract from
            b: Number to subtract

        Returns:
            Difference of a and b, rounded to precision
        """
        result = a - b
        return round(result, self.precision)

    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.

        Args:
            a: First factor
            b: Second factor

        Returns:
            Product of a and b, rounded to precision
        """
        result = a * b
        # Uses helper_function from utils
        formatted = helper_function(result)
        return round(float(formatted), self.precision)

    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.

        Args:
            a: Dividend
            b: Divisor

        Returns:
            Quotient of a and b, rounded to precision

        Raises:
            ZeroDivisionError: If b is zero
        """
        # Uses safe_divide from utils
        result = safe_divide(a, b)
        return round(result, self.precision)


class AdvancedCalculator(Calculator):
    """
    Extended calculator with advanced mathematical operations.

    Inherits basic operations from Calculator and adds
    more complex functionality like power and statistics.
    """

    def power(self, base: float, exponent: float) -> float:
        """
        Calculate base raised to the power of exponent.

        Args:
            base: The base number
            exponent: The power to raise to

        Returns:
            base^exponent, rounded to precision
        """
        result = base ** exponent
        return round(result, self.precision)

    def compute_sum(self, values: Sequence[float]) -> float:
        """
        Compute the sum of a sequence of values.

        This method demonstrates calling inherited methods in a loop.

        Args:
            values: Sequence of numbers to sum

        Returns:
            Sum of all values

        Raises:
            ValueError: If values is empty
        """
        if not values:
            raise ValueError("Cannot compute sum of empty sequence")

        total = 0.0
        for value in values:
            # Calls inherited add method
            total = self.add(total, value)
        return total

    def compute_product(self, values: Sequence[float]) -> float:
        """
        Compute the product of a sequence of values.

        Args:
            values: Sequence of numbers to multiply

        Returns:
            Product of all values

        Raises:
            ValueError: If values is empty
        """
        if not values:
            raise ValueError("Cannot compute product of empty sequence")

        result = 1.0
        for value in values:
            # Calls inherited multiply method
            result = self.multiply(result, value)
        return result

    def mean(self, values: Sequence[float]) -> float:
        """
        Calculate the arithmetic mean of values.

        Args:
            values: Sequence of numbers

        Returns:
            Mean of the values

        Raises:
            ValueError: If values is empty
        """
        total = self.compute_sum(values)
        return self.divide(total, len(values))

    def format_result(self, value: float) -> str:
        """
        Format a result value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        # Uses format_output from utils
        return format_output(value, self.precision)


class ScientificCalculator(AdvancedCalculator):
    """
    Scientific calculator with trigonometric and logarithmic functions.

    Demonstrates deeper inheritance hierarchy.
    """

    def __init__(self, precision: int = 6) -> None:
        """
        Initialize with higher default precision.

        Args:
            precision: Decimal precision (default: 6 for scientific use)
        """
        super().__init__(precision)

    def square_root(self, value: float) -> float:
        """
        Calculate the square root of a value.

        Args:
            value: Non-negative number

        Returns:
            Square root of value

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Cannot compute square root of negative number")
        return self.power(value, 0.5)

    def factorial(self, n: int) -> int:
        """
        Calculate factorial of n.

        Args:
            n: Non-negative integer

        Returns:
            n! (n factorial)

        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result = int(self.multiply(result, i))
        return result
