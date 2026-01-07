"""
Advanced Python patterns for testing edge cases.

This module demonstrates patterns that require special handling:
- Decorators (function and class decorators)
- Async/await functions
- Nested classes
- Properties and descriptors
- Context managers
- Generators and async generators
- Multiple inheritance
- Metaclasses
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Generator
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# Decorators
# =============================================================================


def log_calls(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that logs function calls.

    This decorator wraps a function to print its name and arguments
    before execution.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that logs calls
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)

    return wrapper


def retry(max_attempts: int = 3):
    """
    Decorator factory that retries a function on failure.

    Args:
        max_attempts: Maximum number of attempts (default: 3)

    Returns:
        Decorator that implements retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            for _attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
            raise RuntimeError(f"Failed after {max_attempts} attempts") from last_error

        return wrapper

    return decorator


def cache_result(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that caches function results.

    Simple memoization decorator for functions with hashable arguments.

    Args:
        func: Function to cache

    Returns:
        Cached version of the function
    """
    _cache: dict[tuple, Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]

    return wrapper


# =============================================================================
# Async Functions
# =============================================================================


async def fetch_data(url: str, timeout: float = 10.0) -> dict:
    """
    Asynchronously fetch data from a URL.

    This is a mock function demonstrating async patterns.

    Args:
        url: URL to fetch data from
        timeout: Request timeout in seconds

    Returns:
        Dictionary with fetched data

    Raises:
        TimeoutError: If fetch takes longer than timeout
        ValueError: If URL is invalid
    """
    if not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url}")

    await asyncio.sleep(0.01)  # Simulate network delay
    return {"url": url, "status": "success", "data": {}}


async def process_items_async(items: list[dict]) -> list[dict]:
    """
    Process multiple items asynchronously.

    Args:
        items: List of items to process

    Returns:
        List of processed items
    """

    async def process_single(item: dict) -> dict:
        await asyncio.sleep(0.001)
        return {**item, "processed": True}

    tasks = [process_single(item) for item in items]
    return await asyncio.gather(*tasks)


async def stream_results() -> AsyncGenerator[int, None]:
    """
    Async generator that yields results over time.

    Yields:
        Sequential integers with simulated delays
    """
    for i in range(10):
        await asyncio.sleep(0.001)
        yield i


# =============================================================================
# Nested Classes
# =============================================================================


class OuterClass:
    """
    Demonstrates nested class patterns.

    Contains inner classes that need special handling for
    component discovery and documentation.
    """

    CONSTANT = "OUTER_VALUE"

    class InnerClass:
        """Inner class with its own methods."""

        def __init__(self, value: int) -> None:
            """Initialize inner class."""
            self.value = value

        def get_value(self) -> int:
            """Return the stored value."""
            return self.value

        class DeeplyNestedClass:
            """Deeply nested class for edge case testing."""

            def __init__(self, data: str) -> None:
                """Initialize deeply nested class."""
                self.data = data

            def process(self) -> str:
                """Process the data."""
                return self.data.upper()

    class HelperClass:
        """Helper class used by outer class methods."""

        @staticmethod
        def help() -> str:
            """Provide help text."""
            return "Helper available"

    def __init__(self, name: str) -> None:
        """
        Initialize the outer class.

        Args:
            name: Name for this instance
        """
        self.name = name
        self.inner = self.InnerClass(42)
        self.helper = self.HelperClass()

    def use_inner(self) -> int:
        """
        Use the inner class.

        Returns:
            Value from inner class
        """
        return self.inner.get_value()

    def create_nested(self, data: str) -> "OuterClass.InnerClass.DeeplyNestedClass":
        """
        Create a deeply nested class instance.

        Args:
            data: Data for the nested class

        Returns:
            DeeplyNestedClass instance
        """
        return self.InnerClass.DeeplyNestedClass(data)


# =============================================================================
# Properties and Descriptors
# =============================================================================


class Validator:
    """Descriptor that validates assigned values."""

    def __init__(self, min_value: float = 0, max_value: float = 100) -> None:
        """
        Initialize validator with bounds.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner: type, name: str) -> None:
        """Store the attribute name."""
        self.name = name
        self.private_name = f"_validated_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> float:
        """Get the validated value."""
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self.private_name, 0)

    def __set__(self, obj: Any, value: float) -> None:
        """Set and validate the value."""
        if not self.min_value <= value <= self.max_value:
            raise ValueError(f"{self.name} must be between {self.min_value} and {self.max_value}")
        setattr(obj, self.private_name, value)


class ConfigurableComponent:
    """Class demonstrating properties and descriptors."""

    temperature = Validator(0, 100)
    humidity = Validator(0, 100)

    def __init__(self, name: str) -> None:
        """
        Initialize component.

        Args:
            name: Component name
        """
        self._name = name
        self._active = False
        self.temperature = 20.0
        self.humidity = 50.0

    @property
    def name(self) -> str:
        """Get component name (read-only)."""
        return self._name

    @property
    def is_active(self) -> bool:
        """Check if component is active."""
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """
        Set component active state.

        Args:
            value: New active state

        Raises:
            TypeError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("is_active must be a boolean")
        self._active = value

    @property
    def status(self) -> dict:
        """
        Get component status.

        Returns:
            Dictionary with current status
        """
        return {
            "name": self._name,
            "active": self._active,
            "temperature": self.temperature,
            "humidity": self.humidity,
        }


# =============================================================================
# Context Managers
# =============================================================================


class ResourceManager:
    """Context manager for resource handling."""

    def __init__(self, resource_name: str) -> None:
        """
        Initialize resource manager.

        Args:
            resource_name: Name of the resource to manage
        """
        self.resource_name = resource_name
        self.acquired = False

    def __enter__(self) -> "ResourceManager":
        """
        Acquire the resource.

        Returns:
            This manager instance
        """
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Release the resource.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to propagate exceptions
        """
        self.acquired = False
        return False

    def use_resource(self) -> str:
        """
        Use the managed resource.

        Returns:
            Resource status message

        Raises:
            RuntimeError: If resource not acquired
        """
        if not self.acquired:
            raise RuntimeError("Resource not acquired")
        return f"Using {self.resource_name}"


class AsyncResourceManager:
    """Async context manager for async resource handling."""

    def __init__(self, resource_name: str) -> None:
        """Initialize async resource manager."""
        self.resource_name = resource_name
        self.acquired = False

    async def __aenter__(self) -> "AsyncResourceManager":
        """Async acquire the resource."""
        await asyncio.sleep(0.001)
        self.acquired = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async release the resource."""
        await asyncio.sleep(0.001)
        self.acquired = False
        return False


# =============================================================================
# Generators
# =============================================================================


def fibonacci_generator(limit: int) -> Generator[int, None, None]:
    """
    Generate Fibonacci numbers up to a limit.

    Args:
        limit: Maximum value to generate

    Yields:
        Fibonacci numbers not exceeding limit
    """
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b


def bidirectional_generator() -> Generator[int, int | None, str]:
    """
    Generator that receives values and returns a final result.

    Yields:
        Current accumulated value

    Returns:
        Summary string
    """
    total = 0
    count = 0
    while True:
        received = yield total
        if received is None:
            break
        total += received
        count += 1
    return f"Processed {count} values, total={total}"


# =============================================================================
# Multiple Inheritance
# =============================================================================


class Loggable:
    """Mixin that adds logging capability."""

    def log(self, message: str) -> None:
        """Log a message."""
        print(f"[{self.__class__.__name__}] {message}")


class Configurable:
    """Mixin that adds configuration capability."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration."""
        self._config = kwargs
        super().__init__(**kwargs)  # Support MRO

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)


class Cacheable:
    """Mixin that adds caching capability."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize cache."""
        self._cache: dict[str, Any] = {}
        super().__init__(**kwargs)

    def cache_get(self, key: str) -> Any:
        """Get cached value."""
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = value


class AdvancedService(Loggable, Configurable, Cacheable):
    """
    Service class using multiple inheritance.

    Inherits from multiple mixins to demonstrate MRO complexity.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize advanced service.

        Args:
            name: Service name
            **kwargs: Configuration options
        """
        super().__init__(**kwargs)
        self.name = name

    def execute(self, operation: str) -> str:
        """
        Execute an operation.

        Args:
            operation: Operation to execute

        Returns:
            Operation result
        """
        self.log(f"Executing {operation}")

        # Check cache
        cached = self.cache_get(operation)
        if cached:
            self.log("Cache hit")
            return cached

        # Execute
        result = f"{self.name}:{operation}"
        self.cache_set(operation, result)

        return result


# =============================================================================
# Decorated Methods
# =============================================================================


class ServiceWithDecorators:
    """Class demonstrating decorated methods."""

    def __init__(self) -> None:
        """Initialize service."""
        self._calls = 0

    @log_calls
    def logged_method(self, value: int) -> int:
        """Method with logging decorator."""
        return value * 2

    @cache_result
    def cached_method(self, value: int) -> int:
        """Method with caching decorator."""
        self._calls += 1
        return value**2

    @retry(max_attempts=3)
    def unreliable_method(self, fail_count: int = 0) -> str:
        """
        Method that may fail.

        Args:
            fail_count: How many times to fail before succeeding

        Returns:
            Success message
        """
        if self._calls < fail_count:
            self._calls += 1
            raise RuntimeError("Simulated failure")
        return "Success"

    @staticmethod
    @log_calls
    def static_logged_method(value: str) -> str:
        """Static method with decorator."""
        return value.upper()

    @classmethod
    @log_calls
    def class_logged_method(cls, value: str) -> str:
        """Class method with decorator."""
        return f"{cls.__name__}: {value}"
