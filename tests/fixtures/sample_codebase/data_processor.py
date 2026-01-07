"""
Data processing classes for the sample codebase.

This module demonstrates:
- Dataclasses for structured data
- Async methods
- Context managers
- Property decorators
- Static methods and class methods
- Complex call patterns (callbacks, conditionals)
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tests.fixtures.sample_codebase.utils_legacy import helper_function, validate_input


@dataclass
class ProcessingResult:
    """
    Result of a data processing operation.

    Attributes:
        success: Whether processing succeeded
        data: Processed data (if successful)
        error: Error message (if failed)
        metrics: Processing metrics
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if result represents valid processed data."""
        return self.success and self.data is not None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metrics": self.metrics,
        }


class DataProcessor:
    """
    Process and transform data with validation and caching.

    This class demonstrates complex patterns including:
    - Instance state management
    - Async processing
    - Callback invocation
    - Method chaining through internal calls
    """

    def __init__(self, config: dict[str, Any] | None = None, strict_mode: bool = False) -> None:
        """
        Initialize the data processor.

        Args:
            config: Optional configuration dictionary
            strict_mode: If True, fail on any validation warning
        """
        self.config = config or {}
        self.strict_mode = strict_mode
        self._cache: dict[str, ProcessingResult] = {}
        self._callbacks: list[Callable[[ProcessingResult], None]] = []

    @property
    def cache_size(self) -> int:
        """Return the number of cached results."""
        return len(self._cache)

    def register_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """
        Register a callback to be invoked after processing.

        Args:
            callback: Function to call with processing result
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: ProcessingResult) -> None:
        """Invoke all registered callbacks with the result."""
        for callback in self._callbacks:
            callback(result)

    def process(self, data: dict[str, Any]) -> ProcessingResult:
        """
        Process input data synchronously.

        This method:
        1. Validates input
        2. Transforms data
        3. Caches result
        4. Notifies callbacks

        Args:
            data: Input data dictionary

        Returns:
            ProcessingResult with processed data or error

        Raises:
            TypeError: If data is not a dictionary
        """
        # Input validation
        try:
            validate_input(data)
        except (TypeError, ValueError) as e:
            result = ProcessingResult(success=False, error=str(e))
            self._notify_callbacks(result)
            return result

        # Transform data
        transformed = self._transform(data)

        # Build result
        result = ProcessingResult(
            success=True, data=transformed, metrics={"items_processed": len(data)}
        )

        # Cache result
        cache_key = self._compute_cache_key(data)
        self._cache[cache_key] = result

        # Notify callbacks
        self._notify_callbacks(result)

        return result

    async def process_async(self, data: dict[str, Any], delay: float = 0.0) -> ProcessingResult:
        """
        Process input data asynchronously.

        Args:
            data: Input data dictionary
            delay: Optional delay in seconds (for testing)

        Returns:
            ProcessingResult with processed data or error
        """
        if delay > 0:
            await asyncio.sleep(delay)

        # Reuse synchronous processing
        return self.process(data)

    async def process_batch(self, items: list[dict[str, Any]]) -> list[ProcessingResult]:
        """
        Process multiple items concurrently.

        Args:
            items: List of data dictionaries to process

        Returns:
            List of processing results
        """
        tasks = [self.process_async(item) for item in items]
        return await asyncio.gather(*tasks)

    def _transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply transformations to input data.

        Args:
            data: Raw input data

        Returns:
            Transformed data dictionary
        """
        result = {}
        for key, value in data.items():
            # Transform strings to uppercase
            if isinstance(value, str):
                result[key] = value.upper()
            # Format numbers using helper
            elif isinstance(value, (int, float)):
                result[key] = helper_function(value)
            # Recursively transform nested dicts
            elif isinstance(value, dict):
                result[key] = self._transform(value)
            else:
                result[key] = value
        return result

    def _compute_cache_key(self, data: dict[str, Any]) -> str:
        """Compute a cache key for the given data."""
        # Use helper function for string conversion
        return helper_function(sorted(data.items()))

    @staticmethod
    def create_default() -> "DataProcessor":
        """
        Factory method to create a processor with default settings.

        Returns:
            DataProcessor with default configuration
        """
        return DataProcessor(config={"mode": "default"})

    @classmethod
    def create_strict(cls) -> "DataProcessor":
        """
        Factory method to create a strict-mode processor.

        Returns:
            DataProcessor with strict validation
        """
        return cls(strict_mode=True)

    def clear_cache(self) -> int:
        """
        Clear the result cache.

        Returns:
            Number of items cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_cached(self, data: dict[str, Any]) -> ProcessingResult | None:
        """
        Retrieve cached result for given data.

        Args:
            data: Input data to look up

        Returns:
            Cached result if found, None otherwise
        """
        cache_key = self._compute_cache_key(data)
        return self._cache.get(cache_key)


class StreamingProcessor(DataProcessor):
    """
    Data processor that supports streaming operations.

    Extends DataProcessor with generator-based processing.
    """

    def __init__(self, chunk_size: int = 100, **kwargs: Any) -> None:
        """
        Initialize streaming processor.

        Args:
            chunk_size: Number of items per chunk
            **kwargs: Additional arguments for parent
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def process_stream(self, items: list[dict[str, Any]]):
        """
        Process items in a streaming fashion.

        This is a generator that yields results as they're processed.

        Args:
            items: List of items to process

        Yields:
            ProcessingResult for each item
        """
        for item in items:
            yield self.process(item)

    def process_chunks(self, items: list[dict[str, Any]]):
        """
        Process items in chunks.

        Args:
            items: List of items to process

        Yields:
            List of ProcessingResults for each chunk
        """
        for i in range(0, len(items), self.chunk_size):
            chunk = items[i : i + self.chunk_size]
            yield [self.process(item) for item in chunk]
