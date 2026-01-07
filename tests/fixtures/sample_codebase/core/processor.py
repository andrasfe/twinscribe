"""Main data processing logic for the sample codebase.

This module contains the primary processing pipeline that demonstrates:
- Class-based processing with inheritance
- Cross-module dependencies (validators, helpers, formatters, entities)
- Async processing patterns
- Recursive processing for nested data
- Multiple return paths and exception handling

Ground Truth Call Graph Edges:
- DataProcessor.__init__ -> validators.validate_config
- DataProcessor.process -> validators.validate_input
- DataProcessor.process -> helpers.normalize_string
- DataProcessor.process -> ProcessingPipeline.run
- DataProcessor._process_item -> formatters.format_output
- DataProcessor._process_item -> _process_item (recursive for nested)
- ProcessingPipeline.run -> validators.validate_input
- ProcessingPipeline.run -> formatters.format_output
- ProcessingPipeline.add_step -> nothing (leaf)
"""

from collections.abc import Callable
from typing import Any

from tests.fixtures.sample_codebase.core.validators import (
    validate_config,
    validate_input,
)
from tests.fixtures.sample_codebase.models.entities import ProcessingResult, ProcessingStatus
from tests.fixtures.sample_codebase.utils.formatters import format_output
from tests.fixtures.sample_codebase.utils.helpers import normalize_string


class ProcessingPipeline:
    """A sequential processing pipeline with configurable steps.

    This class demonstrates:
    - Step-based processing pattern
    - Callable storage and execution
    - Cross-module calls to validators and formatters

    Attributes:
        steps: List of processing step functions.
        name: Human-readable pipeline name.

    Call Graph Edges:
        - run -> validate_input
        - run -> format_output
        - add_step: leaf function
    """

    def __init__(self, name: str = "default") -> None:
        """Initialize a processing pipeline.

        Args:
            name: Human-readable name for the pipeline.
        """
        self.steps: list[Callable[[Any], Any]] = []
        self.name = name

    def add_step(self, step: Callable[[Any], Any]) -> "ProcessingPipeline":
        """Add a processing step to the pipeline.

        This is a leaf function - no outgoing calls.

        Args:
            step: A callable that takes and returns data.

        Returns:
            Self for method chaining.
        """
        self.steps.append(step)
        return self

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute all pipeline steps on the input data.

        Demonstrates:
        - Loop-based processing
        - Calls to external modules (validators, formatters)
        - Multiple return paths (empty pipeline, processed)

        Args:
            data: Input data dictionary to process.

        Returns:
            Processed data after all steps.

        Raises:
            ValidationError: If input validation fails.

        Call Graph Edges:
            - run -> validate_input
            - run -> format_output (at end)
        """
        # Validate input before processing
        validated_data = validate_input(data, required_fields=None)

        # Run each step
        result = validated_data
        for step in self.steps:
            result = step(result)

        # Format output
        formatted = format_output(result, style="json")
        return {"result": formatted, "pipeline": self.name}


class DataProcessor:
    """Main data processor with configurable pipeline.

    This class is the central component demonstrating:
    - Constructor validation
    - Method calling other methods
    - Cross-module dependencies
    - Recursive processing for nested structures
    - Exception handling and multiple return paths

    Attributes:
        config: Validated configuration dictionary.
        pipeline: Processing pipeline instance.
        processed_count: Number of items processed.

    Ground Truth Call Graph:
        - __init__ -> validate_config
        - process -> validate_input
        - process -> normalize_string
        - process -> pipeline.run
        - _process_item -> format_output
        - _process_item -> _process_item (recursive)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        pipeline: ProcessingPipeline | None = None,
    ) -> None:
        """Initialize the data processor with configuration.

        Args:
            config: Optional configuration dictionary.
            pipeline: Optional pre-configured pipeline.

        Raises:
            ValidationError: If config validation fails.

        Call Graph Edges:
            - __init__ -> validate_config
        """
        self.config = config or {}

        # Validate configuration (cross-module call)
        if self.config:
            validate_config(self.config, strict=False)

        self.pipeline = pipeline or ProcessingPipeline("default")
        self.processed_count = 0

    def process(
        self,
        data: dict[str, Any],
        normalize: bool = True,
        use_pipeline: bool = True,
    ) -> ProcessingResult:
        """Process input data through the configured pipeline.

        This method demonstrates:
        - Multiple optional parameters
        - Conditional calls based on parameters
        - Cross-module dependencies
        - Result wrapping in data class

        Args:
            data: Input data dictionary to process.
            normalize: Whether to normalize string fields.
            use_pipeline: Whether to use the pipeline or direct processing.

        Returns:
            ProcessingResult with status and output data.

        Raises:
            ValidationError: If input validation fails.

        Call Graph Edges:
            - process -> validate_input
            - process -> normalize_string (conditional)
            - process -> pipeline.run (conditional)
            - process -> _process_item (alternative path)
        """
        # Validate input (cross-module call)
        validated = validate_input(data, required_fields=["id"])

        # Optionally normalize strings (cross-module call)
        if normalize:
            for key, value in validated.items():
                if isinstance(value, str):
                    validated[key] = normalize_string(value)

        # Process through pipeline or direct
        if use_pipeline:
            result = self.pipeline.run(validated)
        else:
            result = self._process_item(validated)

        self.processed_count += 1

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=result,
            message=f"Processed item {validated.get('id')}",
        )

    def _process_item(
        self,
        item: dict[str, Any],
        depth: int = 0,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Process a single item, recursively handling nested structures.

        This private method demonstrates:
        - Recursive processing with depth limit
        - Self-reference in call graph
        - Format output call

        Args:
            item: The item dictionary to process.
            depth: Current recursion depth.
            max_depth: Maximum allowed recursion depth.

        Returns:
            Processed item dictionary.

        Raises:
            RecursionError: If max_depth is exceeded.

        Call Graph Edges:
            - _process_item -> format_output
            - _process_item -> _process_item (recursive)
        """
        if depth > max_depth:
            raise RecursionError(f"Max processing depth {max_depth} exceeded")

        result = {}
        for key, value in item.items():
            if isinstance(value, dict):
                # Recursive call for nested dicts
                result[key] = self._process_item(value, depth + 1, max_depth)
            elif isinstance(value, list):
                # Process list items
                result[key] = [
                    self._process_item(v, depth + 1, max_depth) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = value

        # Format the result (cross-module call)
        formatted = format_output(result)
        return {"processed": formatted, "depth": depth}

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics.

        This is a simple leaf method with no outgoing calls.

        Returns:
            Dictionary with processing statistics.
        """
        return {
            "processed_count": self.processed_count,
            "pipeline_name": self.pipeline.name,
            "config_keys": list(self.config.keys()),
        }
