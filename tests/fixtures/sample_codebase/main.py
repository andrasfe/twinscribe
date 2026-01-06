"""Main entry point for the sample codebase.

This module demonstrates typical application entry point patterns:
- Configuration loading
- Component initialization
- Main processing loop
- Error handling
- Cleanup

Ground Truth Call Graph Edges:
- main -> load_config
- main -> DataProcessor.__init__
- main -> process_batch
- load_config -> validators.validate_config
- process_batch -> DataProcessor.process (in loop)
- process_batch -> format_output (for results)
- run_async_processing -> DataProcessor.process (async)
"""

import asyncio
from typing import Any

from tests.fixtures.sample_codebase.core.processor import DataProcessor, ProcessingPipeline
from tests.fixtures.sample_codebase.core.validators import ValidationError, validate_config
from tests.fixtures.sample_codebase.models.entities import ProcessingResult, ProcessingStatus
from tests.fixtures.sample_codebase.utils.formatters import format_output
from tests.fixtures.sample_codebase.utils.helpers import deep_merge, retry_operation


def load_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate application configuration.

    This function demonstrates:
    - Default configuration with overrides
    - Configuration validation
    - Cross-module calls to validators

    Args:
        config_path: Optional path to config file (not implemented).
        overrides: Dictionary of config overrides.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValidationError: If configuration is invalid.

    Call Graph Edges:
        - load_config -> validate_config
        - load_config -> deep_merge
    """
    # Default configuration
    default_config = {
        "batch_size": 100,
        "timeout_seconds": 30,
        "retry_count": 3,
        "output_format": "json",
        "debug": False,
    }

    # Merge with overrides
    if overrides:
        config = deep_merge(default_config, overrides)
    else:
        config = default_config

    # Validate (cross-module call)
    validate_config(
        config,
        schema={
            "batch_size": int,
            "timeout_seconds": int,
            "retry_count": int,
            "output_format": str,
            "debug": bool,
        },
        strict=False,
    )

    return config


def process_batch(
    items: list[dict[str, Any]],
    processor: DataProcessor,
    stop_on_error: bool = False,
) -> list[ProcessingResult[dict[str, Any]]]:
    """Process a batch of items through the processor.

    Demonstrates:
    - Loop-based processing with method calls
    - Error handling and accumulation
    - Conditional control flow

    Args:
        items: List of items to process.
        processor: DataProcessor instance to use.
        stop_on_error: If True, stop on first error.

    Returns:
        List of ProcessingResult for each item.

    Call Graph Edges:
        - process_batch -> DataProcessor.process (in loop)
        - process_batch -> format_output (for error results)
    """
    results: list[ProcessingResult[dict[str, Any]]] = []

    for item in items:
        try:
            result = processor.process(item)
            results.append(result)

        except ValidationError as e:
            error_result = ProcessingResult(
                status=ProcessingStatus.FAILED,
                data=None,
                message=format_output({"error": str(e)}, style="compact"),
                error=e,
            )
            results.append(error_result)

            if stop_on_error:
                break

        except Exception as e:  # noqa: BLE001
            error_result = ProcessingResult(
                status=ProcessingStatus.FAILED,
                data=None,
                message=f"Unexpected error: {e}",
                error=e,
            )
            results.append(error_result)

            if stop_on_error:
                break

    return results


async def run_async_processing(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    concurrency: int = 5,
) -> list[ProcessingResult[dict[str, Any]]]:
    """Process items asynchronously with concurrency limit.

    Demonstrates:
    - Async processing patterns
    - Semaphore-based concurrency control
    - Asyncio gather for parallel execution

    Args:
        items: List of items to process.
        config: Configuration dictionary.
        concurrency: Maximum concurrent operations.

    Returns:
        List of ProcessingResult for each item.

    Call Graph Edges:
        - run_async_processing -> DataProcessor.__init__
        - run_async_processing -> DataProcessor.process (via _process_one)
    """
    processor = DataProcessor(config)
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_one(item: dict[str, Any]) -> ProcessingResult[dict[str, Any]]:
        async with semaphore:
            # Simulate async by running sync in executor
            # In real code, this would be truly async
            return processor.process(item)

    tasks = [_process_one(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to failed results
    processed: list[ProcessingResult[dict[str, Any]]] = []
    for result in results:
        if isinstance(result, Exception):
            processed.append(
                ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error=result,
                    message=str(result),
                )
            )
        else:
            processed.append(result)

    return processed


def main(
    items: list[dict[str, Any]] | None = None,
    config_overrides: dict[str, Any] | None = None,
    use_async: bool = False,
) -> dict[str, Any]:
    """Main entry point for the sample application.

    This function orchestrates the full processing workflow:
    1. Load and validate configuration
    2. Initialize processor with pipeline
    3. Process all items
    4. Format and return results

    Args:
        items: List of items to process (uses sample data if None).
        config_overrides: Configuration overrides.
        use_async: Use async processing (default: False).

    Returns:
        Dictionary with processing results and statistics.

    Raises:
        ValidationError: If configuration is invalid.

    Call Graph Edges:
        - main -> load_config
        - main -> DataProcessor.__init__
        - main -> ProcessingPipeline.__init__
        - main -> process_batch (or run_async_processing)
        - main -> format_output
    """
    # Load configuration
    config = load_config(overrides=config_overrides)

    # Sample data if none provided
    if items is None:
        items = [
            {"id": "1", "name": "Test Item 1", "value": 100},
            {"id": "2", "name": "Test Item 2", "value": 200},
            {"id": "3", "name": "Test Item 3", "value": 300},
        ]

    # Initialize pipeline and processor
    pipeline = ProcessingPipeline(name="main-pipeline")
    processor = DataProcessor(config=config, pipeline=pipeline)

    # Process items
    if use_async:
        results = asyncio.run(run_async_processing(items, config))
    else:
        results = process_batch(items, processor)

    # Summarize results
    success_count = sum(1 for r in results if r.is_success)
    failure_count = sum(1 for r in results if r.is_failure)

    summary = {
        "total_items": len(items),
        "success_count": success_count,
        "failure_count": failure_count,
        "processor_stats": processor.get_stats(),
        "results": [
            {
                "status": r.status.value,
                "message": r.message,
                "data": r.data,
            }
            for r in results
        ],
    }

    # Format output
    formatted = format_output(summary, style=config.get("output_format", "json"))

    return {
        "summary": summary,
        "formatted_output": formatted,
    }


def run_with_retry(
    items: list[dict[str, Any]],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run main processing with retry logic.

    Demonstrates higher-order function usage with retry.

    Args:
        items: Items to process.
        max_retries: Maximum retry attempts.

    Returns:
        Processing results.

    Call Graph Edges:
        - run_with_retry -> retry_operation
        - run_with_retry -> main (via retry_operation callback)
    """
    return retry_operation(
        operation=lambda: main(items=items),
        max_retries=max_retries,
        delay=1.0,
        exceptions=(ValidationError, RuntimeError),
    )


if __name__ == "__main__":
    # Example usage
    result = main()
    print(result["formatted_output"])
