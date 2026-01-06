"""Core module for main application logic.

This module contains the primary business logic including:
- DataProcessor: Main data processing pipeline
- validators: Input validation utilities

Call Graph Patterns:
- processor.py calls validators.py (cross-module)
- processor.py calls utils.helpers (cross-package)
- processor.py calls models.entities (cross-package)
"""

from tests.fixtures.sample_codebase.core.processor import DataProcessor, ProcessingPipeline
from tests.fixtures.sample_codebase.core.validators import (
    ValidationError,
    validate_config,
    validate_input,
    validate_range,
)

__all__ = [
    "DataProcessor",
    "ProcessingPipeline",
    "validate_input",
    "validate_range",
    "validate_config",
    "ValidationError",
]
