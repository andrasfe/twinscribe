"""Sample Codebase Package.

This package provides a realistic sample Python codebase for testing
the dual-stream documentation system. It includes:

Structure:
- core/: Main logic with processor and validators
- utils/: Utility functions (helpers, formatters)
- models/: Data classes and entities
- main.py: Entry point demonstrating typical usage

Testing Patterns:
- Functions with various signatures
- Classes with inheritance
- Call relationships between components
- Various edge cases (decorators, async, generators)
- Direct calls between modules
- Conditional calls (if/else paths)
- Loop-based calls (for/while)
- Recursive calls
- Cross-module calls

The structure mimics a real-world project to test:
- AST parsing and component discovery
- Static call graph analysis
- Documentation generation
- Validation and comparison
"""

# Legacy imports for backward compatibility
from tests.fixtures.sample_codebase.calculator import AdvancedCalculator, Calculator
from tests.fixtures.sample_codebase.data_processor import DataProcessor, ProcessingResult
from tests.fixtures.sample_codebase.utils_legacy import helper_function, validate_input

__version__ = "1.0.0"

__all__ = [
    # Legacy exports
    "Calculator",
    "AdvancedCalculator",
    "DataProcessor",
    "ProcessingResult",
    "helper_function",
    "validate_input",
    # New module exports
    "core",
    "utils",
    "models",
]
