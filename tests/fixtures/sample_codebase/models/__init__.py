"""Data models and entities for the sample codebase.

This module provides data classes and enums used throughout
the application. Demonstrates:
- Dataclass patterns
- Enum definitions
- Type annotations
- Factory methods

These are primarily data containers with minimal logic,
representing leaf nodes in the call graph.
"""

from tests.fixtures.sample_codebase.models.entities import (
    Entity,
    ProcessingResult,
    ProcessingStatus,
    User,
)

__all__ = [
    "Entity",
    "User",
    "ProcessingResult",
    "ProcessingStatus",
]
