"""Entity data classes and enums.

This module provides data models used throughout the sample codebase.
Demonstrates:
- Dataclass with default values
- Enum with string values
- Factory methods on classes
- Property accessors
- Type annotations with generics

Ground Truth Call Graph:
- Entity.from_dict: class method, may call __init__
- Entity.to_dict: instance method, no external calls
- User.__post_init__ -> validators.validate_input (if validation enabled)
- ProcessingResult.is_success: property, no external calls
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ProcessingStatus(Enum):
    """Status codes for processing operations.

    This enum demonstrates string-valued enums commonly
    used in API responses and state tracking.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Entity:
    """Base entity class with common fields.

    This dataclass demonstrates:
    - Field defaults with factory functions
    - Property accessors
    - Class methods for construction
    - Instance methods for serialization

    Attributes:
        id: Unique identifier for the entity.
        created_at: Timestamp when entity was created.
        updated_at: Timestamp when entity was last updated.
        metadata: Optional metadata dictionary.

    Call Graph Edges:
        - from_dict: class method (creates new instance)
        - to_dict: instance method (leaf function)
        - age_seconds: property (leaf function)
    """

    id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Calculate age of entity in seconds.

        This is a computed property with no external calls.

        Returns:
            Seconds since entity creation.
        """
        return (datetime.now() - self.created_at).total_seconds()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create an Entity from a dictionary.

        Factory method demonstrating class method pattern
        for alternative construction.

        Args:
            data: Dictionary with entity fields.

        Returns:
            New Entity instance.

        Raises:
            KeyError: If required 'id' field is missing.

        Note:
            This calls __init__ internally but that's not
            typically shown in call graphs.
        """
        return cls(
            id=data["id"],
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary.

        Leaf method with no external calls.

        Returns:
            Dictionary representation of the entity.
        """
        result = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        return result


@dataclass
class User(Entity):
    """User entity with authentication fields.

    Demonstrates:
    - Dataclass inheritance
    - Additional fields beyond parent
    - Post-init validation hook

    Attributes:
        username: Unique username.
        email: User email address.
        is_active: Whether user account is active.
        roles: List of role names assigned to user.
    """

    username: str = ""
    email: str = ""
    is_active: bool = True
    roles: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate user data after initialization.

        This demonstrates the __post_init__ hook for validation.
        Note: In production, this might call validators.validate_input,
        but here we keep it simple for the example.
        """
        if not self.username and not self.email:
            # At least one identifier required
            pass  # Simplified validation for example

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role.

        Simple leaf method demonstrating role checking.

        Args:
            role: Role name to check.

        Returns:
            True if user has the role.
        """
        return role in self.roles

    def add_role(self, role: str) -> None:
        """Add a role to the user.

        Modifies internal state, no external calls.

        Args:
            role: Role name to add.
        """
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.now()


@dataclass
class ProcessingResult(Generic[T]):
    """Result container for processing operations.

    Generic dataclass demonstrating:
    - Type parameters (generic over T)
    - Status tracking with enum
    - Optional error information
    - Property accessors

    Attributes:
        status: Processing status code.
        data: Result data (type T) if successful.
        message: Human-readable status message.
        error: Exception if processing failed.
        duration_ms: Processing duration in milliseconds.

    Call Graph Edges:
        - is_success: property (leaf)
        - is_failure: property (leaf)
        - unwrap: method that may raise
    """

    status: ProcessingStatus
    data: T | None = None
    message: str = ""
    error: Exception | None = None
    duration_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        """Check if processing was successful.

        Property accessor with no external calls.

        Returns:
            True if status is SUCCESS.
        """
        return self.status == ProcessingStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if processing failed.

        Property accessor with no external calls.

        Returns:
            True if status is FAILED.
        """
        return self.status == ProcessingStatus.FAILED

    def unwrap(self) -> T:
        """Get result data, raising if failed.

        Method demonstrating unwrap pattern from Rust/Option.

        Returns:
            The result data if successful.

        Raises:
            RuntimeError: If processing failed or data is None.
        """
        if self.is_failure:
            raise RuntimeError(f"Cannot unwrap failed result: {self.message}") from self.error

        if self.data is None:
            raise RuntimeError("Result data is None")

        return self.data

    def map(self, func: Any) -> "ProcessingResult[Any]":
        """Transform successful result data.

        Demonstrates functional programming pattern.
        Creates new result with transformed data.

        Args:
            func: Function to apply to data if successful.

        Returns:
            New ProcessingResult with transformed data or same error.
        """
        if self.is_success and self.data is not None:
            return ProcessingResult(
                status=self.status,
                data=func(self.data),
                message=self.message,
                duration_ms=self.duration_ms,
            )
        return ProcessingResult(
            status=self.status,
            data=None,
            message=self.message,
            error=self.error,
            duration_ms=self.duration_ms,
        )
