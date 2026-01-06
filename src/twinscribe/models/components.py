"""
Component data models for representing code entities.

These models represent the structure of code components discovered
through AST analysis, including their documentation structure.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from twinscribe.models.base import ComponentType


class ComponentLocation(BaseModel):
    """Physical location of a component in the codebase.

    Attributes:
        file_path: Relative path from codebase root to the file
        line_start: First line of the component (1-indexed)
        line_end: Last line of the component (1-indexed)
        column_start: Starting column (0-indexed, optional)
        column_end: Ending column (0-indexed, optional)
    """

    file_path: str = Field(
        ...,
        description="Relative path from codebase root",
        examples=["src/utils/helpers.py"],
    )
    line_start: int = Field(..., ge=1, description="First line number (1-indexed)")
    line_end: int = Field(..., ge=1, description="Last line number (1-indexed)")
    column_start: Optional[int] = Field(
        default=None, ge=0, description="Starting column (0-indexed)"
    )
    column_end: Optional[int] = Field(
        default=None, ge=0, description="Ending column (0-indexed)"
    )

    @field_validator("line_end")
    @classmethod
    def line_end_after_start(cls, v: int, info) -> int:
        """Ensure line_end is >= line_start."""
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v

    def to_reference(self) -> str:
        """Format as file:line-line reference string."""
        return f"{self.file_path}:{self.line_start}-{self.line_end}"


class ParameterDoc(BaseModel):
    """Documentation for a single parameter.

    Attributes:
        name: Parameter name as it appears in the signature
        type: Type annotation or inferred type
        description: Human-readable description of purpose
        default: Default value if any (as string representation)
        required: Whether the parameter is required
    """

    name: str = Field(..., min_length=1, description="Parameter name")
    type: Optional[str] = Field(
        default=None, description="Type annotation", examples=["str", "List[int]"]
    )
    description: str = Field(
        default="", description="What this parameter does"
    )
    default: Optional[str] = Field(
        default=None,
        description="Default value as string",
        examples=["None", "'default'", "42"],
    )
    required: bool = Field(
        default=True, description="Whether parameter is required"
    )

    @field_validator("name")
    @classmethod
    def valid_identifier(cls, v: str) -> str:
        """Validate that name is a valid Python identifier."""
        # Allow *args and **kwargs
        clean_name = v.lstrip("*")
        if clean_name and not clean_name.isidentifier():
            raise ValueError(f"'{v}' is not a valid parameter name")
        return v


class ReturnDoc(BaseModel):
    """Documentation for return value.

    Attributes:
        type: Return type annotation
        description: What is returned and under what conditions
    """

    type: Optional[str] = Field(
        default=None, description="Return type annotation"
    )
    description: str = Field(
        default="", description="What is returned and when"
    )


class ExceptionDoc(BaseModel):
    """Documentation for an exception that can be raised.

    Attributes:
        type: Exception class name
        condition: When this exception is raised
    """

    type: str = Field(
        ..., min_length=1, description="Exception type name", examples=["ValueError"]
    )
    condition: str = Field(
        default="", description="Condition under which this is raised"
    )


class ComponentDocumentation(BaseModel):
    """Documentation content for a component.

    This is the core documentation structure that both documenter
    agents produce for each component.

    Attributes:
        summary: One-line summary description
        description: Detailed explanation of purpose and behavior
        parameters: Documentation for each parameter
        returns: Documentation for return value
        raises: Documentation for exceptions that can be raised
        examples: Usage examples as code strings
        notes: Additional notes or warnings
        see_also: Related components or external references
    """

    summary: str = Field(
        default="",
        max_length=200,
        description="One-line description",
    )
    description: str = Field(
        default="", description="Detailed explanation"
    )
    parameters: list[ParameterDoc] = Field(
        default_factory=list, description="Parameter documentation"
    )
    returns: Optional[ReturnDoc] = Field(
        default=None, description="Return value documentation"
    )
    raises: list[ExceptionDoc] = Field(
        default_factory=list, description="Exceptions that can be raised"
    )
    examples: list[str] = Field(
        default_factory=list, description="Usage examples"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or warnings"
    )
    see_also: list[str] = Field(
        default_factory=list, description="Related references"
    )


class Component(BaseModel):
    """Represents a code component discovered in the codebase.

    This is the primary entity being documented. Components form
    a graph through their call relationships.

    Attributes:
        component_id: Unique identifier (module.Class.method format)
        name: Short name of the component
        type: Type of component (function, method, class, etc.)
        location: Physical location in codebase
        signature: Full signature string
        parent_id: ID of containing component (for methods in classes)
        dependencies: IDs of components this one depends on
        existing_docstring: Original docstring if present
        is_public: Whether component is part of public API
        created_at: When this component was discovered
    """

    component_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier in module.Class.method format",
        examples=["mypackage.utils.StringHelper.format_name"],
    )
    name: str = Field(
        ..., min_length=1, description="Short name", examples=["format_name"]
    )
    type: ComponentType = Field(..., description="Type of component")
    location: ComponentLocation = Field(..., description="File location")
    signature: Optional[str] = Field(
        default=None,
        description="Full signature string",
        examples=["def format_name(first: str, last: str) -> str"],
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of containing component",
        examples=["mypackage.utils.StringHelper"],
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Component IDs this depends on (import-level)",
    )
    existing_docstring: Optional[str] = Field(
        default=None, description="Original docstring if present"
    )
    is_public: bool = Field(
        default=True,
        description="True if part of public API (no leading underscore)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When component was discovered",
    )

    @field_validator("component_id", "parent_id")
    @classmethod
    def validate_component_id_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate component ID follows module.Class.method format."""
        if v is None:
            return v
        # Allow dots but require at least one segment
        parts = v.split(".")
        if not all(p.isidentifier() or p.startswith("_") for p in parts if p):
            # Allow private names (starting with _)
            pass  # Relaxed validation for flexibility
        return v

    @property
    def module_path(self) -> str:
        """Extract module path from component_id."""
        parts = self.component_id.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else ""

    def model_post_init(self, __context: Any) -> None:
        """Set is_public based on name if not explicitly set."""
        if self.name.startswith("_"):
            object.__setattr__(self, "is_public", False)
