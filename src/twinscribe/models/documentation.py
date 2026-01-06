"""
Documentation output models from documenter agents.

These models define the output schema produced by documenter agents
(A1 and B1) as specified in section 3.1 of the specification.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from twinscribe.models.base import CallType, StreamId
from twinscribe.models.components import ComponentDocumentation


class CalleeRef(BaseModel):
    """Reference to a component called by the focal component.

    Attributes:
        component_id: ID of the called component
        call_site_line: Line number where the call occurs
        call_type: Type of call (direct, conditional, loop)
    """

    component_id: str = Field(
        ..., min_length=1, description="ID of called component"
    )
    call_site_line: Optional[int] = Field(
        default=None, ge=1, description="Line number of call"
    )
    call_type: CallType = Field(
        default=CallType.DIRECT, description="Type of call"
    )


class CallerRef(BaseModel):
    """Reference to a component that calls the focal component.

    Attributes:
        component_id: ID of the calling component
        call_site_line: Line number in caller where call occurs
        call_type: Type of call
    """

    component_id: str = Field(
        ..., min_length=1, description="ID of calling component"
    )
    call_site_line: Optional[int] = Field(
        default=None, ge=1, description="Line number in caller"
    )
    call_type: CallType = Field(
        default=CallType.DIRECT, description="Type of call"
    )


class CallGraphSection(BaseModel):
    """Call graph information for a component.

    Contains both outgoing calls (callees) and incoming calls (callers)
    to fully describe the component's position in the call graph.

    Attributes:
        callers: Components that call this component
        callees: Components called by this component
    """

    callers: list[CallerRef] = Field(
        default_factory=list,
        description="Components that call this one",
    )
    callees: list[CalleeRef] = Field(
        default_factory=list,
        description="Components this one calls",
    )

    @property
    def total_edges(self) -> int:
        """Total number of call relationships."""
        return len(self.callers) + len(self.callees)


class DocumenterMetadata(BaseModel):
    """Metadata about the documentation generation process.

    Tracks which agent produced the documentation, model used,
    and quality metrics.

    Attributes:
        agent_id: Identifier of the documenter agent (A1, B1)
        stream_id: Which stream this agent belongs to
        model: Model name used for generation
        timestamp: When documentation was generated
        confidence: Agent's confidence score (0.0-1.0)
        processing_order: Order in which component was processed
        token_count: Tokens used for this component
    """

    agent_id: str = Field(
        ...,
        description="Agent identifier",
        examples=["A1", "B1"],
    )
    stream_id: StreamId = Field(
        ..., description="Stream this agent belongs to"
    )
    model: str = Field(
        ...,
        description="Model name used",
        examples=["claude-sonnet-4-5-20250929", "gpt-4o"],
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When generated",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )
    processing_order: int = Field(
        default=0,
        ge=0,
        description="Topological order position",
    )
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Tokens consumed",
    )


class DocumentationOutput(BaseModel):
    """Complete output from a documenter agent for one component.

    This is the main output schema for documenter agents (A1, B1)
    as defined in spec section 3.1.

    Attributes:
        component_id: ID of the documented component
        documentation: The documentation content
        call_graph: Call relationships discovered
        metadata: Agent and processing information
    """

    component_id: str = Field(
        ...,
        min_length=1,
        description="Documented component ID",
        examples=["module.Class.method"],
    )
    documentation: ComponentDocumentation = Field(
        ..., description="Documentation content"
    )
    call_graph: CallGraphSection = Field(
        default_factory=CallGraphSection,
        description="Call graph information",
    )
    metadata: DocumenterMetadata = Field(
        ..., description="Agent metadata"
    )

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Ensure component_id is not empty."""
        if not v.strip():
            raise ValueError("component_id cannot be empty")
        return v

    def get_callee_ids(self) -> list[str]:
        """Extract just the component IDs of callees."""
        return [c.component_id for c in self.call_graph.callees]

    def get_caller_ids(self) -> list[str]:
        """Extract just the component IDs of callers."""
        return [c.component_id for c in self.call_graph.callers]


class StreamOutput(BaseModel):
    """Complete output from one documentation stream.

    Aggregates all component documentation from a single stream
    after the documenter and validator have processed everything.

    Attributes:
        stream_id: Which stream (A or B)
        outputs: Documentation for each component
        total_components: Number of components processed
        total_tokens: Total tokens consumed
        processing_time_seconds: Time taken to process
    """

    stream_id: StreamId = Field(..., description="Stream identifier")
    outputs: dict[str, DocumentationOutput] = Field(
        default_factory=dict,
        description="Component ID -> Documentation mapping",
    )
    total_components: int = Field(
        default=0, ge=0, description="Components processed"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Tokens consumed"
    )
    processing_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Processing duration"
    )

    def get_output(self, component_id: str) -> Optional[DocumentationOutput]:
        """Get documentation output for a specific component."""
        return self.outputs.get(component_id)

    def add_output(self, output: DocumentationOutput) -> None:
        """Add a documentation output to the stream."""
        self.outputs[output.component_id] = output
        self.total_components = len(self.outputs)
        if output.metadata.token_count:
            self.total_tokens += output.metadata.token_count
