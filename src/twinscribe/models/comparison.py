"""
Comparison result models from the comparator agent.

These models define the output schema produced by the comparator agent (C)
as specified in section 3.3 of the specification.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from twinscribe.models.base import DiscrepancyType, ResolutionAction, ResolutionSource


class BeadsTicketRef(BaseModel):
    """Reference to a Beads ticket for human review.

    Attributes:
        summary: Ticket title
        description: Full ticket description
        priority: Ticket priority (Low, Medium, High, Critical)
        ticket_key: Beads ticket key once created (e.g., LEGACY-123)
    """

    summary: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Full description")
    priority: str = Field(
        default="Medium",
        description="Priority level",
        examples=["Low", "Medium", "High", "Critical"],
    )
    ticket_key: str | None = Field(
        default=None,
        description="Beads key once created",
        examples=["LEGACY-123"],
    )


class Discrepancy(BaseModel):
    """Represents a difference found between the two documentation streams.

    Each discrepancy is analyzed and either resolved automatically
    (using ground truth) or escalated to human review via Beads.

    Attributes:
        discrepancy_id: Unique identifier for this discrepancy
        component_id: Component where discrepancy was found
        type: Category of discrepancy
        stream_a_value: Value from Stream A (or None if missing)
        stream_b_value: Value from Stream B (or None if missing)
        ground_truth: Static analysis value if available
        resolution: How the discrepancy was/should be resolved
        confidence: Confidence in the resolution (0.0-1.0)
        requires_beads: True if human review needed
        beads_ticket: Ticket details if requires_beads is True
        iteration_found: Which iteration this was discovered
    """

    discrepancy_id: str = Field(
        ...,
        description="Unique identifier",
        examples=["disc_001"],
    )
    component_id: str = Field(..., min_length=1, description="Affected component")
    type: DiscrepancyType = Field(..., description="Category of discrepancy")
    stream_a_value: Any | None = Field(default=None, description="Value from Stream A")
    stream_b_value: Any | None = Field(default=None, description="Value from Stream B")
    ground_truth: Any | None = Field(
        default=None,
        description="Static analysis ground truth if applicable",
    )
    resolution: ResolutionAction = Field(
        default=ResolutionAction.NEEDS_HUMAN_REVIEW,
        description="How to resolve",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in resolution",
    )
    requires_beads: bool = Field(
        default=False,
        description="True if needs human review ticket",
    )
    beads_ticket: BeadsTicketRef | None = Field(
        default=None,
        description="Ticket details if requires_beads",
    )
    iteration_found: int = Field(
        default=1,
        ge=1,
        description="Iteration when discovered",
    )
    resolution_source: ResolutionSource = Field(
        default=ResolutionSource.UNRESOLVED,
        description="Source of the resolution (consensus, ground_truth_hint, human_review, etc.)",
    )

    @computed_field
    @property
    def is_call_graph_related(self) -> bool:
        """True if discrepancy relates to call graph (resolvable by static analysis)."""
        return self.type in {
            DiscrepancyType.CALL_GRAPH_EDGE,
            DiscrepancyType.CALL_SITE_LINE,
            DiscrepancyType.CALL_TYPE_MISMATCH,
        }

    @computed_field
    @property
    def is_resolved(self) -> bool:
        """True if discrepancy has been resolved."""
        return self.resolution not in {
            ResolutionAction.NEEDS_HUMAN_REVIEW,
            ResolutionAction.DEFERRED,
        }

    @computed_field
    @property
    def is_blocking(self) -> bool:
        """True if this discrepancy blocks convergence."""
        # High confidence resolutions are not blocking
        if self.confidence >= 0.7 and self.is_resolved:
            return False
        return not self.is_resolved


class ConvergenceStatus(BaseModel):
    """Status of convergence between the two streams.

    Attributes:
        converged: True if streams have reached agreement
        blocking_discrepancies: Count of unresolved blocking issues
        recommendation: What to do next
    """

    converged: bool = Field(default=False, description="True if streams agree")
    blocking_discrepancies: int = Field(
        default=0,
        ge=0,
        description="Count of blocking issues",
    )
    recommendation: str = Field(
        default="continue",
        description="Next action",
        examples=[
            "continue",
            "generate_beads_tickets",
            "finalize",
            "max_iterations_reached",
        ],
    )


class ComparisonSummary(BaseModel):
    """Summary statistics from comparison.

    Attributes:
        total_components: Total components compared
        identical: Components with identical documentation
        discrepancies: Total discrepancies found
        resolved_by_ground_truth: Discrepancies resolved via static analysis
        requires_human_review: Discrepancies needing Beads tickets
    """

    total_components: int = Field(default=0, ge=0, description="Total components")
    identical: int = Field(default=0, ge=0, description="Identical components")
    discrepancies: int = Field(default=0, ge=0, description="Total discrepancies")
    resolved_by_ground_truth: int = Field(default=0, ge=0, description="Auto-resolved count")
    requires_human_review: int = Field(default=0, ge=0, description="Needs Beads tickets")

    @computed_field
    @property
    def agreement_rate(self) -> float:
        """Percentage of components that agree."""
        if self.total_components == 0:
            return 1.0
        return self.identical / self.total_components


class ComparatorMetadata(BaseModel):
    """Metadata about the comparison process.

    Attributes:
        agent_id: Identifier of comparator agent (C)
        model: Model used for comparison
        timestamp: When comparison was performed
        comparison_duration_ms: Time taken for comparison
        token_count: Tokens consumed
    """

    agent_id: str = Field(default="C", description="Agent identifier")
    model: str = Field(
        ...,
        description="Model name",
        examples=["claude-opus-4-5-20251101"],
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When compared",
    )
    comparison_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Duration in milliseconds",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Tokens consumed",
    )


class ComparisonResult(BaseModel):
    """Complete output from the comparator agent.

    This is the main output schema for the comparator agent (C)
    as defined in spec section 3.3.

    Attributes:
        comparison_id: Unique identifier for this comparison
        iteration: Which iteration this comparison belongs to
        summary: Statistical summary
        discrepancies: Detailed list of all discrepancies
        convergence_status: Whether streams have converged
        metadata: Comparator agent information
    """

    comparison_id: str = Field(
        ...,
        description="Unique comparison ID",
        examples=["cmp_20260106_001"],
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current iteration number",
    )
    summary: ComparisonSummary = Field(
        default_factory=ComparisonSummary,
        description="Summary statistics",
    )
    discrepancies: list[Discrepancy] = Field(
        default_factory=list,
        description="All discrepancies found",
    )
    convergence_status: ConvergenceStatus = Field(
        default_factory=ConvergenceStatus,
        description="Convergence state",
    )
    metadata: ComparatorMetadata = Field(..., description="Agent metadata")

    def get_discrepancies_for_component(self, component_id: str) -> list[Discrepancy]:
        """Get all discrepancies for a specific component."""
        return [d for d in self.discrepancies if d.component_id == component_id]

    def get_blocking_discrepancies(self) -> list[Discrepancy]:
        """Get all discrepancies that block convergence."""
        return [d for d in self.discrepancies if d.is_blocking]

    def get_beads_required(self) -> list[Discrepancy]:
        """Get discrepancies that need Beads tickets."""
        return [d for d in self.discrepancies if d.requires_beads]

    @computed_field
    @property
    def is_converged(self) -> bool:
        """Convenience property for convergence status."""
        return self.convergence_status.converged
