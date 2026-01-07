"""
Convergence tracking models.

These models track the convergence process across iterations
as defined in spec section 4.2.
"""

from datetime import datetime

from pydantic import BaseModel, Field, computed_field


class ConvergenceCriteria(BaseModel):
    """Criteria for determining when documentation streams have converged.

    Based on spec section 4.2 convergence criteria.

    Attributes:
        max_iterations: Maximum iterations before forced convergence
        call_graph_match_rate: Required match rate for call graphs
        documentation_similarity: Required semantic similarity
        max_open_discrepancies: Max allowed unresolved non-blocking issues
        blocking_discrepancy_types: Discrepancy types that prevent convergence
    """

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum iterations allowed",
    )
    call_graph_match_rate: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Required call graph edge match rate",
    )
    documentation_similarity: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Required semantic similarity threshold",
    )
    max_open_discrepancies: int = Field(
        default=2,
        ge=0,
        description="Max unresolved non-blocking discrepancies",
    )
    blocking_discrepancy_types: list[str] = Field(
        default_factory=lambda: [
            "missing_critical_call",
            "false_critical_call",
            "missing_public_api_doc",
            "security_relevant_gap",
        ],
        description="Types that block convergence",
    )

    def is_satisfied(
        self,
        call_graph_match: float,
        doc_similarity: float,
        open_discrepancies: int,
        has_blocking: bool,
    ) -> bool:
        """Check if all convergence criteria are satisfied.

        Args:
            call_graph_match: Current call graph match rate
            doc_similarity: Current documentation similarity
            open_discrepancies: Count of open non-blocking discrepancies
            has_blocking: Whether any blocking discrepancies exist

        Returns:
            True if all criteria met
        """
        return (
            call_graph_match >= self.call_graph_match_rate
            and doc_similarity >= self.documentation_similarity
            and open_discrepancies <= self.max_open_discrepancies
            and not has_blocking
        )


class ConvergenceHistoryEntry(BaseModel):
    """Record of convergence state at one iteration.

    Attributes:
        iteration: Iteration number
        total_components: Components processed
        identical: Components with identical documentation
        discrepancies: Total discrepancies found
        resolved: Discrepancies resolved this iteration
        blocking: Blocking discrepancies remaining
        call_graph_match_rate: Call graph accuracy
        documentation_similarity: Doc similarity score
        beads_tickets_created: Tickets created this iteration
        timestamp: When iteration completed
    """

    iteration: int = Field(..., ge=1, description="Iteration number")
    total_components: int = Field(default=0, ge=0)
    identical: int = Field(default=0, ge=0)
    discrepancies: int = Field(default=0, ge=0)
    resolved: int = Field(
        default=0,
        ge=0,
        description="Resolved this iteration",
    )
    blocking: int = Field(
        default=0,
        ge=0,
        description="Blocking issues remaining",
    )
    call_graph_match_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    documentation_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    beads_tickets_created: int = Field(
        default=0,
        ge=0,
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
    )

    @computed_field
    @property
    def agreement_rate(self) -> float:
        """Rate of identical components."""
        if self.total_components == 0:
            return 1.0
        return self.identical / self.total_components


class ConvergenceReport(BaseModel):
    """Complete report on the convergence process.

    Generated at the end of documentation run to summarize
    how convergence was achieved (or not).

    Attributes:
        total_iterations: How many iterations were performed
        final_status: Whether convergence was achieved
        history: Per-iteration convergence state
        criteria: Criteria used for convergence
        started_at: When the run started
        completed_at: When the run finished
        forced_convergence: True if max iterations reached
        remaining_discrepancies: Unresolved discrepancy IDs
    """

    total_iterations: int = Field(default=0, ge=0, description="Iterations performed")
    final_status: str = Field(
        default="pending",
        description="Final convergence status",
        examples=["converged", "max_iterations_reached", "pending"],
    )
    history: list[ConvergenceHistoryEntry] = Field(
        default_factory=list,
        description="Per-iteration records",
    )
    criteria: ConvergenceCriteria = Field(
        default_factory=ConvergenceCriteria,
        description="Criteria used",
    )
    started_at: datetime | None = Field(default=None, description="Run start time")
    completed_at: datetime | None = Field(default=None, description="Run end time")
    forced_convergence: bool = Field(
        default=False,
        description="True if max iterations reached",
    )
    remaining_discrepancies: list[str] = Field(
        default_factory=list,
        description="IDs of unresolved discrepancies",
    )

    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        """Total duration of the run in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @computed_field
    @property
    def is_successful(self) -> bool:
        """True if converged without forcing."""
        return self.final_status == "converged" and not self.forced_convergence

    def add_iteration(self, entry: ConvergenceHistoryEntry) -> None:
        """Add an iteration entry to history."""
        self.history.append(entry)
        self.total_iterations = len(self.history)

    def get_latest_entry(self) -> ConvergenceHistoryEntry | None:
        """Get the most recent iteration entry."""
        return self.history[-1] if self.history else None
