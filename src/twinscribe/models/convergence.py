"""
Convergence tracking models.

These models track the convergence process across iterations
as defined in spec section 4.2, with support for dual-stream
consensus-based resolution of call graphs.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    pass


class ConvergenceCriteria(BaseModel):
    """Criteria for determining when documentation streams have converged.

    Based on spec section 4.2 convergence criteria with enhanced support
    for dual-stream consensus.

    Attributes:
        min_agreement_rate: Minimum percentage of components that must have
            identical call graphs (95% = 0.95). Components below this threshold
            trigger continued iteration or escalation.
        max_iterations: Maximum iterations before escalating to Beads.
            After this limit, divergent components are escalated for human review.
        call_graph_match_rate: Required match rate for call graphs (legacy).
        documentation_similarity: Required semantic similarity threshold.
        max_open_discrepancies: Max allowed unresolved non-blocking issues.
        blocking_discrepancy_types: Discrepancy types that prevent convergence.
    """

    min_agreement_rate: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum percentage of components with identical call graphs (95%)",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum iterations before escalating divergent components to Beads",
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

    def is_agreement_sufficient(self, agreement_rate: float) -> bool:
        """Check if the agreement rate meets the minimum threshold.

        Args:
            agreement_rate: Current agreement rate between streams (0.0-1.0)

        Returns:
            True if agreement rate meets or exceeds min_agreement_rate
        """
        return agreement_rate >= self.min_agreement_rate


class ConvergenceStatus(BaseModel):
    """Status of convergence between streams for call graphs.

    This model provides detailed information about which components
    have converged (streams agree) and which are still divergent.

    Attributes:
        is_converged: True if agreement rate meets criteria
        agreement_rate: Percentage of components with identical call graphs
        iteration: Current iteration number
        converged_components: List of component IDs where streams agree
        divergent_components: List of component IDs where streams disagree
    """

    is_converged: bool = Field(
        default=False,
        description="True if streams have achieved consensus",
    )
    agreement_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of components with identical call graphs",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current iteration number",
    )
    converged_components: list[str] = Field(
        default_factory=list,
        description="Component IDs where Stream A == Stream B",
    )
    divergent_components: list[str] = Field(
        default_factory=list,
        description="Component IDs where streams still disagree",
    )

    def should_continue_iterating(self, criteria: ConvergenceCriteria) -> bool:
        """Check if iteration should continue or stop.

        Iteration should continue if:
        - Not yet converged (agreement rate below threshold)
        - Haven't reached max iterations yet

        Args:
            criteria: Convergence criteria to check against

        Returns:
            True if should continue iterating, False if should stop
        """
        # If already converged, no need to continue
        if self.is_converged:
            return False

        # If at or beyond max iterations, stop
        if self.iteration >= criteria.max_iterations:
            return False

        # Continue iterating to try to achieve convergence
        return True

    @computed_field
    @property
    def total_components(self) -> int:
        """Total number of components evaluated."""
        return len(self.converged_components) + len(self.divergent_components)

    @computed_field
    @property
    def divergent_count(self) -> int:
        """Number of components that are still divergent."""
        return len(self.divergent_components)


class ConvergenceResult(BaseModel):
    """Final convergence result after all iterations complete.

    This model captures the outcome of the convergence process,
    indicating which components achieved consensus and which
    need to be escalated to Beads for human review.

    Attributes:
        status: Final status - "converged", "partially_converged", or "divergent"
        final_agreement_rate: Agreement rate at end of iterations
        iterations_used: Number of iterations performed
        accepted_by_consensus: Components where A == B (accepted as truth)
        escalated_to_beads: Components requiring human review via Beads
        iteration_history: Agreement rates per iteration for debugging
    """

    status: str = Field(
        default="pending",
        description="Final convergence status",
        examples=["converged", "partially_converged", "divergent"],
    )
    final_agreement_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final agreement rate between streams",
    )
    iterations_used: int = Field(
        default=0,
        ge=0,
        description="Number of iterations performed",
    )
    accepted_by_consensus: list[str] = Field(
        default_factory=list,
        description="Component IDs accepted via dual-stream consensus (A == B)",
    )
    escalated_to_beads: list[str] = Field(
        default_factory=list,
        description="Component IDs requiring human review via Beads",
    )
    iteration_history: list[float] = Field(
        default_factory=list,
        description="Agreement rate at each iteration",
    )

    @computed_field
    @property
    def is_fully_converged(self) -> bool:
        """True if all components converged (no escalations needed)."""
        return self.status == "converged" and len(self.escalated_to_beads) == 0

    @computed_field
    @property
    def escalation_count(self) -> int:
        """Number of components escalated to Beads."""
        return len(self.escalated_to_beads)

    @computed_field
    @property
    def consensus_count(self) -> int:
        """Number of components accepted by consensus."""
        return len(self.accepted_by_consensus)

    @classmethod
    def from_convergence_status(
        cls,
        status: "ConvergenceStatus",
        criteria: ConvergenceCriteria,
        iteration_history: list[float] | None = None,
    ) -> "ConvergenceResult":
        """Create a ConvergenceResult from a ConvergenceStatus.

        Args:
            status: Current convergence status
            criteria: Convergence criteria used
            iteration_history: Optional list of agreement rates per iteration

        Returns:
            ConvergenceResult summarizing the outcome
        """
        # Determine final status
        if status.is_converged:
            final_status = "converged"
        elif status.agreement_rate >= 0.5:
            final_status = "partially_converged"
        else:
            final_status = "divergent"

        # Components that didn't converge after max iterations go to Beads
        escalated = (
            status.divergent_components if status.iteration >= criteria.max_iterations else []
        )

        return cls(
            status=final_status,
            final_agreement_rate=status.agreement_rate,
            iterations_used=status.iteration,
            accepted_by_consensus=status.converged_components,
            escalated_to_beads=escalated,
            iteration_history=iteration_history or [],
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
