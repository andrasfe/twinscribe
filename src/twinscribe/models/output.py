"""
Final output models for the documentation system.

These models represent the final deliverables produced by the system
including the merged documentation package and run metrics.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import ComponentDocumentation
from twinscribe.models.convergence import ConvergenceReport


class CostBreakdown(BaseModel):
    """Cost breakdown by component of the system.

    Attributes:
        stream_a_documenter: Cost of Stream A documenter agent
        stream_a_validator: Cost of Stream A validator agent
        stream_b_documenter: Cost of Stream B documenter agent
        stream_b_validator: Cost of Stream B validator agent
        comparator: Cost of comparator agent
        total: Total cost
        currency: Currency (default USD)
    """

    stream_a_documenter: float = Field(
        default=0.0, ge=0.0, description="Stream A documenter cost"
    )
    stream_a_validator: float = Field(
        default=0.0, ge=0.0, description="Stream A validator cost"
    )
    stream_b_documenter: float = Field(
        default=0.0, ge=0.0, description="Stream B documenter cost"
    )
    stream_b_validator: float = Field(
        default=0.0, ge=0.0, description="Stream B validator cost"
    )
    comparator: float = Field(
        default=0.0, ge=0.0, description="Comparator cost"
    )
    total: float = Field(
        default=0.0, ge=0.0, description="Total cost"
    )
    currency: str = Field(
        default="USD", description="Currency"
    )

    def compute_total(self) -> float:
        """Compute and set total from components."""
        self.total = (
            self.stream_a_documenter
            + self.stream_a_validator
            + self.stream_b_documenter
            + self.stream_b_validator
            + self.comparator
        )
        return self.total

    @computed_field
    @property
    def stream_a_total(self) -> float:
        """Total cost for Stream A."""
        return self.stream_a_documenter + self.stream_a_validator

    @computed_field
    @property
    def stream_b_total(self) -> float:
        """Total cost for Stream B."""
        return self.stream_b_documenter + self.stream_b_validator


class RunMetrics(BaseModel):
    """Metrics from a documentation run.

    Comprehensive metrics for monitoring and cost tracking.

    Attributes:
        run_id: Unique identifier for this run
        codebase_path: Path to the documented codebase
        language: Primary programming language
        started_at: When run started
        completed_at: When run finished
        components_total: Total components discovered
        components_documented: Successfully documented
        call_graph_precision: Precision vs static analysis
        call_graph_recall: Recall vs static analysis
        call_graph_f1: F1 score
        discrepancies_total: Total discrepancies found
        discrepancies_resolved_auto: Resolved by ground truth
        discrepancies_resolved_beads: Resolved via Beads tickets
        discrepancies_unresolved: Still unresolved
        cost: Cost breakdown
        tokens_total: Total tokens consumed
    """

    run_id: str = Field(
        ...,
        description="Unique run identifier",
        examples=["run_20260106_001"],
    )
    codebase_path: str = Field(
        ..., description="Path to codebase"
    )
    language: str = Field(
        default="python", description="Primary language"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
    )
    completed_at: Optional[datetime] = Field(default=None)
    components_total: int = Field(
        default=0, ge=0, description="Total components"
    )
    components_documented: int = Field(
        default=0, ge=0, description="Successfully documented"
    )
    call_graph_precision: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    call_graph_recall: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    call_graph_f1: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    discrepancies_total: int = Field(
        default=0, ge=0
    )
    discrepancies_resolved_auto: int = Field(
        default=0, ge=0, description="Resolved by ground truth"
    )
    discrepancies_resolved_beads: int = Field(
        default=0, ge=0, description="Resolved via Beads"
    )
    discrepancies_unresolved: int = Field(
        default=0, ge=0
    )
    cost: CostBreakdown = Field(
        default_factory=CostBreakdown
    )
    tokens_total: int = Field(
        default=0, ge=0
    )

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @computed_field
    @property
    def cost_per_component(self) -> float:
        """Average cost per component."""
        if self.components_documented == 0:
            return 0.0
        return self.cost.total / self.components_documented

    @computed_field
    @property
    def documentation_rate(self) -> float:
        """Percentage of components successfully documented."""
        if self.components_total == 0:
            return 0.0
        return self.components_documented / self.components_total


class ComponentFinalDoc(BaseModel):
    """Final documentation for a single component.

    Merged result from both streams after convergence.

    Attributes:
        component_id: Component identifier
        documentation: Merged documentation content
        callers: List of caller component IDs
        callees: List of callee component IDs
        confidence_score: Confidence in the documentation (0-100)
        source_stream: Which stream's output was used (or 'merged')
        validation_passed: Whether validation passed
        beads_ticket_key: Associated Beads ticket if any
    """

    component_id: str = Field(
        ..., min_length=1, description="Component ID"
    )
    documentation: ComponentDocumentation = Field(
        ..., description="Documentation content"
    )
    callers: list[str] = Field(
        default_factory=list,
        description="Caller component IDs",
    )
    callees: list[str] = Field(
        default_factory=list,
        description="Callee component IDs",
    )
    confidence_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Confidence percentage",
    )
    source_stream: str = Field(
        default="merged",
        description="Source of documentation",
        examples=["A", "B", "merged"],
    )
    validation_passed: bool = Field(
        default=True, description="Validation status"
    )
    beads_ticket_key: Optional[str] = Field(
        default=None, description="Associated Beads ticket"
    )

    @computed_field
    @property
    def confidence_bucket(self) -> str:
        """Confidence level bucket for labeling."""
        if self.confidence_score >= 95:
            return "high"
        elif self.confidence_score >= 80:
            return "medium"
        else:
            return "low"


class DocumentationPackage(BaseModel):
    """Complete output package from a documentation run.

    This is the final deliverable containing all documentation,
    call graph, rebuild tickets, and run statistics.

    Attributes:
        documentation: Map of component_id to final documentation
        call_graph: Merged call graph with all relationships
        rebuild_tickets: Beads ticket data for rebuilding components
        convergence_report: Report on convergence process
        metrics: Run metrics and statistics
        version: Schema version for compatibility
    """

    documentation: dict[str, ComponentFinalDoc] = Field(
        default_factory=dict,
        description="Component ID -> Documentation",
    )
    call_graph: CallGraph = Field(
        default_factory=CallGraph,
        description="Final merged call graph",
    )
    rebuild_tickets: list[dict] = Field(
        default_factory=list,
        description="Beads rebuild ticket data",
    )
    convergence_report: ConvergenceReport = Field(
        default_factory=ConvergenceReport,
        description="Convergence process report",
    )
    metrics: RunMetrics = Field(
        ..., description="Run metrics"
    )
    version: str = Field(
        default="2.0.0",
        description="Schema version",
    )

    @computed_field
    @property
    def component_count(self) -> int:
        """Number of documented components."""
        return len(self.documentation)

    @computed_field
    @property
    def edge_count(self) -> int:
        """Number of call graph edges."""
        return self.call_graph.edge_count

    def get_component(self, component_id: str) -> Optional[ComponentFinalDoc]:
        """Get documentation for a specific component."""
        return self.documentation.get(component_id)

    def add_component(self, doc: ComponentFinalDoc) -> None:
        """Add a component's documentation."""
        self.documentation[doc.component_id] = doc

    def to_json_files(self) -> dict[str, str]:
        """Generate filenames for JSON output.

        Returns:
            Dict mapping purpose to suggested filename
        """
        run_id = self.metrics.run_id
        return {
            "documentation": f"{run_id}_documentation.json",
            "call_graph": f"{run_id}_call_graph.json",
            "rebuild_tickets": f"{run_id}_rebuild_tickets.json",
            "convergence_report": f"{run_id}_convergence.json",
            "metrics": f"{run_id}_metrics.json",
        }
