"""
Call graph data models for representing function/method call relationships.

These models represent the ground truth call graph extracted from static
analysis, as well as call graphs inferred by documentation agents.
"""

from collections.abc import Iterator

from pydantic import BaseModel, Field, computed_field

from twinscribe.models.base import CallType


class CallEdge(BaseModel):
    """Represents a call relationship between two components.

    An edge from caller to callee indicates that the caller
    invokes the callee at a specific location in the code.

    Attributes:
        caller: Component ID of the calling component
        callee: Component ID of the called component
        call_site_line: Line number where the call occurs
        call_type: Type of call (direct, conditional, loop, etc.)
        confidence: Confidence score (1.0 for static analysis)
    """

    caller: str = Field(..., min_length=1, description="Component ID of caller")
    callee: str = Field(..., min_length=1, description="Component ID of callee")
    call_site_line: int | None = Field(default=None, ge=1, description="Line number of call site")
    call_type: CallType = Field(default=CallType.DIRECT, description="Type of call")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (1.0 = static analysis verified)",
    )

    def __hash__(self) -> int:
        """Hash for set operations - based on caller/callee pair."""
        return hash((self.caller, self.callee))

    def __eq__(self, other: object) -> bool:
        """Equality based on caller/callee pair."""
        if not isinstance(other, CallEdge):
            return False
        return self.caller == other.caller and self.callee == other.callee

    def to_tuple(self) -> tuple[str, str]:
        """Return (caller, callee) tuple for set operations."""
        return (self.caller, self.callee)


class CallGraph(BaseModel):
    """Collection of call edges forming a directed graph.

    Provides graph operations for querying call relationships,
    which is essential for both documentation and validation.

    Attributes:
        edges: List of all call edges in the graph
        source: Origin of this call graph (static analysis tool name, agent ID)
    """

    edges: list[CallEdge] = Field(default_factory=list, description="All call edges")
    source: str = Field(
        default="unknown",
        description="Origin: 'pycg', 'pyan3', 'agent_A1', etc.",
    )

    @computed_field
    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return len(self.edges)

    @computed_field
    @property
    def node_count(self) -> int:
        """Total number of unique nodes (components) in the graph."""
        nodes = set()
        for edge in self.edges:
            nodes.add(edge.caller)
            nodes.add(edge.callee)
        return len(nodes)

    def get_callees(self, component_id: str) -> list[CallEdge]:
        """Get all components called by a given component.

        Args:
            component_id: The caller component ID

        Returns:
            List of edges where component_id is the caller
        """
        return [e for e in self.edges if e.caller == component_id]

    def get_callers(self, component_id: str) -> list[CallEdge]:
        """Get all components that call a given component.

        Args:
            component_id: The callee component ID

        Returns:
            List of edges where component_id is the callee
        """
        return [e for e in self.edges if e.callee == component_id]

    def has_edge(self, caller: str, callee: str) -> bool:
        """Check if a specific call relationship exists.

        Args:
            caller: Caller component ID
            callee: Callee component ID

        Returns:
            True if the edge exists
        """
        return any(e.caller == caller and e.callee == callee for e in self.edges)

    def add_edge(self, edge: CallEdge) -> bool:
        """Add an edge if it doesn't already exist.

        Args:
            edge: The edge to add

        Returns:
            True if edge was added, False if already existed
        """
        if not self.has_edge(edge.caller, edge.callee):
            self.edges.append(edge)
            return True
        return False

    def remove_edge(self, caller: str, callee: str) -> bool:
        """Remove an edge if it exists.

        Args:
            caller: Caller component ID
            callee: Callee component ID

        Returns:
            True if edge was removed, False if not found
        """
        original_len = len(self.edges)
        self.edges = [e for e in self.edges if not (e.caller == caller and e.callee == callee)]
        return len(self.edges) < original_len

    def all_nodes(self) -> set[str]:
        """Get all unique component IDs in the graph."""
        nodes = set()
        for edge in self.edges:
            nodes.add(edge.caller)
            nodes.add(edge.callee)
        return nodes

    def to_edge_set(self) -> set[tuple[str, str]]:
        """Convert to set of (caller, callee) tuples for comparison."""
        return {e.to_tuple() for e in self.edges}

    def iter_edges(self) -> Iterator[CallEdge]:
        """Iterate over all edges."""
        return iter(self.edges)

    def merge_with(self, other: "CallGraph") -> "CallGraph":
        """Create a new graph merging edges from both graphs.

        Args:
            other: Another CallGraph to merge with

        Returns:
            New CallGraph with combined edges (no duplicates)
        """
        merged = CallGraph(source=f"{self.source}+{other.source}")
        seen = set()
        for edge in list(self.edges) + list(other.edges):
            key = edge.to_tuple()
            if key not in seen:
                merged.edges.append(edge)
                seen.add(key)
        return merged


class CallGraphDiff(BaseModel):
    """Difference between two call graphs.

    Used to compare documented call graphs against ground truth
    from static analysis.

    Attributes:
        missing_in_doc: Edges in ground truth but not in documented
        extra_in_doc: Edges in documented but not in ground truth
        matching: Edges present in both graphs
        precision: Ratio of correct edges to total documented edges
        recall: Ratio of found edges to total ground truth edges
    """

    missing_in_doc: set[tuple[str, str]] = Field(
        default_factory=set,
        description="(caller, callee) in truth but not documented",
    )
    extra_in_doc: set[tuple[str, str]] = Field(
        default_factory=set,
        description="(caller, callee) documented but not in truth",
    )
    matching: set[tuple[str, str]] = Field(
        default_factory=set,
        description="(caller, callee) present in both",
    )
    precision: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of documented edges that are correct",
    )
    recall: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of true edges that were documented",
    )

    @computed_field
    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @computed_field
    @property
    def is_perfect_match(self) -> bool:
        """True if documented graph exactly matches ground truth."""
        return len(self.missing_in_doc) == 0 and len(self.extra_in_doc) == 0

    @classmethod
    def compute(cls, ground_truth: CallGraph, documented: CallGraph) -> "CallGraphDiff":
        """Compute the difference between ground truth and documented graphs.

        Args:
            ground_truth: The authoritative call graph from static analysis
            documented: The call graph from documentation agents

        Returns:
            CallGraphDiff with precision, recall, and edge differences
        """
        truth_edges = ground_truth.to_edge_set()
        doc_edges = documented.to_edge_set()

        matching = truth_edges & doc_edges
        missing = truth_edges - doc_edges
        extra = doc_edges - truth_edges

        precision = len(matching) / len(doc_edges) if doc_edges else 1.0
        recall = len(matching) / len(truth_edges) if truth_edges else 1.0

        return cls(
            missing_in_doc=missing,
            extra_in_doc=extra,
            matching=matching,
            precision=precision,
            recall=recall,
        )
