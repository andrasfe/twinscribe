"""
Call graph data models for representing function/method call relationships.

These models represent the ground truth call graph extracted from static
analysis, as well as call graphs inferred by documentation agents.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

from twinscribe.models.base import CallType

if TYPE_CHECKING:
    from twinscribe.models.convergence import ConvergenceCriteria, ConvergenceStatus


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


class StreamCallGraphComparison(BaseModel):
    """Comparison of call graphs between Stream A and Stream B.

    Used to validate consistency between the two documentation streams
    BEFORE the comparator runs. This is a quality gate to detect
    significant discrepancies early.

    Attributes:
        total_components: Number of components compared
        identical_components: Components with identical call graphs
        differing_components: Components with different call graphs
        only_in_a: Edges found only in Stream A
        only_in_b: Edges found only in Stream B
        agreement_rate: Percentage of components with identical call graphs
    """

    total_components: int = Field(default=0, description="Total components compared")
    identical_components: int = Field(default=0, description="Components with identical call graphs")
    differing_components: int = Field(default=0, description="Components with differences")
    only_in_a: dict[str, set[tuple[str, str]]] = Field(
        default_factory=dict,
        description="Component ID -> edges only in Stream A",
    )
    only_in_b: dict[str, set[tuple[str, str]]] = Field(
        default_factory=dict,
        description="Component ID -> edges only in Stream B",
    )
    component_details: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-component comparison details",
    )

    @computed_field
    @property
    def agreement_rate(self) -> float:
        """Percentage of components with identical call graphs."""
        if self.total_components == 0:
            return 1.0
        return self.identical_components / self.total_components

    @computed_field
    @property
    def is_consistent(self) -> bool:
        """True if all call graphs are identical between streams."""
        return self.differing_components == 0

    @classmethod
    def compare_streams(
        cls,
        stream_a_outputs: dict[str, "DocumentationOutput"],
        stream_b_outputs: dict[str, "DocumentationOutput"],
    ) -> "StreamCallGraphComparison":
        """Compare call graphs from both streams.

        Args:
            stream_a_outputs: Component ID -> DocumentationOutput from Stream A
            stream_b_outputs: Component ID -> DocumentationOutput from Stream B

        Returns:
            StreamCallGraphComparison with detailed comparison results
        """
        # Import here to avoid circular dependency
        from twinscribe.models.documentation import DocumentationOutput

        # Get common components
        common_ids = set(stream_a_outputs.keys()) & set(stream_b_outputs.keys())

        result = cls(total_components=len(common_ids))

        for comp_id in common_ids:
            doc_a = stream_a_outputs[comp_id]
            doc_b = stream_b_outputs[comp_id]

            # Extract call graph edges as sets of (caller/callee, component_id) tuples
            a_callees = {(comp_id, c.component_id) for c in doc_a.call_graph.callees}
            b_callees = {(comp_id, c.component_id) for c in doc_b.call_graph.callees}
            a_callers = {(c.component_id, comp_id) for c in doc_a.call_graph.callers}
            b_callers = {(c.component_id, comp_id) for c in doc_b.call_graph.callers}

            a_edges = a_callees | a_callers
            b_edges = b_callees | b_callers

            if a_edges == b_edges:
                result.identical_components += 1
                result.component_details[comp_id] = {
                    "status": "identical",
                    "edge_count": len(a_edges),
                }
            else:
                result.differing_components += 1
                only_a = a_edges - b_edges
                only_b = b_edges - a_edges

                if only_a:
                    result.only_in_a[comp_id] = only_a
                if only_b:
                    result.only_in_b[comp_id] = only_b

                result.component_details[comp_id] = {
                    "status": "different",
                    "a_edges": len(a_edges),
                    "b_edges": len(b_edges),
                    "only_in_a": len(only_a),
                    "only_in_b": len(only_b),
                    "matching": len(a_edges & b_edges),
                }

        return result

    def get_summary(self) -> str:
        """Get a human-readable summary of the comparison."""
        lines = [
            f"Stream A vs B Call Graph Comparison:",
            f"  Total components: {self.total_components}",
            f"  Identical: {self.identical_components} ({self.agreement_rate:.1%})",
            f"  Different: {self.differing_components}",
        ]

        if self.differing_components > 0:
            lines.append(f"  Components with differences:")
            for comp_id, details in self.component_details.items():
                if details["status"] == "different":
                    lines.append(
                        f"    - {comp_id}: A={details['a_edges']} edges, "
                        f"B={details['b_edges']} edges, "
                        f"only_A={details['only_in_a']}, only_B={details['only_in_b']}"
                    )

        return "\n".join(lines)

    def check_convergence(
        self,
        criteria: "ConvergenceCriteria",
        iteration: int = 1,
    ) -> "ConvergenceStatus":
        """Check convergence status against criteria.

        Evaluates the current comparison results against convergence criteria
        to determine if streams have achieved consensus.

        Args:
            criteria: Convergence criteria to check against
            iteration: Current iteration number

        Returns:
            ConvergenceStatus with detailed convergence information
        """
        # Import here to avoid circular dependency
        from twinscribe.models.convergence import ConvergenceStatus

        # Get lists of converged and divergent components
        converged_components: list[str] = []
        divergent_components: list[str] = []

        for comp_id, details in self.component_details.items():
            if details["status"] == "identical":
                converged_components.append(comp_id)
            else:
                divergent_components.append(comp_id)

        # Check if agreement rate meets criteria
        is_converged = criteria.is_agreement_sufficient(self.agreement_rate)

        return ConvergenceStatus(
            is_converged=is_converged,
            agreement_rate=self.agreement_rate,
            iteration=iteration,
            converged_components=converged_components,
            divergent_components=divergent_components,
        )

    def generate_feedback(self) -> tuple["StreamFeedback", "StreamFeedback"]:
        """Generate feedback for both streams based on discrepancies.

        Creates feedback objects that tell each stream about edges the other
        stream found. This creates a feedback loop where both streams learn
        from each other and can converge.

        When A finds edge X->Y but B doesn't:
        - Tell B: "Stream A found edge X->Y that you missed. Please verify."
        - Tell A: "Stream B did not find edge X->Y. Please verify this is correct."

        Returns:
            Tuple of (feedback_for_stream_a, feedback_for_stream_b)
        """
        # Import here to avoid circular dependency
        from twinscribe.models.feedback import CallGraphFeedback, StreamFeedback

        feedback_a_items: list[CallGraphFeedback] = []
        feedback_b_items: list[CallGraphFeedback] = []

        # Process each component that has differences
        for comp_id in self.only_in_a.keys() | self.only_in_b.keys():
            edges_only_in_a = self.only_in_a.get(comp_id, set())
            edges_only_in_b = self.only_in_b.get(comp_id, set())

            # Feedback for Stream A: edges B found that A missed, and A's edges to verify
            if edges_only_in_b or edges_only_in_a:
                # Build message for Stream A
                message_parts_a = []
                if edges_only_in_b:
                    edge_strs = [f"{c}->{e}" for c, e in edges_only_in_b]
                    message_parts_a.append(
                        f"Stream B found edges that you missed: {', '.join(edge_strs)}. "
                        "Please verify and include if correct."
                    )
                if edges_only_in_a:
                    edge_strs = [f"{c}->{e}" for c, e in edges_only_in_a]
                    message_parts_a.append(
                        f"Stream B did not find edges: {', '.join(edge_strs)}. "
                        "Please verify these are correct."
                    )

                feedback_a_items.append(
                    CallGraphFeedback(
                        component_id=comp_id,
                        edges_only_in_other_stream=list(edges_only_in_b),
                        edges_to_verify=list(edges_only_in_a),
                        other_stream_id="B",
                        message=" ".join(message_parts_a),
                    )
                )

            # Feedback for Stream B: edges A found that B missed, and B's edges to verify
            if edges_only_in_a or edges_only_in_b:
                # Build message for Stream B
                message_parts_b = []
                if edges_only_in_a:
                    edge_strs = [f"{c}->{e}" for c, e in edges_only_in_a]
                    message_parts_b.append(
                        f"Stream A found edges that you missed: {', '.join(edge_strs)}. "
                        "Please verify and include if correct."
                    )
                if edges_only_in_b:
                    edge_strs = [f"{c}->{e}" for c, e in edges_only_in_b]
                    message_parts_b.append(
                        f"Stream A did not find edges: {', '.join(edge_strs)}. "
                        "Please verify these are correct."
                    )

                feedback_b_items.append(
                    CallGraphFeedback(
                        component_id=comp_id,
                        edges_only_in_other_stream=list(edges_only_in_a),
                        edges_to_verify=list(edges_only_in_b),
                        other_stream_id="A",
                        message=" ".join(message_parts_b),
                    )
                )

        feedback_for_a = StreamFeedback(stream_id="A", feedbacks=feedback_a_items)
        feedback_for_b = StreamFeedback(stream_id="B", feedbacks=feedback_b_items)

        return (feedback_for_a, feedback_for_b)
