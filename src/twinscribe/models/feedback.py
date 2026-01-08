"""
Call graph feedback models for inter-stream communication.

These models represent feedback about call graph discrepancies that should be
passed between documentation streams to help them converge. When Stream A
finds edges that Stream B missed (or vice versa), feedback is generated
to help both streams re-document with awareness of the other's findings.
"""

from pydantic import BaseModel, Field


class CallGraphFeedback(BaseModel):
    """Feedback about call graph discrepancies to send to a documenter.

    This feedback is generated when StreamCallGraphComparison detects
    differences between Stream A and Stream B call graphs. It tells a
    documenter about edges that the other stream found, so it can
    verify and potentially update its documentation.

    Attributes:
        component_id: The component this feedback applies to
        edges_only_in_other_stream: Edges (caller, callee) that the other
            stream found but this stream did not
        edges_to_verify: Edges (caller, callee) that this stream found
            but the other stream did not - should be double-checked
        other_stream_id: Identifier of the other stream ("A" or "B")
        message: Human-readable feedback message for the documenter

    Example:
        If Stream A found edge (X, Y) but Stream B did not, Stream B
        would receive feedback with:
            edges_only_in_other_stream=[(X, Y)]
            other_stream_id="A"
            message="Stream A found edge X->Y that you missed. Please verify."
    """

    component_id: str = Field(
        ...,
        min_length=1,
        description="Component ID this feedback applies to",
    )
    edges_only_in_other_stream: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Edges (caller, callee) found only by the other stream",
    )
    edges_to_verify: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Edges this stream found that should be verified",
    )
    other_stream_id: str = Field(
        ...,
        description="Identifier of the other stream (A or B)",
    )
    message: str = Field(
        default="",
        description="Human-readable feedback message",
    )

    def has_feedback(self) -> bool:
        """Check if this feedback contains any actionable information.

        Returns:
            True if there are edges to verify or edges from the other stream
        """
        return bool(self.edges_only_in_other_stream or self.edges_to_verify)

    def to_prompt_section(self) -> str:
        """Convert feedback to a prompt section for the documenter.

        Returns:
            Formatted string suitable for inclusion in a documenter prompt
        """
        lines = []

        if self.edges_only_in_other_stream:
            edge_strs = [f"{caller}->{callee}" for caller, callee in self.edges_only_in_other_stream]
            lines.append(
                f"Stream {self.other_stream_id} found these edges that you missed: "
                f"{', '.join(edge_strs)}. Please verify and update if correct."
            )

        if self.edges_to_verify:
            edge_strs = [f"{caller}->{callee}" for caller, callee in self.edges_to_verify]
            lines.append(
                f"Stream {self.other_stream_id} did not find these edges: "
                f"{', '.join(edge_strs)}. Please verify these are correct."
            )

        return "\n".join(lines) if lines else ""


class StreamFeedback(BaseModel):
    """Aggregated feedback for a documentation stream.

    Contains all call graph feedback items that should be passed to
    a stream's documenter for the next iteration. This allows the
    documenter to be aware of discrepancies and attempt to converge
    with the other stream.

    Attributes:
        stream_id: The stream this feedback is for ("A" or "B")
        feedbacks: List of feedback items for individual components

    Example:
        After comparing Stream A and Stream B outputs, the orchestrator
        generates StreamFeedback for each stream:
        - Stream A's feedback contains info about edges B found but A didn't
        - Stream B's feedback contains info about edges A found but B didn't
    """

    stream_id: str = Field(
        ...,
        description="Stream identifier (A or B)",
    )
    feedbacks: list[CallGraphFeedback] = Field(
        default_factory=list,
        description="List of feedback items for components",
    )

    def get_feedback_for_component(self, component_id: str) -> CallGraphFeedback | None:
        """Get feedback for a specific component.

        Args:
            component_id: The component to look up

        Returns:
            CallGraphFeedback for the component, or None if not found
        """
        for feedback in self.feedbacks:
            if feedback.component_id == component_id:
                return feedback
        return None

    def has_feedback_for_component(self, component_id: str) -> bool:
        """Check if there is feedback for a specific component.

        Args:
            component_id: The component to check

        Returns:
            True if feedback exists for this component
        """
        return self.get_feedback_for_component(component_id) is not None

    def to_corrections(self) -> list[dict]:
        """Convert feedback to corrections format for DocumenterInput.

        This method converts the structured feedback into a list of
        correction dictionaries that can be passed to DocumenterInput.corrections.

        Returns:
            List of correction dictionaries with field, action, and reason keys
        """
        corrections = []
        for feedback in self.feedbacks:
            if not feedback.has_feedback():
                continue

            correction = {
                "field": "call_graph",
                "action": "verify_and_update",
                "reason": feedback.to_prompt_section(),
                "component_id": feedback.component_id,
                "edges_to_add": feedback.edges_only_in_other_stream,
                "edges_to_verify": feedback.edges_to_verify,
            }
            corrections.append(correction)

        return corrections

    def get_component_ids_with_feedback(self) -> list[str]:
        """Get list of component IDs that have feedback.

        Returns:
            List of component IDs with actionable feedback
        """
        return [
            feedback.component_id
            for feedback in self.feedbacks
            if feedback.has_feedback()
        ]

    @property
    def total_edges_to_verify(self) -> int:
        """Total number of edges across all feedback that need verification."""
        return sum(
            len(f.edges_only_in_other_stream) + len(f.edges_to_verify)
            for f in self.feedbacks
        )

    @property
    def component_count(self) -> int:
        """Number of components with feedback."""
        return len([f for f in self.feedbacks if f.has_feedback()])
