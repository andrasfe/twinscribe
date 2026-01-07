"""
Comparator Agent Interface.

Defines the interface for the comparator agent (C) that compares outputs
from both streams, identifies discrepancies, and generates Beads tickets.

Reference: Spec section 3.3
"""

from abc import abstractmethod

from pydantic import BaseModel, Field

from twinscribe.agents.base import AgentConfig, BaseAgent
from twinscribe.models.base import ModelTier
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.comparison import ComparisonResult, Discrepancy
from twinscribe.models.documentation import StreamOutput


class ComparatorInput(BaseModel):
    """Input to the comparator agent.

    Attributes:
        stream_a_output: Validated output from Stream A
        stream_b_output: Validated output from Stream B
        ground_truth_call_graph: Static analysis call graph (authoritative)
        iteration: Current iteration number
        previous_comparison: Previous comparison result if re-comparing
        resolved_discrepancies: Discrepancies resolved since last comparison
    """

    stream_a_output: StreamOutput = Field(..., description="Stream A validated output")
    stream_b_output: StreamOutput = Field(..., description="Stream B validated output")
    ground_truth_call_graph: CallGraph = Field(
        ..., description="Static analysis call graph (ground truth)"
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current iteration number",
    )
    previous_comparison: ComparisonResult | None = Field(
        default=None,
        description="Previous comparison if re-comparing",
    )
    resolved_discrepancies: list[str] = Field(
        default_factory=list,
        description="IDs of resolved discrepancies",
    )


class ComparatorConfig(AgentConfig):
    """Configuration specific to the comparator agent.

    Extends AgentConfig with comparator-specific settings.

    Attributes:
        confidence_threshold: Min confidence to auto-resolve (below = Beads)
        semantic_similarity_threshold: Min similarity for doc content match
        generate_beads_tickets: Whether to generate Beads tickets
        beads_project: Beads project key for tickets
        beads_ticket_priority_default: Default priority for tickets
    """

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for auto-resolution",
    )
    semantic_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for content match",
    )
    generate_beads_tickets: bool = Field(
        default=True,
        description="Generate Beads tickets for discrepancies",
    )
    beads_project: str = Field(
        default="LEGACY_DOC",
        description="Beads project key",
    )
    beads_ticket_priority_default: str = Field(
        default="Medium",
        description="Default ticket priority",
    )


# Default configuration for comparator agent
COMPARATOR_CONFIG = ComparatorConfig(
    agent_id="C",
    stream_id=None,  # Comparator is stream-agnostic
    model_tier=ModelTier.ARBITRATION,
    provider="anthropic",
    model_name="claude-opus-4-5-20251101",
    cost_per_million_input=15.0,
    cost_per_million_output=75.0,
    max_tokens=8192,
    temperature=0.0,
)


class ComparatorAgent(BaseAgent[ComparatorInput, ComparisonResult]):
    """Abstract base class for the comparator agent.

    The comparator agent compares outputs from both documentation streams,
    identifies discrepancies, consults ground truth, and generates Beads
    tickets for issues requiring human review.

    The agent receives:
    - Validated output from Stream A
    - Validated output from Stream B
    - Static analysis call graph (ground truth)
    - Previous iteration context

    The agent produces:
    - Comparison summary statistics
    - List of discrepancies with resolutions
    - Convergence status
    - Beads ticket references

    Decision Logic:
    1. If stream_a == stream_b: ACCEPT (identical)
    2. If discrepancy is call_graph_related: consult ground truth, accept matching
    3. If discrepancy is documentation_content and one is clearly better: accept better
    4. Otherwise: generate Beads ticket for human review

    Reference: Spec section 3.3
    """

    # System prompt template for comparator agent
    SYSTEM_PROMPT = """You are the arbitration agent responsible for comparing documentation outputs
from two independent streams and resolving discrepancies.

YOUR RESPONSIBILITIES:
1. Compare outputs component-by-component
2. Identify all discrepancies (structural and semantic)
3. For call graph discrepancies: consult static analysis (ground truth)
4. For documentation content discrepancies: use judgment or escalate
5. Generate Beads tickets for issues requiring human review
6. Track convergence progress

DECISION HIERARCHY:
1. Static analysis is AUTHORITATIVE for call graph accuracy
2. For semantic/content differences, prefer completeness and accuracy
3. When uncertain (confidence < 0.7), generate Beads ticket
4. Never guess - escalate unclear cases

You have access to:
- Stream A validated output
- Stream B validated output
- Static analysis call graph (GROUND TRUTH)
- Component source code (for context)

Output comparison results in the specified JSON schema.
Be thorough - missing a discrepancy is worse than flagging a false positive."""

    def __init__(self, config: ComparatorConfig) -> None:
        """Initialize the comparator agent.

        Args:
            config: Comparator configuration
        """
        super().__init__(config)
        self._comparator_config = config

    @property
    def comparator_config(self) -> ComparatorConfig:
        """Get comparator-specific configuration."""
        return self._comparator_config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the comparator agent.

        Sets up the LLM client and any required resources.
        """
        pass

    @abstractmethod
    async def process(self, input_data: ComparatorInput) -> ComparisonResult:
        """Compare outputs from both streams.

        Args:
            input_data: Input containing both stream outputs and ground truth

        Returns:
            Comparison result with discrepancies and convergence status

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the comparator agent."""
        pass

    @abstractmethod
    async def compare_component(
        self,
        component_id: str,
        stream_a_doc: dict | None,
        stream_b_doc: dict | None,
        ground_truth: CallGraph,
    ) -> list[Discrepancy]:
        """Compare documentation for a single component.

        Args:
            component_id: Component being compared
            stream_a_doc: Documentation from Stream A (or None if missing)
            stream_b_doc: Documentation from Stream B (or None if missing)
            ground_truth: Static analysis call graph

        Returns:
            List of discrepancies found for this component
        """
        pass

    @abstractmethod
    async def generate_beads_ticket(
        self,
        discrepancy: Discrepancy,
        stream_a_model: str,
        stream_b_model: str,
        source_code: str,
    ) -> dict:
        """Generate a Beads ticket for a discrepancy.

        Args:
            discrepancy: The discrepancy requiring human review
            stream_a_model: Model name for Stream A
            stream_b_model: Model name for Stream B
            source_code: Relevant source code snippet

        Returns:
            Beads ticket data dict ready for API submission
        """
        pass

    def _build_comparison_prompt(
        self,
        component_id: str,
        stream_a_doc: dict,
        stream_b_doc: dict,
        gt_callees: list,
        gt_callers: list,
    ) -> str:
        """Build prompt for comparing a single component.

        Args:
            component_id: Component ID
            stream_a_doc: Stream A documentation
            stream_b_doc: Stream B documentation
            gt_callees: Ground truth callees
            gt_callers: Ground truth callers

        Returns:
            Formatted comparison prompt
        """
        lines = [
            f"## Compare Component: {component_id}",
            "",
            "### Stream A Documentation",
            "```json",
            str(stream_a_doc),
            "```",
            "",
            "### Stream B Documentation",
            "```json",
            str(stream_b_doc),
            "```",
            "",
            "### Ground Truth (Static Analysis)",
            "**Callees:**",
        ]

        for edge in gt_callees:
            lines.append(f"- {edge.callee}")
        if not gt_callees:
            lines.append("_None_")

        lines.extend(
            [
                "",
                "**Callers:**",
            ]
        )

        for edge in gt_callers:
            lines.append(f"- {edge.caller}")
        if not gt_callers:
            lines.append("_None_")

        lines.extend(
            [
                "",
                "## Tasks",
                "1. Compare summaries and descriptions for semantic equivalence",
                "2. Compare parameter documentation",
                "3. Compare return documentation",
                "4. Compare exception documentation",
                "5. Compare call graph against ground truth",
                "6. For each discrepancy, determine resolution or escalate to Beads",
                "",
                "Output discrepancies in the specified JSON format.",
            ]
        )

        return "\n".join(lines)

    def _determine_resolution(
        self,
        discrepancy_type: str,
        stream_a_value: any,
        stream_b_value: any,
        ground_truth: any,
    ) -> tuple[str, float]:
        """Determine resolution for a discrepancy.

        Args:
            discrepancy_type: Type of discrepancy
            stream_a_value: Value from Stream A
            stream_b_value: Value from Stream B
            ground_truth: Ground truth value (if applicable)

        Returns:
            Tuple of (resolution_action, confidence)
        """
        # Call graph discrepancies: use ground truth
        if discrepancy_type.startswith("call_graph"):
            if ground_truth is not None:
                if stream_a_value == ground_truth:
                    return ("accept_stream_a", 0.99)
                elif stream_b_value == ground_truth:
                    return ("accept_stream_b", 0.99)
                else:
                    return ("accept_ground_truth", 0.99)

        # Documentation content: need judgment
        return ("needs_human_review", 0.5)

    def _get_response_schema(self) -> dict:
        """Get the JSON schema for comparison output.

        Returns:
            JSON schema dict for discrepancy list
        """
        return {
            "type": "object",
            "properties": {
                "discrepancies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "discrepancy_id": {"type": "string"},
                            "component_id": {"type": "string"},
                            "type": {"type": "string"},
                            "stream_a_value": {},
                            "stream_b_value": {},
                            "ground_truth": {},
                            "resolution": {"type": "string"},
                            "confidence": {"type": "number"},
                            "requires_beads": {"type": "boolean"},
                            "beads_ticket": {
                                "type": ["object", "null"],
                                "properties": {
                                    "summary": {"type": "string"},
                                    "description": {"type": "string"},
                                    "priority": {"type": "string"},
                                },
                            },
                        },
                        "required": [
                            "discrepancy_id",
                            "component_id",
                            "type",
                            "resolution",
                            "confidence",
                        ],
                    },
                },
                "identical_components": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["discrepancies"],
        }
