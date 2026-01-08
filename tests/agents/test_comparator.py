"""
Tests for ComparatorAgent interface and implementations.

Tests cover:
- ComparatorAgent initialization and lifecycle
- process() comparison logic for stream outputs
- compare_component() for single component comparison
- Call graph discrepancy resolution using ground truth
- Documentation content comparison
- Beads ticket generation for human review
- Convergence status calculation
- Edge cases and error conditions

All tests use mocks to avoid real API calls.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.agents.comparator import (
    COMPARATOR_CONFIG,
    ComparatorAgent,
    ComparatorConfig,
    ComparatorInput,
)
from twinscribe.models.base import (
    CallType,
    DiscrepancyType,
    ModelTier,
    ResolutionAction,
    StreamId,
)
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.models.comparison import (
    BeadsTicketRef,
    ComparatorMetadata,
    ComparisonResult,
    ComparisonSummary,
    ConvergenceStatus,
    Discrepancy,
)
from twinscribe.models.components import (
    ComponentDocumentation,
    ParameterDoc,
    ReturnDoc,
)
from twinscribe.models.documentation import (
    CalleeRef,
    CallerRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
    StreamOutput,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def comparator_config() -> ComparatorConfig:
    """Create a ComparatorConfig for testing."""
    return ComparatorConfig(
        agent_id="C",
        stream_id=None,
        model_tier=ModelTier.ARBITRATION,
        provider="anthropic",
        model_name="claude-opus-4-5-20251101",
        cost_per_million_input=15.0,
        cost_per_million_output=75.0,
        max_tokens=8192,
        temperature=0.0,
        confidence_threshold=0.7,
        semantic_similarity_threshold=0.95,
        generate_beads_tickets=True,
        beads_project="LEGACY_DOC",
        beads_ticket_priority_default="Medium",
    )


@pytest.fixture
def sample_ground_truth_call_graph() -> CallGraph:
    """Create a ground truth CallGraph from static analysis."""
    return CallGraph(
        edges=[
            CallEdge(
                caller="sample_module.Calculator.multiply",
                callee="sample_module.helper_function",
                call_site_line=22,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            CallEdge(
                caller="sample_module.Calculator.add",
                callee="builtins.round",
                call_site_line=18,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            CallEdge(
                caller="sample_module.AdvancedCalculator.compute_complex",
                callee="sample_module.Calculator.add",
                call_site_line=35,
                call_type=CallType.LOOP,
                confidence=1.0,
            ),
            CallEdge(
                caller="sample_module.AdvancedCalculator.compute_complex",
                callee="sample_module.Calculator.multiply",
                call_site_line=36,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
        ],
        source="pycg",
    )


@pytest.fixture
def stream_a_documentation_output() -> DocumentationOutput:
    """Create a sample DocumentationOutput from Stream A."""
    return DocumentationOutput(
        component_id="sample_module.Calculator.add",
        documentation=ComponentDocumentation(
            summary="Add two numbers.",
            description="Adds two floating point numbers and rounds the result.",
            parameters=[
                ParameterDoc(name="a", type="float", description="First number"),
                ParameterDoc(name="b", type="float", description="Second number"),
            ],
            returns=ReturnDoc(type="float", description="Sum of a and b"),
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=[
                CallerRef(
                    component_id="sample_module.AdvancedCalculator.compute_complex",
                    call_site_line=35,
                    call_type=CallType.LOOP,
                ),
            ],
            callees=[
                CalleeRef(
                    component_id="builtins.round",
                    call_site_line=18,
                    call_type=CallType.DIRECT,
                ),
            ],
        ),
        metadata=DocumenterMetadata(
            agent_id="A1",
            stream_id=StreamId.STREAM_A,
            model="claude-sonnet-4-5-20250929",
            confidence=0.92,
            processing_order=1,
            token_count=500,
        ),
    )


@pytest.fixture
def stream_b_documentation_output() -> DocumentationOutput:
    """Create a sample DocumentationOutput from Stream B (identical to A)."""
    return DocumentationOutput(
        component_id="sample_module.Calculator.add",
        documentation=ComponentDocumentation(
            summary="Add two numbers.",
            description="Adds two floating point numbers and rounds the result.",
            parameters=[
                ParameterDoc(name="a", type="float", description="First number"),
                ParameterDoc(name="b", type="float", description="Second number"),
            ],
            returns=ReturnDoc(type="float", description="Sum of a and b"),
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=[
                CallerRef(
                    component_id="sample_module.AdvancedCalculator.compute_complex",
                    call_site_line=35,
                    call_type=CallType.LOOP,
                ),
            ],
            callees=[
                CalleeRef(
                    component_id="builtins.round",
                    call_site_line=18,
                    call_type=CallType.DIRECT,
                ),
            ],
        ),
        metadata=DocumenterMetadata(
            agent_id="B1",
            stream_id=StreamId.STREAM_B,
            model="gpt-4o",
            confidence=0.90,
            processing_order=1,
            token_count=480,
        ),
    )


@pytest.fixture
def stream_b_documentation_with_discrepancy() -> DocumentationOutput:
    """Create Stream B output with call graph discrepancy."""
    return DocumentationOutput(
        component_id="sample_module.Calculator.add",
        documentation=ComponentDocumentation(
            summary="Add two numbers.",
            description="Adds two floating point numbers and rounds the result.",
            parameters=[
                ParameterDoc(name="a", type="float", description="First number"),
                ParameterDoc(name="b", type="float", description="Second number"),
            ],
            returns=ReturnDoc(type="float", description="Sum of a and b"),
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=[
                CallerRef(
                    component_id="sample_module.AdvancedCalculator.compute_complex",
                    call_site_line=35,
                    call_type=CallType.LOOP,
                ),
            ],
            callees=[
                # Missing builtins.round - discrepancy
                CalleeRef(
                    component_id="sample_module.nonexistent_function",
                    call_site_line=20,
                    call_type=CallType.DIRECT,
                ),
            ],
        ),
        metadata=DocumenterMetadata(
            agent_id="B1",
            stream_id=StreamId.STREAM_B,
            model="gpt-4o",
            confidence=0.85,
            processing_order=1,
            token_count=480,
        ),
    )


@pytest.fixture
def stream_b_documentation_with_content_discrepancy() -> DocumentationOutput:
    """Create Stream B output with documentation content discrepancy."""
    return DocumentationOutput(
        component_id="sample_module.Calculator.add",
        documentation=ComponentDocumentation(
            summary="Add two numbers together.",  # Slightly different
            description="This method adds two floating point numbers and returns the rounded result to precision.",  # Different
            parameters=[
                ParameterDoc(
                    name="a", type="float", description="First operand"
                ),  # Different description
                ParameterDoc(
                    name="b", type="float", description="Second operand"
                ),  # Different description
            ],
            returns=ReturnDoc(type="float", description="Rounded sum"),  # Different
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=[
                CallerRef(
                    component_id="sample_module.AdvancedCalculator.compute_complex",
                    call_site_line=35,
                    call_type=CallType.LOOP,
                ),
            ],
            callees=[
                CalleeRef(
                    component_id="builtins.round",
                    call_site_line=18,
                    call_type=CallType.DIRECT,
                ),
            ],
        ),
        metadata=DocumenterMetadata(
            agent_id="B1",
            stream_id=StreamId.STREAM_B,
            model="gpt-4o",
            confidence=0.88,
            processing_order=1,
            token_count=490,
        ),
    )


@pytest.fixture
def stream_a_output(
    stream_a_documentation_output: DocumentationOutput,
) -> StreamOutput:
    """Create a StreamOutput for Stream A."""
    output = StreamOutput(stream_id=StreamId.STREAM_A)
    output.add_output(stream_a_documentation_output)
    return output


@pytest.fixture
def stream_b_output(
    stream_b_documentation_output: DocumentationOutput,
) -> StreamOutput:
    """Create a StreamOutput for Stream B (identical)."""
    output = StreamOutput(stream_id=StreamId.STREAM_B)
    output.add_output(stream_b_documentation_output)
    return output


@pytest.fixture
def stream_b_output_with_discrepancy(
    stream_b_documentation_with_discrepancy: DocumentationOutput,
) -> StreamOutput:
    """Create a StreamOutput for Stream B with call graph discrepancy."""
    output = StreamOutput(stream_id=StreamId.STREAM_B)
    output.add_output(stream_b_documentation_with_discrepancy)
    return output


@pytest.fixture
def stream_b_output_with_content_discrepancy(
    stream_b_documentation_with_content_discrepancy: DocumentationOutput,
) -> StreamOutput:
    """Create a StreamOutput for Stream B with content discrepancy."""
    output = StreamOutput(stream_id=StreamId.STREAM_B)
    output.add_output(stream_b_documentation_with_content_discrepancy)
    return output


@pytest.fixture
def comparator_input_identical(
    stream_a_output: StreamOutput,
    stream_b_output: StreamOutput,
    sample_ground_truth_call_graph: CallGraph,
) -> ComparatorInput:
    """Create ComparatorInput with identical stream outputs."""
    return ComparatorInput(
        stream_a_output=stream_a_output,
        stream_b_output=stream_b_output,
        ground_truth_call_graph=sample_ground_truth_call_graph,
        iteration=1,
    )


@pytest.fixture
def comparator_input_with_call_graph_discrepancy(
    stream_a_output: StreamOutput,
    stream_b_output_with_discrepancy: StreamOutput,
    sample_ground_truth_call_graph: CallGraph,
) -> ComparatorInput:
    """Create ComparatorInput with call graph discrepancy."""
    return ComparatorInput(
        stream_a_output=stream_a_output,
        stream_b_output=stream_b_output_with_discrepancy,
        ground_truth_call_graph=sample_ground_truth_call_graph,
        iteration=1,
    )


@pytest.fixture
def comparator_input_with_content_discrepancy(
    stream_a_output: StreamOutput,
    stream_b_output_with_content_discrepancy: StreamOutput,
    sample_ground_truth_call_graph: CallGraph,
) -> ComparatorInput:
    """Create ComparatorInput with documentation content discrepancy."""
    return ComparatorInput(
        stream_a_output=stream_a_output,
        stream_b_output=stream_b_output_with_content_discrepancy,
        ground_truth_call_graph=sample_ground_truth_call_graph,
        iteration=1,
    )


@pytest.fixture
def sample_discrepancy_call_graph() -> Discrepancy:
    """Create a sample call graph discrepancy."""
    return Discrepancy(
        discrepancy_id="disc_001",
        component_id="sample_module.Calculator.add",
        type=DiscrepancyType.CALL_GRAPH_EDGE,
        stream_a_value={"callee": "builtins.round", "line": 18},
        stream_b_value={"callee": "sample_module.nonexistent_function", "line": 20},
        ground_truth={"callee": "builtins.round", "line": 18},
        resolution=ResolutionAction.ACCEPT_STREAM_A,
        confidence=0.99,
        requires_beads=False,
        iteration_found=1,
    )


@pytest.fixture
def sample_discrepancy_content() -> Discrepancy:
    """Create a sample documentation content discrepancy."""
    return Discrepancy(
        discrepancy_id="disc_002",
        component_id="sample_module.Calculator.add",
        type=DiscrepancyType.DOCUMENTATION_CONTENT,
        stream_a_value="Add two numbers.",
        stream_b_value="Add two numbers together.",
        ground_truth=None,
        resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
        confidence=0.5,
        requires_beads=True,
        beads_ticket=BeadsTicketRef(
            summary="[AI-DOC] documentation_content: sample_module.Calculator.add",
            description="Stream A and B have different summaries.",
            priority="Medium",
        ),
        iteration_found=1,
    )


@pytest.fixture
def sample_comparison_result() -> ComparisonResult:
    """Create a sample ComparisonResult."""
    return ComparisonResult(
        comparison_id="cmp_20260107_001",
        iteration=1,
        summary=ComparisonSummary(
            total_components=5,
            identical=4,
            discrepancies=1,
            resolved_by_ground_truth=1,
            requires_human_review=0,
        ),
        discrepancies=[],
        convergence_status=ConvergenceStatus(
            converged=True,
            blocking_discrepancies=0,
            recommendation="finalize",
        ),
        metadata=ComparatorMetadata(
            agent_id="C",
            model="claude-opus-4-5-20251101",
            comparison_duration_ms=2500,
            token_count=1500,
        ),
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for comparator."""
    client = MagicMock()
    client.send_message = AsyncMock()
    return client


# =============================================================================
# Test Classes
# =============================================================================


class TestComparatorConfig:
    """Tests for ComparatorConfig model."""

    def test_valid_config(self, comparator_config: ComparatorConfig):
        """Test creating a valid ComparatorConfig."""
        assert comparator_config.agent_id == "C"
        assert comparator_config.stream_id is None
        assert comparator_config.model_tier == ModelTier.ARBITRATION
        assert comparator_config.confidence_threshold == 0.7
        assert comparator_config.semantic_similarity_threshold == 0.95
        assert comparator_config.generate_beads_tickets is True
        assert comparator_config.beads_project == "LEGACY_DOC"

    def test_default_config(self):
        """Test that COMPARATOR_CONFIG has correct defaults."""
        assert COMPARATOR_CONFIG.agent_id == "C"
        assert COMPARATOR_CONFIG.model_name == "claude-opus-4-5-20251101"
        assert COMPARATOR_CONFIG.temperature == 0.0

    def test_config_validation_confidence_threshold(self):
        """Test that confidence_threshold is validated (0-1)."""
        with pytest.raises(ValueError):
            ComparatorConfig(
                agent_id="C",
                model_tier=ModelTier.ARBITRATION,
                provider="anthropic",
                model_name="claude-opus-4-5-20251101",
                cost_per_million_input=15.0,
                cost_per_million_output=75.0,
                confidence_threshold=1.5,  # Invalid
            )

    def test_config_validation_similarity_threshold(self):
        """Test that semantic_similarity_threshold is validated (0-1)."""
        with pytest.raises(ValueError):
            ComparatorConfig(
                agent_id="C",
                model_tier=ModelTier.ARBITRATION,
                provider="anthropic",
                model_name="claude-opus-4-5-20251101",
                cost_per_million_input=15.0,
                cost_per_million_output=75.0,
                semantic_similarity_threshold=-0.1,  # Invalid
            )


class TestComparatorInput:
    """Tests for ComparatorInput model."""

    def test_valid_input(self, comparator_input_identical: ComparatorInput):
        """Test creating valid ComparatorInput."""
        assert comparator_input_identical.stream_a_output.stream_id == StreamId.STREAM_A
        assert comparator_input_identical.stream_b_output.stream_id == StreamId.STREAM_B
        assert comparator_input_identical.iteration == 1

    def test_input_with_previous_comparison(
        self,
        stream_a_output: StreamOutput,
        stream_b_output: StreamOutput,
        sample_ground_truth_call_graph: CallGraph,
        sample_comparison_result: ComparisonResult,
    ):
        """Test ComparatorInput with previous comparison context."""
        input_data = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=sample_ground_truth_call_graph,
            iteration=2,
            previous_comparison=sample_comparison_result,
            resolved_discrepancies=["disc_001"],
        )
        assert input_data.iteration == 2
        assert input_data.previous_comparison is not None
        assert len(input_data.resolved_discrepancies) == 1


class TestDiscrepancy:
    """Tests for Discrepancy model."""

    def test_call_graph_discrepancy(self, sample_discrepancy_call_graph: Discrepancy):
        """Test creating a call graph discrepancy."""
        disc = sample_discrepancy_call_graph
        assert disc.type == DiscrepancyType.CALL_GRAPH_EDGE
        assert disc.is_call_graph_related is True
        assert disc.resolution == ResolutionAction.ACCEPT_STREAM_A
        assert disc.confidence == 0.99
        assert disc.requires_beads is False

    def test_content_discrepancy(self, sample_discrepancy_content: Discrepancy):
        """Test creating a documentation content discrepancy."""
        disc = sample_discrepancy_content
        assert disc.type == DiscrepancyType.DOCUMENTATION_CONTENT
        assert disc.is_call_graph_related is False
        assert disc.requires_beads is True
        assert disc.beads_ticket is not None

    def test_is_resolved_property(self):
        """Test is_resolved computed property."""
        resolved_disc = Discrepancy(
            discrepancy_id="disc_001",
            component_id="test.component",
            type=DiscrepancyType.CALL_GRAPH_EDGE,
            resolution=ResolutionAction.ACCEPT_STREAM_A,
            confidence=0.99,
        )
        assert resolved_disc.is_resolved is True

        unresolved_disc = Discrepancy(
            discrepancy_id="disc_002",
            component_id="test.component",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
            confidence=0.5,
        )
        assert unresolved_disc.is_resolved is False

    def test_is_blocking_property(self):
        """Test is_blocking computed property."""
        # High confidence resolved = not blocking
        non_blocking = Discrepancy(
            discrepancy_id="disc_001",
            component_id="test.component",
            type=DiscrepancyType.CALL_GRAPH_EDGE,
            resolution=ResolutionAction.ACCEPT_STREAM_A,
            confidence=0.99,
        )
        assert non_blocking.is_blocking is False

        # Low confidence or unresolved = blocking
        blocking = Discrepancy(
            discrepancy_id="disc_002",
            component_id="test.component",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
            confidence=0.5,
        )
        assert blocking.is_blocking is True


class TestConvergenceStatus:
    """Tests for ConvergenceStatus model."""

    def test_converged_status(self):
        """Test creating converged status."""
        status = ConvergenceStatus(
            converged=True,
            blocking_discrepancies=0,
            recommendation="finalize",
        )
        assert status.converged is True
        assert status.blocking_discrepancies == 0

    def test_not_converged_status(self):
        """Test creating not-converged status."""
        status = ConvergenceStatus(
            converged=False,
            blocking_discrepancies=3,
            recommendation="continue_iteration",
        )
        assert status.converged is False
        assert status.blocking_discrepancies == 3


class TestComparisonSummary:
    """Tests for ComparisonSummary model."""

    def test_agreement_rate_calculation(self):
        """Test agreement_rate computed property."""
        summary = ComparisonSummary(
            total_components=10,
            identical=8,
            discrepancies=2,
        )
        assert summary.agreement_rate == 0.8

    def test_agreement_rate_all_identical(self):
        """Test agreement_rate when all components match."""
        summary = ComparisonSummary(
            total_components=10,
            identical=10,
            discrepancies=0,
        )
        assert summary.agreement_rate == 1.0

    def test_agreement_rate_empty(self):
        """Test agreement_rate when no components processed."""
        summary = ComparisonSummary(
            total_components=0,
            identical=0,
        )
        assert summary.agreement_rate == 1.0  # Default to 1.0 when empty


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_valid_result(self, sample_comparison_result: ComparisonResult):
        """Test creating a valid ComparisonResult."""
        assert sample_comparison_result.comparison_id == "cmp_20260107_001"
        assert sample_comparison_result.iteration == 1
        assert sample_comparison_result.is_converged is True

    def test_get_discrepancies_for_component(
        self,
        sample_discrepancy_call_graph: Discrepancy,
        sample_discrepancy_content: Discrepancy,
    ):
        """Test filtering discrepancies by component."""
        result = ComparisonResult(
            comparison_id="cmp_001",
            iteration=1,
            discrepancies=[
                sample_discrepancy_call_graph,
                sample_discrepancy_content,
            ],
            metadata=ComparatorMetadata(
                agent_id="C",
                model="claude-opus-4-5-20251101",
            ),
        )
        component_discs = result.get_discrepancies_for_component("sample_module.Calculator.add")
        assert len(component_discs) == 2

    def test_get_blocking_discrepancies(self, sample_discrepancy_content: Discrepancy):
        """Test getting only blocking discrepancies."""
        result = ComparisonResult(
            comparison_id="cmp_001",
            iteration=1,
            discrepancies=[sample_discrepancy_content],  # This one is blocking
            metadata=ComparatorMetadata(
                agent_id="C",
                model="claude-opus-4-5-20251101",
            ),
        )
        blocking = result.get_blocking_discrepancies()
        assert len(blocking) == 1

    def test_get_beads_required(self, sample_discrepancy_content: Discrepancy):
        """Test getting discrepancies requiring Beads tickets."""
        result = ComparisonResult(
            comparison_id="cmp_001",
            iteration=1,
            discrepancies=[sample_discrepancy_content],  # requires_beads=True
            metadata=ComparatorMetadata(
                agent_id="C",
                model="claude-opus-4-5-20251101",
            ),
        )
        beads_needed = result.get_beads_required()
        assert len(beads_needed) == 1


# =============================================================================
# Tests for ComparatorAgent (Abstract Base Class)
# These tests use a concrete mock implementation
# =============================================================================


class MockComparatorAgent(ComparatorAgent):
    """Concrete mock implementation of ComparatorAgent for testing."""

    def __init__(self, config: ComparatorConfig):
        super().__init__(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self._initialized = True

    async def process(self, input_data: ComparatorInput) -> ComparisonResult:
        """Process comparison (mock implementation)."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        # Mock comparison logic - return empty result
        return ComparisonResult(
            comparison_id=f"cmp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            iteration=input_data.iteration,
            summary=ComparisonSummary(
                total_components=len(input_data.stream_a_output.outputs),
                identical=len(input_data.stream_a_output.outputs),
                discrepancies=0,
            ),
            convergence_status=ConvergenceStatus(
                converged=True,
                blocking_discrepancies=0,
                recommendation="finalize",
            ),
            metadata=ComparatorMetadata(
                agent_id="C",
                model=self._config.model_name,
            ),
        )

    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self._initialized = False

    async def compare(
        self,
        stream_a_output: StreamOutput,
        stream_b_output: StreamOutput,
        ground_truth_call_graph: CallGraph,
        iteration: int = 1,
    ) -> ComparisonResult:
        """Compare outputs from both streams (mock implementation)."""
        input_data = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=iteration,
        )
        return await self.process(input_data)

    async def compare_component(
        self,
        component_id: str,
        stream_a_doc: dict | None,
        stream_b_doc: dict | None,
        ground_truth: CallGraph,
    ) -> list[Discrepancy]:
        """Compare single component (mock implementation)."""
        return []

    async def generate_beads_ticket(
        self,
        discrepancy: Discrepancy,
        stream_a_model: str,
        stream_b_model: str,
        source_code: str,
    ) -> dict:
        """Generate Beads ticket (mock implementation)."""
        return {
            "project": self._comparator_config.beads_project,
            "summary": f"[AI-DOC] {discrepancy.type.value}: {discrepancy.component_id}",
            "description": "Discrepancy found between streams",
            "priority": self._comparator_config.beads_ticket_priority_default,
        }


class TestComparatorAgentBase:
    """Tests for ComparatorAgent base functionality."""

    def test_agent_initialization(self, comparator_config: ComparatorConfig):
        """Test agent initialization with config."""
        agent = MockComparatorAgent(comparator_config)
        assert agent.config == comparator_config
        assert agent.comparator_config == comparator_config

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self, comparator_config: ComparatorConfig):
        """Test that initialize() prepares the agent."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()
        assert agent._initialized is True

    @pytest.mark.asyncio
    async def test_process_requires_initialization(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_identical: ComparatorInput,
    ):
        """Test that process() requires initialization."""
        agent = MockComparatorAgent(comparator_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await agent.process(comparator_input_identical)

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self, comparator_config: ComparatorConfig):
        """Test that shutdown() cleans up resources."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()
        assert agent._initialized is True
        await agent.shutdown()
        assert agent._initialized is False


class TestComparatorAgentProcessing:
    """Tests for ComparatorAgent processing logic."""

    @pytest.mark.asyncio
    async def test_process_identical_streams(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_identical: ComparatorInput,
    ):
        """Test processing when both streams produce identical output."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        result = await agent.process(comparator_input_identical)

        assert result.is_converged is True
        assert result.summary.discrepancies == 0
        assert len(result.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_process_with_call_graph_discrepancy(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_with_call_graph_discrepancy: ComparatorInput,
    ):
        """Test processing when streams have call graph discrepancy."""
        # This test will need implementation with actual comparison logic
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        # With mock implementation, we get empty result
        # Real implementation should detect discrepancy
        result = await agent.process(comparator_input_with_call_graph_discrepancy)
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_with_content_discrepancy(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_with_content_discrepancy: ComparatorInput,
    ):
        """Test processing when streams have documentation content discrepancy."""
        # This test will need implementation with actual comparison logic
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        result = await agent.process(comparator_input_with_content_discrepancy)
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_respects_iteration_number(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_identical: ComparatorInput,
    ):
        """Test that process respects iteration number from input."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        comparator_input_identical.iteration = 3
        result = await agent.process(comparator_input_identical)

        assert result.iteration == 3


class TestComparatorComponentComparison:
    """Tests for compare_component method."""

    @pytest.mark.asyncio
    async def test_compare_component_identical(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test comparing identical component documentation."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        doc = {
            "summary": "Add two numbers.",
            "callees": ["builtins.round"],
        }
        discrepancies = await agent.compare_component(
            component_id="sample_module.Calculator.add",
            stream_a_doc=doc,
            stream_b_doc=doc,  # Identical
            ground_truth=sample_ground_truth_call_graph,
        )

        assert len(discrepancies) == 0

    @pytest.mark.asyncio
    async def test_compare_component_missing_in_stream_a(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test comparing when component missing in Stream A."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        discrepancies = await agent.compare_component(
            component_id="sample_module.Calculator.add",
            stream_a_doc=None,  # Missing
            stream_b_doc={"summary": "Add two numbers."},
            ground_truth=sample_ground_truth_call_graph,
        )

        # Should detect missing component
        # Implementation needed to assert correct behavior
        pass

    @pytest.mark.asyncio
    async def test_compare_component_missing_in_stream_b(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test comparing when component missing in Stream B."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        discrepancies = await agent.compare_component(
            component_id="sample_module.Calculator.add",
            stream_a_doc={"summary": "Add two numbers."},
            stream_b_doc=None,  # Missing
            ground_truth=sample_ground_truth_call_graph,
        )

        # Should detect missing component
        pass

    @pytest.mark.asyncio
    async def test_compare_component_call_graph_against_ground_truth(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test that call graph comparison uses ground truth."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        # Stream A matches ground truth
        stream_a_doc = {
            "callees": [{"component_id": "builtins.round", "line": 18}],
        }
        # Stream B has wrong callee
        stream_b_doc = {
            "callees": [{"component_id": "wrong.function", "line": 20}],
        }

        discrepancies = await agent.compare_component(
            component_id="sample_module.Calculator.add",
            stream_a_doc=stream_a_doc,
            stream_b_doc=stream_b_doc,
            ground_truth=sample_ground_truth_call_graph,
        )

        # Should resolve in favor of Stream A (matches ground truth)
        # Implementation needed
        pass


class TestComparatorGroundTruthResolution:
    """Tests for ground truth resolution logic."""

    def test_determine_resolution_matches_stream_a(self, comparator_config: ComparatorConfig):
        """Test resolution when Stream A matches ground truth."""
        agent = MockComparatorAgent(comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value="correct_callee",
            stream_b_value="wrong_callee",
            ground_truth="correct_callee",
        )

        assert resolution == "accept_stream_a"
        assert confidence == 0.99

    def test_determine_resolution_matches_stream_b(self, comparator_config: ComparatorConfig):
        """Test resolution when Stream B matches ground truth."""
        agent = MockComparatorAgent(comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value="wrong_callee",
            stream_b_value="correct_callee",
            ground_truth="correct_callee",
        )

        assert resolution == "accept_stream_b"
        assert confidence == 0.99

    def test_determine_resolution_neither_matches(self, comparator_config: ComparatorConfig):
        """Test resolution when neither stream matches ground truth."""
        agent = MockComparatorAgent(comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value="wrong_a",
            stream_b_value="wrong_b",
            ground_truth="correct",
        )

        assert resolution == "accept_ground_truth"
        assert confidence == 0.99

    def test_determine_resolution_content_needs_review(self, comparator_config: ComparatorConfig):
        """Test resolution for content discrepancy needs human review."""
        agent = MockComparatorAgent(comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="documentation_content",
            stream_a_value="Summary version A",
            stream_b_value="Summary version B",
            ground_truth=None,  # No ground truth for content
        )

        assert resolution == "needs_human_review"
        assert confidence == 0.5


class TestComparatorBeadsTicketGeneration:
    """Tests for Beads ticket generation."""

    @pytest.mark.asyncio
    async def test_generate_beads_ticket_for_content_discrepancy(
        self,
        comparator_config: ComparatorConfig,
        sample_discrepancy_content: Discrepancy,
    ):
        """Test generating Beads ticket for content discrepancy."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        ticket = await agent.generate_beads_ticket(
            discrepancy=sample_discrepancy_content,
            stream_a_model="claude-sonnet-4-5-20250929",
            stream_b_model="gpt-4o",
            source_code="def add(self, a, b): return a + b",
        )

        assert ticket["project"] == "LEGACY_DOC"
        assert "summary" in ticket
        assert sample_discrepancy_content.component_id in ticket["summary"]

    @pytest.mark.asyncio
    async def test_beads_ticket_includes_both_stream_values(
        self,
        comparator_config: ComparatorConfig,
        sample_discrepancy_content: Discrepancy,
    ):
        """Test that Beads ticket includes values from both streams."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        ticket = await agent.generate_beads_ticket(
            discrepancy=sample_discrepancy_content,
            stream_a_model="claude-sonnet-4-5-20250929",
            stream_b_model="gpt-4o",
            source_code="def add(self, a, b): return a + b",
        )

        # Implementation should include stream values in description
        assert "description" in ticket

    @pytest.mark.asyncio
    async def test_beads_ticket_respects_priority_config(
        self,
        comparator_config: ComparatorConfig,
        sample_discrepancy_content: Discrepancy,
    ):
        """Test that Beads ticket uses configured default priority."""
        comparator_config.beads_ticket_priority_default = "High"
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        ticket = await agent.generate_beads_ticket(
            discrepancy=sample_discrepancy_content,
            stream_a_model="claude-sonnet-4-5-20250929",
            stream_b_model="gpt-4o",
            source_code="def add(self, a, b): return a + b",
        )

        assert ticket["priority"] == "High"


class TestComparatorConvergence:
    """Tests for convergence calculation."""

    @pytest.mark.asyncio
    async def test_convergence_when_all_identical(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_identical: ComparatorInput,
    ):
        """Test convergence when all components are identical."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        result = await agent.process(comparator_input_identical)

        assert result.convergence_status.converged is True
        assert result.convergence_status.blocking_discrepancies == 0
        assert result.convergence_status.recommendation == "finalize"

    @pytest.mark.asyncio
    async def test_no_convergence_with_blocking_discrepancies(
        self, comparator_config: ComparatorConfig
    ):
        """Test that blocking discrepancies prevent convergence."""
        # Implementation needed with real comparison logic
        pass

    @pytest.mark.asyncio
    async def test_convergence_after_resolution(
        self,
        comparator_config: ComparatorConfig,
        stream_a_output: StreamOutput,
        stream_b_output: StreamOutput,
        sample_ground_truth_call_graph: CallGraph,
        sample_comparison_result: ComparisonResult,
    ):
        """Test convergence when previous discrepancies are resolved."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        input_data = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=sample_ground_truth_call_graph,
            iteration=2,
            previous_comparison=sample_comparison_result,
            resolved_discrepancies=["disc_001"],
        )

        result = await agent.process(input_data)
        # Should recognize progress from previous iteration
        assert result.iteration == 2


class TestComparatorPromptBuilding:
    """Tests for prompt building methods."""

    def test_build_comparison_prompt(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test building comparison prompt for LLM."""
        agent = MockComparatorAgent(comparator_config)

        prompt = agent._build_comparison_prompt(
            component_id="sample_module.Calculator.add",
            stream_a_doc={"summary": "Add numbers"},
            stream_b_doc={"summary": "Add numbers together"},
            gt_callees=sample_ground_truth_call_graph.get_callees("sample_module.Calculator.add"),
            gt_callers=sample_ground_truth_call_graph.get_callers("sample_module.Calculator.add"),
        )

        assert "sample_module.Calculator.add" in prompt
        assert "Stream A" in prompt
        assert "Stream B" in prompt
        assert "Ground Truth" in prompt

    def test_get_response_schema(self, comparator_config: ComparatorConfig):
        """Test that response schema is valid."""
        agent = MockComparatorAgent(comparator_config)
        schema = agent._get_response_schema()

        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "discrepancies" in schema["properties"]


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestComparatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_stream_outputs(
        self,
        comparator_config: ComparatorConfig,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test comparison with empty stream outputs."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        input_data = ComparatorInput(
            stream_a_output=StreamOutput(stream_id=StreamId.STREAM_A),
            stream_b_output=StreamOutput(stream_id=StreamId.STREAM_B),
            ground_truth_call_graph=sample_ground_truth_call_graph,
            iteration=1,
        )

        result = await agent.process(input_data)
        assert result.summary.total_components == 0

    @pytest.mark.asyncio
    async def test_mismatched_component_sets(
        self,
        comparator_config: ComparatorConfig,
        stream_a_documentation_output: DocumentationOutput,
        sample_ground_truth_call_graph: CallGraph,
    ):
        """Test comparison when streams have different component sets."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        # Stream A has one component, Stream B is empty
        stream_a = StreamOutput(stream_id=StreamId.STREAM_A)
        stream_a.add_output(stream_a_documentation_output)
        stream_b = StreamOutput(stream_id=StreamId.STREAM_B)

        input_data = ComparatorInput(
            stream_a_output=stream_a,
            stream_b_output=stream_b,
            ground_truth_call_graph=sample_ground_truth_call_graph,
            iteration=1,
        )

        result = await agent.process(input_data)
        # Should detect missing component in Stream B
        # Implementation needed
        pass

    @pytest.mark.asyncio
    async def test_max_iteration_handling(
        self,
        comparator_config: ComparatorConfig,
        comparator_input_identical: ComparatorInput,
    ):
        """Test handling of maximum iteration reached."""
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        # Simulate high iteration count
        comparator_input_identical.iteration = 5

        result = await agent.process(comparator_input_identical)
        # Should handle gracefully
        assert result.iteration == 5

    @pytest.mark.asyncio
    async def test_beads_generation_disabled(
        self,
        comparator_config: ComparatorConfig,
        sample_discrepancy_content: Discrepancy,
    ):
        """Test that Beads generation can be disabled."""
        comparator_config.generate_beads_tickets = False
        agent = MockComparatorAgent(comparator_config)
        await agent.initialize()

        # When disabled, should not generate tickets
        # Implementation will need to respect this config
        pass

    def test_confidence_threshold_boundary(self, comparator_config: ComparatorConfig):
        """Test behavior at confidence threshold boundary."""
        # Test with confidence exactly at threshold
        comparator_config.confidence_threshold = 0.7
        agent = MockComparatorAgent(comparator_config)

        # Resolution with exactly 0.7 confidence
        # Should be considered above threshold
        pass

    @pytest.mark.asyncio
    async def test_process_with_all_discrepancy_types(self, comparator_config: ComparatorConfig):
        """Test processing with multiple types of discrepancies."""
        # Create inputs with call graph, content, parameter, and type annotation discrepancies
        # Implementation needed with complex test data
        pass


class TestComparatorSystemPrompt:
    """Tests for system prompt configuration."""

    def test_system_prompt_content(self, comparator_config: ComparatorConfig):
        """Test that system prompt contains required instructions."""
        assert "arbitration" in MockComparatorAgent.SYSTEM_PROMPT.lower()
        assert "ground truth" in MockComparatorAgent.SYSTEM_PROMPT.lower()
        assert "discrepan" in MockComparatorAgent.SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_static_analysis(self, comparator_config: ComparatorConfig):
        """Test that system prompt emphasizes static analysis authority."""
        assert (
            "static analysis" in MockComparatorAgent.SYSTEM_PROMPT.lower()
            or "authoritative" in MockComparatorAgent.SYSTEM_PROMPT.lower()
        )
