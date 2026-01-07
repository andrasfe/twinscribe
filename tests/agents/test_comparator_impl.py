"""
Tests for ConcreteComparatorAgent implementation.

Tests the comparison logic, discrepancy detection, ground truth resolution,
and Beads ticket generation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinscribe.agents.comparator import ComparatorConfig, ComparatorInput
from twinscribe.agents.comparator_impl import (
    ConcreteComparatorAgent,
    create_comparator_agent,
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
    ComparisonResult,
    Discrepancy,
)
from twinscribe.models.components import ComponentDocumentation
from twinscribe.models.documentation import (
    CalleeRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
    StreamOutput,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def comparator_config() -> ComparatorConfig:
    """Create a test comparator configuration."""
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
        generate_beads_tickets=False,  # Disable for unit tests
    )


@pytest.fixture
def sample_documentation() -> ComponentDocumentation:
    """Create sample component documentation."""
    return ComponentDocumentation(
        summary="Process input data and return results.",
        description="A comprehensive method that processes the given input data.",
        parameters=[],
        returns=None,
        raises=[],
    )


@pytest.fixture
def sample_metadata_a() -> DocumenterMetadata:
    """Create sample metadata for Stream A."""
    return DocumenterMetadata(
        agent_id="A1",
        stream_id=StreamId.STREAM_A,
        model="claude-sonnet-4-5-20250929",
        confidence=0.92,
    )


@pytest.fixture
def sample_metadata_b() -> DocumenterMetadata:
    """Create sample metadata for Stream B."""
    return DocumenterMetadata(
        agent_id="B1",
        stream_id=StreamId.STREAM_B,
        model="gpt-4o",
        confidence=0.90,
    )


@pytest.fixture
def stream_a_output(
    sample_documentation: ComponentDocumentation,
    sample_metadata_a: DocumenterMetadata,
) -> StreamOutput:
    """Create sample Stream A output."""
    output = StreamOutput(stream_id=StreamId.STREAM_A)

    # Add a component with documentation
    doc_output = DocumentationOutput(
        component_id="module.func1",
        documentation=sample_documentation,
        call_graph=CallGraphSection(
            callers=[],
            callees=[
                CalleeRef(
                    component_id="module.helper", call_site_line=10, call_type=CallType.DIRECT
                )
            ],
        ),
        metadata=sample_metadata_a,
    )
    output.add_output(doc_output)

    # Add another component
    doc_output2 = DocumentationOutput(
        component_id="module.func2",
        documentation=sample_documentation,
        call_graph=CallGraphSection(callers=[], callees=[]),
        metadata=sample_metadata_a,
    )
    output.add_output(doc_output2)

    return output


@pytest.fixture
def stream_b_output(
    sample_documentation: ComponentDocumentation,
    sample_metadata_b: DocumenterMetadata,
) -> StreamOutput:
    """Create sample Stream B output."""
    output = StreamOutput(stream_id=StreamId.STREAM_B)

    # Add same component with slightly different call graph
    doc_output = DocumentationOutput(
        component_id="module.func1",
        documentation=sample_documentation,
        call_graph=CallGraphSection(
            callers=[],
            callees=[
                CalleeRef(
                    component_id="module.helper", call_site_line=10, call_type=CallType.DIRECT
                ),
                CalleeRef(
                    component_id="module.extra", call_site_line=15, call_type=CallType.CONDITIONAL
                ),
            ],
        ),
        metadata=sample_metadata_b,
    )
    output.add_output(doc_output)

    # Add matching second component
    doc_output2 = DocumentationOutput(
        component_id="module.func2",
        documentation=sample_documentation,
        call_graph=CallGraphSection(callers=[], callees=[]),
        metadata=sample_metadata_b,
    )
    output.add_output(doc_output2)

    return output


@pytest.fixture
def ground_truth_call_graph() -> CallGraph:
    """Create ground truth call graph from static analysis."""
    return CallGraph(
        source="pycg",
        edges=[
            CallEdge(
                caller="module.func1",
                callee="module.helper",
                call_site_line=10,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            # Note: module.extra is NOT in ground truth, so Stream B has a false positive
        ],
    )


@pytest.fixture
def comparator_input(
    stream_a_output: StreamOutput,
    stream_b_output: StreamOutput,
    ground_truth_call_graph: CallGraph,
) -> ComparatorInput:
    """Create comparator input with both streams and ground truth."""
    return ComparatorInput(
        stream_a_output=stream_a_output,
        stream_b_output=stream_b_output,
        ground_truth_call_graph=ground_truth_call_graph,
        iteration=1,
    )


@pytest.fixture
def mock_llm_response() -> dict:
    """Create mock LLM response for semantic comparison."""
    return {
        "discrepancies": [
            {
                "discrepancy_id": "disc_semantic_001",
                "component_id": "module.func1",
                "type": "documentation_content",
                "stream_a_value": "Process input data",
                "stream_b_value": "Process input data with validation",
                "resolution": "accept_stream_b",
                "confidence": 0.85,
                "requires_beads": False,
            }
        ],
        "identical_components": ["module.func2"],
    }


# =============================================================================
# Unit Tests
# =============================================================================


class TestConcreteComparatorAgentInit:
    """Tests for ComparatorAgent initialization."""

    def test_create_with_default_config(self) -> None:
        """Test agent creation with default configuration."""
        agent = ConcreteComparatorAgent()
        assert agent.agent_id == "C"
        assert agent.is_initialized is False

    def test_create_with_custom_config(self, comparator_config: ComparatorConfig) -> None:
        """Test agent creation with custom configuration."""
        agent = ConcreteComparatorAgent(config=comparator_config)
        assert agent.comparator_config.confidence_threshold == 0.7
        assert agent.comparator_config.generate_beads_tickets is False

    def test_factory_function(self) -> None:
        """Test factory function creates agent correctly."""
        agent = create_comparator_agent()
        assert isinstance(agent, ConcreteComparatorAgent)


class TestComparatorAgentLifecycle:
    """Tests for agent initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_creates_llm_client(self, comparator_config: ComparatorConfig) -> None:
        """Test that initialization sets up LLM client."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()

            assert agent.is_initialized is True
            mock_client.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_clients(self, comparator_config: ComparatorConfig) -> None:
        """Test that shutdown properly closes clients."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            await agent.shutdown()

            assert agent.is_initialized is False
            mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_raises_if_not_initialized(
        self,
        comparator_config: ComparatorConfig,
        comparator_input: ComparatorInput,
    ) -> None:
        """Test that process raises error if not initialized."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await agent.process(comparator_input)


class TestDocumentComparison:
    """Tests for document comparison logic."""

    def test_docs_identical_when_same(self, comparator_config: ComparatorConfig) -> None:
        """Test identical docs detection."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        doc = {
            "documentation": {"summary": "Test"},
            "call_graph": {"callees": [], "callers": []},
        }

        assert agent._are_docs_identical(doc, doc) is True

    def test_docs_not_identical_when_different(self, comparator_config: ComparatorConfig) -> None:
        """Test different docs detection."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        doc_a = {
            "documentation": {"summary": "Test A"},
            "call_graph": {"callees": [], "callers": []},
        }
        doc_b = {
            "documentation": {"summary": "Test B"},
            "call_graph": {"callees": [], "callers": []},
        }

        assert agent._are_docs_identical(doc_a, doc_b) is False

    def test_docs_identical_handles_none(self, comparator_config: ComparatorConfig) -> None:
        """Test identical docs with None values."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert agent._are_docs_identical(None, None) is True
        assert agent._are_docs_identical(None, {}) is False
        assert agent._are_docs_identical({}, None) is False


class TestCallGraphComparison:
    """Tests for call graph comparison and resolution."""

    def test_compare_call_graphs_detects_extra_edge(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ) -> None:
        """Test detection of extra edge in Stream B."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        stream_a_doc = {
            "call_graph": {
                "callees": [{"component_id": "module.helper"}],
                "callers": [],
            }
        }
        stream_b_doc = {
            "call_graph": {
                "callees": [
                    {"component_id": "module.helper"},
                    {"component_id": "module.extra"},
                ],
                "callers": [],
            }
        }

        gt_callees = ground_truth_call_graph.get_callees("module.func1")
        gt_callers = ground_truth_call_graph.get_callers("module.func1")

        discrepancies = agent._compare_call_graphs(
            component_id="module.func1",
            stream_a_doc=stream_a_doc,
            stream_b_doc=stream_b_doc,
            gt_callees=gt_callees,
            gt_callers=gt_callers,
        )

        # Should detect that module.extra is a discrepancy
        assert len(discrepancies) == 1
        disc = discrepancies[0]
        assert disc.type == DiscrepancyType.CALL_GRAPH_EDGE
        # module.extra is NOT in ground truth, so ground truth wins
        assert disc.resolution == ResolutionAction.ACCEPT_GROUND_TRUTH
        assert disc.confidence == 0.99

    def test_compare_call_graphs_no_discrepancy_when_matching(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ) -> None:
        """Test no discrepancy when call graphs match."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        # Both streams have same call graph
        stream_doc = {
            "call_graph": {
                "callees": [{"component_id": "module.helper"}],
                "callers": [],
            }
        }

        gt_callees = ground_truth_call_graph.get_callees("module.func1")
        gt_callers = ground_truth_call_graph.get_callers("module.func1")

        discrepancies = agent._compare_call_graphs(
            component_id="module.func1",
            stream_a_doc=stream_doc,
            stream_b_doc=stream_doc,
            gt_callees=gt_callees,
            gt_callers=gt_callers,
        )

        assert len(discrepancies) == 0


class TestResolutionLogic:
    """Tests for discrepancy resolution logic."""

    def test_determine_resolution_accepts_stream_a_matching_gt(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test resolution accepts Stream A when it matches ground truth."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value=True,
            stream_b_value=False,
            ground_truth=True,
        )

        assert resolution == "accept_stream_a"
        assert confidence == 0.99

    def test_determine_resolution_accepts_stream_b_matching_gt(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test resolution accepts Stream B when it matches ground truth."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value=False,
            stream_b_value=True,
            ground_truth=True,
        )

        assert resolution == "accept_stream_b"
        assert confidence == 0.99

    def test_determine_resolution_accepts_ground_truth_when_neither_match(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test resolution accepts ground truth when neither stream matches."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="call_graph_edge",
            stream_a_value="value_a",
            stream_b_value="value_b",
            ground_truth="ground_truth_value",
        )

        assert resolution == "accept_ground_truth"
        assert confidence == 0.99

    def test_determine_resolution_needs_review_for_doc_content(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test documentation content requires human review."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        resolution, confidence = agent._determine_resolution(
            discrepancy_type="documentation_content",
            stream_a_value="Summary A",
            stream_b_value="Summary B",
            ground_truth=None,
        )

        assert resolution == "needs_human_review"
        assert confidence == 0.5


class TestDiscrepancyTypeMapping:
    """Tests for discrepancy type enum mapping."""

    def test_map_discrepancy_type_call_graph(self, comparator_config: ComparatorConfig) -> None:
        """Test mapping call graph type."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert agent._map_discrepancy_type("call_graph_edge") == DiscrepancyType.CALL_GRAPH_EDGE
        assert agent._map_discrepancy_type("CALL_GRAPH_EDGE") == DiscrepancyType.CALL_GRAPH_EDGE

    def test_map_discrepancy_type_documentation(self, comparator_config: ComparatorConfig) -> None:
        """Test mapping documentation types."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert (
            agent._map_discrepancy_type("documentation_content")
            == DiscrepancyType.DOCUMENTATION_CONTENT
        )
        assert (
            agent._map_discrepancy_type("parameter_description")
            == DiscrepancyType.PARAMETER_DESCRIPTION
        )

    def test_map_discrepancy_type_unknown_defaults(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test unknown type defaults to documentation_content."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert agent._map_discrepancy_type("unknown_type") == DiscrepancyType.DOCUMENTATION_CONTENT


class TestResolutionActionMapping:
    """Tests for resolution action enum mapping."""

    def test_map_resolution_action_all_types(self, comparator_config: ComparatorConfig) -> None:
        """Test mapping all resolution action types."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert agent._map_resolution_action("accept_stream_a") == ResolutionAction.ACCEPT_STREAM_A
        assert agent._map_resolution_action("accept_stream_b") == ResolutionAction.ACCEPT_STREAM_B
        assert (
            agent._map_resolution_action("accept_ground_truth")
            == ResolutionAction.ACCEPT_GROUND_TRUTH
        )
        assert agent._map_resolution_action("merge_both") == ResolutionAction.MERGE_BOTH
        assert (
            agent._map_resolution_action("needs_human_review")
            == ResolutionAction.NEEDS_HUMAN_REVIEW
        )
        assert agent._map_resolution_action("deferred") == ResolutionAction.DEFERRED

    def test_map_resolution_action_unknown_defaults(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test unknown resolution defaults to needs_human_review."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        assert agent._map_resolution_action("unknown") == ResolutionAction.NEEDS_HUMAN_REVIEW


class TestPriorityDetermination:
    """Tests for ticket priority determination."""

    def test_high_priority_for_call_graph(self, comparator_config: ComparatorConfig) -> None:
        """Test call graph issues get high priority."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        discrepancy = Discrepancy(
            discrepancy_id="test",
            component_id="test.component",
            type=DiscrepancyType.CALL_GRAPH_EDGE,
            confidence=0.8,
        )

        priority = agent._get_priority_from_discrepancy(discrepancy)
        assert priority == "High"

    def test_high_priority_for_low_confidence(self, comparator_config: ComparatorConfig) -> None:
        """Test low confidence issues get high priority."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        discrepancy = Discrepancy(
            discrepancy_id="test",
            component_id="test.component",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            confidence=0.4,
        )

        priority = agent._get_priority_from_discrepancy(discrepancy)
        assert priority == "High"

    def test_medium_priority_for_normal_doc_discrepancy(
        self, comparator_config: ComparatorConfig
    ) -> None:
        """Test normal doc issues get medium priority."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        discrepancy = Discrepancy(
            discrepancy_id="test",
            component_id="test.component",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            confidence=0.6,
        )

        priority = agent._get_priority_from_discrepancy(discrepancy)
        assert priority == "Medium"


class TestCompareComponent:
    """Tests for single component comparison."""

    @pytest.mark.asyncio
    async def test_compare_component_missing_in_a(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ) -> None:
        """Test handling when component missing in Stream A."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        discrepancies = await agent.compare_component(
            component_id="test.component",
            stream_a_doc=None,
            stream_b_doc={"documentation": {}},
            ground_truth=ground_truth_call_graph,
        )

        assert len(discrepancies) == 1
        assert discrepancies[0].resolution == ResolutionAction.ACCEPT_STREAM_B
        assert discrepancies[0].stream_a_value is None

    @pytest.mark.asyncio
    async def test_compare_component_missing_in_b(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ) -> None:
        """Test handling when component missing in Stream B."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        discrepancies = await agent.compare_component(
            component_id="test.component",
            stream_a_doc={"documentation": {}},
            stream_b_doc=None,
            ground_truth=ground_truth_call_graph,
        )

        assert len(discrepancies) == 1
        assert discrepancies[0].resolution == ResolutionAction.ACCEPT_STREAM_A
        assert discrepancies[0].stream_b_value is None


class TestProcessIntegration:
    """Integration tests for the full process flow."""

    @pytest.mark.asyncio
    async def test_process_identifies_call_graph_discrepancy(
        self,
        comparator_config: ComparatorConfig,
        comparator_input: ComparatorInput,
        mock_llm_response: dict,
    ) -> None:
        """Test full process identifies call graph discrepancies."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(mock_llm_response)
            mock_response.usage = MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd=0.01,
            )
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        assert isinstance(result, ComparisonResult)
        assert result.iteration == 1
        # module.func2 is identical in both streams
        assert result.summary.total_components == 2
        # Should have discrepancy for module.extra
        assert result.summary.discrepancies >= 1

    @pytest.mark.asyncio
    async def test_process_sets_convergence_status(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
        sample_documentation: ComponentDocumentation,
        sample_metadata_a: DocumenterMetadata,
        sample_metadata_b: DocumenterMetadata,
    ) -> None:
        """Test process correctly sets convergence status."""
        # Create identical outputs
        output_a = StreamOutput(stream_id=StreamId.STREAM_A)
        output_b = StreamOutput(stream_id=StreamId.STREAM_B)

        doc_a = DocumentationOutput(
            component_id="module.func",
            documentation=sample_documentation,
            call_graph=CallGraphSection(callers=[], callees=[]),
            metadata=sample_metadata_a,
        )
        doc_b = DocumentationOutput(
            component_id="module.func",
            documentation=sample_documentation,
            call_graph=CallGraphSection(callers=[], callees=[]),
            metadata=sample_metadata_b,
        )

        output_a.add_output(doc_a)
        output_b.add_output(doc_b)

        input_data = ComparatorInput(
            stream_a_output=output_a,
            stream_b_output=output_b,
            ground_truth_call_graph=CallGraph(source="test", edges=[]),
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(input_data)
            await agent.shutdown()

        # All identical, should converge
        assert result.convergence_status.converged is True
        assert result.convergence_status.blocking_discrepancies == 0
        assert result.convergence_status.recommendation == "finalize"


class TestInputValidation:
    """Tests for input validation."""

    def test_validate_input_passes_with_valid_data(
        self,
        comparator_config: ComparatorConfig,
        comparator_input: ComparatorInput,
    ) -> None:
        """Test validation passes with valid input."""
        agent = ConcreteComparatorAgent(config=comparator_config)

        # Should not raise
        agent._validate_input(comparator_input)

    def test_pydantic_enforces_required_fields(self) -> None:
        """Test that Pydantic validation enforces required fields."""
        # Attempting to create ComparatorInput without required fields should fail
        with pytest.raises(Exception):  # Pydantic ValidationError
            ComparatorInput()  # type: ignore
