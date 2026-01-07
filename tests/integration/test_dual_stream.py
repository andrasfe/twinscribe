"""
Integration tests for the dual-stream documentation workflow.

Tests cover the complete workflow including:
- Stream A and Stream B processing components
- Comparator comparing outputs and detecting discrepancies
- Ground truth (static oracle) resolution
- Multi-iteration convergence

All tests use mocked LLM responses to avoid real API calls.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinscribe.agents.comparator import ComparatorConfig, ComparatorInput
from twinscribe.agents.comparator_impl import ConcreteComparatorAgent
from twinscribe.agents.documenter import DocumenterConfig
from twinscribe.agents.stream import (
    StreamConfig,
)
from twinscribe.agents.validator import ValidatorConfig
from twinscribe.models.base import (
    CallType,
    ComponentType,
    DiscrepancyType,
    ModelTier,
    ResolutionAction,
    StreamId,
)
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.models.comparison import (
    ComparisonResult,
)
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ComponentLocation,
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
# Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_codebase_dir(fixtures_dir: Path) -> Path:
    """Return the sample codebase directory for testing."""
    return fixtures_dir / "sample_codebase"


@pytest.fixture
def stream_a_config() -> StreamConfig:
    """Configuration for Stream A."""
    return StreamConfig(
        stream_id=StreamId.STREAM_A,
        documenter_config=DocumenterConfig(
            agent_id="A1",
            stream_id=StreamId.STREAM_A,
            model_tier=ModelTier.GENERATION,
            provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
            max_tokens=4096,
            temperature=0.0,
        ),
        validator_config=ValidatorConfig(
            agent_id="A2",
            stream_id=StreamId.STREAM_A,
            model_tier=ModelTier.VALIDATION,
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            cost_per_million_input=0.25,
            cost_per_million_output=1.25,
            max_tokens=2048,
            temperature=0.0,
        ),
        batch_size=5,
        max_retries=1,
        continue_on_error=True,
    )


@pytest.fixture
def stream_b_config() -> StreamConfig:
    """Configuration for Stream B."""
    return StreamConfig(
        stream_id=StreamId.STREAM_B,
        documenter_config=DocumenterConfig(
            agent_id="B1",
            stream_id=StreamId.STREAM_B,
            model_tier=ModelTier.GENERATION,
            provider="openai",
            model_name="gpt-4o",
            cost_per_million_input=2.5,
            cost_per_million_output=10.0,
            max_tokens=4096,
            temperature=0.0,
        ),
        validator_config=ValidatorConfig(
            agent_id="B2",
            stream_id=StreamId.STREAM_B,
            model_tier=ModelTier.VALIDATION,
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_million_input=0.15,
            cost_per_million_output=0.60,
            max_tokens=2048,
            temperature=0.0,
        ),
        batch_size=5,
        max_retries=1,
        continue_on_error=True,
    )


@pytest.fixture
def comparator_config() -> ComparatorConfig:
    """Configuration for the comparator agent."""
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
        generate_beads_tickets=False,  # Disable for tests
    )


@pytest.fixture
def sample_components() -> list[Component]:
    """Create sample components for testing."""
    return [
        Component(
            component_id="sample_module.helper_function",
            name="helper_function",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="src/sample_module.py",
                line_start=1,
                line_end=5,
            ),
        ),
        Component(
            component_id="sample_module.Calculator.__init__",
            name="__init__",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="src/sample_module.py",
                line_start=10,
                line_end=14,
            ),
            parent_id="sample_module.Calculator",
        ),
        Component(
            component_id="sample_module.Calculator.add",
            name="add",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="src/sample_module.py",
                line_start=15,
                line_end=25,
            ),
            parent_id="sample_module.Calculator",
        ),
    ]


@pytest.fixture
def sample_source_code_map() -> dict[str, str]:
    """Create a mapping of component_id to source code."""
    return {
        "sample_module.helper_function": '''def helper_function(value):
    """Convert value to string."""
    return str(value)
''',
        "sample_module.Calculator.__init__": '''def __init__(self, precision: int = 2):
    """Initialize calculator."""
    self.precision = precision
''',
        "sample_module.Calculator.add": '''def add(self, a: float, b: float) -> float:
    """Add two numbers."""
    result = a + b
    return round(result, self.precision)
''',
    }


@pytest.fixture
def ground_truth_call_graph() -> CallGraph:
    """Create ground truth call graph from static analysis."""
    return CallGraph(
        source="pycg",
        edges=[
            CallEdge(
                caller="sample_module.Calculator.add",
                callee="builtins.round",
                call_site_line=22,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            CallEdge(
                caller="sample_module.Calculator.multiply",
                callee="sample_module.helper_function",
                call_site_line=35,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
        ],
    )


def create_mock_documentation_output(
    component_id: str,
    stream_id: StreamId,
    callees: list[dict] | None = None,
    callers: list[dict] | None = None,
    summary: str = "Default summary",
    description: str = "Default description",
) -> DocumentationOutput:
    """Helper to create mock DocumentationOutput."""
    callee_refs = []
    if callees:
        for c in callees:
            callee_refs.append(
                CalleeRef(
                    component_id=c.get("component_id", ""),
                    call_site_line=c.get("line"),
                    call_type=CallType(c.get("call_type", "direct")),
                )
            )

    caller_refs = []
    if callers:
        for c in callers:
            caller_refs.append(
                CallerRef(
                    component_id=c.get("component_id", ""),
                    call_site_line=c.get("line"),
                    call_type=CallType(c.get("call_type", "direct")),
                )
            )

    return DocumentationOutput(
        component_id=component_id,
        documentation=ComponentDocumentation(
            summary=summary,
            description=description,
            parameters=[],
            returns=None,
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=caller_refs,
            callees=callee_refs,
        ),
        metadata=DocumenterMetadata(
            agent_id="A1" if stream_id == StreamId.STREAM_A else "B1",
            stream_id=stream_id,
            model="claude-sonnet" if stream_id == StreamId.STREAM_A else "gpt-4o",
            confidence=0.92,
        ),
    )


def create_stream_output(
    stream_id: StreamId,
    docs: list[DocumentationOutput],
) -> StreamOutput:
    """Helper to create StreamOutput from documentation list."""
    output = StreamOutput(stream_id=stream_id)
    for doc in docs:
        output.add_output(doc)
    return output


# =============================================================================
# Test 1: Happy Path - Full Convergence
# =============================================================================


@pytest.mark.integration
class TestHappyPathFullConvergence:
    """Test the happy path where both streams produce identical output."""

    @pytest.mark.asyncio
    async def test_full_convergence_iteration_1(
        self,
        comparator_config: ComparatorConfig,
        sample_components: list[Component],
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test that when both streams produce identical output,
        convergence is achieved in iteration 1.
        """
        # Create identical documentation for both streams
        stream_a_docs = []
        stream_b_docs = []

        for component in sample_components:
            # Both streams produce identical documentation
            doc_a = create_mock_documentation_output(
                component_id=component.component_id,
                stream_id=StreamId.STREAM_A,
                summary=f"Documentation for {component.name}",
                description=f"Detailed description of {component.name}",
                callees=[{"component_id": "builtins.round", "line": 22, "call_type": "direct"}]
                if component.component_id == "sample_module.Calculator.add"
                else None,
            )
            doc_b = create_mock_documentation_output(
                component_id=component.component_id,
                stream_id=StreamId.STREAM_B,
                summary=f"Documentation for {component.name}",
                description=f"Detailed description of {component.name}",
                callees=[{"component_id": "builtins.round", "line": 22, "call_type": "direct"}]
                if component.component_id == "sample_module.Calculator.add"
                else None,
            )
            stream_a_docs.append(doc_a)
            stream_b_docs.append(doc_b)

        stream_a_output = create_stream_output(StreamId.STREAM_A, stream_a_docs)
        stream_b_output = create_stream_output(StreamId.STREAM_B, stream_b_docs)

        # Create comparator input
        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        # Initialize and run comparator with mocked LLM
        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Verify convergence
        assert isinstance(result, ComparisonResult)
        assert result.convergence_status.converged is True
        assert result.convergence_status.blocking_discrepancies == 0
        assert result.convergence_status.recommendation == "finalize"
        assert result.summary.total_components == len(sample_components)
        assert result.summary.identical == len(sample_components)
        assert result.summary.discrepancies == 0

    @pytest.mark.asyncio
    async def test_documentation_package_complete(
        self,
        comparator_config: ComparatorConfig,
        sample_components: list[Component],
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test that after convergence, all components have documentation.
        """
        # Create documentation for all components
        stream_a_docs = []
        for component in sample_components:
            doc = create_mock_documentation_output(
                component_id=component.component_id,
                stream_id=StreamId.STREAM_A,
                summary=f"Complete documentation for {component.name}",
                description=f"Full description with all details for {component.name}",
            )
            stream_a_docs.append(doc)

        stream_a_output = create_stream_output(StreamId.STREAM_A, stream_a_docs)
        stream_b_output = create_stream_output(StreamId.STREAM_B, stream_a_docs)  # Identical

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Verify all components have documentation
        assert result.is_converged
        assert result.summary.total_components == len(sample_components)

        # Verify each component has documentation in the output
        for component in sample_components:
            assert component.component_id in stream_a_output.outputs


# =============================================================================
# Test 2: Ground Truth Resolution
# =============================================================================


@pytest.mark.integration
class TestGroundTruthResolution:
    """Test that call graph discrepancies are resolved using static oracle."""

    @pytest.mark.asyncio
    async def test_call_graph_discrepancy_resolved_by_ground_truth(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that when streams disagree on call graph edges,
        the static oracle (ground truth) is consulted for resolution.
        """
        component_id = "module.process"

        # Ground truth says module.process calls helper.func
        ground_truth = CallGraph(
            source="pycg",
            edges=[
                CallEdge(
                    caller=component_id,
                    callee="helper.func",
                    call_site_line=10,
                    call_type=CallType.DIRECT,
                    confidence=1.0,
                ),
            ],
        )

        # Stream A correctly identifies the callee
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            callees=[{"component_id": "helper.func", "line": 10, "call_type": "direct"}],
        )

        # Stream B incorrectly includes an extra callee
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            callees=[
                {"component_id": "helper.func", "line": 10, "call_type": "direct"},
                {"component_id": "nonexistent.func", "line": 20, "call_type": "direct"},
            ],
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({"discrepancies": []})
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

        # Verify discrepancy was detected and resolved using ground truth
        assert result.summary.discrepancies >= 1

        # Find the call graph discrepancy
        call_graph_discs = [
            d for d in result.discrepancies if d.type == DiscrepancyType.CALL_GRAPH_EDGE
        ]

        # Should have detected the nonexistent.func discrepancy
        assert len(call_graph_discs) >= 1

        # The resolution should favor ground truth (which doesn't have nonexistent.func)
        for disc in call_graph_discs:
            if "nonexistent.func" in str(disc.stream_b_value):
                assert disc.resolution == ResolutionAction.ACCEPT_GROUND_TRUTH
                assert disc.confidence == 0.99  # High confidence from ground truth

    @pytest.mark.asyncio
    async def test_stream_a_accepted_when_matches_ground_truth(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that Stream A is accepted when it matches ground truth
        and Stream B does not.
        """
        component_id = "module.handler"

        # Ground truth
        ground_truth = CallGraph(
            source="pycg",
            edges=[
                CallEdge(
                    caller=component_id,
                    callee="utils.validate",
                    call_site_line=15,
                    call_type=CallType.DIRECT,
                    confidence=1.0,
                ),
            ],
        )

        # Stream A matches ground truth
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            callees=[{"component_id": "utils.validate", "line": 15, "call_type": "direct"}],
        )

        # Stream B is missing the callee
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            callees=[],  # Missing!
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({"discrepancies": []})
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Find the discrepancy
        call_graph_discs = [
            d for d in result.discrepancies if d.type == DiscrepancyType.CALL_GRAPH_EDGE
        ]

        # Should detect the missing callee in Stream B
        assert len(call_graph_discs) >= 1

        # Resolution should accept Stream A (matches ground truth)
        for disc in call_graph_discs:
            assert disc.resolution == ResolutionAction.ACCEPT_STREAM_A
            assert disc.confidence == 0.99


# =============================================================================
# Test 3: Discrepancy Detection
# =============================================================================


@pytest.mark.integration
class TestDiscrepancyDetection:
    """Test detection and categorization of various discrepancies."""

    @pytest.mark.asyncio
    async def test_documentation_content_discrepancy_detected(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test that documentation content discrepancies are detected
        and properly categorized.
        """
        component_id = "module.process"

        # Stream A has one summary
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            summary="Process data synchronously",
            description="Synchronous data processing method",
        )

        # Stream B has different summary
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            summary="Process data asynchronously with validation",
            description="Asynchronous data processing with built-in validation",
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        # Mock LLM to return semantic discrepancy
        mock_llm_response = {
            "discrepancies": [
                {
                    "discrepancy_id": f"disc_{component_id}_content",
                    "component_id": component_id,
                    "type": "documentation_content",
                    "stream_a_value": "Process data synchronously",
                    "stream_b_value": "Process data asynchronously with validation",
                    "resolution": "needs_human_review",
                    "confidence": 0.5,
                    "requires_beads": True,
                }
            ]
        }

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(mock_llm_response)
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should not converge due to content discrepancy
        assert result.convergence_status.converged is False
        assert result.summary.discrepancies >= 1

        # Find content discrepancy
        content_discs = [
            d for d in result.discrepancies if d.type == DiscrepancyType.DOCUMENTATION_CONTENT
        ]
        assert len(content_discs) >= 1

    @pytest.mark.asyncio
    async def test_multiple_discrepancy_types_categorized(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that multiple types of discrepancies are detected
        and correctly categorized.
        """
        component_id = "module.handler"

        # Ground truth
        ground_truth = CallGraph(
            source="pycg",
            edges=[
                CallEdge(
                    caller=component_id,
                    callee="utils.validate",
                    call_site_line=10,
                    call_type=CallType.DIRECT,
                    confidence=1.0,
                ),
            ],
        )

        # Stream A: correct call graph, one summary
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            summary="Handler method",
            callees=[{"component_id": "utils.validate", "line": 10, "call_type": "direct"}],
        )

        # Stream B: wrong call graph, different summary
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            summary="Request handler method",
            callees=[{"component_id": "utils.process", "line": 15, "call_type": "direct"}],
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(
                {
                    "discrepancies": [
                        {
                            "discrepancy_id": "disc_content_1",
                            "component_id": component_id,
                            "type": "documentation_content",
                            "resolution": "needs_human_review",
                            "confidence": 0.5,
                        }
                    ]
                }
            )
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should have multiple discrepancies
        assert result.summary.discrepancies >= 1

        # Check for call graph discrepancy
        call_graph_discs = [
            d for d in result.discrepancies if d.type == DiscrepancyType.CALL_GRAPH_EDGE
        ]

        # Should have detected missing and extra callees
        assert len(call_graph_discs) >= 1

    @pytest.mark.asyncio
    async def test_discrepancy_requires_beads_when_low_confidence(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test that discrepancies with low confidence are flagged
        for Beads ticket generation.
        """
        component_id = "module.ambiguous"

        # Both streams have same call graph but different descriptions
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            summary="Ambiguous function A interpretation",
        )
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            summary="Ambiguous function B interpretation",
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        # Mock LLM to return low confidence discrepancy
        mock_llm_response = {
            "discrepancies": [
                {
                    "discrepancy_id": f"disc_{component_id}_ambiguous",
                    "component_id": component_id,
                    "type": "documentation_content",
                    "stream_a_value": "Ambiguous function A interpretation",
                    "stream_b_value": "Ambiguous function B interpretation",
                    "resolution": "needs_human_review",
                    "confidence": 0.3,  # Low confidence
                    "requires_beads": True,
                }
            ]
        }

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(mock_llm_response)
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should require human review
        assert result.summary.requires_human_review >= 1

        # Find the discrepancy
        beads_discs = [d for d in result.discrepancies if d.requires_beads]
        assert len(beads_discs) >= 1


# =============================================================================
# Test 4: Multi-iteration Convergence
# =============================================================================


@pytest.mark.integration
class TestMultiIterationConvergence:
    """Test convergence over multiple iterations with corrections."""

    @pytest.mark.asyncio
    async def test_convergence_after_correction_feedback(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that streams converge after correction feedback is applied.
        """
        component_id = "module.process"

        ground_truth = CallGraph(
            source="pycg",
            edges=[
                CallEdge(
                    caller=component_id,
                    callee="helper.func",
                    call_site_line=10,
                    call_type=CallType.DIRECT,
                    confidence=1.0,
                ),
            ],
        )

        # === Iteration 1: Initial disagreement ===
        # Stream A has correct callee
        stream_a_doc_iter1 = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            callees=[{"component_id": "helper.func", "line": 10, "call_type": "direct"}],
        )

        # Stream B has wrong callee
        stream_b_doc_iter1 = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            callees=[{"component_id": "wrong.func", "line": 10, "call_type": "direct"}],
        )

        stream_a_output_iter1 = create_stream_output(StreamId.STREAM_A, [stream_a_doc_iter1])
        stream_b_output_iter1 = create_stream_output(StreamId.STREAM_B, [stream_b_doc_iter1])

        comparator_input_iter1 = ComparatorInput(
            stream_a_output=stream_a_output_iter1,
            stream_b_output=stream_b_output_iter1,
            ground_truth_call_graph=ground_truth,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({"discrepancies": []})
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result_iter1 = await agent.process(comparator_input_iter1)

            # Iteration 1 should not converge
            assert result_iter1.convergence_status.converged is False
            assert result_iter1.summary.discrepancies >= 1

            # Collect discrepancy IDs that were resolved
            resolved_disc_ids = [
                d.discrepancy_id for d in result_iter1.discrepancies if d.is_resolved
            ]

            # === Iteration 2: After correction feedback ===
            # Stream B is corrected based on ground truth
            stream_b_doc_iter2 = create_mock_documentation_output(
                component_id=component_id,
                stream_id=StreamId.STREAM_B,
                callees=[{"component_id": "helper.func", "line": 10, "call_type": "direct"}],
            )

            stream_a_output_iter2 = create_stream_output(StreamId.STREAM_A, [stream_a_doc_iter1])
            stream_b_output_iter2 = create_stream_output(StreamId.STREAM_B, [stream_b_doc_iter2])

            comparator_input_iter2 = ComparatorInput(
                stream_a_output=stream_a_output_iter2,
                stream_b_output=stream_b_output_iter2,
                ground_truth_call_graph=ground_truth,
                iteration=2,
                previous_comparison=result_iter1,
                resolved_discrepancies=resolved_disc_ids,
            )

            result_iter2 = await agent.process(comparator_input_iter2)
            await agent.shutdown()

        # Iteration 2 should converge
        assert result_iter2.convergence_status.converged is True
        assert result_iter2.summary.identical == 1
        assert result_iter2.summary.discrepancies == 0

    @pytest.mark.asyncio
    async def test_iteration_tracking(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test that iteration numbers are properly tracked across comparisons.
        """
        component_id = "module.tracked"

        # Identical docs for quick convergence
        doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [doc])

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()

            # Run multiple iterations
            for iteration in [1, 2, 3]:
                comparator_input = ComparatorInput(
                    stream_a_output=stream_a_output,
                    stream_b_output=stream_b_output,
                    ground_truth_call_graph=ground_truth_call_graph,
                    iteration=iteration,
                )

                result = await agent.process(comparator_input)

                # Verify iteration is tracked
                assert result.iteration == iteration

            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_discrepancy_iteration_found_tracking(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that discrepancies track which iteration they were found in.
        """
        component_id = "module.tracked_disc"

        ground_truth = CallGraph(source="pycg", edges=[])

        # Create discrepancy in iteration 2
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            summary="Summary A",
        )
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            summary="Summary B - Different",
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth,
            iteration=2,  # Found in iteration 2
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        mock_llm_response = {
            "discrepancies": [
                {
                    "discrepancy_id": f"disc_{component_id}_summary",
                    "component_id": component_id,
                    "type": "documentation_content",
                    "resolution": "needs_human_review",
                    "confidence": 0.5,
                }
            ]
        }

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(mock_llm_response)
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Check discrepancies track iteration found
        for disc in result.discrepancies:
            assert disc.iteration_found == 2


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases in the dual-stream workflow."""

    @pytest.mark.asyncio
    async def test_component_missing_in_one_stream(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test handling when a component is documented by one stream but not the other.
        """
        component_id = "module.missing_component"

        # Only Stream A has documentation
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = StreamOutput(stream_id=StreamId.STREAM_B)  # Empty

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should detect discrepancy for missing component
        assert result.summary.discrepancies >= 1

        # Find the discrepancy
        missing_discs = [d for d in result.discrepancies if d.component_id == component_id]
        assert len(missing_discs) >= 1

        # Resolution should accept Stream A (since B is missing)
        for disc in missing_discs:
            assert disc.resolution == ResolutionAction.ACCEPT_STREAM_A

    @pytest.mark.asyncio
    async def test_empty_streams(
        self,
        comparator_config: ComparatorConfig,
        ground_truth_call_graph: CallGraph,
    ):
        """
        Test handling of empty stream outputs.
        """
        stream_a_output = StreamOutput(stream_id=StreamId.STREAM_A)
        stream_b_output = StreamOutput(stream_id=StreamId.STREAM_B)

        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=1,
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should converge (both empty = identical)
        assert result.convergence_status.converged is True
        assert result.summary.total_components == 0
        assert result.summary.identical == 0
        assert result.summary.discrepancies == 0

    @pytest.mark.asyncio
    async def test_max_iterations_recommendation(
        self,
        comparator_config: ComparatorConfig,
    ):
        """
        Test that max iterations recommendation is given when appropriate.
        """
        component_id = "module.stubborn"

        ground_truth = CallGraph(source="pycg", edges=[])

        # Persistent disagreement
        stream_a_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_A,
            summary="A interpretation",
        )
        stream_b_doc = create_mock_documentation_output(
            component_id=component_id,
            stream_id=StreamId.STREAM_B,
            summary="B interpretation",
        )

        stream_a_output = create_stream_output(StreamId.STREAM_A, [stream_a_doc])
        stream_b_output = create_stream_output(StreamId.STREAM_B, [stream_b_doc])

        # Simulate reaching max iterations (5)
        comparator_input = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth,
            iteration=5,  # Max iterations
        )

        agent = ConcreteComparatorAgent(config=comparator_config)

        mock_llm_response = {
            "discrepancies": [
                {
                    "discrepancy_id": "disc_stubborn",
                    "component_id": component_id,
                    "type": "documentation_content",
                    "resolution": "needs_human_review",
                    "confidence": 0.5,
                    "requires_beads": True,
                }
            ]
        }

        with patch("twinscribe.agents.comparator_impl.get_comparator_client") as mock_get:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(mock_llm_response)
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
            mock_response.latency_ms = 500
            mock_client.send_message = AsyncMock(return_value=mock_response)
            mock_get.return_value = (mock_client, "test-model")

            await agent.initialize()
            result = await agent.process(comparator_input)
            await agent.shutdown()

        # Should recommend max iterations reached or generate tickets
        assert result.convergence_status.recommendation in [
            "max_iterations_reached",
            "generate_beads_tickets",
        ]
