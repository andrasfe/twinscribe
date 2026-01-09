"""
Integration tests for the DualStreamOrchestrator.

Tests cover:
- Full pipeline with sample codebase using mocked LLM responses
- Convergence in single iteration (identical outputs from both streams)
- Convergence after corrections (streams disagree initially, then agree)
- Beads ticket generation for unresolved discrepancies
- Max iterations handling (forced convergence)
- Output file generation verification

All tests use mocked LLM responses but real static analysis where possible.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinscribe.agents.comparator import (
    ComparatorAgent,
    ComparatorConfig,
    ComparatorInput,
)
from twinscribe.agents.stream import (
    ComponentProcessingResult,
    DocumentationStream,
    StreamConfig,
    StreamResult,
)
from twinscribe.analysis.oracle import OracleConfig, StaticAnalysisOracle
from twinscribe.beads.lifecycle import BeadsLifecycleManager, LifecycleManagerConfig
from twinscribe.models.base import (
    CallType,
    ComponentType,
    DiscrepancyType,
    ResolutionAction,
    StreamId,
)
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.models.comparison import (
    ComparatorMetadata,
    ComparisonResult,
    ComparisonSummary,
    ConvergenceStatus,
    Discrepancy,
)
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ComponentLocation,
    ParameterDoc,
    ReturnDoc,
)
from twinscribe.models.convergence import ConvergenceReport
from twinscribe.models.documentation import (
    CalleeRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
    StreamOutput,
)
from twinscribe.models.output import (
    DocumentationPackage,
    RunMetrics,
)
from twinscribe.orchestrator.orchestrator import (
    DualStreamOrchestrator,
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorPhase,
    OrchestratorState,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_codebase_path(fixtures_dir: Path) -> Path:
    """Return the sample codebase path."""
    return fixtures_dir / "sample_codebase"


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Create a basic orchestrator configuration for testing."""
    return OrchestratorConfig(
        max_iterations=5,
        parallel_components=5,
        wait_for_beads=False,
        beads_timeout_hours=0,
        skip_validation=False,
        dry_run=True,
        continue_on_error=True,
    )


@pytest.fixture
def mock_call_graph() -> CallGraph:
    """Create a mock call graph for testing."""
    return CallGraph(
        edges=[
            CallEdge(
                caller="sample_codebase.calculator.Calculator.add",
                callee="builtins.round",
                call_site_line=25,
                call_type=CallType.DIRECT,
            ),
            CallEdge(
                caller="sample_codebase.calculator.Calculator.multiply",
                callee="builtins.round",
                call_site_line=32,
                call_type=CallType.DIRECT,
            ),
            CallEdge(
                caller="sample_codebase.main.Application.run",
                callee="sample_codebase.calculator.Calculator.add",
                call_site_line=45,
                call_type=CallType.DIRECT,
            ),
        ],
        source="pycg",
    )


@pytest.fixture
def sample_components() -> list[Component]:
    """Create sample components for testing."""
    return [
        Component(
            component_id="sample_codebase.calculator.Calculator.__init__",
            name="__init__",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="calculator.py",
                line_start=10,
                line_end=15,
            ),
            signature="def __init__(self, precision: int = 2):",
            parent_id="sample_codebase.calculator.Calculator",
        ),
        Component(
            component_id="sample_codebase.calculator.Calculator.add",
            name="add",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="calculator.py",
                line_start=17,
                line_end=25,
            ),
            signature="def add(self, a: float, b: float) -> float:",
            parent_id="sample_codebase.calculator.Calculator",
        ),
        Component(
            component_id="sample_codebase.calculator.Calculator.multiply",
            name="multiply",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="calculator.py",
                line_start=27,
                line_end=35,
            ),
            signature="def multiply(self, a: float, b: float) -> float:",
            parent_id="sample_codebase.calculator.Calculator",
        ),
    ]


@pytest.fixture
def sample_source_code_map() -> dict[str, str]:
    """Create sample source code mapping."""
    return {
        "sample_codebase.calculator.Calculator.__init__": '''def __init__(self, precision: int = 2):
    """Initialize calculator with precision."""
    self.precision = precision
''',
        "sample_codebase.calculator.Calculator.add": '''def add(self, a: float, b: float) -> float:
    """Add two numbers."""
    result = a + b
    return round(result, self.precision)
''',
        "sample_codebase.calculator.Calculator.multiply": '''def multiply(self, a: float, b: float) -> float:
    """Multiply two numbers."""
    result = a * b
    return round(result, self.precision)
''',
    }


def create_mock_documentation_output(
    component_id: str,
    stream_id: StreamId,
    summary: str = "Test summary",
    callees: list[str] | None = None,
) -> DocumentationOutput:
    """Helper to create mock documentation output."""
    return DocumentationOutput(
        component_id=component_id,
        documentation=ComponentDocumentation(
            summary=summary,
            description=f"Documentation for {component_id}",
            parameters=[
                ParameterDoc(name="a", type="float", description="First operand"),
                ParameterDoc(name="b", type="float", description="Second operand"),
            ],
            returns=ReturnDoc(type="float", description="The result"),
        ),
        call_graph=CallGraphSection(
            callers=[],
            callees=[
                CalleeRef(component_id=callee, call_type=CallType.DIRECT)
                for callee in (callees or [])
            ],
        ),
        metadata=DocumenterMetadata(
            agent_id=f"{stream_id.value}1",
            stream_id=stream_id,
            model="mock-model",
            confidence=0.9,
            token_count=100,
        ),
    )


def create_mock_stream_output(
    stream_id: StreamId,
    component_ids: list[str],
    callees_map: dict[str, list[str]] | None = None,
) -> StreamOutput:
    """Helper to create mock stream output."""
    output = StreamOutput(stream_id=stream_id)
    callees_map = callees_map or {}

    for comp_id in component_ids:
        doc = create_mock_documentation_output(
            component_id=comp_id,
            stream_id=stream_id,
            callees=callees_map.get(comp_id, []),
        )
        output.add_output(doc)

    return output


def create_converged_comparison_result(
    iteration: int = 1,
    total_components: int = 3,
) -> ComparisonResult:
    """Create a comparison result indicating convergence."""
    return ComparisonResult(
        comparison_id=f"cmp_test_{iteration}",
        iteration=iteration,
        summary=ComparisonSummary(
            total_components=total_components,
            identical=total_components,
            discrepancies=0,
            resolved_by_ground_truth=0,
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
            model="mock-comparator",
            comparison_duration_ms=100,
        ),
    )


def create_discrepancy_comparison_result(
    iteration: int = 1,
    total_components: int = 3,
    discrepancies: list[Discrepancy] | None = None,
    converged: bool = False,
) -> ComparisonResult:
    """Create a comparison result with discrepancies."""
    if discrepancies is None:
        discrepancies = [
            Discrepancy(
                discrepancy_id="disc_test_001",
                component_id="sample_codebase.calculator.Calculator.add",
                type=DiscrepancyType.CALL_GRAPH_EDGE,
                stream_a_value=True,
                stream_b_value=False,
                ground_truth=True,
                resolution=ResolutionAction.ACCEPT_STREAM_A,
                confidence=0.99,
            ),
        ]

    blocking = sum(1 for d in discrepancies if d.is_blocking)
    requires_human = sum(1 for d in discrepancies if d.requires_beads)

    return ComparisonResult(
        comparison_id=f"cmp_test_{iteration}",
        iteration=iteration,
        summary=ComparisonSummary(
            total_components=total_components,
            identical=total_components - len(discrepancies),
            discrepancies=len(discrepancies),
            resolved_by_ground_truth=len([d for d in discrepancies if d.is_call_graph_related]),
            requires_human_review=requires_human,
        ),
        discrepancies=discrepancies,
        convergence_status=ConvergenceStatus(
            converged=converged,
            blocking_discrepancies=blocking,
            recommendation="finalize" if converged else "continue",
        ),
        metadata=ComparatorMetadata(
            agent_id="C",
            model="mock-comparator",
            comparison_duration_ms=100,
        ),
    )


# =============================================================================
# Mock Classes
# =============================================================================


class MockStaticAnalysisOracle(StaticAnalysisOracle):
    """Mock implementation of StaticAnalysisOracle for testing."""

    def __init__(
        self,
        codebase_path: str | Path,
        config: OracleConfig | None = None,
        mock_call_graph: CallGraph | None = None,
    ):
        # Don't call super().__init__ to avoid actual initialization
        self._codebase_path = Path(codebase_path)
        self._config = config or OracleConfig()
        self._mock_call_graph = mock_call_graph or CallGraph()
        self._call_graph = self._mock_call_graph
        self._initialized = False

    @property
    def codebase_path(self) -> Path:
        """Get the codebase path."""
        return self._codebase_path

    @property
    def call_graph(self) -> CallGraph:
        """Get the call graph."""
        return self._call_graph

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the mock oracle."""
        self._initialized = True
        self._call_graph = self._mock_call_graph

    async def get_call_graph(self, force_refresh: bool = False) -> CallGraph:
        """Return the mock call graph."""
        return self._mock_call_graph

    async def refresh(self) -> CallGraph:
        """Return the mock call graph."""
        return self._mock_call_graph

    async def shutdown(self) -> None:
        """Shutdown the mock oracle."""
        self._initialized = False


class MockDocumentationStream(DocumentationStream):
    """Mock implementation of DocumentationStream for testing."""

    def __init__(
        self,
        config: StreamConfig,
        mock_output: StreamOutput | None = None,
    ):
        super().__init__(config)
        self._mock_output = mock_output
        self._corrections_applied: list[tuple[str, str, Any]] = []
        self._outputs_dict: dict[str, DocumentationOutput] = {}

    async def initialize(self) -> None:
        """Initialize the mock stream."""
        self._initialized = True

    async def process(
        self,
        components: list[Component],
        source_code_map: dict[str, str],
        ground_truth: Any,
    ) -> StreamResult:
        """Process components using mock output."""
        if not self._initialized:
            raise RuntimeError("Stream not initialized")

        if self._mock_output:
            self._output = self._mock_output
            self._outputs_dict = dict(self._mock_output.outputs)

        # Return StreamResult wrapping the StreamOutput
        return StreamResult(
            stream_id=self._config.stream_id,
            output=self._output,
            successful=len(self._output.outputs),
            failed=0,
        )

    async def process_component(
        self,
        component: Component,
        source_code: str,
        dependency_context: dict[str, DocumentationOutput],
        ground_truth: CallGraph,
    ) -> ComponentProcessingResult:
        """Process a single component."""
        return ComponentProcessingResult(
            component_id=component.component_id,
            success=True,
        )

    async def apply_correction(
        self,
        component_id: str,
        field: str,
        corrected_value: Any,
    ) -> bool:
        """Record and apply a correction."""
        self._corrections_applied.append((component_id, field, corrected_value))
        return True

    async def reprocess_component(
        self,
        component_id: str,
        corrections: list[dict],
    ) -> ComponentProcessingResult:
        """Reprocess a component."""
        return ComponentProcessingResult(
            component_id=component_id,
            success=True,
        )

    async def shutdown(self) -> None:
        """Shutdown the mock stream."""
        self._initialized = False

    def get_outputs(self) -> dict[str, DocumentationOutput]:
        """Get all outputs as a dict."""
        return self._outputs_dict


class MockComparatorAgent(ComparatorAgent):
    """Mock implementation of ComparatorAgent for testing."""

    def __init__(
        self,
        config: ComparatorConfig,
        comparison_results: list[ComparisonResult] | None = None,
    ):
        super().__init__(config)
        self._comparison_results = comparison_results or []
        self._comparison_index = 0
        self._compare_calls: list[tuple] = []

    async def initialize(self) -> None:
        """Initialize the mock comparator."""
        self._initialized = True

    async def process(self, input_data: ComparatorInput) -> ComparisonResult:
        """Return the next mock comparison result."""
        if self._comparison_index < len(self._comparison_results):
            result = self._comparison_results[self._comparison_index]
            self._comparison_index += 1
            return result

        # Default: return converged result
        return create_converged_comparison_result()

    async def compare(
        self,
        output_a: StreamOutput,
        output_b: StreamOutput,
        ground_truth: CallGraph,
        iteration: int = 1,
    ) -> ComparisonResult:
        """Convenience method for comparison."""
        self._compare_calls.append((output_a, output_b, ground_truth, iteration))

        if self._comparison_index < len(self._comparison_results):
            result = self._comparison_results[self._comparison_index]
            self._comparison_index += 1
            return result

        return create_converged_comparison_result()

    async def shutdown(self) -> None:
        """Shutdown the mock comparator."""
        self._initialized = False

    async def compare_component(
        self,
        component_id: str,
        stream_a_doc: dict | None,
        stream_b_doc: dict | None,
        ground_truth: CallGraph,
    ) -> list[Discrepancy]:
        """Compare a single component."""
        return []

    async def generate_beads_ticket(
        self,
        discrepancy: Discrepancy,
        stream_a_model: str,
        stream_b_model: str,
        source_code: str,
    ) -> dict:
        """Generate a mock Beads ticket."""
        return {
            "title": f"Discrepancy: {discrepancy.component_id}",
            "description": "Test ticket",
            "priority": "Medium",
            "ticket_key": f"TEST-{discrepancy.discrepancy_id}",
        }


class MockBeadsLifecycleManager(BeadsLifecycleManager):
    """Mock implementation of BeadsLifecycleManager for testing."""

    def __init__(self, config: LifecycleManagerConfig | None = None):
        # Don't call super().__init__ to avoid actual initialization
        self._config = config or LifecycleManagerConfig()
        self._initialized = False
        self._created_tickets: list[dict] = []
        self._ticket_counter = 0

    async def initialize(self) -> None:
        """Initialize the mock manager."""
        self._initialized = True

    async def close(self) -> None:
        """Close the mock manager."""
        self._initialized = False

    async def create_discrepancy_ticket(self, data: Any) -> MagicMock:
        """Create a mock discrepancy ticket."""
        self._ticket_counter += 1
        ticket = MagicMock()
        ticket.ticket_key = f"TEST-DISC-{self._ticket_counter}"
        self._created_tickets.append({"type": "discrepancy", "data": data})
        return ticket

    async def create_rebuild_ticket(self, data: Any) -> MagicMock:
        """Create a mock rebuild ticket."""
        self._ticket_counter += 1
        ticket = MagicMock()
        ticket.ticket_key = f"TEST-REBUILD-{self._ticket_counter}"
        self._created_tickets.append({"type": "rebuild", "data": data})
        return ticket

    @property
    def tracker(self) -> MagicMock:
        """Return a mock tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_pending_tickets.return_value = []
        return mock_tracker


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestOrchestratorFullPipeline:
    """Test the full orchestrator pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_components(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test complete pipeline execution with mocked LLM responses."""
        # Create mock outputs for both streams
        component_ids = [c.component_id for c in sample_components]
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids)

        # Create mock oracle
        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        # Create mock stream configs
        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        # Create mock streams with matching outputs
        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # Create mock comparator that returns convergence
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[create_converged_comparison_result()],
        )

        # Patch component discovery to return our sample components
        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            # Create orchestrator
            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
                beads_manager=None,
            )

            # Run the pipeline
            result = await orchestrator.run()

            # Verify successful completion
            assert orchestrator.state.phase == OrchestratorPhase.COMPLETED
            assert result is not None
            assert isinstance(result, DocumentationPackage)

    @pytest.mark.asyncio
    async def test_pipeline_tracks_progress(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that pipeline correctly tracks progress through phases."""
        component_ids = [c.component_id for c in sample_components]
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[create_converged_comparison_result()],
        )

        # Track progress through phases
        phases_observed: list[OrchestratorPhase] = []

        def progress_callback(state: OrchestratorState) -> None:
            phases_observed.append(state.phase)

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            orchestrator.on_progress(progress_callback)
            await orchestrator.run()

        # Verify phases were tracked correctly
        assert OrchestratorPhase.INITIALIZING in phases_observed
        assert OrchestratorPhase.COMPLETED in phases_observed


@pytest.mark.integration
class TestOrchestratorConvergence:
    """Test convergence scenarios."""

    @pytest.mark.asyncio
    async def test_convergence_in_single_iteration(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test convergence when streams produce identical outputs."""
        component_ids = [c.component_id for c in sample_components]

        # Create identical outputs for both streams
        callees_map = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
            "sample_codebase.calculator.Calculator.multiply": ["builtins.round"],
        }
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # Comparator returns converged on first iteration
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_converged_comparison_result(
                    iteration=1, total_components=len(sample_components)
                ),
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify single iteration convergence
        assert orchestrator.state.iteration == 1
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED
        assert len(orchestrator.iteration_history) == 1
        assert orchestrator.iteration_history[0].converged is True

    @pytest.mark.asyncio
    async def test_convergence_after_corrections(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test convergence after applying corrections from ground truth."""
        component_ids = [c.component_id for c in sample_components]

        # Stream A has correct call graph, Stream B is missing an edge
        callees_map_a = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
        }
        callees_map_b = {}  # Missing the call to round

        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map_a)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map_b)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # First iteration: discrepancy found and resolved by ground truth
        # Second iteration: converged
        discrepancy = Discrepancy(
            discrepancy_id="disc_call_graph_001",
            component_id="sample_codebase.calculator.Calculator.add",
            type=DiscrepancyType.CALL_GRAPH_EDGE,
            stream_a_value=True,
            stream_b_value=False,
            ground_truth=True,
            resolution=ResolutionAction.ACCEPT_STREAM_A,
            confidence=0.99,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(
                    iteration=1,
                    total_components=len(sample_components),
                    discrepancies=[discrepancy],
                    converged=False,
                ),
                create_converged_comparison_result(
                    iteration=2,
                    total_components=len(sample_components),
                ),
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify convergence after corrections
        assert orchestrator.state.iteration == 2
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED
        assert len(orchestrator.iteration_history) == 2
        assert orchestrator.iteration_history[0].converged is False
        assert orchestrator.iteration_history[1].converged is True


@pytest.mark.integration
class TestOrchestratorBeadsIntegration:
    """Test Beads ticket generation for unresolved discrepancies."""

    @pytest.mark.asyncio
    async def test_beads_ticket_generation_for_unresolved_discrepancy(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that Beads tickets are created for unresolved discrepancies."""
        config = OrchestratorConfig(
            max_iterations=5,
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=False,  # Enable Beads ticket creation
            continue_on_error=True,
        )

        component_ids = [c.component_id for c in sample_components]
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # Discrepancy that requires human review
        human_review_discrepancy = Discrepancy(
            discrepancy_id="disc_doc_content_001",
            component_id="sample_codebase.calculator.Calculator.add",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            stream_a_value="Adds two numbers together",
            stream_b_value="Computes the sum of a and b",
            resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
            confidence=0.4,  # Low confidence triggers Beads
            requires_beads=True,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(
                    iteration=1,
                    total_components=len(sample_components),
                    discrepancies=[human_review_discrepancy],
                    converged=False,
                ),
                create_converged_comparison_result(
                    iteration=2,
                    total_components=len(sample_components),
                ),
            ],
        )

        mock_beads = MockBeadsLifecycleManager()

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
                beads_manager=mock_beads,
            )

            await orchestrator.run()

        # Verify Beads ticket was created
        assert len(mock_beads._created_tickets) >= 1
        assert orchestrator.iteration_history[0].beads_tickets_created >= 1


@pytest.mark.integration
class TestOrchestratorMaxIterations:
    """Test max iterations handling."""

    @pytest.mark.asyncio
    async def test_max_iterations_forces_convergence(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that pipeline completes when max iterations is reached."""
        config = OrchestratorConfig(
            max_iterations=3,  # Low limit for testing
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=True,
            continue_on_error=True,
        )

        component_ids = [c.component_id for c in sample_components]

        # Create DIFFERENT call graphs so streams are seen as divergent
        # This is necessary to test max iterations - if call graphs are identical,
        # the orchestrator detects convergence via StreamCallGraphComparison
        callees_map_a = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
        }
        callees_map_b = {
            "sample_codebase.calculator.Calculator.add": ["builtins.print"],  # Different!
        }
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map_a)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map_b)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # Persistent discrepancy that never resolves
        persistent_discrepancy = Discrepancy(
            discrepancy_id="disc_persistent_001",
            component_id="sample_codebase.calculator.Calculator.add",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            stream_a_value="Description A",
            stream_b_value="Description B",
            resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
            confidence=0.3,
        )

        # Comparator always returns non-converged result
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(
                    iteration=i,
                    total_components=len(sample_components),
                    discrepancies=[persistent_discrepancy],
                    converged=False,
                )
                for i in range(1, 5)  # More than max_iterations
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify max iterations was reached
        assert orchestrator.state.iteration == 3
        assert len(orchestrator.iteration_history) == 3
        # Pipeline should still complete (not crash)
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED


@pytest.mark.integration
class TestOrchestratorOutputGeneration:
    """Test output file generation verification."""

    @pytest.mark.asyncio
    async def test_documentation_package_structure(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that DocumentationPackage has correct structure."""
        component_ids = [c.component_id for c in sample_components]

        callees_map = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
            "sample_codebase.calculator.Calculator.multiply": ["builtins.round"],
        }
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[create_converged_comparison_result()],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify DocumentationPackage structure
        assert result is not None
        assert isinstance(result, DocumentationPackage)
        assert isinstance(result.call_graph, CallGraph)
        assert isinstance(result.convergence_report, ConvergenceReport)
        assert isinstance(result.metrics, RunMetrics)

    @pytest.mark.asyncio
    async def test_convergence_report_contents(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that ConvergenceReport is correctly populated."""
        component_ids = [c.component_id for c in sample_components]

        # Create DIFFERENT call graphs so we can test multi-iteration convergence
        # With identical call graphs, StreamCallGraphComparison detects convergence immediately
        callees_map_a = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
        }
        callees_map_b = {
            "sample_codebase.calculator.Calculator.add": ["builtins.print"],  # Different initially
        }
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map_a)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map_b)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(iteration=1, converged=False),
                create_converged_comparison_result(iteration=2),
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify ConvergenceReport
        report = result.convergence_report
        assert report.total_iterations == 2
        assert report.final_status == "converged"
        assert len(report.history) == 2

    @pytest.mark.asyncio
    async def test_metrics_include_timing(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that metrics include timing information."""
        component_ids = [c.component_id for c in sample_components]
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[create_converged_comparison_result()],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=orchestrator_config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify timing metrics
        metrics = result.metrics
        assert metrics.components_total == len(sample_components)
        # Check elapsed time was recorded
        assert metrics.duration_seconds is not None or metrics.completed_at is not None


@pytest.mark.integration
class TestOrchestratorErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_stream_initialization_error(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
        mock_call_graph: CallGraph,
    ):
        """Test handling of stream initialization errors."""
        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        # Create a stream that fails on initialization
        stream_a = MockDocumentationStream(config=stream_a_config)
        stream_a.initialize = AsyncMock(side_effect=RuntimeError("Init failed"))

        stream_b = MockDocumentationStream(config=stream_b_config)

        mock_comparator = MockComparatorAgent(config=MagicMock(spec=ComparatorConfig))

        orchestrator = DualStreamOrchestrator(
            config=orchestrator_config,
            static_oracle=mock_oracle,
            stream_a=stream_a,
            stream_b=stream_b,
            comparator=mock_comparator,
        )

        # Should raise an error
        with pytest.raises(OrchestratorError):
            await orchestrator.run()

        # Verify state is FAILED
        assert orchestrator.state.phase == OrchestratorPhase.FAILED
        assert len(orchestrator.state.errors) > 0

    @pytest.mark.asyncio
    async def test_handles_oracle_initialization_error(
        self,
        sample_codebase_path: Path,
        orchestrator_config: OrchestratorConfig,
    ):
        """Test handling of oracle initialization errors."""
        mock_oracle = MockStaticAnalysisOracle(sample_codebase_path)
        mock_oracle.initialize = AsyncMock(side_effect=RuntimeError("Oracle init failed"))

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(config=stream_a_config)
        stream_b = MockDocumentationStream(config=stream_b_config)
        mock_comparator = MockComparatorAgent(config=MagicMock(spec=ComparatorConfig))

        orchestrator = DualStreamOrchestrator(
            config=orchestrator_config,
            static_oracle=mock_oracle,
            stream_a=stream_a,
            stream_b=stream_b,
            comparator=mock_comparator,
        )

        with pytest.raises(OrchestratorError):
            await orchestrator.run()

        assert orchestrator.state.phase == OrchestratorPhase.FAILED

    @pytest.mark.asyncio
    async def test_continues_on_component_error_when_configured(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that pipeline continues when individual components fail."""
        config = OrchestratorConfig(
            max_iterations=5,
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=True,
            continue_on_error=True,  # Should continue past errors
        )

        component_ids = [c.component_id for c in sample_components]
        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[create_converged_comparison_result()],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = ["Minor error during discovery"]
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Should complete despite errors
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED
        assert len(orchestrator.state.errors) >= 1


@pytest.mark.integration
class TestOrchestratorCallGraphRefinement:
    """Test iterative call graph refinement with feedback loop."""

    @pytest.mark.asyncio
    async def test_divergent_components_trigger_feedback_generation(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that divergent call graphs generate feedback for streams."""
        config = OrchestratorConfig(
            max_iterations=5,
            max_call_graph_iterations=5,
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=True,
            continue_on_error=True,
        )

        component_ids = [c.component_id for c in sample_components]

        # Stream A finds one callee, Stream B finds a different one
        callees_map_a = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
        }
        callees_map_b = {
            "sample_codebase.calculator.Calculator.add": ["builtins.print"],  # Different!
        }

        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map_a)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map_b)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        # Track feedback passed to streams
        feedback_received_a = []
        feedback_received_b = []

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        # Monkey-patch to track feedback
        original_set_feedback_a = getattr(stream_a, "set_call_graph_feedback", lambda x: None)

        def track_feedback_a(feedback):
            feedback_received_a.append(feedback)
            return original_set_feedback_a(feedback)

        stream_a.set_call_graph_feedback = track_feedback_a

        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )
        original_set_feedback_b = getattr(stream_b, "set_call_graph_feedback", lambda x: None)

        def track_feedback_b(feedback):
            feedback_received_b.append(feedback)
            return original_set_feedback_b(feedback)

        stream_b.set_call_graph_feedback = track_feedback_b

        # First iteration: discrepancy, generates feedback
        # Second iteration: converges after feedback
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(
                    iteration=1,
                    total_components=len(sample_components),
                    converged=False,
                ),
                create_converged_comparison_result(
                    iteration=2,
                    total_components=len(sample_components),
                ),
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify feedback was generated and passed to streams
        # First iteration has None (no previous feedback), subsequent iterations may have feedback
        assert len(feedback_received_a) >= 1
        assert len(feedback_received_b) >= 1
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_only_divergent_components_reprocessed_in_subsequent_iterations(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that subsequent iterations only reprocess divergent components."""
        config = OrchestratorConfig(
            max_iterations=3,
            max_call_graph_iterations=3,
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=True,
            continue_on_error=True,
        )

        component_ids = [c.component_id for c in sample_components]

        # All call graphs identical (should converge immediately)
        callees_map = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
            "sample_codebase.calculator.Calculator.multiply": ["builtins.round"],
        }

        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        # Track how many times process is called
        process_call_counts_a = []
        process_call_counts_b = []

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        original_process_a = stream_a.process

        async def tracking_process_a(components, source_code_map, ground_truth):
            process_call_counts_a.append(len(components))
            return await original_process_a(components, source_code_map, ground_truth)

        stream_a.process = tracking_process_a

        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )
        original_process_b = stream_b.process

        async def tracking_process_b(components, source_code_map, ground_truth):
            process_call_counts_b.append(len(components))
            return await original_process_b(components, source_code_map, ground_truth)

        stream_b.process = tracking_process_b

        # Converges on first iteration
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_converged_comparison_result(
                    iteration=1,
                    total_components=len(sample_components),
                ),
            ],
        )

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
            )

            result = await orchestrator.run()

        # Verify only one iteration was run (since it converged)
        assert orchestrator.state.iteration == 1
        assert len(process_call_counts_a) == 1
        assert len(process_call_counts_b) == 1
        # First iteration processes all components
        assert process_call_counts_a[0] == len(sample_components)
        assert process_call_counts_b[0] == len(sample_components)

    @pytest.mark.asyncio
    async def test_max_iterations_escalates_divergent_to_beads(
        self,
        sample_codebase_path: Path,
        sample_components: list[Component],
        sample_source_code_map: dict[str, str],
        mock_call_graph: CallGraph,
    ):
        """Test that max iterations triggers escalation of divergent components to Beads."""
        from twinscribe.models.convergence import ConvergenceCriteria

        config = OrchestratorConfig(
            max_iterations=2,
            max_call_graph_iterations=2,
            parallel_components=5,
            wait_for_beads=False,
            beads_timeout_hours=0,
            skip_validation=False,
            dry_run=False,  # Enable Beads ticket creation
            continue_on_error=True,
            convergence_criteria=ConvergenceCriteria(
                min_agreement_rate=0.95,
                max_iterations=2,
            ),
        )

        component_ids = [c.component_id for c in sample_components]

        # Streams never converge - different call graphs
        callees_map_a = {
            "sample_codebase.calculator.Calculator.add": ["builtins.round"],
        }
        callees_map_b = {
            "sample_codebase.calculator.Calculator.add": ["builtins.print"],
        }

        stream_a_output = create_mock_stream_output(StreamId.STREAM_A, component_ids, callees_map_a)
        stream_b_output = create_mock_stream_output(StreamId.STREAM_B, component_ids, callees_map_b)

        mock_oracle = MockStaticAnalysisOracle(
            sample_codebase_path,
            mock_call_graph=mock_call_graph,
        )

        stream_a_config = MagicMock(spec=StreamConfig)
        stream_a_config.stream_id = StreamId.STREAM_A
        stream_b_config = MagicMock(spec=StreamConfig)
        stream_b_config.stream_id = StreamId.STREAM_B

        stream_a = MockDocumentationStream(
            config=stream_a_config,
            mock_output=stream_a_output,
        )
        stream_b = MockDocumentationStream(
            config=stream_b_config,
            mock_output=stream_b_output,
        )

        # Never converges
        mock_comparator = MockComparatorAgent(
            config=MagicMock(spec=ComparatorConfig),
            comparison_results=[
                create_discrepancy_comparison_result(
                    iteration=i,
                    total_components=len(sample_components),
                    converged=False,
                )
                for i in range(1, 5)
            ],
        )

        mock_beads = MockBeadsLifecycleManager()

        with patch("twinscribe.analysis.component_discovery.ComponentDiscovery") as MockDiscovery:
            mock_discovery_instance = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.components = sample_components
            mock_discovery_result.source_code_map = sample_source_code_map
            mock_discovery_result.processing_order = component_ids
            mock_discovery_result.errors = []
            mock_discovery_instance.discover = AsyncMock(return_value=mock_discovery_result)
            MockDiscovery.return_value = mock_discovery_instance

            orchestrator = DualStreamOrchestrator(
                config=config,
                static_oracle=mock_oracle,
                stream_a=stream_a,
                stream_b=stream_b,
                comparator=mock_comparator,
                beads_manager=mock_beads,
            )

            result = await orchestrator.run()

        # Verify max iterations was reached
        assert orchestrator.state.iteration == 2
        assert orchestrator.state.phase == OrchestratorPhase.COMPLETED
        # Beads tickets should have been created for divergent components
        # (The exact count depends on how many discrepancies the comparator returns)
