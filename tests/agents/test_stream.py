"""
Tests for DocumentationStream interface and implementations.

Tests cover:
- DocumentationStream initialization and lifecycle
- process() method with mock components
- process_component() with mock LLM responses
- apply_correction() for discrepancy resolution
- reprocess_component() for iteration support
- Error handling and retry logic
- Progress callbacks
- Edge cases and error conditions

All tests use mocks to avoid real API calls.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.agents.documenter import DocumenterAgent, DocumenterConfig
from twinscribe.agents.stream import (
    ComponentProcessingResult,
    DocumentationStream,
    StreamConfig,
    StreamProgressCallback,
    StreamResult,
)
from twinscribe.agents.validator import ValidatorAgent, ValidatorConfig
from twinscribe.models.base import (
    CallType,
    ComponentType,
    ModelTier,
    StreamId,
    ValidationStatus,
)
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ComponentLocation,
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
from twinscribe.models.validation import (
    CallGraphAccuracy,
    CompletenessCheck,
    ValidationResult,
    ValidatorMetadata,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def stream_config_a() -> StreamConfig:
    """Create a StreamConfig for Stream A."""
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
        max_retries=3,
        continue_on_error=True,
    )


@pytest.fixture
def stream_config_b() -> StreamConfig:
    """Create a StreamConfig for Stream B."""
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
        max_retries=3,
        continue_on_error=True,
    )


@pytest.fixture
def sample_component() -> Component:
    """Create a sample Component for testing."""
    return Component(
        component_id="sample_module.Calculator.add",
        name="add",
        type=ComponentType.METHOD,
        location=ComponentLocation(
            file_path="src/sample_module.py",
            line_start=15,
            line_end=25,
        ),
        signature="def add(self, a: float, b: float) -> float:",
        parent_id="sample_module.Calculator",
        dependencies=["sample_module.helper_function"],
        existing_docstring="Add two numbers.",
    )


@pytest.fixture
def sample_component_list() -> list[Component]:
    """Create a list of sample Components in topological order."""
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
def sample_call_graph() -> CallGraph:
    """Create a sample CallGraph for testing."""
    return CallGraph(
        edges=[
            CallEdge(
                caller="sample_module.Calculator.add",
                callee="builtins.round",
                call_site_line=22,
                call_type=CallType.DIRECT,
            ),
            CallEdge(
                caller="sample_module.AdvancedCalculator.compute",
                callee="sample_module.Calculator.add",
                call_site_line=35,
                call_type=CallType.LOOP,
            ),
        ],
        source="pycg",
    )


@pytest.fixture
def sample_documentation_output() -> DocumentationOutput:
    """Create a sample DocumentationOutput."""
    return DocumentationOutput(
        component_id="sample_module.Calculator.add",
        documentation=ComponentDocumentation(
            summary="Add two numbers.",
            description="Adds two floating point numbers and returns the result rounded to the configured precision.",
            parameters=[
                ParameterDoc(name="a", type="float", description="First number to add"),
                ParameterDoc(name="b", type="float", description="Second number to add"),
            ],
            returns=ReturnDoc(
                type="float", description="Sum of a and b, rounded to configured precision"
            ),
            raises=[],
        ),
        call_graph=CallGraphSection(
            callers=[
                CallerRef(
                    component_id="sample_module.AdvancedCalculator.compute",
                    call_site_line=35,
                    call_type=CallType.LOOP,
                ),
            ],
            callees=[
                CalleeRef(
                    component_id="builtins.round",
                    call_site_line=22,
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
def sample_validation_result() -> ValidationResult:
    """Create a sample ValidationResult."""
    return ValidationResult(
        component_id="sample_module.Calculator.add",
        validation_result=ValidationStatus.PASS,
        completeness=CompletenessCheck(
            score=1.0,
            missing_elements=[],
            extra_elements=[],
        ),
        call_graph_accuracy=CallGraphAccuracy(
            score=1.0,
            verified_callees=["builtins.round"],
            missing_callees=[],
            false_callees=[],
            verified_callers=["sample_module.AdvancedCalculator.compute"],
            missing_callers=[],
            false_callers=[],
        ),
        corrections_applied=[],
        metadata=ValidatorMetadata(
            agent_id="A2",
            stream_id=StreamId.STREAM_A,
            model="claude-haiku-4-5-20251001",
            static_analyzer="pycg",
            token_count=200,
        ),
    )


@pytest.fixture
def mock_documenter_agent() -> MagicMock:
    """Create a mock DocumenterAgent."""
    mock = MagicMock(spec=DocumenterAgent)
    mock.initialize = AsyncMock()
    mock.process = AsyncMock()
    mock.shutdown = AsyncMock()
    return mock


@pytest.fixture
def mock_validator_agent() -> MagicMock:
    """Create a mock ValidatorAgent."""
    mock = MagicMock(spec=ValidatorAgent)
    mock.initialize = AsyncMock()
    mock.process = AsyncMock()
    mock.shutdown = AsyncMock()
    return mock


@pytest.fixture
def mock_progress_callback() -> MagicMock:
    """Create a mock StreamProgressCallback."""
    mock = MagicMock(spec=StreamProgressCallback)
    mock.on_component_start = AsyncMock()
    mock.on_component_complete = AsyncMock()
    mock.on_validation_complete = AsyncMock()
    mock.on_stream_complete = AsyncMock()
    mock.on_error = AsyncMock()
    return mock


# =============================================================================
# Test Classes
# =============================================================================


class TestStreamConfig:
    """Tests for StreamConfig model."""

    def test_valid_config(self, stream_config_a: StreamConfig):
        """Test creating a valid StreamConfig."""
        assert stream_config_a.stream_id == StreamId.STREAM_A
        assert stream_config_a.batch_size == 5
        assert stream_config_a.max_retries == 3
        assert stream_config_a.continue_on_error is True

    def test_config_validation_batch_size(self):
        """Test that batch_size is validated (1-20)."""
        # This test will need implementation once we have a concrete stream class
        pass

    def test_config_validation_max_retries(self):
        """Test that max_retries is validated (>= 0)."""
        # This test will need implementation once we have a concrete stream class
        pass


class TestStreamResult:
    """Tests for StreamResult dataclass."""

    def test_stream_result_creation(self):
        """Test creating a StreamResult."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(
            stream_id=StreamId.STREAM_A,
            output=output,
            successful=5,
            failed=1,
            failed_component_ids=["comp_1"],
            total_tokens=1000,
            total_cost=0.05,
            started_at=datetime(2026, 1, 6, 10, 0, 0),
            completed_at=datetime(2026, 1, 6, 10, 5, 0),
        )
        assert result.stream_id == StreamId.STREAM_A
        assert result.successful == 5
        assert result.failed == 1

    def test_duration_seconds_property(self):
        """Test duration_seconds calculation."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(
            stream_id=StreamId.STREAM_A,
            output=output,
            started_at=datetime(2026, 1, 6, 10, 0, 0),
            completed_at=datetime(2026, 1, 6, 10, 5, 0),
        )
        assert result.duration_seconds == 300.0

    def test_duration_seconds_none_when_incomplete(self):
        """Test duration_seconds is None when times not set."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(
            stream_id=StreamId.STREAM_A,
            output=output,
        )
        assert result.duration_seconds is None

    def test_success_rate_property(self):
        """Test success_rate calculation."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(
            stream_id=StreamId.STREAM_A,
            output=output,
            successful=8,
            failed=2,
        )
        assert result.success_rate == 0.8

    def test_success_rate_zero_total(self):
        """Test success_rate when no components processed."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(
            stream_id=StreamId.STREAM_A,
            output=output,
            successful=0,
            failed=0,
        )
        assert result.success_rate == 0.0


class TestComponentProcessingResult:
    """Tests for ComponentProcessingResult dataclass."""

    def test_successful_result(
        self,
        sample_documentation_output: DocumentationOutput,
        sample_validation_result: ValidationResult,
    ):
        """Test creating a successful processing result."""
        result = ComponentProcessingResult(
            component_id="sample_module.Calculator.add",
            documentation=sample_documentation_output,
            validation=sample_validation_result,
            success=True,
            retries=0,
        )
        assert result.success is True
        assert result.documentation is not None
        assert result.validation is not None
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed processing result."""
        result = ComponentProcessingResult(
            component_id="sample_module.Calculator.add",
            success=False,
            error="LLM API timeout",
            retries=3,
        )
        assert result.success is False
        assert result.documentation is None
        assert result.error == "LLM API timeout"
        assert result.retries == 3


class TestStreamProgressCallback:
    """Tests for StreamProgressCallback interface."""

    @pytest.mark.asyncio
    async def test_callback_on_component_start(self, mock_progress_callback: MagicMock):
        """Test on_component_start callback."""
        await mock_progress_callback.on_component_start(
            stream_id=StreamId.STREAM_A,
            component_id="test.component",
            index=0,
            total=10,
        )
        mock_progress_callback.on_component_start.assert_called_once_with(
            stream_id=StreamId.STREAM_A,
            component_id="test.component",
            index=0,
            total=10,
        )

    @pytest.mark.asyncio
    async def test_callback_on_component_complete(self, mock_progress_callback: MagicMock):
        """Test on_component_complete callback."""
        await mock_progress_callback.on_component_complete(
            stream_id=StreamId.STREAM_A,
            component_id="test.component",
            success=True,
            duration_ms=1500.0,
        )
        mock_progress_callback.on_component_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_on_validation_complete(self, mock_progress_callback: MagicMock):
        """Test on_validation_complete callback."""
        await mock_progress_callback.on_validation_complete(
            stream_id=StreamId.STREAM_A,
            component_id="test.component",
            passed=True,
            corrections=0,
        )
        mock_progress_callback.on_validation_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_on_stream_complete(self, mock_progress_callback: MagicMock):
        """Test on_stream_complete callback."""
        output = StreamOutput(stream_id=StreamId.STREAM_A)
        result = StreamResult(stream_id=StreamId.STREAM_A, output=output)
        await mock_progress_callback.on_stream_complete(
            stream_id=StreamId.STREAM_A,
            result=result,
        )
        mock_progress_callback.on_stream_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_on_error(self, mock_progress_callback: MagicMock):
        """Test on_error callback."""
        error = RuntimeError("Test error")
        await mock_progress_callback.on_error(
            stream_id=StreamId.STREAM_A,
            component_id="test.component",
            error=error,
        )
        mock_progress_callback.on_error.assert_called_once()


# =============================================================================
# Tests for DocumentationStream (Abstract Base Class)
# These tests use a concrete mock implementation
# =============================================================================


class MockDocumentationStream(DocumentationStream):
    """Concrete mock implementation of DocumentationStream for testing."""

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self._mock_documenter = MagicMock()
        self._mock_validator = MagicMock()

    async def initialize(self) -> None:
        """Initialize the mock stream."""
        self._initialized = True

    async def process(
        self,
        components: list[Component],
        source_code_map: dict[str, str],
        ground_truth: CallGraph,
    ) -> StreamResult:
        """Process all components (mock implementation)."""
        if not self._initialized:
            raise RuntimeError("Stream not initialized")
        return StreamResult(
            stream_id=self._config.stream_id,
            output=self._output,
            successful=len(components),
            failed=0,
        )

    async def process_component(
        self,
        component: Component,
        source_code: str,
        dependency_context: dict[str, DocumentationOutput],
        ground_truth: CallGraph,
    ) -> ComponentProcessingResult:
        """Process a single component (mock implementation)."""
        return ComponentProcessingResult(
            component_id=component.component_id,
            success=True,
        )

    async def apply_correction(
        self,
        component_id: str,
        corrected_value: any,
        field_path: str,
    ) -> bool:
        """Apply a correction (mock implementation)."""
        if component_id not in self._output.outputs:
            raise ValueError(f"Component {component_id} not found")
        return True

    async def reprocess_component(
        self,
        component_id: str,
        corrections: list[dict],
    ) -> ComponentProcessingResult:
        """Reprocess a component (mock implementation)."""
        return ComponentProcessingResult(
            component_id=component_id,
            success=True,
        )

    async def shutdown(self) -> None:
        """Shutdown the mock stream."""
        self._initialized = False


class TestDocumentationStreamBase:
    """Tests for DocumentationStream base functionality."""

    def test_stream_initialization(self, stream_config_a: StreamConfig):
        """Test stream initialization with config."""
        stream = MockDocumentationStream(stream_config_a)
        assert stream.config == stream_config_a
        assert stream.stream_id == StreamId.STREAM_A
        assert stream.is_initialized is False

    def test_stream_output_property(self, stream_config_a: StreamConfig):
        """Test that output property returns StreamOutput."""
        stream = MockDocumentationStream(stream_config_a)
        assert isinstance(stream.output, StreamOutput)
        assert stream.output.stream_id == StreamId.STREAM_A

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self, stream_config_a: StreamConfig):
        """Test that initialize() sets the initialized flag."""
        stream = MockDocumentationStream(stream_config_a)
        assert stream.is_initialized is False
        await stream.initialize()
        assert stream.is_initialized is True

    @pytest.mark.asyncio
    async def test_process_requires_initialization(self, stream_config_a: StreamConfig):
        """Test that process() requires initialization."""
        stream = MockDocumentationStream(stream_config_a)
        # Override mock to check initialized flag
        with pytest.raises(RuntimeError, match="not initialized"):
            await stream.process([], {}, CallGraph())

    @pytest.mark.asyncio
    async def test_shutdown_clears_flag(self, stream_config_a: StreamConfig):
        """Test that shutdown() clears the initialized flag."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        assert stream.is_initialized is True
        await stream.shutdown()
        assert stream.is_initialized is False

    def test_get_documentation_returns_none_for_missing(self, stream_config_a: StreamConfig):
        """Test get_documentation returns None for non-existent component."""
        stream = MockDocumentationStream(stream_config_a)
        assert stream.get_documentation("nonexistent.component") is None

    def test_get_documentation_returns_output(
        self,
        stream_config_a: StreamConfig,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test get_documentation returns stored output."""
        stream = MockDocumentationStream(stream_config_a)
        stream._output.add_output(sample_documentation_output)
        doc = stream.get_documentation("sample_module.Calculator.add")
        assert doc is not None
        assert doc.component_id == "sample_module.Calculator.add"

    def test_get_all_component_ids(
        self,
        stream_config_a: StreamConfig,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test get_all_component_ids returns correct IDs."""
        stream = MockDocumentationStream(stream_config_a)
        stream._output.add_output(sample_documentation_output)
        ids = stream.get_all_component_ids()
        assert "sample_module.Calculator.add" in ids

    def test_reset_clears_output(
        self,
        stream_config_a: StreamConfig,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test reset() clears all documentation."""
        stream = MockDocumentationStream(stream_config_a)
        stream._output.add_output(sample_documentation_output)
        assert len(stream.get_all_component_ids()) == 1
        stream.reset()
        assert len(stream.get_all_component_ids()) == 0

    @pytest.mark.asyncio
    async def test_apply_correction_raises_for_missing_component(
        self, stream_config_a: StreamConfig
    ):
        """Test apply_correction raises ValueError for missing component."""
        stream = MockDocumentationStream(stream_config_a)
        with pytest.raises(ValueError, match="not found"):
            await stream.apply_correction(
                component_id="nonexistent.component",
                corrected_value="new_value",
                field_path="documentation.summary",
            )


# =============================================================================
# Tests for Stream Processing Logic (to be filled with implementation)
# =============================================================================


class TestStreamProcessing:
    """Tests for stream processing logic."""

    @pytest.mark.asyncio
    async def test_process_empty_components(self, stream_config_a: StreamConfig):
        """Test processing empty component list."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        result = await stream.process([], {}, CallGraph())
        assert result.successful == 0
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_process_single_component(
        self,
        stream_config_a: StreamConfig,
        sample_component: Component,
        sample_source_code_map: dict[str, str],
        sample_call_graph: CallGraph,
    ):
        """Test processing a single component."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        result = await stream.process(
            [sample_component],
            sample_source_code_map,
            sample_call_graph,
        )
        assert result.successful == 1

    @pytest.mark.asyncio
    async def test_process_multiple_components_in_order(
        self,
        stream_config_a: StreamConfig,
        sample_component_list: list[Component],
        sample_source_code_map: dict[str, str],
        sample_call_graph: CallGraph,
    ):
        """Test processing multiple components respects topological order."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        result = await stream.process(
            sample_component_list,
            sample_source_code_map,
            sample_call_graph,
        )
        assert result.successful == len(sample_component_list)

    @pytest.mark.asyncio
    async def test_process_component_builds_dependency_context(
        self,
        stream_config_a: StreamConfig,
        sample_component: Component,
        sample_call_graph: CallGraph,
    ):
        """Test that process_component receives correct dependency context."""
        # This test verifies that when processing components in order,
        # each component receives documentation from its already-processed dependencies
        pass

    @pytest.mark.asyncio
    async def test_process_with_retries_on_failure(self, stream_config_a: StreamConfig):
        """Test that processing retries on transient failures."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_process_continues_on_error_when_configured(self, stream_config_a: StreamConfig):
        """Test that processing continues after failure when continue_on_error=True."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_process_stops_on_error_when_configured(self, stream_config_b: StreamConfig):
        """Test that processing stops after failure when continue_on_error=False."""
        # Implementation needed for concrete stream class
        stream_config_b.continue_on_error = False
        # Implementation needed
        pass


class TestStreamProcessComponent:
    """Tests for process_component method."""

    @pytest.mark.asyncio
    async def test_process_component_success(
        self,
        stream_config_a: StreamConfig,
        sample_component: Component,
        sample_call_graph: CallGraph,
    ):
        """Test successful component processing."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        result = await stream.process_component(
            component=sample_component,
            source_code="def add(self, a, b): return a + b",
            dependency_context={},
            ground_truth=sample_call_graph,
        )
        assert result.success is True
        assert result.component_id == sample_component.component_id

    @pytest.mark.asyncio
    async def test_process_component_with_dependency_context(
        self,
        stream_config_a: StreamConfig,
        sample_component: Component,
        sample_call_graph: CallGraph,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test processing component with existing dependency documentation."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        dependency_context = {
            "sample_module.helper_function": sample_documentation_output,
        }
        result = await stream.process_component(
            component=sample_component,
            source_code="def add(self, a, b): return a + b",
            dependency_context=dependency_context,
            ground_truth=sample_call_graph,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_component_calls_documenter_and_validator(
        self, stream_config_a: StreamConfig
    ):
        """Test that process_component calls both documenter and validator."""
        # Implementation needed for concrete stream class with mocked agents
        pass

    @pytest.mark.asyncio
    async def test_process_component_handles_documenter_error(self, stream_config_a: StreamConfig):
        """Test handling of documenter agent errors."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_process_component_handles_validator_error(self, stream_config_a: StreamConfig):
        """Test handling of validator agent errors."""
        # Implementation needed for concrete stream class
        pass


class TestStreamCorrections:
    """Tests for correction and reprocessing methods."""

    @pytest.mark.asyncio
    async def test_apply_correction_updates_documentation(
        self,
        stream_config_a: StreamConfig,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test that apply_correction updates the stored documentation."""
        stream = MockDocumentationStream(stream_config_a)
        stream._output.add_output(sample_documentation_output)

        success = await stream.apply_correction(
            component_id="sample_module.Calculator.add",
            corrected_value="Updated summary",
            field_path="documentation.summary",
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_apply_correction_call_graph_field(
        self,
        stream_config_a: StreamConfig,
        sample_documentation_output: DocumentationOutput,
    ):
        """Test applying correction to call graph field."""
        stream = MockDocumentationStream(stream_config_a)
        stream._output.add_output(sample_documentation_output)

        # Test correcting a call graph callee
        success = await stream.apply_correction(
            component_id="sample_module.Calculator.add",
            corrected_value=["new.callee.component"],
            field_path="call_graph.callees",
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_reprocess_component_applies_corrections(self, stream_config_a: StreamConfig):
        """Test that reprocess_component applies provided corrections."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()

        corrections = [
            {"field": "documentation.summary", "value": "Corrected summary"},
            {"field": "call_graph.callees", "value": ["corrected.callee"]},
        ]
        result = await stream.reprocess_component(
            component_id="sample_module.Calculator.add",
            corrections=corrections,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_reprocess_component_increments_iteration(self, stream_config_a: StreamConfig):
        """Test that reprocess_component tracks iteration count."""
        # Implementation needed for concrete stream class
        pass


class TestStreamErrorHandling:
    """Tests for error handling in stream processing."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(self, stream_config_a: StreamConfig):
        """Test handling of LLM API timeout errors."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_handles_llm_rate_limit(self, stream_config_a: StreamConfig):
        """Test handling of LLM rate limit errors."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_response(self, stream_config_a: StreamConfig):
        """Test handling of invalid/malformed LLM responses."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_handles_component_not_in_source_map(
        self,
        stream_config_a: StreamConfig,
        sample_component: Component,
        sample_call_graph: CallGraph,
    ):
        """Test handling when component source code is missing."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        # Empty source map - component source not provided
        result = await stream.process(
            [sample_component],
            {},  # Empty source map
            sample_call_graph,
        )
        # Behavior depends on implementation - may fail or skip
        pass

    @pytest.mark.asyncio
    async def test_retry_logic_respects_max_retries(self, stream_config_a: StreamConfig):
        """Test that retry logic respects max_retries config."""
        # Implementation needed for concrete stream class
        pass


class TestStreamCallbacks:
    """Tests for progress callback integration."""

    @pytest.mark.asyncio
    async def test_callbacks_called_in_correct_order(
        self,
        stream_config_a: StreamConfig,
        sample_component_list: list[Component],
        sample_source_code_map: dict[str, str],
        sample_call_graph: CallGraph,
        mock_progress_callback: MagicMock,
    ):
        """Test that callbacks are called in the expected order."""
        # Implementation needed for concrete stream class with callback support
        pass

    @pytest.mark.asyncio
    async def test_callback_receives_correct_component_index(
        self, mock_progress_callback: MagicMock
    ):
        """Test that on_component_start receives correct index and total."""
        # Implementation needed for concrete stream class
        pass

    @pytest.mark.asyncio
    async def test_error_callback_receives_exception(self, mock_progress_callback: MagicMock):
        """Test that on_error callback receives the actual exception."""
        # Implementation needed for concrete stream class
        pass


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestStreamEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_config_with_batch_size_one(self, stream_config_a: StreamConfig):
        """Test configuration with batch_size=1 (sequential processing)."""
        stream_config_a.batch_size = 1
        stream = MockDocumentationStream(stream_config_a)
        assert stream.config.batch_size == 1

    def test_config_with_zero_retries(self, stream_config_a: StreamConfig):
        """Test configuration with max_retries=0 (no retries)."""
        stream_config_a.max_retries = 0
        stream = MockDocumentationStream(stream_config_a)
        assert stream.config.max_retries == 0

    @pytest.mark.asyncio
    async def test_process_component_with_no_callees(
        self,
        stream_config_a: StreamConfig,
        sample_call_graph: CallGraph,
    ):
        """Test processing a component that has no callees."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()

        # Component with no calls
        leaf_component = Component(
            component_id="sample_module.CONSTANT",
            name="CONSTANT",
            type=ComponentType.MODULE,
            location=ComponentLocation(
                file_path="src/sample_module.py",
                line_start=1,
                line_end=1,
            ),
        )
        result = await stream.process_component(
            component=leaf_component,
            source_code="CONSTANT = 42",
            dependency_context={},
            ground_truth=sample_call_graph,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_component_with_circular_dependency_hint(
        self, stream_config_a: StreamConfig
    ):
        """Test handling components with circular dependency hints."""
        # This should not happen in topological order, but test defensive handling
        pass

    @pytest.mark.asyncio
    async def test_reset_during_processing(self, stream_config_a: StreamConfig):
        """Test that reset() can be called safely."""
        stream = MockDocumentationStream(stream_config_a)
        await stream.initialize()
        stream.reset()
        assert len(stream.get_all_component_ids()) == 0

    def test_multiple_streams_independent(
        self, stream_config_a: StreamConfig, stream_config_b: StreamConfig
    ):
        """Test that Stream A and Stream B are independent."""
        stream_a = MockDocumentationStream(stream_config_a)
        stream_b = MockDocumentationStream(stream_config_b)

        assert stream_a.stream_id != stream_b.stream_id
        assert stream_a.output is not stream_b.output
