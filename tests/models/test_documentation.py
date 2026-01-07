"""Unit tests for twinscribe.models.documentation module.

Tests cover:
- CalleeRef and CallerRef models
- CallGraphSection model
- DocumenterMetadata model
- DocumentationOutput model
- StreamOutput model
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from twinscribe.models.base import CallType, StreamId
from twinscribe.models.components import ComponentDocumentation, ParameterDoc
from twinscribe.models.documentation import (
    CalleeRef,
    CallerRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
    StreamOutput,
)


class TestCalleeRef:
    """Tests for CalleeRef model."""

    def test_valid_callee_ref(self):
        """Test creating a valid callee reference."""
        ref = CalleeRef(
            component_id="pkg.module.function",
            call_site_line=42,
            call_type=CallType.DIRECT,
        )
        assert ref.component_id == "pkg.module.function"
        assert ref.call_site_line == 42
        assert ref.call_type == CallType.DIRECT

    def test_minimal_callee_ref(self):
        """Test callee ref with defaults."""
        ref = CalleeRef(component_id="func")
        assert ref.call_site_line is None
        assert ref.call_type == CallType.DIRECT

    def test_empty_component_id_rejected(self):
        """Test that empty component_id is rejected."""
        with pytest.raises(ValidationError):
            CalleeRef(component_id="")

    def test_call_site_must_be_positive(self):
        """Test that call_site_line must be >= 1."""
        with pytest.raises(ValidationError):
            CalleeRef(component_id="func", call_site_line=0)


class TestCallerRef:
    """Tests for CallerRef model."""

    def test_valid_caller_ref(self):
        """Test creating a valid caller reference."""
        ref = CallerRef(
            component_id="pkg.Class.method",
            call_site_line=100,
            call_type=CallType.LOOP,
        )
        assert ref.component_id == "pkg.Class.method"
        assert ref.call_site_line == 100
        assert ref.call_type == CallType.LOOP

    def test_minimal_caller_ref(self):
        """Test caller ref with defaults."""
        ref = CallerRef(component_id="main")
        assert ref.call_site_line is None
        assert ref.call_type == CallType.DIRECT


class TestCallGraphSection:
    """Tests for CallGraphSection model."""

    def test_empty_section(self):
        """Test creating an empty call graph section."""
        section = CallGraphSection()
        assert section.callers == []
        assert section.callees == []
        assert section.total_edges == 0

    def test_section_with_edges(self):
        """Test section with callers and callees."""
        section = CallGraphSection(
            callers=[
                CallerRef(component_id="caller1"),
                CallerRef(component_id="caller2"),
            ],
            callees=[
                CalleeRef(component_id="callee1"),
            ],
        )
        assert len(section.callers) == 2
        assert len(section.callees) == 1
        assert section.total_edges == 3


class TestDocumenterMetadata:
    """Tests for DocumenterMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        meta = DocumenterMetadata(
            agent_id="A1",
            stream_id=StreamId.STREAM_A,
            model="claude-sonnet-4-5-20250929",
            confidence=0.92,
            processing_order=5,
            token_count=1500,
        )
        assert meta.agent_id == "A1"
        assert meta.stream_id == StreamId.STREAM_A
        assert meta.model == "claude-sonnet-4-5-20250929"
        assert meta.confidence == 0.92
        assert meta.processing_order == 5
        assert meta.token_count == 1500

    def test_timestamp_defaults_to_now(self):
        """Test that timestamp defaults to current time."""
        before = datetime.utcnow()
        meta = DocumenterMetadata(
            agent_id="B1",
            stream_id=StreamId.STREAM_B,
            model="gpt-4o",
        )
        after = datetime.utcnow()
        assert before <= meta.timestamp <= after

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid
        meta = DocumenterMetadata(
            agent_id="A1",
            stream_id=StreamId.STREAM_A,
            model="test",
            confidence=0.0,
        )
        assert meta.confidence == 0.0

        meta = DocumenterMetadata(
            agent_id="A1",
            stream_id=StreamId.STREAM_A,
            model="test",
            confidence=1.0,
        )
        assert meta.confidence == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            DocumenterMetadata(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model="test",
                confidence=1.5,
            )

    def test_processing_order_must_be_non_negative(self):
        """Test processing_order >= 0."""
        with pytest.raises(ValidationError):
            DocumenterMetadata(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model="test",
                processing_order=-1,
            )


class TestDocumentationOutput:
    """Tests for DocumentationOutput model."""

    @pytest.fixture
    def sample_output(self):
        """Create a sample documentation output."""
        return DocumentationOutput(
            component_id="pkg.module.MyClass.process",
            documentation=ComponentDocumentation(
                summary="Process the input data.",
                description="Detailed description.",
                parameters=[ParameterDoc(name="data", type="dict")],
            ),
            call_graph=CallGraphSection(
                callers=[CallerRef(component_id="pkg.main")],
                callees=[
                    CalleeRef(component_id="pkg.utils.validate"),
                    CalleeRef(component_id="pkg.utils.format"),
                ],
            ),
            metadata=DocumenterMetadata(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model="claude-sonnet-4-5",
                confidence=0.95,
            ),
        )

    def test_valid_output(self, sample_output):
        """Test creating valid documentation output."""
        assert sample_output.component_id == "pkg.module.MyClass.process"
        assert sample_output.documentation.summary == "Process the input data."
        assert len(sample_output.call_graph.callees) == 2

    def test_empty_component_id_rejected(self):
        """Test that empty component_id is rejected."""
        with pytest.raises(ValidationError):
            DocumentationOutput(
                component_id="",
                documentation=ComponentDocumentation(),
                metadata=DocumenterMetadata(
                    agent_id="A1",
                    stream_id=StreamId.STREAM_A,
                    model="test",
                ),
            )

    def test_whitespace_component_id_rejected(self):
        """Test that whitespace-only component_id is rejected."""
        with pytest.raises(ValidationError):
            DocumentationOutput(
                component_id="   ",
                documentation=ComponentDocumentation(),
                metadata=DocumenterMetadata(
                    agent_id="A1",
                    stream_id=StreamId.STREAM_A,
                    model="test",
                ),
            )

    def test_get_callee_ids(self, sample_output):
        """Test extracting callee IDs."""
        callee_ids = sample_output.get_callee_ids()
        assert callee_ids == ["pkg.utils.validate", "pkg.utils.format"]

    def test_get_caller_ids(self, sample_output):
        """Test extracting caller IDs."""
        caller_ids = sample_output.get_caller_ids()
        assert caller_ids == ["pkg.main"]

    def test_default_call_graph(self):
        """Test that call_graph defaults to empty section."""
        output = DocumentationOutput(
            component_id="test.func",
            documentation=ComponentDocumentation(),
            metadata=DocumenterMetadata(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model="test",
            ),
        )
        assert output.call_graph.callers == []
        assert output.call_graph.callees == []

    def test_json_serialization(self, sample_output):
        """Test JSON roundtrip."""
        json_str = sample_output.model_dump_json()
        restored = DocumentationOutput.model_validate_json(json_str)
        assert restored.component_id == sample_output.component_id
        assert restored.documentation.summary == sample_output.documentation.summary
        assert len(restored.call_graph.callees) == len(sample_output.call_graph.callees)


class TestStreamOutput:
    """Tests for StreamOutput model."""

    @pytest.fixture
    def sample_doc_output(self):
        """Create a sample documentation output."""
        return DocumentationOutput(
            component_id="test.func",
            documentation=ComponentDocumentation(summary="Test function"),
            metadata=DocumenterMetadata(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model="test",
                token_count=100,
            ),
        )

    def test_empty_stream_output(self):
        """Test creating empty stream output."""
        stream = StreamOutput(stream_id=StreamId.STREAM_A)
        assert stream.outputs == {}
        assert stream.total_components == 0
        assert stream.total_tokens == 0
        assert stream.processing_time_seconds == 0.0

    def test_add_output(self, sample_doc_output):
        """Test adding documentation output."""
        stream = StreamOutput(stream_id=StreamId.STREAM_A)
        stream.add_output(sample_doc_output)

        assert stream.total_components == 1
        assert stream.total_tokens == 100
        assert "test.func" in stream.outputs

    def test_get_output(self, sample_doc_output):
        """Test getting output by component ID."""
        stream = StreamOutput(stream_id=StreamId.STREAM_A)
        stream.add_output(sample_doc_output)

        result = stream.get_output("test.func")
        assert result is not None
        assert result.component_id == "test.func"

        # Non-existent
        assert stream.get_output("nonexistent") is None

    def test_multiple_outputs(self):
        """Test stream with multiple outputs."""
        stream = StreamOutput(stream_id=StreamId.STREAM_B)

        for i in range(5):
            output = DocumentationOutput(
                component_id=f"test.func{i}",
                documentation=ComponentDocumentation(summary=f"Function {i}"),
                metadata=DocumenterMetadata(
                    agent_id="B1",
                    stream_id=StreamId.STREAM_B,
                    model="test",
                    token_count=50,
                ),
            )
            stream.add_output(output)

        assert stream.total_components == 5
        assert stream.total_tokens == 250

    def test_json_serialization(self, sample_doc_output):
        """Test JSON roundtrip."""
        stream = StreamOutput(stream_id=StreamId.STREAM_A)
        stream.add_output(sample_doc_output)

        json_str = stream.model_dump_json()
        restored = StreamOutput.model_validate_json(json_str)

        assert restored.stream_id == StreamId.STREAM_A
        assert restored.total_components == 1
