"""Unit tests for twinscribe.models.call_graph module.

Tests cover:
- CallEdge model validation and operations
- CallGraph model with all graph operations
- CallGraphDiff model and comparison computation
"""

import pytest
from pydantic import ValidationError

from twinscribe.models.base import CallType
from twinscribe.models.call_graph import (
    CallEdge,
    CallGraph,
    CallGraphDiff,
)


class TestCallEdge:
    """Tests for CallEdge model."""

    def test_valid_edge(self):
        """Test creating a valid call edge."""
        edge = CallEdge(
            caller="module.FunctionA",
            callee="module.FunctionB",
            call_site_line=42,
            call_type=CallType.DIRECT,
            confidence=1.0,
        )
        assert edge.caller == "module.FunctionA"
        assert edge.callee == "module.FunctionB"
        assert edge.call_site_line == 42
        assert edge.call_type == CallType.DIRECT
        assert edge.confidence == 1.0

    def test_minimal_edge(self):
        """Test edge with only required fields."""
        edge = CallEdge(caller="a", callee="b")
        assert edge.call_site_line is None
        assert edge.call_type == CallType.DIRECT
        assert edge.confidence == 1.0

    def test_empty_caller_rejected(self):
        """Test that empty caller is rejected."""
        with pytest.raises(ValidationError):
            CallEdge(caller="", callee="b")

    def test_empty_callee_rejected(self):
        """Test that empty callee is rejected."""
        with pytest.raises(ValidationError):
            CallEdge(caller="a", callee="")

    def test_call_site_line_must_be_positive(self):
        """Test that call_site_line must be >= 1 if provided."""
        with pytest.raises(ValidationError):
            CallEdge(caller="a", callee="b", call_site_line=0)

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid bounds
        edge_min = CallEdge(caller="a", callee="b", confidence=0.0)
        edge_max = CallEdge(caller="a", callee="b", confidence=1.0)
        assert edge_min.confidence == 0.0
        assert edge_max.confidence == 1.0

        # Invalid bounds
        with pytest.raises(ValidationError):
            CallEdge(caller="a", callee="b", confidence=-0.1)
        with pytest.raises(ValidationError):
            CallEdge(caller="a", callee="b", confidence=1.1)

    def test_all_call_types(self):
        """Test all call types are valid."""
        for call_type in CallType:
            edge = CallEdge(caller="a", callee="b", call_type=call_type)
            assert edge.call_type == call_type

    def test_hash_based_on_caller_callee(self):
        """Test that hash is based on caller/callee pair."""
        edge1 = CallEdge(caller="a", callee="b", call_site_line=10)
        edge2 = CallEdge(caller="a", callee="b", call_site_line=20)
        edge3 = CallEdge(caller="a", callee="c")

        assert hash(edge1) == hash(edge2)
        assert hash(edge1) != hash(edge3)

    def test_equality_based_on_caller_callee(self):
        """Test equality based on caller/callee pair."""
        edge1 = CallEdge(
            caller="module.func1",
            callee="module.func2",
            call_type=CallType.DIRECT,
        )
        edge2 = CallEdge(
            caller="module.func1",
            callee="module.func2",
            call_type=CallType.CONDITIONAL,  # Different type
        )
        edge3 = CallEdge(caller="module.func1", callee="module.func3")

        assert edge1 == edge2  # Same caller/callee
        assert edge1 != edge3  # Different callee

    def test_to_tuple(self):
        """Test to_tuple method."""
        edge = CallEdge(caller="pkg.A", callee="pkg.B")
        assert edge.to_tuple() == ("pkg.A", "pkg.B")

    def test_json_serialization(self):
        """Test JSON roundtrip."""
        edge = CallEdge(
            caller="module.ClassA.method",
            callee="module.ClassB.helper",
            call_site_line=150,
            call_type=CallType.LOOP,
            confidence=0.85,
        )
        json_str = edge.model_dump_json()
        restored = CallEdge.model_validate_json(json_str)
        assert restored.caller == edge.caller
        assert restored.callee == edge.callee
        assert restored.call_type == edge.call_type


class TestCallGraph:
    """Tests for CallGraph model."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        return CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="a", callee="c"),
                CallEdge(caller="b", callee="d"),
                CallEdge(caller="c", callee="d"),
            ],
            source="test",
        )

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = CallGraph()
        assert graph.edges == []
        assert graph.source == "unknown"
        assert graph.edge_count == 0
        assert graph.node_count == 0

    def test_graph_with_edges(self, sample_graph):
        """Test graph with edges."""
        assert sample_graph.edge_count == 4
        assert sample_graph.node_count == 4  # a, b, c, d

    def test_get_callees(self, sample_graph):
        """Test getting callees of a component."""
        callees = sample_graph.get_callees("a")
        assert len(callees) == 2
        callee_ids = {e.callee for e in callees}
        assert callee_ids == {"b", "c"}

    def test_get_callees_no_results(self, sample_graph):
        """Test getting callees when none exist."""
        callees = sample_graph.get_callees("d")  # d doesn't call anything
        assert callees == []

    def test_get_callers(self, sample_graph):
        """Test getting callers of a component."""
        callers = sample_graph.get_callers("d")
        assert len(callers) == 2
        caller_ids = {e.caller for e in callers}
        assert caller_ids == {"b", "c"}

    def test_get_callers_no_results(self, sample_graph):
        """Test getting callers when none exist."""
        callers = sample_graph.get_callers("a")  # nothing calls a
        assert callers == []

    def test_has_edge(self, sample_graph):
        """Test checking if edge exists."""
        assert sample_graph.has_edge("a", "b") is True
        assert sample_graph.has_edge("b", "a") is False
        assert sample_graph.has_edge("x", "y") is False

    def test_add_edge_new(self, sample_graph):
        """Test adding a new edge."""
        new_edge = CallEdge(caller="d", callee="e")
        result = sample_graph.add_edge(new_edge)
        assert result is True
        assert sample_graph.has_edge("d", "e")
        assert sample_graph.edge_count == 5

    def test_add_edge_duplicate(self, sample_graph):
        """Test adding duplicate edge returns False."""
        dup_edge = CallEdge(caller="a", callee="b")
        result = sample_graph.add_edge(dup_edge)
        assert result is False
        assert sample_graph.edge_count == 4  # Unchanged

    def test_remove_edge(self, sample_graph):
        """Test removing an edge."""
        result = sample_graph.remove_edge("a", "b")
        assert result is True
        assert sample_graph.has_edge("a", "b") is False
        assert sample_graph.edge_count == 3

    def test_remove_edge_not_found(self, sample_graph):
        """Test removing non-existent edge."""
        result = sample_graph.remove_edge("x", "y")
        assert result is False
        assert sample_graph.edge_count == 4  # Unchanged

    def test_all_nodes(self, sample_graph):
        """Test getting all nodes."""
        nodes = sample_graph.all_nodes()
        assert nodes == {"a", "b", "c", "d"}

    def test_to_edge_set(self, sample_graph):
        """Test converting to edge set."""
        edge_set = sample_graph.to_edge_set()
        assert edge_set == {
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("c", "d"),
        }

    def test_iter_edges(self, sample_graph):
        """Test iterating over edges."""
        edges = list(sample_graph.iter_edges())
        assert len(edges) == 4

    def test_merge_with(self):
        """Test merging two graphs."""
        graph1 = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
            source="graph1",
        )
        graph2 = CallGraph(
            edges=[
                CallEdge(caller="b", callee="c"),  # Duplicate
                CallEdge(caller="c", callee="d"),
            ],
            source="graph2",
        )
        merged = graph1.merge_with(graph2)
        assert merged.edge_count == 3  # No duplicates
        assert merged.source == "graph1+graph2"
        assert merged.has_edge("a", "b")
        assert merged.has_edge("b", "c")
        assert merged.has_edge("c", "d")

    def test_json_serialization(self, sample_graph):
        """Test JSON roundtrip."""
        json_str = sample_graph.model_dump_json()
        restored = CallGraph.model_validate_json(json_str)
        assert restored.edge_count == sample_graph.edge_count
        assert restored.source == sample_graph.source


class TestCallGraphDiff:
    """Tests for CallGraphDiff model."""

    def test_perfect_match(self):
        """Test diff when graphs are identical."""
        ground_truth = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ]
        )
        documented = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ]
        )
        diff = CallGraphDiff.compute(ground_truth, documented)

        assert diff.is_perfect_match
        assert diff.precision == 1.0
        assert diff.recall == 1.0
        assert diff.f1_score == 1.0
        assert len(diff.missing_in_doc) == 0
        assert len(diff.extra_in_doc) == 0
        assert len(diff.matching) == 2

    def test_missing_edges(self):
        """Test diff when documented graph is missing edges."""
        ground_truth = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
                CallEdge(caller="c", callee="d"),
            ]
        )
        documented = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
            ]
        )
        diff = CallGraphDiff.compute(ground_truth, documented)

        assert not diff.is_perfect_match
        assert diff.precision == 1.0  # All documented edges are correct
        assert diff.recall == 1/3  # Only 1 of 3 edges found
        assert len(diff.missing_in_doc) == 2
        assert len(diff.extra_in_doc) == 0

    def test_extra_edges(self):
        """Test diff when documented graph has extra edges."""
        ground_truth = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
            ]
        )
        documented = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),  # Extra
                CallEdge(caller="x", callee="y"),  # Extra
            ]
        )
        diff = CallGraphDiff.compute(ground_truth, documented)

        assert not diff.is_perfect_match
        assert diff.precision == 1/3  # Only 1 of 3 documented edges is correct
        assert diff.recall == 1.0  # All ground truth edges found
        assert len(diff.missing_in_doc) == 0
        assert len(diff.extra_in_doc) == 2

    def test_mixed_diff(self):
        """Test diff with both missing and extra edges."""
        ground_truth = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ]
        )
        documented = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),  # Matching
                CallEdge(caller="x", callee="y"),  # Extra
            ]
        )
        diff = CallGraphDiff.compute(ground_truth, documented)

        assert not diff.is_perfect_match
        assert diff.precision == 0.5  # 1 of 2 documented is correct
        assert diff.recall == 0.5  # 1 of 2 ground truth is found
        assert len(diff.missing_in_doc) == 1
        assert len(diff.extra_in_doc) == 1
        assert len(diff.matching) == 1

    def test_empty_graphs(self):
        """Test diff with empty graphs."""
        empty = CallGraph()
        diff = CallGraphDiff.compute(empty, empty)

        assert diff.is_perfect_match
        assert diff.precision == 1.0
        assert diff.recall == 1.0

    def test_empty_documented_vs_non_empty_truth(self):
        """Test diff when documented is empty but truth is not."""
        ground_truth = CallGraph(
            edges=[CallEdge(caller="a", callee="b")]
        )
        documented = CallGraph()

        diff = CallGraphDiff.compute(ground_truth, documented)
        assert not diff.is_perfect_match
        assert diff.precision == 1.0  # No documented edges, so precision is 1
        assert diff.recall == 0.0
        assert len(diff.missing_in_doc) == 1

    def test_f1_score_calculation(self):
        """Test F1 score is computed correctly."""
        diff = CallGraphDiff(
            precision=0.8,
            recall=0.6,
            matching=set(),
            missing_in_doc=set(),
            extra_in_doc=set(),
        )
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(diff.f1_score - expected_f1) < 0.001

    def test_f1_score_zero_division(self):
        """Test F1 score handles zero precision and recall."""
        diff = CallGraphDiff(
            precision=0.0,
            recall=0.0,
            matching=set(),
            missing_in_doc=set(),
            extra_in_doc=set(),
        )
        assert diff.f1_score == 0.0

    def test_json_serialization(self):
        """Test JSON roundtrip with sets."""
        ground_truth = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ]
        )
        documented = CallGraph(
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="x", callee="y"),
            ]
        )
        diff = CallGraphDiff.compute(ground_truth, documented)

        # Sets are serialized as lists in JSON mode
        json_data = diff.model_dump(mode="json")
        assert isinstance(json_data["missing_in_doc"], list)

        # Restore and verify
        restored = CallGraphDiff.model_validate(json_data)
        assert restored.precision == diff.precision
        assert restored.recall == diff.recall
