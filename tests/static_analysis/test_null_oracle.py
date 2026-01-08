"""
Tests for the NullStaticAnalysisOracle.

The NullOracle is used when static analysis is disabled or unavailable,
allowing the pipeline to run solely on dual-stream consensus.
"""

import pytest

from twinscribe.analysis.null_oracle import (
    NullStaticAnalysisOracle,
    create_null_oracle,
)
from twinscribe.models.call_graph import CallEdge, CallGraph


class TestNullStaticAnalysisOracle:
    """Tests for NullStaticAnalysisOracle."""

    def test_oracle_creation(self, tmp_path):
        """Test creating a null oracle."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        assert oracle.codebase_path == tmp_path

    def test_oracle_is_always_initialized(self, tmp_path):
        """Test that null oracle is always considered initialized."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        assert oracle.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_noop(self, tmp_path):
        """Test that initialize is a no-op."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        await oracle.initialize()
        assert oracle.is_initialized is True

    def test_call_graph_is_empty(self, tmp_path):
        """Test that call graph is always empty."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        graph = oracle.call_graph
        assert graph is not None
        assert graph.edge_count == 0
        assert graph.source == "none"

    @pytest.mark.asyncio
    async def test_get_call_graph_returns_empty(self, tmp_path):
        """Test that get_call_graph returns empty graph."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        await oracle.initialize()

        graph = await oracle.get_call_graph()
        assert graph.edge_count == 0
        assert graph.source == "none"

    @pytest.mark.asyncio
    async def test_get_call_graph_with_force_refresh(self, tmp_path):
        """Test that force_refresh parameter is ignored."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        await oracle.initialize()

        graph = await oracle.get_call_graph(force_refresh=True)
        assert graph.edge_count == 0

    def test_get_callees_returns_empty(self, tmp_path):
        """Test that get_callees always returns empty list."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        callees = oracle.get_callees("any.component")
        assert callees == []

    def test_get_callers_returns_empty(self, tmp_path):
        """Test that get_callers always returns empty list."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        callers = oracle.get_callers("any.component")
        assert callers == []

    def test_verify_edge_returns_false(self, tmp_path):
        """Test that verify_edge always returns False."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        assert oracle.verify_edge("a", "b") is False
        assert oracle.verify_edge("any", "edge") is False

    def test_all_nodes_returns_empty(self, tmp_path):
        """Test that all_nodes returns empty set."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        nodes = oracle.all_nodes()
        assert nodes == set()

    def test_diff_against_documented_graph(self, tmp_path):
        """Test diff_against with a documented graph.

        When using null oracle, all documented edges are "extra"
        since there's no ground truth.
        """
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        diff = oracle.diff_against(documented)

        # All documented edges are "extra" (not in ground truth)
        assert len(diff.extra_in_doc) == 2
        assert len(diff.missing_in_doc) == 0
        assert len(diff.matching) == 0

    @pytest.mark.asyncio
    async def test_refresh_returns_empty_graph(self, tmp_path):
        """Test that refresh returns empty graph."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        await oracle.initialize()

        graph = await oracle.refresh()
        assert graph.edge_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_is_noop(self, tmp_path):
        """Test that shutdown is a no-op."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))
        await oracle.initialize()

        # Should not raise
        await oracle.shutdown()

    def test_stats_tracking(self, tmp_path):
        """Test that stats are tracked for queries."""
        oracle = NullStaticAnalysisOracle(codebase_path=str(tmp_path))

        initial_queries = oracle.stats.total_queries

        oracle.get_callees("a")
        oracle.get_callers("b")
        oracle.verify_edge("c", "d")

        assert oracle.stats.total_queries == initial_queries + 3


class TestCreateNullOracle:
    """Tests for the create_null_oracle factory function."""

    def test_creates_null_oracle(self, tmp_path):
        """Test factory creates NullStaticAnalysisOracle."""
        oracle = create_null_oracle(str(tmp_path))
        assert isinstance(oracle, NullStaticAnalysisOracle)

    def test_with_string_path(self, tmp_path):
        """Test factory works with string path."""
        oracle = create_null_oracle(str(tmp_path))
        assert oracle.codebase_path == tmp_path

    def test_with_path_object(self, tmp_path):
        """Test factory works with Path object."""
        oracle = create_null_oracle(tmp_path)
        assert oracle.codebase_path == tmp_path


class TestNullOracleWithNonexistentPath:
    """Test NullOracle behavior with non-existent paths.

    The NullOracle should work fine with any path since it doesn't
    actually analyze anything.
    """

    def test_accepts_nonexistent_path(self):
        """Test that null oracle accepts non-existent path."""
        oracle = NullStaticAnalysisOracle(codebase_path="/nonexistent/path")
        assert oracle.is_initialized is True

    def test_call_graph_with_nonexistent_path(self):
        """Test call graph returns empty with non-existent path."""
        oracle = NullStaticAnalysisOracle(codebase_path="/nonexistent/path")
        graph = oracle.call_graph
        assert graph.edge_count == 0

    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_path(self):
        """Test initialize works with non-existent path."""
        oracle = NullStaticAnalysisOracle(codebase_path="/nonexistent/path")
        # Should not raise
        await oracle.initialize()
        assert oracle.is_initialized is True
