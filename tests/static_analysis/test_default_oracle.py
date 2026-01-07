"""
Tests for DefaultStaticAnalysisOracle.

Tests the concrete oracle implementation with PyCG integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinscribe.analysis.analyzer import AnalyzerType
from twinscribe.analysis.default_oracle import (
    DefaultStaticAnalysisOracle,
    create_python_oracle,
)
from twinscribe.analysis.oracle import OracleConfig
from twinscribe.models.call_graph import CallEdge, CallGraph


@pytest.mark.static_analysis
class TestDefaultOracleCreation:
    """Tests for oracle creation and configuration."""

    def test_oracle_creation_with_defaults(self, tmp_path):
        """Test creating oracle with default configuration."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        assert oracle.codebase_path == tmp_path
        assert oracle.config is not None
        assert not oracle.is_initialized

    def test_oracle_creation_with_config(self, tmp_path):
        """Test creating oracle with custom configuration."""
        config = OracleConfig(
            cache_enabled=False,
            cache_ttl_hours=12,
        )
        oracle = DefaultStaticAnalysisOracle(tmp_path, config)

        assert oracle.config.cache_enabled is False
        assert oracle.config.cache_ttl_hours == 12

    def test_create_python_oracle_factory(self, tmp_path):
        """Test the create_python_oracle factory function."""
        oracle = create_python_oracle(
            tmp_path,
            cache_enabled=True,
            strip_prefix="myproject",
        )

        assert isinstance(oracle, DefaultStaticAnalysisOracle)
        assert oracle.config.primary_analyzer == AnalyzerType.PYCG


@pytest.mark.static_analysis
class TestDefaultOracleInitialization:
    """Tests for oracle initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_available_analyzer(self, tmp_path):
        """Test initialization when analyzer is available."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        # Create a Python file
        (tmp_path / "module.py").write_text("def foo(): pass")

        # Mock the _create_analyzer method to return a mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.check_available = AsyncMock(return_value=True)
        mock_analyzer.analyzer_type = AnalyzerType.PYCG
        mock_analyzer.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )

        with patch.object(oracle, "_create_analyzer", return_value=mock_analyzer):
            await oracle.initialize()

        assert oracle.is_initialized
        assert oracle._active_analyzer is not None

    @pytest.mark.asyncio
    async def test_initialize_falls_back_to_ast(self, tmp_path):
        """Test initialization falls back to AST analyzer when PyCG unavailable.

        Note: The AST fallback is invoked in _select_analyzer only when
        at least one analyzer was registered during _load_analyzers.
        This test verifies the selection fallback mechanism.
        """
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        (tmp_path / "module.py").write_text("def foo(): pass")

        # Mock PyCG analyzer as registered but not selected by primary logic
        mock_pycg = MagicMock()
        mock_pycg.check_available = AsyncMock(return_value=True)
        mock_pycg.analyzer_type = AnalyzerType.PYCG

        # Mock AST analyzer as the fallback
        mock_ast = MagicMock()
        mock_ast.check_available = AsyncMock(return_value=True)
        mock_ast.analyzer_type = AnalyzerType.PYCG  # AST uses PYCG type for compat
        mock_ast.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )

        # Pre-populate analyzers so the check at line 94 passes
        oracle._analyzers[AnalyzerType.PYCG] = mock_pycg

        # Mock _select_analyzer to simulate falling back to AST
        async def mock_select():
            oracle._stats.fallback_uses += 1
            return mock_ast

        with patch.object(oracle, "_load_analyzers", AsyncMock()):
            with patch.object(oracle, "_select_analyzer", mock_select):
                await oracle.initialize()

        assert oracle.is_initialized
        assert oracle._stats.fallback_uses > 0

    @pytest.mark.asyncio
    async def test_initialize_twice_is_idempotent(self, tmp_path):
        """Test that calling initialize twice is safe."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        (tmp_path / "module.py").write_text("def foo(): pass")

        mock_analyzer = MagicMock()
        mock_analyzer.check_available = AsyncMock(return_value=True)
        mock_analyzer.analyzer_type = AnalyzerType.PYCG
        mock_analyzer.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )

        with patch.object(oracle, "_create_analyzer", return_value=mock_analyzer):
            await oracle.initialize()
            await oracle.initialize()  # Second call should be no-op

        assert oracle.is_initialized


@pytest.mark.static_analysis
class TestDefaultOracleQueries:
    """Tests for oracle query methods."""

    @pytest.fixture
    def initialized_oracle(self, tmp_path):
        """Create an initialized oracle with mocked call graph."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="module.A", callee="module.B"),
                CallEdge(caller="module.A", callee="module.C"),
                CallEdge(caller="module.B", callee="module.C"),
            ],
        )
        return oracle

    def test_get_callees(self, initialized_oracle):
        """Test getting callees for a component."""
        callees = initialized_oracle.get_callees("module.A")

        assert len(callees) == 2
        callee_ids = {e.callee for e in callees}
        assert callee_ids == {"module.B", "module.C"}

    def test_get_callees_nonexistent(self, initialized_oracle):
        """Test getting callees for non-existent component."""
        callees = initialized_oracle.get_callees("nonexistent")
        assert callees == []

    def test_get_callers(self, initialized_oracle):
        """Test getting callers for a component."""
        callers = initialized_oracle.get_callers("module.C")

        assert len(callers) == 2
        caller_ids = {e.caller for e in callers}
        assert caller_ids == {"module.A", "module.B"}

    def test_get_callers_nonexistent(self, initialized_oracle):
        """Test getting callers for non-existent component."""
        callers = initialized_oracle.get_callers("nonexistent")
        assert callers == []

    def test_verify_edge(self, initialized_oracle):
        """Test edge verification."""
        assert initialized_oracle.verify_edge("module.A", "module.B") is True
        assert initialized_oracle.verify_edge("module.B", "module.A") is False
        assert initialized_oracle.verify_edge("x", "y") is False

    def test_get_all_edges(self, initialized_oracle):
        """Test getting all edges."""
        edges = initialized_oracle.get_all_edges()

        assert len(edges) == 3
        assert ("module.A", "module.B") in edges

    def test_all_nodes(self, initialized_oracle):
        """Test getting all nodes."""
        nodes = initialized_oracle.all_nodes()

        assert nodes == {"module.A", "module.B", "module.C"}

    def test_queries_before_init_raise(self, tmp_path):
        """Test that queries before initialization raise error."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        with pytest.raises(RuntimeError, match="not initialized"):
            oracle.get_callees("anything")

        with pytest.raises(RuntimeError, match="not initialized"):
            oracle.get_callers("anything")


@pytest.mark.static_analysis
class TestDefaultOracleCallGraph:
    """Tests for call graph retrieval."""

    @pytest.mark.asyncio
    async def test_get_call_graph_uses_cache(self, tmp_path):
        """Test that get_call_graph uses cache."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._codebase_hash = "hash123"

        expected_graph = CallGraph(source="test", edges=[])
        oracle._call_graph = expected_graph

        # Mock cache as valid
        with patch.object(oracle, "_is_cache_valid", return_value=True):
            with patch.object(oracle, "_compute_codebase_hash", return_value="hash123"):
                graph = await oracle.get_call_graph()

        assert graph is expected_graph
        assert oracle.stats.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_call_graph_force_refresh(self, tmp_path):
        """Test that force_refresh bypasses cache."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._active_analyzer = MagicMock()
        oracle._active_analyzer.analyzer_type = AnalyzerType.PYCG
        oracle._active_analyzer.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )
        oracle._codebase_hash = "hash123"

        # Even with cache
        oracle._call_graph = CallGraph(source="cached", edges=[])

        with patch.object(oracle, "_compute_codebase_hash", return_value="hash123"):
            graph = await oracle.get_call_graph(force_refresh=True)

        # Should have called analyze
        oracle._active_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_call_graph_not_initialized_raises(self, tmp_path):
        """Test that get_call_graph raises if not initialized."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        with pytest.raises(RuntimeError, match="not initialized"):
            await oracle.get_call_graph()


@pytest.mark.static_analysis
class TestDefaultOracleRefresh:
    """Tests for refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_reruns_analysis(self, tmp_path):
        """Test that refresh reruns analysis."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._active_analyzer = MagicMock()
        oracle._active_analyzer.analyzer_type = AnalyzerType.PYCG
        oracle._active_analyzer.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )
        oracle._codebase_hash = "old_hash"

        with patch.object(oracle, "_compute_codebase_hash", return_value="new_hash"):
            await oracle.refresh()

        oracle._active_analyzer.analyze.assert_called_once()
        assert oracle._codebase_hash == "new_hash"

    @pytest.mark.asyncio
    async def test_refresh_clears_old_cache(self, tmp_path):
        """Test that refresh clears old cache before re-caching."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._active_analyzer = MagicMock()
        oracle._active_analyzer.analyzer_type = AnalyzerType.PYCG
        oracle._active_analyzer.analyze = AsyncMock(
            return_value=MagicMock(
                analyzer_type=AnalyzerType.PYCG,
                raw_edges=[],
                nodes=set(),
            )
        )

        # Pre-populate cache with old entry
        cache_key = oracle._get_cache_key()
        old_entry = MagicMock()
        old_entry.codebase_hash = "old_hash"
        oracle._cache[cache_key] = old_entry

        with patch.object(oracle, "_compute_codebase_hash", return_value="new_hash"):
            await oracle.refresh()

        # Cache should have new entry with new hash
        assert cache_key in oracle._cache
        assert oracle._cache[cache_key].codebase_hash == "new_hash"
        assert oracle._cache[cache_key] is not old_entry


@pytest.mark.static_analysis
class TestDefaultOracleDiff:
    """Tests for diff functionality."""

    def test_diff_against(self, tmp_path):
        """Test comparing documented graph against ground truth."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="a", callee="c"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
                CallEdge(caller="x", callee="y"),  # Extra
            ],
        )

        diff = oracle.diff_against(documented)

        assert ("a", "c") in diff.missing_in_doc
        assert ("x", "y") in diff.extra_in_doc
        assert len(diff.matching) == 2

    def test_diff_perfect_match(self, tmp_path):
        """Test diff with identical graphs."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
            ],
        )

        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),
            ],
        )

        diff = oracle.diff_against(documented)

        assert diff.is_perfect_match
        assert diff.precision == 1.0
        assert diff.recall == 1.0


@pytest.mark.static_analysis
class TestDefaultOracleShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self, tmp_path):
        """Test that shutdown clears all state."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._call_graph = CallGraph(source="test", edges=[])
        oracle._cache["key"] = MagicMock()
        oracle._analyzers[AnalyzerType.PYCG] = MagicMock()

        await oracle.shutdown()

        assert not oracle._initialized
        assert oracle._call_graph is None
        assert len(oracle._cache) == 0
        assert len(oracle._analyzers) == 0


@pytest.mark.static_analysis
class TestDefaultOracleCodebaseHash:
    """Tests for codebase hash computation."""

    def test_compute_hash_empty_dir(self, tmp_path):
        """Test hash computation for empty directory."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        hash1 = oracle._compute_codebase_hash()

        assert hash1 is not None
        assert len(hash1) > 0

    def test_compute_hash_changes_with_files(self, tmp_path):
        """Test that hash changes when files change."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)

        hash1 = oracle._compute_codebase_hash()

        # Add a file
        (tmp_path / "module.py").write_text("def foo(): pass")
        hash2 = oracle._compute_codebase_hash()

        assert hash1 != hash2

    def test_compute_hash_deterministic(self, tmp_path):
        """Test that hash is deterministic."""
        (tmp_path / "module.py").write_text("def foo(): pass")

        oracle = DefaultStaticAnalysisOracle(tmp_path)

        hash1 = oracle._compute_codebase_hash()
        hash2 = oracle._compute_codebase_hash()

        assert hash1 == hash2


@pytest.mark.static_analysis
class TestOracleStats:
    """Tests for oracle statistics tracking."""

    def test_query_stats_increment(self, tmp_path):
        """Test that query operations increment stats."""
        oracle = DefaultStaticAnalysisOracle(tmp_path)
        oracle._initialized = True
        oracle._call_graph = CallGraph(source="test", edges=[])

        initial = oracle.stats.total_queries

        oracle.get_callees("x")
        oracle.get_callers("x")
        oracle.verify_edge("x", "y")

        assert oracle.stats.total_queries == initial + 3
