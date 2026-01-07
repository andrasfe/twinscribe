"""
Tests for the StaticAnalysisOracle.

These tests use mocking to avoid dependency on actual static analysis tools.
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from twinscribe.models.base import CallType
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.static_analysis.oracle import (
    AnalyzerExecutionError,
    AnalyzerNotAvailableError,
    Pyan3Analyzer,
    PyCGAnalyzer,
    StaticAnalysisOracle,
)


class TestCallEdgeAndGraph:
    """Test basic call graph structures."""

    def test_call_edge_creation(self):
        """Test creating a call edge."""
        edge = CallEdge(
            caller="module.func_a",
            callee="module.func_b",
            call_type=CallType.DIRECT,
            confidence=1.0,
        )
        assert edge.caller == "module.func_a"
        assert edge.callee == "module.func_b"

    def test_call_graph_operations(self):
        """Test call graph methods."""
        edges = [
            CallEdge(caller="a", callee="b"),
            CallEdge(caller="a", callee="c"),
            CallEdge(caller="b", callee="c"),
        ]
        graph = CallGraph(source="test", edges=edges)

        assert graph.edge_count == 3
        assert graph.node_count == 3  # a, b, c
        assert graph.has_edge("a", "b")
        assert not graph.has_edge("c", "a")

        callees = graph.get_callees("a")
        assert len(callees) == 2

        callers = graph.get_callers("c")
        assert len(callers) == 2


class TestPyCGAnalyzer:
    """Tests for PyCGAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating a PyCG analyzer."""
        analyzer = PyCGAnalyzer(
            timeout_seconds=60,
            max_iter=3,
        )
        assert analyzer.timeout_seconds == 60
        assert analyzer.max_iter == 3

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test availability check when PyCG is installed."""
        mock_run.return_value = MagicMock(
            stdout="pycg help",
            stderr="",
            returncode=0,
        )
        analyzer = PyCGAnalyzer()
        assert analyzer.is_available() is True

    @patch("subprocess.run")
    def test_is_available_false(self, mock_run):
        """Test availability check when PyCG is not installed."""
        mock_run.side_effect = FileNotFoundError("pycg not found")
        analyzer = PyCGAnalyzer()
        assert analyzer.is_available() is False

    def test_parse_output(self):
        """Test parsing PyCG JSON output."""
        analyzer = PyCGAnalyzer()

        # Create temp file with PyCG-style output
        pycg_output = {
            "module.ClassA.__init__": ["module.helper.setup"],
            "module.ClassA.process": [
                "module.ClassA.__init__",
                "module.helper.validate",
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pycg_output, f)
            f.flush()

            graph = analyzer._parse_output(f.name)

        assert graph.source == "pycg"
        assert graph.edge_count == 3

        # Check specific edges exist
        assert graph.has_edge("module.ClassA.__init__", "module.helper.setup")
        assert graph.has_edge("module.ClassA.process", "module.helper.validate")


class TestPyan3Analyzer:
    """Tests for Pyan3Analyzer."""

    def test_analyzer_creation(self):
        """Test creating a pyan3 analyzer."""
        analyzer = Pyan3Analyzer(timeout_seconds=120)
        assert analyzer.timeout_seconds == 120

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test availability check when pyan3 is installed."""
        mock_run.return_value = MagicMock(
            stdout="pyan3 help",
            stderr="",
            returncode=0,
        )
        analyzer = Pyan3Analyzer()
        assert analyzer.is_available() is True

    def test_parse_output(self):
        """Test parsing pyan3 JSON output."""
        analyzer = Pyan3Analyzer()

        pyan_output = json.dumps(
            {
                "graph": [
                    {"source": "module.func_a", "target": "module.func_b", "flavor": "uses"},
                    {"source": "module.func_b", "target": "module.func_c", "flavor": "uses"},
                    {"source": "module.Class", "target": "module.func_a", "flavor": "defines"},
                ]
            }
        )

        graph = analyzer._parse_output(pyan_output)

        assert graph.source == "pyan3"
        # Only "uses" edges should be included
        assert graph.edge_count == 2
        assert graph.has_edge("module.func_a", "module.func_b")


@pytest.mark.static_analysis
class TestStaticAnalysisOracle:
    """Tests for StaticAnalysisOracle."""

    def test_oracle_creation(self, tmp_path):
        """Test creating an oracle."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        assert oracle.codebase_path == str(tmp_path)

    def test_oracle_language_normalization(self, tmp_path):
        """Test that language strings are normalized."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="PYTHON",
        )
        assert oracle.language.value == "python"

    @patch.object(PyCGAnalyzer, "is_available", return_value=True)
    @patch.object(PyCGAnalyzer, "analyze")
    def test_oracle_uses_primary_analyzer(self, mock_analyze, mock_available, tmp_path):
        """Test that oracle uses primary analyzer when available."""
        expected_graph = CallGraph(
            source="pycg",
            edges=[CallEdge(caller="a", callee="b")],
        )
        mock_analyze.return_value = expected_graph

        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        graph = oracle.call_graph

        assert graph.source == "pycg"
        mock_analyze.assert_called_once()

    @patch.object(PyCGAnalyzer, "is_available", return_value=False)
    @patch.object(Pyan3Analyzer, "is_available", return_value=True)
    @patch.object(Pyan3Analyzer, "analyze")
    def test_oracle_falls_back_to_secondary(
        self, mock_analyze, mock_pyan_available, mock_pycg_available, tmp_path
    ):
        """Test that oracle falls back when primary is not available."""
        expected_graph = CallGraph(
            source="pyan3",
            edges=[CallEdge(caller="a", callee="b")],
        )
        mock_analyze.return_value = expected_graph

        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        graph = oracle.call_graph

        assert graph.source == "pyan3"
        mock_analyze.assert_called_once()

    @patch.object(PyCGAnalyzer, "is_available", return_value=False)
    @patch.object(Pyan3Analyzer, "is_available", return_value=False)
    def test_oracle_returns_empty_when_no_analyzer(self, mock_pyan, mock_pycg, tmp_path):
        """Test that oracle returns empty graph when no analyzer is available."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        graph = oracle.call_graph

        assert graph.source == "none"
        assert graph.edge_count == 0

    def test_get_callees(self, tmp_path):
        """Test getting callees for a component."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        # Inject a mock call graph
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="module.A", callee="module.B"),
                CallEdge(caller="module.A", callee="module.C"),
                CallEdge(caller="module.B", callee="module.C"),
            ],
        )

        callees = oracle.get_callees("module.A")
        assert len(callees) == 2

        callee_ids = {e.callee for e in callees}
        assert "module.B" in callee_ids
        assert "module.C" in callee_ids

    def test_get_callers(self, tmp_path):
        """Test getting callers for a component."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="module.A", callee="module.C"),
                CallEdge(caller="module.B", callee="module.C"),
            ],
        )

        callers = oracle.get_callers("module.C")
        assert len(callers) == 2

    def test_verify_edge(self, tmp_path):
        """Test edge verification."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        oracle._call_graph = CallGraph(
            source="test",
            edges=[CallEdge(caller="a", callee="b")],
        )

        assert oracle.verify_edge("a", "b") is True
        assert oracle.verify_edge("b", "a") is False
        assert oracle.verify_edge("x", "y") is False

    def test_diff_against(self, tmp_path):
        """Test comparing documented graph against ground truth."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        # Ground truth
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="a", callee="c"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        # Documented graph (missing one edge, has one extra)
        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
                CallEdge(caller="x", callee="y"),  # Extra (false positive)
            ],
        )

        diff = oracle.diff_against(documented)

        # Missing: a->c
        assert ("a", "c") in diff.missing_in_doc
        # Extra: x->y
        assert ("x", "y") in diff.extra_in_doc
        # Matching: a->b, b->c
        assert len(diff.matching) == 2

        # Precision: 2 correct / 3 documented = 0.67
        assert 0.6 < diff.precision < 0.7
        # Recall: 2 found / 3 in truth = 0.67
        assert 0.6 < diff.recall < 0.7

    def test_get_summary(self, tmp_path):
        """Test getting summary statistics."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        oracle._call_graph = CallGraph(
            source="pycg",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        summary = oracle.get_summary()

        assert summary["source"] == "pycg"
        assert summary["edge_count"] == 2
        assert summary["node_count"] == 3
        assert summary["language"] == "python"

    @patch.object(PyCGAnalyzer, "is_available", return_value=True)
    @patch.object(PyCGAnalyzer, "analyze")
    def test_refresh_clears_cache(self, mock_analyze, mock_available, tmp_path):
        """Test that refresh clears the cached call graph."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        # First call
        mock_analyze.return_value = CallGraph(
            source="pycg",
            edges=[CallEdge(caller="a", callee="b")],
        )
        _ = oracle.call_graph
        assert mock_analyze.call_count == 1

        # Refresh
        mock_analyze.return_value = CallGraph(
            source="pycg",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="c", callee="d"),
            ],
        )
        new_graph = oracle.refresh()

        assert mock_analyze.call_count == 2
        assert new_graph.edge_count == 2


class TestAnalyzerErrors:
    """Test analyzer error handling."""

    def test_analyzer_not_available_error(self):
        """Test AnalyzerNotAvailableError."""
        error = AnalyzerNotAvailableError("PyCG not installed")
        assert "PyCG not installed" in str(error)

    def test_analyzer_execution_error(self):
        """Test AnalyzerExecutionError."""
        error = AnalyzerExecutionError("Analysis failed")
        assert "Analysis failed" in str(error)

    @patch("subprocess.run")
    def test_timeout_raises_execution_error(self, mock_run):
        """Test that timeout raises AnalyzerExecutionError."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

        analyzer = PyCGAnalyzer(timeout_seconds=10)

        with pytest.raises(AnalyzerExecutionError) as exc_info:
            analyzer._run_command(["pycg", "test.py"])

        assert "timed out" in str(exc_info.value)


@pytest.mark.static_analysis
class TestStaticAnalysisOracleExtended:
    """Extended tests for StaticAnalysisOracle covering edge cases."""

    def test_oracle_with_invalid_path(self):
        """Test that oracle can be created with non-existent path.

        The path validation happens at analysis time, not creation time.
        """
        oracle = StaticAnalysisOracle(
            codebase_path="/nonexistent/path",
            language="python",
        )
        assert oracle.codebase_path == "/nonexistent/path"

    @patch.object(PyCGAnalyzer, "is_available", return_value=True)
    @patch.object(PyCGAnalyzer, "analyze")
    def test_analyzer_raises_on_invalid_path(self, mock_analyze, mock_available):
        """Test that analyzer raises error for non-existent path."""
        mock_analyze.side_effect = AnalyzerExecutionError("Codebase path not found: /nonexistent")

        oracle = StaticAnalysisOracle(
            codebase_path="/nonexistent",
            language="python",
        )

        # When both primary and fallback fail, returns empty graph
        with patch.object(Pyan3Analyzer, "is_available", return_value=False):
            graph = oracle.call_graph
            assert graph.edge_count == 0

    def test_get_callees_for_nonexistent_component(self, tmp_path):
        """Test get_callees returns empty list for non-existent component."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[CallEdge(caller="a", callee="b")],
        )

        callees = oracle.get_callees("nonexistent")
        assert callees == []

    def test_get_callees_for_leaf_component(self, tmp_path):
        """Test get_callees returns empty for component that calls nothing."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        # 'c' is a leaf - it doesn't call anything
        callees = oracle.get_callees("c")
        assert callees == []

    def test_get_callers_for_nonexistent_component(self, tmp_path):
        """Test get_callers returns empty list for non-existent component."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[CallEdge(caller="a", callee="b")],
        )

        callers = oracle.get_callers("nonexistent")
        assert callers == []

    def test_get_callers_for_entry_point(self, tmp_path):
        """Test get_callers returns empty for entry point (nothing calls it)."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="main", callee="process"),
                CallEdge(caller="process", callee="helper"),
            ],
        )

        # 'main' is an entry point - nothing calls it
        callers = oracle.get_callers("main")
        assert callers == []

    def test_diff_against_identical_graphs(self, tmp_path):
        """Test diff with identical graphs produces perfect match."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        diff = oracle.diff_against(documented)

        assert diff.is_perfect_match
        assert diff.precision == 1.0
        assert diff.recall == 1.0
        assert diff.f1_score == 1.0
        assert len(diff.missing_in_doc) == 0
        assert len(diff.extra_in_doc) == 0

    def test_diff_against_empty_documented_graph(self, tmp_path):
        """Test diff when documented graph is empty."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="c"),
            ],
        )

        documented = CallGraph(source="agent", edges=[])

        diff = oracle.diff_against(documented)

        assert not diff.is_perfect_match
        assert diff.precision == 1.0  # No false positives
        assert diff.recall == 0.0  # Found nothing
        assert len(diff.missing_in_doc) == 2

    def test_diff_against_extra_edges_only(self, tmp_path):
        """Test diff when documented graph has only extra edges."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[CallEdge(caller="a", callee="b")],
        )

        documented = CallGraph(
            source="agent",
            edges=[
                CallEdge(caller="a", callee="b"),  # Correct
                CallEdge(caller="x", callee="y"),  # Extra
                CallEdge(caller="p", callee="q"),  # Extra
            ],
        )

        diff = oracle.diff_against(documented)

        assert diff.recall == 1.0  # All ground truth found
        assert diff.precision == 1 / 3  # 1 correct of 3 documented
        assert len(diff.extra_in_doc) == 2

    @patch.object(PyCGAnalyzer, "is_available", return_value=True)
    @patch.object(PyCGAnalyzer, "analyze")
    def test_call_graph_caching(self, mock_analyze, mock_available, tmp_path):
        """Test that call_graph is cached after first access."""
        mock_analyze.return_value = CallGraph(
            source="pycg",
            edges=[CallEdge(caller="a", callee="b")],
        )

        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        # First access
        graph1 = oracle.call_graph
        assert mock_analyze.call_count == 1

        # Second access - should use cache
        graph2 = oracle.call_graph
        assert mock_analyze.call_count == 1

        # Verify same instance
        assert graph1 is graph2

    def test_oracle_with_empty_codebase(self, tmp_path):
        """Test oracle behavior with empty directory (no Python files)."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        oracle = StaticAnalysisOracle(
            codebase_path=str(empty_dir),
            language="python",
        )

        # Mock analyzer to return empty graph for empty codebase
        with (
            patch.object(PyCGAnalyzer, "is_available", return_value=True),
            patch.object(PyCGAnalyzer, "analyze") as mock_analyze,
        ):
            mock_analyze.return_value = CallGraph(source="pycg", edges=[])
            graph = oracle.call_graph
            assert graph.edge_count == 0

    def test_oracle_language_unknown_falls_back_to_python(self, tmp_path):
        """Test that unknown language falls back to Python analyzers."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="unknown",
        )
        # Should fall back to Python (default)
        assert oracle.language.value == "python"

    @patch.object(PyCGAnalyzer, "is_available", return_value=True)
    @patch.object(PyCGAnalyzer, "analyze")
    def test_oracle_with_analyzer_config(self, mock_analyze, mock_available, tmp_path):
        """Test oracle initialization with custom analyzer config."""
        mock_analyze.return_value = CallGraph(
            source="pycg",
            edges=[CallEdge(caller="a", callee="b")],
        )

        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
            analyzer_config={
                "primary": {"timeout_seconds": 600, "max_iter": 10},
            },
        )

        # Verify config was passed - the analyzer should be configured
        assert oracle._primary_analyzer.timeout_seconds == 600
        assert oracle._primary_analyzer.max_iter == 10

    def test_verify_edge_bidirectional(self, tmp_path):
        """Test verify_edge correctly distinguishes edge direction."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                CallEdge(caller="a", callee="b"),
                CallEdge(caller="b", callee="a"),  # Also has reverse
            ],
        )

        assert oracle.verify_edge("a", "b") is True
        assert oracle.verify_edge("b", "a") is True

    def test_oracle_with_complex_dependency_chain(self, tmp_path):
        """Test oracle with complex dependency relationships."""
        oracle = StaticAnalysisOracle(
            codebase_path=str(tmp_path),
            language="python",
        )

        # Create a complex graph with:
        # - Multiple entry points
        # - Diamond dependency pattern
        # - Recursive call
        # - Shared utility
        oracle._call_graph = CallGraph(
            source="test",
            edges=[
                # Entry points
                CallEdge(caller="main", callee="process"),
                CallEdge(caller="cli", callee="process"),
                # Diamond pattern: process -> [validate, transform] -> save
                CallEdge(caller="process", callee="validate"),
                CallEdge(caller="process", callee="transform"),
                CallEdge(caller="validate", callee="save"),
                CallEdge(caller="transform", callee="save"),
                # Recursive call
                CallEdge(caller="traverse", callee="traverse"),
                # Shared utility
                CallEdge(caller="validate", callee="log"),
                CallEdge(caller="transform", callee="log"),
                CallEdge(caller="save", callee="log"),
            ],
        )

        # Test entry points have no callers
        assert oracle.get_callers("main") == []
        assert oracle.get_callers("cli") == []

        # Test leaf node 'log' has many callers
        log_callers = oracle.get_callers("log")
        assert len(log_callers) == 3

        # Test diamond pattern - save has 2 callers
        save_callers = oracle.get_callers("save")
        caller_ids = {e.caller for e in save_callers}
        assert caller_ids == {"validate", "transform"}

        # Test recursive call
        traverse_callees = oracle.get_callees("traverse")
        assert len(traverse_callees) == 1
        assert traverse_callees[0].callee == "traverse"


@pytest.mark.static_analysis
class TestPyCGAnalyzerAnalysis:
    """Tests for PyCGAnalyzer analyze method."""

    @patch.object(PyCGAnalyzer, "_run_command")
    def test_analyze_with_python_files(self, mock_run, tmp_path):
        """Test analyze method with Python files present."""
        # Create Python files
        py_file = tmp_path / "module.py"
        py_file.write_text("def func(): pass")

        # Create mock output file
        output_data = {
            "module.func": ["module.helper"],
        }
        output_file = tmp_path / "output.json"
        output_file.write_text(json.dumps(output_data))

        mock_run.return_value = ("", "", 0)

        analyzer = PyCGAnalyzer()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = str(output_file)
            with patch.object(analyzer, "_parse_output") as mock_parse:
                mock_parse.return_value = CallGraph(
                    source="pycg",
                    edges=[CallEdge(caller="module.func", callee="module.helper")],
                )
                graph = analyzer.analyze(str(tmp_path))

                assert graph.source == "pycg"
                mock_run.assert_called_once()

    def test_analyze_with_nonexistent_path(self):
        """Test analyze raises error for non-existent path."""
        analyzer = PyCGAnalyzer()
        with pytest.raises(AnalyzerExecutionError) as exc_info:
            analyzer.analyze("/nonexistent/path")
        assert "not found" in str(exc_info.value)

    def test_analyze_with_no_python_files(self, tmp_path):
        """Test analyze returns empty graph when no Python files."""
        # Create directory with no Python files
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Not a Python file")

        analyzer = PyCGAnalyzer()
        graph = analyzer.analyze(str(tmp_path))

        assert graph.source == "pycg"
        assert graph.edge_count == 0

    @patch.object(PyCGAnalyzer, "_run_command")
    def test_analyze_with_non_zero_return(self, mock_run, tmp_path):
        """Test analyze handles non-zero return code."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def func(): pass")

        # Create valid output file despite non-zero return
        output_data = {"module.func": []}
        output_file = tmp_path / "output.json"
        output_file.write_text(json.dumps(output_data))

        mock_run.return_value = ("", "warning message", 1)

        analyzer = PyCGAnalyzer()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = str(output_file)
            graph = analyzer.analyze(str(tmp_path))
            # Should still try to parse output
            assert isinstance(graph, CallGraph)


@pytest.mark.static_analysis
class TestPyan3AnalyzerAnalysis:
    """Tests for Pyan3Analyzer analyze method."""

    def test_analyze_with_nonexistent_path(self):
        """Test analyze raises error for non-existent path."""
        analyzer = Pyan3Analyzer()
        with pytest.raises(AnalyzerExecutionError) as exc_info:
            analyzer.analyze("/nonexistent/path")
        assert "not found" in str(exc_info.value)

    def test_analyze_with_no_python_files(self, tmp_path):
        """Test analyze returns empty graph when no Python files."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Not a Python file")

        analyzer = Pyan3Analyzer()
        graph = analyzer.analyze(str(tmp_path))

        assert graph.source == "pyan3"
        assert graph.edge_count == 0

    @patch.object(Pyan3Analyzer, "_run_command")
    def test_analyze_with_python_files(self, mock_run, tmp_path):
        """Test analyze method with Python files."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def func(): pass")

        pyan_output = json.dumps(
            {
                "graph": [
                    {"source": "module.func", "target": "module.helper", "flavor": "uses"},
                ]
            }
        )
        mock_run.return_value = (pyan_output, "", 0)

        analyzer = Pyan3Analyzer()
        graph = analyzer.analyze(str(tmp_path))

        assert graph.source == "pyan3"
        assert graph.edge_count == 1

    @patch.object(Pyan3Analyzer, "_run_command")
    def test_analyze_with_non_zero_return(self, mock_run, tmp_path):
        """Test analyze handles non-zero return code."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def func(): pass")

        mock_run.return_value = ("", "warning message", 1)

        analyzer = Pyan3Analyzer()
        graph = analyzer.analyze(str(tmp_path))

        # Should return empty graph on parse failure
        assert graph.source == "pyan3"
        assert graph.edge_count == 0


@pytest.mark.static_analysis
class TestBaseAnalyzerRunCommand:
    """Tests for BaseAnalyzer _run_command method."""

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test _run_command with successful execution."""
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )

        analyzer = PyCGAnalyzer()
        stdout, stderr, code = analyzer._run_command(["echo", "test"])

        assert stdout == "output"
        assert code == 0

    @patch("subprocess.run")
    def test_run_command_with_cwd(self, mock_run):
        """Test _run_command with working directory."""
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )

        analyzer = PyCGAnalyzer()
        analyzer._run_command(["echo", "test"], cwd="/tmp")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["cwd"] == "/tmp"

    @patch("subprocess.run")
    def test_run_command_generic_exception(self, mock_run):
        """Test _run_command handles generic exceptions."""
        mock_run.side_effect = RuntimeError("Something went wrong")

        analyzer = PyCGAnalyzer()
        with pytest.raises(AnalyzerExecutionError) as exc_info:
            analyzer._run_command(["cmd"])

        assert "Command failed" in str(exc_info.value)


@pytest.mark.static_analysis
class TestPyCGAnalyzerExtended:
    """Extended tests for PyCG analyzer edge cases."""

    def test_parse_output_empty_file(self):
        """Test parsing empty output file."""
        analyzer = PyCGAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            f.flush()
            graph = analyzer._parse_output(f.name)

        assert graph.source == "pycg"
        assert graph.edge_count == 0

    def test_parse_output_invalid_json(self):
        """Test parsing invalid JSON returns empty graph."""
        analyzer = PyCGAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()
            graph = analyzer._parse_output(f.name)

        assert graph.source == "pycg"
        assert graph.edge_count == 0

    def test_parse_output_missing_file(self):
        """Test parsing non-existent file returns empty graph."""
        analyzer = PyCGAnalyzer()
        graph = analyzer._parse_output("/nonexistent/file.json")
        assert graph.source == "pycg"
        assert graph.edge_count == 0


@pytest.mark.static_analysis
class TestPyan3AnalyzerExtended:
    """Extended tests for pyan3 analyzer edge cases."""

    def test_parse_output_empty_string(self):
        """Test parsing empty string output."""
        analyzer = Pyan3Analyzer()
        graph = analyzer._parse_output("")
        assert graph.source == "pyan3"
        assert graph.edge_count == 0

    def test_parse_output_whitespace_only(self):
        """Test parsing whitespace-only output."""
        analyzer = Pyan3Analyzer()
        graph = analyzer._parse_output("   \n\t  ")
        assert graph.source == "pyan3"
        assert graph.edge_count == 0

    def test_parse_output_invalid_json(self):
        """Test parsing invalid JSON returns empty graph."""
        analyzer = Pyan3Analyzer()
        graph = analyzer._parse_output("invalid json")
        assert graph.source == "pyan3"
        assert graph.edge_count == 0

    def test_parse_output_missing_graph_key(self):
        """Test parsing JSON without 'graph' key."""
        analyzer = Pyan3Analyzer()
        graph = analyzer._parse_output('{"other": "data"}')
        assert graph.source == "pyan3"
        assert graph.edge_count == 0

    def test_parse_output_filters_non_uses_edges(self):
        """Test that only 'uses' flavor edges are included."""
        analyzer = Pyan3Analyzer()

        pyan_output = json.dumps(
            {
                "graph": [
                    {"source": "a", "target": "b", "flavor": "uses"},
                    {"source": "c", "target": "d", "flavor": "defines"},
                    {"source": "e", "target": "f", "flavor": "uses"},
                    {"source": "g", "target": "h", "flavor": "imports"},
                ]
            }
        )

        graph = analyzer._parse_output(pyan_output)

        assert graph.edge_count == 2
        assert graph.has_edge("a", "b")
        assert graph.has_edge("e", "f")
        assert not graph.has_edge("c", "d")
        assert not graph.has_edge("g", "h")
