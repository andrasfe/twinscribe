"""
Static Analysis Oracle.

Provides ground truth call graph extraction from static analysis tools.
This is the authoritative source for call graph validation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

from twinscribe.config.models import Language
from twinscribe.models.base import CallType
from twinscribe.models.call_graph import CallEdge, CallGraph, CallGraphDiff

logger = logging.getLogger(__name__)


class AnalyzerError(Exception):
    """Base exception for analyzer errors."""

    pass


class AnalyzerNotAvailableError(AnalyzerError):
    """Raised when an analyzer tool is not installed."""

    pass


class AnalyzerExecutionError(AnalyzerError):
    """Raised when analyzer execution fails."""

    pass


class BaseAnalyzer(ABC):
    """Base class for call graph analyzers.

    All language-specific analyzers inherit from this class
    and implement the analyze method.
    """

    def __init__(
        self,
        executable: str | None = None,
        timeout_seconds: int = 300,
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            executable: Path to the analyzer executable
            timeout_seconds: Execution timeout in seconds
            extra_args: Additional command-line arguments
        """
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.extra_args = extra_args or []

    @abstractmethod
    def analyze(self, codebase_path: str) -> CallGraph:
        """Analyze the codebase and extract call graph.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            CallGraph with all discovered edges
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this analyzer is available (tool installed).

        Returns:
            True if the analyzer can be used
        """
        pass

    def _run_command(
        self,
        command: list[str],
        cwd: str | None = None,
    ) -> tuple[str, str, int]:
        """Run a command and capture output.

        Args:
            command: Command and arguments
            cwd: Working directory

        Returns:
            Tuple of (stdout, stderr, return_code)

        Raises:
            AnalyzerExecutionError: If command fails
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=cwd,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            raise AnalyzerExecutionError(
                f"Command timed out after {self.timeout_seconds}s: {' '.join(command)}"
            )
        except FileNotFoundError:
            raise AnalyzerNotAvailableError(f"Command not found: {command[0]}")
        except Exception as e:
            raise AnalyzerExecutionError(f"Command failed: {e}")


class PyCGAnalyzer(BaseAnalyzer):
    """Python call graph analyzer using PyCG.

    PyCG is a practical call graph generator for Python that handles
    dynamic features better than AST-based approaches.
    """

    def __init__(
        self,
        executable: str | None = None,
        timeout_seconds: int = 300,
        extra_args: list[str] | None = None,
        max_iter: int = 5,
    ) -> None:
        """Initialize PyCG analyzer.

        Args:
            executable: Path to pycg executable (default: 'pycg')
            timeout_seconds: Execution timeout
            extra_args: Additional arguments
            max_iter: Maximum iterations for analysis
        """
        super().__init__(executable or "pycg", timeout_seconds, extra_args)
        self.max_iter = max_iter

    def is_available(self) -> bool:
        """Check if PyCG is installed."""
        try:
            stdout, _, code = self._run_command(["pycg", "--help"])
            return code == 0
        except (AnalyzerNotAvailableError, AnalyzerExecutionError):
            return False

    def analyze(self, codebase_path: str) -> CallGraph:
        """Analyze Python codebase with PyCG.

        Args:
            codebase_path: Path to Python codebase

        Returns:
            CallGraph with discovered edges
        """
        path = Path(codebase_path)
        if not path.exists():
            raise AnalyzerExecutionError(f"Codebase path not found: {codebase_path}")

        # Find all Python files
        python_files = list(path.rglob("*.py"))
        if not python_files:
            logger.warning(f"No Python files found in {codebase_path}")
            return CallGraph(source="pycg", edges=[])

        # Create output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
            output_path = output_file.name

        try:
            # Build command
            command = [
                self.executable,
                "--max-iter",
                str(self.max_iter),
                "-o",
                output_path,
            ]
            command.extend(self.extra_args)

            # Add package path
            command.append(str(path))

            logger.debug(f"Running PyCG: {' '.join(command)}")
            stdout, stderr, code = self._run_command(command)

            if code != 0:
                logger.warning(f"PyCG returned non-zero: {stderr}")
                # Try to parse output anyway

            # Parse output
            return self._parse_output(output_path)
        finally:
            # Clean up
            try:
                Path(output_path).unlink()
            except Exception:
                pass

    def _parse_output(self, output_path: str) -> CallGraph:
        """Parse PyCG JSON output.

        Args:
            output_path: Path to PyCG output file

        Returns:
            CallGraph
        """
        try:
            with open(output_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to parse PyCG output: {e}")
            return CallGraph(source="pycg", edges=[])

        edges = []
        for caller, callees in data.items():
            for callee in callees:
                edge = CallEdge(
                    caller=caller,
                    callee=callee,
                    call_type=CallType.DIRECT,
                    confidence=1.0,  # Static analysis is authoritative
                )
                edges.append(edge)

        return CallGraph(source="pycg", edges=edges)


class Pyan3Analyzer(BaseAnalyzer):
    """Python call graph analyzer using pyan3.

    Pyan3 is an AST-based call graph generator, used as fallback
    when PyCG is not available.
    """

    def __init__(
        self,
        executable: str | None = None,
        timeout_seconds: int = 300,
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize pyan3 analyzer.

        Args:
            executable: Path to pyan executable (default: 'pyan3')
            timeout_seconds: Execution timeout
            extra_args: Additional arguments
        """
        super().__init__(executable or "pyan3", timeout_seconds, extra_args)

    def is_available(self) -> bool:
        """Check if pyan3 is installed."""
        try:
            stdout, _, code = self._run_command(["pyan3", "--help"])
            return code == 0
        except (AnalyzerNotAvailableError, AnalyzerExecutionError):
            return False

    def analyze(self, codebase_path: str) -> CallGraph:
        """Analyze Python codebase with pyan3.

        Args:
            codebase_path: Path to Python codebase

        Returns:
            CallGraph with discovered edges
        """
        path = Path(codebase_path)
        if not path.exists():
            raise AnalyzerExecutionError(f"Codebase path not found: {codebase_path}")

        # Find all Python files
        python_files = [str(f) for f in path.rglob("*.py")]
        if not python_files:
            logger.warning(f"No Python files found in {codebase_path}")
            return CallGraph(source="pyan3", edges=[])

        # Create output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
            output_path = output_file.name

        try:
            # Build command - pyan3 outputs to stdout
            command = [
                self.executable,
                "--uses",
                "--no-defines",
                "--format=json",
            ]
            command.extend(self.extra_args)
            command.extend(python_files)

            logger.debug(f"Running pyan3: {' '.join(command[:5])}...")
            stdout, stderr, code = self._run_command(command, cwd=str(path))

            if code != 0:
                logger.warning(f"pyan3 returned non-zero: {stderr}")

            # Parse stdout as JSON
            return self._parse_output(stdout)
        finally:
            try:
                Path(output_path).unlink()
            except Exception:
                pass

    def _parse_output(self, json_output: str) -> CallGraph:
        """Parse pyan3 JSON output.

        Args:
            json_output: JSON string from pyan3

        Returns:
            CallGraph
        """
        if not json_output.strip():
            return CallGraph(source="pyan3", edges=[])

        try:
            data = json.loads(json_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pyan3 output: {e}")
            return CallGraph(source="pyan3", edges=[])

        edges = []
        # pyan3 outputs a list of edges
        for edge_data in data.get("graph", []):
            if edge_data.get("flavor") == "uses":
                edge = CallEdge(
                    caller=edge_data.get("source", ""),
                    callee=edge_data.get("target", ""),
                    call_type=CallType.DIRECT,
                    confidence=1.0,
                )
                edges.append(edge)

        return CallGraph(source="pyan3", edges=edges)


class StaticAnalysisOracle:
    """Oracle for ground truth call graph from static analysis.

    This class provides the authoritative call graph extracted using
    static analysis tools. It serves as the source of truth for
    validating LLM-generated call relationships.

    Attributes:
        codebase_path: Path to the codebase root
        language: Primary programming language
        primary_analyzer: Primary analyzer for the language
        fallback_analyzer: Fallback analyzer if primary fails
    """

    # Analyzer registry
    _analyzers: dict[Language, tuple[type[BaseAnalyzer], type[BaseAnalyzer] | None]] = {
        Language.PYTHON: (PyCGAnalyzer, Pyan3Analyzer),
    }

    def __init__(
        self,
        codebase_path: str,
        language: str | Language = Language.PYTHON,
        analyzer_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the oracle.

        Args:
            codebase_path: Path to the codebase root
            language: Primary programming language
            analyzer_config: Optional analyzer configuration
        """
        self.codebase_path = codebase_path

        # Normalize language
        if isinstance(language, str):
            try:
                self.language = Language(language.lower())
            except ValueError:
                self.language = Language.PYTHON
        else:
            self.language = language

        self._config = analyzer_config or {}
        self._call_graph: CallGraph | None = None

        # Initialize analyzers
        self._primary_analyzer: BaseAnalyzer | None = None
        self._fallback_analyzer: BaseAnalyzer | None = None
        self._init_analyzers()

    def _init_analyzers(self) -> None:
        """Initialize analyzers based on language."""
        if self.language not in self._analyzers:
            logger.warning(f"No analyzer registered for {self.language}, using Python")
            analyzer_classes = self._analyzers[Language.PYTHON]
        else:
            analyzer_classes = self._analyzers[self.language]

        primary_class, fallback_class = analyzer_classes

        # Initialize primary analyzer
        primary_config = self._config.get("primary", {})
        self._primary_analyzer = primary_class(**primary_config)

        # Initialize fallback if available
        if fallback_class:
            fallback_config = self._config.get("fallback", {})
            self._fallback_analyzer = fallback_class(**fallback_config)

    @cached_property
    def call_graph(self) -> CallGraph:
        """Extract and cache the ground truth call graph.

        Returns:
            CallGraph from static analysis

        Raises:
            AnalyzerError: If all analyzers fail
        """
        if self._call_graph is not None:
            return self._call_graph

        # Try primary analyzer
        if self._primary_analyzer and self._primary_analyzer.is_available():
            try:
                logger.info(f"Analyzing with {self._primary_analyzer.__class__.__name__}")
                self._call_graph = self._primary_analyzer.analyze(self.codebase_path)
                logger.info(
                    f"Extracted {self._call_graph.edge_count} edges "
                    f"from {self._call_graph.node_count} nodes"
                )
                return self._call_graph
            except AnalyzerError as e:
                logger.warning(f"Primary analyzer failed: {e}")

        # Try fallback analyzer
        if self._fallback_analyzer and self._fallback_analyzer.is_available():
            try:
                logger.info(f"Falling back to {self._fallback_analyzer.__class__.__name__}")
                self._call_graph = self._fallback_analyzer.analyze(self.codebase_path)
                logger.info(
                    f"Extracted {self._call_graph.edge_count} edges "
                    f"from {self._call_graph.node_count} nodes"
                )
                return self._call_graph
            except AnalyzerError as e:
                logger.warning(f"Fallback analyzer failed: {e}")

        # No analyzer available
        logger.error("No static analyzer available")
        self._call_graph = CallGraph(source="none", edges=[])
        return self._call_graph

    def get_callees(self, component_id: str) -> list[CallEdge]:
        """Get all functions/methods called by a component.

        Args:
            component_id: The caller component ID

        Returns:
            List of edges where component_id is the caller
        """
        return self.call_graph.get_callees(component_id)

    def get_callers(self, component_id: str) -> list[CallEdge]:
        """Get all functions/methods that call a component.

        Args:
            component_id: The callee component ID

        Returns:
            List of edges where component_id is the callee
        """
        return self.call_graph.get_callers(component_id)

    def verify_edge(self, caller: str, callee: str) -> bool:
        """Verify if a call relationship exists.

        Args:
            caller: Caller component ID
            callee: Callee component ID

        Returns:
            True if the edge exists in ground truth
        """
        return self.call_graph.has_edge(caller, callee)

    def diff_against(self, documented_graph: CallGraph) -> CallGraphDiff:
        """Compare documented call graph against ground truth.

        Args:
            documented_graph: The call graph from documentation agents

        Returns:
            CallGraphDiff with precision, recall, and edge differences
        """
        return CallGraphDiff.compute(self.call_graph, documented_graph)

    def refresh(self) -> CallGraph:
        """Force refresh of the call graph cache.

        Returns:
            Newly extracted CallGraph
        """
        # Clear cached property
        if "call_graph" in self.__dict__:
            del self.__dict__["call_graph"]
        self._call_graph = None

        return self.call_graph

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the call graph.

        Returns:
            Dict with edge count, node count, and analyzer used
        """
        graph = self.call_graph
        return {
            "source": graph.source,
            "edge_count": graph.edge_count,
            "node_count": graph.node_count,
            "language": self.language.value,
            "codebase_path": self.codebase_path,
        }
