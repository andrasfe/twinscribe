"""
Null Static Analysis Oracle.

Provides a no-op implementation of StaticAnalysisOracle for use when
static analysis is disabled or unavailable. This allows the pipeline
to run without static analysis, relying solely on dual-stream consensus.
"""

import logging
from pathlib import Path

from twinscribe.analysis.oracle import (
    OracleConfig,
    StaticAnalysisOracle,
)
from twinscribe.models.call_graph import CallEdge, CallGraph, CallGraphDiff

logger = logging.getLogger(__name__)


class NullStaticAnalysisOracle(StaticAnalysisOracle):
    """Null implementation of StaticAnalysisOracle.

    Used when static analysis is skipped (e.g., for legacy languages
    without static analyzers, or when explicitly disabled).

    The null oracle:
    - Returns an empty call graph
    - Always reports as initialized
    - Provides no-op implementations for all methods

    This allows the dual-stream consensus to be the sole source of truth
    for call graph generation.

    Usage:
        oracle = NullStaticAnalysisOracle(codebase_path="/path/to/project")
        await oracle.initialize()  # No-op

        # Returns empty graph
        call_graph = await oracle.get_call_graph()
        assert call_graph.edge_count == 0
    """

    def __init__(
        self,
        codebase_path: str | Path,
        config: OracleConfig | None = None,
    ) -> None:
        """Initialize the null oracle.

        Args:
            codebase_path: Path to the codebase (stored for reference only)
            config: Oracle configuration (optional, mostly ignored)
        """
        super().__init__(codebase_path, config)
        # Pre-initialize with empty call graph
        self._call_graph = CallGraph(edges=[], source="none")

    @property
    def is_initialized(self) -> bool:
        """Check if oracle is initialized.

        The null oracle is always considered initialized.
        """
        return True

    async def initialize(self) -> None:
        """Initialize the oracle.

        For NullOracle, this is a no-op since there is no analyzer
        to initialize.
        """
        logger.info(
            "NullOracle initialized - static analysis disabled. "
            "Dual-stream consensus will be the sole source of truth."
        )
        self._initialized = True

    async def get_call_graph(self, force_refresh: bool = False) -> CallGraph:
        """Get the call graph.

        Always returns an empty call graph since no static analysis
        is performed.

        Args:
            force_refresh: Ignored (no analysis to refresh)

        Returns:
            Empty CallGraph
        """
        return self._call_graph

    def get_callees(self, component_id: str) -> list[CallEdge]:
        """Get callees for a component.

        Always returns empty list since no static analysis data exists.

        Args:
            component_id: Component to query

        Returns:
            Empty list
        """
        self._stats.total_queries += 1
        return []

    def get_callers(self, component_id: str) -> list[CallEdge]:
        """Get callers for a component.

        Always returns empty list since no static analysis data exists.

        Args:
            component_id: Component to query

        Returns:
            Empty list
        """
        self._stats.total_queries += 1
        return []

    def verify_edge(self, caller: str, callee: str) -> bool:
        """Verify if an edge exists.

        Always returns False since no static analysis data exists.

        Args:
            caller: Caller component ID
            callee: Callee component ID

        Returns:
            False (no edges in null oracle)
        """
        self._stats.total_queries += 1
        return False

    def diff_against(self, documented_graph: CallGraph) -> CallGraphDiff:
        """Compare against documented call graph.

        When using null oracle, documented graph has no ground truth
        to compare against. Returns a diff showing all documented edges
        as "extra" (not in ground truth).

        Args:
            documented_graph: Call graph from documentation agents

        Returns:
            Diff with all documented edges as "extra"
        """
        self._stats.total_queries += 1
        return CallGraphDiff.compute(self._call_graph, documented_graph)

    def all_nodes(self) -> set[str]:
        """Get all nodes in the call graph.

        Returns:
            Empty set (no nodes in null oracle)
        """
        return set()

    async def refresh(self) -> CallGraph:
        """Refresh the call graph.

        No-op for null oracle.

        Returns:
            Empty CallGraph
        """
        return self._call_graph

    async def shutdown(self) -> None:
        """Clean up resources.

        No-op for null oracle.
        """
        logger.info("NullOracle shutdown")


def create_null_oracle(codebase_path: str | Path) -> NullStaticAnalysisOracle:
    """Create a null oracle for when static analysis is disabled.

    Factory function for creating a null oracle with default configuration.

    Args:
        codebase_path: Path to codebase (stored for reference)

    Returns:
        NullStaticAnalysisOracle instance

    Usage:
        oracle = create_null_oracle("/path/to/cobol/project")
        await oracle.initialize()
        # Pipeline runs without static analysis
    """
    return NullStaticAnalysisOracle(codebase_path)
