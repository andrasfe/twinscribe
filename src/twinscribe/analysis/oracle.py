"""
Static Analysis Oracle.

Provides the ground truth call graph for validation. The oracle manages
analyzer selection, fallback logic, and caching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerError,
    AnalyzerType,
    Language,
)
from twinscribe.analysis.normalizer import (
    CallGraphNormalizer,
    NormalizationConfig,
)
from twinscribe.models.call_graph import CallEdge, CallGraph, CallGraphDiff


class OracleConfig(BaseModel):
    """Configuration for the static analysis oracle.

    Attributes:
        language: Primary language of the codebase
        primary_analyzer: Primary analyzer type to use
        fallback_analyzers: Ordered list of fallback analyzers
        cache_enabled: Whether to cache analysis results
        cache_ttl_hours: Cache time-to-live in hours
        normalization: Normalization configuration
        auto_select_analyzer: Auto-select best analyzer for language
    """

    language: Language = Field(
        default=Language.PYTHON,
        description="Primary codebase language",
    )
    primary_analyzer: Optional[AnalyzerType] = Field(
        default=None,
        description="Primary analyzer (auto-selected if None)",
    )
    fallback_analyzers: list[AnalyzerType] = Field(
        default_factory=list,
        description="Fallback analyzers in order",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable result caching",
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=0,
        description="Cache TTL in hours (0 = no expiry)",
    )
    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description="Normalization settings",
    )
    auto_select_analyzer: bool = Field(
        default=True,
        description="Auto-select analyzer for language",
    )


# Default analyzer mappings by language
DEFAULT_ANALYZERS: dict[Language, tuple[AnalyzerType, list[AnalyzerType]]] = {
    Language.PYTHON: (
        AnalyzerType.PYCG,
        [AnalyzerType.PYAN3, AnalyzerType.SOURCETRAIL],
    ),
    Language.JAVA: (
        AnalyzerType.JAVA_CALLGRAPH,
        [AnalyzerType.WALA, AnalyzerType.SOURCETRAIL],
    ),
    Language.JAVASCRIPT: (
        AnalyzerType.TS_CALLGRAPH,
        [AnalyzerType.SOURCETRAIL],
    ),
    Language.TYPESCRIPT: (
        AnalyzerType.TS_CALLGRAPH,
        [AnalyzerType.SOURCETRAIL],
    ),
    Language.MULTI: (
        AnalyzerType.SOURCETRAIL,
        [],
    ),
}


@dataclass
class CacheEntry:
    """Cache entry for analysis results.

    Attributes:
        call_graph: Cached call graph
        analyzer_type: Analyzer that produced the result
        timestamp: When the result was cached
        codebase_hash: Hash of codebase for invalidation
    """

    call_graph: CallGraph
    analyzer_type: AnalyzerType
    timestamp: datetime
    codebase_hash: str


@dataclass
class OracleStats:
    """Statistics from oracle operations.

    Attributes:
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        primary_successes: Successful primary analyzer runs
        fallback_uses: Number of fallback analyzer uses
        total_analyses: Total analysis runs
        total_queries: Total query operations
    """

    cache_hits: int = 0
    cache_misses: int = 0
    primary_successes: int = 0
    fallback_uses: int = 0
    total_analyses: int = 0
    total_queries: int = 0


class StaticAnalysisOracle(ABC):
    """Abstract interface for the static analysis oracle.

    The oracle provides ground truth call graph information extracted
    from static analysis. It manages analyzer selection, fallback logic,
    and result caching.

    Usage:
        oracle = ConcreteOracle(codebase_path, config)
        await oracle.initialize()

        # Get full call graph
        call_graph = await oracle.get_call_graph()

        # Query specific relationships
        callees = oracle.get_callees("module.Class.method")
        callers = oracle.get_callers("module.Class.method")

        # Verify edge
        exists = oracle.verify_edge("caller.func", "callee.func")

        # Compare against documented graph
        diff = oracle.diff_against(documented_graph)
    """

    def __init__(
        self,
        codebase_path: str | Path,
        config: Optional[OracleConfig] = None,
    ) -> None:
        """Initialize the oracle.

        Args:
            codebase_path: Path to the codebase to analyze
            config: Oracle configuration
        """
        self._codebase_path = Path(codebase_path)
        self._config = config or OracleConfig()
        self._call_graph: Optional[CallGraph] = None
        self._cache: dict[str, CacheEntry] = {}
        self._stats = OracleStats()
        self._normalizer = CallGraphNormalizer(self._config.normalization)
        self._initialized = False

        # Auto-select analyzers if configured
        if self._config.auto_select_analyzer and not self._config.primary_analyzer:
            self._auto_select_analyzers()

    def _auto_select_analyzers(self) -> None:
        """Auto-select primary and fallback analyzers based on language."""
        lang = self._config.language
        if lang in DEFAULT_ANALYZERS:
            primary, fallbacks = DEFAULT_ANALYZERS[lang]
            # Use object.__setattr__ since Pydantic models may be frozen
            self._config = self._config.model_copy(
                update={
                    "primary_analyzer": primary,
                    "fallback_analyzers": fallbacks,
                }
            )

    @property
    def codebase_path(self) -> Path:
        """Get codebase path."""
        return self._codebase_path

    @property
    def config(self) -> OracleConfig:
        """Get oracle configuration."""
        return self._config

    @property
    def stats(self) -> OracleStats:
        """Get oracle statistics."""
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if oracle is initialized."""
        return self._initialized

    @property
    def call_graph(self) -> Optional[CallGraph]:
        """Get cached call graph (may be None if not yet analyzed)."""
        return self._call_graph

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the oracle and run initial analysis.

        Checks analyzer availability and runs the initial analysis
        to populate the call graph.

        Raises:
            RuntimeError: If no analyzer is available
            AnalyzerError: If analysis fails
        """
        pass

    @abstractmethod
    async def get_call_graph(self, force_refresh: bool = False) -> CallGraph:
        """Get the complete call graph.

        Returns cached result unless force_refresh is True or cache
        is expired/invalid.

        Args:
            force_refresh: Force re-analysis ignoring cache

        Returns:
            Complete normalized call graph

        Raises:
            RuntimeError: If oracle not initialized
            AnalyzerError: If analysis fails
        """
        pass

    def get_callees(self, component_id: str) -> list[CallEdge]:
        """Get all functions/methods called by a component.

        Args:
            component_id: Component to query

        Returns:
            List of edges where component_id is the caller

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        self._stats.total_queries += 1
        return self._call_graph.get_callees(component_id)

    def get_callers(self, component_id: str) -> list[CallEdge]:
        """Get all functions/methods that call a component.

        Args:
            component_id: Component to query

        Returns:
            List of edges where component_id is the callee

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        self._stats.total_queries += 1
        return self._call_graph.get_callers(component_id)

    def verify_edge(self, caller: str, callee: str) -> bool:
        """Verify if a call relationship exists.

        Args:
            caller: Caller component ID
            callee: Callee component ID

        Returns:
            True if the edge exists in ground truth

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        self._stats.total_queries += 1
        return self._call_graph.has_edge(caller, callee)

    def diff_against(self, documented_graph: CallGraph) -> CallGraphDiff:
        """Compare documented call graph against ground truth.

        Args:
            documented_graph: Call graph from documentation agents

        Returns:
            Diff showing missing, extra, and matching edges

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        self._stats.total_queries += 1
        return CallGraphDiff.compute(self._call_graph, documented_graph)

    def all_nodes(self) -> set[str]:
        """Get all unique component IDs in the call graph.

        Returns:
            Set of all component IDs

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        return self._call_graph.all_nodes()

    @abstractmethod
    async def refresh(self) -> CallGraph:
        """Force refresh the call graph from source.

        Re-runs analysis ignoring any cache.

        Returns:
            Fresh call graph

        Raises:
            AnalyzerError: If analysis fails
        """
        pass

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache is valid and not expired
        """
        if not self._config.cache_enabled:
            return False

        if cache_key not in self._cache:
            return False

        entry = self._cache[cache_key]

        # Check TTL
        if self._config.cache_ttl_hours > 0:
            expiry = entry.timestamp + timedelta(hours=self._config.cache_ttl_hours)
            if datetime.utcnow() > expiry:
                return False

        # Could also check codebase_hash for invalidation
        return True

    def _get_cache_key(self) -> str:
        """Generate cache key for current configuration.

        Returns:
            Cache key string
        """
        return f"{self._codebase_path}:{self._config.primary_analyzer}"

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources.

        Releases any held resources and clears cache if needed.
        """
        pass


class OracleFactory:
    """Factory for creating StaticAnalysisOracle instances.

    Provides convenient methods to create oracles with common configurations.
    """

    @staticmethod
    def for_python(
        codebase_path: str | Path,
        cache_enabled: bool = True,
        strip_prefix: Optional[str] = None,
    ) -> "StaticAnalysisOracle":
        """Create oracle configured for Python analysis.

        Args:
            codebase_path: Path to Python codebase
            cache_enabled: Enable caching
            strip_prefix: Module prefix to strip

        Returns:
            Configured oracle instance
        """
        config = OracleConfig(
            language=Language.PYTHON,
            primary_analyzer=AnalyzerType.PYCG,
            fallback_analyzers=[AnalyzerType.PYAN3],
            cache_enabled=cache_enabled,
            normalization=NormalizationConfig(
                strip_module_prefix=strip_prefix,
                include_builtins=False,
                include_stdlib=False,
            ),
        )
        # Return concrete implementation (to be created)
        raise NotImplementedError("Concrete implementation required")

    @staticmethod
    def for_java(
        codebase_path: str | Path,
        cache_enabled: bool = True,
    ) -> "StaticAnalysisOracle":
        """Create oracle configured for Java analysis.

        Args:
            codebase_path: Path to Java codebase
            cache_enabled: Enable caching

        Returns:
            Configured oracle instance
        """
        config = OracleConfig(
            language=Language.JAVA,
            primary_analyzer=AnalyzerType.JAVA_CALLGRAPH,
            fallback_analyzers=[AnalyzerType.WALA],
            cache_enabled=cache_enabled,
        )
        raise NotImplementedError("Concrete implementation required")

    @staticmethod
    def for_typescript(
        codebase_path: str | Path,
        cache_enabled: bool = True,
    ) -> "StaticAnalysisOracle":
        """Create oracle configured for TypeScript analysis.

        Args:
            codebase_path: Path to TypeScript codebase
            cache_enabled: Enable caching

        Returns:
            Configured oracle instance
        """
        config = OracleConfig(
            language=Language.TYPESCRIPT,
            primary_analyzer=AnalyzerType.TS_CALLGRAPH,
            fallback_analyzers=[AnalyzerType.SOURCETRAIL],
            cache_enabled=cache_enabled,
        )
        raise NotImplementedError("Concrete implementation required")
