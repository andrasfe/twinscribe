"""
Default Static Analysis Oracle Implementation.

Provides the concrete implementation of StaticAnalysisOracle with support for
PyCG-based analysis and fallback to AST-based analysis when PyCG is unavailable.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerError,
    AnalyzerType,
    Language,
)
from twinscribe.analysis.oracle import (
    CacheEntry,
    OracleConfig,
    StaticAnalysisOracle,
)
from twinscribe.models.call_graph import CallGraph

logger = logging.getLogger(__name__)


class DefaultStaticAnalysisOracle(StaticAnalysisOracle):
    """Default implementation of the StaticAnalysisOracle.

    Provides call graph extraction using PyCG as the primary analyzer
    with support for fallback analyzers. Implements caching and
    incremental update detection.

    Features:
    - Primary analyzer with automatic fallback chain
    - Result caching with TTL and hash-based invalidation
    - Incremental updates when files change
    - Comprehensive query methods

    Usage:
        oracle = DefaultStaticAnalysisOracle(
            codebase_path="/path/to/project",
            config=OracleConfig(language=Language.PYTHON),
        )
        await oracle.initialize()

        # Query call graph
        callees = oracle.get_callees("module.function")
        callers = oracle.get_callers("module.Class.method")
        exists = oracle.verify_edge("caller", "callee")
    """

    def __init__(
        self,
        codebase_path: str | Path,
        config: OracleConfig | None = None,
    ) -> None:
        """Initialize the default oracle.

        Args:
            codebase_path: Path to the codebase to analyze
            config: Oracle configuration
        """
        super().__init__(codebase_path, config)
        self._analyzers: dict[AnalyzerType, Analyzer] = {}
        self._active_analyzer: Analyzer | None = None
        self._codebase_hash: str | None = None

    async def initialize(self) -> None:
        """Initialize the oracle and run initial analysis.

        Checks analyzer availability, selects the best available analyzer,
        and performs initial analysis.

        Raises:
            RuntimeError: If no analyzer is available
            AnalyzerError: If analysis fails
        """
        if self._initialized:
            logger.debug("Oracle already initialized")
            return

        logger.info(f"Initializing oracle for {self._codebase_path}")

        # Load and check analyzers
        await self._load_analyzers()

        if not self._analyzers:
            raise RuntimeError("No analyzers available. Install pycg with: pip install pycg")

        # Select active analyzer
        self._active_analyzer = await self._select_analyzer()
        if self._active_analyzer is None:
            raise RuntimeError("Failed to select an available analyzer")

        logger.info(f"Using {self._active_analyzer.analyzer_type.value} as primary analyzer")

        # Compute initial codebase hash
        self._codebase_hash = self._compute_codebase_hash()

        # Run initial analysis
        await self._run_analysis()

        self._initialized = True
        logger.info("Oracle initialization complete")

    async def _load_analyzers(self) -> None:
        """Load and check availability of configured analyzers."""
        analyzers_to_check = []

        # Add primary analyzer
        if self._config.primary_analyzer:
            analyzers_to_check.append(self._config.primary_analyzer)

        # Add fallback analyzers
        analyzers_to_check.extend(self._config.fallback_analyzers)

        # Check each analyzer
        for analyzer_type in analyzers_to_check:
            analyzer = self._create_analyzer(analyzer_type)
            if analyzer is not None:
                is_available = await analyzer.check_available()
                if is_available:
                    self._analyzers[analyzer_type] = analyzer
                    logger.debug(f"Analyzer {analyzer_type.value} is available")
                else:
                    logger.debug(f"Analyzer {analyzer_type.value} is not available")

    def _create_analyzer(self, analyzer_type: AnalyzerType) -> Analyzer | None:
        """Create an analyzer instance for the given type.

        Args:
            analyzer_type: Type of analyzer to create

        Returns:
            Analyzer instance or None if type not supported
        """
        if analyzer_type == AnalyzerType.PYCG:
            from twinscribe.analysis.pycg_analyzer import PyCGAnalyzer

            return PyCGAnalyzer()

        elif analyzer_type == AnalyzerType.AST:
            from twinscribe.analysis.ast_analyzer import ASTAnalyzer

            return ASTAnalyzer()

        elif analyzer_type == AnalyzerType.PYAN3:
            # Pyan3 analyzer would be implemented similarly
            # For now, return None as it's not implemented
            logger.debug("Pyan3 analyzer not yet implemented")
            return None

        else:
            logger.debug(f"Analyzer type {analyzer_type} not supported")
            return None

    async def _select_analyzer(self) -> Analyzer | None:
        """Select the best available analyzer.

        Returns primary analyzer if available, otherwise falls back
        through the fallback chain.

        Returns:
            Selected analyzer or None if none available
        """
        # Try primary first
        if self._config.primary_analyzer:
            if self._config.primary_analyzer in self._analyzers:
                return self._analyzers[self._config.primary_analyzer]

        # Try fallbacks in order
        for fallback_type in self._config.fallback_analyzers:
            if fallback_type in self._analyzers:
                logger.info(f"Primary analyzer unavailable, using fallback: {fallback_type.value}")
                self._stats.fallback_uses += 1
                return self._analyzers[fallback_type]

        # Try AST fallback if available
        try:
            from twinscribe.analysis.ast_analyzer import ASTAnalyzer

            ast_analyzer = ASTAnalyzer()
            if await ast_analyzer.check_available():
                logger.info("Using AST-based fallback analyzer")
                self._stats.fallback_uses += 1
                return ast_analyzer
        except ImportError:
            pass

        return None

    async def _run_analysis(self) -> CallGraph:
        """Run analysis using the active analyzer.

        Returns:
            Normalized call graph

        Raises:
            AnalyzerError: If analysis fails
        """
        if self._active_analyzer is None:
            raise RuntimeError("No analyzer selected")

        self._stats.total_analyses += 1

        try:
            result = await self._active_analyzer.analyze(self._codebase_path)
            self._stats.primary_successes += 1
        except AnalyzerError as e:
            logger.error(f"Analysis failed: {e}")
            # Try fallback
            fallback_graph = await self._try_fallback_analysis()
            if fallback_graph is not None:
                return fallback_graph
            raise

        # Normalize the result
        call_graph = self._normalizer.normalize(result)

        # Cache the result
        self._cache_result(call_graph)

        # Store as current call graph
        self._call_graph = call_graph

        return call_graph

    async def _try_fallback_analysis(self) -> CallGraph | None:
        """Try fallback analyzers after primary fails.

        Returns:
            CallGraph from fallback or None if all fail
        """
        for fallback_type in self._config.fallback_analyzers:
            if fallback_type in self._analyzers:
                analyzer = self._analyzers[fallback_type]
                try:
                    logger.info(f"Trying fallback analyzer: {fallback_type.value}")
                    result = await analyzer.analyze(self._codebase_path)
                    call_graph = self._normalizer.normalize(result)
                    self._stats.fallback_uses += 1
                    self._cache_result(call_graph)
                    self._call_graph = call_graph
                    return call_graph
                except AnalyzerError as e:
                    logger.warning(f"Fallback {fallback_type.value} also failed: {e}")
                    continue

        return None

    def _cache_result(self, call_graph: CallGraph) -> None:
        """Cache an analysis result.

        Args:
            call_graph: The call graph to cache
        """
        if not self._config.cache_enabled:
            return

        cache_key = self._get_cache_key()
        self._cache[cache_key] = CacheEntry(
            call_graph=call_graph,
            analyzer_type=self._active_analyzer.analyzer_type
            if self._active_analyzer
            else AnalyzerType.PYCG,
            timestamp=datetime.utcnow(),
            codebase_hash=self._codebase_hash or "",
        )

    def _compute_codebase_hash(self) -> str:
        """Compute a hash of the codebase for cache invalidation.

        Uses file modification times and sizes for efficiency.

        Returns:
            Hash string representing codebase state
        """
        hasher = hashlib.md5()

        # Get all Python files
        python_files = sorted(self._codebase_path.glob("**/*.py"))

        for file_path in python_files:
            try:
                stat = file_path.stat()
                # Include path, size, and mtime in hash
                hasher.update(str(file_path).encode())
                hasher.update(str(stat.st_size).encode())
                hasher.update(str(stat.st_mtime).encode())
            except OSError:
                continue

        return hasher.hexdigest()

    async def get_call_graph(self, force_refresh: bool = False) -> CallGraph:
        """Get the complete call graph.

        Returns cached result unless force_refresh is True, cache is
        expired, or codebase has changed.

        Args:
            force_refresh: Force re-analysis ignoring cache

        Returns:
            Complete normalized call graph

        Raises:
            RuntimeError: If oracle not initialized
            AnalyzerError: If analysis fails
        """
        if not self._initialized:
            raise RuntimeError("Oracle not initialized. Call initialize() first.")

        cache_key = self._get_cache_key()

        # Check if we need to refresh
        needs_refresh = force_refresh

        if not needs_refresh:
            # Check cache validity
            if not self._is_cache_valid(cache_key):
                self._stats.cache_misses += 1
                needs_refresh = True
            else:
                # Check if codebase changed
                current_hash = self._compute_codebase_hash()
                if cache_key in self._cache:
                    if self._cache[cache_key].codebase_hash != current_hash:
                        logger.info("Codebase changed, refreshing analysis")
                        needs_refresh = True
                        self._codebase_hash = current_hash

        if needs_refresh:
            return await self._run_analysis()

        # Return cached result
        self._stats.cache_hits += 1
        if self._call_graph is not None:
            return self._call_graph

        # Should not reach here, but handle gracefully
        return await self._run_analysis()

    def get_all_edges(self) -> list[tuple[str, str]]:
        """Get all edges as (caller, callee) tuples.

        Returns:
            List of all edges in the call graph

        Raises:
            RuntimeError: If oracle not initialized
        """
        if not self._initialized or self._call_graph is None:
            raise RuntimeError("Oracle not initialized")

        self._stats.total_queries += 1
        return [(edge.caller, edge.callee) for edge in self._call_graph.edges]

    async def refresh(self) -> CallGraph:
        """Force refresh the call graph from source.

        Re-runs analysis ignoring any cache.

        Returns:
            Fresh call graph

        Raises:
            RuntimeError: If oracle not initialized
            AnalyzerError: If analysis fails
        """
        if not self._initialized:
            raise RuntimeError("Oracle not initialized. Call initialize() first.")

        # Update codebase hash
        self._codebase_hash = self._compute_codebase_hash()

        # Clear cache for this key
        cache_key = self._get_cache_key()
        if cache_key in self._cache:
            del self._cache[cache_key]

        return await self._run_analysis()

    async def update_for_files(self, changed_files: list[Path]) -> CallGraph:
        """Update call graph for specific changed files.

        This is more efficient than full refresh when only a few
        files have changed. Currently performs full refresh but
        could be optimized for incremental updates.

        Args:
            changed_files: List of files that changed

        Returns:
            Updated call graph
        """
        logger.info(f"Updating analysis for {len(changed_files)} changed files")

        # For now, do full refresh
        # Future: implement incremental analysis
        return await self.refresh()

    async def shutdown(self) -> None:
        """Clean up resources.

        Clears cache and releases analyzer resources.
        """
        logger.info("Shutting down oracle")

        # Clear cache
        self._cache.clear()
        self._call_graph = None
        self._initialized = False

        # Clear analyzer references
        self._analyzers.clear()
        self._active_analyzer = None


def create_python_oracle(
    codebase_path: str | Path,
    cache_enabled: bool = True,
    strip_prefix: str | None = None,
    include_stdlib: bool = False,
    include_builtins: bool = False,
) -> DefaultStaticAnalysisOracle:
    """Create an oracle configured for Python analysis.

    Convenience factory function for creating a Python-focused oracle
    with sensible defaults.

    Args:
        codebase_path: Path to Python codebase
        cache_enabled: Enable result caching
        strip_prefix: Module prefix to strip from component IDs
        include_stdlib: Include stdlib calls in the graph
        include_builtins: Include builtin function calls

    Returns:
        Configured DefaultStaticAnalysisOracle instance

    Usage:
        oracle = create_python_oracle("/path/to/project")
        await oracle.initialize()
        callees = oracle.get_callees("mymodule.MyClass.my_method")
    """
    from twinscribe.analysis.normalizer import NormalizationConfig

    config = OracleConfig(
        language=Language.PYTHON,
        primary_analyzer=AnalyzerType.PYCG,
        fallback_analyzers=[AnalyzerType.AST],  # AST is always available
        cache_enabled=cache_enabled,
        normalization=NormalizationConfig(
            strip_module_prefix=strip_prefix,
            include_builtins=include_builtins,
            include_stdlib=include_stdlib,
        ),
    )

    return DefaultStaticAnalysisOracle(codebase_path, config)
