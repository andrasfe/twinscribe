"""
Dual-Stream Documentation System - Static Analysis Integration

This module provides the static analysis integration layer that serves
as the ground truth anchor for call graph validation.

Supported analyzers:
- Python: PyCG (primary), AST-based (fallback), pyan3 (optional)
- Java: java-callgraph-static (primary), WALA (fallback)
- JavaScript/TypeScript: typescript-call-graph
- Multi-language: Sourcetrail (fallback for all)

The StaticAnalysisOracle provides a unified interface for extracting
and querying call graphs regardless of the underlying tool.

Usage:
    from twinscribe.analysis import create_python_oracle

    oracle = create_python_oracle("/path/to/project")
    await oracle.initialize()

    # Query call graph
    callees = oracle.get_callees("module.function")
    callers = oracle.get_callers("module.Class.method")
"""

from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerError,
    AnalyzerResult,
    AnalyzerType,
    Language,
)
from twinscribe.analysis.ast_analyzer import ASTAnalyzer, create_ast_analyzer
from twinscribe.analysis.component_discovery import (
    ComponentDiscovery,
    DiscoveryResult,
    create_component_discovery,
    discover_components,
)
from twinscribe.analysis.default_oracle import (
    DefaultStaticAnalysisOracle,
    create_python_oracle,
)
from twinscribe.analysis.normalizer import (
    CallGraphNormalizer,
    NormalizationConfig,
)
from twinscribe.analysis.oracle import OracleConfig, OracleFactory, StaticAnalysisOracle
from twinscribe.analysis.pycg_analyzer import PyCGAnalyzer, create_pycg_analyzer
from twinscribe.analysis.registry import (
    AnalyzerRegistry,
    get_analyzer,
    get_registry,
    register_analyzer,
)

__all__ = [
    # Main oracle
    "StaticAnalysisOracle",
    "DefaultStaticAnalysisOracle",
    "OracleConfig",
    "OracleFactory",
    "create_python_oracle",
    # Analyzer base
    "Analyzer",
    "AnalyzerConfig",
    "AnalyzerResult",
    "AnalyzerError",
    "AnalyzerType",
    "Language",
    # Concrete analyzers
    "PyCGAnalyzer",
    "create_pycg_analyzer",
    "ASTAnalyzer",
    "create_ast_analyzer",
    # Component discovery
    "ComponentDiscovery",
    "DiscoveryResult",
    "create_component_discovery",
    "discover_components",
    # Normalizer
    "CallGraphNormalizer",
    "NormalizationConfig",
    # Registry
    "AnalyzerRegistry",
    "register_analyzer",
    "get_analyzer",
    "get_registry",
]


def _register_default_analyzers() -> None:
    """Register default analyzer implementations with the global registry."""
    registry = get_registry()

    # Register PyCG as primary for Python
    registry.register(
        AnalyzerType.PYCG,
        PyCGAnalyzer,
        default_for_language=Language.PYTHON,
    )

    # Note: AST analyzer is used as internal fallback in DefaultStaticAnalysisOracle
    # but not registered separately since it reuses PYCG type for compatibility


# Register default analyzers on module import
_register_default_analyzers()
