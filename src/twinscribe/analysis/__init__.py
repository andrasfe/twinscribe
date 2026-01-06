"""
Dual-Stream Documentation System - Static Analysis Integration

This module provides the static analysis integration layer that serves
as the ground truth anchor for call graph validation.

Supported analyzers:
- Python: PyCG (primary), pyan3 (fallback)
- Java: java-callgraph-static (primary), WALA (fallback)
- JavaScript/TypeScript: typescript-call-graph
- Multi-language: Sourcetrail (fallback for all)

The StaticAnalysisOracle provides a unified interface for extracting
and querying call graphs regardless of the underlying tool.
"""

from twinscribe.analysis.oracle import StaticAnalysisOracle
from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerResult,
    AnalyzerError,
)
from twinscribe.analysis.normalizer import (
    CallGraphNormalizer,
    NormalizationConfig,
)
from twinscribe.analysis.registry import (
    AnalyzerRegistry,
    register_analyzer,
    get_analyzer,
)

__all__ = [
    # Main oracle
    "StaticAnalysisOracle",
    # Analyzer base
    "Analyzer",
    "AnalyzerConfig",
    "AnalyzerResult",
    "AnalyzerError",
    # Normalizer
    "CallGraphNormalizer",
    "NormalizationConfig",
    # Registry
    "AnalyzerRegistry",
    "register_analyzer",
    "get_analyzer",
]
