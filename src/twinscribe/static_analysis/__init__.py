"""
TwinScribe Static Analysis Module.

This module provides the ground truth call graph extraction from static analysis tools.

Key Components:
- StaticAnalysisOracle: Main interface for ground truth call graph extraction
- PyCGAnalyzer: Python call graph analysis using PyCG
- Pyan3Analyzer: Fallback Python call graph analysis using pyan3

The static analysis results serve as the authoritative source for call graph
validation, eliminating "consensus of wrong answers" from LLM outputs.

Supported Languages:
- Python (primary: PyCG, fallback: pyan3)
- Java (java-callgraph-static, WALA)
- JavaScript/TypeScript (typescript-call-graph)
- Multi-language (Sourcetrail)
"""

from twinscribe.static_analysis.oracle import (
    StaticAnalysisOracle,
    BaseAnalyzer,
    PyCGAnalyzer,
    Pyan3Analyzer,
    AnalyzerError,
    AnalyzerNotAvailableError,
    AnalyzerExecutionError,
)

__all__ = [
    "StaticAnalysisOracle",
    "BaseAnalyzer",
    "PyCGAnalyzer",
    "Pyan3Analyzer",
    "AnalyzerError",
    "AnalyzerNotAvailableError",
    "AnalyzerExecutionError",
]
