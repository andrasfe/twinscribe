"""
Abstract Analyzer Interface.

Defines the base interface for language-specific static analysis tools
that extract call graphs from source code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AnalyzerType(str, Enum):
    """Type of static analyzer."""

    PYCG = "pycg"  # Python Call Graph
    PYAN3 = "pyan3"  # Python analyzer
    JAVA_CALLGRAPH = "java-callgraph"  # Java call graph
    WALA = "wala"  # IBM WALA for Java
    TS_CALLGRAPH = "typescript-call-graph"  # TypeScript/JavaScript
    SOURCETRAIL = "sourcetrail"  # Multi-language indexer


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    MULTI = "multi"  # Multiple languages


class AnalyzerConfig(BaseModel):
    """Configuration for a static analyzer.

    Attributes:
        analyzer_type: Type of analyzer
        language: Target language
        executable_path: Path to analyzer executable (if external)
        timeout_seconds: Maximum execution time
        max_iterations: Max iterations for iterative analyzers
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
        output_format: Expected output format
        extra_args: Additional command-line arguments
    """

    analyzer_type: AnalyzerType = Field(..., description="Type of analyzer")
    language: Language = Field(..., description="Target language")
    executable_path: str | None = Field(
        default=None,
        description="Path to analyzer executable",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Maximum execution time",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        description="Max iterations for iterative analysis",
    )
    include_patterns: list[str] = Field(
        default_factory=lambda: ["**/*.py"],
        description="File patterns to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/test_*",
            "**/tests/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
        ],
        description="File patterns to exclude",
    )
    output_format: str = Field(
        default="json",
        description="Expected output format",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional arguments",
    )


# Default configurations for supported analyzers
PYCG_CONFIG = AnalyzerConfig(
    analyzer_type=AnalyzerType.PYCG,
    language=Language.PYTHON,
    include_patterns=["**/*.py"],
    output_format="json",
)

PYAN3_CONFIG = AnalyzerConfig(
    analyzer_type=AnalyzerType.PYAN3,
    language=Language.PYTHON,
    include_patterns=["**/*.py"],
    output_format="json",
    extra_args=["--no-defines"],
)

JAVA_CALLGRAPH_CONFIG = AnalyzerConfig(
    analyzer_type=AnalyzerType.JAVA_CALLGRAPH,
    language=Language.JAVA,
    include_patterns=["**/*.java", "**/*.class", "**/*.jar"],
    output_format="text",
)

TS_CALLGRAPH_CONFIG = AnalyzerConfig(
    analyzer_type=AnalyzerType.TS_CALLGRAPH,
    language=Language.TYPESCRIPT,
    include_patterns=["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"],
    exclude_patterns=["**/node_modules/**", "**/*.d.ts"],
    output_format="json",
)

SOURCETRAIL_CONFIG = AnalyzerConfig(
    analyzer_type=AnalyzerType.SOURCETRAIL,
    language=Language.MULTI,
    timeout_seconds=600,  # Sourcetrail can be slow
    output_format="database",
)


class AnalyzerError(Exception):
    """Exception raised when static analysis fails."""

    def __init__(
        self,
        message: str,
        analyzer_type: AnalyzerType,
        exit_code: int | None = None,
        stderr: str | None = None,
    ) -> None:
        """Initialize analyzer error.

        Args:
            message: Error message
            analyzer_type: Type of analyzer that failed
            exit_code: Process exit code if applicable
            stderr: Standard error output if available
        """
        super().__init__(message)
        self.analyzer_type = analyzer_type
        self.exit_code = exit_code
        self.stderr = stderr


@dataclass
class RawCallEdge:
    """Raw call edge from analyzer output before normalization.

    Attributes:
        caller: Caller identifier (format varies by analyzer)
        callee: Callee identifier (format varies by analyzer)
        line_number: Line number if available
        metadata: Additional metadata from analyzer
    """

    caller: str
    callee: str
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    """Result from running a static analyzer.

    Attributes:
        analyzer_type: Type of analyzer used
        codebase_path: Path to analyzed codebase
        raw_edges: Raw call edges (before normalization)
        nodes: All discovered nodes/components
        execution_time_seconds: How long analysis took
        timestamp: When analysis was performed
        warnings: Non-fatal warnings during analysis
        metadata: Additional analyzer-specific metadata
    """

    analyzer_type: AnalyzerType
    codebase_path: str
    raw_edges: list[RawCallEdge] = field(default_factory=list)
    nodes: set[str] = field(default_factory=set)
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_count(self) -> int:
        """Number of edges discovered."""
        return len(self.raw_edges)

    @property
    def node_count(self) -> int:
        """Number of unique nodes."""
        return len(self.nodes)


class Analyzer(ABC):
    """Abstract base class for static analysis tools.

    Each analyzer implementation wraps a specific tool (PyCG, pyan3, etc.)
    and provides a common interface for extracting call graphs.

    Lifecycle:
    1. Create with configuration
    2. check_available() - Verify tool is installed
    3. analyze() - Run analysis on codebase
    4. Results are raw edges that need normalization
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        """Initialize the analyzer.

        Args:
            config: Analyzer configuration
        """
        self._config = config

    @property
    def config(self) -> AnalyzerConfig:
        """Get analyzer configuration."""
        return self._config

    @property
    def analyzer_type(self) -> AnalyzerType:
        """Get analyzer type."""
        return self._config.analyzer_type

    @property
    def language(self) -> Language:
        """Get target language."""
        return self._config.language

    @abstractmethod
    async def check_available(self) -> bool:
        """Check if the analyzer tool is available.

        Verifies the tool is installed and accessible.

        Returns:
            True if analyzer is available
        """
        pass

    @abstractmethod
    async def get_version(self) -> str | None:
        """Get the version of the analyzer tool.

        Returns:
            Version string or None if not available
        """
        pass

    @abstractmethod
    async def analyze(self, codebase_path: Path) -> AnalyzerResult:
        """Run static analysis on the codebase.

        Extracts call graph information from the source code.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            Raw analysis result with edges

        Raises:
            AnalyzerError: If analysis fails
            FileNotFoundError: If codebase_path doesn't exist
            TimeoutError: If analysis exceeds timeout
        """
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> list[RawCallEdge]:
        """Parse raw output from the analyzer tool.

        Converts tool-specific output format to RawCallEdge list.

        Args:
            raw_output: Raw output from the tool

        Returns:
            List of raw call edges
        """
        pass

    def _build_command(self, codebase_path: Path) -> list[str]:
        """Build command line for running the analyzer.

        Override in subclasses for tool-specific command building.

        Args:
            codebase_path: Path to codebase

        Returns:
            Command as list of strings
        """
        raise NotImplementedError("Subclass must implement _build_command")

    def _filter_files(self, codebase_path: Path) -> list[Path]:
        """Get list of files matching include/exclude patterns.

        Args:
            codebase_path: Path to codebase

        Returns:
            List of files to analyze
        """
        import fnmatch

        all_files = []
        for pattern in self._config.include_patterns:
            all_files.extend(codebase_path.glob(pattern))

        # Filter out excluded patterns
        filtered = []
        for file_path in all_files:
            rel_path = str(file_path.relative_to(codebase_path))
            excluded = any(fnmatch.fnmatch(rel_path, exc) for exc in self._config.exclude_patterns)
            if not excluded:
                filtered.append(file_path)

        return filtered
