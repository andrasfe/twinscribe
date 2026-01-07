"""
Call Graph Normalizer.

Converts raw analyzer output to normalized CallGraph format with
consistent component IDs across different tools.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from twinscribe.analysis.analyzer import AnalyzerResult, AnalyzerType, RawCallEdge
from twinscribe.models.base import CallType
from twinscribe.models.call_graph import CallEdge, CallGraph


class NormalizationConfig(BaseModel):
    """Configuration for call graph normalization.

    Attributes:
        strip_module_prefix: Common prefix to strip from IDs
        include_builtins: Whether to include builtin function calls
        include_stdlib: Whether to include standard library calls
        include_external: Whether to include external package calls
        resolve_aliases: Whether to resolve import aliases
        normalize_case: Whether to normalize case for comparison
    """

    strip_module_prefix: str | None = Field(
        default=None,
        description="Module prefix to strip from component IDs",
    )
    include_builtins: bool = Field(
        default=False,
        description="Include builtin function calls",
    )
    include_stdlib: bool = Field(
        default=False,
        description="Include standard library calls",
    )
    include_external: bool = Field(
        default=True,
        description="Include external package calls",
    )
    resolve_aliases: bool = Field(
        default=True,
        description="Resolve import aliases to canonical names",
    )
    normalize_case: bool = Field(
        default=False,
        description="Normalize case for comparison",
    )


# Known Python builtins to filter
PYTHON_BUILTINS = {
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "type",
    "isinstance",
    "issubclass",
    "hasattr",
    "getattr",
    "setattr",
    "delattr",
    "callable",
    "iter",
    "next",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "min",
    "max",
    "sum",
    "any",
    "all",
    "abs",
    "round",
    "pow",
    "divmod",
    "open",
    "input",
    "format",
    "repr",
    "hash",
    "id",
    "object",
    "super",
    "classmethod",
    "staticmethod",
    "property",
}

# Known Python stdlib modules to filter
PYTHON_STDLIB_PREFIXES = {
    "os.",
    "sys.",
    "re.",
    "json.",
    "datetime.",
    "collections.",
    "itertools.",
    "functools.",
    "pathlib.",
    "typing.",
    "abc.",
    "logging.",
    "threading.",
    "multiprocessing.",
    "asyncio.",
    "http.",
    "urllib.",
    "email.",
    "html.",
    "xml.",
    "sqlite3.",
    "pickle.",
    "copy.",
    "math.",
    "random.",
    "statistics.",
}


@dataclass
class NormalizationStats:
    """Statistics from normalization process.

    Attributes:
        total_raw_edges: Edges before normalization
        normalized_edges: Edges after normalization
        filtered_builtins: Edges filtered as builtins
        filtered_stdlib: Edges filtered as stdlib
        filtered_external: Edges filtered as external
        failed_normalization: Edges that couldn't be normalized
    """

    total_raw_edges: int = 0
    normalized_edges: int = 0
    filtered_builtins: int = 0
    filtered_stdlib: int = 0
    filtered_external: int = 0
    failed_normalization: int = 0


class CallGraphNormalizer:
    """Normalizes raw analyzer output to CallGraph format.

    Different analyzers produce call graphs in different formats:
    - PyCG: Full module paths with :: separators
    - pyan3: Different format with -> notation
    - java-callgraph: Java fully qualified names
    - etc.

    The normalizer converts these to a consistent format using
    dotted module.Class.method notation.
    """

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        """Initialize the normalizer.

        Args:
            config: Normalization configuration
        """
        self._config = config or NormalizationConfig()
        self._stats = NormalizationStats()

    @property
    def config(self) -> NormalizationConfig:
        """Get normalization config."""
        return self._config

    @property
    def stats(self) -> NormalizationStats:
        """Get normalization statistics."""
        return self._stats

    def normalize(self, result: AnalyzerResult) -> CallGraph:
        """Normalize analyzer result to CallGraph.

        Args:
            result: Raw analyzer result

        Returns:
            Normalized CallGraph with consistent component IDs
        """
        self._stats = NormalizationStats(total_raw_edges=len(result.raw_edges))

        edges = []
        for raw_edge in result.raw_edges:
            normalized = self._normalize_edge(raw_edge, result.analyzer_type)
            if normalized is not None:
                edges.append(normalized)

        self._stats.normalized_edges = len(edges)

        return CallGraph(
            edges=edges,
            source=result.analyzer_type.value,
        )

    def _normalize_edge(
        self,
        raw_edge: RawCallEdge,
        analyzer_type: AnalyzerType,
    ) -> CallEdge | None:
        """Normalize a single raw edge.

        Args:
            raw_edge: Raw edge from analyzer
            analyzer_type: Type of analyzer

        Returns:
            Normalized CallEdge or None if filtered
        """
        # Normalize caller and callee IDs based on analyzer type
        caller = self._normalize_id(raw_edge.caller, analyzer_type)
        callee = self._normalize_id(raw_edge.callee, analyzer_type)

        if caller is None or callee is None:
            self._stats.failed_normalization += 1
            return None

        # Apply filters
        if not self._should_include(callee):
            return None

        # Determine call type from metadata if available
        call_type = self._determine_call_type(raw_edge)

        return CallEdge(
            caller=caller,
            callee=callee,
            call_site_line=raw_edge.line_number,
            call_type=call_type,
            confidence=1.0,  # Static analysis = full confidence
        )

    def _normalize_id(
        self,
        raw_id: str,
        analyzer_type: AnalyzerType,
    ) -> str | None:
        """Normalize a component ID to standard format.

        Args:
            raw_id: Raw ID from analyzer
            analyzer_type: Type of analyzer

        Returns:
            Normalized ID in module.Class.method format
        """
        if not raw_id:
            return None

        # Apply analyzer-specific normalization
        if analyzer_type == AnalyzerType.PYCG:
            normalized = self._normalize_pycg_id(raw_id)
        elif analyzer_type == AnalyzerType.PYAN3:
            normalized = self._normalize_pyan3_id(raw_id)
        elif analyzer_type == AnalyzerType.JAVA_CALLGRAPH:
            normalized = self._normalize_java_id(raw_id)
        elif analyzer_type == AnalyzerType.TS_CALLGRAPH:
            normalized = self._normalize_ts_id(raw_id)
        else:
            # Default: assume already normalized
            normalized = raw_id

        if normalized is None:
            return None

        # Strip module prefix if configured
        if self._config.strip_module_prefix:
            prefix = self._config.strip_module_prefix
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                if normalized.startswith("."):
                    normalized = normalized[1:]

        # Normalize case if configured
        if self._config.normalize_case:
            normalized = normalized.lower()

        return normalized

    def _normalize_pycg_id(self, raw_id: str) -> str | None:
        """Normalize PyCG format to standard format.

        PyCG format: module.submodule.Class.method or module.function
        Already mostly correct, just need to handle special cases.
        """
        # PyCG uses dot notation which is already standard
        # Handle special markers like <module> or <listcomp>
        if raw_id.startswith("<") or raw_id.endswith(">"):
            # Filter out internal Python constructs
            return None

        return raw_id

    def _normalize_pyan3_id(self, raw_id: str) -> str | None:
        """Normalize pyan3 format to standard format.

        pyan3 may use different notation depending on output format.
        """
        # pyan3 also typically uses dot notation
        # Remove any leading/trailing whitespace
        normalized = raw_id.strip()

        # Handle potential arrow notation in some outputs
        if "->" in normalized:
            # This is likely a call representation, not an ID
            return None

        return normalized

    def _normalize_java_id(self, raw_id: str) -> str | None:
        """Normalize Java fully qualified name to standard format.

        Java format: com.package.Class.method(params)Returntype
        Standard format: com.package.Class.method
        """
        # Remove method signature
        if "(" in raw_id:
            raw_id = raw_id.split("(")[0]

        # Remove return type indicator
        if ")" in raw_id:
            raw_id = raw_id.split(")")[0]

        # Java uses dots which is already standard
        return raw_id

    def _normalize_ts_id(self, raw_id: str) -> str | None:
        """Normalize TypeScript/JavaScript format to standard format.

        TS format varies but typically includes file path and symbol.
        """
        # TypeScript call graphs often include file paths
        # Convert to module.symbol format

        # Remove file extension
        if raw_id.endswith(".ts") or raw_id.endswith(".js"):
            raw_id = raw_id.rsplit(".", 1)[0]

        # Convert slashes to dots
        normalized = raw_id.replace("/", ".").replace("\\", ".")

        return normalized

    def _should_include(self, component_id: str) -> bool:
        """Check if a component should be included based on filters.

        Args:
            component_id: Normalized component ID

        Returns:
            True if should be included
        """
        # Check builtins
        if not self._config.include_builtins:
            base_name = component_id.rsplit(".", 1)[-1]
            if base_name in PYTHON_BUILTINS:
                self._stats.filtered_builtins += 1
                return False

        # Check stdlib
        if not self._config.include_stdlib:
            for prefix in PYTHON_STDLIB_PREFIXES:
                if component_id.startswith(prefix):
                    self._stats.filtered_stdlib += 1
                    return False

        return True

    def _determine_call_type(self, raw_edge: RawCallEdge) -> CallType:
        """Determine the type of call from metadata.

        Args:
            raw_edge: Raw edge with metadata

        Returns:
            Appropriate CallType
        """
        metadata = raw_edge.metadata

        # Check for explicit call type in metadata
        if "call_type" in metadata:
            type_str = metadata["call_type"].lower()
            if "conditional" in type_str or "if" in type_str:
                return CallType.CONDITIONAL
            elif "loop" in type_str or "for" in type_str or "while" in type_str:
                return CallType.LOOP
            elif "try" in type_str or "except" in type_str:
                return CallType.EXCEPTION
            elif "callback" in type_str or "lambda" in type_str:
                return CallType.CALLBACK
            elif "dynamic" in type_str or "getattr" in type_str:
                return CallType.DYNAMIC

        # Default to direct call
        return CallType.DIRECT
