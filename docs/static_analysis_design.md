# Static Analysis Integration Design Document

## Overview

This document describes the static analysis integration layer that serves as the ground truth anchor for call graph validation in the Dual-Stream Documentation System.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      StaticAnalysisOracle                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     AnalyzerRegistry                             │   │
│  │                                                                  │   │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │   │
│  │   │   PyCG      │   │   pyan3     │   │ Sourcetrail │          │   │
│  │   │  Analyzer   │   │  Analyzer   │   │  Analyzer   │          │   │
│  │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │   │
│  │          │                 │                 │                  │   │
│  │          ▼                 ▼                 ▼                  │   │
│  │   ┌─────────────────────────────────────────────────────────┐  │   │
│  │   │              CallGraphNormalizer                        │  │   │
│  │   │                                                         │  │   │
│  │   │   Raw Output ──▶ RawCallEdge ──▶ Normalized CallEdge   │  │   │
│  │   │                                                         │  │   │
│  │   └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Cache                                    │   │
│  │                                                                  │   │
│  │   codebase_hash ──▶ CallGraph                                   │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Public Interface:                                                      │
│  - get_call_graph() -> CallGraph                                       │
│  - get_callees(component_id) -> list[CallEdge]                         │
│  - get_callers(component_id) -> list[CallEdge]                         │
│  - verify_edge(caller, callee) -> bool                                 │
│  - diff_against(documented_graph) -> CallGraphDiff                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/twinscribe/analysis/
├── __init__.py        # Public exports
├── analyzer.py        # Abstract Analyzer base class
├── normalizer.py      # Call graph normalization
├── oracle.py          # StaticAnalysisOracle interface
└── registry.py        # Analyzer registration
```

## Supported Analyzers

| Analyzer | Language | Output Format | Accuracy | Speed |
|----------|----------|---------------|----------|-------|
| PyCG | Python | JSON | High | Fast |
| pyan3 | Python | JSON/Graphviz | Medium | Fast |
| java-callgraph | Java | Text | High | Medium |
| WALA | Java | Various | Very High | Slow |
| typescript-call-graph | JS/TS | JSON | Medium | Fast |
| Sourcetrail | Multi | Database | High | Slow |

## Configuration

### OracleConfig

```python
class OracleConfig(BaseModel):
    language: Language = Language.PYTHON
    primary_analyzer: Optional[AnalyzerType] = None  # Auto-selected
    fallback_analyzers: list[AnalyzerType] = []
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    normalization: NormalizationConfig
    auto_select_analyzer: bool = True
```

### AnalyzerConfig

```python
class AnalyzerConfig(BaseModel):
    analyzer_type: AnalyzerType
    language: Language
    executable_path: Optional[str] = None
    timeout_seconds: int = 300
    max_iterations: int = 5
    include_patterns: list[str] = ["**/*.py"]
    exclude_patterns: list[str] = ["**/test_*", "**/tests/**", "**/__pycache__/**"]
    output_format: str = "json"
    extra_args: list[str] = []
```

### NormalizationConfig

```python
class NormalizationConfig(BaseModel):
    strip_module_prefix: Optional[str] = None
    include_builtins: bool = False
    include_stdlib: bool = False
    include_external: bool = True
    resolve_aliases: bool = True
    normalize_case: bool = False
```

### Default Configurations by Language

```yaml
python:
  primary: pycg
  fallback: [pyan3, sourcetrail]
  config:
    include_patterns: ["**/*.py"]
    exclude_patterns: ["**/test_*", "**/tests/**"]

java:
  primary: java-callgraph
  fallback: [wala, sourcetrail]
  config:
    include_patterns: ["**/*.java", "**/*.class"]

typescript:
  primary: typescript-call-graph
  fallback: [sourcetrail]
  config:
    include_patterns: ["**/*.ts", "**/*.tsx"]
    exclude_patterns: ["**/node_modules/**"]
```

## Interfaces

### StaticAnalysisOracle

Main interface for ground truth extraction.

```python
class StaticAnalysisOracle(ABC):
    def __init__(codebase_path: str | Path, config: OracleConfig = None)

    # Lifecycle
    async def initialize() -> None
    async def shutdown() -> None

    # Call graph access
    async def get_call_graph(force_refresh: bool = False) -> CallGraph
    async def refresh() -> CallGraph

    # Query methods
    def get_callees(component_id: str) -> list[CallEdge]
    def get_callers(component_id: str) -> list[CallEdge]
    def verify_edge(caller: str, callee: str) -> bool
    def all_nodes() -> set[str]

    # Comparison
    def diff_against(documented_graph: CallGraph) -> CallGraphDiff

    # Properties
    @property codebase_path: Path
    @property config: OracleConfig
    @property stats: OracleStats
    @property is_initialized: bool
    @property call_graph: Optional[CallGraph]
```

### Analyzer

Abstract base class for static analysis tools.

```python
class Analyzer(ABC):
    def __init__(config: AnalyzerConfig)

    @property config: AnalyzerConfig
    @property analyzer_type: AnalyzerType
    @property language: Language

    # Check availability
    async def check_available() -> bool
    async def get_version() -> Optional[str]

    # Analysis
    async def analyze(codebase_path: Path) -> AnalyzerResult

    # Output parsing
    def parse_output(raw_output: str) -> list[RawCallEdge]
```

### CallGraphNormalizer

Converts analyzer output to normalized CallGraph.

```python
class CallGraphNormalizer:
    def __init__(config: NormalizationConfig = None)

    @property config: NormalizationConfig
    @property stats: NormalizationStats

    def normalize(result: AnalyzerResult) -> CallGraph
```

### AnalyzerRegistry

Manages analyzer implementations.

```python
class AnalyzerRegistry:
    # Registration
    def register(analyzer_type, implementation, default_for_language=None)
    def unregister(analyzer_type) -> bool

    # Lookup
    def get(analyzer_type, config=None) -> Analyzer
    def get_for_language(language, config=None) -> Analyzer
    def is_registered(analyzer_type) -> bool
    def list_registered() -> list[AnalyzerType]
    def list_for_language(language) -> list[AnalyzerType]
```

## Data Structures

### RawCallEdge

Raw edge before normalization.

```python
@dataclass
class RawCallEdge:
    caller: str             # Analyzer-specific format
    callee: str             # Analyzer-specific format
    line_number: Optional[int]
    metadata: dict[str, Any]
```

### AnalyzerResult

Raw result from analyzer.

```python
@dataclass
class AnalyzerResult:
    analyzer_type: AnalyzerType
    codebase_path: str
    raw_edges: list[RawCallEdge]
    nodes: set[str]
    execution_time_seconds: float
    timestamp: datetime
    warnings: list[str]
    metadata: dict[str, Any]
```

### NormalizationStats

Statistics from normalization.

```python
@dataclass
class NormalizationStats:
    total_raw_edges: int
    normalized_edges: int
    filtered_builtins: int
    filtered_stdlib: int
    filtered_external: int
    failed_normalization: int
```

### OracleStats

Statistics from oracle operations.

```python
@dataclass
class OracleStats:
    cache_hits: int
    cache_misses: int
    primary_successes: int
    fallback_uses: int
    total_analyses: int
    total_queries: int
```

## Normalization Process

Different analyzers produce call graphs in different formats. The normalizer converts these to a consistent format.

### Format Examples

**PyCG Output:**
```json
{
  "mypackage.module.MyClass.method": [
    "mypackage.utils.helper_function",
    "builtins.print"
  ]
}
```

**pyan3 Output:**
```
mypackage.module.MyClass.method -> mypackage.utils.helper_function
mypackage.module.MyClass.method -> builtins.print
```

**java-callgraph Output:**
```
M:com.myapp.MyClass:myMethod (I)V (M)com/myapp/Helper:help()V
```

### Normalization Steps

1. **Parse analyzer-specific format** to `RawCallEdge`
2. **Convert identifiers** to `module.Class.method` format
3. **Apply filters** (builtins, stdlib, external)
4. **Resolve aliases** if configured
5. **Strip prefixes** if configured
6. **Create normalized `CallEdge`** with confidence=1.0

### Filtering Rules

| Filter | Default | Effect |
|--------|---------|--------|
| include_builtins | False | Exclude `print`, `len`, etc. |
| include_stdlib | False | Exclude `os.`, `sys.`, etc. |
| include_external | True | Include third-party packages |

## Fallback Logic

The oracle tries analyzers in order until one succeeds:

```
1. Try primary analyzer
   ├── Success: Use result
   └── Failure: Try fallback 1
       ├── Success: Use result
       └── Failure: Try fallback 2
           ├── Success: Use result
           └── Failure: Raise AnalyzerError
```

Failures that trigger fallback:
- Analyzer not available (not installed)
- Timeout exceeded
- Parse error on output
- Empty result (no edges found)

## Caching

Results are cached to avoid repeated analysis:

```python
@dataclass
class CacheEntry:
    call_graph: CallGraph
    analyzer_type: AnalyzerType
    timestamp: datetime
    codebase_hash: str  # For invalidation
```

Cache invalidation:
- TTL expiry (default 24 hours)
- Codebase hash change (file modifications)
- Explicit refresh request
- Analyzer type change

## Usage Examples

### Basic Usage

```python
from twinscribe.analysis import StaticAnalysisOracle, OracleConfig

# Create oracle for Python codebase
oracle = ConcreteOracle(
    codebase_path="/path/to/codebase",
    config=OracleConfig(language=Language.PYTHON)
)

# Initialize (runs analysis)
await oracle.initialize()

# Get full call graph
call_graph = await oracle.get_call_graph()
print(f"Found {call_graph.edge_count} edges")

# Query specific relationships
callees = oracle.get_callees("mypackage.MyClass.method")
for edge in callees:
    print(f"  calls {edge.callee}")

# Verify an edge
if oracle.verify_edge("caller.func", "callee.func"):
    print("Edge exists")

# Compare against documented graph
diff = oracle.diff_against(documented_graph)
print(f"Precision: {diff.precision:.2%}")
print(f"Recall: {diff.recall:.2%}")
```

### Custom Configuration

```python
config = OracleConfig(
    language=Language.PYTHON,
    primary_analyzer=AnalyzerType.PYAN3,  # Override default
    fallback_analyzers=[AnalyzerType.PYCG],
    cache_enabled=True,
    cache_ttl_hours=48,
    normalization=NormalizationConfig(
        strip_module_prefix="mypackage.",
        include_builtins=False,
        include_stdlib=False,
    ),
)

oracle = ConcreteOracle("/path/to/codebase", config)
```

### Registering Custom Analyzer

```python
from twinscribe.analysis import (
    register_analyzer,
    Analyzer,
    AnalyzerType,
    Language,
)

class CustomAnalyzer(Analyzer):
    async def check_available(self) -> bool:
        # Check if custom tool is installed
        ...

    async def analyze(self, codebase_path: Path) -> AnalyzerResult:
        # Run custom analysis
        ...

    def parse_output(self, raw_output: str) -> list[RawCallEdge]:
        # Parse custom output format
        ...

# Register with global registry
register_analyzer(
    AnalyzerType.CUSTOM,
    CustomAnalyzer,
    default_for_language=Language.CUSTOM,
)
```

## Implementation Notes

1. **Async Design**: All analysis operations are async to support parallel processing and non-blocking I/O.

2. **Subprocess Execution**: External tools (PyCG, pyan3) are executed via subprocess with timeout handling.

3. **Error Handling**: `AnalyzerError` includes exit code and stderr for debugging.

4. **Extensibility**: New analyzers can be added via the registry without modifying core code.

5. **Thread Safety**: The oracle is designed to be used from a single async context. Multiple coroutines can safely share the same oracle instance.

6. **Memory Management**: Large call graphs are loaded once and cached. Results are not duplicated when querying.

## Testing Strategy

1. **Unit Tests**:
   - Normalizer with various analyzer formats
   - Registry registration/lookup
   - Cache invalidation logic

2. **Integration Tests**:
   - Real PyCG/pyan3 execution on test codebases
   - Fallback behavior
   - End-to-end oracle usage

3. **Mock Analyzers**:
   - For testing without external tools
   - Predictable output for verification
