# Data Models Design Document

## Overview

This document describes the Pydantic data models for the Dual-Stream Code Documentation System. All models support JSON serialization and include comprehensive validation.

## Model Architecture

```
                                    ┌─────────────────┐
                                    │    Component    │
                                    │  (discovered)   │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
           ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
           │ Documentation  │      │  CallGraph     │      │  Validation    │
           │    Output      │      │  (ground truth)│      │    Result      │
           │ (per stream)   │      │                │      │ (per stream)   │
           └───────┬────────┘      └────────┬───────┘      └───────┬────────┘
                   │                        │                       │
                   └────────────────────────┼───────────────────────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │  Comparison    │
                                   │    Result      │
                                   │ (discrepancies)│
                                   └───────┬────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
           ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
           │ Discrepancy    │    │  Convergence   │    │ Documentation  │
           │   Ticket       │    │    Report      │    │   Package      │
           │ (Beads)        │    │                │    │ (final output) │
           └────────────────┘    └────────────────┘    └────────────────┘
```

## Module Structure

```
src/twinscribe/models/
├── __init__.py          # Public exports
├── base.py              # Enumerations and constants
├── components.py        # Code component models
├── call_graph.py        # Call graph models
├── documentation.py     # Documenter agent outputs
├── validation.py        # Validator agent outputs
├── comparison.py        # Comparator agent outputs
├── convergence.py       # Convergence tracking
├── output.py            # Final output package
└── beads.py             # Beads ticket models
```

## Model Reference

### Base Enumerations (`base.py`)

| Enum | Description | Values |
|------|-------------|--------|
| `ComponentType` | Type of code entity | FUNCTION, METHOD, CLASS, MODULE, PROPERTY, STATICMETHOD, CLASSMETHOD |
| `CallType` | Type of call relationship | DIRECT, CONDITIONAL, LOOP, EXCEPTION, CALLBACK, DYNAMIC |
| `DiscrepancyType` | Category of stream difference | CALL_GRAPH_EDGE, CALL_SITE_LINE, DOCUMENTATION_CONTENT, etc. |
| `ValidationStatus` | Validation result | PASS, FAIL, WARNING |
| `ResolutionAction` | Discrepancy resolution | ACCEPT_STREAM_A, ACCEPT_STREAM_B, ACCEPT_GROUND_TRUTH, etc. |
| `ModelTier` | LLM cost tier | GENERATION, VALIDATION, ARBITRATION |
| `StreamId` | Stream identifier | STREAM_A, STREAM_B |

### Component Models (`components.py`)

#### `Component`
Primary entity being documented. Discovered through AST analysis.

```python
class Component(BaseModel):
    component_id: str        # Unique ID: "module.Class.method"
    name: str                # Short name: "method"
    type: ComponentType      # FUNCTION, METHOD, CLASS, etc.
    location: ComponentLocation
    signature: Optional[str] # Full signature string
    parent_id: Optional[str] # Containing component
    dependencies: list[str]  # Import-level dependencies
    existing_docstring: Optional[str]
    is_public: bool
    created_at: datetime
```

#### `ComponentDocumentation`
Documentation content structure.

```python
class ComponentDocumentation(BaseModel):
    summary: str             # One-line description
    description: str         # Detailed explanation
    parameters: list[ParameterDoc]
    returns: Optional[ReturnDoc]
    raises: list[ExceptionDoc]
    examples: list[str]
    notes: Optional[str]
    see_also: list[str]
```

### Call Graph Models (`call_graph.py`)

#### `CallEdge`
Represents a single call relationship.

```python
class CallEdge(BaseModel):
    caller: str              # Component ID of caller
    callee: str              # Component ID of callee
    call_site_line: Optional[int]
    call_type: CallType
    confidence: float        # 1.0 for static analysis
```

#### `CallGraph`
Collection of edges with graph operations.

```python
class CallGraph(BaseModel):
    edges: list[CallEdge]
    source: str              # Origin: "pycg", "agent_A1", etc.

    # Methods:
    def get_callees(component_id: str) -> list[CallEdge]
    def get_callers(component_id: str) -> list[CallEdge]
    def has_edge(caller: str, callee: str) -> bool
    def merge_with(other: CallGraph) -> CallGraph
```

#### `CallGraphDiff`
Comparison result between call graphs.

```python
class CallGraphDiff(BaseModel):
    missing_in_doc: set[tuple[str, str]]
    extra_in_doc: set[tuple[str, str]]
    matching: set[tuple[str, str]]
    precision: float
    recall: float

    @computed_field
    def f1_score(self) -> float
```

### Documentation Output Models (`documentation.py`)

#### `DocumentationOutput`
Output from documenter agents (A1, B1). **Spec section 3.1**

```python
class DocumentationOutput(BaseModel):
    component_id: str
    documentation: ComponentDocumentation
    call_graph: CallGraphSection
    metadata: DocumenterMetadata
```

#### `CallGraphSection`
Call relationships for a single component.

```python
class CallGraphSection(BaseModel):
    callers: list[CallerRef]
    callees: list[CalleeRef]
```

### Validation Models (`validation.py`)

#### `ValidationResult`
Output from validator agents (A2, B2). **Spec section 3.2**

```python
class ValidationResult(BaseModel):
    component_id: str
    validation_result: ValidationStatus  # PASS/FAIL/WARNING
    completeness: CompletenessCheck
    call_graph_accuracy: CallGraphAccuracy
    corrections_applied: list[CorrectionApplied]
    metadata: ValidatorMetadata
```

#### `CallGraphAccuracy`
Validation against static analysis ground truth.

```python
class CallGraphAccuracy(BaseModel):
    score: float
    verified_callees: list[str]
    missing_callees: list[str]
    false_callees: list[str]
    verified_callers: list[str]
    missing_callers: list[str]
    false_callers: list[str]
```

### Comparison Models (`comparison.py`)

#### `ComparisonResult`
Output from comparator agent (C). **Spec section 3.3**

```python
class ComparisonResult(BaseModel):
    comparison_id: str
    iteration: int
    summary: ComparisonSummary
    discrepancies: list[Discrepancy]
    convergence_status: ConvergenceStatus
    metadata: ComparatorMetadata
```

#### `Discrepancy`
A difference between streams.

```python
class Discrepancy(BaseModel):
    discrepancy_id: str
    component_id: str
    type: DiscrepancyType
    stream_a_value: Optional[Any]
    stream_b_value: Optional[Any]
    ground_truth: Optional[Any]
    resolution: ResolutionAction
    confidence: float
    requires_beads: bool
    beads_ticket: Optional[BeadsTicketRef]

    @computed_field
    def is_call_graph_related(self) -> bool

    @computed_field
    def is_blocking(self) -> bool
```

### Convergence Models (`convergence.py`)

#### `ConvergenceCriteria`
Thresholds for convergence. **Spec section 4.2**

```python
class ConvergenceCriteria(BaseModel):
    max_iterations: int = 5
    call_graph_match_rate: float = 0.98
    documentation_similarity: float = 0.95
    max_open_discrepancies: int = 2
    blocking_discrepancy_types: list[str]
```

#### `ConvergenceReport`
Complete convergence process report.

```python
class ConvergenceReport(BaseModel):
    total_iterations: int
    final_status: str  # "converged" | "max_iterations_reached"
    history: list[ConvergenceHistoryEntry]
    criteria: ConvergenceCriteria
    forced_convergence: bool
    remaining_discrepancies: list[str]
```

### Output Models (`output.py`)

#### `DocumentationPackage`
Final system output.

```python
class DocumentationPackage(BaseModel):
    documentation: dict[str, ComponentFinalDoc]
    call_graph: CallGraph
    rebuild_tickets: list[dict]
    convergence_report: ConvergenceReport
    metrics: RunMetrics
    version: str = "2.0.0"
```

#### `RunMetrics`
Comprehensive run statistics.

```python
class RunMetrics(BaseModel):
    run_id: str
    codebase_path: str
    language: str
    components_total: int
    components_documented: int
    call_graph_precision: float
    call_graph_recall: float
    call_graph_f1: float
    discrepancies_total: int
    discrepancies_resolved_auto: int
    discrepancies_resolved_beads: int
    cost: CostBreakdown
    tokens_total: int
```

### Beads Models (`beads.py`)

#### `DiscrepancyTicket`
Ticket for human review. **Spec section 5.1**

```python
class DiscrepancyTicket(BaseModel):
    project: str = "LEGACY_DOC"
    issue_type: BeadsTicketType = CLARIFICATION
    priority: BeadsTicketPriority
    summary: str
    component_id: str
    differences: list[StreamComparison]
    ground_truth_available: bool
    ground_truth_value: Optional[str]
    source_code_snippet: str
    discrepancy_id: str
    ticket_key: Optional[str]  # Set after creation

    def to_beads_payload(self) -> dict
```

#### `RebuildTicket`
Ticket for component rebuild. **Spec section 5.1**

```python
class RebuildTicket(BaseModel):
    project: str = "REBUILD"
    issue_type: BeadsTicketType = STORY
    summary: str
    component_id: str
    confidence_score: int
    documentation_summary: str
    documentation_description: str
    parameters: list[dict]
    callees: list[dict]
    callers: list[dict]
    acceptance_criteria: list[str]

    def to_beads_payload(self) -> dict
```

## JSON Serialization

All models use Pydantic v2 and support:

```python
# Serialize to JSON string
model.model_dump_json(indent=2)

# Serialize to dict
model.model_dump()

# Deserialize from JSON
Model.model_validate_json(json_string)

# Deserialize from dict
Model.model_validate(data_dict)
```

## Validation Rules

### Component IDs
- Format: `module.Class.method` or `module.function`
- Minimum 1 character
- Valid Python identifiers (with dots)

### Scores and Rates
- All scores: 0.0 to 1.0
- Confidence levels: 0.0 to 1.0
- Percentages (confidence_score): 0 to 100

### Line Numbers
- All line numbers: >= 1 (1-indexed)
- line_end >= line_start

### Required vs Optional
- `component_id` always required
- `metadata` always required on agent outputs
- Type annotations optional (inferred types supported)
- Default values provided where sensible

## Usage Examples

### Creating a Component

```python
from twinscribe.models import (
    Component, ComponentType, ComponentLocation
)

component = Component(
    component_id="mypackage.utils.format_name",
    name="format_name",
    type=ComponentType.FUNCTION,
    location=ComponentLocation(
        file_path="src/utils.py",
        line_start=45,
        line_end=52
    ),
    signature="def format_name(first: str, last: str) -> str"
)
```

### Creating Documentation Output

```python
from twinscribe.models import (
    DocumentationOutput,
    ComponentDocumentation,
    ParameterDoc,
    ReturnDoc,
    CallGraphSection,
    CalleeRef,
    DocumenterMetadata,
    StreamId
)

output = DocumentationOutput(
    component_id="mypackage.utils.format_name",
    documentation=ComponentDocumentation(
        summary="Format a full name from first and last name.",
        description="Combines first and last name with proper capitalization.",
        parameters=[
            ParameterDoc(name="first", type="str", description="First name"),
            ParameterDoc(name="last", type="str", description="Last name")
        ],
        returns=ReturnDoc(type="str", description="Formatted full name")
    ),
    call_graph=CallGraphSection(
        callees=[
            CalleeRef(component_id="str.title", call_site_line=48)
        ]
    ),
    metadata=DocumenterMetadata(
        agent_id="A1",
        stream_id=StreamId.STREAM_A,
        model="claude-sonnet-4-5-20250929",
        confidence=0.95
    )
)
```

### Comparing Call Graphs

```python
from twinscribe.models import CallGraph, CallEdge, CallGraphDiff

ground_truth = CallGraph(
    edges=[
        CallEdge(caller="a.foo", callee="b.bar"),
        CallEdge(caller="a.foo", callee="c.baz")
    ],
    source="pycg"
)

documented = CallGraph(
    edges=[
        CallEdge(caller="a.foo", callee="b.bar"),
        CallEdge(caller="a.foo", callee="d.wrong")  # False positive
    ],
    source="agent_A1"
)

diff = CallGraphDiff.compute(ground_truth, documented)
print(f"Precision: {diff.precision}")  # 0.5
print(f"Recall: {diff.recall}")        # 0.5
print(f"Missing: {diff.missing_in_doc}")  # {('a.foo', 'c.baz')}
print(f"Extra: {diff.extra_in_doc}")      # {('a.foo', 'd.wrong')}
```

## Implementation Notes

1. **Pydantic v2**: All models use Pydantic v2 features including `computed_field` and `field_validator`.

2. **Type Safety**: Full type hints throughout. Use `mypy` for static checking.

3. **Immutability**: Models are mutable by default but can be made immutable with `model_config = {"frozen": True}`.

4. **JSON Compatibility**: All types are JSON-serializable. Sets are converted to lists during serialization.

5. **Extensibility**: Models can be subclassed for custom behavior.

## Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
]
```
