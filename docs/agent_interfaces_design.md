# Agent Interfaces Design Document

## Overview

This document describes the API contracts and interfaces for all agents in the Dual-Stream Documentation System. The design follows a tiered model architecture where different agents use models of varying cost and capability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocumentationStream                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Stream A (or B)                             │   │
│  │                                                                  │   │
│  │   ┌─────────────────┐           ┌─────────────────┐            │   │
│  │   │ DocumenterAgent │ ───────▶  │ ValidatorAgent  │            │   │
│  │   │     (A1/B1)     │           │    (A2/B2)      │            │   │
│  │   │                 │           │                 │            │   │
│  │   │ - Generate docs │           │ - Verify docs   │            │   │
│  │   │ - Extract calls │           │ - Check ground  │            │   │
│  │   │                 │           │   truth         │            │   │
│  │   └─────────────────┘           │ - Apply fixes   │            │   │
│  │                                 └─────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       ComparatorAgent         │
              │            (C)                │
              │                               │
              │ - Compare A vs B              │
              │ - Consult ground truth        │
              │ - Generate Beads tickets      │
              │ - Track convergence           │
              └───────────────────────────────┘
```

## Module Structure

```
src/twinscribe/agents/
├── __init__.py        # Public exports
├── base.py            # Base classes, configs, metrics
├── documenter.py      # DocumenterAgent interface
├── validator.py       # ValidatorAgent interface
├── comparator.py      # ComparatorAgent interface
└── stream.py          # DocumentationStream interface
```

## Base Classes

### AgentConfig

Configuration shared by all agents.

```python
class AgentConfig(BaseModel):
    agent_id: str              # "A1", "A2", "B1", "B2", "C"
    stream_id: Optional[StreamId]  # STREAM_A, STREAM_B, or None
    model_tier: ModelTier      # GENERATION, VALIDATION, ARBITRATION
    provider: str              # "anthropic", "openai"
    model_name: str            # Full model identifier
    cost_per_million_input: float
    cost_per_million_output: float
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_retries: int = 3
```

### AgentMetrics

Runtime metrics collected by all agents.

```python
@dataclass
class AgentMetrics:
    requests_made: int
    tokens_input: int
    tokens_output: int
    cost_total: float
    errors: int
    avg_latency_ms: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    def record_request(input_tokens, output_tokens, latency_ms, cost) -> None
    def record_error() -> None
```

### BaseAgent[InputT, OutputT]

Abstract base class for all agents.

```python
class BaseAgent(ABC, Generic[InputT, OutputT]):
    def __init__(config: AgentConfig) -> None

    @property config: AgentConfig
    @property metrics: AgentMetrics
    @property agent_id: str
    @property is_initialized: bool

    @abstractmethod
    async def initialize() -> None

    @abstractmethod
    async def process(input_data: InputT) -> OutputT

    @abstractmethod
    async def shutdown() -> None

    async def process_batch(inputs: list[InputT], concurrency: int = 5) -> list[OutputT]
    def reset_metrics() -> None
```

## DocumenterAgent (A1, B1)

**Purpose:** Generate comprehensive documentation with call graph linkages.

**Model Tier:** Generation (Claude Sonnet 4.5, GPT-4o)

### Input Contract

```python
class DocumenterInput(BaseModel):
    component: Component           # Component to document
    source_code: str              # Source code of component
    dependency_context: dict[str, DocumentationOutput]  # Already-documented deps
    static_analysis_hints: Optional[CallGraph]  # Hints for guidance
    iteration: int = 1            # Current iteration
    previous_output: Optional[DocumentationOutput]  # For re-documentation
    corrections: list[dict]       # Corrections to apply
```

### Output Contract

```python
DocumentationOutput:
    component_id: str
    documentation: ComponentDocumentation
    call_graph: CallGraphSection
    metadata: DocumenterMetadata
```

### Configuration

```python
class DocumenterConfig(AgentConfig):
    include_examples: bool = True
    max_example_count: int = 3
    include_see_also: bool = True
    confidence_threshold: float = 0.7
```

### Default Configurations

| Agent | Model | Provider | Input Cost | Output Cost |
|-------|-------|----------|------------|-------------|
| A1 | claude-sonnet-4-5-20250929 | anthropic | $3.00/M | $15.00/M |
| B1 | gpt-4o | openai | $2.50/M | $10.00/M |

### Interface

```python
class DocumenterAgent(BaseAgent[DocumenterInput, DocumentationOutput]):
    SYSTEM_PROMPT: str  # Template for system instructions

    async def initialize() -> None
    async def process(input_data: DocumenterInput) -> DocumentationOutput
    async def shutdown() -> None

    # Helper methods
    def _build_user_prompt(input_data: DocumenterInput) -> str
    def _get_response_schema() -> dict
```

## ValidatorAgent (A2, B2)

**Purpose:** Verify documentation completeness and call graph accuracy.

**Model Tier:** Validation (Claude Haiku 4.5, GPT-4o-mini)

### Input Contract

```python
class ValidatorInput(BaseModel):
    documentation: DocumentationOutput  # Doc to validate
    source_code: str                   # Original source
    ground_truth_call_graph: CallGraph # Static analysis (authoritative)
    component_ast: Optional[dict]      # AST for detailed checks
```

### Output Contract

```python
ValidationResult:
    component_id: str
    validation_result: ValidationStatus  # PASS/FAIL/WARNING
    completeness: CompletenessCheck
    call_graph_accuracy: CallGraphAccuracy
    corrections_applied: list[CorrectionApplied]
    metadata: ValidatorMetadata
```

### Configuration

```python
class ValidatorConfig(AgentConfig):
    auto_correct: bool = True
    false_positive_threshold: float = 0.05
    missing_callee_threshold: float = 0.10
    missing_caller_threshold: float = 0.20
    require_all_parameters: bool = True
    require_return_doc: bool = True
    require_exception_doc: bool = True
```

### Default Configurations

| Agent | Model | Provider | Input Cost | Output Cost |
|-------|-------|----------|------------|-------------|
| A2 | claude-haiku-4-5-20251001 | anthropic | $0.25/M | $1.25/M |
| B2 | gpt-4o-mini | openai | $0.15/M | $0.60/M |

### Interface

```python
class ValidatorAgent(BaseAgent[ValidatorInput, ValidationResult]):
    SYSTEM_PROMPT: str

    async def initialize() -> None
    async def process(input_data: ValidatorInput) -> ValidationResult
    async def shutdown() -> None

    def _build_user_prompt(input_data: ValidatorInput) -> str
    def _get_response_schema() -> dict
```

## ComparatorAgent (C)

**Purpose:** Compare streams, resolve discrepancies, generate Beads tickets.

**Model Tier:** Arbitration (Claude Opus 4.5)

### Input Contract

```python
class ComparatorInput(BaseModel):
    stream_a_output: StreamOutput     # Stream A validated output
    stream_b_output: StreamOutput     # Stream B validated output
    ground_truth_call_graph: CallGraph
    iteration: int = 1
    previous_comparison: Optional[ComparisonResult]
    resolved_discrepancies: list[str]
```

### Output Contract

```python
ComparisonResult:
    comparison_id: str
    iteration: int
    summary: ComparisonSummary
    discrepancies: list[Discrepancy]
    convergence_status: ConvergenceStatus
    metadata: ComparatorMetadata
```

### Configuration

```python
class ComparatorConfig(AgentConfig):
    confidence_threshold: float = 0.7
    semantic_similarity_threshold: float = 0.95
    generate_beads_tickets: bool = True
    beads_project: str = "LEGACY_DOC"
    beads_ticket_priority_default: str = "Medium"
```

### Default Configuration

| Agent | Model | Provider | Input Cost | Output Cost |
|-------|-------|----------|------------|-------------|
| C | claude-opus-4-5-20251101 | anthropic | $15.00/M | $75.00/M |

### Interface

```python
class ComparatorAgent(BaseAgent[ComparatorInput, ComparisonResult]):
    SYSTEM_PROMPT: str

    async def initialize() -> None
    async def process(input_data: ComparatorInput) -> ComparisonResult
    async def shutdown() -> None

    async def compare_component(
        component_id: str,
        stream_a_doc: Optional[dict],
        stream_b_doc: Optional[dict],
        ground_truth: CallGraph,
    ) -> list[Discrepancy]

    async def generate_beads_ticket(
        discrepancy: Discrepancy,
        stream_a_model: str,
        stream_b_model: str,
        source_code: str,
    ) -> dict
```

### Decision Logic

```
FOR each component:
    IF stream_a == stream_b:
        -> ACCEPT (identical)

    ELSE IF discrepancy is call_graph_related:
        -> CONSULT static analysis ground truth
        -> ACCEPT matching stream
        -> LOG correction

    ELSE IF discrepancy is documentation_content:
        IF one stream clearly better (high confidence):
            -> ACCEPT better stream
        ELSE:
            -> GENERATE Beads ticket

    ELSE:
        -> GENERATE Beads ticket
```

## DocumentationStream

**Purpose:** Manage a documenter/validator pair and process components.

### Configuration

```python
class StreamConfig(BaseModel):
    stream_id: StreamId
    documenter_config: DocumenterConfig
    validator_config: ValidatorConfig
    batch_size: int = 5
    max_retries: int = 3
    continue_on_error: bool = True
```

### Results

```python
@dataclass
class StreamResult:
    stream_id: StreamId
    output: StreamOutput
    successful: int
    failed: int
    failed_component_ids: list[str]
    total_tokens: int
    total_cost: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

@dataclass
class ComponentProcessingResult:
    component_id: str
    documentation: Optional[DocumentationOutput]
    validation: Optional[ValidationResult]
    success: bool
    error: Optional[str]
    retries: int
```

### Interface

```python
class DocumentationStream(ABC):
    def __init__(config: StreamConfig) -> None

    @property config: StreamConfig
    @property stream_id: StreamId
    @property output: StreamOutput
    @property is_initialized: bool

    @abstractmethod
    async def initialize() -> None

    @abstractmethod
    async def process(
        components: list[Component],
        source_code_map: dict[str, str],
        ground_truth: CallGraph,
    ) -> StreamResult

    @abstractmethod
    async def process_component(
        component: Component,
        source_code: str,
        dependency_context: dict[str, DocumentationOutput],
        ground_truth: CallGraph,
    ) -> ComponentProcessingResult

    @abstractmethod
    async def apply_correction(
        component_id: str,
        corrected_value: any,
        field_path: str,
    ) -> bool

    @abstractmethod
    async def reprocess_component(
        component_id: str,
        corrections: list[dict],
    ) -> ComponentProcessingResult

    @abstractmethod
    async def shutdown() -> None

    def get_documentation(component_id: str) -> Optional[DocumentationOutput]
    def get_all_component_ids() -> list[str]
    def reset() -> None
```

### Progress Callback

```python
class StreamProgressCallback:
    async def on_component_start(stream_id, component_id, index, total) -> None
    async def on_component_complete(stream_id, component_id, success, duration_ms) -> None
    async def on_validation_complete(stream_id, component_id, passed, corrections) -> None
    async def on_stream_complete(stream_id, result) -> None
    async def on_error(stream_id, component_id, error) -> None
```

## LLM Client Interface

All agents communicate with LLMs through a common interface:

```python
class LLMClient(ABC):
    @abstractmethod
    async def complete(
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[dict] = None,
    ) -> LLMResponse

    @abstractmethod
    async def close() -> None

@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    finish_reason: str = "stop"
```

## Usage Examples

### Processing a Single Component

```python
from twinscribe.agents import (
    DocumenterAgent,
    ValidatorAgent,
    DocumenterInput,
    ValidatorInput,
    STREAM_A_DOCUMENTER_CONFIG,
    STREAM_A_VALIDATOR_CONFIG,
)

# Initialize agents
documenter = ConcreteDocumenterAgent(STREAM_A_DOCUMENTER_CONFIG)
validator = ConcreteValidatorAgent(STREAM_A_VALIDATOR_CONFIG)

await documenter.initialize()
await validator.initialize()

# Document a component
doc_input = DocumenterInput(
    component=component,
    source_code=source_code,
    dependency_context={},
)
doc_output = await documenter.process(doc_input)

# Validate the documentation
val_input = ValidatorInput(
    documentation=doc_output,
    source_code=source_code,
    ground_truth_call_graph=ground_truth,
)
val_result = await validator.process(val_input)

# Cleanup
await documenter.shutdown()
await validator.shutdown()
```

### Using a Documentation Stream

```python
from twinscribe.agents import (
    DocumentationStream,
    StreamConfig,
    STREAM_A_DOCUMENTER_CONFIG,
    STREAM_A_VALIDATOR_CONFIG,
)
from twinscribe.models.base import StreamId

config = StreamConfig(
    stream_id=StreamId.STREAM_A,
    documenter_config=STREAM_A_DOCUMENTER_CONFIG,
    validator_config=STREAM_A_VALIDATOR_CONFIG,
    batch_size=5,
)

stream = ConcreteDocumentationStream(config)
await stream.initialize()

result = await stream.process(
    components=components,  # In topological order
    source_code_map=source_code_map,
    ground_truth=ground_truth,
)

print(f"Processed {result.successful} components")
print(f"Failed: {result.failed}")
print(f"Total cost: ${result.total_cost:.2f}")

await stream.shutdown()
```

### Running the Comparator

```python
from twinscribe.agents import (
    ComparatorAgent,
    ComparatorInput,
    COMPARATOR_CONFIG,
)

comparator = ConcreteComparatorAgent(COMPARATOR_CONFIG)
await comparator.initialize()

comparison_input = ComparatorInput(
    stream_a_output=stream_a_result.output,
    stream_b_output=stream_b_result.output,
    ground_truth_call_graph=ground_truth,
    iteration=1,
)

comparison = await comparator.process(comparison_input)

print(f"Identical: {comparison.summary.identical}")
print(f"Discrepancies: {comparison.summary.discrepancies}")
print(f"Converged: {comparison.convergence_status.converged}")

await comparator.shutdown()
```

## Implementation Notes

1. **Async First**: All agent operations are async to support concurrent processing.

2. **Structured Output**: Agents use JSON schema constraints for reliable structured output from LLMs.

3. **Retry Logic**: Agents handle transient failures with configurable retry logic.

4. **Metrics Collection**: All agents collect metrics for cost tracking and monitoring.

5. **Progress Callbacks**: Streams support progress callbacks for UI integration.

6. **Correction Flow**: The system supports applying corrections and re-processing components.

7. **Batch Processing**: Streams can process multiple components in parallel batches.
