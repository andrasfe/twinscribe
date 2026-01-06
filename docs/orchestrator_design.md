# Orchestrator Design Document

## Overview

The orchestrator module coordinates the entire dual-stream documentation pipeline. It manages the flow from codebase analysis through convergence to final output generation, handling state management, checkpointing, and recovery.

## Architecture

```
+-------------------------------------------------------------------------+
|                         DualStreamOrchestrator                          |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                        INITIALIZATION                             |  |
|  |                                                                   |  |
|  |  1. Parse codebase -> AST                                        |  |
|  |  2. Build dependency graph                                       |  |
|  |  3. Run static analysis -> Ground truth call graph               |  |
|  |  4. Topological sort -> Processing order                         |  |
|  |  5. Initialize streams A and B                                   |  |
|  +-------------------------------------------------------------------+  |
|                                    |                                    |
|                                    v                                    |
|  +-------------------------------------------------------------------+  |
|  |                        ITERATION LOOP                             |  |
|  |                                                                   |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  | STEP 1: PARALLEL DOCUMENTATION                               |  |  |
|  |  |                                                              |  |  |
|  |  |   Stream A              asyncio.gather              Stream B |  |  |
|  |  |   [Documenter A1]  <---------------------> [Documenter B1]   |  |  |
|  |  |   [Validator A2]                           [Validator B2]    |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |                                |                                  |  |
|  |                                v                                  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  | STEP 2: COMPARISON                                           |  |  |
|  |  |                                                              |  |  |
|  |  |   [ComparatorAgent] + [StaticAnalysisOracle]                |  |  |
|  |  |   - Structural comparison (call graphs)                      |  |  |
|  |  |   - Semantic comparison (documentation)                      |  |  |
|  |  |   - Ground truth consultation                                |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |                                |                                  |  |
|  |                                v                                  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  | STEP 3: RESOLUTION                                           |  |  |
|  |  |                                                              |  |  |
|  |  |   [ConvergenceChecker]                                       |  |  |
|  |  |   IF converged -> EXIT                                       |  |  |
|  |  |   ELSE IF ground truth resolvable -> Apply corrections       |  |  |
|  |  |   ELSE -> Create Beads tickets, wait for resolution          |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |                                |                                  |  |
|  |                                v                                  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  | STEP 4: ITERATION CHECK                                      |  |  |
|  |  |                                                              |  |  |
|  |  |   IF iteration >= MAX_ITERATIONS -> Force convergence        |  |  |
|  |  |   ELSE -> Increment iteration, return to Step 1              |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |                                                                   |  |
|  +-------------------------------------------------------------------+  |
|                                    |                                    |
|                                    v                                    |
|  +-------------------------------------------------------------------+  |
|  |                        FINALIZATION                               |  |
|  |                                                                   |  |
|  |  1. Merge converged outputs                                      |  |
|  |  2. Generate final documentation package                         |  |
|  |  3. Generate Beads rebuild tickets                               |  |
|  |  4. Produce convergence report                                   |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Module Structure

```
src/twinscribe/orchestrator/
+-- __init__.py          # Public exports
+-- orchestrator.py      # Main orchestrator class
+-- convergence.py       # Convergence criteria and checking
+-- state.py             # State management and checkpointing
```

## Component Details

### DualStreamOrchestrator (orchestrator.py)

Main orchestrator coordinating all pipeline phases.

```python
class DualStreamOrchestrator:
    """Main orchestrator for the dual-stream documentation system."""

    def __init__(
        config: OrchestratorConfig,
        static_oracle: StaticAnalysisOracle,
        stream_a: DocumentationStream,
        stream_b: DocumentationStream,
        comparator: ComparatorAgent,
        beads_manager: Optional[BeadsLifecycleManager] = None,
    )

    async def run() -> DocumentationPackage
    def on_progress(callback: ProgressCallback) -> None
```

#### Configuration

```python
class OrchestratorConfig(BaseModel):
    max_iterations: int = 5          # Maximum documentation iterations
    parallel_components: int = 10     # Concurrent component processing
    wait_for_beads: bool = True       # Wait for Beads resolution
    beads_timeout_hours: int = 48     # Timeout for Beads tickets
    skip_validation: bool = False     # Skip validation (testing)
    dry_run: bool = False             # No external changes
    continue_on_error: bool = True    # Continue on component errors
```

#### Execution Phases

| Phase | Description | State |
|-------|-------------|-------|
| `NOT_STARTED` | Orchestrator created but not run | Initial |
| `INITIALIZING` | Setting up components | Setup |
| `DISCOVERING` | Finding components to document | Setup |
| `DOCUMENTING` | Running documentation streams | Iteration |
| `COMPARING` | Comparing stream outputs | Iteration |
| `RESOLVING` | Resolving discrepancies | Iteration |
| `FINALIZING` | Generating final output | Cleanup |
| `COMPLETED` | Pipeline finished successfully | Terminal |
| `FAILED` | Pipeline failed | Terminal |

#### State Tracking

```python
@dataclass
class OrchestratorState:
    phase: OrchestratorPhase
    iteration: int
    total_components: int
    processed_components: int
    converged_components: int
    pending_discrepancies: int
    beads_tickets_open: int
    start_time: Optional[datetime]
    errors: list[str]
```

### ConvergenceChecker (convergence.py)

Checks if streams have converged based on configurable criteria.

```python
class ConvergenceChecker:
    """Checks convergence criteria against comparison results."""

    def __init__(criteria: ConvergenceCriteria)
    def check(comparison: ComparisonResult, iteration: int) -> ConvergenceCheck
    def get_required_actions(check: ConvergenceCheck) -> list[str]
```

#### Convergence Criteria

```python
class ConvergenceCriteria(BaseModel):
    max_iterations: int = 5                        # Hard iteration limit
    call_graph_match_threshold: float = 0.98       # 98% edge match
    documentation_similarity_threshold: float = 0.95  # 95% similarity
    max_open_discrepancies: int = 2                # Max non-blocking
    beads_ticket_timeout_hours: int = 48           # Beads timeout
    blocking_types: list[BlockingDiscrepancyType]  # Blocking types
    require_static_validation: bool = True         # Require validation
```

#### Blocking Discrepancy Types

| Type | Description |
|------|-------------|
| `MISSING_CRITICAL_CALL` | Call exists in code but undocumented |
| `FALSE_CRITICAL_CALL` | Documented call doesn't exist |
| `MISSING_PUBLIC_API_DOC` | Public method undocumented |
| `SECURITY_RELEVANT_GAP` | Security-sensitive code undocumented |
| `TYPE_MISMATCH` | Parameter/return type mismatch |
| `SIGNATURE_MISMATCH` | Function signature mismatch |

#### Convergence Status

```
NOT_STARTED
     |
     v
IN_PROGRESS <--+
     |         |
     v         |
+----+----+    |
|    |    |    |
v    v    v    |
CONVERGED  PARTIALLY_CONVERGED --+
           |
           v
         FORCED (max iterations)
```

### CheckpointManager (state.py)

Manages state persistence for recovery.

```python
class CheckpointManager:
    """Manages orchestrator checkpoints."""

    def __init__(
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10,
        auto_checkpoint: bool = True,
    )

    def create_checkpoint(orchestrator, checkpoint_id=None) -> str
    def load_checkpoint(checkpoint_id: str) -> Checkpoint
    def list_checkpoints() -> list[str]
    def get_latest_checkpoint() -> Optional[str]
    def delete_checkpoint(checkpoint_id: str) -> bool
```

#### Checkpoint Structure

```python
class Checkpoint(BaseModel):
    checkpoint_id: str
    created_at: datetime
    iteration: int
    phase: str
    component_states: dict[str, dict]
    stream_a_state: dict
    stream_b_state: dict
    comparison_results: list[dict]
    beads_tickets: dict
    metadata: dict
```

### ProgressTracker (state.py)

Tracks detailed progress and provides estimates.

```python
class ProgressTracker:
    """Tracks detailed progress for the orchestrator."""

    def start() -> None
    def record_component(component_id: str, duration_seconds: float) -> None
    def record_iteration(duration_seconds: float) -> None
    def estimate_remaining(remaining_components, remaining_iterations) -> Optional[float]
    def get_throughput() -> Optional[float]
    def get_summary() -> dict
```

## Data Flow

### Iteration Flow

```
Components to Process
         |
         v
+--------+--------+
|                 |
v                 v
Stream A         Stream B
[Document]       [Document]
[Validate]       [Validate]
|                 |
+--------+--------+
         |
         v
    Comparison
         |
         v
    Discrepancies?
         |
    +----+----+
    |         |
    v         v
Ground Truth  Beads Tickets
Resolution    (wait for human)
    |         |
    +----+----+
         |
         v
    Apply Corrections
         |
         v
    Convergence Check
         |
    +----+----+
    |         |
    v         v
CONVERGED    NOT CONVERGED
    |         |
    v         v
 Finalize   Next Iteration
```

### Component Processing Order

Components are processed in topological order based on the dependency graph:

1. Build dependency graph from static analysis call graph
2. Reverse edges (callees before callers)
3. Apply Kahn's algorithm for topological sort
4. Handle cycles by appending remaining components

```python
def _build_processing_order(self) -> list[str]:
    """Sort components so dependencies are processed first."""
    # A calls B means B should be documented first
    # So we reverse the call graph edges for ordering
```

## Integration Points

### With Static Analysis

```python
# Initialization
await oracle.initialize()
call_graph = oracle.call_graph

# During comparison
ground_truth = oracle.verify_edge(caller, callee)
diff = oracle.diff_against(documented_graph)
```

### With Documentation Streams

```python
# Parallel execution
output_a, output_b = await asyncio.gather(
    stream_a.process(components, processing_order),
    stream_b.process(components, processing_order),
)

# Apply corrections
await stream_a.apply_correction(component_id, field, value)
await stream_b.apply_correction(component_id, field, value)
```

### With Comparator

```python
# Compare outputs
comparison = await comparator.compare(output_a, output_b, ground_truth)

# Access results
for discrepancy in comparison.discrepancies:
    if discrepancy.resolution_source == "ground_truth":
        # Apply automatically
    elif discrepancy.requires_human_review:
        # Create Beads ticket
```

### With Beads Manager

```python
# Create ticket
ticket = await beads_manager.create_discrepancy_ticket(template_data)

# Wait for resolution
resolution = await beads_manager.wait_for_resolution(
    ticket.ticket_key,
    timeout_seconds=48*3600,
)

# Apply resolution
await beads_manager.apply_resolution(resolution, apply_func)
```

## Error Handling

### Component-Level Errors

When `continue_on_error=True`:
- Log error and continue with next component
- Track failed components in state
- Include in final report

### Stream-Level Errors

- Retry with exponential backoff
- If persistent, fail the stream
- Option to continue with single stream

### Beads Timeout

When Beads tickets timeout:
- Mark as expired in tracker
- Continue to next iteration
- Log unresolved discrepancies
- Include in convergence report

## Progress Callbacks

Register callbacks to monitor progress:

```python
def progress_handler(state: OrchestratorState):
    print(f"Phase: {state.phase}")
    print(f"Iteration: {state.iteration}")
    print(f"Progress: {state.processed_components}/{state.total_components}")

orchestrator.on_progress(progress_handler)
await orchestrator.run()
```

## Checkpointing and Recovery

### Automatic Checkpointing

```python
# Enable auto-checkpointing
manager = CheckpointManager(auto_checkpoint=True)

# Checkpoints created at:
# - End of each iteration
# - Before Beads wait
# - After major phase changes
```

### Manual Recovery

```python
# Recover from latest checkpoint
recovery = StateRecovery(checkpoint_manager)
await recovery.recover(orchestrator)

# Or from specific checkpoint
await recovery.recover(orchestrator, "checkpoint_20260106_120000")
```

## Output

### DocumentationPackage

```python
@dataclass
class DocumentationPackage:
    components: list[ComponentFinalDoc]  # Final documentation
    call_graph: CallGraph                # Merged call graph
    rebuild_tickets: list[str]           # Beads ticket keys
    convergence_report: ConvergenceReport
    metrics: dict[str, Any]
```

### Convergence Report

```python
@dataclass
class ConvergenceReport:
    converged: bool
    total_iterations: int
    final_status: str
    history: list[ConvergenceHistoryEntry]
    trends: dict[str, list[float]]
    recommendations: list[str]
```

## Usage Examples

### Basic Usage

```python
from twinscribe.orchestrator import (
    DualStreamOrchestrator,
    OrchestratorConfig,
)

# Configure
config = OrchestratorConfig(
    max_iterations=5,
    wait_for_beads=True,
)

# Create orchestrator
orchestrator = DualStreamOrchestrator(
    config=config,
    static_oracle=oracle,
    stream_a=stream_a,
    stream_b=stream_b,
    comparator=comparator,
    beads_manager=beads_manager,
)

# Run pipeline
result = await orchestrator.run()

# Access results
print(f"Components: {len(result.components)}")
print(f"Converged: {result.convergence_report.converged}")
```

### With Progress Tracking

```python
from twinscribe.orchestrator import ProgressTracker

tracker = ProgressTracker()

def on_progress(state):
    tracker.record_component(state.current_component, state.component_time)
    remaining = tracker.estimate_remaining(
        state.total_components - state.processed_components,
        config.max_iterations - state.iteration,
    )
    print(f"ETA: {remaining:.0f}s")

orchestrator.on_progress(on_progress)
```

### With Checkpointing

```python
from twinscribe.orchestrator import CheckpointManager, StateRecovery

# Setup checkpointing
checkpoint_mgr = CheckpointManager("./checkpoints")

# Create checkpoint manually
checkpoint_id = checkpoint_mgr.create_checkpoint(orchestrator)

# Later, recover from checkpoint
recovery = StateRecovery(checkpoint_mgr)
success = await recovery.recover(orchestrator)
if success:
    # Continue from where we left off
    result = await orchestrator.run()
```

### Dry Run Mode

```python
config = OrchestratorConfig(
    dry_run=True,  # No Beads tickets, no file writes
)

# Useful for testing and validation
result = await orchestrator.run()
```

## Performance Considerations

1. **Parallel Processing**: Components processed in parallel within each stream
2. **Incremental Processing**: Only changed components re-processed after iteration 1
3. **Checkpointing Overhead**: Checkpoints written asynchronously
4. **Memory**: Component results cached in memory; consider streaming for large codebases
5. **Beads Polling**: Configurable poll interval to balance latency vs. API load

## Testing Strategy

### Unit Tests
- Convergence criteria checking
- Topological sort correctness
- Checkpoint serialization/deserialization
- State transitions

### Integration Tests
- Full pipeline with mock components
- Checkpoint recovery
- Beads integration (mock API)
- Error handling scenarios

### End-to-End Tests
- Real codebase documentation
- Multi-iteration convergence
- Beads ticket workflow
