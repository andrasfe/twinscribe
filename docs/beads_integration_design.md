# Beads Integration Design Document

## Overview

The Beads integration module provides human-in-the-loop resolution for edge cases that cannot be automatically resolved through ground truth validation or model consensus. It integrates with Beads, a git-backed issue tracker designed for AI agents, to create, monitor, and apply resolutions from human reviewers.

Beads stores issues as JSONL files in a `.beads/` directory, using git as the underlying database. This allows tasks to be versioned, branched, and merged alongside code.

## Architecture

```
+-------------------------------------------------------------------+
|                    Beads Integration Module                        |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  |                 BeadsLifecycleManager                       |  |
|  |                                                             |  |
|  |  - Creates discrepancy issues                               |  |
|  |  - Creates rebuild issues                                   |  |
|  |  - Monitors for resolutions                                 |  |
|  |  - Applies resolutions                                      |  |
|  |  - Handles timeouts                                         |  |
|  +-------------------------------------------------------------+  |
|           |              |                |                       |
|           v              v                v                       |
|  +--------------+  +--------------+  +--------------+             |
|  | BeadsClient  |  |IssueTracker  |  |TemplateEngine|             |
|  |              |  |              |  |              |             |
|  | - bd CLI     |  | - Track state|  | - Render     |             |
|  | - JSONL ops  |  | - Query      |  |   content    |             |
|  | - Git sync   |  | - Serialize  |  | - Parse      |             |
|  +--------------+  +--------------+  +--------------+             |
|           |                                                       |
|           v                                                       |
|  +-------------------------------------------------------------+  |
|  |              .beads/ directory (JSONL files)                |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

## Module Structure

```
src/twinscribe/beads/
+-- __init__.py        # Public exports
+-- client.py          # Beads CLI wrapper
+-- tracker.py         # Issue tracking and state
+-- templates.py       # Issue content rendering
+-- manager.py         # Lifecycle coordination
```

## Component Details

### BeadsClient (client.py)

Wrapper for interacting with the Beads CLI (`bd`).

```python
class BeadsClient:
    """Client for interacting with Beads via CLI."""

    def __init__(self, beads_dir: str = ".beads"):
        """Initialize with path to Beads directory."""

    async def initialize(self) -> None:
        """Initialize Beads in the repository if needed."""

    async def create_issue(self, request: CreateIssueRequest) -> BeadsIssue:
        """Create a new issue using bd create."""

    async def get_issue(self, issue_id: str) -> BeadsIssue:
        """Get issue details using bd show."""

    async def update_issue(self, issue_id: str, **kwargs) -> BeadsIssue:
        """Update issue status/fields using bd update."""

    async def add_comment(self, issue_id: str, body: str) -> None:
        """Add a comment to an issue."""

    async def list_ready(self) -> list[BeadsIssue]:
        """List issues ready for work (no blocking deps)."""

    async def close_issue(self, issue_id: str) -> BeadsIssue:
        """Close an issue using bd close."""

    async def sync(self) -> None:
        """Sync with git using bd sync."""
```

#### Data Classes

| Class | Purpose |
|-------|---------|
| `BeadsClientConfig` | Configuration (directory, labels, priorities) |
| `BeadsIssue` | Issue representation with id, title, status, etc. |
| `CreateIssueRequest` | Request object for issue creation |

#### Issue IDs

Beads uses hash-based IDs (e.g., `bd-a1b2`) to prevent merge conflicts. Issues can also have hierarchical structure:
- Epic: `bd-a3f8`
- Task: `bd-a3f8.1`
- Subtask: `bd-a3f8.1.1`

### IssueTracker (tracker.py)

Tracks Beads issues and maps them to discrepancies and components.

```python
class IssueTracker:
    """Tracks Beads issues and their resolution status."""

    def track(issue_id, issue_type, ...) -> TrackedIssue
    def get(issue_id: str) -> Optional[TrackedIssue]
    def get_by_discrepancy(discrepancy_id: str) -> Optional[TrackedIssue]
    def get_by_component(component_id: str) -> list[TrackedIssue]
    def query(query: IssueQuery) -> list[TrackedIssue]
    def update_resolution(issue_id, text, action) -> TrackedIssue
    def mark_applied(issue_id: str) -> TrackedIssue
    def expire_issue(issue_id: str) -> TrackedIssue
```

#### Issue States

```
                    +----------+
                    |  PENDING |
                    +----+-----+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    +-----------+  +-----------+  +-----------+
    |IN_PROGRESS|  | RESOLVED  |  |  EXPIRED  |
    +-----------+  +-----+-----+  +-----------+
          |              |
          v              v
    +-----------+  +-----------+
    | RESOLVED  |  |  APPLIED  |
    +-----------+  +-----------+
          |
          v
    +-----------+
    |  APPLIED  |
    +-----------+
```

| State | Description |
|-------|-------------|
| `PENDING` | Issue created, awaiting human review |
| `IN_PROGRESS` | Human is actively working on resolution |
| `RESOLVED` | Human provided resolution, not yet applied |
| `APPLIED` | Resolution applied to documentation |
| `EXPIRED` | Timeout reached without resolution |
| `CANCELLED` | Issue was cancelled |

#### TrackedIssue

```python
@dataclass
class TrackedIssue:
    issue_id: str                  # e.g., "bd-a1b2"
    issue_type: IssueType          # DISCREPANCY or REBUILD
    status: IssueStatus
    discrepancy_id: Optional[str]  # Links to Discrepancy model
    component_id: Optional[str]    # Links to Component model
    created_at: datetime
    updated_at: datetime
    resolution_text: Optional[str]
    resolution_action: Optional[str]
    timeout_at: Optional[datetime]
    metadata: dict[str, Any]
```

### IssueTemplateEngine (templates.py)

Renders issue content from templates.

```python
class IssueTemplateEngine:
    """Renders issue content from templates."""

    def render_discrepancy(data, template_name) -> tuple[str, str]
    def render_rebuild(data, template_name) -> tuple[str, str]
    def register_template(template: IssueTemplate) -> None
    def get_labels(data, template_name) -> list[str]
    def get_priority(data, template_name) -> int
```

#### Template Data Classes

**DiscrepancyTemplateData**
```python
@dataclass
class DiscrepancyTemplateData:
    discrepancy_id: str
    component_name: str
    component_type: str
    file_path: str
    discrepancy_type: str
    stream_a_value: str
    stream_b_value: str
    static_analysis_value: Optional[str]
    context: str
    iteration: int
    previous_attempts: list[str]
    labels: list[str]
    priority: int
```

**RebuildTemplateData**
```python
@dataclass
class RebuildTemplateData:
    component_name: str
    component_type: str
    file_path: str
    documentation: str
    call_graph: dict[str, list[str]]
    dependencies: list[str]
    dependents: list[str]
    complexity_score: float
    rebuild_priority: int
    suggested_approach: str
    labels: list[str]
    parent_id: Optional[str]  # For epic/subtask hierarchy
```

#### Resolution Parsing

```python
class ResolutionParser:
    """Parses human resolutions from issue comments."""

    def parse(comment_text: str) -> Optional[tuple[str, str]]
    def is_resolution_comment(comment_text: str) -> bool
```

Expected resolution format in comments:
```
RESOLUTION: <accept_a|accept_b|merge|manual>
<explanation and/or corrected content>
```

### BeadsLifecycleManager (manager.py)

Coordinates the full lifecycle of Beads issues.

```python
class BeadsLifecycleManager:
    """Manages the lifecycle of Beads issues."""

    async def initialize() -> None
    async def close() -> None

    # Issue Creation
    async def create_discrepancy_issue(data) -> TrackedIssue
    async def create_rebuild_issue(data) -> TrackedIssue
    async def create_rebuild_epic(name, components) -> str

    # Resolution Monitoring
    async def check_for_resolution(issue_id) -> Optional[IssueResolution]
    async def wait_for_resolution(issue_id, timeout) -> Optional[IssueResolution]
    async def start_monitoring() -> None
    async def stop_monitoring() -> None

    # Resolution Application
    async def apply_resolution(resolution, apply_func) -> ResolutionResult

    # State Management
    async def sync_from_beads() -> int
    def get_statistics() -> dict
```

#### Configuration

```python
class ManagerConfig(BaseModel):
    directory: str = ".beads"             # Beads directory
    poll_interval_seconds: int = 60       # Resolution poll interval
    timeout_hours: int = 48               # Issue timeout
    auto_create_issues: bool = True
    default_labels: list[str] = ["ai-documentation", "twinscribe"]
    discrepancy_priority: int = 1         # 0=highest
    rebuild_priority: int = 0
```

## Data Flow

### Discrepancy Issue Flow

```
Comparison Result      DiscrepancyTemplateData       bd create
(discrepancy) -------> (render template) ----------> (CLI) -------+
                                                                   |
                                                                   v
                                                             BeadsIssue
                                                                   |
                                                                   v
      TrackedIssue <-----------------------------------------------+
           |
           | (monitoring loop via bd show)
           v
      Check Status ----> Parse Resolution ----> IssueResolution
                                                      |
                                                      v
                                              Apply to Documentation
                                                      |
                                                      v
                                                Mark APPLIED (bd close)
```

### Rebuild Issue Flow

```
Final Documentation     RebuildTemplateData           bd create
(component docs) -----> (render template) --------> (CLI) -------+
                                                                   |
                                                                   v
                                                             BeadsIssue
                                                                   |
                                                                   v
      TrackedIssue <-----------------------------------------------+
           |
           | (optional: link to epic via dep add)
           v
      Ready for Manual Rebuild
```

## Integration Points

### With Convergence Module

```python
# In convergence loop
if discrepancy.requires_human_resolution:
    issue = await beads_manager.create_discrepancy_issue(
        DiscrepancyTemplateData(
            discrepancy_id=discrepancy.id,
            component_name=component.name,
            ...
        )
    )

    # Option 1: Wait synchronously
    resolution = await beads_manager.wait_for_resolution(
        issue.issue_id,
        timeout_seconds=config.beads_timeout_hours * 3600
    )

    # Option 2: Continue and check later
    beads_manager.on_resolution(handle_resolution_callback)
    await beads_manager.start_monitoring()
```

### With Configuration Module

Beads configuration is loaded from `BeadsConfig`:

```python
class BeadsConfig(BaseModel):
    enabled: bool = True
    directory: str = ".beads"
    labels: list[str] = ["ai-documentation", "twinscribe"]
    auto_create_issues: bool = True
    discrepancy_priority: int = 1
    rebuild_priority: int = 0
```

## CLI Commands Used

| Command | Purpose |
|---------|---------|
| `bd init` | Initialize Beads in repository |
| `bd create "Title" -p <priority>` | Create new issue |
| `bd show <id>` | Get issue details |
| `bd update <id> --status <status>` | Update issue |
| `bd close <id>` | Close resolved issue |
| `bd ready` | List ready issues |
| `bd dep add <id> <dep_id>` | Add dependency |
| `bd sync` | Sync with git |

## Resolution Actions

| Action | Description | Effect |
|--------|-------------|--------|
| `accept_a` | Use Stream A's interpretation | Apply A's value directly |
| `accept_b` | Use Stream B's interpretation | Apply B's value directly |
| `merge` | Combine both interpretations | Use provided merged content |
| `manual` | Override with manual content | Use provided manual content |

## Error Handling

### Transient Errors

- CLI timeouts: Retry with exponential backoff
- Git conflicts: Run `bd sync` and retry
- File system errors: Log and retry

### Permanent Errors

- Beads not initialized: Run `bd init` or raise error
- Invalid issue ID: Raise `NotFoundError`
- Permission denied: Raise filesystem error

### Timeout Handling

When an issue times out without resolution:
1. Mark issue as `EXPIRED` in tracker
2. Log warning with issue details
3. Return to convergence loop for fallback handling
4. Options: auto-select based on static analysis, escalate, or fail

## Serialization

The tracker supports serialization for persistence and recovery:

```python
# Save state
state = tracker.to_dict()
with open("tracker_state.json", "w") as f:
    json.dump(state, f)

# Restore state
with open("tracker_state.json") as f:
    state = json.load(f)
tracker = IssueTracker.from_dict(state)
```

## Usage Examples

### Basic Discrepancy Workflow

```python
from twinscribe.beads import (
    BeadsLifecycleManager,
    ManagerConfig,
    DiscrepancyTemplateData,
)
from twinscribe.beads.client import BeadsClient, BeadsClientConfig

# Initialize
client_config = BeadsClientConfig(
    directory=".beads",
    labels=["ai-documentation", "twinscribe"],
)
client = BeadsClient(client_config)

manager_config = ManagerConfig(
    timeout_hours=24,
)
manager = BeadsLifecycleManager(client, manager_config)
await manager.initialize()

# Create discrepancy issue
data = DiscrepancyTemplateData(
    discrepancy_id="disc-001",
    component_name="calculate_total",
    component_type="function",
    file_path="/src/billing/calculator.py",
    discrepancy_type="call_graph",
    stream_a_value="calls: process_payment, log_transaction",
    stream_b_value="calls: process_payment, send_receipt",
    static_analysis_value="calls: process_payment, log_transaction",
    context="Stream B may have hallucinated send_receipt call",
)

issue = await manager.create_discrepancy_issue(data)
print(f"Created issue: {issue.issue_id}")

# Wait for resolution
resolution = await manager.wait_for_resolution(issue.issue_id)
if resolution:
    print(f"Resolved with action: {resolution.action}")
    # Apply resolution...
else:
    print("Issue timed out")
```

### Async Monitoring

```python
# Register callback for resolutions
def handle_resolution(resolution: IssueResolution):
    print(f"Issue {resolution.issue_id} resolved: {resolution.action}")
    # Queue for processing...

manager.on_resolution(handle_resolution)

# Start background monitoring
await manager.start_monitoring()

# ... continue with other work ...

# Stop when done
await manager.stop_monitoring()
```

### Creating Rebuild Issues

```python
from twinscribe.beads import RebuildTemplateData

# Create epic for the rebuild project
epic_id = await manager.create_rebuild_epic(
    "Legacy Billing Module Rebuild",
    components=[...],
)

# Create individual rebuild issues as subtasks
for component in components:
    data = RebuildTemplateData(
        component_name=component.name,
        component_type=component.type.value,
        file_path=component.file_path,
        documentation=component.final_documentation,
        call_graph={
            "calls": component.call_graph.get_callees(component.id),
            "called_by": component.call_graph.get_callers(component.id),
        },
        dependencies=component.dependencies,
        rebuild_priority=component.rebuild_order,
        complexity_score=component.complexity,
        parent_id=epic_id,
    )

    await manager.create_rebuild_issue(data)
```

## Security Considerations

1. **Git-backed**: All issues stored in git, providing full audit trail
2. **Local-first**: No external API credentials needed
3. **Stealth mode**: Optional local-only mode without committing to shared repos
4. **File permissions**: Respect filesystem permissions

## Testing Strategy

### Unit Tests
- Template rendering with various data combinations
- Resolution parsing with valid and invalid formats
- Tracker state management and serialization
- Error handling for various failure modes

### Integration Tests
- Mock CLI output for end-to-end workflows
- Timeout handling verification
- State recovery after restart

### End-to-End Tests
- Real Beads initialization and issue lifecycle
- Full discrepancy workflow
- Rebuild issue creation with epic hierarchy
